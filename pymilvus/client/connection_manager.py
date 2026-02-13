"""Connection manager for MilvusClient.

This module provides connection pooling, lifecycle management, and strategy-based
connection handling for MilvusClient, bypassing the legacy connections singleton.
"""

import asyncio
import contextlib
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional
from urllib.parse import urlparse

import grpc

if TYPE_CHECKING:
    from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
    from pymilvus.client.grpc_handler import GrpcHandler

from pymilvus.client.global_stub import (
    GLOBAL_CLUSTER_IDENTIFIER,
    GlobalTopology,
    TopologyRefresher,
    fetch_topology,
)
from pymilvus.exceptions import ConnectionConfigException

logger = logging.getLogger(__name__)


DEFAULT_PORT = 19530


@dataclass
class ConnectionConfig:
    """Configuration for a Milvus connection.

    Attributes:
        uri: Original URI string (before parsing)
        address: Parsed host:port address
        token: Authentication token (from param or URI credentials)
        db_name: Database name (from param or URI path)
    """

    uri: str
    address: str
    token: str = ""
    db_name: str = ""

    @property
    def key(self) -> str:
        """Return deduplication key: address|token."""
        return f"{self.address}|{self.token}"

    @property
    def is_global(self) -> bool:
        """Check if this is a global cluster endpoint."""
        return GLOBAL_CLUSTER_IDENTIFIER in self.uri.lower()

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        token: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> "ConnectionConfig":
        """Create ConnectionConfig from URI with optional overrides.

        Args:
            uri: Connection URI (e.g., https://user:pass@host:19530/mydb)
            token: Override token (default: extracted from URI)
            db_name: Override database name (default: extracted from URI path)

        Returns:
            ConnectionConfig with parsed/overridden values

        Raises:
            ConnectionConfigException: If the URI format is invalid
        """
        # Validate URI format
        valid_schemes = ("unix", "http", "https", "tcp")
        is_local_db = uri.endswith(".db")
        parsed = urlparse(uri)

        if not is_local_db and parsed.scheme.lower() not in valid_schemes:
            schemes_str = "[" + ", ".join(valid_schemes) + "]"
            raise ConnectionConfigException(
                message=f"uri: {uri} is illegal, needs start with {schemes_str} "
                f"or a local file endswith [.db]"
            )

        # Extract host:port
        host = parsed.hostname or "localhost"
        port = parsed.port or DEFAULT_PORT
        address = f"{host}:{port}"

        # Extract token from credentials if present
        uri_token = ""
        if parsed.username:
            uri_token = parsed.username
            if parsed.password:
                uri_token = f"{parsed.username}:{parsed.password}"

        # Extract db_name from path (first segment)
        uri_db_name = ""
        if parsed.path and parsed.path != "/":
            path_parts = parsed.path.strip("/").split("/")
            if path_parts and path_parts[0]:
                uri_db_name = path_parts[0]

        # Apply explicit overrides (explicit params win over URI)
        # For token and db_name, empty string means "use URI value"
        final_token = token if token else uri_token
        final_db_name = db_name if db_name else uri_db_name

        return cls(
            uri=uri,
            address=address,
            token=final_token,
            db_name=final_db_name,
        )


@dataclass
class ManagedConnection:
    """A connection managed by ConnectionManager.

    Attributes:
        handler: The underlying GrpcHandler
        config: Original connection configuration
        strategy: Connection strategy (Regular or Global)
        created_at: Unix timestamp when connection was created
        last_used_at: Unix timestamp when connection was last accessed
        clients: WeakSet of MilvusClient instances using this connection
    """

    handler: "GrpcHandler"
    config: ConnectionConfig
    strategy: Any  # ConnectionStrategy, defined later
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    clients: weakref.WeakSet = field(default_factory=weakref.WeakSet)

    def touch(self) -> None:
        """Update last_used_at to current time."""
        self.last_used_at = time.time()

    @property
    def idle_time(self) -> float:
        """Return seconds since last use."""
        return time.time() - self.last_used_at

    @property
    def has_clients(self) -> bool:
        """Return True if any clients reference this connection."""
        return len(self.clients) > 0

    def add_client(self, client: Any) -> None:
        """Add a client to the reference set."""
        self.clients.add(client)

    def remove_client(self, client: Any) -> None:
        """Remove a client from the reference set."""
        self.clients.discard(client)


class ConnectionStrategy(ABC):
    """Abstract base class for connection strategies.

    Strategies define how to create handlers and handle UNAVAILABLE errors.
    """

    @abstractmethod
    def create_handler(self, config: ConnectionConfig) -> "GrpcHandler":
        """Create a new GrpcHandler for the given config.

        Args:
            config: Connection configuration

        Returns:
            New GrpcHandler instance
        """

    @abstractmethod
    def on_unavailable(self, managed: ManagedConnection) -> bool:
        """Handle UNAVAILABLE error for a connection.

        Args:
            managed: The managed connection that encountered the error

        Returns:
            True if recovery should be triggered, False otherwise
        """

    @abstractmethod
    def close(self, managed: ManagedConnection) -> None:
        """Close resources for a managed connection (sync).

        Args:
            managed: The managed connection to close
        """

    async def close_async(self, managed: ManagedConnection) -> None:
        """Close resources for a managed connection (async).

        Default implementation delegates to sync close().
        Async strategies should override this for proper async cleanup.

        Args:
            managed: The managed connection to close
        """
        self.close(managed)


class RegularStrategy(ConnectionStrategy):
    """Strategy for regular (non-global) connections.

    Creates direct connections to a single Milvus endpoint.
    Always triggers recovery on UNAVAILABLE errors.
    """

    def create_handler(self, config: ConnectionConfig) -> "GrpcHandler":
        """Create a GrpcHandler for the configured endpoint."""
        from pymilvus.client.grpc_handler import GrpcHandler  # noqa: PLC0415

        return GrpcHandler(
            uri=config.uri,
            token=config.token,
            db_name=config.db_name,
        )

    def on_unavailable(self, managed: ManagedConnection) -> bool:
        """Regular connections always need recovery on UNAVAILABLE."""
        return True

    def close(self, managed: ManagedConnection) -> None:
        """Close the handler."""
        managed.handler.close()


class _GlobalStrategyMixin:
    """Shared topology management for sync and async global strategies.

    Provides common init, topology refresh callback, on_unavailable logic,
    and topology fetch + refresher setup used by both GlobalStrategy and
    AsyncGlobalStrategy.
    """

    def __init__(self):
        self._topology: Optional[GlobalTopology] = None
        self._config: Optional[ConnectionConfig] = None
        self._refresher: Optional[TopologyRefresher] = None
        self._lock = threading.Lock()

    def _fetch_and_start_refresher(self, config: ConnectionConfig) -> GlobalTopology:
        """Fetch initial topology and start background refresher.

        Returns:
            The fetched GlobalTopology (caller uses it to connect to primary).
        """
        topology = fetch_topology(config.uri, config.token)

        with self._lock:
            self._topology = topology
            self._config = config

        self._refresher = TopologyRefresher(
            global_endpoint=config.uri,
            token=config.token,
            topology=topology,
            on_topology_change=self._on_topology_change,
        )
        self._refresher.start()
        return topology

    def _on_topology_change(self, new_topology: GlobalTopology) -> None:
        """Callback when topology changes via background refresh."""
        with self._lock:
            self._topology = new_topology

    def on_unavailable(self, managed: ManagedConnection) -> bool:
        """Refresh topology and return True if primary changed."""
        if self._config is None:
            return True  # No config, trigger recovery

        with self._lock:
            old_primary = self._topology.primary.endpoint if self._topology else None

        # Fetch fresh topology
        try:
            new_topology = fetch_topology(self._config.uri, self._config.token)
        except Exception:
            logger.warning("Failed to refresh topology on UNAVAILABLE", exc_info=True)
            return True  # Error fetching, trigger recovery anyway

        with self._lock:
            self._topology = new_topology
            new_primary = new_topology.primary.endpoint

        # Only recover if primary actually changed
        if old_primary != new_primary:
            logger.info(f"Primary changed: {old_primary} -> {new_primary}")
            return True

        return False

    def get_topology(self) -> Optional[GlobalTopology]:
        """Get current topology (thread-safe)."""
        with self._lock:
            return self._topology

    def _stop_refresher(self) -> None:
        """Stop the background topology refresher."""
        if self._refresher:
            self._refresher.stop()
            self._refresher = None


class GlobalStrategy(_GlobalStrategyMixin, ConnectionStrategy):
    """Strategy for global cluster connections.

    Manages topology discovery, primary routing, and background refresh.
    On UNAVAILABLE, refreshes topology and triggers recovery only if primary changed.
    """

    def create_handler(self, config: ConnectionConfig) -> "GrpcHandler":
        """Fetch topology and create handler for primary cluster."""
        from pymilvus.client.grpc_handler import GrpcHandler  # noqa: PLC0415

        topology = self._fetch_and_start_refresher(config)
        primary = topology.primary
        return GrpcHandler(
            uri=primary.endpoint,
            token=config.token,
            db_name=config.db_name,
        )

    def close(self, managed: ManagedConnection) -> None:
        """Stop refresher and close handler."""
        self._stop_refresher()
        managed.handler.close()


# Health check constants
IDLE_THRESHOLD_SECONDS = 30.0
HEALTH_CHECK_TIMEOUT = 5.0


class ConnectionManager:
    """Manages connections for MilvusClient.

    Thread-safe singleton that handles:
    - Connection deduplication by address|token
    - Shared vs dedicated connection modes
    - Health checks on idle connections
    - UNAVAILABLE error recovery via strategies
    """

    _instance: Optional["ConnectionManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.RLock()
        self._registry: Dict[str, ManagedConnection] = {}  # key -> ManagedConnection
        self._dedicated: Dict[int, ManagedConnection] = {}  # handler_id -> ManagedConnection

    @classmethod
    def get_instance(cls) -> "ConnectionManager":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.close_all()
            cls._instance = None

    def get_or_create(
        self,
        config: ConnectionConfig,
        dedicated: bool = False,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> "GrpcHandler":
        """Get existing or create new connection.

        Args:
            config: Connection configuration
            dedicated: If True, create a new connection even if one exists
            client: MilvusClient instance to track (for shared connections)
            timeout: Connection timeout in seconds

        Returns:
            GrpcHandler for the connection
        """
        with self._lock:
            if dedicated:
                return self._create_dedicated(config, client, timeout)

            # Check for existing shared connection
            key = config.key
            if key in self._registry:
                managed = self._registry[key]
                if client:
                    managed.add_client(client)

                # Health check if idle too long (before touch so idle_time is accurate)
                if managed.idle_time > IDLE_THRESHOLD_SECONDS and not self._check_health(managed):
                    self._recover(managed)

                managed.touch()
                return managed.handler

            # Create new shared connection
            return self._create_shared(config, client, timeout)

    def _register_error_callback(self, handler: "GrpcHandler") -> None:
        """Register an RPC error callback on the handler.

        The retry_on_rpc_failure decorator calls handler._on_rpc_error(e)
        on gRPC errors. This routes errors to ConnectionManager.handle_error().
        """
        handler._on_rpc_error = lambda error: self.handle_error(handler, error)

    def _create_shared(
        self,
        config: ConnectionConfig,
        client: Optional[Any],
        timeout: Optional[float],
    ) -> "GrpcHandler":
        """Create a new shared connection."""
        strategy = self._get_strategy(config)
        handler = strategy.create_handler(config)
        self._register_error_callback(handler)

        # Wait for channel ready
        handler._wait_for_channel_ready(timeout=timeout)

        managed = ManagedConnection(
            handler=handler,
            config=config,
            strategy=strategy,
        )
        if client:
            managed.add_client(client)

        self._registry[config.key] = managed
        return handler

    def _create_dedicated(
        self,
        config: ConnectionConfig,
        client: Optional[Any],
        timeout: Optional[float],
    ) -> "GrpcHandler":
        """Create a new dedicated connection."""
        strategy = self._get_strategy(config)
        handler = strategy.create_handler(config)
        self._register_error_callback(handler)

        # Wait for channel ready
        handler._wait_for_channel_ready(timeout=timeout)

        managed = ManagedConnection(
            handler=handler,
            config=config,
            strategy=strategy,
        )
        if client:
            managed.add_client(client)

        self._dedicated[id(handler)] = managed
        return handler

    def _get_strategy(self, config: ConnectionConfig) -> ConnectionStrategy:
        """Get appropriate strategy for config."""
        if config.is_global:
            return GlobalStrategy()
        return RegularStrategy()

    def _get_managed(self, handler: "GrpcHandler") -> Optional[ManagedConnection]:
        """Get ManagedConnection for a handler."""
        # Check dedicated first
        handler_id = id(handler)
        if handler_id in self._dedicated:
            return self._dedicated[handler_id]

        # Check shared
        for managed in self._registry.values():
            if managed.handler is handler:
                return managed

        return None

    def release(
        self,
        handler: "GrpcHandler",
        client: Optional[Any] = None,
    ) -> None:
        """Release a connection.

        For shared connections, removes client reference.
        For dedicated connections, closes immediately.

        Args:
            handler: The handler to release
            client: The client releasing the connection
        """
        with self._lock:
            handler_id = id(handler)

            # Dedicated connection - close immediately
            if handler_id in self._dedicated:
                managed = self._dedicated.pop(handler_id)
                managed.strategy.close(managed)
                return

            # Shared connection - remove client reference
            managed = self._get_managed(handler)
            if managed and client:
                managed.remove_client(client)

    def close_all(self) -> None:
        """Close all connections."""
        with self._lock:
            # Close shared connections
            for managed in list(self._registry.values()):
                try:
                    managed.strategy.close(managed)
                except Exception:
                    logger.warning("Failed to close connection", exc_info=True)
            self._registry.clear()

            # Close dedicated connections
            for managed in list(self._dedicated.values()):
                try:
                    managed.strategy.close(managed)
                except Exception:
                    logger.warning("Failed to close dedicated connection", exc_info=True)
            self._dedicated.clear()

    def _check_health(self, managed: ManagedConnection) -> bool:
        """Check if connection is healthy.

        Args:
            managed: Connection to check

        Returns:
            True if healthy, False if needs recovery
        """
        try:
            # Check gRPC channel state
            channel = getattr(managed.handler, "_channel", None)
            if channel is None:
                return False

            state = channel.check_connectivity_state(True)
            if state.name == "SHUTDOWN":
                return False

            # Try GetVersion as health probe
            managed.handler.get_server_version(timeout=HEALTH_CHECK_TIMEOUT)

        except Exception:
            return False
        else:
            return True

    def _recover(self, managed: ManagedConnection) -> None:
        """Recover a connection by creating new handler.

        Uses strategy.close() instead of handler.close() so that strategy-owned
        resources (e.g. GlobalStrategy's TopologyRefresher) are also cleaned up.

        Args:
            managed: Connection to recover
        """
        try:
            old_handler_id = id(managed.handler)

            # Close old handler and strategy resources (e.g. TopologyRefresher)
            with contextlib.suppress(Exception):
                managed.strategy.close(managed)

            # Create new handler via strategy
            new_handler = managed.strategy.create_handler(managed.config)
            self._register_error_callback(new_handler)
            new_handler._wait_for_channel_ready()

            # Update managed connection
            managed.handler = new_handler
            managed.touch()

            # Update dedicated registry key if this is a dedicated connection
            if old_handler_id in self._dedicated:
                self._dedicated.pop(old_handler_id)
                self._dedicated[id(new_handler)] = managed

        except Exception:
            logger.warning("Connection recovery failed", exc_info=True)
            raise

    def handle_error(
        self,
        handler: "GrpcHandler",
        error: Exception,
    ) -> None:
        """Handle an error from a connection.

        For UNAVAILABLE errors, triggers strategy-specific recovery.
        The lock is released before calling on_unavailable() so that
        strategy-level network I/O (e.g. GlobalStrategy topology fetch)
        does not block other threads.

        Args:
            handler: The handler that encountered the error
            error: The exception that occurred
        """

        # Only handle UNAVAILABLE
        if not isinstance(error, grpc.RpcError):
            return

        code = error.code()
        if code != grpc.StatusCode.UNAVAILABLE:
            return

        with self._lock:
            managed = self._get_managed(handler)
            if managed is None:
                return

        # Run strategy decision outside lock (may do network I/O)
        should_recover = managed.strategy.on_unavailable(managed)

        if should_recover:
            with self._lock:
                # Re-verify the handler is still the current one; another
                # thread may have already recovered this connection.
                if managed.handler is not handler:
                    return
                self._recover(managed)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dict with connection counts and details
        """
        with self._lock:
            shared = [
                {
                    "key": key,
                    "address": m.config.address,
                    "idle_time": m.idle_time,
                    "client_count": len(m.clients),
                    "dedicated": False,
                }
                for key, m in self._registry.items()
            ]
            dedicated = [
                {
                    "key": str(handler_id),
                    "address": m.config.address,
                    "idle_time": m.idle_time,
                    "client_count": len(m.clients),
                    "dedicated": True,
                }
                for handler_id, m in self._dedicated.items()
            ]
            return {
                "total_connections": len(self._registry) + len(self._dedicated),
                "shared_connections": len(self._registry),
                "dedicated_connections": len(self._dedicated),
                "connections": shared + dedicated,
            }


class AsyncRegularStrategy(ConnectionStrategy):
    """Async strategy for regular (non-global) connections."""

    # ClassVar set keeps references to fire-and-forget asyncio tasks so they
    # are not garbage-collected before completion.  Shared across all instances
    # intentionally — the discard callback removes each task when it finishes.
    _background_tasks: ClassVar[set] = set()

    def create_handler(self, config: ConnectionConfig) -> "AsyncGrpcHandler":
        """Create an AsyncGrpcHandler for the configured endpoint."""
        from pymilvus.client.async_grpc_handler import AsyncGrpcHandler  # noqa: PLC0415

        return AsyncGrpcHandler(
            uri=config.uri,
            token=config.token,
            db_name=config.db_name,
        )

    def on_unavailable(self, managed: ManagedConnection) -> bool:
        """Regular connections always need recovery on UNAVAILABLE."""
        return True

    async def close_async(self, managed: ManagedConnection) -> None:
        """Close the handler asynchronously."""
        await managed.handler.close()

    def close(self, managed: ManagedConnection) -> None:
        """Sync close - for interface compliance, shouldn't be used."""
        # Create event loop if needed for cleanup
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget task - store reference to avoid gc
            task = loop.create_task(self.close_async(managed))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            asyncio.run(self.close_async(managed))


class AsyncGlobalStrategy(_GlobalStrategyMixin, ConnectionStrategy):
    """Async strategy for global cluster connections.

    Reuses sync fetch_topology and TopologyRefresher (both use threading),
    which is acceptable since topology fetches are short-lived and the
    refresher runs in its own daemon thread.
    """

    # ClassVar set keeps references to fire-and-forget asyncio tasks so they
    # are not garbage-collected before completion.  Shared across all instances
    # intentionally — the discard callback removes each task when it finishes.
    _background_tasks: ClassVar[set] = set()

    def create_handler(self, config: ConnectionConfig) -> "AsyncGrpcHandler":
        """Fetch topology and create async handler for primary cluster."""
        from pymilvus.client.async_grpc_handler import AsyncGrpcHandler  # noqa: PLC0415

        topology = self._fetch_and_start_refresher(config)
        primary = topology.primary
        return AsyncGrpcHandler(
            uri=primary.endpoint,
            token=config.token,
            db_name=config.db_name,
        )

    async def close_async(self, managed: ManagedConnection) -> None:
        """Stop refresher and close handler asynchronously."""
        self._stop_refresher()
        await managed.handler.close()

    def close(self, managed: ManagedConnection) -> None:
        """Sync close - for interface compliance."""
        self._stop_refresher()
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(managed.handler.close())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            asyncio.run(managed.handler.close())


class AsyncConnectionManager:
    """Async connection manager for AsyncMilvusClient.

    Uses asyncio.Lock for concurrency safety.
    """

    _instances: ClassVar[Dict[int, "AsyncConnectionManager"]] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        # Lazy-init: on Python 3.8, asyncio.Lock() requires an event loop to
        # exist in the current thread.  Deferring creation to the first async
        # call avoids RuntimeError when get_instance() is called from sync code.
        self._lock: Optional[asyncio.Lock] = None
        self._registry: Dict[str, ManagedConnection] = {}
        self._dedicated: Dict[int, ManagedConnection] = {}

    def _get_lock(self) -> asyncio.Lock:
        """Return the asyncio.Lock, creating it lazily for Python 3.8 compat."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @classmethod
    def get_instance(cls) -> "AsyncConnectionManager":
        """Get singleton instance for current event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            loop_id = 0

        with cls._instances_lock:
            if loop_id not in cls._instances:
                cls._instances[loop_id] = cls()
            return cls._instances[loop_id]

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset all instances for testing.

        Best-effort cleanup: closes handlers synchronously via strategy.close()
        since we cannot reliably await async methods from a sync classmethod
        (the caller may or may not be inside a running event loop).
        """
        with cls._instances_lock:
            for instance in cls._instances.values():
                for managed in list(instance._registry.values()):
                    with contextlib.suppress(Exception):
                        managed.strategy.close(managed)
                instance._registry.clear()
                for managed in list(instance._dedicated.values()):
                    with contextlib.suppress(Exception):
                        managed.strategy.close(managed)
                instance._dedicated.clear()
            cls._instances.clear()

    async def get_or_create(
        self,
        config: ConnectionConfig,
        dedicated: bool = False,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> "AsyncGrpcHandler":
        """Get existing or create new async connection."""
        async with self._get_lock():
            if dedicated:
                return await self._create_dedicated(config, client, timeout)

            key = config.key
            if key in self._registry:
                managed = self._registry[key]
                if client:
                    managed.add_client(client)

                # Health check if idle too long (before touch so idle_time is accurate)
                if managed.idle_time > IDLE_THRESHOLD_SECONDS and not await self._check_health(
                    managed
                ):
                    await self._recover(managed)

                managed.touch()
                return managed.handler

            return await self._create_shared(config, client, timeout)

    def _register_error_callback(self, handler: "AsyncGrpcHandler") -> None:
        """Register an RPC error callback on the handler.

        The async retry_on_rpc_failure decorator awaits handler._on_rpc_error(e)
        on gRPC errors. This routes errors to AsyncConnectionManager.handle_error().
        The callback is an async def so the entire chain is awaitable.

        The closure captures ``handler`` by reference. After recovery, the old
        handler's closure still points to the old handler object, so
        ``_get_managed(old_handler)`` returns None (the registry now holds the
        new handler). This intentionally prevents double recovery when multiple
        in-flight RPCs fail on the same old handler.
        """

        async def _on_rpc_error(error: Exception) -> None:
            await self.handle_error(handler, error)

        handler._on_rpc_error = _on_rpc_error

    async def _create_shared(
        self,
        config: ConnectionConfig,
        client: Optional[Any],
        timeout: Optional[float],
    ) -> "AsyncGrpcHandler":
        """Create new shared async connection."""
        strategy = self._get_strategy(config)
        handler = strategy.create_handler(config)
        self._register_error_callback(handler)

        await handler.ensure_channel_ready(timeout=timeout)

        managed = ManagedConnection(
            handler=handler,
            config=config,
            strategy=strategy,
        )
        if client:
            managed.add_client(client)

        self._registry[config.key] = managed
        return handler

    async def _create_dedicated(
        self,
        config: ConnectionConfig,
        client: Optional[Any],
        timeout: Optional[float],
    ) -> "AsyncGrpcHandler":
        """Create new dedicated async connection."""
        strategy = self._get_strategy(config)
        handler = strategy.create_handler(config)
        self._register_error_callback(handler)

        await handler.ensure_channel_ready(timeout=timeout)

        managed = ManagedConnection(
            handler=handler,
            config=config,
            strategy=strategy,
        )
        if client:
            managed.add_client(client)

        self._dedicated[id(handler)] = managed
        return handler

    def _get_strategy(self, config: ConnectionConfig) -> ConnectionStrategy:
        """Get appropriate async strategy for config."""
        if config.is_global:
            return AsyncGlobalStrategy()
        return AsyncRegularStrategy()

    def _get_managed(self, handler: "AsyncGrpcHandler") -> Optional[ManagedConnection]:
        """Get ManagedConnection for a handler."""
        handler_id = id(handler)
        if handler_id in self._dedicated:
            return self._dedicated[handler_id]

        for managed in self._registry.values():
            if managed.handler is handler:
                return managed

        return None

    async def _check_health(self, managed: ManagedConnection) -> bool:
        """Check if an async connection is healthy.

        Args:
            managed: Connection to check

        Returns:
            True if healthy, False if needs recovery
        """
        try:
            channel = getattr(managed.handler, "_channel", None)
            if channel is None:
                return False

            state = channel.check_connectivity_state(True)
            if state.name == "SHUTDOWN":
                return False

            await managed.handler.get_server_version(timeout=HEALTH_CHECK_TIMEOUT)

        except Exception:
            return False
        else:
            return True

    async def release(
        self,
        handler: "AsyncGrpcHandler",
        client: Optional[Any] = None,
    ) -> None:
        """Release an async connection."""
        async with self._get_lock():
            handler_id = id(handler)

            if handler_id in self._dedicated:
                managed = self._dedicated.pop(handler_id)
                await managed.strategy.close_async(managed)
                return

            managed = self._get_managed(handler)
            if managed and client:
                managed.remove_client(client)

    async def handle_error(
        self,
        handler: "AsyncGrpcHandler",
        error: Exception,
    ) -> None:
        """Handle an error from an async connection.

        For UNAVAILABLE errors, triggers strategy-specific recovery.
        Uses asyncio.Lock to prevent concurrent recovery attempts from
        multiple coroutines hitting the same UNAVAILABLE error.

        Args:
            handler: The handler that encountered the error
            error: The exception that occurred
        """
        if not isinstance(error, grpc.RpcError):
            return

        code = error.code()
        if code != grpc.StatusCode.UNAVAILABLE:
            return

        async with self._get_lock():
            managed = self._get_managed(handler)
            if managed is None:
                return

            # Run sync on_unavailable in executor to avoid blocking the
            # event loop (GlobalStrategy.on_unavailable does network I/O).
            loop = asyncio.get_running_loop()
            should_recover = await loop.run_in_executor(
                None, managed.strategy.on_unavailable, managed
            )
            if should_recover:
                await self._recover(managed)

    async def _recover(self, managed: ManagedConnection) -> None:
        """Recover an async connection by creating new handler.

        Uses strategy.close_async() to properly clean up old handler and strategy
        resources (e.g. TopologyRefresher). Awaits ensure_channel_ready() on the
        new handler so it's fully ready before being stored.

        Args:
            managed: Connection to recover
        """
        try:
            old_handler_id = id(managed.handler)

            # Close old handler and strategy resources (e.g. TopologyRefresher)
            with contextlib.suppress(Exception):
                await managed.strategy.close_async(managed)

            # Create new handler via strategy (sync - creates handler object)
            new_handler = managed.strategy.create_handler(managed.config)
            self._register_error_callback(new_handler)
            await new_handler.ensure_channel_ready()

            # Update managed connection
            managed.handler = new_handler
            managed.touch()

            # Update dedicated registry key if this is a dedicated connection
            if old_handler_id in self._dedicated:
                self._dedicated.pop(old_handler_id)
                self._dedicated[id(new_handler)] = managed

        except Exception:
            logger.warning("Async connection recovery failed", exc_info=True)
            raise

    async def close_all(self) -> None:
        """Close all async connections."""
        async with self._get_lock():
            for managed in list(self._registry.values()):
                try:
                    await managed.strategy.close_async(managed)
                except Exception:
                    logger.warning("Failed to close async connection", exc_info=True)
            self._registry.clear()

            for managed in list(self._dedicated.values()):
                try:
                    await managed.strategy.close_async(managed)
                except Exception:
                    logger.warning("Failed to close dedicated async connection", exc_info=True)
            self._dedicated.clear()
