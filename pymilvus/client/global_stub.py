import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import requests

from pymilvus.exceptions import MilvusException

if TYPE_CHECKING:
    from pymilvus.client.grpc_handler import GrpcHandler

logger = logging.getLogger(__name__)


def is_global_endpoint(uri: str) -> bool:
    """Check if the URI points to a global cluster endpoint."""
    if not uri:
        return False
    return "global-cluster" in uri.lower()


class ClusterCapability:
    """Bitset flags for cluster capabilities."""

    READABLE = 0b01  # bit 0
    WRITABLE = 0b10  # bit 1
    PRIMARY = 0b11  # read + write


@dataclass
class ClusterInfo:
    """Information about a cluster in the global topology."""

    cluster_id: str
    endpoint: str
    capability: int

    @property
    def is_primary(self) -> bool:
        """Check if this cluster is the primary (writable) cluster."""
        return (self.capability & ClusterCapability.WRITABLE) != 0


@dataclass
class GlobalTopology:
    """Global cluster topology containing all clusters."""

    version: int
    clusters: List[ClusterInfo]

    @property
    def primary(self) -> ClusterInfo:
        """Get the primary cluster from the topology."""
        for cluster in self.clusters:
            if cluster.is_primary:
                return cluster
        msg = "No primary cluster found in topology"
        raise ValueError(msg)


# Constants for retry logic
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 10.0  # seconds
REQUEST_TIMEOUT = 10  # seconds


def _parse_topology_response(data: dict) -> GlobalTopology:
    """Parse the topology response from the REST API."""
    clusters = [
        ClusterInfo(
            cluster_id=c["clusterId"],
            endpoint=c["endpoint"],
            capability=c["capability"],
        )
        for c in data["clusters"]
    ]
    return GlobalTopology(version=int(data["version"]), clusters=clusters)


def fetch_topology(global_endpoint: str, token: str) -> GlobalTopology:
    """Fetch the global cluster topology from the REST API.

    Args:
        global_endpoint: The global cluster endpoint URL
        token: Authentication token

    Returns:
        GlobalTopology object containing cluster information

    Raises:
        MilvusException: If topology cannot be fetched after retries
    """
    # Build the topology URL
    endpoint = global_endpoint.rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    url = f"{endpoint}/global-cluster/topology"

    headers = {"Authorization": f"Bearer {token}"}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code != 200:
                msg = f"Topology request failed with status {response.status_code}: {response.text}"
                raise RuntimeError(msg)  # Retryable HTTP error

            result = response.json()
            if result.get("code", 0) != 0:
                raise MilvusException(message=result.get("message", "Unknown API error"))

            return _parse_topology_response(result["data"])

        except MilvusException:
            raise
        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
                delay += random.uniform(0, delay * 0.1)  # noqa: S311 - jitter doesn't need crypto-grade randomness
                logger.warning(f"Topology fetch attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)

    raise MilvusException(message=f"Failed to fetch global topology after {MAX_RETRIES} attempts: {last_error}")


# Default refresh interval
DEFAULT_REFRESH_INTERVAL = 300  # 5 minutes


class TopologyRefresher:
    """Background thread that periodically refreshes the global cluster topology."""

    def __init__(
        self,
        global_endpoint: str,
        token: str,
        topology: GlobalTopology,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
        on_topology_change: Optional[callable] = None,
    ):
        self._global_endpoint = global_endpoint
        self._token = token
        self._topology = topology
        self._refresh_interval = refresh_interval
        self._on_topology_change = on_topology_change

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background refresh thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background refresh thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_running(self) -> bool:
        """Check if the refresh thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def get_topology(self) -> GlobalTopology:
        """Get the current topology (thread-safe)."""
        with self._lock:
            return self._topology

    def trigger_refresh(self) -> None:
        """Trigger an immediate topology refresh (async)."""
        threading.Thread(target=self._try_refresh, daemon=True).start()

    def _refresh_loop(self) -> None:
        """Main refresh loop running in background thread."""
        while not self._stop_event.wait(self._refresh_interval):
            self._try_refresh()

    def _try_refresh(self) -> None:
        """Attempt to refresh the topology."""
        try:
            new_topology = fetch_topology(self._global_endpoint, self._token)

            with self._lock:
                if new_topology.version > self._topology.version:
                    old_topology = self._topology
                    self._topology = new_topology
                    logger.info(
                        f"Topology updated: version {old_topology.version} -> {new_topology.version}"
                    )

                    if self._on_topology_change:
                        try:
                            self._on_topology_change(new_topology)
                        except Exception as e:
                            logger.warning(f"Topology change callback failed: {e}")

        except Exception as e:
            logger.warning(f"Topology refresh failed: {e}")
            # Keep using cached topology, will retry next interval


class GlobalStub:
    """Global cluster stub that manages topology discovery and primary connection."""

    def __init__(
        self,
        global_endpoint: str,
        token: str,
        *,
        start_refresher: bool = True,
        **handler_kwargs,
    ):
        """Initialize the global stub.

        Args:
            global_endpoint: The global cluster endpoint URL
            token: Authentication token
            start_refresher: Whether to start the background topology refresher
            **handler_kwargs: Additional kwargs to pass to GrpcHandler
        """
        self._global_endpoint = global_endpoint
        self._token = token
        self._handler_kwargs = handler_kwargs
        self._lock = threading.Lock()

        # Fetch initial topology
        self._topology = fetch_topology(global_endpoint, token)

        # Connect to primary cluster
        self._primary_handler = self._create_primary_handler()

        # Start background refresher
        self._refresher = TopologyRefresher(
            global_endpoint=global_endpoint,
            token=token,
            topology=self._topology,
            on_topology_change=self._on_topology_change,
        )
        if start_refresher:
            self._refresher.start()

    def _create_primary_handler(self) -> "GrpcHandler":
        """Create a GrpcHandler for the primary cluster."""
        from pymilvus.client.grpc_handler import GrpcHandler  # noqa: PLC0415

        primary = self._topology.primary
        return GrpcHandler(uri=primary.endpoint, token=self._token, **self._handler_kwargs)

    def _on_topology_change(self, new_topology: GlobalTopology) -> None:
        """Handle topology change - reconnect to new primary if needed."""
        with self._lock:
            old_primary = self._topology.primary
            new_primary = new_topology.primary

            self._topology = new_topology

            if old_primary.endpoint != new_primary.endpoint:
                logger.info(f"Primary changed: {old_primary.endpoint} -> {new_primary.endpoint}")
                old_handler = self._primary_handler
                self._primary_handler = self._create_primary_handler()

                # Close old handler after new one is established
                try:
                    old_handler.close()
                except Exception as e:
                    logger.warning(f"Failed to close old handler: {e}")

    def get_topology(self) -> GlobalTopology:
        """Get the current topology."""
        return self._refresher.get_topology()

    def get_primary_endpoint(self) -> str:
        """Get the primary cluster endpoint."""
        with self._lock:
            return self._topology.primary.endpoint

    def get_primary_handler(self) -> "GrpcHandler":
        """Get the GrpcHandler for the primary cluster."""
        with self._lock:
            return self._primary_handler

    def trigger_refresh(self) -> None:
        """Trigger an immediate topology refresh."""
        self._refresher.trigger_refresh()

    def close(self) -> None:
        """Close the global stub and release resources."""
        self._refresher.stop()
        with self._lock:
            if self._primary_handler:
                try:
                    self._primary_handler.close()
                except Exception as e:
                    logger.warning(f"Failed to close primary handler: {e}")
