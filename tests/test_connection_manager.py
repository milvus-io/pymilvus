# tests/test_connection_manager.py
"""Tests for ConnectionManager and related classes."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import grpc
import pytest
from pymilvus import AsyncMilvusClient, MilvusClient
from pymilvus.client.connection_manager import (
    IDLE_THRESHOLD_SECONDS,
    AsyncConnectionManager,
    AsyncGlobalStrategy,
    AsyncRegularStrategy,
    ConnectionConfig,
    ConnectionManager,
    GlobalStrategy,
    GlobalTopology,
    ManagedConnection,
    RegularStrategy,
    _GlobalStrategyMixin,
)
from pymilvus.client.global_stub import ClusterInfo
from pymilvus.exceptions import ConnectionConfigException, MilvusException

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_grpc_handler():
    """Create a mock GrpcHandler."""
    handler = Mock()
    handler._wait_for_channel_ready = Mock()
    handler.get_server_type = Mock(return_value="milvus")
    handler.close = Mock()
    return handler


@pytest.fixture
def mock_async_handler():
    """Create a mock AsyncGrpcHandler."""
    handler = Mock()
    handler.ensure_channel_ready = AsyncMock()
    handler.get_server_type = AsyncMock(return_value="milvus")
    handler.close = AsyncMock()
    return handler


@pytest.fixture
def sample_topology():
    """Create a sample GlobalTopology for testing."""
    return GlobalTopology(
        version=1,
        clusters=[
            ClusterInfo(cluster_id="c1", endpoint="https://primary:19530", capability=0b11),
        ],
    )


@pytest.fixture(autouse=True)
def reset_connection_managers():
    """Reset connection managers before and after each test."""
    ConnectionManager._reset_instance()
    AsyncConnectionManager._reset_instance()
    yield
    ConnectionManager._reset_instance()
    AsyncConnectionManager._reset_instance()


# =============================================================================
# TestConnectionConfig - URI parsing tests
# =============================================================================


class TestConnectionConfig:
    """Tests for ConnectionConfig dataclass."""

    @pytest.mark.parametrize(
        "uri,token,db_name,expected",
        [
            # Basic URI parsing
            (
                "http://localhost:19530",
                None,
                None,
                {"address": "localhost:19530", "token": "", "db_name": ""},
            ),
            (
                "https://host.example.com:19530",
                None,
                None,
                {"address": "host.example.com:19530", "token": "", "db_name": ""},
            ),
            # URI with credentials
            (
                "https://user:pass@host:19530",
                None,
                None,
                {"address": "host:19530", "token": "user:pass", "db_name": ""},
            ),
            # URI with database path
            (
                "https://host:19530/mydb",
                None,
                None,
                {"address": "host:19530", "token": "", "db_name": "mydb"},
            ),
            # Full URI with all components
            (
                "https://user:pass@host:19530/mydb",
                None,
                None,
                {"address": "host:19530", "token": "user:pass", "db_name": "mydb"},
            ),
            # Default port
            (
                "http://localhost",
                None,
                None,
                {"address": "localhost:19530", "token": "", "db_name": ""},
            ),
            # Explicit params override URI
            (
                "https://user:pass@host:19530/db1",
                "other_token",
                "db2",
                {
                    "address": "host:19530",
                    "token": "other_token",
                    "db_name": "db2",
                },
            ),
        ],
    )
    def test_from_uri(self, uri, token, db_name, expected):
        """Test URI parsing with various formats."""
        kwargs = {}
        if token is not None:
            kwargs["token"] = token
        if db_name is not None:
            kwargs["db_name"] = db_name

        config = ConnectionConfig.from_uri(uri, **kwargs)

        assert config.address == expected["address"]
        assert config.token == expected["token"]
        assert config.db_name == expected["db_name"]

    @pytest.mark.parametrize(
        "uri,expected_key",
        [
            ("https://user:pass@host:19530", "host:19530|user:pass"),
            ("http://localhost:19530", "localhost:19530|"),
        ],
    )
    def test_key_property(self, uri, expected_key):
        """Test key property for connection deduplication."""
        config = ConnectionConfig.from_uri(uri)
        assert config.key == expected_key

    @pytest.mark.parametrize(
        "uri,expected_is_global",
        [
            ("https://global-cluster.example.com:19530", True),
            ("http://localhost:19530", False),
            ("https://in01-xxx.zilliz.com:19530", False),
        ],
    )
    def test_is_global(self, uri, expected_is_global):
        """Test global cluster endpoint detection."""
        config = ConnectionConfig.from_uri(uri)
        assert config.is_global == expected_is_global


# =============================================================================
# TestManagedConnection
# =============================================================================


class TestManagedConnection:
    """Tests for ManagedConnection dataclass."""

    def test_creation_and_timestamps(self):
        """Test ManagedConnection creation and timestamp tracking."""
        config = ConnectionConfig.from_uri("http://localhost:19530")
        managed = ManagedConnection(handler=Mock(), config=config, strategy=Mock())

        assert managed.created_at > 0
        assert managed.last_used_at > 0

        # Test touch updates last_used_at
        original = managed.last_used_at
        time.sleep(0.01)
        managed.touch()
        assert managed.last_used_at > original

        # Test idle_time
        time.sleep(0.02)
        assert managed.idle_time >= 0.02

    def test_client_tracking(self):
        """Test WeakSet client tracking."""
        config = ConnectionConfig.from_uri("http://localhost:19530")
        managed = ManagedConnection(handler=Mock(), config=config, strategy=Mock())

        class FakeClient:
            pass

        assert managed.has_clients is False

        client1, client2 = FakeClient(), FakeClient()
        managed.add_client(client1)
        managed.add_client(client2)
        assert len(managed.clients) == 2
        assert managed.has_clients is True

        # WeakSet releases when client deleted
        del client1
        assert len(managed.clients) == 1

        # Explicit removal
        managed.remove_client(client2)
        assert managed.has_clients is False


# =============================================================================
# TestRegularStrategy
# =============================================================================


class TestRegularStrategy:
    """Tests for RegularStrategy."""

    def test_create_handler(self):
        """Test handler creation with config params."""
        config = ConnectionConfig.from_uri("https://host:19530/mydb", token="mytoken")
        strategy = RegularStrategy()

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            MockHandler.return_value = Mock()
            handler = strategy.create_handler(config)

            MockHandler.assert_called_once_with(
                uri="https://host:19530/mydb", token="mytoken", db_name="mydb"
            )
            assert handler is MockHandler.return_value

    def test_close_and_on_unavailable(self):
        """Test close and on_unavailable behavior."""
        config = ConnectionConfig.from_uri("http://localhost:19530")
        strategy = RegularStrategy()
        handler = Mock()
        managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

        # on_unavailable always returns True for regular connections
        assert strategy.on_unavailable(managed) is True

        # close calls handler.close()
        strategy.close(managed)
        handler.close.assert_called_once()


# =============================================================================
# TestGlobalStrategy
# =============================================================================


class TestGlobalStrategy:
    """Tests for GlobalStrategy."""

    def test_create_handler_fetches_topology(self, sample_topology):
        """Test handler creation fetches topology, connects to primary, starts refresher."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
                MockHandler.return_value = Mock()
                with patch("pymilvus.client.connection_manager.TopologyRefresher") as MockRefresher:
                    mock_refresher = Mock()
                    MockRefresher.return_value = mock_refresher

                    strategy = GlobalStrategy()
                    handler = strategy.create_handler(config)

                    mock_fetch.assert_called_once_with(
                        "https://global-cluster.example.com:19530", "mytoken"
                    )
                    MockHandler.assert_called_once_with(
                        uri="https://primary:19530", token="mytoken", db_name=""
                    )
                    assert strategy.get_topology() is sample_topology
                    assert handler is MockHandler.return_value
                    mock_refresher.start.assert_called_once()

    def test_close_stops_refresher(self, sample_topology):
        """Test close stops the topology refresher."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
                mock_handler = Mock()
                MockHandler.return_value = mock_handler
                with patch("pymilvus.client.connection_manager.TopologyRefresher") as MockRefresher:
                    mock_refresher = Mock()
                    MockRefresher.return_value = mock_refresher

                    strategy = GlobalStrategy()
                    handler = strategy.create_handler(config)
                    managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

                    strategy.close(managed)

                    mock_refresher.stop.assert_called_once()
                    mock_handler.close.assert_called_once()

    @pytest.mark.parametrize(
        "old_endpoint,new_endpoint,expected_recovery",
        [
            ("https://old-primary:19530", "https://new-primary:19530", True),
            ("https://same-primary:19530", "https://same-primary:19530", False),
        ],
    )
    def test_on_unavailable_topology_refresh(self, old_endpoint, new_endpoint, expected_recovery):
        """Test on_unavailable refreshes topology and returns recovery status."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )

        old_topology = GlobalTopology(
            version=1,
            clusters=[ClusterInfo(cluster_id="c1", endpoint=old_endpoint, capability=0b11)],
        )
        new_topology = GlobalTopology(
            version=2,
            clusters=[ClusterInfo(cluster_id="c2", endpoint=new_endpoint, capability=0b11)],
        )

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.side_effect = [old_topology, new_topology]
            with patch("pymilvus.client.grpc_handler.GrpcHandler"):
                with patch("pymilvus.client.connection_manager.TopologyRefresher"):
                    strategy = GlobalStrategy()
                    handler = strategy.create_handler(config)
                    managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

                    result = strategy.on_unavailable(managed)

                    assert result is expected_recovery
                    assert strategy.get_topology() is new_topology


# =============================================================================
# TestConnectionManager
# =============================================================================


class TestConnectionManager:
    """Tests for ConnectionManager."""

    def test_singleton(self):
        """Test singleton pattern."""
        mgr1 = ConnectionManager.get_instance()
        mgr2 = ConnectionManager.get_instance()
        assert mgr1 is mgr2

    @pytest.mark.parametrize("dedicated", [False, True])
    def test_get_or_create_modes(self, mock_grpc_handler, dedicated):
        """Test shared vs dedicated connection modes."""
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")
        mgr = ConnectionManager.get_instance()

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            handler1, handler2 = Mock(), Mock()
            handler1._wait_for_channel_ready = Mock()
            handler2._wait_for_channel_ready = Mock()
            MockHandler.side_effect = [handler1, handler2]

            h1 = mgr.get_or_create(config, dedicated=dedicated)
            h2 = mgr.get_or_create(config, dedicated=dedicated)

            if dedicated:
                assert h1 is not h2
                assert MockHandler.call_count == 2
            else:
                assert h1 is h2
                assert MockHandler.call_count == 1

    def test_release_removes_client_reference(self, mock_grpc_handler):
        """Test release removes client from managed connection."""
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")
        mgr = ConnectionManager.get_instance()

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_grpc_handler):
            client = Mock()
            handler = mgr.get_or_create(config, client=client)

            managed = mgr._get_managed(handler)
            assert client in managed.clients

            mgr.release(handler, client=client)
            assert client not in managed.clients

    def test_close_all(self):
        """Test close_all closes all connections."""
        mgr = ConnectionManager.get_instance()
        configs = [
            ConnectionConfig.from_uri("http://host1:19530", token="t1"),
            ConnectionConfig.from_uri("http://host2:19530", token="t2"),
        ]

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            handlers = [Mock(), Mock()]
            for h in handlers:
                h._wait_for_channel_ready = Mock()
            MockHandler.side_effect = handlers

            for config in configs:
                mgr.get_or_create(config)

            mgr.close_all()

            for h in handlers:
                h.close.assert_called_once()
            assert len(mgr._registry) == 0

    def test_get_stats(self, mock_grpc_handler):
        """Test get_stats returns pool statistics."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_grpc_handler):
            mgr.get_or_create(config)

            stats = mgr.get_stats()
            assert stats["total_connections"] == 1
            assert stats["shared_connections"] == 1
            assert stats["dedicated_connections"] == 0

    def test_uses_global_strategy_for_global_endpoint(self, sample_topology, mock_grpc_handler):
        """Test GlobalStrategy is used for global-cluster URIs."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("https://global-cluster.example.com:19530", token="test")

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
                MockHandler.return_value = mock_grpc_handler
                with patch("pymilvus.client.connection_manager.TopologyRefresher"):
                    mgr.get_or_create(config)

                    mock_fetch.assert_called_once()
                    assert MockHandler.call_args.kwargs["uri"] == "https://primary:19530"


# =============================================================================
# TestAsyncGlobalStrategy
# =============================================================================


class TestAsyncGlobalStrategy:
    """Tests for AsyncGlobalStrategy."""

    def test_create_handler_fetches_topology_and_starts_refresher(self, sample_topology):
        """Test async handler creation fetches topology and starts refresher."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
                MockHandler.return_value = Mock()
                with patch("pymilvus.client.connection_manager.TopologyRefresher") as MockRefresher:
                    mock_refresher = Mock()
                    MockRefresher.return_value = mock_refresher

                    strategy = AsyncGlobalStrategy()
                    handler = strategy.create_handler(config)

                    mock_fetch.assert_called_once()
                    MockHandler.assert_called_once_with(
                        uri="https://primary:19530", token="mytoken", db_name=""
                    )
                    mock_refresher.start.assert_called_once()
                    assert handler is MockHandler.return_value

    @pytest.mark.asyncio
    async def test_close_async_stops_refresher(self, sample_topology):
        """Test close_async stops refresher and closes handler."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
                mock_handler = Mock()
                mock_handler.close = AsyncMock()
                MockHandler.return_value = mock_handler
                with patch("pymilvus.client.connection_manager.TopologyRefresher") as MockRefresher:
                    mock_refresher = Mock()
                    MockRefresher.return_value = mock_refresher

                    strategy = AsyncGlobalStrategy()
                    handler = strategy.create_handler(config)
                    managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

                    await strategy.close_async(managed)

                    mock_refresher.stop.assert_called_once()
                    mock_handler.close.assert_awaited_once()


# =============================================================================
# TestAsyncConnectionManager
# =============================================================================


class TestAsyncConnectionManager:
    """Tests for AsyncConnectionManager."""

    @pytest.mark.asyncio
    async def test_singleton(self):
        """Test singleton pattern."""
        mgr1 = AsyncConnectionManager.get_instance()
        mgr2 = AsyncConnectionManager.get_instance()
        assert mgr1 is mgr2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dedicated", [False, True])
    async def test_get_or_create_modes(self, mock_async_handler, dedicated):
        """Test shared vs dedicated connection modes."""
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")
        mgr = AsyncConnectionManager.get_instance()

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            handler1, handler2 = Mock(), Mock()
            handler1.ensure_channel_ready = AsyncMock()
            handler2.ensure_channel_ready = AsyncMock()
            MockHandler.side_effect = [handler1, handler2]

            h1 = await mgr.get_or_create(config, dedicated=dedicated)
            h2 = await mgr.get_or_create(config, dedicated=dedicated)

            if dedicated:
                assert h1 is not h2
            else:
                assert h1 is h2

    @pytest.mark.asyncio
    async def test_release(self, mock_async_handler):
        """Test release removes client reference."""
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")
        mgr = AsyncConnectionManager.get_instance()

        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_async_handler
        ):
            client = Mock()
            handler = await mgr.get_or_create(config, client=client)

            managed = mgr._get_managed(handler)
            assert client in managed.clients

            await mgr.release(handler, client=client)
            assert client not in managed.clients


# =============================================================================
# TestMilvusClientIntegration
# =============================================================================


class TestMilvusClientIntegration:
    """Integration tests for MilvusClient with ConnectionManager."""

    def test_uses_connection_manager(self, mock_grpc_handler):
        """Test MilvusClient uses ConnectionManager."""
        with patch.object(ConnectionManager, "get_instance") as mock_get_instance:
            mock_manager = Mock()
            mock_manager.get_or_create.return_value = mock_grpc_handler
            mock_get_instance.return_value = mock_manager

            client = MilvusClient(uri="http://localhost:19530")

            mock_manager.get_or_create.assert_called_once()
            config = mock_manager.get_or_create.call_args.args[0]
            assert config.address == "localhost:19530"

            client.close()
            mock_manager.release.assert_called_once_with(mock_grpc_handler, client=client)

    @pytest.mark.parametrize(
        "scenario,uri1,token1,uri2,token2,expect_same_handler",
        [
            (
                "same_config",
                "http://localhost:19530",
                "test",
                "http://localhost:19530",
                "test",
                True,
            ),
            (
                "different_tokens",
                "http://localhost:19530",
                "token1",
                "http://localhost:19530",
                "token2",
                False,
            ),
        ],
    )
    def test_connection_sharing(self, scenario, uri1, token1, uri2, token2, expect_same_handler):
        """Test connection sharing behavior based on config."""
        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            handler1, handler2 = Mock(), Mock()
            for h in [handler1, handler2]:
                h._wait_for_channel_ready = Mock()
                h.get_server_type = Mock(return_value="milvus")
            MockHandler.side_effect = [handler1, handler2]

            client1 = MilvusClient(uri=uri1, token=token1)
            client2 = MilvusClient(uri=uri2, token=token2)

            if expect_same_handler:
                assert client1._handler is client2._handler
                assert MockHandler.call_count == 1
            else:
                assert client1._handler is not client2._handler
                assert MockHandler.call_count == 2

            client1.close()
            client2.close()

    def test_dedicated_mode(self):
        """Test dedicated mode creates separate connections."""
        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            handler1, handler2 = Mock(), Mock()
            for h in [handler1, handler2]:
                h._wait_for_channel_ready = Mock()
                h.get_server_type = Mock(return_value="milvus")
            MockHandler.side_effect = [handler1, handler2]

            client1 = MilvusClient(uri="http://localhost:19530", token="test", dedicated=True)
            client2 = MilvusClient(uri="http://localhost:19530", token="test", dedicated=True)

            assert client1._handler is not client2._handler

            client1.close()
            client2.close()

    def test_global_endpoint_uses_global_strategy(self, sample_topology):
        """Test global cluster URIs use GlobalStrategy."""
        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
                mock_handler = Mock()
                mock_handler._wait_for_channel_ready = Mock()
                mock_handler.get_server_type = Mock(return_value="zilliz")
                MockHandler.return_value = mock_handler
                with patch("pymilvus.client.connection_manager.TopologyRefresher"):
                    client = MilvusClient(
                        uri="https://global-cluster.zilliz.com:19530", token="test"
                    )

                    mock_fetch.assert_called_once()
                    assert MockHandler.call_args.kwargs["uri"] == "https://primary:19530"

                    client.close()

    def test_stats_reflect_connections(self):
        """Test get_stats returns accurate pool information."""
        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler._wait_for_channel_ready = Mock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            MockHandler.return_value = mock_handler

            mgr = ConnectionManager.get_instance()
            assert mgr.get_stats()["total_connections"] == 0

            client = MilvusClient(uri="http://localhost:19530", token="test")

            stats = mgr.get_stats()
            assert stats["total_connections"] == 1
            assert stats["shared_connections"] == 1

            client.close()


# =============================================================================
# TestAsyncMilvusClientIntegration
# =============================================================================


class TestAsyncMilvusClientIntegration:
    """Integration tests for AsyncMilvusClient with AsyncConnectionManager."""

    @pytest.mark.asyncio
    async def test_uses_async_connection_manager(self):
        """Test AsyncMilvusClient uses AsyncConnectionManager."""
        with patch.object(AsyncConnectionManager, "get_instance") as mock_get_instance:
            mock_manager = Mock()
            mock_handler = Mock()
            mock_handler.get_server_type = AsyncMock(return_value="milvus")

            async def mock_get_or_create(*args, **kwargs):
                return mock_handler

            mock_manager.get_or_create = mock_get_or_create
            mock_manager.release = AsyncMock()
            mock_get_instance.return_value = mock_manager

            client = AsyncMilvusClient(uri="http://localhost:19530")
            await client._connect()

            mock_get_instance.assert_called()

            await client.close()
            mock_manager.release.assert_called_once()


# =============================================================================
# TestErrorHandling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.parametrize(
        "error_code,expect_recovery",
        [
            (grpc.StatusCode.UNAVAILABLE, True),
            (grpc.StatusCode.INVALID_ARGUMENT, False),
            (grpc.StatusCode.NOT_FOUND, False),
        ],
    )
    def test_handle_error(self, error_code, expect_recovery):
        """Test error handling triggers recovery only for UNAVAILABLE."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler._wait_for_channel_ready = Mock()
            mock_handler.close = Mock()
            MockHandler.return_value = mock_handler

            handler = mgr.get_or_create(config)

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return error_code

            new_handler = Mock()
            new_handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = new_handler

            mgr.handle_error(handler, MockRpcError())

            if expect_recovery:
                mock_handler.close.assert_called()
                assert MockHandler.call_count >= 2
            else:
                mock_handler.close.assert_not_called()

    def test_error_callback_registered_on_handler(self):
        """Test that _on_rpc_error callback is set on handler after creation."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock(spec=[])  # no spec attrs, so _on_rpc_error must be set
            mock_handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = mock_handler

            handler = mgr.get_or_create(config)

            # Verify callback was registered
            assert hasattr(handler, "_on_rpc_error")
            assert callable(handler._on_rpc_error)

    def test_error_callback_routes_to_handle_error(self):
        """Test that _on_rpc_error callback calls handle_error on the manager."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock(spec=[])
            mock_handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = mock_handler

            handler = mgr.get_or_create(config)

            with patch.object(mgr, "handle_error") as mock_handle:
                error = Mock(spec=grpc.RpcError)
                handler._on_rpc_error(error)
                mock_handle.assert_called_once_with(handler, error)

    def test_error_callback_re_registered_after_recovery(self):
        """Test that _on_rpc_error is re-registered on new handler after recovery."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler._wait_for_channel_ready = Mock()
            old_handler.close = Mock()
            MockHandler.return_value = old_handler

            handler = mgr.get_or_create(config)

            # Trigger recovery
            new_handler = Mock(spec=[])
            new_handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = new_handler

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return grpc.StatusCode.UNAVAILABLE

            mgr.handle_error(handler, MockRpcError())

            # The new handler should also have the callback
            managed = next(iter(mgr._registry.values()))
            assert hasattr(managed.handler, "_on_rpc_error")
            assert callable(managed.handler._on_rpc_error)

    @pytest.mark.asyncio
    async def test_async_error_callback_registered(self):
        """Test that _on_rpc_error callback is set on async handler after creation."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock(spec=[])
            mock_handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = mock_handler

            handler = await mgr.get_or_create(config)

            assert hasattr(handler, "_on_rpc_error")
            assert callable(handler._on_rpc_error)

    @pytest.mark.asyncio
    async def test_async_error_callback_is_async(self):
        """Test that _on_rpc_error callback is awaitable (async)."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock(spec=[])
            mock_handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = mock_handler

            handler = await mgr.get_or_create(config)

            # The callback should be a coroutine function
            assert asyncio.iscoroutinefunction(handler._on_rpc_error)

    @pytest.mark.asyncio
    async def test_async_handle_error_unavailable(self):
        """Test async handle_error triggers recovery for UNAVAILABLE."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler.ensure_channel_ready = AsyncMock()
            old_handler.close = AsyncMock()
            MockHandler.return_value = old_handler

            handler = await mgr.get_or_create(config)

            new_handler = Mock(spec=[])
            new_handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = new_handler

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return grpc.StatusCode.UNAVAILABLE

            await mgr.handle_error(handler, MockRpcError())

            # Should have created a new handler
            managed = next(iter(mgr._registry.values()))
            assert managed.handler is new_handler
            assert hasattr(managed.handler, "_on_rpc_error")

    @pytest.mark.asyncio
    async def test_async_recover_calls_ensure_channel_ready(self):
        """Test that async _recover awaits ensure_channel_ready on new handler."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler.ensure_channel_ready = AsyncMock()
            old_handler.close = AsyncMock()
            MockHandler.return_value = old_handler

            handler = await mgr.get_or_create(config)

            new_handler = Mock(spec=[])
            new_handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = new_handler

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return grpc.StatusCode.UNAVAILABLE

            await mgr.handle_error(handler, MockRpcError())

            # ensure_channel_ready should be called on the new handler during recovery
            new_handler.ensure_channel_ready.assert_called_once()


# =============================================================================
# TestHealthCheck
# =============================================================================


class TestHealthCheck:
    """Tests for health check on idle connections."""

    def test_health_check_triggers_on_idle_connection(self):
        """Test that an idle shared connection triggers health check and recovery."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            old_handler = Mock()
            old_handler._wait_for_channel_ready = Mock()
            old_handler.close = Mock()
            MockHandler.return_value = old_handler

            # Create shared connection
            h1 = mgr.get_or_create(config)
            assert h1 is old_handler

            # Artificially make the connection idle beyond the threshold
            managed = mgr._get_managed(h1)
            managed.last_used_at = time.time() - IDLE_THRESHOLD_SECONDS - 1

            # Prepare a new handler for recovery and make health check fail
            new_handler = Mock()
            new_handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = new_handler

            with patch.object(mgr, "_check_health", return_value=False):
                h2 = mgr.get_or_create(config)

            # Should have recovered: new handler returned
            assert h2 is new_handler
            assert managed.handler is new_handler

    def test_no_health_check_when_recently_used(self):
        """Test that a recently-used connection skips health check."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            handler = Mock()
            handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = handler

            mgr.get_or_create(config)

            with patch.object(mgr, "_check_health") as mock_check:
                mgr.get_or_create(config)
                # Health check should NOT be called since idle_time < threshold
                mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_health_check_triggers_on_idle_connection(self):
        """Test that an idle async shared connection triggers health check and recovery."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler.ensure_channel_ready = AsyncMock()
            old_handler.close = AsyncMock()
            MockHandler.return_value = old_handler

            h1 = await mgr.get_or_create(config)
            assert h1 is old_handler

            # Make the connection idle
            managed = mgr._get_managed(h1)
            managed.last_used_at = time.time() - IDLE_THRESHOLD_SECONDS - 1

            new_handler = Mock(spec=[])
            new_handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = new_handler

            # Replace _check_health with an async function that returns False
            async def _failing_health_check(m):
                return False

            mgr._check_health = _failing_health_check
            h2 = await mgr.get_or_create(config)

            assert h2 is new_handler
            assert managed.handler is new_handler


# =============================================================================
# TestDedicatedRecovery
# =============================================================================


class TestDedicatedRecovery:
    """Tests for dedicated connection recovery key updates."""

    def test_dedicated_connection_findable_after_recovery(self):
        """Test that after recovery, the new handler is still findable in _dedicated."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler._wait_for_channel_ready = Mock()
            old_handler.close = Mock()
            MockHandler.return_value = old_handler

            handler = mgr.get_or_create(config, dedicated=True)
            assert id(handler) in mgr._dedicated

            # Trigger UNAVAILABLE recovery
            new_handler = Mock(spec=[])
            new_handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = new_handler

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return grpc.StatusCode.UNAVAILABLE

            mgr.handle_error(handler, MockRpcError())

            # Old key gone, new key present
            assert id(old_handler) not in mgr._dedicated
            assert id(new_handler) in mgr._dedicated

            # New handler is findable via _get_managed
            assert mgr._get_managed(new_handler) is not None
            assert mgr._get_managed(new_handler).handler is new_handler

    def test_dedicated_connection_releasable_after_recovery(self):
        """Test that a recovered dedicated connection can be released."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler._wait_for_channel_ready = Mock()
            old_handler.close = Mock()
            MockHandler.return_value = old_handler

            handler = mgr.get_or_create(config, dedicated=True)

            new_handler = Mock(spec=[])
            new_handler._wait_for_channel_ready = Mock()
            new_handler.close = Mock()
            MockHandler.return_value = new_handler

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return grpc.StatusCode.UNAVAILABLE

            mgr.handle_error(handler, MockRpcError())

            managed = mgr._get_managed(new_handler)
            assert managed is not None

            # Release the recovered connection
            mgr.release(new_handler)
            assert id(new_handler) not in mgr._dedicated

    @pytest.mark.asyncio
    async def test_async_dedicated_connection_findable_after_recovery(self):
        """Test async dedicated connection key update after recovery."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler.ensure_channel_ready = AsyncMock()
            old_handler.close = AsyncMock()
            MockHandler.return_value = old_handler

            handler = await mgr.get_or_create(config, dedicated=True)
            assert id(handler) in mgr._dedicated

            new_handler = Mock(spec=[])
            new_handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = new_handler

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return grpc.StatusCode.UNAVAILABLE

            await mgr.handle_error(handler, MockRpcError())

            assert id(old_handler) not in mgr._dedicated
            assert id(new_handler) in mgr._dedicated
            assert mgr._get_managed(new_handler).handler is new_handler


# =============================================================================
# TestDoubleClose
# =============================================================================


class TestDoubleClose:
    """Tests for safe double-close behavior."""

    def test_sync_client_double_close_is_safe(self):
        """Test that calling close() twice on MilvusClient doesn't raise."""
        with patch.object(ConnectionManager, "get_instance") as mock_get_instance:
            mock_manager = Mock()
            mock_handler = Mock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            mock_manager.get_or_create.return_value = mock_handler
            mock_get_instance.return_value = mock_manager

            client = MilvusClient(uri="http://localhost:19530")
            client.close()
            client.close()  # Should not raise

            # release should only be called once
            mock_manager.release.assert_called_once()


# =============================================================================
# TestGetStatsWithDedicated
# =============================================================================


class TestGetStatsWithDedicated:
    """Tests for get_stats including dedicated connections."""

    def test_stats_include_dedicated_connections(self):
        """Test that get_stats includes details for dedicated connections."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            handler1, handler2 = Mock(), Mock()
            for h in [handler1, handler2]:
                h._wait_for_channel_ready = Mock()
            MockHandler.side_effect = [handler1, handler2]

            mgr.get_or_create(config)
            mgr.get_or_create(config, dedicated=True)

            stats = mgr.get_stats()
            assert stats["total_connections"] == 2
            assert stats["shared_connections"] == 1
            assert stats["dedicated_connections"] == 1
            assert len(stats["connections"]) == 2

            shared = [c for c in stats["connections"] if not c["dedicated"]]
            dedicated = [c for c in stats["connections"] if c["dedicated"]]
            assert len(shared) == 1
            assert len(dedicated) == 1
            assert dedicated[0]["address"] == "localhost:19530"


# =============================================================================
# TestConnectionConfigInvalidUri
# =============================================================================


class TestConnectionConfigInvalidUri:
    """Tests for invalid URI rejection."""

    def test_invalid_scheme_raises(self):
        """Test that an unsupported URI scheme raises ConnectionConfigException."""
        with pytest.raises(ConnectionConfigException, match="is illegal"):
            ConnectionConfig.from_uri("ftp://host:19530")

    def test_local_db_skips_scheme_check(self):
        """Test that .db files skip scheme validation."""
        config = ConnectionConfig.from_uri("mydata.db")
        assert config.address is not None


# =============================================================================
# TestConnectionStrategyCloseAsyncDefault
# =============================================================================


class TestConnectionStrategyCloseAsyncDefault:
    """Test default close_async delegates to close."""

    @pytest.mark.asyncio
    async def test_default_close_async_delegates_to_close(self):
        """Test base class close_async calls sync close."""
        strategy = RegularStrategy()
        handler = Mock()
        handler.close = Mock()
        config = ConnectionConfig.from_uri("http://localhost:19530")
        managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

        # RegularStrategy doesn't override close_async, so base class is used
        await strategy.close_async(managed)
        handler.close.assert_called_once()


# =============================================================================
# TestGlobalStrategyMixinEdgeCases
# =============================================================================


class TestGlobalStrategyMixinEdgeCases:
    """Tests for _GlobalStrategyMixin edge cases."""

    def test_on_topology_change_callback(self, sample_topology):
        """Test _on_topology_change updates topology."""
        mixin = _GlobalStrategyMixin()
        assert mixin.get_topology() is None

        mixin._on_topology_change(sample_topology)
        assert mixin.get_topology() is sample_topology

    def test_on_unavailable_no_config(self):
        """Test on_unavailable returns True when _config is None."""
        mixin = _GlobalStrategyMixin()
        managed = Mock()
        assert mixin.on_unavailable(managed) is True

    def test_on_unavailable_fetch_fails(self, sample_topology):
        """Test on_unavailable returns True when fetch_topology raises."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )
        strategy = GlobalStrategy()

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.side_effect = [sample_topology, RuntimeError("network error")]
            with patch("pymilvus.client.grpc_handler.GrpcHandler"):
                with patch("pymilvus.client.connection_manager.TopologyRefresher"):
                    handler = strategy.create_handler(config)
                    managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

                    result = strategy.on_unavailable(managed)
                    assert result is True


# =============================================================================
# TestSyncCheckHealthActual
# =============================================================================


class TestSyncCheckHealthActual:
    """Tests for _check_health actual execution (not mocked)."""

    def test_healthy_connection(self):
        """Test _check_health returns True for healthy connection."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock()
        channel = Mock()
        state = Mock()
        state.name = "READY"
        channel.check_connectivity_state.return_value = state
        handler._channel = channel
        handler.get_server_version = Mock(return_value="v2.0")

        managed = ManagedConnection(handler=handler, config=config, strategy=RegularStrategy())
        assert mgr._check_health(managed) is True

    def test_no_channel(self):
        """Test _check_health returns False when no channel."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock(spec=[])  # no _channel attribute
        managed = ManagedConnection(handler=handler, config=config, strategy=RegularStrategy())
        assert mgr._check_health(managed) is False

    def test_shutdown_channel(self):
        """Test _check_health returns False when channel is SHUTDOWN."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock()
        channel = Mock()
        state = Mock()
        state.name = "SHUTDOWN"
        channel.check_connectivity_state.return_value = state
        handler._channel = channel

        managed = ManagedConnection(handler=handler, config=config, strategy=RegularStrategy())
        assert mgr._check_health(managed) is False

    def test_health_probe_exception(self):
        """Test _check_health returns False when get_server_version raises."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock()
        channel = Mock()
        state = Mock()
        state.name = "READY"
        channel.check_connectivity_state.return_value = state
        handler._channel = channel
        handler.get_server_version = Mock(side_effect=Exception("timeout"))

        managed = ManagedConnection(handler=handler, config=config, strategy=RegularStrategy())
        assert mgr._check_health(managed) is False


# =============================================================================
# TestSyncRecoverFailure
# =============================================================================


class TestSyncRecoverFailure:
    """Tests for sync _recover exception path."""

    def test_recover_raises_on_failure(self):
        """Test _recover logs and re-raises when handler creation fails."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            old_handler = Mock()
            old_handler._wait_for_channel_ready = Mock()
            old_handler.close = Mock()
            MockHandler.return_value = old_handler

            handler = mgr.get_or_create(config)
            managed = mgr._get_managed(handler)

            # Make strategy.create_handler fail
            MockHandler.side_effect = RuntimeError("cannot connect")

            with pytest.raises(RuntimeError, match="cannot connect"):
                mgr._recover(managed)


# =============================================================================
# TestSyncHandleErrorEdgeCases
# =============================================================================


class TestSyncHandleErrorEdgeCases:
    """Tests for handle_error edge cases."""

    def test_non_rpc_error_ignored(self):
        """Test handle_error ignores non-RpcError exceptions."""
        mgr = ConnectionManager.get_instance()
        handler = Mock()

        # Should not raise, just return
        mgr.handle_error(handler, ValueError("some error"))

    def test_managed_not_found(self):
        """Test handle_error returns when handler not in registry."""
        mgr = ConnectionManager.get_instance()
        handler = Mock()

        class MockRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

        # Handler not registered - should not raise
        mgr.handle_error(handler, MockRpcError())

    def test_handler_already_recovered(self):
        """Test handle_error skips recovery if handler already replaced."""
        mgr = ConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            old_handler = Mock()
            old_handler._wait_for_channel_ready = Mock()
            MockHandler.return_value = old_handler

            handler = mgr.get_or_create(config)
            managed = mgr._get_managed(handler)

            class MockRpcError(grpc.RpcError):
                def code(self):
                    return grpc.StatusCode.UNAVAILABLE

            # Simulate another thread recovering between lock release and re-acquire
            new_handler = Mock()
            new_handler._wait_for_channel_ready = Mock()
            original_on_unavailable = managed.strategy.on_unavailable

            def swap_handler_then_return_true(m):
                original_on_unavailable(m)
                managed.handler = new_handler  # simulate concurrent recovery
                return True

            managed.strategy.on_unavailable = swap_handler_then_return_true

            mgr.handle_error(handler, MockRpcError())

            # The handler should be new_handler (set by our swap), not re-recovered
            assert managed.handler is new_handler


# =============================================================================
# TestSyncGetManagedReturnsNone
# =============================================================================


class TestSyncGetManagedReturnsNone:
    """Test _get_managed returns None for unknown handler."""

    def test_unknown_handler_returns_none(self):
        """Test _get_managed returns None when handler is not in any registry."""
        mgr = ConnectionManager.get_instance()
        unknown_handler = Mock()
        assert mgr._get_managed(unknown_handler) is None


# =============================================================================
# TestAsyncRegularStrategySyncClose
# =============================================================================


class TestAsyncRegularStrategySyncClose:
    """Tests for AsyncRegularStrategy.close (sync fallback)."""

    @pytest.mark.asyncio
    async def test_close_with_running_loop(self):
        """Test sync close creates fire-and-forget task when loop is running."""
        strategy = AsyncRegularStrategy()
        handler = Mock()
        handler.close = AsyncMock()
        config = ConnectionConfig.from_uri("http://localhost:19530")
        managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

        strategy.close(managed)
        # Give the event loop a chance to execute the task
        await asyncio.sleep(0)
        handler.close.assert_awaited_once()


# =============================================================================
# TestAsyncGlobalStrategySyncClose
# =============================================================================


class TestAsyncGlobalStrategySyncClose:
    """Tests for AsyncGlobalStrategy.close (sync fallback)."""

    @pytest.mark.asyncio
    async def test_close_with_running_loop(self, sample_topology):
        """Test sync close stops refresher and creates fire-and-forget task."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
                mock_handler = Mock()
                mock_handler.close = AsyncMock()
                MockHandler.return_value = mock_handler
                with patch("pymilvus.client.connection_manager.TopologyRefresher") as MockRefresher:
                    mock_refresher = Mock()
                    MockRefresher.return_value = mock_refresher

                    strategy = AsyncGlobalStrategy()
                    handler = strategy.create_handler(config)
                    managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

                    strategy.close(managed)
                    await asyncio.sleep(0)

                    mock_refresher.stop.assert_called_once()
                    mock_handler.close.assert_awaited_once()

    def test_close_without_running_loop(self, sample_topology):
        """Test sync close falls back to asyncio.run when no loop is running."""
        config = ConnectionConfig.from_uri(
            "https://global-cluster.example.com:19530", token="mytoken"
        )

        with patch("pymilvus.client.connection_manager.fetch_topology") as mock_fetch:
            mock_fetch.return_value = sample_topology
            with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
                mock_handler = Mock()
                close_called = []

                async def mock_close():
                    close_called.append(True)

                mock_handler.close = mock_close
                MockHandler.return_value = mock_handler
                with patch("pymilvus.client.connection_manager.TopologyRefresher") as MockRefresher:
                    mock_refresher = Mock()
                    MockRefresher.return_value = mock_refresher

                    strategy = AsyncGlobalStrategy()
                    handler = strategy.create_handler(config)
                    managed = ManagedConnection(handler=handler, config=config, strategy=strategy)

                    strategy.close(managed)

                    mock_refresher.stop.assert_called_once()
                    assert len(close_called) == 1


# =============================================================================
# TestAsyncConnectionManagerEdgeCases
# =============================================================================


class TestAsyncConnectionManagerEdgeCases:
    """Tests for AsyncConnectionManager edge cases."""

    def test_get_instance_without_running_loop(self):
        """Test get_instance uses loop_id=0 when no event loop running."""
        mgr = AsyncConnectionManager.get_instance()
        assert mgr is not None
        # Should be stored under key 0
        assert 0 in AsyncConnectionManager._instances

    @pytest.mark.asyncio
    async def test_get_or_create_adds_client_to_existing(self):
        """Test that get_or_create adds client ref to existing shared connection."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            handler = Mock(spec=[])
            handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = handler

            client1 = Mock()
            client2 = Mock()

            await mgr.get_or_create(config, client=client1)
            await mgr.get_or_create(config, client=client2)

            managed = mgr._get_managed(handler)
            assert client1 in managed.clients
            assert client2 in managed.clients

    @pytest.mark.asyncio
    async def test_create_dedicated_with_client(self):
        """Test _create_dedicated stores client reference."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            handler = Mock(spec=[])
            handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = handler

            client = Mock()
            h = await mgr.get_or_create(config, dedicated=True, client=client)

            managed = mgr._get_managed(h)
            assert client in managed.clients

    @pytest.mark.asyncio
    async def test_get_strategy_returns_async_global(self):
        """Test _get_strategy returns AsyncGlobalStrategy for global endpoint."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("https://global-cluster.example.com:19530", token="test")
        strategy = mgr._get_strategy(config)
        assert isinstance(strategy, AsyncGlobalStrategy)

    @pytest.mark.asyncio
    async def test_get_managed_returns_none_unknown_handler(self):
        """Test _get_managed returns None for unknown handler."""
        mgr = AsyncConnectionManager.get_instance()
        unknown_handler = Mock()
        assert mgr._get_managed(unknown_handler) is None

    @pytest.mark.asyncio
    async def test_error_callback_routes_to_handle_error(self):
        """Test that async _on_rpc_error callback calls handle_error."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock(spec=[])
            mock_handler.ensure_channel_ready = AsyncMock()
            MockHandler.return_value = mock_handler

            handler = await mgr.get_or_create(config)

            with patch.object(mgr, "handle_error", new_callable=AsyncMock) as mock_handle:
                error = Mock(spec=grpc.RpcError)
                await handler._on_rpc_error(error)
                mock_handle.assert_called_once_with(handler, error)


# =============================================================================
# TestAsyncCheckHealthActual
# =============================================================================


class TestAsyncCheckHealthActual:
    """Tests for async _check_health actual execution."""

    @pytest.mark.asyncio
    async def test_healthy_connection(self):
        """Test _check_health returns True for healthy async connection."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock()
        channel = Mock()
        state = Mock()
        state.name = "READY"
        channel.check_connectivity_state.return_value = state
        handler._channel = channel
        handler.get_server_version = AsyncMock(return_value="v2.0")

        managed = ManagedConnection(handler=handler, config=config, strategy=AsyncRegularStrategy())
        assert await mgr._check_health(managed) is True

    @pytest.mark.asyncio
    async def test_no_channel(self):
        """Test _check_health returns False when no channel."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock(spec=[])  # no _channel attribute
        managed = ManagedConnection(handler=handler, config=config, strategy=AsyncRegularStrategy())
        assert await mgr._check_health(managed) is False

    @pytest.mark.asyncio
    async def test_shutdown_channel(self):
        """Test _check_health returns False when channel is SHUTDOWN."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock()
        channel = Mock()
        state = Mock()
        state.name = "SHUTDOWN"
        channel.check_connectivity_state.return_value = state
        handler._channel = channel

        managed = ManagedConnection(handler=handler, config=config, strategy=AsyncRegularStrategy())
        assert await mgr._check_health(managed) is False

    @pytest.mark.asyncio
    async def test_health_probe_exception(self):
        """Test _check_health returns False when get_server_version raises."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        handler = Mock()
        channel = Mock()
        state = Mock()
        state.name = "READY"
        channel.check_connectivity_state.return_value = state
        handler._channel = channel
        handler.get_server_version = AsyncMock(side_effect=Exception("timeout"))

        managed = ManagedConnection(handler=handler, config=config, strategy=AsyncRegularStrategy())
        assert await mgr._check_health(managed) is False


# =============================================================================
# TestAsyncReleaseDedicated
# =============================================================================


class TestAsyncReleaseDedicated:
    """Test async release of dedicated connections."""

    @pytest.mark.asyncio
    async def test_release_closes_dedicated(self):
        """Test that releasing a dedicated async connection closes it."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            handler = Mock(spec=[])
            handler.ensure_channel_ready = AsyncMock()
            handler.close = AsyncMock()
            MockHandler.return_value = handler

            h = await mgr.get_or_create(config, dedicated=True)
            assert id(h) in mgr._dedicated

            await mgr.release(h)
            assert id(h) not in mgr._dedicated
            handler.close.assert_awaited_once()


# =============================================================================
# TestAsyncHandleErrorEdgeCases
# =============================================================================


class TestAsyncHandleErrorEdgeCases:
    """Tests for async handle_error edge cases."""

    @pytest.mark.asyncio
    async def test_non_rpc_error_ignored(self):
        """Test async handle_error ignores non-RpcError exceptions."""
        mgr = AsyncConnectionManager.get_instance()
        handler = Mock()

        # Should not raise, just return
        await mgr.handle_error(handler, ValueError("some error"))

    @pytest.mark.asyncio
    async def test_non_unavailable_ignored(self):
        """Test async handle_error ignores non-UNAVAILABLE RPC errors."""
        mgr = AsyncConnectionManager.get_instance()
        handler = Mock()

        class MockRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.NOT_FOUND

        await mgr.handle_error(handler, MockRpcError())

    @pytest.mark.asyncio
    async def test_managed_not_found(self):
        """Test async handle_error returns when handler not in registry."""
        mgr = AsyncConnectionManager.get_instance()
        handler = Mock()

        class MockRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

        # Handler not registered - should not raise
        await mgr.handle_error(handler, MockRpcError())


# =============================================================================
# TestAsyncRecoverFailure
# =============================================================================


class TestAsyncRecoverFailure:
    """Tests for async _recover exception path."""

    @pytest.mark.asyncio
    async def test_recover_raises_on_failure(self):
        """Test async _recover logs and re-raises when handler creation fails."""
        mgr = AsyncConnectionManager.get_instance()
        config = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            old_handler = Mock(spec=[])
            old_handler.ensure_channel_ready = AsyncMock()
            old_handler.close = AsyncMock()
            MockHandler.return_value = old_handler

            handler = await mgr.get_or_create(config)
            managed = mgr._get_managed(handler)

            # Make strategy.create_handler fail
            MockHandler.side_effect = RuntimeError("cannot connect")

            with pytest.raises(RuntimeError, match="cannot connect"):
                await mgr._recover(managed)


# =============================================================================
# TestAsyncCloseAll
# =============================================================================


class TestAsyncCloseAll:
    """Tests for async close_all."""

    @pytest.mark.asyncio
    async def test_close_all_closes_shared_and_dedicated(self):
        """Test close_all closes both shared and dedicated async connections."""
        mgr = AsyncConnectionManager.get_instance()
        config1 = ConnectionConfig.from_uri("http://host1:19530", token="t1")
        config2 = ConnectionConfig.from_uri("http://host2:19530", token="t2")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            handlers = []
            for _ in range(3):
                h = Mock(spec=[])
                h.ensure_channel_ready = AsyncMock()
                h.close = AsyncMock()
                handlers.append(h)
            MockHandler.side_effect = handlers

            await mgr.get_or_create(config1)
            await mgr.get_or_create(config2)
            await mgr.get_or_create(config1, dedicated=True)

            assert len(mgr._registry) == 2
            assert len(mgr._dedicated) == 1

            await mgr.close_all()

            assert len(mgr._registry) == 0
            assert len(mgr._dedicated) == 0
            for h in handlers:
                h.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_all_handles_errors(self):
        """Test close_all continues despite close errors on both shared and dedicated."""
        mgr = AsyncConnectionManager.get_instance()
        config1 = ConnectionConfig.from_uri("http://localhost:19530", token="test")
        config2 = ConnectionConfig.from_uri("http://localhost:19530", token="test")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            shared_handler = Mock(spec=[])
            shared_handler.ensure_channel_ready = AsyncMock()
            shared_handler.close = AsyncMock(side_effect=RuntimeError("close failed"))
            dedicated_handler = Mock(spec=[])
            dedicated_handler.ensure_channel_ready = AsyncMock()
            dedicated_handler.close = AsyncMock(side_effect=RuntimeError("close failed"))
            MockHandler.side_effect = [shared_handler, dedicated_handler]

            await mgr.get_or_create(config1)
            await mgr.get_or_create(config2, dedicated=True)

            assert len(mgr._registry) == 1
            assert len(mgr._dedicated) == 1

            # Should not raise despite close errors on both shared and dedicated
            await mgr.close_all()
            assert len(mgr._registry) == 0
            assert len(mgr._dedicated) == 0


# =============================================================================
# TestMilvusClientChangedCode
# =============================================================================


class TestMilvusClientChangedCode:
    """Tests for changed code in MilvusClient (sync)."""

    def test_user_password_token_construction(self):
        """Test that user/password are combined into token when no token given."""
        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler._wait_for_channel_ready = Mock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            MockHandler.return_value = mock_handler

            client = MilvusClient(uri="http://localhost:19530", user="admin", password="secret")

            assert client._config.token == "admin:secret"
            client.close()

    def test_explicit_token_overrides_user_password(self):
        """Test that explicit token takes precedence over user/password."""
        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler._wait_for_channel_ready = Mock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            MockHandler.return_value = mock_handler

            client = MilvusClient(
                uri="http://localhost:19530",
                token="my_token",
                user="admin",
                password="secret",
            )

            assert client._config.token == "my_token"
            client.close()

    def test_get_connection_returns_handler(self):
        """Test _get_connection returns the handler."""
        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler._wait_for_channel_ready = Mock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            MockHandler.return_value = mock_handler

            client = MilvusClient(uri="http://localhost:19530")
            assert client._get_connection() is mock_handler
            client.close()

    def test_use_database_updates_config(self):
        """Test use_database updates _config.db_name."""
        with patch("pymilvus.client.grpc_handler.GrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler._wait_for_channel_ready = Mock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            mock_handler.describe_database = Mock(return_value={})
            MockHandler.return_value = mock_handler

            client = MilvusClient(uri="http://localhost:19530")
            assert client._config.db_name == ""

            client.use_database("mydb")
            assert client._config.db_name == "mydb"
            client.close()

    def test_dedicated_kwarg(self):
        """Test that dedicated kwarg is consumed and passed to get_or_create."""
        with patch.object(ConnectionManager, "get_instance") as mock_get_instance:
            mock_manager = Mock()
            mock_handler = Mock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            mock_manager.get_or_create.return_value = mock_handler
            mock_get_instance.return_value = mock_manager

            client = MilvusClient(uri="http://localhost:19530", dedicated=True)

            call_kwargs = mock_manager.get_or_create.call_args
            assert call_kwargs.kwargs.get("dedicated") is True
            client.close()


# =============================================================================
# TestAsyncMilvusClientChangedCode
# =============================================================================


class TestAsyncMilvusClientChangedCode:
    """Tests for changed code in AsyncMilvusClient."""

    def test_init_deferred_state(self):
        """Test __init__ sets deferred state without connecting."""
        client = AsyncMilvusClient(uri="http://localhost:19530", token="test")
        assert client._handler is None
        assert client._manager is None
        assert client._using is None
        assert client.is_self_hosted is None
        assert client._closed is False
        assert client._config.address == "localhost:19530"
        assert client._config.token == "test"

    def test_user_password_token_construction(self):
        """Test that user/password are combined into token when no token given."""
        client = AsyncMilvusClient(uri="http://localhost:19530", user="admin", password="secret")
        assert client._config.token == "admin:secret"

    def test_explicit_token_overrides_user_password(self):
        """Test that explicit token takes precedence over user/password."""
        client = AsyncMilvusClient(
            uri="http://localhost:19530",
            token="my_token",
            user="admin",
            password="secret",
        )
        assert client._config.token == "my_token"

    def test_dedicated_kwarg_stored(self):
        """Test that dedicated kwarg is stored on client."""
        client = AsyncMilvusClient(uri="http://localhost:19530", dedicated=True)
        assert client._dedicated is True

    @pytest.mark.asyncio
    async def test_connect_sets_handler(self):
        """Test _connect establishes connection and sets handler."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            MockHandler.return_value = mock_handler

            await client._connect()

            assert client._handler is mock_handler
            assert client._manager is not None
            assert client._using is not None
            assert client.is_self_hosted is True

            await client.close()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        """Test _connect is a no-op when already connected."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            MockHandler.return_value = mock_handler

            await client._connect()
            first_handler = client._handler

            await client._connect()  # second call should be no-op
            assert client._handler is first_handler
            assert MockHandler.call_count == 1

            await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async with AsyncMilvusClient works."""
        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            mock_handler.close = AsyncMock()
            MockHandler.return_value = mock_handler

            async with AsyncMilvusClient(uri="http://localhost:19530") as client:
                assert client._handler is mock_handler

            # After exiting context, should be closed
            assert client._closed is True
            assert client._handler is None

    @pytest.mark.asyncio
    async def test_get_connection_auto_connects(self):
        """Test _get_connection auto-connects when not yet connected."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            MockHandler.return_value = mock_handler

            conn = await client._get_connection()
            assert conn is mock_handler
            assert client._handler is mock_handler

            await client.close()

    @pytest.mark.asyncio
    async def test_get_connection_raises_when_closed(self):
        """Test _get_connection raises MilvusException when client is closed."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            mock_handler.close = AsyncMock()
            MockHandler.return_value = mock_handler

            await client._connect()
            await client.close()

            with pytest.raises(MilvusException, match="should create connection first"):
                await client._get_connection()

    def test_get_server_type_raises_when_not_connected(self):
        """Test get_server_type raises when client not connected."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with pytest.raises(MilvusException, match="Client not connected"):
            client.get_server_type()

    @pytest.mark.asyncio
    async def test_get_server_type_returns_type(self):
        """Test get_server_type returns handler's server type when connected."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="zilliz")
            MockHandler.return_value = mock_handler

            await client._connect()
            assert client.get_server_type() == "zilliz"

            await client.close()

    @pytest.mark.asyncio
    async def test_close_sets_closed_flag(self):
        """Test close sets _closed=True and releases handler."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            mock_handler.close = AsyncMock()
            MockHandler.return_value = mock_handler

            await client._connect()
            assert client._handler is not None

            await client.close()
            assert client._closed is True
            assert client._handler is None

    @pytest.mark.asyncio
    async def test_close_without_connect_is_safe(self):
        """Test close without connect doesn't raise."""
        client = AsyncMilvusClient(uri="http://localhost:19530")
        await client.close()  # Should not raise
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_use_database_updates_config(self):
        """Test use_database updates _config.db_name."""
        client = AsyncMilvusClient(uri="http://localhost:19530")

        with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler") as MockHandler:
            mock_handler = Mock()
            mock_handler.ensure_channel_ready = AsyncMock()
            mock_handler.get_server_type = Mock(return_value="milvus")
            mock_handler.describe_database = AsyncMock(return_value={})
            MockHandler.return_value = mock_handler

            await client._connect()
            assert client._config.db_name == ""

            await client.use_database("mydb")
            assert client._config.db_name == "mydb"

            await client.close()
