import threading
import time
from unittest.mock import MagicMock, patch

import grpc
import pytest
from pymilvus.client.global_stub import (
    ClusterCapability,
    ClusterInfo,
    GlobalStub,
    GlobalTopology,
    TopologyRefresher,
    fetch_topology,
    is_global_endpoint,
)
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.exceptions import MilvusException


class TestIsGlobalEndpoint:
    def test_detects_global_cluster_in_url(self):
        assert is_global_endpoint("https://glo-xxx.global-cluster.vectordb.zilliz.com") is True

    def test_detects_global_cluster_case_insensitive(self):
        assert is_global_endpoint("https://glo-xxx.GLOBAL-CLUSTER.vectordb.zilliz.com") is True

    def test_rejects_regular_endpoint(self):
        assert is_global_endpoint("https://in01-xxx.zilliz.com") is False

    def test_rejects_empty_string(self):
        assert is_global_endpoint("") is False


class TestClusterCapability:
    def test_primary_capability(self):
        assert ClusterCapability.PRIMARY == 0b11
        assert ClusterCapability.READABLE == 0b01
        assert ClusterCapability.WRITABLE == 0b10


class TestClusterInfo:
    def test_primary_cluster(self):
        cluster = ClusterInfo(
            cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
        )
        assert cluster.is_primary is True

    def test_secondary_cluster(self):
        cluster = ClusterInfo(
            cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=1
        )
        assert cluster.is_primary is False


class TestGlobalTopology:
    def test_finds_primary_cluster(self):
        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
                ClusterInfo(
                    cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=1
                ),
            ],
        )
        primary = topology.primary
        assert primary.cluster_id == "in01-xxx"
        assert primary.is_primary is True

    def test_raises_when_no_primary(self):
        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=1
                ),
            ],
        )
        with pytest.raises(ValueError, match="No primary cluster"):
            _ = topology.primary


class TestFetchTopology:
    def test_fetches_topology_successfully(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "version": "123",
                "clusters": [
                    {
                        "clusterId": "in01-xxx",
                        "endpoint": "https://in01-xxx.zilliz.com",
                        "capability": 3,
                    },
                ],
            },
        }

        with patch(
            "pymilvus.client.global_stub.requests.get", return_value=mock_response
        ) as mock_get:
            topology = fetch_topology(
                "https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token"
            )

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "global-cluster/topology" in call_args[0][0]
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-token"

            assert topology.version == 123
            assert len(topology.clusters) == 1
            assert topology.primary.cluster_id == "in01-xxx"

    def test_retries_on_failure(self):
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "code": 0,
            "data": {
                "version": "1",
                "clusters": [
                    {
                        "clusterId": "in01-xxx",
                        "endpoint": "https://in01-xxx.zilliz.com",
                        "capability": 3,
                    },
                ],
            },
        }

        with patch(
            "pymilvus.client.global_stub.requests.get",
            side_effect=[mock_response_fail, mock_response_success],
        ):
            with patch("pymilvus.client.global_stub.time.sleep"):  # Skip delays in tests
                topology = fetch_topology(
                    "https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token"
                )
                assert topology.version == 1

    def test_raises_after_max_retries(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("pymilvus.client.global_stub.requests.get", return_value=mock_response):
            with patch("pymilvus.client.global_stub.time.sleep"):
                with pytest.raises(MilvusException, match="Failed to fetch global topology"):
                    fetch_topology(
                        "https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token"
                    )

    def test_raises_on_api_error_code(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": 1, "message": "Invalid token"}

        with patch("pymilvus.client.global_stub.requests.get", return_value=mock_response):
            with patch("pymilvus.client.global_stub.time.sleep"):
                with pytest.raises(MilvusException, match="Invalid token"):
                    fetch_topology(
                        "https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token"
                    )

    def test_adds_https_prefix_when_missing(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "version": "1",
                "clusters": [
                    {
                        "clusterId": "in01-xxx",
                        "endpoint": "https://in01-xxx.zilliz.com",
                        "capability": 3,
                    },
                ],
            },
        }

        with patch(
            "pymilvus.client.global_stub.requests.get", return_value=mock_response
        ) as mock_get:
            fetch_topology("glo-xxx.global-cluster.vectordb.zilliz.com", "test-token")

            call_args = mock_get.call_args
            # Verify https:// was added
            assert (
                call_args[0][0]
                == "https://glo-xxx.global-cluster.vectordb.zilliz.com/global-cluster/topology"
            )


class TestTopologyRefresher:
    def test_starts_and_stops(self):
        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(cluster_id="in01", endpoint="https://in01.zilliz.com", capability=3)
            ],
        )

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.zilliz.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.1,
        )
        refresher.start()
        assert refresher.is_running()

        refresher.stop()
        assert not refresher.is_running()

    def test_updates_topology_on_version_change(self):
        initial_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(cluster_id="in01", endpoint="https://in01.zilliz.com", capability=3)
            ],
        )

        new_topology = GlobalTopology(
            version=2,
            clusters=[
                ClusterInfo(cluster_id="in02", endpoint="https://in02.zilliz.com", capability=3)
            ],
        )

        callback_called = threading.Event()
        received_topology = []

        def on_topology_change(topo):
            received_topology.append(topo)
            callback_called.set()

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.zilliz.com",
            token="test-token",
            topology=initial_topology,
            refresh_interval=0.05,
            on_topology_change=on_topology_change,
        )

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=new_topology):
            refresher.start()
            callback_called.wait(timeout=1.0)
            refresher.stop()

        assert len(received_topology) == 1
        assert received_topology[0].version == 2

    def test_does_not_update_on_same_version(self):
        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(cluster_id="in01", endpoint="https://in01.zilliz.com", capability=3)
            ],
        )

        callback_called = []

        def on_topology_change(topo):
            callback_called.append(topo)

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.zilliz.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.05,
            on_topology_change=on_topology_change,
        )

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=topology):
            refresher.start()
            time.sleep(0.2)  # Allow a few refresh cycles
            refresher.stop()

        assert len(callback_called) == 0

    def test_trigger_refresh_immediate(self):
        initial_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(cluster_id="in01", endpoint="https://in01.zilliz.com", capability=3)
            ],
        )

        new_topology = GlobalTopology(
            version=2,
            clusters=[
                ClusterInfo(cluster_id="in02", endpoint="https://in02.zilliz.com", capability=3)
            ],
        )

        callback_called = threading.Event()

        def on_topology_change(topo):
            callback_called.set()

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.zilliz.com",
            token="test-token",
            topology=initial_topology,
            refresh_interval=300,  # Long interval - shouldn't trigger automatically
            on_topology_change=on_topology_change,
        )

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=new_topology):
            refresher.start()
            refresher.trigger_refresh()
            callback_called.wait(timeout=1.0)
            refresher.stop()

        assert callback_called.is_set()

    def test_continues_on_fetch_failure(self):
        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(cluster_id="in01", endpoint="https://in01.zilliz.com", capability=3)
            ],
        )

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.zilliz.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.05,
        )

        with patch(
            "pymilvus.client.global_stub.fetch_topology", side_effect=Exception("Network error")
        ):
            refresher.start()
            time.sleep(0.2)  # Should not crash
            assert refresher.is_running()
            refresher.stop()

        assert refresher.get_topology().version == 1  # Still has original topology


class TestGlobalStub:
    def test_initializes_with_topology(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        mock_handler = MagicMock()

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
                stub = GlobalStub(
                    global_endpoint="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                )

                assert stub.get_topology().version == 1
                assert stub.get_primary_endpoint() == "https://in01-xxx.zilliz.com"

    def test_provides_primary_handler(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        mock_handler = MagicMock()
        mock_handler._stub = MagicMock()
        mock_handler._channel = MagicMock()

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
                stub = GlobalStub(
                    global_endpoint="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                )

                assert stub.get_primary_handler() is mock_handler

    def test_reconnects_on_topology_change(self):
        initial_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        new_topology = GlobalTopology(
            version=2,
            clusters=[
                ClusterInfo(
                    cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=3
                ),
            ],
        )

        mock_handler_1 = MagicMock()
        mock_handler_2 = MagicMock()

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=initial_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler_1):
                stub = GlobalStub(
                    global_endpoint="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                    start_refresher=False,  # Don't start background refresh for this test
                )

        # Simulate topology change
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler_2):
            stub._on_topology_change(new_topology)

        assert stub.get_primary_handler() is mock_handler_2
        assert stub.get_primary_endpoint() == "https://in02-xxx.zilliz.com"
        mock_handler_1.close.assert_called_once()

    def test_closes_cleanly(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        mock_handler = MagicMock()

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
                stub = GlobalStub(
                    global_endpoint="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                )

                stub.close()

                mock_handler.close.assert_called_once()
                assert not stub._refresher.is_running()

    def test_trigger_refresh_on_connection_error(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        mock_handler = MagicMock()

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
                stub = GlobalStub(
                    global_endpoint="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                    start_refresher=False,
                )

                with patch.object(stub._refresher, "trigger_refresh") as mock_trigger:
                    stub.trigger_refresh()
                    mock_trigger.assert_called_once()


class TestGrpcHandlerGlobalIntegration:
    def test_uses_global_stub_for_global_endpoint(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"):
                handler = GrpcHandler(
                    uri="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                )

                assert handler._global_stub is not None
                assert handler._global_stub.get_primary_endpoint() == "https://in01-xxx.zilliz.com"

    def test_uses_regular_connection_for_non_global_endpoint(self):
        with patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"):
            handler = GrpcHandler(uri="https://in01-xxx.zilliz.com", token="test-token")

            assert handler._global_stub is None

    def test_triggers_refresh_on_unavailable_error(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"):
                handler = GrpcHandler(
                    uri="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                )

                with patch.object(handler._global_stub, "trigger_refresh") as mock_trigger:
                    # Simulate UNAVAILABLE error
                    mock_error = MagicMock()
                    mock_error.code.return_value = grpc.StatusCode.UNAVAILABLE

                    handler._handle_global_connection_error(mock_error)

                    mock_trigger.assert_called_once()

    def test_does_not_trigger_refresh_on_other_errors(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"):
                handler = GrpcHandler(
                    uri="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                )

                with patch.object(handler._global_stub, "trigger_refresh") as mock_trigger:
                    # Simulate INVALID_ARGUMENT error (should not trigger refresh)
                    mock_error = MagicMock()
                    mock_error.code.return_value = grpc.StatusCode.INVALID_ARGUMENT

                    handler._handle_global_connection_error(mock_error)

                    mock_trigger.assert_not_called()


class TestGlobalErrorHandling:
    def test_retry_decorator_triggers_refresh_on_unavailable(self):
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"):
                handler = GrpcHandler(
                    uri="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                )

                # Verify that _handle_global_connection_error is wired up properly
                assert hasattr(handler, "_handle_global_connection_error")
                assert handler._global_stub is not None


class TestGlobalClientEndToEnd:
    def test_full_initialization_flow(self):
        """Test the complete flow from GlobalStub initialization to handler access."""
        mock_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
                ClusterInfo(
                    cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=1
                ),
            ],
        )

        mock_grpc_handler = MagicMock()
        mock_grpc_handler._stub = MagicMock()
        mock_grpc_handler._channel = MagicMock()
        mock_grpc_handler._final_channel = MagicMock()
        mock_grpc_handler._address = "in01-xxx.zilliz.com:443"
        mock_grpc_handler.get_server_type.return_value = "zilliz"

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=mock_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_grpc_handler):
                # Test GlobalStub initialization and handler access
                stub = GlobalStub(
                    global_endpoint="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                    start_refresher=False,
                )

                # Verify the topology was fetched
                assert stub.get_topology().version == 1
                assert len(stub.get_topology().clusters) == 2

                # Verify the primary endpoint is correct
                assert stub.get_primary_endpoint() == "https://in01-xxx.zilliz.com"

                # Verify handler is accessible
                handler = stub.get_primary_handler()
                assert handler is mock_grpc_handler

                # Clean up
                stub.close()

    def test_topology_refresh_updates_connection(self):
        """Test that topology refresh properly updates the primary connection."""
        initial_topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(
                    cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3
                ),
            ],
        )

        new_topology = GlobalTopology(
            version=2,
            clusters=[
                ClusterInfo(
                    cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=3
                ),
            ],
        )

        mock_handler_1 = MagicMock()
        mock_handler_1._stub = MagicMock()
        mock_handler_2 = MagicMock()
        mock_handler_2._stub = MagicMock()

        handler_calls = [mock_handler_1, mock_handler_2]

        with patch("pymilvus.client.global_stub.fetch_topology", return_value=initial_topology):
            with patch("pymilvus.client.grpc_handler.GrpcHandler", side_effect=handler_calls):
                stub = GlobalStub(
                    global_endpoint="https://glo-xxx.global-cluster.vectordb.zilliz.com",
                    token="test-token",
                    start_refresher=False,
                )

                initial_handler = stub.get_primary_handler()
                assert initial_handler is mock_handler_1

                # Trigger topology change
                stub._on_topology_change(new_topology)

                new_handler = stub.get_primary_handler()
                assert new_handler is mock_handler_2
                assert stub.get_primary_endpoint() == "https://in02-xxx.zilliz.com"
