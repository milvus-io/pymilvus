import threading
from unittest.mock import MagicMock, patch

import pytest
from pymilvus.client.global_topology import (
    GLOBAL_CLUSTER_IDENTIFIER,
    ClusterCapability,
    ClusterInfo,
    GlobalTopology,
    TopologyRefresher,
    fetch_topology,
    is_global_endpoint,
)
from pymilvus.exceptions import MilvusException

# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_topology(version=1, clusters=None):
    if clusters is None:
        clusters = [
            ClusterInfo(cluster_id="in01", endpoint="https://in01.zilliz.com", capability=3)
        ]
    return GlobalTopology(version=version, clusters=clusters)


def _mock_response(
    version="1",
    cluster_id="in01-xxx",
    endpoint="https://in01-xxx.zilliz.com",
    capability=3,
    code=0,
    message=None,
):
    resp = MagicMock()
    resp.status_code = 200
    body = {
        "code": code,
        "data": {
            "version": version,
            "clusters": [{"clusterId": cluster_id, "endpoint": endpoint, "capability": capability}],
        },
    }
    if message is not None:
        body["message"] = message
        body.pop("data", None)
    resp.json.return_value = body
    return resp


# ── TestIsGlobalEndpoint ──────────────────────────────────────────────────────


class TestIsGlobalEndpoint:
    def test_detects_global_cluster_in_url(self):
        assert is_global_endpoint("https://glo-xxx.global-cluster.vectordb.zilliz.com") is True

    def test_detects_global_cluster_case_insensitive(self):
        assert is_global_endpoint("https://glo-xxx.GLOBAL-CLUSTER.vectordb.zilliz.com") is True

    def test_rejects_regular_endpoint(self):
        assert is_global_endpoint("https://in01-xxx.zilliz.com") is False

    def test_rejects_empty_string(self):
        assert is_global_endpoint("") is False


# ── TestClusterCapability ─────────────────────────────────────────────────────


class TestClusterCapability:
    def test_primary_capability(self):
        assert ClusterCapability.PRIMARY == 0b11
        assert ClusterCapability.READABLE == 0b01
        assert ClusterCapability.WRITABLE == 0b10


# ── TestClusterInfo ───────────────────────────────────────────────────────────


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


# ── TestGlobalTopology ────────────────────────────────────────────────────────


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


# ── TestFetchTopology ─────────────────────────────────────────────────────────

_GLOBAL_URL = "https://glo-xxx.global-cluster.vectordb.zilliz.com"
_TOKEN = "test-token"


class TestFetchTopology:
    def test_fetches_topology_successfully(self):
        mock_response = _mock_response(version="123")
        with patch(
            "pymilvus.client.global_topology.requests.get", return_value=mock_response
        ) as mock_get:
            topology = fetch_topology(_GLOBAL_URL, _TOKEN)
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "global-cluster/topology" in call_args[0][0]
            assert call_args[1]["headers"]["Authorization"] == f"Bearer {_TOKEN}"
            assert topology.version == 123
            assert len(topology.clusters) == 1
            assert topology.primary.cluster_id == "in01-xxx"

    def test_retries_on_failure(self):
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_success = _mock_response(version="1")
        with patch(
            "pymilvus.client.global_topology.requests.get",
            side_effect=[mock_fail, mock_success],
        ), patch("pymilvus.client.global_topology.time.sleep"):
            topology = fetch_topology(_GLOBAL_URL, _TOKEN)
            assert topology.version == 1

    def test_raises_after_max_retries(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        with patch(
            "pymilvus.client.global_topology.requests.get", return_value=mock_response
        ), patch("pymilvus.client.global_topology.time.sleep"):
            with pytest.raises(MilvusException, match="Failed to fetch global topology"):
                fetch_topology(_GLOBAL_URL, _TOKEN)

    def test_raises_on_api_error_code(self):
        mock_response = _mock_response(code=1, message="Invalid token")
        with patch(
            "pymilvus.client.global_topology.requests.get", return_value=mock_response
        ), patch("pymilvus.client.global_topology.time.sleep"):
            with pytest.raises(MilvusException, match="Invalid token"):
                fetch_topology(_GLOBAL_URL, _TOKEN)

    def test_adds_https_prefix_when_missing(self):
        mock_response = _mock_response()
        with patch(
            "pymilvus.client.global_topology.requests.get", return_value=mock_response
        ) as mock_get:
            fetch_topology("glo-xxx.global-cluster.vectordb.zilliz.com", _TOKEN)
            call_args = mock_get.call_args
            assert (
                call_args[0][0]
                == "https://glo-xxx.global-cluster.vectordb.zilliz.com/global-cluster/topology"
            )


# ── TestTopologyRefresher ─────────────────────────────────────────────────────


class TestTopologyRefresher:
    def test_starts_and_stops(self):
        topology = _make_topology()
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
        initial_topology = _make_topology(version=1)
        new_topology = _make_topology(
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

        with patch("pymilvus.client.global_topology.fetch_topology", return_value=new_topology):
            refresher.start()
            callback_called.wait(timeout=1.0)
            refresher.stop()

        assert len(received_topology) == 1
        assert received_topology[0].version == 2

    def test_does_not_update_on_same_version(self):
        topology = _make_topology(version=1)
        callback_called = []
        refresh_count = threading.Event()
        call_count = 0

        def counting_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                refresh_count.set()
            return topology

        def on_topology_change(topo):
            callback_called.append(topo)

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.zilliz.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.05,
            on_topology_change=on_topology_change,
        )

        with patch("pymilvus.client.global_topology.fetch_topology", side_effect=counting_fetch):
            refresher.start()
            refresh_count.wait(timeout=2.0)
            refresher.stop()

        assert len(callback_called) == 0

    def test_trigger_refresh_immediate(self):
        initial_topology = _make_topology(version=1)
        new_topology = _make_topology(
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

        with patch("pymilvus.client.global_topology.fetch_topology", return_value=new_topology):
            refresher.start()
            refresher.trigger_refresh()
            callback_called.wait(timeout=1.0)
            refresher.stop()

        assert callback_called.is_set()

    def test_continues_on_fetch_failure(self):
        topology = _make_topology(version=1)
        fetch_attempted = threading.Event()

        def failing_fetch(*args, **kwargs):
            fetch_attempted.set()
            raise Exception("Network error")

        refresher = TopologyRefresher(
            global_endpoint="https://glo.global-cluster.zilliz.com",
            token="test-token",
            topology=topology,
            refresh_interval=0.05,
        )

        with patch("pymilvus.client.global_topology.fetch_topology", side_effect=failing_fetch):
            refresher.start()
            fetch_attempted.wait(timeout=2.0)
            assert refresher.is_running()
            refresher.stop()

        assert refresher.get_topology().version == 1  # Still has original topology


# ── TestGlobalClusterConstant ─────────────────────────────────────────────────


class TestGlobalClusterConstant:
    def test_global_cluster_identifier_constant(self):
        assert GLOBAL_CLUSTER_IDENTIFIER == "global-cluster"
