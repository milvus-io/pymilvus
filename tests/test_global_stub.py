from unittest.mock import MagicMock, patch

import pytest

from pymilvus.exceptions import MilvusException


class TestIsGlobalEndpoint:
    def test_detects_global_cluster_in_url(self):
        from pymilvus.client.global_stub import is_global_endpoint

        assert is_global_endpoint("https://glo-xxx.global-cluster.vectordb.zilliz.com") is True

    def test_detects_global_cluster_case_insensitive(self):
        from pymilvus.client.global_stub import is_global_endpoint

        assert is_global_endpoint("https://glo-xxx.GLOBAL-CLUSTER.vectordb.zilliz.com") is True

    def test_rejects_regular_endpoint(self):
        from pymilvus.client.global_stub import is_global_endpoint

        assert is_global_endpoint("https://in01-xxx.zilliz.com") is False

    def test_rejects_empty_string(self):
        from pymilvus.client.global_stub import is_global_endpoint

        assert is_global_endpoint("") is False


class TestClusterCapability:
    def test_primary_capability(self):
        from pymilvus.client.global_stub import ClusterCapability

        assert ClusterCapability.PRIMARY == 0b11
        assert ClusterCapability.READABLE == 0b01
        assert ClusterCapability.WRITABLE == 0b10


class TestClusterInfo:
    def test_primary_cluster(self):
        from pymilvus.client.global_stub import ClusterInfo

        cluster = ClusterInfo(cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3)
        assert cluster.is_primary is True

    def test_secondary_cluster(self):
        from pymilvus.client.global_stub import ClusterInfo

        cluster = ClusterInfo(cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=1)
        assert cluster.is_primary is False


class TestGlobalTopology:
    def test_finds_primary_cluster(self):
        from pymilvus.client.global_stub import ClusterInfo, GlobalTopology

        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(cluster_id="in01-xxx", endpoint="https://in01-xxx.zilliz.com", capability=3),
                ClusterInfo(cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=1),
            ],
        )
        primary = topology.primary
        assert primary.cluster_id == "in01-xxx"
        assert primary.is_primary is True

    def test_raises_when_no_primary(self):
        from pymilvus.client.global_stub import ClusterInfo, GlobalTopology

        topology = GlobalTopology(
            version=1,
            clusters=[
                ClusterInfo(cluster_id="in02-xxx", endpoint="https://in02-xxx.zilliz.com", capability=1),
            ],
        )
        with pytest.raises(ValueError, match="No primary cluster"):
            _ = topology.primary


class TestFetchTopology:
    def test_fetches_topology_successfully(self):
        from pymilvus.client.global_stub import fetch_topology

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "version": "123",
                "clusters": [
                    {"clusterId": "in01-xxx", "endpoint": "https://in01-xxx.zilliz.com", "capability": 3},
                ],
            },
        }

        with patch("pymilvus.client.global_stub.requests.get", return_value=mock_response) as mock_get:
            topology = fetch_topology("https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token")

            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "global-cluster/topology" in call_args[0][0]
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-token"

            assert topology.version == 123
            assert len(topology.clusters) == 1
            assert topology.primary.cluster_id == "in01-xxx"

    def test_retries_on_failure(self):
        from pymilvus.client.global_stub import fetch_topology

        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "code": 0,
            "data": {
                "version": "1",
                "clusters": [
                    {"clusterId": "in01-xxx", "endpoint": "https://in01-xxx.zilliz.com", "capability": 3},
                ],
            },
        }

        with patch("pymilvus.client.global_stub.requests.get", side_effect=[mock_response_fail, mock_response_success]):
            with patch("pymilvus.client.global_stub.time.sleep"):  # Skip delays in tests
                topology = fetch_topology("https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token")
                assert topology.version == 1

    def test_raises_after_max_retries(self):
        from pymilvus.client.global_stub import fetch_topology

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("pymilvus.client.global_stub.requests.get", return_value=mock_response):
            with patch("pymilvus.client.global_stub.time.sleep"):
                with pytest.raises(MilvusException, match="Failed to fetch global topology"):
                    fetch_topology("https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token")

    def test_raises_on_api_error_code(self):
        from pymilvus.client.global_stub import fetch_topology

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": 1, "message": "Invalid token"}

        with patch("pymilvus.client.global_stub.requests.get", return_value=mock_response):
            with patch("pymilvus.client.global_stub.time.sleep"):
                with pytest.raises(MilvusException, match="Invalid token"):
                    fetch_topology("https://glo-xxx.global-cluster.vectordb.zilliz.com", "test-token")

    def test_adds_https_prefix_when_missing(self):
        from pymilvus.client.global_stub import fetch_topology

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "version": "1",
                "clusters": [
                    {"clusterId": "in01-xxx", "endpoint": "https://in01-xxx.zilliz.com", "capability": 3},
                ],
            },
        }

        with patch("pymilvus.client.global_stub.requests.get", return_value=mock_response) as mock_get:
            fetch_topology("glo-xxx.global-cluster.vectordb.zilliz.com", "test-token")

            call_args = mock_get.call_args
            # Verify https:// was added
            assert call_args[0][0] == "https://glo-xxx.global-cluster.vectordb.zilliz.com/global-cluster/topology"
