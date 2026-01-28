import pytest


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
