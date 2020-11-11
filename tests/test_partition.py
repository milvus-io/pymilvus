import pytest

from milvus import BaseError


class TestCreatePartition:
    def test_create_partition_normal(self, connect, vcollection):
        try:
            connect.create_partition(vcollection, "p_1")
        except Exception as e:
            pytest.fail(f'Create partition {"p_1"} fail: {e}')

    def test_create_partition_exceed_limit(self, connect, vcollection):
        with pytest.raises(BaseError):
            for i in range(10000):
                connect.create_partition(vcollection, f"p_{i}")

    def test_create_partition_default_partition(self, connect, vcollection):
        with pytest.raises(BaseError):
            connect.create_partition(vcollection, "_default")


class TestListPartitions:
    def test_list_partitions_normal(self, connect, vcollection):
        for i in range(100):
            connect.create_partition(vcollection, f"p_{i}")

        pars = connect.list_partitions(connect)
        assert len(pars) == 100 + 1

    def test_list_partitions_with_nonexist_collection(self, connect):
        with pytest.raises(BaseError):
            connect.list_partitions("test_xxxxxxxxxxxx")


class TestDropPartition:
    def test_drop_partition_normal(self, connect, vcollection):
        connect.create_partition(vcollection, "p_1")
        connect.drop_partition(vcollection, "p_1")

    def test_drop_partition_much(self, connect, vcollection):
        for i in range(100):
            connect.create_partition(vcollection, f"p_{i}")

        for j in range(100):
            connect.drop_partition(vcollection, f"p_{j}")
