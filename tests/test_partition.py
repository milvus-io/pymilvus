import pytest

from milvus import ParamError


class TestCreatePartition:
    def test_create_partition_normal(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "new_tag")
        assert status.OK()

    @pytest.mark.parametrize("tag", [[], 1234353, {}, [1, 2]])
    def test_create_partition_invalid_tag(self, tag, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.create_partition(gcollection, tag)

    @pytest.mark.skip(reason="Bug here. See #1762(https://github.com/milvus-io/milvus/issues/1762)")
    def test_create_partition_default(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "_default")
        assert not status.OK()

    def test_create_partition_repeat(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "tag01")
        assert status.OK()

        status = gcon.create_partition(gcollection, "tag01")
        assert not status.OK()


class TestHadPartition:
    @pytest.mark.skip(reason="default partition tag is not exists")
    def test_has_partition_default(self, gcon, gcollection):
        status, exits = gcon.has_partition(gcollection, "_default")
        assert status.OK()
        assert exits

    def test_has_partition_created(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "test")
        assert status.OK()
        status, exists = gcon.has_partition(gcollection, "test")
        assert status.OK()
        assert exists

    def test_has_partition_with_tag_non_existent(self, gcon, gcollection):
        status, exists = gcon.has_partition(gcollection, "test_has_partition_with_tag_non_existent")
        assert status.OK()
        assert not exists

    @pytest.mark.parametrize("tag", [[], None, 123, {}, True])
    def test_drop_partition_invalid_tag(self, tag, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.drop_partition(gcollection, tag)


class TestShowPartitions:
    def test_list_partitions_normal(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "tag01")
        assert status.OK()

        status, partitions = gcon.list_partitions(gcollection)
        assert status.OK()
        for partition in partitions:
            assert partition.collection_name == gcollection
            assert partition.tag in ("tag01", "_default")


class TestDropPartition:
    def test_drop_partition(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "tag01")
        assert status.OK()

        status = gcon.drop_partition(gcollection, "tag01")
        assert status.OK()

    @pytest.mark.parametrize("tag", [[], None, 123, {}])
    def test_drop_partition_invalid_tag(self, tag, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.drop_partition(gcollection, tag)

    def test_drop_partition_non_existent(self, gcon, gcollection):
        status = gcon.drop_partition(gcollection, "non_existent")
        assert not status.OK()
