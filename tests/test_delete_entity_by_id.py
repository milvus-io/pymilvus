import pytest

from milvus import DataType, BaseError


class TestDeleteEntityByID:
    def test_delete_entity_by_id_normal(self, connect, vrecords):
        stats = connect.get_collection_stats(vrecords)
        segment_id = int(stats['partitions'][0]['segments'][0]['id'])
        ids = connect.list_id_in_segment(vrecords, segment_id)
        delete_ids = [ids[0], ids[10], ids[100]]
        connect.delete_entity_by_id(vrecords, delete_ids)
        connect.flush([vrecords])
        entities = connect.get_entity_by_id(vrecords, delete_ids)
        for e in entities:
            assert e is None

    def test_delete_entity_by_id_with_nonexist_id(self, connect, vrecords):
        stats = connect.get_collection_stats(vrecords)
        segment_id = int(stats['partitions'][0]['segments'][0]['id'])
        ids = connect.list_id_in_segment(vrecords, segment_id)
        delete_ids = [ids[0], ids[10], ids[100], ids[100] + 500000]
        try:
            connect.delete_entity_by_id(vrecords, delete_ids)
        except Exception as e:
            pytest.fail(f"Delete entity fail: {e}")

    def test_delete_entity_by_id_with_nonexist_collection(self, connect):
        with pytest.raises(BaseError):
            connect.delete_entity_by_id("test_xxx", [0])
