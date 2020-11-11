import pytest

from milvus import DataType


class TestDeleteEntityByID:
    def test_delete_entity_by_id_normal(self, connect, vrecords):
        stats = connect.get_collection_stats(vrecords)
        segment_id = int(stats['partitions'][0]['segments'][0]['id'])
        ids = connect.list_id_in_segment(vrecords, segment_id)
        delete_ids = [ids[0], ids[10], ids[100]]
        connect.delete_entity_by_id(vrecords, delete_ids)
        entities = connect.get_entity_by_id(vrecords, delete_ids)
        for e in entities:
            assert e is None
