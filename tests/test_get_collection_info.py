import pytest

from milvus import DataType


class TestGetCollectionInfo:
    def test_get_collection_info_normal(self, connect, ivrecords):
        info = connect.get_collection_info(ivrecords)

        assert info['auto_id'] is True
        for f in info['fields']:
            assert f['name'] in ('Vec', 'Int')
        assert info['segment_row_limit'] == 100000
