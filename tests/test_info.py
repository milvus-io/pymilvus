import pytest

from milvus import ParamError, BaseError

from factorys import records_factory


class TestCountEntities:
    def test_count_entities_normal(self, connect, vrecords):
        count = connect.count_entities(vrecords)
        assert count == 10000

    def test_count_entities_after_delete(self, connect, vcollection, dim):
        count = 10000
        entities = [{"Vec": vector} for vector in records_factory(dim, count)]
        try:
            ids = connect.insert(vcollection, entities)
            connect.flush([vcollection])
            assert connect.count_entities(vcollection) == count

            connect.delete_entity_by_id(vcollection, [ids[0], ids[len(ids) - 1]])
            assert connect.count_entities(vcollection) == count

            connect.flush([vcollection])
            assert connect.count_entities(vcollection) == count - 2
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")

    def test_count_entities_without_existing_collection(self, connect):
        with pytest.raises(BaseError):
            connect.count_entities("test_count_entities_xxxxxxxxxx")

    @pytest.mark.parametrize("name", [1, None, False, {"a": 1}])
    def test_count_entities_with_invalid_collection_name(self, name, connect):
        with pytest.raises(ParamError):
            connect.count_entities(name)


class TestGetCollectionInfo:
    def test_get_collection_info_with_empty_collection(self, connect, vcollection):
        info = connect.get_collection_info(vcollection)

        assert bool(info['auto_id']) == True
        assert len(info['fields']) == 1
        assert info['fields'][0]['name'] == "Vec"
        assert 'dim' in info['fields'][0]['params']
        assert int(info['segment_row_limit']) >= 4096

    def test_get_collection_info_after_indexed(self, connect, vrecords):
        connect.create_index(vrecords, "Vec", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 512}})
        info = connect.get_collection_info(vrecords)
        indexes = info['fields'][0]['indexes']
        assert len(indexes) == 1

        index0 = indexes[0]
        assert index0['index_type'] == "IVF_FLAT"
        assert index0['metric_type'] == "L2"
        assert index0['params']['nlist'] == 512

    def test_get_collection_info_without_existing_collection(self, connect):
        with pytest.raises(BaseError):
            connect.get_collection_info("test_count_entities_xxxxxxxxxx")

    @pytest.mark.parametrize("name", [1, None, False, {"a": 1}])
    def test_get_collection_info_with_invalid_collection_name(self, name, connect):
        with pytest.raises(ParamError):
            connect.get_collection_info(name)


class TestGetCollectionStats:
    def test_get_collection_stats_normal(self, connect, vcollection):
        stats = connect.get_collection_stats(vcollection)
        assert stats['data_size'] < 1e-5
        assert stats['partition_count'] == 1
        assert stats['row_count'] == 0
        assert len(stats['partitions']) == 1

        p0 = stats['partitions'][0]

        assert p0['data_size'] < 1e-5
        assert p0['row_count'] == 0
        assert p0['segment_count'] == 0
        assert p0['tag'] == '_default'
        assert not p0['segments']

    def test_get_collection_stats_after_inserted(self, connect, vrecords):
        stats = connect.get_collection_stats(vrecords)
        assert stats['data_size'] > 1e-5
        assert stats['partition_count'] == 1
        assert stats['row_count'] == 10000
        assert len(stats['partitions']) == 1

        p0 = stats['partitions'][0]

        assert p0['data_size'] > 1e-5
        assert p0['row_count'] == 10000
        assert p0['segment_count'] > 0
        assert p0['tag'] == '_default'
        assert len(p0['segments']) > 0

        s0 = p0['segments'][0]
        assert s0['data_size'] > 1e-5
        assert s0['row_count'] > 0

        for f in s0['files']:
            assert f['field'] in ['_id', 'Vec']
            assert f['name'] in ['_raw', '_blf']

    def test_get_collection_stats_without_existing_collection(self, connect):
        with pytest.raises(BaseError):
            connect.get_collection_stats("test_count_entities_xxxxxxxxxx")

    @pytest.mark.parametrize("name", [1, None, False, {"a": 1}])
    def test_get_collection_stats_with_invalid_collection_name(self, name, connect):
        with pytest.raises(ParamError):
            connect.get_collection_stats(name)
