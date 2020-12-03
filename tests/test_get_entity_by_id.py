import pytest

from milvus import BaseError, ParamError

from factorys import records_factory


class TestGetEntityByID:
    def test_get_entity_by_id_normal(self, connect, vcollection, dim):
        count = 10000
        vectors = records_factory(dim, count)
        entities = [{"Vec": vector} for vector in vectors]
        try:
            ids = connect.insert(vcollection, entities, partition_tag='_default')
            connect.flush([vcollection])

            got_entities = connect.get_entity_by_id(vcollection, ids[:1])
            assert got_entities[0].id == ids[0]
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")

    def test_get_entity_by_id_with_some_id_noexist(self, connect, vcollection, dim):
        count = 10000
        vectors = records_factory(dim, count)
        entities = [{"Vec": vector} for vector in vectors]
        try:
            ids = connect.insert(vcollection, entities, partition_tag='_default')
            connect.flush([vcollection])

            got_entities = connect.get_entity_by_id(vcollection, [ids[0] - 1])
            assert got_entities[0] is None

            got_entities2 = connect.get_entity_by_id(vcollection, [ids[0], ids[0] - 1])
            assert got_entities2[0].id == ids[0]
            assert got_entities2[1] is None
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")

    def test_get_entity_by_id_without_existing_collection(self, connect):
        with pytest.raises(BaseError):
            connect.get_entity_by_id("test_count_entities_xxxxxxxxxx", [0])

    @pytest.mark.parametrize("name", [1, None, False, {"a": 1}])
    def test_get_entity_by_id_with_invalid_collection_name(self, name, connect):
        with pytest.raises(ParamError):
            connect.get_entity_by_id(name, [0])
