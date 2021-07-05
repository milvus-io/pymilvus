import logging
import numpy
import pytest
from utils import *
from pymilvus_orm import Collection, connections

LOGGER = logging.getLogger(__name__)


class TestCollections:
    @pytest.fixture(scope="function",)
    def collection(self):
        name = gen_collection_name()
        schema = gen_schema()
        yield Collection(name, schema=schema)
        if connections.get_connection().has_collection(name):
            connections.get_connection().drop_collection(name)

    def test_collection_by_DataFrame(self):
        from pymilvus_orm import Collection, connections
        from pymilvus_orm.schema import FieldSchema, CollectionSchema
        from pymilvus_orm.types import DataType
        fields = [
            FieldSchema("int64", DataType.INT64),
            FieldSchema("float", DataType.FLOAT),
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=128)
        ]
        collection_schema = CollectionSchema(fields, primary_field="int64")
        connections.connect()
        collection = Collection(name=gen_collection_name(), schema=collection_schema)
        collection.drop()

    def test_constructor(self, collection):
        assert type(collection) is Collection

    def test_construct_from_dataframe(self):
        assert type(Collection.construct_from_dataframe(gen_collection_name(), gen_pd_data(default_nb), primary_field="int64")[0]) is Collection

    def test_schema(self, collection):
        schema = collection.schema
        description = "This is new description"
        with pytest.raises(AttributeError):
            schema.description = description
        with pytest.raises(AttributeError):
            collection.schema = schema

    def test_description(self, collection):
        LOGGER.info(collection.description)
        description = "This is new description"
        with pytest.raises(AttributeError):
            collection.description = description

    def test_name(self, collection):
        LOGGER.info(collection.name)
        with pytest.raises(AttributeError):
            collection.name = gen_collection_name()

    def test_is_empty(self, collection):
        assert collection.is_empty is True

    def test_num_entities(self, collection):
        assert collection.num_entities == 0

    def test_drop(self, collection):
        collection.drop()

    def test_load(self, collection):
        collection.load()

    def test_release(self, collection):
        collection.release()

    @pytest.mark.xfail
    def test_insert(self, collection):
        data = gen_list_data(default_nb)
        collection.insert(data)

    def test_insert_ret(self, collection):
        vectors = gen_vectors(1, default_dim, bool(0))
        data = [
            [1],
            [numpy.float32(1.0)],
            vectors
        ]
        result = collection.insert(data)
        print(result)
        assert "insert count" in str(result)
        assert "delete count" in str(result)
        assert "upsert count" in str(result)
        assert "timestamp" in str(result)

    @pytest.mark.xfail
    def test_search(self, collection):
        collection.search()

    @pytest.mark.xfail
    def test_get(self, collection):
        data = gen_list_data(default_nb)
        ids = collection.insert(data)
        assert len(ids) == default_nb
        res = collection.get(ids[0:10])

    @pytest.mark.xfail
    def test_query(self, collection):
        data = gen_list_data(default_nb)
        ids = collection.insert(data)
        assert len(ids) == default_nb
        ids_expr = ",".join(str(x) for x in ids)
        expr = "id in [ " + ids_expr + " ]"
        res = collection.query(expr)

    def test_partitions(self, collection):
        assert len(collection.partitions) == 1

    def test_partition(self, collection):
        collection.partition(gen_partition_name())

    def test_has_partition(self, collection):
        assert collection.has_partition("_default") is True
        assert collection.has_partition(gen_partition_name()) is False

    @pytest.mark.xfail
    def test_drop_partition(self, collection):
        collection.drop_partition(gen_partition_name())

    def test_indexes(self, collection):
        assert type(collection.indexes) is list
        assert len(collection.indexes) == 0

    @pytest.mark.xfail
    def test_index(self, collection):
        collection.index()

    @pytest.mark.xfail
    def test_create_index(self, collection, defa):
        collection.create_index(gen_field_name(), gen_index_name())

    def test_has_index(self, collection):
        assert collection.has_index() is False

    @pytest.mark.xfail
    def test_drop_index(self, collection):
        collection.drop_index()

    def test_dummy(self):
        pass
