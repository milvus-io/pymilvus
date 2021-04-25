import logging
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

    def test_constructor(self, collection):
        assert type(collection) is Collection

    @pytest.mark.xfail
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
        data = gen_data(default_nb)
        collection.insert(data)

    @pytest.mark.xfail
    def test_search(self, collection):
        collection.search()

    @pytest.mark.xfail
    def test_partitions(self, collection):
        assert len(collection.partitions) == 0

    @pytest.mark.xfail
    def test_partition(self, collection):
        collection.partition(gen_partition_name())

    @pytest.mark.xfail
    def test_has_partition(self, collection):
        collection.has_partition(gen_partition_name())

    @pytest.mark.xfail
    def test_drop_partition(self, collection):
        collection.drop_partition(gen_partition_name())

    @pytest.mark.xfail
    def test_indexes(self, collection):
        indexes = collection.indexes

    @pytest.mark.xfail
    def test_index(self, collection):
        collection.index(gen_index_name())

    @pytest.mark.xfail
    def test_create_index(self, collection, defa):
        collection.create_index(gen_field_name(), gen_index_name())

    @pytest.mark.xfail
    def test_has_index(self, collection):
        collection.has_index(gen_index_name())

    @pytest.mark.xfail
    def test_drop_index(self, collection):
        collection.drop_index(gen_index_name())

    def test_dummy(self):
        pass
