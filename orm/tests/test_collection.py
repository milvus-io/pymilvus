import logging
import pytest
from utils import *
from pymilvus_orm import Collection

LOGGER = logging.getLogger(__name__)


class TestCollections:
    @pytest.fixture(
        scope="function",
    )
    def params(self):
        params = {
            "default": {"host": "localhost", "port": "19530"},
            "dev": {"host": "localhost", "port": "19530"}
        }
        return params

    @pytest.fixture(
        scope="function",
    )
    def collection(self, params):
        name = gen_collection_name()
        schema = gen_schema()
        return Collection(name, schema, **params)

    def test_constructor(self, collection):
        LOGGER.info(type(collection))

    def test_schema(self, collection):
        schema = collection.schema
        description = "This is new description"
        schema.description = description
        collection.schema(schema)
        new_schema = collection.schema()
        assert new_schema.description == description

    def test_description(self, collection):
        LOGGER.info(collection.description)
        description = "This is new description"
        collection.description = description
        assert collection.description == description

    def test_name(self, collection):
        LOGGER.info(collection.name)
        collection.name = gen_collection_name()

    def test_is_empty(self, collection):
        is_empty = collection.is_empty

    def test_num_entities(self, collection):
        num = collection.num_entities

    def test_drop(self, collection):
        collection.drop()

    def test_load(self, collection):
        collection.load()

    def test_release(self, collection):
        collection.release()

    def test_insert(self, collection):
        data = gen_data(default_nb)
        collection.insert(data)

    def test_search(self, collection):
        collection.search()

    def test_partitions(self, collection):
        partitions = collection.partitions

    def test_partition(self, collection):
        collection.partition(gen_partition_name())

    def test_has_partition(self, collection):
        collection.has_partition(gen_partition_name())

    def test_drop_partition(self, collection):
        collection.drop_partition(gen_partition_name())

    def test_indexes(self, collection):
        indexes = collection.indexes

    def test_index(self, collection):
        collection.index(gen_index_name())

    def test_create_index(self, collection, defa):
        collection.create_index(gen_field_name(), gen_index_name())

    def test_has_index(self, collection):
        collection.has_index(gen_index_name())

    def test_drop_index(self, collection):
        collection.drop_index(gen_index_name())
