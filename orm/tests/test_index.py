import logging
import pytest
from utils import *
from pymilvus_orm import Collection, Index

LOGGER = logging.getLogger(__name__)


class TestIndex:
    @pytest.fixture(
        scope="function",
    )
    def name(self):
        return gen_index_name()

    @pytest.fixture(
        scope="function",
    )
    def field_name(self):
        return gen_field_name()

    @pytest.fixture(
        scope="function",
    )
    def collection_name(self):
        return gen_collection_name()

    @pytest.fixture(
        scope="function",
    )
    def schema(self):
        return gen_schema()

    @pytest.fixture(
        scope="function",
    )
    def index_param(self):
        return gen_index()

    @pytest.fixture(
        scope="function",
    )
    def index(self, name, field_name, collection_name, schema, index_param):
        # from pymilvus_orm.collection import Collection
        collection = Collection(collection_name, schema=schema)
        return Index(collection, field_name, index_param, name)

    def test_name(self, index, name):
        assert index.name == name

    def test_params(self, index, index_param):
        assert index.params == index_param

    def test_collection_name(self, index, collection_name):
        assert index.collection_name == collection_name

    def test_field_name(self, index, field_name):
        assert index.field_name == field_name

    def test_drop(self, index):
        with pytest.raises(Exception):
            index.drop()
