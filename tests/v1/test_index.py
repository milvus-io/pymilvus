import pytest
from pymilvus import Collection, Index, connections

from utils import (
    gen_index_name,
    gen_field_name,
    gen_collection_name,
    gen_schema,
    gen_index,
    gen_simple_index,
)


@pytest.mark.skip("Connect with the real server")
class TestIndex:
    @pytest.fixture(scope="function")
    def name(self):
        return gen_index_name()

    @pytest.fixture(scope="function")
    def field_name(self):
        return gen_field_name()

    @pytest.fixture(scope="function")
    def collection_name(self):
        return gen_collection_name()

    @pytest.fixture(scope="function")
    def schema(self):
        return gen_schema()

    @pytest.fixture(scope="function")
    def index_param(self):
        return gen_index()

    @pytest.fixture(
        scope="function",
        params=gen_simple_index()
    )
    def get_simple_index(self, request):
        return request.param

    @pytest.fixture(scope="function")
    def index(self, name, field_name, collection_name, schema, get_simple_index):
        connections.connect()
        collection = Collection(collection_name, schema=schema)
        return Index(collection, field_name, get_simple_index)

    def test_params(self, index, get_simple_index):
        assert index.params == get_simple_index

    def test_collection_name(self, index, collection_name):
        assert index.collection_name == collection_name

    def test_field_name(self, index, field_name):
        assert index.field_name == field_name

    def test_drop(self, index):
        index.drop()
