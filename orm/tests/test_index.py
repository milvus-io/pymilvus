import logging
import pytest
from utils import *
from pymilvus_orm import Collection, Index

LOGGER = logging.getLogger(__name__)


class TestIndex:
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
    def index(self, params):
        collection = Collection(gen_collection_name(), schema=gen_schema(), **params)
        return Index(collection, gen_index_name(), gen_field_name(), gen_index(), **params)

    def test_name(self, index):
        LOGGER.info(index.name)

    def test_params(self, index):
        LOGGER.info(index.params)

    def test_collection_name(self, index):
        LOGGER.info(index.collection_name)

    def test_field_name(self, index):
        LOGGER.info(index.field_name)

    @pytest.mark.xfail
    def test_drop(self, index):
        index.drop()
