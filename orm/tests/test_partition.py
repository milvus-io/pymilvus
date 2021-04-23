import logging
import unittest
import pytest
from utils import *

try:
    from pymilvus_orm import Collection, Partition
except ImportError:
    from os.path import dirname, abspath
    import sys

    from pymilvus_orm import Collection, Partition

LOGGER = logging.getLogger(__name__)


class TestPartition():
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
    def partition(self, params):
        collection = Collection(gen_collection_name(), schema=gen_schema(), **params)
        return Partition(collection, gen_partition_name(), **params)

    def test_constructor(self, partition):
        LOGGER.info(type(partition))

    def test_description(self, partition):
        description = partition.description
        partition.description = description

    def test_name(self, partition):
        name = partition.name
        partition.name = name

    @pytest.mark.xfail
    def test_is_empty(self, partition):
        is_empty = partition.is_empty

    def test_num_entities(self, partition):
        num = partition.num_entities

    @pytest.mark.xfail
    def test_drop(self, partition):
        partition.drop()

    @pytest.mark.xfail
    def test_load(self, partition):
        partition.load()

    @pytest.mark.xfail
    def test_release(self, partition):
        partition.release()

    @pytest.mark.xfail
    def test_insert(self, partition):
        data = gen_data(default_nb)
        partition.insert(data)
