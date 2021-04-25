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
    def partition_name(self):
        return gen_partition_name()

    @pytest.fixture(
        scope="function",
    )
    def description(self):
        return "TestPartition_description"

    @pytest.fixture(
        scope="function",
    )
    def partition(self, collection_name, schema, partition_name, description):
        collection = Collection(collection_name, schema=schema)
        params = {
            "description": description,
        }
        return Partition(collection, partition_name, **params)

    def test_constructor(self, partition):
        LOGGER.info(type(partition))

    def test_description(self, partition, description):
        assert partition.description == description

    def test_name(self, partition, partition_name):
        assert partition.name == partition_name

    @pytest.mark.xfail
    def test_is_empty(self, partition):
        assert partition.is_empty is True

    @pytest.mark.xfail
    def test_num_entities(self, partition):
        assert partition.num_entities == 0

    @pytest.mark.xfail
    def test_drop(self, partition):
        partition.drop()

    def test_load(self, partition):
        try:
            partition.load()
        except:
            assert False

    def test_release(self, partition):
        try:
            partition.release()
        except:
            assert False

    def test_insert(self, partition):
        data = gen_list_data(default_nb)
        partition.insert(data)
