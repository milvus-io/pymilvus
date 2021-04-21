import logging
import unittest
import pytest
from tests.utils import *

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
    def partition(self):
        collection = Collection(gen_collection_name(), gen_schema(), {})
        return Partition(collection, gen_partition_name(), "", {})

    def test_constructor(self, partition):
        LOGGER.info(type(partition))

    def test_description(self, partition):
        description = partition.description
        partition.description= description

    def test_name(self, partition):
        name = partition.name
        partition.name = name

    def test_is_empty(self, partition):
        is_empty = partition.is_empty

    def test_num_entities(self, partition):
        num = partition.num_entities

    def test_drop(self, partition):
        partition.drop()

    def test_load(self, partition):
        partition.load()

    def test_release(self, partition):
        partition.release()

    def test_insert(self, partition):
        data = gen_data(default_nb)
        partition.insert(data)
