import logging
import unittest
import pytest
from utils import *
from pymilvus_orm import Collection, Partition

LOGGER = logging.getLogger(__name__)


class TestPartition:
    @pytest.fixture(scope="function")
    def collection_name(self):
        return gen_collection_name()

    @pytest.fixture(scope="function")
    def schema(self):
        return gen_schema()

    @pytest.fixture(scope="function")
    def partition_name(self):
        return gen_partition_name()

    @pytest.fixture(scope="function")
    def description(self):
        return "TestPartition_description"

    @pytest.fixture(scope="function")
    def collection(self, collection_name, schema):
        c = Collection(collection_name, schema=schema)
        yield c
        c.drop()

    @pytest.fixture(scope="function")
    def partition(self, collection, partition_name, description):
        params = {
            "description": description,
        }
        yield Partition(collection, partition_name, **params)
        if collection.has_partition(partition_name):
            collection.drop_partition(partition_name)

    def test_constructor(self, partition):
        assert type(partition) is Partition

    def test_description(self, partition, description):
        assert partition.description == description

    def test_name(self, partition, partition_name):
        assert partition.name == partition_name

    def test_is_empty(self, partition):
        assert partition.is_empty is True

    def test_num_entities(self, partition):
        assert partition.num_entities == 0

    def test_drop(self, collection, partition, partition_name):
        assert collection.has_partition(partition_name) is True
        partition.drop()
        assert collection.has_partition(partition_name) is False

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

    @pytest.mark.xfail
    def test_get(self, partition):
        data = gen_list_data(default_nb)
        ids = partition.insert(data)
        assert len(ids) == default_nb
        res = partition.get(ids[0:10])

    @pytest.mark.xfail
    def test_query(self, partition):
        data = gen_list_data(default_nb)
        ids = partition.insert(data)
        assert len(ids) == default_nb
        ids_expr = ",".join(str(x) for x in ids)
        expr = "id in [ " + ids_expr + " ]"
        res = partition.query(expr)
