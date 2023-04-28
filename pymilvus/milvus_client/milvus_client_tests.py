"""Test the MilvusClient"""
import logging
import random
import sys
from uuid import uuid4

import numpy as np

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from pymilvus.milvus_client.milvus_client import MilvusClient

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
MILVUS_URI = None
COLLECTION_NAME = "test"


def valid_data(seed: int):
    "Generate valid data"
    datas = []
    count = 10
    for cur in range(count):
        float_num = seed + (cur / 10)
        int_num = (seed * 10) + cur
        temp = {
            "varchar": str(float_num)[:5],
            "float": np.float32(float_num),
            "int": int_num,
            "float_vector": [float_num] * 3,
        }
        datas.append(temp)

    return datas


def invalid_data(seed: int):
    """Generate wrong keyed data"""
    datas = []
    count = 10
    for cur in range(count):
        float_num = seed + (cur / 10)
        int_num = (seed * 10) + cur
        temp = {
            "varcha": str(float_num)[:5],
            "floa": np.float32(float_num),
            "in": int_num,
            "float_vecto": [float_num] * 3,
        }
        datas.append(temp)

    return datas


def create_existing_collection(uri, collection_name):
    alias = uuid4().hex
    connections.connect(uri=uri, alias=alias)
    if utility.has_collection(collection_name=collection_name, using=alias):
        utility.drop_collection(collection_name=collection_name, using=alias)
    fields = [
        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=3),
        FieldSchema(name="int", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="float", dtype=DataType.FLOAT),
        FieldSchema(name="varchar", dtype=DataType.VARCHAR, max_length=65_535),
    ]
    schema = CollectionSchema(fields)

    ret = {
        "col": Collection(collection_name, schema, using=alias),
        "fields": ["float_vector", "int", "float", "varchar"],
        "primary_field": "int",
        "vector_field": "float_vector",
    }

    return ret


class TestMilvusClient:
    """
    Tests to Run:
    Construct non existant collection
    Construct existant collection
    Insert data existant collection
    Insert data nonexistant collection
    Insert non matching data existant collection
    Insert non matching data nonexistant collection
    Insert insert data into auto_id with pk field
    insert data into auto_id without pk field
    Test Search
    Test Query
    Test get vector
    test delete vector
    test add partition
    test remove partition
    """

    @staticmethod
    def test_construct_from_existing_collection():
        info = create_existing_collection(MILVUS_URI, COLLECTION_NAME)
        client = MilvusClient(collection_name=COLLECTION_NAME, uri=MILVUS_URI)
        assert list(client.fields.keys()) == info["fields"]
        assert client.pk_field == info["primary_field"]
        assert client.vector_field == info["vector_field"]

    @staticmethod
    def test_construct_from_nonexistant_collection():
        client = MilvusClient(
            collection_name=COLLECTION_NAME, uri=MILVUS_URI, overwrite=True
        )
        assert client.fields is None
        assert client.pk_field is None
        assert client.vector_field is None

    @staticmethod
    def test_insert_in_existing_collection_valid():
        create_existing_collection(MILVUS_URI, COLLECTION_NAME)
        client = MilvusClient(collection_name=COLLECTION_NAME, uri=MILVUS_URI)
        client.insert_data(valid_data(1))
        assert len(client) == 10

    @staticmethod
    def test_insert_in_nonexistant_collection_valid():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            vector_field="float_vector",
            overwrite=True,
        )
        client.insert_data(valid_data(1))
        assert len(client) == 10

    @staticmethod
    def test_insert_in_existing_collection_invalid():
        create_existing_collection(MILVUS_URI, COLLECTION_NAME)
        client = MilvusClient(collection_name=COLLECTION_NAME, uri=MILVUS_URI)
        try:
            client.insert_data(invalid_data(1))
            raise ValueError("Failed")
        except KeyError:
            return

    @staticmethod
    def test_insert_in_nonexistant_collection_invalid():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            vector_field="float_vector",
            overwrite=True,
        )
        try:
            client.insert_data(invalid_data(1))
            raise AssertionError("Failed")
        except ValueError:
            return

    @staticmethod
    def test_insert_pk_with_autoid():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            vector_field="float_vector",
            overwrite=True,
        )
        client.insert_data(valid_data(1))
        pk = client.pk_field
        data = valid_data(2)
        for d in data:
            d[pk] = int(random.random() * 100)
        client.insert_data(data)
        assert len(client) == 20

    @staticmethod
    def test_insert_with_missing_pk_without_autoid():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            pk_field="int",
            vector_field="float_vector",
            overwrite=True,
        )
        client.insert_data(valid_data(1))
        data = valid_data(2)
        for d in data:
            d.pop("int")
        try:
            client.insert_data(data)
            raise ValueError("Failed")
        except KeyError:
            return

    @staticmethod
    def test_custom_index_existing():
        col_info = create_existing_collection(MILVUS_URI, COLLECTION_NAME)
        col: Collection = col_info["col"]
        col.create_index(
            field_name=col_info["vector_field"],
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            },
        )
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            consistency_level="Session",
            index_params={
                "index_type": "IVF_SQ8",
                "metric_type": "L2",
                "params": {"nlist": 128},
            },
        )

        assert client.collection.indexes[0].params["index_type"] == "IVF_SQ8"
        assert client.default_search_params["params"] == {"nprobe": 10}

    @staticmethod
    def test_search_default_params():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            pk_field="int",
            vector_field="float_vector",
            overwrite=True,
            consistency_level="Session",
        )
        client.insert_data(valid_data(1))
        res = client.search_data([0, 0, 0], top_k=3)
        assert len(res[0]) == 3
        assert res[0][0]["data"]["int"] == 10

    @staticmethod
    def test_query():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            pk_field="int",
            vector_field="float_vector",
            overwrite=True,
            consistency_level="Session",
        )
        client.insert_data(valid_data(1))
        res = client.query_data('varchar in ["1.1"]')
        assert res[0]["int"] == 11

    @staticmethod
    def test_delete_by_pk():
        # Test int pk
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            pk_field="int",
            vector_field="float_vector",
            overwrite=True,
            consistency_level="Session",
        )
        client.insert_data(valid_data(1))
        res = client.query_data('varchar in ["1.1"]')
        key = res[0]["int"]
        client.delete_by_pk(key)
        res = client.query_data('varchar in ["1.1"]')
        assert len(res) == 0
        # Test varchar pk
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            pk_field="varchar",
            vector_field="float_vector",
            overwrite=True,
            consistency_level="Session",
        )
        client.insert_data(valid_data(1))
        res = client.query_data('varchar in ["1.1"]')
        key = res[0]["varchar"]
        client.delete_by_pk(key)
        res = client.query_data('varchar in ["1.1"]')
        assert len(res) == 0

    @staticmethod
    def test_get_vector_by_pk():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            pk_field="int",
            vector_field="float_vector",
            overwrite=True,
            consistency_level="Session",
        )
        client.insert_data(valid_data(1))
        res = client.query_data('varchar in ["1.1"]')
        vector = client.get_vectors_by_pk(res[0]["int"])
        assert str(vector[0]["float_vector"]) == "[1.1, 1.1, 1.1]"

    @staticmethod
    def test_partition_modification():
        client = MilvusClient(
            collection_name=COLLECTION_NAME,
            uri=MILVUS_URI,
            pk_field="int",
            vector_field="float_vector",
            overwrite=True,
            consistency_level="Session",
            partitions="2",
        )
        client.insert_data(valid_data(1))
        assert len(client.collection.partitions) == 2
        client.add_partitions(["lol"])
        assert len(client.collection.partitions) == 3
        client.delete_partitions(["lol"])
        assert len(client.collection.partitions) == 2


# if __name__ == "__main__":
#     MILVUS_URI = "http://localhost:19530"
#     TestMilvusClient.test_construct_from_existing_collection()
#     TestMilvusClient.test_construct_from_nonexistant_collection()
#     TestMilvusClient.test_insert_in_existing_collection_valid()
#     TestMilvusClient.test_insert_in_nonexistant_collection_valid()
#     TestMilvusClient.test_insert_in_existing_collection_invalid()
#     TestMilvusClient.test_insert_in_nonexistant_collection_invalid()
#     TestMilvusClient.test_insert_pk_with_autoid()
#     TestMilvusClient.test_insert_with_missing_pk_without_autoid()
#     TestMilvusClient.test_search_default_params()
#     TestMilvusClient.test_custom_index_existing()
#     TestMilvusClient.test_query()
#     TestMilvusClient.test_delete_by_pk()
#     TestMilvusClient.test_get_vector_by_pk()
#     TestMilvusClient.test_partition_modification()

#     MILVUS_URI = "https://user:password@in01-bbd08105d3a44f9.aws-us-west-2.vectordb" \
#       ".zillizcloud.com:19538"
#     TestMilvusClient.test_construct_from_existing_collection()
#     TestMilvusClient.test_construct_from_nonexistant_collection()
#     TestMilvusClient.test_insert_in_existing_collection_valid()
#     TestMilvusClient.test_insert_in_nonexistant_collection_valid()
#     TestMilvusClient.test_insert_in_existing_collection_invalid()
#     TestMilvusClient.test_insert_in_nonexistant_collection_invalid()
#     TestMilvusClient.test_insert_pk_with_autoid()
#     TestMilvusClient.test_insert_with_missing_pk_without_autoid()
#     TestMilvusClient.test_search_default_params()
#     # TestMilvusClient.test_custom_index_existing() <- Not supported on Zilliz Cloud
#     TestMilvusClient.test_query()
#     TestMilvusClient.test_delete_by_pk()
#     TestMilvusClient.test_get_vector_by_pk()
#     # TestMilvusClient.test_partition_modification() <- Not supported on Zilliz Cloud
