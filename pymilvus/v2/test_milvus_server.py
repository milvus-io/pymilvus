import os
import random

import pytest

from .milvus_server import GrpcServer
from .types import DataType


@pytest.fixture
def server_instance():
    # just for debug
    host = os.getenv("host")
    host = host if host else "127.0.0.1"
    port = os.getenv("port")
    port = port if port else "19530"
    return GrpcServer(host=host, port=port)


@pytest.fixture
def collection_name():
    # just for develop
    return f"collection_{random.randint(100000000, 999999999)}"


@pytest.fixture
def partition_name():
    # just for develop
    return f"partition_{random.randint(100000000, 999999999)}"


class TestCreateCollection:
    def test_create_collection(self, server_instance, collection_name):
        response = server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        assert response.error_code == 0


class TestDropCollection:
    def test_drop_collection(self, server_instance, collection_name):
        server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        response = server_instance.drop_collection(collection_name)
        assert response.error_code == 0


class TestHasCollection:
    def test_has_collection(self, server_instance, collection_name):
        server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        response = server_instance.has_collection(collection_name)
        assert response.status.error_code == 0
        assert response.value is True


class TestDescribeCollection:
    def test_describe_collection(self, server_instance, collection_name):
        server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        response = server_instance.describe_collection(collection_name)
        assert response.status.error_code == 0


class TestListCollections:
    def test_list_collections(self, server_instance, collection_name):
        server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        response = server_instance.list_collections()
        assert response.status.error_code == 0
        assert collection_name in list(response.collection_names)


class TestCreatePartition:
    def test_create_partition(self, server_instance, collection_name, partition_name):
        server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        response = server_instance.create_partition(collection_name, partition_name)
        assert response.error_code == 0


class TestDropPartition:
    def test_drop_partition(self, server_instance, collection_name, partition_name):
        server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        server_instance.create_partition(collection_name, partition_name)
        response = server_instance.drop_partition(collection_name, partition_name)
        assert response.error_code == 0


class TestHasPartition:
    def test_has_partition(self, server_instance, collection_name, partition_name):
        server_instance.create_collection(collection_name, {"fields": [
            {
                "name": "my_id",
                "type": DataType.INT64,
                "auto_id": True,
                "is_primary": True,
            },
            {
                "name": "my_vector",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 64},
            }
        ], "description": "this is a description"}, 2)
        server_instance.create_partition(collection_name, partition_name)
        response = server_instance.has_partition(collection_name, partition_name)
        assert response.status.error_code == 0
        assert response.value is True
