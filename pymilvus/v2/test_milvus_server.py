import os

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


class TestCreateCollection:
    def test_create_collection(self, server_instance):
        server_instance.create_collection("name", {"fields": [
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


class TestDropCollection:
    def test_drop_collection(self, server_instance):
        server_instance.create_collection("name", {"fields": [
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
        server_instance.drop_collection("name")


class TestHasCollection:
    def test_has_collection(self, server_instance):
        server_instance.create_collection("name", {"fields": [
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
        response = server_instance.has_collection("name")
        assert response.status.error_code == 0
        assert response.value is True


class TestDescribeCollection:
    def test_describe_collection(self, server_instance):
        server_instance.create_collection("name", {"fields": [
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
        response = server_instance.describe_collection("name")
        assert response.status.error_code == 0
