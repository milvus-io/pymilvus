import pytest

from .milvus_server import GrpcServer
from .types import DataType
import os


@pytest.fixture
def grpc_server():
    # just for debug
    host = os.getenv("host")
    host = host if host else "127.0.0.1"
    port = os.getenv("port")
    port = port if port else "19530"
    return GrpcServer(host=host, port=port)


class TestCreateCollection:
    def test_create_collection(self, grpc_server):
        grpc_server.create_collection("name", {"fields": [
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
