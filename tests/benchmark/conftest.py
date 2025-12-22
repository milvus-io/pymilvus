from unittest.mock import MagicMock, patch

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient, StructFieldSchema
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2

from . import mock_responses


def setup_search_mock(client, mock_fn):
    client._get_connection()._stub.Search = MagicMock(side_effect=mock_fn)


def setup_query_mock(client, mock_fn):
    client._get_connection()._stub.Query = MagicMock(side_effect=mock_fn)


def setup_hybrid_search_mock(client, mock_fn):
    client._get_connection()._stub.HybridSearch = MagicMock(side_effect=mock_fn)


def get_default_test_schema() -> CollectionSchema:
    schema = MilvusClient.create_schema()
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=128)
    schema.add_field(field_name='name', datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name='bool_field', datatype=DataType.BOOL)
    schema.add_field(field_name='int8_field', datatype=DataType.INT8)
    schema.add_field(field_name='int16_field', datatype=DataType.INT16)
    schema.add_field(field_name='int32_field', datatype=DataType.INT32)
    schema.add_field(field_name='age', datatype=DataType.INT32)
    schema.add_field(field_name='float_field', datatype=DataType.FLOAT)
    schema.add_field(field_name='score', datatype=DataType.FLOAT)
    schema.add_field(field_name='double_field', datatype=DataType.DOUBLE)
    schema.add_field(field_name='varchar_field', datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name='json_field', datatype=DataType.JSON)
    schema.add_field(field_name='array_field', datatype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
    schema.add_field(field_name='geometry_field', datatype=DataType.GEOMETRY)
    schema.add_field(field_name='timestamptz_field', datatype=DataType.TIMESTAMPTZ)
    schema.add_field(field_name='binary_vector', datatype=DataType.BINARY_VECTOR, dim=128)
    schema.add_field(field_name='float16_vector', datatype=DataType.FLOAT16_VECTOR, dim=128)
    schema.add_field(field_name='bfloat16_vector', datatype=DataType.BFLOAT16_VECTOR, dim=128)
    schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name='int8_vector', datatype=DataType.INT8_VECTOR, dim=128)

    struct_schema = StructFieldSchema()
    struct_schema.add_field('struct_int', DataType.INT32)
    struct_schema.add_field('struct_str', DataType.VARCHAR, max_length=100)
    schema.add_field(field_name='struct_array_field', datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema, max_capacity=10)
    return schema


@pytest.fixture
def mocked_milvus_client():
    with patch('grpc.insecure_channel') as mock_channel_func, \
         patch('grpc.secure_channel') as mock_secure_channel_func, \
         patch('grpc.channel_ready_future') as mock_ready_future, \
         patch('pymilvus.grpc_gen.milvus_pb2_grpc.MilvusServiceStub') as mock_stub_class:

        mock_channel = MagicMock()
        mock_channel_func.return_value = mock_channel
        mock_secure_channel_func.return_value = mock_channel

        mock_future = MagicMock()
        mock_future.result = MagicMock(return_value=None)
        mock_ready_future.return_value = mock_future

        mock_connect_response = milvus_pb2.ConnectResponse()
        mock_connect_response.status.error_code = common_pb2.ErrorCode.Success
        mock_connect_response.status.code = 0
        mock_connect_response.identifier = 12345

        mock_stub = MagicMock()
        mock_stub.Connect = MagicMock(return_value=mock_connect_response)
        mock_stub.Search = MagicMock()
        mock_stub.Query = MagicMock()
        mock_stub.HybridSearch = MagicMock()

        mock_stub_class.return_value = mock_stub

        client = MilvusClient()

        yield client
