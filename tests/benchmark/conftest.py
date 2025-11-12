from unittest.mock import MagicMock, patch
import pytest

from pymilvus import MilvusClient
from . import mock_responses
from pymilvus.grpc_gen import common_pb2, milvus_pb2


@pytest.fixture
def mock_search_stub():
    def _mock_search(request, timeout=None, metadata=None):
        return mock_responses.create_search_results(
            num_queries=1,
            top_k=10,
            output_fields=["id", "age", "score", "name"]
        )
    return _mock_search


@pytest.fixture
def mock_query_stub():
    def _mock_query(request, timeout=None, metadata=None):
        return mock_responses.create_query_results(
            num_rows=100,
            output_fields=["id", "age", "score", "name", "active", "metadata"]
        )
    return _mock_query


@pytest.fixture
def mocked_milvus_client(mock_search_stub, mock_query_stub):
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
        
        mock_stub = MagicMock()
        
        
        mock_connect_response = milvus_pb2.ConnectResponse()
        mock_connect_response.status.error_code = common_pb2.ErrorCode.Success
        mock_connect_response.status.code = 0
        mock_connect_response.identifier = 12345
        mock_stub.Connect = MagicMock(return_value=mock_connect_response)
        
        mock_stub.Search = MagicMock(side_effect=mock_search_stub)
        mock_stub.Query = MagicMock(side_effect=mock_query_stub)
        mock_stub.HybridSearch = MagicMock(side_effect=mock_search_stub)
        mock_stub.DescribeCollection = MagicMock(return_value=_create_describe_collection_response())
        
        mock_stub_class.return_value = mock_stub
        
        client = MilvusClient(uri="http://localhost:19530")
        
        yield client


def _create_describe_collection_response():
    from pymilvus.grpc_gen import milvus_pb2, schema_pb2, common_pb2
    
    response = milvus_pb2.DescribeCollectionResponse()
    response.status.error_code = common_pb2.ErrorCode.Success
    
    schema = response.schema
    schema.name = "test_collection"
    
    id_field = schema.fields.add()
    id_field.fieldID = 1
    id_field.name = "id"
    id_field.data_type = schema_pb2.DataType.Int64
    id_field.is_primary_key = True
    
    embedding_field = schema.fields.add()
    embedding_field.fieldID = 2
    embedding_field.name = "embedding"
    embedding_field.data_type = schema_pb2.DataType.FloatVector
    
    dim_param = embedding_field.type_params.add()
    dim_param.key = "dim"
    dim_param.value = "128"
    
    age_field = schema.fields.add()
    age_field.fieldID = 3
    age_field.name = "age"
    age_field.data_type = schema_pb2.DataType.Int32
    
    score_field = schema.fields.add()
    score_field.fieldID = 4
    score_field.name = "score"
    score_field.data_type = schema_pb2.DataType.Float
    
    name_field = schema.fields.add()
    name_field.fieldID = 5
    name_field.name = "name"
    name_field.data_type = schema_pb2.DataType.VarChar
    
    return response
