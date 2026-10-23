from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.client.call_context import CallContext
from pymilvus.client.entity_helper import _convert_to_vector_bytes
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import milvus_pb2


@pytest.fixture(params=[DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR])
def half_vector_client(request):
    vector_type = request.param
    schema = CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vector", vector_type, dim=4),
        ]
    )
    handler = GrpcHandler.__new__(GrpcHandler)
    handler._address = "local-test:19530"
    handler._invalidate_schema = Mock()
    handler._get_schema = Mock(return_value=(schema.to_dict(), 0))
    handler._stub = Mock()
    handler._stub.Insert.return_value = milvus_pb2.MutationResult(insert_cnt=1)
    client = MilvusClient.__new__(MilvusClient)
    client._get_connection = lambda: handler
    client._generate_call_context = lambda **kwargs: CallContext("default")
    return client, handler._stub, vector_type


@pytest.mark.parametrize("layout", ["contiguous", "positive_stride", "negative_stride"])
def test_insert_half_vector_preserves_logical_order(half_vector_client, layout):
    client, stub, vector_type = half_vector_client
    dtype = np.float16 if vector_type == DataType.FLOAT16_VECTOR else ml_dtypes.bfloat16
    if layout == "positive_stride":
        vector = np.arange(8, dtype=np.float32).astype(dtype)[::2]
    else:
        vector = np.arange(4, dtype=np.float32).astype(dtype)
        if layout == "negative_stride":
            vector = vector[::-1]
    original = vector.copy()

    client.insert("local_repro", [{"pk": 1, "vector": vector}])

    stub.Insert.assert_called_once()
    request = stub.Insert.call_args.kwargs["request"]
    field = next(field for field in request.fields_data if field.field_name == "vector")
    payload = (
        field.vectors.float16_vector
        if vector_type == DataType.FLOAT16_VECTOR
        else field.vectors.bfloat16_vector
    )
    assert field.vectors.dim == 4
    assert payload == original.tobytes()
    np.testing.assert_array_equal(np.frombuffer(payload, dtype=dtype), original)
    np.testing.assert_array_equal(vector, original)


def test_half_vector_serializer_rejects_wrong_dtype(half_vector_client):
    _, _, vector_type = half_vector_client
    vector = np.arange(8, dtype=np.float32)[::2]

    with pytest.raises(ParamError, match="invalid input for"):
        _convert_to_vector_bytes(vector, vector_type)
