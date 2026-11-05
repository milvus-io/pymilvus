"""Sync and async handlers preserve server error details and classification."""

import pytest
from pymilvus.exceptions import DescribeCollectionException, MilvusException
from pymilvus.grpc_gen import common_pb2, milvus_pb2

COLLECTION = "test_collection"
ERROR_CASES = [
    pytest.param(
        "describe_collection",
        "DescribeCollection",
        milvus_pb2.DescribeCollectionResponse,
        DescribeCollectionException,
        id="describe-collection",
    ),
    pytest.param(
        "has_collection",
        "DescribeCollection",
        milvus_pb2.DescribeCollectionResponse,
        MilvusException,
        id="has-collection",
    ),
    pytest.param(
        "list_indexes",
        "DescribeIndex",
        milvus_pb2.DescribeIndexResponse,
        MilvusException,
        id="list-indexes",
    ),
]


def _input_error_status():
    # Use an ordinary error, outside the handlers' not-found special cases.
    return common_pb2.Status(
        code=1100,
        reason="invalid parameter",
        extra_info={"is_input_error": "true"},
        retriable=False,
    )


@pytest.mark.parametrize("method_name,rpc_name,response_type,error_type", ERROR_CASES)
def test_sync_handler_classification(
    mock_grpc_handler, method_name, rpc_name, response_type, error_type
):
    getattr(mock_grpc_handler._stub, rpc_name).return_value = response_type(
        status=_input_error_status()
    )

    with pytest.raises(error_type) as exc:
        getattr(mock_grpc_handler, method_name)(COLLECTION)

    assert exc.value.code == 1100
    assert exc.value.message == "invalid parameter"
    assert exc.value.compatible_code == common_pb2.Success
    assert exc.value.is_input_error is True
    assert exc.value.retriable is False


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name,rpc_name,response_type,error_type", ERROR_CASES)
async def test_async_handler_classification(
    mock_async_grpc_handler, method_name, rpc_name, response_type, error_type
):
    getattr(mock_async_grpc_handler._async_stub, rpc_name).return_value = response_type(
        status=_input_error_status()
    )

    with pytest.raises(error_type) as exc:
        await getattr(mock_async_grpc_handler, method_name)(COLLECTION)

    assert exc.value.code == 1100
    assert exc.value.message == "invalid parameter"
    assert exc.value.compatible_code == common_pb2.Success
    assert exc.value.is_input_error is True
    assert exc.value.retriable is False
