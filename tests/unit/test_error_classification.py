"""Cover the handler raise sites that surface the error classification
(is_input_error / retriable) via MilvusException.from_status."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pymilvus.exceptions import DescribeCollectionException, MilvusException
from pymilvus.grpc_gen import common_pb2

COLLECTION = "test_collection"


def _input_error_status():
    # An error status that is NOT any of the not-found special cases the
    # handlers short-circuit on, so the code falls through to the raise.
    return common_pb2.Status(
        code=1100,
        reason="invalid parameter",
        extra_info={"is_input_error": "true"},
        retriable=False,
    )


def _error_response():
    resp = MagicMock()
    resp.status = _input_error_status()
    return resp


class TestSyncHandlerClassification:
    def test_describe_collection_surfaces_classification(self, mock_grpc_handler):
        mock_grpc_handler._stub.DescribeCollection = MagicMock(return_value=_error_response())
        with pytest.raises(DescribeCollectionException) as exc:
            mock_grpc_handler.describe_collection(COLLECTION)
        assert exc.value.code == 1100
        assert exc.value.is_input_error is True
        assert exc.value.retriable is False

    def test_has_collection_surfaces_classification(self, mock_grpc_handler):
        mock_grpc_handler._stub.DescribeCollection = MagicMock(return_value=_error_response())
        with pytest.raises(MilvusException) as exc:
            mock_grpc_handler.has_collection(COLLECTION)
        assert exc.value.code == 1100
        assert exc.value.is_input_error is True

    def test_list_indexes_surfaces_classification(self, mock_grpc_handler):
        mock_grpc_handler._stub.DescribeIndex = MagicMock(return_value=_error_response())
        with pytest.raises(MilvusException) as exc:
            mock_grpc_handler.list_indexes(COLLECTION)
        assert exc.value.code == 1100
        assert exc.value.is_input_error is True


class TestAsyncHandlerClassification:
    @pytest.mark.asyncio
    async def test_describe_collection_surfaces_classification(self, mock_async_grpc_handler):
        mock_async_grpc_handler._async_stub.DescribeCollection = AsyncMock(
            return_value=_error_response()
        )
        with pytest.raises(DescribeCollectionException) as exc:
            await mock_async_grpc_handler.describe_collection(COLLECTION)
        assert exc.value.code == 1100
        assert exc.value.is_input_error is True
        assert exc.value.retriable is False

    @pytest.mark.asyncio
    async def test_has_collection_surfaces_classification(self, mock_async_grpc_handler):
        mock_async_grpc_handler._async_stub.DescribeCollection = AsyncMock(
            return_value=_error_response()
        )
        with pytest.raises(MilvusException) as exc:
            await mock_async_grpc_handler.has_collection(COLLECTION)
        assert exc.value.code == 1100
        assert exc.value.is_input_error is True

    @pytest.mark.asyncio
    async def test_list_indexes_surfaces_classification(self, mock_async_grpc_handler):
        mock_async_grpc_handler._async_stub.DescribeIndex = AsyncMock(
            return_value=_error_response()
        )
        with pytest.raises(MilvusException) as exc:
            await mock_async_grpc_handler.list_indexes(COLLECTION)
        assert exc.value.code == 1100
        assert exc.value.is_input_error is True
