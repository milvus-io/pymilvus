"""Tests for bytes vector type auto-detection in search/hybrid_search paths.

Covers the schema auto-fetch logic in GrpcHandler and AsyncGrpcHandler
when search data contains bytes vectors (float16/bfloat16/binary).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pymilvus import DataType
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.grpc_handler import GrpcHandler

MOCK_SCHEMA = {
    "fields": [
        {"name": "id", "type": DataType.INT64},
        {"name": "vec", "type": DataType.FLOAT16_VECTOR},
    ],
    "update_timestamp": 12345,
}


def _make_grpc_handler():
    """Create a GrpcHandler with mocked internals."""
    handler = GrpcHandler.__new__(GrpcHandler)
    handler._channel = MagicMock()
    handler._address = "localhost:19530"
    handler._schema_dict = {}
    handler._get_schema = MagicMock(return_value=(MOCK_SCHEMA, 12345))
    handler._execute_search = MagicMock(return_value=MagicMock())
    handler._execute_hybrid_search = MagicMock(return_value=MagicMock())
    return handler


def _make_async_grpc_handler():
    """Create an AsyncGrpcHandler with mocked internals."""
    handler = AsyncGrpcHandler.__new__(AsyncGrpcHandler)
    handler._channel = MagicMock()
    handler._address = "localhost:19530"
    handler._schema_dict = {}
    handler._get_schema = AsyncMock(return_value=(MOCK_SCHEMA, 12345))
    handler._execute_search = AsyncMock(return_value=MagicMock())
    handler._execute_hybrid_search = AsyncMock(return_value=MagicMock())
    handler.ensure_channel_ready = AsyncMock()
    return handler


class TestGrpcHandlerSearchBytesVector:
    """Test GrpcHandler.search auto-fetches schema for bytes vectors."""

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_search_bytes_data_fetches_schema(self, mock_ts_utils):
        """When data[0] is bytes and no schema kwarg, _get_schema is called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=[b"\x01\x02\x03\x04"],
        )

        handler._get_schema.assert_called_once()

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_search_bytes_data_with_existing_schema_skips_fetch(self, mock_ts_utils):
        """When schema kwarg already present, _get_schema is NOT called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=[b"\x01\x02\x03\x04"],
            schema=MOCK_SCHEMA,
        )

        handler._get_schema.assert_not_called()

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_search_float_data_skips_schema_fetch(self, mock_ts_utils):
        """When data is float vectors, _get_schema is NOT called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=[[1.0, 2.0, 3.0]],
        )

        handler._get_schema.assert_not_called()


class TestGrpcHandlerSearchNumpyArray:
    """Test GrpcHandler.search does not raise ValueError with numpy array data."""

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_search_numpy_array_does_not_raise(self, mock_ts_utils):
        """When data is a numpy array, the truthiness check must not raise ValueError."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        data = np.random.default_rng(seed=19530).random((1, 128))
        # This must NOT raise:
        # ValueError: The truth value of an array with more than one element is ambiguous
        handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=data,
        )

        handler._get_schema.assert_not_called()

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_search_numpy_array_multi_vectors(self, mock_ts_utils):
        """Multiple numpy vectors should also work without raising."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        data = np.random.default_rng(seed=42).random((3, 128))
        handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=data,
        )

        handler._get_schema.assert_not_called()


class TestGrpcHandlerHybridSearchBytesVector:
    """Test GrpcHandler.hybrid_search auto-fetches schema for bytes vectors."""

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_hybrid_search_bytes_data_fetches_schema(self, mock_ts_utils):
        """When any req has bytes data, _get_schema is called once."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        req1 = AnnSearchRequest(
            data=[b"\x01\x02\x03\x04"],
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )
        req2 = AnnSearchRequest(
            data=[[1.0, 2.0, 3.0]],
            anns_field="vec2",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        handler.hybrid_search(
            collection_name="test_col",
            reqs=[req1, req2],
            rerank=None,
            limit=10,
        )

        handler._get_schema.assert_called_once()

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_hybrid_search_no_bytes_skips_schema_fetch(self, mock_ts_utils):
        """When no req has bytes data, _get_schema is NOT called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        req = AnnSearchRequest(
            data=[[1.0, 2.0, 3.0]],
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        handler.hybrid_search(
            collection_name="test_col",
            reqs=[req],
            rerank=None,
            limit=10,
        )

        handler._get_schema.assert_not_called()

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_hybrid_search_with_existing_schema_skips_fetch(self, mock_ts_utils):
        """When schema kwarg present, _get_schema is NOT called even with bytes."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        req = AnnSearchRequest(
            data=[b"\x01\x02\x03\x04"],
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        handler.hybrid_search(
            collection_name="test_col",
            reqs=[req],
            rerank=None,
            limit=10,
            schema=MOCK_SCHEMA,
        )

        handler._get_schema.assert_not_called()


class TestAsyncGrpcHandlerSearchBytesVector:
    """Test AsyncGrpcHandler.search auto-fetches schema for bytes vectors."""

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_search_bytes_data_fetches_schema(self, mock_ts_utils):
        """When data[0] is bytes and no schema kwarg, _get_schema is called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        await handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=[b"\x01\x02\x03\x04"],
        )

        handler._get_schema.assert_called_once()

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_search_with_existing_schema_skips_fetch(self, mock_ts_utils):
        """When schema kwarg already present, _get_schema is NOT called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        await handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=[b"\x01\x02\x03\x04"],
            schema=MOCK_SCHEMA,
        )

        handler._get_schema.assert_not_called()

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_search_float_data_skips_schema_fetch(self, mock_ts_utils):
        """When data is float vectors, _get_schema is NOT called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        await handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=[[1.0, 2.0, 3.0]],
        )

        handler._get_schema.assert_not_called()


class TestAsyncGrpcHandlerHybridSearchBytesVector:
    """Test AsyncGrpcHandler.hybrid_search auto-fetches schema for bytes vectors."""

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_hybrid_search_bytes_data_fetches_schema(self, mock_ts_utils):
        """When any req has bytes data, _get_schema is called once."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        req = AnnSearchRequest(
            data=[b"\x01\x02\x03\x04"],
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        await handler.hybrid_search(
            collection_name="test_col",
            reqs=[req],
            rerank=None,
            limit=10,
        )

        handler._get_schema.assert_called_once()

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_hybrid_search_no_bytes_skips_fetch(self, mock_ts_utils):
        """When no req has bytes data, _get_schema is NOT called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        req = AnnSearchRequest(
            data=[[1.0, 2.0, 3.0]],
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        await handler.hybrid_search(
            collection_name="test_col",
            reqs=[req],
            rerank=None,
            limit=10,
        )

        handler._get_schema.assert_not_called()

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_hybrid_search_with_schema_skips_fetch(self, mock_ts_utils):
        """When schema kwarg present, _get_schema is NOT called."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        req = AnnSearchRequest(
            data=[b"\x01\x02\x03\x04"],
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        await handler.hybrid_search(
            collection_name="test_col",
            reqs=[req],
            rerank=None,
            limit=10,
            schema=MOCK_SCHEMA,
        )

        handler._get_schema.assert_not_called()


class TestAsyncGrpcHandlerSearchNumpyArray:
    """Test AsyncGrpcHandler.search does not raise ValueError with numpy array data."""

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_search_numpy_array_does_not_raise(self, mock_ts_utils):
        """When data is a numpy array, the truthiness check must not raise ValueError."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        data = np.random.default_rng(seed=19530).random((1, 128))
        await handler.search(
            collection_name="test_col",
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            data=data,
        )

        handler._get_schema.assert_not_called()


class TestGrpcHandlerHybridSearchNumpyArray:
    """Test GrpcHandler.hybrid_search does not raise ValueError with numpy array data."""

    @patch("pymilvus.client.grpc_handler.ts_utils")
    def test_hybrid_search_numpy_array_does_not_raise(self, mock_ts_utils):
        """When a request has numpy array data, hybrid_search must not raise."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_grpc_handler()

        req = AnnSearchRequest(
            data=np.random.default_rng(seed=19530).random((1, 128)),
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        handler.hybrid_search(
            collection_name="test_col",
            reqs=[req],
            rerank=None,
            limit=10,
        )

        handler._get_schema.assert_not_called()


class TestAsyncGrpcHandlerHybridSearchNumpyArray:
    """Test AsyncGrpcHandler.hybrid_search does not raise ValueError with numpy array data."""

    @pytest.mark.asyncio
    @patch("pymilvus.client.async_grpc_handler.ts_utils")
    async def test_async_hybrid_search_numpy_array_does_not_raise(self, mock_ts_utils):
        """When a request has numpy array data, hybrid_search must not raise."""
        mock_ts_utils.construct_guarantee_ts.return_value = True
        handler = _make_async_grpc_handler()

        req = AnnSearchRequest(
            data=np.random.default_rng(seed=19530).random((1, 128)),
            anns_field="vec",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
        )

        await handler.hybrid_search(
            collection_name="test_col",
            reqs=[req],
            rerank=None,
            limit=10,
        )

        handler._get_schema.assert_not_called()
