from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.exceptions import MilvusException, ParamError


@pytest.fixture
def handler():
    """Create an AsyncGrpcHandler with a mocked channel."""
    mock_channel = MagicMock()
    mock_channel.channel_ready = AsyncMock()
    mock_channel.close = AsyncMock()
    mock_channel._unary_unary_interceptors = []

    h = AsyncGrpcHandler(channel=mock_channel)
    h._is_channel_ready = True

    mock_stub = AsyncMock()
    h._async_stub = mock_stub
    return h


def make_flush_response(seg_ids_map, flush_ts_map):
    """Build a mock flush response with given segment IDs and flush timestamps."""
    mock_flush_response = MagicMock()
    mock_flush_status = MagicMock()
    mock_flush_status.code = 0
    mock_flush_status.error_code = 0
    mock_flush_status.reason = ""
    mock_flush_response.status = mock_flush_status
    mock_flush_response.coll_segIDs = seg_ids_map
    mock_flush_response.coll_flush_ts = flush_ts_map
    return mock_flush_response


class TestAsyncFlush:
    """Test cases for async flush functionality"""

    @pytest.mark.asyncio
    async def test_flush_waits_for_segments_to_be_flushed(self, handler) -> None:
        """
        Test that async flush() waits for all segments to be flushed before returning.

        This test verifies the fix for the bug where async flush() would return
        immediately after the RPC call, without waiting for segments to be actually flushed.
        """
        mock_seg_ids = MagicMock()
        mock_seg_ids.data = [1, 2, 3]
        mock_flush_response = make_flush_response(
            {"test_collection": mock_seg_ids},
            {"test_collection": 12345},
        )
        handler._async_stub.Flush = AsyncMock(return_value=mock_flush_response)

        # Mock get_flush_state to return False first, then True (simulating flush completion)
        call_count = 0

        async def mock_get_flush_state(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count > 1

        handler.get_flush_state = AsyncMock(side_effect=mock_get_flush_state)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.flush_param.return_value = MagicMock()

            result = await handler.flush(["test_collection"], timeout=10)

            handler._async_stub.Flush.assert_called_once()
            assert (
                handler.get_flush_state.call_count >= 1
            ), "get_flush_state should be called to wait for segments to be flushed"
            assert result == mock_flush_response

    @pytest.mark.asyncio
    async def test_flush_waits_for_multiple_collections(self, handler) -> None:
        """
        Test that async flush() waits for all collections' segments to be flushed.
        """
        mock_seg_ids_1 = MagicMock()
        mock_seg_ids_1.data = [1, 2]
        mock_seg_ids_2 = MagicMock()
        mock_seg_ids_2.data = [3, 4]
        mock_flush_response = make_flush_response(
            {"collection1": mock_seg_ids_1, "collection2": mock_seg_ids_2},
            {"collection1": 12345, "collection2": 12346},
        )
        handler._async_stub.Flush = AsyncMock(return_value=mock_flush_response)

        checked_collections = []

        async def mock_get_flush_state(
            segment_ids, collection_name, flush_ts, timeout=None, context=None, **kwargs
        ):
            checked_collections.append(collection_name)
            return True

        handler.get_flush_state = AsyncMock(side_effect=mock_get_flush_state)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.flush_param.return_value = MagicMock()

            await handler.flush(["collection1", "collection2"], timeout=10)

            assert "collection1" in checked_collections, "collection1 should be checked"
            assert "collection2" in checked_collections, "collection2 should be checked"
            assert handler.get_flush_state.call_count == 2

    @pytest.mark.asyncio
    async def test_flush_timeout(self, handler) -> None:
        """
        Test that async flush() raises timeout exception when segments don't flush in time.
        """
        mock_seg_ids = MagicMock()
        mock_seg_ids.data = [1, 2, 3]
        mock_flush_response = make_flush_response(
            {"test_collection": mock_seg_ids},
            {"test_collection": 12345},
        )
        handler._async_stub.Flush = AsyncMock(return_value=mock_flush_response)
        handler.get_flush_state = AsyncMock(return_value=False)

        start_time = 1000.0
        current_time = start_time

        def mock_time():
            nonlocal current_time
            current_time += 0.6
            return current_time

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"), patch(
            "pymilvus.client.async_grpc_handler.time.time", side_effect=mock_time
        ):
            mock_prepare.flush_param.return_value = MagicMock()

            with pytest.raises(MilvusException) as exc_info:
                await handler.flush(["test_collection"], timeout=0.5)

            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_flush_parameter_validation(self, handler) -> None:
        """
        Test that async flush() validates parameters correctly.
        """
        with pytest.raises(ParamError):
            await handler.flush([])

        with pytest.raises(ParamError):
            await handler.flush(None)

        with pytest.raises(ParamError):
            await handler.flush("not_a_list")

    @pytest.mark.asyncio
    async def test_get_flush_state(self, handler) -> None:
        """
        Test the get_flush_state() method.
        """
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_response.flushed = True

        handler._async_stub.GetFlushState = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_flush_state_request.return_value = MagicMock()

            result = await handler.get_flush_state(
                segment_ids=[1, 2, 3], collection_name="test_collection", flush_ts=12345, timeout=10
            )

            handler._async_stub.GetFlushState.assert_called_once()
            assert result is True, "get_flush_state should return the flushed status"
