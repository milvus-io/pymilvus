from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.exceptions import MilvusException


class TestAsyncFlush:
    """Test cases for async flush functionality"""

    @pytest.mark.asyncio
    async def test_flush_waits_for_segments_to_be_flushed(self) -> None:
        """
        Test that async flush() waits for all segments to be flushed before returning.
        
        This test verifies the fix for the bug where async flush() would return
        immediately after the RPC call, without waiting for segments to be actually flushed.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock the async stub
        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Create mock flush response with segment IDs and flush timestamp
        mock_flush_response = MagicMock()
        mock_flush_status = MagicMock()
        mock_flush_status.code = 0
        mock_flush_status.error_code = 0
        mock_flush_status.reason = ""
        mock_flush_response.status = mock_flush_status
        
        # Mock collection segment IDs and flush timestamp
        mock_seg_ids = MagicMock()
        mock_seg_ids.data = [1, 2, 3]  # Segment IDs
        mock_flush_ts = 12345  # Flush timestamp
        
        mock_flush_response.coll_segIDs = {"test_collection": mock_seg_ids}
        mock_flush_response.coll_flush_ts = {"test_collection": mock_flush_ts}
        
        mock_stub.Flush = AsyncMock(return_value=mock_flush_response)

        # Mock get_flush_state to return False first, then True (simulating flush completion)
        call_count = 0
        async def mock_get_flush_state(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return False first time (not flushed), True second time (flushed)
            return call_count > 1
        
        handler.get_flush_state = AsyncMock(side_effect=mock_get_flush_state)

        # Mock Prepare.flush_param
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_prepare.flush_param.return_value = MagicMock()

            # Call flush
            result = await handler.flush(["test_collection"], timeout=10)

            # Verify Flush RPC was called
            mock_stub.Flush.assert_called_once()
            
            # Verify get_flush_state was called (waiting for flush to complete)
            assert handler.get_flush_state.call_count >= 1, \
                "get_flush_state should be called to wait for segments to be flushed"
            
            # Verify the response is returned
            assert result == mock_flush_response

    @pytest.mark.asyncio
    async def test_flush_waits_for_multiple_collections(self) -> None:
        """
        Test that async flush() waits for all collections' segments to be flushed.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Create mock flush response for multiple collections
        mock_flush_response = MagicMock()
        mock_flush_status = MagicMock()
        mock_flush_status.code = 0
        mock_flush_status.error_code = 0
        mock_flush_status.reason = ""
        mock_flush_response.status = mock_flush_status
        
        # Mock segment IDs and flush timestamps for two collections
        mock_seg_ids_1 = MagicMock()
        mock_seg_ids_1.data = [1, 2]
        mock_seg_ids_2 = MagicMock()
        mock_seg_ids_2.data = [3, 4]
        
        mock_flush_response.coll_segIDs = {
            "collection1": mock_seg_ids_1,
            "collection2": mock_seg_ids_2
        }
        mock_flush_response.coll_flush_ts = {
            "collection1": 12345,
            "collection2": 12346
        }
        
        mock_stub.Flush = AsyncMock(return_value=mock_flush_response)

        # Track which collections were checked
        checked_collections = []

        async def mock_get_flush_state(segment_ids, collection_name, flush_ts, timeout=None, **kwargs):
            checked_collections.append(collection_name)
            return True  # Always return True (already flushed)
        
        handler.get_flush_state = AsyncMock(side_effect=mock_get_flush_state)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_prepare.flush_param.return_value = MagicMock()

            await handler.flush(["collection1", "collection2"], timeout=10)

            # Verify both collections were checked
            assert "collection1" in checked_collections, \
                "collection1 should be checked for flush completion"
            assert "collection2" in checked_collections, \
                "collection2 should be checked for flush completion"
            assert handler.get_flush_state.call_count == 2, \
                "get_flush_state should be called for each collection"

    @pytest.mark.asyncio
    async def test_flush_timeout(self) -> None:
        """
        Test that async flush() raises timeout exception when segments don't flush in time.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Create mock flush response
        mock_flush_response = MagicMock()
        mock_flush_status = MagicMock()
        mock_flush_status.code = 0
        mock_flush_status.error_code = 0
        mock_flush_status.reason = ""
        mock_flush_response.status = mock_flush_status
        
        mock_seg_ids = MagicMock()
        mock_seg_ids.data = [1, 2, 3]
        
        mock_flush_response.coll_segIDs = {"test_collection": mock_seg_ids}
        mock_flush_response.coll_flush_ts = {"test_collection": 12345}
        
        mock_stub.Flush = AsyncMock(return_value=mock_flush_response)

        # Mock get_flush_state to always return False (never flushed)
        handler.get_flush_state = AsyncMock(return_value=False)

        # Mock time to simulate timeout
        import time
        original_time = time.time
        start_time = 1000.0
        current_time = start_time
        
        def mock_time():
            nonlocal current_time
            # Increment time by 0.6 seconds each call to exceed timeout
            current_time += 0.6
            return current_time
        
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}), \
             patch('pymilvus.client.async_grpc_handler.time.time', side_effect=mock_time):
            mock_prepare.flush_param.return_value = MagicMock()

            # Call flush with short timeout
            with pytest.raises(MilvusException) as exc_info:
                await handler.flush(["test_collection"], timeout=0.5)

            # Verify timeout exception was raised
            assert "wait for flush timeout" in str(exc_info.value).lower(), \
                "Should raise timeout exception when flush takes too long"

    @pytest.mark.asyncio
    async def test_flush_parameter_validation(self) -> None:
        """
        Test that async flush() validates parameters correctly.
        """
        # Setup mock channel
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Test empty collection names
        with pytest.raises(Exception):  # Should raise ParamError
            await handler.flush([])

        # Test None collection names
        with pytest.raises(Exception):  # Should raise ParamError
            await handler.flush(None)  # type: ignore

        # Test invalid type
        with pytest.raises(Exception):  # Should raise ParamError
            await handler.flush("not_a_list")  # type: ignore

    @pytest.mark.asyncio
    async def test_get_flush_state(self) -> None:
        """
        Test the get_flush_state() method.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Create mock GetFlushState response
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_response.flushed = True
        
        mock_stub.GetFlushState = AsyncMock(return_value=mock_response)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_prepare.get_flush_state_request.return_value = MagicMock()

            # Call get_flush_state
            result = await handler.get_flush_state(
                segment_ids=[1, 2, 3],
                collection_name="test_collection",
                flush_ts=12345,
                timeout=10
            )

            # Verify GetFlushState RPC was called
            mock_stub.GetFlushState.assert_called_once()
            
            # Verify result
            assert result is True, "get_flush_state should return the flushed status"

