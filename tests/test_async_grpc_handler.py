from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.exceptions import MilvusException


class TestAsyncGrpcHandler:
    """Test cases for AsyncGrpcHandler class"""

    @pytest.mark.asyncio
    async def test_load_partitions_refresh_attribute(self) -> None:
        """
        Test that load_partitions correctly accesses request.refresh instead of request.is_refresh.
        This test verifies the fix for issue #2796.
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

        # Create a mock response for LoadPartitions
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        # Mock wait_for_loading_partitions to avoid actual waiting
        handler.wait_for_loading_partitions = AsyncMock()

        # Mock Prepare.load_partitions to return a request with refresh attribute
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            # Create mock request with refresh attribute (not is_refresh)
            mock_request = MagicMock()
            mock_request.refresh = True  # This is the correct attribute name
            mock_prepare.load_partitions.return_value = mock_request

            # Call load_partitions
            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                replica_number=1,
                timeout=30,
                refresh=True
            )

            # Verify that Prepare.load_partitions was called correctly
            mock_prepare.load_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                replica_number=1,
                refresh=True
            )

            # Verify that wait_for_loading_partitions was called with is_refresh parameter
            # correctly set from request.refresh (not request.is_refresh)
            handler.wait_for_loading_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                is_refresh=True,  # Should be the value from request.refresh
                timeout=30,
                refresh=True
            )

    @pytest.mark.asyncio
    async def test_load_partitions_without_refresh(self) -> None:
        """Test load_partitions when refresh parameter is not provided"""
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

        # Create a mock response for LoadPartitions
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        # Mock wait_for_loading_partitions
        handler.wait_for_loading_partitions = AsyncMock()

        # Mock Prepare.load_partitions
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            # Create mock request with default refresh value
            mock_request = MagicMock()
            mock_request.refresh = False  # Default value when not specified
            mock_prepare.load_partitions.return_value = mock_request

            # Call load_partitions without refresh parameter
            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                timeout=30
            )

            # Verify that wait_for_loading_partitions was called with is_refresh=False
            handler.wait_for_loading_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1"],
                is_refresh=False,  # Should be False when not specified
                timeout=30
            )

    @pytest.mark.asyncio
    async def test_wait_for_loading_partitions(self) -> None:
        """Test wait_for_loading_partitions method"""
        # Setup mock channel
        mock_channel = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock get_loading_progress to return 100 immediately
        handler.get_loading_progress = AsyncMock(return_value=100)

        # Call wait_for_loading_partitions
        await handler.wait_for_loading_partitions(
            collection_name="test_collection",
            partition_names=["partition1", "partition2"],
            is_refresh=True,
            timeout=30
        )

        # Verify that get_loading_progress was called
        handler.get_loading_progress.assert_called_once_with(
            "test_collection",
            ["partition1", "partition2"],
            timeout=30,
            is_refresh=True
        )

    @pytest.mark.asyncio
    async def test_wait_for_loading_partitions_timeout(self) -> None:
        """Test that wait_for_loading_partitions raises exception on timeout"""

        # Setup mock channel
        mock_channel = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock get_loading_progress to always return less than 100
        handler.get_loading_progress = AsyncMock(return_value=50)

        # Call wait_for_loading_partitions with very short timeout
        with pytest.raises(MilvusException) as exc_info:
            await handler.wait_for_loading_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                is_refresh=False,
                timeout=0.001  # Very short timeout to trigger timeout error
            )

        assert "wait for loading partition timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_partitions_with_resource_groups(self) -> None:
        """Test load_partitions with additional parameters like resource_groups"""
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

        # Create a mock response for LoadPartitions
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        # Mock wait_for_loading_partitions
        handler.wait_for_loading_partitions = AsyncMock()

        # Mock Prepare.load_partitions
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            # Create mock request
            mock_request = MagicMock()
            mock_request.refresh = False
            mock_prepare.load_partitions.return_value = mock_request

            # Call load_partitions with resource_groups
            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                replica_number=2,
                resource_groups=["rg1", "rg2"],
                timeout=30
            )

            # Verify that Prepare.load_partitions was called with resource_groups
            mock_prepare.load_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1"],
                replica_number=2,
                resource_groups=["rg1", "rg2"]
            )
