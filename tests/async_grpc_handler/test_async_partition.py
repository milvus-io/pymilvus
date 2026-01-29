"""Tests for AsyncGrpcHandler partition operations.

Coverage: Partition create, drop, has, list, load, release operations.
"""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.exceptions import MilvusException


class TestAsyncGrpcHandlerPartition:
    """Tests for partition operations."""

    @pytest.mark.asyncio
    async def test_create_partition(self) -> None:
        """Test create_partition async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.CreatePartition = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.create_partition_request.return_value = mock_request

            await handler.create_partition("test_collection", "test_partition", timeout=30)

            mock_stub.CreatePartition.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_partition(self) -> None:
        """Test drop_partition async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.DropPartition = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.drop_partition_request.return_value = mock_request

            await handler.drop_partition("test_collection", "test_partition", timeout=30)

            mock_stub.DropPartition.assert_called_once()

    @pytest.mark.asyncio
    async def test_has_partition(self) -> None:
        """Test has_partition async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.value = True
        mock_stub.HasPartition = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.has_partition_request.return_value = mock_request

            result = await handler.has_partition("test_collection", "test_partition", timeout=30)

            assert result is True

    @pytest.mark.asyncio
    async def test_list_partitions(self) -> None:
        """Test list_partitions async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.partition_names = ["_default", "partition1"]
        mock_stub.ShowPartitions = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.show_partitions_request.return_value = mock_request

            result = await handler.list_partitions("test_collection", timeout=30)

            assert result == ["_default", "partition1"]


class TestAsyncGrpcHandlerPartitionLoad:
    """Tests for partition load/release operations."""

    @pytest.mark.asyncio
    async def test_load_partitions_refresh_attribute(self) -> None:
        """Test that load_partitions correctly accesses request.refresh."""
        mock_channel = MagicMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        handler.wait_for_loading_partitions = AsyncMock()

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_request.refresh = True
            mock_prepare.load_partitions.return_value = mock_request

            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                replica_number=1,
                timeout=30,
                refresh=True,
            )

            mock_prepare.load_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                replica_number=1,
                refresh=True,
            )

            handler.wait_for_loading_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                is_refresh=True,
                timeout=30,
                refresh=True,
                context=ANY,
            )

    @pytest.mark.asyncio
    async def test_load_partitions_without_refresh(self) -> None:
        """Test load_partitions when refresh parameter is not provided"""
        mock_channel = MagicMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        handler.wait_for_loading_partitions = AsyncMock()

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_request.refresh = False
            mock_prepare.load_partitions.return_value = mock_request

            await handler.load_partitions(
                collection_name="test_collection", partition_names=["partition1"], timeout=30
            )

            handler.wait_for_loading_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1"],
                is_refresh=False,
                timeout=30,
                context=ANY,
            )

    @pytest.mark.asyncio
    async def test_wait_for_loading_partitions(self) -> None:
        """Test wait_for_loading_partitions method"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        handler.get_loading_progress = AsyncMock(return_value=100)

        await handler.wait_for_loading_partitions(
            collection_name="test_collection",
            partition_names=["partition1", "partition2"],
            is_refresh=True,
            timeout=30,
        )

        handler.get_loading_progress.assert_called_once_with(
            "test_collection",
            ["partition1", "partition2"],
            timeout=30,
            is_refresh=True,
            context=ANY,
        )

    @pytest.mark.asyncio
    async def test_wait_for_loading_partitions_timeout(self) -> None:
        """Test that wait_for_loading_partitions raises exception on timeout"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        handler.get_loading_progress = AsyncMock(return_value=50)

        with pytest.raises(MilvusException) as exc_info:
            await handler.wait_for_loading_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                is_refresh=False,
                timeout=0.001,
            )

        assert "wait for loading partition timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_partitions_with_resource_groups(self) -> None:
        """Test load_partitions with additional parameters like resource_groups"""
        mock_channel = MagicMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        handler.wait_for_loading_partitions = AsyncMock()

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_request.refresh = False
            mock_prepare.load_partitions.return_value = mock_request

            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                replica_number=2,
                resource_groups=["rg1", "rg2"],
                timeout=30,
            )

            mock_prepare.load_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1"],
                replica_number=2,
                resource_groups=["rg1", "rg2"],
            )

    @pytest.mark.asyncio
    async def test_release_partitions(self) -> None:
        """Test release_partitions async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.ReleasePartitions = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.release_partitions_request.return_value = MagicMock()
            await handler.release_partitions("test_coll", ["partition1"])
            mock_stub.ReleasePartitions.assert_called_once()
