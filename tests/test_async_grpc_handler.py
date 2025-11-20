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

    @pytest.mark.asyncio
    async def test_create_snapshot(self) -> None:
        """Test create_snapshot async API"""
        mock_channel = AsyncMock()
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
        mock_stub.CreateSnapshot = AsyncMock(return_value=mock_status)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_request = MagicMock()
            mock_prepare.create_snapshot_req.return_value = mock_request

            await handler.create_snapshot(
                snapshot_name="test_snapshot",
                db_name="",
                collection_name="test_collection",
                description="test description",
                timeout=30
            )

            mock_prepare.create_snapshot_req.assert_called_once_with(
                snapshot_name="test_snapshot",
                db_name="",
                collection_name="test_collection",
                description="test description"
            )

    @pytest.mark.asyncio
    async def test_drop_snapshot(self) -> None:
        """Test drop_snapshot async API"""
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.DropSnapshot = AsyncMock(return_value=mock_status)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_request = MagicMock()
            mock_prepare.drop_snapshot_req.return_value = mock_request

            await handler.drop_snapshot(snapshot_name="test_snapshot", timeout=30)

            mock_prepare.drop_snapshot_req.assert_called_once_with("test_snapshot")

    @pytest.mark.asyncio
    async def test_list_snapshots(self) -> None:
        """Test list_snapshots async API"""
        mock_channel = AsyncMock()
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
        mock_response.snapshots = [
            MagicMock(name="snapshot1", collection_name="coll1"),
            MagicMock(name="snapshot2", collection_name="coll2")
        ]
        mock_stub.ListSnapshots = AsyncMock(return_value=mock_response)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_request = MagicMock()
            mock_prepare.list_snapshots_req.return_value = mock_request

            result = await handler.list_snapshots(db_name="", collection_name="test_collection", timeout=30)

            mock_prepare.list_snapshots_req.assert_called_once_with(
                db_name="",
                collection_name="test_collection"
            )
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_describe_snapshot(self) -> None:
        """Test describe_snapshot async API"""
        mock_channel = AsyncMock()
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
        mock_response.name = "test_snapshot"
        mock_response.description = "test description"
        mock_response.collection_name = "test_collection"
        mock_response.partition_names = ["_default"]
        mock_response.create_ts = 1234567890
        mock_response.s3_location = "s3://bucket/path"
        mock_stub.DescribeSnapshot = AsyncMock(return_value=mock_response)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_request = MagicMock()
            mock_prepare.describe_snapshot_req.return_value = mock_request

            result = await handler.describe_snapshot(snapshot_name="test_snapshot", timeout=30)

            mock_prepare.describe_snapshot_req.assert_called_once_with("test_snapshot")
            assert result["name"] == "test_snapshot"
            assert result["description"] == "test description"
            assert result["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_restore_snapshot(self) -> None:
        """Test restore_snapshot async API"""
        mock_channel = AsyncMock()
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
        mock_response.job_id = 12345
        mock_stub.RestoreSnapshot = AsyncMock(return_value=mock_response)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_request = MagicMock()
            mock_prepare.restore_snapshot_req.return_value = mock_request

            job_id = await handler.restore_snapshot(
                snapshot_name="test_snapshot",
                db_name="",
                collection_name="new_collection",
                rewrite_data=False,
                timeout=30
            )

            mock_prepare.restore_snapshot_req.assert_called_once_with(
                snapshot_name="test_snapshot",
                db_name="",
                collection_name="new_collection",
                rewrite_data=False
            )
            assert job_id == 12345

    @pytest.mark.asyncio
    async def test_get_restore_snapshot_state(self) -> None:
        """Test get_restore_snapshot_state async API"""
        mock_channel = AsyncMock()
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

        # Mock response.info object
        mock_info = MagicMock()
        mock_info.job_id = 12345
        mock_info.snapshot_name = "test_snapshot"
        mock_info.db_name = ""
        mock_info.collection_name = "test_collection"
        mock_info.state = "InProgress"
        mock_info.progress = 50
        mock_info.reason = ""
        mock_info.start_time = 1234567890
        mock_info.time_cost = 120
        mock_response.info = mock_info

        mock_stub.GetRestoreSnapshotState = AsyncMock(return_value=mock_response)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_request = MagicMock()
            mock_prepare.get_restore_snapshot_state_req.return_value = mock_request

            result = await handler.get_restore_snapshot_state(job_id=12345, timeout=30)

            mock_prepare.get_restore_snapshot_state_req.assert_called_once_with(12345)
            assert result["state"] == "InProgress"
            assert result["progress"] == 50
            assert result["time_cost"] == 120

    @pytest.mark.asyncio
    async def test_list_restore_snapshot_jobs(self) -> None:
        """Test list_restore_snapshot_jobs async API"""
        mock_channel = AsyncMock()
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
        mock_job1 = MagicMock(job_id=1, snapshot_name="snap1")
        mock_job2 = MagicMock(job_id=2, snapshot_name="snap2")
        mock_response.jobs = [mock_job1, mock_job2]
        mock_stub.ListRestoreSnapshotJobs = AsyncMock(return_value=mock_response)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            mock_request = MagicMock()
            mock_prepare.list_restore_snapshot_jobs_req.return_value = mock_request

            result = await handler.list_restore_snapshot_jobs(collection_name="test_collection", timeout=30)

            mock_prepare.list_restore_snapshot_jobs_req.assert_called_once_with(collection_name="test_collection")
            assert len(result) == 2
