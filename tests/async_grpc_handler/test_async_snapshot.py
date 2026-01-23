"""Tests for AsyncGrpcHandler snapshot operations.

Coverage: Snapshot create, drop, list, describe, restore operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler


class TestAsyncGrpcHandlerSnapshot:
    """Tests for snapshot operations."""

    @pytest.mark.asyncio
    async def test_create_snapshot(self) -> None:
        """Test create_snapshot async API"""
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
        mock_stub.CreateSnapshot = AsyncMock(return_value=mock_status)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.create_snapshot_req.return_value = mock_request

            await handler.create_snapshot(
                snapshot_name="test_snapshot",
                collection_name="test_collection",
                description="test description",
                timeout=30,
            )

            mock_prepare.create_snapshot_req.assert_called_once_with(
                snapshot_name="test_snapshot",
                collection_name="test_collection",
                description="test description",
            )

    @pytest.mark.asyncio
    async def test_drop_snapshot(self) -> None:
        """Test drop_snapshot async API"""
        mock_channel = MagicMock()
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

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.drop_snapshot_req.return_value = mock_request

            await handler.drop_snapshot(snapshot_name="test_snapshot", timeout=30)

            mock_prepare.drop_snapshot_req.assert_called_once_with("test_snapshot")

    @pytest.mark.asyncio
    async def test_list_snapshots(self) -> None:
        """Test list_snapshots async API"""
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
        mock_response.snapshots = [
            MagicMock(name="snapshot1", collection_name="coll1"),
            MagicMock(name="snapshot2", collection_name="coll2"),
        ]
        mock_stub.ListSnapshots = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.list_snapshots_req.return_value = mock_request

            result = await handler.list_snapshots(collection_name="test_collection", timeout=30)

            mock_prepare.list_snapshots_req.assert_called_once_with(
                collection_name="test_collection"
            )
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_describe_snapshot(self) -> None:
        """Test describe_snapshot async API"""
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
        mock_response.name = "test_snapshot"
        mock_response.description = "test description"
        mock_response.collection_name = "test_collection"
        mock_response.partition_names = ["_default"]
        mock_response.create_ts = 1234567890
        mock_response.s3_location = "s3://bucket/path"
        mock_stub.DescribeSnapshot = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.describe_snapshot_req.return_value = mock_request

            result = await handler.describe_snapshot(snapshot_name="test_snapshot", timeout=30)

            mock_prepare.describe_snapshot_req.assert_called_once_with("test_snapshot")
            assert result.name == "test_snapshot"
            assert result.description == "test description"
            assert result.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_restore_snapshot(self) -> None:
        """Test restore_snapshot async API"""
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
        mock_response.job_id = 12345
        mock_stub.RestoreSnapshot = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.restore_snapshot_req.return_value = mock_request

            job_id = await handler.restore_snapshot(
                snapshot_name="test_snapshot",
                collection_name="new_collection",
                rewrite_data=False,
                timeout=30,
            )

            mock_prepare.restore_snapshot_req.assert_called_once_with(
                snapshot_name="test_snapshot", collection_name="new_collection", rewrite_data=False
            )
            assert job_id == 12345

    @pytest.mark.asyncio
    async def test_get_restore_snapshot_state(self) -> None:
        """Test get_restore_snapshot_state async API"""
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

        mock_info = MagicMock()
        mock_info.job_id = 12345
        mock_info.snapshot_name = "test_snapshot"
        mock_info.db_name = ""
        mock_info.collection_name = "test_collection"
        mock_info.state = 1
        mock_info.progress = 50
        mock_info.reason = ""
        mock_info.start_time = 1234567890
        mock_info.time_cost = 120
        mock_response.info = mock_info

        mock_stub.GetRestoreSnapshotState = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ), patch(
            "pymilvus.client.async_grpc_handler.milvus_types.RestoreSnapshotState.Name",
            return_value="RestoreSnapshotExecuting",
        ):
            mock_request = MagicMock()
            mock_prepare.get_restore_snapshot_state_req.return_value = mock_request

            result = await handler.get_restore_snapshot_state(job_id=12345, timeout=30)

            mock_prepare.get_restore_snapshot_state_req.assert_called_once_with(12345)
            assert result.state == "RestoreSnapshotExecuting"
            assert result.progress == 50
            assert result.time_cost == 120

    @pytest.mark.asyncio
    async def test_list_restore_snapshot_jobs(self) -> None:
        """Test list_restore_snapshot_jobs async API"""
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
        mock_job1 = MagicMock(
            job_id=1,
            snapshot_name="snap1",
            db_name="",
            collection_name="col1",
            state=2,
            progress=100,
            reason="",
            start_time=1234567890,
            time_cost=1000,
        )
        mock_job2 = MagicMock(
            job_id=2,
            snapshot_name="snap2",
            db_name="",
            collection_name="col2",
            state=1,
            progress=50,
            reason="",
            start_time=1234567890,
            time_cost=500,
        )
        mock_response.jobs = [mock_job1, mock_job2]
        mock_stub.ListRestoreSnapshotJobs = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ), patch(
            "pymilvus.client.async_grpc_handler.milvus_types.RestoreSnapshotState.Name",
            side_effect=lambda x: f"State{x}",
        ):
            mock_request = MagicMock()
            mock_prepare.list_restore_snapshot_jobs_req.return_value = mock_request

            result = await handler.list_restore_snapshot_jobs(
                collection_name="test_collection", timeout=30
            )

            mock_prepare.list_restore_snapshot_jobs_req.assert_called_once_with(
                collection_name="test_collection"
            )
            assert len(result) == 2
            assert result[0].job_id == 1
            assert result[1].job_id == 2
