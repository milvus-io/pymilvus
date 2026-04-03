"""Tests for AsyncGrpcHandler external collection operations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.types import RefreshExternalCollectionJobInfo


def _make_async_handler():
    """Create an AsyncGrpcHandler with mocked channel and stub."""
    mock_channel = MagicMock()
    mock_channel.channel_ready = AsyncMock()
    mock_channel.close = AsyncMock()
    mock_channel._unary_unary_interceptors = []

    handler = AsyncGrpcHandler(channel=mock_channel)
    handler._is_channel_ready = True

    mock_stub = AsyncMock()
    handler._async_stub = mock_stub
    return handler, mock_stub


def _make_job_info_pb(job_id=1, state=2, progress=100):
    """Create a mock job info protobuf object."""
    info = MagicMock()
    info.job_id = job_id
    info.collection_name = "ext_coll"
    info.state = state
    info.progress = progress
    info.reason = ""
    info.external_source = "s3://bucket/path"
    info.start_time = 1000
    info.end_time = 2000
    return info


class TestAsyncGrpcHandlerExternalCollection:
    """Tests for async external collection refresh operations."""

    @pytest.mark.asyncio
    async def test_refresh_external_collection(self):
        handler, mock_stub = _make_async_handler()

        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_id = 42
        mock_stub.RefreshExternalCollection = AsyncMock(return_value=mock_resp)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.refresh_external_collection_request.return_value = MagicMock()

            result = await handler.refresh_external_collection(
                collection_name="ext_coll",
                timeout=30,
            )

            assert result == 42
            mock_prepare.refresh_external_collection_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_with_new_source(self):
        handler, mock_stub = _make_async_handler()

        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_id = 43
        mock_stub.RefreshExternalCollection = AsyncMock(return_value=mock_resp)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.refresh_external_collection_request.return_value = MagicMock()

            result = await handler.refresh_external_collection(
                collection_name="ext_coll",
                external_source="s3://new-path",
                external_spec='{"format": "iceberg"}',
            )

            assert result == 43

    @pytest.mark.asyncio
    async def test_get_refresh_progress(self):
        handler, mock_stub = _make_async_handler()

        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_info = _make_job_info_pb(job_id=42, state=2, progress=100)
        mock_stub.GetRefreshExternalCollectionProgress = AsyncMock(return_value=mock_resp)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_refresh_external_collection_progress_request.return_value = MagicMock()

            result = await handler.get_refresh_external_collection_progress(job_id=42)

            assert isinstance(result, RefreshExternalCollectionJobInfo)
            assert result.job_id == 42
            assert result.state == "RefreshCompleted"
            assert result.progress == 100

    @pytest.mark.asyncio
    async def test_list_refresh_jobs(self):
        handler, mock_stub = _make_async_handler()

        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.jobs = [
            _make_job_info_pb(job_id=1, state=2, progress=100),
            _make_job_info_pb(job_id=2, state=1, progress=50),
        ]
        mock_stub.ListRefreshExternalCollectionJobs = AsyncMock(return_value=mock_resp)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.list_refresh_external_collection_jobs_request.return_value = MagicMock()

            result = await handler.list_refresh_external_collection_jobs(collection_name="ext_coll")

            assert len(result) == 2
            assert result[0].state == "RefreshCompleted"
            assert result[1].state == "RefreshInProgress"
