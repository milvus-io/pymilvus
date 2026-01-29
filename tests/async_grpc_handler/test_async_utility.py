"""Tests for AsyncGrpcHandler utility operations.

Coverage: Server version, compaction, analyzer, replica operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler


class TestAsyncGrpcHandlerUtility:
    """Tests for utility operations."""

    @pytest.mark.asyncio
    async def test_get_server_version(self) -> None:
        """Test get_server_version async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.version = "v2.4.0"
        mock_stub.GetVersion = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result = await handler.get_server_version(timeout=30)

            assert result == "v2.4.0"

    @pytest.mark.asyncio
    async def test_get_server_version_with_detail(self) -> None:
        """Test get_server_version returns server info dict when detail=True"""
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

        mock_server_info = MagicMock()
        mock_server_info.build_tags = "2.6.6"
        mock_server_info.build_time = "Fri Jan 23 03:05:45 UTC 2026"
        mock_server_info.git_commit = "cebbe1e4da"
        mock_server_info.go_version = "go version go1.24.11 linux/amd64"
        mock_server_info.deploy_mode = "STANDALONE"
        mock_response.server_info = mock_server_info

        mock_stub.Connect = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result = await handler.get_server_version(detail=True)

        expected = {
            "version": "2.6.6",
            "build_time": "Fri Jan 23 03:05:45 UTC 2026",
            "git_commit": "cebbe1e4da",
            "go_version": "go version go1.24.11 linux/amd64",
            "deploy_mode": "STANDALONE",
        }
        assert result == expected
        mock_stub.Connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_server_version_with_detail_uses_cache(self) -> None:
        """Test get_server_version caches server info and returns cached value"""
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

        mock_server_info = MagicMock()
        mock_server_info.build_tags = "2.6.6"
        mock_server_info.build_time = "Fri Jan 23 03:05:45 UTC 2026"
        mock_server_info.git_commit = "cebbe1e4da"
        mock_server_info.go_version = "go version go1.24.11 linux/amd64"
        mock_server_info.deploy_mode = "STANDALONE"
        mock_response.server_info = mock_server_info

        mock_stub.Connect = AsyncMock(return_value=mock_response)

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result1 = await handler.get_server_version(detail=True)
            result2 = await handler.get_server_version(detail=True)

        assert result1 == result2
        assert mock_stub.Connect.call_count == 1

    @pytest.mark.asyncio
    async def test_run_analyzer(self) -> None:
        """Test run_analyzer async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.token_results = []
        mock_stub.RunAnalyzer = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.run_analyzer_request.return_value = MagicMock()
            result = await handler.run_analyzer("hello world")
            assert result is not None


class TestAsyncGrpcHandlerCompaction:
    """Tests for compaction operations."""

    @pytest.mark.asyncio
    async def test_compact(self) -> None:
        """Test compact async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.compactionID = 12345
        mock_stub.ManualCompaction = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.describe_collection_request.return_value = mock_request

            handler.describe_collection = AsyncMock(return_value={"collection_id": 1})

            result = await handler.compact("test_collection", timeout=30)

            assert result == 12345

    @pytest.mark.asyncio
    async def test_compact_with_describe(self) -> None:
        """Test compact async API with describe_collection"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_desc_resp = MagicMock()
        mock_desc_resp.status.code = 0
        mock_desc_resp.collectionID = 12345
        mock_stub.DescribeCollection = AsyncMock(return_value=mock_desc_resp)
        mock_comp_resp = MagicMock()
        mock_comp_resp.status.code = 0
        mock_comp_resp.compactionID = 67890
        mock_stub.ManualCompaction = AsyncMock(return_value=mock_comp_resp)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_collection_request.return_value = MagicMock()
            mock_prepare.manual_compaction.return_value = MagicMock()
            result = await handler.compact("test_coll")
            assert result == 67890

    @pytest.mark.asyncio
    async def test_get_compaction_state(self) -> None:
        """Test get_compaction_state async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.state = 2  # Completed
        mock_stub.GetCompactionState = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result = await handler.get_compaction_state(12345, timeout=30)

            assert result is not None


class TestAsyncGrpcHandlerReplica:
    """Tests for replica operations."""

    @pytest.mark.asyncio
    async def test_describe_replica(self) -> None:
        """Test describe_replica async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        # Mock _get_schema to avoid describe_collection call
        mock_schema = {"fields": [], "update_timestamp": 100, "collection_id": 100}
        handler._get_schema = AsyncMock(return_value=(mock_schema, 100))

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_replica = MagicMock()
        mock_replica.replicaID = 1
        mock_replica.collectionID = 100
        mock_replica.partition_ids = [1]
        mock_replica.shard_replicas = []
        mock_replica.node_ids = [1, 2]
        mock_replica.resource_group_name = "default"
        mock_replica.num_outbound_node = {}
        mock_response.replicas = [mock_replica]
        mock_stub.GetReplicas = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_replicas.return_value = MagicMock()
            result = await handler.describe_replica("test_coll")
            assert len(result) == 1


class TestAsyncGrpcHandlerSegment:
    """Tests for segment operations."""

    @pytest.mark.asyncio
    async def test_get_persistent_segment_infos(self) -> None:
        """Test get_persistent_segment_infos async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.infos = []
        mock_stub.GetPersistentSegmentInfo = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.get_persistent_segment_info_request.return_value = MagicMock()
            result = await handler.get_persistent_segment_infos("test_coll")
            assert result == []
