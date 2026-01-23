"""Tests for AsyncGrpcHandler collection operations.

Coverage: Collection create/drop/load/release, schema caching, collection properties.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.cache import GlobalCache
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2


class TestAsyncGrpcHandlerCollection:
    """Tests for AsyncGrpcHandler collection operations."""

    @pytest.mark.asyncio
    async def test_create_collection(self) -> None:
        """Test create_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.CreateCollection = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.create_collection_request.return_value = MagicMock()
            await handler.create_collection("test_coll", fields=[])
            mock_stub.CreateCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_collection(self) -> None:
        """Test drop_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.DropCollection = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.drop_collection_request.return_value = MagicMock()
            await handler.drop_collection("test_coll")
            mock_stub.DropCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_truncate_collection(self) -> None:
        """Test truncate_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_stub.TruncateCollection = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.truncate_collection_request.return_value = MagicMock()
            await handler.truncate_collection("test_coll", timeout=30)
            mock_stub.TruncateCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_collection(self) -> None:
        """Test load_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()
        handler.wait_for_loading_collection = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.LoadCollection = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_request.refresh = False
            mock_prepare.load_collection.return_value = mock_request
            await handler.load_collection("test_coll", timeout=30)
            mock_stub.LoadCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_collection(self) -> None:
        """Test release_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.ReleaseCollection = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.release_collection.return_value = MagicMock()
            await handler.release_collection("test_coll")
            mock_stub.ReleaseCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_loading_collection(self) -> None:
        """Test wait_for_loading_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.get_loading_progress = AsyncMock(return_value=100)

        await handler.wait_for_loading_collection(collection_name="test_coll", timeout=30)
        handler.get_loading_progress.assert_called()

    @pytest.mark.asyncio
    async def test_wait_for_loading_collection_timeout(self) -> None:
        """Test wait_for_loading_collection times out"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.get_loading_progress = AsyncMock(return_value=50)

        with pytest.raises(MilvusException) as exc_info:
            await handler.wait_for_loading_collection(collection_name="test_coll", timeout=0.001)
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_loading_progress(self) -> None:
        """Test get_loading_progress async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.progress = 75
        mock_response.refresh_progress = 80
        mock_stub.GetLoadingProgress = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_loading_progress.return_value = MagicMock()
            result = await handler.get_loading_progress("test_coll")
            assert result == 75

    @pytest.mark.asyncio
    async def test_get_loading_progress_refresh(self) -> None:
        """Test get_loading_progress with is_refresh=True"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.progress = 75
        mock_response.refresh_progress = 80
        mock_stub.GetLoadingProgress = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_loading_progress.return_value = MagicMock()
            result = await handler.get_loading_progress("test_coll", is_refresh=True)
            assert result == 80

    @pytest.mark.asyncio
    async def test_describe_collection(self) -> None:
        """Test describe_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.collection_name = "test_coll"
        mock_response.schema.fields = []
        mock_response.schema.name = "test_coll"
        mock_response.schema.enable_dynamic_field = False
        mock_response.properties = []
        mock_response.shards_num = 1
        mock_response.consistency_level = 0
        mock_response.collectionID = 1
        mock_stub.DescribeCollection = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.is_successful", return_value=True), patch(
            "pymilvus.client.async_grpc_handler.CollectionSchema"
        ) as mock_schema:
            mock_schema.return_value.dict.return_value = {"collection_name": "test_coll"}
            mock_prepare.describe_collection_request.return_value = MagicMock()
            result = await handler.describe_collection("test_coll")
            assert result["collection_name"] == "test_coll"

    @pytest.mark.asyncio
    async def test_has_collection(self) -> None:
        """Test has_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_stub.DescribeCollection = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.is_successful", return_value=True):
            mock_prepare.describe_collection_request.return_value = MagicMock()
            result = await handler.has_collection("test_coll")
            assert result is True

    @pytest.mark.asyncio
    async def test_has_collection_not_found(self) -> None:
        """Test has_collection returns False when collection not found"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = common_pb2.CollectionNotExists
        mock_response.status.reason = ""
        mock_stub.DescribeCollection = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ):
            mock_prepare.describe_collection_request.return_value = MagicMock()
            result = await handler.has_collection("nonexistent")
            assert result is False

    @pytest.mark.asyncio
    async def test_has_collection_unexpected_error_with_reason(self) -> None:
        """Test has_collection returns False for UnexpectedError with 'can't find' reason"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = common_pb2.UnexpectedError
        mock_response.status.reason = "can't find collection"
        mock_stub.DescribeCollection = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ):
            mock_prepare.describe_collection_request.return_value = MagicMock()
            result = await handler.has_collection("nonexistent")
            assert result is False

    @pytest.mark.asyncio
    async def test_has_collection_raises_on_other_error(self) -> None:
        """Test has_collection raises MilvusException for other errors"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 500
        mock_response.status.error_code = 0
        mock_response.status.reason = "internal error"
        mock_stub.DescribeCollection = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.is_successful", return_value=False):
            mock_prepare.describe_collection_request.return_value = MagicMock()
            with pytest.raises(MilvusException):
                await handler.has_collection("test_coll")

    @pytest.mark.asyncio
    async def test_list_collections(self) -> None:
        """Test list_collections async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.collection_names = ["coll1", "coll2"]
        mock_stub.ShowCollections = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.show_collections_request.return_value = MagicMock()
            result = await handler.list_collections()
            assert result == ["coll1", "coll2"]

    @pytest.mark.asyncio
    async def test_get_collection_stats(self) -> None:
        """Test get_collection_stats async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.stats = [MagicMock(key="row_count", value="100")]
        mock_stub.GetCollectionStatistics = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.get_collection_stats_request.return_value = MagicMock()
            result = await handler.get_collection_stats("test_coll")
            assert result is not None

    @pytest.mark.asyncio
    async def test_get_load_state(self) -> None:
        """Test get_load_state async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.state = 3
        mock_stub.GetLoadState = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_load_state.return_value = MagicMock()
            result = await handler.get_load_state("test_coll")
            assert result is not None

    @pytest.mark.asyncio
    async def test_refresh_load(self) -> None:
        """Test refresh_load async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.refresh_progress = 100
        mock_stub.GetLoadingProgress = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_loading_progress.return_value = MagicMock()
            result = await handler.refresh_load("test_coll")
            assert result == 100

    @pytest.mark.asyncio
    async def test_rename_collection(self) -> None:
        """Test rename_collection async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.RenameCollection = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.rename_collections_request.return_value = MagicMock()
            await handler.rename_collection("old_name", "new_name")
            mock_stub.RenameCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_describe_replica(self) -> None:
        """Test describe_replica async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()
        handler._get_schema = AsyncMock(return_value=({"collection_id": 1}, 0))

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_replica = MagicMock()
        mock_replica.replicaID = 1
        mock_replica.node_ids = [1, 2]
        mock_replica.resource_group_name = "default"
        mock_replica.num_outbound_node = {}
        mock_shard = MagicMock()
        mock_shard.dm_channel_name = "channel1"
        mock_shard.node_ids = [1]
        mock_shard.leaderID = 1
        mock_replica.shard_replicas = [mock_shard]
        mock_response.replicas = [mock_replica]
        mock_stub.GetReplicas = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_replicas.return_value = MagicMock()
            result = await handler.describe_replica("test_coll")
            assert len(result) == 1


class TestAsyncGrpcHandlerSchemaCache:
    """Tests for schema caching operations."""

    @pytest.mark.asyncio
    async def test_get_schema_cached(self) -> None:
        """Test _get_schema uses cache when available"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel, address="localhost:19530")
        handler._is_channel_ready = True

        cached_schema = {"fields": [], "update_timestamp": 100}
        GlobalCache.schema.set("localhost:19530", "", "test_coll", cached_schema)

        try:
            schema, ts = await handler._get_schema("test_coll")
            assert schema == cached_schema
            assert ts == 100
        finally:
            GlobalCache.schema.invalidate("localhost:19530", "", "test_coll")

    @pytest.mark.asyncio
    async def test_get_schema_fetch_on_miss(self) -> None:
        """Test _get_schema fetches from server on cache miss"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel, address="localhost:19530")
        handler._is_channel_ready = True

        remote_schema = {"fields": [], "update_timestamp": 200}
        handler.describe_collection = AsyncMock(return_value=remote_schema)

        GlobalCache.schema.invalidate("localhost:19530", "", "cache_miss_coll")

        try:
            schema, ts = await handler._get_schema("cache_miss_coll")
            assert schema == remote_schema
            assert ts == 200
            handler.describe_collection.assert_called_once()
        finally:
            GlobalCache.schema.invalidate("localhost:19530", "", "cache_miss_coll")

    @pytest.mark.asyncio
    async def test_get_info(self) -> None:
        """Test _get_info method"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        schema = {
            "fields": [{"name": "id", "type": 5}],
            "struct_array_fields": [],
            "enable_dynamic_field": True,
            "update_timestamp": 100,
        }
        handler._get_schema = AsyncMock(return_value=(schema, 100))

        fields, _struct_fields, enable_dynamic, ts = await handler._get_info("test_coll")
        assert fields == schema["fields"]
        assert enable_dynamic is True
        assert ts == 100

    def test_invalidate_schema(self) -> None:
        """Test _invalidate_schema method"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel, address="localhost:19530")

        GlobalCache.schema.set("localhost:19530", "", "test_coll", {"fields": []})
        handler._invalidate_schema("test_coll")
        assert GlobalCache.schema.get("localhost:19530", "", "test_coll") is None

    def test_invalidate_db_schemas(self) -> None:
        """Test _invalidate_db_schemas method"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel, address="localhost:19530")

        GlobalCache.schema.set("localhost:19530", "test_db", "coll1", {"fields": []})
        GlobalCache.schema.set("localhost:19530", "test_db", "coll2", {"fields": []})
        handler._invalidate_db_schemas("test_db")
        assert GlobalCache.schema.get("localhost:19530", "test_db", "coll1") is None
        assert GlobalCache.schema.get("localhost:19530", "test_db", "coll2") is None


class TestAsyncGrpcHandlerCollectionProperties:
    """Tests for collection property operations."""

    @pytest.mark.asyncio
    async def test_alter_collection_properties(self) -> None:
        """Test alter_collection_properties async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterCollection = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.alter_collection_request.return_value = MagicMock()
            await handler.alter_collection_properties("test_coll", {"key": "value"})
            mock_stub.AlterCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_collection_properties(self) -> None:
        """Test drop_collection_properties async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterCollection = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.alter_collection_request.return_value = MagicMock()
            await handler.drop_collection_properties("test_coll", ["key1", "key2"])
            mock_stub.AlterCollection.assert_called_once()

    @pytest.mark.asyncio
    async def test_alter_collection_field(self) -> None:
        """Test alter_collection_field async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterCollectionField = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.alter_collection_field_request.return_value = MagicMock()
            await handler.alter_collection_field("test_coll", "field1", {"mmap": True})
            mock_stub.AlterCollectionField.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_collection_field(self) -> None:
        """Test add_collection_field async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AddCollectionField = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        mock_field_schema = MagicMock()
        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.add_collection_field_request.return_value = MagicMock()
            await handler.add_collection_field("test_coll", mock_field_schema)
            mock_stub.AddCollectionField.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_collection_function(self) -> None:
        """Test drop_collection_function async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.DropCollectionFunction = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.drop_collection_function_request.return_value = MagicMock()
            await handler.drop_collection_function("test_coll", "func1")
            mock_stub.DropCollectionFunction.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_collection_function(self) -> None:
        """Test add_collection_function async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AddCollectionFunction = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        mock_function = MagicMock()
        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.add_collection_function_request.return_value = MagicMock()
            await handler.add_collection_function("test_coll", mock_function)
            mock_stub.AddCollectionFunction.assert_called_once()

    @pytest.mark.asyncio
    async def test_alter_collection_function(self) -> None:
        """Test alter_collection_function async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterCollectionFunction = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        mock_function = MagicMock()
        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.alter_collection_function_request.return_value = MagicMock()
            await handler.alter_collection_function("test_coll", "func1", mock_function)
            mock_stub.AlterCollectionFunction.assert_called_once()
