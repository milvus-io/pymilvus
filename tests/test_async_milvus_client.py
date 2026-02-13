"""Test cases for AsyncMilvusClient new features"""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from pymilvus import AsyncMilvusClient, DataType
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.connection_manager import AsyncConnectionManager
from pymilvus.client.types import CompactionPlans, LoadState
from pymilvus.orm.collection import Function
from pymilvus.orm.schema import StructFieldSchema


@pytest.fixture(autouse=True)
def reset_async_connection_manager():
    """Reset AsyncConnectionManager singleton before and after each test."""
    AsyncConnectionManager._reset_instance()
    yield
    AsyncConnectionManager._reset_instance()


class TestAsyncMilvusClientNewFeatures:
    """Test cases for newly added features in AsyncMilvusClient"""

    @pytest.mark.asyncio
    async def test_get_server_type(self):
        """Test get_server_type returns correct value"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.get_server_type.return_value = "milvus"

            mock_handler.ensure_channel_ready = AsyncMock()

            client = AsyncMilvusClient()
            await client._connect()
            # get_server_type is called during __init__ for is_self_hosted, and again here
            assert mock_handler.get_server_type.call_count >= 1
            result = client.get_server_type()
            assert result == "milvus"
            # Verify it was called at least once more (during the explicit call)
            assert mock_handler.get_server_type.call_count >= 2

    @pytest.mark.asyncio
    async def test_get_load_state_with_progress_when_loading(self):
        """Test get_load_state returns progress when state is Loading"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.get_load_state = AsyncMock(return_value=LoadState.Loading)
            mock_handler.get_loading_progress = AsyncMock(return_value=75)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.get_load_state("test_collection")

            assert result["state"] == LoadState.Loading
            assert result["progress"] == 75
            mock_handler.get_load_state.assert_called_once()
            mock_handler.get_loading_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_load_state_without_progress_when_not_loading(self):
        """Test get_load_state does not return progress when state is not Loading"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.get_load_state = AsyncMock(return_value=LoadState.Loaded)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.get_load_state("test_collection")

            assert result["state"] == LoadState.Loaded
            assert "progress" not in result
            mock_handler.get_load_state.assert_called_once()
            mock_handler.get_loading_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_describe_collection_with_struct_array_fields(self):
        """Test describe_collection converts struct_array_fields to user-friendly format"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_result = {
                "fields": [{"name": "field1", "type": "INT64"}],
                "struct_array_fields": [
                    {"name": "struct_field", "type": "STRUCT", "element_type": "INT64"}
                ],
            }
            mock_handler.describe_collection = AsyncMock(return_value=mock_result)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.describe_collection("test_collection")

            # Verify struct_array_fields is converted and added to fields
            assert "struct_array_fields" not in result
            assert len(result["fields"]) == 2  # original field + converted struct field
            mock_handler.describe_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_persistent_segments(self):
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()

        # Create a mock segment info
        mock_segment_info = MagicMock()
        mock_segment_info.segmentID = 1001
        mock_segment_info.collectionID = 2001
        mock_segment_info.num_rows = 1000
        mock_segment_info.is_sorted = True
        mock_segment_info.state = 3  # FLUSHED
        mock_segment_info.level = 1
        mock_segment_info.storage_version = 1

        mock_handler.get_persistent_segment_infos = AsyncMock(return_value=[mock_segment_info])

        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            # Initialize AsyncMilvusClient
            client = AsyncMilvusClient(uri="http://localhost:19530")

            await client._connect()

            # Call list_persistent_segments
            result = await client.list_persistent_segments("test_collection")

            # Verify the result
            assert len(result) == 1
            segment_info = result[0]
            assert segment_info.segment_id == 1001
            assert segment_info.collection_id == 2001
            assert segment_info.collection_name == "test_collection"
            assert segment_info.num_rows == 1000
            assert segment_info.is_sorted is True
            assert segment_info.state == 3
            assert segment_info.level == 1
            assert segment_info.storage_version == 1

            # Verify call arguments
            mock_handler.get_persistent_segment_infos.assert_called_once_with(
                "test_collection", timeout=None, context=ANY
            )

    @pytest.mark.asyncio
    async def test_describe_collection_without_struct_array_fields(self):
        """Test describe_collection works normally when no struct_array_fields"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_result = {"fields": [{"name": "field1", "type": "INT64"}]}
            mock_handler.describe_collection = AsyncMock(return_value=mock_result)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.describe_collection("test_collection")

            assert result == mock_result
            assert "struct_array_fields" not in result
            mock_handler.describe_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_ranker(self):
        """Test search method accepts ranker parameter"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.search = AsyncMock(return_value=[[{"id": 1, "distance": 0.1}]])

            mock_ranker = MagicMock(spec=Function)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.search("test_collection", data=[[0.1, 0.2]], ranker=mock_ranker)

            assert result == [[{"id": 1, "distance": 0.1}]]
            # Verify ranker was passed to handler
            call_args = mock_handler.search.call_args
            # ranker is passed as keyword argument
            assert call_args.kwargs.get("ranker") == mock_ranker

    @pytest.mark.asyncio
    async def test_hybrid_search_with_function_ranker(self):
        """Test hybrid_search accepts Function type ranker"""

        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.hybrid_search = AsyncMock(return_value=[[{"id": 1}]])

            mock_function_ranker = MagicMock(spec=Function)
            mock_req = MagicMock(spec=AnnSearchRequest)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.hybrid_search(
                "test_collection", reqs=[mock_req], ranker=mock_function_ranker
            )

            assert result == [[{"id": 1}]]
            # ranker is passed as positional argument (3rd arg), check args[2]
            call_args = mock_handler.hybrid_search.call_args
            assert call_args[0][2] == mock_function_ranker  # ranker is 3rd positional arg

    @pytest.mark.asyncio
    async def test_compact_with_is_l0(self):
        """Test compact method accepts is_l0 parameter"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.compact = AsyncMock(return_value=123)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.compact("test_collection", is_l0=True)

            assert result == 123
            call_args = mock_handler.compact.call_args
            assert call_args[1]["is_l0"] is True

    @pytest.mark.asyncio
    async def test_flush_all(self):
        """Test flush_all method"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.flush_all = AsyncMock()

            client = AsyncMilvusClient()
            await client._connect()
            await client.flush_all(timeout=10)
            mock_handler.flush_all.assert_called_once_with(timeout=10, context=ANY)

    @pytest.mark.asyncio
    async def test_get_flush_all_state(self):
        """Test get_flush_all_state method"""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.get_flush_all_state = AsyncMock(return_value=True)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.get_flush_all_state(timeout=10)

            assert result is True
            mock_handler.get_flush_all_state.assert_called_once_with(timeout=10, context=ANY)

    @pytest.mark.asyncio
    async def test_get_compaction_plans(self):
        """Test get_compaction_plans method"""

        mock_plans = MagicMock(spec=CompactionPlans)

        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.get_compaction_plans = AsyncMock(return_value=mock_plans)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.get_compaction_plans(job_id=123, timeout=10)

            assert result == mock_plans
            mock_handler.get_compaction_plans.assert_called_once_with(123, timeout=10, context=ANY)

    @pytest.mark.asyncio
    async def test_update_replicate_configuration(self):
        """Test update_replicate_configuration method"""
        clusters = [
            {
                "cluster_id": "cluster1",
                "connection_param": {"uri": "http://localhost:19530", "token": "token1"},
            }
        ]
        cross_cluster_topology = [
            {"source_cluster_id": "cluster1", "target_cluster_id": "cluster2"}
        ]

        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_status = MagicMock()
            mock_handler.update_replicate_configuration = AsyncMock(return_value=mock_status)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.update_replicate_configuration(
                clusters=clusters, cross_cluster_topology=cross_cluster_topology, timeout=10
            )

            assert result == mock_status
            mock_handler.update_replicate_configuration.assert_called_once_with(
                clusters=clusters,
                cross_cluster_topology=cross_cluster_topology,
                timeout=10,
                context=ANY,
            )

    def test_create_struct_field_schema(self):
        """Test create_struct_field_schema class method"""
        result = AsyncMilvusClient.create_struct_field_schema()
        assert isinstance(result, StructFieldSchema)

    @pytest.mark.parametrize(
        "uri, db_name, expected_db_name",
        [
            # Issue #3236: db_name passed in URI path should be used when no explicit db_name
            ("http://localhost:19530/test_db", "", "test_db"),
            ("http://localhost:19530/production_db", "", "production_db"),
            ("https://localhost:19530/test_db", "", "test_db"),
            ("http://localhost:19530/mydb", "", "mydb"),
            # URI ending with slash should still extract db_name correctly
            ("http://localhost:19530/mydb/", "", "mydb"),
            ("https://localhost:19530/test_db/", "", "test_db"),
            # Mixed scenarios: explicit db_name takes precedence over URI path
            ("http://localhost:19530/uri_db", "explicit_db", "explicit_db"),
            ("http://localhost:19530/uri_db/", "explicit_db", "explicit_db"),
            # URI without path, no explicit db_name (should remain empty)
            ("http://localhost:19530", "", ""),
            ("https://localhost:19530", "", ""),
            # Multiple path segments - only first should be used as db_name
            ("http://localhost:19530/db1/collection1", "", "db1"),
            ("http://localhost:19530/db1/collection1/", "", "db1"),
            # Empty path segments should be handled correctly
            ("http://localhost:19530//", "", ""),
            ("http://localhost:19530///", "", ""),
        ],
    )
    @pytest.mark.asyncio
    async def test_async_milvus_client_extract_db_name_from_uri(
        self, uri: str, db_name: str, expected_db_name: str
    ):
        """
        Test that AsyncMilvusClient extracts db_name from URI path when db_name is not explicitly provided.
        This fixes issue #3236: v2.6.7 db name do not work
        """
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.ensure_channel_ready = AsyncMock()

        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            client = AsyncMilvusClient(uri=uri, db_name=db_name)
            await client._connect()
            assert client._config.db_name == expected_db_name, (
                f"Expected db_name to be '{expected_db_name}', "
                f"but got '{client._config.db_name}' for uri='{uri}' and db_name='{db_name}'"
            )


# ============================================================
# AsyncMilvusClient Collection Operations Tests
# ============================================================


class TestAsyncMilvusClientCollectionOps:
    """Tests for AsyncMilvusClient collection operations."""

    @pytest.mark.asyncio
    async def test_drop_collection(self):
        """Test drop_collection method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.drop_collection = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.drop_collection("test_collection")

            mock_handler.drop_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_has_collection(self):
        """Test has_collection method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.has_collection = AsyncMock(return_value=True)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.has_collection("test_collection")

            assert result is True
            mock_handler.has_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_collections(self):
        """Test list_collections method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.list_collections = AsyncMock(return_value=["coll1", "coll2", "coll3"])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.list_collections()

            assert result == ["coll1", "coll2", "coll3"]

    @pytest.mark.asyncio
    async def test_get_collection_stats(self):
        """Test get_collection_stats method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_stat = MagicMock()
            mock_stat.key = "row_count"
            mock_stat.value = "1000"
            mock_handler.get_collection_stats = AsyncMock(return_value=[mock_stat])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.get_collection_stats("test_collection")

            assert result == {"row_count": 1000}

    @pytest.mark.asyncio
    async def test_rename_collection(self):
        """Test rename_collection method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.rename_collection = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.rename_collection("old_name", "new_name")

            mock_handler.rename_collection.assert_called_once()


# ============================================================
# AsyncMilvusClient Partition Operations Tests
# ============================================================


class TestAsyncMilvusClientPartitionOps:
    """Tests for AsyncMilvusClient partition operations."""

    @pytest.mark.asyncio
    async def test_create_partition(self):
        """Test create_partition method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.create_partition = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.create_partition("test_collection", "test_partition")

            mock_handler.create_partition.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_partition(self):
        """Test drop_partition method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.drop_partition = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.drop_partition("test_collection", "test_partition")

            mock_handler.drop_partition.assert_called_once()

    @pytest.mark.asyncio
    async def test_has_partition(self):
        """Test has_partition method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.has_partition = AsyncMock(return_value=True)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.has_partition("test_collection", "test_partition")

            assert result is True

    @pytest.mark.asyncio
    async def test_list_partitions(self):
        """Test list_partitions method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.list_partitions = AsyncMock(return_value=["_default", "partition1"])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.list_partitions("test_collection")

            assert result == ["_default", "partition1"]


# ============================================================
# AsyncMilvusClient Index Operations Tests
# ============================================================


class TestAsyncMilvusClientIndexOps:
    """Tests for AsyncMilvusClient index operations."""

    @pytest.mark.asyncio
    async def test_drop_index(self):
        """Test drop_index method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.drop_index = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.drop_index("test_collection", "test_index")

            mock_handler.drop_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_indexes(self):
        """Test list_indexes method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_index1 = MagicMock()
            mock_index1.field_name = "vector"
            mock_index1.index_name = "index1"
            mock_index2 = MagicMock()
            mock_index2.field_name = "vector"
            mock_index2.index_name = "index2"
            mock_handler.list_indexes = AsyncMock(return_value=[mock_index1, mock_index2])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.list_indexes("test_collection")

            assert result == ["index1", "index2"]


# ============================================================
# AsyncMilvusClient Alias Operations Tests
# ============================================================


class TestAsyncMilvusClientAliasOps:
    """Tests for AsyncMilvusClient alias operations."""

    @pytest.mark.asyncio
    async def test_create_alias(self):
        """Test create_alias method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.create_alias = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.create_alias("test_collection", "test_alias")

            mock_handler.create_alias.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_alias(self):
        """Test drop_alias method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.drop_alias = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.drop_alias("test_alias")

            mock_handler.drop_alias.assert_called_once()

    @pytest.mark.asyncio
    async def test_alter_alias(self):
        """Test alter_alias method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.alter_alias = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.alter_alias("test_collection", "test_alias")

            mock_handler.alter_alias.assert_called_once()


# ============================================================
# AsyncMilvusClient User/Role Operations Tests
# ============================================================


class TestAsyncMilvusClientUserOps:
    """Tests for AsyncMilvusClient user and role operations."""

    @pytest.mark.asyncio
    async def test_create_user(self):
        """Test create_user method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.create_user = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.create_user("test_user", "password123")

            mock_handler.create_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_user(self):
        """Test drop_user method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.drop_user = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.drop_user("test_user")

            mock_handler.drop_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_users(self):
        """Test list_users method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.list_users = AsyncMock(return_value=["root", "user1"])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.list_users()

            assert result == ["root", "user1"]

    @pytest.mark.asyncio
    async def test_create_role(self):
        """Test create_role method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.create_role = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.create_role("test_role")

            mock_handler.create_role.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_role(self):
        """Test drop_role method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.drop_role = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.drop_role("test_role")

            mock_handler.drop_role.assert_called_once()

    @pytest.mark.asyncio
    async def test_grant_role(self):
        """Test grant_role method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.grant_role = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.grant_role("test_user", "test_role")

            mock_handler.grant_role.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_role(self):
        """Test revoke_role method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.revoke_role = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.revoke_role("test_user", "test_role")

            mock_handler.revoke_role.assert_called_once()


# ============================================================
# AsyncMilvusClient Utility Operations Tests
# ============================================================


class TestAsyncMilvusClientUtilityOps:
    """Tests for AsyncMilvusClient utility operations."""

    @pytest.mark.asyncio
    async def test_flush(self):
        """Test flush method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.flush = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.flush("test_collection")

            mock_handler.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_collection(self):
        """Test load_collection method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.load_collection = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.load_collection("test_collection")

            mock_handler.load_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_collection(self):
        """Test release_collection method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.release_collection = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.release_collection("test_collection")

            mock_handler.release_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_partitions(self):
        """Test load_partitions method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.load_partitions = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.load_partitions("test_collection", ["partition1", "partition2"])

            mock_handler.load_partitions.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_partitions(self):
        """Test release_partitions method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.release_partitions = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.release_partitions("test_collection", ["partition1", "partition2"])

            mock_handler.release_partitions.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_partition_stats(self):
        """Test get_partition_stats method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_stat = MagicMock()
            mock_stat.key = "row_count"
            mock_stat.value = "500"
            mock_handler.get_partition_stats = AsyncMock(return_value=[mock_stat])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.get_partition_stats("test_collection", "test_partition")

            assert result == {"row_count": 500}

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            client = AsyncMilvusClient()
            await client._connect()

            # Verify handler is set before close
            assert client._handler is not None

            # Mock the manager's release method
            client._manager.release = AsyncMock()

            await client.close()

            # Verify release was called and handler is cleared
            client._manager.release.assert_called_once()
            assert client._handler is None


# ============================================================
# AsyncMilvusClient Database Operations Tests
# ============================================================


class TestAsyncMilvusClientDatabaseOps:
    """Tests for AsyncMilvusClient database operations."""

    @pytest.mark.asyncio
    async def test_create_database(self):
        """Test create_database method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.create_database = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.create_database("test_db")

            mock_handler.create_database.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_database(self):
        """Test drop_database method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.drop_database = AsyncMock(return_value=None)

            client = AsyncMilvusClient()
            await client._connect()
            await client.drop_database("test_db")

            mock_handler.drop_database.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_databases(self):
        """Test list_databases method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.list_database = AsyncMock(return_value=["default", "test_db"])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.list_databases()

            assert result == ["default", "test_db"]


# ============================================================
# AsyncMilvusClient Data Operations Tests
# ============================================================


class TestAsyncMilvusClientDataOps:
    """Tests for AsyncMilvusClient data operations."""

    @pytest.mark.asyncio
    async def test_insert_single_dict(self):
        """Test insert with single dict."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_result = MagicMock()
            mock_result.insert_count = 1
            mock_result.primary_keys = [1]
            mock_result.cost = 0
            mock_handler.insert_rows = AsyncMock(return_value=mock_result)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.insert("test_collection", {"id": 1, "vector": [0.1, 0.2]})

            assert result["insert_count"] == 1
            assert result["ids"] == [1]

    @pytest.mark.asyncio
    async def test_insert_list_of_dicts(self):
        """Test insert with list of dicts."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_result = MagicMock()
            mock_result.insert_count = 2
            mock_result.primary_keys = [1, 2]
            mock_result.cost = 0
            mock_handler.insert_rows = AsyncMock(return_value=mock_result)

            client = AsyncMilvusClient()
            await client._connect()
            data = [{"id": 1, "vector": [0.1, 0.2]}, {"id": 2, "vector": [0.3, 0.4]}]
            result = await client.insert("test_collection", data)

            assert result["insert_count"] == 2

    @pytest.mark.asyncio
    async def test_insert_empty_data(self):
        """Test insert with empty data returns zero count."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.insert("test_collection", [])

            assert result["insert_count"] == 0
            assert result["ids"] == []

    @pytest.mark.asyncio
    async def test_insert_invalid_type(self):
        """Test insert with invalid type raises TypeError."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):

            client = AsyncMilvusClient()

            with pytest.raises(TypeError, match="wrong type of argument"):
                await client.insert("test_collection", "invalid")

    @pytest.mark.asyncio
    async def test_upsert_single_dict(self):
        """Test upsert with single dict."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_result = MagicMock()
            mock_result.upsert_count = 1
            mock_result.primary_keys = [1]
            mock_result.cost = 0
            mock_handler.upsert_rows = AsyncMock(return_value=mock_result)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.upsert("test_collection", {"id": 1, "vector": [0.1, 0.2]})

            assert result["upsert_count"] == 1

    @pytest.mark.asyncio
    async def test_upsert_empty_data(self):
        """Test upsert with empty data returns zero count."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.upsert("test_collection", [])

            assert result["upsert_count"] == 0
            assert result["ids"] == []

    @pytest.mark.asyncio
    async def test_upsert_invalid_type(self):
        """Test upsert with invalid type raises TypeError."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):

            client = AsyncMilvusClient()

            with pytest.raises(TypeError, match="wrong type of argument"):
                await client.upsert("test_collection", "invalid")

    @pytest.mark.asyncio
    async def test_delete_with_ids(self):
        """Test delete with IDs."""

        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_result = MagicMock()
            mock_result.delete_count = 2
            mock_result.cost = 0
            mock_result.primary_keys = []
            mock_handler.delete = AsyncMock(return_value=mock_result)
            # Mock _get_schema to return schema with primary key field
            mock_schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
            mock_handler._get_schema = AsyncMock(return_value=(mock_schema, 100))

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.delete("test_collection", ids=[1, 2])

            assert result["delete_count"] == 2

    @pytest.mark.asyncio
    async def test_delete_with_filter(self):
        """Test delete with filter expression."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_result = MagicMock()
            mock_result.delete_count = 5
            mock_result.cost = 0
            mock_result.primary_keys = []
            mock_handler.delete = AsyncMock(return_value=mock_result)

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.delete("test_collection", filter="age > 20")

            assert result["delete_count"] == 5

    @pytest.mark.asyncio
    async def test_search(self):
        """Test search method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.search = AsyncMock(return_value=[[{"id": 1, "distance": 0.1}]])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.search("test_collection", data=[[0.1, 0.2]])

            assert result == [[{"id": 1, "distance": 0.1}]]
            mock_handler.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query(self):
        """Test query method."""
        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.query = AsyncMock(return_value=[{"id": 1, "name": "test"}])

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.query("test_collection", filter="id > 0")

            assert result == [{"id": 1, "name": "test"}]
            mock_handler.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get(self):
        """Test get method (query by ids)."""

        mock_handler = MagicMock()
        mock_handler.ensure_channel_ready = AsyncMock()
        with patch(
            "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
        ):
            mock_handler.query = AsyncMock(return_value=[{"id": 1, "name": "test"}])
            # Mock _get_schema to return schema with primary key field
            mock_schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
            mock_handler._get_schema = AsyncMock(return_value=(mock_schema, 100))

            client = AsyncMilvusClient()
            await client._connect()
            result = await client.get("test_collection", ids=[1])

            assert result == [{"id": 1, "name": "test"}]
