"""Test cases for AsyncMilvusClient new features"""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pymilvus import AsyncMilvusClient, DataType
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.connection_manager import AsyncConnectionManager
from pymilvus.client.types import (
    CompactionPlans,
    LoadState,
    RefreshExternalCollectionJobInfo,
    RestoreSnapshotJobInfo,
    SnapshotInfo,
)
from pymilvus.orm.collection import Function
from pymilvus.orm.schema import StructFieldSchema


@pytest.fixture(autouse=True)
def reset_async_connection_manager():
    """Reset AsyncConnectionManager singleton before and after each test."""
    AsyncConnectionManager._reset_instance()
    yield
    AsyncConnectionManager._reset_instance()


@pytest_asyncio.fixture
async def client_and_handler():
    """Yield (client, mock_handler) with channel ready and connected."""
    mock_handler = MagicMock()
    mock_handler.ensure_channel_ready = AsyncMock()
    with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler):
        client = AsyncMilvusClient()
        await client._connect()
        yield client, mock_handler


class TestAsyncMilvusClientNewFeatures:
    """Test cases for newly added features in AsyncMilvusClient"""

    @pytest.mark.asyncio
    async def test_get_server_type(self, client_and_handler):
        """Test get_server_type returns correct value"""
        client, mock_handler = client_and_handler
        mock_handler.get_server_type.return_value = "milvus"
        # get_server_type is called during __init__ for is_self_hosted, and again here
        assert mock_handler.get_server_type.call_count >= 1
        result = client.get_server_type()
        assert result == "milvus"
        # Verify it was called at least once more (during the explicit call)
        assert mock_handler.get_server_type.call_count >= 2

    @pytest.mark.asyncio
    async def test_get_load_state_with_progress_when_loading(self, client_and_handler):
        """Test get_load_state returns progress when state is Loading"""
        client, mock_handler = client_and_handler
        mock_handler.get_load_state = AsyncMock(return_value=LoadState.Loading)
        mock_handler.get_loading_progress = AsyncMock(return_value=75)

        result = await client.get_load_state("test_collection")

        assert result["state"] == LoadState.Loading
        assert result["progress"] == 75
        mock_handler.get_load_state.assert_called_once()
        mock_handler.get_loading_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_load_state_without_progress_when_not_loading(self, client_and_handler):
        """Test get_load_state does not return progress when state is not Loading"""
        client, mock_handler = client_and_handler
        mock_handler.get_load_state = AsyncMock(return_value=LoadState.Loaded)

        result = await client.get_load_state("test_collection")

        assert result["state"] == LoadState.Loaded
        assert "progress" not in result
        mock_handler.get_load_state.assert_called_once()
        mock_handler.get_loading_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_describe_collection_with_struct_array_fields(self, client_and_handler):
        """Test describe_collection converts struct_array_fields to user-friendly format"""
        client, mock_handler = client_and_handler
        mock_result = {
            "fields": [{"name": "field1", "type": "INT64"}],
            "struct_array_fields": [
                {"name": "struct_field", "type": "STRUCT", "element_type": "INT64"}
            ],
        }
        mock_handler.describe_collection = AsyncMock(return_value=mock_result)

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
    async def test_describe_collection_without_struct_array_fields(self, client_and_handler):
        """Test describe_collection works normally when no struct_array_fields"""
        client, mock_handler = client_and_handler
        mock_result = {"fields": [{"name": "field1", "type": "INT64"}]}
        mock_handler.describe_collection = AsyncMock(return_value=mock_result)

        result = await client.describe_collection("test_collection")

        assert result == mock_result
        assert "struct_array_fields" not in result
        mock_handler.describe_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_ranker(self, client_and_handler):
        """Test search method accepts ranker parameter"""
        client, mock_handler = client_and_handler
        mock_handler.search = AsyncMock(return_value=[[{"id": 1, "distance": 0.1}]])
        mock_ranker = MagicMock(spec=Function)

        result = await client.search("test_collection", data=[[0.1, 0.2]], ranker=mock_ranker)

        assert result == [[{"id": 1, "distance": 0.1}]]
        # Verify ranker was passed to handler
        call_args = mock_handler.search.call_args
        # ranker is passed as keyword argument
        assert call_args.kwargs.get("ranker") == mock_ranker

    @pytest.mark.asyncio
    async def test_hybrid_search_with_function_ranker(self, client_and_handler):
        """Test hybrid_search accepts Function type ranker"""
        client, mock_handler = client_and_handler
        mock_handler.hybrid_search = AsyncMock(return_value=[[{"id": 1}]])
        mock_function_ranker = MagicMock(spec=Function)
        mock_req = MagicMock(spec=AnnSearchRequest)

        result = await client.hybrid_search(
            "test_collection", reqs=[mock_req], ranker=mock_function_ranker
        )

        assert result == [[{"id": 1}]]
        # ranker is passed as positional argument (3rd arg), check args[2]
        call_args = mock_handler.hybrid_search.call_args
        assert call_args[0][2] == mock_function_ranker  # ranker is 3rd positional arg

    @pytest.mark.asyncio
    async def test_compact_with_is_l0(self, client_and_handler):
        """Test compact method accepts is_l0 parameter"""
        client, mock_handler = client_and_handler
        mock_handler.compact = AsyncMock(return_value=123)

        result = await client.compact("test_collection", is_l0=True)

        assert result == 123
        call_args = mock_handler.compact.call_args
        assert call_args[1]["is_l0"] is True

    @pytest.mark.asyncio
    async def test_flush_all(self, client_and_handler):
        """Test flush_all method"""
        client, mock_handler = client_and_handler
        mock_handler.flush_all = AsyncMock()

        await client.flush_all(timeout=10)
        mock_handler.flush_all.assert_called_once_with(timeout=10, context=ANY)

    @pytest.mark.asyncio
    async def test_get_flush_all_state(self, client_and_handler):
        """Test get_flush_all_state method"""
        client, mock_handler = client_and_handler
        mock_handler.get_flush_all_state = AsyncMock(return_value=True)

        result = await client.get_flush_all_state(timeout=10)

        assert result is True
        mock_handler.get_flush_all_state.assert_called_once_with(timeout=10, context=ANY)

    @pytest.mark.asyncio
    async def test_get_compaction_plans(self, client_and_handler):
        """Test get_compaction_plans method"""
        client, mock_handler = client_and_handler
        mock_plans = MagicMock(spec=CompactionPlans)
        mock_handler.get_compaction_plans = AsyncMock(return_value=mock_plans)

        result = await client.get_compaction_plans(job_id=123, timeout=10)

        assert result == mock_plans
        mock_handler.get_compaction_plans.assert_called_once_with(123, timeout=10, context=ANY)

    @pytest.mark.asyncio
    async def test_update_replicate_configuration(self, client_and_handler):
        """Test update_replicate_configuration method"""
        client, mock_handler = client_and_handler
        clusters = [
            {
                "cluster_id": "cluster1",
                "connection_param": {"uri": "http://localhost:19530", "token": "token1"},
            }
        ]
        cross_cluster_topology = [
            {"source_cluster_id": "cluster1", "target_cluster_id": "cluster2"}
        ]
        mock_status = MagicMock()
        mock_handler.update_replicate_configuration = AsyncMock(return_value=mock_status)

        result = await client.update_replicate_configuration(
            clusters=clusters, cross_cluster_topology=cross_cluster_topology, timeout=10
        )

        assert result == mock_status
        mock_handler.update_replicate_configuration.assert_called_once_with(
            clusters=clusters,
            cross_cluster_topology=cross_cluster_topology,
            force_promote=False,
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
# Simple delegation test table
# ============================================================
_SIMPLE_DELEGATION_CASES = [
    # Collection ops
    ("drop_collection", ("test_collection",), {}, "drop_collection"),
    ("rename_collection", ("old_name", "new_name"), {}, "rename_collection"),
    # Partition ops
    ("create_partition", ("test_collection", "test_partition"), {}, "create_partition"),
    ("drop_partition", ("test_collection", "test_partition"), {}, "drop_partition"),
    # Index ops
    ("drop_index", ("test_collection", "test_index"), {}, "drop_index"),
    # Alias ops
    ("create_alias", ("test_collection", "test_alias"), {}, "create_alias"),
    ("drop_alias", ("test_alias",), {}, "drop_alias"),
    ("alter_alias", ("test_collection", "test_alias"), {}, "alter_alias"),
    # User/role ops
    ("create_user", ("test_user", "password123"), {}, "create_user"),
    ("drop_user", ("test_user",), {}, "drop_user"),
    ("create_role", ("test_role",), {}, "create_role"),
    ("drop_role", ("test_role",), {}, "drop_role"),
    ("grant_role", ("test_user", "test_role"), {}, "grant_role"),
    ("revoke_role", ("test_user", "test_role"), {}, "revoke_role"),
    # Utility ops
    ("flush", ("test_collection",), {}, "flush"),
    ("load_collection", ("test_collection",), {}, "load_collection"),
    ("release_collection", ("test_collection",), {}, "release_collection"),
    ("load_partitions", ("test_collection", ["partition1", "partition2"]), {}, "load_partitions"),
    (
        "release_partitions",
        ("test_collection", ["partition1", "partition2"]),
        {},
        "release_partitions",
    ),
    # Database ops
    ("create_database", ("test_db",), {}, "create_database"),
    ("drop_database", ("test_db",), {}, "drop_database"),
]


class TestSimpleDelegation:
    """Parametrized tests for simple handler-delegation methods."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "client_method, args, kwargs, handler_method",
        _SIMPLE_DELEGATION_CASES,
        ids=[c[0] for c in _SIMPLE_DELEGATION_CASES],
    )
    async def test_delegation(
        self, client_and_handler, client_method, args, kwargs, handler_method
    ):
        client, mock_handler = client_and_handler
        setattr(mock_handler, handler_method, AsyncMock(return_value=None))

        await getattr(client, client_method)(*args, **kwargs)

        getattr(mock_handler, handler_method).assert_called_once()


# ============================================================
# Tests with return-value or non-trivial assertions
# ============================================================


class TestAsyncMilvusClientOps:
    """Tests for AsyncMilvusClient operations with return-value assertions."""

    @pytest.mark.asyncio
    async def test_has_collection(self, client_and_handler):
        """Test has_collection method."""
        client, mock_handler = client_and_handler
        mock_handler.has_collection = AsyncMock(return_value=True)

        result = await client.has_collection("test_collection")

        assert result is True
        mock_handler.has_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_collections(self, client_and_handler):
        """Test list_collections method."""
        client, mock_handler = client_and_handler
        mock_handler.list_collections = AsyncMock(return_value=["coll1", "coll2", "coll3"])

        result = await client.list_collections()

        assert result == ["coll1", "coll2", "coll3"]

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, client_and_handler):
        """Test get_collection_stats method."""
        client, mock_handler = client_and_handler
        mock_stat = MagicMock()
        mock_stat.key = "row_count"
        mock_stat.value = "1000"
        mock_handler.get_collection_stats = AsyncMock(return_value=[mock_stat])

        result = await client.get_collection_stats("test_collection")

        assert result == {"row_count": 1000}

    @pytest.mark.asyncio
    async def test_has_partition(self, client_and_handler):
        """Test has_partition method."""
        client, mock_handler = client_and_handler
        mock_handler.has_partition = AsyncMock(return_value=True)

        result = await client.has_partition("test_collection", "test_partition")

        assert result is True

    @pytest.mark.asyncio
    async def test_list_partitions(self, client_and_handler):
        """Test list_partitions method."""
        client, mock_handler = client_and_handler
        mock_handler.list_partitions = AsyncMock(return_value=["_default", "partition1"])

        result = await client.list_partitions("test_collection")

        assert result == ["_default", "partition1"]

    @pytest.mark.asyncio
    async def test_list_indexes(self, client_and_handler):
        """Test list_indexes method."""
        client, mock_handler = client_and_handler
        mock_index1 = MagicMock()
        mock_index1.field_name = "vector"
        mock_index1.index_name = "index1"
        mock_index2 = MagicMock()
        mock_index2.field_name = "vector"
        mock_index2.index_name = "index2"
        mock_handler.list_indexes = AsyncMock(return_value=[mock_index1, mock_index2])

        result = await client.list_indexes("test_collection")

        assert result == ["index1", "index2"]

    @pytest.mark.asyncio
    async def test_list_users(self, client_and_handler):
        """Test list_users method."""
        client, mock_handler = client_and_handler
        mock_handler.list_users = AsyncMock(return_value=["root", "user1"])

        result = await client.list_users()

        assert result == ["root", "user1"]

    @pytest.mark.asyncio
    async def test_get_partition_stats(self, client_and_handler):
        """Test get_partition_stats method."""
        client, mock_handler = client_and_handler
        mock_stat = MagicMock()
        mock_stat.key = "row_count"
        mock_stat.value = "500"
        mock_handler.get_partition_stats = AsyncMock(return_value=[mock_stat])

        result = await client.get_partition_stats("test_collection", "test_partition")

        assert result == {"row_count": 500}

    @pytest.mark.asyncio
    async def test_close(self, client_and_handler):
        """Test close method."""
        client, _mock_handler = client_and_handler

        # Verify handler is set before close
        assert client._handler is not None

        # Mock the manager's release method
        client._manager.release = AsyncMock()

        await client.close()

        # Verify release was called and handler is cleared
        client._manager.release.assert_called_once()
        assert client._handler is None

    @pytest.mark.asyncio
    async def test_list_databases(self, client_and_handler):
        """Test list_databases method."""
        client, mock_handler = client_and_handler
        mock_handler.list_database = AsyncMock(return_value=["default", "test_db"])

        result = await client.list_databases()

        assert result == ["default", "test_db"]


# ============================================================
# AsyncMilvusClient Data Operations Tests
# ============================================================


class TestAsyncMilvusClientDataOps:
    """Tests for AsyncMilvusClient data operations."""

    @pytest.mark.asyncio
    async def test_insert_single_dict(self, client_and_handler):
        """Test insert with single dict."""
        client, mock_handler = client_and_handler
        mock_result = MagicMock()
        mock_result.insert_count = 1
        mock_result.primary_keys = [1]
        mock_result.cost = 0
        mock_handler.insert_rows = AsyncMock(return_value=mock_result)

        result = await client.insert("test_collection", {"id": 1, "vector": [0.1, 0.2]})

        assert result["insert_count"] == 1
        assert result["ids"] == [1]

    @pytest.mark.asyncio
    async def test_insert_list_of_dicts(self, client_and_handler):
        """Test insert with list of dicts."""
        client, mock_handler = client_and_handler
        mock_result = MagicMock()
        mock_result.insert_count = 2
        mock_result.primary_keys = [1, 2]
        mock_result.cost = 0
        mock_handler.insert_rows = AsyncMock(return_value=mock_result)

        data = [{"id": 1, "vector": [0.1, 0.2]}, {"id": 2, "vector": [0.3, 0.4]}]
        result = await client.insert("test_collection", data)

        assert result["insert_count"] == 2

    @pytest.mark.asyncio
    async def test_insert_empty_data(self, client_and_handler):
        """Test insert with empty data returns zero count."""
        client, _mock_handler = client_and_handler

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
    async def test_upsert_single_dict(self, client_and_handler):
        """Test upsert with single dict."""
        client, mock_handler = client_and_handler
        mock_result = MagicMock()
        mock_result.upsert_count = 1
        mock_result.primary_keys = [1]
        mock_result.cost = 0
        mock_handler.upsert_rows = AsyncMock(return_value=mock_result)

        result = await client.upsert("test_collection", {"id": 1, "vector": [0.1, 0.2]})

        assert result["upsert_count"] == 1

    @pytest.mark.asyncio
    async def test_upsert_empty_data(self, client_and_handler):
        """Test upsert with empty data returns zero count."""
        client, _mock_handler = client_and_handler

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
    async def test_delete_with_ids(self, client_and_handler):
        """Test delete with IDs."""
        client, mock_handler = client_and_handler
        mock_result = MagicMock()
        mock_result.delete_count = 2
        mock_result.cost = 0
        mock_result.primary_keys = []
        mock_handler.delete = AsyncMock(return_value=mock_result)
        # Mock _get_schema to return schema with primary key field
        mock_schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        mock_handler._get_schema = AsyncMock(return_value=(mock_schema, 100))

        result = await client.delete("test_collection", ids=[1, 2])

        assert result["delete_count"] == 2

    @pytest.mark.asyncio
    async def test_delete_with_filter(self, client_and_handler):
        """Test delete with filter expression."""
        client, mock_handler = client_and_handler
        mock_result = MagicMock()
        mock_result.delete_count = 5
        mock_result.cost = 0
        mock_result.primary_keys = []
        mock_handler.delete = AsyncMock(return_value=mock_result)

        result = await client.delete("test_collection", filter="age > 20")

        assert result["delete_count"] == 5

    @pytest.mark.asyncio
    async def test_search(self, client_and_handler):
        """Test search method."""
        client, mock_handler = client_and_handler
        mock_handler.search = AsyncMock(return_value=[[{"id": 1, "distance": 0.1}]])

        result = await client.search("test_collection", data=[[0.1, 0.2]])

        assert result == [[{"id": 1, "distance": 0.1}]]
        mock_handler.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query(self, client_and_handler):
        """Test query method."""
        client, mock_handler = client_and_handler
        mock_handler.query = AsyncMock(return_value=[{"id": 1, "name": "test"}])

        result = await client.query("test_collection", filter="id > 0")

        assert result == [{"id": 1, "name": "test"}]
        mock_handler.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get(self, client_and_handler):
        """Test get method (query by ids)."""
        client, mock_handler = client_and_handler
        mock_handler.query = AsyncMock(return_value=[{"id": 1, "name": "test"}])
        # Mock _get_schema to return schema with primary key field
        mock_schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        mock_handler._get_schema = AsyncMock(return_value=(mock_schema, 100))

        result = await client.get("test_collection", ids=[1])

        assert result == [{"id": 1, "name": "test"}]


# ============================================================
# Snapshot tests (parametrized)
# ============================================================

_SNAPSHOT_CASES = [
    (
        "create_snapshot",
        {
            "snapshot_name": "test_snapshot",
            "collection_name": "test_collection",
            "description": "Test description",
        },
        {
            "snapshot_name": "test_snapshot",
            "collection_name": "test_collection",
            "description": "Test description",
            "db_name": "",
            "compaction_protection_seconds": 0,
            "timeout": None,
            "context": ANY,
        },
        None,
        None,
    ),
    (
        "drop_snapshot",
        {"snapshot_name": "test_snapshot", "collection_name": "test_collection"},
        {
            "snapshot_name": "test_snapshot",
            "collection_name": "test_collection",
            "db_name": "",
            "timeout": None,
            "context": ANY,
        },
        None,
        None,
    ),
    (
        "list_snapshots",
        {"collection_name": "test_collection"},
        {
            "collection_name": "test_collection",
            "db_name": "",
            "timeout": None,
            "context": ANY,
        },
        ["snapshot1", "snapshot2"],
        None,
    ),
    (
        "restore_snapshot",
        {
            "snapshot_name": "test_snapshot",
            "target_collection_name": "restored_collection",
            "source_collection_name": "test_collection",
        },
        {
            "snapshot_name": "test_snapshot",
            "target_collection_name": "restored_collection",
            "source_collection_name": "test_collection",
            "target_db_name": "",
            "source_db_name": "",
            "timeout": None,
            "context": ANY,
        },
        12345,
        None,
    ),
    (
        "get_restore_snapshot_state",
        {"job_id": 12345},
        {"job_id": 12345, "timeout": None, "context": ANY},
        None,  # will be overridden below with a real object
        None,
    ),
    (
        "list_restore_snapshot_jobs",
        {"collection_name": "test_collection"},
        {
            "collection_name": "test_collection",
            "db_name": "",
            "timeout": None,
            "context": ANY,
        },
        [],
        None,
    ),
]


class TestAsyncMilvusClientSnapshot:
    """Test snapshot-related APIs in AsyncMilvusClient."""

    @pytest.mark.asyncio
    async def test_describe_snapshot(self, client_and_handler):
        """Test describe_snapshot passes context."""
        client, mock_handler = client_and_handler
        mock_info = SnapshotInfo(
            name="test_snapshot",
            description="Test description",
            collection_name="test_collection",
            partition_names=["_default"],
            create_ts=1234567890,
            s3_location="s3://bucket/path",
        )
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.describe_snapshot = AsyncMock(return_value=mock_info)

        info = await client.describe_snapshot(
            snapshot_name="test_snapshot", collection_name="test_collection"
        )

        assert info.name == "test_snapshot"
        mock_handler.describe_snapshot.assert_called_once_with(
            snapshot_name="test_snapshot",
            collection_name="test_collection",
            db_name="",
            timeout=None,
            context=ANY,
        )

    @pytest.mark.asyncio
    async def test_get_restore_snapshot_state(self, client_and_handler):
        """Test get_restore_snapshot_state passes context."""
        client, mock_handler = client_and_handler
        mock_state = RestoreSnapshotJobInfo(
            job_id=12345,
            snapshot_name="test_snapshot",
            db_name="default",
            collection_name="test_collection",
            state="RestoreSnapshotCompleted",
            progress=100,
            reason="",
            start_time=1234567890,
            time_cost=5000,
        )
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.get_restore_snapshot_state = AsyncMock(return_value=mock_state)

        state = await client.get_restore_snapshot_state(job_id=12345)

        assert state.job_id == 12345
        mock_handler.get_restore_snapshot_state.assert_called_once_with(
            job_id=12345, timeout=None, context=ANY
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name, call_kwargs, expected_handler_kwargs, return_value",
        [
            (
                "create_snapshot",
                {
                    "snapshot_name": "test_snapshot",
                    "collection_name": "test_collection",
                    "description": "Test description",
                },
                {
                    "snapshot_name": "test_snapshot",
                    "collection_name": "test_collection",
                    "description": "Test description",
                    "db_name": "",
                    "compaction_protection_seconds": 0,
                    "timeout": None,
                    "context": ANY,
                },
                None,
            ),
            (
                "drop_snapshot",
                {"snapshot_name": "test_snapshot", "collection_name": "test_collection"},
                {
                    "snapshot_name": "test_snapshot",
                    "collection_name": "test_collection",
                    "db_name": "",
                    "timeout": None,
                    "context": ANY,
                },
                None,
            ),
            (
                "list_snapshots",
                {"collection_name": "test_collection"},
                {
                    "collection_name": "test_collection",
                    "db_name": "",
                    "timeout": None,
                    "context": ANY,
                },
                ["snapshot1", "snapshot2"],
            ),
            (
                "restore_snapshot",
                {
                    "snapshot_name": "test_snapshot",
                    "target_collection_name": "restored_collection",
                    "source_collection_name": "test_collection",
                },
                {
                    "snapshot_name": "test_snapshot",
                    "target_collection_name": "restored_collection",
                    "source_collection_name": "test_collection",
                    "target_db_name": "",
                    "source_db_name": "",
                    "timeout": None,
                    "context": ANY,
                },
                12345,
            ),
            (
                "list_restore_snapshot_jobs",
                {"collection_name": "test_collection"},
                {
                    "collection_name": "test_collection",
                    "db_name": "",
                    "timeout": None,
                    "context": ANY,
                },
                [],
            ),
        ],
        ids=[
            "create_snapshot",
            "drop_snapshot",
            "list_snapshots",
            "restore_snapshot",
            "list_restore_snapshot_jobs",
        ],
    )
    async def test_snapshot_method(
        self,
        client_and_handler,
        method_name,
        call_kwargs,
        expected_handler_kwargs,
        return_value,
    ):
        """Parametrized test for snapshot methods that pass context."""
        client, mock_handler = client_and_handler
        mock_handler.get_server_type.return_value = "milvus"
        setattr(mock_handler, method_name, AsyncMock(return_value=return_value))

        result = await getattr(client, method_name)(**call_kwargs)

        getattr(mock_handler, method_name).assert_called_once_with(**expected_handler_kwargs)
        if return_value is not None:
            assert result == return_value

    @pytest.mark.asyncio
    async def test_pin_snapshot_data(self, client_and_handler):
        """Test async pin_snapshot_data passes context."""
        client, mock_handler = client_and_handler
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.pin_snapshot_data = AsyncMock(return_value=42)

        pin_id = await client.pin_snapshot_data(
            snapshot_name="test_snapshot",
            collection_name="test_collection",
            ttl_seconds=60,
        )

        assert pin_id == 42
        mock_handler.pin_snapshot_data.assert_called_once_with(
            snapshot_name="test_snapshot",
            collection_name="test_collection",
            db_name="",
            ttl_seconds=60,
            timeout=None,
            context=ANY,
        )

    @pytest.mark.asyncio
    async def test_unpin_snapshot_data(self, client_and_handler):
        """Test async unpin_snapshot_data passes context."""
        client, mock_handler = client_and_handler
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.unpin_snapshot_data = AsyncMock(return_value=None)

        await client.unpin_snapshot_data(pin_id=42)

        mock_handler.unpin_snapshot_data.assert_called_once_with(
            pin_id=42, timeout=None, context=ANY
        )


class TestAsyncFileResourceMethods:
    """Tests for async file_resource API methods with context passing."""

    @pytest.mark.asyncio
    async def test_add_file_resource(self, client_and_handler):
        client, mock_handler = client_and_handler
        mock_handler.add_file_resource = AsyncMock(return_value=None)

        await client.add_file_resource(name="test_file", path="/data/test.csv")

        mock_handler.add_file_resource.assert_called_once()
        call_kwargs = mock_handler.add_file_resource.call_args.kwargs
        assert call_kwargs["name"] == "test_file"
        assert call_kwargs["path"] == "/data/test.csv"
        assert "context" in call_kwargs

    @pytest.mark.asyncio
    async def test_add_file_resource_with_timeout(self, client_and_handler):
        client, mock_handler = client_and_handler
        mock_handler.add_file_resource = AsyncMock(return_value=None)

        await client.add_file_resource(name="f", path="/p", timeout=30)

        call_kwargs = mock_handler.add_file_resource.call_args.kwargs
        assert call_kwargs["timeout"] == 30

    @pytest.mark.asyncio
    async def test_remove_file_resource(self, client_and_handler):
        client, mock_handler = client_and_handler
        mock_handler.remove_file_resource = AsyncMock(return_value=None)

        await client.remove_file_resource(name="test_file")

        mock_handler.remove_file_resource.assert_called_once()
        call_kwargs = mock_handler.remove_file_resource.call_args.kwargs
        assert call_kwargs["name"] == "test_file"
        assert "context" in call_kwargs

    @pytest.mark.asyncio
    async def test_list_file_resources(self, client_and_handler):
        client, mock_handler = client_and_handler
        mock_handler.list_file_resources = AsyncMock(return_value=["file1", "file2"])

        result = await client.list_file_resources()

        mock_handler.list_file_resources.assert_called_once()
        assert result == ["file1", "file2"]
        assert "context" in mock_handler.list_file_resources.call_args.kwargs


class TestAsyncMilvusClientExternalCollection:
    """Test external collection refresh APIs in AsyncMilvusClient."""

    @pytest.mark.asyncio
    async def test_refresh_external_collection(self, client_and_handler):
        client, mock_handler = client_and_handler
        mock_handler.refresh_external_collection = AsyncMock(return_value=42)

        result = await client.refresh_external_collection(collection_name="ext_coll")

        assert result == 42
        mock_handler.refresh_external_collection.assert_called_once_with(
            collection_name="ext_coll",
            timeout=None,
            context=ANY,
            external_source="",
            external_spec="",
        )

    @pytest.mark.asyncio
    async def test_refresh_with_new_source(self, client_and_handler):
        client, mock_handler = client_and_handler
        mock_handler.refresh_external_collection = AsyncMock(return_value=43)

        result = await client.refresh_external_collection(
            collection_name="ext_coll",
            external_source="s3://new-path",
            external_spec='{"format": "iceberg"}',
        )

        assert result == 43

    @pytest.mark.asyncio
    async def test_get_refresh_progress(self, client_and_handler):
        client, mock_handler = client_and_handler
        job_info = RefreshExternalCollectionJobInfo(
            job_id=42,
            collection_name="ext_coll",
            state="RefreshCompleted",
            progress=100,
            reason="",
            external_source="s3://bucket",
            start_time=1000,
            end_time=2000,
        )
        mock_handler.get_refresh_external_collection_progress = AsyncMock(return_value=job_info)

        result = await client.get_refresh_external_collection_progress(job_id=42)

        assert result.job_id == 42
        assert result.state == "RefreshCompleted"

    @pytest.mark.asyncio
    async def test_list_refresh_jobs(self, client_and_handler):
        client, mock_handler = client_and_handler
        mock_handler.list_refresh_external_collection_jobs = AsyncMock(return_value=[])

        result = await client.list_refresh_external_collection_jobs(collection_name="ext_coll")

        assert result == []
        mock_handler.list_refresh_external_collection_jobs.assert_called_once_with(
            collection_name="ext_coll", timeout=None, context=ANY
        )
