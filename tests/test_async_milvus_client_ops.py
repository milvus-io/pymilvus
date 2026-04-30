from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import DataType
from pymilvus.client.types import LoadState
from pymilvus.exceptions import DataTypeNotMatchException, MilvusException, ParamError
from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClient
from pymilvus.milvus_client.async_optimize_task import AsyncOptimizeTask
from pymilvus.milvus_client.optimize_task import OptimizeResult


def _make_client():
    client = AsyncMilvusClient()
    handler = AsyncMock()
    handler.get_server_type.return_value = "milvus"
    client._handler = handler
    return client, handler


class TestAsyncClientCreateCollection:
    @pytest.mark.asyncio
    async def test_fast_create_collection_int_id(self):
        client, handler = _make_client()
        with patch.object(client, "create_index"), patch.object(client, "load_collection"):
            await client._fast_create_collection("col", 128, id_type="int")
        handler.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_fast_create_collection_string_id(self):
        client, handler = _make_client()
        with patch.object(client, "create_index"), patch.object(client, "load_collection"):
            await client._fast_create_collection("col", 128, id_type="string")
        handler.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_fast_create_collection_varchar_max_length(self):
        client, handler = _make_client()
        with patch.object(client, "create_index"), patch.object(client, "load_collection"):
            await client._fast_create_collection("col", 128, id_type="string", max_length=256)
        handler.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_fast_create_collection_missing_dimension_raises(self):
        client, _ = _make_client()
        with pytest.raises(TypeError, match="dimension"):
            await client._fast_create_collection("col", None)

    @pytest.mark.asyncio
    async def test_create_collection_with_schema_no_index_params(self):
        client, handler = _make_client()
        schema = AsyncMilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        await client._create_collection_with_schema("col", schema, index_params=None)
        handler.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_with_schema_with_index_params(self):
        client, handler = _make_client()
        schema = AsyncMilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
        ip = AsyncMilvusClient.prepare_index_params("vec", index_type="FLAT", metric_type="L2")
        with patch.object(client, "create_index"), patch.object(client, "load_collection"):
            await client._create_collection_with_schema("col", schema, index_params=ip)
        handler.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_truncate_collection(self):
        client, handler = _make_client()
        await client.truncate_collection("col")
        handler.truncate_collection.assert_called_once()


class TestAsyncClientIndexOps:
    @pytest.mark.asyncio
    async def test_create_index_empty_raises(self):
        client, _ = _make_client()
        ip = AsyncMilvusClient.prepare_index_params()
        with pytest.raises(ParamError, match="empty"):
            await client.create_index("col", ip)

    @pytest.mark.asyncio
    async def test_create_index_iterates(self):
        client, handler = _make_client()
        ip = AsyncMilvusClient.prepare_index_params("vec", index_type="FLAT", metric_type="L2")
        await client.create_index("col", ip)
        handler.create_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_alter_index_properties(self):
        client, handler = _make_client()
        await client.alter_index_properties("col", "idx", {"mmap.enabled": True})
        handler.alter_index_properties.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_index_properties(self):
        client, handler = _make_client()
        await client.drop_index_properties("col", "idx", ["mmap.enabled"])
        handler.drop_index_properties.assert_called_once()


class TestAsyncClientPartitionOps:
    @pytest.mark.asyncio
    async def test_load_partitions_str_converts(self):
        client, handler = _make_client()
        await client.load_partitions("col", "part1")
        args, _ = handler.load_partitions.call_args
        assert args[1] == ["part1"]

    @pytest.mark.asyncio
    async def test_release_partitions_str_converts(self):
        client, handler = _make_client()
        await client.release_partitions("col", "part1")
        handler.release_partitions.assert_called_once()


# Each tuple: (client_method, args, kwargs, handler_method)
_SIMPLE_ASYNC_DELEGATION_CASES = [
    # Collection management
    ("alter_collection_properties", ("col", {"key": "val"}), {}, "alter_collection_properties"),
    ("drop_collection_properties", ("col", ["key"]), {}, "drop_collection_properties"),
    (
        "alter_collection_field",
        ("col", "vec", {"mmap.enabled": True}),
        {},
        "alter_collection_field",
    ),
    ("drop_collection_function", ("col", "fn"), {}, "drop_collection_function"),
    ("drop_collection_field", ("col",), {"field_name": "f"}, "drop_collection_field"),
    ("add_collection_function", ("col", MagicMock()), {}, "add_collection_function"),
    ("alter_collection_function", ("col", "fn", MagicMock()), {}, "alter_collection_function"),
    # Server ops
    ("refresh_load", ("col",), {}, "refresh_load"),
    ("run_analyzer", ("hello world",), {}, "run_analyzer"),
    ("update_replicate_configuration", (), {"clusters": []}, "update_replicate_configuration"),
    ("get_replicate_configuration", (), {}, "get_replicate_configuration"),
    ("get_server_version", (), {}, "get_server_version"),
    ("describe_replica", ("col",), {}, "describe_replica"),
    ("describe_resource_group", ("rg1",), {}, "describe_resource_group"),
    # User/role ops
    ("update_password", ("user", "old", "new"), {}, "update_password"),
    # Privilege ops
    ("grant_privilege", ("admin", "Collection", "Insert", "col"), {}, "grant_privilege"),
    ("revoke_privilege", ("admin", "Collection", "Insert", "col"), {}, "revoke_privilege"),
    ("grant_privilege_v2", ("admin", "Insert", "col"), {}, "grant_privilege_v2"),
    ("revoke_privilege_v2", ("admin", "Insert", "col"), {}, "revoke_privilege_v2"),
    # Privilege groups
    ("create_privilege_group", ("grp",), {}, "create_privilege_group"),
    ("drop_privilege_group", ("grp",), {}, "drop_privilege_group"),
    ("add_privileges_to_group", ("grp", ["Insert"]), {}, "add_privileges_to_group"),
    ("remove_privileges_from_group", ("grp", ["Insert"]), {}, "remove_privileges_from_group"),
    # Resource groups
    ("create_resource_group", ("rg1",), {}, "create_resource_group"),
    ("drop_resource_group", ("rg1",), {}, "drop_resource_group"),
    ("update_resource_groups", ({},), {}, "update_resource_groups"),
    ("transfer_replica", ("rg1", "rg2", "col", 1), {}, "transfer_replica"),
    # Database ops
    ("alter_database_properties", ("mydb", {"key": "val"}), {}, "alter_database"),
    ("drop_database_properties", ("mydb", ["key"]), {}, "drop_database_properties"),
]


class TestAsyncClientSimpleDelegation:
    @pytest.mark.parametrize(
        "method,args,kwargs,handler_method",
        _SIMPLE_ASYNC_DELEGATION_CASES,
        ids=[c[0] for c in _SIMPLE_ASYNC_DELEGATION_CASES],
    )
    @pytest.mark.asyncio
    async def test_delegation(self, method, args, kwargs, handler_method):
        client, handler = _make_client()
        await getattr(client, method)(*args, **kwargs)
        getattr(handler, handler_method).assert_called_once()


class TestAsyncClientAliasAndServerOps:
    @pytest.mark.asyncio
    async def test_describe_alias(self):
        client, handler = _make_client()
        handler.describe_alias.return_value = {"alias": "a1"}
        await client.describe_alias("a1")
        handler.describe_alias.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_aliases(self):
        client, handler = _make_client()
        handler.list_aliases.return_value = ["a1"]
        result = await client.list_aliases("col")
        assert "a1" in result

    @pytest.mark.asyncio
    async def test_using_database(self):
        client, handler = _make_client()
        handler.describe_database.return_value = {"db_name": "mydb"}
        await client.using_database("mydb")
        assert client._config.db_name == "mydb"

    @pytest.mark.asyncio
    async def test_list_resource_groups(self):
        client, handler = _make_client()
        handler.list_resource_groups.return_value = ["rg1"]
        assert "rg1" in await client.list_resource_groups()

    @pytest.mark.asyncio
    async def test_get_compaction_state(self):
        client, handler = _make_client()
        state = MagicMock()
        state.state_name = "Completed"
        handler.get_compaction_state.return_value = state
        assert await client.get_compaction_state(42) == "Completed"

    @pytest.mark.asyncio
    async def test_get_compaction_plans(self):
        client, handler = _make_client()
        handler.get_compaction_plans.return_value = MagicMock()
        await client.get_compaction_plans(42)
        handler.get_compaction_plans.assert_called_once()


class TestAsyncClientRBACOps:
    @pytest.mark.asyncio
    async def test_describe_user_with_groups(self):
        client, handler = _make_client()
        role = MagicMock()
        role.role_name = "admin"
        item = MagicMock()
        item.roles = ["admin"]
        group_info = MagicMock()
        group_info.groups = [item]
        res = MagicMock()
        res.results = [MagicMock()]
        handler.describe_user.return_value = res
        with patch("pymilvus.milvus_client.async_milvus_client.UserInfo") as mock_ui:
            mock_ui.return_value = group_info
            result = await client.describe_user("alice")
            assert result["user_name"] == "alice"

    @pytest.mark.asyncio
    async def test_describe_user_empty_results(self):
        client, handler = _make_client()
        res = MagicMock()
        res.results = []
        handler.describe_user.return_value = res
        assert await client.describe_user("alice") == {}

    @pytest.mark.asyncio
    async def test_describe_role(self):
        client, handler = _make_client()
        res = MagicMock()
        res.groups = []
        handler.select_grant_for_one_role.return_value = res
        result = await client.describe_role("admin")
        assert result["role"] == "admin"
        assert result["privileges"] == []

    @pytest.mark.asyncio
    async def test_list_roles(self):
        client, handler = _make_client()
        with patch("pymilvus.milvus_client.async_milvus_client.RoleInfo") as mock_ri:
            g = MagicMock()
            g.role_name = "admin"
            mock_ri.return_value = MagicMock(groups=[g])
            handler.list_roles.return_value = MagicMock()
            assert "admin" in await client.list_roles()

    @pytest.mark.asyncio
    async def test_list_privilege_groups(self):
        client, handler = _make_client()
        grp = MagicMock()
        grp.group_name = "grp1"
        priv = MagicMock()
        priv.name = "Insert"
        grp.privileges = [priv]
        handler.list_privilege_groups.return_value = [grp]
        result = await client.list_privilege_groups()
        assert len(result) == 1
        assert result[0]["privilege_group"] == "grp1"
        assert result[0]["privileges"] == ["Insert"]


class TestAsyncClientQueryBranches:
    @pytest.mark.asyncio
    async def test_query_filter_and_ids_raises(self):
        client, _ = _make_client()
        with pytest.raises(ParamError):
            await client.query("col", filter="id > 0", ids=[1, 2])

    @pytest.mark.asyncio
    async def test_query_non_string_filter_raises(self):
        client, _ = _make_client()
        with pytest.raises(DataTypeNotMatchException):
            await client.query("col", filter=123)

    @pytest.mark.asyncio
    async def test_query_ids_scalar_converts(self):
        client, handler = _make_client()
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        handler._get_schema.return_value = (schema, 100)
        handler.query.return_value = [{"id": 42}]
        assert await client.query("col", ids=42) == [{"id": 42}]

    @pytest.mark.asyncio
    async def test_get_empty_ids_returns_early(self):
        client, handler = _make_client()
        assert await client.get("col", ids=[]) == []
        handler.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_scalar_id(self):
        client, handler = _make_client()
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        handler._get_schema.return_value = (schema, 100)
        handler.query.return_value = [{"id": 1}]
        await client.get("col", ids=1)
        handler.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_filter_non_str_raises(self):
        client, _ = _make_client()
        with pytest.raises(DataTypeNotMatchException):
            await client.delete("col", filter=123)

    @pytest.mark.asyncio
    async def test_delete_pks_kwarg_invalid_raises(self):
        client, _ = _make_client()
        with pytest.raises(TypeError):
            await client.delete("col", pks=3.14)

    @pytest.mark.asyncio
    async def test_upsert_exception_propagates(self):
        client, handler = _make_client()
        handler.upsert_rows.side_effect = RuntimeError("upsert failed")
        with pytest.raises(RuntimeError, match="upsert failed"):
            await client.upsert("col", [{"id": 1}])


class TestAsyncClientOptimize:
    @pytest.mark.asyncio
    async def test_is_collection_loaded_true(self):
        client, handler = _make_client()
        handler.get_load_state.return_value = LoadState.Loaded
        assert await client._is_collection_loaded("col") is True

    @pytest.mark.asyncio
    async def test_is_collection_loaded_false(self):
        client, handler = _make_client()
        handler.get_load_state.return_value = LoadState.NotLoad
        assert await client._is_collection_loaded("col") is False

    @pytest.mark.asyncio
    async def test_wait_for_indexes_empty_returns(self):
        client, handler = _make_client()
        task = MagicMock()
        await client._wait_for_indexes(task, "col", [])
        handler.wait_for_creating_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_for_indexes_calls_wait(self):
        client, handler = _make_client()
        task = MagicMock()
        task.check_cancelled = MagicMock()
        await client._wait_for_indexes(task, "col", ["vec_idx"])
        handler.wait_for_creating_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_compaction_completed(self):
        client, handler = _make_client()
        state = MagicMock()
        state.state = 2
        handler.get_compaction_state.return_value = state
        task = MagicMock()
        task.check_cancelled = MagicMock()
        await client._wait_for_compaction_with_cancel(task, 42)
        handler.get_compaction_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_compaction_failed_raises(self):
        client, handler = _make_client()
        state = MagicMock()
        state.state = 3
        handler.get_compaction_state.return_value = state
        task = MagicMock()
        task.check_cancelled = MagicMock()
        with pytest.raises(MilvusException, match="Compaction 42 failed"):
            await client._wait_for_compaction_with_cancel(task, 42)

    @pytest.mark.asyncio
    async def test_execute_optimize_success(self):
        client, handler = _make_client()
        handler.compact.return_value = 99
        state = MagicMock()
        state.state = 2
        handler.get_compaction_state.return_value = state
        handler.get_load_state.return_value = LoadState.NotLoad
        schema = {"fields": [{"name": "id", "type": DataType.INT64}]}
        handler._get_schema.return_value = schema
        handler.list_indexes.return_value = []
        task = MagicMock()
        task.check_cancelled = MagicMock()
        task.set_progress = MagicMock()
        task.progress_history = MagicMock(return_value=[])
        task._target_size = None
        result = await client._execute_optimize(task, "col", None, None)
        assert result.collection_name == "col"

    @pytest.mark.asyncio
    async def test_optimize_wait_true(self):
        client, handler = _make_client()
        handler.compact.return_value = 99
        state = MagicMock()
        state.state = 2
        handler.get_compaction_state.return_value = state
        handler.get_load_state.return_value = LoadState.NotLoad
        schema = {"fields": [{"name": "id", "type": DataType.INT64}]}
        handler._get_schema.return_value = schema
        handler.list_indexes.return_value = []
        result = await client.optimize("col", wait=True)
        assert isinstance(result, OptimizeResult)

    @pytest.mark.asyncio
    async def test_optimize_wait_false_returns_task(self):
        client, _handler = _make_client()
        with patch.object(client, "_execute_optimize", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = MagicMock()
            task = await client.optimize("col", wait=False)
            assert isinstance(task, AsyncOptimizeTask)
