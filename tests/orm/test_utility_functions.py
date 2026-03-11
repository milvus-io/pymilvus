"""Tests for pymilvus/orm/utility.py -- thin wrapper functions that delegate to connection handlers.

Each utility function follows the pattern:
    context = connections._generate_call_context(using)
    return _get_connection(using).handler_method(args, timeout=timeout, context=context)

We mock connections._fetch_handler and connections._generate_call_context to verify
correct delegation without needing a real Milvus server.
"""

from unittest import mock
from unittest.mock import Mock

import pytest
from pymilvus.orm.utility import (
    alter_alias,
    create_alias,
    create_resource_group,
    create_user,
    delete_user,
    describe_resource_group,
    drop_alias,
    drop_collection,
    drop_resource_group,
    flush_all,
    get_server_type,
    get_server_version,
    has_collection,
    has_partition,
    index_building_progress,
    list_aliases,
    list_collections,
    list_indexes,
    list_resource_groups,
    list_roles,
    list_user,
    list_usernames,
    list_users,
    load_state,
    loading_progress,
    rename_collection,
    reset_password,
    transfer_node,
    transfer_replica,
    truncate_collection,
    update_password,
    update_resource_groups,
    wait_for_index_building_complete,
    wait_for_loading_complete,
)


@pytest.fixture
def mock_conn():
    """Mock the connections singleton used by utility functions.

    Yields (mock_handler, mock_connections) where mock_handler is the object
    returned by connections._fetch_handler(), and mock_connections is the
    patched connections module-level singleton.
    """
    with mock.patch("pymilvus.orm.utility.connections") as mock_connections:
        mock_handler = Mock()
        mock_connections._fetch_handler.return_value = mock_handler
        mock_connections._generate_call_context.return_value = Mock()
        yield mock_handler, mock_connections


# ---------------------------------------------------------------------------
# Collection operations
# ---------------------------------------------------------------------------


class TestHasCollection:
    def test_has_collection_true(self, mock_conn):
        handler, _conns = mock_conn
        handler.has_collection.return_value = True
        result = has_collection("test_coll")
        handler.has_collection.assert_called_once()
        assert result is True

    def test_has_collection_false(self, mock_conn):
        handler, _conns = mock_conn
        handler.has_collection.return_value = False
        result = has_collection("nonexistent")
        assert result is False

    def test_has_collection_custom_using(self, mock_conn):
        handler, conns = mock_conn
        handler.has_collection.return_value = True
        has_collection("coll", using="custom")
        conns._fetch_handler.assert_called_with("custom")


class TestHasPartition:
    def test_has_partition_true(self, mock_conn):
        handler, _ = mock_conn
        handler.has_partition.return_value = True
        result = has_partition("coll", "part")
        handler.has_partition.assert_called_once()
        assert result is True

    def test_has_partition_false(self, mock_conn):
        handler, _ = mock_conn
        handler.has_partition.return_value = False
        assert has_partition("coll", "missing") is False


class TestDropCollection:
    def test_drop_collection(self, mock_conn):
        handler, _ = mock_conn
        drop_collection("coll", timeout=5.0)
        handler.drop_collection.assert_called_once()
        call_kwargs = handler.drop_collection.call_args
        assert call_kwargs[0][0] == "coll"
        assert call_kwargs[1]["timeout"] == 5.0


class TestTruncateCollection:
    def test_truncate_collection(self, mock_conn):
        handler, _ = mock_conn
        truncate_collection("coll", timeout=3.0)
        handler.truncate_collection.assert_called_once()


class TestRenameCollection:
    def test_rename_collection(self, mock_conn):
        handler, _ = mock_conn
        rename_collection("old", "new", new_db_name="db2", timeout=2.0)
        handler.rename_collections.assert_called_once()
        kwargs = handler.rename_collections.call_args[1]
        assert kwargs["old_name"] == "old"
        assert kwargs["new_name"] == "new"
        assert kwargs["new_db_name"] == "db2"


class TestListCollections:
    def test_list_collections(self, mock_conn):
        handler, _ = mock_conn
        handler.list_collections.return_value = ["coll1", "coll2"]
        result = list_collections(timeout=1.0)
        handler.list_collections.assert_called_once()
        assert result == ["coll1", "coll2"]


# ---------------------------------------------------------------------------
# Loading / index progress
# ---------------------------------------------------------------------------


class TestLoadingProgress:
    def test_loading_progress(self, mock_conn):
        handler, _ = mock_conn
        handler.get_loading_progress.return_value = 100.0
        result = loading_progress("coll")
        handler.get_loading_progress.assert_called_once()
        assert result == {"loading_progress": "100%"}

    def test_loading_progress_partial(self, mock_conn):
        handler, _ = mock_conn
        handler.get_loading_progress.return_value = 42.5
        result = loading_progress("coll", partition_names=["p1"])
        assert result == {"loading_progress": "42%"}


class TestLoadState:
    def test_load_state(self, mock_conn):
        handler, _ = mock_conn
        handler.get_load_state.return_value = "Loaded"
        result = load_state("coll")
        handler.get_load_state.assert_called_once()
        assert result == "Loaded"

    def test_load_state_with_partitions(self, mock_conn):
        handler, _ = mock_conn
        handler.get_load_state.return_value = "Loading"
        result = load_state("coll", partition_names=["p1", "p2"])
        assert result == "Loading"


class TestWaitForLoadingComplete:
    def test_wait_no_partitions(self, mock_conn):
        handler, _ = mock_conn
        wait_for_loading_complete("coll")
        handler.wait_for_loading_collection.assert_called_once()
        handler.wait_for_loading_partitions.assert_not_called()

    def test_wait_with_empty_partitions(self, mock_conn):
        handler, _ = mock_conn
        wait_for_loading_complete("coll", partition_names=[])
        handler.wait_for_loading_collection.assert_called_once()
        handler.wait_for_loading_partitions.assert_not_called()

    def test_wait_with_partitions(self, mock_conn):
        handler, _ = mock_conn
        wait_for_loading_complete("coll", partition_names=["p1"])
        handler.wait_for_loading_partitions.assert_called_once()
        handler.wait_for_loading_collection.assert_not_called()


class TestIndexBuildingProgress:
    def test_index_building_progress(self, mock_conn):
        handler, _ = mock_conn
        handler.get_index_build_progress.return_value = {"total_rows": 100, "indexed_rows": 50}
        result = index_building_progress("coll", index_name="idx")
        handler.get_index_build_progress.assert_called_once()
        assert result == {"total_rows": 100, "indexed_rows": 50}


class TestWaitForIndexBuildingComplete:
    def test_wait_for_index_complete(self, mock_conn):
        handler, _ = mock_conn
        handler.wait_for_creating_index.return_value = (True,)
        result = wait_for_index_building_complete("coll", index_name="idx")
        handler.wait_for_creating_index.assert_called_once()
        assert result is True


# ---------------------------------------------------------------------------
# Alias operations
# ---------------------------------------------------------------------------


class TestCreateAlias:
    def test_create_alias(self, mock_conn):
        handler, _ = mock_conn
        create_alias("coll", "my_alias", timeout=2.0)
        handler.create_alias.assert_called_once()
        args = handler.create_alias.call_args[0]
        assert args[0] == "coll"
        assert args[1] == "my_alias"


class TestDropAlias:
    def test_drop_alias(self, mock_conn):
        handler, _ = mock_conn
        drop_alias("my_alias", timeout=2.0)
        handler.drop_alias.assert_called_once()


class TestAlterAlias:
    def test_alter_alias(self, mock_conn):
        handler, _ = mock_conn
        alter_alias("coll", "my_alias", timeout=2.0)
        handler.alter_alias.assert_called_once()


class TestListAliases:
    def test_list_aliases(self, mock_conn):
        handler, _ = mock_conn
        handler.list_aliases.return_value = {"aliases": ["a1", "a2"]}
        result = list_aliases("coll")
        handler.list_aliases.assert_called_once()
        assert result == ["a1", "a2"]


# ---------------------------------------------------------------------------
# Credential / user operations
# ---------------------------------------------------------------------------


class TestCreateUser:
    def test_create_user(self, mock_conn):
        handler, _ = mock_conn
        create_user("admin", "pass123", timeout=5.0)
        handler.create_user.assert_called_once_with("admin", "pass123", timeout=5.0)


class TestDeleteUser:
    def test_delete_user(self, mock_conn):
        handler, _ = mock_conn
        delete_user("admin", timeout=5.0)
        handler.delete_user.assert_called_once_with("admin", timeout=5.0)


class TestListUsernames:
    def test_list_usernames(self, mock_conn):
        handler, _ = mock_conn
        handler.list_usernames.return_value = ["root", "admin"]
        result = list_usernames()
        handler.list_usernames.assert_called_once()
        assert result == ["root", "admin"]


class TestResetPassword:
    def test_reset_password(self, mock_conn):
        handler, _ = mock_conn
        reset_password("user1", "old", "new", timeout=3.0)
        handler.reset_password.assert_called_once_with("user1", "old", "new", timeout=3.0)


class TestUpdatePassword:
    def test_update_password(self, mock_conn):
        handler, _ = mock_conn
        update_password("user1", "old", "new", timeout=3.0)
        handler.update_password.assert_called_once_with("user1", "old", "new", timeout=3.0)


# ---------------------------------------------------------------------------
# Role / user info
# ---------------------------------------------------------------------------


class TestListRoles:
    def test_list_roles(self, mock_conn):
        handler, _ = mock_conn
        handler.select_all_role.return_value = ["admin", "readonly"]
        result = list_roles(include_user_info=True)
        handler.select_all_role.assert_called_once_with(True, timeout=None)
        assert result == ["admin", "readonly"]


class TestListUser:
    def test_list_user(self, mock_conn):
        handler, _ = mock_conn
        handler.select_one_user.return_value = {"username": "root", "roles": ["admin"]}
        result = list_user("root", include_role_info=True)
        handler.select_one_user.assert_called_once_with("root", True, timeout=None)
        assert result == {"username": "root", "roles": ["admin"]}


class TestListUsers:
    def test_list_users(self, mock_conn):
        handler, _ = mock_conn
        handler.select_all_user.return_value = [{"username": "root"}]
        result = list_users(include_role_info=False)
        handler.select_all_user.assert_called_once_with(False, timeout=None)
        assert result == [{"username": "root"}]


class TestGetServerVersion:
    def test_get_server_version(self, mock_conn):
        handler, _ = mock_conn
        handler.get_server_version.return_value = "2.3.0"
        result = get_server_version()
        handler.get_server_version.assert_called_once_with(timeout=None)
        assert result == "2.3.0"


# ---------------------------------------------------------------------------
# Resource group operations
# ---------------------------------------------------------------------------


class TestCreateResourceGroup:
    def test_create_resource_group(self, mock_conn):
        handler, _ = mock_conn
        create_resource_group("rg1", timeout=2.0)
        handler.create_resource_group.assert_called_once_with("rg1", 2.0)


class TestDropResourceGroup:
    def test_drop_resource_group(self, mock_conn):
        handler, _ = mock_conn
        drop_resource_group("rg1", timeout=2.0)
        handler.drop_resource_group.assert_called_once()


class TestDescribeResourceGroup:
    def test_describe_resource_group(self, mock_conn):
        handler, _ = mock_conn
        handler.describe_resource_group.return_value = {"name": "rg1", "num_nodes": 3}
        result = describe_resource_group("rg1", timeout=2.0)
        handler.describe_resource_group.assert_called_once()
        assert result == {"name": "rg1", "num_nodes": 3}


class TestListResourceGroups:
    def test_list_resource_groups(self, mock_conn):
        handler, _ = mock_conn
        handler.list_resource_groups.return_value = ["rg1", "rg2"]
        result = list_resource_groups()
        handler.list_resource_groups.assert_called_once()
        assert result == ["rg1", "rg2"]


class TestUpdateResourceGroups:
    def test_update_resource_groups(self, mock_conn):
        handler, _ = mock_conn
        configs = {"rg1": Mock()}
        update_resource_groups(configs, timeout=2.0)
        handler.update_resource_groups.assert_called_once_with(configs, 2.0)


# ---------------------------------------------------------------------------
# Transfer operations
# ---------------------------------------------------------------------------


class TestTransferNode:
    def test_transfer_node(self, mock_conn):
        handler, _ = mock_conn
        transfer_node("src", "dst", 3, timeout=2.0)
        handler.transfer_node.assert_called_once()
        args = handler.transfer_node.call_args[0]
        assert args[0] == "src"
        assert args[1] == "dst"
        assert args[2] == 3


class TestTransferReplica:
    def test_transfer_replica(self, mock_conn):
        handler, _ = mock_conn
        transfer_replica("src", "dst", "coll", 2, timeout=2.0)
        handler.transfer_replica.assert_called_once()
        args = handler.transfer_replica.call_args[0]
        assert args[0] == "src"
        assert args[1] == "dst"
        assert args[2] == "coll"
        assert args[3] == 2


# ---------------------------------------------------------------------------
# Flush / server type
# ---------------------------------------------------------------------------


class TestFlushAll:
    def test_flush_all(self, mock_conn):
        handler, _ = mock_conn
        flush_all(timeout=5.0)
        handler.flush_all.assert_called_once()


class TestGetServerType:
    def test_get_server_type(self, mock_conn):
        handler, _ = mock_conn
        handler.get_server_type.return_value = "milvus"
        result = get_server_type()
        handler.get_server_type.assert_called_once()
        assert result == "milvus"

    def test_get_server_type_zilliz(self, mock_conn):
        handler, _ = mock_conn
        handler.get_server_type.return_value = "zilliz"
        result = get_server_type(using="cloud")
        assert result == "zilliz"


# ---------------------------------------------------------------------------
# List indexes
# ---------------------------------------------------------------------------


class TestListIndexes:
    def test_list_indexes_no_field_filter(self, mock_conn):
        handler, _ = mock_conn
        idx1 = Mock()
        idx1.index_name = "idx_vec"
        idx1.field_name = "vec"
        idx2 = Mock()
        idx2.index_name = "idx_text"
        idx2.field_name = "text"
        handler.list_indexes.return_value = [idx1, idx2]

        result = list_indexes("coll")
        assert result == ["idx_vec", "idx_text"]

    def test_list_indexes_with_field_filter(self, mock_conn):
        handler, _ = mock_conn
        idx1 = Mock()
        idx1.index_name = "idx_vec"
        idx1.field_name = "vec"
        idx2 = Mock()
        idx2.index_name = "idx_text"
        idx2.field_name = "text"
        handler.list_indexes.return_value = [idx1, idx2]

        result = list_indexes("coll", field_name="vec")
        assert result == ["idx_vec"]

    def test_list_indexes_with_none_entries(self, mock_conn):
        handler, _ = mock_conn
        idx1 = Mock()
        idx1.index_name = "idx_vec"
        idx1.field_name = "vec"
        handler.list_indexes.return_value = [None, idx1, None]

        result = list_indexes("coll")
        assert result == ["idx_vec"]

    def test_list_indexes_empty(self, mock_conn):
        handler, _ = mock_conn
        handler.list_indexes.return_value = []
        result = list_indexes("coll")
        assert result == []


# ---------------------------------------------------------------------------
# Parametrized tests for simple delegating functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func, args, handler_method",
    [
        (has_collection, ("coll",), "has_collection"),
        (has_partition, ("coll", "part"), "has_partition"),
        (drop_collection, ("coll",), "drop_collection"),
        (truncate_collection, ("coll",), "truncate_collection"),
        (list_usernames, (), "list_usernames"),
        (get_server_version, (), "get_server_version"),
    ],
    ids=[
        "has_collection",
        "has_partition",
        "drop_collection",
        "truncate_collection",
        "list_usernames",
        "get_server_version",
    ],
)
class TestDelegatingFunctions:
    """Verify that thin wrapper functions call the right handler method."""

    def test_delegates_to_handler(self, func, args, handler_method, mock_conn):
        handler, _conns = mock_conn
        func(*args)
        assert getattr(handler, handler_method).called


@pytest.mark.parametrize(
    "func, args, handler_method",
    [
        (load_state, ("coll",), "get_load_state"),
        (index_building_progress, ("coll",), "get_index_build_progress"),
        (drop_resource_group, ("rg",), "drop_resource_group"),
        (describe_resource_group, ("rg",), "describe_resource_group"),
        (list_resource_groups, (), "list_resource_groups"),
        (flush_all, (), "flush_all"),
    ],
    ids=[
        "load_state",
        "index_building_progress",
        "drop_resource_group",
        "describe_resource_group",
        "list_resource_groups",
        "flush_all",
    ],
)
class TestContextDelegatingFunctions:
    """Verify that functions that generate call context also delegate correctly."""

    def test_generates_context_and_delegates(self, func, args, handler_method, mock_conn):
        handler, conns = mock_conn
        func(*args)
        conns._generate_call_context.assert_called()
        assert getattr(handler, handler_method).called
