"""Tests for pymilvus/orm/role.py — Role class."""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus.orm.role import Role

CONNECTIONS_PREFIX = "pymilvus.orm.role.connections"


@pytest.fixture
def mock_handler():
    """Patch connections._fetch_handler to return a MagicMock handler."""
    handler = MagicMock()
    with patch(f"{CONNECTIONS_PREFIX}._fetch_handler", return_value=handler):
        yield handler


@pytest.fixture
def role(mock_handler):
    """Create a Role instance with the default alias."""
    return Role("test_role")


@pytest.fixture
def role_custom(mock_handler):
    """Create a Role instance with a custom alias."""
    return Role("admin_role", using="custom_alias")


class TestRoleInit:
    """Tests for Role.__init__."""

    def test_stores_name_and_using(self, mock_handler):
        r = Role("myrole")
        assert r._name == "myrole"
        assert r._using == "default"

    def test_custom_using(self, mock_handler):
        r = Role("myrole", using="other")
        assert r._using == "other"

    def test_stores_extra_kwargs(self, mock_handler):
        r = Role("myrole", foo="bar", baz=42)
        assert r._kwargs == {"foo": "bar", "baz": 42}


class TestRoleNameProperty:
    """Tests for Role.name property."""

    def test_name_returns_role_name(self, role):
        assert role.name == "test_role"


class TestRoleGetConnection:
    """Tests for Role._get_connection."""

    def test_delegates_to_fetch_handler(self, role, mock_handler):
        conn = role._get_connection()
        assert conn is mock_handler


class TestRoleCreate:
    """Tests for Role.create."""

    def test_calls_create_role(self, role, mock_handler):
        role.create()
        mock_handler.create_role.assert_called_once_with("test_role")


class TestRoleDrop:
    """Tests for Role.drop."""

    def test_calls_drop_role(self, role, mock_handler):
        role.drop()
        mock_handler.drop_role.assert_called_once_with("test_role")


class TestRoleAddUser:
    """Tests for Role.add_user."""

    def test_calls_add_user_to_role(self, role, mock_handler):
        role.add_user("alice")
        mock_handler.add_user_to_role.assert_called_once_with("alice", "test_role")


class TestRoleRemoveUser:
    """Tests for Role.remove_user."""

    def test_calls_remove_user_from_role(self, role, mock_handler):
        role.remove_user("bob")
        mock_handler.remove_user_from_role.assert_called_once_with("bob", "test_role")


class TestRoleGetUsers:
    """Tests for Role.get_users."""

    def test_returns_users_from_first_group(self, role, mock_handler):
        group = MagicMock()
        group.users = ["alice", "bob"]
        roles_result = MagicMock()
        roles_result.groups = [group]
        mock_handler.select_one_role.return_value = roles_result

        result = role.get_users()

        mock_handler.select_one_role.assert_called_once_with("test_role", True)
        assert result == ["alice", "bob"]

    def test_returns_empty_list_when_no_groups(self, role, mock_handler):
        roles_result = MagicMock()
        roles_result.groups = []
        mock_handler.select_one_role.return_value = roles_result

        result = role.get_users()

        mock_handler.select_one_role.assert_called_once_with("test_role", True)
        assert result == []


class TestRoleIsExist:
    """Tests for Role.is_exist."""

    def test_returns_true_when_groups_present(self, role, mock_handler):
        roles_result = MagicMock()
        roles_result.groups = [MagicMock()]
        mock_handler.select_one_role.return_value = roles_result

        assert role.is_exist() is True
        mock_handler.select_one_role.assert_called_once_with("test_role", False)

    def test_returns_false_when_no_groups(self, role, mock_handler):
        roles_result = MagicMock()
        roles_result.groups = []
        mock_handler.select_one_role.return_value = roles_result

        assert role.is_exist() is False
        mock_handler.select_one_role.assert_called_once_with("test_role", False)


class TestRoleGrant:
    """Tests for Role.grant."""

    def test_calls_grant_privilege(self, role, mock_handler):
        role.grant("Collection", "my_collection", "Insert")
        mock_handler.grant_privilege.assert_called_once_with(
            "test_role", "Collection", "my_collection", "Insert", ""
        )

    def test_with_db_name(self, role, mock_handler):
        role.grant("Collection", "my_collection", "Insert", db_name="mydb")
        mock_handler.grant_privilege.assert_called_once_with(
            "test_role", "Collection", "my_collection", "Insert", "mydb"
        )


class TestRoleRevoke:
    """Tests for Role.revoke."""

    def test_calls_revoke_privilege(self, role, mock_handler):
        role.revoke("Collection", "my_collection", "Insert")
        mock_handler.revoke_privilege.assert_called_once_with(
            "test_role", "Collection", "my_collection", "Insert", ""
        )

    def test_with_db_name(self, role, mock_handler):
        role.revoke("Collection", "my_collection", "Insert", db_name="mydb")
        mock_handler.revoke_privilege.assert_called_once_with(
            "test_role", "Collection", "my_collection", "Insert", "mydb"
        )


class TestRoleGrantV2:
    """Tests for Role.grant_v2."""

    def test_calls_grant_privilege_v2(self, role, mock_handler):
        role.grant_v2("Insert", "my_collection")
        mock_handler.grant_privilege_v2.assert_called_once_with(
            "test_role", "Insert", "my_collection", db_name=None
        )

    def test_with_db_name(self, role, mock_handler):
        role.grant_v2("Insert", "my_collection", db_name="mydb")
        mock_handler.grant_privilege_v2.assert_called_once_with(
            "test_role", "Insert", "my_collection", db_name="mydb"
        )


class TestRoleRevokeV2:
    """Tests for Role.revoke_v2."""

    def test_calls_revoke_privilege_v2(self, role, mock_handler):
        role.revoke_v2("Insert", "my_collection")
        mock_handler.revoke_privilege_v2.assert_called_once_with(
            "test_role", "Insert", "my_collection", db_name=None
        )

    def test_with_db_name(self, role, mock_handler):
        role.revoke_v2("Insert", "my_collection", db_name="mydb")
        mock_handler.revoke_privilege_v2.assert_called_once_with(
            "test_role", "Insert", "my_collection", db_name="mydb"
        )


class TestRoleListGrant:
    """Tests for Role.list_grant."""

    def test_calls_select_grant_for_role_and_object(self, role, mock_handler):
        sentinel = MagicMock(name="grant_info")
        mock_handler.select_grant_for_role_and_object.return_value = sentinel

        result = role.list_grant("Collection", "my_collection")

        mock_handler.select_grant_for_role_and_object.assert_called_once_with(
            "test_role", "Collection", "my_collection", ""
        )
        assert result is sentinel

    def test_with_db_name(self, role, mock_handler):
        role.list_grant("Collection", "my_collection", db_name="mydb")
        mock_handler.select_grant_for_role_and_object.assert_called_once_with(
            "test_role", "Collection", "my_collection", "mydb"
        )


class TestRoleListGrants:
    """Tests for Role.list_grants."""

    def test_calls_select_grant_for_one_role(self, role, mock_handler):
        sentinel = MagicMock(name="grant_info")
        mock_handler.select_grant_for_one_role.return_value = sentinel

        result = role.list_grants()

        mock_handler.select_grant_for_one_role.assert_called_once_with("test_role", "")
        assert result is sentinel

    def test_with_db_name(self, role, mock_handler):
        role.list_grants(db_name="mydb")
        mock_handler.select_grant_for_one_role.assert_called_once_with("test_role", "mydb")


class TestRoleCreatePrivilegeGroup:
    """Tests for Role.create_privilege_group."""

    def test_calls_create_privilege_group(self, role, mock_handler):
        role.create_privilege_group("my_group")
        mock_handler.create_privilege_group.assert_called_once_with("my_group")


class TestRoleDropPrivilegeGroup:
    """Tests for Role.drop_privilege_group."""

    def test_calls_drop_privilege_group(self, role, mock_handler):
        role.drop_privilege_group("my_group")
        mock_handler.drop_privilege_group.assert_called_once_with("my_group")


class TestRoleListPrivilegeGroups:
    """Tests for Role.list_privilege_groups."""

    def test_calls_list_privilege_groups(self, role, mock_handler):
        sentinel = MagicMock(name="privilege_groups")
        mock_handler.list_privilege_groups.return_value = sentinel

        result = role.list_privilege_groups()

        mock_handler.list_privilege_groups.assert_called_once_with()
        assert result is sentinel


class TestRoleAddPrivilegesToGroup:
    """Tests for Role.add_privileges_to_group."""

    def test_calls_add_privileges_to_group(self, role, mock_handler):
        privs = ["Insert", "Release"]
        role.add_privileges_to_group("my_group", privs)
        mock_handler.add_privileges_to_group.assert_called_once_with("my_group", privs)


class TestRoleRemovePrivilegesFromGroup:
    """Tests for Role.remove_privileges_from_group."""

    def test_calls_remove_privileges_from_group(self, role, mock_handler):
        privs = ["Insert", "Release"]
        role.remove_privileges_from_group("my_group", privs)
        mock_handler.remove_privileges_from_group.assert_called_once_with("my_group", privs)


class TestRoleCustomAlias:
    """Tests that verify methods use the custom alias for _fetch_handler."""

    def test_create_uses_custom_alias(self, mock_handler):
        r = Role("admin_role", using="custom_alias")
        r.create()
        mock_handler.create_role.assert_called_once_with("admin_role")

    def test_drop_uses_custom_alias(self, mock_handler):
        r = Role("admin_role", using="custom_alias")
        r.drop()
        mock_handler.drop_role.assert_called_once_with("admin_role")
