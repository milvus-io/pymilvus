"""Tests for GrpcHandler authentication and authorization operations."""

from unittest.mock import MagicMock

from pymilvus.client.types import ResourceGroupConfig

from .conftest import make_response, make_status


class TestGrpcHandlerUserOps:
    """Tests for user and role operations."""

    def test_create_user(self, handler):
        handler._stub.CreateCredential.return_value = make_status()
        handler.create_user("user", "pass")
        handler._stub.CreateCredential.assert_called_once()

    def test_delete_user(self, handler):
        handler._stub.DeleteCredential.return_value = make_status()
        handler.delete_user("user")
        handler._stub.DeleteCredential.assert_called_once()

    def test_update_password(self, handler):
        handler._stub.UpdateCredential.return_value = make_status()
        handler.update_password("user", "old", "new")
        handler._stub.UpdateCredential.assert_called_once()

    def test_list_usernames(self, handler):
        handler._stub.ListCredUsers.return_value = make_response(usernames=["u1", "u2"])
        assert handler.list_usernames() == ["u1", "u2"]

    def test_create_role(self, handler):
        handler._stub.CreateRole.return_value = make_status()
        handler.create_role("role")
        handler._stub.CreateRole.assert_called_once()

    def test_drop_role(self, handler):
        handler._stub.DropRole.return_value = make_status()
        handler.drop_role("role")
        handler._stub.DropRole.assert_called_once()

    def test_add_user_to_role(self, handler):
        handler._stub.OperateUserRole.return_value = make_status()
        handler.add_user_to_role("user", "role")
        handler._stub.OperateUserRole.assert_called_once()

    def test_remove_user_from_role(self, handler):
        handler._stub.OperateUserRole.return_value = make_status()
        handler.remove_user_from_role("user", "role")
        handler._stub.OperateUserRole.assert_called_once()

    def test_select_one_role(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = []
        handler._stub.SelectRole.return_value = mock_resp
        result = handler.select_one_role("admin", include_user_info=True)
        assert result is not None

    def test_select_all_role(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = []
        handler._stub.SelectRole.return_value = mock_resp
        result = handler.select_all_role(include_user_info=False)
        assert result is not None

    def test_select_one_user(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = []
        handler._stub.SelectUser.return_value = mock_resp
        result = handler.select_one_user("admin", include_role_info=True)
        assert result is not None

    def test_select_all_user(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = []
        handler._stub.SelectUser.return_value = mock_resp
        result = handler.select_all_user(include_role_info=False)
        assert result is not None


class TestGrpcHandlerPrivilegeOps:
    """Tests for privilege operations."""

    def test_grant_privilege(self, handler):
        handler._stub.OperatePrivilege.return_value = make_status()
        handler.grant_privilege("role", "Collection", "*", "Query", "default")
        handler._stub.OperatePrivilege.assert_called_once()

    def test_revoke_privilege(self, handler):
        handler._stub.OperatePrivilege.return_value = make_status()
        handler.revoke_privilege("role", "Collection", "*", "Query", "default")
        handler._stub.OperatePrivilege.assert_called_once()

    def test_select_grant_for_one_role(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.entities = []
        handler._stub.SelectGrant.return_value = mock_resp
        result = handler.select_grant_for_one_role("admin", "default")
        assert result is not None

    def test_select_grant_for_role_and_object(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.entities = []
        handler._stub.SelectGrant.return_value = mock_resp
        result = handler.select_grant_for_role_and_object("admin", "Collection", "test", "default")
        assert result is not None


class TestGrpcHandlerPrivilegeV2Ops:
    """Tests for privilege v2 operations."""

    def test_grant_privilege_v2(self, handler):
        handler._stub.OperatePrivilegeV2.return_value = make_status()
        handler.grant_privilege_v2("role", "Query", "coll", "default")
        handler._stub.OperatePrivilegeV2.assert_called_once()

    def test_revoke_privilege_v2(self, handler):
        handler._stub.OperatePrivilegeV2.return_value = make_status()
        handler.revoke_privilege_v2("role", "Query", "coll", "default")
        handler._stub.OperatePrivilegeV2.assert_called_once()


class TestGrpcHandlerPrivilegeGroupOps:
    """Tests for privilege group operations."""

    def test_create_privilege_group(self, handler):
        handler._stub.CreatePrivilegeGroup.return_value = make_status()
        handler.create_privilege_group("pg")
        handler._stub.CreatePrivilegeGroup.assert_called_once()

    def test_drop_privilege_group(self, handler):
        handler._stub.DropPrivilegeGroup.return_value = make_status()
        handler.drop_privilege_group("pg")
        handler._stub.DropPrivilegeGroup.assert_called_once()

    def test_list_privilege_groups(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.privilege_groups = []
        handler._stub.ListPrivilegeGroups.return_value = mock_resp
        result = handler.list_privilege_groups()
        assert result is not None

    def test_add_privileges_to_group(self, handler):
        handler._stub.OperatePrivilegeGroup.return_value = make_status()
        handler.add_privileges_to_group("pg", ["Query", "Insert"])
        handler._stub.OperatePrivilegeGroup.assert_called_once()

    def test_remove_privileges_from_group(self, handler):
        handler._stub.OperatePrivilegeGroup.return_value = make_status()
        handler.remove_privileges_from_group("pg", ["Query"])
        handler._stub.OperatePrivilegeGroup.assert_called_once()


class TestGrpcHandlerResourceGroupOps:
    """Tests for resource group operations."""

    def test_create_resource_group(self, handler):
        handler._stub.CreateResourceGroup.return_value = make_status()
        handler.create_resource_group("rg")
        handler._stub.CreateResourceGroup.assert_called_once()

    def test_drop_resource_group(self, handler):
        handler._stub.DropResourceGroup.return_value = make_status()
        handler.drop_resource_group("rg")
        handler._stub.DropResourceGroup.assert_called_once()

    def test_list_resource_groups(self, handler):
        handler._stub.ListResourceGroups.return_value = make_response(
            resource_groups=["__default", "rg1"]
        )
        assert handler.list_resource_groups() == ["__default", "rg1"]

    def test_describe_resource_group(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.resource_group = MagicMock()
        handler._stub.DescribeResourceGroup.return_value = mock_resp
        result = handler.describe_resource_group("rg")
        assert result is not None

    def test_transfer_replica(self, handler):
        handler._stub.TransferReplica.return_value = make_status()
        handler.transfer_replica("rg1", "rg2", "coll", 1)
        handler._stub.TransferReplica.assert_called_once()

    def test_transfer_node(self, handler):
        handler._stub.TransferNode.return_value = make_status()
        handler.transfer_node("rg1", "rg2", 1)
        handler._stub.TransferNode.assert_called_once()

    def test_update_resource_groups(self, handler):
        handler._stub.UpdateResourceGroups.return_value = make_status()

        config = ResourceGroupConfig(requests={"node_num": 1})
        handler.update_resource_groups({"rg": config})
        handler._stub.UpdateResourceGroups.assert_called_once()


class TestGrpcHandlerLoadBalance:
    """Tests for load balance operations."""

    def test_load_balance(self, handler):
        handler._stub.LoadBalance.return_value = make_status()
        handler.load_balance("coll", 1, [2, 3], [100, 101])
        handler._stub.LoadBalance.assert_called_once()
