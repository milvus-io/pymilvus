"""Tests for AsyncGrpcHandler authentication and authorization operations.

Coverage: User, role, privilege, grant, resource group operations.
"""

import base64
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pymilvus import AsyncMilvusClient, MilvusClient
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.call_context import CallContext


class TestAsyncGrpcHandlerUser:
    """Tests for user operations."""

    @pytest.mark.asyncio
    async def test_create_user(self) -> None:
        """Test create_user async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.CreateCredential = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.create_credential_request.return_value = mock_request

            await handler.create_user("test_user", "password123", timeout=30)

            mock_stub.CreateCredential.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_with_description(self) -> None:
        """Test create_user async API with description."""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.CreateCredential = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            await handler.create_user("test_user", "password123", description="reader account")

            req = mock_stub.CreateCredential.call_args.args[0]
            assert req.description == "reader account"

    @pytest.mark.asyncio
    async def test_drop_user(self) -> None:
        """Test drop_user async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.DeleteCredential = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.delete_credential_request.return_value = MagicMock()
            await handler.drop_user("test_user")
            mock_stub.DeleteCredential.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_password(self) -> None:
        """Test update_password async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.UpdateCredential = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.update_credential_request.return_value = MagicMock()
            await handler.update_password("user", "old_pass", "new_pass")
            mock_stub.UpdateCredential.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_password_with_description(self) -> None:
        """Test update_password async API with description."""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.UpdateCredential = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            await handler.update_password(
                "user", "old_pass", "new_pass", description="updated account"
            )
            req = mock_stub.UpdateCredential.call_args.args[0]
            assert req.username == "user"
            assert req.description == "updated account"

    @pytest.mark.asyncio
    async def test_update_user_description_only(self) -> None:
        """Test update_user async API with description only."""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.UpdateCredential = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            await handler.update_user("user", description="updated account")
            req = mock_stub.UpdateCredential.call_args.args[0]
            assert req.username == "user"
            assert req.oldPassword == ""
            assert req.newPassword == ""
            assert req.description == "updated account"

    @pytest.mark.asyncio
    async def test_update_user_requires_description(self) -> None:
        parameter = inspect.signature(AsyncGrpcHandler.update_user).parameters["description"]
        assert parameter.default is inspect.Parameter.empty

    @pytest.mark.asyncio
    async def test_list_users(self) -> None:
        """Test list_users async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.usernames = ["user1", "user2"]
        mock_stub.ListCredUsers = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result = await handler.list_users()
            assert result == ["user1", "user2"]

    @pytest.mark.asyncio
    async def test_describe_user(self) -> None:
        """Test describe_user async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_result = MagicMock()
        mock_result.user.name = "test_user"
        mock_role = MagicMock()
        mock_role.name = "admin"
        mock_result.roles = [mock_role]
        mock_response.results = [mock_result]
        mock_stub.SelectUser = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.select_user_request.return_value = MagicMock()
            result = await handler.describe_user("test_user", include_role_info=True)
            assert result is not None


class TestAsyncGrpcHandlerRole:
    """Tests for role operations."""

    @pytest.mark.asyncio
    async def test_create_role(self) -> None:
        """Test create_role async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.CreateRole = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.create_role_request.return_value = mock_request

            await handler.create_role("test_role", description="reader role", timeout=30)

            mock_stub.CreateRole.assert_called_once()
            mock_prepare.create_role_request.assert_called_once_with("test_role", "reader role")

    @pytest.mark.asyncio
    async def test_create_role_preserves_positional_context(self) -> None:
        """Test create_role positional context compatibility"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.CreateRole = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        context = CallContext(db_name="db1", client_request_id="req1")
        await handler.create_role("test_role", 30, context)

        req = mock_stub.CreateRole.call_args.args[0]
        kwargs = mock_stub.CreateRole.call_args.kwargs
        assert req.entity.description == ""
        assert kwargs["timeout"] == 30
        assert ("dbname", "db1") in kwargs["metadata"]
        assert ("client-request-id", "req1") in kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_alter_role(self) -> None:
        """Test alter_role async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.AlterRole = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.alter_role_request.return_value = mock_request

            await handler.alter_role("test_role", "updated reader role", timeout=30)

            mock_stub.AlterRole.assert_called_once()
            mock_prepare.alter_role_request.assert_called_once_with(
                "test_role", "updated reader role"
            )

    @pytest.mark.asyncio
    async def test_drop_role(self) -> None:
        """Test drop_role async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.DropRole = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.drop_role_request.return_value = mock_request

            await handler.drop_role("test_role", timeout=30)

            mock_stub.DropRole.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_roles(self) -> None:
        """Test list_roles async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_result = MagicMock()
        mock_result.role.name = "admin"
        mock_response.results = [mock_result]
        mock_stub.SelectRole = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.select_role_request.return_value = MagicMock()
            result = await handler.list_roles(include_user_info=False)
            assert result == [mock_result]

    @pytest.mark.asyncio
    async def test_describe_role(self) -> None:
        """Test describe_role async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_result = MagicMock()
        mock_result.role.name = "admin"
        mock_response.results = [mock_result]
        mock_stub.SelectRole = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.select_role_request.return_value = MagicMock()
            result = await handler.describe_role("admin", include_user_info=True)
            assert result == [mock_result]


class TestAsyncGrpcHandlerGrant:
    """Tests for grant operations."""

    @pytest.mark.asyncio
    async def test_grant_role(self) -> None:
        """Test grant_role async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperateUserRole = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_user_role_request.return_value = MagicMock()
            await handler.grant_role("user1", "admin")
            mock_stub.OperateUserRole.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_role(self) -> None:
        """Test revoke_role async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperateUserRole = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_user_role_request.return_value = MagicMock()
            await handler.revoke_role("user1", "admin")
            mock_stub.OperateUserRole.assert_called_once()

    @pytest.mark.asyncio
    async def test_grant_privilege(self) -> None:
        """Test grant_privilege async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperatePrivilege = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_privilege_request.return_value = MagicMock()
            await handler.grant_privilege("admin", "Collection", "*", "Insert", "default")
            mock_stub.OperatePrivilege.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_privilege(self) -> None:
        """Test revoke_privilege async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperatePrivilege = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_privilege_request.return_value = MagicMock()
            await handler.revoke_privilege("admin", "Collection", "*", "Insert", "default")
            mock_stub.OperatePrivilege.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_grant_for_one_role(self) -> None:
        """Test select_grant_for_one_role async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.entities = []
        mock_stub.SelectGrant = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.select_grant_request.return_value = MagicMock()
            result = await handler.select_grant_for_one_role("admin", "default")
            assert result is not None

    @pytest.mark.asyncio
    async def test_grant_privilege_v2(self) -> None:
        """Test grant_privilege_v2 async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperatePrivilegeV2 = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_privilege_v2_request.return_value = MagicMock()
            await handler.grant_privilege_v2("admin", "Insert", "test_coll", db_name="default")
            mock_stub.OperatePrivilegeV2.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_privilege_v2(self) -> None:
        """Test revoke_privilege_v2 async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperatePrivilegeV2 = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_privilege_v2_request.return_value = MagicMock()
            await handler.revoke_privilege_v2("admin", "Insert", "test_coll", db_name="default")
            mock_stub.OperatePrivilegeV2.assert_called_once()


class TestAsyncGrpcHandlerPrivilegeGroup:
    """Tests for privilege group operations."""

    @pytest.mark.asyncio
    async def test_create_privilege_group(self) -> None:
        """Test create_privilege_group async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.CreatePrivilegeGroup = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.create_privilege_group_req.return_value = MagicMock()
            await handler.create_privilege_group("test_group")
            mock_stub.CreatePrivilegeGroup.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_privilege_group(self) -> None:
        """Test drop_privilege_group async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.DropPrivilegeGroup = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.drop_privilege_group_req.return_value = MagicMock()
            await handler.drop_privilege_group("test_group")
            mock_stub.DropPrivilegeGroup.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_privilege_groups(self) -> None:
        """Test list_privilege_groups async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.privilege_groups = []
        mock_stub.ListPrivilegeGroups = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.list_privilege_groups_req.return_value = MagicMock()
            result = await handler.list_privilege_groups()
            assert result == []

    @pytest.mark.asyncio
    async def test_add_privileges_to_group(self) -> None:
        """Test add_privileges_to_group async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperatePrivilegeGroup = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_privilege_group_req.return_value = MagicMock()
            await handler.add_privileges_to_group("test_group", ["Insert", "Query"])
            mock_stub.OperatePrivilegeGroup.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_privileges_from_group(self) -> None:
        """Test remove_privileges_from_group async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.OperatePrivilegeGroup = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.operate_privilege_group_req.return_value = MagicMock()
            await handler.remove_privileges_from_group("test_group", ["Insert"])
            mock_stub.OperatePrivilegeGroup.assert_called_once()


class TestAsyncGrpcHandlerResourceGroup:
    """Tests for resource group operations."""

    @pytest.mark.asyncio
    async def test_create_resource_group(self) -> None:
        """Test create_resource_group async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.CreateResourceGroup = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.create_resource_group.return_value = MagicMock()
            await handler.create_resource_group("test_rg")
            mock_stub.CreateResourceGroup.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_resource_group(self) -> None:
        """Test drop_resource_group async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.DropResourceGroup = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.drop_resource_group.return_value = MagicMock()
            await handler.drop_resource_group("test_rg")
            mock_stub.DropResourceGroup.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_resource_groups(self) -> None:
        """Test list_resource_groups async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.resource_groups = ["rg1", "rg2"]
        mock_stub.ListResourceGroups = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result = await handler.list_resource_groups()
            assert result == ["rg1", "rg2"]

    @pytest.mark.asyncio
    async def test_describe_resource_group(self) -> None:
        """Test describe_resource_group async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_rg = MagicMock()
        mock_rg.name = "test_rg"
        mock_rg.capacity = 10
        mock_rg.num_available_node = 5
        mock_rg.num_loaded_replica = {}
        mock_rg.num_outgoing_node = {}
        mock_rg.num_incoming_node = {}
        mock_rg.config = None
        mock_rg.nodes = []
        mock_response.resource_group = mock_rg
        mock_stub.DescribeResourceGroup = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_resource_group.return_value = MagicMock()
            result = await handler.describe_resource_group("test_rg")
            assert result is not None

    @pytest.mark.asyncio
    async def test_update_resource_groups(self) -> None:
        """Test update_resource_groups async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.UpdateResourceGroups = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.update_resource_groups.return_value = MagicMock()
            await handler.update_resource_groups({})
            mock_stub.UpdateResourceGroups.assert_called_once()

    @pytest.mark.asyncio
    async def test_transfer_replica(self) -> None:
        """Test transfer_replica async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.TransferReplica = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.transfer_replica.return_value = MagicMock()
            await handler.transfer_replica("src_rg", "tgt_rg", "test_coll", 1)
            mock_stub.TransferReplica.assert_called_once()


class TestAsyncGrpcHandlerCredentialRotation:
    """Tests for rotating the credential attached to a live async channel."""

    @staticmethod
    def _installed_authorizations(channel) -> list:
        """Decode the authorization headers every installed interceptor would add."""
        authorizations = []
        for interceptor in channel._unary_unary_interceptors:
            details = MagicMock()
            details.metadata = None
            new_details, _ = interceptor._fn(details, MagicMock())
            authorizations.extend(
                base64.b64decode(value).decode()
                for header, value in new_details.metadata
                if header == "authorization"
            )
        return authorizations

    def _handler(self) -> AsyncGrpcHandler:
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        return AsyncGrpcHandler(channel=mock_channel, user="root", password="old_pass")

    def test_rotating_the_credential_replaces_the_stale_one(self) -> None:
        """A rotated credential must not leave the superseded one on the channel.

        Interceptors are appended to the channel's list, and each authorization
        interceptor appends its own header, so a stale one left in place keeps
        sending the old credential ahead of the new one.
        """
        handler = self._handler()
        assert self._installed_authorizations(handler._async_channel) == ["root:old_pass"]

        handler._setup_authorization_interceptor("root", "new_pass", None)
        handler._setup_grpc_channel()
        assert self._installed_authorizations(handler._async_channel) == ["root:new_pass"]

        handler._setup_authorization_interceptor("root", "newer_pass", None)
        handler._setup_grpc_channel()
        assert self._installed_authorizations(handler._async_channel) == ["root:newer_pass"]

    def test_rotating_the_credential_installs_exactly_one_interceptor(self) -> None:
        handler = self._handler()
        for password in ("new_pass", "newer_pass", "newest_pass"):
            handler._setup_authorization_interceptor("root", password, None)
            handler._setup_grpc_channel()
            interceptors = handler._async_channel._unary_unary_interceptors
            assert len(interceptors) == 1
            assert handler._async_authorization_interceptor in interceptors

    def test_uninstall_is_a_no_op_without_an_installed_interceptor(self) -> None:
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        assert handler._async_authorization_interceptor is None

        handler._uninstall_authorization_interceptor()

        assert handler._async_authorization_interceptor is None
        assert handler._async_channel._unary_unary_interceptors == []


class TestAsyncMilvusClientUpdatePassword:
    """Tests for AsyncMilvusClient.update_password's reset_connection flag."""

    @staticmethod
    def _client_with_connection() -> tuple:
        client = AsyncMilvusClient.__new__(AsyncMilvusClient)
        conn = MagicMock()
        conn.update_password = AsyncMock()
        client._get_connection = AsyncMock(return_value=conn)
        client._generate_call_context = MagicMock(return_value=None)
        return client, conn

    @pytest.mark.asyncio
    async def test_reset_connection_rebinds_the_channel(self) -> None:
        """reset_connection must re-authorize the live connection, as the sync client does.

        Without it, every later call on the same client keeps sending the password
        that was just replaced.
        """
        client, conn = self._client_with_connection()

        await client.update_password("root", "old_pass", "new_pass", reset_connection=True)

        conn.update_password.assert_awaited_once()
        conn._setup_authorization_interceptor.assert_called_once_with("root", "new_pass", None)
        conn._setup_grpc_channel.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_connection_is_left_alone_by_default(self) -> None:
        client, conn = self._client_with_connection()

        await client.update_password("root", "old_pass", "new_pass")

        conn.update_password.assert_awaited_once()
        conn._setup_authorization_interceptor.assert_not_called()
        conn._setup_grpc_channel.assert_not_called()

    def test_signature_matches_the_sync_client(self) -> None:
        sync_params = inspect.signature(MilvusClient.update_password).parameters
        async_params = inspect.signature(AsyncMilvusClient.update_password).parameters
        assert set(sync_params) == set(async_params)
