"""Tests for AsyncGrpcHandler database operations.

Coverage: Database create, drop, list, describe, alter operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler


class TestAsyncGrpcHandlerDatabase:
    """Tests for database operations."""

    @pytest.mark.asyncio
    async def test_create_database(self) -> None:
        """Test create_database async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.CreateDatabase = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.create_database_req.return_value = mock_request

            await handler.create_database("test_db", timeout=30)

            mock_stub.CreateDatabase.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_database(self) -> None:
        """Test drop_database async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.DropDatabase = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.drop_database_req.return_value = mock_request

            await handler.drop_database("test_db", timeout=30)

            mock_stub.DropDatabase.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_database(self) -> None:
        """Test list_database async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.db_names = ["default", "test_db"]
        mock_stub.ListDatabases = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_request = MagicMock()
            mock_prepare.list_database_req.return_value = mock_request

            result = await handler.list_database(timeout=30)

            assert result == ["default", "test_db"]

    @pytest.mark.asyncio
    async def test_describe_database(self) -> None:
        """Test describe_database async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.db_name = "test_db"
        mock_response.properties = []
        mock_response.db_id = 1
        mock_response.created_timestamp = 123456
        mock_stub.DescribeDatabase = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_database_req.return_value = MagicMock()
            result = await handler.describe_database("test_db")
            assert result is not None

    @pytest.mark.asyncio
    async def test_alter_database(self) -> None:
        """Test alter_database async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterDatabase = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.alter_database_req.return_value = MagicMock()
            await handler.alter_database("test_db", properties={"key": "value"})
            mock_stub.AlterDatabase.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_database_properties(self) -> None:
        """Test drop_database_properties async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterDatabase = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.drop_database_properties_req.return_value = MagicMock()
            await handler.drop_database_properties("test_db", ["key1", "key2"])
            mock_stub.AlterDatabase.assert_called_once()
