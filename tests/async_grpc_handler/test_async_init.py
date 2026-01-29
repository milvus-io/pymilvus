"""Tests for AsyncGrpcHandler initialization and setup.

Coverage: Initialization, context manager, secure channel, close operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.exceptions import ParamError


class TestAsyncGrpcHandlerInit:
    """Tests for AsyncGrpcHandler initialization and setup."""

    def test_init_with_channel(self) -> None:
        """Test AsyncGrpcHandler initialization with channel"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        assert handler._async_channel == mock_channel

    def test_init_with_host_port(self) -> None:
        """Test AsyncGrpcHandler initialization with host and port"""
        with patch("pymilvus.client.async_grpc_handler.grpc.aio.insecure_channel") as mock_ch:
            with patch("pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
                mock_channel = MagicMock()
                mock_channel._unary_unary_interceptors = []
                mock_ch.return_value = mock_channel
                handler = AsyncGrpcHandler(host="127.0.0.1", port="19530")
                assert handler._address == "127.0.0.1:19530"

    def test_init_secure_not_bool_raises(self) -> None:
        """Test that non-bool secure parameter raises ParamError"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        with pytest.raises(ParamError):
            AsyncGrpcHandler(channel=mock_channel, secure="yes")

    def test_init_with_token(self) -> None:
        """Test initialization with token creates authorization interceptor"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        with patch("pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
            AsyncGrpcHandler(channel=mock_channel, token="test_token")
            assert len(mock_channel._unary_unary_interceptors) > 0

    def test_init_with_user_password(self) -> None:
        """Test initialization with user/password creates authorization interceptor"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        with patch("pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
            AsyncGrpcHandler(channel=mock_channel, user="admin", password="pass")
            assert len(mock_channel._unary_unary_interceptors) > 0

    def test_context_manager(self) -> None:
        """Test __enter__ and __exit__"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        with handler as h:
            assert h == handler

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test close method"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        mock_channel.close = AsyncMock()
        handler = AsyncGrpcHandler(channel=mock_channel)
        await handler.close()
        mock_channel.close.assert_called_once()
        assert handler._async_channel is None

    def test_server_address_property(self) -> None:
        """Test server_address property"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel, address="localhost:19530")
        assert handler.server_address == "localhost:19530"

    def test_get_server_type(self) -> None:
        """Test get_server_type method"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel, address="localhost:19530")
        with patch("pymilvus.client.async_grpc_handler.get_server_type", return_value="milvus"):
            result = handler.get_server_type()
            assert result == "milvus"

    @pytest.mark.asyncio
    async def test_ensure_channel_ready(self) -> None:
        """Test ensure_channel_ready method"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        mock_channel.channel_ready = AsyncMock()
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = False

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.identifier = 12345
        mock_stub.Connect = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ), patch("pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
            mock_prepare.register_request.return_value = MagicMock()
            await handler.ensure_channel_ready()
            assert handler._is_channel_ready is True

    @pytest.mark.asyncio
    async def test_ensure_channel_ready_already_ready(self) -> None:
        """Test ensure_channel_ready when already ready"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        await handler.ensure_channel_ready()
        mock_channel.channel_ready.assert_not_called()

    def test_setup_secure_channel(self) -> None:
        """Test setting up secure channel"""
        with patch("pymilvus.client.async_grpc_handler.grpc.aio.secure_channel") as mock_sec, patch(
            "pymilvus.client.async_grpc_handler.grpc.ssl_channel_credentials"
        ) as mock_creds, patch(
            "pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"
        ):
            mock_channel = MagicMock()
            mock_channel._unary_unary_interceptors = []
            mock_sec.return_value = mock_channel
            mock_creds.return_value = MagicMock()

            AsyncGrpcHandler(uri="localhost:19530", secure=True)
            mock_sec.assert_called_once()
