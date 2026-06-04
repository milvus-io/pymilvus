"""Tests for AsyncGrpcHandler initialization and setup.

Coverage: Initialization, context manager, secure channel, close operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.exceptions import MilvusException, ParamError


def _mock_channel() -> MagicMock:
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    return channel


def _setup_secure_channel_mocks(
    mock_secure_channel: MagicMock, mock_credentials: MagicMock
) -> None:
    mock_secure_channel.return_value = _mock_channel()
    mock_credentials.return_value = MagicMock()


class TestAsyncGrpcHandlerInit:
    """Tests for AsyncGrpcHandler initialization and setup."""

    def test_init_with_channel(self) -> None:
        """Test AsyncGrpcHandler initialization with channel"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        assert handler._async_channel == mock_channel

    def test_init_stores_connect_reserved_from_option(self) -> None:
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel, option={"cluster_id": "c1"})
        assert handler._connect_reserved == {"cluster_id": "c1"}

    def test_init_connect_reserved_defaults_empty(self) -> None:
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        assert handler._connect_reserved == {}

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

    @pytest.mark.asyncio
    async def test_close_retired_channel_logs_close_failure(self) -> None:
        """Retiring an old reconnect channel logs close failures without raising."""
        mock_channel = _mock_channel()
        handler = AsyncGrpcHandler(channel=mock_channel)
        retired_channel = MagicMock()
        retired_channel.close = AsyncMock(side_effect=RuntimeError("close failed"))

        with patch("pymilvus.client.async_grpc_handler.logger.warning") as mock_warning:
            await handler._close_retired_channel(retired_channel, grace=1)

        mock_warning.assert_called_once()

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
    async def test_ensure_channel_ready_forwards_connect_reserved(self) -> None:
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        mock_channel.channel_ready = AsyncMock()
        handler = AsyncGrpcHandler(channel=mock_channel, option={"cluster_id": "c1"})
        handler._is_channel_ready = False

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.identifier = 1
        mock_stub.Connect = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ), patch("pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
            mock_prepare.register_request.return_value = MagicMock()
            await handler.ensure_channel_ready()
            mock_prepare.register_request.assert_called_once()
            _, kwargs = mock_prepare.register_request.call_args
            assert kwargs.get("cluster_id") == "c1"

    @pytest.mark.asyncio
    async def test_ensure_channel_ready_already_ready(self) -> None:
        """Test ensure_channel_ready when already ready"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        await handler.ensure_channel_ready()
        mock_channel.channel_ready.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_channel_ready_rpc_failure_raises_milvus_exception(self) -> None:
        """Connect RPC failures map to the public async connection error."""
        mock_channel = _mock_channel()
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = False
        handler._setup_identifier_interceptor_for_channel = AsyncMock(
            side_effect=grpc.RpcError("unavailable")
        )

        with pytest.raises(MilvusException, match="Fail connecting to server on"):
            await handler.ensure_channel_ready()

    def test_setup_authorization_interceptor_appends_header(self) -> None:
        """Authorization setup appends a generated interceptor to the final channel."""
        mock_channel = _mock_channel()
        handler = AsyncGrpcHandler(channel=mock_channel)

        handler._setup_authorization_interceptor("root", "Milvus", "")

        assert handler._async_authorization_interceptor is not None
        assert handler._async_authorization_interceptor in mock_channel._unary_unary_interceptors

    def test_build_stub_reuses_authorization_and_one_time_log_level(self) -> None:
        """Stub building reuses existing async interceptors and consumes log level."""
        mock_channel = _mock_channel()
        handler = AsyncGrpcHandler(channel=mock_channel)
        authorization_interceptor = MagicMock()
        handler._async_authorization_interceptor = authorization_interceptor
        handler._log_level = "debug"
        next_channel = _mock_channel()

        with patch("pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
            handler._build_stub(next_channel)

        assert authorization_interceptor in next_channel._unary_unary_interceptors
        assert len(next_channel._unary_unary_interceptors) == 2
        assert handler._log_level is None

    def test_setup_secure_channel(self) -> None:
        """Test setting up secure channel"""
        with patch("pymilvus.client.async_grpc_handler.grpc.aio.secure_channel") as mock_sec, patch(
            "pymilvus.client.async_grpc_handler.grpc.ssl_channel_credentials"
        ) as mock_creds, patch(
            "pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"
        ):
            _setup_secure_channel_mocks(mock_sec, mock_creds)

            AsyncGrpcHandler(uri="localhost:19530", secure=True)
            mock_sec.assert_called_once()

    def test_setup_secure_channel_with_server_pem_and_server_name(self, tmp_path) -> None:
        """Secure channel setup reads server PEM and forwards server name override."""
        server_pem = tmp_path / "server.pem"
        server_pem.write_bytes(b"server-cert")
        with patch("pymilvus.client.async_grpc_handler.grpc.aio.secure_channel") as mock_sec, patch(
            "pymilvus.client.async_grpc_handler.grpc.ssl_channel_credentials"
        ) as mock_creds, patch(
            "pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"
        ):
            _setup_secure_channel_mocks(mock_sec, mock_creds)

            AsyncGrpcHandler(
                uri="localhost:19530",
                secure=True,
                server_pem_path=str(server_pem),
                server_name="milvus.test",
            )

            mock_creds.assert_called_once()
            assert mock_creds.call_args.kwargs["root_certificates"] == b"server-cert"
            assert ("grpc.ssl_target_name_override", "milvus.test") in mock_sec.call_args.kwargs[
                "options"
            ]

    def test_setup_secure_channel_with_client_cert_chain(self, tmp_path) -> None:
        """Secure channel setup reads CA, private key, and client certificate files."""
        ca_pem = tmp_path / "ca.pem"
        client_key = tmp_path / "client.key"
        client_pem = tmp_path / "client.pem"
        ca_pem.write_bytes(b"ca-cert")
        client_key.write_bytes(b"client-key")
        client_pem.write_bytes(b"client-cert")
        with patch("pymilvus.client.async_grpc_handler.grpc.aio.secure_channel") as mock_sec, patch(
            "pymilvus.client.async_grpc_handler.grpc.ssl_channel_credentials"
        ) as mock_creds, patch(
            "pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub"
        ):
            _setup_secure_channel_mocks(mock_sec, mock_creds)

            AsyncGrpcHandler(
                uri="localhost:19530",
                secure=True,
                ca_pem_path=str(ca_pem),
                client_key_path=str(client_key),
                client_pem_path=str(client_pem),
            )

            mock_creds.assert_called_once()
            assert mock_creds.call_args.kwargs["root_certificates"] == b"ca-cert"
            assert mock_creds.call_args.kwargs["private_key"] == b"client-key"
            assert mock_creds.call_args.kwargs["certificate_chain"] == b"client-cert"
