"""Tests for GrpcHandler initialization and connection management."""

from unittest.mock import MagicMock, patch

import grpc
import pytest
from pymilvus.client.grpc_handler import GrpcHandler, ReconnectHandler
from pymilvus.exceptions import MilvusException, ParamError


class TestReconnectHandler:
    """Tests for ReconnectHandler class."""

    def test_init(self):
        mock_conns = MagicMock()
        handler = ReconnectHandler(mock_conns, "test_conn", {"host": "localhost"})
        assert handler.connection_name == "test_conn"
        assert handler._kwargs == {"host": "localhost"}
        assert handler.is_idle_state is False

    def test_reset_db_name(self):
        handler = ReconnectHandler(MagicMock(), "test", {"db_name": "old"})
        handler.reset_db_name("new")
        assert handler._kwargs["db_name"] == "new"

    def test_reconnect_on_idle_sets_state(self):
        handler = ReconnectHandler(MagicMock(), "test", {})
        mock_state = MagicMock(value=(None, "idle"))
        with patch.object(handler, "check_state_and_reconnect_later"):
            handler.reconnect_on_idle(mock_state)
            assert handler.is_idle_state is True

    def test_reconnect_on_idle_non_idle_clears_state(self):
        handler = ReconnectHandler(MagicMock(), "test", {})
        handler.is_idle_state = True
        handler.reconnect_on_idle(MagicMock(value=(None, "ready")))
        assert handler.is_idle_state is False


class TestGrpcHandlerInit:
    """Tests for GrpcHandler initialization."""

    def test_init_with_channel(self):
        mock_channel = MagicMock()
        handler = GrpcHandler(channel=mock_channel)
        assert handler._channel == mock_channel

    def test_init_stores_connect_reserved_from_option(self):
        handler = GrpcHandler(channel=MagicMock(), option={"cluster_id": "c1"})
        assert handler._connect_reserved == {"cluster_id": "c1"}

    def test_init_connect_reserved_defaults_empty(self):
        handler = GrpcHandler(channel=MagicMock())
        assert handler._connect_reserved == {}

    def test_init_with_uri(self):
        with patch("pymilvus.client.grpc_handler.grpc.insecure_channel") as mock_ch:
            with patch("pymilvus.client.grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
                mock_ch.return_value = MagicMock()
                handler = GrpcHandler(uri="http://localhost:19530")
                assert handler._address == "localhost:19530"

    def test_init_with_host_port(self):
        with patch("pymilvus.client.grpc_handler.grpc.insecure_channel") as mock_ch:
            with patch("pymilvus.client.grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
                mock_ch.return_value = MagicMock()
                handler = GrpcHandler(host="127.0.0.1", port="19530")
                assert handler._address == "127.0.0.1:19530"

    def test_init_secure_channel(self):
        with patch("pymilvus.client.grpc_handler.grpc.secure_channel") as mock_sec:
            with patch("pymilvus.client.grpc_handler.grpc.ssl_channel_credentials"):
                with patch("pymilvus.client.grpc_handler.milvus_pb2_grpc.MilvusServiceStub"):
                    mock_sec.return_value = MagicMock()
                    GrpcHandler(uri="localhost:19530", secure=True)
                    mock_sec.assert_called_once()

    def test_init_with_token(self):
        handler = GrpcHandler(channel=MagicMock(), token="test_token")
        assert handler._authorization_interceptor is not None

    def test_init_with_user_password(self):
        handler = GrpcHandler(channel=MagicMock(), user="admin", password="pass")
        assert handler._authorization_interceptor is not None

    def test_init_secure_not_bool_raises(self):
        with pytest.raises(ParamError):
            GrpcHandler(uri="localhost:19530", secure="yes")

    def test_context_manager(self):
        handler = GrpcHandler(channel=MagicMock())
        with handler as h:
            assert h == handler


class TestGrpcHandlerConnectionMgmt:
    """Tests for connection management."""

    def test_close(self):
        mock_ch = MagicMock()
        handler = GrpcHandler(channel=mock_ch)
        handler.close()
        mock_ch.close.assert_called_once()
        assert handler._channel is None

    def test_server_address_property(self):
        handler = GrpcHandler(channel=MagicMock())
        handler._address = "localhost:19530"
        assert handler.server_address == "localhost:19530"

    def test_get_server_type(self):
        handler = GrpcHandler(channel=MagicMock())
        handler._address = "localhost:19530"
        with patch("pymilvus.client.grpc_handler.get_server_type", return_value="milvus"):
            assert handler.get_server_type() == "milvus"

    def test_register_state_change_callback(self):
        mock_ch = MagicMock()
        handler = GrpcHandler(channel=mock_ch)
        cb = MagicMock()
        handler.register_state_change_callback(cb)
        assert cb in handler.callbacks
        mock_ch.subscribe.assert_called_once()

    def test_deregister_state_change_callbacks(self):
        mock_ch = MagicMock()
        handler = GrpcHandler(channel=mock_ch)
        handler.callbacks = [MagicMock(), MagicMock()]
        handler.deregister_state_change_callbacks()
        assert len(handler.callbacks) == 0
        assert mock_ch.unsubscribe.call_count == 2

    def test_wait_for_channel_ready_timeout(self):
        handler = GrpcHandler(channel=MagicMock())
        with patch("pymilvus.client.grpc_handler.grpc.channel_ready_future") as mock_ready:
            mock_ready.return_value.result.side_effect = grpc.FutureTimeoutError()
            with pytest.raises(MilvusException):
                handler._wait_for_channel_ready(timeout=0.1)

    def test_wait_for_channel_ready_no_channel(self):
        handler = GrpcHandler(channel=None)
        handler._channel = None
        with pytest.raises(MilvusException):
            handler._wait_for_channel_ready()

    def test_wait_for_channel_ready_generic_exception(self):
        handler = GrpcHandler(channel=MagicMock())
        with patch("pymilvus.client.grpc_handler.grpc.channel_ready_future") as mock_ready:
            mock_ready.return_value.result.side_effect = RuntimeError("unexpected error")
            with pytest.raises(RuntimeError, match="unexpected error"):
                handler._wait_for_channel_ready(timeout=1.0)

    def test_internal_register_forwards_connect_reserved(self):
        handler = GrpcHandler(channel=MagicMock(), option={"cluster_id": "c1"})
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.identifier = 42
        handler._stub = MagicMock()
        handler._stub.Connect.return_value = mock_response
        with patch("pymilvus.client.grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.grpc_handler.check_status"
        ):
            mock_prepare.register_request.return_value = MagicMock()
            handler._GrpcHandler__internal_register("user", "host")
            mock_prepare.register_request.assert_called_once_with("user", "host", cluster_id="c1")
