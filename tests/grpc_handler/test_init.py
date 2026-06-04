"""Tests for GrpcHandler initialization and connection management."""

from unittest.mock import MagicMock, patch

import grpc
import pytest
from pymilvus.client.grpc_handler import GrpcHandler, ReconnectHandler
from pymilvus.exceptions import MilvusException, ParamError


def _connect_response(identifier=42):
    response = MagicMock()
    response.status.code = 0
    response.status.error_code = 0
    response.identifier = identifier
    return response


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
        with patch.object(handler, "_setup_identifier_interceptor", side_effect=grpc.RpcError()):
            with pytest.raises(MilvusException, match="Fail connecting"):
                handler._wait_for_channel_ready(timeout=0.1)

    def test_wait_for_channel_ready_no_channel(self):
        handler = GrpcHandler(channel=None)
        handler._channel = None
        with pytest.raises(MilvusException):
            handler._wait_for_channel_ready()

    def test_wait_for_channel_ready_generic_exception(self):
        handler = GrpcHandler(channel=MagicMock())
        with patch.object(
            handler, "_setup_identifier_interceptor", side_effect=RuntimeError("unexpected error")
        ):
            with pytest.raises(RuntimeError, match="unexpected error"):
                handler._wait_for_channel_ready(timeout=1.0)

    def test_wait_for_channel_ready_none_timeout_uses_default(self):
        """_wait_for_channel_ready(timeout=None) must use a finite default, not block forever.

        When MilvusClient is constructed with the default timeout=None and the URI is
        unreachable, the call should raise MilvusException instead of hanging indefinitely.
        This test ensures the implementation substitutes a finite timeout before issuing
        the validation Connect RPC.
        """
        handler = GrpcHandler(channel=MagicMock())
        with patch.object(
            handler, "_setup_identifier_interceptor", side_effect=grpc.RpcError()
        ) as mock_setup:
            with pytest.raises(MilvusException, match="Fail connecting"):
                handler._wait_for_channel_ready(timeout=None)
            assert mock_setup.call_args.kwargs["timeout"] == 10

    def test_wait_for_channel_ready_success(self):
        handler = GrpcHandler(channel=MagicMock())
        final_channel = MagicMock()
        stub = MagicMock()
        with patch.object(
            handler, "_setup_identifier_interceptor", return_value=(final_channel, stub)
        ) as mock_setup:
            assert handler._wait_for_channel_ready(timeout=None) == (final_channel, stub)
            mock_setup.assert_called_once_with(
                handler._user,
                timeout=10,
            )

    def test_wait_for_channel_ready_with_replacement_channel_uses_local_state(self):
        handler = GrpcHandler(channel=MagicMock())
        replacement_channel = MagicMock()
        replacement_final_channel = MagicMock()
        replacement_stub = MagicMock()
        ready_final_channel = MagicMock()
        ready_stub = MagicMock()

        with patch.object(
            handler, "_setup_identifier_interceptor", return_value=(ready_final_channel, ready_stub)
        ) as mock_setup:
            assert handler._wait_for_channel_ready(
                timeout=2,
                channel=replacement_channel,
                final_channel=replacement_final_channel,
                stub=replacement_stub,
                address="newhost:19530",
            ) == (ready_final_channel, ready_stub)

        mock_setup.assert_called_once_with(
            handler._user,
            timeout=2,
            final_channel=replacement_final_channel,
            stub=replacement_stub,
        )
        assert handler._channel is not replacement_channel

    def test_wait_for_channel_ready_with_replacement_channel_uses_target_address_in_error(self):
        handler = GrpcHandler(channel=MagicMock())

        with patch.object(handler, "_setup_identifier_interceptor", side_effect=grpc.RpcError()):
            with pytest.raises(MilvusException, match="newhost:19530"):
                handler._wait_for_channel_ready(
                    timeout=2,
                    channel=MagicMock(),
                    final_channel=MagicMock(),
                    stub=MagicMock(),
                    address="newhost:19530",
                )

        assert handler._channel is not None

    def test_close_channel_safely_handles_none_and_close_failure(self):
        handler = GrpcHandler(channel=MagicMock())
        channel = MagicMock()
        channel.close.side_effect = RuntimeError("already closed")

        handler._close_channel_safely(None)
        with patch("pymilvus.client.grpc_handler.logger.warning") as mock_warning:
            handler._close_channel_safely(channel)

        mock_warning.assert_called_once()

    def test_move_state_change_callbacks_moves_callbacks_and_logs_failures(self):
        handler = GrpcHandler(channel=MagicMock())
        callbacks = [MagicMock(), MagicMock()]
        handler.callbacks = callbacks
        old_channel = MagicMock()
        new_channel = MagicMock()
        old_channel.unsubscribe.side_effect = [None, RuntimeError("unsubscribe failed")]
        new_channel.subscribe.side_effect = [None, RuntimeError("subscribe failed")]

        with patch("pymilvus.client.grpc_handler.logger.warning") as mock_warning:
            handler._move_state_change_callbacks(old_channel, new_channel)

        old_channel.unsubscribe.assert_any_call(callbacks[0])
        old_channel.unsubscribe.assert_any_call(callbacks[1])
        new_channel.subscribe.assert_any_call(callbacks[0], try_to_connect=True)
        new_channel.subscribe.assert_any_call(callbacks[1], try_to_connect=True)
        assert mock_warning.call_count == 2

    def test_create_secure_channel_with_server_name_and_server_cert(self, tmp_path):
        server_pem = tmp_path / "server.pem"
        server_pem.write_bytes(b"root")
        handler = GrpcHandler(
            channel=MagicMock(),
            secure=True,
            server_name="server.example",
            server_pem_path=str(server_pem),
        )

        with patch("pymilvus.client.grpc_handler.grpc.ssl_channel_credentials") as mock_creds:
            with patch("pymilvus.client.grpc_handler.grpc.secure_channel") as mock_secure:
                creds = MagicMock()
                mock_creds.return_value = creds
                assert handler._create_grpc_channel("securehost:19530") is mock_secure.return_value

        mock_creds.assert_called_once_with(
            root_certificates=b"root",
            private_key=None,
            certificate_chain=None,
        )
        assert ("grpc.ssl_target_name_override", "server.example") in mock_secure.call_args.kwargs[
            "options"
        ]

    def test_create_secure_channel_with_client_cert_chain(self, tmp_path):
        ca_pem = tmp_path / "ca.pem"
        client_key = tmp_path / "client.key"
        client_pem = tmp_path / "client.pem"
        ca_pem.write_bytes(b"ca")
        client_key.write_bytes(b"key")
        client_pem.write_bytes(b"cert")
        handler = GrpcHandler(
            channel=MagicMock(),
            secure=True,
            ca_pem_path=str(ca_pem),
            client_key_path=str(client_key),
            client_pem_path=str(client_pem),
        )

        with patch("pymilvus.client.grpc_handler.grpc.ssl_channel_credentials") as mock_creds:
            with patch("pymilvus.client.grpc_handler.grpc.secure_channel"):
                handler._create_grpc_channel("securehost:19530")

        mock_creds.assert_called_once_with(
            root_certificates=b"ca",
            private_key=b"key",
            certificate_chain=b"cert",
        )

    def test_set_onetime_loglevel_rebuilds_stub_with_loglevel_interceptor(self):
        handler = GrpcHandler(channel=MagicMock())
        log_interceptor = MagicMock()
        final_channel = MagicMock()
        stub = MagicMock()

        with patch(
            "pymilvus.client.grpc_handler.interceptor.header_adder_interceptor",
            return_value=log_interceptor,
        ) as mock_header:
            with patch(
                "pymilvus.client.grpc_handler.grpc.intercept_channel",
                return_value=final_channel,
            ) as mock_intercept:
                with patch(
                    "pymilvus.client.grpc_handler.milvus_pb2_grpc.MilvusServiceStub",
                    return_value=stub,
                ):
                    handler.set_onetime_loglevel("debug")

        mock_header.assert_called_once_with(["log_level"], ["debug"])
        mock_intercept.assert_called_once_with(handler._channel, log_interceptor)
        assert handler._log_level is None
        assert handler._stub is stub

    def test_setup_identifier_interceptor_updates_current_handler_state(self):
        handler = GrpcHandler(channel=MagicMock())
        handler._stub = MagicMock()
        original_stub = handler._stub
        original_stub.Connect.return_value = _connect_response(identifier=99)
        identifier_interceptor = MagicMock()
        final_channel = MagicMock()
        stub = MagicMock()

        with patch(
            "pymilvus.client.grpc_handler.interceptor.header_adder_interceptor",
            return_value=identifier_interceptor,
        ) as mock_header:
            with patch(
                "pymilvus.client.grpc_handler.grpc.intercept_channel",
                return_value=final_channel,
            ):
                with patch(
                    "pymilvus.client.grpc_handler.milvus_pb2_grpc.MilvusServiceStub",
                    return_value=stub,
                ):
                    assert handler._setup_identifier_interceptor("user", timeout=3) == (
                        final_channel,
                        stub,
                    )

        mock_header.assert_called_once_with(["identifier"], ["99"])
        original_stub.Connect.assert_called_once()
        assert original_stub.Connect.call_args.kwargs["timeout"] == 3
        assert handler._identifier == 99
        assert handler._identifier_interceptor is identifier_interceptor
        assert handler._final_channel is final_channel
        assert handler._stub is stub

    def test_setup_identifier_interceptor_can_validate_replacement_stub(self):
        handler = GrpcHandler(channel=MagicMock())
        original_stub = handler._stub
        replacement_stub = MagicMock()
        replacement_stub.Connect.return_value = _connect_response(identifier=100)
        replacement_final_channel = MagicMock()
        ready_final_channel = MagicMock()
        ready_stub = MagicMock()

        with patch("pymilvus.client.grpc_handler.grpc.intercept_channel") as mock_intercept:
            with patch(
                "pymilvus.client.grpc_handler.milvus_pb2_grpc.MilvusServiceStub",
                return_value=ready_stub,
            ):
                mock_intercept.return_value = ready_final_channel
                assert handler._setup_identifier_interceptor(
                    "user",
                    timeout=4,
                    final_channel=replacement_final_channel,
                    stub=replacement_stub,
                ) == (ready_final_channel, ready_stub)

        replacement_stub.Connect.assert_called_once()
        assert replacement_stub.Connect.call_args.kwargs["timeout"] == 4
        assert handler._stub is original_stub

    def test_internal_register_forwards_connect_reserved(self):
        handler = GrpcHandler(channel=MagicMock(), option={"cluster_id": "c1"})
        handler._stub = MagicMock()
        handler._stub.Connect.return_value = _connect_response()
        with patch("pymilvus.client.grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.grpc_handler.check_status"
        ):
            mock_prepare.register_request.return_value = MagicMock()
            handler._GrpcHandler__internal_register("user", "host")
            mock_prepare.register_request.assert_called_once_with("user", "host", cluster_id="c1")

    def test_internal_register_forwards_timeout_to_connect(self):
        handler = GrpcHandler(channel=MagicMock())
        handler._stub = MagicMock()
        handler._stub.Connect.return_value = _connect_response()

        assert handler._internal_register("user", "host", timeout=0.25) == 42
        handler._stub.Connect.assert_called_once()
        assert handler._stub.Connect.call_args.kwargs["timeout"] == 0.25
