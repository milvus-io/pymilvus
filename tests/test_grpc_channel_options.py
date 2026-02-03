"""
Tests for gRPC channel options: keepalive defaults and user-configurable grpc_options.
"""

from unittest.mock import MagicMock, patch

import grpc
import pytest
from grpc._cython import cygrpc

from pymilvus.client.grpc_handler import GrpcHandler


class TestGrpcHandlerChannelOptions:
    """Tests for GrpcHandler gRPC channel options."""

    def test_default_grpc_options_empty(self):
        """grpc_options defaults to empty dict when not provided."""
        handler = GrpcHandler(channel=MagicMock())
        assert handler._grpc_options == {}

    def test_custom_grpc_options_stored(self):
        """User-provided grpc_options are stored on the handler."""
        custom_opts = {"grpc.keepalive_time_ms": 3000}
        handler = GrpcHandler(channel=MagicMock(), grpc_options=custom_opts)
        assert handler._grpc_options == custom_opts

    @patch("pymilvus.client.grpc_handler.grpc.insecure_channel")
    def test_default_keepalive_values(self, mock_insecure_channel):
        """Default channel options include optimized keepalive settings."""
        mock_insecure_channel.return_value = MagicMock()

        handler = GrpcHandler.__new__(GrpcHandler)
        handler._channel = None
        handler._secure = False
        handler._grpc_options = {}
        handler._address = "localhost:19530"
        handler._log_level = None
        handler._authorization_interceptor = None
        handler._setup_grpc_channel()

        mock_insecure_channel.assert_called_once()
        _, call_kwargs = mock_insecure_channel.call_args
        opts = dict(call_kwargs["options"])

        assert opts["grpc.keepalive_time_ms"] == 10000
        assert opts["grpc.keepalive_timeout_ms"] == 5000
        assert opts["grpc.keepalive_permit_without_calls"] is True
        assert opts["grpc.enable_retries"] == 1
        assert opts[cygrpc.ChannelArgKey.max_send_message_length] == -1
        assert opts[cygrpc.ChannelArgKey.max_receive_message_length] == -1

    @patch("pymilvus.client.grpc_handler.grpc.insecure_channel")
    def test_user_override_keepalive(self, mock_insecure_channel):
        """User-provided grpc_options override default keepalive values."""
        mock_insecure_channel.return_value = MagicMock()

        handler = GrpcHandler.__new__(GrpcHandler)
        handler._channel = None
        handler._secure = False
        handler._grpc_options = {"grpc.keepalive_time_ms": 3000, "grpc.keepalive_timeout_ms": 3000}
        handler._address = "localhost:19530"
        handler._log_level = None
        handler._authorization_interceptor = None
        handler._setup_grpc_channel()

        _, call_kwargs = mock_insecure_channel.call_args
        opts = dict(call_kwargs["options"])

        assert opts["grpc.keepalive_time_ms"] == 3000
        assert opts["grpc.keepalive_timeout_ms"] == 3000
        # Non-overridden defaults remain
        assert opts["grpc.keepalive_permit_without_calls"] is True

    @patch("pymilvus.client.grpc_handler.grpc.insecure_channel")
    def test_user_adds_new_option(self, mock_insecure_channel):
        """User can add new gRPC options not in the defaults."""
        mock_insecure_channel.return_value = MagicMock()

        handler = GrpcHandler.__new__(GrpcHandler)
        handler._channel = None
        handler._secure = False
        handler._grpc_options = {"grpc.http2.max_pings_without_data": 0}
        handler._address = "localhost:19530"
        handler._log_level = None
        handler._authorization_interceptor = None
        handler._setup_grpc_channel()

        _, call_kwargs = mock_insecure_channel.call_args
        opts = dict(call_kwargs["options"])

        assert opts["grpc.http2.max_pings_without_data"] == 0
        # Defaults still present
        assert opts["grpc.keepalive_time_ms"] == 10000

    @patch("pymilvus.client.grpc_handler.grpc.secure_channel")
    def test_secure_channel_has_keepalive(self, mock_secure_channel):
        """Secure channel also receives keepalive options."""
        mock_secure_channel.return_value = MagicMock()

        handler = GrpcHandler.__new__(GrpcHandler)
        handler._channel = None
        handler._secure = True
        handler._server_name = ""
        handler._server_pem_path = ""
        handler._client_pem_path = ""
        handler._client_key_path = ""
        handler._ca_pem_path = ""
        handler._grpc_options = {}
        handler._address = "localhost:19530"
        handler._log_level = None
        handler._authorization_interceptor = None
        handler._setup_grpc_channel()

        _, call_kwargs = mock_secure_channel.call_args
        opts = dict(call_kwargs["options"])

        assert opts["grpc.keepalive_time_ms"] == 10000
        assert opts["grpc.keepalive_timeout_ms"] == 5000
        assert opts["grpc.keepalive_permit_without_calls"] is True


class TestAsyncGrpcHandlerChannelOptions:
    """Tests for AsyncGrpcHandler gRPC channel options."""

    def test_async_default_grpc_options_empty(self):
        """grpc_options defaults to empty dict when not provided."""
        from pymilvus.client.async_grpc_handler import AsyncGrpcHandler

        handler = AsyncGrpcHandler.__new__(AsyncGrpcHandler)
        handler._async_channel = MagicMock()
        handler._grpc_options = {}
        assert handler._grpc_options == {}

    @patch("pymilvus.client.async_grpc_handler.grpc.aio.insecure_channel")
    def test_async_default_keepalive_values(self, mock_insecure_channel):
        """Async handler default channel options include optimized keepalive settings."""
        mock_insecure_channel.return_value = MagicMock()

        handler = self._create_handler()
        handler._grpc_options = {}
        handler._setup_grpc_channel()

        _, call_kwargs = mock_insecure_channel.call_args
        opts = dict(call_kwargs["options"])

        assert opts["grpc.keepalive_time_ms"] == 10000
        assert opts["grpc.keepalive_timeout_ms"] == 5000
        assert opts["grpc.keepalive_permit_without_calls"] is True

    @patch("pymilvus.client.async_grpc_handler.grpc.aio.insecure_channel")
    def test_async_user_override(self, mock_insecure_channel):
        """User-provided grpc_options override defaults in async handler."""
        mock_insecure_channel.return_value = MagicMock()

        handler = self._create_handler()
        handler._grpc_options = {"grpc.keepalive_time_ms": 3000}
        handler._setup_grpc_channel()

        _, call_kwargs = mock_insecure_channel.call_args
        opts = dict(call_kwargs["options"])

        assert opts["grpc.keepalive_time_ms"] == 3000
        assert opts["grpc.keepalive_timeout_ms"] == 5000

    def _create_handler(self):
        from pymilvus.client.async_grpc_handler import AsyncGrpcHandler

        handler = AsyncGrpcHandler.__new__(AsyncGrpcHandler)
        handler._async_channel = None
        handler._secure = False
        handler._grpc_options = {}
        handler._address = "localhost:19530"
        handler._log_level = None
        handler._async_authorization_interceptor = None
        return handler
