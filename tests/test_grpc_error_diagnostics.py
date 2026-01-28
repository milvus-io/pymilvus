"""
Tests for gRPC error diagnostics: debug_error_string and channel_state.

Unit tests use mocks and run quickly.
Integration tests require network access and are marked with @pytest.mark.integration.
"""

from typing import Optional

import grpc
import pytest
from pymilvus.decorators import _CONNECTIVITY_INT_TO_ENUM, _get_rpc_error_info, _try_get_channel


class MockRpcError(grpc.RpcError):
    """Mock gRPC RpcError for unit testing."""

    def __init__(
        self,
        code: grpc.StatusCode = grpc.StatusCode.UNAVAILABLE,
        details: str = "mock error details",
        debug_error_string: Optional[str] = None,
    ):
        self._code = code
        self._details = details
        self._debug_error_string = debug_error_string

    def code(self):
        return self._code

    def details(self):
        return self._details

    def debug_error_string(self):
        return self._debug_error_string


class MockChannel:
    """Mock gRPC channel for unit testing."""

    def __init__(self, state: grpc.ChannelConnectivity = grpc.ChannelConnectivity.READY):
        self._channel = self
        self._state = state

    def check_connectivity_state(self, try_to_connect: bool):
        return self._state  # Return enum value like real gRPC


class MockGrpcHandler:
    """Mock GrpcHandler with _channel attribute."""

    def __init__(self, channel_state: grpc.ChannelConnectivity = grpc.ChannelConnectivity.READY):
        self._channel = MockChannel(channel_state)


class TestGetRpcErrorInfo:
    """Unit tests for _get_rpc_error_info function."""

    def test_basic_error_info(self):
        """Test that basic error info includes code and details."""
        error = MockRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            details="connection refused",
        )
        result = _get_rpc_error_info(error)

        assert "StatusCode.UNAVAILABLE" in result
        assert "connection refused" in result

    def test_includes_debug_error_string(self):
        """Test that debug_error_string is included when available."""
        error = MockRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            details="connection refused",
            debug_error_string='{"grpc_status":14,"grpc_message":"Connection refused"}',
        )
        result = _get_rpc_error_info(error)

        assert "debug=" in result
        assert "Connection refused" in result

    def test_handles_missing_debug_error_string(self):
        """Test graceful handling when debug_error_string is None."""
        error = MockRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            details="some error",
            debug_error_string=None,
        )
        result = _get_rpc_error_info(error)

        assert "StatusCode.UNAVAILABLE" in result
        assert "debug=" not in result

    def test_includes_channel_state_when_provided(self):
        """Test that channel_state is included when channel is provided."""
        error = MockRpcError(code=grpc.StatusCode.DEADLINE_EXCEEDED)
        channel = MockChannel(state=grpc.ChannelConnectivity.CONNECTING)

        result = _get_rpc_error_info(error, channel)

        assert "channel_state=CONNECTING" in result

    def test_channel_state_ready(self):
        """Test channel_state=READY for connected channel."""
        error = MockRpcError(code=grpc.StatusCode.DEADLINE_EXCEEDED)
        channel = MockChannel(state=grpc.ChannelConnectivity.READY)

        result = _get_rpc_error_info(error, channel)

        assert "channel_state=READY" in result

    def test_channel_state_transient_failure(self):
        """Test channel_state=TRANSIENT_FAILURE for failed connection."""
        error = MockRpcError(code=grpc.StatusCode.UNAVAILABLE)
        channel = MockChannel(state=grpc.ChannelConnectivity.TRANSIENT_FAILURE)

        result = _get_rpc_error_info(error, channel)

        assert "channel_state=TRANSIENT_FAILURE" in result

    def test_no_channel_state_when_channel_is_none(self):
        """Test that channel_state is not included when channel is None."""
        error = MockRpcError(code=grpc.StatusCode.UNAVAILABLE)

        result = _get_rpc_error_info(error, channel=None)

        assert "channel_state=" not in result

    def test_full_error_info_format(self):
        """Test complete error info with all components."""
        error = MockRpcError(
            code=grpc.StatusCode.DEADLINE_EXCEEDED,
            details="Deadline Exceeded",
            debug_error_string='{"grpc_status":4}',
        )
        channel = MockChannel(state=grpc.ChannelConnectivity.CONNECTING)

        result = _get_rpc_error_info(error, channel)

        assert "StatusCode.DEADLINE_EXCEEDED" in result
        assert "Deadline Exceeded" in result
        assert "channel_state=CONNECTING" in result
        assert "debug=" in result


class TestTryGetChannel:
    """Unit tests for _try_get_channel function."""

    def test_extracts_channel_from_grpc_handler(self):
        """Test that channel is extracted from GrpcHandler-like object."""
        handler = MockGrpcHandler(channel_state=grpc.ChannelConnectivity.READY)

        channel = _try_get_channel((handler,))

        assert channel is not None
        assert channel._state == grpc.ChannelConnectivity.READY

    def test_returns_none_for_empty_args(self):
        """Test that None is returned when args is empty."""
        channel = _try_get_channel(())

        assert channel is None

    def test_returns_none_when_no_channel_attribute(self):
        """Test that None is returned when object has no _channel."""

        class NoChannelObject:
            pass

        channel = _try_get_channel((NoChannelObject(),))

        assert channel is None


class TestConnectivityStateMapping:
    """Unit tests for connectivity state mapping."""

    def test_all_states_have_mapping(self):
        """Test that all connectivity states have int-to-enum mapping."""
        assert _CONNECTIVITY_INT_TO_ENUM[0] == grpc.ChannelConnectivity.IDLE
        assert _CONNECTIVITY_INT_TO_ENUM[1] == grpc.ChannelConnectivity.CONNECTING
        assert _CONNECTIVITY_INT_TO_ENUM[2] == grpc.ChannelConnectivity.READY
        assert _CONNECTIVITY_INT_TO_ENUM[3] == grpc.ChannelConnectivity.TRANSIENT_FAILURE
        assert _CONNECTIVITY_INT_TO_ENUM[4] == grpc.ChannelConnectivity.SHUTDOWN
        assert len(_CONNECTIVITY_INT_TO_ENUM) == 5


@pytest.mark.integration
class TestGrpcErrorDiagnosticsIntegration:
    """
    Integration tests that make real gRPC connections.
    These tests require network access and may be slow.

    Run with: pytest tests/test_grpc_error_diagnostics.py -m integration
    """

    def _test_connection(self, uri: str, timeout: float = 3.0):
        """Helper to test a connection and return error info."""
        channel = None
        try:
            addr = uri.replace("http://", "").replace("https://", "")
            if uri.startswith("https://"):
                channel = grpc.secure_channel(
                    addr,
                    grpc.ssl_channel_credentials(),
                    options=[("grpc.enable_retries", 0)],
                )
            else:
                channel = grpc.insecure_channel(addr, options=[("grpc.enable_retries", 0)])

            method = channel.unary_unary(
                "/test/Method",
                request_serializer=lambda x: b"",
                response_deserializer=lambda x: x,
            )
            method(b"", timeout=timeout)
        except grpc.RpcError as e:
            error_info = _get_rpc_error_info(e, channel)
            return e, error_info
        else:
            return None, None  # No error
        finally:
            if channel:
                channel.close()

    def test_dns_failure_includes_debug_and_channel_state(self):
        """Test DNS resolution failure includes diagnostic info."""
        error, error_info = self._test_connection("http://baddomain.invalid:19530")

        assert error is not None
        assert error.code() == grpc.StatusCode.UNAVAILABLE
        assert "debug=" in error_info
        assert "channel_state=" in error_info
        assert "DNS" in error_info or "name" in error_info.lower()

    def test_connection_refused_includes_debug_and_channel_state(self):
        """Test connection refused includes diagnostic info."""
        error, error_info = self._test_connection("http://127.0.0.1:19999")

        assert error is not None
        assert error.code() == grpc.StatusCode.UNAVAILABLE
        assert "debug=" in error_info
        assert "channel_state=TRANSIENT_FAILURE" in error_info
        assert "refused" in error_info.lower() or "connect" in error_info.lower()

    def test_deadline_exceeded_shows_connecting_state(self):
        """Test that deadline exceeded with unresponsive server shows CONNECTING state."""
        # 8.8.8.8:80 is likely to timeout (firewall drops packets)
        error, error_info = self._test_connection("http://8.8.8.8:80", timeout=3.0)

        assert error is not None
        assert error.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        assert "debug=" in error_info
        assert "channel_state=CONNECTING" in error_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
