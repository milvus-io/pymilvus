from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pymilvus.bulk_writer.constants import ConnectType
from pymilvus.bulk_writer.endpoint_resolver import EndpointResolver


def _make_conn(status: int = 200, request_error: Optional[Exception] = None) -> MagicMock:
    conn = MagicMock()
    if request_error is not None:
        conn.request.side_effect = request_error
    else:
        resp = MagicMock()
        resp.status = status
        conn.getresponse.return_value = resp
    return conn


class TestEndpointResolver:
    def test_non_ali_cloud_returns_default_endpoint(self):
        with patch("http.client.HTTPSConnection") as mock_cls:
            got = EndpointResolver.resolve_endpoint(
                "s3.us-west-2.amazonaws.com", "aws", "us-west-2", ConnectType.AUTO
            )
            assert got == "s3.us-west-2.amazonaws.com"
            mock_cls.assert_not_called()

    def test_forced_internal_reachable_closes_connection(self):
        conn = _make_conn(status=200)
        with patch("http.client.HTTPSConnection", return_value=conn) as mock_cls:
            got = EndpointResolver.resolve_endpoint(
                "default.example.com", "ali", "cn-hangzhou", ConnectType.INTERNAL
            )
        assert got == "oss-cn-hangzhou-internal.aliyuncs.com"
        mock_cls.assert_called_once_with("oss-cn-hangzhou-internal.aliyuncs.com", timeout=5)
        conn.close.assert_called_once()

    @pytest.mark.parametrize("status", [400, 403, 404, 500, 503])
    def test_forced_internal_raises_on_unhealthy_http_status(self, status: int):
        conn = _make_conn(status=status)
        with patch("http.client.HTTPSConnection", return_value=conn):
            with pytest.raises(RuntimeError, match=str(status)):
                EndpointResolver.resolve_endpoint(
                    "default.example.com", "ali", "cn-hangzhou", ConnectType.INTERNAL
                )
        conn.close.assert_called_once()

    def test_forced_public_raises_on_unhealthy_http_status(self):
        conn = _make_conn(status=502)
        with patch("http.client.HTTPSConnection", return_value=conn):
            with pytest.raises(RuntimeError, match="502"):
                EndpointResolver.resolve_endpoint(
                    "default.example.com", "ali", "cn-hangzhou", ConnectType.PUBLIC
                )
        conn.close.assert_called_once()

    def test_auto_mode_uses_internal_when_reachable(self):
        conn = _make_conn(status=301)
        with patch("http.client.HTTPSConnection", return_value=conn) as mock_cls:
            got = EndpointResolver.resolve_endpoint(
                "default.example.com", "ali", "cn-shanghai", ConnectType.AUTO
            )
        assert got == "oss-cn-shanghai-internal.aliyuncs.com"
        mock_cls.assert_called_once_with("oss-cn-shanghai-internal.aliyuncs.com", timeout=5)
        conn.close.assert_called_once()

    def test_auto_mode_falls_back_to_public_when_internal_returns_error_status(self):
        internal_conn = _make_conn(status=503)
        public_conn = _make_conn(status=200)
        with patch(
            "http.client.HTTPSConnection", side_effect=[internal_conn, public_conn]
        ) as mock_cls:
            got = EndpointResolver.resolve_endpoint(
                "default.example.com", "ali", "cn-hangzhou", ConnectType.AUTO
            )
        assert got == "oss-cn-hangzhou.aliyuncs.com"
        assert mock_cls.call_count == 2
        internal_conn.close.assert_called_once()
        public_conn.close.assert_called_once()

    def test_auto_mode_falls_back_on_network_error_and_closes_both_connections(self):
        internal_conn = _make_conn(request_error=OSError("network unreachable"))
        public_conn = _make_conn(status=200)
        with patch("http.client.HTTPSConnection", side_effect=[internal_conn, public_conn]):
            got = EndpointResolver.resolve_endpoint(
                "default.example.com", "ali", "cn-hangzhou", ConnectType.AUTO
            )
        assert got == "oss-cn-hangzhou.aliyuncs.com"
        internal_conn.close.assert_called_once()
        public_conn.close.assert_called_once()

    def test_auto_mode_raises_when_both_endpoints_return_error_status(self):
        internal_conn = _make_conn(status=503)
        public_conn = _make_conn(status=500)
        with patch("http.client.HTTPSConnection", side_effect=[internal_conn, public_conn]):
            with pytest.raises(RuntimeError, match="500"):
                EndpointResolver.resolve_endpoint(
                    "default.example.com", "ali", "cn-hangzhou", ConnectType.AUTO
                )
        internal_conn.close.assert_called_once()
        public_conn.close.assert_called_once()

    def test_check_endpoint_reachable_returns_false_without_raising(self):
        conn = _make_conn(status=500)
        with patch("http.client.HTTPSConnection", return_value=conn):
            assert EndpointResolver._check_endpoint_reachable("oss.example.com", False) is False
        conn.close.assert_called_once()

    def test_check_endpoint_reachable_closes_connection_on_request_exception(self):
        conn = _make_conn(request_error=TimeoutError("timed out"))
        with patch("http.client.HTTPSConnection", return_value=conn):
            with pytest.raises(RuntimeError, match="timed out"):
                EndpointResolver._check_endpoint_reachable("oss.example.com", True)
        conn.close.assert_called_once()
