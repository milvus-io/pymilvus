import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from grpc._interceptor import _ClientCallDetails as GrpcClientCallDetails
from pymilvus.client.async_interceptor import (
    _GenericAsyncClientInterceptor,
    async_header_adder_interceptor,
)
from pymilvus.client.interceptor import (
    _ClientCallDetails,
    _GenericClientInterceptor,
    header_adder_interceptor,
)


def _make_call_details(method="TestMethod", timeout=None, metadata=None, credentials=None):
    return _ClientCallDetails(method, timeout, metadata, credentials)


_INTERCEPT_METHODS = [
    "intercept_unary_unary",
    "intercept_unary_stream",
    "intercept_stream_unary",
    "intercept_stream_stream",
]

# Methods that take an iterator as request arg (stream request)
_STREAM_REQUEST_METHODS = {"intercept_stream_unary", "intercept_stream_stream"}


class TestGenericClientInterceptor:
    @pytest.mark.parametrize("method", _INTERCEPT_METHODS)
    def test_intercept_no_postprocess(self, method):
        def fn(details, req_iter):
            return details, req_iter, None

        interceptor = _GenericClientInterceptor(fn)
        continuation = MagicMock(return_value="response")
        details = _make_call_details()
        request = iter(["req"]) if method in _STREAM_REQUEST_METHODS else "request"
        result = getattr(interceptor, method)(continuation, details, request)
        # For stream-returning methods, consume the iterator
        if method in ("intercept_unary_stream", "intercept_stream_stream"):
            result = list(result)
        continuation.assert_called_once()

    @pytest.mark.parametrize("method", _INTERCEPT_METHODS)
    def test_intercept_with_postprocess(self, method):
        def fn(details, req_iter):
            return details, req_iter, lambda r: "processed"

        interceptor = _GenericClientInterceptor(fn)
        continuation = MagicMock(return_value="response")
        details = _make_call_details()
        request = iter(["req"]) if method in _STREAM_REQUEST_METHODS else "request"
        result = getattr(interceptor, method)(continuation, details, request)
        assert result == "processed"


class TestHeaderAdderInterceptor:
    def test_adds_headers_no_existing_metadata(self):
        interceptor = header_adder_interceptor(["Authorization"], ["Bearer token"])
        details = _make_call_details(metadata=None)
        continuation = MagicMock(return_value="ok")
        interceptor.intercept_unary_unary(continuation, details, "req")
        call_details = continuation.call_args[0][0]
        assert ("Authorization", "Bearer token") in call_details.metadata

    def test_adds_headers_with_existing_metadata(self):
        interceptor = header_adder_interceptor(["X-New"], ["value"])
        details = _make_call_details(metadata=[("existing", "meta")])
        continuation = MagicMock(return_value="ok")
        interceptor.intercept_unary_unary(continuation, details, "req")
        call_details = continuation.call_args[0][0]
        assert ("existing", "meta") in call_details.metadata
        assert ("X-New", "value") in call_details.metadata

    def test_multiple_headers(self):
        interceptor = header_adder_interceptor(["h1", "h2"], ["v1", "v2"])
        details = _make_call_details()
        continuation = MagicMock(return_value="ok")
        interceptor.intercept_unary_unary(continuation, details, "req")
        call_details = continuation.call_args[0][0]
        assert ("h1", "v1") in call_details.metadata
        assert ("h2", "v2") in call_details.metadata

    def test_preserves_wait_for_ready_and_compression(self):
        interceptor = header_adder_interceptor(["identifier"], ["12345"])
        creds = MagicMock(name="call_credentials")
        details = GrpcClientCallDetails(
            method="/milvus.proto.milvus.MilvusService/Search",
            timeout=15.0,
            metadata=(("existing", "meta"),),
            credentials=creds,
            wait_for_ready=True,
            compression=grpc.Compression.Gzip,
        )
        continuation = MagicMock(return_value="ok")
        interceptor.intercept_unary_unary(continuation, details, "req")
        call_details = continuation.call_args[0][0]
        assert call_details.method == details.method
        assert call_details.timeout == 15.0
        assert call_details.credentials is creds
        assert call_details.wait_for_ready is True
        assert call_details.compression == grpc.Compression.Gzip
        assert ("existing", "meta") in call_details.metadata
        assert ("identifier", "12345") in call_details.metadata

    def test_chained_interceptors_preserve_wait_for_ready_and_compression(self):
        first = header_adder_interceptor(["identifier"], ["12345"])
        second = header_adder_interceptor(["authorization"], ["Bearer tok"])
        details = GrpcClientCallDetails(
            method="/milvus.proto.milvus.MilvusService/Connect",
            timeout=30.0,
            metadata=None,
            credentials=None,
            wait_for_ready=True,
            compression=grpc.Compression.Deflate,
        )
        final_continuation = MagicMock(return_value="ok")

        def run_second(next_details, next_req):
            return second.intercept_unary_unary(final_continuation, next_details, next_req)

        first.intercept_unary_unary(run_second, details, "req")
        call_details = final_continuation.call_args[0][0]
        assert call_details.wait_for_ready is True
        assert call_details.compression == grpc.Compression.Deflate
        assert ("identifier", "12345") in call_details.metadata
        assert ("authorization", "Bearer tok") in call_details.metadata

    def test_four_field_call_details_defaults_optional_fields_to_none(self):
        interceptor = header_adder_interceptor(["h1"], ["v1"])
        details = SimpleNamespace(
            method="TestMethod", timeout=None, metadata=None, credentials=None
        )
        continuation = MagicMock(return_value="ok")
        interceptor.intercept_unary_unary(continuation, details, "req")
        call_details = continuation.call_args[0][0]
        assert call_details.wait_for_ready is None
        assert call_details.compression is None


class TestGenericAsyncClientInterceptor:
    @pytest.mark.parametrize("method", _INTERCEPT_METHODS)
    def test_intercept_delegates(self, method):
        def fn(details, request):
            return details, request

        interceptor = _GenericAsyncClientInterceptor(fn)

        async def run():
            continuation = AsyncMock(return_value="async_response")
            details = MagicMock()
            request = iter(["r"]) if method in _STREAM_REQUEST_METHODS else "req"
            result = await getattr(interceptor, method)(continuation, details, request)
            assert result == "async_response"

        asyncio.run(run())


class TestAsyncHeaderAdderInterceptor:
    def test_adds_headers_to_metadata(self):
        interceptor = async_header_adder_interceptor(["Authorization"], [b"Bearer token"])

        async def run():
            details = MagicMock()
            details.metadata = None
            details.method = "/Test"
            details.timeout = None
            details.credentials = None
            details.wait_for_ready = False
            continuation = AsyncMock(return_value="ok")
            result = await interceptor.intercept_unary_unary(continuation, details, "req")
            assert result == "ok"

        asyncio.run(run())

    def test_preserves_existing_metadata(self):
        interceptor = async_header_adder_interceptor(["x-new"], ["val"])

        async def run():
            details = MagicMock()
            details.metadata = [("existing", "meta")]
            details.method = "/Test"
            details.timeout = None
            details.credentials = None
            details.wait_for_ready = False
            continuation = AsyncMock(return_value="ok")
            await interceptor.intercept_unary_unary(continuation, details, "req")
            call_details = continuation.call_args[0][0]
            assert ("existing", "meta") in call_details.metadata
            assert ("x-new", "val") in call_details.metadata

        asyncio.run(run())
