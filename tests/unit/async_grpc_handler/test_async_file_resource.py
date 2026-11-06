"""Tests for AsyncGrpcHandler file resource operations."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.call_context import CallContext


def _make_async_handler():
    """Create an AsyncGrpcHandler with mocked channel and stub."""
    mock_channel = MagicMock()
    mock_channel.channel_ready = AsyncMock()
    mock_channel.close = AsyncMock()
    mock_channel._unary_unary_interceptors = []

    handler = AsyncGrpcHandler(channel=mock_channel)
    handler._is_channel_ready = True

    mock_stub = AsyncMock()
    handler._async_stub = mock_stub
    return handler, mock_stub


def _ok_response():
    resp = MagicMock()
    resp.code = 0
    resp.error_code = 0
    resp.reason = ""
    resp.status = resp
    resp.resources = []
    return resp


class TestAsyncGrpcHandlerFileResource:
    """AsyncMilvusClient forwards caller kwargs (client_request_id, retry_times, ...) next to context."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,rpc,args",
        [
            ("add_file_resource", "AddFileResource", {"name": "udf", "path": "s3://bucket/udf.py"}),
            ("remove_file_resource", "RemoveFileResource", {"name": "udf"}),
            ("list_file_resources", "ListFileResources", {}),
        ],
    )
    async def test_extra_kwargs_do_not_break_call_metadata(self, method, rpc, args):
        handler, mock_stub = _make_async_handler()
        setattr(mock_stub, rpc, AsyncMock(return_value=_ok_response()))
        context = CallContext(db_name="db1", client_request_id="req-1")

        await getattr(handler, method)(
            **args, timeout=5, context=context, client_request_id="req-1", retry_times=3
        )

        call_kwargs = getattr(mock_stub, rpc).call_args.kwargs
        assert call_kwargs["metadata"] == context.to_grpc_metadata()
        assert call_kwargs["timeout"] == 5
