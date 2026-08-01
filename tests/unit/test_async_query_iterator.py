"""Unit tests for AsyncQueryIterator (no server required)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import AsyncMilvusClient
from pymilvus.client.connection_manager import AsyncConnectionManager
from pymilvus.client.constants import (
    COLLECTION_ID,
    ITERATOR_SESSION_TS_FIELD,
)
from pymilvus.client.iterator import AsyncQueryIterator
from pymilvus.client.types import DataType
from pymilvus.exceptions import DataTypeNotMatchException

_SCHEMA_DICT = {
    "fields": [
        {"name": "pk", "type": DataType.INT64, "is_primary": True},
        {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
    ],
}

_VARCHAR_SCHEMA_DICT = {
    "fields": [
        {"name": "pk", "type": DataType.VARCHAR, "is_primary": True},
        {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
    ],
}


class _QueryResult(list):
    """Stand-in for HybridExtraList: a list that also carries `.extra`."""

    def __init__(self, rows, session_ts=100):
        super().__init__(rows)
        self.extra = {ITERATOR_SESSION_TS_FIELD: session_ts}


class _FakeAsyncHandler:
    """Async handler double. `pages` is consumed one entry per query() call."""

    def __init__(self, pages=None, session_ts=100):
        self.pages = list(pages or [])
        self.session_ts = session_ts
        self.describe_calls = []
        self.query_calls = []

    async def describe_collection(self, collection_name, **kwargs):
        self.describe_calls.append((collection_name, kwargs))
        return {COLLECTION_ID: 999}

    async def query(self, collection_name, **kwargs):
        self.query_calls.append((collection_name, kwargs))
        rows = self.pages.pop(0) if self.pages else []
        return _QueryResult(rows, session_ts=self.session_ts)


async def _make_iterator(handler, **overrides):
    kwargs = {
        "handler": handler,
        "context": None,
        "collection_name": "test",
        "batch_size": 2,
        "expr": "pk > 0",
        "output_fields": ["pk"],
        "schema": _SCHEMA_DICT,
    }
    kwargs.update(overrides)
    return await AsyncQueryIterator.create(**kwargs)


@pytest.mark.asyncio
async def test_next_returns_batches_then_empty():
    # first query() call is the mvccTs probe, the rest are data pages
    handler = _FakeAsyncHandler(
        pages=[
            [],  # mvccTs probe
            [{"pk": 1}, {"pk": 2}],
            [{"pk": 3}],
            [],
        ]
    )
    it = await _make_iterator(handler)

    assert await it.next() == [{"pk": 1}, {"pk": 2}]
    assert await it.next() == [{"pk": 3}]
    assert await it.next() == []
    await it.close()


@pytest.mark.asyncio
async def test_describe_collection_called_once():
    handler = _FakeAsyncHandler(pages=[[], [{"pk": 1}], []])
    it = await _make_iterator(handler)
    await it.next()
    await it.next()
    await it.close()

    assert len(handler.describe_calls) == 1


@pytest.mark.asyncio
async def test_cursor_expr_advances_with_int_pk():
    handler = _FakeAsyncHandler(pages=[[], [{"pk": 7}, {"pk": 9}], []])
    it = await _make_iterator(handler)
    await it.next()
    await it.next()
    await it.close()

    # query_calls[0] is the mvccTs probe, [1] the first page, [2] the second page
    assert handler.query_calls[1][1]["expr"] == "pk > 0"
    second_expr = handler.query_calls[2][1]["expr"]
    assert "pk > 9" in second_expr
    assert "(pk > 0)" in second_expr
    assert second_expr.index("pk > 9") < second_expr.index("(pk > 0)")


@pytest.mark.asyncio
async def test_cursor_expr_quotes_varchar_pk():
    handler = _FakeAsyncHandler(pages=[[], [{"pk": "abc"}], []])
    it = await _make_iterator(handler, expr='pk != ""', schema=_VARCHAR_SCHEMA_DICT)
    await it.next()
    await it.next()
    await it.close()

    assert 'pk > "abc"' in handler.query_calls[2][1]["expr"]


@pytest.fixture(autouse=True)
def _reset_async_connection_manager():
    AsyncConnectionManager._reset_instance()
    yield
    AsyncConnectionManager._reset_instance()


async def _client_with_handler(handler):
    mock_handler = MagicMock()
    mock_handler.ensure_channel_ready = AsyncMock()
    mock_handler.describe_collection = handler.describe_collection
    mock_handler.query = handler.query
    mock_handler._get_schema = AsyncMock(return_value=(_SCHEMA_DICT, 1))
    with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler):
        client = AsyncMilvusClient()
        await client._connect()
        return client


@pytest.mark.asyncio
async def test_client_query_iterator_returns_async_iterator():
    handler = _FakeAsyncHandler(pages=[[], [{"pk": 1}, {"pk": 2}], []])
    client = await _client_with_handler(handler)

    it = await client.query_iterator(
        collection_name="test", batch_size=2, filter="pk > 0", output_fields=["pk"]
    )
    assert isinstance(it, AsyncQueryIterator)
    assert await it.next() == [{"pk": 1}, {"pk": 2}]
    assert await it.next() == []
    await it.close()


@pytest.mark.asyncio
async def test_client_query_iterator_rejects_non_string_filter():
    handler = _FakeAsyncHandler()
    client = await _client_with_handler(handler)

    with pytest.raises(DataTypeNotMatchException):
        await client.query_iterator(collection_name="test", filter=123)


def test_async_client_session_exposes_query_iterator():
    from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClientSession  # noqa: PLC0415

    assert hasattr(AsyncMilvusClientSession, "query_iterator")
