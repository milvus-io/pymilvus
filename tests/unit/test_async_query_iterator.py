"""Unit tests for AsyncQueryIterator (no server required)."""

import pytest
from pymilvus.client.constants import (
    COLLECTION_ID,
    ITERATOR_SESSION_TS_FIELD,
)
from pymilvus.client.iterator import AsyncQueryIterator
from pymilvus.client.types import DataType

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
