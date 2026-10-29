"""Sync and async grep share query semantics and preserve raw rg arguments."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pymilvus import AsyncMilvusClient, FunctionChainStage, MilvusClient
from pymilvus.exceptions import MilvusException, ParamError


@pytest.fixture(params=[MilvusClient, AsyncMilvusClient])
def grep_client(request):
    client = object.__new__(request.param)
    is_async = request.param is AsyncMilvusClient
    client.query = AsyncMock() if is_async else MagicMock()

    async def call(**kwargs):
        result = client.grep(**kwargs)
        return await result if is_async else result

    return client, call


@pytest.mark.asyncio
async def test_grep_forwards_query_options_and_preserves_result(grep_client):
    client, call = grep_client
    expected = [{"id": 1, "hits": [{"content": "timeout", "starts": [0], "ends": [7]}]}]
    client.query.return_value = expected
    arguments = r'-i -C 2 -e "user_id=[0-9]{3,6}\b.*timeout"'
    output_fields = ["id", "path"]
    result = await call(
        collection_name="code",
        text_field="docs",
        args=arguments,
        filter='repo == {repo} and path like "src/%"',
        output_fields=output_fields,
        limit=50,
        offset=10,
        timeout=2.5,
        partition_names=["p"],
        match_field="hits",
        filter_params={"repo": "demo"},
        consistency_level="Strong",
    )
    assert result is expected
    client.query.assert_called_once()
    forwarded = client.query.call_args.kwargs
    chain = forwarded.pop("function_chains")
    assert forwarded == {
        "collection_name": "code",
        "filter": 'repo == {repo} and path like "src/%"',
        "output_fields": output_fields,
        "limit": 50,
        "offset": 10,
        "timeout": 2.5,
        "partition_names": ["p"],
        "filter_params": {"repo": "demo"},
        "consistency_level": "Strong",
    }
    assert forwarded["output_fields"] is output_fields
    proto = chain.to_proto()
    assert proto.name == "grep"
    assert proto.stage == FunctionChainStage.QUERY_SCAN
    assert list(proto.output_columns) == ["hits"]
    assert len(proto.ops) == 1
    op = proto.ops[0]
    assert op.op == "grep"
    assert not op.HasField("expr")
    assert list(op.inputs) == ["docs"]
    assert list(op.outputs) == ["hits"]
    assert op.params["arguments"].string_value == arguments
    if isinstance(client.query, AsyncMock):
        client.query.assert_awaited_once()


@pytest.mark.asyncio
async def test_grep_defaults_and_no_matches(grep_client):
    client, call = grep_client
    client.query.return_value = []
    assert await call(collection_name="code", text_field="docs", args="timeout") == []
    kwargs = client.query.call_args.kwargs
    assert kwargs["filter"] == ""
    assert kwargs["output_fields"] is None
    assert kwargs["limit"] == 100
    assert kwargs["offset"] == 0
    assert kwargs["timeout"] is None
    assert kwargs["partition_names"] is None
    assert kwargs["function_chains"].output_columns == ("matches",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override, message",
    [
        ({"text_field": ""}, "text_field"),
        ({"text_field": "  "}, "text_field"),
        ({"text_field": 1}, "text_field"),
        ({"args": None}, "args"),
        ({"args": ["-i", "timeout"]}, "args"),
        ({"match_field": ""}, "match_field"),
        ({"match_field": None}, "match_field"),
        ({"function_chains": None}, "function_chains"),
        ({"function_chains": []}, "function_chains"),
    ],
)
async def test_invalid_grep_fails_before_query(grep_client, override, message):
    client, call = grep_client
    kwargs = {"collection_name": "code", "text_field": "docs", "args": "timeout"}
    kwargs.update(override)
    with pytest.raises(ParamError, match=message):
        await call(**kwargs)
    client.query.assert_not_called()


@pytest.mark.asyncio
async def test_grep_preserves_query_errors(grep_client):
    client, call = grep_client
    error = MilvusException(message="unsupported grep flag")
    client.query.side_effect = error
    with pytest.raises(MilvusException) as exc:
        await call(collection_name="code", text_field="docs", args="-C 2 timeout")
    assert exc.value is error
