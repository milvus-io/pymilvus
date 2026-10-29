"""QueryScan request transport and computed JSON result decoding."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pymilvus import AsyncMilvusClient, FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError
from pymilvus.function_chain import col
from pymilvus.grpc_gen import milvus_pb2, schema_pb2


def _chain(export=True, arguments="-i -e 'timeout'"):
    return FunctionChain(
        FunctionChainStage.QUERY_SCAN,
        name="text_hits" if export else "",
        output_columns=["matches"] if export else None,
    ).grep(col("content"), arguments=arguments)


def test_grep_wire_contract_and_projection_ownership():
    columns = ["matches"]
    chain = FunctionChain(FunctionChainStage.QUERY_SCAN, "hits", output_columns=columns)
    chain.grep("content", arguments="-i -e '超时'", output="matches")
    columns.append("other")
    proto = chain.to_proto()
    assert proto.stage == schema_pb2.FunctionChainStageQueryScan
    assert list(proto.output_columns) == ["matches"]
    assert chain.output_columns == ("matches",)
    op = proto.ops[0]
    assert op.op == "grep"
    assert not op.HasField("expr")
    assert list(op.inputs) == ["content"]
    assert list(op.outputs) == ["matches"]
    assert op.params["arguments"].string_value == "-i -e '超时'"


@pytest.mark.parametrize("columns", ["matches", [""], [1], ["matches", "matches"]])
def test_invalid_projection(columns):
    with pytest.raises(ParamError, match="output_columns"):
        FunctionChain(FunctionChainStage.QUERY_SCAN, "hits", output_columns=columns)


@pytest.mark.parametrize(
    "stage, name", [(FunctionChainStage.QUERY_SCAN, ""), (FunctionChainStage.L0_RERANK, "hits")]
)
def test_projection_requires_named_query_scan(stage, name):
    with pytest.raises(ParamError, match="named QUERY_SCAN"):
        FunctionChain(stage, name, output_columns=["matches"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": "", "arguments": "x"},
        {"value": "content", "arguments": 1},
        {"value": "content", "arguments": "x", "output": ""},
    ],
)
def test_invalid_grep_arguments(kwargs):
    with pytest.raises(ParamError):
        FunctionChain(FunctionChainStage.QUERY_SCAN).grep(**kwargs)


def test_grep_requires_query_scan():
    with pytest.raises(ParamError, match="QUERY_SCAN"):
        FunctionChain(FunctionChainStage.L0_RERANK).grep("content", arguments="x")


@pytest_asyncio.fixture(params=["sync", "async"])
async def query_call(request, monkeypatch):
    details = [
        [{"content": "timeout timeout", "starts": [0, 8], "ends": [7, 15]}],
        [{"content": "超时", "starts": [0], "ends": [6]}],
    ]
    response = milvus_pb2.QueryResults()
    ids = response.fields_data.add(field_name="id", type=schema_pb2.Int64)
    ids.scalars.long_data.data.extend([10, 20])
    matches = response.fields_data.add(field_name="matches", type=schema_pb2.JSON)
    matches.scalars.json_data.data.extend(json.dumps(d).encode() for d in details)
    if request.param == "async":
        channel = MagicMock()
        channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=channel)
        handler._is_channel_ready = True
        handler._async_stub = MagicMock()
        rpc = AsyncMock(return_value=response)
        handler._async_stub.Query = rpc
        client = AsyncMilvusClient()
        monkeypatch.setattr(client, "_get_connection", AsyncMock(return_value=handler))
        module = "pymilvus.client.async_grpc_handler"
    else:
        handler = GrpcHandler(channel=MagicMock())
        handler._stub = MagicMock()
        rpc = handler._stub.Query
        rpc.return_value = response
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=MagicMock()):
            client = MilvusClient()
        monkeypatch.setattr(client, "_get_connection", lambda: handler)
        module = "pymilvus.client.grpc_handler"
    monkeypatch.setattr(f"{module}.ts_utils.construct_guarantee_ts", lambda *a, **kw: True)

    async def call(use_grep=False, **kwargs):
        method = client.grep if use_grep else client.query
        result = method(
            "docs",
            filter="id > {minimum}",
            filter_params={"minimum": 0},
            output_fields=["id"],
            limit=2,
            offset=1,
            timeout=5,
            **kwargs,
        )
        if request.param == "async":
            result = await result
        wire = rpc.call_args.args[0]
        return milvus_pb2.QueryRequest.FromString(wire.SerializeToString()), list(result)

    yield call, rpc, details


@pytest.mark.asyncio
async def test_grep_reaches_query_rpc(query_call):
    call, _, details = query_call
    wire, rows = await call(use_grep=True, text_field="content", args="-i -e 'timeout'")
    chain = wire.function_chains[0]
    assert chain.name == "grep"
    assert chain.stage == schema_pb2.FunctionChainStageQueryScan
    assert list(chain.output_columns) == ["matches"]
    assert chain.ops[0].params["arguments"].string_value == "-i -e 'timeout'"
    assert rows == [{"id": 10, "matches": details[0]}, {"id": 20, "matches": details[1]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("as_list", [False, True])
async def test_query_chain_reaches_rpc_and_decodes_details(query_call, as_list):
    call, rpc, details = query_call
    chain = _chain()
    wire, rows = await call(function_chains=[chain] if as_list else chain)
    assert list(wire.function_chains) == [chain.to_proto()]
    assert list(wire.output_fields) == ["id"]
    assert wire.expr_template_values["minimum"].int64_val == 0
    params = {p.key: p.value for p in wire.query_params}
    assert params["limit"] == "2"
    assert params["offset"] == "1"
    assert rpc.call_args.kwargs["timeout"] == 5
    assert rows == [{"id": 10, "matches": details[0]}, {"id": 20, "matches": details[1]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("chains", [None, []])
async def test_query_without_chains(query_call, chains):
    call, _, _ = query_call
    wire, _ = await call(function_chains=chains)
    assert not wire.function_chains


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chains",
    [
        "bad",
        [object()],
        [_chain(), _chain()],
        FunctionChain(FunctionChainStage.L0_RERANK).limit(1),
        FunctionChain(FunctionChainStage.QUERY_SCAN),
    ],
)
async def test_invalid_query_chain_rejected_before_rpc(query_call, chains):
    call, rpc, _ = query_call
    with pytest.raises(ParamError):
        await call(function_chains=chains)
    rpc.assert_not_called()


@pytest.mark.parametrize("arguments", ["", "-C 2", "-e '['", "-i -e '超时'"])
def test_filter_only_and_raw_arguments(arguments):
    chain = _chain(export=False, arguments=arguments)
    request = Prepare.query_request("docs", "", ["id"], [], function_chains=chain)
    assert not request.function_chains[0].output_columns
    assert request.function_chains[0].ops[0].params["arguments"].string_value == arguments
