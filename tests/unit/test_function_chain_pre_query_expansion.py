"""PRE expansion must never be silently accepted by a row-preserving server."""

from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest
from pymilvus import AnnSearchRequest, FunctionChain, FunctionChainStage
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.function_chain_capability import (
    POST_TABLE_VERSION_KEY,
    PRE_QUERY_EXPANSION_VERSION_KEY,
    TRANSFORM_VERSION_KEY,
    requires_pre_query_expansion_capability,
)
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.exceptions import ParamError
from pymilvus.function_chain import agg, col, fn
from pymilvus.grpc_gen import milvus_pb2, schema_pb2


def expand_queries(operation="flat_map"):
    chain = FunctionChain(FunctionChainStage.PRE_PROCESS)
    expr = fn.py_udf(col("query"), resource_name="expand_queries")
    if operation == "flat_map":
        return chain.flat_map("query", expr, position="variant")
    return chain.map("variants", expr).explode("variants", "query", position="variant")


@pytest.mark.parametrize("operation", ["flat_map", "explode"])
def test_pre_expansion_roundtrip_preserves_operator_and_query_binding(operation):
    chain = expand_queries(operation).select(["query", "variant"])
    assert FunctionChain.from_proto(chain.to_proto()).to_proto() == chain.to_proto()
    assert FunctionChain.from_dict(chain.to_dict()).to_proto() == chain.to_proto()
    op = next(op for op in chain.to_proto().ops if op.op == operation)
    assert list(op.outputs) == ["query", "variant"]
    assert op.params["outer"].bool_value is False


@pytest.mark.parametrize("operation", ["flat_map", "explode"])
@pytest.mark.parametrize("stage", [FunctionChainStage.L0_RERANK, FunctionChainStage.L1_RERANK])
def test_pre_expansion_does_not_relax_rerank_cardinality(stage, operation):
    chain = FunctionChain(stage)
    with pytest.raises(ParamError, match="row expansion requires"):
        if operation == "flat_map":
            chain.flat_map("query", fn.py_udf(col("query"), resource_name="expand_queries"))
        else:
            chain.explode("queries")


def test_expansion_capability_follows_nested_ann_plan_and_stage():
    pre = expand_queries().to_proto()
    branch = milvus_pb2.SearchRequest(function_chains=[pre])
    assert requires_pre_query_expansion_capability(branch)
    assert requires_pre_query_expansion_capability(
        milvus_pb2.HybridSearchRequest(requests=[milvus_pb2.SearchRequest(), branch])
    )
    for stage in (schema_pb2.FunctionChainStageIngestion, schema_pb2.FunctionChainStagePostProcess):
        pre.stage = stage
        assert not requires_pre_query_expansion_capability(
            milvus_pb2.SearchRequest(function_chains=[pre])
        )
    identity = (
        FunctionChain(FunctionChainStage.PRE_PROCESS)
        .map("query", fn.py_udf(col("query"), resource_name="translate"))
        .select(["query"])
    )
    assert not requires_pre_query_expansion_capability(
        milvus_pb2.SearchRequest(function_chains=[identity.to_proto()])
    )


def table_response():
    table = pa.table({"$query_index": [0, 0, 1], "$subquery_index": [0, 1, 2], "$id": [10, 11, 12]})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    response = milvus_pb2.SearchResults()
    response.post_process_result.CopyFrom(
        schema_pb2.FunctionChainTableResult(
            arrow_ipc=sink.getvalue().to_pybytes(),
            row_count=3,
        )
    )
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "future"])
@pytest.mark.parametrize("endpoint", ["search", "hybrid"])
@pytest.mark.parametrize("operation", ["flat_map", "explode"])
async def test_expanded_search_checks_new_capability_before_rpc_and_preserves_mapping(
    mode,
    endpoint,
    operation,
    monkeypatch,
):
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel) if mode == "async" else GrpcHandler(channel=channel)
    version = milvus_pb2.GetVersionResponse()
    version.status.extra_info.update({TRANSFORM_VERSION_KEY: "1", POST_TABLE_VERSION_KEY: "1"})
    stub = MagicMock()
    response = table_response()
    if mode == "async":
        handler._async_stub = stub
        stub.GetVersion = AsyncMock(return_value=version)
        rpc = AsyncMock(return_value=response)
        module = "pymilvus.client.async_grpc_handler"
    else:
        handler._stub = stub
        stub.GetVersion.return_value = version
        rpc = MagicMock(return_value=response)
        rpc.future.return_value.result.return_value = response
        module = "pymilvus.client.grpc_handler"
    setattr(stub, "Search" if endpoint == "search" else "HybridSearch", rpc)
    monkeypatch.setattr(f"{module}.ts_utils.construct_guarantee_ts", lambda *args, **kwargs: True)
    pre = expand_queries(operation)
    post = FunctionChain(
        FunctionChainStage.POST_PROCESS,
        result_mode="table",
        input="ann_inputs" if endpoint == "hybrid" else "fused",
    ).select(["$query_index", "$subquery_index", "$id"])
    kwargs = {"_async": True} if mode == "future" else {}

    async def call():
        if endpoint == "search":
            result = handler.search(
                "c",
                "vector",
                {},
                3,
                data=[[1, 0], [0, 1]],
                function_chains=[pre, post],
                **kwargs,
            )
        else:
            branch = AnnSearchRequest(
                [[1, 0], [0, 1]],
                "vector",
                {},
                3,
                function_chains=pre,
            )
            result = handler.hybrid_search("c", [branch], None, 3, function_chains=post, **kwargs)
        if mode == "async":
            return await result
        if mode == "future":
            return result.result()
        return result

    for value in (None, "0", "invalid"):
        if value is None:
            version.status.extra_info.pop(PRE_QUERY_EXPANSION_VERSION_KEY, None)
        else:
            version.status.extra_info[PRE_QUERY_EXPANSION_VERSION_KEY] = value
        with pytest.raises(ParamError, match="PRE query expansion support"):
            await call()
        rpc.assert_not_called()
        rpc.future.assert_not_called()
    version.status.extra_info[PRE_QUERY_EXPANSION_VERSION_KEY] = "1"
    result = await call()
    assert result.rows == [
        {"$query_index": 0, "$subquery_index": 0, "$id": 10},
        {"$query_index": 0, "$subquery_index": 1, "$id": 11},
        {"$query_index": 1, "$subquery_index": 2, "$id": 12},
    ]
    request = rpc.call_args.args[0] if mode != "future" else rpc.future.call_args.args[0]
    assert requires_pre_query_expansion_capability(request)
    assert version.status.extra_info[PRE_QUERY_EXPANSION_VERSION_KEY] == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "future"])
async def test_identity_preprocessing_keeps_existing_capability_contract(mode, monkeypatch):
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel) if mode == "async" else GrpcHandler(channel=channel)
    version = milvus_pb2.GetVersionResponse()
    version.status.extra_info[TRANSFORM_VERSION_KEY] = "1"
    response = milvus_pb2.SearchResults(
        results=schema_pb2.SearchResultData(
            num_queries=1,
            topks=[1],
            scores=[0.9],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[10])),
        )
    )
    stub = MagicMock()
    if mode == "async":
        handler._async_stub = stub
        stub.GetVersion = AsyncMock(return_value=version)
        stub.Search = AsyncMock(return_value=response)
        module = "pymilvus.client.async_grpc_handler"
    else:
        handler._stub = stub
        stub.GetVersion.return_value = version
        stub.Search.return_value = response
        stub.Search.future.return_value.result.return_value = response
        module = "pymilvus.client.grpc_handler"
    monkeypatch.setattr(f"{module}.ts_utils.construct_guarantee_ts", lambda *args, **kwargs: True)
    chain = (
        FunctionChain(FunctionChainStage.PRE_PROCESS)
        .map(
            "query",
            fn.py_udf(col("query"), resource_name="translate"),
        )
        .select(["query"])
    )
    result = handler.search(
        "c",
        "vector",
        {},
        1,
        data=[[1, 0]],
        function_chains=chain,
        **({"_async": True} if mode == "future" else {}),
    )
    if mode == "async":
        result = await result
    elif mode == "future":
        result = result.result()
    assert result[0][0]["id"] == 10


def test_subquery_index_can_be_preserved_but_not_forged():
    def post():
        return FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table")

    projection = post().select(["$subquery_index"])
    assert list(projection.to_proto().ops[0].outputs) == ["$subquery_index"]
    grouped = post().union_all().group_by("$subquery_index").aggregate({"count": agg.count()})
    assert list(grouped.to_proto().ops[-1].outputs) == ["$subquery_index", "count"]
    for builder in (
        lambda: post().map("$subquery_index", fn.py_udf(col("text"), resource_name="forge")),
        lambda: post().select(["text"], ["$subquery_index"]),
        lambda: post().flat_map("$subquery_index", fn.py_udf(col("text"), resource_name="forge")),
    ):
        with pytest.raises(ParamError, match="read-only"):
            builder()
