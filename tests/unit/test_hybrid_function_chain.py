"""Hybrid chains must stay attached to their individual recall sources."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pymilvus import (
    AnnSearchRequest,
    Function,
    FunctionChain,
    FunctionChainStage,
    FunctionType,
    RRFRanker,
    WeightedRanker,
)
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.exceptions import ParamError
from pymilvus.function_chain import col, fn
from pymilvus.grpc_gen import milvus_pb2


def _l0_chain():
    return FunctionChain(FunctionChainStage.L0_RERANK).map(
        "$score", fn.num_combine(col("$score"), col("quality"), mode="sum")
    )


def _l1_chain():
    return FunctionChain(FunctionChainStage.L1_RERANK).sort("$score").limit(5)


def _ann_request(chains=None):
    return AnnSearchRequest(
        data=[[0.1, 0.2]],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=10,
        function_chains=chains,
    )


def _function_ranker():
    return Function(
        name="rerank",
        function_type=FunctionType.RERANK,
        input_field_names=["text"],
        params={"provider": "mock"},
    )


@pytest_asyncio.fixture(params=["sync", "async", "future"])
async def hybrid_call(request, monkeypatch):
    """Exercise actual request preparation and capture the protobuf at the RPC."""
    mode = request.param
    response = milvus_pb2.SearchResults()
    if mode == "async":
        channel = MagicMock()
        channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=channel)
        handler._async_stub = MagicMock()
        rpc = AsyncMock(return_value=response)
        handler._async_stub.HybridSearch = rpc
        module = "pymilvus.client.async_grpc_handler"
    else:
        handler = GrpcHandler(channel=MagicMock())
        handler._stub = MagicMock()
        rpc = handler._stub.HybridSearch
        rpc.return_value = response
        rpc.future.return_value.result.return_value = response
        module = "pymilvus.client.grpc_handler"
    monkeypatch.setattr(f"{module}.ts_utils.construct_guarantee_ts", lambda *a, **kw: True)

    async def call(reqs, ranker=None, **kwargs):
        if mode == "async":
            await handler.hybrid_search("c", reqs, ranker, 10, **kwargs)
        elif mode == "future":
            handler.hybrid_search("c", reqs, ranker, 10, _async=True, **kwargs).result()
        else:
            handler.hybrid_search("c", reqs, ranker, 10, **kwargs)
        wire_request = (rpc.future if mode == "future" else rpc).call_args.args[0]
        return milvus_pb2.HybridSearchRequest.FromString(wire_request.SerializeToString())

    yield call, rpc


@pytest.mark.asyncio
@pytest.mark.parametrize("rerank_source", ["default", "rrf", "weighted", "l2"])
async def test_per_request_chains_are_isolated(hybrid_call, rerank_source):
    call, _ = hybrid_call
    l0, l1 = _l0_chain(), _l1_chain()
    top_chain = FunctionChain(FunctionChainStage.L2_RERANK).merge("rrf").limit(3)
    rankers = {"default": None, "rrf": RRFRanker(), "weighted": WeightedRanker(0.4, 0.3, 0.3)}
    kwargs = {"function_chains": top_chain} if rerank_source == "l2" else {}
    reqs = [_ann_request([l1, l0]), _ann_request(l1), _ann_request()]

    request = await call(reqs, rankers.get(rerank_source), **kwargs)

    assert list(request.requests[0].function_chains) == [l1.to_proto(), l0.to_proto()]
    assert list(request.requests[1].function_chains) == [l1.to_proto()]
    assert not request.requests[2].function_chains
    assert list(request.function_chains) == ([top_chain.to_proto()] if kwargs else [])
    assert reqs[0].function_chains == [l1, l0]
    assert not request.HasField("function_score")


@pytest.mark.asyncio
async def test_nested_chains_reject_typed_ranker_before_rpc(hybrid_call):
    call, rpc = hybrid_call
    with pytest.raises(ParamError, match=r"function_score.*sub-search\[1\]"):
        await call([_ann_request(), _ann_request(_l0_chain())], _function_ranker())
    rpc.assert_not_called()
    rpc.future.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("chains", [None, []])
async def test_typed_ranker_still_accepts_requests_without_chains(hybrid_call, chains):
    call, _ = hybrid_call
    request = await call([_ann_request(chains)], _function_ranker())
    assert request.HasField("function_score")
    assert not request.requests[0].function_chains


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chains, message",
    [
        (FunctionChain(FunctionChainStage.UNSPECIFIED).limit(1), "not supported"),
        ([_l0_chain(), _l0_chain()], "appears more than once"),
        ([_l1_chain(), _l1_chain()], "appears more than once"),
        (FunctionChain(FunctionChainStage.L0_RERANK), "at least one op"),
        (FunctionChain(FunctionChainStage.L1_RERANK), "at least one op"),
        ([object()], "must be a FunctionChain"),
        ("invalid", "must be a FunctionChain"),
    ],
)
async def test_invalid_nested_chains_fail_before_rpc(hybrid_call, chains, message):
    call, rpc = hybrid_call
    with pytest.raises(ParamError, match=message):
        await call([_ann_request(chains)], RRFRanker())
    rpc.assert_not_called()
    rpc.future.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        FunctionChainStage.PRE_PROCESS,
        FunctionChainStage.POST_PROCESS,
        FunctionChainStage.L2_RERANK,
        FunctionChainStage.INGESTION,
    ],
)
async def test_nested_stage_support_is_validated_by_server(hybrid_call, stage):
    call, _ = hybrid_call
    chain = FunctionChain(stage).limit(1)
    # Verify transport only; acceptance of this stage/op belongs to the server.
    request = await call([_ann_request([chain, _l0_chain()])], RRFRanker())
    assert list(request.requests[0].function_chains) == [
        chain.to_proto(),
        _l0_chain().to_proto(),
    ]


@pytest.mark.asyncio
async def test_top_level_chain_is_not_inherited(hybrid_call):
    call, _ = hybrid_call
    top_chain = FunctionChain(FunctionChainStage.L2_RERANK).merge("rrf")
    request = await call([_ann_request(), _ann_request([])], function_chains=top_chain)
    assert list(request.function_chains) == [top_chain.to_proto()]
    assert all(not sub.function_chains for sub in request.requests)
