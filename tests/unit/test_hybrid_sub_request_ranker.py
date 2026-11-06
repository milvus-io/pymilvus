"""Per-sub-request function scores must stay attached to their own recall path."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pymilvus import (
    AnnSearchRequest,
    Function,
    FunctionChain,
    FunctionChainStage,
    FunctionScore,
    FunctionType,
    WeightedRanker,
)
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import milvus_pb2


def _l0_chain():
    return FunctionChain(FunctionChainStage.L0_RERANK).limit(5)


def _ann_request(ranker=None, chains=None):
    return AnnSearchRequest(
        data=[[0.1, 0.2]],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=10,
        function_chains=chains,
        ranker=ranker,
    )


def _boost_function(name="boost", weight=2.0):
    return Function(
        name,
        FunctionType.RERANK,
        input_field_names=["text"],
        output_field_names=[],
        params={"reranker": "boost", "weight": weight},
    )


def _sub_function_score(name="boost"):
    return FunctionScore(
        functions=[_boost_function(name)],
        params={"boost_mode": "multiply"},
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

    async def call(reqs, rerank=None, **kwargs):
        if mode == "async":
            await handler.hybrid_search("c", reqs, rerank, 10, **kwargs)
        elif mode == "future":
            handler.hybrid_search("c", reqs, rerank, 10, _async=True, **kwargs).result()
        else:
            handler.hybrid_search("c", reqs, rerank, 10, **kwargs)
        wire_request = (rpc.future if mode == "future" else rpc).call_args.args[0]
        return milvus_pb2.HybridSearchRequest.FromString(wire_request.SerializeToString())

    yield call, rpc


@pytest.mark.asyncio
async def test_sub_request_function_score_lands_on_own_request(hybrid_call):
    """Only the sub-request carrying a ranker gets a function score."""
    call, _ = hybrid_call
    reqs = [_ann_request(_sub_function_score()), _ann_request()]

    request = await call(reqs, WeightedRanker(0.9, 0.1))

    assert request.requests[0].HasField("function_score")
    assert len(request.requests[0].function_score.functions) == 1
    assert request.requests[0].function_score.functions[0].name == "boost"
    assert not request.requests[1].HasField("function_score")
    assert not request.HasField("function_score")
    params = {kv.key: kv.value for kv in request.rank_params}
    assert params["strategy"] == "weighted"


@pytest.mark.asyncio
async def test_sub_request_function_ranker_lands_on_own_request(hybrid_call):
    """A plain Function ranker is also scoped to its own sub-request."""
    call, _ = hybrid_call
    reqs = [_ann_request(), _ann_request(_boost_function("rerank"))]

    request = await call(reqs, WeightedRanker(0.5, 0.5))

    assert not request.requests[0].HasField("function_score")
    assert request.requests[1].HasField("function_score")
    assert request.requests[1].function_score.functions[0].name == "rerank"


@pytest.mark.asyncio
async def test_distinct_function_scores_per_recall_path(hybrid_call):
    """Each recall path can carry a different function score."""
    call, _ = hybrid_call
    reqs = [
        _ann_request(_sub_function_score("boost_title")),
        _ann_request(_sub_function_score("boost_content")),
    ]

    request = await call(reqs, WeightedRanker(0.9, 0.1))

    names = [sub.function_score.functions[0].name for sub in request.requests]
    assert names == ["boost_title", "boost_content"]


@pytest.mark.asyncio
async def test_top_level_function_score_applies_after_fusion(hybrid_call):
    """A FunctionScore as the hybrid ranker is applied to the fused result."""
    call, _ = hybrid_call
    ranker = FunctionScore(
        functions=[_boost_function("boost")],
        params={"boost_mode": "multiply"},
    )

    request = await call([_ann_request(), _ann_request()], ranker)

    assert request.HasField("function_score")
    assert request.function_score.functions[0].name == "boost"
    assert all(not sub.HasField("function_score") for sub in request.requests)


@pytest.mark.asyncio
async def test_sub_request_ranker_conflicts_with_own_function_chains(hybrid_call):
    """A ranker cannot be combined with function_chains on the same request."""
    call, rpc = hybrid_call
    reqs = [_ann_request(), _ann_request(_sub_function_score(), chains=_l0_chain())]

    with pytest.raises(ParamError, match="function_chains and ranker cannot be used together"):
        await call(reqs, WeightedRanker(0.5, 0.5))

    rpc.assert_not_called()
    rpc.future.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_top_level_ranker_rejected_before_rpc(hybrid_call):
    call, rpc = hybrid_call
    with pytest.raises(ParamError, match="must be a Function, a FunctionScore or a Ranker"):
        await call([_ann_request()], "invalid")

    rpc.assert_not_called()
    rpc.future.assert_not_called()


@pytest.mark.asyncio
async def test_kwargs_ranker_still_applies_to_plain_requests(hybrid_call):
    """A ranker forwarded through kwargs (ORM path) still reaches every request."""
    call, _ = hybrid_call
    shared = _boost_function("shared")

    request = await call([_ann_request(), _ann_request()], None, ranker=shared)

    assert all(sub.HasField("function_score") for sub in request.requests)
    assert all(sub.function_score.functions[0].name == "shared" for sub in request.requests)


@pytest.mark.asyncio
async def test_request_ranker_takes_precedence_over_kwargs_ranker(hybrid_call):
    """An explicit per-request ranker wins over the one inherited via kwargs."""
    call, _ = hybrid_call
    shared = _boost_function("shared")
    reqs = [_ann_request(_sub_function_score("own"))]

    request = await call(reqs, None, ranker=shared)

    assert request.requests[0].function_score.functions[0].name == "own"


@pytest.mark.parametrize(
    "ranker",
    [
        Function(
            "decay",
            FunctionType.RERANK,
            input_field_names=["ts"],
            params={"reranker": "decay", "function": "gauss", "origin": 0, "scale": 10},
        ),
        FunctionScore(
            functions=[
                _boost_function(),
                Function(
                    "rrf", FunctionType.RERANK, input_field_names=[], params={"reranker": "rrf"}
                ),
            ]
        ),
        Function(
            "untyped", FunctionType.RERANK, input_field_names=["text"], params={"weight": 2.0}
        ),
    ],
    ids=["decay", "mixed_boost_rrf", "missing_reranker"],
)
def test_sub_request_ranker_rejects_non_boost(ranker):
    """The server only applies boost per sub-request, so others fail before the RPC."""
    with pytest.raises(ParamError, match="only supports boost functions"):
        _ann_request(ranker)


def test_sub_request_ranker_rejects_empty_function_score():
    with pytest.raises(ParamError, match="at least one boost function"):
        _ann_request(FunctionScore(functions=[]))


def test_sub_request_ranker_rejects_other_types():
    with pytest.raises(ParamError, match="must be a Function or a FunctionScore"):
        _ann_request(WeightedRanker(0.5, 0.5))


def test_sub_request_ranker_reranker_param_is_case_insensitive():
    function = Function(
        "b", FunctionType.RERANK, input_field_names=["text"], params={"Reranker": "BOOST"}
    )
    assert _ann_request(function).ranker is function
