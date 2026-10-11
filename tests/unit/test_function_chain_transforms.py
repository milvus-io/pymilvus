from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    FunctionChain,
    FunctionChainStage,
    RRFRanker,
)
from pymilvus.client.abstract import CollectionSchema as WireSchema
from pymilvus.client.abstract import MutationResult
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import DataNotMatchException, IngestionFunctionChainError, ParamError
from pymilvus.function_chain import col, fn
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2


def chunking():
    return (
        FunctionChain(FunctionChainStage.PRE_INSERT, name="chunk_and_embed")
        .flat_map(
            "chunks",
            fn.py_udf(col("raw", DataType.VARCHAR), resource_name="chunker"),
            position="part",
        )
        .select(["chunks.text", "chunks.vector"], ["text", "vector"])
    )


def schema(chain=None):
    return CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("text", DataType.VARCHAR, max_length=4096),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
        ],
        function_chains=[chain or chunking()],
    )


def test_chain_schema_survives_create_describe_and_reconstruction():
    original = schema()
    assert [field.name for field in original.fields if field.is_function_output] == [
        "text",
        "vector",
    ]
    create = Prepare.create_collection_request("chunks", original)
    wire = schema_pb2.CollectionSchema.FromString(create.schema)
    assert wire.function_chains[0] == chunking().to_proto()
    # The server reserves its output flag for native schema Functions.
    for field in wire.fields:
        field.is_function_output = False
    described = WireSchema(milvus_pb2.DescribeCollectionResponse(schema=wire)).dict()
    restored = CollectionSchema.construct_from_dict(described)
    assert restored.function_chains[0].to_proto() == chunking().to_proto()
    assert (
        Prepare.get_schema_from_collection_schema("chunks", restored).function_chains
        == wire.function_chains
    )


def test_persisted_binding_is_not_mutated_by_builder_changes():
    chain = chunking()
    collection = schema(chain)
    saved = collection.function_chains[0].to_proto()
    chain.map("another", fn.py_udf(col("text"), resource_name="labels"))
    assert collection.function_chains[0].to_proto() == saved
    copy = collection.function_chains
    copy[0].map("another", fn.py_udf(col("text"), resource_name="labels"))
    assert collection.function_chains[0].to_proto() == saved


def test_original_input_is_required_for_in_place_map():
    chain = (
        FunctionChain(FunctionChainStage.INGESTION, name="translate")
        .map("text", fn.py_udf(col("text"), resource_name="translate"))
        .map("vector", fn.py_udf(col("text"), resource_name="embed"))
    )
    collection = schema(chain)
    assert [field.name for field in collection.fields if field.is_function_output] == ["vector"]
    request = Prepare.row_insert_param(
        "c",
        [{"text": "hello"}],
        "",
        collection.to_dict()["fields"],
        function_chains=collection.function_chains,
    )
    assert [field.field_name for field in request.fields_data] == ["text"]


def test_ephemeral_inputs_are_encoded_without_generated_destination_values():
    collection = schema()
    request = Prepare.row_insert_param(
        "c",
        [{"raw": "one two"}, {"raw": "three"}],
        "",
        collection.to_dict()["fields"],
        function_chains=collection.to_dict()["function_chains"],
    )
    assert request.num_rows == 2
    assert [field.field_name for field in request.fields_data] == ["raw"]
    assert list(request.fields_data[0].scalars.string_data.data) == ["one two", "three"]
    with pytest.raises(DataNotMatchException):
        Prepare.row_insert_param(
            "c",
            [{"raw": "one", "vector": [1, 2]}],
            "",
            collection.to_dict()["fields"],
            function_chains=collection.function_chains,
        )


@pytest.mark.parametrize(
    "stage",
    [FunctionChainStage.PRE_PROCESS, FunctionChainStage.POST_PROCESS, FunctionChainStage.L2_RERANK],
)
@pytest.mark.parametrize("operation", ["flat_map", "explode"])
def test_query_stages_reject_row_expansion(stage, operation):
    chain = FunctionChain(stage)
    with pytest.raises(ParamError, match="row expansion requires INGESTION"):
        if operation == "explode":
            chain.explode("list")
        else:
            chain.flat_map("rows", fn.py_udf(col("text"), resource_name="chunker"))


def test_multi_output_map_and_nested_parameters_roundtrip():
    chain = FunctionChain(FunctionChainStage.INGESTION, "labels").map(
        ["text", "vector"],
        fn.py_udf(
            col("raw", DataType.VARCHAR),
            resource_name="enrich",
            udf_params={"options": {"tags": ["a", "b"], "limit": 8}},
        ),
    )
    assert FunctionChain.from_dict(chain.to_dict()).to_proto() == chain.to_proto()
    assert list(chain.to_proto().ops[0].outputs) == ["text", "vector"]


def test_postprocess_coexists_with_legacy_hybrid_ranker():
    post = FunctionChain(FunctionChainStage.POST_PROCESS).map(
        "highlight", fn.py_udf(col("text"), resource_name="highlight")
    )
    request = Prepare.hybrid_search_request_with_ranker(
        "c", [milvus_pb2.SearchRequest()], RRFRanker(), 10, function_chains=post
    )
    assert request.function_chains[0].stage == FunctionChainStage.POST_PROCESS
    assert any(param.key == "strategy" for param in request.rank_params)


def test_ingestion_is_rejected_as_search_stage():
    with pytest.raises(ParamError, match=r"INGESTION.*not supported"):
        Prepare.function_chains_schema(chunking())


def test_mutation_result_exposes_one_lineage_entry_per_confirmed_id():
    raw = milvus_pb2.MutationResult(
        IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[100, 101, 102])),
        insert_cnt=3,
        succ_index=[0, 1],
        ingestion_result=milvus_pb2.IngestionFunctionChainResult(
            input_rows=2,
            output_rows=3,
            output_sources=milvus_pb2.FunctionChainRowMapping(
                source_offsets=[0, 1, 2, 3], source_row_indexes=[0, 0, 1]
            ),
        ),
    )
    result = MutationResult(raw)
    assert list(result.primary_keys) == [100, 101, 102]
    assert result.insert_count == 3
    assert result.ingestion_result == {
        "input_rows": 2,
        "output_rows": 3,
        "source_offsets": [0, 1, 2, 3],
        "source_row_indexes": [0, 0, 1],
        "outcome_unknown": False,
    }
    assert MutationResult(milvus_pb2.MutationResult()).ingestion_result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("supported", [False, True])
async def test_old_server_is_rejected_before_create_rpc(asynchronous, supported):
    version = milvus_pb2.GetVersionResponse()
    if supported:
        version.status.extra_info["function_chain_transform_version"] = "1"
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel) if asynchronous else GrpcHandler(channel=channel)
    stub = MagicMock()
    if asynchronous:
        handler._async_stub = stub
        stub.GetVersion = AsyncMock(return_value=version)
        stub.CreateCollection = AsyncMock(return_value=common_pb2.Status())
    else:
        handler._stub = stub
        stub.GetVersion.return_value = version
        stub.CreateCollection.future.return_value.result.return_value = common_pb2.Status()
    if supported:
        if asynchronous:
            await handler.create_collection("c", schema())
        else:
            handler.create_collection("c", schema())
    else:
        with pytest.raises(ParamError, match="does not advertise"):
            if asynchronous:
                await handler.create_collection("c", schema())
            else:
                handler.create_collection("c", schema())
        stub.CreateCollection.assert_not_called()
        stub.CreateCollection.future.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["partial_rate_limit", "lost_rpc"])
async def test_ingestion_failure_preserves_partial_result_and_never_replays(asynchronous, failure):
    class LostResponse(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "response unavailable"

    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel) if asynchronous else GrpcHandler(channel=channel)
    collection_schema = schema().to_dict()
    version = milvus_pb2.GetVersionResponse()
    version.status.extra_info["function_chain_transform_version"] = "1"
    response = milvus_pb2.MutationResult(
        status=common_pb2.Status(
            error_code=common_pb2.RateLimit, code=8, reason="partial append", retriable=False
        ),
        IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[100])),
        insert_cnt=1,
        err_index=[0],
        ingestion_result=milvus_pb2.IngestionFunctionChainResult(
            input_rows=1,
            output_rows=1,
            output_sources=milvus_pb2.FunctionChainRowMapping(
                source_offsets=[0, 1], source_row_indexes=[0]
            ),
        ),
    )
    if asynchronous:
        handler._get_schema = AsyncMock(return_value=(collection_schema, 0))
        handler._async_stub = MagicMock()
        handler._async_stub.GetVersion = AsyncMock(return_value=version)
        rpc = AsyncMock(
            return_value=response, side_effect=LostResponse() if failure == "lost_rpc" else None
        )
        handler._async_stub.Insert = rpc
    else:
        handler._get_schema = MagicMock(return_value=(collection_schema, 0))
        handler._stub = MagicMock()
        handler._stub.GetVersion.return_value = version
        rpc = MagicMock(
            return_value=response, side_effect=LostResponse() if failure == "lost_rpc" else None
        )
        handler._stub.Insert = rpc
    with pytest.raises(IngestionFunctionChainError) as caught:
        if asynchronous:
            await handler.insert_rows("c", [{"raw": "one two"}], retry_times=2)
        else:
            handler.insert_rows("c", [{"raw": "one two"}], retry_times=2)
    assert rpc.call_count == 1
    assert caught.value.retriable is False
    if failure == "partial_rate_limit":
        result = caught.value.mutation_result
        assert list(result.primary_keys) == [100]
        assert result.ingestion_result["source_row_indexes"] == [0]
        assert caught.value.unknown_outcome is False
    else:
        assert caught.value.mutation_result is None
        assert caught.value.unknown_outcome is True
