"""POST tables acknowledge execution and preserve an independent Arrow schema."""

import struct
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest
from pymilvus import (
    AnnSearchRequest,
    DataType,
    FunctionChain,
    FunctionChainResultMode,
    FunctionChainStage,
    FunctionChainTableResult,
    PostProcessInput,
    RRFRanker,
    SearchResult,
)
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.function_chain_capability import (
    POST_TABLE_VERSION_KEY,
    TRANSFORM_VERSION_KEY,
)
from pymilvus.client.function_chain_table import parse_table_response
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.function_chain import FunctionChainOp, agg, col, fn
from pymilvus.grpc_gen import milvus_pb2, schema_pb2


def post(input_source="fused"):
    return FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table", input=input_source)


def ipc_table(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return schema_pb2.FunctionChainTableResult(
        arrow_ipc=sink.getvalue().to_pybytes(), row_count=table.num_rows
    )


def table_response(endpoint="search"):
    response = milvus_pb2.QueryResults() if endpoint == "query" else milvus_pb2.SearchResults()
    response.post_process_result.CopyFrom(
        ipc_table(
            pa.table(
                {
                    "summary": pa.array(
                        [{"texts": ["你好", None, "world"], "count": 2**53 + 1}, None],
                        type=pa.struct(
                            [
                                pa.field("texts", pa.list_(pa.string())),
                                pa.field("count", pa.int64()),
                            ]
                        ),
                    )
                }
            )
        )
    )
    return response


def test_post_options_roundtrip_and_legacy_defaults():
    legacy = FunctionChain(FunctionChainStage.POST_PROCESS).select(["text"])
    assert not legacy.to_proto().HasField("post_process_options")
    chain = post("ann_inputs").union_all().aggregate({"count": agg.count()})
    wire = chain.to_proto()
    assert wire.post_process_options.result_mode == FunctionChainResultMode.TABLE
    assert wire.post_process_options.input == PostProcessInput.ANN_INPUTS
    assert [operation.op for operation in wire.ops] == ["union_all", "aggregate"]
    assert FunctionChain.from_proto(wire).to_proto() == wire
    assert FunctionChain.from_dict(chain.to_dict()).to_proto() == wire


@pytest.mark.parametrize(
    "kwargs",
    [
        {"result_mode": "wrong"},
        {"input": "wrong"},
        {"result_mode": True},
        {"input": True},
        {"input": "ann_inputs"},
    ],
)
def test_invalid_post_options_are_rejected(kwargs):
    with pytest.raises(ParamError):
        FunctionChain(FunctionChainStage.POST_PROCESS, **kwargs)


@pytest.mark.parametrize("stage", [FunctionChainStage.INGESTION, FunctionChainStage.L2_RERANK])
def test_post_options_are_stage_scoped(stage):
    with pytest.raises(ParamError, match="require POST_PROCESS"):
        FunctionChain(stage, result_mode="table")


def test_grouping_lowers_to_one_aggregate_and_keeps_occurrence_hints():
    price = col("price", DataType.DOUBLE)
    chain = (
        post()
        .group_by(col("tenant", DataType.VARCHAR), aliases=["group"])
        .aggregate(
            {
                "total": agg.sum(price),
                "average": agg.avg(price),
                "count": agg.count(),
                "values": agg.collect_list(col("tags")),
            }
        )
    )
    wire = chain.to_proto()
    assert [op.op for op in wire.ops] == ["aggregate"]
    operation = wire.ops[0]
    assert not operation.HasField("expr")
    assert list(operation.inputs) == ["tenant", "price", "price", "tags"]
    assert list(operation.outputs) == ["group", "total", "average", "count", "values"]
    assert [v.int64_value for v in operation.params["$input_data_types"].array_value.values] == [
        DataType.VARCHAR,
        DataType.DOUBLE,
        DataType.DOUBLE,
        0,
    ]
    assert (
        operation.params["aggregates"].array_value.values[2].object_value.fields["op"].string_value
        == "count"
    )
    assert "input" not in operation.params["aggregates"].array_value.values[2].object_value.fields


def test_custom_aggregate_key_hints_precede_expression_hints_without_explicit_inputs():
    chain = (
        post()
        .union_all()
        .group_by(col("tenant", DataType.VARCHAR))
        .aggregate("summary", fn.py_udf(col("text", DataType.VARCHAR), resource_name="summary"))
    )
    operation = chain.to_proto().ops[1]
    assert operation.op == "aggregate"
    assert not operation.inputs
    assert operation.expr.name == "py_udf"
    assert "aggregates" not in operation.params
    assert [v.string_value for v in operation.params["keys"].array_value.values] == ["tenant"]
    assert list(operation.outputs) == ["tenant", "summary"]
    assert [v.int64_value for v in operation.params["$input_data_types"].array_value.values] == [
        DataType.VARCHAR,
        DataType.VARCHAR,
    ]
    assert FunctionChain.from_dict(chain.to_dict()).to_proto() == chain.to_proto()


@pytest.mark.parametrize("grouped", [False, True])
def test_custom_aggregate_multiple_outputs_preserve_repeated_arguments_and_key_aliases(grouped):
    chain = post()
    builder = (
        chain.group_by(col("tenant", DataType.VARCHAR), aliases=["group"]) if grouped else chain
    )
    text = col("text", DataType.VARCHAR)
    builder.aggregate(
        ["summary", "keywords"],
        fn.py_udf(text, col("$query_index", DataType.INT64), text, resource_name="summary"),
    )
    wire = chain.to_proto()
    assert len(wire.ops) == 1
    operation = wire.ops[0]
    assert operation.op == "aggregate"
    assert not operation.inputs
    assert list(operation.outputs) == (["group"] if grouped else []) + ["summary", "keywords"]
    assert [arg.column.name for arg in operation.expr.args] == ["text", "$query_index", "text"]
    assert [v.int64_value for v in operation.params["$input_data_types"].array_value.values] == (
        ([DataType.VARCHAR] if grouped else [])
        + [DataType.VARCHAR, DataType.INT64, DataType.VARCHAR]
    )
    assert "aggregates" not in operation.params
    assert FunctionChain.from_proto(wire).to_proto() == wire


@pytest.mark.parametrize(
    "outputs,expr,message",
    [
        ([], fn.py_udf(col("text"), resource_name="summary"), "non-empty sequence"),
        ({"n": agg.count()}, fn.py_udf(col("text"), resource_name="summary"), "sequence"),
        (["a", "a"], fn.py_udf(col("text"), resource_name="summary"), "unique"),
        (["tenant"], fn.py_udf(col("text"), resource_name="summary"), "unique"),
        (["$score"], fn.py_udf(col("text"), resource_name="summary"), "ordinary"),
        ("summary", fn.round_decimal(col("score"), decimal=2), "fn.py_udf"),
        ("summary", "invalid", "fn.py_udf"),
    ],
)
def test_invalid_custom_aggregate_does_not_append_operation(outputs, expr, message):
    chain = post()
    with pytest.raises(ParamError, match=message):
        chain.group_by("tenant").aggregate(outputs, expr)
    assert not chain.ops


def test_native_aggregate_keyword_form_is_unchanged():
    operation = post().aggregate(aggregates={"n": agg.count()}).to_proto().ops[0]
    assert operation.op == "aggregate"
    assert not operation.HasField("expr")
    assert list(operation.outputs) == ["n"]
    assert "aggregates" in operation.params


@pytest.mark.parametrize("legacy,replacement", [("reduce", "aggregate"), ("concat", "union_all")])
def test_removed_operations_are_not_an_api_or_serializable(legacy, replacement):
    assert not hasattr(post(), legacy)
    assert not hasattr(post().group_by("tenant"), legacy)
    message = f"{legacy} is not supported; use {replacement}"
    with pytest.raises(ParamError, match=message):
        FunctionChainOp(op=legacy)
    wire = post().aggregate({"n": agg.count()}).to_proto()
    wire.ops[0].op = legacy
    with pytest.raises(ParamError, match=message):
        FunctionChain.from_proto(wire)
    definition = post().aggregate({"n": agg.count()}).to_dict()
    definition["ops"][0]["op"] = legacy
    with pytest.raises(ParamError, match=message):
        FunctionChain.from_dict(definition)


@pytest.mark.parametrize(
    "factory", [agg.sum, agg.avg, agg.min, agg.max, agg.first, agg.last, agg.collect_list]
)
def test_aggregate_arguments_use_column_refs(factory):
    with pytest.raises(ParamError, match="col"):
        factory("text")


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: c.union_all(),
        lambda c: c.aggregate({"count": agg.count()}),
        lambda c: c.group_by("tenant"),
        lambda c: c.aggregate("summary", fn.py_udf(col("text"), resource_name="summary")),
    ],
)
def test_new_group_operations_require_explicit_table(operation):
    with pytest.raises(ParamError, match="POST TABLE"):
        operation(FunctionChain(FunctionChainStage.POST_PROCESS))


@pytest.mark.parametrize(
    "stage",
    [
        FunctionChainStage.INGESTION,
        FunctionChainStage.PRE_PROCESS,
        FunctionChainStage.L0_RERANK,
        FunctionChainStage.L1_RERANK,
        FunctionChainStage.L2_RERANK,
    ],
)
@pytest.mark.parametrize(
    "operation",
    [
        lambda c: c.union_all(),
        lambda c: c.aggregate({"count": agg.count()}),
        lambda c: c.aggregate("summary", fn.py_udf(col("text"), resource_name="summary")),
    ],
)
def test_union_all_and_both_aggregate_modes_reject_other_stages(stage, operation):
    with pytest.raises(ParamError, match="POST TABLE"):
        operation(FunctionChain(stage))


def test_table_expansion_and_source_projection():
    post().explode("values").flat_map("rows", fn.py_udf(col("text"), resource_name="chunker"))
    post().select(["$query_index", "$ann_index"])
    post().group_by("$query_index").aggregate({"count": agg.count()})
    for name in ["$query_index", "$ann_index"]:
        with pytest.raises(ParamError, match="read-only"):
            post().map(name, fn.py_udf(col("text"), resource_name="model"))
        with pytest.raises(ParamError, match="read-only"):
            post().select(["text"], [name])
    with pytest.raises(ParamError, match="unique"):
        post().group_by("tenant").aggregate({"tenant": agg.count()})


def test_table_result_has_independent_nested_schema_and_integer_precision():
    response = table_response()
    result = parse_table_response(response, expected=True)
    assert isinstance(result, FunctionChainTableResult)
    assert result.row_count == len(result) == 2
    assert result.columns == ["summary"]
    assert result.rows == [
        {"summary": {"texts": ["你好", None, "world"], "count": 2**53 + 1}},
        {"summary": None},
    ]
    assert result.to_arrow().schema == result.schema
    assert result.arrow_ipc == response.post_process_result.arrow_ipc


def test_typed_empty_table_is_not_a_missing_response():
    table = pa.table({"summary": pa.array([], type=pa.struct([pa.field("text", pa.string())]))})
    response = milvus_pb2.QueryResults(post_process_result=ipc_table(table))
    result = parse_table_response(response, expected=True)
    assert result.row_count == 0
    assert result.rows == []
    assert result.schema == table.schema


def test_all_null_column_retains_its_arrow_type_and_field_metadata():
    schema = pa.schema([pa.field("value", pa.int64(), metadata={b"description": b"nullable"})])
    table = pa.Table.from_arrays([pa.array([None, None], type=pa.int64())], schema=schema)
    result = parse_table_response(
        milvus_pb2.QueryResults(post_process_result=ipc_table(table)), expected=True
    )
    assert result.schema == schema
    assert result.schema.field("value").metadata == {b"description": b"nullable"}
    assert result.rows == [{"value": None}, {"value": None}]


def test_nonnullable_column_cannot_publish_null_rows():
    schema = pa.schema([pa.field("value", pa.int64(), nullable=False)])
    table = pa.Table.from_arrays([pa.array([None], type=pa.int64())], schema=schema)
    with pytest.raises(MilvusException, match="non-nullable column"):
        parse_table_response(
            milvus_pb2.QueryResults(post_process_result=ipc_table(table)), expected=True
        )


def test_duplicate_columns_and_query_payload_conflicts_fail_closed():
    duplicate = pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=["value", "value"])
    with pytest.raises(MilvusException, match="unique"):
        parse_table_response(
            milvus_pb2.QueryResults(post_process_result=ipc_table(duplicate)), expected=True
        )
    response = table_response("query")
    response.fields_data.add(field_name="legacy", type=DataType.INT64)
    with pytest.raises(MilvusException, match="ordinary query fields"):
        parse_table_response(response, expected=True)


def test_nested_duplicate_members_fail_before_exposing_schema_or_rows():
    duplicate = pa.StructArray.from_arrays(
        [pa.array([1]), pa.array([2])], names=["member", "member"]
    )
    table = pa.table({"payload": pa.ListArray.from_arrays(pa.array([0, 1]), duplicate)})
    with pytest.raises(MilvusException, match="Struct member names"):
        parse_table_response(
            milvus_pb2.QueryResults(post_process_result=ipc_table(table)), expected=True
        )


def test_invalid_utf8_nested_in_struct_is_fully_validated():
    bad = pa.Array.from_buffers(
        pa.string(), 1, [None, pa.py_buffer(struct.pack("<ii", 0, 1)), pa.py_buffer(b"\xff")]
    )
    table = pa.table({"payload": pa.StructArray.from_arrays([bad], names=["text"])})
    with pytest.raises(MilvusException, match="Invalid POST table Arrow IPC"):
        parse_table_response(
            milvus_pb2.QueryResults(post_process_result=ipc_table(table)), expected=True
        )


def test_native_and_custom_aggregate_aliases_are_validated_as_strings():
    with pytest.raises(ParamError, match="aliases must be non-empty strings"):
        post().aggregate({col("bad_alias"): agg.count()})
    with pytest.raises(ParamError, match="aliases must be non-empty strings"):
        post().aggregate([col("bad_alias")], fn.py_udf(col("text"), resource_name="summary"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.ClearField("post_process_result"),
        lambda r: setattr(r.post_process_result, "row_count", -1),
        lambda r: setattr(r.post_process_result, "row_count", 999),
        lambda r: setattr(r.post_process_result, "arrow_ipc", b"invalid"),
        lambda r: setattr(
            r.post_process_result, "arrow_ipc", r.post_process_result.arrow_ipc + b"extra"
        ),
        lambda r: r.results.SetInParent(),
    ],
)
def test_invalid_or_unacknowledged_table_response_never_falls_back(mutation):
    response = table_response()
    mutation(response)
    with pytest.raises(MilvusException):
        parse_table_response(response, expected=True)


def test_unsolicited_table_response_is_rejected():
    with pytest.raises(MilvusException, match="Unexpected"):
        parse_table_response(table_response(), expected=False)


def test_legacy_search_results_keep_their_nq_and_hit_shape():
    response = milvus_pb2.SearchResults(
        results=schema_pb2.SearchResultData(
            num_queries=2,
            topks=[1, 1],
            scores=[0.9, 0.8],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2])),
        )
    )
    assert parse_table_response(response, expected=False) is None
    result = SearchResult(response.results)
    assert len(result) == 2
    assert result[0][0]["id"] == 1
    assert result[1][0]["id"] == 2


def test_ann_inputs_has_no_top_level_rank_params_and_keeps_branch_budget():
    branches = [milvus_pb2.SearchRequest(search_params=[])]
    chain = post("ann_inputs").union_all().aggregate({"count": agg.count()})
    request = Prepare.hybrid_search_request_with_ranker(
        "c", branches, None, 1, function_chains=chain
    )
    assert not request.rank_params
    for ranker, chains in [
        (RRFRanker(), chain),
        (None, [chain, FunctionChain(FunctionChainStage.L2_RERANK).merge("rrf")]),
    ]:
        with pytest.raises(ParamError, match="ANN_INPUTS"):
            Prepare.hybrid_search_request_with_ranker(
                "c", branches, ranker, 1, function_chains=chains
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"function_chains": FunctionChain(FunctionChainStage.POST_PROCESS).select(["text"])},
        {"function_chains": post("ann_inputs").select(["text"])},
        {"function_chains": post().select(["text"]), "iterator": True},
        {"function_chains": post().select(["text"]), "group_by_fields": ["tenant"]},
    ],
)
def test_unsupported_query_table_combinations_are_rejected(kwargs):
    with pytest.raises(ParamError):
        Prepare.query_request("c", "", ["text"], [], **kwargs)


@pytest.mark.parametrize(
    "mode,endpoint",
    [
        (mode, endpoint)
        for mode in ["sync", "async", "future"]
        for endpoint in ["search", "hybrid", "query"]
        if (mode, endpoint) != ("future", "query")
    ],
)
@pytest.mark.asyncio
async def test_handlers_table_success_and_missing_ack_fail_closed(mode, endpoint, monkeypatch):
    response = table_response(endpoint)
    version = milvus_pb2.GetVersionResponse()
    version.status.extra_info.update({TRANSFORM_VERSION_KEY: "1", POST_TABLE_VERSION_KEY: "1"})
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel) if mode == "async" else GrpcHandler(channel=channel)
    stub = MagicMock()
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
    setattr(stub, {"search": "Search", "hybrid": "HybridSearch", "query": "Query"}[endpoint], rpc)
    monkeypatch.setattr(f"{module}.ts_utils.construct_guarantee_ts", lambda *a, **kw: True)
    chain = post().select(["text"])
    kwargs = {"function_chains": chain}
    if mode == "future":
        kwargs["_async"] = True

    async def call():
        if endpoint == "search":
            result = handler.search("c", "vector", {}, 3, data=[[1, 0], [0, 1]], **kwargs)
        elif endpoint == "hybrid":
            branch = AnnSearchRequest([[1, 0], [0, 1]], "vector", {}, 5)
            result = handler.hybrid_search("c", [branch], None, 3, **kwargs)
        else:
            result = handler.query("c", "", ["text"], **kwargs)
        if mode == "async":
            return await result
        if mode == "future":
            return result.result()
        return result

    assert (await call()).rows == parse_table_response(response, expected=True).rows
    response.ClearField("post_process_result")
    with pytest.raises(MilvusException, match="missing post_process_result"):
        await call()
    version.status.extra_info.pop(POST_TABLE_VERSION_KEY)
    rpc.reset_mock()
    with pytest.raises(ParamError, match="POST TABLE support"):
        await call()
    rpc.assert_not_called()
    rpc.future.assert_not_called()
    version.status.extra_info[POST_TABLE_VERSION_KEY] = "1"
    monkeypatch.setattr(
        "pymilvus.client.function_chain_table.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(ParamError, match=r"pymilvus\[arrow\]"):
        await call()
    rpc.assert_not_called()
    rpc.future.assert_not_called()


def test_native_query_aggregation_cannot_be_silently_treated_as_table():
    with pytest.raises(ParamError, match="native Query aggregation"):
        Prepare.query_request(
            "c", "", ["count(*)"], [], function_chains=post().aggregate({"n": agg.count()})
        )


def test_ordinary_sdk_import_does_not_require_pyarrow():
    script = """
import sys
class BlockArrow:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            raise ImportError('optional Arrow intentionally unavailable')
sys.meta_path.insert(0, BlockArrow())
from pymilvus import FunctionChain, FunctionChainStage, SearchResult
FunctionChain(FunctionChainStage.POST_PROCESS).select(['text'])
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
