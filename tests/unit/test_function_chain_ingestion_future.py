from unittest.mock import MagicMock

import grpc
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, FunctionChain, FunctionChainStage
from pymilvus.client.abstract import MutationResult
from pymilvus.client.asynch import MutationFuture
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.exceptions import IngestionFunctionChainError, MilvusException, ParamError
from pymilvus.function_chain import col, fn
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2
from pymilvus.orm.mutation import MutationResult as OrmMutationResult


class LostInsertResponse(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "INSERT response unavailable"


def collection_schema():
    chain = (
        FunctionChain(FunctionChainStage.INGESTION, name="chunk_and_embed")
        .flat_map("chunks", fn.py_udf(col("raw", DataType.VARCHAR), resource_name="chunker"))
        .select(["chunks.text", "chunks.vector"], ["text", "vector"])
    )
    return CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("text", DataType.VARCHAR, max_length=4096),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
        ],
        function_chains=[chain],
    ).to_dict()


def mutation_response(*, partial=False):
    ids = [100] if partial else [100, 101, 102]
    source_rows = [0] if partial else [0, 0, 1]
    return milvus_pb2.MutationResult(
        status=(
            common_pb2.Status(
                code=8,
                error_code=common_pb2.RateLimit,
                reason="partial append",
                retriable=True,
            )
            if partial
            else common_pb2.Status()
        ),
        IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=ids)),
        insert_cnt=len(ids),
        timestamp=12345,
        succ_index=[] if partial else [0, 1],
        err_index=[0, 1] if partial else [],
        ingestion_result=milvus_pb2.IngestionFunctionChainResult(
            input_rows=2,
            output_rows=len(ids),
            output_sources=milvus_pb2.FunctionChainRowMapping(
                source_offsets=list(range(len(ids) + 1)), source_row_indexes=source_rows
            ),
        ),
    )


def grpc_future(response=None, error=None):
    future = MagicMock()
    future.result.return_value = response
    future.result.side_effect = error
    future.exception.return_value = error
    future.code.return_value = grpc.StatusCode.UNAVAILABLE
    future.details.return_value = "INSERT response unavailable"
    return future


@pytest.fixture
def handler():
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    value = GrpcHandler(channel=channel)
    value._stub = MagicMock()
    version = milvus_pb2.GetVersionResponse()
    version.status.extra_info["function_chain_transform_version"] = "1"
    value._stub.GetVersion.return_value = version
    value.describe_collection = MagicMock(return_value=collection_schema())
    return value


@pytest.mark.parametrize("asynchronous", [False, True])
def test_column_insert_prepares_transient_input_and_preserves_expansion_metadata(
    handler, asynchronous
):
    response = mutation_response()
    rpc = grpc_future(response)
    handler._stub.Insert.future.return_value = rpc
    callback_results = []
    result = handler.batch_insert(
        "c",
        [{"name": "raw", "type": DataType.VARCHAR, "values": ["one two", "three"]}],
        _async=asynchronous,
        _callback=callback_results.append,
    )
    if asynchronous:
        future = result
        result = future.result()
        assert future.result(raw=True) is response
        assert callback_results == [result]
        assert future.is_done() is True
    else:
        assert callback_results == []
    assert list(result.primary_keys) == [100, 101, 102]
    assert result.ingestion_result == {
        "input_rows": 2,
        "output_rows": 3,
        "source_offsets": [0, 1, 2, 3],
        "source_row_indexes": [0, 0, 1],
        "outcome_unknown": False,
    }
    request = handler._stub.Insert.future.call_args.args[0]
    assert request.num_rows == 2
    assert [field.field_name for field in request.fields_data] == ["raw"]
    assert list(request.fields_data[0].scalars.string_data.data) == ["one two", "three"]
    assert handler.describe_collection.call_count == 1
    assert handler._stub.Insert.future.call_count == 1
    assert rpc.result.call_count == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["partial", "lost_response", "lost_dispatch"])
def test_column_ingestion_failure_never_replays_and_retains_confirmed_result(
    handler, asynchronous, failure
):
    response = mutation_response(partial=True)
    rpc = grpc_future(response, LostInsertResponse() if failure == "lost_response" else None)
    handler._stub.Insert.future.return_value = rpc
    if failure == "lost_dispatch":
        handler._stub.Insert.future.side_effect = LostInsertResponse()
    callbacks = []
    future = None
    with pytest.raises(IngestionFunctionChainError) as caught:
        result = handler.batch_insert(
            "c",
            [{"name": "raw", "type": DataType.VARCHAR, "values": ["one two", "three"]}],
            _async=asynchronous,
            _callback=callbacks.append,
            retry_times=2,
        )
        if asynchronous:
            future = result
            future.result()
    error = caught.value
    assert error.retriable is False
    assert handler._stub.Insert.future.call_count == 1
    assert callbacks == []
    if failure == "partial":
        assert error.unknown_outcome is False
        assert list(error.mutation_result.primary_keys) == [100]
        assert error.mutation_result.ingestion_result["source_row_indexes"] == [0]
        assert error.mutation_result._raw is response
        assert error.compatible_code == common_pb2.RateLimit
    else:
        assert error.unknown_outcome is True
        assert error.mutation_result is None
        assert error.code == grpc.StatusCode.UNAVAILABLE
    if future is not None:
        with pytest.raises(IngestionFunctionChainError) as repeated:
            future.result()
        assert repeated.value is error
        assert rpc.result.call_count <= 1


@pytest.mark.parametrize("failure", ["partial", "lost_response", "cancelled", "none"])
def test_ingestion_future_done_retains_failure_for_result_and_exception(failure):
    response = mutation_response(partial=True) if failure == "partial" else None
    transport_error = {
        "lost_response": LostInsertResponse(),
        "cancelled": grpc.FutureCancelledError(),
    }.get(failure)
    rpc = grpc_future(response, transport_error)
    callback_results = []
    future = MutationFuture(rpc, callback_results.append, _ingestion_function_chain=True)
    future.done()
    assert future.is_done() is True
    with pytest.raises(IngestionFunctionChainError) as caught:
        future.result()
    with pytest.raises(IngestionFunctionChainError) as same:
        future.exception()
    assert caught.value is same.value
    assert callback_results == []
    assert future._response is response
    assert rpc.result.call_count == 1
    assert caught.value.unknown_outcome is (failure != "partial")


@pytest.mark.parametrize("mode", ["sync", "future_result", "future_done"])
def test_server_partial_unknown_outcome_retains_confirmed_ids_and_lineage(handler, mode):
    response = mutation_response(partial=True)
    response.ingestion_result.outcome_unknown = True
    rpc = grpc_future(response)
    handler._stub.Insert.future.return_value = rpc
    with pytest.raises(IngestionFunctionChainError) as caught:
        result = handler.batch_insert(
            "c",
            [{"name": "raw", "type": DataType.VARCHAR, "values": ["one two", "three"]}],
            _async=mode != "sync",
            retry_times=2,
        )
        if mode == "future_done":
            result.done()
            result.exception()
        elif mode == "future_result":
            result.result()
    error = caught.value
    assert error.unknown_outcome is True
    assert error.retriable is False
    assert list(error.mutation_result.primary_keys) == [100]
    assert error.mutation_result.ingestion_result["source_row_indexes"] == [0]
    assert error.mutation_result.ingestion_result["outcome_unknown"] is True
    assert error.mutation_result._raw is response
    assert handler._stub.Insert.future.call_count == 1
    assert rpc.result.call_count == 1


def test_local_wait_timeout_can_later_recover_response_without_replaying_insert():
    response = mutation_response()
    rpc = grpc_future(response)
    rpc.result.side_effect = [grpc.FutureTimeoutError(), response]
    callbacks = []
    future = MutationFuture(rpc, callbacks.append, _ingestion_function_chain=True)
    with pytest.raises(IngestionFunctionChainError) as caught:
        future.result(timeout=0.01)
    assert caught.value.unknown_outcome is True
    assert caught.value.code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert future.is_done() is False
    result = future.result(timeout=1)
    assert list(result.primary_keys) == [100, 101, 102]
    assert callbacks == [result]
    assert rpc.result.call_args_list[0].kwargs == {"timeout": 0.01}
    assert rpc.result.call_args_list[1].kwargs == {"timeout": 1}
    # result(timeout=...) must not first block on Future.exception().
    rpc.exception.assert_not_called()


def test_ingestion_future_done_parses_success_and_calls_callbacks_once():
    response = mutation_response()
    rpc = grpc_future(response)
    callbacks = []
    future = MutationFuture(rpc, callbacks.append, _ingestion_function_chain=True)
    future.done()
    future.done()
    result = future.result()
    assert future.result(raw=True) is response
    assert callbacks == [result]
    assert rpc.result.call_count == 1
    assert result.ingestion_result["source_row_indexes"] == [0, 0, 1]


def test_ingestion_future_exception_marks_rpc_failure_unknown():
    rpc = grpc_future(error=LostInsertResponse())
    future = MutationFuture(rpc, _ingestion_function_chain=True)
    with pytest.raises(IngestionFunctionChainError) as caught:
        future.exception()
    assert caught.value.unknown_outcome is True
    assert future.is_done() is True
    with pytest.raises(IngestionFunctionChainError) as same:
        future.result()
    assert same.value is caught.value
    rpc.result.assert_not_called()


def test_unbound_future_still_preserves_server_ingestion_partial_metadata():
    response = mutation_response(partial=True)
    rpc = grpc_future(response)
    future = MutationFuture(rpc)
    with pytest.raises(IngestionFunctionChainError) as caught:
        future.result()
    assert caught.value.mutation_result._raw is response
    with pytest.raises(IngestionFunctionChainError) as same:
        future.result()
    assert same.value is caught.value
    assert rpc.result.call_count == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_column_ingestion_rejects_unsupported_server_before_dispatch(handler, asynchronous):
    handler._stub.GetVersion.return_value = milvus_pb2.GetVersionResponse()
    with pytest.raises(ParamError, match="does not advertise"):
        result = handler.batch_insert(
            "c",
            [{"name": "raw", "type": DataType.VARCHAR, "values": ["hello"]}],
            _async=asynchronous,
        )
        if asynchronous:
            result.result()
    handler._stub.Insert.future.assert_not_called()


def test_typed_transient_vector_column_uses_dimension_and_keeps_cached_schema(handler):
    chain = (
        FunctionChain(FunctionChainStage.INGESTION, name="translate_vector")
        .map("text", fn.py_udf(col("raw_vector", DataType.FLOAT_VECTOR), resource_name="describe"))
        .map("vector", fn.py_udf(col("text"), resource_name="embed"))
    )
    schema = collection_schema()
    schema["function_chains"] = [chain.to_dict()]
    original_fields = [dict(field) for field in schema["fields"]]
    columns = [{"name": "raw_vector", "type": DataType.FLOAT_VECTOR, "values": [[1.0, 2.0]]}]
    request = handler._prepare_batch_insert_request("c", columns, schema=schema)
    assert request.num_rows == 1
    assert request.fields_data[0].field_name == "raw_vector"
    assert request.fields_data[0].vectors.dim == 2
    assert list(request.fields_data[0].vectors.float_vector.data) == [1.0, 2.0]
    assert schema["fields"] == original_fields
    assert columns[0]["values"] == [[1.0, 2.0]]


@pytest.mark.parametrize(
    ("dtype", "value", "dimension"),
    [
        (DataType.FLOAT16_VECTOR, b"\x00\x00\x00\x00", 2),
        (DataType.BFLOAT16_VECTOR, b"\x00\x00\x00\x00", 2),
        (DataType.INT8_VECTOR, b"\x01\x02", 2),
        (DataType.BINARY_VECTOR, b"\x01\x02", 16),
    ],
)
def test_typed_transient_byte_vector_column_uses_logical_dimension(
    handler, dtype, value, dimension
):
    chain = (
        FunctionChain(FunctionChainStage.INGESTION, name="describe_vector")
        .map("text", fn.py_udf(col("raw_vector", dtype), resource_name="describe"))
        .map("vector", fn.py_udf(col("text"), resource_name="embed"))
    )
    schema = collection_schema()
    schema["function_chains"] = [chain.to_dict()]
    request = handler._prepare_batch_insert_request(
        "c", [{"name": "raw_vector", "type": dtype, "values": [value]}], schema=schema
    )
    field = request.fields_data[0]
    assert field.type == dtype
    assert field.vectors.dim == dimension
    assert (
        field.vectors.WhichOneof("data")
        == {
            DataType.FLOAT16_VECTOR: "float16_vector",
            DataType.BFLOAT16_VECTOR: "bfloat16_vector",
            DataType.INT8_VECTOR: "int8_vector",
            DataType.BINARY_VECTOR: "binary_vector",
        }[dtype]
    )


def test_in_place_column_map_accepts_original_input_field(handler):
    chain = (
        FunctionChain(FunctionChainStage.INGESTION, name="translate")
        .map("text", fn.py_udf(col("text"), resource_name="translate"))
        .map("vector", fn.py_udf(col("text"), resource_name="embed"))
    )
    schema = collection_schema()
    schema["function_chains"] = [chain.to_dict()]
    # The server can flag a final output that is also read by the chain.
    request = handler._prepare_batch_insert_request(
        "c", [{"name": "text", "type": DataType.VARCHAR, "values": ["hello"]}], schema=schema
    )
    assert [field.field_name for field in request.fields_data] == ["text"]
    assert schema["fields"][1]["is_function_output"] is True


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unbound_column_insert_preserves_legacy_result_and_skips_capability(handler, asynchronous):
    schema = collection_schema()
    schema["function_chains"] = []
    for field in schema["fields"]:
        field["is_function_output"] = False
    handler.describe_collection.return_value = schema
    response = milvus_pb2.MutationResult(
        IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[42])), insert_cnt=1
    )
    handler._stub.Insert.future.return_value = grpc_future(response)
    result = handler.batch_insert(
        "c",
        [
            {"name": "text", "type": DataType.VARCHAR, "values": ["hello"]},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "values": [[1.0, 2.0]]},
        ],
        _async=asynchronous,
    )
    if asynchronous:
        result = result.result()
    assert list(result.primary_keys) == [42]
    assert result.ingestion_result is None
    handler._stub.GetVersion.assert_not_called()


def test_unbound_column_insert_keeps_legacy_rpc_retry(handler):
    schema = collection_schema()
    schema["function_chains"] = []
    schema["fields"] = [
        {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False}
    ]
    response = milvus_pb2.MutationResult(
        IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[42])), insert_cnt=1
    )
    rpc = grpc_future(response)
    rpc.result.side_effect = [LostInsertResponse(), response]
    handler._stub.Insert.future.return_value = rpc
    result = handler.batch_insert(
        "c",
        [{"name": "id", "type": DataType.INT64, "values": [42]}],
        schema=schema,
        retry_times=2,
        initial_back_off=0,
    )
    assert list(result.primary_keys) == [42]
    assert handler._stub.Insert.future.call_count == 2
    assert result.ingestion_result is None


def test_unbound_future_keeps_legacy_transport_error_type():
    future = MutationFuture(grpc_future(error=LostInsertResponse()))
    with pytest.raises(MilvusException) as caught:
        future.result()
    assert not isinstance(caught.value, IngestionFunctionChainError)


@pytest.mark.parametrize("has_ingestion", [False, True])
def test_orm_mutation_result_exposes_optional_lineage(has_ingestion):
    response = mutation_response() if has_ingestion else milvus_pb2.MutationResult()
    client_result = MutationResult(response)
    result = OrmMutationResult(client_result)
    assert result.ingestion_result == client_result.ingestion_result
    assert OrmMutationResult(None).ingestion_result is None
