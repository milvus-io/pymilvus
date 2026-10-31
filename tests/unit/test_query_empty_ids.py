import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import AsyncMilvusClient, DataType, MilvusClient
from pymilvus.exceptions import DataTypeNotMatchException, MilvusException, ParamError
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2


def _query_response(ids):
    return milvus_pb2.QueryResults(
        status=common_pb2.Status(),
        fields_data=[
            schema_pb2.FieldData(
                type=DataType.INT64,
                field_name="id",
                scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=ids)),
            )
        ],
    )


@pytest.fixture(params=["sync", "async"])
def query_client(request, mock_grpc_handler, mock_async_grpc_handler, monkeypatch):
    schema = {"fields": [{"name": "id", "type": DataType.INT64, "is_primary": True}]}
    if request.param == "sync":
        handler = mock_grpc_handler
        with patch("pymilvus.milvus_client.milvus_client.ConnectionManager.get_instance") as cm:
            cm.return_value.get_or_create.return_value = handler
            client = MilvusClient()
        get_connection = MagicMock(return_value=handler)
        monkeypatch.setattr(handler, "_get_schema", MagicMock(return_value=(schema, 1)))
        rpc = handler._stub.Query
    else:
        handler = mock_async_grpc_handler
        client = AsyncMilvusClient()
        get_connection = AsyncMock(return_value=handler)
        monkeypatch.setattr(handler, "_get_schema", AsyncMock(return_value=(schema, 1)))
        rpc = handler._async_stub.Query

    monkeypatch.setattr(client, "_get_connection", get_connection)
    rpc.return_value = _query_response([1, 2])
    return client, get_connection, rpc


async def _query(client, **kwargs):
    result = client.query("records", limit=2, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.mark.asyncio
async def test_query_empty_ids_does_not_request_unfiltered_rows(query_client):
    client, get_connection, rpc = query_client
    assert await _query(client, ids=[]) == []
    get_connection.assert_called_once()
    rpc.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,expected_expr,expected_ids",
    [
        ({}, "", [1, 2]),
        ({"ids": None}, "", [1, 2]),
        ({"ids": 0}, "id in [0]", [0]),
        ({"ids": ""}, "id in ['']", [""]),
        ({"ids": [1, 2]}, "id in [1,2]", [1, 2]),
        ({"filter": "id > 0"}, "id > 0", [1, 2]),
        (
            {"filter": "id > {minimum}", "filter_params": {"minimum": 0}},
            "id > {minimum}",
            [1, 2],
        ),
    ],
)
async def test_query_other_selectors_keep_real_request_path(
    query_client, kwargs, expected_expr, expected_ids
):
    client, get_connection, rpc = query_client
    # Keep the string-primary-key control's schema and response correctly typed.
    if expected_ids == [""]:
        handler = get_connection.return_value
        handler._get_schema.return_value = (
            {"fields": [{"name": "id", "type": DataType.VARCHAR, "is_primary": True}]},
            1,
        )
        response = _query_response([])
        response.fields_data[0].type = DataType.VARCHAR
        response.fields_data[0].scalars.ClearField("long_data")
        response.fields_data[0].scalars.string_data.data.extend(expected_ids)
        rpc.return_value = response
    else:
        rpc.return_value = _query_response(expected_ids)

    result = await _query(client, **kwargs)
    assert [row["id"] for row in result] == expected_ids
    request = rpc.call_args.args[0]
    assert isinstance(request, milvus_pb2.QueryRequest)
    assert milvus_pb2.QueryRequest.FromString(request.SerializeToString()).expr == expected_expr
    assert request.collection_name == "records"
    if "filter_params" in kwargs:
        assert request.expr_template_values["minimum"].int64_val == 0
    get_connection.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filter_value,error",
    [("id > 0", ParamError), (123, DataTypeNotMatchException)],
)
async def test_empty_ids_preserves_filter_validation(query_client, filter_value, error):
    client, get_connection, rpc = query_client
    with pytest.raises(error):
        await _query(client, ids=[], filter=filter_value)
    get_connection.assert_not_called()
    rpc.assert_not_called()


@pytest.mark.asyncio
async def test_query_empty_ids_preserves_closed_client_guard(query_client, monkeypatch):
    client, _, rpc = query_client
    monkeypatch.setattr(client, "_get_connection", type(client)._get_connection.__get__(client))
    closed = client.close()
    if inspect.isawaitable(closed):
        await closed
    with pytest.raises(MilvusException, match="should create connection first"):
        await _query(client, ids=[])
    rpc.assert_not_called()
