"""RLS contracts through public clients, real request builders and protobuf serialization."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pymilvus import AnnSearchRequest, AsyncMilvusClient, DataType, MilvusClient, RRFRanker
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.connection_manager import ConnectionConfig
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2

TAGS = {
    "department": "研发",
    "level": 2**63 - 1,
    "ratio": 3.0,
    "teams": ["red", "blue"],
    "levels": [1, 2],
    "ratios": [1.5, 2.0],
    "empty": [],
}
POLICY = {
    "policy_name": "owner",
    "policy_type": "permissive",
    "actions": ["query", "search_iterator"],
    "using_expr": "owner == $current_principal",
    "description": "Owners may read their rows",
}
SCHEMA = {
    "fields": [
        {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
        {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
    ],
    "enable_dynamic_field": False,
}


@pytest_asyncio.fixture(params=[False, True], ids=["sync", "async"])
async def client_and_stub(request):
    is_async = request.param
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler_type = AsyncGrpcHandler if is_async else GrpcHandler
    client_type = AsyncMilvusClient if is_async else MilvusClient
    handler = handler_type(channel=channel)
    mock = AsyncMock if is_async else MagicMock
    stub = mock()
    handler._get_schema = mock(return_value=(SCHEMA, 0))
    if is_async:
        handler.ensure_channel_ready = AsyncMock()
        handler._async_stub = stub
    else:
        handler._stub = stub
    client = client_type.__new__(client_type)
    client._config = ConnectionConfig.from_uri("http://localhost:19530", db_name="original")
    client._get_connection = AsyncMock(return_value=handler) if is_async else lambda: handler
    return client, handler, stub, is_async


MANAGEMENT = [
    ("create_row_policy", "CreateRowPolicy", POLICY, common_pb2.Status()),
    ("update_row_policy", "UpdateRowPolicy", POLICY, common_pb2.Status()),
    ("drop_row_policy", "DropRowPolicy", {"policy_name": "owner"}, common_pb2.Status()),
    (
        "list_row_policies",
        "ListRowPolicies",
        {},
        milvus_pb2.ListRowPoliciesResponse(
            policies=[
                milvus_pb2.RowPolicy(
                    policy_name="owner",
                    policy_type=milvus_pb2.RowPolicyTypePermissive,
                    actions=[milvus_pb2.Query, milvus_pb2.SearchIterator],
                    using_expr=POLICY["using_expr"],
                )
            ]
        ),
    ),
    (
        "set_rls_principal_tags",
        "SetRLSPrincipalTags",
        {"principal_name": "alice", "tags": TAGS},
        common_pb2.Status(),
    ),
    (
        "get_rls_principal_tags",
        "GetRLSPrincipalTags",
        {"principal_name": "alice"},
        milvus_pb2.GetRLSPrincipalTagsResponse(tags=json.dumps(TAGS)),
    ),
    (
        "list_rls_principals",
        "ListRLSPrincipals",
        {},
        milvus_pb2.ListRLSPrincipalsResponse(principal_names=["alice"]),
    ),
    (
        "delete_rls_principal_tags",
        "DeleteRLSPrincipalTags",
        {"principal_name": "alice", "tag_keys": ["department"]},
        common_pb2.Status(),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "rpc_name", "kwargs", "response"), MANAGEMENT)
@pytest.mark.parametrize("db_name", [None, "other"])
async def test_management_wire(client_and_stub, method, rpc_name, kwargs, response, db_name):
    client, _, stub, is_async = client_and_stub
    rpc = getattr(stub, rpc_name)
    rpc.return_value = response
    result = getattr(client, method)(
        "documents", db_name=db_name, timeout=2, client_request_id="rls-test", **kwargs
    )
    if is_async:
        result = await result
    rpc.assert_called_once()
    request = rpc.call_args.args[0]
    wire = type(request).FromString(request.SerializeToString())
    assert wire.db_name == ("original" if db_name is None else db_name)
    assert wire.collection_name == "documents"
    metadata = dict(rpc.call_args.kwargs["metadata"])
    assert metadata["dbname"] == wire.db_name
    assert metadata["client-request-id"] == "rls-test"
    assert rpc.call_args.kwargs["timeout"] == 2
    if method in ("create_row_policy", "update_row_policy"):
        assert list(wire.actions) == [0, 6]
        assert wire.policy_type == 1
        assert wire.using_expr == POLICY["using_expr"]
        assert wire.description == POLICY["description"]
        assert not wire.roles
    elif method == "set_rls_principal_tags":
        assert json.loads(wire.tags) == TAGS
        assert type(json.loads(wire.tags)["ratio"]) is float
    elif method == "get_rls_principal_tags":
        assert result == TAGS
        assert type(result["level"]) is int
        assert type(result["ratio"]) is float
    elif method == "list_row_policies":
        assert result[0]["actions"] == ["Query", "SearchIterator"]
        assert result[0]["policy_type"] == "RowPolicyTypePermissive"
    elif method == "list_rls_principals":
        assert result == ["alice"]
    elif method == "delete_rls_principal_tags":
        assert list(wire.tag_keys) == ["department"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "rpc_name", "kwargs", "response"), MANAGEMENT)
async def test_management_preserves_server_errors(
    client_and_stub, method, rpc_name, kwargs, response
):
    client, _, stub, is_async = client_and_stub
    response = type(response).FromString(response.SerializeToString())
    status = response if isinstance(response, common_pb2.Status) else response.status
    status.code = 1800
    status.error_code = common_pb2.PermissionDenied
    status.reason = "RLS permission denied"
    getattr(stub, rpc_name).return_value = response
    with pytest.raises(MilvusException, match="RLS permission denied") as exc:
        result = getattr(client, method)("documents", **kwargs)
        if is_async:
            await result
    assert exc.value.code == 1800
    assert exc.value.compatible_code == common_pb2.PermissionDenied
    getattr(stub, rpc_name).assert_called_once()


DATA_OPS = [
    ("insert", "Insert", {"data": [{"id": 1, "vector": [0.1, 0.2]}]}),
    ("upsert", "Upsert", {"data": [{"id": 1, "vector": [0.1, 0.2]}]}),
    ("delete", "Delete", {"filter": "id > 0"}),
    ("query", "Query", {"filter": "id > 0"}),
    ("get", "Query", {"ids": [1]}),
    ("search", "Search", {"data": [[0.1, 0.2]], "anns_field": "vector"}),
    (
        "hybrid_search",
        "HybridSearch",
        {
            "reqs": [AnnSearchRequest([[0.1, 0.2]], "vector", {}, 2)],
            "ranker": RRFRanker(),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "rpc_name", "kwargs"), DATA_OPS)
@pytest.mark.parametrize(
    "rls", [{}, {"rls_principal": "用户'alice", "skip_rls": False}, {"skip_rls": True}]
)
async def test_runtime_wire(client_and_stub, method, rpc_name, kwargs, rls):
    client, _, stub, is_async = client_and_stub
    # A failure response stops before result decoding, while exercising the real status path.
    status = common_pb2.Status(code=1800, reason="denied by RLS")
    response_type = {
        "Insert": milvus_pb2.MutationResult,
        "Upsert": milvus_pb2.MutationResult,
        "Delete": milvus_pb2.MutationResult,
        "Query": milvus_pb2.QueryResults,
        "Search": milvus_pb2.SearchResults,
        "HybridSearch": milvus_pb2.SearchResults,
    }[rpc_name]
    rpc = getattr(stub, rpc_name)
    rpc.return_value = response_type(status=status)
    if not is_async:
        rpc.future.return_value.result.return_value = rpc.return_value
    with pytest.raises(MilvusException, match="denied by RLS"):
        result = getattr(client, method)("documents", timeout=2, **kwargs, **rls)
        if is_async:
            await result
    call = rpc.call_args or rpc.future.call_args
    request = call.args[0] if call.args else call.kwargs["request"]
    wire = type(request).FromString(request.SerializeToString())
    assert wire.rls_principal == rls.get("rls_principal", "")
    assert wire.skip_rls is rls.get("skip_rls", False)
    if rpc_name == "HybridSearch":
        assert len(wire.requests) == 1
        assert wire.requests[0].rls_principal == wire.rls_principal


@pytest.mark.parametrize(
    "tags",
    [
        {},
        [],
        {1: "bad"},
        {"k": True},
        {"k": None},
        {"k": {}},
        {"k": [[1]]},
        {"k": [False]},
        {"k": float("nan")},
        {"k": [float("inf")]},
    ],
)
def test_invalid_tags(tags):
    with pytest.raises(ParamError):
        Prepare.set_rls_principal_tags_request("", "documents", "alice", tags)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rls_principal": 1},
        {"rls_principal": None},
        {"skip_rls": "false"},
        {"skip_rls": 1},
        {"skip_rls": None},
    ],
)
def test_invalid_runtime_options(kwargs):
    with pytest.raises(ParamError):
        Prepare.query_request("documents", "", [], [], **kwargs)


@pytest.mark.parametrize(
    "method", [Prepare.create_row_policy_request, Prepare.update_row_policy_request]
)
def test_policy_validation(method):
    for action in ("get", "RowPolicyActionGet", 8, True, [], "bad"):
        with pytest.raises(ParamError):
            method("", "documents", "owner", "permissive", action)
    for policy_type in (0, True, "unknown", None):
        with pytest.raises(ParamError):
            method("", "documents", "owner", policy_type, "query")
    assert list(method("", "documents", "owner", 1, 0).actions) == [milvus_pb2.Query]


def test_bulk_import_and_delete_all_tags():
    request = Prepare.do_bulk_insert(
        "documents", "", ["rows.json"], rls_principal="Alice", skip_rls=False
    )
    assert {v.key: v.value for v in request.options} == {
        "rls_principal": "Alice",
        "skip_rls": "false",
    }
    request = Prepare.delete_rls_principal_tags_request("", "documents", "alice", None)
    assert not request.tag_keys


@pytest.mark.parametrize("method", [Prepare.batch_insert_param, Prepare.batch_upsert_param])
def test_column_writes(method):
    request = method(
        "documents",
        [
            {"name": "id", "type": DataType.INT64, "values": [1]},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "values": [[0.1, 0.2]]},
        ],
        "",
        SCHEMA["fields"],
        rls_principal="alice",
        skip_rls=True,
    )
    assert request.rls_principal == "alice"
    assert request.skip_rls


@pytest.mark.asyncio
async def test_prebuilt_writes_preserve_request_context(client_and_stub):
    _, handler, _, is_async = client_and_stub
    handler.describe_collection = MagicMock(return_value=SCHEMA)
    kinds = ["upsert"] if is_async else ["insert", "upsert"]
    for kind in kinds:
        request_type = milvus_pb2.InsertRequest if kind == "insert" else milvus_pb2.UpsertRequest
        original = request_type(collection_name="documents", rls_principal="alice")
        builder = getattr(handler, f"_prepare_batch_{kind}_request")
        for options in ({}, {"rls_principal": "bob", "skip_rls": True}):
            request = builder("documents", [], **{f"{kind}_param": original}, **options)
            if is_async:
                request = await request
            assert request.rls_principal == options.get("rls_principal", "alice")
            assert request.skip_rls is options.get("skip_rls", False)
            assert original.rls_principal == "alice"
            assert not original.skip_rls


@pytest.mark.asyncio
@pytest.mark.parametrize("client_and_stub", [False], indirect=True)
@pytest.mark.parametrize("kind", ["query", "search"])
async def test_iterator_context_survives_probe_and_pages(client_and_stub, kind):
    client, handler, stub, _ = client_and_stub
    handler.describe_collection = MagicMock(return_value={**SCHEMA, "collection_id": 123})
    context = {"rls_principal": "alice", "skip_rls": False}
    if kind == "query":
        stub.Query.return_value = milvus_pb2.QueryResults(
            fields_data=[
                schema_pb2.FieldData(
                    type=DataType.INT64,
                    field_name="id",
                    scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[1])),
                )
            ],
            session_ts=123,
        )
        iterator = client.query_iterator("documents", batch_size=1, **context)
        rpc = stub.Query
    else:
        stub.Search.return_value = milvus_pb2.SearchResults(
            results=schema_pb2.SearchResultData(
                num_queries=1,
                top_k=1,
                topks=[1],
                scores=[1.0],
                ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
                search_iterator_v2_results=schema_pb2.SearchIteratorV2Results(
                    token="cursor", last_bound=1.0
                ),
            ),
        )
        iterator = client.search_iterator("documents", [[0.1, 0.2]], batch_size=1, **context)
        rpc = stub.Search
    try:
        iterator.next()
        iterator.next()
        assert rpc.call_count == 3
        for call in rpc.call_args_list:
            wire = call.args[0]
            assert wire.rls_principal == "alice"
            assert not wire.skip_rls
    finally:
        iterator.close()
