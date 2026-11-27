"""Unit tests for ExecuteDQL (DQL, DQLRequest, DQLResult, sync & async execute_dql)."""

import ast
import asyncio
import inspect
import math
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pymilvus
import pymilvus.client.dql as client_dql_mod
import pytest
from pymilvus import DQL, AsyncMilvusClient, DQLRequest, DQLResult, MilvusClient, MilvusException
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.cache import GlobalCache
from pymilvus.client.call_context import CallContext
from pymilvus.client.dql import (
    prepare_execute_dql_batch,
    unpack_execute_dql_response,
    validate_execute_dql_params,
)
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.client.search_aggregation import SearchAggregation
from pymilvus.client.search_result import SearchResult
from pymilvus.client.types import DataType
from pymilvus.exceptions import ErrorCode, ParamError
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2
from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClientSession
from pymilvus.milvus_client.milvus_client import MilvusClientSession


def _make_search_result_data(
    ids,
    scores,
    topks=None,
    pk_name="id",
    output_fields=None,
    fields_data=None,
):
    topks = topks if topks is not None else [len(ids)]
    if ids and isinstance(ids[0], str):
        id_proto = schema_pb2.IDs(str_id=schema_pb2.StringArray(data=list(ids)))
    else:
        id_proto = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(ids)))
    return schema_pb2.SearchResultData(
        num_queries=len(topks),
        top_k=max(topks) if topks else 0,
        ids=id_proto,
        scores=list(scores),
        topks=list(topks),
        primary_field_name=pk_name,
        output_fields=list(output_fields or []),
        fields_data=list(fields_data or []),
    )


def _ok_status(extra_info=None):
    return common_pb2.Status(code=0, error_code=common_pb2.Success, extra_info=extra_info or {})


def _err_status(code, reason, error_code=common_pb2.UnexpectedError, retriable=False):
    return common_pb2.Status(
        code=code,
        error_code=error_code,
        reason=reason,
        retriable=retriable,
    )


class TestPublicExports:
    """Top-level pymilvus exports expose only DQL, DQLRequest, DQLResult."""

    def test_top_level_exports_exact(self):
        for sym in ("DQL", "DQLRequest", "DQLResult"):
            assert sym in pymilvus.__all__
            assert hasattr(pymilvus, sym)

        for forbidden in ("multi_search", "SearchRequest", "SubSearchResult"):
            assert forbidden not in pymilvus.__all__
            assert not hasattr(pymilvus, forbidden)

    def test_dql_namespace_only_exposes_search_in_v1(self):
        assert callable(DQL.search)
        assert not hasattr(DQL, "query")
        assert not hasattr(DQL, "hybrid_search")

    def test_dql_module_imports_without_typing_extensions_or_310_only_symbols(self):
        """Ensure Python 3.9 compatibility: runtime import never requires ParamSpec/Concatenate."""
        tree = ast.parse(Path(client_dql_mod.__file__).read_text(encoding="utf-8"))
        for child in ast.walk(tree):
            if isinstance(child, ast.ImportFrom):
                assert child.module != "typing_extensions"
                imported = {alias.name for alias in child.names}
                assert not (imported & {"ParamSpec", "Concatenate"})

        code = (
            "import sys, grpc\n"
            "sys.modules['typing_extensions'] = None\n"
            "import pymilvus.client.dql as dql_mod\n"
            "req = dql_mod.DQL.search('col', data=[[0.1, 0.2]])\n"
            "assert isinstance(req, dql_mod.DQLRequest)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    def test_execute_dql_example_parses_cleanly(self):
        example_path = (
            Path(client_dql_mod.__file__).resolve().parents[2] / "examples" / "execute_dql.py"
        )
        assert example_path.is_file()
        ast.parse(example_path.read_text(encoding="utf-8"))


class TestDQLSearchBuilder:
    """DQL.search exposes explicit keyword-only signature aligned with MilvusClient.search."""

    def test_signature_stays_aligned_with_milvus_client_search(self):
        """Drift guard: DQL.search matches MilvusClient.search minus self/timeout plus db_name."""
        client_params = inspect.signature(MilvusClient.search).parameters
        dql_sig = inspect.signature(DQL.search)
        dql_params = dql_sig.parameters

        expected_names = (set(client_params) - {"self", "timeout"}) | {"db_name"}
        assert set(dql_params) == expected_names
        assert "timeout" not in dql_params
        assert "db_name" in dql_params
        assert dql_params["collection_name"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert dql_params["kwargs"].kind == inspect.Parameter.VAR_KEYWORD

        for name, param in dql_params.items():
            if name in ("collection_name", "kwargs"):
                continue
            assert param.kind == inspect.Parameter.KEYWORD_ONLY
            if name in client_params:
                assert param.default == client_params[name].default
        assert dql_params["db_name"].default is None
        assert dql_sig.return_annotation is DQLRequest

    def test_positional_args_after_collection_name_rejected(self):
        with pytest.raises(TypeError):
            DQL.search("papers", [[0.1, 0.2]])  # type: ignore[misc]

    def test_defaults_and_kwargs_bound_into_immutable_request(self):
        req = DQL.search(
            "papers",
            data=[[0.1, 0.2]],
            anns_field="emb",
            filter="year >= 2020",
            limit=5,
            output_fields=["title"],
            db_name="archive",
            filter_params={"min_year": 2020},
            round_decimal=3,
        )
        assert isinstance(req, DQLRequest)
        assert req.op == "search"
        assert req.collection_name == "papers"
        assert req.data == [[0.1, 0.2]]
        assert req.anns_field == "emb"
        assert req.filter == "year >= 2020"
        assert req.limit == 5
        assert req.output_fields == ["title"]
        assert req.db_name == "archive"
        assert req.extra_kwargs["filter_params"] == {"min_year": 2020}
        assert req.extra_kwargs["round_decimal"] == 3

        # Immutable dataclass with shallow-copied extra_kwargs
        with pytest.raises(AttributeError):
            req.limit = 99  # type: ignore[misc]
        assert isinstance(req.extra_kwargs, dict)

    def test_dql_instance_call_does_not_bind_self(self):
        req = DQL().search("papers", data=[[0.1, 0.2]])
        assert req.collection_name == "papers"

    def test_missing_required_collection_name_raises_type_error(self):
        with pytest.raises(TypeError):
            DQL.search()  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad_timeout", [0, 0.0, 1.5, -1.0, "5"])
    def test_per_request_non_none_timeout_rejected_immediately(self, bad_timeout):
        with pytest.raises(ParamError, match="execute_dql"):
            DQL.search("papers", data=[[0.1, 0.2]], timeout=bad_timeout)

    def test_explicit_none_timeout_allowed(self):
        req = DQL.search("papers", data=[[0.1, 0.2]], timeout=None)
        assert "timeout" not in req.extra_kwargs

    def test_collection_name_and_db_name_validation_aligns_with_search(self):
        with pytest.raises(ParamError, match="db_name"):
            DQL.search("papers", data=[[0.1, 0.2]], db_name=123)  # type: ignore[arg-type]
        with pytest.raises(ParamError, match="collection_name"):
            DQL.search(None, data=[[0.1, 0.2]])  # type: ignore[arg-type]
        with pytest.raises(ParamError, match="collection_name"):
            DQL.search(123, data=[[0.1, 0.2]])  # type: ignore[arg-type]
        req = DQL.search("papers", data=[[0.1, 0.2]], db_name=None)
        assert req.db_name is None

    def test_bare_dql_request_rejected_as_unsupported_op(self):
        with pytest.raises(ParamError, match="Unsupported DQL operation"):
            validate_execute_dql_params([DQLRequest()])


class TestDQLResult:
    """DQLResult models ok / result / error / unwrap with frozen immutability."""

    def test_success_result(self):
        sr = SearchResult(_make_search_result_data([1], [0.1]))
        item = DQLResult(result=sr)
        assert item.ok is True
        assert item.result is sr
        assert item.error is None
        assert item.unwrap() is sr
        with pytest.raises(AttributeError):
            item.result = None  # type: ignore[misc]

    def test_error_result_unwrap_raises_exact_exception(self):
        err = MilvusException(code=10001, message="timeout")
        item = DQLResult(error=err)
        assert item.ok is False
        assert item.result is None
        assert item.error is err
        with pytest.raises(MilvusException) as exc_info:
            item.unwrap()
        assert exc_info.value is err
        assert exc_info.value.code == 10001


class TestValidateAndPrepareExecuteDQL:
    """Local validation and ExecuteDQLRequest wire serialization."""

    def test_validate_rejects_empty_and_bad_reqs(self):
        with pytest.raises(ParamError, match="empty"):
            validate_execute_dql_params([])
        with pytest.raises(ParamError, match="sequence"):
            validate_execute_dql_params("not-a-seq")
        with pytest.raises(ParamError, match="DQLRequest"):
            validate_execute_dql_params([object()])

        class _UnsupportedRequest(DQLRequest):
            op = "query"

        with pytest.raises(ParamError, match="Unsupported DQL operation"):
            validate_execute_dql_params([_UnsupportedRequest()])

    def test_validate_rejects_batch_db_name(self):
        req = DQL.search("papers", data=[[0.1, 0.2]])
        with pytest.raises(ParamError, match="db_name"):
            validate_execute_dql_params([req], batch_kwargs={"db_name": "archive"})
        with pytest.raises(ParamError, match="db_name"):
            validate_execute_dql_params([req], batch_kwargs={"db_name": None})

    @pytest.mark.parametrize("bad_conc", [-1, True, False, 1.5, "4", None, 2**32])
    def test_validate_rejects_invalid_max_concurrency(self, bad_conc):
        req = DQL.search("papers", data=[[0.1, 0.2]])
        with pytest.raises(ParamError, match="max_concurrency"):
            validate_execute_dql_params([req], max_concurrency=bad_conc)

    @pytest.mark.parametrize(
        "bad_timeout",
        [-0.001, -5, math.nan, math.inf, -math.inf, True, "5", 10_000_000_000_000.0],
    )
    def test_validate_rejects_invalid_timeout(self, bad_timeout):
        req = DQL.search("papers", data=[[0.1, 0.2]])
        with pytest.raises(ParamError, match="timeout"):
            validate_execute_dql_params([req], timeout=bad_timeout)

    def test_prepare_execute_dql_request_timeout_semantics(self):
        req = DQL.search("c1", data=[[0.1, 0.2]])
        assert validate_execute_dql_params([req], max_concurrency=0, timeout=None) == (0, None)
        assert validate_execute_dql_params([req], max_concurrency=4, timeout=0) == (4, 0)
        assert validate_execute_dql_params([req], max_concurrency=2, timeout=0.0001) == (2, 1)
        assert validate_execute_dql_params([req], max_concurrency=2, timeout=2.5) == (2, 2500)

        sr = milvus_pb2.DQLRequest(search=milvus_pb2.SearchRequest(collection_name="c1"))

        req_none = Prepare.execute_dql_request([sr], max_concurrency=0, timeout_ms=None)
        assert req_none.max_concurrency == 0
        assert not req_none.HasField("timeout_ms")
        assert len(req_none.requests) == 1

        req_zero = Prepare.execute_dql_request([sr], max_concurrency=4, timeout_ms=0)
        assert req_zero.max_concurrency == 4
        assert req_zero.HasField("timeout_ms")
        assert req_zero.timeout_ms == 0

        req_small = Prepare.execute_dql_request([sr], max_concurrency=2, timeout_ms=1)
        assert req_small.HasField("timeout_ms")
        assert req_small.timeout_ms == 1

        req_sec = Prepare.execute_dql_request([sr], max_concurrency=2, timeout_ms=2500)
        assert req_sec.timeout_ms == 2500


class TestUnpackExecuteDQLResponse:
    """unpack_execute_dql_response preserves order, round_decimal, session_ts, and status authority."""

    def test_mixed_success_failure_and_order_preservation(self):
        sr1 = milvus_pb2.SearchResults(
            status=_ok_status({"report_value": "7"}),
            results=_make_search_result_data([10, 20], [0.12345, 0.98765]),
            session_ts=4242,
        )
        sr3 = milvus_pb2.SearchResults(
            status=_ok_status(),
            results=_make_search_result_data([], [], topks=[0]),
            session_ts=99,
        )
        resp = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[
                milvus_pb2.DQLResult(status=_ok_status(), search=sr1),
                milvus_pb2.DQLResult(status=_err_status(10001, "execution budget expired")),
                milvus_pb2.DQLResult(
                    status=_ok_status({"report_value": "3"}),
                    search=sr3,
                ),
            ],
        )

        unpacked = unpack_execute_dql_response(resp, expected_count=3, round_decimals=[2, -1, -1])
        assert len(unpacked) == 3

        assert unpacked[0].ok is True
        assert isinstance(unpacked[0].result, SearchResult)
        assert unpacked[0].result[0].ids == [10, 20]
        assert unpacked[0].result[0].distances == [0.12, 0.99]
        assert unpacked[0].result.extra == {"cost": 7}
        assert unpacked[0].result.get_session_ts() == 4242

        assert unpacked[1].ok is False
        assert unpacked[1].result is None
        assert unpacked[1].error.code == 10001
        assert "execution budget expired" in unpacked[1].error.message

        assert unpacked[2].ok is True
        assert len(unpacked[2].result[0]) == 0
        assert unpacked[2].result.extra == {"cost": 3}
        assert unpacked[2].result.get_session_ts() == 99

    def test_length_mismatch_raises(self):
        resp = milvus_pb2.ExecuteDQLResponse(status=_ok_status(), results=[])
        with pytest.raises(MilvusException, match="mismatch"):
            unpack_execute_dql_response(resp, expected_count=2, round_decimals=[-1, -1])

    def test_ok_status_without_search_payload_becomes_item_error(self):
        resp = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[milvus_pb2.DQLResult(status=_ok_status())],
        )
        unpacked = unpack_execute_dql_response(resp, expected_count=1, round_decimals=[-1])
        assert len(unpacked) == 1
        assert unpacked[0].ok is False
        assert "without a search result payload" in unpacked[0].error.message


class _FakeRpcError(grpc.RpcError):
    def __init__(self, status_code, details_str="rpc error"):
        super().__init__()
        self._code = status_code
        self._details = details_str

    def code(self):
        return self._code

    def details(self):
        return self._details


def _make_sync_handler():
    handler = GrpcHandler.__new__(GrpcHandler)
    handler._channel = MagicMock()
    handler._address = "localhost:19530"
    handler._stub = MagicMock()
    handler._get_schema = MagicMock()
    handler.set_onetime_loglevel = MagicMock()
    return handler


def _make_async_handler():
    handler = AsyncGrpcHandler.__new__(AsyncGrpcHandler)
    handler._channel = MagicMock()
    handler._address = "localhost:19530"
    handler._async_stub = MagicMock()
    handler._async_stub.ExecuteDQL = AsyncMock()
    handler._async_stub.Search = AsyncMock()
    handler._get_schema = AsyncMock()
    handler.set_onetime_loglevel = MagicMock()
    return handler


def _extract_search_param_map(search_req: milvus_pb2.SearchRequest) -> dict:
    return {kv.key: kv.value for kv in search_req.search_params}


class TestClientAndSessionSignatures:
    """MilvusClient, AsyncMilvusClient, and both Sessions expose isomorphic execute_dql signatures."""

    @pytest.mark.parametrize(
        "cls",
        [MilvusClient, AsyncMilvusClient, MilvusClientSession, AsyncMilvusClientSession],
    )
    def test_isomorphic_signature_and_keyword_only_controls(self, cls):
        assert not hasattr(cls, "multi_search")
        sig = inspect.signature(cls.execute_dql)
        params = list(sig.parameters.values())
        assert [p.name for p in params] == [
            "self",
            "reqs",
            "max_concurrency",
            "timeout",
            "kwargs",
        ]
        assert params[1].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params[2].kind == inspect.Parameter.KEYWORD_ONLY
        assert params[2].default == 0
        assert params[3].kind == inspect.Parameter.KEYWORD_ONLY
        assert params[3].default is None
        assert params[4].kind == inspect.Parameter.VAR_KEYWORD

    def test_sync_client_rejects_positional_max_concurrency_and_batch_db_name(self):
        client = MilvusClient.__new__(MilvusClient)
        client._config = MagicMock(db_name="app")
        client._cluster_id = ""
        handler = _make_sync_handler()
        client._get_connection = MagicMock(return_value=handler)

        req = DQL.search("papers", data=[[0.1, 0.2]])
        with pytest.raises(TypeError):
            client.execute_dql([req], 4)  # type: ignore[misc]
        with pytest.raises(ParamError, match="db_name"):
            client.execute_dql([req], db_name="archive")
        handler._stub.ExecuteDQL.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_client_rejects_positional_and_batch_db_name(self):
        client = AsyncMilvusClient.__new__(AsyncMilvusClient)
        client._config = MagicMock(db_name="app")
        client._cluster_id = ""
        handler = _make_async_handler()
        client._get_connection = AsyncMock(return_value=handler)

        req = DQL.search("papers", data=[[0.1, 0.2]])
        with pytest.raises(TypeError):
            await client.execute_dql([req], 4)  # type: ignore[misc]
        with pytest.raises(ParamError, match="db_name"):
            await client.execute_dql([req], db_name="archive")
        handler._async_stub.ExecuteDQL.assert_not_awaited()

    def test_sync_session_forwards_cluster_id_and_enforces_open(self):
        parent = MagicMock()
        parent.execute_dql.return_value = [DQLResult(result="ok")]
        session = MilvusClientSession(parent, "cluster-sync-1")

        req = DQL.search("papers", data=[[0.1, 0.2]])
        out = session.execute_dql([req], max_concurrency=2, timeout=3.0)
        assert out[0].unwrap() == "ok"
        parent.execute_dql.assert_called_once_with(
            [req], max_concurrency=2, timeout=3.0, cluster_id="cluster-sync-1"
        )

        session.close()
        with pytest.raises(MilvusException, match="session is closed"):
            session.execute_dql([req])

    def test_sync_client_forwards_client_request_id_in_call_context(self):
        client = MilvusClient.__new__(MilvusClient)
        client._config = MagicMock(db_name="app")
        client._cluster_id = ""
        mock_conn = MagicMock()
        mock_conn.execute_dql.return_value = []
        client._get_connection = MagicMock(return_value=mock_conn)

        req = DQL.search("papers", data=[[0.1, 0.2]])
        client.execute_dql([req], client_request_id="cid-sync-42")
        ctx = mock_conn.execute_dql.call_args.kwargs["context"]
        md = dict(ctx.to_grpc_metadata())
        assert md["dbname"] == "app"
        assert md["client-request-id"] == "cid-sync-42"

    @pytest.mark.asyncio
    async def test_async_client_forwards_client_request_id_in_call_context(self):
        client = AsyncMilvusClient.__new__(AsyncMilvusClient)
        client._config = MagicMock(db_name="app")
        client._cluster_id = ""
        mock_conn = MagicMock()
        mock_conn.execute_dql = AsyncMock(return_value=[])
        client._get_connection = AsyncMock(return_value=mock_conn)

        req = DQL.search("papers", data=[[0.1, 0.2]])
        await client.execute_dql([req], client_request_id="cid-async-42")
        ctx = mock_conn.execute_dql.call_args.kwargs["context"]
        md = dict(ctx.to_grpc_metadata())
        assert md["dbname"] == "app"
        assert md["client-request-id"] == "cid-async-42"

    def test_batch_cluster_id_and_rls_override_item_values(self):
        batch = prepare_execute_dql_batch(
            [DQL.search("papers", data=[[0.1, 0.2]], cluster_id="item-cls", rls_principal="bob")],
            max_concurrency=0,
            timeout=None,
            server_address="localhost:19530",
            context=None,
            batch_kwargs={"cluster_id": "session-cls", "rls_principal": "alice", "skip_rls": False},
        )
        prepared = batch.items[0]
        assert prepared.extra_kwargs["cluster_id"] == "session-cls"
        assert prepared.extra_kwargs["rls_principal"] == "alice"
        assert prepared.extra_kwargs["skip_rls"] is False

    @pytest.mark.asyncio
    async def test_async_session_forwards_cluster_id_and_enforces_open(self):
        parent = MagicMock()
        parent.execute_dql = AsyncMock(return_value=[DQLResult(result="ok")])
        session = AsyncMilvusClientSession(parent, "cluster-async-1")

        req = DQL.search("papers", data=[[0.1, 0.2]])
        out = await session.execute_dql([req], max_concurrency=3, timeout=4.0)
        assert out[0].unwrap() == "ok"
        parent.execute_dql.assert_awaited_once_with(
            [req], max_concurrency=3, timeout=4.0, cluster_id="cluster-async-1"
        )

        await session.close()
        with pytest.raises(MilvusException, match="session is closed"):
            await session.execute_dql([req])


class TestSharedPrepareAndLocalValidation:
    """Shared search preparation between single search() and execute_dql()."""

    def test_single_search_and_execute_dql_produce_matching_search_request(self):
        """Regression: single GrpcHandler.search and execute_dql share identical preparation."""
        GlobalCache._reset_for_testing()
        handler = _make_sync_handler()
        ctx = CallContext(db_name="app", client_request_id="req-1")

        sr_proto = milvus_pb2.SearchResults(
            status=_ok_status(),
            results=_make_search_result_data([1], [0.5]),
            session_ts=10,
        )
        handler._stub.Search.return_value = sr_proto
        handler._stub.ExecuteDQL.return_value = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[milvus_pb2.DQLResult(status=_ok_status(), search=sr_proto)],
        )

        emb = EmbeddingList()
        emb.add([0.1, 0.2])
        agg = SearchAggregation(fields=["category"], size=5)

        handler.search(
            collection_name="docs",
            anns_field="emb",
            param={"metric_type": "COSINE"},
            limit=7,
            data=[emb],
            expression="year >= {y}",
            partition_names=["p1"],
            output_fields=["title"],
            round_decimal=2,
            expr_params={"y": 2024},
            search_aggregation=agg,
            context=ctx,
        )
        single_req = handler._stub.Search.call_args.args[0]

        handler.execute_dql(
            [
                DQL.search(
                    "docs",
                    data=[emb],
                    anns_field="emb",
                    search_params={"metric_type": "COSINE"},
                    limit=7,
                    filter="year >= {y}",
                    partition_names=["p1"],
                    output_fields=["title"],
                    search_aggregation=agg,
                    round_decimal=2,
                    filter_params={"y": 2024},
                )
            ],
            context=ctx,
        )
        batch_req = handler._stub.ExecuteDQL.call_args.args[0].requests[0].search

        assert batch_req.SerializeToString() == single_req.SerializeToString()

    def test_two_phase_validation_prevents_schema_or_dql_rpc_on_later_item_error(self):
        handler = _make_sync_handler()
        reqs = [
            DQL.search("col_bytes", data=[b"\x01\x02\x03\x04"], anns_field="vec"),
            DQL.search("col_bad", data=[[0.1, 0.2]], limit=-1),
        ]
        with pytest.raises(ParamError):
            handler.execute_dql(reqs)
        handler._get_schema.assert_not_called()
        handler._stub.ExecuteDQL.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_two_phase_validation_prevents_schema_or_dql_rpc(self):
        handler = _make_async_handler()
        reqs = [
            DQL.search("col_bytes", data=[b"\x01\x02\x03\x04"], anns_field="vec"),
            DQL.search("col_bad", data=None, ids=None),
        ]
        with pytest.raises(ParamError):
            await handler.execute_dql(reqs)
        handler._get_schema.assert_not_awaited()
        handler._async_stub.ExecuteDQL.assert_not_awaited()


class TestSyncExecuteDQLHandler:
    """Sync GrpcHandler.execute_dql end-to-end behavior, DB isolation, budget, and retries."""

    def test_cross_db_schema_and_session_ts_isolation_and_dedup(self):
        GlobalCache._reset_for_testing()
        handler = _make_sync_handler()
        # Different session timestamps for same collection name across two DBs
        GlobalCache.collection_ts.set("localhost:19530", "app", "shared_col", 1111)
        GlobalCache.collection_ts.set("localhost:19530", "archive", "shared_col", 2222)

        schema_app = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "vec", "type": DataType.FLOAT16_VECTOR},
            ]
        }
        schema_archive = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "vec", "type": DataType.BFLOAT16_VECTOR},
            ]
        }

        def fake_get_schema(collection_name, timeout=None, context=None, **kwargs):
            assert timeout is None
            db = context.get_db_name() if context else ""
            return (schema_archive if db == "archive" else schema_app), 0

        handler._get_schema.side_effect = fake_get_schema
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([1], [0.1])
        )
        handler._stub.ExecuteDQL.return_value = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[
                milvus_pb2.DQLResult(status=_ok_status(), search=sr),
                milvus_pb2.DQLResult(status=_ok_status(), search=sr),
                milvus_pb2.DQLResult(status=_ok_status(), search=sr),
            ],
        )

        ctx = CallContext(db_name="app", client_request_id="cid-99")
        reqs = [
            DQL.search("shared_col", data=[b"\x01\x02\x03\x04"], anns_field="vec"),
            DQL.search(
                "shared_col", data=[b"\x01\x02\x03\x04"], anns_field="vec", db_name="archive"
            ),
            # Duplicate (effective_db="app", "shared_col") should reuse local_schemas
            DQL.search("shared_col", data=[b"\x01\x02\x03\x04"], anns_field="vec"),
        ]

        out = handler.execute_dql(
            reqs,
            max_concurrency=4,
            timeout=5.0,
            context=ctx,
            cluster_id="cls-1",
            rls_principal="alice",
        )
        assert len(out) == 3
        assert all(item.ok for item in out)

        # Schema fetched once per distinct (effective_db, collection_name)
        assert handler._get_schema.call_count == 2
        fetched_dbs = [
            c.kwargs["context"].get_db_name() for c in handler._get_schema.call_args_list
        ]
        fetched_cids = [
            dict(c.kwargs["context"].to_grpc_metadata())["client-request-id"]
            for c in handler._get_schema.call_args_list
        ]
        assert fetched_dbs == ["app", "archive"]
        assert fetched_cids == ["cid-99", "cid-99"]

        wire_req = handler._stub.ExecuteDQL.call_args.args[0]
        assert wire_req.max_concurrency == 4
        assert wire_req.timeout_ms == 5000
        # Default-db items keep db_name empty so server resolves RPC metadata dbname
        assert wire_req.requests[0].search.db_name == ""
        assert wire_req.requests[1].search.db_name == "archive"
        assert wire_req.requests[2].search.db_name == ""
        # Isolated session ts per DB
        assert wire_req.requests[0].search.guarantee_timestamp == 1111
        assert wire_req.requests[1].search.guarantee_timestamp == 2222
        assert wire_req.requests[2].search.guarantee_timestamp == 1111
        # Batch cluster_id propagated
        assert _extract_search_param_map(wire_req.requests[0].search)["cluster_id"] == "cls-1"
        assert _extract_search_param_map(wire_req.requests[1].search)["cluster_id"] == "cls-1"
        # RPC metadata retains client's default db
        rpc_md = dict(handler._stub.ExecuteDQL.call_args.kwargs["metadata"])
        assert rpc_md["dbname"] == "app"
        assert rpc_md["client-request-id"] == "cid-99"

    def test_timeout_is_execution_budget_not_grpc_deadline(self):
        handler = _make_sync_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([7], [0.2])
        )

        def slow_execute_dql(request, **kwargs):
            # Must NOT set gRPC transport deadline
            assert "timeout" not in kwargs
            assert request.HasField("timeout_ms")
            assert request.timeout_ms == 10
            time.sleep(0.03)
            return milvus_pb2.ExecuteDQLResponse(
                status=_ok_status(),
                results=[
                    milvus_pb2.DQLResult(status=_ok_status(), search=sr),
                    milvus_pb2.DQLResult(status=_err_status(10001, "timeout")),
                ],
            )

        handler._stub.ExecuteDQL.side_effect = slow_execute_dql
        reqs = [
            DQL.search("c1", data=[[0.1, 0.2]]),
            DQL.search("c2", data=[[0.3, 0.4]]),
        ]
        res = handler.execute_dql(reqs, timeout=0.01)
        assert res[0].ok is True
        assert res[0].unwrap()[0].ids == [7]
        assert res[1].ok is False
        assert res[1].error.code == 10001

    @pytest.mark.parametrize("code,reason", [(1102, "batch too large"), (3002, "request canceled")])
    def test_non_retriable_1102_and_3002_do_not_retry(self, code, reason):
        handler = _make_sync_handler()
        handler._stub.ExecuteDQL.return_value = milvus_pb2.ExecuteDQLResponse(
            status=_err_status(code, reason, retriable=False),
        )
        with pytest.raises(MilvusException) as exc_info:
            handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])], retry_times=3)
        assert exc_info.value.code == code
        assert handler._stub.ExecuteDQL.call_count == 1

    def test_unimplemented_propagates_rpc_error_without_retry(self):
        handler = _make_sync_handler()
        handler._stub.ExecuteDQL.side_effect = _FakeRpcError(
            grpc.StatusCode.UNIMPLEMENTED, "unknown service method ExecuteDQL"
        )
        with pytest.raises(grpc.RpcError) as exc_info:
            handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])], retry_times=3)
        assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
        assert handler._stub.ExecuteDQL.call_count == 1

    def test_bytes_vector_without_anns_field_skips_schema_fetch(self):
        handler = _make_sync_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([1], [0.1])
        )
        handler._stub.ExecuteDQL.return_value = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[milvus_pb2.DQLResult(status=_ok_status(), search=sr)],
        )
        handler._stub.Search.return_value = sr

        handler.execute_dql([DQL.search("c1", data=[b"\x01\x02\x03\x04"])])
        handler._get_schema.assert_not_called()

        handler.search(
            collection_name="c1",
            anns_field="",
            param={"metric_type": "L2"},
            limit=5,
            data=[b"\x01\x02\x03\x04"],
        )
        handler._get_schema.assert_not_called()

    def test_batch_kwargs_passthrough_and_error_handler_wrapping(self):
        handler = _make_sync_handler()
        schema = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "vec", "type": DataType.FLOAT16_VECTOR},
            ]
        }
        handler._get_schema.return_value = (schema, 0)
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([1], [0.1])
        )
        handler._stub.ExecuteDQL.return_value = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[milvus_pb2.DQLResult(status=_ok_status(), search=sr)],
        )

        handler.execute_dql(
            [DQL.search("c1", data=[b"\x01\x02\x03\x04"], anns_field="vec")],
            timeout=2.0,
            retry_times=2,
            log_level="info",
            custom_decorator_opt="passthrough",
        )
        schema_kwargs = handler._get_schema.call_args.kwargs
        assert schema_kwargs["timeout"] is None
        assert schema_kwargs["retry_times"] == 2
        assert schema_kwargs["log_level"] == "info"
        assert schema_kwargs["custom_decorator_opt"] == "passthrough"

        # Invalid RLS batch kwargs fail locally via Prepare._rls_context before RPC
        handler._stub.ExecuteDQL.reset_mock()
        with pytest.raises(ParamError, match="rls_principal"):
            handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])], rls_principal=123)
        handler._stub.ExecuteDQL.assert_not_called()

        # Unexpected exception wrapped by @error_handler(func_name="execute_dql")
        handler._get_schema.side_effect = RuntimeError("unexpected boom")
        with pytest.raises(MilvusException, match="unexpected boom"):
            handler.execute_dql([DQL.search("c1", data=[b"\x01\x02\x03\x04"], anns_field="vec")])

    def test_partial_or_all_item_failures_never_resend_batch(self):
        handler = _make_sync_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([1], [0.1])
        )
        handler._stub.ExecuteDQL.return_value = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[
                milvus_pb2.DQLResult(status=_ok_status(), search=sr),
                milvus_pb2.DQLResult(
                    status=_err_status(
                        ErrorCode.RATE_LIMIT,
                        "rate limited",
                        error_code=common_pb2.RateLimit,
                        retriable=True,
                    )
                ),
            ],
        )
        out = handler.execute_dql(
            [DQL.search("c1", data=[[0.1, 0.2]]), DQL.search("c2", data=[[0.3, 0.4]])],
            retry_times=5,
            retry_on_rate_limit=True,
        )
        assert handler._stub.ExecuteDQL.call_count == 1
        assert out[0].ok is True
        assert out[1].ok is False
        assert out[1].error.code == ErrorCode.RATE_LIMIT

    def test_none_response_raises_milvus_exception(self):
        handler = _make_sync_handler()
        handler._stub.ExecuteDQL.return_value = None
        with pytest.raises(MilvusException, match="None response"):
            handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])])
        assert handler._stub.ExecuteDQL.call_count == 1

    def test_outer_rate_limit_retries_when_enabled_and_stops_when_disabled(self):
        handler = _make_sync_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([1], [0.1])
        )
        rate_limited = milvus_pb2.ExecuteDQLResponse(
            status=_err_status(
                ErrorCode.RATE_LIMIT, "queue full", error_code=common_pb2.RateLimit, retriable=True
            )
        )
        ok_resp = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[milvus_pb2.DQLResult(status=_ok_status(), search=sr)],
        )
        handler._stub.ExecuteDQL.side_effect = [rate_limited, ok_resp]
        with patch("pymilvus.decorators.time.sleep"):
            out = handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])], retry_times=3)
        assert len(out) == 1
        assert out[0].ok is True
        assert handler._stub.ExecuteDQL.call_count == 2

        handler._stub.ExecuteDQL.reset_mock()
        handler._stub.ExecuteDQL.side_effect = [rate_limited]
        with pytest.raises(MilvusException) as exc_info:
            handler.execute_dql(
                [DQL.search("c1", data=[[0.1, 0.2]])],
                retry_times=3,
                retry_on_rate_limit=False,
            )
        assert exc_info.value.code == ErrorCode.RATE_LIMIT
        assert handler._stub.ExecuteDQL.call_count == 1

    def test_transient_unavailable_retries_and_recovers(self):
        handler = _make_sync_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([5], [0.2])
        )
        ok_resp = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[milvus_pb2.DQLResult(status=_ok_status(), search=sr)],
        )
        handler._stub.ExecuteDQL.side_effect = [
            _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "transient"),
            ok_resp,
        ]
        with patch("pymilvus.decorators.time.sleep"):
            out = handler.execute_dql([DQL.search("c1", ids=123)], retry_times=3, log_level="debug")
        assert out[0].unwrap()[0].ids == [5]
        assert handler._stub.ExecuteDQL.call_count == 2
        handler.set_onetime_loglevel.assert_called_with("debug")


class TestAsyncExecuteDQLHandler:
    """Async AsyncGrpcHandler.execute_dql end-to-end behavior, DB isolation, budget, and retries."""

    @pytest.mark.asyncio
    async def test_async_cross_db_isolation_and_dedup(self):
        GlobalCache._reset_for_testing()
        handler = _make_async_handler()
        GlobalCache.collection_ts.set("localhost:19530", "", "shared_col", 3333)
        GlobalCache.collection_ts.set("localhost:19530", "analytics", "shared_col", 4444)

        schema = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "vec", "type": DataType.FLOAT16_VECTOR},
            ]
        }
        handler._get_schema = AsyncMock(return_value=(schema, 0))
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data(["k1"], [0.4])
        )
        handler._async_stub.ExecuteDQL = AsyncMock(
            return_value=milvus_pb2.ExecuteDQLResponse(
                status=_ok_status(),
                results=[
                    milvus_pb2.DQLResult(status=_ok_status(), search=sr),
                    milvus_pb2.DQLResult(status=_ok_status(), search=sr),
                ],
            )
        )

        reqs = [
            DQL.search("shared_col", data=[b"\x01\x02\x03\x04"], anns_field="vec"),
            DQL.search(
                "shared_col", data=[b"\x01\x02\x03\x04"], anns_field="vec", db_name="analytics"
            ),
        ]
        out = await handler.execute_dql(reqs, max_concurrency=2, timeout=0)
        assert len(out) == 2
        assert out[0].unwrap()[0].ids == ["k1"]
        assert handler._get_schema.await_count == 2

        wire_req = handler._async_stub.ExecuteDQL.call_args.args[0]
        assert wire_req.max_concurrency == 2
        assert wire_req.HasField("timeout_ms")
        assert wire_req.timeout_ms == 0
        assert wire_req.requests[0].search.db_name == ""
        assert wire_req.requests[1].search.db_name == "analytics"
        assert wire_req.requests[0].search.guarantee_timestamp == 3333
        assert wire_req.requests[1].search.guarantee_timestamp == 4444

    @pytest.mark.asyncio
    async def test_async_timeout_not_enforced_by_asyncio_wait_for_or_grpc_deadline(self):
        handler = _make_async_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([9], [0.1])
        )

        async def slow_async_execute_dql(request, **kwargs):
            assert "timeout" not in kwargs
            assert request.timeout_ms == 10
            await asyncio.sleep(0.04)
            return milvus_pb2.ExecuteDQLResponse(
                status=_ok_status(),
                results=[
                    milvus_pb2.DQLResult(status=_ok_status(), search=sr),
                    milvus_pb2.DQLResult(status=_err_status(10001, "budget expired")),
                ],
            )

        handler._async_stub.ExecuteDQL = AsyncMock(side_effect=slow_async_execute_dql)
        out = await handler.execute_dql(
            [DQL.search("c1", data=[[0.1, 0.2]]), DQL.search("c2", data=[[0.3, 0.4]])],
            timeout=0.01,
        )
        assert out[0].ok is True
        assert out[0].unwrap()[0].ids == [9]
        assert out[1].ok is False
        assert out[1].error.code == 10001

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code,reason", [(1102, "result too large"), (3002, "canceled")])
    async def test_async_non_retriable_1102_and_3002_do_not_retry(self, code, reason):
        handler = _make_async_handler()
        handler._async_stub.ExecuteDQL = AsyncMock(
            return_value=milvus_pb2.ExecuteDQLResponse(
                status=_err_status(code, reason, retriable=False)
            )
        )
        with pytest.raises(MilvusException) as exc_info:
            await handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])], retry_times=3)
        assert exc_info.value.code == code
        assert handler._async_stub.ExecuteDQL.await_count == 1

    @pytest.mark.asyncio
    async def test_async_unimplemented_propagates_rpc_error_without_retry(self):
        handler = _make_async_handler()
        handler._async_stub.ExecuteDQL = AsyncMock(
            side_effect=_FakeRpcError(grpc.StatusCode.UNIMPLEMENTED, "unimplemented")
        )
        with pytest.raises(grpc.RpcError) as exc_info:
            await handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])], retry_times=3)
        assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
        assert handler._async_stub.ExecuteDQL.await_count == 1

    @pytest.mark.asyncio
    async def test_async_bytes_vector_without_anns_field_skips_schema_fetch(self):
        handler = _make_async_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([1], [0.1])
        )
        handler._async_stub.ExecuteDQL = AsyncMock(
            return_value=milvus_pb2.ExecuteDQLResponse(
                status=_ok_status(),
                results=[milvus_pb2.DQLResult(status=_ok_status(), search=sr)],
            )
        )
        await handler.execute_dql([DQL.search("c1", data=[b"\x01\x02\x03\x04"])])
        handler._get_schema.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_async_error_handler_wraps_unexpected_exception(self):
        handler = _make_async_handler()
        handler._get_schema = AsyncMock(side_effect=RuntimeError("async boom"))
        with pytest.raises(MilvusException, match="async boom"):
            await handler.execute_dql(
                [DQL.search("c1", data=[b"\x01\x02\x03\x04"], anns_field="vec")]
            )

    @pytest.mark.asyncio
    async def test_async_partial_failures_never_resend_batch(self):
        handler = _make_async_handler()
        handler._async_stub.ExecuteDQL = AsyncMock(
            return_value=milvus_pb2.ExecuteDQLResponse(
                status=_ok_status(),
                results=[
                    milvus_pb2.DQLResult(
                        status=_err_status(
                            ErrorCode.RATE_LIMIT,
                            "rate limit",
                            error_code=common_pb2.RateLimit,
                            retriable=True,
                        )
                    ),
                ],
            )
        )
        out = await handler.execute_dql(
            [DQL.search("c1", data=[[0.1, 0.2]])],
            retry_times=4,
            retry_on_rate_limit=True,
        )
        assert handler._async_stub.ExecuteDQL.await_count == 1
        assert out[0].ok is False
        assert out[0].error.code == ErrorCode.RATE_LIMIT

    @pytest.mark.asyncio
    async def test_async_none_response_raises_milvus_exception(self):
        handler = _make_async_handler()
        handler._async_stub.ExecuteDQL = AsyncMock(return_value=None)
        with pytest.raises(MilvusException, match="None response"):
            await handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])])
        assert handler._async_stub.ExecuteDQL.await_count == 1

    @pytest.mark.asyncio
    async def test_async_transient_unavailable_retries_and_recovers(self):
        handler = _make_async_handler()
        sr = milvus_pb2.SearchResults(
            status=_ok_status(), results=_make_search_result_data([3], [0.5])
        )
        ok_resp = milvus_pb2.ExecuteDQLResponse(
            status=_ok_status(),
            results=[milvus_pb2.DQLResult(status=_ok_status(), search=sr)],
        )
        handler._async_stub.ExecuteDQL = AsyncMock(
            side_effect=[_FakeRpcError(grpc.StatusCode.UNAVAILABLE, "transient"), ok_resp]
        )
        with patch("pymilvus.decorators.asyncio.sleep", new_callable=AsyncMock):
            out = await handler.execute_dql([DQL.search("c1", data=[[0.1, 0.2]])], retry_times=3)
        assert out[0].unwrap()[0].ids == [3]
        assert handler._async_stub.ExecuteDQL.await_count == 2
