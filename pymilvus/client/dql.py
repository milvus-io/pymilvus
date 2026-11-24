"""ExecuteDQL request/result models, ``DQL`` constructors, and shared search preparation."""

import logging
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import grpc

from pymilvus.exceptions import (
    MilvusException,
    ParamError,
    ServerVersionIncompatibleException,
)
from pymilvus.function_chain import FunctionChain
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.orm.schema import Function, FunctionScore, Highlighter

from . import entity_helper, ts_utils
from .call_context import CallContext
from .check import check_id_and_data, check_pass_param
from .constants import CLUSTER_ID
from .embedding_list import EmbeddingList
from .prepare import Prepare
from .search_aggregation import SearchAggregation
from .search_result import SearchResult
from .utils import check_status, is_successful

logger = logging.getLogger(__name__)

SUPPORTED_DQL_OPS = frozenset({"search"})
_BATCH_CONTEXT_KEYS = (CLUSTER_ID, "rls_principal", "skip_rls")
_EXECUTE_DQL_RPC_KWARG_KEYS = ("retry_times", "retry_on_rate_limit", "log_level", "log-level")
_UINT32_MAX = 2**32 - 1
# Maximum non-negative execution budget in milliseconds supported by Go time.Duration
# (math.MaxInt64 / time.Millisecond).
_MAX_EXECUTE_DQL_TIMEOUT_MS = 9_223_372_036_854


@dataclass(frozen=True)
class DQLRequest:
    """Base class for immutable read-only DQL sub-requests created by ``DQL.*`` constructors."""

    op: ClassVar[str] = ""


@dataclass(frozen=True)
class _SearchRequest(DQLRequest):
    """Immutable search sub-request returned by ``DQL.search``."""

    op: ClassVar[str] = "search"

    collection_name: str
    data: Any = None
    filter: str = ""
    limit: int = 10
    output_fields: Optional[List[str]] = None
    search_params: Optional[Dict[str, Any]] = None
    partition_names: Optional[List[str]] = None
    anns_field: Optional[str] = None
    ranker: Optional[Union[Function, FunctionScore]] = None
    function_chains: Optional[Union[FunctionChain, List[FunctionChain]]] = None
    highlighter: Optional[Highlighter] = None
    ids: Optional[Union[List[int], List[str], str, int]] = None
    search_aggregation: Optional[SearchAggregation] = None
    db_name: Optional[str] = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_kwargs", MappingProxyType(dict(self.extra_kwargs)))


@dataclass(frozen=True)
class DQLResult:
    """Per-item outcome returned by ``execute_dql``."""

    result: Optional[Any] = None
    error: Optional[MilvusException] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.result


class DQL:
    """Namespace of read-only DQL sub-request constructors for ``execute_dql``."""

    @staticmethod
    def search(
        collection_name: str,
        *,
        data: Optional[Union[List[list], list]] = None,
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        ranker: Optional[Union[Function, FunctionScore]] = None,
        function_chains: Optional[Union[FunctionChain, List[FunctionChain]]] = None,
        highlighter: Optional[Highlighter] = None,
        ids: Optional[Union[List[int], List[str], str, int]] = None,
        search_aggregation: Optional[SearchAggregation] = None,
        db_name: Optional[str] = None,
        **kwargs,
    ) -> DQLRequest:
        """Build a search sub-request for ``execute_dql``.

        Parameters mirror ``MilvusClient.search`` (as keyword-only arguments after
        ``collection_name``), with an explicit per-request ``db_name`` and without
        per-request ``timeout`` (execution budget is configured at the ``execute_dql``
        batch level).

        Args:
            collection_name (str): Target collection name.
            data (Optional[Union[List[list], list]]): Query vectors or embedding lists.
            filter (str, optional): Boolean filter expression. Defaults to ``""``.
            limit (int, optional): Top-K result count per query vector. Defaults to ``10``.
            output_fields (List[str], optional): Fields to return in each hit.
            search_params (dict, optional): Search parameters (e.g. ``metric_type``, ``params``).
            partition_names (List[str], optional): Partitions to restrict the search to.
            anns_field (str, optional): Vector field name to search.
            ranker (Function or FunctionScore, optional): Reranker for the search.
            function_chains (FunctionChain or List[FunctionChain], optional): Function chain(s)
                to apply to the search. Mutually exclusive with ``ranker``.
            highlighter (Highlighter, optional): Highlighting configuration.
            ids (Optional[Union[List[int], List[str], str, int]]): Primary keys for search-by-PK.
            search_aggregation (SearchAggregation, optional): Bucket aggregation spec.
            db_name (str, optional): Per-request database override. When omitted or empty,
                falls back to the client's current database and then the server default.
            **kwargs: Additional search options accepted by ``MilvusClient.search``, such as
                ``filter_params``, ``round_decimal``, ``consistency_level``, ``group_by_field``,
                ``offset``, or ``namespace``. Passing a non-``None`` ``timeout`` raises
                ``ParamError``.

        Returns:
            DQLRequest: Immutable sub-request to pass in ``client.execute_dql([...])``.
        """
        if not isinstance(collection_name, str) or not collection_name:
            raise ParamError(message="collection_name must be a non-empty string")
        if db_name is not None and not isinstance(db_name, str):
            raise ParamError(message=f"db_name must be a str or None, got {type(db_name).__name__}")

        extra_kwargs = dict(kwargs)
        timeout = extra_kwargs.pop("timeout", None)
        if timeout is not None:
            raise ParamError(
                message=(
                    "Per-request 'timeout' is not supported in DQL.search(); "
                    "pass 'timeout' to execute_dql() instead."
                )
            )
        if "param" in extra_kwargs:
            raise ParamError(message="Use 'search_params' instead of 'param' in DQL.search().")
        if "use_default_consistency" in extra_kwargs:
            raise ParamError(
                message=(
                    "'use_default_consistency' is managed automatically based on consistency_level."
                )
            )
        if "filter_params" in extra_kwargs and "expr_params" in extra_kwargs:
            raise ParamError(message="Provide either 'filter_params' or 'expr_params', not both.")

        return _SearchRequest(
            collection_name=collection_name,
            data=data,
            filter=filter,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
            partition_names=partition_names,
            anns_field=anns_field,
            ranker=ranker,
            function_chains=function_chains,
            highlighter=highlighter,
            ids=ids,
            search_aggregation=search_aggregation,
            db_name=db_name,
            extra_kwargs=extra_kwargs,
        )


def check_no_batch_db_name(batch_kwargs: Optional[Mapping[str, Any]]) -> None:
    """Reject batch-level ``db_name`` in ``execute_dql`` kwargs."""
    if batch_kwargs and "db_name" in batch_kwargs:
        raise ParamError(
            message=(
                "Batch-level 'db_name' is not supported in execute_dql(); "
                "specify 'db_name' on individual DQL requests instead."
            )
        )


def _parse_max_concurrency(max_concurrency: int) -> int:
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise ParamError(
            message=(
                f"max_concurrency must be a non-negative int, "
                f"got {type(max_concurrency).__name__}"
            )
        )
    if max_concurrency < 0 or max_concurrency > _UINT32_MAX:
        raise ParamError(
            message=f"max_concurrency must be in [0, {_UINT32_MAX}], got {max_concurrency}"
        )
    return max_concurrency


def _parse_timeout_ms(timeout: Optional[float]) -> Optional[int]:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ParamError(
            message=f"timeout must be a non-negative number or None, got {type(timeout).__name__}"
        )
    if math.isnan(timeout) or math.isinf(timeout):
        raise ParamError(message=f"timeout must be a finite number, got {timeout}")
    if timeout < 0:
        raise ParamError(message=f"timeout must be non-negative, got {timeout}")
    if timeout == 0:
        return 0
    timeout_ms = max(1, round(timeout * 1000))
    if timeout_ms > _MAX_EXECUTE_DQL_TIMEOUT_MS:
        raise ParamError(
            message=(
                f"timeout exceeds maximum supported execution budget "
                f"({_MAX_EXECUTE_DQL_TIMEOUT_MS} ms): {timeout}"
            )
        )
    return timeout_ms


def validate_execute_dql_params(
    reqs: Any,
    max_concurrency: int = 0,
    timeout: Optional[float] = None,
    batch_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, Optional[int]]:
    """Validate batch-level ``execute_dql`` inputs locally before issuing any RPCs."""
    check_no_batch_db_name(batch_kwargs)
    if not isinstance(reqs, Sequence) or isinstance(reqs, (str, bytes, bytearray)):
        raise ParamError(
            message=f"reqs must be a non-empty sequence of DQLRequest, got {type(reqs).__name__}"
        )
    if len(reqs) == 0:
        raise ParamError(message="reqs must not be empty")
    for idx, req in enumerate(reqs):
        if not isinstance(req, DQLRequest):
            raise ParamError(
                message=f"reqs[{idx}] must be a DQLRequest instance, got {type(req).__name__}"
            )
        if not isinstance(req, _SearchRequest):
            supported = ", ".join(sorted(SUPPORTED_DQL_OPS))
            raise ParamError(
                message=(
                    f"Unsupported DQL operation at reqs[{idx}]: {req.op!r}. "
                    f"Supported operations: {supported}"
                )
            )
    return _parse_max_concurrency(max_concurrency), _parse_timeout_ms(timeout)


def normalize_search_inputs(
    data: Any,
    ids: Any,
    *,
    limit: int,
    extra_kwargs: Dict[str, Any],
) -> Tuple[Any, Any, bool]:
    """Normalize search ``data``/``ids``, log aggregation warning, and detect schema need."""
    if isinstance(ids, (int, str)):
        ids = [ids]
    check_id_and_data(ids, data)

    if extra_kwargs.get("search_aggregation") is not None:
        logger.warning(
            "search_aggregation is set; search limit=%s is ignored and bucket count comes from SearchAggregation.size",
            limit,
        )

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], EmbeddingList):
        data = [emb_list.to_flat_array() for emb_list in data]
        extra_kwargs["is_embedding_list"] = True

    needs_schema = (
        not extra_kwargs.get("schema")
        and data is not None
        and not entity_helper.entity_is_sparse_matrix(data)
        and isinstance(data, (list, tuple))
        and len(data) > 0
        and isinstance(data[0], bytes)
    )
    return data, ids, needs_schema


def resolve_search_db_context(
    context: Optional[CallContext],
    db_name: Optional[str] = None,
) -> Tuple[str, Optional[CallContext]]:
    """Resolve effective DB name and per-item ``CallContext`` without mutating parent context."""
    explicit_db = db_name if db_name else None
    if explicit_db:
        item_context = (
            context.with_db_name(explicit_db)
            if context is not None
            else CallContext(db_name=explicit_db)
        )
        return explicit_db, item_context
    return (context.get_db_name() if context else ""), context


@dataclass
class _PreparedSearchItem:
    """Normalized, locally validated search sub-request ready for schema fetch and proto build."""

    req: _SearchRequest
    data: Any
    ids: Any
    round_decimal: int
    use_default_consistency: bool
    effective_db: str
    context: Optional[CallContext]
    needs_schema: bool
    extra_kwargs: Dict[str, Any]

    @property
    def collection_name(self) -> str:
        return self.req.collection_name


@dataclass(frozen=True)
class _PreparedBatch:
    """Validated and normalized ExecuteDQL batch ready for schema fetch and proto assembly."""

    items: Tuple[_PreparedSearchItem, ...]
    schema_targets: Tuple[Tuple[Tuple[str, str], Optional[CallContext]], ...]
    max_concurrency: int
    timeout_ms: Optional[int]
    rpc_kwargs: Mapping[str, Any]

    @property
    def round_decimals(self) -> List[int]:
        return [item.round_decimal for item in self.items]


def _prepare_search_item(
    req: _SearchRequest,
    *,
    server_address: str,
    context: Optional[CallContext],
    batch_kwargs: Mapping[str, Any],
) -> _PreparedSearchItem:
    """Validate and prepare a search sub-request using the shared search preparation helpers."""
    item_kwargs = dict(req.extra_kwargs)

    expr_params = item_kwargs.pop("filter_params", item_kwargs.pop("expr_params", {}))
    item_kwargs["expr_params"] = {} if expr_params is None else expr_params

    for batch_key in _BATCH_CONTEXT_KEYS:
        if batch_key in batch_kwargs:
            item_kwargs[batch_key] = batch_kwargs[batch_key]

    if req.search_aggregation is not None:
        item_kwargs["search_aggregation"] = req.search_aggregation

    round_decimal = item_kwargs.pop("round_decimal", -1)
    data, ids, needs_schema = normalize_search_inputs(
        req.data, req.ids, limit=req.limit, extra_kwargs=item_kwargs
    )
    check_pass_param(
        limit=req.limit,
        round_decimal=round_decimal,
        anns_field=req.anns_field or "",
        partition_name_array=req.partition_names,
        output_fields=req.output_fields,
        guarantee_timestamp=item_kwargs.get("guarantee_timestamp"),
    )
    effective_db, item_context = resolve_search_db_context(context, req.db_name)
    use_default_consistency = ts_utils.construct_guarantee_ts(
        req.collection_name, item_kwargs, server_address, effective_db
    )

    return _PreparedSearchItem(
        req=req,
        data=data,
        ids=ids,
        round_decimal=round_decimal,
        use_default_consistency=use_default_consistency,
        effective_db=effective_db,
        context=item_context,
        needs_schema=needs_schema,
        extra_kwargs=item_kwargs,
    )


def prepare_execute_dql_batch(
    reqs: Sequence[DQLRequest],
    *,
    max_concurrency: int,
    timeout: Optional[float],
    server_address: str,
    context: Optional[CallContext],
    batch_kwargs: Mapping[str, Any],
) -> _PreparedBatch:
    """Run Pass 1 batch/item validation and normalization before any network RPCs."""
    normalized_max_concurrency, timeout_ms = validate_execute_dql_params(
        reqs, max_concurrency=max_concurrency, timeout=timeout, batch_kwargs=batch_kwargs
    )
    rpc_kwargs = {k: batch_kwargs[k] for k in _EXECUTE_DQL_RPC_KWARG_KEYS if k in batch_kwargs}
    prepared_items = tuple(
        _prepare_search_item(
            req,  # type: ignore[arg-type]
            server_address=server_address,
            context=context,
            batch_kwargs=batch_kwargs,
        )
        for req in reqs
    )

    schema_targets: List[Tuple[Tuple[str, str], Optional[CallContext]]] = []
    seen_keys = set()
    for item in prepared_items:
        if item.needs_schema:
            schema_key = (item.effective_db, item.collection_name)
            if schema_key not in seen_keys:
                seen_keys.add(schema_key)
                schema_targets.append((schema_key, item.context))

    return _PreparedBatch(
        items=prepared_items,
        schema_targets=tuple(schema_targets),
        max_concurrency=normalized_max_concurrency,
        timeout_ms=timeout_ms,
        rpc_kwargs=MappingProxyType(rpc_kwargs),
    )


def build_execute_dql_request(
    batch: _PreparedBatch,
    local_schemas: Mapping[Tuple[str, str], dict],
) -> milvus_types.ExecuteDQLRequest:
    """Assemble ``ExecuteDQLRequest`` from a prepared batch and DB-isolated schemas."""
    dql_requests: List[milvus_types.DQLRequest] = []
    for item in batch.items:
        req = item.req
        req_kwargs = dict(item.extra_kwargs)
        if item.needs_schema:
            schema = local_schemas.get((item.effective_db, req.collection_name))
            if schema is not None:
                req_kwargs["schema"] = schema
        search_req = Prepare.search_requests_with_expr(
            collection_name=req.collection_name,
            anns_field=req.anns_field or "",
            param=req.search_params or {},
            limit=req.limit,
            data=item.data,
            ids=item.ids,
            expr=req.filter,
            partition_names=req.partition_names,
            output_fields=req.output_fields,
            round_decimal=item.round_decimal,
            ranker=req.ranker,
            function_chains=req.function_chains,
            highlighter=req.highlighter,
            use_default_consistency=item.use_default_consistency,
            db_name=req.db_name or None,
            **req_kwargs,
        )
        dql_requests.append(milvus_types.DQLRequest(search=search_req))

    return Prepare.execute_dql_request(
        dql_requests,
        max_concurrency=batch.max_concurrency,
        timeout_ms=batch.timeout_ms,
    )


def raise_execute_dql_rpc_error(exc: grpc.RpcError) -> None:
    """Translate ``UNIMPLEMENTED`` to ``ServerVersionIncompatibleException``; re-raise others."""
    if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
        raise ServerVersionIncompatibleException(
            code=exc.code(),
            message=(
                "ExecuteDQL is not supported by this Milvus server "
                "(received gRPC UNIMPLEMENTED). Please upgrade Milvus server to a version "
                "that supports ExecuteDQL."
            ),
        ) from exc
    raise exc


def unpack_execute_dql_response(
    response: Optional[milvus_types.ExecuteDQLResponse],
    expected_count: int,
    round_decimals: Sequence[int],
) -> List[DQLResult]:
    """Validate outer ``ExecuteDQLResponse`` status and unpack ordered per-item results.

    ``DQLResult.status`` is authoritative for each item.
    """
    if response is None:
        raise MilvusException(message="Received None response from server during execute_dql")
    check_status(response.status)

    if len(response.results) != expected_count:
        raise MilvusException(
            message=(
                f"ExecuteDQL response result count mismatch: "
                f"expected {expected_count}, got {len(response.results)}"
            )
        )

    results: List[DQLResult] = []
    for idx, dql_result in enumerate(response.results):
        if not is_successful(dql_result.status):
            results.append(DQLResult(error=MilvusException.from_status(dql_result.status)))
            continue

        if dql_result.WhichOneof("result") != "search":
            results.append(
                DQLResult(
                    error=MilvusException(
                        message=f"ExecuteDQL item {idx} succeeded without a search result payload"
                    )
                )
            )
            continue

        search_resp = dql_result.search
        status_for_extra = (
            search_resp.status
            if search_resp.status and search_resp.status.extra_info
            else dql_result.status
        )
        results.append(
            DQLResult(
                result=SearchResult(
                    search_resp.results,
                    round_decimal=round_decimals[idx],
                    status=status_for_extra,
                    session_ts=search_resp.session_ts,
                )
            )
        )
    return results


__all__ = ["DQL", "DQLRequest", "DQLResult"]
