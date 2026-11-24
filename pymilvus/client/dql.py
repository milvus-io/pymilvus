"""ExecuteDQL request/result models, operation binding, and shared search preparation."""

import inspect
import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import grpc

from pymilvus.exceptions import (
    MilvusException,
    ParamError,
    ServerVersionIncompatibleException,
)
from pymilvus.grpc_gen import milvus_pb2 as milvus_types

from . import entity_helper, ts_utils
from .call_context import CallContext
from .check import check_id_and_data, check_pass_param
from .constants import CLUSTER_ID
from .embedding_list import EmbeddingList
from .prepare import Prepare
from .search_result import SearchResult
from .utils import check_status, is_successful

logger = logging.getLogger(__name__)

SUPPORTED_DQL_OPS = frozenset({"search"})
_BATCH_CONTEXT_KEYS = (CLUSTER_ID, "rls_principal", "skip_rls")
_EXECUTE_DQL_RPC_KWARG_KEYS = ("retry_times", "retry_on_rate_limit", "log_level", "log-level")
_NO_TIMEOUT_CHECK = object()


@dataclass(frozen=True)
class DQLRequest:
    """Immutable DQL sub-request created by ``DQL.*`` constructors for ``execute_dql``."""

    op: str
    params: Mapping[str, Any]
    kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not self.op:
            raise ParamError(message="DQLRequest.op must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise ParamError(
                message=f"DQLRequest.params must be a mapping, got {type(self.params).__name__}"
            )
        if not isinstance(self.kwargs, Mapping):
            raise ParamError(
                message=f"DQLRequest.kwargs must be a mapping, got {type(self.kwargs).__name__}"
            )
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))


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


if TYPE_CHECKING:
    import sys
    from typing import TypeVar

    if sys.version_info >= (3, 10):
        from typing import Concatenate, ParamSpec
    else:
        from typing_extensions import Concatenate, ParamSpec

    _Self = TypeVar("_Self")
    _P = ParamSpec("_P")

    def _bind_dql_op(
        op_name: str,
        method: Callable[Concatenate[_Self, _P], Any],
    ) -> Callable[_P, DQLRequest]: ...

else:

    def _bind_dql_op(op_name: str, method: Callable[..., Any]) -> Callable[..., DQLRequest]:
        """Bind a MilvusClient DQL method into a pure constructor returning ``DQLRequest``."""
        raw_sig = inspect.signature(method)
        raw_params = list(raw_sig.parameters.values())
        if not raw_params or raw_params[0].name != "self":
            msg = f"Expected a method with leading 'self' parameter, got {method!r}"
            raise ValueError(msg)

        op_params = raw_params[1:]
        bound_sig = raw_sig.replace(parameters=op_params, return_annotation=DQLRequest)
        var_kw_name = next(
            (p.name for p in op_params if p.kind == inspect.Parameter.VAR_KEYWORD),
            None,
        )

        def _dql_builder(*args: Any, **kwargs: Any) -> DQLRequest:
            bound = bound_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            extra_kwargs = dict(arguments.pop(var_kw_name, {})) if var_kw_name is not None else {}

            if arguments.get("timeout") is not None or extra_kwargs.get("timeout") is not None:
                raise ParamError(
                    message=(
                        f"Per-request 'timeout' is not supported in DQL.{op_name}(); "
                        "pass 'timeout' to execute_dql() instead."
                    )
                )

            if "db_name" in extra_kwargs:
                db_name = extra_kwargs["db_name"]
                if db_name is not None and not isinstance(db_name, str):
                    raise ParamError(
                        message=f"db_name must be a str or None, got {type(db_name).__name__}"
                    )

            return DQLRequest(op=op_name, params=arguments, kwargs=extra_kwargs)

        _dql_builder.__name__ = op_name
        _dql_builder.__qualname__ = f"DQL.{op_name}"
        _dql_builder.__doc__ = method.__doc__
        _dql_builder.__signature__ = bound_sig
        return _dql_builder


def check_no_batch_db_name(batch_kwargs: Optional[Mapping[str, Any]]) -> None:
    """Reject batch-level ``db_name`` in ``execute_dql`` kwargs."""
    if batch_kwargs and "db_name" in batch_kwargs:
        raise ParamError(
            message=(
                "Batch-level 'db_name' is not supported in execute_dql(); "
                "specify 'db_name' on individual DQL requests instead."
            )
        )


def extract_execute_dql_rpc_kwargs(batch_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract RPC control kwargs for ExecuteDQL (never forwarding execution budget ``timeout``)."""
    check_no_batch_db_name(batch_kwargs)
    return {k: batch_kwargs[k] for k in _EXECUTE_DQL_RPC_KWARG_KEYS if k in batch_kwargs}


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
        if req.op not in SUPPORTED_DQL_OPS:
            supported = ", ".join(sorted(SUPPORTED_DQL_OPS))
            raise ParamError(
                message=(
                    f"Unsupported DQL operation at reqs[{idx}]: {req.op!r}. "
                    f"Supported operations: {supported}"
                )
            )
    normalized_concurrency = Prepare.parse_execute_dql_max_concurrency(max_concurrency)
    timeout_ms = Prepare.parse_execute_dql_timeout_ms(timeout)
    return normalized_concurrency, timeout_ms


@dataclass
class _PreparedSearchItem:
    """Normalized, locally validated search inputs ready for schema fetch and proto build."""

    collection_name: str
    anns_field: str
    param: Dict[str, Any]
    limit: int
    data: Any
    ids: Any
    expression: Optional[str]
    partition_names: Optional[List[str]]
    output_fields: Optional[List[str]]
    round_decimal: int
    ranker: Any
    function_chains: Any
    highlighter: Any
    use_default_consistency: bool
    explicit_db_name: Optional[str]
    effective_db: str
    context: Optional[CallContext]
    needs_schema: bool
    extra_kwargs: Dict[str, Any]


def prepare_search_input(
    *,
    collection_name: str,
    anns_field: str,
    param: Dict[str, Any],
    limit: int,
    data: Any = None,
    ids: Any = None,
    expression: Optional[str] = None,
    partition_names: Optional[List[str]] = None,
    output_fields: Optional[List[str]] = None,
    round_decimal: int = -1,
    timeout: Any = _NO_TIMEOUT_CHECK,
    ranker: Any = None,
    function_chains: Any = None,
    highlighter: Any = None,
    db_name: Optional[str] = None,
    server_address: str = "",
    context: Optional[CallContext] = None,
    extra_kwargs: Optional[Mapping[str, Any]] = None,
    check_pass_param_fn: Callable[..., Any] = check_pass_param,
    construct_guarantee_ts_fn: Callable[..., bool] = ts_utils.construct_guarantee_ts,
) -> _PreparedSearchItem:
    """Validate and normalize one search for single ``search`` and ``execute_dql``."""
    item_kwargs = dict(extra_kwargs) if extra_kwargs is not None else {}

    if isinstance(ids, (int, str)):
        ids = [ids]
    check_id_and_data(ids, data)

    if item_kwargs.get("search_aggregation") is not None:
        logger.warning(
            "search_aggregation is set; search limit=%s is ignored and bucket count comes from SearchAggregation.size",
            limit,
        )

    pass_param_kwargs: Dict[str, Any] = {
        "limit": limit,
        "round_decimal": round_decimal,
        "anns_field": anns_field,
        "partition_name_array": partition_names,
        "output_fields": output_fields,
        "guarantee_timestamp": item_kwargs.get("guarantee_timestamp"),
    }
    if timeout is not _NO_TIMEOUT_CHECK:
        pass_param_kwargs["timeout"] = timeout
    check_pass_param_fn(**pass_param_kwargs)

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], EmbeddingList):
        data = [emb_list.to_flat_array() for emb_list in data]
        item_kwargs["is_embedding_list"] = True

    default_db = context.get_db_name() if context else ""
    explicit_db = db_name if db_name else None
    if explicit_db:
        effective_db = explicit_db
        item_context = (
            context.with_db_name(explicit_db)
            if context is not None
            else CallContext(db_name=explicit_db)
        )
    else:
        effective_db = default_db
        item_context = context

    use_default_consistency = construct_guarantee_ts_fn(
        collection_name, item_kwargs, server_address, effective_db
    )

    needs_schema = (
        not item_kwargs.get("schema")
        and data is not None
        and not entity_helper.entity_is_sparse_matrix(data)
        and isinstance(data, (list, tuple))
        and len(data) > 0
        and isinstance(data[0], bytes)
    )

    return _PreparedSearchItem(
        collection_name=collection_name,
        anns_field=anns_field,
        param=param,
        limit=limit,
        data=data,
        ids=ids,
        expression=expression,
        partition_names=partition_names,
        output_fields=output_fields,
        round_decimal=round_decimal,
        ranker=ranker,
        function_chains=function_chains,
        highlighter=highlighter,
        use_default_consistency=use_default_consistency,
        explicit_db_name=explicit_db,
        effective_db=effective_db,
        context=item_context,
        needs_schema=needs_schema,
        extra_kwargs=item_kwargs,
    )


def build_search_request(
    item: _PreparedSearchItem,
    *,
    schema: Optional[dict] = None,
    prepare_cls: Any = Prepare,
) -> milvus_types.SearchRequest:
    """Build a ``milvus_pb2.SearchRequest`` from a prepared search item."""
    req_kwargs = dict(item.extra_kwargs)
    req_kwargs.pop("db_name", None)
    if schema is not None:
        req_kwargs["schema"] = schema
    if item.explicit_db_name:
        req_kwargs["db_name"] = item.explicit_db_name
    return prepare_cls.search_requests_with_expr(
        collection_name=item.collection_name,
        anns_field=item.anns_field,
        param=item.param,
        limit=item.limit,
        data=item.data,
        ids=item.ids,
        expr=item.expression,
        partition_names=item.partition_names,
        output_fields=item.output_fields,
        round_decimal=item.round_decimal,
        ranker=item.ranker,
        function_chains=item.function_chains,
        highlighter=item.highlighter,
        use_default_consistency=item.use_default_consistency,
        **req_kwargs,
    )


def prepare_dql_item(
    req: DQLRequest,
    *,
    server_address: str,
    context: Optional[CallContext],
    batch_kwargs: Mapping[str, Any],
    check_pass_param_fn: Callable[..., Any] = check_pass_param,
    construct_guarantee_ts_fn: Callable[..., bool] = ts_utils.construct_guarantee_ts,
) -> _PreparedSearchItem:
    """Validate and prepare a ``DQLRequest`` item using the shared search preparation path."""
    if req.op != "search":
        supported = ", ".join(sorted(SUPPORTED_DQL_OPS))
        raise ParamError(
            message=f"Unsupported DQL operation: {req.op!r}. Supported operations: {supported}"
        )

    params = req.params
    item_kwargs = dict(req.kwargs)

    if params.get("timeout") is not None or item_kwargs.get("timeout") is not None:
        raise ParamError(
            message=(
                "Per-request 'timeout' is not supported in DQL.search(); "
                "pass 'timeout' to execute_dql() instead."
            )
        )
    if "param" in item_kwargs:
        raise ParamError(message="Use 'search_params' instead of 'param' in DQL.search().")
    if "use_default_consistency" in item_kwargs:
        raise ParamError(
            message=(
                "'use_default_consistency' is managed automatically based on consistency_level."
            )
        )

    collection_name = params.get("collection_name")
    if not isinstance(collection_name, str) or not collection_name:
        raise ParamError(message="collection_name must be a non-empty string")

    if "filter_params" in item_kwargs and "expr_params" in item_kwargs:
        raise ParamError(message="Provide either 'filter_params' or 'expr_params', not both.")
    expr_params = item_kwargs.pop("filter_params", item_kwargs.pop("expr_params", {}))
    item_kwargs["expr_params"] = {} if expr_params is None else expr_params

    db_name = item_kwargs.pop("db_name", None)
    if db_name is not None and not isinstance(db_name, str):
        raise ParamError(message=f"db_name must be a str or None, got {type(db_name).__name__}")

    for batch_key in _BATCH_CONTEXT_KEYS:
        if batch_key in batch_kwargs:
            item_kwargs[batch_key] = batch_kwargs[batch_key]

    search_aggregation = params.get("search_aggregation")
    if search_aggregation is not None:
        item_kwargs["search_aggregation"] = search_aggregation

    round_decimal = item_kwargs.pop("round_decimal", -1)

    return prepare_search_input(
        collection_name=collection_name,
        anns_field=params.get("anns_field") or "",
        param=params.get("search_params") or {},
        limit=params.get("limit", 10),
        data=params.get("data"),
        ids=params.get("ids"),
        expression=params.get("filter", ""),
        partition_names=params.get("partition_names"),
        output_fields=params.get("output_fields"),
        round_decimal=round_decimal,
        ranker=params.get("ranker"),
        function_chains=params.get("function_chains"),
        highlighter=params.get("highlighter"),
        db_name=db_name,
        server_address=server_address,
        context=context,
        extra_kwargs=item_kwargs,
        check_pass_param_fn=check_pass_param_fn,
        construct_guarantee_ts_fn=construct_guarantee_ts_fn,
    )


def build_dql_request_proto(
    item: _PreparedSearchItem,
    *,
    schema: Optional[dict] = None,
    prepare_cls: Any = Prepare,
) -> milvus_types.DQLRequest:
    """Wrap a prepared search item into a protobuf ``DQLRequest``."""
    return milvus_types.DQLRequest(
        search=build_search_request(item, schema=schema, prepare_cls=prepare_cls)
    )


def prepare_execute_dql_batch(
    reqs: Sequence[DQLRequest],
    *,
    max_concurrency: int,
    timeout: Optional[float],
    server_address: str,
    context: Optional[CallContext],
    batch_kwargs: Mapping[str, Any],
    check_pass_param_fn: Callable[..., Any] = check_pass_param,
    construct_guarantee_ts_fn: Callable[..., bool] = ts_utils.construct_guarantee_ts,
) -> Tuple[int, Dict[str, Any], List[_PreparedSearchItem]]:
    """Run Pass 1 batch/item validation and normalization before any network RPCs."""
    normalized_max_concurrency, _ = validate_execute_dql_params(
        reqs, max_concurrency=max_concurrency, timeout=timeout, batch_kwargs=batch_kwargs
    )
    rpc_kwargs = extract_execute_dql_rpc_kwargs(batch_kwargs)
    prepared_items = [
        prepare_dql_item(
            req,
            server_address=server_address,
            context=context,
            batch_kwargs=batch_kwargs,
            check_pass_param_fn=check_pass_param_fn,
            construct_guarantee_ts_fn=construct_guarantee_ts_fn,
        )
        for req in reqs
    ]
    return normalized_max_concurrency, rpc_kwargs, prepared_items


def build_execute_dql_request(
    prepared_items: Sequence[_PreparedSearchItem],
    local_schemas: Mapping[Tuple[str, str], dict],
    *,
    max_concurrency: int,
    timeout: Optional[float],
    prepare_cls: Any = Prepare,
) -> milvus_types.ExecuteDQLRequest:
    """Assemble ``ExecuteDQLRequest`` from prepared items and DB-isolated schemas."""
    dql_requests = [
        build_dql_request_proto(
            item,
            schema=(
                local_schemas.get((item.effective_db, item.collection_name))
                if item.needs_schema
                else None
            ),
            prepare_cls=prepare_cls,
        )
        for item in prepared_items
    ]
    return prepare_cls.execute_dql_request(
        dql_requests,
        max_concurrency=max_concurrency,
        timeout=timeout,
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


def finalize_execute_dql_response(
    response: Optional[milvus_types.ExecuteDQLResponse],
    expected_count: int,
    round_decimals: Sequence[int],
) -> List[DQLResult]:
    """Validate outer ``ExecuteDQLResponse`` status and unpack ordered per-item results."""
    if response is None:
        raise MilvusException(message="Received None response from server during execute_dql")
    check_status(response.status)
    return unpack_execute_dql_response(response, expected_count, round_decimals)


def unpack_execute_dql_response(
    response: milvus_types.ExecuteDQLResponse,
    expected_count: int,
    round_decimals: Sequence[int],
) -> List[DQLResult]:
    """Unpack an outer-success ``ExecuteDQLResponse`` in exact positional order.

    ``DQLResult.status`` is authoritative for each item.
    """
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
