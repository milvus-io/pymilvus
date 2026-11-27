"""ExecuteDQL request/result models, ``DQL`` constructors, and batch helpers."""

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.function_chain import FunctionChain
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.orm.schema import Function, FunctionScore, Highlighter

from .call_context import CallContext
from .check import validate_noneable_param, validate_param
from .constants import CLUSTER_ID
from .prepare import Prepare, PreparedSearch, prepare_search
from .search_aggregation import SearchAggregation
from .search_result import SearchResult
from .utils import check_status, is_successful


@dataclass(frozen=True)
class DQLRequest:
    """Base descriptor for immutable read-only DQL sub-requests.

    Do not instantiate ``DQLRequest`` directly. Construct concrete sub-requests using
    the factory methods on :class:`DQL` (for example, ``DQL.search(...)``) and pass
    them to ``MilvusClient.execute_dql()`` or ``AsyncMilvusClient.execute_dql()``.
    """

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
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DQLResult:
    """Per-item outcome returned by ``execute_dql()``.

    Each element in the list returned by ``execute_dql(reqs, ...)`` corresponds
    positionally to the sub-request at the same index in ``reqs``. For every item,
    either ``result`` is populated (when ``ok`` is ``True`` and ``error`` is ``None``)
    or ``error`` holds a :class:`~pymilvus.MilvusException` (when ``ok`` is ``False``
    and ``result`` is ``None``).

    Attributes:
        result (Optional[Any]): The sub-request's result payload on success (for
            ``DQL.search``, a :class:`~pymilvus.client.search_result.SearchResult`),
            or ``None`` when the item failed.
        error (Optional[MilvusException]): The per-item error reported by the server,
            or ``None`` when the item succeeded.

    Examples:
        Fault-tolerant iteration using ``ok``::

            results = client.execute_dql([
                DQL.search("col_a", data=[[0.1, 0.2]], limit=5),
                DQL.search("col_b", data=[[0.3, 0.4]], limit=5),
            ])
            for idx, item in enumerate(results):
                if item.ok:
                    print(idx, item.result)
                else:
                    print(idx, "failed:", item.error)

        Strict unpacking using ``unwrap()``::

            hits_a, hits_b = [item.unwrap() for item in results]
    """

    result: Optional[Any] = None
    error: Optional[MilvusException] = None

    @property
    def ok(self) -> bool:
        """Whether this sub-request succeeded (``True`` iff ``error is None``)."""
        return self.error is None

    def unwrap(self) -> Any:
        """Return ``result`` if the item succeeded, otherwise raise ``error``.

        Returns:
            Any: The item's result payload (e.g. ``SearchResult`` for ``DQL.search``).

        Raises:
            MilvusException: The per-item error stored in ``self.error``.
        """
        if self.error is not None:
            raise self.error
        return self.result


class DQL:
    """Namespace of read-only DQL sub-request builders for ``execute_dql()``.

    Use the static factory methods on ``DQL`` to build immutable :class:`DQLRequest`
    objects and execute them as an ordered batch via ``MilvusClient.execute_dql()``
    or ``AsyncMilvusClient.execute_dql()``.

    Examples:
        >>> reqs = [
        ...     DQL.search("articles", data=[[0.1, 0.2]], limit=5),
        ...     DQL.search("archive", data=[[0.3, 0.4]], limit=5, db_name="analytics"),
        ... ]
        >>> results = client.execute_dql(reqs, max_concurrency=4, timeout=5.0)
    """

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
        """Build a search sub-request for ``execute_dql()``.

        Parameters mirror ``MilvusClient.search`` (as keyword-only arguments after
        ``collection_name``), with an explicit per-request ``db_name`` and without
        per-request ``timeout`` (execution budget is configured at the ``execute_dql``
        batch level).

        Note:
            Only raw ``bytes`` vectors with ``anns_field`` specified and no explicit
            ``schema`` trigger ``DescribeCollection`` before ``ExecuteDQL``. For large
            multi-collection batches, prefer passing ``np.ndarray`` vectors or an
            explicit ``schema`` to avoid extra schema lookups.

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
                ``offset``, ``schema``, or ``namespace``. Passing a non-``None`` ``timeout``
                raises ``ParamError``.

        Returns:
            DQLRequest: Immutable sub-request to pass in ``client.execute_dql([...])``.
        """
        validate_param("collection_name", collection_name, str)
        validate_noneable_param("db_name", db_name, str)

        extra_kwargs = dict(kwargs)
        timeout = extra_kwargs.pop("timeout", None)
        if timeout is not None:
            raise ParamError(
                message=(
                    "Per-request 'timeout' is not supported in DQL.search(); "
                    "pass 'timeout' to execute_dql() instead."
                )
            )

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


def _parse_max_concurrency(max_concurrency: int) -> int:
    uint32_max = 2**32 - 1
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise ParamError(
            message=(
                f"max_concurrency must be a non-negative int, "
                f"got {type(max_concurrency).__name__}"
            )
        )
    if max_concurrency < 0 or max_concurrency > uint32_max:
        raise ParamError(
            message=f"max_concurrency must be in [0, {uint32_max}], got {max_concurrency}"
        )
    return max_concurrency


def _parse_timeout_ms(timeout: Optional[float]) -> Optional[int]:
    # Maximum non-negative execution budget in milliseconds supported by Go time.Duration
    # (math.MaxInt64 / time.Millisecond).
    max_timeout_ms = 9_223_372_036_854
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
    if timeout_ms > max_timeout_ms:
        raise ParamError(
            message=(
                f"timeout exceeds maximum supported execution budget "
                f"({max_timeout_ms} ms): {timeout}"
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
    if batch_kwargs and "db_name" in batch_kwargs:
        raise ParamError(
            message=(
                "Batch-level 'db_name' is not supported in execute_dql(); "
                "specify 'db_name' on individual DQL requests instead."
            )
        )
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
            raise ParamError(
                message=(
                    f"Unsupported DQL operation at reqs[{idx}]: {req.op!r}. "
                    f"Supported operations: search"
                )
            )
    return _parse_max_concurrency(max_concurrency), _parse_timeout_ms(timeout)


@dataclass(frozen=True)
class _PreparedBatch:
    """Validated and normalized ExecuteDQL batch ready for schema fetch and proto assembly."""

    items: Tuple[PreparedSearch, ...]
    schema_targets: Tuple[Tuple[Tuple[str, str], Optional[CallContext]], ...]
    max_concurrency: int
    timeout_ms: Optional[int]

    @property
    def round_decimals(self) -> List[int]:
        return [item.round_decimal for item in self.items]


def _prepare_search_item(
    req: _SearchRequest,
    *,
    server_address: str,
    context: Optional[CallContext],
    batch_kwargs: Mapping[str, Any],
) -> PreparedSearch:
    """Prepare a search sub-request via the shared ``prepare_search()`` helper."""
    item_kwargs = dict(req.extra_kwargs)

    expr_params = item_kwargs.pop("filter_params", item_kwargs.get("expr_params", {}))
    item_kwargs["expr_params"] = {} if expr_params is None else expr_params

    if CLUSTER_ID in batch_kwargs:
        item_kwargs[CLUSTER_ID] = batch_kwargs[CLUSTER_ID]
    item_kwargs.update(Prepare._rls_context(batch_kwargs))

    if req.search_aggregation is not None:
        item_kwargs["search_aggregation"] = req.search_aggregation

    round_decimal = item_kwargs.pop("round_decimal", -1)
    return prepare_search(
        collection_name=req.collection_name,
        anns_field=req.anns_field or "",
        param=req.search_params or {},
        limit=req.limit,
        data=req.data,
        ids=req.ids,
        expr=req.filter,
        partition_names=req.partition_names,
        output_fields=req.output_fields,
        round_decimal=round_decimal,
        ranker=req.ranker,
        function_chains=req.function_chains,
        highlighter=req.highlighter,
        db_name=req.db_name or None,
        server_address=server_address,
        context=context,
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
    )


def build_execute_dql_request(
    batch: _PreparedBatch,
    local_schemas: Mapping[Tuple[str, str], dict],
) -> milvus_types.ExecuteDQLRequest:
    """Assemble ``ExecuteDQLRequest`` from a prepared batch and DB-isolated schemas."""
    dql_requests = [
        milvus_types.DQLRequest(
            search=item.to_request(
                schema=(
                    local_schemas.get((item.effective_db, item.collection_name))
                    if item.needs_schema
                    else None
                )
            )
        )
        for item in batch.items
    ]
    return Prepare.execute_dql_request(
        dql_requests,
        max_concurrency=batch.max_concurrency,
        timeout_ms=batch.timeout_ms,
    )


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
