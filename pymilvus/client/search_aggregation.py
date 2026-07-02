"""Search aggregation API — user-facing spec for hierarchical bucket aggregation.

Mirrors common.SearchAggregationSpec / TopHitsSpec proto. Users build nested
SearchAggregation + TopHits objects client-side; to_proto() produces the proto message
that is assigned to SearchRequest.search_aggregation.
"""

from typing import Any, Dict, List, Optional, Union

from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import common_pb2, schema_pb2

_VALID_METRIC_OPS = {"avg", "sum", "count", "min", "max"}
_VALID_DIRECTIONS = {"asc", "desc"}
_SPECIAL_ORDER_KEYS = {"_count", "_key"}


def _validate_single_kv(item: Dict[str, str], parent: str) -> None:
    if not isinstance(item, dict) or len(item) != 1:
        raise ParamError(
            message=f"{parent} entries must be single-key dicts like {{'alias': 'asc'}}, got {item!r}"
        )
    key, value = next(iter(item.items()))
    if not isinstance(key, str) or not isinstance(value, str):
        raise ParamError(message=f"{parent} entries must map str to str, got {item!r}")


class TopHits:
    """Document snapshot spec inside a SearchAggregation bucket.

    sort entries use the compact form [{"field_name": "asc"}, ...]; the special
    field "_score" refers to the vector similarity score.
    """

    def __init__(self, size: int, sort: Optional[List[Dict[str, str]]] = None):
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ParamError(message=f"TopHits.size must be a positive int, got {size!r}")
        self._size = size
        self._sort: List[Dict[str, str]] = []
        if sort is not None:
            if not isinstance(sort, list):
                raise ParamError(message=f"TopHits.sort must be a list, got {type(sort).__name__}")
            for item in sort:
                _validate_single_kv(item, "TopHits.sort")
                direction = next(iter(item.values()))
                if direction not in _VALID_DIRECTIONS:
                    raise ParamError(
                        message=f"TopHits.sort direction must be 'asc' or 'desc', got {direction!r}"
                    )
                self._sort.append(dict(item))

    @property
    def size(self) -> int:
        return self._size

    @property
    def sort(self) -> List[Dict[str, str]]:
        return [dict(item) for item in self._sort]

    def to_proto(self) -> common_pb2.TopHitsSpec:
        spec = common_pb2.TopHitsSpec(size=self._size)
        for item in self._sort:
            field_name, direction = next(iter(item.items()))
            spec.sort.append(common_pb2.SortSpec(field_name=field_name, direction=direction))
        return spec


class SearchAggregation:
    """One level of bucket aggregation. Recursively nested via sub_aggregation."""

    def __init__(
        self,
        fields: List[str],
        size: int,
        metrics: Optional[Dict[str, Dict[str, str]]] = None,
        order: Optional[List[Dict[str, str]]] = None,
        top_hits: Optional[TopHits] = None,
        sub_aggregation: Optional["SearchAggregation"] = None,
    ):
        if not isinstance(fields, list) or len(fields) == 0:
            raise ParamError(message="SearchAggregation.fields must be a non-empty list of str")
        for f in fields:
            if not isinstance(f, str) or not f:
                raise ParamError(
                    message=f"SearchAggregation.fields must contain non-empty str, got {f!r}"
                )
            # JSON path expressions (e.g. "meta['region']") are not yet supported by
            # the server in search_aggregation. Reject client-side until enabled.
            if "[" in f or "]" in f:
                raise ParamError(
                    message=(
                        f"SearchAggregation.fields does not yet support bracketed JSON path "
                        f"expressions, got {f!r}"
                    )
                )
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ParamError(message=f"SearchAggregation.size must be a positive int, got {size!r}")
        if top_hits is not None and not isinstance(top_hits, TopHits):
            raise ParamError(
                message=f"SearchAggregation.top_hits must be a TopHits instance, got {type(top_hits).__name__}"
            )
        if sub_aggregation is not None and not isinstance(sub_aggregation, SearchAggregation):
            raise ParamError(
                message=f"SearchAggregation.sub_aggregation must be a SearchAggregation instance, got {type(sub_aggregation).__name__}"
            )

        self._fields = list(fields)
        self._size = size
        self._metrics = self._validate_metrics(metrics)
        self._order = self._validate_order(order, self._metrics)
        self._top_hits = top_hits
        self._sub_aggregation = sub_aggregation

    @staticmethod
    def _validate_metrics(
        metrics: Optional[Dict[str, Dict[str, str]]],
    ) -> Dict[str, Dict[str, str]]:
        if metrics is None:
            return {}
        if not isinstance(metrics, dict):
            raise ParamError(message="SearchAggregation.metrics must be a dict keyed by alias")
        out: Dict[str, Dict[str, str]] = {}
        for alias, spec in metrics.items():
            if not isinstance(alias, str) or not alias:
                raise ParamError(message=f"metric alias must be non-empty str, got {alias!r}")
            _validate_single_kv(spec, f"metrics[{alias!r}]")
            op, field_name = next(iter(spec.items()))
            if op not in _VALID_METRIC_OPS:
                raise ParamError(
                    message=f"metrics[{alias!r}] op must be one of {sorted(_VALID_METRIC_OPS)}, got {op!r}"
                )
            if not field_name:
                raise ParamError(message=f"metrics[{alias!r}] field_name must be non-empty")
            if op != "count" and field_name == "*":
                raise ParamError(
                    message=f"metrics[{alias!r}] '*' is only valid for 'count' op, got op={op!r}"
                )
            out[alias] = dict(spec)
        return out

    @staticmethod
    def _validate_order(
        order: Optional[List[Dict[str, str]]],
        metrics: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if order is None:
            return []
        if not isinstance(order, list):
            raise ParamError(message="SearchAggregation.order must be a list of single-key dicts")
        out: List[Dict[str, str]] = []
        allowed_keys = set(metrics.keys()) | _SPECIAL_ORDER_KEYS
        for item in order:
            _validate_single_kv(item, "SearchAggregation.order")
            key, direction = next(iter(item.items()))
            if direction not in _VALID_DIRECTIONS:
                raise ParamError(
                    message=f"SearchAggregation.order direction must be 'asc' or 'desc', got {direction!r}"
                )
            if key not in allowed_keys:
                raise ParamError(
                    message=(
                        f"SearchAggregation.order key {key!r} must be a metric alias or one of "
                        f"{sorted(_SPECIAL_ORDER_KEYS)}; defined metrics: {sorted(metrics.keys())}"
                    )
                )
            out.append(dict(item))
        return out

    def to_proto(self) -> common_pb2.SearchAggregationSpec:
        spec = common_pb2.SearchAggregationSpec(size=self._size)
        spec.fields.extend(self._fields)
        for alias, metric in self._metrics.items():
            op, field_name = next(iter(metric.items()))
            spec.metrics[alias].CopyFrom(common_pb2.MetricAggSpec(op=op, field_name=field_name))
        for item in self._order:
            key, direction = next(iter(item.items()))
            spec.order.append(common_pb2.OrderSpec(key=key, direction=direction))
        if self._top_hits is not None:
            spec.top_hits.CopyFrom(self._top_hits.to_proto())
        if self._sub_aggregation is not None:
            spec.sub_aggregation.CopyFrom(self._sub_aggregation.to_proto())
        return spec


def _oneof_value(msg: Any) -> Any:
    """Resolve the populated oneof branch on a proto message; returns None if none set."""
    which = msg.WhichOneof("value") if msg.DESCRIPTOR.oneofs_by_name.get("value") else None
    return getattr(msg, which) if which else None


class AggregationHit:
    """One document inside an AggregationBucket. Mirrors schema.AggHit proto.

    Field values are keyed by field_name (server-filled). field_ids() gives
    the numeric id view for consumers that need the stable wire identity.
    """

    def __init__(self, proto: schema_pb2.AggHit):
        pk_branch = proto.WhichOneof("pk")
        self._pk: Optional[Union[int, str]] = getattr(proto, pk_branch) if pk_branch else None
        self._score: float = proto.score
        self._fields: Dict[str, Any] = {}
        self._field_ids: Dict[str, int] = {}
        for f in proto.fields:
            value = _oneof_value(f)
            if value is None:
                continue
            name = f.field_name or str(f.field_id)
            self._fields[name] = value
            self._field_ids[name] = f.field_id

    @property
    def pk(self) -> Optional[Union[int, str]]:
        return self._pk

    @property
    def score(self) -> float:
        return self._score

    @property
    def fields(self) -> Dict[str, Any]:
        return dict(self._fields)

    def field_ids(self) -> Dict[str, int]:
        """Map of field_name -> field_id for the fields present on this hit."""
        return dict(self._field_ids)

    def __repr__(self) -> str:
        return f"AggregationHit(pk={self._pk!r}, score={self._score}, fields={self._fields!r})"


class AggregationBucket:
    """One bucket in the aggregation result tree. Mirrors schema.AggBucket proto.

    - key: list of {field_name, field_id, value} entries forming the composite grouping key.
    - count: total docs in retrieval pool falling into this bucket.
    - metrics: alias -> typed metric value (int/float/str/bool).
    - hits: top-hits document snapshot (empty if level had no top_hits).
    - sub_groups: child buckets (empty if leaf level).
    """

    def __init__(self, proto: schema_pb2.AggBucket):
        self._key: List[Dict[str, Any]] = []
        for entry in proto.key:
            value = _oneof_value(entry)
            self._key.append(
                {
                    "field_name": entry.field_name or str(entry.field_id),
                    "field_id": entry.field_id,
                    "value": value,
                }
            )

        self._count: int = proto.count

        self._metrics: Dict[str, Any] = {}
        for alias, metric in proto.metrics.items():
            self._metrics[alias] = _oneof_value(metric)

        self._hits: List[AggregationHit] = [AggregationHit(h) for h in proto.hits]
        self._sub_groups: List[AggregationBucket] = [AggregationBucket(b) for b in proto.sub_groups]

    @property
    def key(self) -> List[Dict[str, Any]]:
        return [dict(k) for k in self._key]

    @property
    def count(self) -> int:
        return self._count

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    @property
    def hits(self) -> List[AggregationHit]:
        return list(self._hits)

    @property
    def sub_groups(self) -> List["AggregationBucket"]:
        return list(self._sub_groups)

    def __repr__(self) -> str:
        return (
            f"AggregationBucket(key={self._key!r}, count={self._count}, "
            f"metrics={self._metrics!r}, hits={len(self._hits)}, "
            f"sub_groups={len(self._sub_groups)})"
        )


def parse_agg_buckets(proto_buckets: Any) -> List[AggregationBucket]:
    """Parse repeated AggBucket from SearchResultData into AggregationBucket list."""
    return [AggregationBucket(b) for b in proto_buckets]
