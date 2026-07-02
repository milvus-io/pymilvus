# Function Chain API Design for PyMilvus

- **Created:** 2026-06-23
- **Updated:** 2026-06-23
- **Author(s):** @junjie.jiang

## Overview

This document describes the PyMilvus client API for Milvus Function Chain search reranking.
Function Chain is a typed, ordered, stage-aware plan sent through public proto
`schema.FunctionChain` and `SearchRequest.function_chains`. It lets users describe score
adjustment and rerank operations such as numeric combination, decay, model rerank,
rounding, explicit sorting, and per-query limiting.

The PyMilvus responsibility is to provide a natural Python builder and compile it to
Function Chain proto. Server-side planning, field fetching, execution placement, and final
search result projection remain Milvus responsibilities.

This design is based on the current Milvus implementation in
`/home/junjie.jiangjjj/WorkSpace/dev/milvus`, not only on the high-level chain API design
notes. The public PyMilvus API should expose only operations and expressions that the
server can currently deserialize and execute for ordinary Search L2 rerank.

**Key Principles:**

- Expose Function Chain as a new Search rerank API, not as a collection schema function.
- Keep Function Chain separate from existing `Function`, `FunctionScore`, and `ranker` APIs.
- Match the proto shape directly: `FunctionChainExpr = name + ordered args + typed params`.
- Use explicit `col(...)` objects for expression arguments; do not support `inputs=[...]` in
  the public API.
- Use typed proto values instead of JSON strings for nested parameters.
- Keep `fn` as a lightweight module of factory functions, not a class and not a dynamic DSL.
- First version supports ordinary `search` L2 rerank only; `hybrid_search` is out of scope.
- `$score` is a system value passed as a typed column reference name, not as an output field.
- Do not document or expose unimplemented server expressions such as generic `expr` or
  `boost` in the first PyMilvus version.

## Current Server Support Snapshot

### Public proto shape

The public proto models expressions as ordered typed arguments:

```proto
message FunctionChainExpr {
  string name = 1;
  repeated FunctionChainExprArg args = 2;
  map<string, FunctionParamValue> params = 3;
}

message FunctionChainExprArg {
  oneof arg {
    FunctionChainColumnArg column = 1;
    FunctionParamValue literal = 2;
  }
}

message FunctionChainColumnArg {
  string name = 1;
}
```

PyMilvus should model ordered `args` directly. For the current supported server functions,
public SDK examples should use `col(...)` args. Literal expression args are in proto, but
current implemented Milvus builtin expressions do not require literal args; their scalar
configuration is passed through typed `params`.

### Server-registered operator names

Milvus currently declares/registers these public FunctionChain op names:

| Op name | Server status | PyMilvus first-version exposure |
|---------|---------------|----------------------------------|
| `map` | Implemented | Expose as `FunctionChain.map(output, expr)` |
| `sort` | Implemented | Expose as `FunctionChain.sort(by, desc=True, tie_break_col=None)` |
| `limit` | Implemented | Expose as `FunctionChain.limit(limit, offset=0)` |
| `filter` | Implemented runtime operator, but no current boolean expression is exposed for FunctionChain | Do not expose first |
| `select` | Implemented runtime operator, but final projection is Search-owned | Do not expose first |
| `group_by` | Implemented runtime operator, but writes `$group_score`, which current L2 system-output validation rejects | Do not expose first |

Evidence in Milvus:

- Op constants: `internal/util/function/chain/types/constants.go:24-31`
- Unknown op rejection: `internal/util/function/chain/repr.go:358-370`
- Search L2 stage restriction: `internal/proxy/function_chain_validator.go:50-79`
- L2 readable/writable system names: `internal/proxy/function_chain_validator.go:127-168`

### Server-registered expression names

Milvus currently registers these FunctionChain expression names:

| Expr name sent to server | Server params | Expected args | PyMilvus helper |
|--------------------------|---------------|---------------|-----------------|
| `num_combine` | `mode`, optional `weights` | At least two numeric columns | `fn.num_combine(...)` |
| `decay` | `function`, `origin`, `scale`, optional `offset`, optional `decay` | One numeric column | `fn.decay(...)` |
| `round_decimal` | `decimal` | One score/numeric column | `fn.round_decimal(...)` |
| `rerank_model` | `queries`, plus provider-specific scalar params | One text column | `fn.rerank_model(...)` |

Anything else is rejected by the function registry as an unknown function.

Evidence in Milvus:

- Function registry unknown-function rejection: `internal/util/function/chain/types/registry.go:81-88`
- `num_combine`: `internal/util/function/chain/expr/num_combine_expr.go:53-127`
- `decay`: `internal/util/function/chain/expr/decay_expr.go:114-190`
- `round_decimal`: `internal/util/function/chain/expr/round_decimal_expr.go:31-64`
- `rerank_model`: `internal/util/function/chain/expr/rerank_model_expr.go:35-89`

### Not supported by current server implementation

The following should not be part of first-version PyMilvus public docs:

- `fn.expr(col("$score") * 100)` because Milvus does not currently register an `expr`
  FunctionChain expression.
- Arithmetic AST operator overloading for FunctionChain expressions, because there is no
  current server `expr` implementation to consume the compiled expression string.
- `fn.boost(...)` because Milvus does not currently register a `boost` expression.
- Generic `fn.call(name, ...)` as a documented normal path, because unknown names are rejected.
  It can be added later if/when server-side UDF or custom-function registration exists.
- Public `merge` op deserialization. `MergeOp` exists internally for legacy rerank builders,
  but is not publicly deserializable from `FunctionChainOp` yet.
- Hybrid search `function_chains`; Proxy rejects it for now.

## Scope

### In Scope for the First Version

- `MilvusClient.search(..., function_chains=...)`
- `Collection.search(..., function_chains=...)`
- `AsyncMilvusClient.search(..., function_chains=...)`
- Single chain or list of chains as input.
- `FunctionChainStage.L2_RERANK` for ordinary Search.
- Builder operations:
  - `map(output, expr)`
  - `sort(by, desc=True, tie_break_col=None)`
  - `limit(limit, offset=0)`
- Argument helper:
  - `col(name)` for collection fields, temporary variables, and supported system values.
- Expression factories:
  - `fn.num_combine(*cols, mode="sum", weights=None)`
  - `fn.decay(col, function, origin, scale, offset=0, decay=0.5)`
  - `fn.round_decimal(col, decimal)`
  - `fn.rerank_model(col, queries=[...], **provider_params)`

### Out of Scope for the First Version

- `hybrid_search(..., function_chains=...)`
- Function chains inside `AnnSearchRequest` / sub-search requests.
- Ingestion, preprocess, L0, L1, and postprocess stages from the Python public API contract.
- Generic arithmetic expression DSL and operator overloading.
- `filter`, `select`, and `group_by` public builder methods.
- Dynamic `fn.<any_name>` generation through `__getattr__`.
- Returning temporary chain variables as user-facing output fields.

## User Experience

Recommended score-combination usage:

```python
from pymilvus import FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.function_chain import col, fn

client = MilvusClient(uri="http://localhost:19530")

chain = (
    FunctionChain(stage=FunctionChainStage.L2_RERANK, name="l2_score_plus_ts")
    .map("$score", fn.num_combine(col("$score"), col("ts"), mode="sum"))
    .sort(col("$score"), desc=True)
)

res = client.search(
    collection_name="my_collection",
    data=[query_vector],
    anns_field="vector",
    search_params={},
    output_fields=["doctype"],
    function_chains=[chain],
)
```

The above compiles to server expr `num_combine`, not `sum`:

```text
expr.name = "num_combine"
expr.args = [column("$score"), column("ts")]
expr.params["mode"] = "sum"
```

Temporary values can be chained by writing a non-system output name and reading it later:

```python
chain = (
    FunctionChain(FunctionChainStage.L2_RERANK)
    .map("freshness", fn.decay(col("ts"), function="linear", origin=now, scale=86400))
    .map("$score", fn.num_combine(col("$score"), col("freshness"), mode="sum"))
    .sort(col("$score"), desc=True)
)
```

Model rerank usage:

```python
chain = (
    FunctionChain(FunctionChainStage.L2_RERANK)
    .map(
        "model_score",
        fn.rerank_model(
            col("doc"),
            queries=["renewable energy developments"],
            provider="voyageai",
            model_name="rerank-2.5",
            max_client_batch_size=128,
            truncation=True,
        ),
    )
    .map("$score", fn.num_combine(col("$score"), col("model_score"), mode="sum"))
    .sort(col("$score"), desc=True)
)
```

Rounding and limiting usage:

```python
chain = (
    FunctionChain(FunctionChainStage.L2_RERANK)
    .map("$score", fn.round_decimal(col("$score"), decimal=3))
    .sort(col("$score"), desc=True)
    .limit(20)
)
```

## Public API Reference

### Module Layout

```text
pymilvus/
  function_chain/
    __init__.py
    chain.py
    fn.py
```

Top-level exports:

```python
from pymilvus import FunctionChain, FunctionChainStage
from pymilvus.function_chain import col, fn
```

Optionally, PyMilvus may also export `col` and `fn` from the top level, but the canonical
import is `from pymilvus.function_chain import col, fn`.

### FunctionChainStage

```python
class FunctionChainStage(IntEnum):
    UNSPECIFIED = schema_pb2.FunctionChainStageUnspecified
    INGESTION = schema_pb2.FunctionChainStageIngestion
    PRE_PROCESS = schema_pb2.FunctionChainStagePreProcess
    L0_RERANK = schema_pb2.FunctionChainStageL0Rerank
    L1_RERANK = schema_pb2.FunctionChainStageL1Rerank
    L2_RERANK = schema_pb2.FunctionChainStageL2Rerank
    POST_PROCESS = schema_pb2.FunctionChainStagePostProcess
```

The first PyMilvus release validates `L2_RERANK` as the only supported stage for Search.
Other enum values exist because they exist in public proto, but they should not be accepted
by `search(..., function_chains=...)` until the server supports them for that request path.

### FunctionChain

```python
class FunctionChain:
    def __init__(self, stage: FunctionChainStage, name: str = ""):
        ...

    def map(self, output: str, expr: FunctionChainExpr) -> "FunctionChain":
        ...

    def sort(
        self,
        by: Union[str, ColumnRef],
        desc: bool = True,
        tie_break_col: Optional[Union[str, ColumnRef]] = None,
    ) -> "FunctionChain":
        ...

    def limit(self, limit: int, offset: int = 0) -> "FunctionChain":
        ...

    def to_proto(self) -> schema_pb2.FunctionChain:
        ...
```

`FunctionChain` is a mutable builder that returns `self` from builder methods so users can
chain calls fluently. The stored operations are converted to proto only when a search
request is prepared.

`map(output, expr)` produces a `FunctionChainOp` with:

```text
op = "map"
outputs = [output]
expr = expr.to_proto()
```

Because `ProtoOpToRepr` derives map inputs from `expr.args`, PyMilvus does not need to set
`op.inputs` for `map`; column args in the expression are enough.

`sort(by, desc=True, tie_break_col=None)` produces a `FunctionChainOp` with:

```text
op = "sort"
inputs = [by]
params = {"column": by, "desc": desc, "tie_break_col": tie_break_col?}
```

The current server `SortOp` reads the sort column from `params["column"]`; tests also set
`inputs = ["$score"]` so PyMilvus should set both for structural dependency tracking and
runtime construction.

`limit(limit, offset=0)` produces:

```text
op = "limit"
params = {"limit": limit, "offset": offset?}
```

### Args

```python
@dataclass(frozen=True)
class ColumnRef:
    name: str


def col(name: str) -> ColumnRef:
    ...
```

`col("ts")`, `col("score1")`, and `col("$score")` all compile to
`FunctionChainExprArg.column.name`. PyMilvus does not classify the name as a collection
field, temporary variable, or system value. That is server-side planning responsibility.

Bare strings are not accepted as positional expression args, because they are ambiguous:

```python
fn.num_combine("$score", mode="sum")  # invalid
```

Users must write:

```python
fn.num_combine(col("$score"), col("score1"), mode="sum")
```

For `sort`, bare strings are accepted because a sort key can only be a field, temporary
variable, or system name:

```python
.sort("$score", desc=True)
.sort(col("$score"), desc=True)
```

### FunctionChainExpr

```python
@dataclass(frozen=True)
class FunctionChainExpr:
    name: str
    args: Tuple[ColumnRef, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_proto(self) -> schema_pb2.FunctionChainExpr:
        ...
```

`FunctionChainExpr` is immutable data. It is not responsible for server-side semantic
validation such as whether a field exists, whether a temp variable has been defined, or
whether an expression is allowed in L2. It only validates local Python structure and
compiles to proto.

### fn Module

`fn` is implemented as a Python module containing ordinary factory functions. It is not a
class, not an instance, and not a dynamic namespace.

```python
def num_combine(*cols: ColumnRef, mode: str = "sum", weights: Optional[List[float]] = None) -> FunctionChainExpr:
    ...


def decay(
    value: ColumnRef,
    *,
    function: str,
    origin: Union[int, float],
    scale: Union[int, float],
    offset: Union[int, float] = 0,
    decay: float = 0.5,
) -> FunctionChainExpr:
    ...


def round_decimal(value: ColumnRef, *, decimal: int) -> FunctionChainExpr:
    ...


def rerank_model(value: ColumnRef, *, queries: List[str], **provider_params) -> FunctionChainExpr:
    ...
```

Generated server expression names:

- `fn.num_combine` sends `expr.name = "num_combine"`.
- `fn.decay` sends `expr.name = "decay"`.
- `fn.round_decimal` sends `expr.name = "round_decimal"`.
- `fn.rerank_model` sends `expr.name = "rerank_model"`.

## Data Model and Proto Compilation

### Operation Model

```python
@dataclass(frozen=True)
class FunctionChainOp:
    op: str
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    expr: Optional[FunctionChainExpr] = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_proto(self) -> schema_pb2.FunctionChainOp:
        ...
```

`FunctionChainOp.to_proto()` should fill the corresponding proto fields:

- `op`
- `expr`, when present
- `inputs`
- `outputs`
- `params`, recursively encoded as `FunctionParamValue`

### FunctionParamValue Encoding

PyMilvus must recursively encode params to `schema.FunctionParamValue`, not JSON strings.

| Python value | Proto field |
|--------------|-------------|
| `bool` | `bool_value` |
| `int` | `int64_value` |
| `float` | `double_value` |
| `str` | `string_value` |
| `bytes` | `bytes_value` |
| `list` / `tuple` | `array_value` |
| `dict` | `object_value` |

Implementation notes:

- Check `bool` before `int`, because `bool` is a Python `int` subclass.
- Support numpy scalar equivalents where PyMilvus already accepts numpy values:
  - `np.bool_`
  - `np.integer`
  - `np.floating`
- Reject `None`, because current proto has no null value.
- Reject unsupported objects with a `ParamError` that includes the key path when possible.

## Search Integration

### MilvusClient.search

Add `function_chains` to the client-level search API:

```python
def search(
    self,
    collection_name: str,
    data: Optional[Union[List[list], list]] = None,
    filter: str = "",
    limit: int = 10,
    output_fields: Optional[List[str]] = None,
    search_params: Optional[dict] = None,
    timeout: Optional[float] = None,
    partition_names: Optional[List[str]] = None,
    anns_field: Optional[str] = None,
    ranker: Optional[Union[Function, FunctionScore]] = None,
    function_chains: Optional[Union[FunctionChain, List[FunctionChain]]] = None,
    **kwargs,
) -> List[List[dict]]:
    ...
```

Forward `function_chains` to the gRPC handler and request preparation layer.

### Collection.search

Add the same optional argument:

```python
def search(
    self,
    data,
    anns_field,
    param,
    limit,
    expr=None,
    partition_names=None,
    output_fields=None,
    round_decimal=-1,
    timeout=None,
    ranker=None,
    function_chains=None,
    **kwargs,
):
    ...
```

### Prepare.search_requests_with_expr

Add `function_chains` to request preparation:

```python
def search_requests_with_expr(
    ...,
    ranker: Optional[Union[Function, FunctionScore]] = None,
    function_chains: Optional[Union[FunctionChain, List[FunctionChain]]] = None,
    ...,
) -> milvus_types.SearchRequest:
    ...
```

After constructing `SearchRequest`:

```python
if function_chains:
    if ranker is not None:
        raise ParamError(
            message="ranker/function_score and function_chains cannot be used together"
        )

    chains = _normalize_function_chains(function_chains)
    for chain in chains:
        request.function_chains.append(chain.to_proto())
elif isinstance(ranker, Function):
    request.function_score.CopyFrom(Prepare.ranker_to_function_score(ranker))
elif isinstance(ranker, FunctionScore):
    request.function_score.CopyFrom(Prepare.function_score_schema(ranker))
elif ranker is not None:
    raise ParamError(message="The search ranker must be a Function or FunctionScore.")
```

Normalization:

```python
def _normalize_function_chains(function_chains):
    if isinstance(function_chains, FunctionChain):
        return [function_chains]
    if not isinstance(function_chains, list):
        raise ParamError(
            message="function_chains must be a FunctionChain or a list of FunctionChain"
        )
    if not function_chains:
        return []
    if not all(isinstance(chain, FunctionChain) for chain in function_chains):
        raise ParamError(
            message="function_chains must be a FunctionChain or a list of FunctionChain"
        )
    return function_chains
```

### Hybrid Search

The first version rejects Function Chain for hybrid search:

```python
if kwargs.get("function_chains") is not None:
    raise ParamError(message="function_chains is not supported for hybrid_search yet")
```

This mirrors the server-side first-version scope. Hybrid Function Chain support requires a
separate design for sub-searches, merge/rank, global post-hybrid rerank, and final
projection.

## Validation Rules

PyMilvus should perform structural validation that produces clear local errors. It should
not duplicate server-owned semantic planning.

Client-side validation:

1. `FunctionChain.name` must be a string.
2. `FunctionChain.stage` must be convertible to `FunctionChainStage.L2_RERANK` for Search.
3. `map(output, expr)` requires non-empty string `output` and `FunctionChainExpr` `expr`.
4. `sort(by, desc=True)` requires a non-empty string or `ColumnRef`, boolean `desc`, and an
   optional non-empty `tie_break_col`.
5. `limit(limit, offset=0)` requires positive integer `limit` and non-negative integer `offset`.
6. `FunctionChainExpr.name` must be one of the server-supported expression names generated
   by the PyMilvus helpers.
7. Positional expression args must be `ColumnRef`; bare strings are rejected as ambiguous.
8. `params` must be recursively encodable to `FunctionParamValue`.
9. `ranker` / `FunctionScore` and `function_chains` are mutually exclusive.
10. `hybrid_search` rejects `function_chains` in the first version.

Server-side validation remains responsible for:

- Duplicate stages.
- Unsupported chain stages by request type.
- Unknown collection fields.
- Unsupported system references such as unknown `$xxx` names.
- Temporary variable read-before-write errors.
- Field type compatibility for each expression.
- Fetch/requery planning and final projection.

## L2 `$score` and System Name Semantics

For public Search L2 rerank chains, the server currently allows:

- readable system inputs: `$id`, `$score`
- writable system outputs: `$score` only

Users can read `$score`, rewrite it, and sort by it:

```python
FunctionChain(FunctionChainStage.L2_RERANK) \
    .map("freshness", fn.decay(col("ts"), function="linear", origin=now, scale=86400)) \
    .map("$score", fn.num_combine(col("$score"), col("freshness"), mode="sum")) \
    .sort(col("$score"), desc=True)
```

PyMilvus sends `$score` as `FunctionChainColumnArg.name = "$score"`. It should not add
`$score` to `output_fields`, and users should not need to request `$score` explicitly. The
final reranked value is returned through the existing search score/distance result field,
so users observe it as `Hit.distance` / score depending on the API surface.

Temporary values must not use the `$` system namespace:

```python
.map("tmp_score", fn.num_combine(col("$score"), col("ts"), mode="sum"))   # valid temp name
.map("$tmp_score", fn.num_combine(col("$score"), col("ts"), mode="sum"))  # invalid in L2 Search
```

## Proto Generation Requirement

Before implementing this API, PyMilvus generated protobuf files must be regenerated from a
Milvus proto version that contains:

- `schema.FunctionChainStage`
- `schema.FunctionChain`
- `schema.FunctionChainOp`
- `schema.FunctionChainExpr`
- `schema.FunctionChainExprArg`
- `schema.FunctionChainColumnArg`
- `schema.FunctionParamValue`
- `milvus.SearchRequest.function_chains`

Without regenerated `pymilvus/grpc_gen/*_pb2.py` and `*_pb2.pyi`, the client cannot build
or type-check Function Chain requests.

## Error Handling

Use `ParamError` for local input errors, consistent with existing PyMilvus request
preparation behavior.

Recommended messages:

- `function_chains must be a FunctionChain or a list of FunctionChain`
- `ranker/function_score and function_chains cannot be used together`
- `function_chains is not supported for hybrid_search yet`
- `column name must be a non-empty string`
- `String positional argument is ambiguous; use col(name) for a column reference`
- `function chain expression <name> is not supported by this PyMilvus version`
- `params must be a dict`
- `Duplicated function params: [...]`
- `Function Chain params do not support None`
- `Unsupported Function Chain param type at <path>: <type>`

## Testing

### Unit Tests

1. `FunctionParamValue` encoding:
   - `bool`
   - `int`
   - `float`
   - `str`
   - `bytes`
   - `list` / `tuple`
   - nested `dict`
   - numpy scalar values
   - `None` rejection
   - unsupported object rejection
2. `col` helper:
   - `col("ts")` creates a column arg.
   - `col("$score")` creates a system-name column arg.
   - empty or non-string column names are rejected.
3. `fn` factories:
   - `fn.num_combine(col("$score"), col("ts"), mode="sum")`
   - `fn.num_combine(col("a"), col("b"), mode="weighted", weights=[0.7, 0.3])`
   - `fn.decay(col("ts"), function="linear", origin=now, scale=86400)`
   - `fn.round_decimal(col("$score"), decimal=3)`
   - `fn.rerank_model(col("doc"), queries=[...], provider="...")`
   - bare string positional arg rejection
4. `FunctionChain` builder:
   - `map` emits `op="map"`, output, and expr.
   - `sort(col("$score"))` emits `op="sort"`, input, and typed `column`/`desc` params.
   - `limit(2)` emits `op="limit"` and typed `limit` param.
   - method chaining preserves operation order.
5. `SearchRequest` preparation:
   - single `FunctionChain` is accepted.
   - list of `FunctionChain` is accepted.
   - function chains populate `request.function_chains`.
   - `ranker` plus `function_chains` is rejected.
   - invalid `function_chains` input is rejected.
6. Hybrid search:
   - `function_chains` is rejected with a clear `ParamError`.

### Integration Tests

When a Milvus server with Function Chain support is available:

1. `$score`-only chain rewrites score and changes result distance/score.
2. Chain requiring a scalar field succeeds even when that field is not in user
   `output_fields`.
3. Temporary variables such as `tmp_score` are not returned as output fields.
4. Explicit `sort(col("$score"), desc=True)` controls output order.
5. `limit(2)` trims per-query search results.
6. Existing `ranker` / `FunctionScore` search behavior remains unchanged.

## Open Questions

1. Should PyMilvus top-level export `col` and `fn` directly, or only expose
   `from pymilvus.function_chain import col, fn`?
2. Should `FunctionChain` be mutable as proposed, or should builder methods return a new
   chain for immutability? The current recommendation is mutable to match common PyMilvus
   builder style and keep the fluent API simple.
3. Should PyMilvus add Python-only convenience aliases such as `fn.sum` or `fn.weighted`
   later? The current recommendation is no: expose only server expression names first.
4. Should unsupported but registered runtime ops (`filter`, `select`, `group_by`) remain
   completely hidden, or be exposed as experimental only after targeted server-side tests are
   added for public Search L2 `function_chains`?
