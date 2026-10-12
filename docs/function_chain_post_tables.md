# FunctionChain POST tables

This experimental extension returns an independent table from Search,
HybridSearch or Query. It requires matching server, protocol and SDK revisions.
An ordinary search still returns `SearchResult`; an explicit POST TABLE call
returns `TableResult` (`FunctionChainTableResult`).

From the matching SDK checkout, install the optional Arrow dependency before
using TABLE mode:

```sh
pip install -e '.[arrow]'
```

The SDK checks `GetVersion.status.extra_info["function_chain_post_table_version"]`
equals `"1"`, and requires `post_process_result` in every TABLE response. A
Proxy that ignores the new option cannot silently return ordinary hits or query
rows. Upgrade every serving Proxy; a capability response from one node does
not prove all nodes behind a load balancer support TABLE. A Python model also
requires the enabled PyUDF runtime/worker and registered wheel resource.

## Query expansion before Search

PRE_PROCESS accepts `flat_map` and `explode`: one original query can produce
zero, one or many prepared subqueries. `map` still keeps the current row count;
use `map` returning a List followed by `explode`, or the combined `flat_map`.
Bind the expanded query values to `query` and retain it in the final projection.
Every generated row retains its original query index, and Search uses the
prepared query count. A branch whose total prepared count is zero is rejected
before dispatch; an individual original query may produce no subqueries.

```python
from pymilvus import FunctionChain, FunctionChainStage
from pymilvus.function_chain import agg, col, fn

pre = (
    FunctionChain(FunctionChainStage.PRE_PROCESS)
    .flat_map(
        "query",
        fn.py_udf(col("query"), resource_name="query_expander"),
    )
    .map("query", fn.py_udf(col("query"), resource_name="embedder"))
    .select(["query"])
)
post = (
    FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table")
    .union_all()
    .group_by("$query_index")
    .aggregate({
        "ids": agg.collect_list(col("$id")),
        "scores": agg.collect_list(col("$score")),
        "subqueries": agg.collect_list(col("$subquery_index")),
    })
)
result = client.search(
    "documents", data=["compare both systems", "explain indexing"],
    anns_field="vector", limit=10, function_chains=[pre, post],
)
```

For example, prepared queries `[q0a, q0b, q1a]` have original query indices
`[0, 0, 1]` and prepared subquery indices `[0, 1, 2]`. POST `$query_index`
identifies the original request row; `$subquery_index` identifies the prepared
query within that ANN input. `union_all().group_by("$query_index").aggregate(...)`
combines the hits for each original question. Aggregate collection helpers retain
duplicate IDs; use a custom aggregate for deduplication or score fusion.

The SDK requires
`GetVersion.status.extra_info["function_chain_pre_query_expansion_version"] == "1"`
only when a PRE chain contains `flat_map` or `explode`, including nested Hybrid
ANN chains. Existing PRE `map`/`select` calls keep their prior capability check.
This check runs before Search/HybridSearch dispatch for synchronous, asynchronous
and Future APIs. Upgrade all serving Proxies to the matching revision.

A nonidentity expansion currently requires explicit POST TABLE. Ordinary HITS
does not silently expose prepared NQ as if it were the original NQ. Hybrid
nonidentity expansion requires top-level POST TABLE with `input="ann_inputs"`;
ANN branches can have different prepared counts and query mappings. It does not
support a top-level ranker/L2 chain in this mode. Each branch's search filter,
parameters and limit apply to all subqueries produced by that branch; generating
per-subquery filters or additional search requests is outside this interface.

UDF `context.query_count` remains original NQ. `context.original_queries` retains
the original ANN inputs; `context.queries` contains prepared ANN inputs (the
stage-entry snapshot during PRE and the final PRE output during POST).
`context.prepared_query_counts` lists prepared counts by ANN input, and
`context.query_origins` maps each prepared query back to its original query.
During PRE, `context.row_query_origins` maps the current operator rows. These
fields are request data; they do not provide a Milvus client or connector API.

See [function_chain_pre_query_expansion.py](../examples/function_chain_pre_query_expansion.py)
for a deterministic query expansion example using the matching transform demo
wheel, without an external model API.

## Cross-query aggregation

POST starts with one logical group per prepared query (equal to original NQ
when PRE does not expand it). Aggregation normally
operates within each current group. Explicit `union_all()` combines all groups
in stable order and retains duplicate rows; it does not deduplicate IDs.
It follows SQL `UNION ALL` semantics, rather than the duplicate elimination
performed by SQL `UNION`. The input groups have the same runtime schema.

```python
from pymilvus import FunctionChain, FunctionChainStage
from pymilvus.function_chain import agg, col

post = (
    FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table")
    .union_all()
    .group_by("tenant")
    .aggregate({
        "matches": agg.count(),
        "texts": agg.collect_list(col("text")),
        "queries": agg.collect_list(col("$query_index")),
    })
)
result = client.search(
    "documents", data=query_vectors, anns_field="vector", limit=20,
    function_chains=post,
)
print(result.schema, result.row_count, result.rows)
table = result.to_arrow()
```

`group_by(...).aggregate(...)` emits one native `aggregate` operation, separate
from hit-retaining rerank grouping. Keys can be strings or `col(...)`; optional
`aliases=[...]` renames the grouping columns. Aggregate aliases must be unique
and cannot overwrite the keys. Available helpers are `count`, `sum`, `avg`,
`min`, `max`, `first`, `last` and `collect_list`. `count()` counts rows;
`count(col(...))` counts non-null values. `first`/`last` include null values at
their selected position. `collect_list` retains nulls and input order.

An unkeyed empty aggregate emits one row (`count=0`, `collect_list=[]`, other
empty aggregate values are null). A keyed empty aggregate emits zero typed
rows. Null grouping keys form a group, and groups follow first occurrence.
Without `union_all`, repeated keys in different query groups stay separate.

`$query_index` is the original query index; `$subquery_index` is the prepared
query index, and `$ann_index` is `-1` for FUSED input. These source columns are
readable/projectable and cannot be overwritten.
Keys can retain them after concatenation when separate query/route results are
needed. TABLE output does not automatically include IDs or scores; explicitly
select/aggregate `col("$id")` or `col("$score")` to retain them as data.

## Hybrid inputs before fusion

Use `input="ann_inputs"` to process every reduced ANN branch before top-level
fusion and the final Hybrid limit:

```python
post = (
    FunctionChain(
        FunctionChainStage.POST_PROCESS, result_mode="table", input="ann_inputs"
    )
    .union_all()
    .group_by("$ann_index")
    .aggregate({"matches": agg.count(), "ids": agg.collect_list(col("$id"))})
)
result = client.hybrid_search("documents", ann_requests, function_chains=post)
```

Each `AnnSearchRequest.limit` bounds that branch's input. Groups are ordered by
ANN branch, then prepared query, including empty groups. `$ann_index`,
`$query_index` and `$subquery_index` identify their sources. The same ID in two branches remains two
rows with its branch-specific score. This mode rejects top-level rankers and
L2 chains. Branch PRE/L0/L1 chains remain valid. FUSED is the default and uses
the ordinary final candidate pool after optional ranker/L2 processing.

## Query tables

Query accepts POST TABLE with FUSED input. The existing filter, limit and offset
bound its input; a POST aggregate does not scan every matching stored row.
All three source indices are `-1`.

```python
post = (
    FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table")
    .group_by("tenant")
    .aggregate({"documents": agg.count(), "texts": agg.collect_list(col("text"))})
)
result = client.query("documents", filter="", limit=100, function_chains=post)
```

Query TABLE currently rejects native Query aggregation and Query Iterator.
Search TABLE rejects `search_aggregation`, search-by-IDs and iterators.
ANN_INPUTS is only supported on top-level HybridSearch. These combinations
fail instead of returning a different result shape.

## A Python aggregate with an independent nested schema

```python
from pymilvus.function_chain import fn

post = (
    FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table")
    .union_all()
    .aggregate(
        "summary",
        fn.py_udf(
            col("text"), resource_name="transform_demo_table_v3",
            udf_params={"mode": "summary"},
        ),
    )
    .select(["summary"])
)
```

The current transform_demo 0.5 wheel's summary returns a Struct containing `texts`
(List of strings), `count` and `query_count`. Arrow IPC preserves this nested
runtime schema without binding it to collection fields or converting it into
a hit. `.rows`/`.to_rows()` materialize Python dictionaries;
`.to_arrow()` retains Arrow types and metadata. `.to_pandas()` is also available.
Typed empty/all-null results retain their schema.
TABLE uses underlying Arrow values; the ordinary hits/query display formatting
for geometry and time fields is not applied to an independent output schema.

For a custom aggregate UDF, implement the POST `transform_batch(params, batches,
*, context)` callable. Each input batch produces exactly one row, and
`context.table_result` and `context.reduction` are true. `context.query_count`
is original NQ (zero for Query); it is independent of the current group count.
`context.result_offsets` describes the batches for this call. Grouped custom
aggregation uses `.group_by(...).aggregate(outputs, expr)` and keeps the keys beside the UDF
outputs. Output aliases may be a string or a list of strings. Native aggregation
uses `.aggregate({alias: agg.count(), ...})`; both forms emit the same `aggregate`
operation. A keyed empty custom aggregation invokes the UDF once on typed empty
input to establish its output schema, then discards that schema-probe row.

TABLE map uses the same batch callable with `context.reduction=False` and
preserves each input batch's row count. Projection, filter, sort, limit and list
expansion are available in TABLE pipelines. HITS remains the default POST mode
and preserves candidate IDs, scores, query boundaries and cardinality.

## Running the example

[function_chain_post_tables.py](../examples/function_chain_post_tables.py)
creates three deterministic rows and exercises cross-NQ grouping, ANN inputs,
Query, empty aggregates and unchanged ordinary hits:

```sh
python examples/function_chain_post_tables.py --uri http://127.0.0.1:19530
```

To include Python aggregation, register the transform_demo 0.5 wheel and pass its
resource name. `--wheel-path` registers an object-storage key, not a local file:

```sh
python examples/function_chain_post_tables.py \
  --uri http://127.0.0.1:19530 \
  --collection function_chain_post_table_demo_v3 \
  --resource-name transform_demo_table_v3 \
  --wheel-path examples/milvus_transform_demo-0.5.0-py3-none-any.whl
```

Upload the wheel beforehand using your deployment's object-storage workflow.
The script keeps its collection; use `--reuse` to rerun against its existing
three rows. It uses deterministic vectors and makes no paid model API calls.
Synchronous clients, `AsyncMilvusClient`, and Search/Hybrid `SearchFuture`
calls all use the same table decoder. There is no QueryFuture API.

## Recorded local validation

After unifying native/custom aggregation under `aggregate` and renaming the
group combination operation to `union_all`, the full SDK unit suite passed with
5407 passed and 3 skipped (26.43s). Tests cover operation serialization,
native/UDF input occurrence hints, multi-column UDF output aliases, grouping,
stage restrictions, and rejection of the removed `reduce`/`concat` APIs and
wire names. Evidence:
`/tmp/milvus-function-chain-unified-aggregate-union-all-sdk-final-tests.log`.
This SDK validation does not rerun the standalone server example below.

The full PyMilvus unit suite passed with 5380 passed and 3 skipped (26.08s),
including capability/response checks for synchronous, coroutine and Future
paths, malformed nested UTF-8/IPC/schema data, nullability and ordinary import
without Arrow. Evidence: `/tmp/milvus-function-chain-post-table-sdk-full-tests.log`.

The complete example passed against the matching final local server, including
the TABLE sort and empty ANN source fixes: cross-NQ groups, branch counts 6/4 despite a top-level Hybrid limit of
1, Query groups, empty global/keyed aggregates, legacy hits and nested Python
summary output. Evidence: `/tmp/milvus-function-chain-post-table-e2e/sdk-demo-empty-fix-final.log`.
These are local checks, not a published release or a complete Milvus repository
test result.

## Request scope and logging

The current matching worker is runtime 0.6.0. In addition to Arrow schema and
query metadata, each call receives `context.database`, `context.collection`,
`context.user`, `context.resource_name` and a request-owned `context.logger`.
Scope is supplied by the server; `user` is empty when authentication is disabled
or the call has no verified principal. Alias calls use the canonical collection.
UDF params cannot override scope. Operator log labels are separate from the
registered wheel resource used by factory initialization. Existing Go
context/gRPC controls deadline and cancellation. No hosted connector is added.
