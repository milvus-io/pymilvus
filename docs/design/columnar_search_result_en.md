# Columnar Search Result

> silas.jiang · 2026-01-29 · Implemented

## Background

While profiling large TopK searches, we noticed that `SearchResult.__init__` was taking a significant amount of time. The root cause is that it pre-creates all Hit objects during initialization—for a `nq=100, topk=10000` scenario, that means creating 1 million Python objects upfront.

This design was fine when TopK was small, but as user scenarios became more complex, it turned into a client-side performance bottleneck.

## Core Idea

Switch to columnar storage with lazy object creation.

Specifically:
1. During initialization, just store references to the Protobuf—don't iterate through the data
2. When the user accesses `result[0][5]`, create a lightweight RowProxy on demand
3. When accessing `hit['field']`, that's when we actually extract the value from Protobuf

This reduces initialization complexity from O(nq × topk) to O(nq).

## Overall Structure

```
ColumnarSearchResult (extends list)
    │
    └── ColumnarHits (one per query)
            │
            ├── ids, distances (lazy slicing)
            │
            └── get_value(field, idx)
                    │
                    └── AccessorFactory → type-specific Accessors
```

**ColumnarSearchResult** is the top-level container that holds a reference to the raw Protobuf data. During initialization, it does one thing: create nq ColumnarHits based on the `topks` array, each recording its own `[start, end)` range.

**ColumnarHits** stores results for a single query. `ids` and `distances` are lazy properties—slicing happens only on first access. Field access goes through `get_value()`, which internally uses AccessorFactory to create the appropriate type-specific accessor.

**RowProxy** is an extremely lightweight object (just 3 fields) that implements the `Mapping` interface. All data access is delegated to ColumnarHits—it doesn't store any data itself.

**AccessorFactory** uses the strategy pattern to select the right accessor based on DataType. For example, `FLOAT_VECTOR` needs to slice by dimension, `JSON` requires orjson parsing, while `VARCHAR` can just index directly.

## Why This Design

**Why use RowProxy instead of just returning a dict?**

Mainly for compatibility. Existing code extensively uses patterns like `hit.id` and `hit.entity['field']`—a plain dict can't do that. RowProxy simulates all the original Hit access patterns through `__getattr__` and `__getitem__`.

**Why cache Accessors?**

Creating an accessor requires extracting the payload from Protobuf (e.g., `scalars.long_data.data`), which has some overhead. After caching, subsequent accesses to the same field become O(1) array indexing.

**Why share `fields_data_map` across ColumnarHits?**

The `fields_data` in Protobuf is flat—data from all queries is concatenated together. Converting it to a `{field_name: field_data}` map only needs to happen once; there's no need for each ColumnarHits to do it separately.

## Type Support

Vector types: FLOAT_VECTOR, BINARY_VECTOR, FLOAT16, BFLOAT16, INT8, and SPARSE are all supported.

Scalar types: The usual BOOL, INT8/16/32/64, FLOAT, DOUBLE, VARCHAR are covered.

Complex types: JSON uses orjson for parsing, ARRAY goes through entity_helper. Dynamic fields are stored in the `$meta` JSON and parsed on demand.

Nullable fields are handled via a `NullableAccessor` wrapper that checks the `valid_data` bitmap before returning the value or None.

## New API

Besides being fully compatible with the existing interface, we added a `get_column()` method:

```python
ids = hits.get_column("id")                             # List[int]
scores = hits.get_column("score", return_type="numpy")  # np.ndarray
vectors = hits.get_column("embedding", return_type="numpy")  # shape: (n, dim)
```

This is useful when you need to process data in bulk, like feeding it directly to numpy or pandas.

## Compatibility

Fully API-compatible. Existing code doesn't need any changes:

```python
for hits in result:
    for hit in hits:
        print(hit.id, hit.distance)
        print(hit['field'])
        print(hit.entity.field)
```

**The only breaking change**: results are read-only. `hit['field'] = value` will raise `TypeError`. In practice, nobody should be doing this anyway—search results shouldn't be modified.

## Integration

The change is minimal. Just swap `SearchResult` for `ColumnarSearchResult` in GrpcHandler:

```python
# grpc_handler.py
- from .search_result import SearchResult
+ from .columnar_search_result import ColumnarSearchResult
```

The ORM layer doesn't require any changes.

## Future Work

- Support `to_arrow()` and `to_pandas()` for easier integration with data processing frameworks
- Consider adding `get_columns()` for batch multi-column retrieval

## Related Files

- `pymilvus/client/columnar_search_result.py` - Implementation
- `tests/test_columnar_search_result.py` - Unit tests
- `tests/test_columnar_compat.py` - Compatibility tests
