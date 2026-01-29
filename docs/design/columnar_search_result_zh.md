# Columnar Search Result

> silas.jiang · 2026-01-29 · 已实现

## 背景

我们在 profiling 大 TopK 搜索时发现，`SearchResult.__init__` 占用了相当多的时间。根本原因是它在初始化时就把所有 Hit 对象都创建好了——对于 `nq=100, topk=10000` 的场景，这意味着要预创建 100 万个 Python 对象。

这个设计在早期 TopK 较小时没什么问题，但随着用户场景变复杂，它成了客户端的性能瓶颈。

## 核心思路

换成列式存储，延迟创建对象。

具体来说：
1. 初始化时只存 Protobuf 的引用，不遍历数据
2. 用户访问 `result[0][5]` 时才创建一个轻量的 RowProxy
3. 访问 `hit['field']` 时才真正从 Protobuf 里取值

这样初始化复杂度从 O(nq × topk) 降到 O(nq)。

## 整体结构

```
ColumnarSearchResult (继承 list)
    │
    └── ColumnarHits (每个 query 一个)
            │
            ├── ids, distances (惰性切片)
            │
            └── get_value(field, idx)
                    │
                    └── AccessorFactory → 各类型的 Accessor
```

**ColumnarSearchResult** 是顶层容器，持有原始 Protobuf 数据的引用。初始化时只做一件事：根据 `topks` 数组创建 nq 个 ColumnarHits，每个记录自己的 `[start, end)` 范围。

**ColumnarHits** 存储单个 query 的结果。`ids` 和 `distances` 是 lazy property，首次访问时才做切片。字段访问走 `get_value()`，内部通过 AccessorFactory 创建对应类型的 accessor。

**RowProxy** 是个极轻量的对象（只有 3 个字段），实现了 `Mapping` 接口。所有数据访问都委托给 ColumnarHits，它自己不存任何数据。

**AccessorFactory** 用策略模式，根据 DataType 选择合适的 accessor。比如 `FLOAT_VECTOR` 需要按 dim 切片，`JSON` 需要 orjson 解析，`VARCHAR` 直接索引就行。

## 为什么这么设计

**为什么用 RowProxy 而不是直接返回 dict？**

主要是为了兼容性。现有代码大量使用 `hit.id`、`hit.entity['field']` 这种写法，dict 做不到。RowProxy 通过 `__getattr__` 和 `__getitem__` 模拟了原来 Hit 的所有访问方式。

**为什么 Accessor 要缓存？**

创建 accessor 需要从 Protobuf 里提取 payload（比如 `scalars.long_data.data`），这个操作有一定开销。缓存之后，同一个字段的后续访问就是 O(1) 的数组索引。

**为什么 ColumnarHits 之间要共享 `fields_data_map`？**

Protobuf 里的 `fields_data` 是平铺的，所有 query 的数据连在一起。把它转成 `{field_name: field_data}` 的 map 只需要做一次，没必要每个 ColumnarHits 都做。

## 类型支持

向量类型：FLOAT_VECTOR、BINARY_VECTOR、FLOAT16、BFLOAT16、INT8、SPARSE 都支持。

标量类型：常规的 BOOL、INT8/16/32/64、FLOAT、DOUBLE、VARCHAR 都有。

复杂类型：JSON 用 orjson 解析，ARRAY 走 entity_helper。动态字段存在 `$meta` JSON 里，访问时按需解析。

Nullable 字段通过 `NullableAccessor` 包装处理，检查 `valid_data` 位图后返回值或 None。

## 新增的 API

除了完全兼容原有接口，还加了 `get_column()` 方法：

```python
ids = hits.get_column("id")                           # List[int]
scores = hits.get_column("score", return_type="numpy")  # np.ndarray
vectors = hits.get_column("embedding", return_type="numpy")  # shape: (n, dim)
```

这个在需要批量处理数据时很有用，比如直接喂给 numpy 或 pandas。

## 兼容性

API 层面完全兼容，现有代码不需要改动：

```python
for hits in result:
    for hit in hits:
        print(hit.id, hit.distance)
        print(hit['field'])
        print(hit.entity.field)
```

**唯一的 breaking change**：结果是只读的。`hit['field'] = value` 会抛 `TypeError`。实际上应该没人这么用，搜索结果本来就不该被修改。

## 集成

改动很小，只需要把 GrpcHandler 里的 `SearchResult` 换成 `ColumnarSearchResult`：

```python
# grpc_handler.py
- from .search_result import SearchResult
+ from .columnar_search_result import ColumnarSearchResult
```

ORM 层不需要任何改动。

## 后续计划

- 支持 `to_arrow()` 和 `to_pandas()`，方便和数据处理框架集成
- 考虑加 `get_columns()` 批量获取多列

## 相关文件

- `pymilvus/client/columnar_search_result.py` - 实现
- `tests/test_columnar_search_result.py` - 单测
- `tests/test_columnar_compat.py` - 兼容性测试
