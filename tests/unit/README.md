# pymilvus Unit Test Guidelines

Rules for writing concise, maintainable unit tests. Follow these to keep test files short.

## 1. Use conftest.py Fixtures

`conftest.py` provides shared fixtures and helpers. **Always check before writing setup code.**

Available fixtures:
- `mock_grpc_handler` / `mock_async_grpc_handler` -- fully wired GrpcHandler/AsyncGrpcHandler
- `mock_grpc_stub` / `mock_async_stub` -- gRPC stubs with all methods pre-mocked
- `mock_milvus_client` -- yields `(client, handler)` tuple with mocked connection
- `mock_milvus_client_handler` -- standalone mock handler for MilvusClient
- `mock_grpc_channel` / `mock_async_channel` -- mock channels
- `mock_field_data_int64`, `mock_field_data_array`, `mock_field_data_float_vector`, `mock_collection_schema` -- proto fixtures

Available helpers:
- `make_status(code, reason, error_code)` -- mock status response
- `make_response(status_code, reason, **kwargs)` -- mock gRPC response with arbitrary fields
- `make_collection_schema_response(...)` -- mock DescribeCollection response
- `make_mutation_response(...)` -- mock insert/upsert/delete response
- `make_search_response(...)` / `make_query_response(...)` -- mock search/query responses

```python
# WRONG: inline setup (5+ lines per test)
def test_something(self):
    handler = MagicMock()
    handler.get_server_type.return_value = "milvus"
    handler._wait_for_channel_ready = MagicMock()
    with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
        client = MilvusClient()
        ...

# RIGHT: use conftest fixture (0 setup lines)
def test_something(self, mock_milvus_client):
    client, handler = mock_milvus_client
    ...
```

If a file needs a specialized mock that conftest doesn't cover, define a **file-level `_make_X()` factory** or `@pytest.fixture` once, then reuse it everywhere in that file.

## 2. Parametrize Instead of Copy-Paste

If two or more tests share the same structure and differ only in inputs/expected values, use `@pytest.mark.parametrize`.

```python
# WRONG: N tests x M lines each
def test_is_legal_host_valid(self):
    assert is_legal_host("localhost") is True

def test_is_legal_host_empty(self):
    assert is_legal_host("") is False

def test_is_legal_host_none(self):
    assert is_legal_host(None) is False

# RIGHT: 1 test, N rows
@pytest.mark.parametrize("host,expected", [
    ("localhost", True),
    ("", False),
    (None, False),
])
def test_is_legal_host(self, host, expected):
    assert is_legal_host(host) is expected
```

## 3. Use Delegation Tables for Simple Call-and-Assert Tests

When testing that a client method delegates to the handler, use a parametrized table instead of one test per method.

```python
_SIMPLE_DELEGATION_CASES = [
    # (client_method, args, kwargs, handler_method)
    ("drop_collection", ("col",), {}, "drop_collection"),
    ("has_collection", ("col",), {}, "has_collection"),
    ("create_partition", ("col", "part"), {}, "create_partition"),
    ...
]

class TestDelegation:
    @pytest.mark.parametrize(
        "method,args,kwargs,handler_method",
        _SIMPLE_DELEGATION_CASES,
        ids=[c[0] for c in _SIMPLE_DELEGATION_CASES],
    )
    def test_delegation(self, method, args, kwargs, handler_method, mock_milvus_client):
        client, handler = mock_milvus_client
        getattr(client, method)(*args, **kwargs)
        getattr(handler, handler_method).assert_called_once()
```

Adding a new method? Add one row to the table, not a new test method.

## 4. Use `@pytest.mark.asyncio` for Async Tests

```python
# WRONG: manual event loop
def test_async_op(self):
    async def go():
        result = await client.search(...)
        assert result == expected
    asyncio.run(go())

# RIGHT: native async test
@pytest.mark.asyncio
async def test_async_op(self):
    result = await client.search(...)
    assert result == expected
```

## 5. Extract Repeated Inner Classes to Module Level

If the same helper class (e.g., `MockRpcError`) appears in multiple test methods, define it once at module level.

```python
# WRONG: defined inside each method
def test_recover(self):
    class MockRpcError(grpc.RpcError):
        def code(self): return grpc.StatusCode.UNAVAILABLE
    ...

def test_handle_error(self):
    class MockRpcError(grpc.RpcError):   # same thing again
        def code(self): return grpc.StatusCode.UNAVAILABLE
    ...

# RIGHT: once at module level
class _MockRpcError(grpc.RpcError):
    def code(self): return grpc.StatusCode.UNAVAILABLE
```

## 6. Keep Test Classes Meaningful

Don't create a class for 1-2 tests. Either:
- Add them to an existing thematic class
- Use top-level functions

Merge small classes that test the same component.

## 7. Don't Duplicate Coverage

Before writing a new test, check:
- Is this case already covered by a parametrized table?
- Does another test class already exercise this code path?
- Is this a strict subset of an existing test?

If yes, don't write it.

## Quick Checklist

Before submitting a test file, verify:

- [ ] No inline setup that conftest fixtures already provide
- [ ] No 2+ tests with identical structure (parametrize them)
- [ ] No helper classes/functions duplicated across methods
- [ ] No test classes with only 1 trivial test method
- [ ] No `asyncio.run()` wrappers (use `@pytest.mark.asyncio`)
- [ ] No tests that duplicate coverage of existing parametrized tables
