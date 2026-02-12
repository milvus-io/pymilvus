"""Benchmarks for general performance improvements in pymilvus.

Each benchmark compares the old (before) implementation against the new (after)
implementation to demonstrate the performance improvement.

Run with:
    pytest tests/benchmark/test_perf_improvements.py -v --benchmark-sort=name
"""

import logging
import struct
import time
import traceback
from copy import deepcopy
from typing import Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. sparse_float_row_to_bytes: bytes concat vs struct.pack batch
# ---------------------------------------------------------------------------
def _sparse_row_old(indices, values):
    """Old: O(n^2) bytes concatenation in loop."""
    data = b""
    for i, v in sorted(zip(indices, values), key=lambda x: x[0]):
        data += struct.pack("I", i)
        data += struct.pack("f", v)
    return data


def _sparse_row_new(indices, values):
    """New: single struct.pack call with batch format."""
    sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
    n = len(sorted_pairs)
    fmt = f"<{'If' * n}"
    flat = []
    for i, v in sorted_pairs:
        flat.append(i)
        flat.append(v)
    return struct.pack(fmt, *flat)


@pytest.mark.benchmark(group="sparse_row_to_bytes")
class TestSparseRowToBytes:
    SIZES: tuple = (10, 100, 1000)

    @pytest.fixture(params=SIZES)
    def sparse_data(self, request):
        n = request.param
        indices = list(range(n))
        values = [float(i) * 0.1 for i in range(n)]
        return indices, values

    def test_old_bytes_concat(self, benchmark, sparse_data):
        indices, values = sparse_data
        benchmark(_sparse_row_old, indices, values)

    def test_new_struct_pack_batch(self, benchmark, sparse_data):
        indices, values = sparse_data
        result = benchmark(_sparse_row_new, indices, values)
        expected = _sparse_row_old(indices, values)
        assert result == expected


# ---------------------------------------------------------------------------
# 2. convert_to_array_of_vector: bytes concat vs b"".join
# ---------------------------------------------------------------------------
def _bytes_concat_old(chunks):
    """Old: O(n^2) bytes concatenation."""
    all_bytes = b""
    for chunk in chunks:
        all_bytes += chunk
    return all_bytes


def _bytes_concat_new(chunks):
    """New: b''.join (O(n))."""
    return b"".join(chunks)


@pytest.mark.benchmark(group="bytes_concat")
class TestBytesConcat:
    SIZES: tuple = (10, 100, 1000)

    @pytest.fixture(params=SIZES)
    def byte_chunks(self, request):
        n = request.param
        return [b"\x00" * 128 for _ in range(n)]

    def test_old_bytes_concat(self, benchmark, byte_chunks):
        benchmark(_bytes_concat_old, byte_chunks)

    def test_new_bytes_join(self, benchmark, byte_chunks):
        result = benchmark(_bytes_concat_new, byte_chunks)
        expected = _bytes_concat_old(byte_chunks)
        assert result == expected


# ---------------------------------------------------------------------------
# 3. get_params: deepcopy vs dict()
# ---------------------------------------------------------------------------
def _get_params_old(search_params):
    params = deepcopy(search_params.get("params", {}))
    for key, value in search_params.items():
        if key in params:
            pass
        elif key != "params":
            params[key] = value
    return params


def _get_params_new(search_params):
    params = dict(search_params.get("params", {}))
    for key, value in search_params.items():
        if key in params:
            pass
        elif key != "params":
            params[key] = value
    return params


@pytest.mark.benchmark(group="get_params")
class TestGetParams:
    @pytest.fixture
    def search_params(self):
        return {
            "metric_type": "L2",
            "params": {"nprobe": 10, "ef": 64},
            "offset": 0,
            "limit": 100,
        }

    def test_old_deepcopy(self, benchmark, search_params):
        benchmark(_get_params_old, search_params)

    def test_new_dict_copy(self, benchmark, search_params):
        result = benchmark(_get_params_new, search_params)
        expected = _get_params_old(search_params)
        assert result == expected


# ---------------------------------------------------------------------------
# 4. Membership check: list vs frozenset
# ---------------------------------------------------------------------------
_valid_index_types_list = [
    "GPU_IVF_FLAT",
    "GPU_IVF_PQ",
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_PQ",
    "HNSW",
    "BIN_FLAT",
    "BIN_IVF_FLAT",
    "DISKANN",
    "AUTOINDEX",
    "GPU_CAGRA",
    "GPU_BRUTE_FORCE",
]
_valid_index_types_frozenset = frozenset(_valid_index_types_list)


def _membership_list(items, container):
    count = 0
    for item in items:
        if item in container:
            count += 1
    return count


@pytest.mark.benchmark(group="membership_check")
class TestMembershipCheck:
    @pytest.fixture
    def test_items(self):
        return [
            "HNSW",
            "MISSING1",
            "FLAT",
            "MISSING2",
            "GPU_CAGRA",
            "MISSING3",
            "IVF_PQ",
            "MISSING4",
            "DISKANN",
            "MISSING5",
        ] * 100

    def test_old_list_membership(self, benchmark, test_items):
        benchmark(_membership_list, test_items, _valid_index_types_list)

    def test_new_frozenset_membership(self, benchmark, test_items):
        result = benchmark(_membership_list, test_items, _valid_index_types_frozenset)
        expected = _membership_list(test_items, _valid_index_types_list)
        assert result == expected


# ---------------------------------------------------------------------------
# 5. len_of: cascading HasField vs WhichOneof
# ---------------------------------------------------------------------------
class MockScalarData:
    def __init__(self, kind, length):
        self._kind = kind
        self._data = MagicMock()
        self._data.data = list(range(length))
        self._fields = {kind: True}

    def HasField(self, name):
        return name in self._fields

    def WhichOneof(self, name):
        return self._kind

    def __getattr__(self, name):
        if name == self._kind:
            return self._data
        raise AttributeError(name)


class MockFieldData:
    def __init__(self, kind="int_data", length=100):
        self.scalars = MockScalarData(kind, length)
        self.valid_data = []
        self._field = "scalars"

    def HasField(self, name):
        return name == self._field

    def WhichOneof(self, name):
        return self._field


_SCALAR_KINDS = [
    "bool_data",
    "int_data",
    "long_data",
    "float_data",
    "double_data",
    "string_data",
    "bytes_data",
    "json_data",
    "array_data",
]


def _len_of_old(field_data):
    """Old: cascading HasField calls."""
    if field_data.HasField("scalars"):
        for kind in _SCALAR_KINDS:
            if field_data.scalars.HasField(kind):
                return len(field_data.scalars.__getattr__(kind).data)
    return 0


def _len_of_new(field_data):
    """New: WhichOneof dispatch."""
    field_kind = field_data.WhichOneof("field")
    if field_kind == "scalars":
        scalar_kind = field_data.scalars.WhichOneof("data")
        if scalar_kind is None:
            return 0
        return len(getattr(field_data.scalars, scalar_kind).data)
    return 0


@pytest.mark.benchmark(group="len_of")
class TestLenOf:
    KINDS: tuple = ("int_data", "json_data", "array_data")

    @pytest.fixture(params=KINDS)
    def field_data(self, request):
        return MockFieldData(kind=request.param, length=1000)

    def test_old_cascading_hasfield(self, benchmark, field_data):
        benchmark(_len_of_old, field_data)

    def test_new_whichoneof(self, benchmark, field_data):
        result = benchmark(_len_of_new, field_data)
        expected = _len_of_old(field_data)
        assert result == expected


# ---------------------------------------------------------------------------
# 6. error_handler: datetime.now + traceback on happy path vs deferred
# ---------------------------------------------------------------------------
import datetime


def _error_handler_old_happy_path(func):
    """Old: creates datetime.now() and dict on every call."""
    record_dict = {}
    record_dict["RPC start"] = str(datetime.datetime.now())
    return func()


def _error_handler_new_happy_path(func):
    """New: only time.monotonic() on happy path."""
    _start_ts = time.monotonic()
    return func()


@pytest.mark.benchmark(group="error_handler_happy_path")
class TestErrorHandlerHappyPath:
    def test_old_datetime_now(self, benchmark):
        benchmark(_error_handler_old_happy_path, lambda: 42)

    def test_new_monotonic(self, benchmark):
        result = benchmark(_error_handler_new_happy_path, lambda: 42)
        assert result == 42


# ---------------------------------------------------------------------------
# 7. error_handler: traceback.format_exc with vs without log level check
# ---------------------------------------------------------------------------
def _error_handler_old_error_path():
    """Old: always calls traceback.format_exc()."""
    try:
        raise ValueError("test error")
    except Exception:
        tb_str = traceback.format_exc()
        _ = f"Error: {tb_str}"
        return tb_str


def _error_handler_new_error_path():
    """New: checks log level before traceback.format_exc()."""
    logger = logging.getLogger("benchmark_test_logger")
    logger.setLevel(logging.CRITICAL)
    try:
        raise ValueError("test error")
    except Exception:
        if logger.isEnabledFor(logging.ERROR):
            tb_str = traceback.format_exc()
            _ = f"Error: {tb_str}"
            return tb_str
        return None


@pytest.mark.benchmark(group="error_handler_error_path")
class TestErrorHandlerErrorPath:
    def test_old_always_format_traceback(self, benchmark):
        benchmark(_error_handler_old_error_path)

    def test_new_skip_traceback_when_not_logging(self, benchmark):
        benchmark(_error_handler_new_error_path)


# ---------------------------------------------------------------------------
# 8. isinstance: typing.Dict/List vs builtins dict/list
# ---------------------------------------------------------------------------
def _isinstance_typing(data):
    """Old: isinstance with typing generics."""
    if isinstance(data, Dict):
        return "dict"
    if isinstance(data, List):
        return "list"
    return "other"


def _isinstance_builtin(data):
    """New: isinstance with builtins."""
    if isinstance(data, dict):
        return "dict"
    if isinstance(data, list):
        return "list"
    return "other"


@pytest.mark.benchmark(group="isinstance_check")
class TestIsinstanceCheck:
    TEST_DATA: tuple = (
        {"key": "value"},
        [1, 2, 3],
        "string",
    )

    @pytest.fixture(params=TEST_DATA)
    def data(self, request):
        return request.param

    def test_old_typing_isinstance(self, benchmark, data):
        benchmark(_isinstance_typing, data)

    def test_new_builtin_isinstance(self, benchmark, data):
        result = benchmark(_isinstance_builtin, data)
        expected = _isinstance_typing(data)
        assert result == expected


# ---------------------------------------------------------------------------
# 9. json_type dispatch: if/elif chain vs dict lookup
# ---------------------------------------------------------------------------
class _MockDataType:
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4
    BOOL = 5
    VARCHAR = 6
    STRING = 7


def _json_type_old(json_type):
    """Old: if/elif chain."""
    if json_type == _MockDataType.INT8:
        return "Int8"
    if json_type == _MockDataType.INT16:
        return "Int16"
    if json_type == _MockDataType.INT32:
        return "Int32"
    if json_type == _MockDataType.INT64:
        return "Int64"
    if json_type == _MockDataType.BOOL:
        return "Bool"
    if json_type in (_MockDataType.VARCHAR, _MockDataType.STRING):
        return "VarChar"
    return None


_JSON_TYPE_MAP = {
    _MockDataType.INT8: "Int8",
    _MockDataType.INT16: "Int16",
    _MockDataType.INT32: "Int32",
    _MockDataType.INT64: "Int64",
    _MockDataType.BOOL: "Bool",
    _MockDataType.VARCHAR: "VarChar",
    _MockDataType.STRING: "VarChar",
}


def _json_type_new(json_type):
    """New: dict lookup."""
    return _JSON_TYPE_MAP.get(json_type)


@pytest.mark.benchmark(group="json_type_dispatch")
class TestJsonTypeDispatch:
    TYPES: tuple = (
        _MockDataType.INT8,
        _MockDataType.INT64,
        _MockDataType.VARCHAR,
        _MockDataType.STRING,
    )

    @pytest.fixture(params=TYPES)
    def json_type(self, request):
        return request.param

    def test_old_if_elif_chain(self, benchmark, json_type):
        benchmark(_json_type_old, json_type)

    def test_new_dict_lookup(self, benchmark, json_type):
        result = benchmark(_json_type_new, json_type)
        expected = _json_type_old(json_type)
        assert result == expected


# ---------------------------------------------------------------------------
# 10. extend([item]) vs append(item)
# ---------------------------------------------------------------------------
def _extend_single_old(container, n):
    """Old: extend with single-element list."""
    for i in range(n):
        container.extend([i])
    return container


def _append_single_new(container, n):
    """New: append single item."""
    for i in range(n):
        container.append(i)
    return container


@pytest.mark.benchmark(group="extend_vs_append")
class TestExtendVsAppend:
    def test_old_extend_single(self, benchmark):
        def run():
            return _extend_single_old([], 1000)

        benchmark(run)

    def test_new_append_single(self, benchmark):
        def run():
            return _append_single_new([], 1000)

        result = benchmark(run)
        assert len(result) == 1000


# ---------------------------------------------------------------------------
# 11. _num_input_fields: len(list comprehension) vs sum(generator)
# ---------------------------------------------------------------------------
def _num_fields_old(fields, predicate):
    """Old: create full list then len()."""
    return len([f for f in fields if predicate(f)])


def _num_fields_new(fields, predicate):
    """New: sum with generator (no intermediate list)."""
    return sum(1 for f in fields if predicate(f))


@pytest.mark.benchmark(group="num_input_fields")
class TestNumInputFields:
    @pytest.fixture
    def fields_data(self):
        fields = []
        for i in range(50):
            fields.append({"name": f"field_{i}", "is_input": i % 3 != 0})
        return fields

    def test_old_len_list_comp(self, benchmark, fields_data):
        benchmark(_num_fields_old, fields_data, lambda f: f["is_input"])

    def test_new_sum_generator(self, benchmark, fields_data):
        result = benchmark(_num_fields_new, fields_data, lambda f: f["is_input"])
        expected = _num_fields_old(fields_data, lambda f: f["is_input"])
        assert result == expected


# ---------------------------------------------------------------------------
# 12. Dynamic field json_dict: always iterate vs skip when disabled
# ---------------------------------------------------------------------------
def _dynamic_field_old(entity, fields_data, enable_dynamic):
    """Old: always builds dict comprehension, checks enable_dynamic inside."""
    json_dict = {k: v for k, v in entity.items() if k not in fields_data and enable_dynamic}
    if enable_dynamic:
        return json_dict
    return None


def _dynamic_field_new(entity, fields_data, enable_dynamic):
    """New: skip comprehension entirely when dynamic is disabled."""
    if enable_dynamic:
        return {k: v for k, v in entity.items() if k not in fields_data}
    return None


@pytest.mark.benchmark(group="dynamic_field_json")
class TestDynamicFieldJson:
    @pytest.fixture
    def entity_data(self):
        entity = {f"field_{i}": i for i in range(20)}
        fields_data = {f"field_{i}": None for i in range(10)}
        return entity, fields_data

    def test_old_always_iterate_disabled(self, benchmark, entity_data):
        entity, fields_data = entity_data
        benchmark(_dynamic_field_old, entity, fields_data, False)

    def test_new_skip_when_disabled(self, benchmark, entity_data):
        entity, fields_data = entity_data
        benchmark(_dynamic_field_new, entity, fields_data, False)

    def test_old_always_iterate_enabled(self, benchmark, entity_data):
        entity, fields_data = entity_data
        benchmark(_dynamic_field_old, entity, fields_data, True)

    def test_new_skip_when_enabled(self, benchmark, entity_data):
        entity, fields_data = entity_data
        result = benchmark(_dynamic_field_new, entity, fields_data, True)
        expected = _dynamic_field_old(entity, fields_data, True)
        assert result == expected


# ===========================================================================
# Iteration 2 benchmarks
# ===========================================================================


# ---------------------------------------------------------------------------
# 13. Nested function re-creation vs module-level functions
# ---------------------------------------------------------------------------
def _with_nested_functions(items):
    """Old: nested function definitions recreated per call."""

    def is_type_in_str(v, t):
        if not isinstance(v, str):
            return False
        try:
            t(v)
        except ValueError:
            return False
        return True

    def is_int_type(v):
        return isinstance(v, (int,)) or is_type_in_str(v, int)

    def is_float_type(v):
        return isinstance(v, (float,)) or is_type_in_str(v, float)

    count = 0
    for i, f in items:
        if is_int_type(i) and is_float_type(f):
            count += 1
    return count


def _module_is_type_in_str(v, t):
    if not isinstance(v, str):
        return False
    try:
        t(v)
    except ValueError:
        return False
    return True


def _module_is_int_type(v):
    return isinstance(v, (int,)) or _module_is_type_in_str(v, int)


def _module_is_float_type(v):
    return isinstance(v, (float,)) or _module_is_type_in_str(v, float)


def _with_module_functions(items):
    """New: module-level function references (no re-creation)."""
    count = 0
    for i, f in items:
        if _module_is_int_type(i) and _module_is_float_type(f):
            count += 1
    return count


@pytest.mark.benchmark(group="nested_vs_module_functions")
class TestNestedVsModuleFunctions:
    @pytest.fixture
    def sparse_items(self):
        return [(i, float(i) * 0.5) for i in range(100)]

    def test_old_nested_functions(self, benchmark, sparse_items):
        benchmark(_with_nested_functions, sparse_items)

    def test_new_module_functions(self, benchmark, sparse_items):
        result = benchmark(_with_module_functions, sparse_items)
        expected = _with_nested_functions(sparse_items)
        assert result == expected


# ---------------------------------------------------------------------------
# 14. Dict re-creation per call vs module-level constant (vector_attr_map)
# ---------------------------------------------------------------------------
def _dict_per_call(field_type, n):
    """Old: recreate dict on every call."""
    for _ in range(n):
        vector_attr_map = {
            1: "int8_vector",
            2: "binary_vector",
            3: "float16_vector",
            4: "bfloat16_vector",
        }
        result = vector_attr_map.get(field_type)
    return result


_MODULE_VECTOR_ATTR_MAP = {
    1: "int8_vector",
    2: "binary_vector",
    3: "float16_vector",
    4: "bfloat16_vector",
}


def _dict_module_level(field_type, n):
    """New: use module-level constant dict."""
    for _ in range(n):
        result = _MODULE_VECTOR_ATTR_MAP.get(field_type)
    return result


@pytest.mark.benchmark(group="dict_creation_per_call")
class TestDictCreationPerCall:
    def test_old_dict_per_call(self, benchmark):
        benchmark(_dict_per_call, 2, 1000)

    def test_new_module_level_dict(self, benchmark):
        result = benchmark(_dict_module_level, 2, 1000)
        expected = _dict_per_call(2, 1000)
        assert result == expected


# ---------------------------------------------------------------------------
# 15. hasattr per call vs __init__ initialization
# ---------------------------------------------------------------------------
class _OldCache:
    def lookup(self, key):
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key not in self._cache:
            self._cache[key] = key * 2
        return self._cache[key]


class _NewCache:
    def __init__(self):
        self._cache = {}

    def lookup(self, key):
        if key not in self._cache:
            self._cache[key] = key * 2
        return self._cache[key]


@pytest.mark.benchmark(group="hasattr_vs_init")
class TestHasattrVsInit:
    def test_old_hasattr_per_call(self, benchmark):
        obj = _OldCache()

        def run():
            for i in range(100):
                obj.lookup(i % 10)

        benchmark(run)

    def test_new_init_once(self, benchmark):
        obj = _NewCache()

        def run():
            for i in range(100):
                obj.lookup(i % 10)

        benchmark(run)


# ---------------------------------------------------------------------------
# 16. str.lower() redundant comparison
# ---------------------------------------------------------------------------
def _lower_both_sides(n):
    """Old: call .lower() on both sides every time."""
    count = 0
    for _ in range(n):
        if "utf-8".lower() != "utf-8".lower():
            count += 1
    return count


def _lower_one_side(n):
    """New: only call .lower() on the variable side."""
    protocol = "utf-8"
    count = 0
    for _ in range(n):
        if protocol.lower() != "utf-8":
            count += 1
    return count


@pytest.mark.benchmark(group="lower_comparison")
class TestLowerComparison:
    def test_old_lower_both(self, benchmark):
        benchmark(_lower_both_sides, 1000)

    def test_new_lower_one(self, benchmark):
        result = benchmark(_lower_one_side, 1000)
        expected = _lower_both_sides(1000)
        assert result == expected


# ---------------------------------------------------------------------------
# 17. list membership vs frozenset for type checking (search results)
# ---------------------------------------------------------------------------
_TYPE_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_TYPE_FROZENSET = frozenset(_TYPE_LIST)


def _type_check_list(types, container):
    count = 0
    for t in types:
        if t in container:
            count += 1
    return count


@pytest.mark.benchmark(group="type_check_list_vs_frozenset")
class TestTypeCheckListVsFrozenset:
    @pytest.fixture
    def types_to_check(self):
        return [1, 5, 10, 11, 3, 15, 7, 20] * 125

    def test_old_list(self, benchmark, types_to_check):
        benchmark(_type_check_list, types_to_check, _TYPE_LIST)

    def test_new_frozenset(self, benchmark, types_to_check):
        result = benchmark(_type_check_list, types_to_check, _TYPE_FROZENSET)
        expected = _type_check_list(types_to_check, _TYPE_LIST)
        assert result == expected


# ===========================================================================
# Iteration 3 benchmarks
# ===========================================================================


# ---------------------------------------------------------------------------
# 18. Protobuf attribute chain caching
# ---------------------------------------------------------------------------
class _MockProtoChain:
    """Simulates protobuf nested attribute access overhead."""

    class _Scalars:
        class _ArrayData:
            def __init__(self):
                self.element_type = 5
                self.data = list(range(100))

        def __init__(self):
            self.array_data = self._ArrayData()

    def __init__(self):
        self.scalars = self._Scalars()


def _proto_chain_uncached(proto, n):
    """Old: access chain repeatedly."""
    total = 0
    for _ in range(n):
        et = proto.scalars.array_data.element_type
        d = proto.scalars.array_data.data
        total += et + len(d)
    return total


def _proto_chain_cached(proto, n):
    """New: cache intermediate object."""
    array_data = proto.scalars.array_data
    total = 0
    for _ in range(n):
        et = array_data.element_type
        d = array_data.data
        total += et + len(d)
    return total


@pytest.mark.benchmark(group="proto_chain_caching")
class TestProtoChainCaching:
    @pytest.fixture
    def proto(self):
        return _MockProtoChain()

    def test_old_uncached_chain(self, benchmark, proto):
        benchmark(_proto_chain_uncached, proto, 1000)

    def test_new_cached_chain(self, benchmark, proto):
        result = benchmark(_proto_chain_cached, proto, 1000)
        expected = _proto_chain_uncached(proto, 1000)
        assert result == expected


# ---------------------------------------------------------------------------
# 19. valid_data caching in check_append pattern
# ---------------------------------------------------------------------------
class _MockFieldDataForCheck:
    def __init__(self, n):
        self.valid_data = [True] * n
        self.field_name = "test_field"
        self.type = 5
        self.scalars = MagicMock()
        self.scalars.int_data.data = list(range(n))


def _check_no_cache(field_data, index, row_data):
    """Old: repeated len() and attribute access."""
    if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
        row_data[field_data.field_name] = None
        return
    row_data[field_data.field_name] = field_data.scalars.int_data.data[index]


def _check_with_cache(field_data, index, row_data):
    """New: cached local variables."""
    valid_data = field_data.valid_data
    field_name = field_data.field_name
    has_valid = len(valid_data) > 0
    is_null = has_valid and valid_data[index] is False
    scalar_data = field_data.scalars.int_data.data
    row_data[field_name] = None if is_null else scalar_data[index]


@pytest.mark.benchmark(group="check_append_caching")
class TestCheckAppendCaching:
    @pytest.fixture
    def setup(self):
        return _MockFieldDataForCheck(100)

    def test_old_no_cache(self, benchmark, setup):
        def run():
            row = {}
            for i in range(100):
                _check_no_cache(setup, i, row)
            return row

        benchmark(run)

    def test_new_with_cache(self, benchmark, setup):
        def run():
            row = {}
            for i in range(100):
                _check_with_cache(setup, i, row)
            return row

        benchmark(run)


# ===========================================================================
# Iteration 4-10 benchmarks
# ===========================================================================

# ---------------------------------------------------------------------------
# 20. json.loads vs orjson.loads
# ---------------------------------------------------------------------------
import json

import orjson as _orjson


def _json_stdlib_loads(data, n):
    """Old: stdlib json.loads."""
    for _ in range(n):
        result = json.loads(data)
    return result


def _orjson_loads(data, n):
    """New: orjson.loads (faster C implementation)."""
    for _ in range(n):
        result = _orjson.loads(data)
    return result


@pytest.mark.benchmark(group="json_loads")
class TestJsonLoads:
    @pytest.fixture
    def json_data(self):
        return b'{"nprobe": 10, "ef": 64, "metric_type": "L2"}'

    def test_old_stdlib_json(self, benchmark, json_data):
        benchmark(_json_stdlib_loads, json_data, 100)

    def test_new_orjson(self, benchmark, json_data):
        result = benchmark(_orjson_loads, json_data, 100)
        expected = _json_stdlib_loads(json_data, 100)
        assert result == expected


# ---------------------------------------------------------------------------
# 21. dict.fromkeys per iteration vs template.copy()
# ---------------------------------------------------------------------------
def _dict_fromkeys_per_iter(keys, n):
    """Old: call dict.fromkeys for each entity."""
    return [dict.fromkeys(keys) for _ in range(n)]


def _dict_template_copy(keys, n):
    """New: create template once, then copy."""
    template = dict.fromkeys(keys)
    return [template.copy() for _ in range(n)]


@pytest.mark.benchmark(group="dict_fromkeys_vs_copy")
class TestDictFromkeysVsCopy:
    SIZES: tuple = (10, 100, 1000)

    @pytest.fixture(params=SIZES)
    def setup(self, request):
        n = request.param
        keys = [f"field_{i}" for i in range(15)]
        return keys, n

    def test_old_fromkeys_per_iter(self, benchmark, setup):
        keys, n = setup
        benchmark(_dict_fromkeys_per_iter, keys, n)

    def test_new_template_copy(self, benchmark, setup):
        keys, n = setup
        result = benchmark(_dict_template_copy, keys, n)
        expected = _dict_fromkeys_per_iter(keys, n)
        assert result == expected


# ---------------------------------------------------------------------------
# 22. get_field_data: if/elif chain vs dict dispatch
# ---------------------------------------------------------------------------
class _MockScalars:
    def __init__(self):
        self.bool_data = MagicMock()
        self.bool_data.data = [True, False]
        self.int_data = MagicMock()
        self.int_data.data = [1, 2, 3]
        self.long_data = MagicMock()
        self.long_data.data = [100, 200]
        self.string_data = MagicMock()
        self.string_data.data = ["a", "b"]


_DISPATCH_MAP = {
    1: "bool_data",
    2: "int_data",
    5: "long_data",
    21: "string_data",
}


def _get_field_data_chain(field_type, scalars):
    """Old: if/elif chain."""
    if field_type == 1:
        return scalars.bool_data.data
    if field_type == 2:
        return scalars.int_data.data
    if field_type == 5:
        return scalars.long_data.data
    if field_type == 21:
        return scalars.string_data.data
    return None


def _get_field_data_dispatch(field_type, scalars):
    """New: dict dispatch + getattr."""
    attr = _DISPATCH_MAP.get(field_type)
    if attr is not None:
        return getattr(scalars, attr).data
    return None


@pytest.mark.benchmark(group="get_field_data_dispatch")
class TestGetFieldDataDispatch:
    TYPES: tuple = (1, 5, 21)

    @pytest.fixture(params=TYPES)
    def setup(self, request):
        return request.param, _MockScalars()

    def test_old_chain(self, benchmark, setup):
        ft, scalars = setup
        benchmark(_get_field_data_chain, ft, scalars)

    def test_new_dispatch(self, benchmark, setup):
        ft, scalars = setup
        result = benchmark(_get_field_data_dispatch, ft, scalars)
        expected = _get_field_data_chain(ft, scalars)
        assert result == expected


# ---------------------------------------------------------------------------
# 23. hasattr chain vs tuple iteration for array data
# ---------------------------------------------------------------------------
class _MockArrayItem:
    def __init__(self):
        self.string_data = MagicMock()
        self.string_data.data = []
        self.int_data = MagicMock()
        self.int_data.data = [1, 2, 3]
        self.long_data = MagicMock()
        self.long_data.data = []
        self.float_data = MagicMock()
        self.float_data.data = []
        self.double_data = MagicMock()
        self.double_data.data = []
        self.bool_data = MagicMock()
        self.bool_data.data = []


def _get_array_length_old(array_item):
    """Old: hasattr chain."""
    if hasattr(array_item, "string_data") and hasattr(array_item.string_data, "data"):
        length = len(array_item.string_data.data)
        if length > 0:
            return length
    if hasattr(array_item, "int_data") and hasattr(array_item.int_data, "data"):
        length = len(array_item.int_data.data)
        if length > 0:
            return length
    if hasattr(array_item, "long_data") and hasattr(array_item.long_data, "data"):
        length = len(array_item.long_data.data)
        if length > 0:
            return length
    return 0


_ATTRS = ("string_data", "int_data", "long_data", "float_data", "double_data", "bool_data")


def _get_array_length_new(array_item):
    """New: tuple iteration with getattr."""
    for attr_name in _ATTRS:
        data = getattr(array_item, attr_name, None)
        if data is not None:
            length = len(data.data)
            if length > 0:
                return length
    return 0


@pytest.mark.benchmark(group="array_length_hasattr")
class TestArrayLengthHasattr:
    @pytest.fixture
    def array_item(self):
        return _MockArrayItem()

    def test_old_hasattr_chain(self, benchmark, array_item):
        benchmark(_get_array_length_old, array_item)

    def test_new_tuple_iteration(self, benchmark, array_item):
        result = benchmark(_get_array_length_new, array_item)
        expected = _get_array_length_old(array_item)
        assert result == expected


# ---------------------------------------------------------------------------
# 24. len(self) cached vs uncached in loop
# ---------------------------------------------------------------------------
class _MockList(list):
    pass


def _loop_uncached_len(lst):
    """Old: call len(self) in each range()."""
    total = 0
    for _ in range(5):
        for i in range(len(lst)):
            total += lst[i]
    return total


def _loop_cached_len(lst):
    """New: cache len once."""
    n = len(lst)
    total = 0
    for _ in range(5):
        for i in range(n):
            total += lst[i]
    return total


@pytest.mark.benchmark(group="cached_len")
class TestCachedLen:
    @pytest.fixture
    def data(self):
        return _MockList(range(200))

    def test_old_uncached_len(self, benchmark, data):
        benchmark(_loop_uncached_len, data)

    def test_new_cached_len(self, benchmark, data):
        result = benchmark(_loop_cached_len, data)
        expected = _loop_uncached_len(data)
        assert result == expected
