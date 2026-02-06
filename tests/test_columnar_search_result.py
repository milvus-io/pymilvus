"""
Comprehensive unit tests for ColumnarSearchResult.

Tests all supported data types and ensures compatibility with SearchResult.
"""

import collections.abc
import struct
from unittest.mock import MagicMock

import pytest
from pymilvus.client.columnar_search_result import (
    BytesVectorAccessor,
    ColumnarSearchResult,
    FloatVectorAccessor,
    Int8VectorAccessor,
    JsonAccessor,
    NullableAccessor,
    RowProxy,
    ScalarAccessor,
)
from pymilvus.client.search_result import SearchResult
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, schema_pb2

# ==============================================================================
# Test Fixtures and Mock Data Builders
# ==============================================================================


def build_base_result(nq: int = 2, topk: int = 5) -> schema_pb2.SearchResultData:
    """Create a base SearchResultData with IDs and scores."""
    total = nq * topk
    res = schema_pb2.SearchResultData()
    res.num_queries = nq
    res.top_k = topk
    res.topks.extend([topk] * nq)
    res.ids.int_id.data.extend(list(range(total)))
    res.scores.extend([float(i) * 0.1 for i in range(total)])
    res.primary_field_name = "id"
    return res


def add_bool_field(res: schema_pb2.SearchResultData, name: str = "bool_field"):
    """Add BOOL field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Bool
    field.scalars.bool_data.data.extend([i % 2 == 0 for i in range(total)])
    res.output_fields.append(name)


def add_int8_field(res: schema_pb2.SearchResultData, name: str = "int8_field"):
    """Add INT8 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int8
    field.scalars.int_data.data.extend([i % 128 for i in range(total)])
    res.output_fields.append(name)


def add_int16_field(res: schema_pb2.SearchResultData, name: str = "int16_field"):
    """Add INT16 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int16
    field.scalars.int_data.data.extend([i * 100 for i in range(total)])
    res.output_fields.append(name)


def add_int32_field(res: schema_pb2.SearchResultData, name: str = "int32_field"):
    """Add INT32 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int32
    field.scalars.int_data.data.extend([i * 1000 for i in range(total)])
    res.output_fields.append(name)


def add_int64_field(res: schema_pb2.SearchResultData, name: str = "int64_field"):
    """Add INT64 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int64
    field.scalars.long_data.data.extend([i * 10000 for i in range(total)])
    res.output_fields.append(name)


def add_float_field(res: schema_pb2.SearchResultData, name: str = "float_field"):
    """Add FLOAT field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Float
    field.scalars.float_data.data.extend([float(i) * 0.1 for i in range(total)])
    res.output_fields.append(name)


def add_double_field(res: schema_pb2.SearchResultData, name: str = "double_field"):
    """Add DOUBLE field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Double
    field.scalars.double_data.data.extend([float(i) * 0.001 for i in range(total)])
    res.output_fields.append(name)


def add_varchar_field(res: schema_pb2.SearchResultData, name: str = "varchar_field"):
    """Add VARCHAR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.VarChar
    field.scalars.string_data.data.extend([f"str_{i}" for i in range(total)])
    res.output_fields.append(name)


def add_json_field(res: schema_pb2.SearchResultData, name: str = "json_field"):
    """Add JSON field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.JSON
    for i in range(total):
        field.scalars.json_data.data.append(f'{{"value": {i}, "name": "item_{i}"}}'.encode())
    res.output_fields.append(name)


def add_array_int64_field(res: schema_pb2.SearchResultData, name: str = "array_int64"):
    """Add ARRAY<INT64> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.Int64
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.long_data.data.extend([i, i + 1, i + 2])
    res.output_fields.append(name)


def add_array_varchar_field(res: schema_pb2.SearchResultData, name: str = "array_varchar"):
    """Add ARRAY<VARCHAR> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.VarChar
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.string_data.data.extend([f"a{i}", f"b{i}"])
    res.output_fields.append(name)


def add_float_vector_field(
    res: schema_pb2.SearchResultData, dim: int = 4, name: str = "float_vector"
):
    """Add FLOAT_VECTOR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.FloatVector
    field.vectors.dim = dim
    for i in range(total * dim):
        field.vectors.float_vector.data.append(float(i) * 0.01)
    res.output_fields.append(name)


def add_binary_vector_field(
    res: schema_pb2.SearchResultData, dim: int = 32, name: str = "binary_vector"
):
    """Add BINARY_VECTOR field to mock data (dim is in bits)."""
    total = sum(res.topks)
    bytes_per_vec = dim // 8
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.BinaryVector
    field.vectors.dim = dim
    field.vectors.binary_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append(name)


def add_float16_vector_field(
    res: schema_pb2.SearchResultData, dim: int = 4, name: str = "float16_vector"
):
    """Add FLOAT16_VECTOR field to mock data."""
    total = sum(res.topks)
    bytes_per_vec = dim * 2
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Float16Vector
    field.vectors.dim = dim
    field.vectors.float16_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append(name)


def add_bfloat16_vector_field(
    res: schema_pb2.SearchResultData, dim: int = 4, name: str = "bfloat16_vector"
):
    """Add BFLOAT16_VECTOR field to mock data."""
    total = sum(res.topks)
    bytes_per_vec = dim * 2
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.BFloat16Vector
    field.vectors.dim = dim
    field.vectors.bfloat16_vector = bytes([i % 256 for i in range(total * bytes_per_vec)])
    res.output_fields.append(name)


def add_int8_vector_field(
    res: schema_pb2.SearchResultData, dim: int = 4, name: str = "int8_vector"
):
    """Add INT8_VECTOR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int8Vector
    field.vectors.dim = dim
    field.vectors.int8_vector = bytes([i % 256 for i in range(total * dim)])
    res.output_fields.append(name)


def add_sparse_vector_field(res: schema_pb2.SearchResultData, name: str = "sparse_vector"):
    """Add SPARSE_FLOAT_VECTOR field to mock data."""

    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.SparseFloatVector
    # Create sparse vectors in the format expected by Milvus
    for _ in range(total):
        # Simple sparse vector: {0: 0.1, 1: 0.2}
        indices = [0, 1]
        values = [0.1, 0.2]
        content = struct.pack(f"<{len(indices)}I{len(values)}f", *indices, *values)
        field.vectors.sparse_float_vector.contents.append(content)
    res.output_fields.append(name)


def add_dynamic_field(res: schema_pb2.SearchResultData):
    """Add $meta field for dynamic fields."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = "$meta"
    field.type = schema_pb2.DataType.JSON
    field.is_dynamic = True
    for i in range(total):
        field.scalars.json_data.data.append(f'{{"dyn_field": {i * 100}}}'.encode())


def add_element_indices(res: schema_pb2.SearchResultData):
    """Add element_indices to mock data."""
    total = sum(res.topks)
    res.element_indices.data.extend([i * 10 for i in range(total)])


# ==============================================================================
# Test Classes
# ==============================================================================


class TestColumnarSearchResultBasic:
    """Basic functionality tests."""

    def test_empty_result(self):
        """Test empty search result."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 0
        res.top_k = 0
        cr = ColumnarSearchResult(res)
        assert len(cr) == 0

    def test_single_query_single_result(self):
        """Test single query with single result."""
        res = build_base_result(nq=1, topk=1)
        cr = ColumnarSearchResult(res)
        assert len(cr) == 1
        assert len(cr[0]) == 1
        assert cr[0][0].id == 0

    def test_multiple_queries(self):
        """Test multiple queries."""
        res = build_base_result(nq=3, topk=5)
        cr = ColumnarSearchResult(res)
        assert len(cr) == 3
        for hits in cr:
            assert len(hits) == 5

    def test_uneven_topk(self):
        """Test when different queries have different topk."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 3
        res.top_k = 5
        res.topks.extend([5, 3, 2])  # Different topk per query
        res.ids.int_id.data.extend(list(range(10)))
        res.scores.extend([float(i) for i in range(10)])
        res.primary_field_name = "id"

        cr = ColumnarSearchResult(res)
        assert len(cr) == 3
        assert len(cr[0]) == 5
        assert len(cr[1]) == 3
        assert len(cr[2]) == 2


class TestColumnarSearchResultScalarTypes:
    """Tests for all scalar data types."""

    def test_bool_field(self):
        """Test BOOL field type."""
        res = build_base_result(nq=1, topk=3)
        add_bool_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["bool_field"] == cr[0][i]["bool_field"]

    def test_int8_field(self):
        """Test INT8 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int8_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["int8_field"] == cr[0][i]["int8_field"]

    def test_int16_field(self):
        """Test INT16 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int16_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["int16_field"] == cr[0][i]["int16_field"]

    def test_int32_field(self):
        """Test INT32 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int32_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["int32_field"] == cr[0][i]["int32_field"]

    def test_int64_field(self):
        """Test INT64 field type."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["int64_field"] == cr[0][i]["int64_field"]

    def test_float_field(self):
        """Test FLOAT field type."""
        res = build_base_result(nq=1, topk=3)
        add_float_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["float_field"] == cr[0][i]["float_field"]

    def test_double_field(self):
        """Test DOUBLE field type."""
        res = build_base_result(nq=1, topk=3)
        add_double_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["double_field"] == cr[0][i]["double_field"]

    def test_varchar_field(self):
        """Test VARCHAR field type."""
        res = build_base_result(nq=1, topk=3)
        add_varchar_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["varchar_field"] == cr[0][i]["varchar_field"]

    def test_json_field(self):
        """Test JSON field type."""
        res = build_base_result(nq=1, topk=3)
        add_json_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["json_field"]
            cr_val = cr[0][i]["json_field"]
            assert isinstance(sr_val, dict) and isinstance(cr_val, dict)
            assert sr_val == cr_val


class TestColumnarSearchResultArrayTypes:
    """Tests for ARRAY data types."""

    def test_array_int64_field(self):
        """Test ARRAY<INT64> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_int64_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = list(sr[0][i]["array_int64"])  # Convert to list for comparison
            cr_val = cr[0][i]["array_int64"]
            assert sr_val == cr_val

    def test_array_varchar_field(self):
        """Test ARRAY<VARCHAR> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_varchar_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = list(sr[0][i]["array_varchar"])
            cr_val = cr[0][i]["array_varchar"]
            assert sr_val == cr_val


class TestColumnarSearchResultVectorTypes:
    """Tests for all vector data types."""

    def test_float_vector_field(self):
        """Test FLOAT_VECTOR field type - should return list."""
        res = build_base_result(nq=1, topk=3)
        add_float_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["float_vector"]
            cr_val = cr[0][i]["float_vector"]
            # Both should be list type
            assert isinstance(cr_val, list)
            assert list(sr_val) == cr_val

    def test_binary_vector_field(self):
        """Test BINARY_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_binary_vector_field(res, dim=32)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["binary_vector"]
            cr_val = cr[0][i]["binary_vector"]
            assert isinstance(sr_val, bytes) and isinstance(cr_val, bytes)
            assert sr_val == cr_val

    def test_float16_vector_field(self):
        """Test FLOAT16_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_float16_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["float16_vector"]
            cr_val = cr[0][i]["float16_vector"]
            assert isinstance(sr_val, bytes) and isinstance(cr_val, bytes)
            assert sr_val == cr_val

    def test_bfloat16_vector_field(self):
        """Test BFLOAT16_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_bfloat16_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["bfloat16_vector"]
            cr_val = cr[0][i]["bfloat16_vector"]
            assert isinstance(sr_val, bytes) and isinstance(cr_val, bytes)
            assert sr_val == cr_val

    def test_int8_vector_field(self):
        """Test INT8_VECTOR field type - should return bytes."""
        res = build_base_result(nq=1, topk=3)
        add_int8_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["int8_vector"]
            cr_val = cr[0][i]["int8_vector"]
            assert isinstance(sr_val, bytes) and isinstance(cr_val, bytes)
            assert sr_val == cr_val

    def test_sparse_vector_field(self):
        """Test SPARSE_FLOAT_VECTOR field type."""
        res = build_base_result(nq=1, topk=3)
        add_sparse_vector_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["sparse_vector"]
            cr_val = cr[0][i]["sparse_vector"]
            # Sparse vector should be returned as a dictionary {idx: val}
            assert isinstance(cr_val, dict) or (
                hasattr(cr_val, "shape") and hasattr(cr_val, "data")
            )
            # Both should be dict-like sparse representations
            assert type(sr_val) is type(cr_val)


class TestColumnarHits:
    """Tests for ColumnarHits class."""

    def test_length(self):
        """Test ColumnarHits length."""
        res = build_base_result(nq=2, topk=10)
        cr = ColumnarSearchResult(res)

        assert len(cr) == 2
        assert len(cr[0]) == 10
        assert len(cr[1]) == 10

    def test_iteration(self):
        """Test iteration over ColumnarHits."""
        res = build_base_result(nq=1, topk=5)
        cr = ColumnarSearchResult(res)

        hits = cr[0]
        count = 0
        for hit in hits:
            assert isinstance(hit, RowProxy)
            count += 1
        assert count == 5

    def test_ids_and_distances(self):
        """Test that ColumnarHits exposes ids and distances properties."""
        res = build_base_result(nq=2, topk=5)
        cr = ColumnarSearchResult(res)

        hits = cr[0]
        assert len(hits.ids) == 5
        assert len(hits.distances) == 5
        assert hits.ids == list(range(5))

    def test_slice_access(self):
        """Test slice access on ColumnarHits."""
        res = build_base_result(nq=1, topk=10)
        cr = ColumnarSearchResult(res)

        hits = cr[0]
        sliced = hits[2:5]
        assert len(sliced) == 3
        assert all(isinstance(h, RowProxy) for h in sliced)
        assert sliced[0].id == 2
        assert sliced[2].id == 4

    def test_negative_index(self):
        """Test negative indexing on ColumnarHits."""
        res = build_base_result(nq=1, topk=5)
        cr = ColumnarSearchResult(res)

        hits = cr[0]
        assert hits[-1].id == 4
        assert hits[-2].id == 3


class TestRowProxy:
    """Tests for RowProxy class."""

    def test_basic_access(self):
        """Test basic field access."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        assert hit.id == 0
        assert hit.distance == 0.0
        assert hit["int64_field"] == 0

    def test_dict_like_interface(self):
        """Test dict-like interface."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        add_varchar_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]

        keys = hit.keys()
        assert "int64_field" in keys
        assert "varchar_field" in keys

        assert "int64_field" in hit
        assert "nonexistent" not in hit

        assert hit.get("int64_field") == 0
        assert hit.get("nonexistent", "default") == "default"

        items = dict(hit.items())
        assert "int64_field" in items
        assert "varchar_field" in items

    def test_entity_access(self):
        """Test entity property for nested access."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        # hit.entity should return self for compatibility
        assert hit.entity["int64_field"] == 0
        assert hit["entity"]["int64_field"] == 0

    def test_to_dict(self):
        """Test to_dict() method."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        add_varchar_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        d = hit.to_dict()

        assert d["id"] == 0
        assert d["distance"] == 0.0
        assert "entity" in d
        assert d["entity"]["int64_field"] == 0
        assert d["entity"]["varchar_field"] == "str_0"

    def test_read_only(self):
        """Test that RowProxy is read-only."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        with pytest.raises(TypeError):
            hit["int64_field"] = 999

    def test_properties(self):
        """Test id, distance, pk, score properties."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        hit = cr[0][1]
        assert hit.id == 1
        assert hit.pk == 1
        assert hit.distance == pytest.approx(0.1, rel=1e-5)
        assert hit.score == pytest.approx(0.1, rel=1e-5)


class TestOffsetField:
    """Tests for offset (element_indices) support."""

    def test_offset_field_access(self):
        """Test accessing offset field when element_indices are present."""
        res = build_base_result(nq=1, topk=3)
        add_element_indices(res)

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        # Should be able to access offset via property and dict
        assert hit.offset == 0
        assert hit["offset"] == 0
        assert "offset" in hit
        assert "offset" in hit.keys()

        hit2 = cr[0][1]
        assert hit2.offset == 10
        assert hit2["offset"] == 10

    def test_offset_field_missing(self):
        """Test accessing offset field when element_indices are NOT present."""
        res = build_base_result(nq=1, topk=3)
        # NOT calling add_element_indices(res)

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        # Should NOT be able to access offset
        assert hit.offset is None
        assert "offset" not in hit
        with pytest.raises(KeyError):
            _ = hit["offset"]

    def test_offset_in_to_dict(self):
        """Test that offset is included in to_dict() output when present."""
        res = build_base_result(nq=1, topk=3)
        add_element_indices(res)
        cr = ColumnarSearchResult(res)

        d = cr[0][0].to_dict()
        assert "offset" in d
        assert d["offset"] == 0

    def test_offset_not_in_to_dict(self):
        """Test that offset is NOT included in to_dict() output when missing."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        d = cr[0][0].to_dict()
        assert "offset" not in d


class TestDynamicFields:
    """Tests for dynamic field support."""

    def test_dynamic_field_access(self):
        """Test accessing dynamic fields from $meta."""
        res = build_base_result(nq=1, topk=3)
        add_dynamic_field(res)
        res.output_fields.append("dyn_field")

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        # Should be able to access dynamic field
        assert hit["dyn_field"] == 0

    def test_dynamic_field_in_keys(self):
        """Test that dynamic fields appear in keys()."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        add_dynamic_field(res)
        res.output_fields.append("dyn_field")

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        keys = hit.keys()
        assert "int64_field" in keys
        assert "dyn_field" in keys


class TestCompatibilityWithSearchResult:
    """Tests ensuring full compatibility with SearchResult."""

    def test_iteration_pattern(self):
        """Test common iteration pattern."""
        res = build_base_result(nq=2, topk=5)
        add_int64_field(res)
        add_float_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        sr_results = []
        for hits in sr:
            for hit in hits:
                sr_results.append((hit.id, hit.distance, hit["int64_field"]))

        cr_results = []
        for hits in cr:
            for hit in hits:
                cr_results.append((hit.id, hit.distance, hit["int64_field"]))

        assert sr_results == cr_results

    def test_all_scalar_types_together(self):
        """Test all scalar types in a single result."""
        res = build_base_result(nq=1, topk=3)
        add_bool_field(res)
        add_int8_field(res)
        add_int16_field(res)
        add_int32_field(res)
        add_int64_field(res)
        add_float_field(res)
        add_double_field(res)
        add_varchar_field(res)
        add_json_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            for field in res.output_fields:
                sr_val = sr[0][i][field]
                cr_val = cr[0][i][field]
                # Values should be equal (may need type conversion for some)
                if hasattr(sr_val, "__iter__") and not isinstance(sr_val, (str, dict)):
                    assert list(sr_val) == list(cr_val), f"Mismatch for {field}"
                else:
                    assert sr_val == cr_val, f"Mismatch for {field}"

    def test_all_vector_types_together(self):
        """Test all vector types in a single result."""
        res = build_base_result(nq=1, topk=3)
        add_float_vector_field(res, dim=4)
        add_binary_vector_field(res, dim=32)
        add_float16_vector_field(res, dim=4)
        add_bfloat16_vector_field(res, dim=4)
        add_int8_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            # FLOAT_VECTOR returns list
            sr_float = list(sr[0][i]["float_vector"])
            cr_float = cr[0][i]["float_vector"]
            assert sr_float == cr_float

            # Other vectors return bytes
            for field in ["binary_vector", "float16_vector", "bfloat16_vector", "int8_vector"]:
                sr_val = sr[0][i][field]
                cr_val = cr[0][i][field]
                assert sr_val == cr_val


class TestPerformance:
    """Performance-related tests (not measuring time, just ensuring efficiency)."""

    def test_initialization_creates_minimal_objects(self):
        """Test that initialization doesn't create many objects."""
        res = build_base_result(nq=100, topk=100)
        add_float_vector_field(res, dim=128)

        # This should complete quickly as it doesn't create 10,000 Hit objects
        cr = ColumnarSearchResult(res)

        assert len(cr) == 100
        assert len(cr[0]) == 100

    def test_materialize_is_noop(self):
        """Test that materialize() is a no-op."""
        res = build_base_result(nq=1, topk=5)
        cr = ColumnarSearchResult(res)

        # Should not raise any error
        cr.materialize()

        # Data should still be accessible
        assert cr[0][0].id == 0


class TestSpecialCases:
    """Tests for special/edge cases."""

    def test_string_primary_key(self):
        """Test with string primary key."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 3
        res.topks.extend([3])
        res.ids.str_id.data.extend(["a", "b", "c"])
        res.scores.extend([0.1, 0.2, 0.3])
        res.primary_field_name = "pk"

        cr = ColumnarSearchResult(res)
        assert cr[0][0].id == "a"
        assert cr[0][1].id == "b"
        assert cr[0][2].id == "c"

    def test_round_decimal(self):
        """Test score rounding."""
        res = build_base_result(nq=1, topk=3)
        res.scores[:] = [0.123456, 0.234567, 0.345678]

        cr = ColumnarSearchResult(res, round_decimal=2)
        assert cr[0][0].distance == 0.12
        assert cr[0][1].distance == 0.23
        assert cr[0][2].distance == 0.35

    def test_recalls(self):
        """Test recalls attribute."""
        res = build_base_result(nq=2, topk=3)
        res.recalls.extend([0.95, 0.98])

        cr = ColumnarSearchResult(res)
        assert cr.recalls is not None
        recalls_list = list(cr.recalls)
        assert recalls_list[0] == pytest.approx(0.95, rel=1e-5)
        assert recalls_list[1] == pytest.approx(0.98, rel=1e-5)

    def test_extra_info(self):
        """Test extra info from status."""
        res = build_base_result(nq=1, topk=3)

        # Without status, extra should be empty
        cr = ColumnarSearchResult(res)
        assert cr.extra == {}

    def test_index_out_of_range(self):
        """Test IndexError is raised for out of range access."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        with pytest.raises(IndexError):
            _ = cr[0][10]

        with pytest.raises(IndexError):
            _ = cr[0][-10]

    def test_field_not_found(self):
        """Test KeyError is raised for non-existent field."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        with pytest.raises(KeyError):
            _ = cr[0][0]["nonexistent_field"]


# ==============================================================================
# Sparse Vector Tests
# ==============================================================================


class TestSparseVectorField:
    """Tests for SPARSE_FLOAT_VECTOR field type."""

    def test_sparse_vector_field(self):
        """Test SPARSE_FLOAT_VECTOR field access."""
        res = build_base_result(nq=1, topk=3)
        add_sparse_vector_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["sparse_vector"]
            cr_val = cr[0][i]["sparse_vector"]
            # Both should be dict-like sparse representations
            assert type(sr_val) is type(cr_val)
            assert sr_val == cr_val

    def test_sparse_vector_multiple_queries(self):
        """Test sparse vector with multiple queries."""
        res = build_base_result(nq=3, topk=5)
        add_sparse_vector_field(res)

        cr = ColumnarSearchResult(res)

        # Access from different queries
        for q in range(3):
            for i in range(5):
                val = cr[q][i]["sparse_vector"]
                assert val is not None


# ==============================================================================
# Nullable Field Tests
# ==============================================================================


def add_nullable_int64_field(res: schema_pb2.SearchResultData, name: str = "nullable_int64"):
    """Add nullable INT64 field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Int64
    field.scalars.long_data.data.extend([i * 100 for i in range(total)])
    # Mark every other value as null
    field.valid_data.extend([i % 2 == 0 for i in range(total)])
    res.output_fields.append(name)


def add_nullable_varchar_field(res: schema_pb2.SearchResultData, name: str = "nullable_varchar"):
    """Add nullable VARCHAR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.VarChar
    field.scalars.string_data.data.extend([f"str_{i}" for i in range(total)])
    # Mark every third value as null
    field.valid_data.extend([i % 3 != 0 for i in range(total)])
    res.output_fields.append(name)


def add_nullable_json_field(res: schema_pb2.SearchResultData, name: str = "nullable_json"):
    """Add nullable JSON field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.JSON
    for i in range(total):
        field.scalars.json_data.data.append(f'{{"value": {i}}}'.encode())
    # Mark some values as null
    field.valid_data.extend([i % 4 != 0 for i in range(total)])
    res.output_fields.append(name)


class TestNullableFields:
    """Tests for nullable field handling."""

    def test_nullable_int64_field(self):
        """Test nullable INT64 field returns None for null values."""
        res = build_base_result(nq=1, topk=6)
        add_nullable_int64_field(res)

        cr = ColumnarSearchResult(res)

        # Even indices should have values, odd indices should be None
        assert cr[0][0]["nullable_int64"] == 0
        assert cr[0][1]["nullable_int64"] is None
        assert cr[0][2]["nullable_int64"] == 200
        assert cr[0][3]["nullable_int64"] is None
        assert cr[0][4]["nullable_int64"] == 400
        assert cr[0][5]["nullable_int64"] is None

    def test_nullable_varchar_field(self):
        """Test nullable VARCHAR field returns None for null values."""
        res = build_base_result(nq=1, topk=6)
        add_nullable_varchar_field(res)

        cr = ColumnarSearchResult(res)

        # Index 0, 3 should be None (i % 3 == 0)
        assert cr[0][0]["nullable_varchar"] is None
        assert cr[0][1]["nullable_varchar"] == "str_1"
        assert cr[0][2]["nullable_varchar"] == "str_2"
        assert cr[0][3]["nullable_varchar"] is None
        assert cr[0][4]["nullable_varchar"] == "str_4"
        assert cr[0][5]["nullable_varchar"] == "str_5"

    def test_nullable_json_field(self):
        """Test nullable JSON field returns None for null values."""
        res = build_base_result(nq=1, topk=8)
        add_nullable_json_field(res)

        cr = ColumnarSearchResult(res)

        # Index 0, 4 should be None (i % 4 == 0)
        assert cr[0][0]["nullable_json"] is None
        assert cr[0][1]["nullable_json"] == {"value": 1}
        assert cr[0][4]["nullable_json"] is None
        assert cr[0][5]["nullable_json"] == {"value": 5}

    def test_nullable_field_compatibility(self):
        """Test nullable field compatibility with SearchResult."""
        res = build_base_result(nq=1, topk=4)
        add_nullable_int64_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(4):
            sr_val = sr[0][i]["nullable_int64"]
            cr_val = cr[0][i]["nullable_int64"]
            assert sr_val == cr_val, f"Mismatch at index {i}: SR={sr_val}, CR={cr_val}"


# ==============================================================================
# Accessor Classes Direct Tests
# ==============================================================================


class TestAccessorClasses:
    """Direct tests for accessor helper classes."""

    def test_scalar_accessor(self):
        """Test ScalarAccessor directly."""
        data = [10, 20, 30, 40, 50]
        accessor = ScalarAccessor(data, start=1)

        assert accessor.get(0) == 20  # data[0 + 1]
        assert accessor.get(1) == 30  # data[1 + 1]
        assert accessor.get(2) == 40  # data[2 + 1]

    def test_float_vector_accessor(self):
        """Test FloatVectorAccessor directly."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 2 vectors of dim=4
        accessor = FloatVectorAccessor(data, start=0, dim=4)

        vec0 = accessor.get(0)
        vec1 = accessor.get(1)

        assert vec0 == [0.1, 0.2, 0.3, 0.4]
        assert vec1 == [0.5, 0.6, 0.7, 0.8]

    def test_float_vector_accessor_with_offset(self):
        """Test FloatVectorAccessor with start offset."""
        data = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4]  # Skip first vector
        accessor = FloatVectorAccessor(data, start=1, dim=4)

        vec = accessor.get(0)  # Should get second vector
        assert vec == [0.1, 0.2, 0.3, 0.4]

    def test_bytes_vector_accessor(self):
        """Test BytesVectorAccessor directly."""
        data = bytes([0, 1, 2, 3, 4, 5, 6, 7])  # 2 vectors of 4 bytes each
        accessor = BytesVectorAccessor(data, start=0, bpv=4)

        vec0 = accessor.get(0)
        vec1 = accessor.get(1)

        assert vec0 == bytes([0, 1, 2, 3])
        assert vec1 == bytes([4, 5, 6, 7])

    def test_int8_vector_accessor(self):
        """Test Int8VectorAccessor directly."""
        data = bytes([10, 20, 30, 40, 50, 60, 70, 80])  # 2 vectors of dim=4
        accessor = Int8VectorAccessor(data, start=0, dim=4)

        vec0 = accessor.get(0)
        vec1 = accessor.get(1)

        assert vec0 == bytes([10, 20, 30, 40])
        assert vec1 == bytes([50, 60, 70, 80])

    def test_json_accessor(self):
        """Test JsonAccessor directly."""
        data = [b'{"a": 1}', b'{"b": 2}', b'{"c": 3}']
        accessor = JsonAccessor(data, start=0)

        assert accessor.get(0) == {"a": 1}
        assert accessor.get(1) == {"b": 2}
        assert accessor.get(2) == {"c": 3}

    def test_json_accessor_empty_bytes(self):
        """Test JsonAccessor with empty bytes returns None."""
        data = [b'{"a": 1}', b"", b'{"c": 3}']
        accessor = JsonAccessor(data, start=0)

        assert accessor.get(0) == {"a": 1}
        assert accessor.get(1) is None
        assert accessor.get(2) == {"c": 3}

    def test_nullable_accessor(self):
        """Test NullableAccessor directly."""
        raw_data = [100, 200, 300, 400]
        valid_data = [True, False, True, False]

        def raw_get(i: int) -> int:
            return raw_data[i]

        accessor = NullableAccessor(raw_get, valid_data, start=0)

        assert accessor.get(0) == 100  # valid
        assert accessor.get(1) is None  # invalid
        assert accessor.get(2) == 300  # valid
        assert accessor.get(3) is None  # invalid

    def test_nullable_accessor_with_offset(self):
        """Test NullableAccessor with start offset."""
        raw_data = [100, 200, 300, 400]
        valid_data = [True, False, True, False]

        # Raw accessor doesn't know about offset, NullableAccessor handles it
        def raw_get(i: int) -> int:
            return raw_data[i + 1]  # Offset already applied by raw_accessor

        accessor = NullableAccessor(raw_get, valid_data, start=1)

        assert accessor.get(0) is None  # valid_data[0+1] = False
        assert accessor.get(1) == 300  # valid_data[1+1] = True


# ==============================================================================
# More Array Element Types Tests
# ==============================================================================


def add_array_bool_field(res: schema_pb2.SearchResultData, name: str = "array_bool"):
    """Add ARRAY<BOOL> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.Bool
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.bool_data.data.extend([True, False, i % 2 == 0])
    res.output_fields.append(name)


def add_array_float_field(res: schema_pb2.SearchResultData, name: str = "array_float"):
    """Add ARRAY<FLOAT> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.Float
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.float_data.data.extend([float(i) * 0.1, float(i) * 0.2])
    res.output_fields.append(name)


def add_array_double_field(res: schema_pb2.SearchResultData, name: str = "array_double"):
    """Add ARRAY<DOUBLE> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.Double
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.double_data.data.extend([float(i) * 0.001, float(i) * 0.002])
    res.output_fields.append(name)


def add_array_int32_field(res: schema_pb2.SearchResultData, name: str = "array_int32"):
    """Add ARRAY<INT32> field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Array
    field.scalars.array_data.element_type = schema_pb2.DataType.Int32
    for i in range(total):
        arr = field.scalars.array_data.data.add()
        arr.int_data.data.extend([i * 10, i * 20, i * 30])
    res.output_fields.append(name)


class TestArrayElementTypes:
    """Tests for various ARRAY element types."""

    def test_array_bool_field(self):
        """Test ARRAY<BOOL> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_bool_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = list(sr[0][i]["array_bool"])
            cr_val = cr[0][i]["array_bool"]
            assert sr_val == cr_val

    def test_array_float_field(self):
        """Test ARRAY<FLOAT> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_float_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = list(sr[0][i]["array_float"])
            cr_val = cr[0][i]["array_float"]
            assert len(sr_val) == len(cr_val)
            for sv, cv in zip(sr_val, cr_val):
                assert sv == pytest.approx(cv, rel=1e-5)

    def test_array_double_field(self):
        """Test ARRAY<DOUBLE> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_double_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = list(sr[0][i]["array_double"])
            cr_val = cr[0][i]["array_double"]
            assert len(sr_val) == len(cr_val)
            for sv, cv in zip(sr_val, cr_val):
                assert sv == pytest.approx(cv, rel=1e-5)

    def test_array_int32_field(self):
        """Test ARRAY<INT32> field type."""
        res = build_base_result(nq=1, topk=3)
        add_array_int32_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = list(sr[0][i]["array_int32"])
            cr_val = cr[0][i]["array_int32"]
            assert sr_val == cr_val


# ==============================================================================
# String Representation Tests
# ==============================================================================


class TestStringRepresentations:
    """Tests for __str__ and __repr__ methods."""

    def test_columnar_search_result_str(self):
        """Test ColumnarSearchResult string representation."""
        res = build_base_result(nq=2, topk=3)
        cr = ColumnarSearchResult(res)

        s = str(cr)
        assert "data:" in s
        assert len(s) > 0

    def test_columnar_search_result_repr(self):
        """Test ColumnarSearchResult repr is same as str."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        assert str(cr) == repr(cr)

    def test_columnar_hits_str(self):
        """Test ColumnarHits string representation."""
        res = build_base_result(nq=1, topk=5)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hits = cr[0]
        s = str(hits)
        assert len(s) > 0
        # Should contain dict-like representation of hits

    def test_columnar_hits_str_truncation(self):
        """Test ColumnarHits truncates output for large results."""
        res = build_base_result(nq=1, topk=20)
        cr = ColumnarSearchResult(res)

        hits = cr[0]
        s = str(hits)
        # Should have truncation message for >10 results
        assert "remaining" in s

    def test_row_proxy_str(self):
        """Test RowProxy string representation."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        s = str(hit)
        assert "id" in s or "0" in s
        assert "distance" in s

    def test_row_proxy_repr(self):
        """Test RowProxy repr is same as str."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        assert str(hit) == repr(hit)


# ==============================================================================
# Multi-Query Scenarios Tests
# ==============================================================================


class TestMultiQueryScenarios:
    """Test scenarios with multiple queries."""

    def test_multi_query_independent_access(self):
        """Test that each query's hits are independent."""
        res = build_base_result(nq=5, topk=10)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        # Verify each query has its own offset
        for q in range(5):
            for i in range(10):
                expected_id = q * 10 + i
                assert cr[q][i].id == expected_id
                assert cr[q][i]["int64_field"] == expected_id * 10000

    def test_multi_query_slice_access(self):
        """Test slicing across different queries."""
        res = build_base_result(nq=3, topk=10)
        cr = ColumnarSearchResult(res)

        # Slice first 3 from each query
        for q in range(3):
            sliced = cr[q][:3]
            assert len(sliced) == 3
            for i, hit in enumerate(sliced):
                assert hit.id == q * 10 + i

    def test_multi_query_reverse_iteration(self):
        """Test reverse iteration over queries."""
        res = build_base_result(nq=3, topk=5)
        cr = ColumnarSearchResult(res)

        collected = []
        for q in range(2, -1, -1):  # Iterate queries in reverse
            for hit in cr[q]:
                collected.append(hit.id)

        # Should have collected IDs in reverse query order
        expected = list(range(10, 15)) + list(range(5, 10)) + list(range(5))
        assert collected == expected

    def test_accessor_cache_shared_across_queries(self):
        """Test that accessor cache is properly shared across ColumnarHits."""
        res = build_base_result(nq=3, topk=5)
        add_float_vector_field(res, dim=4)
        cr = ColumnarSearchResult(res)

        # Access field from different queries
        _ = cr[0][0]["float_vector"]
        _ = cr[1][0]["float_vector"]
        _ = cr[2][0]["float_vector"]

        # All should work correctly with proper values
        for q in range(3):
            vec = cr[q][0]["float_vector"]
            assert len(vec) == 4
            assert isinstance(vec, list)


# ==============================================================================
# Iterator Compatibility Tests
# ==============================================================================


class TestIteratorCompatibility:
    """Test iterator-related functionality."""

    def test_get_session_ts(self):
        """Test get_session_ts method."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res, session_ts=12345)

        assert cr.get_session_ts() == 12345

    def test_get_session_ts_default(self):
        """Test get_session_ts returns default value."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        assert cr.get_session_ts() == 0

    def test_get_search_iterator_v2_results_info(self):
        """Test get_search_iterator_v2_results_info method."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        # Should return the protobuf field (may be empty)
        result = cr.get_search_iterator_v2_results_info()
        assert result is not None


# ==============================================================================
# Edge Cases and Boundary Conditions
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_topk_for_some_queries(self):
        """Test handling when some queries have zero results."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 3
        res.top_k = 5
        res.topks.extend([5, 0, 3])  # Second query has no results
        res.ids.int_id.data.extend(list(range(8)))  # 5 + 0 + 3 = 8
        res.scores.extend([float(i) for i in range(8)])
        res.primary_field_name = "id"

        cr = ColumnarSearchResult(res)

        assert len(cr) == 3
        assert len(cr[0]) == 5
        assert len(cr[1]) == 0  # Empty
        assert len(cr[2]) == 3

        # Empty hits should be iterable but produce nothing
        count = 0
        for _ in cr[1]:
            count += 1
        assert count == 0

    def test_single_result_single_query(self):
        """Test minimal case: 1 query, 1 result."""
        res = build_base_result(nq=1, topk=1)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        assert len(cr) == 1
        assert len(cr[0]) == 1
        assert cr[0][0].id == 0
        assert cr[0][0]["int64_field"] == 0

    def test_large_dimension_vector(self):
        """Test with large dimension vector."""
        res = build_base_result(nq=1, topk=2)
        add_float_vector_field(res, dim=1024, name="large_vector")

        cr = ColumnarSearchResult(res)
        vec = cr[0][0]["large_vector"]

        assert len(vec) == 1024
        assert isinstance(vec, list)

    def test_contains_special_keys(self):
        """Test __contains__ for special keys."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]

        # Special keys should always be contained
        assert "id" in hit
        assert "distance" in hit
        assert "entity" in hit

        # Primary key name should be contained
        assert hit._pk_name in hit

        # Added field should be contained
        assert "int64_field" in hit

        # Non-existent field should not be contained
        assert "nonexistent" not in hit

    def test_empty_json_in_results(self):
        """Test handling of empty JSON values."""
        res = build_base_result(nq=1, topk=3)

        # Add JSON field with some empty values
        sum(res.topks)
        field = res.fields_data.add()
        field.field_name = "json_field"
        field.type = schema_pb2.DataType.JSON
        field.scalars.json_data.data.append(b'{"key": "value"}')
        field.scalars.json_data.data.append(b"")  # Empty
        field.scalars.json_data.data.append(b"{}")  # Empty object
        res.output_fields.append("json_field")

        cr = ColumnarSearchResult(res)

        assert cr[0][0]["json_field"] == {"key": "value"}
        assert cr[0][1]["json_field"] is None  # Empty bytes -> None
        assert cr[0][2]["json_field"] == {}  # Empty object

    def test_iter_on_row_proxy(self):
        """Test iterating over RowProxy like a dict."""
        res = build_base_result(nq=1, topk=3)
        add_int64_field(res)
        add_varchar_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]

        # Should be able to iterate over field names
        field_names = list(hit)
        assert "int64_field" in field_names
        assert "varchar_field" in field_names

    def test_values_method(self):
        """Test values() method on RowProxy."""
        res = build_base_result(nq=1, topk=1)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        values = hit.values()

        assert isinstance(values, list)
        assert 0 in values  # int64_field value

    def test_items_method(self):
        """Test items() method on RowProxy."""
        res = build_base_result(nq=1, topk=1)
        add_int64_field(res)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        items = hit.items()

        assert isinstance(items, list)
        assert ("int64_field", 0) in items


# ==============================================================================
# Extra Info and Status Tests
# ==============================================================================


class TestExtraInfoAndStatus:
    """Tests for extra info extraction from status."""

    def test_extra_info_with_cost(self):
        """Test extra info parsing with cost."""

        res = build_base_result(nq=1, topk=3)

        status = common_pb2.Status()
        status.extra_info["report_value"] = "100"

        cr = ColumnarSearchResult(res, status=status)

        assert "cost" in cr.extra
        assert cr.extra["cost"] == 100

    def test_extra_info_with_scan_bytes(self):
        """Test extra info parsing with scanned bytes."""

        res = build_base_result(nq=1, topk=3)

        status = common_pb2.Status()
        status.extra_info["scanned_remote_bytes"] = "1024"
        status.extra_info["scanned_total_bytes"] = "2048"

        cr = ColumnarSearchResult(res, status=status)

        assert cr.extra["scanned_remote_bytes"] == 1024
        assert cr.extra["scanned_total_bytes"] == 2048

    def test_extra_info_with_cache_hit_ratio(self):
        """Test extra info parsing with cache hit ratio."""

        res = build_base_result(nq=1, topk=3)

        status = common_pb2.Status()
        status.extra_info["cache_hit_ratio"] = "0.85"

        cr = ColumnarSearchResult(res, status=status)

        assert cr.extra["cache_hit_ratio"] == pytest.approx(0.85)


# ==============================================================================
# Highlight Tests
# ==============================================================================


def add_highlight_results(res: schema_pb2.SearchResultData, field_name: str = "text_field"):
    """Add highlight results to mock data."""

    total = sum(res.topks)

    highlight_result = common_pb2.HighlightResult()
    highlight_result.field_name = field_name

    for i in range(total):
        highlight_data = common_pb2.HighlightData()
        highlight_data.fragments.extend([f"fragment_{i}_a", f"fragment_{i}_b"])
        highlight_data.scores.extend([0.9 - i * 0.01, 0.8 - i * 0.01])
        highlight_result.datas.append(highlight_data)

    res.highlight_results.append(highlight_result)


class TestHighlightSupport:
    """Tests for highlight functionality."""

    def test_highlight_basic(self):
        """Test basic highlight access."""
        res = build_base_result(nq=1, topk=3)
        add_varchar_field(res, name="text_field")
        add_highlight_results(res, field_name="text_field")

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        highlight = hit.highlight
        assert highlight is not None
        assert "text_field" in highlight
        assert "fragments" in highlight["text_field"]
        assert "scores" in highlight["text_field"]
        assert highlight["text_field"]["fragments"] == ["fragment_0_a", "fragment_0_b"]

    def test_highlight_multiple_hits(self):
        """Test highlight across multiple hits."""
        res = build_base_result(nq=1, topk=5)
        add_varchar_field(res, name="text_field")
        add_highlight_results(res, field_name="text_field")

        cr = ColumnarSearchResult(res)

        for i in range(5):
            hit = cr[0][i]
            highlight = hit.highlight
            assert highlight is not None
            assert highlight["text_field"]["fragments"] == [f"fragment_{i}_a", f"fragment_{i}_b"]

    def test_highlight_multiple_queries(self):
        """Test highlight with multiple queries."""
        res = build_base_result(nq=3, topk=5)
        add_varchar_field(res, name="text_field")
        add_highlight_results(res, field_name="text_field")

        cr = ColumnarSearchResult(res)

        for q in range(3):
            for i in range(5):
                hit = cr[q][i]
                highlight = hit.highlight
                assert highlight is not None
                # Calculate absolute index
                abs_idx = q * 5 + i
                expected = [f"fragment_{abs_idx}_a", f"fragment_{abs_idx}_b"]
                assert highlight["text_field"]["fragments"] == expected

    def test_highlight_no_results(self):
        """Test that highlight returns None when no highlight results."""
        res = build_base_result(nq=1, topk=3)
        cr = ColumnarSearchResult(res)

        hit = cr[0][0]
        assert hit.highlight is None

    def test_highlight_compatibility_with_search_result(self):
        """Test highlight compatibility with SearchResult."""
        res = build_base_result(nq=1, topk=3)
        add_varchar_field(res, name="text_field")
        add_highlight_results(res, field_name="text_field")

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_highlight = sr[0][i].highlight
            cr_highlight = cr[0][i].highlight

            assert sr_highlight is not None
            assert cr_highlight is not None
            assert (
                sr_highlight["text_field"]["fragments"] == cr_highlight["text_field"]["fragments"]
            )
            assert sr_highlight["text_field"]["scores"] == cr_highlight["text_field"]["scores"]


# ==============================================================================
# GEOMETRY Type Tests
# ==============================================================================


def add_geometry_field(res: schema_pb2.SearchResultData, name: str = "geometry_field"):
    """Add GEOMETRY field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Geometry
    for i in range(total):
        field.scalars.geometry_wkt_data.data.append(f"POINT({i} {i * 2})")
    res.output_fields.append(name)


class TestGeometryField:
    """Tests for GEOMETRY field type."""

    def test_geometry_field_basic(self):
        """Test GEOMETRY field access."""
        res = build_base_result(nq=1, topk=3)
        add_geometry_field(res)

        cr = ColumnarSearchResult(res)

        assert cr[0][0]["geometry_field"] == "POINT(0 0)"
        assert cr[0][1]["geometry_field"] == "POINT(1 2)"
        assert cr[0][2]["geometry_field"] == "POINT(2 4)"

    def test_geometry_field_compatibility(self):
        """Test GEOMETRY field compatibility with SearchResult."""
        res = build_base_result(nq=1, topk=3)
        add_geometry_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["geometry_field"] == cr[0][i]["geometry_field"]


# ==============================================================================
# TIMESTAMPTZ Type Tests
# ==============================================================================


def add_timestamptz_field(res: schema_pb2.SearchResultData, name: str = "timestamptz_field"):
    """Add TIMESTAMPTZ field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.Timestamptz
    for i in range(total):
        field.scalars.string_data.data.append(f"2026-01-15T10:00:{i:02d}+08:00")
    res.output_fields.append(name)


class TestTimestampTZField:
    """Tests for TIMESTAMPTZ field type."""

    def test_timestamptz_field_basic(self):
        """Test TIMESTAMPTZ field access."""
        res = build_base_result(nq=1, topk=3)
        add_timestamptz_field(res)

        cr = ColumnarSearchResult(res)

        assert cr[0][0]["timestamptz_field"] == "2026-01-15T10:00:00+08:00"
        assert cr[0][1]["timestamptz_field"] == "2026-01-15T10:00:01+08:00"
        assert cr[0][2]["timestamptz_field"] == "2026-01-15T10:00:02+08:00"

    def test_timestamptz_field_compatibility(self):
        """Test TIMESTAMPTZ field compatibility with SearchResult."""
        res = build_base_result(nq=1, topk=3)
        add_timestamptz_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            assert sr[0][i]["timestamptz_field"] == cr[0][i]["timestamptz_field"]


# ==============================================================================
# ==============================================================================
# _ARRAY_OF_STRUCT Type Tests
# ==============================================================================


def add_array_of_struct_field(res: schema_pb2.SearchResultData, name: str = "struct_array_field"):
    """Add _ARRAY_OF_STRUCT field to mock data.
    In Milvus Columnar format, _ARRAY_OF_STRUCT stores sub-fields as ARRAY fields.
    Each row's 'array of struct' is represented by taking the i-th element from
    each sub-field's array.
    """
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.ArrayOfStruct

    # Sub-field 1: nested_int (as an ARRAY of INT64)
    int_field = field.struct_arrays.fields.add()
    int_field.field_name = "nested_int"
    int_field.type = schema_pb2.DataType.Array
    int_field.scalars.array_data.element_type = schema_pb2.DataType.Int64

    # Sub-field 2: nested_str (as an ARRAY of VARCHAR)
    str_field = field.struct_arrays.fields.add()
    str_field.field_name = "nested_str"
    str_field.type = schema_pb2.DataType.Array
    str_field.scalars.array_data.element_type = schema_pb2.DataType.VarChar

    for i in range(total):
        # Row i has an array of 2 structs
        # We populate the ScalarArray for this row
        row_ints = int_field.scalars.array_data.data.add()
        row_ints.long_data.data.extend([i * 10, i * 10 + 1])

        row_strs = str_field.scalars.array_data.data.add()
        row_strs.string_data.data.extend([f"row_{i}_a", f"row_{i}_b"])

    res.output_fields.append(name)


class TestArrayOfStructField:
    """Tests for _ARRAY_OF_STRUCT field type."""

    def test_array_of_struct_field_basic(self):
        """Test _ARRAY_OF_STRUCT field access."""
        res = build_base_result(nq=1, topk=2)
        add_array_of_struct_field(res)

        cr = ColumnarSearchResult(res)
        val = cr[0][0]["struct_array_field"]

        assert isinstance(val, list)
        assert len(val) == 2
        assert val[0]["nested_int"] == 0
        assert val[0]["nested_str"] == "row_0_a"
        assert val[1]["nested_int"] == 1
        assert val[1]["nested_str"] == "row_0_b"

    def test_array_of_struct_compatibility(self):
        """Test _ARRAY_OF_STRUCT field compatibility with SearchResult."""
        res = build_base_result(nq=1, topk=3)
        add_array_of_struct_field(res)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["struct_array_field"]
            cr_val = cr[0][i]["struct_array_field"]
            assert sr_val == cr_val
            assert len(cr_val) == 2
            assert cr_val[0]["nested_int"] == i * 10


# ==============================================================================
# _ARRAY_OF_VECTOR Type Tests
# ==============================================================================


def add_array_of_vector_field(
    res: schema_pb2.SearchResultData, dim: int = 4, name: str = "vector_array_field"
):
    """Add _ARRAY_OF_VECTOR field to mock data."""
    total = sum(res.topks)
    field = res.fields_data.add()
    field.field_name = name
    field.type = schema_pb2.DataType.ArrayOfVector

    # Add vector array data for each row
    for i in range(total):
        vector_data = field.vectors.vector_array.data.add()
        vector_data.dim = dim
        # Each row has 2 vectors
        for v in range(2):
            for d in range(dim):
                vector_data.float_vector.data.append(float(i * 100 + v * 10 + d))

    res.output_fields.append(name)


class TestArrayOfVectorField:
    """Tests for _ARRAY_OF_VECTOR field type."""

    def test_array_of_vector_field_basic(self):
        """Test _ARRAY_OF_VECTOR field access."""
        res = build_base_result(nq=1, topk=3)
        add_array_of_vector_field(res, dim=4)

        cr = ColumnarSearchResult(res)

        val = cr[0][0]["vector_array_field"]
        assert val is not None
        # Should be a list of vectors
        assert isinstance(val, list)
        if len(val) > 0:
            assert isinstance(val[0], list)

    def test_array_of_vector_compatibility(self):
        """Test _ARRAY_OF_VECTOR field compatibility with SearchResult."""
        res = build_base_result(nq=1, topk=3)
        add_array_of_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        for i in range(3):
            sr_val = sr[0][i]["vector_array_field"]
            cr_val = cr[0][i]["vector_array_field"]
            assert sr_val == cr_val


# ==============================================================================
# Complete Type Coverage Summary Test
# ==============================================================================


class TestCompleteTypeCoverage:
    """Ensure all types from SearchResult are supported in ColumnarSearchResult."""

    def test_all_supported_types_together(self):
        """Test all supported types in a single result for comprehensive coverage."""
        res = build_base_result(nq=1, topk=3)

        # Scalar types
        add_bool_field(res)
        add_int8_field(res)
        add_int16_field(res)
        add_int32_field(res)
        add_int64_field(res)
        add_float_field(res)
        add_double_field(res)
        add_varchar_field(res)
        add_json_field(res)
        add_geometry_field(res)
        add_timestamptz_field(res)

        # Vector types
        add_float_vector_field(res, dim=4)
        add_binary_vector_field(res, dim=32)
        add_float16_vector_field(res, dim=4)
        add_bfloat16_vector_field(res, dim=4)
        add_int8_vector_field(res, dim=4)
        add_sparse_vector_field(res)

        # Array types
        add_array_int64_field(res)
        add_array_varchar_field(res)

        # Complex types
        add_array_of_struct_field(res)
        add_array_of_vector_field(res, dim=4)

        sr = SearchResult(res)
        cr = ColumnarSearchResult(res)

        # Verify all fields can be accessed without error
        for i in range(3):
            sr_hit = sr[0][i]
            cr_hit = cr[0][i]

            # Top-level properties
            assert sr_hit.id == cr_hit.id
            assert sr_hit.distance == cr_hit.distance

            # All output fields should be accessible
            for field_name in res.output_fields:
                sr_val = sr_hit[field_name]
                cr_val = cr_hit[field_name]
                # For most types, values should be equal
                # Some container types may need special comparison
                if isinstance(sr_val, (list, bytes, dict)):
                    assert sr_val == cr_val, f"Mismatch for {field_name}: SR={sr_val}, CR={cr_val}"
                else:
                    assert sr_val == cr_val, f"Mismatch for {field_name}"


from pymilvus.grpc_gen import schema_pb2


def test_row_proxy_mapping_compatibility():
    # Setup mock data
    res = schema_pb2.SearchResultData()
    res.num_queries = 1
    res.top_k = 1
    res.topks.extend([1])
    res.ids.int_id.data.extend([100])
    res.scores.extend([0.99])

    # Add a field
    field = res.fields_data.add()
    field.field_name = "age"
    field.type = schema_pb2.DataType.Int64
    field.scalars.long_data.data.extend([25])
    res.output_fields.append("age")

    cr = ColumnarSearchResult(res)
    hit = cr[0][0]

    # 1. Check isinstance
    assert isinstance(hit, collections.abc.Mapping), "RowProxy must be instance of Mapping"
    assert isinstance(hit, collections.abc.Collection), "RowProxy must be instance of Collection"

    # 2. Check strict dict compatibility (it is NOT a dict, but many libraries use Mapping)

    # 3. Check behavior
    assert len(hit) > 0
    assert "age" in hit
    assert hit["age"] == 25

    # 4. Check interaction with things expecting Mapping
    # e.g. dict() constructor
    d = dict(hit)
    assert d["id"] == 100
    assert d["age"] == 25
    assert abs(d["distance"] - 0.99) < 0.0001


class TestCoverageGaps:
    """Targeted tests to reach 100% coverage."""

    def test_to_dict_with_meta_field_explicitly_in_fields(self):
        """Cover line 154: `if field_name == "$meta": continue` in to_dict."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        # Add $meta field explicitly to fields_data
        field = res.fields_data.add()
        field.field_name = "$meta"
        field.type = schema_pb2.DataType.JSON
        field.scalars.json_data.data.extend([b'{"dynamic_a": 1}'])

        # Add a regular field
        field2 = res.fields_data.add()
        field2.field_name = "reg_field"
        field2.type = schema_pb2.DataType.Int64
        field2.scalars.long_data.data.extend([10])

        # Crucial: Add $meta to output_fields so it appears in `self._hits.fields`
        res.output_fields.append("$meta")
        res.output_fields.append("reg_field")

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        # Check to_dict
        d = hit.to_dict()
        assert "entity" in d
        ent = d["entity"]
        assert "reg_field" in ent
        assert ent["reg_field"] == 10
        assert "dynamic_a" in ent
        assert ent["dynamic_a"] == 1
        # $meta should NOT be in the materialized dict
        assert "$meta" not in ent

    def test_to_dict_dynamic_field_conflict(self):
        """Cover line 161-162: Conflict between static and dynamic field names."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        # Static field "conflict_name" = 10
        field = res.fields_data.add()
        field.field_name = "conflict_name"
        field.type = schema_pb2.DataType.Int64
        field.scalars.long_data.data.extend([10])

        # Dynamic field "$meta" contains "conflict_name" = 99
        meta_field = res.fields_data.add()
        meta_field.field_name = "$meta"
        meta_field.type = schema_pb2.DataType.JSON
        meta_field.scalars.json_data.data.extend([b'{"conflict_name": 99, "other_dyn": 2}'])

        res.output_fields.append("conflict_name")
        res.output_fields.append("$meta")

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        d = hit.to_dict()
        # Static field should win
        ent = d["entity"]
        assert ent["conflict_name"] == 10
        assert ent["other_dyn"] == 2

    def test_pk_name_not_id_keys_method(self):
        """Cover line 209: `if self._pk_name != "id": keys.append("id")`."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])
        res.primary_field_name = "custom_pk"

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        k = hit.keys()
        assert "custom_pk" in k
        assert "id" in k
        assert "distance" in k

    def test_meta_cache_hit(self):
        """Cover line 403: `if abs_idx in self._meta_cache:`."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        meta_field = res.fields_data.add()
        meta_field.field_name = "$meta"
        meta_field.type = schema_pb2.DataType.JSON
        # Two dynamic fields to ensure we can access twice
        meta_field.scalars.json_data.data.extend([b'{"a": 1, "b": 2}'])

        # Treat 'a' and 'b' as output fields so they are accessed via get_value
        res.output_fields.extend(["a", "b"])

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        # Access 'a' -> Cache Miss -> Parse JSON -> Store in Cache
        val_a = hit["a"]
        assert val_a == 1

        # Access 'b' -> Cache Hit -> Retrieve from Cache
        val_b = hit["b"]
        assert val_b == 2

    def test_array_of_struct_fallback(self):
        """Cover line 528: `entity_helper.extract_struct_array_from_column_data`."""
        # We need to trigger the fallback.
        # The primary way ARRAY_OF_STRUCT is handled is via the fallback in bind_accessor.
        # But we need to ensure we reach the specific line.
        # The condition is: if dtype == DataType._ARRAY_OF_STRUCT:
        # And ensure struct_arrays is present.

        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        field = res.fields_data.add()
        field.field_name = "struct_arr"
        field.type = schema_pb2.DataType.ArrayOfStruct

        # Setup struct_arrays
        sub_field = field.struct_arrays.fields.add()
        sub_field.field_name = "sub1"
        sub_field.type = schema_pb2.DataType.Array
        sub_field.scalars.array_data.element_type = schema_pb2.DataType.Int64
        # Add data for 1 row
        arr_dat = sub_field.scalars.array_data.data.add()
        arr_dat.long_data.data.extend([1, 2])

        res.output_fields.append("struct_arr")

        cr = ColumnarSearchResult(res)
        val = cr[0][0]["struct_arr"]

        # This should trigger line 528
        assert len(val) == 2
        assert val[0]["sub1"] == 1
        assert val[1]["sub1"] == 2

    def test_array_of_vector_fallback_logic(self):
        """Cover lines 539-540: `f_data = ...; num_vecs = ...`."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        field = res.fields_data.add()
        field.field_name = "vec_arr"
        field.type = schema_pb2.DataType.ArrayOfVector

        # Setup vectors
        vec_data = field.vectors.vector_array.data.add()
        vec_data.dim = 2
        # 2 vectors of dim 2: [1.0, 2.0], [3.0, 4.0]
        vec_data.float_vector.data.extend([1.0, 2.0, 3.0, 4.0])

        res.output_fields.append("vec_arr")

        cr = ColumnarSearchResult(res)
        val = cr[0][0]["vec_arr"]

        # This should trigger lines 539-540
        assert len(val) == 2
        assert val[0] == [1.0, 2.0]
        assert val[1] == [3.0, 4.0]

    def test_batch_access_methods(self):
        """Cover lines 560, 564: get_all_ids, get_all_distances."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 2
        res.topks.extend([2])
        res.ids.int_id.data.extend([100, 101])
        res.scores.extend([0.9, 0.8])

        cr = ColumnarSearchResult(res)
        hits = cr[0]

        assert hits.get_all_ids() == [100, 101]
        assert hits.get_all_distances() == pytest.approx([0.9, 0.8], rel=1e-5)

    def test_array_of_struct_fallback_missing_data(self):
        """Cover fallback returning None for ArrayOfStruct when data is missing."""

        res = schema_pb2.SearchResultData()

        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        # We can't easily make a real Protobuf object not have a field attribute that is in its descriptor
        # So we use a Mock for this specific field data
        mock_field_data = MagicMock()
        mock_field_data.type = schema_pb2.DataType.ArrayOfStruct
        # Ensure 'struct_arrays' attribute is missing or Falsy to trigger the else block
        # Option 1: Mock doesn't have it (raise AttributeError? No, hasattr returns False)
        del mock_field_data.struct_arrays

        # We need to inject this mock into the ColumnarSearchResult
        # Since fields_data_map is built inside __init__, we have to patch it AFTER init
        # OR we can pass a dummy real field in init, and then swap it in the private map.

        field = res.fields_data.add()
        field.field_name = "struct_arr_empty"
        field.type = schema_pb2.DataType.ArrayOfStruct

        res.output_fields.append("struct_arr_empty")

        cr = ColumnarSearchResult(res)

        # SWAP the real field data with our mock
        # The map is _fields_data_map
        hits = cr[0]
        hits._fields_data_map["struct_arr_empty"] = mock_field_data

        hit = hits[0]
        # Should return None because mock_field_data doesn't have struct_arrays
        assert hit["struct_arr_empty"] is None

    def test_array_of_vector_fallback_missing_data(self):
        """Cover fallback returning [] for ArrayOfVector when data is missing."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        field = res.fields_data.add()
        field.field_name = "vec_arr_bad"
        field.type = schema_pb2.DataType.ArrayOfVector
        # Do NOT set vectors.vector_array

        res.output_fields.append("vec_arr_bad")

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        # Should return empty list [] instead of raising Exception
        assert hit["vec_arr_bad"] == []

    def test_unknown_field_type_fallback(self):
        """Cover fallback raising Exception for unknown type."""
        res = schema_pb2.SearchResultData()
        res.num_queries = 1
        res.top_k = 1
        res.topks.extend([1])
        res.ids.int_id.data.extend([100])
        res.scores.extend([0.99])

        field = res.fields_data.add()
        field.field_name = "unknown_field"
        field.type = 9999  # Unknown type

        res.output_fields.append("unknown_field")

        cr = ColumnarSearchResult(res)
        hit = cr[0][0]

        with pytest.raises(MilvusException, match="Unsupported field type: 9999"):
            _ = hit["unknown_field"]
