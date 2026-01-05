"""
Unit tests for get_column() API in ColumnarHits.

These tests verify correctness of the get_column() method for:
- Different data types (scalar, vector, complex)
- Different return types (list, numpy, arrow)
- Edge cases (empty results, nullable fields, dynamic fields)
- Error handling
"""

from typing import List, Optional

import orjson
import pytest
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2


def create_field_data(
    name: str,
    dtype: int,
    values: list,
    dim: int = 0,
    valid_data: Optional[list] = None,
) -> schema_pb2.FieldData:
    """Helper to create FieldData protobuf message."""
    field_data = schema_pb2.FieldData()
    field_data.field_name = name
    field_data.type = dtype

    if valid_data:
        field_data.valid_data.extend(valid_data)

    if dtype == DataType.BOOL:
        field_data.scalars.bool_data.data.extend(values)
    elif dtype in (DataType.INT8, DataType.INT16, DataType.INT32):
        field_data.scalars.int_data.data.extend(values)
    elif dtype == DataType.INT64:
        field_data.scalars.long_data.data.extend(values)
    elif dtype == DataType.FLOAT:
        field_data.scalars.float_data.data.extend(values)
    elif dtype == DataType.DOUBLE:
        field_data.scalars.double_data.data.extend(values)
    elif dtype in (DataType.VARCHAR, DataType.STRING):
        field_data.scalars.string_data.data.extend(values)
    elif dtype == DataType.JSON:
        field_data.scalars.json_data.data.extend(values)
    elif dtype == DataType.FLOAT_VECTOR:
        field_data.vectors.dim = dim
        # Flatten the vectors
        flat = [v for vec in values for v in vec]
        field_data.vectors.float_vector.data.extend(flat)
    elif dtype == DataType.BINARY_VECTOR:
        field_data.vectors.dim = dim
        binary_data = b"".join(values)
        field_data.vectors.binary_vector = binary_data

    return field_data


def create_search_result_data(
    ids: list,
    scores: list,
    fields_data: List[schema_pb2.FieldData],
    topks: List[int],
    pk_name: str = "id",
    output_fields: Optional[List[str]] = None,
    use_str_id: bool = False,
    offsets: Optional[List[int]] = None,
) -> schema_pb2.SearchResultData:
    """Helper to create SearchResultData protobuf message."""
    res = schema_pb2.SearchResultData()
    res.num_queries = len(topks)
    res.top_k = max(topks) if topks else 0
    res.topks.extend(topks)
    res.primary_field_name = pk_name
    res.output_fields.extend(output_fields or [])

    if use_str_id:
        res.ids.str_id.data.extend([str(i) for i in ids])
    else:
        res.ids.int_id.data.extend(ids)

    res.scores.extend(scores)

    if offsets:
        res.element_indices.data.extend(offsets)

    for fd in fields_data:
        res.fields_data.append(fd)

    return res


class TestGetColumnBasic:
    """Basic tests for get_column() API."""

    def test_get_column_id(self):
        """Test getting id column."""
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[],
            topks=[3],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        # Default list
        ids = hits.get_column("id")
        assert ids == [1, 2, 3]
        assert isinstance(ids, list)

    def test_get_column_distance(self):
        """Test getting distance column."""
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[],
            topks=[3],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        distances = hits.get_column("distance")
        assert distances == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)

    def test_get_column_pk_name(self):
        """Test getting column by pk_name."""
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[],
            topks=[3],
            pk_name="custom_pk",
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        ids = hits.get_column("custom_pk")
        assert ids == [1, 2, 3]

    def test_get_column_offset(self):
        """Test getting offset column."""
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[],
            topks=[3],
            offsets=[10, 20, 30],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        offsets = hits.get_column("offset")
        assert offsets == [10, 20, 30]

        # Test numpy return
        np = pytest.importorskip("numpy")
        offsets_np = hits.get_column("offset", return_type="numpy")
        assert isinstance(offsets_np, np.ndarray)
        assert list(offsets_np) == [10, 20, 30]

    def test_get_column_not_found(self):
        """Test KeyError for non-existent field."""
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[],
            topks=[3],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        with pytest.raises(KeyError, match="Field 'nonexistent' not found"):
            hits.get_column("nonexistent")

    def test_get_column_invalid_return_type(self):
        """Test ValueError for invalid return_type."""
        field = create_field_data("test_field", DataType.INT64, [1, 2, 3])
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[field],
            topks=[3],
            output_fields=["test_field"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        with pytest.raises(ValueError, match="Invalid return_type"):
            hits.get_column("test_field", return_type="invalid")


class TestGetColumnScalarTypes:
    """Tests for scalar type columns."""

    @pytest.mark.parametrize(
        "dtype,values,expected_np_dtype",
        [
            (DataType.BOOL, [True, False, True], "bool"),
            (DataType.INT8, [1, 2, 3], "int8"),
            (DataType.INT16, [100, 200, 300], "int16"),
            (DataType.INT32, [1000, 2000, 3000], "int32"),
            (DataType.INT64, [10000, 20000, 30000], "int64"),
            (DataType.FLOAT, [1.1, 2.2, 3.3], "float32"),
            (DataType.DOUBLE, [1.11, 2.22, 3.33], "float64"),
        ],
    )
    def test_get_column_numeric_types_numpy(self, dtype, values, expected_np_dtype):
        """Test numeric scalar types with numpy return."""
        np = pytest.importorskip("numpy")

        field = create_field_data("test_field", dtype, values)
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[field],
            topks=[3],
            output_fields=["test_field"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        arr = hits.get_column("test_field", return_type="numpy")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype.name == expected_np_dtype
        assert len(arr) == 3

    def test_get_column_varchar_list(self):
        """Test VARCHAR column as list."""
        field = create_field_data("name", DataType.VARCHAR, ["alice", "bob", "charlie"])
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[field],
            topks=[3],
            output_fields=["name"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        names = hits.get_column("name", return_type="list")
        assert names == ["alice", "bob", "charlie"]

    def test_get_column_varchar_numpy(self):
        """Test VARCHAR column as numpy (object dtype)."""
        np = pytest.importorskip("numpy")

        field = create_field_data("name", DataType.VARCHAR, ["alice", "bob", "charlie"])
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[field],
            topks=[3],
            output_fields=["name"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        arr = hits.get_column("name", return_type="numpy")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == object
        assert list(arr) == ["alice", "bob", "charlie"]


class TestGetColumnVectorTypes:
    """Tests for vector type columns."""

    def test_get_column_float_vector_list(self):
        """Test FLOAT_VECTOR as list."""
        vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        field = create_field_data("embedding", DataType.FLOAT_VECTOR, vectors, dim=4)
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[field],
            topks=[3],
            output_fields=["embedding"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        vecs = hits.get_column("embedding", return_type="list")
        assert len(vecs) == 3
        assert vecs[0] == [1.0, 2.0, 3.0, 4.0]

    def test_get_column_float_vector_numpy(self):
        """Test FLOAT_VECTOR as numpy with correct shape."""
        np = pytest.importorskip("numpy")

        vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        field = create_field_data("embedding", DataType.FLOAT_VECTOR, vectors, dim=4)
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[field],
            topks=[3],
            output_fields=["embedding"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        arr = hits.get_column("embedding", return_type="numpy")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float32
        np.testing.assert_array_almost_equal(arr[0], [1.0, 2.0, 3.0, 4.0])

    def test_get_column_binary_vector_numpy(self):
        """Test BINARY_VECTOR as numpy."""
        np = pytest.importorskip("numpy")

        # dim=8 means 1 byte per vector
        vectors = [b"\x01", b"\x02", b"\x03"]
        field = create_field_data("bvec", DataType.BINARY_VECTOR, vectors, dim=8)
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[field],
            topks=[3],
            output_fields=["bvec"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        arr = hits.get_column("bvec", return_type="numpy")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 1)  # 3 vectors, 1 byte each
        assert arr.dtype == np.uint8


class TestGetColumnEdgeCases:
    """Tests for edge cases."""

    def test_get_column_empty_result(self):
        """Test with empty results."""
        res = create_search_result_data(
            ids=[],
            scores=[],
            fields_data=[],
            topks=[0],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        ids = hits.get_column("id")
        assert ids == []

    def test_get_column_single_result(self):
        """Test with single result."""
        field = create_field_data("value", DataType.INT64, [42])
        res = create_search_result_data(
            ids=[1],
            scores=[0.99],
            fields_data=[field],
            topks=[1],
            output_fields=["value"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        values = hits.get_column("value")
        assert values == [42]

    def test_get_column_string_ids(self):
        """Test with string IDs."""
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[],
            topks=[3],
            use_str_id=True,
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        ids = hits.get_column("id")
        assert ids == ["1", "2", "3"]

    def test_get_column_string_ids_numpy(self):
        """Test string IDs as numpy."""
        np = pytest.importorskip("numpy")

        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[],
            topks=[3],
            use_str_id=True,
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        arr = hits.get_column("id", return_type="numpy")
        assert isinstance(arr, np.ndarray)
        # String IDs won't be int64
        assert list(arr) == ["1", "2", "3"]

    def test_get_column_multiple_queries(self):
        """Test with multiple queries (nq > 1)."""
        field = create_field_data("score", DataType.FLOAT, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        res = create_search_result_data(
            ids=[1, 2, 3, 4, 5, 6],
            scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            fields_data=[field],
            topks=[3, 3],  # 2 queries, 3 results each
            output_fields=["score"],
        )
        result = ColumnarSearchResult(res)

        # First query results
        hits0 = result[0]
        scores0 = hits0.get_column("score")
        assert scores0 == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)

        # Second query results
        hits1 = result[1]
        scores1 = hits1.get_column("score")
        assert scores1 == pytest.approx([0.4, 0.5, 0.6], rel=1e-5)


class TestGetColumnDynamic:
    """Tests for dynamic field columns."""

    def test_get_column_dynamic_field_list(self):
        """Test dynamic field from $meta JSON."""

        meta_values = [
            orjson.dumps({"extra_field": 100}),
            orjson.dumps({"extra_field": 200}),
            orjson.dumps({"extra_field": 300}),
        ]
        meta_field = create_field_data("$meta", DataType.JSON, meta_values)
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[meta_field],
            topks=[3],
            output_fields=["extra_field"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        values = hits.get_column("extra_field", return_type="list")
        assert values == [100, 200, 300]

    def test_get_column_dynamic_field_numpy(self):
        """Test dynamic field as numpy returns object array."""
        np = pytest.importorskip("numpy")

        meta_values = [
            orjson.dumps({"dyn": "a"}),
            orjson.dumps({"dyn": "b"}),
            orjson.dumps({"dyn": "c"}),
        ]
        meta_field = create_field_data("$meta", DataType.JSON, meta_values)
        res = create_search_result_data(
            ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            fields_data=[meta_field],
            topks=[3],
            output_fields=["dyn"],
        )
        result = ColumnarSearchResult(res)
        hits = result[0]

        arr = hits.get_column("dyn", return_type="numpy")
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == object
        assert list(arr) == ["a", "b", "c"]
