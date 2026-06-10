import numpy as np
import pytest
from pymilvus.client.types import DataType
from pymilvus.orm.types import (
    infer_dtype_by_scalar_data,
    infer_dtype_bydata,
    is_float_datatype,
    is_integer_datatype,
    is_numeric_datatype,
    map_numpy_dtype_to_datatype,
)


# ---------------------------------------------------------------------------
# is_integer_datatype
# ---------------------------------------------------------------------------
class TestIsIntegerDatatype:
    @pytest.mark.parametrize(
        "dt",
        [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64],
    )
    def test_true_for_integer_types(self, dt):
        assert is_integer_datatype(dt) is True

    @pytest.mark.parametrize(
        "dt",
        [DataType.FLOAT, DataType.DOUBLE, DataType.VARCHAR, DataType.BOOL, DataType.UNKNOWN],
    )
    def test_false_for_non_integer_types(self, dt):
        assert is_integer_datatype(dt) is False


# ---------------------------------------------------------------------------
# is_float_datatype
# ---------------------------------------------------------------------------
class TestIsFloatDatatype:
    def test_true_for_float(self):
        assert is_float_datatype(DataType.FLOAT) is True

    @pytest.mark.parametrize(
        "dt",
        [DataType.DOUBLE, DataType.INT64, DataType.VARCHAR, DataType.BOOL],
    )
    def test_false_for_non_float(self, dt):
        assert is_float_datatype(dt) is False


# ---------------------------------------------------------------------------
# is_numeric_datatype
# ---------------------------------------------------------------------------
class TestIsNumericDatatype:
    @pytest.mark.parametrize(
        "dt",
        [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.FLOAT],
    )
    def test_true_for_numeric(self, dt):
        assert is_numeric_datatype(dt) is True

    @pytest.mark.parametrize(
        "dt",
        [DataType.DOUBLE, DataType.VARCHAR, DataType.BOOL, DataType.UNKNOWN],
    )
    def test_false_for_non_numeric(self, dt):
        assert is_numeric_datatype(dt) is False


# ---------------------------------------------------------------------------
# infer_dtype_by_scalar_data
# ---------------------------------------------------------------------------
class TestInferDtypeByScalarData:
    @pytest.mark.parametrize(
        "data, dtype, expected",
        [
            # list -> ARRAY
            ([1, 2, 3], None, DataType.ARRAY),
            # np.float32 -> FLOAT (regardless of dtype arg)
            (np.float32(1.0), None, DataType.FLOAT),
            # plain float with dtype=FLOAT -> FLOAT
            (1.0, DataType.FLOAT, DataType.FLOAT),
            # plain float without dtype -> DOUBLE
            (1.0, None, DataType.DOUBLE),
            # np.float64 -> DOUBLE
            (np.float64(1.0), None, DataType.DOUBLE),
            # np.double -> DOUBLE
            (np.double(1.0), None, DataType.DOUBLE),
            # bool -> BOOL (must come before int check)
            (True, None, DataType.BOOL),
            (False, None, DataType.BOOL),
            # int -> INT64
            (42, None, DataType.INT64),
            # np.int64 -> INT64
            (np.int64(42), None, DataType.INT64),
            # np.int32 -> INT32
            (np.int32(42), None, DataType.INT32),
            # np.int16 -> INT16
            (np.int16(42), None, DataType.INT16),
            # np.int8 -> INT8
            (np.int8(42), None, DataType.INT8),
            # str -> VARCHAR
            ("hello", None, DataType.VARCHAR),
            # bytes -> BINARY_VECTOR
            (b"\x00\x01", None, DataType.BINARY_VECTOR),
            # unknown type -> UNKNOWN
            (object(), None, DataType.UNKNOWN),
        ],
    )
    def test_inference(self, data, dtype, expected):
        result = infer_dtype_by_scalar_data(data, dtype)
        assert result == expected


# ---------------------------------------------------------------------------
# infer_dtype_bydata
# ---------------------------------------------------------------------------
class TestInferDtypeBydata:
    def test_scalar_int(self):
        assert infer_dtype_bydata(42) == DataType.INT64

    def test_scalar_float(self):
        assert infer_dtype_bydata(3.14) == DataType.DOUBLE

    def test_scalar_str(self):
        assert infer_dtype_bydata("hello") == DataType.VARCHAR

    def test_scalar_bool(self):
        assert infer_dtype_bydata(True) == DataType.BOOL

    def test_dict_returns_json(self):
        assert infer_dtype_bydata({"key": "value"}) == DataType.JSON

    def test_list_of_ints_returns_float_vector(self):
        # infer_dtype on [1, 2, 3] gives "integer" -> INT64 -> is_numeric -> FLOAT_VECTOR
        assert infer_dtype_bydata([1, 2, 3]) == DataType.FLOAT_VECTOR

    def test_list_of_floats_returns_float_vector(self):
        assert infer_dtype_bydata([1.0, 2.0, 3.0]) == DataType.FLOAT_VECTOR

    def test_list_of_strings_returns_array(self):
        # infer_dtype on ["a", "b"] gives "string" -> VARCHAR -> not numeric, not UNKNOWN -> ARRAY
        assert infer_dtype_bydata(["a", "b"]) == DataType.ARRAY

    def test_numpy_int_array(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        assert infer_dtype_bydata(arr) == DataType.FLOAT_VECTOR

    def test_numpy_float_array(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        assert infer_dtype_bydata(arr) == DataType.FLOAT_VECTOR

    def test_list_of_mixed_types_returns_float_vector(self):
        # Any list input is treated as a vector
        assert infer_dtype_bydata([1, "a", 2.0]) == DataType.FLOAT_VECTOR

    def test_empty_list_fallback(self):
        # Empty list: infer_dtype may raise or return unexpected; fallback path
        # empty list -> infer_dtype gives "empty" which is not in dtype_str_map -> UNKNOWN
        # Then element check: data[0] raises IndexError -> elem is None -> skip
        result = infer_dtype_bydata([])
        assert result == DataType.UNKNOWN

    def test_list_like_with_scalar_elements_returns_float_vector(self):
        """A list-like with numeric elements is treated as FLOAT_VECTOR."""

        class NumericList:
            def __iter__(self):
                return iter([42])

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                if idx == 0:
                    return 42
                raise IndexError

        result = infer_dtype_bydata(NumericList())
        assert result == DataType.FLOAT_VECTOR

    def test_numpy_like_with_dtype_returns_float_vector(self):
        """Object with numpy dtype and __len__ is treated as a list-like numeric → FLOAT_VECTOR."""

        class NumpyLike:
            dtype = np.dtype("float32")

            def __iter__(self):
                raise TypeError

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                raise IndexError

        result = infer_dtype_bydata(NumpyLike())
        assert result == DataType.FLOAT_VECTOR


# ---------------------------------------------------------------------------
# infer_dtype_bydata — fallback paths (lines 111-132)
# ---------------------------------------------------------------------------


class TestInferDtypeBydataFallbacks:
    def test_type_error_falls_to_element_check(self):
        """When infer_dtype raises TypeError, fall through to element[0] check."""

        class RaiseOnIter:
            def __len__(self):
                return 1

            def __iter__(self):
                raise TypeError("bad")

            def __getitem__(self, idx):
                if idx == 0:
                    return 42
                raise IndexError

        assert infer_dtype_bydata(RaiseOnIter()) == DataType.INT64

    def test_element_index_error_gives_none(self):
        """When element[0] raises IndexError, elem is None -> stays UNKNOWN."""

        class EmptyRaiseOnIter:
            def __len__(self):
                return 0

            def __iter__(self):
                raise TypeError("bad")

            def __getitem__(self, idx):
                raise IndexError

        assert infer_dtype_bydata(EmptyRaiseOnIter()) == DataType.UNKNOWN

    def test_numpy_dtype_fallback(self):
        """When not list-like but has .dtype, use numpy dtype mapping."""

        class NotListLikeWithDtype:
            dtype = np.dtype("int64")

            def __getitem__(self, idx):
                raise IndexError

        assert infer_dtype_bydata(NotListLikeWithDtype()) == DataType.INT64

    def test_numpy_dtype_fallback_float(self):
        """Numpy dtype fallback for float32."""

        class NotListLikeFloat:
            dtype = np.dtype("float32")

            def __getitem__(self, idx):
                raise IndexError

        assert infer_dtype_bydata(NotListLikeFloat()) == DataType.FLOAT


# ---------------------------------------------------------------------------
# map_numpy_dtype_to_datatype
# ---------------------------------------------------------------------------
class TestMapNumpyDtypeToDatatype:
    @pytest.mark.parametrize(
        "np_dtype, expected",
        [
            (np.dtype("bool"), DataType.BOOL),
            (np.dtype("int8"), DataType.INT8),
            (np.dtype("int16"), DataType.INT16),
            (np.dtype("int32"), DataType.INT32),
            (np.dtype("int64"), DataType.INT64),
            (np.dtype("float32"), DataType.FLOAT),
            (np.dtype("float64"), DataType.DOUBLE),
        ],
    )
    def test_known_dtypes(self, np_dtype, expected):
        assert map_numpy_dtype_to_datatype(np_dtype) == expected

    def test_unknown_dtype_returns_unknown(self):
        assert map_numpy_dtype_to_datatype(np.dtype("complex128")) == DataType.UNKNOWN
