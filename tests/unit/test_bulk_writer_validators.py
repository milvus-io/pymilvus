import numpy as np
import pytest
from pymilvus.bulk_writer.validators import (
    binary_vector_validator,
    float16_vector_validator,
    float_vector_validator,
    int8_vector_validator,
    sparse_vector_validator,
    struct_validator,
)
from pymilvus.exceptions import MilvusException

# ---------------------------------------------------------------------------
# Parametrized cross-validator tests: invalid_type, invalid_shape, invalid_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validator_func,bad_value,dim,kwargs,message",
    [
        (float_vector_validator, "invalid", 3, {}, r"only accept numpy\.ndarray"),
        (int8_vector_validator, "invalid", 3, {}, r"only accept numpy\.ndarray"),
        (
            float16_vector_validator,
            "invalid",
            3,
            {"is_bfloat": False},
            r"only accept numpy\.ndarray",
        ),
        (
            binary_vector_validator,
            "invalid",
            8,
            {},
            r"only accept numpy\.ndarray, list, bytes",
        ),
        (
            float16_vector_validator,
            "invalid",
            3,
            {"is_bfloat": True},
            r"only accept numpy\.ndarray",
        ),
    ],
    ids=["float", "int8", "float16", "binary", "bfloat16"],
)
def test_invalid_type(validator_func, bad_value, dim, kwargs, message):
    """Validators should raise MilvusException for non-array/non-list input."""
    with pytest.raises(MilvusException, match=message):
        validator_func(bad_value, dim, **kwargs)


@pytest.mark.parametrize(
    "validator_func,arr,dim,kwargs",
    [
        (float_vector_validator, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 4, {}),
        (int8_vector_validator, np.array([[1, 2], [3, 4]], dtype=np.int8), 4, {}),
        (
            float16_vector_validator,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
            4,
            {"is_bfloat": False},
        ),
        (binary_vector_validator, np.array([[0, 1], [2, 3]], dtype=np.uint8), 32, {}),
        (
            float16_vector_validator,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            4,
            {"is_bfloat": True},
        ),
    ],
    ids=["float", "int8", "float16", "binary", "bfloat16"],
)
def test_invalid_numpy_shape(validator_func, arr, dim, kwargs):
    """Validators should raise MilvusException for 2D numpy arrays."""
    if kwargs.get("is_bfloat"):
        arr = arr.astype(pytest.importorskip("ml_dtypes").bfloat16)
    with pytest.raises(MilvusException, match="shape must"):
        validator_func(arr, dim, **kwargs)


@pytest.mark.parametrize(
    "validator_func,arr,dim,kwargs",
    [
        (float_vector_validator, np.array([1.0, 2.0], dtype=np.float32), 3, {}),
        (int8_vector_validator, np.array([1, 2], dtype=np.int8), 3, {}),
        (float16_vector_validator, np.array([1.0, 2.0], dtype=np.float16), 3, {"is_bfloat": False}),
        (binary_vector_validator, np.array([0, 1], dtype=np.uint8), 24, {}),
        (float16_vector_validator, np.array([1.0, 2.0], dtype=np.float32), 3, {"is_bfloat": True}),
    ],
    ids=["float", "int8", "float16", "binary", "bfloat16"],
)
def test_invalid_numpy_length(validator_func, arr, dim, kwargs):
    """Validators should raise MilvusException for numpy arrays with wrong length."""
    if kwargs.get("is_bfloat"):
        arr = arr.astype(pytest.importorskip("ml_dtypes").bfloat16)
    with pytest.raises(MilvusException, match="length must be equal"):
        validator_func(arr, dim, **kwargs)


# ---------------------------------------------------------------------------
# Float vector validator
# ---------------------------------------------------------------------------


class TestFloatVectorValidator:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([1.0, 2.0, 3.0], id="list"),
            pytest.param(np.array([1.0, 2.0, 3.0], dtype=np.float32), id="float32"),
            pytest.param(np.array([1.0, 2.0, 3.0], dtype=np.float64), id="float64"),
        ],
    )
    def test_valid_values(self, value):
        assert float_vector_validator(value, 3) == [1.0, 2.0, 3.0]

    def test_invalid_list_length(self):
        """Test list with wrong dimension"""
        with pytest.raises(
            MilvusException, match="array's length must be equal to vector dimension"
        ):
            float_vector_validator([1.0, 2.0], 3)

    def test_invalid_list_type(self):
        """Test list with non-float elements"""
        with pytest.raises(MilvusException, match="array's element must be float value"):
            float_vector_validator([1.0, 2, 3.0], 3)

    def test_invalid_numpy_dtype(self):
        """Test numpy array with invalid dtype"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(MilvusException, match='dtype must be "float32" or "float64"'):
            float_vector_validator(arr, 3)


# ---------------------------------------------------------------------------
# Binary vector validator
# ---------------------------------------------------------------------------


class TestBinaryVectorValidator:
    @pytest.mark.parametrize(
        "value,dim,expected",
        [
            pytest.param([1, 0, 1, 1, 0, 0, 1, 0], 8, [0xB2], id="bits-msb-first"),
            pytest.param(
                [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
                16,
                [0xB2, 0x4D],
                id="multiple-bytes-msb-first",
            ),
            pytest.param(b"\x00\x01\x80\xff", 32, [0, 1, 128, 255], id="bytes"),
            pytest.param(
                np.array([0, 1, 2, 128, 255], dtype=np.uint8),
                40,
                [0, 1, 2, 128, 255],
                id="uint8",
            ),
        ],
    )
    def test_valid_values(self, value, dim, expected):
        assert binary_vector_validator(value, dim) == expected

    def test_invalid_list_length(self):
        """Test list with wrong dimension"""
        with pytest.raises(
            MilvusException, match="length of the list must be equal to vector dimension"
        ):
            binary_vector_validator([1, 0, 1], 8)

    def test_invalid_bytes_length(self):
        """Test bytes with wrong length"""
        data = b"\x00"
        with pytest.raises(
            MilvusException, match="length of the bytes must be equal to 8x of vector dimension"
        ):
            binary_vector_validator(data, 16)

    def test_invalid_numpy_dtype(self):
        """Test numpy array with invalid dtype"""
        arr = np.array([0, 1, 2], dtype=np.int32)
        with pytest.raises(MilvusException, match='dtype must be "uint8"'):
            binary_vector_validator(arr, 24)


# ---------------------------------------------------------------------------
# Float16 vector validator
# ---------------------------------------------------------------------------


class TestFloat16VectorValidator:
    @pytest.mark.parametrize(
        "is_bfloat,expected",
        [
            pytest.param(False, b"\x00\x3c\x00\x40\x00\x42", id="float16"),
            pytest.param(True, b"\x80\x3f\x00\x40\x40\x40", id="bfloat16"),
        ],
    )
    def test_valid_list(self, is_bfloat, expected):
        if is_bfloat:
            pytest.importorskip("ml_dtypes")
        assert float16_vector_validator([1.0, 2.0, 3.0], 3, is_bfloat=is_bfloat) == expected

    def test_invalid_list_length(self):
        with pytest.raises(
            MilvusException, match="array's length must be equal to vector dimension"
        ):
            float16_vector_validator([1.0, 2.0], 3, is_bfloat=False)

    def test_invalid_list_type(self):
        with pytest.raises(MilvusException, match="array's element must be float value"):
            float16_vector_validator([1.0, 2, 3.0], 3, is_bfloat=False)

    @pytest.mark.parametrize(
        "dtype,is_bfloat,expected",
        [
            pytest.param("float16", False, b"\x00\x3c\x00\x40\x00\x42", id="float16-bytes"),
            pytest.param("bfloat16", True, b"\x80\x3f\x00\x40\x40\x40", id="bfloat16-bytes"),
        ],
    )
    def test_valid_numpy(self, dtype, is_bfloat, expected):
        if dtype == "bfloat16":
            pytest.importorskip("ml_dtypes")
        value = np.array([1.0, 2.0, 3.0], dtype=dtype)

        result = float16_vector_validator(value, 3, is_bfloat=is_bfloat)

        assert isinstance(result, type(expected))
        assert result == expected

    @pytest.mark.parametrize(
        "dtype,is_bfloat,dtype_name",
        [
            pytest.param("int32", False, "float16", id="int32-for-float16"),
            pytest.param("float64", False, "float16", id="float64-for-float16"),
            pytest.param("float32", False, "float16", id="float32-for-float16"),
            pytest.param("int32", True, "bfloat16", id="int32-for-bfloat16"),
            pytest.param("float64", True, "bfloat16", id="float64-for-bfloat16"),
            pytest.param("float32", True, "bfloat16", id="float32-for-bfloat16"),
            pytest.param("float16", True, "bfloat16", id="float16-is-not-bfloat16"),
            pytest.param("bfloat16", False, "float16", id="bfloat16-is-not-float16"),
        ],
    )
    def test_invalid_numpy_dtype(self, dtype, is_bfloat, dtype_name):
        if dtype == "bfloat16":
            pytest.importorskip("ml_dtypes")
        value = np.array([1, 2, 3], dtype=dtype)
        with pytest.raises(MilvusException, match=f'dtype must be "{dtype_name}"'):
            float16_vector_validator(value, 3, is_bfloat=is_bfloat)


# ---------------------------------------------------------------------------
# Int8 vector validator
# ---------------------------------------------------------------------------


class TestInt8VectorValidator:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([-128, -1, 0, 127], id="list"),
            pytest.param(np.array([-128, -1, 0, 127], dtype=np.int8), id="int8"),
        ],
    )
    def test_valid_values(self, value):
        assert int8_vector_validator(value, 4) == [-128, -1, 0, 127]

    def test_invalid_list_length(self):
        """Test list with wrong dimension"""
        with pytest.raises(
            MilvusException, match="array's length must be equal to vector dimension"
        ):
            int8_vector_validator([1, 2], 3)

    def test_invalid_list_type(self):
        """Test list with non-int elements"""
        with pytest.raises(MilvusException, match="array's element must be int value"):
            int8_vector_validator([1, 2.0, 3], 3)

    def test_invalid_numpy_dtype(self):
        """Test numpy array with invalid dtype"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(MilvusException, match='dtype must be "int8"'):
            int8_vector_validator(arr, 3)


# ---------------------------------------------------------------------------
# Sparse vector validator
# ---------------------------------------------------------------------------


class TestSparseVectorValidator:
    @pytest.mark.parametrize(
        "data,expected",
        [
            pytest.param({2: 13.23, 45: 0.54}, {2: 13.23, 45: 0.54}, id="mapping"),
            pytest.param(
                {"indices": [1, 2], "values": [0.1, 0.2]},
                {"indices": [1, 2], "values": [0.1, 0.2]},
                id="indices-values",
            ),
        ],
    )
    def test_valid_values(self, data, expected):
        assert sparse_vector_validator(data) == expected

    def test_invalid_type(self):
        """Test with non-dict input"""
        with pytest.raises(MilvusException, match="only accept dict"):
            sparse_vector_validator([1, 2, 3])

    @pytest.mark.parametrize("index", ["a", True, False], ids=["string", "true", "false"])
    @pytest.mark.parametrize("as_arrays", [False, True], ids=["mapping", "indices-values"])
    def test_invalid_index_type(self, index, as_arrays):
        # bool is an int subclass but is not a sparse-vector index.
        data = {"indices": [2, index], "values": [0.1, 0.2]} if as_arrays else {2: 0.1, index: 0.2}
        with pytest.raises(MilvusException, match="index must be integer"):
            sparse_vector_validator(data)

    @pytest.mark.parametrize("value", ["invalid", 3, True], ids=["string", "int", "bool"])
    @pytest.mark.parametrize("as_arrays", [False, True], ids=["mapping", "indices-values"])
    def test_invalid_value_type(self, value, as_arrays):
        data = {"indices": [1, 2], "values": [0.1, value]} if as_arrays else {1: 0.1, 2: value}
        with pytest.raises(MilvusException, match="value must be float"):
            sparse_vector_validator(data)

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param({}, id="mapping"),
            pytest.param({"indices": [], "values": []}, id="indices-values"),
        ],
    )
    def test_empty_vector(self, data):
        with pytest.raises(MilvusException, match="empty sparse vector is not allowed"):
            sparse_vector_validator(data)

    @pytest.mark.parametrize("field", ["indices", "values"])
    @pytest.mark.parametrize("value", ["invalid", (1, 2)], ids=["string", "tuple"])
    def test_requires_lists(self, field, value):
        data = {"indices": [1, 2], "values": [0.1, 0.2]}
        data[field] = value
        with pytest.raises(MilvusException, match=f"{field} of sparse vector must be a list"):
            sparse_vector_validator(data)

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param({"indices": [1, 2, 3], "values": [0.1, 0.2]}, id="more-indices"),
            pytest.param({"indices": [1, 2], "values": [0.1, 0.2, 0.3]}, id="more-values"),
        ],
    )
    def test_mismatched_indices_values_length(self, data):
        with pytest.raises(MilvusException, match="length of indices and values"):
            sparse_vector_validator(data)


class TestStructValidator:
    @pytest.mark.parametrize(
        "value,max_capacity",
        [
            pytest.param([], 2, id="empty"),
            pytest.param([{"id": 1}], 2, id="below-capacity"),
            pytest.param([{"id": 1}, {"id": 2}], 2, id="at-capacity"),
        ],
    )
    def test_valid_values(self, value, max_capacity):
        assert struct_validator(value, max_capacity) is True

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param({"id": 1}, id="not-a-list"),
            pytest.param([1], id="not-a-dict"),
            pytest.param([{"id": 1}, 2], id="later-element-not-a-dict"),
        ],
    )
    def test_invalid_type(self, value):
        with pytest.raises(MilvusException, match="only accept list of dict"):
            struct_validator(value, 2)

    def test_exceeds_capacity(self):
        with pytest.raises(MilvusException, match="length must be less or equal than max_capacity"):
            struct_validator([{"id": 1}, {"id": 2}], 1)
