import numpy as np
import pytest
from pymilvus.bulk_writer.validators import (
    binary_vector_validator,
    float16_vector_validator,
    float_vector_validator,
    int8_vector_validator,
    sparse_vector_validator,
)
from pymilvus.exceptions import MilvusException

# ---------------------------------------------------------------------------
# Parametrized cross-validator tests: invalid_type, invalid_shape, invalid_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validator_func,bad_value,dim,kwargs",
    [
        (float_vector_validator, "invalid", 3, {}),
        (int8_vector_validator, "invalid", 3, {}),
        (float16_vector_validator, "invalid", 3, {"is_bfloat": False}),
        (binary_vector_validator, "invalid", 8, {}),
    ],
)
def test_invalid_type(validator_func, bad_value, dim, kwargs):
    """Validators should raise MilvusException for non-array/non-list input."""
    with pytest.raises(MilvusException, match=r"only accept numpy\.ndarray"):
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
    ],
)
def test_invalid_numpy_shape(validator_func, arr, dim, kwargs):
    """Validators should raise MilvusException for 2D numpy arrays."""
    with pytest.raises(MilvusException, match="shape must"):
        validator_func(arr, dim, **kwargs)


@pytest.mark.parametrize(
    "validator_func,arr,dim,kwargs",
    [
        (float_vector_validator, np.array([1.0, 2.0], dtype=np.float32), 3, {}),
        (int8_vector_validator, np.array([1, 2], dtype=np.int8), 3, {}),
        (float16_vector_validator, np.array([1.0, 2.0], dtype=np.float16), 3, {"is_bfloat": False}),
        (binary_vector_validator, np.array([0, 1], dtype=np.uint8), 24, {}),
    ],
)
def test_invalid_numpy_length(validator_func, arr, dim, kwargs):
    """Validators should raise MilvusException for numpy arrays with wrong length."""
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
            pytest.param(b"\x00\x01", 16, [0, 1], id="bytes"),
            pytest.param(np.array([0, 1, 2], dtype=np.uint8), 24, [0, 1, 2], id="uint8"),
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

    def test_invalid_type(self):
        """Test with invalid input type"""
        with pytest.raises(MilvusException, match=r"only accept numpy\.ndarray, list, bytes"):
            binary_vector_validator("invalid", 8)


# ---------------------------------------------------------------------------
# Float16 vector validator
# ---------------------------------------------------------------------------


class TestFloat16VectorValidator:
    @pytest.mark.parametrize("is_bfloat", [False, True], ids=["float16", "bfloat16"])
    def test_valid_list(self, is_bfloat):
        assert float16_vector_validator([1.0, 2.0, 3.0], 3, is_bfloat=is_bfloat) == [1.0, 2.0, 3.0]

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
            pytest.param("float32", False, [1.0, 2.0, 3.0], id="float32-for-float16"),
            pytest.param("float64", False, [1.0, 2.0, 3.0], id="float64-for-float16"),
            pytest.param("float32", True, [1.0, 2.0, 3.0], id="float32-for-bfloat16"),
            pytest.param("float64", True, [1.0, 2.0, 3.0], id="float64-for-bfloat16"),
        ],
    )
    def test_valid_numpy(self, dtype, is_bfloat, expected):
        if dtype == "bfloat16":
            pytest.importorskip("ml_dtypes")
        value = np.array([1.0, 2.0, 3.0], dtype=dtype)

        result = float16_vector_validator(value, 3, is_bfloat=is_bfloat)

        assert isinstance(result, type(expected))
        assert result == expected

    @pytest.mark.parametrize("is_bfloat,dtype_name", [(False, "float16"), (True, "bfloat16")])
    def test_invalid_numpy_dtype(self, is_bfloat, dtype_name):
        value = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(
            MilvusException, match=f'dtype must be "{dtype_name}", "float32" or "float64"'
        ):
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
    def test_valid_dict(self):
        """Test valid dict format"""
        data = {2: 13.23, 45: 0.54}
        result = sparse_vector_validator(data)
        assert result == data

    def test_valid_indices_values_format(self):
        """Test valid indices/values format"""
        data = {"indices": [1, 2], "values": [0.1, 0.2]}
        result = sparse_vector_validator(data)
        assert result == data

    def test_invalid_type(self):
        """Test with non-dict input"""
        with pytest.raises(MilvusException, match="only accept dict"):
            sparse_vector_validator([1, 2, 3])

    def test_invalid_index_type(self):
        """Test dict with non-integer index"""
        data = {"a": 0.5, 2: 0.3}
        with pytest.raises(MilvusException, match="index must be integer"):
            sparse_vector_validator(data)

    def test_invalid_value_type(self):
        """Test dict with non-float value"""
        data = {1: 0.5, 2: 3}
        with pytest.raises(MilvusException, match="value must be float"):
            sparse_vector_validator(data)

    def test_empty_dict(self):
        """Test empty dict"""
        with pytest.raises(MilvusException, match="empty sparse vector is not allowed"):
            sparse_vector_validator({})

    def test_invalid_indices_type(self):
        """Test with non-list indices"""
        data = {"indices": "invalid", "values": [0.1, 0.2]}
        with pytest.raises(MilvusException, match="indices of sparse vector must be a list"):
            sparse_vector_validator(data)

    def test_invalid_values_type(self):
        """Test with non-list values"""
        data = {"indices": [1, 2], "values": "invalid"}
        with pytest.raises(MilvusException, match="values of sparse vector must be a list"):
            sparse_vector_validator(data)

    def test_bool_index_in_dict_format(self):
        """Test dict with bool index (True/False should be rejected)"""
        data = {True: 0.5, 2: 0.3}
        with pytest.raises(MilvusException, match="index must be integer"):
            sparse_vector_validator(data)

    def test_bool_index_in_indices_format(self):
        """Test indices/values format with bool index"""
        data = {"indices": [True, 2], "values": [0.1, 0.2]}
        with pytest.raises(MilvusException, match="index must be integer"):
            sparse_vector_validator(data)

    def test_mismatched_indices_values_length(self):
        """Test with mismatched indices and values length"""
        data = {"indices": [1, 2, 3], "values": [0.1, 0.2]}
        with pytest.raises(MilvusException, match="length of indices and values"):
            sparse_vector_validator(data)

    def test_empty_indices_values(self):
        """Test with empty indices and values"""
        data = {"indices": [], "values": []}
        with pytest.raises(MilvusException, match="empty sparse vector is not allowed"):
            sparse_vector_validator(data)

    def test_invalid_index_in_indices_format(self):
        """Test with invalid index type in indices/values format"""
        data = {"indices": ["a", 2], "values": [0.1, 0.2]}
        with pytest.raises(MilvusException, match="index must be integer"):
            sparse_vector_validator(data)

    def test_invalid_value_in_indices_format(self):
        """Test with invalid value type in indices/values format"""
        data = {"indices": [1, 2], "values": [0.1, "invalid"]}
        with pytest.raises(MilvusException, match="value must be float"):
            sparse_vector_validator(data)
