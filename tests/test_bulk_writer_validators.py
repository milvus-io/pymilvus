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


class TestFloatVectorValidator:
    def test_valid_list(self):
        """Test valid list of floats"""
        result = float_vector_validator([1.0, 2.0, 3.0], 3)
        assert result == [1.0, 2.0, 3.0]

    def test_invalid_list_length(self):
        """Test list with wrong dimension"""
        with pytest.raises(MilvusException, match="array's length must be equal to vector dimension"):
            float_vector_validator([1.0, 2.0], 3)

    def test_invalid_list_type(self):
        """Test list with non-float elements"""
        with pytest.raises(MilvusException, match="array's element must be float value"):
            float_vector_validator([1.0, 2, 3.0], 3)

    def test_valid_numpy_float32(self):
        """Test valid numpy array with float32"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = float_vector_validator(arr, 3)
        assert result == [1.0, 2.0, 3.0]

    def test_valid_numpy_float64(self):
        """Test valid numpy array with float64"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = float_vector_validator(arr, 3)
        assert result == [1.0, 2.0, 3.0]

    def test_invalid_numpy_dtype(self):
        """Test numpy array with invalid dtype"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(MilvusException, match='dtype must be "float32" or "float64"'):
            float_vector_validator(arr, 3)

    def test_invalid_numpy_shape(self):
        """Test numpy array with wrong shape"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with pytest.raises(MilvusException, match="shape must not be one dimension"):
            float_vector_validator(arr, 4)

    def test_invalid_numpy_length(self):
        """Test numpy array with wrong dimension"""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(MilvusException, match="length must be equal to vector dimension"):
            float_vector_validator(arr, 3)

    def test_invalid_type(self):
        """Test with invalid input type"""
        with pytest.raises(MilvusException, match="only accept numpy.ndarray or list"):
            float_vector_validator("invalid", 3)


class TestBinaryVectorValidator:
    def test_valid_list(self):
        """Test valid list of binary values"""
        result = binary_vector_validator([1, 0, 1, 1, 0, 0, 1, 0], 8)
        expected = np.packbits([1, 0, 1, 1, 0, 0, 1, 0], axis=-1).tolist()
        assert result == expected

    def test_invalid_list_length(self):
        """Test list with wrong dimension"""
        with pytest.raises(MilvusException, match="length of the list must be equal to vector dimension"):
            binary_vector_validator([1, 0, 1], 8)

    def test_valid_bytes(self):
        """Test valid bytes input"""
        data = b'\x00\x01'
        result = binary_vector_validator(data, 16)
        assert result == [0, 1]

    def test_invalid_bytes_length(self):
        """Test bytes with wrong length"""
        data = b'\x00'
        with pytest.raises(MilvusException, match="length of the bytes must be equal to 8x of vector dimension"):
            binary_vector_validator(data, 16)

    def test_valid_numpy_uint8(self):
        """Test valid numpy array with uint8"""
        arr = np.array([0, 1, 2], dtype=np.uint8)
        result = binary_vector_validator(arr, 24)
        assert result == [0, 1, 2]

    def test_invalid_numpy_dtype(self):
        """Test numpy array with invalid dtype"""
        arr = np.array([0, 1, 2], dtype=np.int32)
        with pytest.raises(MilvusException, match='dtype must be "uint8"'):
            binary_vector_validator(arr, 24)

    def test_invalid_numpy_shape(self):
        """Test numpy array with wrong shape"""
        arr = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        with pytest.raises(MilvusException, match="shape must be one dimension"):
            binary_vector_validator(arr, 32)

    def test_invalid_numpy_length(self):
        """Test numpy array with wrong dimension"""
        arr = np.array([0, 1], dtype=np.uint8)
        with pytest.raises(MilvusException, match="length must be equal to 8x of vector dimension"):
            binary_vector_validator(arr, 24)

    def test_invalid_type(self):
        """Test with invalid input type"""
        with pytest.raises(MilvusException, match="only accept numpy.ndarray, list, bytes"):
            binary_vector_validator("invalid", 8)


class TestFloat16VectorValidator:
    def test_valid_list_float16(self):
        """Test valid list of floats for float16"""
        result = float16_vector_validator([1.0, 2.0, 3.0], 3, is_bfloat=False)
        assert isinstance(result, bytes)
        # Verify we can reconstruct the array
        arr = np.frombuffer(result, dtype=np.float16)
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not hasattr(np, 'bfloat16'), reason="bfloat16 not available")
    def test_valid_list_bfloat16(self):
        """Test valid list of floats for bfloat16"""
        result = float16_vector_validator([1.0, 2.0, 3.0], 3, is_bfloat=True)
        assert isinstance(result, bytes)

    def test_invalid_list_length(self):
        """Test list with wrong dimension"""
        with pytest.raises(MilvusException, match="array's length must be equal to vector dimension"):
            float16_vector_validator([1.0, 2.0], 3, is_bfloat=False)

    def test_invalid_list_type(self):
        """Test list with non-float elements"""
        with pytest.raises(MilvusException, match="array's element must be float value"):
            float16_vector_validator([1.0, 2, 3.0], 3, is_bfloat=False)

    def test_valid_numpy_float16(self):
        """Test valid numpy array with float16"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = float16_vector_validator(arr, 3, is_bfloat=False)
        assert isinstance(result, bytes)
        assert result == arr.tobytes()

    @pytest.mark.skipif(not hasattr(np, 'bfloat16'), reason="bfloat16 not available")
    def test_valid_numpy_bfloat16(self):
        """Test valid numpy array with bfloat16"""
        arr = np.array([1.0, 2.0, 3.0], dtype='bfloat16')
        result = float16_vector_validator(arr, 3, is_bfloat=True)
        assert isinstance(result, bytes)
        assert result == arr.tobytes()

    def test_invalid_numpy_dtype_float16(self):
        """Test numpy array with wrong dtype for float16"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(MilvusException, match='dtype must be "float16"'):
            float16_vector_validator(arr, 3, is_bfloat=False)

    def test_invalid_numpy_dtype_bfloat16(self):
        """Test numpy array with wrong dtype for bfloat16"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(MilvusException, match='dtype must be "bfloat16"'):
            float16_vector_validator(arr, 3, is_bfloat=True)

    def test_invalid_numpy_shape(self):
        """Test numpy array with wrong shape"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        with pytest.raises(MilvusException, match="shape must not be one dimension"):
            float16_vector_validator(arr, 4, is_bfloat=False)

    def test_invalid_numpy_length(self):
        """Test numpy array with wrong dimension"""
        arr = np.array([1.0, 2.0], dtype=np.float16)
        with pytest.raises(MilvusException, match="length must be equal to vector dimension"):
            float16_vector_validator(arr, 3, is_bfloat=False)

    def test_invalid_type(self):
        """Test with invalid input type"""
        with pytest.raises(MilvusException, match="only accept numpy.ndarray or list"):
            float16_vector_validator("invalid", 3, is_bfloat=False)


class TestInt8VectorValidator:
    def test_valid_list(self):
        """Test valid list of integers"""
        result = int8_vector_validator([1, 2, 3], 3)
        assert result == [1, 2, 3]

    def test_invalid_list_length(self):
        """Test list with wrong dimension"""
        with pytest.raises(MilvusException, match="array's length must be equal to vector dimension"):
            int8_vector_validator([1, 2], 3)

    def test_invalid_list_type(self):
        """Test list with non-int elements"""
        with pytest.raises(MilvusException, match="array's element must be int value"):
            int8_vector_validator([1, 2.0, 3], 3)

    def test_valid_numpy_int8(self):
        """Test valid numpy array with int8"""
        arr = np.array([1, 2, 3], dtype=np.int8)
        result = int8_vector_validator(arr, 3)
        assert result == [1, 2, 3]

    def test_invalid_numpy_dtype(self):
        """Test numpy array with invalid dtype"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(MilvusException, match='dtype must be "int8"'):
            int8_vector_validator(arr, 3)

    def test_invalid_numpy_shape(self):
        """Test numpy array with wrong shape"""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int8)
        with pytest.raises(MilvusException, match="shape must not be one dimension"):
            int8_vector_validator(arr, 4)

    def test_invalid_numpy_length(self):
        """Test numpy array with wrong dimension"""
        arr = np.array([1, 2], dtype=np.int8)
        with pytest.raises(MilvusException, match="length must be equal to vector dimension"):
            int8_vector_validator(arr, 3)

    def test_invalid_type(self):
        """Test with invalid input type"""
        with pytest.raises(MilvusException, match="only accept numpy.ndarray or list"):
            int8_vector_validator("invalid", 3)


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
