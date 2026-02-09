import numpy as np
import pytest
from pymilvus.client.entity_helper import (
    convert_to_array,
    convert_to_array_of_vector,
    extract_array_row_data_no_validity,
    extract_array_row_data_with_validity,
    get_array_length,
    get_array_value_at_index,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2 as schema_types


class TestExtractArrayRowDataFunctions:
    """Test extract_array_row_data_with_validity and extract_array_row_data_no_validity"""

    def test_extract_array_row_data_int8_no_validity(self):
        """Test extracting INT8 array data without validity"""

        field_data = schema_types.FieldData()
        field_data.field_name = "arr_field"
        field_data.scalars.array_data.element_type = DataType.INT8

        # Add array data
        arr1 = schema_types.ScalarField()
        arr1.int_data.data.extend([1, 2, 3])
        field_data.scalars.array_data.data.append(arr1)

        arr2 = schema_types.ScalarField()
        arr2.int_data.data.extend([4, 5])
        field_data.scalars.array_data.data.append(arr2)

        entity_rows = [{}, {}]
        extract_array_row_data_no_validity(field_data, entity_rows, 2)

        assert entity_rows[0]["arr_field"] == [1, 2, 3]
        assert entity_rows[1]["arr_field"] == [4, 5]

    def test_extract_array_row_data_int16_no_validity(self):
        """Test extracting INT16 array data without validity"""

        field_data = schema_types.FieldData()
        field_data.field_name = "arr_field"
        field_data.scalars.array_data.element_type = DataType.INT16

        arr1 = schema_types.ScalarField()
        arr1.int_data.data.extend([100, 200])
        field_data.scalars.array_data.data.append(arr1)

        entity_rows = [{}]
        extract_array_row_data_no_validity(field_data, entity_rows, 1)

        assert entity_rows[0]["arr_field"] == [100, 200]

    def test_extract_array_row_data_int32_no_validity(self):
        """Test extracting INT32 array data without validity"""

        field_data = schema_types.FieldData()
        field_data.field_name = "arr_field"
        field_data.scalars.array_data.element_type = DataType.INT32

        arr1 = schema_types.ScalarField()
        arr1.int_data.data.extend([1000, 2000, 3000])
        field_data.scalars.array_data.data.append(arr1)

        entity_rows = [{}]
        extract_array_row_data_no_validity(field_data, entity_rows, 1)

        assert entity_rows[0]["arr_field"] == [1000, 2000, 3000]

    def test_extract_array_row_data_with_validity_int64(self):
        """Test extracting INT64 array data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.field_name = "arr_field"
        field_data.scalars.array_data.element_type = DataType.INT64
        field_data.valid_data.extend([True, False, True])

        arr1 = schema_types.ScalarField()
        arr1.long_data.data.extend([1, 2, 3])
        field_data.scalars.array_data.data.append(arr1)

        arr2 = schema_types.ScalarField()  # Will be None due to valid_data[1] = False
        field_data.scalars.array_data.data.append(arr2)

        arr3 = schema_types.ScalarField()
        arr3.long_data.data.extend([7, 8, 9])
        field_data.scalars.array_data.data.append(arr3)

        entity_rows = [{}, {}, {}]
        extract_array_row_data_with_validity(field_data, entity_rows, 3)

        assert entity_rows[0]["arr_field"] == [1, 2, 3]
        assert entity_rows[1]["arr_field"] is None
        assert entity_rows[2]["arr_field"] == [7, 8, 9]

    def test_extract_array_row_data_with_validity_bool(self):
        """Test extracting BOOL array data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.field_name = "bool_arr"
        field_data.scalars.array_data.element_type = DataType.BOOL
        field_data.valid_data.extend([True, True])

        arr1 = schema_types.ScalarField()
        arr1.bool_data.data.extend([True, False])
        field_data.scalars.array_data.data.append(arr1)

        arr2 = schema_types.ScalarField()
        arr2.bool_data.data.extend([False, True, False])
        field_data.scalars.array_data.data.append(arr2)

        entity_rows = [{}, {}]
        extract_array_row_data_with_validity(field_data, entity_rows, 2)

        assert entity_rows[0]["bool_arr"] == [True, False]
        assert entity_rows[1]["bool_arr"] == [False, True, False]

    def test_extract_array_row_data_with_validity_varchar(self):
        """Test extracting VARCHAR array data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.field_name = "str_arr"
        field_data.scalars.array_data.element_type = DataType.VARCHAR
        field_data.valid_data.extend([True, False])

        arr1 = schema_types.ScalarField()
        arr1.string_data.data.extend(["hello", "world"])
        field_data.scalars.array_data.data.append(arr1)

        arr2 = schema_types.ScalarField()
        field_data.scalars.array_data.data.append(arr2)

        entity_rows = [{}, {}]
        extract_array_row_data_with_validity(field_data, entity_rows, 2)

        assert entity_rows[0]["str_arr"] == ["hello", "world"]
        assert entity_rows[1]["str_arr"] is None

    def test_extract_array_row_data_with_validity_float(self):
        """Test extracting FLOAT array data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.field_name = "float_arr"
        field_data.scalars.array_data.element_type = DataType.FLOAT
        field_data.valid_data.extend([True])

        arr1 = schema_types.ScalarField()
        arr1.float_data.data.extend([1.5, 2.5, 3.5])
        field_data.scalars.array_data.data.append(arr1)

        entity_rows = [{}]
        extract_array_row_data_with_validity(field_data, entity_rows, 1)

        assert entity_rows[0]["float_arr"][0] == pytest.approx(1.5)

    def test_extract_array_row_data_with_validity_double(self):
        """Test extracting DOUBLE array data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.field_name = "double_arr"
        field_data.scalars.array_data.element_type = DataType.DOUBLE
        field_data.valid_data.extend([True])

        arr1 = schema_types.ScalarField()
        arr1.double_data.data.extend([1.111, 2.222])
        field_data.scalars.array_data.data.append(arr1)

        entity_rows = [{}]
        extract_array_row_data_with_validity(field_data, entity_rows, 1)

        assert entity_rows[0]["double_arr"][0] == pytest.approx(1.111)


class TestArrayConversion:
    """Test convert_to_array for scalar types"""

    def test_convert_bool_array(self):

        result = convert_to_array([True, False], {"element_type": DataType.BOOL})
        assert list(result.bool_data.data) == [True, False]

    def test_convert_float_array(self):

        result = convert_to_array([1.5, 2.5], {"element_type": DataType.FLOAT})
        assert list(result.float_data.data) == pytest.approx([1.5, 2.5])

    def test_convert_double_array(self):

        result = convert_to_array([1.11, 2.22], {"element_type": DataType.DOUBLE})
        assert list(result.double_data.data) == pytest.approx([1.11, 2.22])

    def test_get_array_value_at_index_extended(self):

        # Test Double
        arr_data = convert_to_array([1.1, 2.2], {"element_type": DataType.DOUBLE})
        assert get_array_value_at_index(arr_data, 0) == pytest.approx(1.1)
        assert get_array_value_at_index(arr_data, 1) == pytest.approx(2.2)
        assert get_array_value_at_index(arr_data, 2) is None

        # Test Bool
        arr_data = convert_to_array([True, False], {"element_type": DataType.BOOL})
        assert get_array_value_at_index(arr_data, 0) is True
        assert get_array_value_at_index(arr_data, 1) is False


class TestConvertToArrayExtended:
    """Test convert_to_array and convert_to_array_arr for various element types"""

    def test_convert_to_array_int8(self):
        """Test converting INT8 array"""
        field_info = {"name": "arr_field", "element_type": DataType.INT8}
        result = convert_to_array([1, 2, 3], field_info)
        assert list(result.int_data.data) == [1, 2, 3]

    def test_convert_to_array_int16(self):
        """Test converting INT16 array"""
        field_info = {"name": "arr_field", "element_type": DataType.INT16}
        result = convert_to_array([100, 200, 300], field_info)
        assert list(result.int_data.data) == [100, 200, 300]

    def test_convert_to_array_int32(self):
        """Test converting INT32 array"""
        field_info = {"name": "arr_field", "element_type": DataType.INT32}
        result = convert_to_array([1000, 2000], field_info)
        assert list(result.int_data.data) == [1000, 2000]

    def test_convert_to_array_string_type(self):
        """Test converting STRING (alias) array"""
        field_info = {"name": "arr_field", "element_type": DataType.STRING}
        result = convert_to_array(["a", "b"], field_info)
        assert list(result.string_data.data) == ["a", "b"]

    def test_convert_to_array_numpy_input(self):
        """Test converting numpy array"""
        field_info = {"name": "arr_field", "element_type": DataType.INT64}
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = convert_to_array(arr, field_info)
        assert list(result.long_data.data) == [1, 2, 3]


class TestConvertToArrayOfVector:
    """Test convert_to_array_of_vector function"""

    def test_convert_empty_array_of_vector(self):
        """Test converting empty array of vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT_VECTOR,
            "params": {"dim": "4"},
        }
        result = convert_to_array_of_vector([], field_info)
        assert result.dim == 4
        assert len(result.float_vector.data) == 0

    def test_convert_array_of_float_vectors(self):
        """Test converting array of float vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT_VECTOR,
            "params": {"dim": 2},
        }
        vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = convert_to_array_of_vector(vectors, field_info)
        assert result.dim == 2
        assert list(result.float_vector.data) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_convert_array_of_float_vectors_numpy(self):
        """Test converting numpy array of vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT_VECTOR,
            "params": {"dim": 2},
        }
        vectors = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        result = convert_to_array_of_vector(vectors, field_info)
        assert list(result.float_vector.data) == [1.0, 2.0, 3.0, 4.0]

    def test_convert_array_of_float16_vectors(self):
        """Test converting array of float16 vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT16_VECTOR,
            "params": {"dim": 2},
        }
        v1 = np.array([1.0, 2.0], dtype=np.float16)
        v2 = np.array([3.0, 4.0], dtype=np.float16)
        result = convert_to_array_of_vector([v1, v2], field_info)
        assert result.dim == 2
        expected = v1.view(np.uint8).tobytes() + v2.view(np.uint8).tobytes()
        assert result.float16_vector == expected

    def test_convert_array_of_float16_vectors_bytes(self):
        """Test converting array of float16 vectors from raw bytes"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT16_VECTOR,
            "params": {"dim": 2},
        }
        v1_bytes = np.array([1.0, 2.0], dtype=np.float16).view(np.uint8).tobytes()
        v2_bytes = np.array([3.0, 4.0], dtype=np.float16).view(np.uint8).tobytes()
        result = convert_to_array_of_vector([v1_bytes, v2_bytes], field_info)
        assert result.float16_vector == v1_bytes + v2_bytes

    def test_convert_empty_array_of_float16_vectors(self):
        """Test converting empty array of float16 vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT16_VECTOR,
            "params": {"dim": 4},
        }
        result = convert_to_array_of_vector([], field_info)
        assert result.dim == 4
        assert result.float16_vector == b""

    def test_convert_array_of_bfloat16_vectors(self):
        """Test converting array of bfloat16 vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.BFLOAT16_VECTOR,
            "params": {"dim": 2},
        }
        # Use raw bytes since bfloat16 may not be available in all numpy versions
        v1_bytes = b"\x00\x3f\x00\x40"  # 4 bytes for dim=2
        v2_bytes = b"\x00\x41\x00\x42"
        result = convert_to_array_of_vector([v1_bytes, v2_bytes], field_info)
        assert result.dim == 2
        assert result.bfloat16_vector == v1_bytes + v2_bytes

    def test_convert_empty_array_of_bfloat16_vectors(self):
        """Test converting empty array of bfloat16 vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.BFLOAT16_VECTOR,
            "params": {"dim": 4},
        }
        result = convert_to_array_of_vector([], field_info)
        assert result.dim == 4
        assert result.bfloat16_vector == b""

    def test_convert_array_of_int8_vectors(self):
        """Test converting array of int8 vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.INT8_VECTOR,
            "params": {"dim": 3},
        }
        v1 = np.array([1, -2, 3], dtype=np.int8)
        v2 = np.array([4, -5, 6], dtype=np.int8)
        result = convert_to_array_of_vector([v1, v2], field_info)
        assert result.dim == 3
        expected = v1.view(np.uint8).tobytes() + v2.view(np.uint8).tobytes()
        assert result.int8_vector == expected

    def test_convert_empty_array_of_int8_vectors(self):
        """Test converting empty array of int8 vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.INT8_VECTOR,
            "params": {"dim": 3},
        }
        result = convert_to_array_of_vector([], field_info)
        assert result.dim == 3
        assert result.int8_vector == b""

    def test_convert_array_of_binary_vectors(self):
        """Test converting array of binary vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.BINARY_VECTOR,
            "params": {"dim": 8},
        }
        v1 = b"\xff"
        v2 = b"\x0f"
        result = convert_to_array_of_vector([v1, v2], field_info)
        assert result.dim == 8
        assert result.binary_vector == b"\xff\x0f"

    def test_convert_empty_array_of_binary_vectors(self):
        """Test converting empty array of binary vectors"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.BINARY_VECTOR,
            "params": {"dim": 16},
        }
        result = convert_to_array_of_vector([], field_info)
        assert result.dim == 16
        assert result.binary_vector == b""

    def test_convert_array_of_vector_unsupported_type(self):
        """Test unsupported element type raises error"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.SPARSE_FLOAT_VECTOR,
            "params": {"dim": 8},
        }
        with pytest.raises(ParamError, match="Unsupported element type"):
            convert_to_array_of_vector([[1, 2]], field_info)

    def test_convert_array_of_float16_vectors_invalid_dtype(self):
        """Test numpy array with invalid dtype raises error for float16"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT16_VECTOR,
            "params": {"dim": 2},
        }
        vectors = [np.array([1.0, 2.0], dtype=np.float32)]
        with pytest.raises(ParamError, match="invalid input for float16 vector"):
            convert_to_array_of_vector(vectors, field_info)

    def test_convert_array_of_int8_vectors_invalid_dtype(self):
        """Test numpy array with invalid dtype raises error for int8"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.INT8_VECTOR,
            "params": {"dim": 2},
        }
        vectors = [np.array([1, 2], dtype=np.int32)]
        with pytest.raises(ParamError, match="invalid input for int8 vector"):
            convert_to_array_of_vector(vectors, field_info)

    def test_convert_array_of_float16_vectors_invalid_type(self):
        """Test non-bytes non-ndarray input raises error for float16"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT16_VECTOR,
            "params": {"dim": 2},
        }
        with pytest.raises(ParamError, match="invalid input type"):
            convert_to_array_of_vector([[1.0, 2.0]], field_info)

    def test_convert_array_of_float_vectors_invalid_dtype(self):
        """Test numpy array with invalid dtype raises error"""
        field_info = {
            "name": "vec_arr_field",
            "element_type": DataType.FLOAT_VECTOR,
            "params": {"dim": 2},
        }
        # int64 dtype should raise error
        vectors = [np.array([1, 2], dtype=np.int64)]
        with pytest.raises(ParamError, match="invalid input for float32 vector"):
            convert_to_array_of_vector(vectors, field_info)


class TestGetArrayHelpers:
    """Test get_array_length and get_array_value_at_index functions"""

    def test_get_array_length_int64(self):
        """Test get_array_length for INT64 array"""

        array_item = schema_types.ScalarField()
        array_item.long_data.data.extend([1, 2, 3, 4, 5])

        result = get_array_length(array_item)
        assert result == 5

    def test_get_array_length_int_types(self):
        """Test get_array_length for INT8/16/32 array"""

        array_item = schema_types.ScalarField()
        array_item.int_data.data.extend([10, 20, 30])

        result = get_array_length(array_item)
        assert result == 3

    def test_get_array_length_float(self):
        """Test get_array_length for FLOAT array"""

        array_item = schema_types.ScalarField()
        array_item.float_data.data.extend([1.0, 2.0])

        result = get_array_length(array_item)
        assert result == 2

    def test_get_array_length_double(self):
        """Test get_array_length for DOUBLE array"""

        array_item = schema_types.ScalarField()
        array_item.double_data.data.extend([1.1, 2.2, 3.3])

        result = get_array_length(array_item)
        assert result == 3

    def test_get_array_length_string(self):
        """Test get_array_length for string array"""

        array_item = schema_types.ScalarField()
        array_item.string_data.data.extend(["a", "b", "c", "d"])

        result = get_array_length(array_item)
        assert result == 4

    def test_get_array_length_bool(self):
        """Test get_array_length for bool array"""

        array_item = schema_types.ScalarField()
        array_item.bool_data.data.extend([True, False])

        result = get_array_length(array_item)
        assert result == 2

    def test_get_array_value_at_index_int64(self):
        """Test get_array_value_at_index for INT64 array"""

        array_item = schema_types.ScalarField()
        array_item.long_data.data.extend([100, 200, 300])

        assert get_array_value_at_index(array_item, 0) == 100
        assert get_array_value_at_index(array_item, 1) == 200
        assert get_array_value_at_index(array_item, 2) == 300

    def test_get_array_value_at_index_float(self):
        """Test get_array_value_at_index for FLOAT array"""

        array_item = schema_types.ScalarField()
        array_item.float_data.data.extend([1.5, 2.5])

        assert get_array_value_at_index(array_item, 0) == pytest.approx(1.5)
        assert get_array_value_at_index(array_item, 1) == pytest.approx(2.5)

    def test_get_array_value_at_index_string(self):
        """Test get_array_value_at_index for string array"""

        array_item = schema_types.ScalarField()
        array_item.string_data.data.extend(["hello", "world"])

        assert get_array_value_at_index(array_item, 0) == "hello"
        assert get_array_value_at_index(array_item, 1) == "world"
