import numpy as np
import pytest
from pymilvus.client.entity_helper import (
    convert_to_array,
    convert_to_array_of_vector,
    extract_array_rows,
    get_array_length,
    get_array_value_at_index,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.grpc_gen import schema_pb2 as schema_types


class TestExtractArrayRowDataFunctions:
    """Decode scalar arrays without losing row positions or null values."""

    @pytest.mark.parametrize(
        "element_type,attr,arrays,valid_data,expected",
        [
            pytest.param(
                DataType.INT8,
                "int_data",
                [[1, 2, 3], [4, 5]],
                [],
                [[1, 2, 3], [4, 5]],
                id="int8",
            ),
            pytest.param(
                DataType.INT16,
                "int_data",
                [[100, 200]],
                [],
                [[100, 200]],
                id="int16",
            ),
            pytest.param(
                DataType.INT32,
                "int_data",
                [[1000, 2000, 3000]],
                [],
                [[1000, 2000, 3000]],
                id="int32",
            ),
            pytest.param(
                DataType.INT64,
                "long_data",
                [[1, 2, 3], [], [], [7, 8, 9]],
                [True, False, True, True],
                [[1, 2, 3], None, [], [7, 8, 9]],
                id="int64-null-empty-and-values",
            ),
            pytest.param(
                DataType.BOOL,
                "bool_data",
                [[True, False], [False, True, False]],
                [True, True],
                [[True, False], [False, True, False]],
                id="bool-all-valid",
            ),
            pytest.param(
                DataType.VARCHAR,
                "string_data",
                [["hello", "world"], []],
                [True, False],
                [["hello", "world"], None],
                id="varchar-null",
            ),
            pytest.param(
                DataType.FLOAT,
                "float_data",
                [[1.5, 2.5, 3.5]],
                [True],
                [pytest.approx([1.5, 2.5, 3.5])],
                id="float",
            ),
            pytest.param(
                DataType.DOUBLE,
                "double_data",
                [[1.111, 2.222]],
                [True],
                [pytest.approx([1.111, 2.222])],
                id="double",
            ),
            pytest.param(
                DataType.TEXT,
                "string_data",
                [["alpha", "beta"]],
                [],
                [["alpha", "beta"]],
                id="text",
            ),
        ],
    )
    def test_extract_array_rows(self, element_type, attr, arrays, valid_data, expected):
        field_data = schema_types.FieldData(
            type=DataType.ARRAY, field_name="arr_field", valid_data=valid_data
        )
        field_data.scalars.array_data.element_type = element_type
        for values in arrays:
            scalar = field_data.scalars.array_data.data.add()
            getattr(scalar, attr).data.extend(values)
        entity_rows = [{"id": i} for i in range(len(arrays))]

        extract_array_rows(field_data, entity_rows, len(arrays), has_valid=bool(valid_data))

        assert entity_rows == [{"id": i, "arr_field": value} for i, value in enumerate(expected)]

    @pytest.mark.parametrize(
        "element_type,values",
        [
            pytest.param(999, [], id="unknown"),
            pytest.param(DataType.TIMESTAMPTZ, ["2026-05-27T00:00:00Z"], id="timestamptz"),
        ],
    )
    def test_extract_array_rows_unsupported_element(self, element_type, values):
        field_data = schema_types.FieldData(type=DataType.ARRAY, field_name="arr_field")
        field_data.scalars.array_data.element_type = element_type
        field_data.scalars.array_data.data.add().string_data.data.extend(values)

        with pytest.raises(MilvusException, match=f"Unsupported data type: {element_type}"):
            extract_array_rows(field_data, [{}], 1, has_valid=False)


class TestArrayConversion:
    @pytest.mark.parametrize(
        "element_type,values,attr",
        [
            pytest.param(DataType.BOOL, [True, False], "bool_data", id="bool"),
            pytest.param(DataType.INT8, [-128, 0, 127], "int_data", id="int8"),
            pytest.param(DataType.INT16, [-32768, 0, 32767], "int_data", id="int16"),
            pytest.param(DataType.INT32, [-2147483648, 0, 2147483647], "int_data", id="int32"),
            pytest.param(DataType.INT64, [-(2**63), 0, 2**63 - 1], "long_data", id="int64"),
            pytest.param(DataType.FLOAT, [1.5, 2.5], "float_data", id="float"),
            pytest.param(DataType.DOUBLE, [1.11, 2.22], "double_data", id="double"),
            pytest.param(DataType.STRING, ["a", "b"], "string_data", id="string"),
            pytest.param(DataType.VARCHAR, ["varchar-a", "varchar-b"], "string_data", id="varchar"),
            pytest.param(DataType.TEXT, ["text-a", "text-b"], "string_data", id="text"),
        ],
    )
    def test_convert_to_array(self, element_type, values, attr):
        result = convert_to_array(values, {"name": "arr_field", "element_type": element_type})

        # The expected wire field is independent of the production type registry.
        assert result.WhichOneof("data") == attr
        assert list(getattr(result, attr).data) == values

    def test_convert_to_array_numpy_input(self):
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
        assert result.int8_vector == b"\x01\xfe\x03\x04\xfb\x06"

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
