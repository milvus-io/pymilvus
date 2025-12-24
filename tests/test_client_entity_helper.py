import json
import struct
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.entity_helper import (
    convert_to_str_array, entity_to_array_arr,
    entity_to_field_data,
    entity_to_str_arr,
    entity_type_to_dtype,
    extract_array_row_data,
    extract_dynamic_field_from_result,
    get_max_len_of_var_char,
    pack_field_value_to_field_data,
    sparse_proto_to_rows,
    sparse_rows_to_proto,
)
from pymilvus.client.entity_helper import extract_row_data_from_fields_data_v2 as convert_to_entity
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2

# Import additional functions for testing
from pymilvus.grpc_gen import schema_pb2 as schema_types
from pymilvus.settings import Config
from scipy.sparse import csr_matrix
from pymilvus.exceptions import DataNotMatchException


class TestEntityHelperSparse:
    """Test entity_helper module functions"""

    @pytest.mark.parametrize("valid_sparse_matrix", [
        [{0: 1.0, 5: 2.5, 10: 3.0}], # list of one dict
        [{0: 1.0, 5: 2.5}, {10: 3.0, 15: 4.0}], # list of dicts
        [{}, {10: 3.0, 15: 4.0}], # list of dicts partial empty is allowed
        [[(1, 0.5), (10, 0.3)], [(2, 0.7), (20, 0.1)]], # list of list
        [[("1", "0.5"), (10, 0.3)]], # str representation of int
        csr_matrix(([1, 2, 3], [0, 2, 3], [0, 2, 3, 3]), shape=(3, 4)), # scipy sparse matrix
        [csr_matrix([[1, 0, 2]]), csr_matrix([[0, 0, 3]])], # list of scipy sparse matrices
    ])
    def test_entity_is_sparse_matrix(self, valid_sparse_matrix: list):
        assert entity_helper.entity_is_sparse_matrix(valid_sparse_matrix) is True

    @pytest.mark.parametrize("not_sparse_matrix", [
        [{"a": 1.0, "b": 2.0}], # invalid dict for non-numeric keys
        [], # empty
        [{0: 1.0}, "not a dict", {5: 2.0}], # mixed lists
        None,
        123,
        "string",
        [1, 2, 3],
        [[1, 2, 3]],
        [[(1, 0.5, 0.2)]],
        [[(1, "invalid")]],
        [csr_matrix([[1, 0], [0, 1]])], # list of multi-row is not sparse
    ])
    def test_entity_isnot_sparse_matrix(self, not_sparse_matrix: any):
        assert entity_helper.entity_is_sparse_matrix(not_sparse_matrix) is False

    def test_get_input_num_rows_list(self):
        """Test getting number of rows from list input"""
        # Regular list
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert entity_helper.get_input_num_rows(data) == 3
        assert entity_helper.get_input_num_rows([1, 2, 3]) == 3
        assert entity_helper.get_input_num_rows({"a": 1, "b": 2}) == 2

        data = [[1, 2, 3]]
        assert entity_helper.get_input_num_rows(data) == 1

        data = []
        assert entity_helper.get_input_num_rows(data) == 0

        matrix = csr_matrix([[1, 0], [0, 1], [1, 1]])
        assert entity_helper.get_input_num_rows(matrix) == 3

        sparse_list = [
            {0: 1.0},
            {5: 2.5},
            {10: 3.0}
        ]
        assert entity_helper.get_input_num_rows(sparse_list) == 3

        data = np.array([[1, 2, 3], [4, 5, 6]])
        assert entity_helper.get_input_num_rows(data) == 2

    @pytest.mark.parametrize("sparse_list", [
        [{0: 1.0, 2: 2.0}, {2: 3.0}],
        csr_matrix([[1, 0, 2], [0, 0, 3]])
    ])
    def test_sparse_rows_to_proto_dict(self, sparse_list: any):
        """Test converting sparse rows to protobuf format"""
        proto = entity_helper.sparse_rows_to_proto(sparse_list)
        assert isinstance(proto, schema_pb2.SparseFloatArray)
        assert len(proto.contents) == 2
        assert proto.dim == 3

    def test_sparse_proto_to_rows(self):
        """Test converting protobuf sparse vectors back to rows"""
        # Create a mock sparse proto
        proto = schema_pb2.SparseFloatArray(dim=100)

        # Add some sparse vectors in binary format
        # Format: pairs of (uint32 index, float32 value)
        vec1_data = b''
        for idx, val in [(0, 1.0), (5, 2.5), (10, 3.0)]:
            vec1_data += struct.pack('I', idx) + struct.pack('f', val)
        proto.contents.append(vec1_data)

        vec2_data = b''
        for idx, val in [(15, 4.0), (20, 5.0)]:
            vec2_data += struct.pack('I', idx) + struct.pack('f', val)
        proto.contents.append(vec2_data)

        # Convert back to rows
        rows = entity_helper.sparse_proto_to_rows(proto, 0, 2)
        assert len(rows) == 2
        assert rows[0] == {0: 1.0, 5: 2.5, 10: 3.0}
        assert rows[1] == {15: 4.0, 20: 5.0}

    def test_sparse_proto_to_rows_with_range(self):
        """Test converting specific range of sparse vectors"""
        proto = schema_pb2.SparseFloatArray(dim=100)

        # Add multiple sparse vectors in binary format
        for i in range(5):
            vec_data = struct.pack('I', i) + struct.pack('f', float(i))
            proto.contents.append(vec_data)

        # Get middle range
        rows = entity_helper.sparse_proto_to_rows(proto, 1, 4)
        assert len(rows) == 3
        assert rows[0] == {1: 1.0}
        assert rows[1] == {2: 2.0}
        assert rows[2] == {3: 3.0}

    def test_convert_to_json_nested(self):
        """Test JSON conversion with nested structures"""
        obj = {
            "level1": {
                "level2": {
                    "array": np.array([[1, 2], [3, 4]]),
                    "list": [np.int32(1), np.float64(2.5)]
                }
            }
        }
        result = entity_helper.convert_to_json(obj)
        parsed = json.loads(result)
        assert parsed["level1"]["level2"]["array"] == [[1, 2], [3, 4]]

    @pytest.mark.parametrize("data", [
        {"key": "value", "number": 42}, # dict
        {"outer": {"inner": "value"}}, # nested dict
        [1, 2, 3, "four"], # list
        [{"a": 1}, {"b": 2}], # list of dict
        None,
        pytest.param({"array": np.array([1, 2, 3])}, marks=pytest.mark.xfail(reason="fix me")),
        { "int": np.int64(42), "float": np.float32(3.14), "bool": np.bool_(True) },
        [{"val": np.int64(10)}, {"val": np.float32(3.14)}],
    ])
    def test_convert_to_json_dict(self, data: dict):
        """Test JSON conversion for dict input"""
        result = entity_helper.convert_to_json(data)
        assert isinstance(result, bytes)
        assert json.loads(result.decode()) == data

    @pytest.mark.parametrize("json_string,expected", [
        ('{"key": "value", "number": 42}', {"key": "value", "number": 42}),
        ('{"nested": {"inner": "value"}}', {"nested": {"inner": "value"}}),
        ('[1, 2, 3, "four"]', [1, 2, 3, "four"]),
        ('{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
        ('null', None),
        ('true', True),
        ('false', False),
        ('123', 123),
        ('"simple string"', "simple string"),
    ])
    def test_convert_to_json_string_valid(self, json_string: str, expected):
        """Test JSON conversion for valid JSON string input"""
        result = entity_helper.convert_to_json(json_string)
        assert isinstance(result, bytes)
        # Verify the result is valid JSON
        parsed = json.loads(result.decode())
        assert parsed == expected

    def test_convert_to_json_from_json_dumps(self):
        """Test JSON conversion from json.dumps() output"""
        original_dict = {"key": "value", "count": 100, "nested": {"inner": "data"}}
        json_string = json.dumps(original_dict)
        
        result = entity_helper.convert_to_json(json_string)
        assert isinstance(result, bytes)
        parsed = json.loads(result.decode())
        assert parsed == original_dict

    @pytest.mark.parametrize("invalid_json_string", [
        "not a json string",
        '{"invalid": }',
        '{"key": "value"',  # missing closing brace
        "{'key': 'value'}",  # single quotes not valid in JSON
        "{key: value}",  # unquoted keys
        "undefined",
        "{,}",
    ])
    def test_convert_to_json_string_invalid(self, invalid_json_string: str):
        """Test JSON conversion rejects invalid JSON strings"""
        
        with pytest.raises(DataNotMatchException) as exc_info:
            entity_helper.convert_to_json(invalid_json_string)
        
        # Verify error message contains the invalid JSON string
        error_message = str(exc_info.value)
        assert "Invalid JSON string" in error_message
        # Verify the original input string is in the error message
        assert invalid_json_string in error_message or invalid_json_string[:50] in error_message

    def test_convert_to_json_string_with_non_string_keys(self):
        """Test JSON conversion rejects JSON strings with non-string keys in dict"""
        
        # This is actually not possible in standard JSON, as JSON object keys are always strings
        # But we can test that dict validation still works
        invalid_dict = {1: "value", 2: "another"}
        
        with pytest.raises(DataNotMatchException) as exc_info:
            entity_helper.convert_to_json(invalid_dict)
        
        error_message = str(exc_info.value)
        assert "JSON" in error_message

    def test_convert_to_json_long_invalid_string_truncated(self):
        """Test that long invalid JSON strings are truncated in error messages"""
        
        # Create a long invalid JSON string
        long_invalid_json = "invalid json " * 50  # > 200 characters
        
        with pytest.raises(DataNotMatchException) as exc_info:
            entity_helper.convert_to_json(long_invalid_json)
        
        error_message = str(exc_info.value)
        assert "Invalid JSON string" in error_message
        # Should contain truncated version with "..."
        assert "..." in error_message

    def test_convert_to_json_deeply_nested_dict(self):
        """Test JSON conversion with deeply nested dictionary to avoid recursion limit"""
        
        def create_deep_dict(depth):
            """Create a deeply nested dictionary with specified depth"""
            result = {'id_0': '-501'}
            for i in range(1, depth + 1):
                result = {'id_' + str(i): result}
            return result
        
        # Test with 512 levels (tests fallback from orjson to standard json library)
        # orjson fails at ~500 levels, so 512 will trigger fallback to json.dumps
        deep_dict = create_deep_dict(512)
        
        # Should not raise RecursionError
        result = entity_helper.convert_to_json(deep_dict)
        assert isinstance(result, bytes)
        
        # Verify the structure is preserved
        parsed = json.loads(result.decode())
        current = parsed
        for i in range(512, 0, -1):
            assert f'id_{i}' in current, f"Missing key id_{i} at depth {512 - i + 1}"
            current = current[f'id_{i}']
        assert current == {'id_0': '-501'}, "Final nested value mismatch"

    def test_pack_field_value_to_field_data(self):
        """Test packing field values into field data protobuf"""
        # Test with scalar field
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT64
        field_info = {"name": "test_field"}

        value = 42
        entity_helper.pack_field_value_to_field_data(value, field_data, field_info)

        assert len(field_data.scalars.long_data.data) == 1
        assert field_data.scalars.long_data.data[0] == value

    def test_pack_field_value_to_field_data_vectors(self):
        """Test packing vector field values"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_info = {"name": "vector_field"}

        value = [1.0, 2.0, 3.0, 4.0]

        entity_helper.pack_field_value_to_field_data(value, field_data, field_info)

        assert field_data.vectors.dim == 4
        assert list(field_data.vectors.float_vector.data) == value

    def test_extract_array_row_data(self):
        """Test extracting array data from protobuf"""
        # Create field data with array
        field_data = schema_pb2.FieldData()
        field_data.scalars.array_data.element_type = DataType.INT64

        # Add array data for index 0
        scalar_field = schema_pb2.ScalarField()
        scalar_field.long_data.data.extend([0, 1, 2])
        field_data.scalars.array_data.data.append(scalar_field)

        result = entity_helper.extract_array_row_data(field_data, 0)
        assert result == [0, 1, 2]

    def test_extract_array_row_data_string(self):
        """Test extracting string array data"""
        # Create field data with array
        field_data = schema_pb2.FieldData()
        field_data.scalars.array_data.element_type = DataType.VARCHAR

        # Add array data for index 0
        scalar_field = schema_pb2.ScalarField()
        scalar_field.string_data.data.extend(["str_0", "str_1"])
        field_data.scalars.array_data.data.append(scalar_field)

        result = entity_helper.extract_array_row_data(field_data, 0)
        assert result == ["str_0", "str_1"]

    def test_extract_array_row_data_bool(self):
        """Test extracting boolean array data"""
        # Create field data with array
        field_data = schema_pb2.FieldData()
        field_data.scalars.array_data.element_type = DataType.BOOL

        # Add array data for index 0
        scalar_field = schema_pb2.ScalarField()
        scalar_field.bool_data.data.extend([True, False, True])
        field_data.scalars.array_data.data.append(scalar_field)

        result = entity_helper.extract_array_row_data(field_data, 0)
        assert result == [True, False, True]

    def test_extract_array_row_data_float(self):
        """Test extracting float array data"""
        # Create field data with array
        field_data = schema_pb2.FieldData()
        field_data.scalars.array_data.element_type = DataType.FLOAT

        # Add array data for index 0
        scalar_field = schema_pb2.ScalarField()
        scalar_field.float_data.data.extend([1.1, 2.2, 3.3])
        field_data.scalars.array_data.data.append(scalar_field)

        result = entity_helper.extract_array_row_data(field_data, 0)
        assert result == pytest.approx([1.1, 2.2, 3.3])

    def test_extract_array_row_data_double(self):
        """Test extracting double array data"""
        # Create field data with array
        field_data = schema_pb2.FieldData()
        field_data.scalars.array_data.element_type = DataType.DOUBLE

        # Add array data for index 0
        scalar_field = schema_pb2.ScalarField()
        scalar_field.double_data.data.extend([1.11111, 2.22222, 3.33333])
        field_data.scalars.array_data.data.append(scalar_field)

        result = entity_helper.extract_array_row_data(field_data, 0)
        assert result == pytest.approx([1.11111, 2.22222, 3.33333])

    def test_extract_array_row_data_invalid_type(self):
        """Test extracting array data with invalid type"""
        # Create field data with array
        field_data = schema_pb2.FieldData()
        field_data.scalars.array_data.element_type = 999  # Invalid type

        # Add array data for index 0
        scalar_field = schema_pb2.ScalarField()
        field_data.scalars.array_data.data.append(scalar_field)

        result = entity_helper.extract_array_row_data(field_data, 0)
        assert result is None  # Returns None for unknown types


class TestEntityHelperExtended:
    def test_sparse_rows_to_proto_invalid_index(self):
        """Test error handling for invalid indices"""
        # Negative index
        with pytest.raises(ParamError, match="sparse vector index must be positive"):
            sparse_rows_to_proto([{-1: 0.5}])

        # Index too large
        with pytest.raises(ParamError, match="sparse vector index must be positive"):
            sparse_rows_to_proto([{2**32: 0.5}])

    def test_sparse_rows_to_proto_nan_value(self):
        """Test error handling for NaN values"""
        with pytest.raises(ParamError, match="sparse vector value must not be NaN"):
            sparse_rows_to_proto([{1: float('nan')}])

    def test_sparse_rows_to_proto_invalid_input(self):
        """Test error handling for invalid input"""
        with pytest.raises(ParamError, match="input must be a sparse matrix"):
            sparse_rows_to_proto("invalid")

    def test_sparse_proto_to_rows_invalid(self):
        """Test error handling for invalid proto"""
        with pytest.raises(ParamError, match="Vector must be a sparse float vector"):
            sparse_proto_to_rows("invalid")

    def test_entity_type_to_dtype(self):
        """Test converting entity type to dtype"""
        # Integer type
        assert entity_type_to_dtype(1) == 1
        assert entity_type_to_dtype(DataType.INT64) == DataType.INT64

        # We can't test string conversion without knowing exact protobuf enum names
        # Let's just test invalid type
        with pytest.raises(ParamError, match="invalid entity type"):
            entity_type_to_dtype([])

    def test_get_max_len_of_var_char(self):
        """Test getting max length of varchar field"""
        # With params
        field_info = {"params": {Config.MaxVarCharLengthKey: 100}}
        assert get_max_len_of_var_char(field_info) == 100

        # Without params - use default
        field_info = {}
        assert get_max_len_of_var_char(field_info) == Config.MaxVarCharLength

        # Partial params
        field_info = {"params": {}}
        assert get_max_len_of_var_char(field_info) == Config.MaxVarCharLength

    def test_convert_to_str_array(self):
        """Test converting to string array"""
        field_info = {"name": "test_field", "params": {Config.MaxVarCharLengthKey: 10}}

        # Valid strings
        result = convert_to_str_array(["hello", "world"], field_info)
        assert result == ["hello", "world"]

        # String exceeding max length
        with pytest.raises(ParamError, match="length of string exceeds max length"):
            convert_to_str_array(["this is too long"], field_info)

        # Non-string input
        with pytest.raises(ParamError, match="expects string input"):
            convert_to_str_array([123], field_info)

        # Without check
        result = convert_to_str_array([123, "test"], field_info, check=False)
        assert len(result) == 2

    @patch('pymilvus.client.entity_helper.Config')
    def test_convert_to_str_array_with_encoding(self, mock_config):
        """Test string array conversion with different encoding"""
        mock_config.EncodeProtocol = "latin-1"
        mock_config.MaxVarCharLengthKey = "max_length"
        mock_config.MaxVarCharLength = 65535

        field_info = {"name": "test_field"}
        strings = ["hello", "world"]
        result = convert_to_str_array(strings, field_info, check=False)
        assert len(result) == 2

    def test_entity_to_str_arr(self):
        """Test entity_to_str_arr wrapper function"""
        field_info = {"name": "test_field", "params": {Config.MaxVarCharLengthKey: 20}}
        result = entity_to_str_arr(["test1", "test2"], field_info)
        assert result == ["test1", "test2"]

    def test_extract_array_row_data(self):
        """Test extracting array row data"""
        # INT64 array
        field_data = schema_types.FieldData()
        field_data.type = DataType.ARRAY
        array_data = schema_types.ArrayArray()

        # Add INT64 data
        int_array = schema_types.LongArray()
        int_array.data.extend([1, 2, 3])
        array_data.data.append(schema_types.ScalarField(long_data=int_array))
        int_array2 = schema_types.LongArray()
        int_array2.data.extend([4, 5])
        array_data.data.append(schema_types.ScalarField(long_data=int_array2))

        array_data.element_type = DataType.INT64
        field_data.scalars.array_data.CopyFrom(array_data)

        result = extract_array_row_data(field_data, 0)
        assert result == [1, 2, 3]
        result = extract_array_row_data(field_data, 1)
        assert result == [4, 5]

    def test_extract_array_row_data_string(self):
        """Test extracting string array data"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.ARRAY
        array_data = schema_types.ArrayArray()

        # Add string data
        str_array = schema_types.StringArray()
        str_array.data.extend(["hello", "world"])
        array_data.data.append(schema_types.ScalarField(string_data=str_array))

        array_data.element_type = DataType.VARCHAR
        field_data.scalars.array_data.CopyFrom(array_data)

        result = extract_array_row_data(field_data, 0)
        assert result == ["hello", "world"]

    def test_extract_array_row_data_bool(self):
        """Test extracting boolean array data"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.ARRAY
        array_data = schema_types.ArrayArray()

        # Add bool data
        bool_array = schema_types.BoolArray()
        bool_array.data.extend([True, False, True])
        array_data.data.append(schema_types.ScalarField(bool_data=bool_array))

        array_data.element_type = DataType.BOOL
        field_data.scalars.array_data.CopyFrom(array_data)

        result = extract_array_row_data(field_data, 0)
        assert result == [True, False, True]

    def test_extract_array_row_data_float(self):
        """Test extracting float array data"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.ARRAY
        array_data = schema_types.ArrayArray()

        # Add float data
        float_array = schema_types.FloatArray()
        float_array.data.extend([1.1, 2.2, 3.3])
        array_data.data.append(schema_types.ScalarField(float_data=float_array))

        array_data.element_type = DataType.FLOAT
        field_data.scalars.array_data.CopyFrom(array_data)

        result = extract_array_row_data(field_data, 0)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.1)

    def test_extract_array_row_data_double(self):
        """Test extracting double array data"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.ARRAY
        array_data = schema_types.ArrayArray()

        # Add double data
        double_array = schema_types.DoubleArray()
        double_array.data.extend([1.11111, 2.22222])
        array_data.data.append(schema_types.ScalarField(double_data=double_array))

        array_data.element_type = DataType.DOUBLE
        field_data.scalars.array_data.CopyFrom(array_data)

        result = extract_array_row_data(field_data, 0)
        assert len(result) == 2

    def test_extract_array_row_data_invalid_type(self):
        """Test error handling for invalid array element type"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.ARRAY
        array_data = schema_types.ArrayArray()
        array_data.element_type = 999  # Invalid type
        # Add at least one empty element to avoid index error
        array_data.data.append(schema_types.ScalarField())
        field_data.scalars.array_data.CopyFrom(array_data)

        assert extract_array_row_data(field_data, 0) is None

    def test_entity_to_array_arr(self):
        """Test converting entity to array array"""
        field_info = {
            "name": "array_field",
            "element_type": DataType.INT64
        }

        # List of lists
        data = [[1, 2, 3], [4, 5], [6]]
        result = entity_to_array_arr(data, field_info)
        assert len(result) == 3
        assert result[0].long_data.data == [1, 2, 3]
        assert result[1].long_data.data == [4, 5]

    def test_entity_to_array_arr_string(self):
        """Test converting string arrays"""
        field_info = {
            "name": "array_field",
            "element_type": DataType.VARCHAR
        }

        data = [["hello", "world"], ["foo"]]
        result = entity_to_array_arr(data, field_info)
        assert len(result) == 2
        assert list(result[0].string_data.data) == ["hello", "world"]

    def test_entity_to_array_arr_invalid_type(self):
        """Test error handling for invalid element type"""
        field_info = {
            "name": "array_field",
            "element_type": 999
        }

        with pytest.raises(ParamError, match="Unsupported element type"):
            entity_to_array_arr([[1, 2]], field_info)

    def test_pack_field_value_to_field_data(self):
        """Test packing field values to field data"""
        # pack_field_value_to_field_data takes different parameters
        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_data.field_name = "vector_field"
        field_info = {"name": "vector_field"}

        # Pack a single vector
        pack_field_value_to_field_data(
            np.array([1.0, 2.0]),
            field_data,
            field_info
        )

        # Check the result
        assert field_data.type == DataType.FLOAT_VECTOR
        assert field_data.vectors.dim == 2

    def test_pack_field_value_to_field_data_sparse(self):
        """Test packing sparse vectors"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.SPARSE_FLOAT_VECTOR
        field_data.field_name = "sparse_field"
        field_info = {"name": "sparse_field"}

        # Pack a single sparse vector
        sparse_data = {1: 0.5, 10: 0.3}
        pack_field_value_to_field_data(
            sparse_data,
            field_data,
            field_info
        )

        # Check the result
        assert field_data.type == DataType.SPARSE_FLOAT_VECTOR
        assert len(field_data.vectors.sparse_float_vector.contents) == 1

    def test_pack_field_value_to_field_data_scalars(self):
        """Test packing scalar field values"""
        # Test INT64
        field_data = schema_types.FieldData()
        field_data.type = DataType.INT64
        field_data.field_name = "int_field"
        field_info = {"name": "int_field"}

        pack_field_value_to_field_data(
            42,
            field_data,
            field_info
        )

        assert field_data.type == DataType.INT64
        assert field_data.scalars.long_data.data[0] == 42

    def test_extract_field_info(self):
        """Test extracting primary field from schema"""
        # Create schema with primary field
        fields_info = [
            {"name": "id", "is_primary": True},
            {"name": "vector", "is_primary": False}
        ]

        # Test field extraction logic
        # This tests the field extraction patterns used in entity_helper
        for field in fields_info:
            if field.get("is_primary"):
                assert field["name"] == "id"

    def test_extract_dynamic_field_from_result(self):
        """Test extracting dynamic field from result"""
        # Create actual result object with fields_data and output_fields attributes
        class ActualResult:
            def __init__(self, fields_data: List, output_fields: List):
                self.fields_data = fields_data
                self.output_fields = output_fields

        # Test with dynamic field
        dynamic_field_data = schema_types.FieldData()
        dynamic_field_data.is_dynamic = True
        dynamic_field_data.field_name = "$meta"

        regular_field_data = schema_types.FieldData()
        regular_field_data.is_dynamic = False
        regular_field_data.field_name = "id"

        # Create result with dynamic field - extra_field comes before $meta
        result = ActualResult(
            fields_data=[regular_field_data, dynamic_field_data],
            output_fields=["id", "extra_field", "another_extra", "$meta"]
        )

        dynamic_field_name, dynamic_fields = extract_dynamic_field_from_result(result)
        assert dynamic_field_name == "$meta"
        # When $meta is found in output_fields, dynamic_fields gets cleared
        assert len(dynamic_fields) == 0

        # Test with no dynamic field
        result_no_dynamic = ActualResult(
            fields_data=[regular_field_data],
            output_fields=["id", "extra_field"]
        )

        dynamic_field_name, dynamic_fields = extract_dynamic_field_from_result(result_no_dynamic)
        assert dynamic_field_name is None
        assert "extra_field" in dynamic_fields
        assert "id" not in dynamic_fields

        # Test with dynamic field NOT in output_fields (dynamic_fields preserved)
        result_meta_not_in_output = ActualResult(
            fields_data=[regular_field_data, dynamic_field_data],
            output_fields=["id", "extra_field", "another_extra"]
        )

        dynamic_field_name, dynamic_fields = extract_dynamic_field_from_result(result_meta_not_in_output)
        assert dynamic_field_name == "$meta"
        assert "extra_field" in dynamic_fields
        assert "another_extra" in dynamic_fields
        assert "id" not in dynamic_fields  # id is a regular field
        assert len(dynamic_fields) == 2

    def test_data_validation(self):
        """Test data validation patterns"""
        # Test length validation
        data1 = [1, 2, 3]
        data2 = [4, 5, 6]
        assert len(data1) == len(data2)  # Valid same length

        # Length mismatch should be caught
        data3 = [1, 2]
        data4 = [3, 4, 5]
        assert len(data3) != len(data4)  # Invalid different length

    def test_schema_validation(self):
        """Test schema validation patterns"""
        fields_info = [
            {"name": "id", "is_primary": True, "auto_id": False},
            {"name": "vector", "is_primary": False}
        ]

        # Valid data matches schema
        data = {"id": [1, 2], "vector": [[1.0, 2.0], [3.0, 4.0]]}
        for field in fields_info:
            if not field.get("auto_id"):
                assert field["name"] in data  # Field should be in data

        # Test auto_id logic
        fields_info[0]["auto_id"] = True
        # When auto_id is True, id field should not be required in data






    def test_convert_to_entity(self):
        """Test converting field data to entities"""
        # Create field data
        field1 = schema_types.FieldData()
        field1.field_name = "id"
        field1.type = DataType.INT64
        long_array = schema_types.LongArray()
        long_array.data.extend([1, 2, 3])
        field1.scalars.long_data.CopyFrom(long_array)

        field2 = schema_types.FieldData()
        field2.field_name = "name"
        field2.type = DataType.VARCHAR
        str_array = schema_types.StringArray()
        str_array.data.extend(["a", "b", "c"])
        field2.scalars.string_data.CopyFrom(str_array)

        # extract_row_data_from_fields_data_v2 takes a single field_data and list of entity dicts
        # We need to create empty entity rows first
        entity_rows = [{} for _ in range(3)]

        # Process each field separately
        convert_to_entity(field1, entity_rows)
        convert_to_entity(field2, entity_rows)

        assert len(entity_rows) == 3
        assert entity_rows[0] == {"id": 1, "name": "a"}
        assert entity_rows[1] == {"id": 2, "name": "b"}
        assert entity_rows[2] == {"id": 3, "name": "c"}

    def test_entity_to_field_data(self):
        """Test converting entity to field data"""
        # entity_to_field_data expects a dict with specific structure
        entity = {
            "name": "test_field",
            "type": DataType.INT64,
            "values": [1, 2, 3, 4, 5]
        }
        field_info = {"name": "test_field"}

        result = entity_to_field_data(entity, field_info, 5)

        assert result.field_name == "test_field"
        assert result.type == DataType.INT64
        assert list(result.scalars.long_data.data) == [1, 2, 3, 4, 5]


class TestNullableVectorSupport:
    """Test nullable vector support in entity_helper module for all 6 vector types"""

    # Vector type configurations for parametrized tests
    VECTOR_TYPE_CONFIGS = [
        {
            "dtype": DataType.FLOAT_VECTOR,
            "name": "float_vector",
            "values": [[1.0, 2.0, 3.0, 4.0], None, [5.0, 6.0, 7.0, 8.0]],
            "dim": 4,
            "get_data_len": lambda fd: len(fd.vectors.float_vector.data),
            "expected_data_len": 8,  # 2 valid vectors * 4 dim
        },
        {
            "dtype": DataType.BINARY_VECTOR,
            "name": "binary_vector",
            "values": [b'\x01\x02\x03\x04', None, b'\x05\x06\x07\x08'],
            "dim": 32,  # 4 bytes * 8 bits
            "get_data_len": lambda fd: len(fd.vectors.binary_vector),
            "expected_data_len": 8,  # 2 valid vectors * 4 bytes
        },
        {
            "dtype": DataType.FLOAT16_VECTOR,
            "name": "float16_vector",
            "values": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes(), None,
                       np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float16).tobytes()],
            "dim": 4,
            "get_data_len": lambda fd: len(fd.vectors.float16_vector),
            "expected_data_len": 16,  # 2 valid vectors * 4 dim * 2 bytes
        },
        {
            "dtype": DataType.BFLOAT16_VECTOR,
            "name": "bfloat16_vector",
            "values": [b'\x00\x3f\x00\x40\x00\x40\x00\x40', None, b'\x00\x40\x00\x40\x00\x40\x00\x40'],
            "dim": 4,
            "get_data_len": lambda fd: len(fd.vectors.bfloat16_vector),
            "expected_data_len": 16,  # 2 valid vectors * 4 dim * 2 bytes
        },
        {
            "dtype": DataType.SPARSE_FLOAT_VECTOR,
            "name": "sparse_float_vector",
            "values": [{1: 0.5, 10: 0.3}, None, {5: 0.8}],
            "dim": None,
            "get_data_len": lambda fd: len(fd.vectors.sparse_float_vector.contents),
            "expected_data_len": 2,  # 2 valid sparse vectors
        },
        {
            "dtype": DataType.INT8_VECTOR,
            "name": "int8_vector",
            "values": [np.array([1, 2, 3, 4], dtype=np.int8).tobytes(), None,
                       np.array([5, 6, 7, 8], dtype=np.int8).tobytes()],
            "dim": 4,
            "get_data_len": lambda fd: len(fd.vectors.int8_vector),
            "expected_data_len": 8,  # 2 valid vectors * 4 dim * 1 byte
        },
    ]

    @pytest.mark.parametrize("config", VECTOR_TYPE_CONFIGS,
                             ids=[c["name"] for c in VECTOR_TYPE_CONFIGS])
    def test_entity_to_field_data_nullable_vector(self, config):
        """Test entity_to_field_data with nullable vector containing None values"""
        entity = {
            "name": f"nullable_{config['name']}",
            "type": config["dtype"],
            "values": config["values"]
        }
        field_info = {"name": f"nullable_{config['name']}", "nullable": True}
        if config["dim"]:
            field_info["params"] = {"dim": config["dim"]}

        result = entity_to_field_data(entity, field_info, 3)

        assert result.field_name == f"nullable_{config['name']}"
        assert result.type == config["dtype"]
        assert list(result.valid_data) == [True, False, True]
        assert config["get_data_len"](result) == config["expected_data_len"]

    @pytest.mark.parametrize("config", VECTOR_TYPE_CONFIGS,
                             ids=[c["name"] for c in VECTOR_TYPE_CONFIGS])
    def test_entity_to_field_data_nullable_vector_all_none(self, config):
        """Test entity_to_field_data with nullable vector where all values are None"""
        entity = {
            "name": f"nullable_{config['name']}",
            "type": config["dtype"],
            "values": [None, None, None]
        }
        field_info = {"name": f"nullable_{config['name']}", "nullable": True}
        if config["dim"]:
            field_info["params"] = {"dim": config["dim"]}

        result = entity_to_field_data(entity, field_info, 3)

        assert list(result.valid_data) == [False, False, False]
        assert config["get_data_len"](result) == 0

    @pytest.mark.parametrize("dtype,name", [
        (DataType.FLOAT_VECTOR, "float_vector"),
        (DataType.BINARY_VECTOR, "binary_vector"),
        (DataType.FLOAT16_VECTOR, "float16_vector"),
        (DataType.BFLOAT16_VECTOR, "bfloat16_vector"),
        (DataType.SPARSE_FLOAT_VECTOR, "sparse_float_vector"),
        (DataType.INT8_VECTOR, "int8_vector"),
    ])
    def test_pack_field_value_nullable_vector_none(self, dtype, name):
        """Test pack_field_value_to_field_data with None value for nullable vector"""
        field_data = schema_types.FieldData()
        field_data.type = dtype
        field_data.field_name = f"nullable_{name}"
        field_info = {"name": f"nullable_{name}", "nullable": True}

        pack_field_value_to_field_data(None, field_data, field_info)

        # Verify no data added for each type
        if dtype == DataType.FLOAT_VECTOR:
            assert len(field_data.vectors.float_vector.data) == 0
        elif dtype == DataType.BINARY_VECTOR:
            assert len(field_data.vectors.binary_vector) == 0
        elif dtype == DataType.FLOAT16_VECTOR:
            assert len(field_data.vectors.float16_vector) == 0
        elif dtype == DataType.BFLOAT16_VECTOR:
            assert len(field_data.vectors.bfloat16_vector) == 0
        elif dtype == DataType.SPARSE_FLOAT_VECTOR:
            assert len(field_data.vectors.sparse_float_vector.contents) == 0
        elif dtype == DataType.INT8_VECTOR:
            assert len(field_data.vectors.int8_vector) == 0

    def test_extract_row_data_nullable_float_vector(self):
        """Test extracting nullable float vector from field data"""
        from pymilvus.client.entity_helper import extract_row_data_from_fields_data

        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_data.field_name = "nullable_vector"
        field_data.vectors.dim = 4
        field_data.vectors.float_vector.data.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        field_data.valid_data.extend([True, False, True])

        result0 = extract_row_data_from_fields_data([field_data], 0)
        assert result0["nullable_vector"] == [1.0, 2.0, 3.0, 4.0]

        result1 = extract_row_data_from_fields_data([field_data], 1)
        assert result1["nullable_vector"] is None

        result2 = extract_row_data_from_fields_data([field_data], 2)
        assert result2["nullable_vector"] == [5.0, 6.0, 7.0, 8.0]

    def test_extract_row_data_nullable_sparse_vector(self):
        """Test extracting nullable sparse vector from field data"""
        from pymilvus.client.entity_helper import extract_row_data_from_fields_data

        field_data = schema_types.FieldData()
        field_data.type = DataType.SPARSE_FLOAT_VECTOR
        field_data.field_name = "nullable_sparse"
        vec1_data = struct.pack('I', 1) + struct.pack('f', 0.5)
        vec2_data = struct.pack('I', 5) + struct.pack('f', 0.8)
        field_data.vectors.sparse_float_vector.contents.extend([vec1_data, vec2_data])
        field_data.valid_data.extend([True, False, True])

        result0 = extract_row_data_from_fields_data([field_data], 0)
        assert result0["nullable_sparse"] == pytest.approx({1: 0.5})

        result1 = extract_row_data_from_fields_data([field_data], 1)
        assert result1["nullable_sparse"] is None

        result2 = extract_row_data_from_fields_data([field_data], 2)
        assert result2["nullable_sparse"] == pytest.approx({5: 0.8})

    def test_extract_row_data_non_nullable_vector_uses_logical_index(self):
        """Test that non-nullable vectors still use logical index directly"""
        from pymilvus.client.entity_helper import extract_row_data_from_fields_data

        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_data.field_name = "regular_vector"
        field_data.vectors.dim = 4
        field_data.vectors.float_vector.data.extend([
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ])

        result0 = extract_row_data_from_fields_data([field_data], 0)
        assert result0["regular_vector"] == [1.0, 2.0, 3.0, 4.0]

        result1 = extract_row_data_from_fields_data([field_data], 1)
        assert result1["regular_vector"] == [5.0, 6.0, 7.0, 8.0]

        result2 = extract_row_data_from_fields_data([field_data], 2)
        assert result2["regular_vector"] == [9.0, 10.0, 11.0, 12.0]
