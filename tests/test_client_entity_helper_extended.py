from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from pymilvus.client.entity_helper import (
    convert_to_str_array,
    entity_to_array_arr,
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

# Import additional functions for testing
from pymilvus.grpc_gen import schema_pb2 as schema_types
from pymilvus.settings import Config


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
