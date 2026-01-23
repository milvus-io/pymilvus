import numpy as np
import pytest
from pymilvus.client.entity_helper import extract_struct_array_from_column_data
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2 as schema_types


class TestExtractStructArray:
    """Test extract_struct_array_from_column_data function"""

    def test_extract_basic_struct_array(self):
        """Test extracting struct array with scalar fields"""

        # Create struct array field structure
        struct_field = schema_types.FieldData()
        struct_field.type = DataType._ARRAY_OF_STRUCT

        # Add sub-field: name (string)
        name_field = schema_types.FieldData()
        name_field.field_name = "name"
        name_field.type = DataType.ARRAY

        # Helper to create array data for a row
        def create_string_array(values):
            arr = schema_types.ScalarField()
            arr.string_data.data.extend(values)
            return arr

        name_field.scalars.array_data.data.append(create_string_array(["Alice", "Bob"]))
        struct_field.struct_arrays.fields.append(name_field)

        # Add sub-field: age (int)
        age_field = schema_types.FieldData()
        age_field.field_name = "age"
        age_field.type = DataType.ARRAY

        def create_int_array(values):
            arr = schema_types.ScalarField()
            arr.int_data.data.extend(values)
            return arr

        age_field.scalars.array_data.data.append(create_int_array([30, 25]))
        struct_field.struct_arrays.fields.append(age_field)

        # Extract row 0
        result = extract_struct_array_from_column_data(struct_field.struct_arrays, 0)

        assert len(result) == 2
        assert result[0] == {"name": "Alice", "age": 30}
        assert result[1] == {"name": "Bob", "age": 25}

    def test_extract_struct_array_invalid_input(self):
        """Test extracting with invalid input"""

        assert extract_struct_array_from_column_data(None, 0) == []

        empty_field = schema_types.FieldData()
        assert extract_struct_array_from_column_data(empty_field, 0) == []

    def test_extract_struct_array_with_vectors(self):
        """Test extracting struct array containing vectors"""

        struct_field = schema_types.FieldData()
        struct_field.type = DataType._ARRAY_OF_STRUCT

        # Add sub-field: vector (FLOAT_VECTOR)
        vec_field = schema_types.FieldData()
        vec_field.field_name = "vector"
        vec_field.type = DataType._ARRAY_OF_VECTOR

        # Create vector array data for row 0
        # 2 vectors of dim 2: [1.0, 2.0] and [3.0, 4.0]
        vec_data = schema_types.VectorField()
        vec_data.dim = 2
        vec_data.float_vector.data.extend([1.0, 2.0, 3.0, 4.0])

        vec_field.vectors.vector_array.element_type = DataType.FLOAT_VECTOR
        vec_field.vectors.vector_array.data.append(vec_data)
        struct_field.struct_arrays.fields.append(vec_field)

        result = extract_struct_array_from_column_data(struct_field.struct_arrays, 0)

        assert len(result) == 2
        assert result[0]["vector"] == [1.0, 2.0]
        assert result[1]["vector"] == [3.0, 4.0]

    def test_extract_struct_array_all_vector_types(self):
        """Test extracting struct array with all vector types"""

        # FLOAT16
        struct_field = schema_types.FieldData()
        f16_field = schema_types.FieldData()
        f16_field.field_name = "f16"
        f16_field.type = DataType._ARRAY_OF_VECTOR

        vec_data = schema_types.VectorField()
        vec_data.dim = 2
        # 1.0 -> 0x3c00, 2.0 -> 0x4000 (roughly, float16 logic)
        vec_data.float16_vector = np.array([1.0, 2.0], dtype=np.float16).tobytes()

        f16_field.vectors.vector_array.element_type = DataType.FLOAT16_VECTOR
        f16_field.vectors.vector_array.data.append(vec_data)
        struct_field.struct_arrays.fields.append(f16_field)

        result = extract_struct_array_from_column_data(struct_field.struct_arrays, 0)
        assert len(result) == 1
        assert len(result[0]["f16"]) == 2

        # BINARY
        struct_field = schema_types.FieldData()
        bin_field = schema_types.FieldData()
        bin_field.field_name = "bin"
        bin_field.type = DataType._ARRAY_OF_VECTOR

        vec_data = schema_types.VectorField()
        vec_data.dim = 8
        vec_data.binary_vector = b"\x01"

        bin_field.vectors.vector_array.element_type = DataType.BINARY_VECTOR
        bin_field.vectors.vector_array.data.append(vec_data)
        struct_field.struct_arrays.fields.append(bin_field)

        result = extract_struct_array_from_column_data(struct_field.struct_arrays, 0)
        assert len(result) == 1
        assert result[0]["bin"] == [b"\x01"]

        # INT8
        struct_field = schema_types.FieldData()
        int8_field = schema_types.FieldData()
        int8_field.field_name = "int8"
        int8_field.type = DataType._ARRAY_OF_VECTOR

        vec_data = schema_types.VectorField()
        vec_data.dim = 2
        vec_data.int8_vector = b"\x01\x02"

        int8_field.vectors.vector_array.element_type = DataType.INT8_VECTOR
        int8_field.vectors.vector_array.data.append(vec_data)
        struct_field.struct_arrays.fields.append(int8_field)

        result = extract_struct_array_from_column_data(struct_field.struct_arrays, 0)
        assert len(result) == 1
        assert result[0]["int8"] == [1, 2]

    def test_extract_struct_array_unexpected_field_type(self):
        """Test error when struct contains unexpected field type"""

        struct_field = schema_types.FieldData()

        # Invalid sub-field type (INT64 is not ARRAY or ARRAY_OF_VECTOR)
        bad_field = schema_types.FieldData()
        bad_field.field_name = "bad"
        bad_field.type = DataType.INT64

        struct_field.struct_arrays.fields.append(bad_field)

        # Should detect 0 structs because type is not one of checked ones for counting
        result = extract_struct_array_from_column_data(struct_field.struct_arrays, 0)
        assert len(result) == 0

        # Force a count > 0 to test the inner loop error
        # Add a valid field first so num_structs > 0
        valid_field = schema_types.FieldData()
        valid_field.field_name = "valid"
        valid_field.type = DataType.ARRAY
        arr = schema_types.ScalarField()
        arr.int_data.data.extend([1])
        valid_field.scalars.array_data.data.append(arr)
        struct_field.struct_arrays.fields.insert(0, valid_field)

        with pytest.raises(ParamError, match="Unexpected field type"):
            extract_struct_array_from_column_data(struct_field.struct_arrays, 0)
