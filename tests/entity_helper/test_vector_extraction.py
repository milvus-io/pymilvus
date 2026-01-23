import numpy as np
import pytest
from pymilvus.client.entity_helper import extract_vector_array_row_data
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2 as schema_types


class TestExtractVectorArrayRowDataExtended:
    """Test extract_vector_array_row_data for remaining types"""

    def test_extract_float16_vector_array(self):
        """Test extracting FLOAT16_VECTOR array"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR

        vector_data = schema_types.VectorField()
        # 2 elements: 1.0, 2.0
        vector_data.float16_vector = np.array([1.0, 2.0], dtype=np.float16).tobytes()
        field_data.vectors.vector_array.element_type = DataType.FLOAT16_VECTOR
        field_data.vectors.vector_array.data.append(vector_data)

        result = extract_vector_array_row_data(field_data, 0)
        assert result == pytest.approx([1.0, 2.0])

    def test_extract_bfloat16_vector_array(self):
        """Test extracting BFLOAT16_VECTOR array"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR

        vector_data = schema_types.VectorField()
        # Mock bfloat16 bytes
        vector_data.bfloat16_vector = b"\x01\x02\x03\x04"
        field_data.vectors.vector_array.element_type = DataType.BFLOAT16_VECTOR
        field_data.vectors.vector_array.data.append(vector_data)

        result = extract_vector_array_row_data(field_data, 0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_extract_int8_vector_array(self):
        """Test extracting INT8_VECTOR array"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR

        vector_data = schema_types.VectorField()
        vector_data.int8_vector = b"\x01\x02"
        field_data.vectors.vector_array.element_type = DataType.INT8_VECTOR
        field_data.vectors.vector_array.data.append(vector_data)

        result = extract_vector_array_row_data(field_data, 0)
        assert result == [1, 2]

    def test_extract_vector_array_unimplemented(self):
        """Test extracting unsupported vector type"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR

        vector_data = schema_types.VectorField()
        field_data.vectors.vector_array.element_type = 999
        field_data.vectors.vector_array.data.append(vector_data)

        with pytest.raises(ParamError, match="Unimplemented type"):
            extract_vector_array_row_data(field_data, 0)
