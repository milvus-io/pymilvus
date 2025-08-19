import json
import struct

import numpy as np
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2
from scipy.sparse import csr_matrix


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
        #  {"array": np.array([1, 2, 3])}, # TODO
        { "int": np.int64(42), "float": np.float32(3.14), "bool": np.bool_(True) },
        [{"val": np.int64(10)}, {"val": np.float32(3.14)}],
    ])
    def test_convert_to_json_dict(self, data: dict):
        """Test JSON conversion for dict input"""
        result = entity_helper.convert_to_json(data)
        assert isinstance(result, bytes)
        assert json.loads(result.decode()) == data

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
