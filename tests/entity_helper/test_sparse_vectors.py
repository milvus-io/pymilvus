import json
import struct
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException
from pymilvus.grpc_gen import schema_pb2
from pymilvus.grpc_gen import schema_pb2 as schema_types
from scipy.sparse import csr_matrix


class TestEntityHelperSparse:
    """Test entity_helper module functions"""

    @pytest.mark.parametrize(
        "valid_sparse_matrix",
        [
            [{0: 1.0, 5: 2.5, 10: 3.0}],  # list of one dict
            [{0: 1.0, 5: 2.5}, {10: 3.0, 15: 4.0}],  # list of dicts
            [{}, {10: 3.0, 15: 4.0}],  # list of dicts partial empty is allowed
            [[(1, 0.5), (10, 0.3)], [(2, 0.7), (20, 0.1)]],  # list of list
            [[("1", "0.5"), (10, 0.3)]],  # str representation of int
            csr_matrix(
                ([1.0, 2.0, 3.0], [0, 2, 3], [0, 2, 3, 3]), shape=(3, 4)
            ),  # scipy sparse matrix
            [
                csr_matrix([[1.0, 0, 2.0]]),
                csr_matrix([[0, 0, 3.0]]),
            ],  # list of scipy sparse matrices
        ],
    )
    def test_entity_is_sparse_matrix(self, valid_sparse_matrix: list):
        assert entity_helper.entity_is_sparse_matrix(valid_sparse_matrix) is True

    @pytest.mark.parametrize(
        "not_sparse_matrix",
        [
            [{"a": 1.0, "b": 2.0}],  # invalid dict for non-numeric keys
            [],  # empty
            [{0: 1.0}, "not a dict", {5: 2.0}],  # mixed lists
            None,
            123,
            "string",
            [1, 2, 3],
            [[1, 2, 3]],
            [[(1, 0.5, 0.2)]],
            [[(1, "invalid")]],
            [csr_matrix([[1, 0], [0, 1]])],  # list of multi-row is not sparse
        ],
    )
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

        sparse_list = [{0: 1.0}, {5: 2.5}, {10: 3.0}]
        assert entity_helper.get_input_num_rows(sparse_list) == 3

        data = np.array([[1, 2, 3], [4, 5, 6]])
        assert entity_helper.get_input_num_rows(data) == 2

    @pytest.mark.parametrize(
        "sparse_list", [[{0: 1.0, 2: 2.0}, {2: 3.0}], csr_matrix([[1, 0, 2], [0, 0, 3]])]
    )
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
        vec1_data = b""
        for idx, val in [(0, 1.0), (5, 2.5), (10, 3.0)]:
            vec1_data += struct.pack("I", idx) + struct.pack("f", val)
        proto.contents.append(vec1_data)

        vec2_data = b""
        for idx, val in [(15, 4.0), (20, 5.0)]:
            vec2_data += struct.pack("I", idx) + struct.pack("f", val)
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
            vec_data = struct.pack("I", i) + struct.pack("f", float(i))
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
                    "list": [np.int32(1), np.float64(2.5)],
                }
            }
        }
        result = entity_helper.convert_to_json(obj)
        parsed = json.loads(result)
        assert parsed["level1"]["level2"]["array"] == [[1, 2], [3, 4]]

    @pytest.mark.parametrize(
        "data",
        [
            {"key": "value", "number": 42},  # dict
            {"outer": {"inner": "value"}},  # nested dict
            [1, 2, 3, "four"],  # list
            [{"a": 1}, {"b": 2}],  # list of dict
            None,
            {"int": np.int64(42), "float": np.float32(3.14), "bool": np.bool_(True)},
            [{"val": np.int64(10)}, {"val": np.float32(3.14)}],
        ],
    )
    def test_convert_to_json_dict(self, data: dict):
        """Test JSON conversion for dict input"""
        result = entity_helper.convert_to_json(data)
        assert isinstance(result, bytes)
        assert json.loads(result.decode()) == data

    @pytest.mark.parametrize(
        "json_string,expected",
        [
            ('{"key": "value", "number": 42}', {"key": "value", "number": 42}),
            ('{"nested": {"inner": "value"}}', {"nested": {"inner": "value"}}),
            ('[1, 2, 3, "four"]', [1, 2, 3, "four"]),
            ('{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
            ("null", None),
            ("true", True),
            ("false", False),
            ("123", 123),
            ('"simple string"', "simple string"),
        ],
    )
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

    @pytest.mark.parametrize(
        "invalid_json_string",
        [
            "not a json string",
            '{"invalid": }',
            '{"key": "value"',  # missing closing brace
            "{'key': 'value'}",  # single quotes not valid in JSON
            "{key: value}",  # unquoted keys
            "undefined",
            "{,}",
        ],
    )
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
            result = {"id_0": "-501"}
            for i in range(1, depth + 1):
                result = {"id_" + str(i): result}
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
            assert f"id_{i}" in current, f"Missing key id_{i} at depth {512 - i + 1}"
            current = current[f"id_{i}"]
        assert current == {"id_0": "-501"}, "Final nested value mismatch"

    def test_pack_field_value_to_field_data(self):
        """Test packing field values into field data protobuf"""
        # Test with scalar field
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT64
        field_info = {"name": "test_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        value = 42
        entity_helper.pack_field_value_to_field_data(
            value, field_data, field_info, vector_bytes_cache
        )

        assert len(field_data.scalars.long_data.data) == 1
        assert field_data.scalars.long_data.data[0] == value

    def test_pack_field_value_to_field_data_vectors(self):
        """Test packing vector field values"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_info = {"name": "vector_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        value = [1.0, 2.0, 3.0, 4.0]

        entity_helper.pack_field_value_to_field_data(
            value, field_data, field_info, vector_bytes_cache
        )

        assert field_data.vectors.dim == 4
        assert list(field_data.vectors.float_vector.data) == value

    @pytest.mark.parametrize(
        "element_type,data_attr,data_values,expected,use_approx",
        [
            (DataType.INT64, "long_data", [0, 1, 2], [0, 1, 2], False),
            (DataType.VARCHAR, "string_data", ["str_0", "str_1"], ["str_0", "str_1"], False),
            (DataType.BOOL, "bool_data", [True, False, True], [True, False, True], False),
            (DataType.FLOAT, "float_data", [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], True),
            (
                DataType.DOUBLE,
                "double_data",
                [1.11111, 2.22222, 3.33333],
                [1.11111, 2.22222, 3.33333],
                True,
            ),
        ],
    )
    def test_extract_array_row_data_types(
        self, element_type, data_attr, data_values, expected, use_approx
    ):
        """Test extracting array data of various types from protobuf"""
        field_data = schema_pb2.FieldData()
        field_data.scalars.array_data.element_type = element_type

        scalar_field = schema_pb2.ScalarField()
        getattr(scalar_field, data_attr).data.extend(data_values)
        field_data.scalars.array_data.data.append(scalar_field)

        result = entity_helper.extract_array_row_data(field_data, 0)
        if use_approx:
            assert result == pytest.approx(expected)
        else:
            assert result == expected

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


class TestMockedSparseMatrix:
    """Test sparse matrix functions using mocks to avoid environment issues"""

    def test_entity_is_sparse_matrix_mocked(self):
        """Test entity_is_sparse_matrix with mocked scipy sparse matrix"""

        # Create a mock that acts like a scipy sparse matrix
        mock_sparse = MagicMock()
        mock_sparse.shape = (1, 10)

        def is_sparse_side_effect(arg):
            return isinstance(arg, MagicMock)

        # Patch the name in entity_helper namespace to be sure
        with patch(
            "pymilvus.client.entity_helper.SciPyHelper.is_scipy_sparse",
            side_effect=is_sparse_side_effect,
        ):
            assert entity_helper.entity_is_sparse_matrix(mock_sparse) is True

            # Test list of sparse matrices
            mock_sparse2 = MagicMock()
            mock_sparse2.shape = (1, 10)
            assert entity_helper.entity_is_sparse_matrix([mock_sparse, mock_sparse2]) is True

    def test_pack_field_value_to_field_data_sparse_mocked(self):
        """Test packing sparse vector using mocks"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.SPARSE_FLOAT_VECTOR
        field_data.field_name = "sparse"
        field_info = {"name": "sparse"}
        vector_bytes_cache = {}

        # Mock sparse matrix (scipy.sparse.csr_matrix style)
        mock_sparse = MagicMock()
        mock_sparse.shape = (1, 10)
        mock_sparse.tocsr.return_value = mock_sparse
        mock_sparse.indptr = [0, 2]
        mock_sparse.indices = [1, 5]
        mock_sparse.data = [0.5, 0.8]

        def is_sparse_side_effect(arg):
            return isinstance(arg, MagicMock)

        with patch(
            "pymilvus.client.entity_helper.SciPyHelper.is_scipy_sparse",
            side_effect=is_sparse_side_effect,
        ):
            entity_helper.pack_field_value_to_field_data(
                mock_sparse, field_data, field_info, vector_bytes_cache
            )

        assert len(field_data.vectors.sparse_float_vector.contents) == 1
        # content is bytes, we can parse it back to check
        row = entity_helper.sparse_parse_single_row(
            field_data.vectors.sparse_float_vector.contents[0]
        )
        assert len(row) == 2
        assert row[1] == pytest.approx(0.5)
        assert row[5] == pytest.approx(0.8)

    def test_sparse_rows_to_proto_with_list_of_mocks(self):
        """Test sparse_rows_to_proto with list of mocked sparse matrices"""

        mock_sparse1 = MagicMock()
        mock_sparse1.shape = (1, 10)
        mock_sparse1.indices = [1]
        mock_sparse1.data = [0.5]

        mock_sparse2 = MagicMock()
        mock_sparse2.shape = (1, 10)
        mock_sparse2.indices = [2]
        mock_sparse2.data = [0.6]

        def is_sparse_side_effect(arg):
            return isinstance(arg, MagicMock)

        with patch(
            "pymilvus.client.entity_helper.SciPyHelper.is_scipy_sparse",
            side_effect=is_sparse_side_effect,
        ):
            # Directly test the function that handles batch of sparse matrices
            proto = entity_helper.sparse_rows_to_proto([mock_sparse1, mock_sparse2])

        assert len(proto.contents) == 2
        row1 = entity_helper.sparse_parse_single_row(proto.contents[0])
        row2 = entity_helper.sparse_parse_single_row(proto.contents[1])
        assert row1[1] == pytest.approx(0.5)
        assert row2[2] == pytest.approx(0.6)
