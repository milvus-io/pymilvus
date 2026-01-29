import struct
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.entity_helper import (
    convert_to_json,
    entity_to_field_data,
    entity_to_json_arr,
    extract_row_data_from_fields_data,
    extract_struct_array_from_column_data,
    pack_field_value_to_field_data,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ParamError
from pymilvus.grpc_gen import schema_pb2 as schema_types


class TestEntityHelperEdgeCases:
    """Consolidated edge case tests for entity_helper functions.

    This class consolidates tests from several smaller test classes:
    - TestStructArrayTruncated
    - TestJsonStrInput
    - TestEntityToFieldDataExtended
    - TestPackFieldValueNone
    - TestJsonEdgeCases
    - TestPackExceptionsMock
    - TestSparseEdgeCases
    - TestExtractRowDataV1Int8
    """

    # Tests from TestStructArrayTruncated
    def test_struct_array_truncated_vectors_all_types(self):
        """Test extract_struct_array_from_column_data with truncated/invalid vector data"""

        def set_float(vec_data):
            vec_data.float_vector.data.extend([1.0, 2.0])

        def set_float16(vec_data):
            vec_data.float16_vector = b"\x00\x00\x00\x00"

        def set_bfloat16(vec_data):
            vec_data.bfloat16_vector = b"\x00\x00\x00\x00"

        def set_int8(vec_data):
            vec_data.int8_vector = b"\x00\x00"

        def set_binary(vec_data):
            vec_data.binary_vector = b"\x00"

        types_setup = [
            (DataType.FLOAT_VECTOR, "float_vector", set_float, 2),
            (DataType.FLOAT16_VECTOR, "float16_vector", set_float16, 2),
            (DataType.BFLOAT16_VECTOR, "bfloat16_vector", set_bfloat16, 2),
            (DataType.INT8_VECTOR, "int8_vector", set_int8, 2),
            (DataType.BINARY_VECTOR, "binary_vector", set_binary, 8),
        ]

        for dtype, _attr_name, fill_func, dim in types_setup:
            sa_field = schema_types.FieldData()
            sa_field.type = DataType._ARRAY_OF_STRUCT

            vec_field = schema_types.FieldData()
            vec_field.type = DataType._ARRAY_OF_VECTOR
            vec_field.field_name = "vec"
            vec_field.vectors.vector_array.element_type = dtype
            vec_f = schema_types.VectorField()
            vec_field.vectors.vector_array.data.append(vec_f)

            vector_data = vec_field.vectors.vector_array.data[0]
            vector_data.dim = dim

            fill_func(vector_data)

            arr_field = schema_types.FieldData()
            arr_field.type = DataType.ARRAY
            arr_field.field_name = "ids"
            arr_scalar = schema_types.ScalarField()
            arr_field.scalars.array_data.data.append(arr_scalar)
            arr_field.scalars.array_data.data[0].int_data.data.extend([1, 2])

            sa_field.struct_arrays.fields.add().CopyFrom(arr_field)
            sa_field.struct_arrays.fields.add().CopyFrom(vec_field)

            result = extract_struct_array_from_column_data(sa_field.struct_arrays, 0)
            assert len(result) == 2
            assert result[1]["vec"] is None

    # Tests from TestJsonStrInput
    def test_json_str_input_valid(self):
        """Test string input for convert_to_json"""
        res = convert_to_json('{"key": 1}')
        assert res == b'{"key": 1}'

    # Tests from TestEntityToFieldDataExtended
    def test_entity_to_field_data_timestamptz(self):
        """Test entity_to_field_data for TIMESTAMPTZ type"""

        entity = {
            "type": DataType.TIMESTAMPTZ,
            "name": "ts_field",
            "values": ["2024-01-01T00:00:00Z", "2024-06-15T12:30:00Z"],
        }
        field_info = {"name": "ts_field"}

        result = entity_to_field_data(entity, field_info, 2)
        assert result.field_name == "ts_field"
        assert list(result.scalars.string_data.data) == [
            "2024-01-01T00:00:00Z",
            "2024-06-15T12:30:00Z",
        ]

    def test_entity_to_field_data_geometry(self):
        """Test entity_to_field_data for GEOMETRY type"""

        entity = {
            "type": DataType.GEOMETRY,
            "name": "geo_field",
            "values": ["POINT(0 0)", "LINESTRING(0 0, 1 1)"],
        }
        field_info = {"name": "geo_field", "params": {"max_length": 1000}}

        result = entity_to_field_data(entity, field_info, 2)
        assert result.field_name == "geo_field"
        assert len(result.scalars.geometry_wkt_data.data) == 2

    # Tests from TestPackFieldValueNone
    def test_pack_none_values_scalars(self):
        """Test packing None values for scalar types"""

        types_to_test = [
            (DataType.BOOL, "bool_data"),
            (DataType.INT8, "int_data"),
            (DataType.INT16, "int_data"),
            (DataType.INT32, "int_data"),
            (DataType.INT64, "long_data"),
            (DataType.FLOAT, "float_data"),
            (DataType.DOUBLE, "double_data"),
            (DataType.VARCHAR, "string_data"),
            (DataType.TIMESTAMPTZ, "string_data"),
            (DataType.JSON, "json_data"),
        ]

        for dtype, field_attr in types_to_test:
            field_data = schema_types.FieldData()
            field_data.type = dtype
            field_data.field_name = "f"
            field_info = {"name": "f"}

            pack_field_value_to_field_data(None, field_data, field_info, {})
            pack_field_value_to_field_data(None, field_data, field_info, {})

            field_container = getattr(field_data.scalars, field_attr)
            assert len(field_container.data) == 0

    def test_pack_none_values_vectors(self):
        """Test packing None values for vector types"""

        vector_types = [
            DataType.FLOAT_VECTOR,
            DataType.BINARY_VECTOR,
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
            DataType.SPARSE_FLOAT_VECTOR,
            DataType.INT8_VECTOR,
        ]

        for dtype in vector_types:
            field_data = schema_types.FieldData()
            field_data.type = dtype
            field_data.field_name = "v"
            field_info = {"name": "v", "params": {"dim": 8}}

            pack_field_value_to_field_data(None, field_data, field_info, {})

            if dtype != DataType.SPARSE_FLOAT_VECTOR:
                assert field_data.vectors.dim == 8
            else:
                assert field_data.vectors.dim == 0

    # Tests from TestJsonEdgeCases
    def test_json_dict_with_int_keys(self):
        """Test JSON conversion rejects dict with int keys"""

        data = {1: "value"}
        with pytest.raises(DataNotMatchException, match="JSON key must be str"):
            convert_to_json(data)

    def test_json_arr_with_none(self):
        """Test entity_to_json_arr rejects None values"""

        with pytest.raises(ParamError):
            entity_to_json_arr([None], {"name": "json_field"})

    # Tests from TestPackExceptionsMock
    def test_pack_exceptions_mock(self):
        """Test exception handling in pack_field_value_to_field_data using mocks"""

        fail_targets = [
            (DataType.BOOL, "bool_data", True),
            (DataType.INT32, "int_data", 1),
            (DataType.INT64, "long_data", 1),
            (DataType.FLOAT, "float_data", 1.0),
            (DataType.DOUBLE, "double_data", 1.0),
            (DataType.VARCHAR, "string_data", "s"),
            (DataType.TIMESTAMPTZ, "string_data", "2024-01-01"),
            (DataType.JSON, "json_data", {"x": 1}),
            (DataType.ARRAY, "array_data", [1, 2]),
            (DataType.GEOMETRY, "geometry_wkt_data", "POINT(0 0)"),
            (DataType.FLOAT_VECTOR, "float_vector", [1.0] * 8),
            (DataType.BINARY_VECTOR, "binary_vector", b"\x00"),
            (DataType.FLOAT16_VECTOR, "float16_vector", b"\x00\x00"),
            (DataType.BFLOAT16_VECTOR, "bfloat16_vector", b"\x00\x00"),
            (DataType.SPARSE_FLOAT_VECTOR, "sparse_float_vector", {0: 1.0}),
            (DataType.INT8_VECTOR, "int8_vector", b"\x00"),
        ]

        for dtype, attr_name, val in fail_targets:
            field_data_mock = MagicMock()
            field_data_mock.type = dtype
            field_data_mock.vectors.dim = 0
            field_info = {
                "name": "f",
                "element_type": DataType.INT64,
            }

            valid_val = val

            if "vector" in attr_name:
                if attr_name == "float_vector":
                    getattr(field_data_mock.vectors, attr_name).data.extend.side_effect = TypeError(
                        "Mock Error"
                    )
                elif attr_name == "sparse_float_vector":
                    getattr(field_data_mock.vectors, attr_name).contents.append.side_effect = (
                        TypeError("Mock Error")
                    )
            else:
                getattr(field_data_mock.scalars, attr_name).data.append.side_effect = TypeError(
                    "Mock Error"
                )

            if dtype in (
                DataType.BINARY_VECTOR,
                DataType.FLOAT16_VECTOR,
                DataType.BFLOAT16_VECTOR,
                DataType.INT8_VECTOR,
            ):
                continue

            if dtype == DataType.SPARSE_FLOAT_VECTOR:
                with patch(
                    "pymilvus.client.entity_helper.SciPyHelper.is_scipy_sparse", return_value=False
                ):
                    with pytest.raises(DataNotMatchException):
                        pack_field_value_to_field_data(valid_val, field_data_mock, field_info, {})
            else:
                with pytest.raises(DataNotMatchException):
                    pack_field_value_to_field_data(valid_val, field_data_mock, field_info, {})

    def test_pack_vector_exceptions_value_trigger(self):
        """Trigger exceptions in vector packing using object that raises error"""

        class ErrorObj:
            def __len__(self):
                raise TypeError("Len Error")

            def __bytes__(self):
                raise TypeError("Bytes Error")

            def __iter__(self):
                raise TypeError("Iter Error")

        field_info = {"name": "f"}

        fd = schema_types.FieldData()
        fd.type = DataType.BINARY_VECTOR
        with pytest.raises(DataNotMatchException):
            pack_field_value_to_field_data(ErrorObj(), fd, field_info, {})

    # Tests from TestSparseEdgeCases
    def test_sparse_proto_to_rows_empty_range(self):
        """Test sparse_proto_to_rows with same start and end"""
        proto = schema_types.SparseFloatArray(dim=100)
        proto.contents.append(struct.pack("I", 0) + struct.pack("f", 1.0))

        rows = entity_helper.sparse_proto_to_rows(proto, 0, 0)
        assert len(rows) == 0

    def test_entity_is_sparse_matrix_exception_handling(self):
        """Test entity_is_sparse_matrix exception branch"""

        class BadIterator:
            def __iter__(self):
                raise RuntimeError("Cannot iterate")

            def __len__(self):
                return 1

        result = entity_helper.entity_is_sparse_matrix(BadIterator())
        assert result is False

    # Tests from TestExtractRowDataV1Int8
    def test_extract_v1_int8_vector(self):
        """Test extraction of INT8 vectors in V1"""

        fd = schema_types.FieldData()
        fd.type = DataType.INT8_VECTOR
        fd.field_name = "i8"
        fd.vectors.dim = 2

        fd.vectors.int8_vector = b"\x01\x02\x03\x04"

        row0 = extract_row_data_from_fields_data([fd], 0)

        assert row0["i8"] == [b"\x01\x02"]


class TestEntityHelperCoverage:
    """Test class for remaining coverage gaps in entity_helper.py"""

    def test_entity_is_sparse_matrix_inner_functions(self):
        """Cover inner functions of entity_is_sparse_matrix"""
        # Test case where row is valid list/dict but elements fail type check
        # Non-string fallback in is_type_in_str
        data = [[(1, object())]]
        assert entity_helper.entity_is_sparse_matrix(data) is False

        # inner is_type_in_str: passed string that is not parseable
        data = [[(1, "not_a_number")]]
        assert entity_helper.entity_is_sparse_matrix(data) is False

    def test_sparse_rows_to_proto_errors(self):
        """Cover sparse_rows_to_proto error paths"""
        # Length mismatch - hard to trigger via public API due to zip usage/dict conversion
        # Skipping length mismatch test

        with pytest.raises(ParamError, match=r"positive and less than 2\^32-1"):
            # Index -1
            entity_helper.sparse_rows_to_proto([{-1: 0.5}])

        # NaN value
        with pytest.raises(ParamError, match="NaN"):
            entity_helper.sparse_rows_to_proto([{1: float("nan")}])

    def test_sparse_proto_to_rows_defaults(self):
        """Cover sparse_proto_to_rows default arguments"""
        proto = schema_types.SparseFloatArray()
        # Add dummy content
        proto.contents.append(b"dummy")

        with mock.patch("pymilvus.client.entity_helper.sparse_parse_single_row") as mock_parse:
            entity_helper.sparse_proto_to_rows(proto)  # defaults start=None, end=None
            mock_parse.assert_called()

        with pytest.raises(ParamError, match="Vector must be a sparse float vector"):
            entity_helper.sparse_proto_to_rows("not_a_proto")

    def test_get_input_num_rows_mock(self):
        """Cover get_input_num_rows with scipy sparse"""
        with mock.patch(
            "pymilvus.client.entity_helper.SciPyHelper.is_scipy_sparse", return_value=True
        ):
            mock_entity = mock.MagicMock()
            mock_entity.shape = (5, 10)
            assert entity_helper.get_input_num_rows(mock_entity) == 5

    def test_entity_type_to_dtype(self):
        """Cover entity_type_to_dtype"""
        assert entity_helper.entity_type_to_dtype(100) == 100  # int pass through
        assert entity_helper.entity_type_to_dtype("Int64") == DataType.INT64
        with pytest.raises(ParamError):
            entity_helper.entity_type_to_dtype(None)

    def test_convert_to_str_array_errors(self):
        """Cover convert_to_str_array errors"""
        field_info = {"name": "test", "params": {"max_length": 5}}

        # Not a string
        with pytest.raises(ParamError, match="expects string input"):
            entity_helper.convert_to_str_array([123], field_info)

        # Too long
        with pytest.raises(ParamError, match="exceeds max length"):
            entity_helper.convert_to_str_array(["123456"], field_info)

    def test_convert_to_json_errors(self):
        """Cover convert_to_json errors"""
        # non-string key in dict
        with pytest.raises(DataNotMatchException, match="JSON key must be str"):
            entity_helper.convert_to_json({1: "v"})

        # non-string key in parsed json string
        # Mock orjson.loads to return a dict with non-string keys from a string input
        with mock.patch("pymilvus.client.entity_helper.orjson.loads", return_value={1: "v"}):
            with pytest.raises(DataNotMatchException, match="Invalid JSON string"):
                entity_helper.convert_to_json('{"1": "v"}')

    def test_convert_to_json_arr_none(self):
        """Cover convert_to_json_arr None check"""
        field_info = {"name": "test"}
        with pytest.raises(ParamError, match="expects a non-None input"):
            entity_helper.convert_to_json_arr([None], field_info)

    def test_convert_to_array_unsupported(self):
        """Cover convert_to_array unsupported type"""
        field_info = {"name": "test", "element_type": DataType.UNKNOWN}
        with pytest.raises(ParamError, match="Unsupported element type"):
            entity_helper.convert_to_array([], field_info)

    def test_convert_to_array_of_vector_errors(self):
        """Cover convert_to_array_of_vector errors"""
        field_info = {"name": "test", "element_type": DataType.FLOAT_VECTOR, "params": {"dim": 4}}

        # Invalid dtype
        arr = [np.array([1, 2], dtype="int")]
        with pytest.raises(ParamError, match="dtype=float32"):
            entity_helper.convert_to_array_of_vector(arr, field_info)

        # Unsupported element type
        field_info["element_type"] = DataType.INT64
        with pytest.raises(ParamError, match="Unsupported element type"):
            entity_helper.convert_to_array_of_vector([], field_info)

    def test_flush_vector_bytes(self):
        """Cover flush_vector_bytes"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.INT8_VECTOR
        cache = {id(field_data): [b"\x01"]}

        entity_helper.flush_vector_bytes(field_data, cache)
        assert field_data.vectors.int8_vector == b"\x01"
        assert id(field_data) not in cache

        # Empty cache
        entity_helper.flush_vector_bytes(field_data, {})
        # Should do nothing
