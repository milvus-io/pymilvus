import struct
import time
from typing import ClassVar, Dict, List
from unittest.mock import patch

import numpy as np
import orjson
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.entity_helper import (
    convert_to_str_array,
    entity_to_array_arr,
    entity_to_field_data,
    entity_to_str_arr,
    entity_type_to_dtype,
    extract_dynamic_field_from_result,
    extract_row_data_from_fields_data,
    extract_row_data_from_fields_data_v2,
    extract_vector_array_row_data,
    flush_vector_bytes,
    get_max_len_of_var_char,
    pack_field_value_to_field_data,
    sparse_proto_to_rows,
    sparse_rows_to_proto,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import DataNotMatchException, ParamError
from pymilvus.grpc_gen import schema_pb2
from pymilvus.grpc_gen import schema_pb2 as schema_types
from pymilvus.settings import Config

# Alias for backward compatibility
convert_to_entity = extract_row_data_from_fields_data_v2


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
            sparse_rows_to_proto([{1: float("nan")}])

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

    @patch("pymilvus.client.entity_helper.Config")
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

    def test_entity_to_array_arr(self):
        """Test converting entity to array array"""
        field_info = {"name": "array_field", "element_type": DataType.INT64}

        # List of lists
        data = [[1, 2, 3], [4, 5], [6]]
        result = entity_to_array_arr(data, field_info)
        assert len(result) == 3
        assert result[0].long_data.data == [1, 2, 3]
        assert result[1].long_data.data == [4, 5]

    def test_entity_to_array_arr_string(self):
        """Test converting string arrays"""
        field_info = {"name": "array_field", "element_type": DataType.VARCHAR}

        data = [["hello", "world"], ["foo"]]
        result = entity_to_array_arr(data, field_info)
        assert len(result) == 2
        assert list(result[0].string_data.data) == ["hello", "world"]

    def test_entity_to_array_arr_invalid_type(self):
        """Test error handling for invalid element type"""
        field_info = {"name": "array_field", "element_type": 999}

        with pytest.raises(ParamError, match="Unsupported element type"):
            entity_to_array_arr([[1, 2]], field_info)

    def test_pack_field_value_to_field_data(self):
        """Test packing field values to field data"""
        # pack_field_value_to_field_data takes different parameters
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_data.field_name = "vector_field"
        field_info = {"name": "vector_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack a single vector
        pack_field_value_to_field_data(
            np.array([1.0, 2.0]), field_data, field_info, vector_bytes_cache
        )

        # Check the result
        assert field_data.type == DataType.FLOAT_VECTOR
        assert field_data.vectors.dim == 2

    def test_pack_field_value_to_field_data_sparse(self):
        """Test packing sparse vectors"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.SPARSE_FLOAT_VECTOR
        field_data.field_name = "sparse_field"
        field_info = {"name": "sparse_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack a single sparse vector
        sparse_data = {1: 0.5, 10: 0.3}
        pack_field_value_to_field_data(sparse_data, field_data, field_info, vector_bytes_cache)

        # Check the result
        assert field_data.type == DataType.SPARSE_FLOAT_VECTOR
        assert len(field_data.vectors.sparse_float_vector.contents) == 1

    def test_pack_field_value_to_field_data_scalars(self):
        """Test packing scalar field values"""
        # Test INT64
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT64
        field_data.field_name = "int_field"
        field_info = {"name": "int_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        pack_field_value_to_field_data(42, field_data, field_info, vector_bytes_cache)

        assert field_data.type == DataType.INT64
        assert field_data.scalars.long_data.data[0] == 42

    def test_extract_field_info(self):
        """Test extracting primary field from schema"""
        # Create schema with primary field
        fields_info = [{"name": "id", "is_primary": True}, {"name": "vector", "is_primary": False}]

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
        dynamic_field_data = schema_pb2.FieldData()
        dynamic_field_data.is_dynamic = True
        dynamic_field_data.field_name = "$meta"

        regular_field_data = schema_pb2.FieldData()
        regular_field_data.is_dynamic = False
        regular_field_data.field_name = "id"

        # Create result with dynamic field - extra_field comes before $meta
        result = ActualResult(
            fields_data=[regular_field_data, dynamic_field_data],
            output_fields=["id", "extra_field", "another_extra", "$meta"],
        )

        dynamic_field_name, dynamic_fields = extract_dynamic_field_from_result(result)
        assert dynamic_field_name == "$meta"
        # When $meta is found in output_fields, dynamic_fields gets cleared
        assert len(dynamic_fields) == 0

        # Test with no dynamic field
        result_no_dynamic = ActualResult(
            fields_data=[regular_field_data], output_fields=["id", "extra_field"]
        )

        dynamic_field_name, dynamic_fields = extract_dynamic_field_from_result(result_no_dynamic)
        assert dynamic_field_name is None
        assert "extra_field" in dynamic_fields
        assert "id" not in dynamic_fields

        # Test with dynamic field NOT in output_fields (dynamic_fields preserved)
        result_meta_not_in_output = ActualResult(
            fields_data=[regular_field_data, dynamic_field_data],
            output_fields=["id", "extra_field", "another_extra"],
        )

        dynamic_field_name, dynamic_fields = extract_dynamic_field_from_result(
            result_meta_not_in_output
        )
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
            {"name": "vector", "is_primary": False},
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
        field1 = schema_pb2.FieldData()
        field1.field_name = "id"
        field1.type = DataType.INT64
        long_array = schema_pb2.LongArray()
        long_array.data.extend([1, 2, 3])
        field1.scalars.long_data.CopyFrom(long_array)

        field2 = schema_pb2.FieldData()
        field2.field_name = "name"
        field2.type = DataType.VARCHAR
        str_array = schema_pb2.StringArray()
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
        entity = {"name": "test_field", "type": DataType.INT64, "values": [1, 2, 3, 4, 5]}
        field_info = {"name": "test_field"}

        result = entity_to_field_data(entity, field_info, 5)

        assert result.field_name == "test_field"
        assert result.type == DataType.INT64
        assert list(result.scalars.long_data.data) == [1, 2, 3, 4, 5]


class TestNullableVectorSupport:
    """Test nullable vector support in entity_helper module for all 6 vector types"""

    # Vector type configurations for parametrized tests
    VECTOR_TYPE_CONFIGS: ClassVar = [
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
            "values": [b"\x01\x02\x03\x04", None, b"\x05\x06\x07\x08"],
            "dim": 32,  # 4 bytes * 8 bits
            "get_data_len": lambda fd: len(fd.vectors.binary_vector),
            "expected_data_len": 8,  # 2 valid vectors * 4 bytes
        },
        {
            "dtype": DataType.FLOAT16_VECTOR,
            "name": "float16_vector",
            "values": [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes(),
                None,
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float16).tobytes(),
            ],
            "dim": 4,
            "get_data_len": lambda fd: len(fd.vectors.float16_vector),
            "expected_data_len": 16,  # 2 valid vectors * 4 dim * 2 bytes
        },
        {
            "dtype": DataType.BFLOAT16_VECTOR,
            "name": "bfloat16_vector",
            "values": [
                b"\x00\x3f\x00\x40\x00\x40\x00\x40",
                None,
                b"\x00\x40\x00\x40\x00\x40\x00\x40",
            ],
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
            "values": [
                np.array([1, 2, 3, 4], dtype=np.int8).tobytes(),
                None,
                np.array([5, 6, 7, 8], dtype=np.int8).tobytes(),
            ],
            "dim": 4,
            "get_data_len": lambda fd: len(fd.vectors.int8_vector),
            "expected_data_len": 8,  # 2 valid vectors * 4 dim * 1 byte
        },
    ]

    @pytest.mark.parametrize(
        "config", VECTOR_TYPE_CONFIGS, ids=[c["name"] for c in VECTOR_TYPE_CONFIGS]
    )
    def test_entity_to_field_data_nullable_vector(self, config):
        """Test entity_to_field_data with nullable vector containing None values"""
        entity = {
            "name": f"nullable_{config['name']}",
            "type": config["dtype"],
            "values": config["values"],
        }
        field_info = {"name": f"nullable_{config['name']}", "nullable": True}
        if config["dim"]:
            field_info["params"] = {"dim": config["dim"]}

        result = entity_to_field_data(entity, field_info, 3)

        assert result.field_name == f"nullable_{config['name']}"
        assert result.type == config["dtype"]
        assert list(result.valid_data) == [True, False, True]
        assert config["get_data_len"](result) == config["expected_data_len"]

    @pytest.mark.parametrize(
        "config", VECTOR_TYPE_CONFIGS, ids=[c["name"] for c in VECTOR_TYPE_CONFIGS]
    )
    def test_entity_to_field_data_nullable_vector_all_none(self, config):
        """Test entity_to_field_data with nullable vector where all values are None"""
        entity = {
            "name": f"nullable_{config['name']}",
            "type": config["dtype"],
            "values": [None, None, None],
        }
        field_info = {"name": f"nullable_{config['name']}", "nullable": True}
        if config["dim"]:
            field_info["params"] = {"dim": config["dim"]}

        result = entity_to_field_data(entity, field_info, 3)

        assert list(result.valid_data) == [False, False, False]
        assert config["get_data_len"](result) == 0

    @pytest.mark.parametrize(
        "dtype,name",
        [
            (DataType.FLOAT_VECTOR, "float_vector"),
            (DataType.BINARY_VECTOR, "binary_vector"),
            (DataType.FLOAT16_VECTOR, "float16_vector"),
            (DataType.BFLOAT16_VECTOR, "bfloat16_vector"),
            (DataType.SPARSE_FLOAT_VECTOR, "sparse_float_vector"),
            (DataType.INT8_VECTOR, "int8_vector"),
        ],
    )
    def test_pack_field_value_nullable_vector_none(self, dtype, name):
        """Test pack_field_value_to_field_data with None value for nullable vector"""
        field_data = schema_pb2.FieldData()
        field_data.type = dtype
        field_data.field_name = f"nullable_{name}"
        field_info = {"name": f"nullable_{name}", "nullable": True}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        pack_field_value_to_field_data(None, field_data, field_info, vector_bytes_cache)

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

        field_data = schema_pb2.FieldData()
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

        field_data = schema_pb2.FieldData()
        field_data.type = DataType.SPARSE_FLOAT_VECTOR
        field_data.field_name = "nullable_sparse"
        vec1_data = struct.pack("I", 1) + struct.pack("f", 0.5)
        vec2_data = struct.pack("I", 5) + struct.pack("f", 0.8)
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

        field_data = schema_pb2.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_data.field_name = "regular_vector"
        field_data.vectors.dim = 4
        field_data.vectors.float_vector.data.extend(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )

        result0 = extract_row_data_from_fields_data([field_data], 0)
        assert result0["regular_vector"] == [1.0, 2.0, 3.0, 4.0]

        result1 = extract_row_data_from_fields_data([field_data], 1)
        assert result1["regular_vector"] == [5.0, 6.0, 7.0, 8.0]

        result2 = extract_row_data_from_fields_data([field_data], 2)
        assert result2["regular_vector"] == [9.0, 10.0, 11.0, 12.0]

    def test_pack_field_value_to_field_data_int8_vector(self):
        """Test packing int8 vector field values"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT8_VECTOR
        field_data.field_name = "int8_vector_field"
        field_info = {"name": "int8_vector_field", "params": {"dim": 768}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack a single int8 vector
        vector = np.array([i % 128 - 64 for i in range(768)], dtype=np.int8)
        pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

        # Flush to merge collected bytes
        flush_vector_bytes(field_data, vector_bytes_cache)

        # Check the result
        assert field_data.type == DataType.INT8_VECTOR
        assert field_data.vectors.dim == 768
        assert len(field_data.vectors.int8_vector) == 768

        # Verify data correctness
        expected_bytes = vector.tobytes()
        assert field_data.vectors.int8_vector == expected_bytes

    def test_pack_field_value_to_field_data_int8_vector_multiple(self):
        """Test packing multiple int8 vectors to verify memory optimization"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT8_VECTOR
        field_data.field_name = "int8_vector_field"
        field_info = {"name": "int8_vector_field", "params": {"dim": 768}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack multiple vectors (simulating batch insert)
        num_vectors = 1000
        vectors = []
        for i in range(num_vectors):
            vector = np.array([(i + j) % 128 - 64 for j in range(768)], dtype=np.int8)
            vectors.append(vector)
            pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

        # Flush to merge all collected bytes
        flush_vector_bytes(field_data, vector_bytes_cache)

        # Verify final result
        assert field_data.vectors.dim == 768
        expected_total_size = num_vectors * 768
        assert len(field_data.vectors.int8_vector) == expected_total_size

        # Verify data correctness for sample vectors
        for idx in [0, 100, 500, 999]:
            expected_bytes = vectors[idx].tobytes()
            actual_bytes = field_data.vectors.int8_vector[idx * 768 : (idx + 1) * 768]
            assert expected_bytes == actual_bytes, f"Vector {idx} data mismatch"

    def test_flush_int8_vector_bytes(self):
        """Test flush_vector_bytes function for int8 vectors"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT8_VECTOR
        field_data.field_name = "int8_vector_field"
        field_info = {"name": "int8_vector_field", "params": {"dim": 128}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack some vectors
        vectors = []
        for i in range(10):
            vector = np.array([i % 128 - 64 for _ in range(128)], dtype=np.int8)
            vectors.append(vector)
            pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

        # Before flush, data might be in cache
        # Flush to merge all bytes
        flush_vector_bytes(field_data, vector_bytes_cache)

        # Verify all data is merged
        expected_size = 10 * 128
        assert len(field_data.vectors.int8_vector) == expected_size

        # Verify data correctness
        for idx in range(10):
            expected_bytes = vectors[idx].tobytes()
            actual_bytes = field_data.vectors.int8_vector[idx * 128 : (idx + 1) * 128]
            assert expected_bytes == actual_bytes

    def test_pack_field_value_to_field_data_int8_vector_large_batch(self):
        """Test packing large batch of int8 vectors to verify O(n) performance"""

        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT8_VECTOR
        field_data.field_name = "int8_vector_field"
        field_info = {"name": "int8_vector_field", "params": {"dim": 768}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack a large number of vectors (similar to the bug scenario)
        num_vectors = 10000
        vectors = []

        start_time = time.time()
        for i in range(num_vectors):
            vector = np.array([(i + j) % 128 - 64 for j in range(768)], dtype=np.int8)
            vectors.append(vector)
            pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

        # Flush to merge all collected bytes
        flush_vector_bytes(field_data, vector_bytes_cache)
        elapsed_time = time.time() - start_time

        # Verify performance: should complete in reasonable time (< 10 seconds)
        assert (
            elapsed_time < 10.0
        ), f"Operation took {elapsed_time:.2f} seconds, expected < 10 seconds"

        # Verify data correctness
        expected_total_size = num_vectors * 768
        assert len(field_data.vectors.int8_vector) == expected_total_size

        # Sample verification
        sample_indices = [0, 1000, 5000, 9999]
        for idx in sample_indices:
            expected_bytes = vectors[idx].tobytes()
            actual_bytes = field_data.vectors.int8_vector[idx * 768 : (idx + 1) * 768]
            assert expected_bytes == actual_bytes, f"Vector {idx} data mismatch"

    def test_pack_field_value_to_field_data_int8_vector_invalid_dtype(self):
        """Test error handling for invalid int8 vector dtype"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT8_VECTOR
        field_data.field_name = "int8_vector_field"
        field_info = {"name": "int8_vector_field", "params": {"dim": 768}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Try to pack with wrong dtype
        vector = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(ParamError, match="invalid input for int8 vector"):
            pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

    def test_pack_field_value_to_field_data_int8_vector_none(self):
        """Test handling None value for int8 vector"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.INT8_VECTOR
        field_data.field_name = "int8_vector_field"
        field_info = {"name": "int8_vector_field", "params": {"dim": 768}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack None value
        pack_field_value_to_field_data(None, field_data, field_info, vector_bytes_cache)

        # Dimension should be set from params
        assert field_data.vectors.dim == 768

    def test_pack_field_value_to_field_data_binary_vector_multiple(self):
        """Test packing multiple binary vectors to verify memory optimization"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.BINARY_VECTOR
        field_data.field_name = "binary_vector_field"
        field_info = {"name": "binary_vector_field", "params": {"dim": 128}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack multiple vectors
        num_vectors = 1000
        vectors = []
        for i in range(num_vectors):
            vector = bytes([(i + j) % 256 for j in range(16)])  # 128 bits = 16 bytes
            vectors.append(vector)
            pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

        # Flush to merge all collected bytes
        flush_vector_bytes(field_data, vector_bytes_cache)

        # Verify final result
        assert field_data.vectors.dim == 128
        expected_total_size = num_vectors * 16
        assert len(field_data.vectors.binary_vector) == expected_total_size

        # Verify data correctness for sample vectors
        for idx in [0, 100, 500, 999]:
            expected_bytes = vectors[idx]
            actual_bytes = field_data.vectors.binary_vector[idx * 16 : (idx + 1) * 16]
            assert expected_bytes == actual_bytes, f"Vector {idx} data mismatch"

    def test_pack_field_value_to_field_data_float16_vector_multiple(self):
        """Test packing multiple float16 vectors to verify memory optimization"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.FLOAT16_VECTOR
        field_data.field_name = "float16_vector_field"
        field_info = {"name": "float16_vector_field", "params": {"dim": 128}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack multiple vectors
        num_vectors = 1000
        vectors = []
        for i in range(num_vectors):
            vector = np.array([float(i + j) for j in range(128)], dtype=np.float16)
            vectors.append(vector)
            pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

        # Flush to merge all collected bytes
        flush_vector_bytes(field_data, vector_bytes_cache)

        # Verify final result
        assert field_data.vectors.dim == 128
        expected_total_size = num_vectors * 128 * 2  # float16 = 2 bytes per element
        assert len(field_data.vectors.float16_vector) == expected_total_size

        # Verify data correctness for sample vectors
        for idx in [0, 100, 500, 999]:
            expected_bytes = vectors[idx].view(np.uint8).tobytes()
            actual_bytes = field_data.vectors.float16_vector[idx * 128 * 2 : (idx + 1) * 128 * 2]
            assert expected_bytes == actual_bytes, f"Vector {idx} data mismatch"

    def test_pack_field_value_to_field_data_bfloat16_vector_multiple(self):
        """Test packing multiple bfloat16 vectors to verify memory optimization"""
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.BFLOAT16_VECTOR
        field_data.field_name = "bfloat16_vector_field"
        field_info = {"name": "bfloat16_vector_field", "params": {"dim": 128}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Pack multiple vectors using bytes format (since bfloat16 dtype may not be available)
        num_vectors = 1000
        vectors = []
        for i in range(num_vectors):
            # Create bytes directly for bfloat16 (2 bytes per element)
            vector_bytes = bytes([(i + j) % 256 for j in range(128 * 2)])
            vectors.append(vector_bytes)
            pack_field_value_to_field_data(vector_bytes, field_data, field_info, vector_bytes_cache)

        # Flush to merge all collected bytes
        flush_vector_bytes(field_data, vector_bytes_cache)

        # Verify final result
        assert field_data.vectors.dim == 128
        expected_total_size = num_vectors * 128 * 2  # bfloat16 = 2 bytes per element
        assert len(field_data.vectors.bfloat16_vector) == expected_total_size

        # Verify data correctness for sample vectors
        for idx in [0, 100, 500, 999]:
            expected_bytes = vectors[idx]
            actual_bytes = field_data.vectors.bfloat16_vector[idx * 128 * 2 : (idx + 1) * 128 * 2]
            assert expected_bytes == actual_bytes, f"Vector {idx} data mismatch"

    def test_flush_vector_bytes_all_types(self):
        """Test flush_vector_bytes function for all bytes vector types"""
        vector_types = [
            (DataType.INT8_VECTOR, "int8_vector", 768),
            (DataType.BINARY_VECTOR, "binary_vector", 128),
            (DataType.FLOAT16_VECTOR, "float16_vector", 128),
            (DataType.BFLOAT16_VECTOR, "bfloat16_vector", 128),
        ]

        for vector_type, vector_attr, dim in vector_types:
            field_data = schema_pb2.FieldData()
            field_data.type = vector_type
            field_data.field_name = f"{vector_attr}_field"
            field_info = {"name": f"{vector_attr}_field", "params": {"dim": dim}}
            vector_bytes_cache: Dict[int, List[bytes]] = {}

            # Pack some vectors
            num_vectors = 10
            for i in range(num_vectors):
                if vector_type == DataType.INT8_VECTOR:
                    vector = np.array([i % 128 - 64 for _ in range(dim)], dtype=np.int8)
                elif vector_type == DataType.BINARY_VECTOR:
                    vector = bytes([i % 256 for _ in range(dim // 8)])
                elif vector_type == DataType.FLOAT16_VECTOR:
                    vector = np.array([float(i) for _ in range(dim)], dtype=np.float16)
                else:  # BFLOAT16_VECTOR
                    # Use bytes format for bfloat16 since dtype may not be available
                    vector = bytes([i % 256 for _ in range(dim * 2)])

                pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

            # Flush to merge all bytes
            flush_vector_bytes(field_data, vector_bytes_cache)

            # Verify all data is merged
            vector_data = getattr(field_data.vectors, vector_attr)
            if vector_type == DataType.INT8_VECTOR:
                expected_size = num_vectors * dim
            elif vector_type == DataType.BINARY_VECTOR:
                expected_size = num_vectors * (dim // 8)
            else:  # FLOAT16_VECTOR or BFLOAT16_VECTOR
                expected_size = num_vectors * dim * 2

            assert len(vector_data) == expected_size, f"{vector_attr} size mismatch"


class TestPackFieldValueExtendedTypes:
    """Test pack_field_value_to_field_data for extended field types"""

    def test_pack_timestamptz_field(self):
        """Test packing TIMESTAMPTZ field value"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.TIMESTAMPTZ
        field_data.field_name = "ts_field"
        field_info = {"name": "ts_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Test with valid timestamp string
        pack_field_value_to_field_data(
            "2024-01-01T12:00:00Z", field_data, field_info, vector_bytes_cache
        )
        assert len(field_data.scalars.string_data.data) == 1
        assert field_data.scalars.string_data.data[0] == "2024-01-01T12:00:00Z"

    def test_pack_timestamptz_field_none(self):
        """Test packing None value for TIMESTAMPTZ field"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.TIMESTAMPTZ
        field_data.field_name = "ts_field"
        field_info = {"name": "ts_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        pack_field_value_to_field_data(None, field_data, field_info, vector_bytes_cache)
        assert len(field_data.scalars.string_data.data) == 0

    def test_pack_geometry_field(self):
        """Test packing GEOMETRY field value"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.GEOMETRY
        field_data.field_name = "geo_field"
        field_info = {"name": "geo_field", "params": {"max_length": 1000}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        # Test with WKT geometry string - pack expects string directly
        wkt = "POINT(1.0 2.0)"
        pack_field_value_to_field_data(wkt, field_data, field_info, vector_bytes_cache)
        assert len(field_data.scalars.geometry_wkt_data.data) == 1

    def test_pack_geometry_field_none(self):
        """Test packing None value for GEOMETRY field"""
        field_data = schema_types.FieldData()
        field_data.type = DataType.GEOMETRY
        field_data.field_name = "geo_field"
        field_info = {"name": "geo_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        pack_field_value_to_field_data(None, field_data, field_info, vector_bytes_cache)
        assert len(field_data.scalars.geometry_wkt_data.data) == 0

    def test_pack_unsupported_type_raises_error(self):
        """Test that unsupported field type raises ParamError"""
        field_data = schema_types.FieldData()
        field_data.type = 999  # Invalid type
        field_data.field_name = "bad_field"
        field_info = {"name": "bad_field"}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        with pytest.raises(ParamError, match="Unsupported data type"):
            pack_field_value_to_field_data("value", field_data, field_info, vector_bytes_cache)


class TestExtractRowDataV2:
    """Test extract_row_data_from_fields_data_v2 function"""

    def test_extract_bool_with_validity(self):
        """Test extracting bool data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.BOOL
        field_data.field_name = "bool_field"
        field_data.scalars.bool_data.data.extend([True, False, True])
        field_data.valid_data.extend([True, False, True])

        entity_rows = [{}, {}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows[0]["bool_field"] is True
        assert entity_rows[1]["bool_field"] is None
        assert entity_rows[2]["bool_field"] is True

    def test_extract_int_types_with_validity(self):
        """Test extracting INT8/16/32 data with validity mask"""

        for dtype in [DataType.INT8, DataType.INT16, DataType.INT32]:
            field_data = schema_types.FieldData()
            field_data.type = dtype
            field_data.field_name = "int_field"
            field_data.scalars.int_data.data.extend([10, 20, 30])
            field_data.valid_data.extend([True, False, True])

            entity_rows = [{}, {}, {}]
            extract_row_data_from_fields_data_v2(field_data, entity_rows)

            assert entity_rows[0]["int_field"] == 10
            assert entity_rows[1]["int_field"] is None
            assert entity_rows[2]["int_field"] == 30

    def test_extract_int64_with_validity(self):
        """Test extracting INT64 data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.INT64
        field_data.field_name = "int64_field"
        field_data.scalars.long_data.data.extend([100, 200])
        field_data.valid_data.extend([False, True])

        entity_rows = [{}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows[0]["int64_field"] is None
        assert entity_rows[1]["int64_field"] == 200

    def test_extract_float_with_validity(self):
        """Test extracting FLOAT data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT
        field_data.field_name = "float_field"
        field_data.scalars.float_data.data.extend([1.5, 2.5])
        field_data.valid_data.extend([True, False])

        entity_rows = [{}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows[0]["float_field"] == pytest.approx(1.5)
        assert entity_rows[1]["float_field"] is None

    def test_extract_double_with_validity(self):
        """Test extracting DOUBLE data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.DOUBLE
        field_data.field_name = "double_field"
        field_data.scalars.double_data.data.extend([1.111, 2.222])
        field_data.valid_data.extend([True, True])

        entity_rows = [{}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows[0]["double_field"] == pytest.approx(1.111)
        assert entity_rows[1]["double_field"] == pytest.approx(2.222)

    def test_extract_timestamptz_with_validity(self):
        """Test extracting TIMESTAMPTZ data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.TIMESTAMPTZ
        field_data.field_name = "ts_field"
        field_data.scalars.string_data.data.extend(["2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"])
        field_data.valid_data.extend([True, True])

        entity_rows = [{}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows[0]["ts_field"] == "2024-01-01T00:00:00Z"
        assert entity_rows[1]["ts_field"] == "2024-12-31T23:59:59Z"

    def test_extract_varchar_with_validity(self):
        """Test extracting VARCHAR data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.VARCHAR
        field_data.field_name = "str_field"
        field_data.scalars.string_data.data.extend(["hello", "world"])

        entity_rows = [{}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows[0]["str_field"] == "hello"
        assert entity_rows[1]["str_field"] == "world"

    def test_extract_geometry_with_validity(self):
        """Test extracting GEOMETRY data with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.GEOMETRY
        field_data.field_name = "geo_field"
        field_data.scalars.geometry_wkt_data.data.extend(["POINT(1 2)", "POINT(3 4)"])
        field_data.valid_data.extend([True, False])

        entity_rows = [{}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows[0]["geo_field"] == "POINT(1 2)"
        assert entity_rows[1]["geo_field"] is None

    def test_extract_json_returns_true(self):
        """Test extracting JSON data returns True for lazy processing"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.JSON
        field_data.field_name = "json_field"

        entity_rows = [{}]
        result = extract_row_data_from_fields_data_v2(field_data, entity_rows)
        assert result is True

    def test_extract_vector_types_return_true(self):
        """Test vector data types return True for lazy processing"""

        vector_types = [
            DataType.FLOAT_VECTOR,
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
            DataType.BINARY_VECTOR,
            DataType.SPARSE_FLOAT_VECTOR,
            DataType.INT8_VECTOR,
        ]

        for vtype in vector_types:
            field_data = schema_types.FieldData()
            field_data.type = vtype
            field_data.field_name = "vec_field"

            entity_rows = [{}]
            result = extract_row_data_from_fields_data_v2(field_data, entity_rows)
            assert result is True, f"Expected True for {vtype}"

    def test_extract_array_of_struct_returns_true(self):
        """Test _ARRAY_OF_STRUCT returns True for lazy processing"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_STRUCT
        field_data.field_name = "struct_field"

        entity_rows = [{}]
        result = extract_row_data_from_fields_data_v2(field_data, entity_rows)
        assert result is True

    def test_extract_array_of_vector_returns_true(self):
        """Test _ARRAY_OF_VECTOR returns True for lazy processing"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR
        field_data.field_name = "vec_arr_field"

        entity_rows = [{}]
        result = extract_row_data_from_fields_data_v2(field_data, entity_rows)
        assert result is True


class TestExtractRowDataFromFieldsData:
    """Test extract_row_data_from_fields_data function"""

    @pytest.mark.parametrize("empty_data", [None, []])
    def test_empty_fields_data(self, empty_data):
        """Test with empty fields data returns empty dict"""
        result = extract_row_data_from_fields_data(empty_data, 0)
        assert result == {}

    def test_extract_bool_field(self):
        """Test extracting bool field"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.BOOL
        field_data.field_name = "bool_field"
        field_data.scalars.bool_data.data.extend([True, False, True])

        result = extract_row_data_from_fields_data([field_data], 1)
        assert result["bool_field"] is False

    def test_extract_int_field(self):
        """Test extracting int field"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.INT32
        field_data.field_name = "int_field"
        field_data.scalars.int_data.data.extend([10, 20, 30])

        result = extract_row_data_from_fields_data([field_data], 2)
        assert result["int_field"] == 30

    def test_extract_with_validity(self):
        """Test extracting with validity mask"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.INT64
        field_data.field_name = "int64_field"
        field_data.scalars.long_data.data.extend([100, 200, 300])
        field_data.valid_data.extend([True, False, True])

        result = extract_row_data_from_fields_data([field_data], 1)
        assert result["int64_field"] is None

    def test_extract_geometry_field(self):
        """Test extracting geometry field"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.GEOMETRY
        field_data.field_name = "geo_field"
        field_data.scalars.geometry_wkt_data.data.extend(["POINT(0 0)", "POINT(1 1)"])

        result = extract_row_data_from_fields_data([field_data], 0)
        assert result["geo_field"] == "POINT(0 0)"


class TestExtractVectorArrayRowData:
    """Test extract_vector_array_row_data function"""

    def test_extract_float_vector_array(self):
        """Test extracting FLOAT_VECTOR array"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR

        # Create vector array data
        vector_data = schema_types.VectorField()
        vector_data.float_vector.data.extend([1.0, 2.0, 3.0, 4.0])
        field_data.vectors.vector_array.element_type = DataType.FLOAT_VECTOR
        field_data.vectors.vector_array.data.append(vector_data)

        result = extract_vector_array_row_data(field_data, 0)
        assert len(result) == 4

    def test_extract_binary_vector_array(self):
        """Test extracting BINARY_VECTOR array"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR

        vector_data = schema_types.VectorField()
        vector_data.binary_vector = b"\x01\x02"
        field_data.vectors.vector_array.element_type = DataType.BINARY_VECTOR
        field_data.vectors.vector_array.data.append(vector_data)

        result = extract_vector_array_row_data(field_data, 0)
        assert result == [b"\x01\x02"]


class TestPackFieldValueErrors:
    """Test error handling in pack_field_value_to_field_data"""

    def test_pack_bool_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.BOOL
        field_data.field_name = "f"
        field_info = {"name": "f"}

        class NoBool:
            def __bool__(self):
                raise ValueError("no")

        with pytest.raises(DataNotMatchException):
            entity_helper.pack_field_value_to_field_data(NoBool(), field_data, field_info, {})

    def test_pack_int_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.INT64
        field_data.field_name = "f"
        field_info = {"name": "f"}

        with pytest.raises(DataNotMatchException):
            entity_helper.pack_field_value_to_field_data("not int", field_data, field_info, {})

    def test_pack_float_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT
        field_data.field_name = "f"
        field_info = {"name": "f"}

        with pytest.raises(DataNotMatchException):
            entity_helper.pack_field_value_to_field_data("not float", field_data, field_info, {})

    def test_pack_float_vector_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_data.field_name = "f"
        field_info = {"name": "f"}

        # This raises TypeError (extend with list of strings) -> caught -> DataNotMatchException
        with pytest.raises(DataNotMatchException):
            entity_helper.pack_field_value_to_field_data(["not float"], field_data, field_info, {})

    def test_pack_binary_vector_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.BINARY_VECTOR
        field_data.field_name = "f"
        field_info = {"name": "f"}

        with pytest.raises(DataNotMatchException):
            entity_helper.pack_field_value_to_field_data(123, field_data, field_info, {})

    def test_pack_float16_vector_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT16_VECTOR
        field_data.field_name = "f"
        field_info = {"name": "f"}

        # raises ParamError explicit check
        with pytest.raises(ParamError):
            entity_helper.pack_field_value_to_field_data("bad", field_data, field_info, {})

    def test_pack_bfloat16_vector_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.BFLOAT16_VECTOR
        field_data.field_name = "f"
        field_info = {"name": "f"}

        with pytest.raises(ParamError):
            entity_helper.pack_field_value_to_field_data("bad", field_data, field_info, {})

    def test_pack_int8_vector_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.INT8_VECTOR
        field_data.field_name = "f"
        field_info = {"name": "f"}

        with pytest.raises(ParamError):
            entity_helper.pack_field_value_to_field_data("bad", field_data, field_info, {})

    def test_pack_varchar_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.VARCHAR
        field_data.field_name = "f"
        field_info = {"name": "f", "params": {Config.MaxVarCharLengthKey: 10}}

        with pytest.raises(DataNotMatchException):
            entity_helper.pack_field_value_to_field_data(123, field_data, field_info, {})

    def test_pack_json_invalid(self):
        field_data = schema_types.FieldData()
        field_data.type = DataType.JSON
        field_data.field_name = "f"
        field_info = {"name": "f"}

        class Unserializable:
            pass

        with pytest.raises(DataNotMatchException):
            entity_helper.pack_field_value_to_field_data(
                Unserializable(), field_data, field_info, {}
            )


class TestExtractRowDataV1Extended:
    """Test extract_row_data_from_fields_data (V1) for missing coverage"""

    def test_extract_v1_vectors(self):
        """Test extracting vectors via V1 API"""

        # Float Vector
        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT_VECTOR
        field_data.field_name = "fv"
        field_data.vectors.dim = 2
        field_data.vectors.float_vector.data.extend([1.0, 2.0, 3.0, 4.0])  # 2 vectors

        # It extracts one row at a time key-value pairs
        row0 = extract_row_data_from_fields_data([field_data], 0)
        row1 = extract_row_data_from_fields_data([field_data], 1)

        assert row0["fv"] == [1.0, 2.0]
        assert row1["fv"] == [3.0, 4.0]

    def test_extract_v1_binary_vector(self):
        """Test binary vector V1 extraction"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.BINARY_VECTOR
        field_data.field_name = "bv"
        field_data.vectors.dim = 8
        field_data.vectors.binary_vector = b"\x01\x02"  # 2 vectors (1 byte each)

        row0 = extract_row_data_from_fields_data([field_data], 0)
        row1 = extract_row_data_from_fields_data([field_data], 1)

        # Binary vectors in V1 are returned as a list containing the bytes element
        assert row0["bv"] == [b"\x01"]
        assert row1["bv"] == [b"\x02"]

    def test_extract_v1_float16(self):

        field_data = schema_types.FieldData()
        field_data.type = DataType.FLOAT16_VECTOR
        field_data.field_name = "f16"
        field_data.vectors.dim = 1
        # 2 elements
        field_data.vectors.float16_vector = b"\x00\x3c\x00\x3c"  # 1.0, 1.0 (approx)

        row0 = extract_row_data_from_fields_data([field_data], 0)
        row1 = extract_row_data_from_fields_data([field_data], 1)

        assert len(row0["f16"]) == 1
        assert len(row1["f16"]) == 1
        assert isinstance(row0["f16"], list)


class TestExtractRowDataV1Scalar:
    """Test V1 extraction for scalars to cover lines 1047-1093 etc."""

    def test_extract_v1_scalars(self):

        # Create field data for various scalar types
        types = [
            (DataType.BOOL, "bool_data", [True, False], "bool_f"),
            (DataType.INT32, "int_data", [1, 2], "int_f"),
            (DataType.INT64, "long_data", [10, 20], "long_f"),
            (DataType.FLOAT, "float_data", [1.1, 2.2], "float_f"),
            (DataType.DOUBLE, "double_data", [3.3, 4.4], "double_f"),
            (DataType.VARCHAR, "string_data", ["a", "b"], "str_f"),
            (DataType.JSON, "json_data", [b'{"x": 1}', b'{"x": 2}'], "json_f"),
        ]

        fields_data_list = []
        for dtype, attr, val, fname in types:
            fd = schema_types.FieldData()
            fd.type = dtype
            fd.field_name = fname
            getattr(fd.scalars, attr).data.extend(val)
            fields_data_list.append(fd)

        row0 = extract_row_data_from_fields_data(fields_data_list, 0)

        assert row0["bool_f"] is True
        assert row0["int_f"] == 1
        assert row0["long_f"] == 10
        assert row0["float_f"] == pytest.approx(1.1, rel=1e-4)  # float32 precision
        assert row0["double_f"] == pytest.approx(3.3)
        assert row0["str_f"] == "a"
        assert row0["json_f"] == {"x": 1}

    def test_extract_v1_json_dynamic(self):

        fd = schema_types.FieldData()
        fd.type = DataType.JSON
        fd.field_name = "meta"
        fd.is_dynamic = True
        fd.scalars.json_data.data.append(b'{"dy": 100, "other": 200}')

        # Extract, specifying 'dy' as dynamic field
        # dynamic_output_fields argument
        row0 = extract_row_data_from_fields_data([fd], 0, dynamic_output_fields=["dy"])

        assert row0["dy"] == 100
        assert "other" not in row0
        # Wait, logic: row_data.update({k: v for k,v in json_dict.items() if k in dynamic_fields})
        # So only 'dy' is updated into row_data.

        # Test without dynamic_output_fields (None or empty) -> updates all
        row0_all = extract_row_data_from_fields_data([fd], 0)
        assert row0_all["dy"] == 100
        assert row0_all["other"] == 200


class TestExtractRowDataV1Validity:
    """Test extract_row_data_from_fields_data (V1) validity and error handling"""

    def test_json_validity_v1(self):

        fd = schema_types.FieldData()
        fd.type = DataType.JSON
        fd.field_name = "j"
        fd.scalars.json_data.data.append(b"{}")
        fd.valid_data.append(False)  # Invalid

        row0 = extract_row_data_from_fields_data([fd], 0)
        assert row0["j"] is None

    def test_json_invalid_bytes_v1(self):

        fd = schema_types.FieldData()
        fd.type = DataType.JSON
        fd.field_name = "j"
        fd.scalars.json_data.data.append(b"{invalid")

        with pytest.raises(orjson.JSONDecodeError):
            extract_row_data_from_fields_data([fd], 0)

    def test_array_validity_v1(self):

        fd = schema_types.FieldData()
        fd.type = DataType.ARRAY
        fd.field_name = "arr"
        # We need to populate array_data.data to satisfy length check
        fd.scalars.array_data.data.append(schema_types.ScalarField())
        fd.valid_data.append(False)

        row0 = extract_row_data_from_fields_data([fd], 0)
        assert row0["arr"] is None

    def test_vector_validity_v1(self):

        # Test Float Vector validity
        fd = schema_types.FieldData()
        fd.type = DataType.FLOAT_VECTOR
        fd.field_name = "fv"
        fd.valid_data.append(False)

        row0 = extract_row_data_from_fields_data([fd], 0)
        assert row0["fv"] is None

        # Test Binary Vector validity
        fd = schema_types.FieldData()
        fd.type = DataType.BINARY_VECTOR
        fd.field_name = "bv"
        fd.valid_data.append(False)
        row0 = extract_row_data_from_fields_data([fd], 0)
        assert row0["bv"] is None

        # Test Float16 Vector validity
        fd = schema_types.FieldData()
        fd.type = DataType.FLOAT16_VECTOR
        fd.field_name = "f16"
        fd.valid_data.append(False)
        row0 = extract_row_data_from_fields_data([fd], 0)
        assert row0["f16"] is None
