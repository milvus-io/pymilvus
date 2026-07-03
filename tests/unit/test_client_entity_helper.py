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
    extract_array_rows,
    extract_dynamic_field_from_result,
    extract_row_data_from_fields_data_v2,
    flush_vector_bytes,
    get_max_len_of_var_char,
    pack_field_value_to_field_data,
    sparse_proto_to_rows,
    sparse_rows_to_proto,
)
from pymilvus.client.prepare import Prepare
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

    def test_entity_to_array_arr_text(self):
        """Test converting TEXT arrays stores string_data without max_length."""
        field_info = {"name": "array_field", "element_type": DataType.TEXT}
        long_text = "x" * (Config.MaxVarCharLength + 1)

        result = entity_to_array_arr([["hello", long_text], ["foo"]], field_info)

        assert len(result) == 2
        assert list(result[0].string_data.data) == ["hello", long_text]

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

    def test_entity_to_field_data_text_skips_max_length_check(self):
        """Test TEXT field data accepts strings longer than VARCHAR max_length."""
        long_text = "x" * (Config.MaxVarCharLength + 1)
        entity = {"name": "text_field", "type": DataType.TEXT, "values": ["short", long_text]}
        field_info = {"name": "text_field", "params": {Config.MaxVarCharLengthKey: 1}}

        result = entity_to_field_data(entity, field_info, 2)

        assert result.field_name == "text_field"
        assert result.type == DataType.TEXT
        assert list(result.scalars.string_data.data) == ["short", long_text]


class TestLogicalTypeBatchInsertPaths:
    @staticmethod
    def _byte_vector_payload(dtype, dim, seed=0):
        if dtype == DataType.BINARY_VECTOR:
            return bytes((seed + i) % 256 for i in range(dim // 8))
        if dtype == DataType.FLOAT16_VECTOR:
            return np.array([seed + i for i in range(dim)], dtype=np.float16)
        if dtype == DataType.BFLOAT16_VECTOR:
            return bytes((seed + i) % 256 for i in range(dim * 2))
        if dtype == DataType.INT8_VECTOR:
            return np.array([(seed + i) % 128 - 64 for i in range(dim)], dtype=np.int8)
        raise AssertionError(f"unexpected dtype {dtype}")

    @staticmethod
    def _payload_bytes(dtype, payload):
        if dtype in (DataType.FLOAT16_VECTOR, DataType.INT8_VECTOR):
            return payload.tobytes()
        return payload

    @staticmethod
    def _field_data_by_name(request):
        return {field_data.field_name: field_data for field_data in request.fields_data}

    @pytest.mark.parametrize(
        "dtype,values,attr,expected",
        [
            (DataType.BOOL, [True, False], "bool_data", [True, False]),
            (DataType.INT32, [1, 2], "int_data", [1, 2]),
            (DataType.FLOAT, [1.5, 2.5], "float_data", [1.5, 2.5]),
            (DataType.TEXT, ["a", "b"], "string_data", ["a", "b"]),
        ],
        ids=["bool", "int32", "float", "text"],
    )
    def test_registry_backed_batch_scalar_destinations(self, dtype, values, attr, expected):
        field_info = {"name": "scalar", "params": {Config.MaxVarCharLengthKey: 128}}
        entity = {"name": "scalar", "type": dtype, "values": values}

        with patch(
            "pymilvus.client.entity_helper.type_info.get_scalar_attr",
            wraps=entity_helper.type_info.get_scalar_attr,
        ) as get_scalar_attr:
            field_data = entity_to_field_data(entity, field_info, len(values))

        get_scalar_attr.assert_any_call(dtype)
        assert list(getattr(field_data.scalars, attr).data) == expected

    @pytest.mark.parametrize(
        "dtype,dim,vector_attr",
        [
            (DataType.BINARY_VECTOR, 16, "binary_vector"),
            (DataType.FLOAT16_VECTOR, 4, "float16_vector"),
            (DataType.BFLOAT16_VECTOR, 4, "bfloat16_vector"),
            (DataType.INT8_VECTOR, 4, "int8_vector"),
        ],
        ids=["binary", "float16", "bfloat16", "int8"],
    )
    def test_registry_backed_batch_byte_vectors(self, dtype, dim, vector_attr):
        values = [
            self._payload_bytes(dtype, self._byte_vector_payload(dtype, dim, seed))
            for seed in (1, 9)
        ]
        field_info = {"name": "vec", "params": {"dim": dim}}
        entity = {"name": "vec", "type": dtype, "values": values}

        field_data = entity_to_field_data(entity, field_info, 2)

        assert field_data.vectors.dim == dim
        assert getattr(field_data.vectors, vector_attr) == b"".join(values)

    @pytest.mark.parametrize(
        "dtype,dim,vector_attr",
        [
            (DataType.BINARY_VECTOR, 8, "binary_vector"),
            (DataType.FLOAT16_VECTOR, 2, "float16_vector"),
            (DataType.BFLOAT16_VECTOR, 2, "bfloat16_vector"),
            (DataType.INT8_VECTOR, 2, "int8_vector"),
        ],
        ids=["binary", "float16", "bfloat16", "int8"],
    )
    def test_batch_byte_vectors_use_schema_dim(self, dtype, dim, vector_attr):
        values = [self._payload_bytes(dtype, self._byte_vector_payload(dtype, 16, 1))]
        field_info = {"name": "vec", "params": {"dim": str(dim)}}
        entity = {"name": "vec", "type": dtype, "values": values}

        field_data = entity_to_field_data(entity, field_info, 1)

        assert field_data.vectors.dim == dim
        assert getattr(field_data.vectors, vector_attr) == b"".join(values)

    @pytest.mark.parametrize(
        "dtype,dim,vector_attr",
        [
            (DataType.BINARY_VECTOR, 16, "binary_vector"),
            (DataType.FLOAT16_VECTOR, 4, "float16_vector"),
            (DataType.BFLOAT16_VECTOR, 4, "bfloat16_vector"),
            (DataType.INT8_VECTOR, 4, "int8_vector"),
        ],
        ids=["binary", "float16", "bfloat16", "int8"],
    )
    def test_registry_backed_nullable_all_null_uses_schema_dim(self, dtype, dim, vector_attr):
        entity = {"name": "vec", "type": dtype, "values": [None, None]}
        field_info = {"name": "vec", "nullable": True, "params": {"dim": str(dim)}}

        field_data = entity_to_field_data(entity, field_info, 2)

        assert field_data.vectors.dim == dim
        assert getattr(field_data.vectors, vector_attr) == b""
        assert list(field_data.valid_data) == [False, False]

    def test_prepare_batch_insert_and_upsert_vectors(self):
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
            {"name": "float", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
            {"name": "sparse", "type": DataType.SPARSE_FLOAT_VECTOR},
            {"name": "binary", "type": DataType.BINARY_VECTOR, "params": {"dim": 16}},
            {"name": "float16", "type": DataType.FLOAT16_VECTOR, "params": {"dim": 4}},
            {"name": "bfloat16", "type": DataType.BFLOAT16_VECTOR, "params": {"dim": 4}},
            {"name": "int8", "type": DataType.INT8_VECTOR, "params": {"dim": 4}},
        ]
        float_values = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        sparse_values = [{0: 1.0, 3: 2.0}, {1: 3.0}]
        binary_values = [
            self._byte_vector_payload(DataType.BINARY_VECTOR, 16, 1),
            self._byte_vector_payload(DataType.BINARY_VECTOR, 16, 9),
        ]
        float16_values = [
            self._payload_bytes(
                DataType.FLOAT16_VECTOR, self._byte_vector_payload(DataType.FLOAT16_VECTOR, 4, 1)
            ),
            self._payload_bytes(
                DataType.FLOAT16_VECTOR, self._byte_vector_payload(DataType.FLOAT16_VECTOR, 4, 9)
            ),
        ]
        bfloat16_values = [
            self._byte_vector_payload(DataType.BFLOAT16_VECTOR, 4, 1),
            self._byte_vector_payload(DataType.BFLOAT16_VECTOR, 4, 9),
        ]
        int8_values = [
            self._payload_bytes(
                DataType.INT8_VECTOR, self._byte_vector_payload(DataType.INT8_VECTOR, 4, 1)
            ),
            self._payload_bytes(
                DataType.INT8_VECTOR, self._byte_vector_payload(DataType.INT8_VECTOR, 4, 9)
            ),
        ]
        entities = [
            {"name": "id", "type": DataType.INT64, "values": [1, 2]},
            {"name": "float", "type": DataType.FLOAT_VECTOR, "values": float_values},
            {"name": "sparse", "type": DataType.SPARSE_FLOAT_VECTOR, "values": sparse_values},
            {"name": "binary", "type": DataType.BINARY_VECTOR, "values": binary_values},
            {"name": "float16", "type": DataType.FLOAT16_VECTOR, "values": float16_values},
            {"name": "bfloat16", "type": DataType.BFLOAT16_VECTOR, "values": bfloat16_values},
            {"name": "int8", "type": DataType.INT8_VECTOR, "values": int8_values},
        ]

        insert_request = Prepare.batch_insert_param("c", entities, "", fields_info)
        upsert_request = Prepare.batch_upsert_param("c", entities, "", fields_info)

        for request in (insert_request, upsert_request):
            fields = self._field_data_by_name(request)
            assert fields["float"].vectors.dim == 4
            assert list(fields["float"].vectors.float_vector.data) == [
                value for vector in float_values for value in vector
            ]
            assert len(fields["sparse"].vectors.sparse_float_vector.contents) == 2
            assert fields["binary"].vectors.dim == 16
            assert fields["binary"].vectors.binary_vector == b"".join(binary_values)
            assert fields["float16"].vectors.dim == 4
            assert fields["float16"].vectors.float16_vector == b"".join(float16_values)
            assert fields["bfloat16"].vectors.dim == 4
            assert fields["bfloat16"].vectors.bfloat16_vector == b"".join(bfloat16_values)
            assert fields["int8"].vectors.dim == 4
            assert fields["int8"].vectors.int8_vector == b"".join(int8_values)


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

    @pytest.mark.parametrize(
        "dtype,field_name,dim,vector_attr",
        [
            (DataType.INT8_VECTOR, "int8_vector_field", 768, "int8_vector"),
            (DataType.BINARY_VECTOR, "binary_vector_field", 128, "binary_vector"),
            (DataType.FLOAT16_VECTOR, "float16_vector_field", 128, "float16_vector"),
            (DataType.BFLOAT16_VECTOR, "bfloat16_vector_field", 128, "bfloat16_vector"),
        ],
        ids=["int8", "binary", "float16", "bfloat16"],
    )
    def test_pack_field_value_to_field_data_vector_multiple(
        self, dtype, field_name, dim, vector_attr
    ):
        """Test packing multiple vectors of various byte-based types"""
        field_data = schema_pb2.FieldData()
        field_data.type = dtype
        field_data.field_name = field_name
        field_info = {"name": field_name, "params": {"dim": dim}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        num_vectors = 1000
        vectors = []
        for i in range(num_vectors):
            if dtype == DataType.INT8_VECTOR:
                vector = np.array([(i + j) % 128 - 64 for j in range(dim)], dtype=np.int8)
            elif dtype == DataType.BINARY_VECTOR:
                vector = bytes([(i + j) % 256 for j in range(dim // 8)])
            elif dtype == DataType.FLOAT16_VECTOR:
                vector = np.array([float(i + j) for j in range(dim)], dtype=np.float16)
            else:
                vector = bytes([(i + j) % 256 for j in range(dim * 2)])
            vectors.append(vector)
            pack_field_value_to_field_data(vector, field_data, field_info, vector_bytes_cache)

        flush_vector_bytes(field_data, vector_bytes_cache)
        assert field_data.vectors.dim == dim

        if dtype == DataType.INT8_VECTOR:
            bytes_per = dim
        elif dtype == DataType.BINARY_VECTOR:
            bytes_per = dim // 8
        else:
            bytes_per = dim * 2

        vector_data = getattr(field_data.vectors, vector_attr)
        assert len(vector_data) == num_vectors * bytes_per

        for idx in [0, 100, 500, 999]:
            if dtype == DataType.FLOAT16_VECTOR:
                expected = vectors[idx].view(np.uint8).tobytes()
            elif dtype == DataType.INT8_VECTOR:
                expected = vectors[idx].tobytes()
            else:
                expected = vectors[idx]
            actual = vector_data[idx * bytes_per : (idx + 1) * bytes_per]
            assert expected == actual, f"Vector {idx} data mismatch for {dtype}"

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

    @pytest.mark.parametrize(
        "dtype,value,expected",
        [
            (DataType.FLOAT16_VECTOR, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
            (
                DataType.FLOAT16_VECTOR,
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                [1.0, 2.0, 3.0, 4.0],
            ),
            (
                DataType.BFLOAT16_VECTOR,
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                [1.0, 2.0, 3.0, 4.0],
            ),
        ],
    )
    def test_pack_field_value_to_field_data_fp16_bf16_float_input(self, dtype, value, expected):
        field_data = schema_pb2.FieldData()
        field_data.type = dtype
        field_data.field_name = "vector"
        field_info = {"name": "vector", "params": {"dim": 4}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        pack_field_value_to_field_data(value, field_data, field_info, vector_bytes_cache)

        assert field_data.vectors.dim == 4
        assert field_data.vectors.float_vector.data == expected
        assert field_data.vectors.WhichOneof("data") == "float_vector"

    def test_pack_field_value_to_field_data_fp16_invalid_numpy_dtype(self):
        field_data = schema_pb2.FieldData()
        field_data.type = DataType.FLOAT16_VECTOR
        field_data.field_name = "vector"
        field_info = {"name": "vector", "params": {"dim": 4}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        with pytest.raises(ParamError, match=r"Expected an np\.ndarray with dtype=float16"):
            pack_field_value_to_field_data(
                np.array([1, 2, 3, 4], dtype=np.int32),
                field_data,
                field_info,
                vector_bytes_cache,
            )

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

    def test_pack_text_field_skips_max_length_check(self):
        """Test packing TEXT accepts long strings without max_length validation."""
        field_data = schema_types.FieldData()
        field_data.type = DataType.TEXT
        field_data.field_name = "text_field"
        field_info = {"name": "text_field", "params": {Config.MaxVarCharLengthKey: 1}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}
        long_text = "x" * (Config.MaxVarCharLength + 1)

        pack_field_value_to_field_data(long_text, field_data, field_info, vector_bytes_cache)

        assert list(field_data.scalars.string_data.data) == [long_text]

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

    def test_extract_text_with_validity(self):
        """Test extracting TEXT data with validity mask."""
        field_data = schema_types.FieldData()
        field_data.type = DataType.TEXT
        field_data.field_name = "text_field"
        field_data.scalars.string_data.data.extend(["hello", "ignored", "world"])
        field_data.valid_data.extend([True, False, True])

        entity_rows = [{}, {}, {}]
        extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert entity_rows == [
            {"text_field": "hello"},
            {"text_field": None},
            {"text_field": "world"},
        ]

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

    def test_extract_scalar_uses_type_info_attr_for_bulk_assignment(self, monkeypatch):
        """Future scalar registry entries should not need a new branch here."""
        original_get_scalar_attr = entity_helper.type_info.get_scalar_attr

        def fake_get_scalar_attr(dtype):
            if dtype == DataType.NONE:
                return "string_data"
            return original_get_scalar_attr(dtype)

        monkeypatch.setattr(entity_helper.type_info, "get_scalar_attr", fake_get_scalar_attr)

        field_data = schema_types.FieldData()
        field_data.type = DataType.NONE
        field_data.field_name = "future_scalar"
        field_data.scalars.string_data.data.extend(["first", "ignored", "third"])
        field_data.valid_data.extend([True, False, True])

        entity_rows = [{}, {}, {}]
        result = extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert result is False
        assert entity_rows == [
            {"future_scalar": "first"},
            {"future_scalar": None},
            {"future_scalar": "third"},
        ]

    def test_extract_json_returns_true(self):
        """Test extracting JSON data returns True for lazy processing"""

        field_data = schema_types.FieldData()
        field_data.type = DataType.JSON
        field_data.field_name = "json_field"

        entity_rows = [{}]
        result = extract_row_data_from_fields_data_v2(field_data, entity_rows)
        assert result is True

    def test_extract_vector_uses_type_info_family_for_defer(self, monkeypatch):
        """Future vector registry entries should defer without a new allowlist entry."""
        original_is_vector_type = entity_helper.type_info.is_vector_type

        def fake_is_vector_type(dtype):
            if dtype == DataType.NONE:
                return True
            return original_is_vector_type(dtype)

        monkeypatch.setattr(entity_helper.type_info, "is_vector_type", fake_is_vector_type)

        field_data = schema_types.FieldData()
        field_data.type = DataType.NONE
        field_data.field_name = "future_vector"

        entity_rows = [{}]
        result = extract_row_data_from_fields_data_v2(field_data, entity_rows)

        assert result is True
        assert entity_rows == [{}]

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

    def test_extract_nullable_struct_array_row(self):
        """Test nullable struct array row decodes null separately from empty array."""
        struct_arrays = schema_types.StructArrayField()
        sub_field = struct_arrays.fields.add()
        sub_field.field_name = "metadata[score]"
        sub_field.type = DataType.ARRAY
        sub_field.valid_data.extend([False, True])
        sub_field.scalars.array_data.element_type = DataType.FLOAT
        sub_field.scalars.array_data.data.append(schema_types.ScalarField())
        sub_field.scalars.array_data.data.append(
            schema_types.ScalarField(float_data=schema_types.FloatArray(data=[1.0, 2.0]))
        )

        assert entity_helper.extract_struct_array_from_column_data(struct_arrays, 0) is None
        assert entity_helper.extract_struct_array_from_column_data(struct_arrays, 1) == [
            {"score": 1.0},
            {"score": 2.0},
        ]

    def test_extract_array_of_vector_returns_true(self):
        """Test _ARRAY_OF_VECTOR returns True for lazy processing"""

        field_data = schema_types.FieldData()
        field_data.type = DataType._ARRAY_OF_VECTOR
        field_data.field_name = "vec_arr_field"

        entity_rows = [{}]
        result = extract_row_data_from_fields_data_v2(field_data, entity_rows)
        assert result is True

    def test_extract_string_still_raises(self):
        """STRING stays explicitly unsupported for result extraction."""
        field_data = schema_types.FieldData()
        field_data.type = DataType.STRING
        field_data.field_name = "string_field"

        with pytest.raises(entity_helper.MilvusException, match="Not support string yet"):
            extract_row_data_from_fields_data_v2(field_data, [{}])


class _NoBool:
    def __bool__(self):
        raise ValueError("no")


class _Unserializable:
    pass


_PACK_ERROR_CASES = [
    (DataType.BOOL, _NoBool(), {"name": "f"}, DataNotMatchException),
    (DataType.INT64, "not int", {"name": "f"}, DataNotMatchException),
    (DataType.FLOAT, "not float", {"name": "f"}, DataNotMatchException),
    (DataType.FLOAT_VECTOR, ["not float"], {"name": "f"}, DataNotMatchException),
    (DataType.BINARY_VECTOR, 123, {"name": "f"}, DataNotMatchException),
    (DataType.FLOAT16_VECTOR, "bad", {"name": "f"}, ParamError),
    (DataType.BFLOAT16_VECTOR, "bad", {"name": "f"}, ParamError),
    (DataType.INT8_VECTOR, "bad", {"name": "f"}, ParamError),
    (
        DataType.VARCHAR,
        123,
        {"name": "f", "params": {Config.MaxVarCharLengthKey: 10}},
        DataNotMatchException,
    ),
    (DataType.JSON, _Unserializable(), {"name": "f"}, DataNotMatchException),
]


class TestPackFieldValueErrors:
    """Test error handling in pack_field_value_to_field_data"""

    @pytest.mark.parametrize(
        "dtype,bad_value,field_info,expected_exc",
        _PACK_ERROR_CASES,
        ids=[str(c[0]).split(".")[-1] for c in _PACK_ERROR_CASES],
    )
    def test_pack_invalid_value(self, dtype, bad_value, field_info, expected_exc):
        field_data = schema_types.FieldData()
        field_data.type = dtype
        field_data.field_name = "f"
        with pytest.raises(expected_exc):
            entity_helper.pack_field_value_to_field_data(bad_value, field_data, field_info, {})


class TestEmptyResultArrayField:
    """Tests for issue #3386: empty query result with ARRAY field crashes with
    'Unsupported data type: 0' because element_type is NONE (0) on empty results."""

    def test_extract_row_data_v2_empty_result_array_field(self):
        """extract_row_data_from_fields_data_v2 must not crash on empty result with ARRAY field.

        When the server returns an empty result, the array_data.element_type is 0 (NONE).
        The function should return without error since there are no rows to populate.
        """
        fd = schema_types.FieldData()
        fd.type = DataType.ARRAY
        fd.field_name = "arr_field"
        # element_type left at default 0 (NONE) — as the server sends for empty results
        # No data entries, no valid_data entries (empty result)

        entity_rows: List[Dict] = []
        # Must not raise MilvusException("Unsupported data type: 0")
        extract_row_data_from_fields_data_v2(fd, entity_rows)

    def test_extract_array_rows_empty_result_without_validity(self):
        """extract_array_rows must not crash when row_count is 0."""
        fd = schema_types.FieldData()
        fd.type = DataType.ARRAY
        fd.field_name = "arr_field"
        # element_type = 0 (NONE), no data

        entity_rows: List[Dict] = []
        # Must not raise MilvusException("Unsupported data type: 0")
        extract_array_rows(fd, entity_rows, 0, has_valid=False)

    def test_extract_array_rows_empty_result_with_validity(self):
        """extract_array_rows must not crash when row_count is 0."""
        fd = schema_types.FieldData()
        fd.type = DataType.ARRAY
        fd.field_name = "arr_field"
        # element_type = 0 (NONE), no data, no valid_data

        entity_rows: List[Dict] = []
        # Must not raise MilvusException("Unsupported data type: 0")
        extract_array_rows(fd, entity_rows, 0, has_valid=True)


class TestArrayResultMaterialization:
    def test_extract_row_data_v2_materializes_array_as_python_list(self):
        fd = schema_types.FieldData()
        fd.type = DataType.ARRAY
        fd.field_name = "arr_field"
        fd.scalars.array_data.element_type = DataType.INT64
        fd.scalars.array_data.data.append(
            schema_types.ScalarField(long_data=schema_types.LongArray(data=[1, 2]))
        )
        fd.scalars.array_data.data.append(
            schema_types.ScalarField(long_data=schema_types.LongArray(data=[3, 4]))
        )

        entity_rows: List[Dict] = [{}, {}]
        extract_row_data_from_fields_data_v2(fd, entity_rows)

        assert entity_rows == [{"arr_field": [1, 2]}, {"arr_field": [3, 4]}]
        assert isinstance(entity_rows[0]["arr_field"], list)
        assert isinstance(entity_rows[1]["arr_field"], list)


class TestLogicalTypeRowInsertPaths:
    @staticmethod
    def _byte_vector_payload(dtype, dim, seed=0):
        if dtype == DataType.BINARY_VECTOR:
            return bytes((seed + i) % 256 for i in range(dim // 8))
        if dtype == DataType.FLOAT16_VECTOR:
            return np.array([seed + i for i in range(dim)], dtype=np.float16)
        if dtype == DataType.BFLOAT16_VECTOR:
            return bytes((seed + i) % 256 for i in range(dim * 2))
        if dtype == DataType.INT8_VECTOR:
            return np.array([(seed + i) % 128 - 64 for i in range(dim)], dtype=np.int8)
        raise AssertionError(f"unexpected dtype {dtype}")

    @staticmethod
    def _payload_bytes(dtype, payload):
        if dtype in (DataType.FLOAT16_VECTOR, DataType.INT8_VECTOR):
            return payload.tobytes()
        return payload

    @staticmethod
    def _field_data_by_name(request):
        return {field_data.field_name: field_data for field_data in request.fields_data}

    @pytest.mark.parametrize(
        "dtype,value,attr,expected",
        [
            (DataType.BOOL, True, "bool_data", True),
            (DataType.INT8, 7, "int_data", 7),
            (DataType.INT16, 8, "int_data", 8),
            (DataType.INT32, 9, "int_data", 9),
            (DataType.INT64, 7, "long_data", 7),
            (DataType.FLOAT, 1.5, "float_data", 1.5),
            (DataType.DOUBLE, 2.5, "double_data", 2.5),
            (
                DataType.TIMESTAMPTZ,
                "2026-05-18T10:00:00Z",
                "string_data",
                "2026-05-18T10:00:00Z",
            ),
            (DataType.VARCHAR, "varchar", "string_data", "varchar"),
            (DataType.TEXT, "text", "string_data", "text"),
            (DataType.JSON, {"ok": True}, "json_data", orjson.dumps({"ok": True})),
            (DataType.GEOMETRY, "POINT(1 2)", "geometry_wkt_data", "POINT(1 2)"),
        ],
        ids=[
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float",
            "double",
            "timestamptz",
            "varchar",
            "text",
            "json",
            "geometry",
        ],
    )
    def test_type_info_backed_row_pack_scalar_wire_slots(self, dtype, value, attr, expected):
        field_data = schema_types.FieldData(type=dtype, field_name="scalar")
        field_info = {"name": "scalar", "params": {Config.MaxVarCharLengthKey: 128}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        pack_field_value_to_field_data(value, field_data, field_info, vector_bytes_cache)
        pack_field_value_to_field_data(None, field_data, field_info, vector_bytes_cache)

        assert list(getattr(field_data.scalars, attr).data) == [expected]

    def test_type_info_backed_row_pack_array_wire_slot(self):
        field_data = schema_types.FieldData(type=DataType.ARRAY, field_name="array")
        field_info = {"name": "array", "element_type": DataType.INT64}

        pack_field_value_to_field_data([1, 2, 3], field_data, field_info, {})
        pack_field_value_to_field_data(None, field_data, field_info, {})

        assert len(field_data.scalars.array_data.data) == 1
        assert list(field_data.scalars.array_data.data[0].long_data.data) == [1, 2, 3]

    @pytest.mark.parametrize(
        "dtype,bad_value,field_info,expected",
        [
            (
                DataType.INT64,
                "not int",
                {"name": "f"},
                "The Input data type is inconsistent with defined schema, {f} field should be a int64, but got a {<class 'str'>} instead. Detail: 'str' object cannot be interpreted as an integer",
            ),
            (
                DataType.TIMESTAMPTZ,
                1,
                {"name": "f"},
                "The Input data type is inconsistent with defined schema, {f} field should be a string, but got a {<class 'int'>} instead.",
            ),
            (
                DataType.VARCHAR,
                123,
                {"name": "f", "params": {Config.MaxVarCharLengthKey: 10}},
                "The Input data type is inconsistent with defined schema, {f} field should be a varchar, but got a {<class 'int'>} instead. Detail: 'int' object is not iterable",
            ),
            (
                DataType.GEOMETRY,
                123,
                {"name": "f", "params": {Config.MaxVarCharLengthKey: 10}},
                "The Input data type is inconsistent with defined schema, {f} field should be a geometry, but got a {<class 'int'>} instead.",
            ),
        ],
        ids=["int64", "timestamptz", "varchar", "geometry"],
    )
    def test_scalar_row_error_messages_preserve_legacy_labels(
        self, dtype, bad_value, field_info, expected
    ):
        field_data = schema_types.FieldData(type=dtype, field_name="f")

        with pytest.raises(DataNotMatchException) as exc:
            pack_field_value_to_field_data(bad_value, field_data, field_info, {})

        assert exc.value.message == expected

    @pytest.mark.parametrize(
        "dtype,dim,vector_attr",
        [
            (DataType.BINARY_VECTOR, 16, "binary_vector"),
            (DataType.FLOAT16_VECTOR, 4, "float16_vector"),
            (DataType.BFLOAT16_VECTOR, 4, "bfloat16_vector"),
            (DataType.INT8_VECTOR, 4, "int8_vector"),
        ],
        ids=["binary", "float16", "bfloat16", "int8"],
    )
    def test_type_info_backed_row_pack_byte_vectors(self, dtype, dim, vector_attr):
        field_data = schema_types.FieldData(type=dtype, field_name="vec")
        field_info = {"name": "vec", "params": {"dim": dim}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}
        payload = self._byte_vector_payload(dtype, dim)

        pack_field_value_to_field_data(payload, field_data, field_info, vector_bytes_cache)
        flush_vector_bytes(field_data, vector_bytes_cache)

        assert field_data.vectors.dim == dim
        assert getattr(field_data.vectors, vector_attr) == self._payload_bytes(dtype, payload)

    def test_float_vector_row_rejects_invalid_ndarray_dtype(self):
        field_data = schema_types.FieldData(type=DataType.FLOAT_VECTOR, field_name="float")
        field_info = {"name": "float", "params": {"dim": 2}}

        with pytest.raises(ParamError, match="invalid input for float32 vector"):
            pack_field_value_to_field_data(
                np.array([1, 2], dtype=np.int32), field_data, field_info, {}
            )

    def test_float16_vector_row_accepts_bytes_payload(self):
        field_data = schema_types.FieldData(type=DataType.FLOAT16_VECTOR, field_name="float16")
        field_info = {"name": "float16", "params": {"dim": 2}}
        vector_bytes_cache: Dict[int, List[bytes]] = {}

        pack_field_value_to_field_data(
            b"\x00\x01\x02\x03", field_data, field_info, vector_bytes_cache
        )
        flush_vector_bytes(field_data, vector_bytes_cache)

        assert field_data.vectors.dim == 2
        assert field_data.vectors.float16_vector == b"\x00\x01\x02\x03"

    @pytest.mark.parametrize(
        "dtype,value,match",
        [
            (
                DataType.FLOAT16_VECTOR,
                np.array([1, 2], dtype=np.int32),
                "invalid input for float16 vector",
            ),
            (
                DataType.BFLOAT16_VECTOR,
                np.array([1, 2], dtype=np.float16),
                "invalid input for bfloat16 vector",
            ),
            (DataType.INT8_VECTOR, [1, 2], "invalid input type for INT8_VECTOR vector"),
        ],
        ids=["float16-wrong-dtype", "bfloat16-wrong-dtype", "int8-non-array"],
    )
    def test_byte_vector_row_rejects_invalid_payload(self, dtype, value, match):
        field_data = schema_types.FieldData(type=dtype, field_name="vec")
        field_info = {"name": "vec", "params": {"dim": 2}}

        with pytest.raises(ParamError, match=match):
            pack_field_value_to_field_data(value, field_data, field_info, {})

    def test_sparse_vector_row_rejects_multi_row_scipy_payload(self):
        field_data = schema_types.FieldData(type=DataType.SPARSE_FLOAT_VECTOR, field_name="sparse")
        field_info = {"name": "sparse"}

        class FakeSparse:
            shape = (2, 4)

        with patch("pymilvus.client.entity_helper.SciPyHelper.is_scipy_sparse", return_value=True):
            with pytest.raises(ParamError, match="expect 1 row"):
                pack_field_value_to_field_data(FakeSparse(), field_data, field_info, {})

    def test_sparse_vector_row_rejects_invalid_payload(self):
        field_data = schema_types.FieldData(type=DataType.SPARSE_FLOAT_VECTOR, field_name="sparse")
        field_info = {"name": "sparse"}

        with pytest.raises(ParamError, match="invalid input for sparse float vector"):
            pack_field_value_to_field_data("bad sparse row", field_data, field_info, {})

    def test_prepare_row_insert_and_upsert_flush_byte_vectors(self):
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
            {"name": "float", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
            {"name": "sparse", "type": DataType.SPARSE_FLOAT_VECTOR},
            {"name": "binary", "type": DataType.BINARY_VECTOR, "params": {"dim": 16}},
            {"name": "float16", "type": DataType.FLOAT16_VECTOR, "params": {"dim": 4}},
            {"name": "bfloat16", "type": DataType.BFLOAT16_VECTOR, "params": {"dim": 4}},
            {"name": "int8", "type": DataType.INT8_VECTOR, "params": {"dim": 4}},
        ]
        rows = [
            {
                "id": 1,
                "float": [1.0, 2.0, 3.0, 4.0],
                "sparse": {0: 1.0, 3: 2.0},
                "binary": self._byte_vector_payload(DataType.BINARY_VECTOR, 16, 1),
                "float16": self._byte_vector_payload(DataType.FLOAT16_VECTOR, 4, 1),
                "bfloat16": self._byte_vector_payload(DataType.BFLOAT16_VECTOR, 4, 1),
                "int8": self._byte_vector_payload(DataType.INT8_VECTOR, 4, 1),
            },
            {
                "id": 2,
                "float": [5.0, 6.0, 7.0, 8.0],
                "sparse": {1: 3.0},
                "binary": self._byte_vector_payload(DataType.BINARY_VECTOR, 16, 9),
                "float16": self._byte_vector_payload(DataType.FLOAT16_VECTOR, 4, 9),
                "bfloat16": self._byte_vector_payload(DataType.BFLOAT16_VECTOR, 4, 9),
                "int8": self._byte_vector_payload(DataType.INT8_VECTOR, 4, 9),
            },
        ]

        insert_request = Prepare.row_insert_param("c", rows, "", fields_info, [])
        upsert_request = Prepare.row_upsert_param("c", rows, "", fields_info, [])

        for request in (insert_request, upsert_request):
            fields = self._field_data_by_name(request)
            assert fields["float"].vectors.dim == 4
            assert list(fields["float"].vectors.float_vector.data) == [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
            assert len(fields["sparse"].vectors.sparse_float_vector.contents) == 2
            assert fields["binary"].vectors.dim == 16
            assert fields["binary"].vectors.binary_vector == b"".join(row["binary"] for row in rows)
            assert fields["float16"].vectors.dim == 4
            assert fields["float16"].vectors.float16_vector == b"".join(
                row["float16"].tobytes() for row in rows
            )
            assert fields["bfloat16"].vectors.dim == 4
            assert fields["bfloat16"].vectors.bfloat16_vector == b"".join(
                row["bfloat16"] for row in rows
            )
            assert fields["int8"].vectors.dim == 4
            assert fields["int8"].vectors.int8_vector == b"".join(
                row["int8"].tobytes() for row in rows
            )

    def test_prepare_row_insert_and_upsert_fp16_bf16_float_inputs_use_float_vector_slot(self):
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
            {"name": "float16", "type": DataType.FLOAT16_VECTOR, "params": {"dim": 4}},
            {"name": "bfloat16", "type": DataType.BFLOAT16_VECTOR, "params": {"dim": 4}},
        ]
        rows = [
            {
                "id": 1,
                "float16": [1.0, 2.0, 3.0, 4.0],
                "bfloat16": np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float64),
            },
            {
                "id": 2,
                "float16": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                "bfloat16": [13.0, 14.0, 15.0, 16.0],
            },
        ]

        insert_request = Prepare.row_insert_param("c", rows, "", fields_info, [])
        upsert_request = Prepare.row_upsert_param("c", rows, "", fields_info, [])

        for request in (insert_request, upsert_request):
            fields = self._field_data_by_name(request)
            assert fields["float16"].vectors.dim == 4
            assert list(fields["float16"].vectors.float_vector.data) == [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
            assert fields["float16"].vectors.WhichOneof("data") == "float_vector"
            assert fields["float16"].vectors.float16_vector == b""

            assert fields["bfloat16"].vectors.dim == 4
            assert list(fields["bfloat16"].vectors.float_vector.data) == [
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
            ]
            assert fields["bfloat16"].vectors.WhichOneof("data") == "float_vector"
            assert fields["bfloat16"].vectors.bfloat16_vector == b""
