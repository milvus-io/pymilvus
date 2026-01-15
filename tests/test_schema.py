import copy

import numpy as np
import pandas as pd
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.orm.schema import Function, FunctionType, StructFieldSchema


class TestCollectionSchema:
    @pytest.fixture(scope="function")
    def raw_dict(self):
        return {
            "description": "TestCollectionSchema_description",
            "enable_dynamic_field": True,
            "enable_namespace": True,
            "fields": [
                {
                    "name": "vec1",
                    "description": "desc1",
                    "type": DataType.FLOAT_VECTOR,
                    "params": {"dim": 128},
                },
                {
                    "name": "vec2",
                    "description": "desc2",
                    "type": DataType.BINARY_VECTOR,
                    "params": {"dim": 128},
                },
                {
                    "name": "ID",
                    "description": "ID",
                    "type": DataType.INT64,
                    "is_primary": True,
                    "auto_id": False
                },
            ]
        }

    def test_constructor_from_dict(self, raw_dict):
        schema = CollectionSchema.construct_from_dict(raw_dict)
        assert schema.enable_dynamic_field == raw_dict.get("enable_dynamic_field", False)
        assert schema.description, raw_dict['description']
        assert len(schema.fields) == len(raw_dict['fields'])
        f = schema.primary_field
        assert isinstance(f, FieldSchema)
        assert f.name == raw_dict['fields'][2]['name']

    def test_to_dict(self, raw_dict):
        schema = CollectionSchema.construct_from_dict(raw_dict)
        target = schema.to_dict()
        target.pop("auto_id", None)
        assert target == raw_dict
        assert target is not raw_dict

    def test_init_with_functions(self, raw_dict):
        functions = [
            Function("func1", FunctionType.BM25, ["field1"], ["field2"])
        ]
        schema = CollectionSchema.construct_from_dict(raw_dict)
        schema_with_func = CollectionSchema(schema.fields, schema.description, functions=functions)
        assert schema_with_func.functions == functions


class TestFieldSchema:
    @pytest.fixture(scope="function")
    def raw_dict_float_vector(self):
        return  {
            "name": "TestFieldSchema_name_floatvector",
            "description": "TestFieldSchema_description_floatvector",
            "type": DataType.FLOAT_VECTOR,
            "params": {"dim": 128},
        }

    @pytest.fixture(scope="function")
    def raw_dict_binary_vector(self):
        return {
            "name": "TestFieldSchema_name_binary_vector",
            "description": "TestFieldSchema_description_binary_vector",
            "type": DataType.BINARY_VECTOR,
            "params": {"dim": 128},
        }

    @pytest.fixture(scope="function")
    def raw_dict_float16_vector(self):
        return {
            "name": "TestFieldSchema_name_float16_vector",
            "description": "TestFieldSchema_description_float16_vector",
            "type": DataType.FLOAT16_VECTOR,
            "params": {"dim": 128},
        }

    @pytest.fixture(scope="function")
    def raw_dict_bfloat16_vector(self):
        return {
            "name": "TestFieldSchema_name_bfloat16_vector",
            "description": "TestFieldSchema_description_bfloat16_vector",
            "type": DataType.BFLOAT16_VECTOR,
            "params": {"dim": 128},
        }

    @pytest.fixture(scope="function")
    def raw_dict_int8_vector(self):
        return {
            "name": "TestFieldSchema_name_int8_vector",
            "description": "TestFieldSchema_description_int8_vector",
            "type": DataType.INT8_VECTOR,
            "params": {"dim": 128},
        }

    @pytest.fixture(scope="function")
    def raw_dict_norm(self):
        return {
            "name": "TestFieldSchema_name_norm",
            "description": "TestFieldSchema_description_norm",
            "type": DataType.INT64,
        }

    @pytest.fixture(scope="function")
    def dataframe1(self):
        data = {
            'float': [1.0],
            'int32': [2],
            'float_vec': [np.array([3, 4.0], np.float32)]
        }
        return pd.DataFrame(data)

    def test_constructor_from_float_dict(self, raw_dict_float_vector):
        field = FieldSchema.construct_from_dict(raw_dict_float_vector)
        assert field.dtype == DataType.FLOAT_VECTOR
        assert field.description == raw_dict_float_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_float_vector['name']
        assert field.dim == raw_dict_float_vector['params']['dim']

    def test_constructor_from_binary_dict(self, raw_dict_binary_vector):
        field = FieldSchema.construct_from_dict(raw_dict_binary_vector)
        assert field.dtype == DataType.BINARY_VECTOR
        assert field.description == raw_dict_binary_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_binary_vector['name']
        assert field.dim == raw_dict_binary_vector['params']['dim']

    def test_constructor_from_float16_dict(self, raw_dict_float16_vector):
        field = FieldSchema.construct_from_dict(raw_dict_float16_vector)
        assert field.dtype == DataType.FLOAT16_VECTOR
        assert field.description == raw_dict_float16_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_float16_vector['name']
        assert field.dim == raw_dict_float16_vector['params']['dim']

    def test_constructor_from_bfloat16_dict(self, raw_dict_bfloat16_vector):
        field = FieldSchema.construct_from_dict(raw_dict_bfloat16_vector)
        assert field.dtype == DataType.BFLOAT16_VECTOR
        assert field.description == raw_dict_bfloat16_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_bfloat16_vector['name']
        assert field.dim == raw_dict_bfloat16_vector['params']['dim']

    def test_constructor_from_int8_dict(self, raw_dict_int8_vector):
        field = FieldSchema.construct_from_dict(raw_dict_int8_vector)
        assert field.dtype == DataType.INT8_VECTOR
        assert field.description == raw_dict_int8_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_int8_vector['name']
        assert field.dim == raw_dict_int8_vector['params']['dim']

    def test_constructor_from_norm_dict(self, raw_dict_norm):
        field = FieldSchema.construct_from_dict(raw_dict_norm)
        assert field.dtype == DataType.INT64
        assert field.description == raw_dict_norm['description']
        assert field.is_primary is False
        assert field.name == raw_dict_norm['name']
        assert field.dim is None
        assert field.dummy is None

    def test_cmp(self, raw_dict_binary_vector):
        field1 = FieldSchema.construct_from_dict(raw_dict_binary_vector)
        field2 = FieldSchema.construct_from_dict(raw_dict_binary_vector)
        assert field1 == field2
        dict1 = copy.deepcopy(raw_dict_binary_vector)
        dict1["name"] = dict1["name"] + "_"
        field3 = FieldSchema.construct_from_dict(dict1)
        assert field1 != field3

    def test_to_dict(self, raw_dict_norm, raw_dict_float_vector, raw_dict_binary_vector):
        fields = []
        dicts = [raw_dict_norm, raw_dict_float_vector, raw_dict_binary_vector]
        fields.append(FieldSchema.construct_from_dict(raw_dict_norm))
        fields.append(FieldSchema.construct_from_dict(raw_dict_float_vector))
        fields.append(FieldSchema.construct_from_dict(raw_dict_binary_vector))

        for i, f in enumerate(fields):
            target = f.to_dict()
            assert target == dicts[i]
            assert target is not dicts[i]


class TestStructFieldSchema:
    def test_add_field_with_reused_struct_schema(self):
        """Test that reusing the same struct_schema for multiple fields works correctly.

        This test verifies the fix for issue #3104 where reusing a struct_schema
        object across multiple ARRAY<STRUCT> fields caused "duplicated field name" errors.
        """
        # Create a struct schema to reuse
        struct_schema = StructFieldSchema()
        struct_schema.add_field("score", DataType.FLOAT)
        struct_schema.add_field("content", DataType.VARCHAR, max_length=512)

        # Create collection schema
        schema = CollectionSchema(
            fields=[
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128),
            ]
        )

        # Add two ARRAY<STRUCT> fields using the same struct_schema
        schema.add_field(
            "field1",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=10,
        )
        schema.add_field(
            "field2",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=20,
        )

        # Verify both struct fields have different names
        assert len(schema.struct_fields) == 2
        assert schema.struct_fields[0].name == "field1"
        assert schema.struct_fields[1].name == "field2"
        assert schema.struct_fields[0].max_capacity == 10
        assert schema.struct_fields[1].max_capacity == 20

        # Verify original struct_schema was not modified
        assert struct_schema.name == ""
        assert struct_schema.max_capacity is None

    def test_struct_field_schema_to_dict(self):
        """Test StructFieldSchema to_dict method."""
        struct_schema = StructFieldSchema()
        struct_schema.name = "test_struct"
        struct_schema.max_capacity = 10
        struct_schema.add_field("score", DataType.FLOAT)
        struct_schema.add_field("content", DataType.VARCHAR, max_length=512)

        result = struct_schema.to_dict()
        assert result["name"] == "test_struct"
        assert result["max_capacity"] == 10
        assert len(result["fields"]) == 2
        assert result["fields"][0]["name"] == "score"
        assert result["fields"][1]["name"] == "content"

    def test_struct_field_schema_construct_from_dict(self):
        """Test StructFieldSchema construct_from_dict method."""
        raw_dict = {
            "name": "test_struct",
            "description": "test description",
            "max_capacity": 10,
            "fields": [
                {"name": "score", "type": DataType.FLOAT, "description": ""},
                {
                    "name": "content",
                    "type": DataType.VARCHAR,
                    "description": "",
                    "params": {"max_length": 512},
                },
            ],
        }

        struct_schema = StructFieldSchema.construct_from_dict(raw_dict)
        assert struct_schema.name == "test_struct"
        assert struct_schema.max_capacity == 10
        assert len(struct_schema.fields) == 2
        assert struct_schema.fields[0].name == "score"
        assert struct_schema.fields[1].name == "content"
