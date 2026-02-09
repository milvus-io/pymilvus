"""Comprehensive tests for pymilvus/orm/schema.py.

Merged and refactored from:
- test_orm_schema.py
- test_schema.py
- test_orm_schema_coverage.py
"""

import copy

import numpy as np
import pandas as pd
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.client.types import HighlightType
from pymilvus.exceptions import (
    AutoIDException,
    CannotInferSchemaException,
    ClusteringKeyException,
    DataNotMatchException,
    DataTypeNotSupportException,
    FieldsTypeException,
    FieldTypeException,
    FunctionsTypeException,
    ParamError,
    PartitionKeyException,
    PrimaryKeyException,
    SchemaNotReadyException,
)
from pymilvus.orm.schema import (
    Function,
    FunctionScore,
    FunctionType,
    LexicalHighlighter,
    SemanticHighlighter,
    StructFieldSchema,
    check_insert_schema,
    check_is_row_based,
    check_schema,
    check_upsert_schema,
    infer_default_value_bydata,
    is_row_based,
    is_valid_insert_data,
    isVectorDataType,
    prepare_fields_from_dataframe,
    validate_clustering_key,
    validate_partition_key,
    validate_primary_key,
)

# ============================================================
# FieldSchema Tests
# ============================================================


class TestFieldSchemaCreation:
    """Tests for FieldSchema creation with various data types."""

    @pytest.mark.parametrize(
        "name,dtype,kwargs,expected_attrs",
        [
            pytest.param("id", DataType.INT64, {}, {"is_primary": False}, id="int64_basic"),
            pytest.param(
                "pk",
                DataType.INT64,
                {"is_primary": True},
                {"is_primary": True},
                id="int64_primary",
            ),
            pytest.param(
                "pk",
                DataType.INT64,
                {"is_primary": True, "auto_id": True},
                {"auto_id": True},
                id="int64_auto_id",
            ),
            pytest.param(
                "name",
                DataType.VARCHAR,
                {"max_length": 256},
                {"max_length": 256},
                id="varchar",
            ),
            pytest.param(
                "vec",
                DataType.FLOAT_VECTOR,
                {"dim": 128},
                {"dim": 128},
                id="float_vector",
            ),
            pytest.param(
                "vec",
                DataType.BINARY_VECTOR,
                {"dim": 256},
                {"dim": 256},
                id="binary_vector",
            ),
            pytest.param(
                "vec",
                DataType.FLOAT16_VECTOR,
                {"dim": 64},
                {"dim": 64},
                id="float16_vector",
            ),
            pytest.param(
                "vec",
                DataType.BFLOAT16_VECTOR,
                {"dim": 64},
                {"dim": 64},
                id="bfloat16_vector",
            ),
            pytest.param(
                "vec",
                DataType.INT8_VECTOR,
                {"dim": 32},
                {"dim": 32},
                id="int8_vector",
            ),
            pytest.param("vec", DataType.SPARSE_FLOAT_VECTOR, {}, {}, id="sparse_float_vector"),
            pytest.param("meta", DataType.JSON, {}, {}, id="json"),
            pytest.param(
                "tags",
                DataType.ARRAY,
                {"element_type": DataType.VARCHAR, "max_capacity": 100, "max_length": 64},
                {"element_type": DataType.VARCHAR, "max_capacity": 100},
                id="array",
            ),
            pytest.param(
                "opt",
                DataType.VARCHAR,
                {"max_length": 100, "nullable": True},
                {"nullable": True},
                id="nullable",
            ),
            pytest.param(
                "desc",
                DataType.INT64,
                {"description": "Test field"},
                {"description": "Test field"},
                id="with_description",
            ),
        ],
    )
    def test_create_field(self, name, dtype, kwargs, expected_attrs):
        """Test creating fields with various configurations."""
        field = FieldSchema(name, dtype, **kwargs)
        assert field.name == name
        assert field.dtype == dtype
        for attr, expected in expected_attrs.items():
            assert getattr(field, attr) == expected


class TestFieldSchemaValidation:
    """Tests for FieldSchema validation errors."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param("invalid_type", id="string"),
            pytest.param(DataType.UNKNOWN, id="unknown"),
        ],
    )
    def test_invalid_dtype(self, dtype):
        """Test field with invalid dtype raises error."""
        with pytest.raises(DataTypeNotSupportException):
            FieldSchema("field", dtype)

    @pytest.mark.parametrize(
        "kwargs,error_type,error_match",
        [
            pytest.param(
                {"is_primary": "true"},
                PrimaryKeyException,
                None,
                id="is_primary_not_bool",
            ),
            pytest.param(
                {"is_primary": True, "auto_id": "true"},
                AutoIDException,
                None,
                id="auto_id_not_bool",
            ),
            pytest.param(
                {"auto_id": True},
                PrimaryKeyException,
                "primary",
                id="auto_id_not_primary",
            ),
            pytest.param(
                {"is_partition_key": "true"},
                PartitionKeyException,
                None,
                id="partition_key_not_bool",
            ),
            pytest.param(
                {"is_clustering_key": "true"},
                ClusteringKeyException,
                None,
                id="clustering_key_not_bool",
            ),
        ],
    )
    def test_invalid_field_options(self, kwargs, error_type, error_match):
        """Test field with invalid options raises appropriate error."""
        if error_match:
            with pytest.raises(error_type, match=error_match):
                FieldSchema("field", DataType.INT64, **kwargs)
        else:
            with pytest.raises(error_type):
                FieldSchema("field", DataType.INT64, **kwargs)

    def test_default_value_none_not_nullable(self):
        """Test default_value=None on non-nullable field."""
        with pytest.raises(ParamError, match=r"[Dd]efault"):
            FieldSchema("field", DataType.INT64, default_value=None, nullable=False)


class TestFieldSchemaEquality:
    """Tests for FieldSchema equality comparison."""

    @pytest.mark.parametrize(
        "field1_kwargs,field2_kwargs,expected_equal",
        [
            pytest.param(
                {"name": "id", "dtype": DataType.INT64, "is_primary": True},
                {"name": "id", "dtype": DataType.INT64, "is_primary": True},
                True,
                id="equal_fields",
            ),
            pytest.param(
                {"name": "id1", "dtype": DataType.INT64},
                {"name": "id2", "dtype": DataType.INT64},
                False,
                id="different_names",
            ),
            pytest.param(
                {"name": "f", "dtype": DataType.INT64},
                {"name": "f", "dtype": DataType.FLOAT},
                False,
                id="different_types",
            ),
            pytest.param(
                {"name": "id", "dtype": DataType.INT64, "is_primary": True},
                {"name": "id", "dtype": DataType.INT64, "is_primary": False},
                False,
                id="different_primary",
            ),
        ],
    )
    def test_field_equality(self, field1_kwargs, field2_kwargs, expected_equal):
        """Test field equality comparison."""
        field1 = FieldSchema(**field1_kwargs)
        field2 = FieldSchema(**field2_kwargs)
        assert (field1 == field2) == expected_equal

    def test_not_equal_to_non_field(self):
        """Test inequality with non-FieldSchema object."""
        field = FieldSchema("id", DataType.INT64)
        assert field != "not_a_field"
        assert field != {"name": "id"}


class TestFieldSchemaToDict:
    """Tests for FieldSchema to_dict and construct_from_dict methods."""

    @pytest.mark.parametrize(
        "raw_dict",
        [
            pytest.param(
                {"name": "id", "type": DataType.INT64, "description": "test"},
                id="int64",
            ),
            pytest.param(
                {
                    "name": "vec",
                    "type": DataType.FLOAT_VECTOR,
                    "description": "",
                    "params": {"dim": 128},
                },
                id="float_vector",
            ),
            pytest.param(
                {
                    "name": "vec",
                    "type": DataType.BINARY_VECTOR,
                    "description": "",
                    "params": {"dim": 128},
                },
                id="binary_vector",
            ),
            pytest.param(
                {
                    "name": "vec",
                    "type": DataType.FLOAT16_VECTOR,
                    "description": "",
                    "params": {"dim": 128},
                },
                id="float16_vector",
            ),
            pytest.param(
                {
                    "name": "vec",
                    "type": DataType.BFLOAT16_VECTOR,
                    "description": "",
                    "params": {"dim": 128},
                },
                id="bfloat16_vector",
            ),
            pytest.param(
                {
                    "name": "vec",
                    "type": DataType.INT8_VECTOR,
                    "description": "",
                    "params": {"dim": 128},
                },
                id="int8_vector",
            ),
            pytest.param(
                {
                    "name": "tags",
                    "type": DataType.ARRAY,
                    "description": "",
                    "element_type": DataType.VARCHAR,
                    "params": {"max_capacity": 100, "max_length": 64},
                },
                id="array",
            ),
        ],
    )
    def test_construct_from_dict_roundtrip(self, raw_dict):
        """Test constructing field from dict and converting back."""
        field = FieldSchema.construct_from_dict(raw_dict)
        assert field.name == raw_dict["name"]
        assert field.dtype == raw_dict["type"]

        result = field.to_dict()
        assert result["name"] == raw_dict["name"]
        assert result["type"] == raw_dict["type"]


class TestFieldSchemaDeepCopy:
    """Tests for FieldSchema __deepcopy__ method."""

    def test_deepcopy(self):
        """Test deepcopy creates independent copy."""
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128)
        field_copy = copy.deepcopy(field)
        assert field == field_copy
        assert field is not field_copy

    def test_deepcopy_with_memodict_none(self):
        """Test deepcopy with None memodict."""
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128)
        field_copy = field.__deepcopy__(None)
        assert field == field_copy


class TestFieldSchemaTypeParams:
    """Tests for FieldSchema type parameter handling."""

    @pytest.mark.parametrize(
        "enable_match,expected",
        [
            pytest.param("true", True, id="string_true"),
            pytest.param("false", False, id="string_false"),
        ],
    )
    def test_type_params_string_bool_conversion(self, enable_match, expected):
        """Test string 'true'/'false' converted to bool."""
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128, enable_match=enable_match)
        assert field._type_params.get("enable_match") is expected

    def test_mmap_enabled_param(self):
        """Test mmap_enabled type param."""
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128, mmap_enabled=True)
        assert field._type_params.get("mmap_enabled") is True

    def test_warmup_param(self):
        """Test warmup type param."""
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128, warmup=True)
        assert field._type_params.get("warmup") is True

    def test_warmup_param_false(self):
        """Test warmup type param with False value."""
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128, warmup=False)
        assert field._type_params.get("warmup") is False

    def test_analyzer_params_dict(self):
        """Test analyzer_params as dict gets serialized."""
        field = FieldSchema(
            "text", DataType.VARCHAR, max_length=1000, analyzer_params={"type": "standard"}
        )
        assert field._kwargs["analyzer_params"] == '{"type":"standard"}'


# ============================================================
# CollectionSchema Tests
# ============================================================


class TestCollectionSchemaCreation:
    """Tests for CollectionSchema creation."""

    @pytest.mark.parametrize(
        "fields,kwargs,expected_attrs",
        [
            pytest.param(
                [
                    FieldSchema("id", DataType.INT64, is_primary=True),
                    FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
                ],
                {},
                {"primary_field_name": "id"},
                id="basic",
            ),
            pytest.param(
                [
                    FieldSchema("id", DataType.INT64, is_primary=True),
                    FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
                ],
                {"description": "Test collection"},
                {"description": "Test collection"},
                id="with_description",
            ),
            pytest.param(
                [
                    FieldSchema("id", DataType.INT64, is_primary=True),
                    FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
                ],
                {"enable_dynamic_field": True},
                {"enable_dynamic_field": True},
                id="dynamic_field",
            ),
            pytest.param(
                [
                    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
                ],
                {},
                {"auto_id": True},
                id="auto_id",
            ),
        ],
    )
    def test_create_schema(self, fields, kwargs, expected_attrs):
        """Test creating schema with various configurations."""
        schema = CollectionSchema(fields, **kwargs)
        for attr, expected in expected_attrs.items():
            if attr == "primary_field_name":
                assert schema.primary_field.name == expected
            else:
                assert getattr(schema, attr) == expected


class TestCollectionSchemaValidation:
    """Tests for CollectionSchema validation errors."""

    def test_invalid_fields_type(self):
        """Test with non-list fields."""
        with pytest.raises(FieldsTypeException):
            CollectionSchema("not_a_list")

    def test_invalid_field_type_in_list(self):
        """Test with non-FieldSchema in fields list."""
        with pytest.raises(FieldTypeException):
            CollectionSchema([{"name": "id", "type": DataType.INT64}])

    def test_invalid_functions_type(self):
        """Test with non-list functions."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        with pytest.raises(FunctionsTypeException):
            CollectionSchema(fields, functions="not_a_list")

    def test_invalid_function_in_list(self):
        """Test with non-Function in functions list."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        with pytest.raises(SchemaNotReadyException):
            CollectionSchema(fields, functions=[{"name": "func"}])

    def test_invalid_struct_field_in_list(self):
        """Test with non-StructFieldSchema in struct_fields list."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        with pytest.raises(FieldTypeException):
            CollectionSchema(fields, struct_fields=[{"name": "struct"}])

    @pytest.mark.parametrize(
        "kwarg,value,error_type",
        [
            pytest.param("primary_field", 123, PrimaryKeyException, id="primary_field_not_str"),
            pytest.param(
                "partition_key_field", 123, PartitionKeyException, id="partition_key_not_str"
            ),
            pytest.param(
                "clustering_key_field_name",
                123,
                ClusteringKeyException,
                id="clustering_key_not_str",
            ),
            pytest.param("auto_id", "true", AutoIDException, id="auto_id_not_bool"),
        ],
    )
    def test_invalid_kwarg_types(self, kwarg, value, error_type):
        """Test invalid kwarg types in schema creation."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        with pytest.raises(error_type):
            CollectionSchema(fields, **{kwarg: value})

    def test_no_primary_key_raises_exception(self):
        """Test creating schema without primary key raises exception."""
        with pytest.raises(PrimaryKeyException, match=r"[Pp]rimary"):
            CollectionSchema(
                [
                    FieldSchema("id", DataType.INT64),
                    FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
                ]
            )

    @pytest.mark.parametrize(
        "fields,error_type,error_match",
        [
            pytest.param(
                [
                    FieldSchema("id1", DataType.INT64, is_primary=True),
                    FieldSchema("id2", DataType.INT64, is_primary=True),
                ],
                PrimaryKeyException,
                "only one",
                id="multiple_primary_keys",
            ),
            pytest.param(
                [
                    FieldSchema("id", DataType.INT64, is_primary=True),
                    FieldSchema("cat1", DataType.VARCHAR, max_length=100, is_partition_key=True),
                    FieldSchema("cat2", DataType.VARCHAR, max_length=100, is_partition_key=True),
                ],
                PartitionKeyException,
                "only one",
                id="multiple_partition_keys",
            ),
            pytest.param(
                [
                    FieldSchema("id", DataType.INT64, is_primary=True),
                    FieldSchema("ts1", DataType.INT64, is_clustering_key=True),
                    FieldSchema("ts2", DataType.INT64, is_clustering_key=True),
                ],
                ClusteringKeyException,
                "only one",
                id="multiple_clustering_keys",
            ),
        ],
    )
    def test_multiple_special_keys_error(self, fields, error_type, error_match):
        """Test error when multiple special keys specified."""
        with pytest.raises(error_type, match=error_match):
            CollectionSchema(fields)


class TestCollectionSchemaKeyFields:
    """Tests for CollectionSchema key field handling."""

    def test_primary_field_from_kwarg(self):
        """Test setting primary field from kwarg."""
        fields = [
            FieldSchema("id", DataType.INT64),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        schema = CollectionSchema(fields, primary_field="id")
        assert schema.primary_field.name == "id"
        assert schema.primary_field.is_primary is True

    def test_partition_key_field_from_kwarg(self):
        """Test setting partition key field from kwarg."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("category", DataType.VARCHAR, max_length=100),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        schema = CollectionSchema(fields, partition_key_field="category")
        assert schema.partition_key_field.name == "category"
        assert schema.partition_key_field.is_partition_key is True

    def test_clustering_key_field_from_kwarg(self):
        """Test setting clustering key field from kwarg."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("timestamp", DataType.INT64),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        schema = CollectionSchema(fields, clustering_key_field="timestamp")
        assert schema._clustering_key_field.name == "timestamp"
        assert schema._clustering_key_field.is_clustering_key is True


class TestCollectionSchemaAddField:
    """Tests for CollectionSchema.add_field method."""

    def test_add_field_basic(self):
        """Test adding basic field."""
        schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=128)
        assert len(schema.fields) == 2
        assert schema.fields[1].name == "vec"

    def test_add_field_with_params(self):
        """Test adding field with parameters."""
        schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
        schema.add_field("name", DataType.VARCHAR, max_length=256, nullable=True)
        field = schema.fields[1]
        assert field.max_length == 256
        assert field.nullable is True

    def test_add_struct_field_missing_struct_schema(self):
        """Test adding struct field without struct_schema raises error."""
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        with pytest.raises(ParamError, match="struct_schema"):
            schema.add_field("struct", DataType.ARRAY, element_type=DataType.STRUCT)

    def test_add_struct_field_missing_max_capacity(self):
        """Test adding struct field without max_capacity raises error."""
        struct = StructFieldSchema()
        struct.add_field("score", DataType.FLOAT)
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        with pytest.raises(ParamError, match="max_capacity"):
            schema.add_field(
                "struct", DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct
            )

    def test_add_struct_field_with_mmap(self):
        """Test adding struct field with mmap_enabled."""
        struct = StructFieldSchema()
        struct.add_field("score", DataType.FLOAT)
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        schema.add_field(
            "struct",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct,
            max_capacity=10,
            mmap_enabled=True,
        )
        assert len(schema.struct_fields) == 1
        assert schema.struct_fields[0]._type_params.get("mmap_enabled") is True

    def test_add_struct_field_with_warmup(self):
        """Test adding struct field with warmup."""
        struct = StructFieldSchema()
        struct.add_field("score", DataType.FLOAT)
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        schema.add_field(
            "struct",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct,
            max_capacity=10,
            warmup=True,
        )
        assert len(schema.struct_fields) == 1
        assert schema.struct_fields[0]._type_params.get("warmup") is True


class TestCollectionSchemaToDict:
    """Tests for CollectionSchema to_dict and construct_from_dict methods."""

    def test_to_dict_basic(self, raw_dict_schema):
        """Test to_dict for basic schema."""
        schema = CollectionSchema.construct_from_dict(raw_dict_schema)
        result = schema.to_dict()
        assert result["description"] == raw_dict_schema["description"]
        assert len(result["fields"]) == len(raw_dict_schema["fields"])

    def test_to_dict_with_functions(self):
        """Test to_dict includes functions when present."""
        func = Function("f", FunctionType.BM25, ["text"], ["sparse"])
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=1000),
                FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
            ],
            functions=[func],
        )
        d = schema.to_dict()
        assert "functions" in d
        assert len(d["functions"]) == 1

    def test_to_dict_with_struct_fields(self):
        """Test to_dict includes struct_fields when present."""
        struct = StructFieldSchema()
        struct.name = "metadata"
        struct.add_field("score", DataType.FLOAT)
        struct.max_capacity = 10

        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ],
            struct_fields=[struct],
        )
        d = schema.to_dict()
        assert "struct_fields" in d
        assert len(d["struct_fields"]) == 1

    def test_construct_from_dict_with_dynamic(self, raw_dict_schema):
        """Test constructing schema with dynamic field from dict."""
        schema = CollectionSchema.construct_from_dict(raw_dict_schema)
        assert schema.enable_dynamic_field == raw_dict_schema.get("enable_dynamic_field", False)

    @pytest.mark.parametrize(
        "struct_key",
        [
            pytest.param("struct_fields", id="struct_fields"),
            pytest.param("struct_array_fields", id="struct_array_fields"),
        ],
    )
    def test_construct_with_struct_fields_formats(self, struct_key):
        """Test constructing schema with different struct_fields formats."""
        d = {
            "description": "",
            "fields": [{"name": "id", "type": DataType.INT64, "is_primary": True}],
            struct_key: [
                {
                    "name": "metadata",
                    "max_capacity": 10,
                    "fields" if struct_key == "struct_fields" else "struct_fields": [
                        {"name": "score", "type": DataType.FLOAT, "description": ""},
                    ],
                }
            ],
        }
        schema = CollectionSchema.construct_from_dict(d)
        assert len(schema.struct_fields) == 1


class TestCollectionSchemaProperties:
    """Tests for CollectionSchema properties."""

    def test_repr(self, basic_schema):
        """Test __repr__ method."""
        repr_str = repr(basic_schema)
        assert "id" in repr_str

    def test_len(self, basic_schema):
        """Test __len__ method."""
        assert len(basic_schema) == 2

    def test_eq(self):
        """Test __eq__ method."""
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
        schema1 = CollectionSchema(fields)
        schema2 = CollectionSchema(fields)
        assert schema1 == schema2

    @pytest.mark.parametrize(
        "attr,value",
        [
            pytest.param("auto_id", True, id="auto_id"),
            pytest.param("enable_dynamic_field", True, id="enable_dynamic_field"),
            pytest.param("enable_namespace", True, id="enable_namespace"),
        ],
    )
    def test_property_setters(self, basic_schema, attr, value):
        """Test property setters."""
        setattr(basic_schema, attr, value)
        assert getattr(basic_schema, attr) is value

    def test_verify_method(self, basic_schema):
        """Test verify method."""
        basic_schema.verify()  # Should not raise


# ============================================================
# Function Tests
# ============================================================


class TestFunctionCreation:
    """Tests for Function class creation."""

    @pytest.mark.parametrize(
        "name,func_type,input_fields,output_fields,params",
        [
            pytest.param(
                "bm25_func",
                FunctionType.BM25,
                ["text"],
                ["sparse"],
                None,
                id="bm25",
            ),
            pytest.param(
                "text_embed",
                FunctionType.TEXTEMBEDDING,
                ["text"],
                ["vector"],
                {"model_name": "test"},
                id="text_embedding",
            ),
            pytest.param(
                "rerank",
                FunctionType.RERANK,
                ["query", "doc"],
                ["score"],
                None,
                id="rerank",
            ),
        ],
    )
    def test_create_function(self, name, func_type, input_fields, output_fields, params):
        """Test creating functions with various configurations."""
        kwargs = {"params": params} if params else {}
        func = Function(name, func_type, input_fields, output_fields, **kwargs)
        assert func.name == name
        assert func.type == func_type
        assert func.input_field_names == input_fields
        assert func.output_field_names == output_fields

    def test_input_as_string(self):
        """Test input_field_names as single string."""
        func = Function("func", FunctionType.BM25, "input", ["output"])
        assert func.input_field_names == ["input"]

    def test_output_as_string(self):
        """Test output_field_names as single string."""
        func = Function("func", FunctionType.BM25, ["input"], "output")
        assert func.output_field_names == ["output"]

    def test_output_none_default(self):
        """Test output_field_names defaults to empty list."""
        func = Function("func", FunctionType.BM25, ["input"])
        assert func.output_field_names == []


class TestFunctionValidation:
    """Tests for Function validation errors."""

    @pytest.mark.parametrize(
        "name,error_match",
        [
            pytest.param(123, "string", id="int_name"),
            pytest.param(None, "string", id="none_name"),
        ],
    )
    def test_invalid_name_type(self, name, error_match):
        """Test invalid name types."""
        with pytest.raises(ParamError, match=error_match):
            Function(name, FunctionType.BM25, ["input"], ["output"])

    def test_invalid_description_type(self):
        """Test invalid description type."""
        with pytest.raises(ParamError, match="string"):
            Function("func", FunctionType.BM25, ["input"], ["output"], description=123)

    def test_invalid_input_field_names_type(self):
        """Test invalid input_field_names type."""
        with pytest.raises(ParamError, match="string or a list"):
            Function("func", FunctionType.BM25, 123, ["output"])

    def test_invalid_output_field_names_type(self):
        """Test invalid output_field_names type."""
        with pytest.raises(ParamError, match="string or a list"):
            Function("func", FunctionType.BM25, ["input"], 123)

    def test_invalid_function_type(self):
        """Test invalid function type."""
        with pytest.raises(ParamError, match=r"[Uu]nknown"):
            Function("func", "invalid_type", ["input"], ["output"])

    @pytest.mark.parametrize(
        "input_fields,output_fields,error_match",
        [
            pytest.param(["dup", "dup"], ["out"], r"[Dd]uplicate", id="duplicate_input"),
            pytest.param(["in"], ["dup", "dup"], r"[Dd]uplicate", id="duplicate_output"),
            pytest.param(["field"], ["field"], "different", id="common_input_output"),
        ],
    )
    def test_duplicate_fields(self, input_fields, output_fields, error_match):
        """Test duplicate field names."""
        with pytest.raises(ParamError, match=error_match):
            Function("func", FunctionType.BM25, input_fields, output_fields)

    def test_invalid_params_type(self):
        """Test invalid params type."""
        with pytest.raises(ParamError, match="dictionary"):
            Function("func", FunctionType.BM25, ["input"], ["output"], params="not_a_dict")


class TestFunctionVerify:
    """Tests for Function.verify method."""

    @pytest.mark.parametrize(
        "func_type,input_fields,output_fields,input_dtype,output_dtype,should_raise,error_match",
        [
            pytest.param(
                FunctionType.BM25,
                ["in1", "in2"],
                ["out"],
                DataType.VARCHAR,
                DataType.SPARSE_FLOAT_VECTOR,
                True,
                None,
                id="bm25_wrong_input_count",
            ),
            pytest.param(
                FunctionType.BM25,
                ["in"],
                ["out"],
                DataType.INT64,
                DataType.SPARSE_FLOAT_VECTOR,
                True,
                r"[Ii]nput",
                id="bm25_wrong_input_type",
            ),
            pytest.param(
                FunctionType.BM25,
                ["in"],
                ["out"],
                DataType.VARCHAR,
                DataType.FLOAT_VECTOR,
                True,
                r"[Oo]utput",
                id="bm25_wrong_output_type",
            ),
            pytest.param(
                FunctionType.TEXTEMBEDDING,
                ["in"],
                ["out"],
                DataType.VARCHAR,
                DataType.SPARSE_FLOAT_VECTOR,
                True,
                r"[Oo]utput",
                id="textembedding_wrong_output_type",
            ),
            pytest.param(
                FunctionType.UNKNOWN,
                ["in"],
                ["out"],
                DataType.VARCHAR,
                DataType.FLOAT_VECTOR,
                True,
                r"[Uu]nknown",
                id="unknown_type",
            ),
            pytest.param(
                FunctionType.RERANK,
                ["in"],
                ["out"],
                DataType.VARCHAR,
                DataType.FLOAT_VECTOR,
                False,
                None,
                id="rerank_passes",
            ),
            pytest.param(
                FunctionType.MINHASH,
                ["in"],
                ["out"],
                DataType.VARCHAR,
                DataType.FLOAT_VECTOR,
                False,
                None,
                id="minhash_passes",
            ),
        ],
    )
    def test_function_verify(
        self,
        func_type,
        input_fields,
        output_fields,
        input_dtype,
        output_dtype,
        should_raise,
        error_match,
    ):
        """Test Function.verify with various configurations."""
        func = Function("f", func_type, input_fields, output_fields)
        fields = [FieldSchema("id", DataType.INT64, is_primary=True)]
        for field in input_fields:
            kwargs = {"max_length": 1000} if input_dtype == DataType.VARCHAR else {}
            if input_dtype == DataType.FLOAT_VECTOR:
                kwargs["dim"] = 128
            fields.append(FieldSchema(field, input_dtype, **kwargs))
        for field in output_fields:
            kwargs = {}
            if output_dtype == DataType.FLOAT_VECTOR:
                kwargs["dim"] = 128
            fields.append(FieldSchema(field, output_dtype, **kwargs))

        schema = CollectionSchema(fields, check_fields=False)

        if should_raise:
            if error_match:
                with pytest.raises(ParamError, match=error_match):
                    func.verify(schema)
            else:
                with pytest.raises(ParamError):
                    func.verify(schema)
        else:
            func.verify(schema)  # Should not raise


class TestFunctionEquality:
    """Tests for Function __eq__ method."""

    def test_function_equality(self):
        """Test function equality."""
        func1 = Function("f", FunctionType.BM25, ["in"], ["out"])
        func2 = Function("f", FunctionType.BM25, ["in"], ["out"])
        assert func1 == func2

    def test_function_not_equal_to_non_function(self):
        """Test function not equal to non-Function object."""
        func = Function("f", FunctionType.BM25, ["in"], ["out"])
        assert func != "not_a_function"


class TestFunctionToDict:
    """Tests for Function to_dict and construct_from_dict methods."""

    def test_to_dict_basic(self):
        """Test Function.to_dict method."""
        func = Function(
            "test_func",
            FunctionType.BM25,
            ["input"],
            ["output"],
            description="Test function",
        )
        result = func.to_dict()
        assert result["name"] == "test_func"
        assert result["type"] == FunctionType.BM25

    def test_construct_from_dict(self):
        """Test Function.construct_from_dict."""
        d = {
            "name": "test_func",
            "type": FunctionType.BM25,
            "input_field_names": ["input"],
            "output_field_names": ["output"],
            "description": "test",
            "params": {},
        }
        func = Function.construct_from_dict(d)
        assert func.name == "test_func"
        assert func.type == FunctionType.BM25


# ============================================================
# FunctionScore Tests
# ============================================================


class TestFunctionScore:
    """Tests for FunctionScore class."""

    def test_function_score_single_function(self):
        """Test FunctionScore with single function."""
        func = Function("f", FunctionType.BM25, ["text"], ["sparse"])
        score = FunctionScore(func)
        assert len(score.functions) == 1
        assert score.params is None

    def test_function_score_list_of_functions(self):
        """Test FunctionScore with list of functions."""
        funcs = [
            Function("f1", FunctionType.BM25, ["text1"], ["sparse1"]),
            Function("f2", FunctionType.BM25, ["text2"], ["sparse2"]),
        ]
        score = FunctionScore(funcs, params={"weight": 0.5})
        assert len(score.functions) == 2
        assert score.params == {"weight": 0.5}


# ============================================================
# StructFieldSchema Tests
# ============================================================


class TestStructFieldSchemaCreation:
    """Tests for StructFieldSchema creation."""

    def test_create_struct_schema(self):
        """Test creating StructFieldSchema."""
        struct = StructFieldSchema()
        struct.add_field("name", DataType.VARCHAR, max_length=256)
        struct.add_field("age", DataType.INT32)
        assert len(struct.fields) == 2

    def test_struct_dtype_property(self):
        """Test dtype property returns STRUCT."""
        struct = StructFieldSchema()
        assert struct.dtype == DataType.STRUCT


class TestStructFieldSchemaValidation:
    """Tests for StructFieldSchema validation."""

    def test_check_fields_empty(self):
        """Test struct with no fields raises error."""
        struct = StructFieldSchema()
        struct.name = "test"
        with pytest.raises(ParamError, match="at least one"):
            struct._check_fields()

    @pytest.mark.parametrize(
        "kwarg,error_match",
        [
            pytest.param({"is_primary": True}, "primary key", id="primary"),
            pytest.param({"is_partition_key": True}, "partition key", id="partition"),
            pytest.param({"is_clustering_key": True}, "clustering key", id="clustering"),
            pytest.param({"is_dynamic": True}, "dynamic", id="dynamic"),
            pytest.param({"nullable": True}, "nullable", id="nullable"),
        ],
    )
    def test_check_fields_invalid_properties(self, kwarg, error_match):
        """Test struct fields cannot have certain properties."""
        struct = StructFieldSchema()
        struct.name = "test"
        struct._fields = [FieldSchema("field", DataType.INT64, **kwarg)]
        with pytest.raises(ParamError, match=error_match):
            struct._check_fields()

    def test_check_fields_duplicate_names(self):
        """Test struct with duplicate field names raises error."""
        struct = StructFieldSchema()
        struct.name = "test"
        struct._fields = [
            FieldSchema("dup", DataType.INT64),
            FieldSchema("dup", DataType.FLOAT),
        ]
        with pytest.raises(ParamError, match=r"[Dd]uplicate"):
            struct._check_fields()

    def test_add_field_unsupported_types(self):
        """Test adding unsupported types to struct."""
        struct = StructFieldSchema()
        with pytest.raises(ParamError, match="does not support"):
            struct.add_field("arr", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)


class TestStructFieldSchemaToDict:
    """Tests for StructFieldSchema to_dict and construct_from_dict methods."""

    def test_to_dict(self):
        """Test StructFieldSchema.to_dict."""
        struct = StructFieldSchema()
        struct.name = "test_struct"
        struct.max_capacity = 10
        struct.add_field("score", DataType.FLOAT)

        result = struct.to_dict()
        assert result["name"] == "test_struct"
        assert result["max_capacity"] == 10
        assert len(result["fields"]) == 1

    @pytest.mark.parametrize(
        "d",
        [
            pytest.param(
                {
                    "name": "test",
                    "max_capacity": 5,
                    "fields": [{"name": "score", "type": DataType.FLOAT, "description": ""}],
                },
                id="with_fields",
            ),
            pytest.param(
                {
                    "name": "test",
                    "max_capacity": 5,
                    "struct_fields": [{"name": "score", "type": DataType.FLOAT, "params": {}}],
                },
                id="with_struct_fields",
            ),
            pytest.param(
                {
                    "name": "test",
                    "params": {"max_capacity": 10, "mmap_enabled": True},
                    "fields": [{"name": "score", "type": DataType.FLOAT, "description": ""}],
                },
                id="with_params",
            ),
        ],
    )
    def test_construct_from_dict(self, d):
        """Test StructFieldSchema.construct_from_dict."""
        struct = StructFieldSchema.construct_from_dict(d)
        assert struct.name == d["name"]
        assert len(struct.fields) >= 1

    def test_reused_struct_schema(self):
        """Test reusing struct_schema for multiple fields works correctly."""
        struct_schema = StructFieldSchema()
        struct_schema.add_field("score", DataType.FLOAT)

        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )

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

        assert len(schema.struct_fields) == 2
        assert schema.struct_fields[0].name == "field1"
        assert schema.struct_fields[1].name == "field2"
        # Original not modified
        assert struct_schema.name == ""


# ============================================================
# Highlighter Tests
# ============================================================


class TestLexicalHighlighter:
    """Tests for LexicalHighlighter class."""

    @pytest.mark.parametrize(
        "kwargs,expected_key",
        [
            pytest.param({"fragment_offset": 10}, "fragment_offset", id="fragment_offset"),
            pytest.param({"fragment_size": 100}, "fragment_size", id="fragment_size"),
            pytest.param({"num_of_fragments": 3}, "num_of_fragments", id="num_of_fragments"),
            pytest.param(
                {"highlight_search_text": True}, "highlight_search_text", id="highlight_search_text"
            ),
            pytest.param({"pre_tags": ["<b>"], "post_tags": ["</b>"]}, "pre_tags", id="tags"),
        ],
    )
    def test_params_included(self, kwargs, expected_key):
        """Test various params are included in output."""
        highlighter = LexicalHighlighter(**kwargs)
        params = highlighter.params
        assert expected_key in params

    def test_type_property(self):
        """Test type property returns LEXICAL."""
        highlighter = LexicalHighlighter()
        assert highlighter.type == HighlightType.LEXICAL

    def test_with_query(self):
        """Test with_query method."""
        highlighter = LexicalHighlighter()
        highlighter.with_query("text", "search term", "match")
        assert highlighter.highlight_query is not None
        assert len(highlighter.highlight_query) == 1
        assert highlighter.highlight_query[0]["field"] == "text"

    def test_with_query_appends(self):
        """Test with_query appends to existing list."""
        highlighter = LexicalHighlighter(highlight_query=[{"field": "existing"}])
        highlighter.with_query("text", "hello", "match")
        assert len(highlighter.highlight_query) == 2


class TestSemanticHighlighter:
    """Tests for SemanticHighlighter."""

    def test_create_basic(self):
        """Test creating basic SemanticHighlighter."""
        highlighter = SemanticHighlighter(queries=["test"], input_fields=["text"])
        assert highlighter.queries == ["test"]
        assert highlighter.input_fields == ["text"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"pre_tags": ["<b>"], "post_tags": ["</b>"]}, id="tags"),
            pytest.param({"threshold": 0.5}, id="threshold"),
            pytest.param({"highlight_only": True}, id="highlight_only"),
            pytest.param({"model_deployment_id": "model-123"}, id="model_id"),
            pytest.param({"max_client_batch_size": 100}, id="batch_size"),
        ],
    )
    def test_optional_params(self, kwargs):
        """Test optional parameters."""
        highlighter = SemanticHighlighter(queries=["test"], input_fields=["text"], **kwargs)
        params = highlighter.params
        for key in kwargs:
            assert key in params

    def test_type_property(self):
        """Test type property returns SEMANTIC."""
        highlighter = SemanticHighlighter(queries=["test"], input_fields=["text"])
        assert highlighter.type == HighlightType.SEMANTIC


# ============================================================
# Validation Functions Tests
# ============================================================


class TestValidatePrimaryKey:
    """Tests for validate_primary_key function."""

    def test_none_primary_field(self):
        """Test validation with None primary field."""
        with pytest.raises(PrimaryKeyException, match=r"[Pp]rimary"):
            validate_primary_key(None)

    @pytest.mark.parametrize(
        "dtype,valid",
        [
            pytest.param(DataType.INT64, True, id="int64_valid"),
            pytest.param(DataType.VARCHAR, True, id="varchar_valid"),
            pytest.param(DataType.FLOAT, False, id="float_invalid"),
            pytest.param(DataType.DOUBLE, False, id="double_invalid"),
            pytest.param(DataType.BOOL, False, id="bool_invalid"),
            pytest.param(DataType.INT32, False, id="int32_invalid"),
        ],
    )
    def test_primary_key_types(self, dtype, valid):
        """Test valid/invalid primary key types."""
        kwargs = {"max_length": 100} if dtype == DataType.VARCHAR else {}
        field = FieldSchema("pk", dtype, is_primary=True, **kwargs)
        if valid:
            validate_primary_key(field)  # Should not raise
        else:
            with pytest.raises(PrimaryKeyException):
                validate_primary_key(field)


class TestValidatePartitionKey:
    """Tests for validate_partition_key function."""

    @pytest.mark.parametrize(
        "dtype,valid",
        [
            pytest.param(DataType.INT64, True, id="int64_valid"),
            pytest.param(DataType.VARCHAR, True, id="varchar_valid"),
            pytest.param(DataType.FLOAT, False, id="float_invalid"),
            pytest.param(DataType.DOUBLE, False, id="double_invalid"),
            pytest.param(DataType.BOOL, False, id="bool_invalid"),
        ],
    )
    def test_partition_key_types(self, dtype, valid):
        """Test valid/invalid partition key types."""
        kwargs = {"max_length": 100} if dtype == DataType.VARCHAR else {}
        field = FieldSchema("pk", dtype, is_partition_key=True, **kwargs)
        if valid:
            validate_partition_key("pk", field, "id")  # Should not raise
        else:
            with pytest.raises(PartitionKeyException):
                validate_partition_key("pk", field, "id")

    def test_partition_key_field_not_exist(self):
        """Test partition key field not found."""
        with pytest.raises(PartitionKeyException, match="not exist"):
            validate_partition_key("missing_field", None, "id")


class TestValidateClusteringKey:
    """Tests for validate_clustering_key function."""

    @pytest.mark.parametrize(
        "dtype,valid",
        [
            pytest.param(DataType.INT8, True, id="int8_valid"),
            pytest.param(DataType.INT16, True, id="int16_valid"),
            pytest.param(DataType.INT32, True, id="int32_valid"),
            pytest.param(DataType.INT64, True, id="int64_valid"),
            pytest.param(DataType.FLOAT, True, id="float_valid"),
            pytest.param(DataType.DOUBLE, True, id="double_valid"),
            pytest.param(DataType.VARCHAR, True, id="varchar_valid"),
            pytest.param(DataType.FLOAT_VECTOR, True, id="float_vector_valid"),
            pytest.param(DataType.BOOL, False, id="bool_invalid"),
            pytest.param(DataType.JSON, False, id="json_invalid"),
        ],
    )
    def test_clustering_key_types(self, dtype, valid):
        """Test valid/invalid clustering key types."""
        kwargs = {"max_length": 100} if dtype == DataType.VARCHAR else {}
        if dtype == DataType.FLOAT_VECTOR:
            kwargs["dim"] = 128
        field = FieldSchema("ck", dtype, is_clustering_key=True, **kwargs)
        if valid:
            validate_clustering_key("ck", field)  # Should not raise
        else:
            with pytest.raises(ClusteringKeyException):
                validate_clustering_key("ck", field)

    def test_clustering_key_field_not_exist(self):
        """Test clustering key field not found."""
        with pytest.raises(ClusteringKeyException, match="not exist"):
            validate_clustering_key("missing_field", None)


# ============================================================
# Helper Functions Tests
# ============================================================


class TestIsVectorDataType:
    """Tests for isVectorDataType function."""

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            pytest.param(DataType.FLOAT_VECTOR, True, id="float_vector"),
            pytest.param(DataType.BINARY_VECTOR, True, id="binary_vector"),
            pytest.param(DataType.FLOAT16_VECTOR, True, id="float16_vector"),
            pytest.param(DataType.BFLOAT16_VECTOR, True, id="bfloat16_vector"),
            pytest.param(DataType.INT8_VECTOR, True, id="int8_vector"),
            pytest.param(DataType.SPARSE_FLOAT_VECTOR, True, id="sparse_vector"),
            pytest.param(DataType.INT64, False, id="int64"),
            pytest.param(DataType.VARCHAR, False, id="varchar"),
        ],
    )
    def test_is_vector_data_type(self, dtype, expected):
        """Test isVectorDataType for various types."""
        assert isVectorDataType(dtype) == expected


class TestIsValidInsertData:
    """Tests for is_valid_insert_data function."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            pytest.param(pd.DataFrame({"a": [1]}), True, id="dataframe"),
            pytest.param([1, 2, 3], True, id="list"),
            pytest.param({"a": 1}, True, id="dict"),
            pytest.param("string", False, id="string"),
            pytest.param(123, False, id="int"),
        ],
    )
    def test_is_valid_insert_data(self, data, expected):
        """Test is_valid_insert_data for various types."""
        assert is_valid_insert_data(data) == expected


class TestIsRowBased:
    """Tests for is_row_based and check_is_row_based functions."""

    @pytest.mark.parametrize(
        "data,expected",
        [
            pytest.param({"a": 1}, True, id="dict"),
            pytest.param([{"a": 1}], True, id="list_of_dict"),
            pytest.param([[1, 2]], False, id="list_of_list"),
            pytest.param([], False, id="empty_list"),
        ],
    )
    def test_is_row_based(self, data, expected):
        """Test is_row_based for various types."""
        assert is_row_based(data) == expected

    @pytest.mark.parametrize(
        "data,expected",
        [
            pytest.param({"a": 1}, True, id="dict"),
            pytest.param([{"a": 1}], True, id="list_of_dict"),
            pytest.param([[1, 2]], False, id="list_of_list"),
            pytest.param(pd.DataFrame({"a": [1]}), False, id="dataframe"),
            pytest.param([], False, id="empty_list"),
        ],
    )
    def test_check_is_row_based(self, data, expected):
        """Test check_is_row_based for various types."""
        assert check_is_row_based(data) == expected

    def test_check_is_row_based_invalid_type(self):
        """Test check_is_row_based with invalid type."""
        with pytest.raises(DataTypeNotSupportException):
            check_is_row_based("invalid")


class TestCheckInsertSchema:
    """Tests for check_insert_schema function."""

    def test_check_insert_schema_none(self):
        """Test with None schema."""
        with pytest.raises(SchemaNotReadyException):
            check_insert_schema(None, [[1]])

    def test_check_insert_schema_valid(self, basic_schema):
        """Test with valid schema and data."""
        data = [[1, 2], [[1.0] * 128, [2.0] * 128]]
        check_insert_schema(basic_schema, data)  # Should not raise

    def test_check_insert_schema_auto_id_with_data(self):
        """Test with auto_id and provided pk data in DataFrame."""
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=2),
            ]
        )
        insert_data = pd.DataFrame({"id": [1, 2], "vec": [[1.0, 2.0], [3.0, 4.0]]})
        with pytest.raises(DataNotMatchException, match="auto_id"):
            check_insert_schema(schema, insert_data)


class TestCheckUpsertSchema:
    """Tests for check_upsert_schema function."""

    def test_check_upsert_schema_none(self):
        """Test with None schema."""
        with pytest.raises(SchemaNotReadyException):
            check_upsert_schema(None, [[1]])

    def test_check_upsert_schema_missing_pk(self, basic_schema):
        """Test upsert with missing primary key in DataFrame."""
        upsert_data = pd.DataFrame({"vec": [[1.0] * 128, [2.0] * 128]})
        with pytest.raises(DataNotMatchException, match="pk"):
            check_upsert_schema(basic_schema, upsert_data)


class TestCheckSchema:
    """Tests for check_schema function."""

    def test_check_schema_none(self):
        """Test with None schema."""
        with pytest.raises(SchemaNotReadyException):
            check_schema(None)

    def test_check_schema_empty_fields(self):
        """Test with empty fields."""
        schema = CollectionSchema(
            [FieldSchema("id", DataType.INT64, is_primary=True)], check_fields=False
        )
        schema._fields = []
        with pytest.raises(SchemaNotReadyException, match=r"[Ee]mpty"):
            check_schema(schema)

    def test_check_schema_no_vector(self):
        """Test schema without vector field."""
        schema = CollectionSchema(
            [FieldSchema("id", DataType.INT64, is_primary=True)], check_fields=False
        )
        with pytest.raises(SchemaNotReadyException, match=r"[Vv]ector"):
            check_schema(schema)


class TestPrepareFieldsFromDataframe:
    """Tests for prepare_fields_from_dataframe function."""

    def test_empty_dataframe_unknown_dtype(self):
        """Test with empty DataFrame having unknown dtype."""
        empty_frame = pd.DataFrame({"col": pd.array([], dtype=object)})
        with pytest.raises(CannotInferSchemaException):
            prepare_fields_from_dataframe(empty_frame)

    @pytest.mark.parametrize(
        "vec_data,expected_dtype",
        [
            pytest.param(b"\x01\x02", DataType.BINARY_VECTOR, id="binary_vector"),
            pytest.param(
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                DataType.FLOAT_VECTOR,
                id="float_vector",
            ),
        ],
    )
    def test_dataframe_with_vectors(self, vec_data, expected_dtype):
        """Test DataFrame with vector columns."""
        vec_frame = pd.DataFrame({"id": [1], "vec": [vec_data]})
        _, data_types, params = prepare_fields_from_dataframe(vec_frame)
        assert expected_dtype in data_types
        assert "vec" in params


class TestInferDefaultValueByData:
    """Tests for infer_default_value_bydata function."""

    def test_infer_none(self):
        """Test with None returns None."""
        assert infer_default_value_bydata(None) is None

    @pytest.mark.parametrize(
        "data,dtype,attr",
        [
            pytest.param(True, DataType.BOOL, "bool_data", id="bool"),
            pytest.param(42, DataType.INT64, "long_data", id="int64"),
            pytest.param(3.14159, DataType.DOUBLE, "double_data", id="double"),
            pytest.param("hello", DataType.VARCHAR, "string_data", id="varchar"),
        ],
    )
    def test_infer_scalar_types(self, data, dtype, attr):
        """Test inferring default value for scalar types."""
        result = infer_default_value_bydata(data, dtype)
        assert getattr(result, attr) == data

    def test_infer_float(self):
        """Test inferring FLOAT default value."""
        result = infer_default_value_bydata(3.14, DataType.FLOAT)
        assert abs(result.float_data - 3.14) < 0.001

    def test_infer_unsupported_type(self):
        """Test unsupported type raises error."""
        with pytest.raises(ParamError, match=r"[Uu]nsupported"):
            infer_default_value_bydata([1, 2, 3], DataType.ARRAY)
