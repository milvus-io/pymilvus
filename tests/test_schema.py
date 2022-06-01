import logging
import pytest

from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusException
from pymilvus.client.configs import DefaultConfigs

LOGGER = logging.getLogger(__name__)


class TestCollectionSchema:
    @pytest.fixture(scope="function")
    def default_fields(self, request):
        return [
            FieldSchema("vector_field", DataType.FLOAT_VECTOR, dim=128),
            FieldSchema("primary_field", DataType.INT64, is_primary=True, auto_id=True),
        ]

    def test_init_collcetion_schema(self, default_fields):
        s = CollectionSchema(default_fields, description="Normal cases")
        assert s.auto_id == default_fields[1].auto_id
        assert s.primary_field == default_fields[1]
        assert s.description == "Normal cases"
        assert s.fields == default_fields

    def test_init_primary_field_failed(self, default_fields):
        with pytest.raises(MilvusException):
            CollectionSchema(default_fields, description="name not in fields", primary_field="random-name")

        default_fields.append(FieldSchema("double_field", DataType.DOUBLE))
        with pytest.raises(MilvusException):
            CollectionSchema(default_fields, description="field_cannot_be_primary_field", primary_field="double_field")

        default_fields.append(FieldSchema("int64_field", DataType.INT64))
        with pytest.raises(MilvusException):
            CollectionSchema(default_fields, description="two_pk_field", primary_field="int64_field")


class TestFieldSchema:
    get_fields_dict = [
        {
            "name": "TestFieldSchema_name_floatvector",
            "type": DataType.FLOAT_VECTOR,
            "params": {"dim": 128},
            "description": "TestFieldSchema_description_floatvector",
        },
        {
            "name": "TestFieldSchema_name_varchar",
            "type": DataType.VARCHAR,
            "params": {DefaultConfigs.MaxVarCharLengthKey: 128},
            "description": "TestFieldSchema_description_varchar",
        },
        {
            "name": "TestFieldSchema_name_binary_vector",
            "type": DataType.BINARY_VECTOR,
            "params": {"dim": 128},
            "description": "TestFieldSchema_description_binary_vector",
        },
        {
            "name": "TestFieldSchema_name_norm",
            "type": DataType.INT64,
            "description": "TestFieldSchema_description_norm",
        },
        {
            "name": "TestFieldSchema_name_primary",
            "type": DataType.INT64,
            "is_primary": True,
            "description": "TestFieldSchema_description_primary",
        },
        {
            "name": "TestFieldSchema_name_primary_auto_id",
            "type": DataType.INT64,
            "is_primary": True,
            "auto_id": True,
            "description": "TestFieldSchema_description_primary_auto_id",
        },
    ]

    @pytest.fixture(scope="function", params=get_fields_dict)
    def field_dict(self, request):
        yield request.param

    def test_constructor_from_dict(self, field_dict):
        field = FieldSchema.construct_from_dict(field_dict)
        assert field.name == field_dict['name']
        assert field.dtype == field_dict["type"]
        assert field.description == field_dict['description']
        assert field.is_primary is field_dict.get("is_primary", False)
        assert field.params == field_dict.get('params', {})

    @pytest.mark.parametrize("invalid_params", [
        {"is_primary": "not bool", "dtype": DataType.INT8},
        {"is_primary": True, "dtype": DataType.BOOL},
        {"is_primary": True, "dtype": DataType.NONE},
        {"is_primary": True, "dtype": DataType.INT8},
        {"is_primary": True, "dtype": DataType.INT16},
        {"is_primary": True, "dtype": DataType.INT32},
        {"is_primary": True, "dtype": DataType.FLOAT},
        {"is_primary": True, "dtype": DataType.DOUBLE},
        {"is_primary": True, "dtype": DataType.FLOAT_VECTOR},
        {"is_primary": True, "dtype": DataType.BINARY_VECTOR},
    ])
    def test_check_valid_is_primary_error(self, invalid_params):
        with pytest.raises(MilvusException):
            FieldSchema._init_is_primary(**invalid_params)

    @pytest.mark.parametrize("invalid_params", [
        {"auto_id": True, "is_primary": False, "dtype": DataType.INT64},
        {"auto_id": "not bool", "is_primary": True, "dtype": DataType.INT64},
    ])
    def test_check_valid_auto_id(self, invalid_params):
        with pytest.raises(MilvusException):
            FieldSchema._init_auto_id(**invalid_params)
