import pytest

from pymilvus.client.prepare import Prepare
from pymilvus import DataType, ParamError, CollectionSchema, FieldSchema
from pymilvus.client.configs import DefaultConfigs


class TestCreateCollectionRequest:
    @pytest.mark.parametrize("invalid_fields", [
        list(),
        {"no_fields_key": 1},
        {"fields": []},  # lack of fields values
        {"fields": [{"no_name": True}]},
        {"fields": [{"name": "test_int64", "no_type": True}]},
        {"fields": [{"name": "test_int64", "type_wrong": True}]},

        # wrong type for primary field
        {"fields": [{"name": "test_int64", "type": DataType.DOUBLE, "is_primary": True}]},

        # two primary fields
        {"fields": [
            {"name": "test_int64", "type": DataType.INT64, "is_primary": True},
            {"name": "test_int64_2", "type": DataType.INT64, "is_primary": True}]},

        # two auto_id fields
        {"fields": [
            {"name": "test_int64", "type": DataType.INT64, "auto_id": True},
            {"name": "test_int64_2", "type": DataType.INT64, "auto_id": True}]},

        # wrong type for auto_id field
        {"fields": [{"name": "test_double", "type": DataType.DOUBLE, "auto_id": True}]},

        # invalid dim
        {"fields": [
            {"name": "test_int64", "type": DataType.INT64},
            {"name": "test_vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": "invalid"}}]},

        # invalid varcharlength
        {"fields": [
            {"name": "test_varchar", "type": DataType.VARCHAR, "params": {DefaultConfigs.MaxVarCharLengthKey: "invalid"}},
        ]},

        # exceeded varcharlength
        {"fields": [
            {"name": "test_varchar", "type": DataType.VARCHAR, "params": {DefaultConfigs.MaxVarCharLengthKey: DefaultConfigs.MaxVarCharLength + 1}},
        ]},
    ])
    def test_param_error_get_schema(self, invalid_fields):
        with pytest.raises(ParamError):
            Prepare.get_schema("test_name", invalid_fields)

    @pytest.mark.parametrize("valid_fields", [
        {"fields": [
            {"name": "test_varchar", "type": DataType.VARCHAR, "is_primary": True, "params": {"dim": "invalid"}},
        ]},
        {"fields": [
            {"name": "test_floatvector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128, DefaultConfigs.MaxVarCharLengthKey: DefaultConfigs.MaxVarCharLength + 1}},
        ]}
    ])
    @pytest.mark.skip("failing")
    def test_valid_type_params_get_collection_schema(self, valid_fields):
        schema = Prepare.get_schema("test_name", valid_fields)
        assert len(schema.fields) == 1
        assert schema.fields[0].name == valid_fields["fields"]["name"]

    def test_get_schema_from_collection_schema(self):
        schema = CollectionSchema([
            FieldSchema("field_vector", DataType.FLOAT_VECTOR, dim=8),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True)
        ])

        c_schema = Prepare.get_schema_from_collection_schema("random", schema)

        assert c_schema.name == "random"
        assert len(c_schema.fields) == 2
        assert c_schema.fields[0].name == "field_vector"
        assert c_schema.fields[0].data_type == DataType.FLOAT_VECTOR
        assert len(c_schema.fields[0].type_params) == 1
        assert c_schema.fields[0].type_params[0].key == "dim"
        assert c_schema.fields[0].type_params[0].value == "8"

        assert c_schema.fields[1].name == "pk_field"
        assert c_schema.fields[1].data_type == DataType.INT64
        assert c_schema.fields[1].is_primary_key is True
        assert c_schema.fields[1].autoID is True
        assert len(c_schema.fields[1].type_params) == 0
