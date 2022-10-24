import pytest

from pymilvus.client.prepare import Prepare
from pymilvus import DataType, MilvusException, CollectionSchema, FieldSchema
from pymilvus.client.configs import DefaultConfigs


class TestPrepare:
    def test_search_requests_with_expr_offset(self):
        fields = [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("v", DataType.FLOAT_VECTOR, dim=2),
        ]

        schema = CollectionSchema(fields).to_dict()
        data = [
            [1., 2.],
            [1., 2.],
            [1., 2.],
            [1., 2.],
            [1., 2.],
        ]

        search_params = {
            "metric_type": "L2",
            "offset": 10,
        }

        ret = Prepare.search_requests_with_expr("name", data, "v", search_params, 100, schema=schema)

        offset_exists = False
        for p in ret[0].search_params:
            if p.key == "offset":
                offset_exists = True
                assert p.value == "10"

        assert offset_exists is True


class TestCreateCollectionRequest:
    @pytest.mark.parametrize("valid_properties", [
        {"properties": {"p1": "o1"}},
        {"properties": {}},
        {"properties": {"p2": "o2", "p3": "o3"}},
    ])
    def test_create_collection_with_properties(self, valid_properties):
        schema = CollectionSchema([
            FieldSchema("field_vector", DataType.FLOAT_VECTOR, dim=8),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True)
        ])
        req = Prepare.create_collection_request("c_name", schema, **valid_properties)
        
        assert len(valid_properties.get("properties")) == len(req.properties)

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
    ])
    def test_param_error_get_schema(self, invalid_fields):
        with pytest.raises(MilvusException):
            Prepare.get_schema("test_name", invalid_fields)

    @pytest.mark.parametrize("valid_fields", [
        {"fields": [
            {"name": "test_varchar", "type": DataType.VARCHAR, "is_primary": True, "params": {"dim": "invalid"}},
        ]},
        {"fields": [
            {"name": "test_floatvector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128, DefaultConfigs.MaxVarCharLengthKey: DefaultConfigs.MaxVarCharLength + 1}},
        ]}
    ])
    def test_valid_type_params_get_collection_schema(self, valid_fields):
        schema = Prepare.get_schema("test_name", valid_fields)
        assert len(schema.fields) == 1
        assert schema.fields[0].name == valid_fields["fields"][0]["name"]

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
