import pytest
import json

from pymilvus.client.constants import PAGE_RETAIN_ORDER_FIELD
from pymilvus.client.prepare import Prepare
from pymilvus import DataType, MilvusException, CollectionSchema, FieldSchema
from pymilvus import DefaultConfig


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
            "params": {"page_retain_order": True}
        }

        ret = Prepare.search_requests_with_expr("name", data, "v", search_params, 100)

        offset_exists = False
        page_retain_order_exists = False
        print(ret.search_params)
        for p in ret.search_params:
            if p.key == "offset":
                offset_exists = True
                assert p.value == "10"
            elif p.key == "params":
                params = json.loads(p.value)
                if PAGE_RETAIN_ORDER_FIELD in params:
                    page_retain_order_exists = True
                    assert  params[PAGE_RETAIN_ORDER_FIELD] == True

        assert offset_exists is True
        assert page_retain_order_exists is True


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
            {"name": "test_floatvector", "type": DataType.FLOAT_VECTOR,
             "params": {"dim": 128, DefaultConfig.MaxVarCharLengthKey: DefaultConfig.MaxVarCharLength + 1}},
        ]}
    ])
    def test_valid_type_params_get_collection_schema(self, valid_fields):
        schema = Prepare.get_schema("test_name", valid_fields)
        assert len(schema.fields) == 1
        assert schema.fields[0].name == valid_fields["fields"][0]["name"]

    def test_get_schema_from_collection_schema(self):
        schema = CollectionSchema([
            FieldSchema("field_vector", DataType.FLOAT_VECTOR, dim=8),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True),
        ])

        c_schema = Prepare.get_schema_from_collection_schema("random", schema)

        assert c_schema.enable_dynamic_field == False
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

    def test_get_schema_from_collection_schema_with_enable_dynamic_field(self):
        schema = CollectionSchema([
            FieldSchema("field_vector", DataType.FLOAT_VECTOR, dim=8),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True),
        ], enable_dynamic_field=True)

        c_schema = Prepare.get_schema_from_collection_schema("random", schema)

        assert c_schema.enable_dynamic_field, "should enable dynamic field"
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

    @pytest.mark.parametrize("kv", [
        {"shards_num": 1},
        {"num_shards": 2},
    ])
    def test_create_collection_request_num_shards(self, kv):
        schema = CollectionSchema([
            FieldSchema("field_vector", DataType.FLOAT_VECTOR, dim=8),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True)
        ])
        req = Prepare.create_collection_request("c_name", schema, **kv)
        assert req.shards_num == list(kv.values())[0]

    @pytest.mark.parametrize("kv", [
        {"shards_num": 1, "num_shards": 1},
        {"num_shards": "2"},
    ])
    def test_create_collection_request_num_shards_error(self, kv):
        schema = CollectionSchema([
            FieldSchema("field_vector", DataType.FLOAT_VECTOR, dim=8),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True)
        ])

        with pytest.raises(MilvusException):
            req = Prepare.create_collection_request("c_name", schema, **kv)

    def test_row_insert_param_with_auto_id(self):
        import numpy as np
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE)
        ])
        rows = [
            {"float": 1.0, "float_vector": rng.random((1, dim))[0], "a": 1},
            {"float": 1.0, "float_vector": rng.random((1, dim))[0], "b": 1},
        ]

        Prepare.row_insert_param("", rows, "", fields_info=schema.to_dict()["fields"], enable_dynamic=True)

    def test_row_upsert_param_with_auto_id(self):
        import numpy as np
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE)
        ])
        rows = [
            {"pk_field":1, "float": 1.0, "float_vector": rng.random((1, dim))[0], "a": 1},
            {"pk_field":2, "float": 1.0, "float_vector": rng.random((1, dim))[0], "b": 1},
        ]

        Prepare.row_upsert_param("", rows, "", fields_info=schema.to_dict()["fields"], enable_dynamic=True)

class TestAlterCollectionRequest:
    def test_alter_collection_request(self):
        schema = Prepare.alter_collection_request('foo', {'collection.ttl.seconds': 1800})

