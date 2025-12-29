import pytest
import json
import logging

import numpy as np
import pytest

from pymilvus import Function, FunctionType
from pymilvus import CollectionSchema, DataType, DefaultConfig, FieldSchema, MilvusException
from pymilvus.client.constants import PAGE_RETAIN_ORDER_FIELD
from pymilvus.client.prepare import Prepare

LOGGER = logging.getLogger(__name__)


class TestPrepare:
    @pytest.mark.parametrize("coll_name", [None, "", -1, 1.1, []])
    @pytest.mark.parametrize("expr", [None, "", -1, 1.1, []])
    def test_delete_request_wrong_coll_name(self, coll_name: str, expr: str):
        with pytest.raises(MilvusException):
            Prepare.delete_request(coll_name, expr, None, 0)

    @pytest.mark.parametrize("part_name", [])
    def test_delete_request_wrong_part_name(self, part_name):
        with pytest.raises(MilvusException):
            Prepare.delete_request("coll", "id>1", part_name, 0)

    def test_search_requests_with_expr_offset(self):
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

        ret = Prepare.search_requests_with_expr(
            collection_name="name", 
            data=data, 
            anns_field="v", 
            param=search_params, 
            limit=100,
        )

        offset_exists = False
        page_retain_order_exists = False
        LOGGER.info(ret.search_params)
        for p in ret.search_params:
            if p.key == "offset":
                offset_exists = True
                assert p.value == "10"
            elif p.key == "params":
                params = json.loads(p.value)
                if PAGE_RETAIN_ORDER_FIELD in params:
                    page_retain_order_exists = True
                    assert params[PAGE_RETAIN_ORDER_FIELD] is True

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
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True)
        ])
        req = Prepare.create_collection_request(
            "c_name", schema, **valid_properties)
        assert len(valid_properties.get("properties")) == len(req.properties)

    @pytest.mark.parametrize("invalid_fields", [
        [],
        {"no_fields_key": 1},
        {"fields": []},  # lack of fields values
        {"fields": [{"no_name": True}]},
        {"fields": [{"name": "test_int64", "no_type": True}]},
        {"fields": [{"name": "test_int64", "type_wrong": True}]},

        # wrong type for primary field
        {"fields": [{"name": "test_int64",
                     "type": DataType.DOUBLE, "is_primary": True}]},

        # two primary fields
        {"fields": [
            {"name": "test_int64", "type": DataType.INT64, "is_primary": True},
            {"name": "test_int64_2", "type": DataType.INT64, "is_primary": True}]},

        # two auto_id fields
        {"fields": [
            {"name": "test_int64", "type": DataType.INT64, "auto_id": True},
            {"name": "test_int64_2", "type": DataType.INT64, "auto_id": True}]},

        # wrong type for auto_id field
        {"fields": [{"name": "test_double",
                     "type": DataType.DOUBLE, "auto_id": True}]},
    ])
    def test_param_error_get_schema(self, invalid_fields):
        with pytest.raises(MilvusException):
            Prepare.get_schema("test_name", invalid_fields)

    @pytest.mark.parametrize("valid_fields", [
        {"fields": [
            {"name": "test_varchar", "type": DataType.VARCHAR,
                "is_primary": True, "params": {"dim": "invalid"}},
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
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
        ])

        c_schema = Prepare.get_schema_from_collection_schema("random", schema)

        assert c_schema.enable_dynamic_field is False
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
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
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
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True)
        ])
        req = Prepare.create_collection_request("c_name", schema, **kv)
        assert req.shards_num == next(iter(kv.values()))

    @pytest.mark.parametrize("kv", [
        {"shards_num": 1, "num_shards": 1},
        {"num_shards": "2"},
    ])
    def test_create_collection_request_num_shards_error(self, kv):
        schema = CollectionSchema([
            FieldSchema("field_vector", DataType.FLOAT_VECTOR, dim=8),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True)
        ])

        with pytest.raises(MilvusException):
            Prepare.create_collection_request("c_name", schema, **kv)

    def test_row_insert_param_with_auto_id(self):
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE)
        ])
        rows = [
            {"float": 1.0, "float_vector": rng.random((1, dim))[0], "a": 1},
            {"float": 1.0, "float_vector": rng.random((1, dim))[0], "b": 1},
        ]

        Prepare.row_insert_param("", rows, "", fields_info=schema.to_dict()[
                                 "fields"], enable_dynamic=True)

    def test_row_insert_param_with_none(self):
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("nullable_field", DataType.INT64, nullable=True),
            FieldSchema("default_field", DataType.FLOAT, default_value=10),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE),
        ])
        rows = [
            {"float": 1.0, "nullable_field": None, "default_field": None,
                "float_vector": rng.random((1, dim))[0], "a": 1},
            {"float": 1.0, "float_vector": rng.random((1, dim))[0], "b": 1},
        ]

        Prepare.row_insert_param("", rows, "", fields_info=schema.to_dict()[
                                 "fields"], enable_dynamic=True)

    def test_row_upsert_param_with_auto_id(self):
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE)
        ])
        rows = [
            {"pk_field": 1, "float": 1.0, "float_vector": rng.random((1, dim))[
                0], "a": 1},
            {"pk_field": 2, "float": 1.0, "float_vector": rng.random((1, dim))[
                0], "b": 1},
        ]

        Prepare.row_upsert_param("", rows, "", fields_info=schema.to_dict()[
                                 "fields"], enable_dynamic=True)

    def test_upsert_param_with_none(self):
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("nullable_field", DataType.INT64, nullable=True),
            FieldSchema("default_field", DataType.FLOAT, default_value=10),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE),
        ])
        rows = [
            {"pk_field": 1, "float": 1.0, "nullable_field": None,
                "default_field": None, "float_vector": rng.random((1, dim))[0], "a": 1},
            {"pk_field": 2, "float": 1.0, "float_vector": rng.random((1, dim))[
                0], "b": 1},
        ]

        Prepare.row_upsert_param("", rows, "", fields_info=schema.to_dict()[
                                 "fields"], enable_dynamic=True)

    def test_row_upsert_param_with_partial_update_true(self):
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE)
        ])
        rows = [
            {"pk_field": 1, "float": 1.0, "float_vector": rng.random((1, dim))[
                0], "a": 1},
            {"pk_field": 2, "float": 1.0, "float_vector": rng.random((1, dim))[
                0], "b": 1},
        ]

        request = Prepare.row_upsert_param("test_collection", rows, "",
                                           fields_info=schema.to_dict()[
                                               "fields"],
                                           enable_dynamic=True,
                                           partial_update=True)

        # Check that partial_update is set correctly
        assert hasattr(
            request, 'partial_update'), "UpsertRequest should have partial_update field"
        assert request.partial_update is True, "partial_update should be True when explicitly set to True"

    def test_row_upsert_param_partial_update_default(self):
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE)
        ])
        rows = [
            {"pk_field": 1, "float": 1.0, "float_vector": rng.random((1, dim))[
                0], "a": 1},
            {"pk_field": 2, "float": 1.0, "float_vector": rng.random((1, dim))[
                0], "b": 1},
        ]

        request = Prepare.row_upsert_param("test_collection", rows, "",
                                           fields_info=schema.to_dict()[
                                               "fields"],
                                           enable_dynamic=True)

        # Check that partial_update defaults to False
        assert hasattr(
            request, 'partial_update'), "UpsertRequest should have partial_update field"
        assert request.partial_update is False, "partial_update should default to False"

    def test_batch_upsert_param_with_partial_update_true(self):
        entities = [
            {"name": "id", "type": DataType.INT64, "values": [1, 2, 3]},
            {"name": "name", "type": DataType.VARCHAR,
                "values": ["a", "b", "c"]},
            {"name": "float_vector", "type": DataType.FLOAT_VECTOR,
                "values": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
        ]

        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "name", "type": DataType.VARCHAR},
            {"name": "float_vector", "type": DataType.FLOAT_VECTOR, "dim": 2}
        ]

        request = Prepare.batch_upsert_param("test_collection", entities, "",
                                             fields_info, partial_update=True)

        # Check that partial_update is set correctly
        assert hasattr(
            request, 'partial_update'), "UpsertRequest should have partial_update field"
        assert request.partial_update is True, "partial_update should be True when explicitly set to True"

    def test_batch_upsert_param_partial_update_default(self):
        entities = [
            {"name": "id", "type": DataType.INT64, "values": [1, 2, 3]},
            {"name": "name", "type": DataType.VARCHAR,
                "values": ["a", "b", "c"]},
            {"name": "float_vector", "type": DataType.FLOAT_VECTOR,
                "values": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}
        ]

        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "name", "type": DataType.VARCHAR},
            {"name": "float_vector", "type": DataType.FLOAT_VECTOR, "dim": 2}
        ]

        request = Prepare.batch_upsert_param(
            "test_collection", entities, "", fields_info)

        # Check that partial_update defaults to False
        assert hasattr(
            request, 'partial_update'), "UpsertRequest should have partial_update field"
        assert request.partial_update is False, "partial_update should default to False"

    def test_batch_upsert_param_partial_fields_with_partial_update_true(self):
        # Test partial update with only some fields provided
        entities = [
            {"name": "id", "type": DataType.INT64, "values": [1, 2, 3]},
            {"name": "name", "type": DataType.VARCHAR, "values": [
                "updated_a", "updated_b", "updated_c"]}
            # Note: float_vector field is intentionally omitted
        ]

        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "name", "type": DataType.VARCHAR},
            {"name": "float_vector", "type": DataType.FLOAT_VECTOR, "dim": 2},
            {"name": "optional_field", "type": DataType.VARCHAR}
        ]

        # This should succeed with partial_update=True
        request = Prepare.batch_upsert_param("test_collection", entities, "",
                                             fields_info, partial_update=True)

        # Check that partial_update is set correctly
        assert hasattr(
            request, 'partial_update'), "UpsertRequest should have partial_update field"
        assert request.partial_update is True, "partial_update should be True when explicitly set to True"
        assert len(
            request.fields_data) == 2, "Should only contain data for provided fields"

    def test_batch_upsert_param_partial_fields_with_partial_update_false_should_fail(self):
        # Test that partial fields fail when partial_update=False
        entities = [
            {"name": "id", "type": DataType.INT64, "values": [1, 2, 3]},
            {"name": "name", "type": DataType.VARCHAR, "values": [
                "updated_a", "updated_b", "updated_c"]}
            # Note: float_vector field is intentionally omitted
        ]

        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "name", "type": DataType.VARCHAR},
            {"name": "float_vector", "type": DataType.FLOAT_VECTOR, "dim": 2}
        ]

        # This should fail with partial_update=False due to field count mismatch
        with pytest.raises(MilvusException, match="expected number of fields"):
            Prepare.batch_upsert_param("test_collection", entities, "",
                                       fields_info, partial_update=False)

    def test_row_upsert_param_missing_fields_partial_update_true(self):
        """Test that missing non-nullable fields are handled correctly with partial_update=True"""
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            # Non-nullable, no default
            FieldSchema("required_field", DataType.DOUBLE),
            FieldSchema("nullable_field", DataType.VARCHAR, nullable=True)
        ])

        # Entities missing the required_field - should work with partial_update=True
        rows = [
            {"pk_field": 1, "float_vector": rng.random(
                (1, dim))[0]},  # Missing required_field
            {"pk_field": 2, "float_vector": rng.random((1, dim))[0]}
        ]

        # This should work because partial_update=True skips missing field validation
        request = Prepare.row_upsert_param("test_collection", rows, "",
                                           fields_info=schema.to_dict()[
                                               "fields"],
                                           enable_dynamic=False,
                                           partial_update=True)

        assert hasattr(
            request, 'partial_update'), "UpsertRequest should have partial_update field"
        assert request.partial_update is True, "partial_update should be True"

    def test_row_upsert_param_missing_fields_partial_update_false_should_fail(self):
        """Test that missing non-nullable fields fail with partial_update=False"""
        from pymilvus.exceptions import DataNotMatchException
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            # Non-nullable, no default
            FieldSchema("required_field", DataType.DOUBLE),
        ])

        # Entity missing the required_field - should fail with partial_update=False
        rows = [
            {"pk_field": 1, "float_vector": rng.random(
                (1, dim))[0]}  # Missing required_field
        ]

        # This should fail because required_field is missing and not nullable
        with pytest.raises(DataNotMatchException, match="Insert missed an field"):
            Prepare.row_upsert_param("test_collection", rows, "",
                                     fields_info=schema.to_dict()["fields"],
                                     enable_dynamic=False,
                                     partial_update=False)

    def test_row_upsert_param_field_length_inconsistency_error(self):
        """Test error when entities have inconsistent field lengths (validation in line 559-562)"""
        from pymilvus.exceptions import DataNotMatchException
        rng = np.random.default_rng(seed=19530)
        dim = 8
        schema = CollectionSchema([
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("pk_field", DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema("float", DataType.DOUBLE)
        ])

        # Create entities where the entity field count changes between entries
        # This tests the specific logic at lines 559-562 in prepare.py
        rows = [
            {"pk_field": 1, "float_vector": rng.random((1, dim))[
                0]},  # 3 fields
            # Only 2 fields - this should trigger the error
            {"pk_field": 2, "float": 2.0}
        ]

        # This should raise DataNotMatchException due to field length inconsistency
        # This validation happens regardless of partial_update value
        with pytest.raises(DataNotMatchException, match="The data fields length is inconsistent"):
            Prepare.row_upsert_param("test_collection", rows, "",
                                     fields_info=schema.to_dict()["fields"],
                                     enable_dynamic=False,
                                     partial_update=True)

    def test_batch_upsert_param_partial_update_field_count_validation_skip(self):
        """Test that field count validation is skipped with partial_update=True for batch operations"""
        # This tests the logic at lines 632-635 in prepare.py
        entities = [
            {"name": "id", "type": DataType.INT64, "values": [1, 2, 3]},
            {"name": "name", "type": DataType.VARCHAR,
                "values": ["a", "b", "c"]}
            # Intentionally omitting other fields to test partial update
        ]

        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "name", "type": DataType.VARCHAR},
            {"name": "float_vector", "type": DataType.FLOAT_VECTOR, "dim": 2},
            {"name": "optional_field", "type": DataType.VARCHAR, "nullable": True}
        ]

        # This should work with partial_update=True even though not all fields are provided
        request = Prepare.batch_upsert_param("test_collection", entities, "",
                                             fields_info, partial_update=True)

        assert hasattr(
            request, 'partial_update'), "UpsertRequest should have partial_update field"
        assert request.partial_update is True, "partial_update should be True"
        assert len(
            request.fields_data) == 2, "Should only contain data for provided fields"


class TestCreateIndexRequest:
    def test_create_index_request_with_false_boolean_param(self):
        """Test that boolean False values are preserved in index parameters"""
        params = {
            "index_type": "SCANN",
            "metric_type": "L2",
            "with_raw_data": False,
            "nlist": 1024,
        }
        req = Prepare.create_index_request("test_collection", "vector_field", params)
        
        # Verify that with_raw_data=False is included in extra_params
        param_keys = [p.key for p in req.extra_params]
        param_values = {p.key: p.value for p in req.extra_params}
        
        assert "with_raw_data" in param_keys, "with_raw_data parameter should be included"
        # Short-term fix: booleans are serialized as uppercase "False"/"True" for e2e compatibility
        assert param_values["with_raw_data"] == "False", "with_raw_data should be serialized as 'False'"
        assert "index_type" in param_keys
        # String values are serialized as plain strings (without quotes) for compatibility
        assert param_values["index_type"] == "SCANN", "index_type should be serialized as plain string"
        assert param_values["metric_type"] == "L2", "metric_type should be serialized as plain string"
        assert "nlist" in param_keys
        # Numbers are serialized as plain strings
        assert param_values["nlist"] == "1024", "nlist should be serialized as string"

    def test_create_index_request_with_true_boolean_param(self):
        """Test that boolean True values are preserved in index parameters"""
        params = {
            "index_type": "SCANN",
            "metric_type": "L2",
            "with_raw_data": True,
        }
        req = Prepare.create_index_request("test_collection", "vector_field", params)
        
        param_keys = [p.key for p in req.extra_params]
        param_values = {p.key: p.value for p in req.extra_params}
        
        assert "with_raw_data" in param_keys, "with_raw_data parameter should be included"
        # Short-term fix: booleans are serialized as uppercase "False"/"True" for e2e compatibility
        assert param_values["with_raw_data"] == "True", "with_raw_data should be serialized as 'True'"
        assert param_values["index_type"] == "SCANN", "index_type should be serialized as plain string"
        assert param_values["metric_type"] == "L2", "metric_type should be serialized as plain string"

    def test_create_index_request_filters_none_values(self):
        """Test that None values are filtered out from index parameters"""
        params = {
            "index_type": "SCANN",
            "metric_type": "L2",
            "with_raw_data": None,
            "nlist": 1024,
        }
        req = Prepare.create_index_request("test_collection", "vector_field", params)
        
        param_keys = [p.key for p in req.extra_params]
        
        assert "with_raw_data" not in param_keys, "None values should be filtered out"
        assert "index_type" in param_keys
        assert "metric_type" in param_keys
        assert "nlist" in param_keys


class TestAlterCollectionRequest:
    def test_alter_collection_request(self):
        req = Prepare.alter_collection_request(
            'foo', {'collection.ttl.seconds': 1800})
        assert req.collection_name == 'foo'
        assert len(req.properties) == 1
        assert req.properties[0].key == 'collection.ttl.seconds'
        assert req.properties[0].value == '1800'


class TestLoadCollectionRequest:
    def test_load_collection_request(self):
        kwargs = {'load_fields': [
            'pk', 'float_vector', 'string_load', 'int64_load']}
        req = Prepare.load_collection('foo', **kwargs)
        assert req.load_fields == [
            'pk', 'float_vector', 'string_load', 'int64_load']


class TestFunctionEditor:
    def test_add_function(self):
        req = Prepare.add_collection_function_request("test_collection", Function(
            "test", FunctionType.TEXTEMBEDDING, input_field_names=["text"], output_field_names=["embedding"]))
        assert req.collection_name == "test_collection"
        assert req.functionSchema.name == "test"
        assert req.functionSchema.type == FunctionType.TEXTEMBEDDING
        assert req.functionSchema.input_field_names == ["text"]
        assert req.functionSchema.output_field_names == ["embedding"]
        assert req.functionSchema.description == ""

    def test_alter_function(self):
        req = Prepare.alter_collection_function_request("test_collection", "test", Function(
            "test", FunctionType.TEXTEMBEDDING, input_field_names=["text"], output_field_names=["embedding"]))
        assert req.collection_name == "test_collection"
        assert req.functionSchema.name == "test"
        assert req.functionSchema.type == FunctionType.TEXTEMBEDDING
        assert req.functionSchema.input_field_names == ["text"]
        assert req.functionSchema.output_field_names == ["embedding"]
        assert req.functionSchema.description == ""

    def test_drop_function(self):
        req = Prepare.drop_collection_function_request(
            "test_collection", "test")
        assert req.collection_name == "test_collection"
        assert req.function_name == "test"
