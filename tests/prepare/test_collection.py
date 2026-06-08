"""Tests for collection-related Prepare methods."""

import inspect

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2
from pymilvus.orm.schema import StructFieldSchema


class TestCreateCollectionRequest:
    """Tests for create_collection_request."""

    @pytest.mark.parametrize(
        "num_partitions,expected_error",
        [
            pytest.param("10", "invalid num_partitions type", id="string"),
            pytest.param(True, "invalid num_partitions type", id="bool_true"),
            pytest.param(False, "invalid num_partitions type", id="bool_false"),
            pytest.param(0, "should be greater than or equal to 1", id="zero"),
            pytest.param(-1, "should be greater than or equal to 1", id="negative"),
        ],
    )
    def test_create_collection_invalid_num_partitions(
        self, basic_schema, num_partitions, expected_error
    ):
        """Test create_collection_request with invalid num_partitions."""
        with pytest.raises(ParamError, match=expected_error):
            Prepare.create_collection_request(
                "test_coll", basic_schema, num_partitions=num_partitions
            )

    def test_create_collection_valid_num_partitions(self, basic_schema):
        """Test create_collection_request with valid num_partitions."""
        req = Prepare.create_collection_request("test_coll", basic_schema, num_partitions=16)
        assert req.num_partitions == 16

    @pytest.mark.parametrize(
        "consistency_level",
        [
            pytest.param(0, id="strong"),
            pytest.param(1, id="session"),
            pytest.param(2, id="bounded"),
            pytest.param(3, id="eventually"),
            pytest.param("Strong", id="strong_str"),
            pytest.param("Session", id="session_str"),
        ],
    )
    def test_create_collection_consistency_level(self, basic_schema, consistency_level):
        """Test create_collection_request with different consistency levels."""
        req = Prepare.create_collection_request(
            "test_coll", basic_schema, consistency_level=consistency_level
        )
        assert req is not None


class TestGetSchemaFromCollectionSchema:
    """Tests for get_schema_from_collection_schema."""

    @pytest.mark.parametrize(
        "description",
        [
            pytest.param(123, id="int"),
            pytest.param([], id="list"),
            pytest.param({}, id="dict"),
            pytest.param(None, id="none"),
        ],
    )
    def test_invalid_description_type(self, description):
        """Test with invalid description types."""
        # Create schema with invalid description
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        schema._description = description

        with pytest.raises(ParamError, match="description"):
            Prepare.get_schema_from_collection_schema("test", schema)

    def test_schema_with_struct_fields(self):
        """Test schema with struct array fields."""
        struct_field = StructFieldSchema()
        struct_field.name = "metadata"
        struct_field.add_field("score", DataType.FLOAT)
        struct_field.add_field("label", DataType.VARCHAR, max_length=100)
        struct_field.max_capacity = 10

        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        schema.add_struct_field(struct_field)

        result = Prepare.get_schema_from_collection_schema("test", schema)
        assert len(result.struct_array_fields) == 1
        assert result.struct_array_fields[0].name == "metadata"

    def test_schema_with_struct_missing_max_capacity(self):
        """Test struct field without max_capacity raises error."""
        struct_field = StructFieldSchema()
        struct_field.name = "metadata"
        struct_field.add_field("score", DataType.FLOAT)
        # max_capacity is None by default

        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        schema.add_struct_field(struct_field)

        with pytest.raises(ParamError, match="max_capacity not set"):
            Prepare.get_schema_from_collection_schema("test", schema)

    def test_schema_with_vector_struct_field(self):
        """Test struct field with vector type."""
        struct_field = StructFieldSchema()
        struct_field.name = "embeddings"
        struct_field.add_field("vec", DataType.FLOAT_VECTOR, dim=8)
        struct_field.max_capacity = 5

        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        schema.add_struct_field(struct_field)

        result = Prepare.get_schema_from_collection_schema("test", schema)
        assert len(result.struct_array_fields) == 1

    def test_schema_with_functions(self):
        """Test schema with function definitions."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=1000),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128, is_function_output=True),
            ],
            functions=[
                Function(
                    "text_embedding",
                    FunctionType.TEXTEMBEDDING,
                    input_field_names=["text"],
                    output_field_names=["embedding"],
                    params={"model": "test"},
                )
            ],
        )

        result = Prepare.get_schema_from_collection_schema("test", schema)
        assert len(result.functions) == 1
        assert result.functions[0].name == "text_embedding"

    def test_schema_enable_namespace(self):
        """Test schema with enable_namespace."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ],
            enable_namespace=True,
        )

        result = Prepare.get_schema_from_collection_schema("test", schema)
        assert result.enable_namespace is True


class TestGetFieldSchema:
    """Tests for get_field_schema."""

    def test_missing_field_name(self):
        """Test field without name raises error."""
        with pytest.raises(ParamError, match="specify the name"):
            Prepare.get_field_schema({"type": DataType.INT64})

    def test_missing_field_type(self):
        """Test field without type raises error."""
        with pytest.raises(ParamError, match="specify the data type"):
            Prepare.get_field_schema({"name": "test"})

    def test_invalid_field_type(self):
        """Test field with invalid type raises error."""
        with pytest.raises(ParamError, match="Field type must be"):
            Prepare.get_field_schema({"name": "test", "type": "invalid"})

    @pytest.mark.parametrize(
        "is_primary",
        [
            pytest.param("true", id="string"),
            pytest.param(1, id="int"),
            pytest.param([], id="list"),
        ],
    )
    def test_invalid_is_primary_type(self, is_primary):
        """Test is_primary with non-boolean type."""
        with pytest.raises(ParamError, match="is_primary must be boolean"):
            Prepare.get_field_schema(
                {"name": "test", "type": DataType.INT64, "is_primary": is_primary}
            )

    @pytest.mark.parametrize(
        "nullable",
        [
            pytest.param("true", id="string"),
            pytest.param(1, id="int"),
        ],
    )
    def test_invalid_nullable_type(self, nullable):
        """Test nullable with non-boolean type."""
        with pytest.raises(ParamError, match="nullable must be boolean"):
            Prepare.get_field_schema({"name": "test", "type": DataType.INT64, "nullable": nullable})

    @pytest.mark.parametrize(
        "auto_id",
        [
            pytest.param("true", id="string"),
            pytest.param(1, id="int"),
        ],
    )
    def test_invalid_auto_id_type(self, auto_id):
        """Test auto_id with non-boolean type."""
        with pytest.raises(ParamError, match="auto_id must be boolean"):
            Prepare.get_field_schema({"name": "test", "type": DataType.INT64, "auto_id": auto_id})

    def test_two_primary_fields(self):
        """Test multiple primary fields raises error."""
        Prepare.get_field_schema({"name": "pk1", "type": DataType.INT64, "is_primary": True})
        with pytest.raises(ParamError, match="only have one primary"):
            Prepare.get_field_schema(
                {"name": "pk2", "type": DataType.INT64, "is_primary": True},
                primary_field="pk1",
            )

    def test_two_auto_id_fields(self):
        """Test multiple auto_id fields raises error."""
        with pytest.raises(ParamError, match="only have one autoID"):
            Prepare.get_field_schema(
                {"name": "id2", "type": DataType.INT64, "auto_id": True},
                auto_id_field="id1",
            )

    def test_invalid_primary_type(self):
        """Test primary field with invalid data type."""
        with pytest.raises(ParamError, match="int64 and varChar"):
            Prepare.get_field_schema({"name": "pk", "type": DataType.FLOAT, "is_primary": True})

    def test_auto_id_invalid_data_type(self):
        """Test auto_id with non-INT64 type."""
        with pytest.raises(ParamError, match="int64 is the only supported"):
            Prepare.get_field_schema({"name": "id", "type": DataType.VARCHAR, "auto_id": True})

    def test_invalid_params_type(self):
        """Test field with non-dict params."""
        with pytest.raises(ParamError, match="params should be dictionary"):
            Prepare.get_field_schema({"name": "test", "type": DataType.INT64, "params": "invalid"})

    def test_valid_field_with_all_options(self):
        """Test valid field with all options."""
        field_schema, primary, auto_id = Prepare.get_field_schema(
            {
                "name": "pk",
                "type": DataType.INT64,
                "is_primary": True,
                "auto_id": True,
                "nullable": False,
                "description": "Primary key",
                "is_partition_key": False,
                "is_clustering_key": False,
            }
        )
        assert field_schema.name == "pk"
        assert primary == "pk"
        assert auto_id == "pk"

    def test_external_field_is_copied_to_proto(self):
        """External field mapping should survive dict -> proto conversion."""
        field_schema, primary, auto_id = Prepare.get_field_schema(
            {
                "name": "score",
                "type": DataType.DOUBLE,
                "nullable": True,
                "external_field": "score",
            }
        )

        assert field_schema.name == "score"
        assert field_schema.data_type == DataType.DOUBLE
        assert field_schema.nullable is True
        assert field_schema.external_field == "score"
        assert primary is None
        assert auto_id is None


class TestGetSchema:
    """Tests for get_schema."""

    def test_invalid_fields_type(self):
        """Test with non-dict fields."""
        with pytest.raises(ParamError, match="must be a dict"):
            Prepare.get_schema("test", [])

    def test_missing_fields_key(self):
        """Test without 'fields' key."""
        with pytest.raises(ParamError, match="must contain key 'fields'"):
            Prepare.get_schema("test", {"other": []})

    def test_empty_fields(self):
        """Test with empty fields list."""
        with pytest.raises(ParamError, match="cannot be empty"):
            Prepare.get_schema("test", {"fields": []})

    def test_enable_dynamic_field_from_dict(self):
        """Test enable_dynamic_field from fields dict."""
        schema = Prepare.get_schema(
            "test",
            {
                "fields": [
                    {"name": "pk", "type": DataType.INT64, "is_primary": True},
                    {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
                ],
                "enable_dynamic_field": True,
            },
        )
        assert schema.enable_dynamic_field is True

    def test_enable_namespace_from_dict(self):
        """Test enable_namespace from fields dict."""
        schema = Prepare.get_schema(
            "test",
            {
                "fields": [
                    {"name": "pk", "type": DataType.INT64, "is_primary": True},
                ],
                "enable_namespace": True,
            },
        )
        assert schema.enable_namespace is True


class TestAddCollectionFieldRequest:
    """Tests for add_collection_field_request."""

    def test_add_field_request(self):
        """Test adding a field to collection."""
        field = FieldSchema("new_field", DataType.VARCHAR, max_length=100)
        req = Prepare.add_collection_field_request("test_coll", field)
        assert req.collection_name == "test_coll"

    def test_add_external_field_request_serializes_mapping(self):
        """AddCollectionFieldRequest.schema should include external_field."""
        field = FieldSchema(
            "score",
            DataType.DOUBLE,
            nullable=True,
            external_field="score",
        )

        req = Prepare.add_collection_field_request("ext_coll", field)
        parsed = schema_pb2.FieldSchema()
        parsed.ParseFromString(req.schema)

        assert req.collection_name == "ext_coll"
        assert parsed.name == "score"
        assert parsed.data_type == DataType.DOUBLE
        assert parsed.nullable is True
        assert parsed.external_field == "score"

    def test_add_struct_field_request(self):
        """Test adding a nullable struct field to collection."""
        struct_field = StructFieldSchema(nullable=True, description="new struct")
        struct_field.name = "metadata"
        struct_field.max_capacity = 16
        struct_field.add_field("score", DataType.FLOAT)
        struct_field.add_field("embedding", DataType.FLOAT_VECTOR, dim=4)

        req = Prepare.add_collection_struct_field_request("test_coll", struct_field)
        struct_schema = req.struct_array_field_schema

        assert req.collection_name == "test_coll"
        assert struct_schema.name == "metadata"
        assert struct_schema.description == "new struct"
        assert struct_schema.nullable is True
        assert [field.name for field in struct_schema.fields] == ["score", "embedding"]
        assert [field.nullable for field in struct_schema.fields] == [True, True]
        assert struct_schema.fields[0].data_type == DataType.ARRAY
        assert struct_schema.fields[0].element_type == DataType.FLOAT
        assert struct_schema.fields[1].data_type == DataType._ARRAY_OF_VECTOR
        assert struct_schema.fields[1].element_type == DataType.FLOAT_VECTOR
        assert any(
            kv.key == "max_capacity" and kv.value == "16"
            for kv in struct_schema.fields[0].type_params
        )
        assert any(
            kv.key == "dim" and kv.value == "4" for kv in struct_schema.fields[1].type_params
        )

    def test_add_struct_field_request_with_struct_params(self):
        """Test struct-level params are serialized on add struct field."""
        struct_field = StructFieldSchema(nullable=True)
        struct_field.name = "metadata"
        struct_field.max_capacity = 16
        struct_field._type_params["mmap_enabled"] = True
        struct_field._type_params["warmup"] = {"policy": "async"}
        struct_field.add_field("score", DataType.FLOAT)

        req = Prepare.add_collection_struct_field_request("test_coll", struct_field)
        params = {kv.key: kv.value for kv in req.struct_array_field_schema.type_params}

        assert params["mmap.enabled"] == "true"
        assert params["warmup"] == '{"policy":"async"}'


class TestAlterCollectionSchemaRequest:
    """Tests for alter_collection_schema_request."""

    def test_request_builder_signature_excludes_physical_backfill(self):
        signature = inspect.signature(Prepare.alter_collection_schema_request)
        assert "do_physical_backfill" not in signature.parameters

    def test_add_with_field_and_function(self):
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128)
        func = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
        req = Prepare.alter_collection_schema_request(
            collection_name="coll",
            field_schema=field,
            func=func,
        )
        assert req.collection_name == "coll"
        assert req.action.HasField("add_request")
        assert len(req.action.add_request.field_infos) == 1
        assert req.action.add_request.field_infos[0].index_name == ""
        assert len(req.action.add_request.field_infos[0].extra_params) == 0
        assert len(req.action.add_request.func_schema) == 1

    def test_add_with_field_only(self):
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128)
        req = Prepare.alter_collection_schema_request(
            collection_name="coll",
            field_schema=field,
        )
        assert req.action.HasField("add_request")
        assert len(req.action.add_request.field_infos) == 1
        assert len(req.action.add_request.func_schema) == 0

    def test_drop_by_field_name(self):
        req = Prepare.alter_collection_schema_request(
            collection_name="coll",
            drop_field_name="old_field",
        )
        assert req.collection_name == "coll"
        assert req.action.HasField("drop_request")
        assert req.action.drop_request.field_name == "old_field"

    def test_drop_by_field_id(self):
        req = Prepare.alter_collection_schema_request(
            collection_name="coll",
            drop_field_id=42,
        )
        assert req.action.HasField("drop_request")
        assert req.action.drop_request.field_id == 42

    def test_drop_by_function_name(self):
        req = Prepare.alter_collection_schema_request(
            collection_name="coll",
            drop_function_name="bm25",
        )
        assert req.action.HasField("drop_request")
        assert req.action.drop_request.function_name == "bm25"

    def test_error_on_both_add_and_drop(self):
        field = FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128)
        with pytest.raises(ParamError, match="Cannot perform both"):
            Prepare.alter_collection_schema_request(
                collection_name="coll",
                field_schema=field,
                drop_field_name="old",
            )

    def test_error_on_neither_add_nor_drop(self):
        with pytest.raises(ParamError, match="Must specify"):
            Prepare.alter_collection_schema_request(collection_name="coll")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"drop_field_name": ""},
            {"drop_field_id": 0},
            {"drop_function_name": ""},
        ],
    )
    def test_error_on_empty_drop_identifier(self, kwargs):
        with pytest.raises(ParamError, match="exactly one valid Drop identifier"):
            Prepare.alter_collection_schema_request(collection_name="coll", **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"drop_field_name": "old_field", "drop_field_id": 42},
            {"drop_field_name": "old_field", "drop_function_name": "bm25"},
            {"drop_field_id": 42, "drop_function_name": "bm25"},
        ],
    )
    def test_error_on_multiple_drop_identifiers(self, kwargs):
        with pytest.raises(ParamError, match="exactly one valid Drop identifier"):
            Prepare.alter_collection_schema_request(collection_name="coll", **kwargs)
