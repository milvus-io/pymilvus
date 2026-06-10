"""Tests for collection-related Prepare methods."""

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError
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
