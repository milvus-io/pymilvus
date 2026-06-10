"""Tests for data operation Prepare methods (insert, upsert, delete)."""

import numpy as np
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import DataNotMatchException, ParamError
from pymilvus.orm.schema import StructFieldSchema

from .conftest import make_fields_info, make_struct_fields_info


class TestRowInsertParam:
    """Tests for row_insert_param."""

    def test_missing_fields_info(self):
        """Test row insert with no fields_info."""
        with pytest.raises(ParamError, match="Missing collection meta"):
            Prepare.row_insert_param("coll", [{"pk": 1}], "", fields_info=None)

    def test_empty_fields_info(self):
        """Test row insert with empty fields_info."""
        with pytest.raises(ParamError, match="Missing collection meta"):
            Prepare.row_insert_param("coll", [{"pk": 1}], "", fields_info=[])

    def test_insert_with_auto_id_pk_provided(self):
        """Test insert with auto_id when pk is provided in entities."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        rows = [
            {"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0]},
            {"pk": 2, "vector": [5.0, 6.0, 7.0, 8.0]},
        ]
        req = Prepare.row_insert_param("test_coll", rows, "", fields_info=make_fields_info(schema))
        assert req.num_rows == 2

    def test_insert_entity_not_dict(self):
        """Test insert with non-dict entity raises error."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        rows = ["not_a_dict"]
        with pytest.raises(DataNotMatchException):
            Prepare.row_insert_param("test_coll", rows, "", fields_info=make_fields_info(schema))

    def test_insert_unexpected_field_no_dynamic(self):
        """Test insert with unexpected field when dynamic is disabled."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "extra": "field"}]
        with pytest.raises(DataNotMatchException, match="unexpected field"):
            Prepare.row_insert_param(
                "test_coll",
                rows,
                "",
                fields_info=make_fields_info(schema),
                enable_dynamic=False,
            )

    def test_insert_with_dynamic_field(self):
        """Test insert with dynamic field enabled."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ],
            enable_dynamic_field=True,
        )
        rows = [
            {"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "extra": "field"},
        ]
        req = Prepare.row_insert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
        )
        assert req.num_rows == 1

    def test_insert_missing_required_field(self):
        """Test insert with missing required field."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
                FieldSchema("required", DataType.VARCHAR, max_length=100),
            ]
        )
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0]}]
        with pytest.raises(DataNotMatchException, match="Insert missed an field"):
            Prepare.row_insert_param("test_coll", rows, "", fields_info=make_fields_info(schema))

    def test_insert_with_nullable_field_missing(self):
        """Test insert with nullable field missing."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
                FieldSchema("nullable", DataType.VARCHAR, nullable=True, max_length=100),
            ]
        )
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0]}]
        req = Prepare.row_insert_param("test_coll", rows, "", fields_info=make_fields_info(schema))
        assert req.num_rows == 1

    def test_insert_with_default_value_field_missing(self):
        """Test insert with default value field missing."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
                FieldSchema("with_default", DataType.INT64, default_value=0),
            ]
        )
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0]}]
        req = Prepare.row_insert_param("test_coll", rows, "", fields_info=make_fields_info(schema))
        assert req.num_rows == 1

    def test_insert_function_output_field(self):
        """Test insert with function output field raises error."""
        # Manually construct fields_info with is_function_output to test the logic
        fields_info = [
            {"name": "pk", "type": DataType.INT64, "is_primary": True},
            {"name": "text", "type": DataType.VARCHAR, "params": {"max_length": 1000}},
            {
                "name": "embedding",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 4},
                "is_function_output": True,
            },
        ]
        rows = [{"pk": 1, "text": "hello", "embedding": [1.0, 2.0, 3.0, 4.0]}]
        with pytest.raises(DataNotMatchException, match="function output field"):
            Prepare.row_insert_param("test_coll", rows, "", fields_info=fields_info)

    @pytest.mark.parametrize(
        "vector_dtype,vector_data",
        [
            pytest.param(
                DataType.FLOAT16_VECTOR,
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16),
                id="float16",
            ),
            pytest.param(DataType.INT8_VECTOR, np.array([1, 2, 3, 4], dtype=np.int8), id="int8"),
        ],
    )
    def test_insert_special_vector_types(self, vector_dtype, vector_data):
        """Test insert with special vector types."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", vector_dtype, dim=4),
            ]
        )
        rows = [{"pk": 1, "vector": vector_data}]
        req = Prepare.row_insert_param("test_coll", rows, "", fields_info=make_fields_info(schema))
        assert req.num_rows == 1


class TestRowUpsertParam:
    """Tests for row_upsert_param."""

    def test_missing_fields_info(self):
        """Test row upsert with no fields_info."""
        with pytest.raises(ParamError, match="Missing collection meta"):
            Prepare.row_upsert_param("coll", [{"pk": 1}], "", fields_info=None)

    def test_upsert_entity_not_dict(self):
        """Test upsert with non-dict entity raises error."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        rows = ["not_a_dict"]
        with pytest.raises(DataNotMatchException):
            Prepare.row_upsert_param("test_coll", rows, "", fields_info=make_fields_info(schema))

    def test_upsert_function_output_field(self):
        """Test upsert with function output field raises error."""
        # Manually construct fields_info with is_function_output to test the logic
        fields_info = [
            {"name": "pk", "type": DataType.INT64, "is_primary": True},
            {"name": "text", "type": DataType.VARCHAR, "params": {"max_length": 1000}},
            {
                "name": "embedding",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 4},
                "is_function_output": True,
            },
        ]
        rows = [{"pk": 1, "text": "hello", "embedding": [1.0, 2.0, 3.0, 4.0]}]
        with pytest.raises(DataNotMatchException, match="function output field"):
            Prepare.row_upsert_param("test_coll", rows, "", fields_info=fields_info)

    def test_upsert_unexpected_field_no_dynamic(self):
        """Test upsert with unexpected field when dynamic is disabled."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "extra": "field"}]
        with pytest.raises(DataNotMatchException, match="unexpected field"):
            Prepare.row_upsert_param(
                "test_coll",
                rows,
                "",
                fields_info=make_fields_info(schema),
                enable_dynamic=False,
            )

    def test_upsert_with_dynamic_field(self):
        """Test upsert with dynamic field enabled."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "extra": "field"}]
        req = Prepare.row_upsert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
        )
        assert req.num_rows == 1

    def test_upsert_partial_update_struct_not_supported(self):
        """Test partial update with struct fields raises error."""
        struct_field = StructFieldSchema()
        struct_field.name = "metadata"
        struct_field.add_field("score", DataType.FLOAT)
        struct_field.max_capacity = 10

        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        schema.add_struct_field(struct_field)

        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0]}]
        with pytest.raises(ParamError, match="Struct fields are not supported"):
            Prepare.row_upsert_param(
                "test_coll",
                rows,
                "",
                fields_info=make_fields_info(schema),
                struct_fields_info=make_struct_fields_info(schema),
                partial_update=True,
            )

    def test_upsert_partial_update_field_len_inconsistent(self):
        """Test partial update with inconsistent field lengths."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
                FieldSchema("name", DataType.VARCHAR, max_length=100),
            ]
        )
        # Different rows have different fields
        rows = [
            {"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0]},
            {"pk": 2, "name": "test"},
        ]
        with pytest.raises(DataNotMatchException, match="inconsistent"):
            Prepare.row_upsert_param(
                "test_coll",
                rows,
                "",
                fields_info=make_fields_info(schema),
                partial_update=True,
            )


class TestBatchInsertParam:
    """Tests for batch_insert_param."""

    @pytest.mark.parametrize(
        "entity,missing_key",
        [
            pytest.param({"values": [1, 2], "type": DataType.INT64}, "name", id="missing_name"),
            pytest.param({"name": "id", "type": DataType.INT64}, "values", id="missing_values"),
            pytest.param({"name": "id", "values": [1, 2]}, "type", id="missing_type"),
        ],
    )
    def test_missing_entity_field(self, entity, missing_key):
        """Test batch insert with missing entity field."""
        fields_info = [{"name": "id", "type": DataType.INT64, "is_primary": True}]
        with pytest.raises(ParamError, match="must have type, name and values"):
            Prepare.batch_insert_param("coll", [entity], "", fields_info)

    def test_missing_fields_info(self):
        """Test batch insert with no fields_info."""
        entities = [{"name": "id", "values": [1, 2], "type": DataType.INT64}]
        with pytest.raises(ParamError, match="Missing collection meta"):
            Prepare.batch_insert_param("coll", entities, "", None)

    def test_field_count_mismatch(self):
        """Test batch insert with field count mismatch."""
        entities = [{"name": "id", "values": [1, 2], "type": DataType.INT64}]
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
        ]
        with pytest.raises(ParamError, match="expected number of fields"):
            Prepare.batch_insert_param("coll", entities, "", fields_info)

    def test_field_size_mismatch(self):
        """Test batch insert with different field sizes."""
        entities = [
            {"name": "id", "values": [1, 2, 3], "type": DataType.INT64},
            {
                "name": "vector",
                "values": [[1.0, 2.0], [3.0, 4.0]],
                "type": DataType.FLOAT_VECTOR,
            },
        ]
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
        ]
        with pytest.raises(ParamError, match="misaligned"):
            Prepare.batch_insert_param("coll", entities, "", fields_info)

    def test_zero_rows(self):
        """Test batch insert with zero rows."""
        entities = [
            {"name": "id", "values": [], "type": DataType.INT64},
            {"name": "vector", "values": [], "type": DataType.FLOAT_VECTOR},
        ]
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
        ]
        with pytest.raises(ParamError):
            Prepare.batch_insert_param("coll", entities, "", fields_info)


class TestBatchUpsertParam:
    """Tests for batch_upsert_param."""

    def test_upsert_field_count_mismatch_no_partial(self):
        """Test batch upsert with field count mismatch without partial update."""
        entities = [{"name": "id", "values": [1, 2], "type": DataType.INT64}]
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
        ]
        with pytest.raises(ParamError, match="expected number of fields"):
            Prepare.batch_upsert_param("coll", entities, "", fields_info, partial_update=False)

    def test_upsert_partial_update_skip_count_check(self):
        """Test batch upsert with partial update skips field count check."""
        entities = [{"name": "id", "values": [1, 2], "type": DataType.INT64}]
        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
        ]
        # This should work with partial_update=True
        req = Prepare.batch_upsert_param("coll", entities, "", fields_info, partial_update=True)
        assert req.partial_update is True


class TestDeleteRequest:
    """Tests for delete_request."""

    @pytest.mark.parametrize(
        "collection_name",
        [
            pytest.param(None, id="none"),
            pytest.param("", id="empty"),
        ],
    )
    def test_invalid_collection_name(self, collection_name):
        """Test delete with invalid collection name."""
        with pytest.raises(ParamError):
            Prepare.delete_request(collection_name, "id > 0", None, 0)

    @pytest.mark.parametrize(
        "filter_expr",
        [
            pytest.param(None, id="none"),
            pytest.param("", id="empty"),
        ],
    )
    def test_invalid_filter(self, filter_expr):
        """Test delete with invalid filter."""
        with pytest.raises(ParamError):
            Prepare.delete_request("coll", filter_expr, None, 0)

    def test_delete_with_expr_params(self):
        """Test delete with expression parameters."""
        req = Prepare.delete_request(
            "coll",
            "id in {ids}",
            None,
            0,
            expr_params={"ids": [1, 2, 3]},
        )
        assert req.collection_name == "coll"
        assert req.expr == "id in {ids}"


class TestStructFieldProcessing:
    """Tests for struct field processing methods."""

    def test_process_struct_field_not_list(self):
        """Test struct field with non-list value."""
        struct_info = {
            "name": "metadata",
            "fields": [{"name": "score", "type": DataType.FLOAT}],
        }
        with pytest.raises(TypeError, match="Expected list"):
            Prepare._process_struct_field(
                "metadata",
                "not_a_list",
                struct_info,
                {"metadata": {"score": {"type": DataType.ARRAY}}},
                {"metadata": {"score": None}},
            )

    @pytest.mark.parametrize(
        "values,required_fields,field_name,error_type,error_match",
        [
            pytest.param(
                ["not_a_dict"],
                {"score"},
                "metadata",
                TypeError,
                "must be dict",
                id="not_dict",
            ),
            pytest.param(
                [{"other": 1.0}],
                {"score"},
                "metadata",
                ValueError,
                "missing required fields",
                id="missing_fields",
            ),
            pytest.param(
                [{"score": 1.0, "extra": "field"}],
                {"score"},
                "metadata",
                ValueError,
                "unexpected fields",
                id="extra_fields",
            ),
            pytest.param(
                [{"score": None}],
                {"score"},
                "metadata",
                ValueError,
                "cannot be None",
                id="none_value",
            ),
        ],
    )
    def test_validate_struct_values_errors(
        self, values, required_fields, field_name, error_type, error_match
    ):
        """Test struct values validation error cases."""
        with pytest.raises(error_type, match=error_match):
            Prepare._validate_and_collect_struct_values(values, required_fields, field_name)

    @pytest.mark.parametrize(
        "field_info,expected",
        [
            pytest.param({"params": {"dim": "128"}}, 128, id="string_dim"),
            pytest.param({"params": {"dim": 128}}, 128, id="int_dim"),
            pytest.param({"params": {"dim": 256}}, 256, id="larger_dim"),
            pytest.param({"params": {}}, 0, id="missing_dim"),
            pytest.param({}, 0, id="no_params"),
        ],
    )
    def test_get_dim_value(self, field_info, expected):
        """Test get_dim_value with various inputs."""
        assert Prepare._get_dim_value(field_info) == expected

    def test_process_struct_values_unsupported_type(self):
        """Test process_struct_values with unsupported type."""
        with pytest.raises(ParamError, match="Unsupported data type"):
            Prepare._process_struct_values(
                {"field": [1, 2]},
                {"field": {"type": DataType.JSON}},
                {"field": type("MockFieldData", (), {"scalars": None, "vectors": None})()},
            )
