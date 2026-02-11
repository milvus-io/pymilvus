"""Tests for $meta['key'] syntax support on the write path."""

import json

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import DataNotMatchException

from .conftest import make_fields_info


def _get_dynamic_blob(req, idx=0):
    """Extract the dynamic field JSON blob from a request."""
    for fd in req.fields_data:
        if fd.field_name == "$meta":
            return json.loads(fd.scalars.json_data.data[idx])
    return None


class TestMetaKeySyntaxInsert:
    """Tests for $meta['key'] syntax in row_insert_param."""

    def _make_dynamic_schema(self):
        return CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ],
            enable_dynamic_field=True,
        )

    def test_meta_key_single_quotes(self):
        """$meta['key'] should store value under 'key' in dynamic blob."""
        schema = self._make_dynamic_schema()
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "$meta['end_timestamp']": 114514}]
        req = Prepare.row_insert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
        )
        blob = _get_dynamic_blob(req)
        assert blob is not None
        assert blob.get("end_timestamp") == 114514
        assert "$meta['end_timestamp']" not in blob

    def test_meta_key_double_quotes(self):
        """$meta["key"] should store value under 'key' in dynamic blob."""
        schema = self._make_dynamic_schema()
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], '$meta["end_timestamp"]': 114514}]
        req = Prepare.row_insert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
        )
        blob = _get_dynamic_blob(req)
        assert blob is not None
        assert blob.get("end_timestamp") == 114514

    def test_meta_key_mixed_with_normal_dynamic(self):
        """$meta['key'] and normal dynamic fields should coexist."""
        schema = self._make_dynamic_schema()
        rows = [
            {
                "pk": 1,
                "vector": [1.0, 2.0, 3.0, 4.0],
                "normal_dynamic": 42,
                "$meta['end_timestamp']": 114514,
            }
        ]
        req = Prepare.row_insert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
        )
        blob = _get_dynamic_blob(req)
        assert blob["normal_dynamic"] == 42
        assert blob["end_timestamp"] == 114514

    def test_meta_key_not_parsed_when_dynamic_disabled(self):
        """$meta['key'] should be rejected as unexpected field when dynamic is off."""
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ],
        )
        rows = [{"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "$meta['extra']": 1}]
        with pytest.raises(DataNotMatchException):
            Prepare.row_insert_param(
                "test_coll",
                rows,
                "",
                fields_info=make_fields_info(schema),
                enable_dynamic=False,
            )


class TestMetaKeySyntaxUpsert:
    """Tests for $meta['key'] syntax in row_upsert_param."""

    def _make_dynamic_schema(self):
        return CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            ],
            enable_dynamic_field=True,
        )

    def test_upsert_meta_key(self):
        """$meta['key'] should work in upsert."""
        schema = self._make_dynamic_schema()
        rows = [
            {
                "pk": 1,
                "vector": [1.0, 2.0, 3.0, 4.0],
                "$meta['end_timestamp']": 114514,
                '$meta["other_key"]': "hello",
            }
        ]
        req = Prepare.row_upsert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
        )
        blob = _get_dynamic_blob(req)
        assert blob["end_timestamp"] == 114514
        assert blob["other_key"] == "hello"

    def test_upsert_partial_update_meta_key(self):
        """$meta['key'] should work in partial update upsert."""
        schema = self._make_dynamic_schema()
        rows = [{"pk": 1, "$meta['end_timestamp']": 114514}]
        req = Prepare.row_upsert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
            partial_update=True,
        )
        blob = _get_dynamic_blob(req)
        assert blob["end_timestamp"] == 114514

    def test_upsert_multiple_rows_meta_key(self):
        """$meta['key'] should work across multiple rows."""
        schema = self._make_dynamic_schema()
        rows = [
            {"pk": 1, "vector": [1.0, 2.0, 3.0, 4.0], "$meta['ts']": 100},
            {"pk": 2, "vector": [5.0, 6.0, 7.0, 8.0], "$meta['ts']": 200},
        ]
        req = Prepare.row_upsert_param(
            "test_coll",
            rows,
            "",
            fields_info=make_fields_info(schema),
            enable_dynamic=True,
        )
        blob0 = _get_dynamic_blob(req, idx=0)
        blob1 = _get_dynamic_blob(req, idx=1)
        assert blob0["ts"] == 100
        assert blob1["ts"] == 200
