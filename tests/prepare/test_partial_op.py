"""Tests for FieldPartialUpdateOp wiring through Prepare."""

import pytest
from pymilvus import CollectionSchema, DataType, FieldOp, FieldOpType, FieldSchema
from pymilvus.client.field_ops import normalize_field_ops
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import milvus_pb2, schema_pb2

# ============================================================
# FieldOp factory + normalize_field_ops helpers
# ============================================================


def test_fieldop_factories_return_expected_enum_values():
    assert FieldOp.array_append().op == FieldOpType.ARRAY_APPEND
    assert FieldOp.array_remove().op == FieldOpType.ARRAY_REMOVE
    assert FieldOp.replace().op == FieldOpType.REPLACE


def test_normalize_field_ops_accepts_proto_message():
    ops = normalize_field_ops({"tags": FieldOp.array_append()})
    assert list(ops) == ["tags"]
    assert ops["tags"].op == FieldOpType.ARRAY_APPEND


def test_normalize_field_ops_accepts_enum():
    ops = normalize_field_ops({"tags": FieldOpType.ARRAY_REMOVE})
    assert ops["tags"].op == FieldOpType.ARRAY_REMOVE


def test_normalize_field_ops_accepts_string_alias():
    ops = normalize_field_ops({"tags": "array_append"})
    assert ops["tags"].op == FieldOpType.ARRAY_APPEND


def test_normalize_field_ops_accepts_raw_int_enum():
    ops = normalize_field_ops({"tags": int(FieldOpType.ARRAY_APPEND)})
    assert ops["tags"].op == FieldOpType.ARRAY_APPEND


def test_normalize_field_ops_drops_replace():
    assert normalize_field_ops({"tags": FieldOp.replace()}) == {}
    assert normalize_field_ops({"tags": "replace"}) == {}
    assert normalize_field_ops({"tags": FieldOpType.REPLACE}) == {}


def test_normalize_field_ops_ignores_none_value():
    assert normalize_field_ops({"tags": None}) == {}


def test_normalize_field_ops_rejects_unknown_string():
    with pytest.raises(ParamError, match="unknown field_op alias"):
        normalize_field_ops({"tags": "does_not_exist"})


def test_normalize_field_ops_rejects_unsupported_type():
    with pytest.raises(ParamError, match="unsupported field_op type"):
        normalize_field_ops({"tags": object()})


def test_normalize_field_ops_handles_empty_and_none():
    assert normalize_field_ops(None) == {}
    assert normalize_field_ops({}) == {}


# ============================================================
# Prepare.row_upsert_param wiring
# ============================================================


@pytest.fixture
def array_schema():
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("tags", DataType.ARRAY, element_type=DataType.INT64, max_capacity=32),
        ]
    )


def _fields_info(schema: CollectionSchema):
    return [f.to_dict() for f in schema.fields]


def _find_field_op(request, name: str):
    for op in request.field_ops:
        if op.field_name == name:
            return op
    return None


def test_row_upsert_param_emits_field_op_on_request(array_schema):
    rows = [{"pk": 1, "tags": [1, 2]}, {"pk": 2, "tags": [3]}]
    req = Prepare.row_upsert_param(
        "test",
        rows,
        "",
        _fields_info(array_schema),
        field_ops={"tags": FieldOp.array_append()},
    )
    assert req.partial_update is True
    tags_op = _find_field_op(req, "tags")
    assert tags_op is not None
    assert tags_op.op == FieldOpType.ARRAY_APPEND
    # FieldData carries data only — no op leakage into individual rows.
    for fd in req.fields_data:
        # The proto no longer exposes partial_op on FieldData; assert that
        # no field_ops have inadvertently been attached through some other
        # path by checking the message has no populated sub-message.
        assert fd.DESCRIPTOR.fields_by_name.get("partial_op") is None


def test_row_upsert_param_remove_op(array_schema):
    req = Prepare.row_upsert_param(
        "test",
        [{"pk": 1, "tags": [1]}],
        "",
        _fields_info(array_schema),
        field_ops={"tags": "array_remove"},
    )
    assert req.partial_update is True
    op = _find_field_op(req, "tags")
    assert op.op == FieldOpType.ARRAY_REMOVE


def test_row_upsert_param_without_ops_leaves_field_ops_empty(array_schema):
    req = Prepare.row_upsert_param("test", [{"pk": 1, "tags": [1]}], "", _fields_info(array_schema))
    assert req.partial_update is False
    assert len(req.field_ops) == 0


def test_row_upsert_param_unknown_field_op_is_still_emitted(array_schema):
    req = Prepare.row_upsert_param(
        "test",
        [{"pk": 1, "tags": [1]}],
        "",
        _fields_info(array_schema),
        field_ops={"non_existent": FieldOp.array_append()},
    )
    assert req.partial_update is True
    # Op is forwarded so the server can surface a validation error.
    assert len(req.field_ops) == 1
    assert req.field_ops[0].field_name == "non_existent"


def test_row_upsert_param_explicit_partial_update_preserved(array_schema):
    req = Prepare.row_upsert_param(
        "test",
        [{"pk": 1, "tags": [1]}],
        "",
        _fields_info(array_schema),
        partial_update=True,
    )
    assert req.partial_update is True


def test_row_upsert_param_op_auto_promotes_partial_update(array_schema):
    req = Prepare.row_upsert_param(
        "test",
        [{"pk": 1, "tags": [1]}],
        "",
        _fields_info(array_schema),
        partial_update=False,
        field_ops={"tags": FieldOp.array_append()},
    )
    assert req.partial_update is True


# ============================================================
# Prepare.batch_upsert_param wiring
# ============================================================


def _batch_entities():
    return [
        {"name": "pk", "type": DataType.INT64, "values": [1, 2]},
        {"name": "tags", "type": DataType.ARRAY, "values": [[1, 2], [3]]},
    ]


def test_batch_upsert_param_emits_field_op(array_schema):
    req = Prepare.batch_upsert_param(
        "test",
        _batch_entities(),
        "",
        _fields_info(array_schema),
        field_ops={"tags": FieldOp.array_append()},
    )
    assert req.partial_update is True
    op = _find_field_op(req, "tags")
    assert op is not None
    assert op.op == FieldOpType.ARRAY_APPEND


def test_batch_upsert_param_without_ops_default(array_schema):
    req = Prepare.batch_upsert_param("test", _batch_entities(), "", _fields_info(array_schema))
    assert req.partial_update is False
    assert len(req.field_ops) == 0


# ============================================================
# Direct tests for the internal _apply_field_ops helper
# ============================================================


def test_apply_field_ops_appends_each_to_request():
    req = milvus_pb2.UpsertRequest()
    ops = {
        "a": schema_pb2.FieldPartialUpdateOp(op=FieldOpType.ARRAY_APPEND),
        "b": schema_pb2.FieldPartialUpdateOp(op=FieldOpType.ARRAY_REMOVE),
    }
    Prepare._apply_field_ops(req, ops)
    assert len(req.field_ops) == 2
    byname = {o.field_name: o.op for o in req.field_ops}
    assert byname == {"a": FieldOpType.ARRAY_APPEND, "b": FieldOpType.ARRAY_REMOVE}


def test_apply_field_ops_empty_is_noop():
    req = milvus_pb2.UpsertRequest()
    Prepare._apply_field_ops(req, {})
    Prepare._apply_field_ops(req, None)
    assert len(req.field_ops) == 0


def test_field_data_message_has_no_partial_op_field():
    """Regression guard: FieldData must remain a pure data carrier."""
    assert "partial_op" not in schema_pb2.FieldData.DESCRIPTOR.fields_by_name
