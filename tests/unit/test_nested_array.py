"""Nested ARRAY contracts across schema, mutation requests and result decoding."""

import copy
from array import array
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.client.abstract import FieldSchema as ResponseFieldSchema
from pymilvus.client.abstract import StructArrayFieldSchema
from pymilvus.client.entity_helper import convert_to_array, extract_struct_array_from_column_data
from pymilvus.client.field_data_extractors import (
    array_cell_length,
    decode_array,
    decode_array_value,
    decode_range,
)
from pymilvus.client.prepare import Prepare
from pymilvus.client.search_result import extract_array_row_data
from pymilvus.client.utils import convert_struct_fields_to_user_format
from pymilvus.exceptions import DataTypeNotSupportException, MilvusException, ParamError
from pymilvus.grpc_gen import schema_pb2
from pymilvus.orm.schema import StructFieldSchema


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("element_type", [None, DataType.ARRAY])
def test_create_collection_preserves_complete_type_tree(depth, element_type):
    type_schema = {"leaf_type": DataType.INT64}
    for capacity in range(depth, 0, -1):
        type_schema = {
            "array_element": type_schema,
            "type_params": {"max_capacity": capacity},
        }
    field = FieldSchema(
        "nested", DataType.ARRAY, type_schema=type_schema, element_type=element_type
    )
    schema = CollectionSchema([FieldSchema("pk", DataType.INT64, is_primary=True), field])
    for source in (schema, schema.to_dict()):
        request = Prepare.create_collection_request("test", source)
        wire = schema_pb2.CollectionSchema.FromString(request.schema).fields[1]
        assert wire.data_type == DataType.ARRAY
        assert wire.element_type == DataType.ARRAY
        node = wire.type_schema
        for capacity in range(1, depth + 1):
            assert node.WhichOneof("kind") == "array_element"
            assert {param.key: param.value for param in node.type_params} == {
                "max_capacity": str(capacity)
            }
            node = node.array_element
        assert node.WhichOneof("kind") == "leaf_type"
        assert node.leaf_type == DataType.INT64
        assert FieldSchema.construct_from_dict(ResponseFieldSchema(wire).dict()) == field


@pytest.mark.parametrize("root_params_in_schema", [False, True])
def test_dictionary_and_orm_schema_preserve_nested_params(root_params_in_schema):
    raw = {
        "name": "nested",
        "type": DataType.ARRAY,
        "nullable": True,
        "type_schema": {
            "array_element": {
                "array_element": {
                    "leaf_type": DataType.VARCHAR,
                    "type_params": {
                        "max_length": 32,
                        "enable_analyzer": True,
                        "analyzer_params": {"tokenizer": "standard"},
                    },
                },
                "nullable": True,
                "type_params": {"max_capacity": 8},
            },
        },
    }
    if root_params_in_schema:
        raw["type_schema"]["type_params"] = {"max_capacity": 16, "mmap_enabled": True}
    else:
        raw["params"] = {"max_capacity": 16, "mmap_enabled": True}
    original = copy.deepcopy(raw)
    field = FieldSchema.construct_from_dict(raw)
    schema = CollectionSchema([FieldSchema("pk", DataType.INT64, is_primary=True), field])
    orm_wire = Prepare.get_schema_from_collection_schema("test", schema).fields[1]
    dict_wire, _, _ = Prepare.get_field_schema(raw)

    assert dict_wire == orm_wire
    assert raw == original
    assert dict_wire.element_type == DataType.ARRAY
    assert dict_wire.type_schema.array_element.array_element.leaf_type == DataType.VARCHAR
    assert dict_wire.type_schema.array_element.array_element.WhichOneof("kind") == "leaf_type"
    assert dict_wire.nullable and not dict_wire.type_schema.nullable
    assert dict_wire.type_schema.array_element.nullable
    assert {kv.key: kv.value for kv in dict_wire.type_schema.array_element.type_params} == {
        "max_capacity": "8"
    }
    assert FieldSchema.construct_from_dict(ResponseFieldSchema(dict_wire).dict()) == field


@pytest.mark.parametrize(
    "type_schema",
    [
        [],
        {},
        {"leaf_type": DataType.INT32},
        {"array_element": {"leaf_type": DataType.INT32}},
        {"array_element": {}},
        {"array_element": {"array_element": {"leaf_type": DataType.ARRAY}}},
        {"array_element": {"array_element": {"leaf_type": DataType.STRUCT}}},
        {"array_element": {"array_element": {"leaf_type": DataType.FLOAT_VECTOR}}},
        {"array_element": {"array_element": {"leaf_type": DataType.INT32}, "nullable": "true"}},
        {"array_element": {"array_element": {"leaf_type": DataType.INT32}, "type_params": []}},
        {
            "leaf_type": DataType.INT32,
            "array_element": {"array_element": {"leaf_type": DataType.INT32}},
        },
        {
            "array_element": {"array_element": {"leaf_type": DataType.INT32}},
            "nullable": True,
        },
    ],
)
def test_schema_entry_points_reject_invalid_nested_types(type_schema):
    raw = {"name": "nested", "type": DataType.ARRAY, "type_schema": type_schema}
    for create in (FieldSchema.construct_from_dict, Prepare.get_field_schema):
        with pytest.raises((ParamError, DataTypeNotSupportException)):
            create(raw)


@pytest.mark.parametrize(
    "field_options",
    [
        {"type": DataType.INT32},
        {"element_type": DataType.INT32},
        {"params": []},
        {"params": {"max_capacity": 32}},
    ],
)
def test_dictionary_schema_rejects_conflicting_field_options(field_options):
    raw = {
        "name": "nested",
        "type": DataType.ARRAY,
        "type_schema": {
            "array_element": {"array_element": {"leaf_type": DataType.INT32}},
            "type_params": {"max_capacity": 16},
        },
        **field_options,
    }
    with pytest.raises(ParamError):
        Prepare.get_field_schema(raw)


def test_describe_reads_capacity_from_recursive_schema():
    field = FieldSchema(
        "nested",
        DataType.ARRAY,
        max_capacity=16,
        type_schema={
            "array_element": {
                "array_element": {"leaf_type": DataType.INT32},
                "type_params": {"max_capacity": 8},
            },
        },
    )
    wire, _, _ = Prepare.get_field_schema(field.to_dict())
    wire.ClearField("type_params")
    assert FieldSchema.construct_from_dict(ResponseFieldSchema(wire).dict()) == field


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (DataType.BOOL, [True, False]),
        (DataType.INT32, [1, -2]),
        (DataType.INT64, [2**40]),
        (DataType.DOUBLE, [1.25, 2.5]),
        (DataType.VARCHAR, ["hello", "你好"]),
    ],
)
def test_array_wire_roundtrip_preserves_shape_and_empty_arrays(depth, dtype, values):
    info = {"name": "array", "data_type": DataType.ARRAY, "element_type": dtype}
    type_schema = {"array_element": {"leaf_type": dtype}}
    for _ in range(depth - 1):
        type_schema = {"array_element": type_schema}
        info = {"name": "array", "type_schema": type_schema}
        values = [values, []]
    element_type = dtype if depth == 1 else DataType.ARRAY
    encoded = convert_to_array(values, info)
    restored = schema_pb2.ScalarField.FromString(encoded.SerializeToString())
    assert decode_array(restored, element_type) == values
    assert extract_array_row_data([restored], element_type) == [values]


@pytest.mark.parametrize("value", [None, [None]])
def test_nested_array_does_not_replace_none_with_empty_array(value):
    info = {
        "name": "array",
        "type_schema": {"array_element": {"array_element": {"leaf_type": DataType.INT32}}},
    }
    with pytest.raises(TypeError):
        convert_to_array(value, info)


@pytest.mark.parametrize("value", [((1, 2), (3, 4)), np.array([[1, 2], [3, 4]])])
def test_nested_array_accepts_tuples_and_numpy(value):
    info = {
        "name": "array",
        "type_schema": {"array_element": {"array_element": {"leaf_type": DataType.INT32}}},
    }
    assert decode_array(convert_to_array(value, info), DataType.ARRAY) == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    "method", ["row_insert_param", "row_upsert_param", "batch_insert_param", "batch_upsert_param"]
)
def test_mutation_requests_preserve_nested_rows(method):
    fields = [
        FieldSchema("pk", DataType.INT64, is_primary=True).to_dict(),
        FieldSchema(
            "nested",
            DataType.ARRAY,
            max_capacity=8,
            type_schema={
                "array_element": {
                    "array_element": {"leaf_type": DataType.INT32},
                    "type_params": {"max_capacity": 4},
                },
            },
        ).to_dict(),
    ]
    values = [[[1, 2], []], [], [[3]]]
    if method.startswith("row"):
        entities = [{"pk": i, "nested": value} for i, value in enumerate(values)]
    else:
        entities = [
            {"name": "pk", "type": DataType.INT64, "values": list(range(3))},
            {"name": "nested", "type": DataType.ARRAY, "values": values},
        ]
    request = getattr(Prepare, method)("test", entities, "", fields_info=fields)
    data = next(field for field in request.fields_data if field.field_name == "nested")
    data = schema_pb2.FieldData.FromString(data.SerializeToString())
    assert request.num_rows == 3
    assert data.scalars.array_data.element_type == DataType.ARRAY
    assert decode_range(data, 0, 3) == values


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("method", ["row_insert_param", "row_upsert_param"])
def test_struct_array_schema_and_data_roundtrip(nested, method):
    struct = StructFieldSchema()
    struct.name, struct.max_capacity = "metadata", 16
    options = {"element_type": DataType.VARCHAR, "max_length": 32}
    value = ["hello", "你好"]
    if nested:
        options = {
            "type_schema": {
                "array_element": {
                    "array_element": {
                        "leaf_type": DataType.VARCHAR,
                        "type_params": {"max_length": 32},
                    },
                    "type_params": {"max_capacity": 4},
                },
            },
        }
        value = [value, []]
    struct.add_field("tags", DataType.ARRAY, max_capacity=8, mmap_enabled=True, **options)
    wire_schema = Prepare.get_struct_array_field_schema(struct)
    internal = StructArrayFieldSchema(wire_schema).dict()
    public = convert_struct_fields_to_user_format([internal])[0]
    rebuilt = StructFieldSchema.construct_from_dict(public)
    assert rebuilt.fields[0] == struct.fields[0]
    assert rebuilt.max_capacity == 16
    assert Prepare.get_struct_array_field_schema(rebuilt) == wire_schema

    values = [[{"tags": value}, {"tags": []}], []]
    fields = [FieldSchema("pk", DataType.INT64, is_primary=True).to_dict()]
    request = getattr(Prepare, method)(
        "test",
        [{"pk": i, "metadata": value} for i, value in enumerate(values)],
        "",
        fields_info=fields,
        struct_fields_info=[internal],
    )
    data = next(field for field in request.fields_data if field.field_name == "metadata")
    data = schema_pb2.FieldData.FromString(data.SerializeToString())
    assert data.struct_arrays.fields[0].scalars.array_data.element_type == DataType.ARRAY
    assert [
        extract_struct_array_from_column_data(data.struct_arrays, i) for i in range(2)
    ] == values


@pytest.mark.parametrize("element_type", [None, DataType.ARRAY, DataType.FLOAT_VECTOR])
def test_struct_array_requires_encodable_element_type(element_type):
    struct = StructFieldSchema().add_field("tags", DataType.ARRAY, element_type=element_type)
    with pytest.raises(ParamError, match="Unsupported element type"):
        struct._check_fields()


@pytest.mark.parametrize(
    "make_values",
    [list, tuple, lambda v: array("i", v), pd.Series, iter, np.array, lambda v: range(len(v))],
)
def test_flat_array_retains_iterable_inputs_and_wire_format(make_values):
    field = FieldSchema("values", DataType.ARRAY, element_type=DataType.INT32).to_dict()
    request = Prepare.row_insert_param(
        "test",
        [{"pk": 1, "values": make_values([0, 1, 2])}],
        "",
        fields_info=[FieldSchema("pk", DataType.INT64, is_primary=True).to_dict(), field],
    )
    data = next(field for field in request.fields_data if field.field_name == "values")
    expected = schema_pb2.FieldData(type=DataType.ARRAY, field_name="values")
    expected.scalars.array_data.data.add().int_data.data.extend([0, 1, 2])
    assert data.SerializeToString() == expected.SerializeToString()


@pytest.mark.parametrize(
    "method", ["row_insert_param", "row_upsert_param", "batch_insert_param", "batch_upsert_param"]
)
def test_flat_array_mutation_requests_do_not_add_element_type(method):
    fields = [
        FieldSchema("pk", DataType.INT64, is_primary=True).to_dict(),
        FieldSchema("values", DataType.ARRAY, element_type=DataType.INT32).to_dict(),
    ]
    values = [[1, 2], []]
    if method.startswith("row"):
        entities = [{"pk": i, "values": value} for i, value in enumerate(values)]
    else:
        entities = [
            {"name": "pk", "type": DataType.INT64, "values": [0, 1]},
            {"name": "values", "type": DataType.ARRAY, "values": values},
        ]
    request = getattr(Prepare, method)("test", entities, "", fields_info=fields)
    data = next(field for field in request.fields_data if field.field_name == "values")
    assert data.scalars.array_data.element_type == DataType.NONE
    assert [list(cell.int_data.data) for cell in data.scalars.array_data.data] == values


def test_flat_array_results_keep_empty_arrays_and_protobuf_containers():
    data = schema_pb2.FieldData(type=DataType.ARRAY)
    data.scalars.array_data.element_type = DataType.INT32
    empty = data.scalars.array_data.data.add()
    populated = data.scalars.array_data.data.add()
    populated.int_data.data.extend([1, 2])
    assert decode_range(data, 0, 2) == [[], [1, 2]]
    result = extract_array_row_data([empty, populated, None], DataType.INT32)
    assert result == [[], [1, 2], None]
    assert result[1] is populated.int_data.data


def test_flat_array_results_use_declared_element_type():
    data = schema_pb2.FieldData(type=DataType.ARRAY)
    data.scalars.array_data.element_type = DataType.INT32
    data.scalars.array_data.data.add().string_data.data.append("a")
    assert decode_range(data, 0, 1) == [[]]
    assert extract_array_row_data(data.scalars.array_data.data, DataType.INT32) == [[]]


def test_flat_array_empty_results_keep_unsupported_type_error():
    with pytest.raises(MilvusException, match="Unsupported data type"):
        extract_array_row_data([], DataType.JSON)


def test_flat_array_helpers_keep_support_for_plain_objects():
    cell = SimpleNamespace(int_data=SimpleNamespace(data=[1, 2]))
    assert array_cell_length(cell) == 2
    assert decode_array_value(cell, 1) == 2
    assert array_cell_length(SimpleNamespace()) == 0
    assert decode_array_value(SimpleNamespace(), 0) is None
