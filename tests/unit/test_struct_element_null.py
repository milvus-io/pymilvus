"""Struct element null contracts at the public schema and protobuf boundaries."""

import copy
from unittest.mock import AsyncMock, MagicMock

import grpc
import numpy as np
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.client.abstract import StructArrayFieldSchema
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.cache import GlobalCache
from pymilvus.client.entity_helper import (
    convert_to_array,
    convert_to_array_of_vector,
    entity_to_field_data,
    extract_struct_array_from_column_data,
)
from pymilvus.client.field_data_extractors import (
    decode_cell,
    decode_scalar_array,
)
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.prepare import Prepare
from pymilvus.client.search_result import SearchResult
from pymilvus.client.types import HybridExtraList
from pymilvus.client.utils import convert_struct_fields_to_user_format, len_of
from pymilvus.exceptions import DataNotMatchException, MilvusException, ParamError
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2
from pymilvus.orm.schema import StructFieldSchema

PK = {"name": "id", "type": DataType.INT64, "is_primary": True}


def make_schema(row_nullable=True, child_nullable=True, leaf_nullable=True):
    struct = StructFieldSchema()
    struct.add_field("embedding", DataType.FLOAT_VECTOR, dim=2, nullable=child_nullable)
    struct.add_field("score", DataType.INT64, nullable=child_nullable)
    struct.add_field(
        "tags",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=8,
        max_length=32,
        nullable=child_nullable,
        element_nullable=leaf_nullable,
    )
    schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=struct,
        max_capacity=16,
        nullable=row_nullable,
    )
    return schema


def wire_info(schema):
    proto = Prepare.get_schema_from_collection_schema("videos", schema)
    return StructArrayFieldSchema(proto.struct_array_fields[0]).dict()


@pytest.mark.parametrize("row_nullable", [False, True])
@pytest.mark.parametrize("child_nullable", [False, True])
@pytest.mark.parametrize("leaf_nullable", [False, True])
def test_schema_roundtrip(row_nullable, child_nullable, leaf_nullable):
    schema = make_schema(row_nullable, child_nullable, leaf_nullable)
    snapshot = schema.to_dict()
    wire = Prepare.get_schema_from_collection_schema("videos", schema)
    scalar, nested = wire.struct_array_fields[0].fields[1:]
    assert scalar.nullable == row_nullable
    assert scalar.element_nullable == child_nullable
    assert not scalar.HasField("type_schema")
    assert nested.data_type == DataType.ARRAY
    assert nested.element_type == DataType.ARRAY
    root = nested.type_schema
    assert root.nullable == row_nullable
    assert root.array_element.nullable == child_nullable
    assert root.array_element.array_element.nullable == leaf_nullable
    assert {kv.key: kv.value for kv in root.type_params} == {"max_capacity": "16"}
    assert {kv.key: kv.value for kv in root.array_element.type_params} == {"max_capacity": "8"}
    assert root.array_element.array_element.leaf_type == DataType.VARCHAR
    user_fields = convert_struct_fields_to_user_format([wire_info(schema)])
    restored = CollectionSchema.construct_from_dict({"fields": [PK, *user_fields]})
    assert Prepare.get_schema_from_collection_schema("videos", restored) == wire
    assert copy.deepcopy(schema).to_dict() == snapshot
    assert schema.to_dict() == snapshot


@pytest.mark.parametrize("value", [1, "true", None])
def test_element_nullable_requires_boolean(value):
    with pytest.raises(ParamError, match="boolean"):
        FieldSchema("tags", DataType.ARRAY, element_type=DataType.INT64, element_nullable=value)


@pytest.mark.parametrize("struct_format", ["struct_fields", "struct_array_fields", "fields"])
def test_dict_schema_rejects_struct_fields(struct_format):
    schema = make_schema()
    if struct_format == "struct_fields":
        raw = schema.to_dict()
    elif struct_format == "struct_array_fields":
        raw = {"fields": [PK], "struct_array_fields": [wire_info(schema)]}
    else:
        raw = {"fields": [PK, *convert_struct_fields_to_user_format([wire_info(schema)])]}
    with pytest.raises(ParamError, match="CollectionSchema and StructFieldSchema"):
        Prepare.get_schema("videos", raw)


def test_dict_schema_without_struct_keeps_field_params():
    price = {"name": "price", "type": DataType.INT64, "params": {"local_format": "vortex"}}
    proto = Prepare.get_schema("videos", {"fields": [PK, price]}, enable_dynamic_field=True)
    expected = schema_pb2.CollectionSchema(
        name="videos",
        enable_dynamic_field=True,
        fields=[
            schema_pb2.FieldSchema(name="id", data_type=DataType.INT64, is_primary_key=True),
            schema_pb2.FieldSchema(
                name="price",
                data_type=DataType.INT64,
                type_params=[common_pb2.KeyValuePair(key="local_format", value="vortex")],
            ),
        ],
    )
    assert proto.SerializeToString() == expected.SerializeToString()


@pytest.mark.parametrize("upsert", [False, True])
def test_ordinary_array_tuple_input_is_preserved(upsert):
    array = {
        "name": "values",
        "type": DataType.ARRAY,
        "element_type": DataType.INT64,
        "params": {"max_capacity": 8},
    }
    prepare = Prepare.row_upsert_param if upsert else Prepare.row_insert_param
    request = prepare("videos", [{"id": 1, "values": (1, 2)}], "", fields_info=[PK, array])
    assert list(request.fields_data[1].scalars.array_data.data[0].long_data.data) == [1, 2]
    column = entity_to_field_data(
        {"name": "values", "type": DataType.ARRAY, "values": [(1, 2)]}, array, 1
    )
    assert list(column.scalars.array_data.data[0].long_data.data) == [1, 2]
    with pytest.raises(DataNotMatchException):
        prepare("videos", [{"id": 1, "values": (1, None)}], "", fields_info=[PK, array])


def test_ordinary_array_round_trips_without_validity():
    info = {"name": "values", "element_type": DataType.INT64}
    cell = convert_to_array([1, 2, 3], info)
    assert cell.WhichOneof("data") == "long_data"
    assert not cell.valid_data
    assert decode_scalar_array(cell) == [1, 2, 3]
    assert convert_to_array([], info).WhichOneof("data") == "long_data"
    with pytest.raises(TypeError, match="does not allow null elements"):
        convert_to_array([1, None], info)


@pytest.mark.parametrize("value", ["ab", b"ab"])
def test_ordinary_array_string_input_remains_rejected(value):
    array = {
        "name": "values",
        "type": DataType.ARRAY,
        "element_type": DataType.VARCHAR,
        "params": {"max_capacity": 8, "max_length": 16},
    }
    with pytest.raises(ParamError, match="expects a list"):
        Prepare.row_insert_param(
            "videos", [{"id": 1, "values": value}], "", fields_info=[PK, array]
        )


@pytest.mark.parametrize("make", [lambda: range(1, 4), lambda: (v for v in (1, 2, 3))])
def test_ordinary_array_accepts_any_iterable(make):
    array = {
        "name": "values",
        "type": DataType.ARRAY,
        "element_type": DataType.INT64,
        "params": {"max_capacity": 8},
    }
    request = Prepare.row_insert_param(
        "videos", [{"id": 1, "values": make()}], "", fields_info=[PK, array]
    )
    assert list(request.fields_data[1].scalars.array_data.data[0].long_data.data) == [1, 2, 3]
    column = entity_to_field_data(
        {"name": "values", "type": DataType.ARRAY, "values": [make()]}, array, 1
    )
    assert list(column.scalars.array_data.data[0].long_data.data) == [1, 2, 3]


def test_element_nullable_array_rejects_set():
    info = {"name": "values", "element_type": DataType.INT64, "element_nullable": True}
    assert list(convert_to_array((1, None), info).valid_data) == [True, False]
    with pytest.raises(ParamError, match="expects a list"):
        convert_to_array({1, 2}, info)


@pytest.mark.parametrize(
    "leaf", [DataType.ARRAY, DataType.JSON, DataType.STRUCT, DataType.FLOAT_VECTOR]
)
def test_array_subfield_rejects_unsupported_leaf(leaf):
    struct = StructFieldSchema()
    struct.add_field("tags", DataType.ARRAY, element_type=leaf, max_capacity=8)
    with pytest.raises(ParamError, match="scalar element_type"):
        struct._check_fields()


@pytest.mark.parametrize("upsert", [False, True])
def test_row_write_compacts_each_nullable_level(upsert):
    structs = [wire_info(make_schema())]
    rows = [
        {"id": 1, "clips": None},
        {
            "id": 2,
            "clips": [
                {"embedding": [1, 2], "score": 10, "tags": ["a", None]},
                {"embedding": None, "score": None, "tags": None},
                {"embedding": [3, 4], "score": 30, "tags": []},
                {},
            ],
        },
        {"id": 3, "clips": []},
    ]
    before = copy.deepcopy(rows)
    prepare = Prepare.row_upsert_param if upsert else Prepare.row_insert_param
    request = prepare("videos", rows, "", fields_info=[PK], struct_fields_info=structs)
    fields = request.fields_data[1].struct_arrays.fields
    vector, score, tags = fields
    for field in fields:
        assert list(field.valid_data) == [False, True, True]
        payload = field.vectors if field.type == DataType._ARRAY_OF_VECTOR else field.scalars
        assert list(payload.valid_data) == [False, True, True]
    vector_row = vector.vectors.vector_array.data[0]
    assert list(vector_row.valid_data) == [True, False, True, False]
    assert list(vector_row.float_vector.data) == [1, 2, 3, 4]
    score_row = score.scalars.array_data.data[0]
    assert list(score_row.long_data.data) == [10, 30]
    assert list(score_row.valid_data) == [True, False, True, False]
    tags_row = tags.scalars.array_data.data[0]
    assert tags.scalars.array_data.element_type == DataType.ARRAY
    assert tags_row.array_data.element_type == DataType.VARCHAR
    assert list(tags_row.valid_data) == [True, False, True, False]
    assert len(tags_row.array_data.data) == 2
    leaf = tags_row.array_data.data[0]
    assert list(leaf.string_data.data) == ["a"]
    assert list(leaf.valid_data) == [True, False]
    assert tags_row.array_data.data[1].WhichOneof("data") == "string_data"
    assert rows == before


def test_partial_upsert_omitted_struct_is_unchanged_and_present_struct_uses_nulls():
    structs = [wire_info(make_schema())]
    omitted = Prepare.row_upsert_param(
        "videos",
        [{"id": 1}],
        "",
        fields_info=[PK],
        struct_fields_info=structs,
        partial_update=True,
    )
    assert [field.field_name for field in omitted.fields_data] == ["id"]
    present = Prepare.row_upsert_param(
        "videos",
        [{"id": 1, "clips": [{}, {}]}],
        "",
        fields_info=[PK],
        struct_fields_info=structs,
        partial_update=True,
    )
    vector = present.fields_data[1].struct_arrays.fields[0].vectors.vector_array.data[0]
    assert list(vector.valid_data) == [False, False]
    assert len(vector.float_vector.data) == 0


@pytest.mark.parametrize("clips", [[None], [{"unknown": 1}], [{"score": None}]])
def test_row_write_rejects_unknown_or_nonnullable_values(clips):
    with pytest.raises(DataNotMatchException):
        Prepare.row_insert_param(
            "videos",
            [{"id": 1, "clips": clips}],
            "",
            fields_info=[PK],
            struct_fields_info=[wire_info(make_schema(child_nullable=False))],
        )


def test_row_write_rejects_nonnullable_leaf():
    info = wire_info(make_schema(leaf_nullable=False))["fields"][2]
    with pytest.raises(ParamError, match="null elements"):
        convert_to_array([[None]], info)


@pytest.mark.parametrize(
    "dtype,attr,first,last,zero",
    [
        (DataType.BOOL, "bool_data", True, False, False),
        (DataType.INT8, "int_data", 1, 2, 0),
        (DataType.INT16, "int_data", 1, 2, 0),
        (DataType.INT32, "int_data", 1, 2, 0),
        (DataType.INT64, "long_data", 1, 2, 0),
        (DataType.FLOAT, "float_data", 1.0, 2.0, 0.0),
        (DataType.DOUBLE, "double_data", 1.0, 2.0, 0.0),
        (DataType.VARCHAR, "string_data", "a", "b", ""),
    ],
)
def test_scalar_input_is_compact_and_output_is_dense(dtype, attr, first, last, zero):
    info = {"name": "scalar", "element_type": dtype, "element_nullable": True}
    encoded = convert_to_array([first, None, last], info)
    assert list(getattr(encoded, attr).data) == [first, last]
    assert list(encoded.valid_data) == [True, False, True]
    getattr(encoded, attr).data.insert(1, zero)
    assert decode_scalar_array(encoded) == [first, None, last]
    empty = convert_to_array([], info)
    assert empty.WhichOneof("data") == attr
    assert decode_scalar_array(empty) == []


def test_null_struct_output_keeps_later_payload_row_aligned():
    field = dense_struct_output()
    for child in field.struct_arrays.fields:
        child.valid_data.extend([False, True])
        if child.type == DataType._ARRAY_OF_VECTOR:
            child.vectors.valid_data.extend([False, True])
            placeholder = schema_pb2.VectorField(dim=2, float_vector=schema_pb2.FloatArray())
            child.vectors.vector_array.data.insert(0, placeholder)
        else:
            child.scalars.valid_data.extend([False, True])
            child.scalars.array_data.data.insert(0, schema_pb2.ScalarField())
    assert extract_struct_array_from_column_data(field.struct_arrays, 0) is None
    assert extract_struct_array_from_column_data(field.struct_arrays, 1) == EXPECTED


VECTOR_CASES = [
    (DataType.FLOAT_VECTOR, [1.0, 2.0], [3.0, 4.0], 2),
    (DataType.BINARY_VECTOR, b"\x01", b"\x02", 8),
    (
        DataType.FLOAT16_VECTOR,
        np.array([1, 2], dtype="float16"),
        np.array([3, 4], dtype="float16"),
        2,
    ),
    (DataType.BFLOAT16_VECTOR, b"\x80\x3f\x00\x40", b"\x40\x40\x80\x40", 2),
    (DataType.INT8_VECTOR, np.array([1, 2], dtype="int8"), np.array([3, 4], dtype="int8"), 2),
]


@pytest.mark.parametrize("dtype,first,last,dim", VECTOR_CASES)
def test_vector_nulls_keep_logical_positions(dtype, first, last, dim):
    info = {"name": "v", "element_type": dtype, "element_nullable": True, "params": {"dim": dim}}
    row = convert_to_array_of_vector([None, first, None, last], info)
    field = schema_pb2.FieldData(type=DataType._ARRAY_OF_VECTOR)
    field.vectors.vector_array.element_type = dtype
    field.vectors.vector_array.data.append(row)
    values = decode_cell(field, 0)
    assert len(values) == 4
    assert values[0] is None and values[2] is None
    assert values[1] is not None and values[3] is not None
    empty = convert_to_array_of_vector([None, None], info)
    field.vectors.vector_array.data[0].CopyFrom(empty)
    assert decode_cell(field, 0) == [None, None]


@pytest.mark.parametrize("dtype,first,last,dim", VECTOR_CASES)
def test_vector_rejects_bad_width_before_flattening(dtype, first, last, dim):
    info = {"name": "v", "element_type": dtype, "element_nullable": True, "params": {"dim": dim}}
    bad = [1] if dtype == DataType.FLOAT_VECTOR else b""
    with pytest.raises(ParamError, match=r"dimension|width"):
        convert_to_array_of_vector([None, bad], info)


def dense_struct_output():
    result = schema_pb2.FieldData(field_name="clips", type=DataType._ARRAY_OF_STRUCT)
    vector = result.struct_arrays.fields.add(
        field_name="clips[embedding]", type=DataType._ARRAY_OF_VECTOR
    )
    vector.vectors.vector_array.element_type = DataType.FLOAT_VECTOR
    vector.vectors.vector_array.data.add(
        dim=2, valid_data=[False, True, False]
    ).float_vector.data.extend([3, 4])
    score = result.struct_arrays.fields.add(field_name="clips[score]", type=DataType.ARRAY)
    score.scalars.array_data.data.add(valid_data=[True, False, True]).long_data.data.extend(
        [10, 0, 30]
    )
    tags = result.struct_arrays.fields.add(field_name="clips[tags]", type=DataType.ARRAY)
    row = tags.scalars.array_data.data.add(valid_data=[True, False, True])
    row.array_data.element_type = DataType.VARCHAR
    row.array_data.data.add(valid_data=[True, False]).string_data.data.extend(["a", ""])
    row.array_data.data.add().string_data.SetInParent()
    row.array_data.data.add().string_data.SetInParent()
    return result


EXPECTED = [
    {"embedding": None, "score": 10, "tags": ["a", None]},
    {"embedding": [3, 4], "score": None, "tags": None},
    {"embedding": None, "score": 30, "tags": []},
]


def test_dense_scalar_and_compact_vector_struct_output():
    field = dense_struct_output()
    assert extract_struct_array_from_column_data(field.struct_arrays, 0) == EXPECTED
    lazy = HybridExtraList([field], [{}])
    assert lazy[0]["clips"] == EXPECTED
    response = schema_pb2.SearchResultData(
        num_queries=1,
        topks=[1],
        scores=[1],
        fields_data=[field],
        ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
        element_indices=schema_pb2.LongArray(data=[1]),
    )
    hits = SearchResult(response)
    assert hits[0][0]["offset"] == 1
    assert hits[0][0]["entity"]["clips"] == EXPECTED


@pytest.mark.parametrize("current", [False, True])
def test_row_validity_fallback_and_vector_prefix(current):
    field = schema_pb2.FieldData(type=DataType.FLOAT_VECTOR, field_name="v")
    field.vectors.dim = 2
    field.vectors.float_vector.data.extend([1, 2, 3, 4])
    mask = field.vectors.valid_data if current else field.valid_data
    mask.extend([False, True, False, True])
    assert len_of(field) == 4
    assert [decode_cell(field, i) for i in range(4)] == [None, [1, 2], None, [3, 4]]
    lazy = HybridExtraList([field], [{}, {}, {}, {}])
    assert [item["v"] for item in lazy] == [None, [1, 2], None, [3, 4]]


def test_legacy_row_bitmap_wins_over_field_specific():
    field = schema_pb2.FieldData(type=DataType.ARRAY, valid_data=[False])
    field.scalars.valid_data.extend([True])
    field.scalars.array_data.data.add().long_data.data.extend([10])
    assert decode_cell(field, 0) is None


def test_legacy_vector_row_bitmap_controls_compact_index():
    field = schema_pb2.FieldData(
        type=DataType.FLOAT_VECTOR, field_name="v", valid_data=[True, False]
    )
    field.vectors.dim = 2
    field.vectors.valid_data.extend([False, True])
    field.vectors.float_vector.data.extend([1, 2])
    assert [decode_cell(field, index) for index in range(2)] == [[1, 2], None]
    lazy = HybridExtraList([field], [{}, {}])
    assert [row["v"] for row in lazy] == [[1, 2], None]


def test_element_bitmap_stays_local_when_row_bitmaps_differ():
    field = schema_pb2.FieldData(type=DataType.ARRAY, valid_data=[True])
    field.scalars.valid_data.append(False)
    field.scalars.array_data.data.add(valid_data=[False, True]).long_data.data.extend([0, 10])
    assert decode_cell(field, 0) == [None, 10]


def test_malformed_struct_lengths_are_errors():
    field = dense_struct_output()
    field.struct_arrays.fields[1].scalars.array_data.data[0].valid_data.append(True)
    with pytest.raises(MilvusException, match="lengths differ"):
        extract_struct_array_from_column_data(field.struct_arrays, 0)
    field = dense_struct_output()
    field.struct_arrays.fields[1].scalars.array_data.data[0].long_data.data.append(40)
    field.struct_arrays.fields[1].scalars.array_data.data[0].valid_data.append(True)
    with pytest.raises(MilvusException, match="element lengths"):
        extract_struct_array_from_column_data(field.struct_arrays, 0)


def test_malformed_compact_vector_and_row_validity_are_errors():
    field = dense_struct_output()
    field.struct_arrays.fields[0].vectors.vector_array.data[0].float_vector.data.append(5)
    with pytest.raises(MilvusException, match="width"):
        extract_struct_array_from_column_data(field.struct_arrays, 0)
    field = dense_struct_output()
    field.struct_arrays.fields[0].vectors.valid_data.append(False)
    with pytest.raises(MilvusException, match="row validity"):
        extract_struct_array_from_column_data(field.struct_arrays, 0)


@pytest.mark.parametrize("offsets", [[0, 2], [0], [0, 2, 3]])
def test_search_offsets_are_logical_and_validated(offsets):
    response = schema_pb2.SearchResultData(
        num_queries=3,
        topks=[1, 0, 1],
        scores=[1, 2],
        ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[7, 7])),
        element_indices=schema_pb2.LongArray(data=offsets),
    )
    if len(offsets) == 2:
        result = SearchResult(response)
        assert result[0][0]["offset"] == 0
        assert result[1] == []
        assert result[2][0]["offset"] == 2
    else:
        with pytest.raises(MilvusException, match="element_indices"):
            SearchResult(response)


@pytest.mark.parametrize(
    "name", ["clips[]", "[lang]", "clips[lang][x]", "clips[lang]suffix", "clips[lang"]
)
def test_add_rejects_malformed_paths(name):
    with pytest.raises(ParamError, match="path"):
        Prepare.struct_field_path(name)


@pytest.mark.parametrize("array", [False, True])
def test_add_subfield_request_wraps_one_element_without_parent_schema(array):
    kwargs = {"nullable": True}
    if array:
        kwargs.update(
            element_type=DataType.VARCHAR, element_nullable=True, max_capacity=8, max_length=16
        )
    field = FieldSchema("clips[tags]", DataType.ARRAY if array else DataType.INT64, **kwargs)
    wrapped = Prepare.prepare_collection_field(field)
    request = Prepare.alter_collection_schema_request("videos", field_schema=wrapped)
    info = request.action.add_request.field_infos[0]
    assert info.field_schema.name == "clips[tags]"
    assert not info.field_schema.nullable
    assert info.field_schema.element_nullable
    if array:
        assert info.field_schema.type_schema.array_element.array_element.nullable
        assert not info.field_schema.type_schema.type_params
    assert field.name == "clips[tags]"
    drop = Prepare.alter_collection_schema_request("videos", drop_field_name="clips[tags]")
    assert drop.action.drop_request.field_name == "clips[tags]"


def test_add_subfield_requires_nullable_and_rejects_index_binding():
    with pytest.raises(ParamError, match="nullable=True"):
        Prepare.prepare_collection_field(FieldSchema("clips[x]", DataType.INT64))
    field = Prepare.prepare_collection_field(FieldSchema("clips[x]", DataType.INT64, nullable=True))
    with pytest.raises(ParamError, match="function or index"):
        Prepare.alter_collection_schema_request("videos", field_schema=field, index_name="index")


def test_add_scalar_preserves_capacity_for_server_validation():
    field = FieldSchema(
        "clips[lang]", DataType.VARCHAR, max_length=16, max_capacity=16, nullable=True
    )
    proto = Prepare.prepare_collection_field(field)
    assert {kv.key: kv.value for kv in proto.type_params} == {
        "max_length": "16",
        "max_capacity": "16",
    }
    field = FieldSchema("clips[lang]", DataType.INT64, max_capacity=8, nullable=True)
    proto = Prepare.prepare_collection_field(field)
    assert {kv.key: kv.value for kv in proto.type_params} == {"max_capacity": "8"}
    assert proto.name == "clips[lang]"


def bare_handler(asynchronous=False):
    handler_type = AsyncGrpcHandler if asynchronous else GrpcHandler
    handler = handler_type.__new__(handler_type)
    handler._invalidate_schema = MagicMock()
    handler._get_schema = (AsyncMock if asynchronous else MagicMock)(
        return_value=({"struct_array_fields": [wire_info(make_schema(False))]}, 0)
    )
    stub = MagicMock()
    stub.AlterCollectionSchema = (AsyncMock if asynchronous else MagicMock)(
        return_value=milvus_pb2.AlterCollectionSchemaResponse(alter_status=common_pb2.Status())
    )
    stub.AddCollectionField = AsyncMock() if asynchronous else MagicMock()
    if asynchronous:
        handler._async_stub = stub
    else:
        handler._stub = stub
    return handler, stub


def test_sync_row_prepare_uses_element_nullable_metadata():
    handler, _ = bare_handler()
    handler._get_schema.return_value = (
        {"fields": [PK], "struct_array_fields": [wire_info(make_schema())]},
        0,
    )
    rows = [{"id": 1, "clips": [{}, {}]}]
    insert = handler._prepare_row_insert_request("videos", rows)
    upsert = handler._prepare_row_upsert_request("videos", rows)
    for request in (insert, upsert):
        vector = request.fields_data[1].struct_arrays.fields[0].vectors.vector_array.data[0]
        assert list(vector.valid_data) == [False, False]


@pytest.mark.asyncio
async def test_async_row_prepare_uses_element_nullable_metadata():
    handler, _ = bare_handler(True)
    handler._get_schema.return_value = (
        {"fields": [PK], "struct_array_fields": [wire_info(make_schema())]},
        0,
    )
    rows = [{"id": 1, "clips": [{}, {}]}]
    insert = await handler._prepare_row_insert_request("videos", rows)
    upsert = await handler._prepare_row_upsert_request("videos", rows)
    for request in (insert, upsert):
        vector = request.fields_data[1].struct_arrays.fields[0].vectors.vector_array.data[0]
        assert list(vector.valid_data) == [False, False]


def test_sync_add_subfield_and_unimplemented_never_uses_legacy():
    handler, stub = bare_handler()
    field = FieldSchema("clips[lang]", DataType.INT64, nullable=True)
    handler.add_collection_field.__wrapped__(handler, "videos", field)
    assert (
        stub.AlterCollectionSchema.call_args.args[0]
        .action.add_request.field_infos[0]
        .field_schema.name
        == "clips[lang]"
    )
    error = grpc.RpcError()
    error.code = lambda: grpc.StatusCode.UNIMPLEMENTED
    stub.AlterCollectionSchema.side_effect = error
    with pytest.raises(grpc.RpcError):
        handler.add_collection_field.__wrapped__(handler, "videos", field)
    stub.AddCollectionField.assert_not_called()


@pytest.mark.asyncio
async def test_async_add_subfield_and_unimplemented_never_uses_legacy():
    handler, stub = bare_handler(True)
    field = FieldSchema("clips[lang]", DataType.INT64, nullable=True)
    await handler.add_collection_field.__wrapped__(handler, "videos", field)
    assert (
        stub.AlterCollectionSchema.call_args.args[0]
        .action.add_request.field_infos[0]
        .field_schema.name
        == "clips[lang]"
    )
    error = grpc.RpcError()
    error.code = lambda: grpc.StatusCode.UNIMPLEMENTED
    stub.AlterCollectionSchema.side_effect = error
    with pytest.raises(grpc.RpcError):
        await handler.add_collection_field.__wrapped__(handler, "videos", field)
    stub.AddCollectionField.assert_not_called()


@pytest.mark.parametrize("operation", ["add_collection_field", "alter_collection_schema"])
def test_sync_add_subfield_reaches_server_when_cache_predates_parent(operation):
    handler = GrpcHandler(channel=MagicMock())
    handler._stub = MagicMock()
    handler._stub.AlterCollectionSchema.return_value = milvus_pb2.AlterCollectionSchemaResponse(
        alter_status=common_pb2.Status()
    )
    GlobalCache._reset_for_testing()
    GlobalCache.schema.set(handler.server_address, "", "videos", {"fields": [PK]})
    try:
        field = FieldSchema("clips[lang]", DataType.INT64, nullable=True)
        method = getattr(handler, operation)
        method.__wrapped__(handler, "videos", field)
        info = handler._stub.AlterCollectionSchema.call_args.args[0].action.add_request.field_infos[
            0
        ]
        assert info.field_schema.name == "clips[lang]"
        handler._stub.DescribeCollection.assert_not_called()
        assert GlobalCache.schema.get(handler.server_address, "", "videos") is None
    finally:
        GlobalCache._reset_for_testing()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add_collection_field", "alter_collection_schema"])
async def test_async_add_subfield_reaches_server_when_cache_predates_parent(operation):
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel)
    handler._async_stub = AsyncMock()
    handler._async_stub.AlterCollectionSchema.return_value = (
        milvus_pb2.AlterCollectionSchemaResponse(alter_status=common_pb2.Status())
    )
    GlobalCache._reset_for_testing()
    GlobalCache.schema.set(handler.server_address, "", "videos", {"fields": [PK]})
    try:
        field = FieldSchema("clips[lang]", DataType.INT64, nullable=True)
        method = getattr(handler, operation)
        await method.__wrapped__(handler, "videos", field)
        info = handler._async_stub.AlterCollectionSchema.call_args.args[
            0
        ].action.add_request.field_infos[0]
        assert info.field_schema.name == "clips[lang]"
        handler._async_stub.DescribeCollection.assert_not_called()
        assert GlobalCache.schema.get(handler.server_address, "", "videos") is None
    finally:
        GlobalCache._reset_for_testing()
