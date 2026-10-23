import csv
import gc
import json
import sys
from copy import deepcopy
from pathlib import Path

import ml_dtypes
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, StructFieldSchema
from pymilvus.bulk_writer import BulkFileType, LocalBulkWriter
from pymilvus.bulk_writer.buffer import Buffer
from pymilvus.exceptions import MilvusException

FORMATS = [BulkFileType.JSON, BulkFileType.JSONL, BulkFileType.PARQUET, BulkFileType.CSV]


def make_schema(children, *, nullable=True, capacity=4):
    struct = StructFieldSchema(nullable=nullable)
    for name, dtype, kwargs in children:
        struct.add_field(name, dtype, **kwargs)
    schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=struct,
        nullable=nullable,
        max_capacity=capacity,
    )
    return schema


def read_rows(file_path, file_type, config=None):
    config = config or {}
    if file_type == BulkFileType.PARQUET:
        return pq.read_table(file_path).to_pylist()
    if file_type == BulkFileType.JSON:
        return json.loads(file_path.read_text())["rows"]
    if file_type == BulkFileType.JSONL:
        return [json.loads(line) for line in file_path.read_text().splitlines()]
    with file_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=config.get("sep", ",")))
    for row in rows:
        row["id"] = int(row["id"])
        row["clips"] = (
            None if row["clips"] == config.get("nullkey", "") else json.loads(row["clips"])
        )
    return rows


def write_rows(tmp_path, schema, file_type, rows, config=None):
    with LocalBulkWriter(schema, str(tmp_path), file_type=file_type, config=config) as writer:
        for row in deepcopy(rows):
            writer.append_row(row)
        assert writer.total_row_count == len(rows)
        assert writer.buffer_row_count == len(rows)
        writer.commit()
        assert len(writer.batch_files) == 1
        assert len(writer.batch_files[0]) == 1
        return Path(writer.batch_files[0][0])


@pytest.mark.parametrize("file_type", FORMATS)
@pytest.mark.parametrize("config", [{}, {"sep": ";", "nullkey": "NULL"}])
def test_struct_nulls_and_missing_keys_round_trip(tmp_path, file_type, config):
    schema = make_schema(
        [
            ("seq", DataType.INT64, {}),
            ("emb", DataType.FLOAT_VECTOR, {"dim": 2, "nullable": True}),
            ("score", DataType.FLOAT, {"nullable": True}),
            (
                "tags",
                DataType.ARRAY,
                {
                    "element_type": DataType.INT64,
                    "max_capacity": 3,
                    "nullable": True,
                    "element_nullable": True,
                },
            ),
            ("label", DataType.VARCHAR, {"max_length": 20, "nullable": True}),
        ],
        capacity=3,
    )
    rows = [
        {"id": 1, "clips": None},
        {"id": 2},
        {"id": 3, "clips": []},
        {
            "id": 4,
            "clips": [
                {"seq": 1, "emb": None, "score": None, "tags": [10, None], "label": 'a,"b;null'},
                {"seq": 2, "emb": [0.25, 0.5], "tags": None, "deleted": object()},
                {"seq": 3, "score": 0.75, "tags": [], "label": "NULL"},
            ],
        },
    ]
    expected = [
        {"id": 1, "clips": None},
        {"id": 2, "clips": None},
        {"id": 3, "clips": []},
        {
            "id": 4,
            "clips": [
                {"seq": 1, "emb": None, "score": None, "tags": [10, None], "label": 'a,"b;null'},
                {"seq": 2, "emb": [0.25, 0.5], "score": None, "tags": None, "label": None},
                {"seq": 3, "emb": None, "score": 0.75, "tags": [], "label": "NULL"},
            ],
        },
    ]
    path = write_rows(tmp_path, schema, file_type, rows, config)
    assert read_rows(path, file_type, config) == expected


@pytest.mark.parametrize("file_type", FORMATS)
@pytest.mark.parametrize(
    "dtype,value,kwargs",
    [
        (DataType.BOOL, True, {}),
        (DataType.INT8, -12, {}),
        (DataType.INT16, 300, {}),
        (DataType.INT32, 70000, {}),
        (DataType.INT64, 2**40, {}),
        (DataType.FLOAT, 0.25, {}),
        (DataType.DOUBLE, 0.125, {}),
        (DataType.VARCHAR, "null", {"max_length": 8}),
    ],
)
def test_nullable_scalar_subfields(tmp_path, file_type, dtype, value, kwargs):
    schema = make_schema([("value", dtype, {**kwargs, "nullable": True})])
    rows = [{"id": 1, "clips": [{"value": value}, {"value": None}, {}, {"value": value}]}]
    path = write_rows(tmp_path, schema, file_type, rows)
    assert read_rows(path, file_type) == [
        {"id": 1, "clips": [{"value": value}, {"value": None}, {"value": None}, {"value": value}]}
    ]


@pytest.mark.parametrize("file_type", FORMATS)
@pytest.mark.parametrize("numpy_input", [False, True])
@pytest.mark.parametrize(
    "dtype,value,numpy_dtype,parquet_value",
    [
        (DataType.FLOAT_VECTOR, [1.0, 2.0], np.float32, [1.0, 2.0]),
        (DataType.FLOAT16_VECTOR, [1.0, 2.0], np.float16, [0, 60, 0, 64]),
        (DataType.BFLOAT16_VECTOR, [1.0, 2.0], ml_dtypes.bfloat16, [128, 63, 0, 64]),
        (DataType.INT8_VECTOR, [-1, 2], np.int8, [-1, 2]),
        (DataType.BINARY_VECTOR, [0, 1, 0, 1, 1, 0, 1, 0], np.uint8, [90]),
    ],
)
def test_null_vectors_keep_logical_positions(
    tmp_path, file_type, numpy_input, dtype, value, numpy_dtype, parquet_value
):
    schema = make_schema(
        [("v", dtype, {"dim": 8 if dtype == DataType.BINARY_VECTOR else 2, "nullable": True})]
    )
    if numpy_input:
        input_value = np.array(
            [90] if dtype == DataType.BINARY_VECTOR else value, dtype=numpy_dtype
        )
    else:
        input_value = value
    rows = [
        {"id": 1, "clips": [{"v": None}, {"v": input_value}, {}, {"v": input_value}]},
        {"id": 2, "clips": [{}, {"v": None}]},
    ]
    path = write_rows(tmp_path, schema, file_type, rows)
    expected_value = (
        parquet_value
        if file_type == BulkFileType.PARQUET or dtype == DataType.BINARY_VECTOR
        else value
    )
    assert read_rows(path, file_type) == [
        {
            "id": 1,
            "clips": [{"v": None}, {"v": expected_value}, {"v": None}, {"v": expected_value}],
        },
        {"id": 2, "clips": [{"v": None}, {"v": None}]},
    ]


@pytest.mark.parametrize("file_type", FORMATS)
def test_binary_vector_bytes_and_null(tmp_path, file_type):
    schema = make_schema([("v", DataType.BINARY_VECTOR, {"dim": 8, "nullable": True})])
    path = write_rows(tmp_path, schema, file_type, [{"id": 1, "clips": [{"v": b"\x5a"}, {}]}])
    assert read_rows(path, file_type) == [{"id": 1, "clips": [{"v": [90]}, {"v": None}]}]


@pytest.mark.parametrize("file_type", FORMATS)
@pytest.mark.parametrize(
    "leaf,value,kwargs",
    [
        (DataType.BOOL, np.bool_(True), {}),
        (DataType.INT8, np.int8(-12), {}),
        (DataType.INT16, np.int16(300), {}),
        (DataType.INT32, np.int32(70000), {}),
        (DataType.INT64, np.int64(2**40), {}),
        (DataType.FLOAT, np.float32(0.25), {}),
        (DataType.DOUBLE, np.float64(0.125), {}),
        (DataType.VARCHAR, np.str_("null"), {"max_length": 8}),
    ],
)
def test_nested_array_leaf_types(tmp_path, file_type, leaf, value, kwargs):
    schema = make_schema(
        [
            (
                "tags",
                DataType.ARRAY,
                {
                    **kwargs,
                    "element_type": leaf,
                    "max_capacity": 2,
                    "nullable": True,
                    "element_nullable": True,
                },
            )
        ]
    )
    rows = [{"id": 1, "clips": [{"tags": [value, None]}, {"tags": None}, {"tags": []}, {}]}]
    path = write_rows(tmp_path, schema, file_type, rows)
    assert read_rows(path, file_type) == [
        {
            "id": 1,
            "clips": [{"tags": [value.item(), None]}, {"tags": None}, {"tags": []}, {"tags": None}],
        }
    ]


@pytest.mark.parametrize("inner_nullable", [False, True])
@pytest.mark.parametrize("leaf_nullable", [False, True])
def test_nested_array_nullable_layers(tmp_path, inner_nullable, leaf_nullable):
    schema = make_schema(
        [
            (
                "tags",
                DataType.ARRAY,
                {
                    "element_type": DataType.INT64,
                    "max_capacity": 2,
                    "nullable": inner_nullable,
                    "element_nullable": leaf_nullable,
                },
            )
        ]
    )
    with LocalBulkWriter(schema, str(tmp_path)) as writer:
        writer.append_row({"id": 1, "clips": [{"tags": []}, {"tags": np.array([10, 20])}]})
        for row_id, value, accepted in [(2, None, inner_nullable), (3, [None], leaf_nullable)]:
            row = {"id": row_id, "clips": [{"tags": value}]}
            if accepted:
                writer.append_row(row)
            else:
                count = writer.total_row_count
                with pytest.raises(MilvusException, match=r"clips\[0\]\.tags"):
                    writer.append_row(row)
                assert writer.total_row_count == count
                assert writer._buffer.row_count == count


@pytest.mark.parametrize("file_type", FORMATS)
def test_all_children_missing_preserves_objects(tmp_path, file_type):
    schema = make_schema(
        [
            ("v", DataType.FLOAT_VECTOR, {"dim": 2, "nullable": True}),
            ("n", DataType.INT64, {"nullable": True}),
            (
                "tags",
                DataType.ARRAY,
                {"element_type": DataType.INT64, "max_capacity": 2, "nullable": True},
            ),
        ]
    )
    path = write_rows(tmp_path, schema, file_type, [{"id": 1, "clips": [{}, {}]}])
    assert read_rows(path, file_type) == [
        {"id": 1, "clips": [{"v": None, "n": None, "tags": None}] * 2}
    ]
    if file_type == BulkFileType.PARQUET:
        child = pq.read_schema(path).field("clips").type.value_type
        assert child.names == ["v", "n", "tags"]
        tags_type = child.field("tags").type
        assert tags_type.value_type == pa.int64()
        assert not tags_type.value_field.nullable


@pytest.mark.parametrize("row_nullable", [False, True])
@pytest.mark.parametrize("child_nullable", [False, True])
@pytest.mark.parametrize("leaf_nullable", [False, True])
def test_arrow_schema_nullable_layers(row_nullable, child_nullable, leaf_nullable):
    schema = make_schema(
        [
            ("n", DataType.INT64, {"nullable": child_nullable}),
            ("v", DataType.FLOAT_VECTOR, {"dim": 2, "nullable": child_nullable}),
            (
                "tags",
                DataType.ARRAY,
                {
                    "element_type": DataType.INT64,
                    "max_capacity": 2,
                    "nullable": child_nullable,
                    "element_nullable": leaf_nullable,
                },
            ),
        ],
        nullable=row_nullable,
    )
    field = Buffer(schema, BulkFileType.PARQUET)._deduce_arrow_schema().field("clips")
    assert field.nullable == row_nullable
    assert not field.type.value_field.nullable
    struct = field.type.value_type
    assert all(struct.field(name).nullable == child_nullable for name in ["n", "v", "tags"])
    assert not struct.field("v").type.value_field.nullable
    assert struct.field("tags").type.value_field.nullable == leaf_nullable


@pytest.mark.parametrize("clips", [None, [], [{"tags": []}], [{"tags": None}]])
def test_parquet_batches_without_leaf_values(tmp_path, clips):
    schema = make_schema(
        [
            (
                "tags",
                DataType.ARRAY,
                {
                    "element_type": DataType.INT64,
                    "max_capacity": 2,
                    "nullable": True,
                    "element_nullable": True,
                },
            )
        ]
    )
    path = write_rows(tmp_path, schema, BulkFileType.PARQUET, [{"id": 1, "clips": clips}])
    assert pq.read_table(path).to_pylist() == [{"id": 1, "clips": clips}]
    tags = pq.read_schema(path).field("clips").type.value_type.field("tags")
    assert tags.type.value_type == pa.int64()


@pytest.mark.parametrize(
    "dtype,kwargs,value,error",
    [
        (DataType.INT64, {}, None, "not nullable"),
        (DataType.INT8, {}, 128, "overflow"),
        (DataType.VARCHAR, {"max_length": 2}, "long", "length exceeds"),
        (DataType.FLOAT_VECTOR, {"dim": 2, "nullable": True}, [None, 1.0], "element must"),
        (DataType.BINARY_VECTOR, {"dim": 8, "nullable": True}, [None] * 8, "components"),
        (DataType.FLOAT16_VECTOR, {"dim": 2, "nullable": True}, [None, 1.0], "element must"),
        (DataType.BFLOAT16_VECTOR, {"dim": 2, "nullable": True}, [None, 1.0], "element must"),
        (DataType.INT8_VECTOR, {"dim": 2, "nullable": True}, [None, 1], "element must"),
        (DataType.FLOAT_VECTOR, {"dim": 2, "nullable": True}, [1.0], "dimension"),
        (DataType.ARRAY, {"element_type": DataType.INT64, "max_capacity": 2}, None, "not nullable"),
        (
            DataType.ARRAY,
            {"element_type": DataType.INT64, "max_capacity": 2},
            [None],
            "not nullable",
        ),
        (
            DataType.ARRAY,
            {"element_type": DataType.INT64, "max_capacity": 2},
            [[1]],
            "type mismatch",
        ),
        (
            DataType.ARRAY,
            {"element_type": DataType.INT64, "max_capacity": 2, "element_nullable": True},
            [None, None, None],
            "capacity",
        ),
        (
            DataType.ARRAY,
            {"element_type": DataType.VARCHAR, "max_capacity": 2, "max_length": 2},
            ["long"],
            "length exceeds",
        ),
    ],
)
def test_invalid_subfields_do_not_append(tmp_path, dtype, kwargs, value, error):
    schema = make_schema([("value", dtype, kwargs)])
    with LocalBulkWriter(schema, str(tmp_path)) as writer:
        with pytest.raises(MilvusException, match=error) as caught:
            writer.append_row({"id": 1, "clips": [{"value": value}]})
        assert "clips[0].value" in caught.value.message
        if dtype == DataType.ARRAY and isinstance(value, list) and value == [None]:
            assert "value[0]" in caught.value.message
        assert writer.total_row_count == writer.buffer_row_count == writer._buffer.row_count == 0


def test_required_key_and_outer_capacity(tmp_path):
    schema = make_schema([("value", DataType.INT64, {})], capacity=2)
    with LocalBulkWriter(schema, str(tmp_path)) as writer:
        for value, error in [
            ([{}], "missed"),
            ([None], "list of dict"),
            ([{"value": 1}] * 3, "capacity"),
        ]:
            with pytest.raises(MilvusException, match=error):
                writer.append_row({"id": 1, "clips": value})
            assert writer.total_row_count == writer._buffer.row_count == 0
        writer.append_row({"id": 1, "clips": [{"value": 1}] * 2})


@pytest.mark.parametrize("nullable", [False, True])
def test_numpy_struct_rejected_before_files(tmp_path, nullable, recwarn, monkeypatch):
    # Collect leftovers from earlier tests before checking constructor warnings.
    gc.collect()
    recwarn.clear()
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    schema = make_schema([("value", DataType.INT64, {})], nullable=nullable)
    with pytest.raises(MilvusException, match="NumPy bulk writer does not support struct"):
        Buffer(schema, BulkFileType.NUMPY)
    with pytest.raises(MilvusException, match="NumPy bulk writer does not support struct"):
        LocalBulkWriter(schema, str(tmp_path / "new"), file_type=BulkFileType.NUMPY)
    gc.collect()
    assert not unraisable
    assert not list(tmp_path.iterdir())
    assert not recwarn.list, "\n".join(
        f"{warning.category.__module__}.{warning.category.__qualname__}: {warning.message}\n"
        f"{warning.filename}:{warning.lineno}"
        for warning in recwarn
    )


def test_csv_buffer_serializes_nested_numpy_without_mutation(tmp_path):
    schema = make_schema(
        [("tags", DataType.ARRAY, {"element_type": DataType.INT64, "max_capacity": 3})]
    )
    value = [np.int64(1), None, np.int64(3)]
    buffer = Buffer(schema, BulkFileType.CSV)
    buffer.append_row({"id": 1, "clips": [{"tags": value}]})
    path = Path(buffer.persist(str(tmp_path / "nested"))[0])
    assert read_rows(path, BulkFileType.CSV) == [{"id": 1, "clips": [{"tags": [1, None, 3]}]}]
    assert isinstance(value[0], np.int64)


def test_top_level_array_still_rejects_none(tmp_path):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("values", DataType.ARRAY, element_type=DataType.INT64, max_capacity=2),
        ]
    )
    with LocalBulkWriter(schema, str(tmp_path)) as writer:
        with pytest.raises(MilvusException, match="not nullable"):
            writer.append_row({"id": 1, "values": [None]})
