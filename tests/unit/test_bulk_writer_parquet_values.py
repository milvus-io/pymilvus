import ml_dtypes
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, StructFieldSchema
from pymilvus.bulk_writer import BulkFileType, LocalBulkWriter
from pymilvus.exceptions import MilvusException


def write_parquet(tmp_path, schema, rows):
    with LocalBulkWriter(schema, str(tmp_path), file_type=BulkFileType.PARQUET) as writer:
        for row in rows:
            writer.append_row(row)
        path = writer._buffer.persist(str(tmp_path / "values"))[0]
    return pq.read_table(path)


@pytest.mark.parametrize("dtype", [DataType.FLOAT, DataType.DOUBLE])
@pytest.mark.parametrize("nullable", [False, True])
def test_parquet_float_preserves_finite_values_and_null(tmp_path, dtype, nullable):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", dtype, nullable=nullable),
        ]
    )
    values = [-0.5, 0.25] + ([None] if nullable else [])
    rows = [{"id": i, "value": value} for i, value in enumerate(values, start=1)]
    table = write_parquet(tmp_path, schema, rows)
    actual = table.column("value").to_pylist()
    assert actual == values
    assert table.column("value").is_null().to_pylist() == [False, False] + (
        [True] if nullable else []
    )


@pytest.mark.parametrize("large", [2**53 + 1, -(2**53 + 1), 2**63 - 1])
def test_parquet_nullable_integer_preserves_precision(tmp_path, large):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", DataType.INT64, nullable=True),
        ]
    )
    values = [large, None, 0]
    rows = [{"id": i, "value": value} for i, value in enumerate(values, start=1)]
    column = write_parquet(tmp_path, schema, rows).column("value")
    assert column.type == pa.int64()
    assert column.to_pylist() == values
    assert column.is_null().to_pylist() == [False, True, False]


@pytest.mark.parametrize("dtype", [DataType.FLOAT, DataType.DOUBLE])
@pytest.mark.parametrize("nullable_struct", [False, True])
def test_parquet_struct_preserves_finite_values_and_integers(tmp_path, dtype, nullable_struct):
    struct = StructFieldSchema()
    struct.add_field("value", dtype)
    struct.add_field("integer", DataType.INT64)
    schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=struct,
        nullable=nullable_struct,
        max_capacity=4,
    )
    large = 2**53 + 1
    empty = None if nullable_struct else []
    rows = [
        {
            "id": 1,
            "clips": [
                {"value": -0.5, "integer": large},
                {"value": 0.25, "integer": -large},
            ],
        },
        {"id": 2, "clips": empty},
    ]
    table = write_parquet(tmp_path, schema, rows)
    actual = table.to_pylist()
    elements = actual[0]["clips"]
    assert elements[0]["value"] == -0.5
    assert elements[1]["value"] == 0.25
    assert [element["integer"] for element in elements] == [large, -large]
    assert actual[1] == {"id": 2, "clips": empty}
    values = table.column("clips").chunk(0).values
    assert values.field("value").null_count == 0
    assert values.field("integer").null_count == 0


@pytest.mark.parametrize("dtype", [DataType.FLOAT, DataType.DOUBLE])
def test_parquet_array_preserves_finite_values_and_null_rows(tmp_path, dtype):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", DataType.ARRAY, element_type=dtype, max_capacity=4, nullable=True),
        ]
    )
    table = write_parquet(
        tmp_path,
        schema,
        [
            {"id": 1, "value": [-0.5, 0.25]},
            {"id": 2, "value": None},
            {"id": 3, "value": []},
        ],
    )
    column = table.column("value")
    actual = column.to_pylist()
    assert actual == [[-0.5, 0.25], None, []]
    assert column.chunk(0).values.null_count == 0


@pytest.mark.parametrize(
    "dtype", [DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR]
)
def test_parquet_vector_lists_preserve_finite_values(tmp_path, dtype):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", dtype, dim=2),
        ]
    )
    column = write_parquet(tmp_path, schema, [{"id": 1, "value": [-0.5, 0.25]}]).column("value")
    actual = column.to_pylist()
    assert actual == [[-0.5, 0.25]]
    assert column.chunk(0).values.null_count == 0


def test_parquet_struct_float_vector_preserves_finite_values(tmp_path):
    struct = StructFieldSchema()
    struct.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
    schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=struct,
        max_capacity=4,
    )
    table = write_parquet(tmp_path, schema, [{"id": 1, "clips": [{"vector": [-0.5, 0.25]}]}])
    actual = table.to_pylist()[0]["clips"][0]["vector"]
    assert actual == [-0.5, 0.25]
    assert table.column("clips").chunk(0).values.field("vector").values.null_count == 0


NON_FINITE = [float("nan"), float("inf"), -float("inf")]
FLOAT_VECTORS = [DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR]


def assert_rejected_without_appending(tmp_path, schema, file_type, valid, invalid, path, bad):
    with LocalBulkWriter(schema, str(tmp_path), file_type=file_type) as writer:
        writer.append_row({"id": 1, **valid})
        counts = (writer.total_row_count, writer.buffer_row_count, writer.buffer_size)
        with pytest.raises(MilvusException) as exc:
            writer.append_row({"id": 2, **invalid})
        assert path in exc.value.message
        assert str(bad) in exc.value.message
        assert "Non-finite" in exc.value.message
        assert (writer.total_row_count, writer.buffer_row_count, writer.buffer_size) == counts
        assert writer._buffer.row_count == 1
        assert writer._buffer._buffer["id"] == [1]
        assert all(len(column) == 1 for column in writer._buffer._buffer.values())


@pytest.mark.parametrize("file_type", list(BulkFileType))
@pytest.mark.parametrize("dtype", [DataType.FLOAT, DataType.DOUBLE])
@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("numpy_scalar", [False, True])
@pytest.mark.parametrize("bad", NON_FINITE)
def test_append_rejects_non_finite_scalar(tmp_path, file_type, dtype, nullable, numpy_scalar, bad):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", dtype, nullable=nullable),
        ]
    )
    value = np.float32(bad) if numpy_scalar else bad
    assert_rejected_without_appending(
        tmp_path, schema, file_type, {"value": 0.25}, {"value": value}, "value", bad
    )


@pytest.mark.parametrize("file_type", list(BulkFileType))
@pytest.mark.parametrize("dtype", [DataType.FLOAT, DataType.DOUBLE])
@pytest.mark.parametrize("numpy_array", [False, True])
@pytest.mark.parametrize("bad", NON_FINITE)
def test_append_rejects_non_finite_array(tmp_path, file_type, dtype, numpy_array, bad):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", DataType.ARRAY, element_type=dtype, max_capacity=4, nullable=True),
        ]
    )
    value = [0.25, bad]
    if numpy_array:
        value = np.array(value, dtype=np.float32 if dtype == DataType.FLOAT else np.float64)
    assert_rejected_without_appending(
        tmp_path, schema, file_type, {"value": [0.25, -0.5]}, {"value": value}, "value", bad
    )


def vector_input(values, dtype, representation):
    if representation == "list":
        return values
    if representation == "native":
        numpy_dtype = {
            DataType.FLOAT_VECTOR: np.float32,
            DataType.FLOAT16_VECTOR: np.float16,
            DataType.BFLOAT16_VECTOR: ml_dtypes.bfloat16,
        }[dtype]
    else:
        numpy_dtype = np.dtype(representation)
    return np.array(values, dtype=numpy_dtype)


@pytest.mark.parametrize("file_type", list(BulkFileType))
@pytest.mark.parametrize("dtype", FLOAT_VECTORS)
@pytest.mark.parametrize("representation", ["list", "native", "float32", "float64"])
@pytest.mark.parametrize("bad", NON_FINITE)
def test_append_rejects_non_finite_vector(tmp_path, file_type, dtype, representation, bad):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", dtype, dim=2),
        ]
    )
    assert_rejected_without_appending(
        tmp_path,
        schema,
        file_type,
        {"value": vector_input([0.25, -0.5], dtype, representation)},
        {"value": vector_input([0.25, bad], dtype, representation)},
        "value[1]",
        bad,
    )


@pytest.mark.parametrize("file_type", list(BulkFileType))
@pytest.mark.parametrize("dtype", [DataType.FLOAT, DataType.DOUBLE, *FLOAT_VECTORS])
@pytest.mark.parametrize("representation", ["list", "native"])
@pytest.mark.parametrize("bad", NON_FINITE)
def test_append_rejects_non_finite_struct_subfield(tmp_path, file_type, dtype, representation, bad):
    struct = StructFieldSchema()
    struct.add_field("value", dtype, **({"dim": 2} if dtype in FLOAT_VECTORS else {}))
    schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=struct,
        max_capacity=4,
        nullable=True,
    )
    if dtype in FLOAT_VECTORS:
        valid = vector_input([0.25, -0.5], dtype, representation)
        invalid = vector_input([0.25, bad], dtype, representation)
    else:
        valid = 0.25
        invalid = np.float32(bad) if representation == "native" else bad
    assert_rejected_without_appending(
        tmp_path,
        schema,
        file_type,
        {"clips": [{"value": valid}]},
        {"clips": [{"value": valid}, {"value": invalid}]},
        "clips[1].value",
        bad,
    )


@pytest.mark.parametrize("dtype", [DataType.FLOAT, DataType.DOUBLE])
@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("bad", NON_FINITE)
@pytest.mark.parametrize("row", [{"id": 1}, {"id": 1, "value": None}])
def test_append_rejects_non_finite_default(tmp_path, dtype, nullable, bad, row):
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("value", dtype, nullable=nullable, default_value=bad),
        ]
    )
    with LocalBulkWriter(schema, str(tmp_path), file_type=BulkFileType.PARQUET) as writer:
        with pytest.raises(MilvusException) as exc:
            writer.append_row(dict(row))
        assert "value" in exc.value.message
        assert str(bad) in exc.value.message
        assert writer.total_row_count == 0
        assert writer.buffer_row_count == 0
        assert writer.buffer_size == 0
        assert writer._buffer.row_count == 0
