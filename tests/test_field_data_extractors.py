import struct

import orjson
import pytest
from pymilvus.client.field_data_extractors import (
    decode_cell,
    decode_range,
    dense_vector_width,
    get_field_data,
    physical_index,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import schema_pb2


def _scalar_field(dtype, name, data, valid_data=None):
    scalar_map = {
        DataType.BOOL: lambda d: schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=d)),
        DataType.INT32: lambda d: schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=d)),
        DataType.INT64: lambda d: schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=d)),
        DataType.FLOAT: lambda d: schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=d)),
        DataType.DOUBLE: lambda d: schema_pb2.ScalarField(
            double_data=schema_pb2.DoubleArray(data=d)
        ),
        DataType.VARCHAR: lambda d: schema_pb2.ScalarField(
            string_data=schema_pb2.StringArray(data=d)
        ),
        DataType.JSON: lambda d: schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=d)),
    }
    return schema_pb2.FieldData(
        type=dtype,
        field_name=name,
        scalars=scalar_map[dtype](data),
        valid_data=valid_data or [],
    )


def _vector_field(dtype, name, dim, data, valid_data=None):
    vector_args = {"dim": dim}
    if dtype == DataType.FLOAT_VECTOR:
        vector_args["float_vector"] = schema_pb2.FloatArray(data=data)
    elif dtype == DataType.BINARY_VECTOR:
        vector_args["binary_vector"] = data
    elif dtype == DataType.FLOAT16_VECTOR:
        vector_args["float16_vector"] = data
    elif dtype == DataType.BFLOAT16_VECTOR:
        vector_args["bfloat16_vector"] = data
    elif dtype == DataType.INT8_VECTOR:
        vector_args["int8_vector"] = data
    return schema_pb2.FieldData(
        type=dtype,
        field_name=name,
        vectors=schema_pb2.VectorField(**vector_args),
        valid_data=valid_data or [],
    )


def test_decode_scalar_json_and_array_cells():
    scalar = _scalar_field(DataType.VARCHAR, "text", ["a", "b"], valid_data=[True, False])
    json_field = _scalar_field(DataType.JSON, "json", [orjson.dumps({"x": 1})])
    array_field = schema_pb2.FieldData(type=DataType.ARRAY, field_name="array")
    array_field.scalars.array_data.element_type = DataType.INT32
    array_field.scalars.array_data.data.append(
        schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[1, 2]))
    )

    assert get_field_data(scalar) == ["a", "b"]
    assert decode_range(scalar, 0, 2) == ["a", None]
    assert decode_cell(json_field, 0) == {"x": 1}
    assert decode_cell(array_field, 0) == [1, 2]


@pytest.mark.parametrize(
    ("dtype", "dim", "payload", "expected_width", "expected"),
    [
        (DataType.FLOAT_VECTOR, 2, [1.0, 2.0, 3.0, 4.0], 2, [3.0, 4.0]),
        (DataType.BINARY_VECTOR, 16, b"abcd", 2, b"cd"),
        (DataType.FLOAT16_VECTOR, 2, b"abcdefgh", 4, b"efgh"),
        (DataType.BFLOAT16_VECTOR, 2, b"ijklmnop", 4, b"mnop"),
        (DataType.INT8_VECTOR, 2, b"qrst", 2, b"st"),
    ],
)
def test_decode_dense_vector_cells_from_type_info_widths(
    dtype, dim, payload, expected_width, expected
):
    field = _vector_field(dtype, "vector", dim, payload)

    assert dense_vector_width(dtype, dim) == expected_width
    assert decode_cell(field, 1) == expected


def test_decode_dense_vector_cell_wraps_byte_vectors_when_requested():
    field = _vector_field(DataType.INT8_VECTOR, "vector", 2, b"abcd")

    assert decode_cell(field, 1, wrap_byte_vectors=True) == [b"cd"]


def test_decode_nullable_vector_cell_uses_physical_index():
    field = _vector_field(
        DataType.FLOAT_VECTOR,
        "vector",
        2,
        [1.0, 2.0, 3.0, 4.0],
        valid_data=[True, False, True],
    )

    assert physical_index(field, 2) == 1
    assert decode_cell(field, 1) is None
    assert decode_cell(field, 2) == [3.0, 4.0]


def test_decode_sparse_vector_cell():
    sparse = schema_pb2.SparseFloatArray(
        contents=[struct.pack("If", 7, 0.5), struct.pack("If", 9, 1.5)]
    )
    field = schema_pb2.FieldData(
        type=DataType.SPARSE_FLOAT_VECTOR,
        field_name="sparse",
        vectors=schema_pb2.VectorField(sparse_float_vector=sparse),
    )

    assert decode_cell(field, 1) == {9: pytest.approx(1.5)}


def test_decode_array_of_vector_cell_splits_rows():
    field = schema_pb2.FieldData(type=DataType._ARRAY_OF_VECTOR, field_name="vectors")
    field.vectors.vector_array.element_type = DataType.FLOAT_VECTOR
    vector = schema_pb2.VectorField(dim=2)
    vector.float_vector.data.extend([1.0, 2.0, 3.0, 4.0])
    field.vectors.vector_array.data.append(vector)

    assert decode_cell(field, 0) == [[1.0, 2.0], [3.0, 4.0]]


def test_decode_unsupported_field_type_raises():
    with pytest.raises(MilvusException, match="Unsupported field type"):
        decode_cell(schema_pb2.FieldData(type=DataType.NONE), 0)
