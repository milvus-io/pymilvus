import numpy as np
import pytest
from pymilvus.client.entity_helper import _convert_to_vector_bytes, pack_field_value_to_field_data
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2
from pymilvus.orm.prepare import Prepare
from pymilvus.orm.schema import CollectionSchema, FieldSchema


def _storage_bytes(values, dtype):
    return np.array(values, dtype=dtype).tobytes()


def test_float16_slice_and_reverse_keep_element_order():
    base = np.arange(8, dtype=np.float16)
    sliced = base[::2]
    assert not sliced.flags.c_contiguous
    assert _convert_to_vector_bytes(sliced, DataType.FLOAT16_VECTOR) == _storage_bytes(
        [0, 2, 4, 6], np.float16
    )

    reversed_prefix = base[:4][::-1]
    assert not reversed_prefix.flags.c_contiguous
    assert _convert_to_vector_bytes(reversed_prefix, DataType.FLOAT16_VECTOR) == _storage_bytes(
        [3, 2, 1, 0], np.float16
    )

    contiguous = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    assert (
        _convert_to_vector_bytes(contiguous, DataType.FLOAT16_VECTOR)
        == contiguous.view(np.uint8).tobytes()
    )


def test_float16_slice_packs_into_row_field_data():
    sliced = np.arange(8, dtype=np.float16)[::2]
    field_data = schema_pb2.FieldData(type=DataType.FLOAT16_VECTOR, field_name="vec")
    cache = {}
    pack_field_value_to_field_data(
        sliced,
        field_data,
        {"name": "vec", "params": {"dim": 4}},
        cache,
    )
    assert cache[id(field_data)] == [_storage_bytes([0, 2, 4, 6], np.float16)]


def test_prepare_data_accepts_strided_float16():
    schema = CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT16_VECTOR, dim=4),
        ]
    )
    sliced = np.arange(8, dtype=np.float16)[::2]
    result = Prepare.prepare_data([[1], [sliced]], schema)
    vector = next(entity for entity in result if entity["name"] == "vec")
    assert vector["values"] == [_storage_bytes([0, 2, 4, 6], np.float16)]


def test_bfloat16_slice_matches_contiguous_bytes():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    base = np.arange(8, dtype=ml_dtypes.bfloat16)
    sliced = base[::2]
    assert not sliced.flags.c_contiguous
    expected = _storage_bytes([0, 2, 4, 6], ml_dtypes.bfloat16)
    assert _convert_to_vector_bytes(sliced, DataType.BFLOAT16_VECTOR) == expected

    field_data = schema_pb2.FieldData(type=DataType.BFLOAT16_VECTOR, field_name="vec")
    cache = {}
    pack_field_value_to_field_data(
        sliced,
        field_data,
        {"name": "vec", "params": {"dim": 4}},
        cache,
    )
    assert cache[id(field_data)] == [expected]

    schema = CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.BFLOAT16_VECTOR, dim=4),
        ]
    )
    result = Prepare.prepare_data([[1], [sliced]], schema)
    vector = next(entity for entity in result if entity["name"] == "vec")
    assert vector["values"] == [expected]


def test_wrong_half_precision_dtype_still_rejected():
    with pytest.raises(ParamError, match="dtype=float16"):
        _convert_to_vector_bytes(np.arange(4, dtype=np.float32), DataType.FLOAT16_VECTOR)
