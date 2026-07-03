"""Low-level FieldData result decoders.

This module owns policy-free reads from protobuf ``FieldData`` cells. Result
consumers decide materialization policy and call these helpers for the shared
type facts: protobuf slot, row width, and validity-mask handling.
"""

from typing import Any, List, Optional

import orjson

from pymilvus.client import type_info
from pymilvus.client.types import DataType
from pymilvus.client.utils import sparse_parse_single_row
from pymilvus.exceptions import MilvusException, ParamError


def get_field_data(field_data: Any) -> Any:
    """Return the protobuf payload recorded by TypeInfo for ``field_data``."""
    field_type = field_data.type
    try:
        scalar_attr = type_info.get_scalar_attr(field_type)
        if scalar_attr is not None:
            return getattr(field_data.scalars, scalar_attr).data
        vector_attr = type_info.get_vector_attr(field_type)
        if vector_attr is not None:
            obj = getattr(field_data.vectors, vector_attr)
            return obj.data if field_type == DataType.FLOAT_VECTOR else obj
        field_attr = type_info.get_field_attr(field_type)
        if field_attr is not None:
            return getattr(field_data, field_attr)
    except ParamError:
        pass
    msg = f"Unsupported field type: {field_type}"
    raise MilvusException(msg)


def is_null_cell(field_data: Any, logical_index: int) -> bool:
    return len(field_data.valid_data) > 0 and field_data.valid_data[logical_index] is False


def physical_index(field_data: Any, logical_index: int) -> int:
    if len(field_data.valid_data) == 0:
        return logical_index
    return sum(1 for valid in field_data.valid_data[:logical_index] if valid)


def dense_vector_width(dtype: DataType, dim: int) -> Optional[int]:
    if not type_info.is_dense_vector_type(dtype):
        return None
    if not type_info.is_byte_vector_type(dtype):
        return dim
    return type_info.row_width(dtype, dim)


def array_cell_length(array_cell: Any) -> int:
    """Return the populated length of one ScalarField array cell."""
    attr = _populated_array_attr(array_cell)
    if attr is None:
        return 0
    return len(getattr(array_cell, attr).data)


def decode_array_value(array_cell: Any, index: int) -> Any:
    """Decode one value from a ScalarField array cell."""
    attr = _populated_array_attr(array_cell)
    if attr is None:
        return None
    data = getattr(array_cell, attr).data
    if len(data) <= index:
        return None
    return data[index]


def vector_array_length(vector_data: Any, element_type: DataType) -> int:
    """Return the number of vector values stored in one VectorArray row."""
    element_type = _resolve_vector_array_element_type(vector_data, element_type)
    if element_type == DataType.NONE:
        return 0
    if not type_info.is_dense_vector_type(element_type):
        raise ParamError(message=f"Unsupported vector array element type: {element_type}")

    width = dense_vector_width(element_type, vector_data.dim)
    if not width:
        return 0
    payload = _vector_array_payload(vector_data, element_type)
    return len(payload) // width


def decode_vector_array_value(vector_data: Any, element_type: DataType, index: int) -> Any:
    """Decode one vector value from a VectorArray row using TypeInfo width facts."""
    element_type = _resolve_vector_array_element_type(vector_data, element_type)
    if element_type == DataType.NONE:
        return None
    if not type_info.is_dense_vector_type(element_type):
        raise ParamError(message=f"Unsupported vector array element type: {element_type}")

    width = dense_vector_width(element_type, vector_data.dim)
    if not width:
        return None
    payload = _vector_array_payload(vector_data, element_type)
    start = index * width
    end = start + width
    if len(payload) < end:
        return None
    value = payload[start:end]
    if element_type == DataType.FLOAT_VECTOR:
        return list(value)
    return value


def decode_array_cell(field_data: Any, logical_index: int) -> Any:
    array_data = field_data.scalars.array_data
    if logical_index >= len(array_data.data):
        return None
    attr = type_info.get_array_element_attr(array_data.element_type)
    if attr is None:
        raise MilvusException(message=f"Unsupported data type: {array_data.element_type}")
    return list(getattr(array_data.data[logical_index], attr).data)


def decode_vector_array_cell(
    field_data: Any,
    logical_index: int,
    *,
    split_vectors: bool = True,
) -> Any:
    vector_array = field_data.vectors.vector_array
    if logical_index >= len(vector_array.data):
        return []

    vector_data = vector_array.data[logical_index]
    element_type = vector_array.element_type
    if element_type == DataType.NONE:
        element_type = _resolve_vector_array_element_type(vector_data, element_type)
    if element_type == DataType.NONE:
        return []

    if not split_vectors:
        return _vector_array_payload(vector_data, element_type)

    length = vector_array_length(vector_data, element_type)
    return [
        decode_vector_array_value(vector_data, element_type, vector_index)
        for vector_index in range(length)
    ]


def decode_cell(
    field_data: Any,
    logical_index: int,
    *,
    physical_index_override: Optional[int] = None,
    dense_vector_data: Optional[Any] = None,
    wrap_byte_vectors: bool = False,
    split_array_vectors: bool = True,
) -> Any:
    """Decode one FieldData cell into the current public Python value shape."""
    if is_null_cell(field_data, logical_index):
        return None

    field_type = field_data.type
    physical = (
        physical_index(field_data, logical_index)
        if physical_index_override is None
        else physical_index_override
    )

    if field_type == DataType.JSON:
        data = field_data.scalars.json_data.data
        return None if logical_index >= len(data) else orjson.loads(data[logical_index])

    if field_type == DataType.ARRAY:
        return decode_array_cell(field_data, logical_index)

    if type_info.is_scalar_type(field_type):
        data = get_field_data(field_data)
        return None if logical_index >= len(data) else data[logical_index]

    if type_info.is_dense_vector_type(field_type):
        dim = field_data.vectors.dim
        width = dense_vector_width(field_type, dim)
        data = dense_vector_data if dense_vector_data is not None else get_field_data(field_data)
        start = physical * width
        end = start + width
        if len(data) < end:
            return None
        value = data[start:end]
        if wrap_byte_vectors and type_info.is_byte_vector_type(field_type):
            return [value]
        return value

    if type_info.is_sparse_vector_type(field_type):
        contents = field_data.vectors.sparse_float_vector.contents
        return None if physical >= len(contents) else sparse_parse_single_row(contents[physical])

    if field_type == DataType._ARRAY_OF_VECTOR:
        return decode_vector_array_cell(
            field_data,
            logical_index,
            split_vectors=split_array_vectors,
        )

    msg = f"Unsupported field type: {field_type}"
    raise MilvusException(msg)


def decode_range(field_data: Any, start: int, end: int) -> List[Any]:
    return [decode_cell(field_data, index) for index in range(start, end)]


def _resolve_vector_array_element_type(vector_data: Any, element_type: DataType) -> DataType:
    if element_type != DataType.NONE:
        return element_type
    if len(vector_data.float_vector.data) > 0:
        return DataType.FLOAT_VECTOR
    if len(vector_data.float16_vector) > 0:
        return DataType.FLOAT16_VECTOR
    if len(vector_data.bfloat16_vector) > 0:
        return DataType.BFLOAT16_VECTOR
    if len(vector_data.binary_vector) > 0:
        return DataType.BINARY_VECTOR
    if len(vector_data.int8_vector) > 0:
        return DataType.INT8_VECTOR
    return DataType.NONE


def _populated_array_attr(array_cell: Any) -> Optional[str]:
    for info in type_info.TYPE_INFO.values():
        attr = info.array_element_attr
        if attr is None:
            continue
        data = getattr(array_cell, attr, None)
        if data is not None and len(data.data) > 0:
            return attr
    return None


def _vector_array_payload(vector_data: Any, element_type: DataType) -> Any:
    if element_type == DataType.FLOAT_VECTOR:
        return vector_data.float_vector.data
    attr = type_info.get_vector_attr(element_type)
    if attr is not None:
        return getattr(vector_data, attr)
    return []
