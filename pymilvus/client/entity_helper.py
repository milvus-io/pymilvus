import itertools
import json
import math
import struct
from collections.abc import Sized
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import orjson

from pymilvus.exceptions import (
    DataNotMatchException,
    ExceptionsMessage,
    MilvusException,
    ParamError,
)
from pymilvus.grpc_gen import schema_pb2 as schema_types
from pymilvus.settings import Config

from . import field_data_extractors, type_info
from .types import DataType
from .utils import (
    SciPyHelper,
    SparseMatrixInputType,
    SparseRowOutputType,
    sparse_parse_single_row,
)

CHECK_STR_ARRAY = True

_LEGACY_ROW_ERROR_LABELS = {
    DataType.INT8: "int",
    DataType.INT16: "int",
    DataType.INT32: "int",
    DataType.TIMESTAMPTZ: "string",
}


def _error_label(dtype: DataType) -> str:
    try:
        dtype = DataType(dtype)
    except (TypeError, ValueError):
        return str(dtype)
    return _LEGACY_ROW_ERROR_LABELS.get(dtype, dtype.name.lower())


def _scalar_row_error_message(
    dtype: DataType, field_name: str, value: Any, error: Exception
) -> str:
    message = ExceptionsMessage.FieldDataInconsistent % (
        field_name,
        _error_label(dtype),
        type(value),
    )
    # Preserve the legacy row-pack error text for these two types. Improving
    # the diagnostic detail can be handled separately from this TypeInfo refactor.
    if dtype in (DataType.TIMESTAMPTZ, DataType.GEOMETRY):
        return message
    return message + f" Detail: {error!s}"


def _get_dim(field_info: Dict) -> int:
    """Extract vector dimension from field info, handling both int and str values."""
    dim = field_info.get("params", {}).get("dim", 0)
    return int(dim) if isinstance(dim, str) else dim


def _is_type_in_str(v: Any, t: Any):
    if not isinstance(v, str):
        return False
    try:
        t(v)
    except ValueError:
        return False
    return True


def _is_int_type(v: Any):
    return isinstance(v, (int, np.integer)) or _is_type_in_str(v, int)


def _is_float_type(v: Any):
    return isinstance(v, (float, np.floating)) or _is_type_in_str(v, float)


def entity_is_sparse_matrix(entity: Any):
    if SciPyHelper.is_scipy_sparse(entity):
        return True
    try:
        # must be of multiple rows
        if len(entity) == 0:
            return False
        for item in entity:
            if SciPyHelper.is_scipy_sparse(item):
                return item.shape[0] == 1
            if not isinstance(item, dict) and not isinstance(item, list):
                return False
            pairs = item.items() if isinstance(item, dict) else item
            # each row must be a list of Tuple[int, float]. we allow empty sparse row
            for pair in pairs:
                if len(pair) != 2 or not _is_int_type(pair[0]) or not _is_float_type(pair[1]):
                    return False
    except Exception:
        return False
    return True


# converts supported sparse matrix to schemapb.SparseFloatArray proto
def sparse_rows_to_proto(data: SparseMatrixInputType) -> schema_types.SparseFloatArray:
    # converts a sparse float vector to plain bytes. the format is the same as how
    # milvus interprets/persists the data.
    def sparse_float_row_to_bytes(indices: Iterable[int], values: Iterable[float]):
        if len(indices) != len(values):
            raise ParamError(
                message=f"length of indices and values must be the same, got {len(indices)} and {len(values)}"
            )
        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
        for i, v in sorted_pairs:
            if not (0 <= i < 2**32 - 1):
                raise ParamError(
                    message=f"sparse vector index must be positive and less than 2^32-1: {i}"
                )
            if math.isnan(v):
                raise ParamError(message="sparse vector value must not be NaN")
        n = len(sorted_pairs)
        # Pack all pairs at once: interleaved (uint32 index, float value) pairs
        fmt = f"<{'If' * n}"
        return struct.pack(fmt, *itertools.chain.from_iterable(sorted_pairs))

    if not entity_is_sparse_matrix(data):
        raise ParamError(message="input must be a sparse matrix in supported format")

    result = schema_types.SparseFloatArray()

    if SciPyHelper.is_scipy_sparse(data):
        csr = data.tocsr()
        result.dim = csr.shape[1]
        for start, end in zip(csr.indptr[:-1], csr.indptr[1:]):
            result.contents.append(
                sparse_float_row_to_bytes(csr.indices[start:end], csr.data[start:end])
            )
    else:
        dim = 0
        for row_data in data:
            if SciPyHelper.is_scipy_sparse(row_data):
                if row_data.shape[0] != 1:
                    raise ParamError(message="invalid input for sparse float vector: expect 1 row")
                dim = max(dim, row_data.shape[1])
                result.contents.append(sparse_float_row_to_bytes(row_data.indices, row_data.data))
            else:
                indices = []
                values = []
                row = row_data.items() if isinstance(row_data, dict) else row_data
                for index, value in row:
                    indices.append(int(index))
                    values.append(float(value))
                result.contents.append(sparse_float_row_to_bytes(indices, values))
                row_dim = 0
                if len(indices) > 0:
                    row_dim = indices[-1] + 1
                dim = max(dim, row_dim)
        result.dim = dim
    return result


# converts schema_types.SparseFloatArray proto to Iterable[SparseRowOutputType]
def sparse_proto_to_rows(
    sfv: schema_types.SparseFloatArray, start: Optional[int] = None, end: Optional[int] = None
) -> Iterable[SparseRowOutputType]:
    if not isinstance(sfv, schema_types.SparseFloatArray):
        raise ParamError(message="Vector must be a sparse float vector")
    if start is None:
        start = 0
    if end is None:
        end = len(sfv.contents)
    return [sparse_parse_single_row(row_bytes) for row_bytes in sfv.contents[start:end]]


def get_input_num_rows(entity: Any) -> int:
    if SciPyHelper.is_scipy_sparse(entity):
        return entity.shape[0]
    return len(entity)


def entity_type_to_dtype(entity_type: Any):
    if isinstance(entity_type, int):
        return entity_type
    if isinstance(entity_type, str):
        # case sensitive
        return schema_types.DataType.Value(entity_type)
    raise ParamError(message=f"invalid entity type: {entity_type}")


def get_max_len_of_var_char(field_info: Dict) -> int:
    k = Config.MaxVarCharLengthKey
    v = Config.MaxVarCharLength
    return field_info.get("params", {}).get(k, v)


def convert_to_str_array(orig_str_arr: Any, field_info: Dict, check: bool = True):
    arr = []
    if Config.EncodeProtocol.lower() != "utf-8":
        for s in orig_str_arr:
            arr.append(s.encode(Config.EncodeProtocol))
    else:
        arr = orig_str_arr
    max_len = int(get_max_len_of_var_char(field_info))
    if check:
        for s in arr:
            if not isinstance(s, str):
                raise ParamError(
                    message=f"field ({field_info['name']}) expects string input, got: {type(s)}"
                )
            if len(s) > max_len:
                raise ParamError(
                    message=f"invalid input of field ({field_info['name']}), "
                    f"length of string exceeds max length. length: {len(s)}, max length: {max_len}"
                )
    return arr


def entity_to_str_arr(entity_values: Any, field_info: Any, check: bool = True):
    return convert_to_str_array(entity_values, field_info, check=check)


def convert_to_json(obj: object):
    def preprocess_numpy_types(obj: Any):
        """
        Convert numpy types to Python native types using iterative approach
        to avoid recursion limit for deeply nested structures.
        """
        # Use a stack to process nested structures iteratively
        # Each entry: (value, parent_container, key_or_index)
        stack = [(obj, None, None)]
        result = None

        def assign_to_parent(value: Any) -> None:
            """Helper function to assign value to parent container or result"""
            nonlocal result
            if parent is None:
                result = value
            elif isinstance(parent, (dict, list)):
                parent[key_or_idx] = value

        while stack:
            current, parent, key_or_idx = stack.pop()

            # Handle numpy types (leaf nodes)
            if isinstance(current, np.ndarray):
                assign_to_parent(current.tolist())
            elif isinstance(current, np.integer):
                assign_to_parent(int(current))
            elif isinstance(current, np.floating):
                assign_to_parent(float(current))
            elif isinstance(current, np.bool_):
                assign_to_parent(bool(current))
            elif isinstance(current, dict):
                # Process dict: create new dict first
                processed = {}
                assign_to_parent(processed)
                # Add items to stack for processing (reverse order to maintain original order)
                for k, v in reversed(tuple(current.items())):
                    stack.append((v, processed, k))
            elif isinstance(current, list):
                # Process list: create new list with placeholders first
                processed = [None] * len(current)
                assign_to_parent(processed)
                # Add items to stack for processing (reverse order to maintain original order)
                for i in reversed(range(len(current))):
                    stack.append((current[i], processed, i))
            else:
                # Primitive type, no processing needed
                assign_to_parent(current)

        return result

    # Handle JSON string input
    if isinstance(obj, str):
        try:
            # Validate JSON string by parsing it
            parsed_obj = orjson.loads(obj)
            # If it's a valid JSON string, validate dict keys if it's a dict
            if isinstance(parsed_obj, dict):
                for k in parsed_obj:
                    if not isinstance(k, str):
                        raise DataNotMatchException(message=ExceptionsMessage.JSONKeyMustBeStr)
            # Return the original string encoded as bytes (since it's already valid JSON)
            return obj.encode(Config.EncodeProtocol)
        except Exception as e:
            # Truncate the string if it's too long for better readability
            max_len = 200
            json_str_display = obj if len(obj) <= max_len else obj[:max_len] + "..."
            raise DataNotMatchException(
                message=f"Invalid JSON string: {e!s}. Input string: {json_str_display!r}"
            ) from e

    # Handle dict input
    if isinstance(obj, dict):
        for k in obj:
            if not isinstance(k, str):
                raise DataNotMatchException(message=ExceptionsMessage.JSONKeyMustBeStr)

    processed_obj = preprocess_numpy_types(obj)

    # Try orjson first (faster), fallback to standard json for deeply nested structures
    try:
        return orjson.dumps(processed_obj)
    except (TypeError, RecursionError) as e:
        # orjson has recursion limits (~500 levels), fallback to standard json library
        # Standard json.dumps can handle up to ~999 levels with default recursion limit (1000)
        if "Recursion limit" in str(e) or isinstance(e, RecursionError):
            return json.dumps(processed_obj).encode(Config.EncodeProtocol)
        raise


def convert_to_json_arr(objs: List[object], field_info: Any):
    field_name = field_info["name"]
    arr = []
    for obj in objs:
        if obj is None:
            raise ParamError(message=f"field ({field_name}) expects a non-None input")
        arr.append(convert_to_json(obj))
    return arr


def entity_to_json_arr(entity_values: Dict, field_info: Any):
    return convert_to_json_arr(entity_values, field_info)


def convert_to_array_arr(objs: List[Any], field_info: Any):
    return [convert_to_array(obj, field_info) for obj in objs]


def convert_to_array(obj: List[Any], field_info: Any):
    # Convert numpy ndarray to list if needed
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    field_data = schema_types.ScalarField()
    element_type = field_info.get("element_type", None)
    attr_name = type_info.get_array_element_attr(element_type)
    if attr_name is not None:
        getattr(field_data, attr_name).data.extend(obj)
        return field_data
    raise ParamError(
        message=f"Unsupported element type: {element_type} for Array field: {field_info.get('name')}"
    )


def _is_fp32_vector_value(field_value: Any) -> bool:
    if isinstance(field_value, list):
        return True
    if isinstance(field_value, np.ndarray):
        return field_value.dtype in ("float32", "float64")
    return False


def _pack_fp32_vector_value(field_value: Any, field_data: schema_types.FieldData) -> None:
    if isinstance(field_value, np.ndarray) and field_value.dtype not in ("float32", "float64"):
        raise ParamError(
            message="invalid input for float32 vector. Expected an np.ndarray with dtype=float32 or float64"
        )
    f_value = np.asarray(field_value, dtype=np.float32).tolist()
    field_data.vectors.dim = len(f_value)
    field_data.vectors.float_vector.data.extend(f_value)


def _convert_to_vector_bytes(field_value: Any, element_type: DataType) -> bytes:
    """Convert a single vector value to bytes for byte-based vector types."""
    if isinstance(field_value, bytes):
        return field_value

    if element_type == DataType.FLOAT16_VECTOR:
        if isinstance(field_value, np.ndarray):
            if field_value.dtype == "float16":
                return field_value.view(np.uint8).tobytes()
            raise ParamError(
                message="invalid input for float16 vector. Expected an np.ndarray with dtype=float16"
            )
        raise ParamError(
            message="invalid input type for float16 vector. Expected bytes or np.ndarray(dtype=float16)"
        )

    if element_type == DataType.BFLOAT16_VECTOR:
        if isinstance(field_value, np.ndarray):
            if field_value.dtype == "bfloat16":
                return field_value.view(np.uint8).tobytes()
            raise ParamError(
                message="invalid input for bfloat16 vector. Expected an np.ndarray with dtype=bfloat16"
            )
        raise ParamError(
            message="invalid input type for bfloat16 vector. Expected bytes or np.ndarray(dtype=bfloat16)"
        )

    if isinstance(field_value, np.ndarray):
        expected_dtypes = {
            DataType.INT8_VECTOR: "int8",
        }
        expected = expected_dtypes.get(element_type)
        if expected and field_value.dtype != expected:
            raise ParamError(
                message=f"invalid input for {expected} vector. Expected an np.ndarray with dtype={expected}"
            )
        return field_value.view(np.uint8).tobytes()
    raise ParamError(
        message=f"invalid input type for {element_type.name} vector. Expected bytes or np.ndarray"
    )


def convert_to_array_of_vector(obj: List[Any], field_info: Any):
    # Create a single VectorField that contains all vectors flattened
    field_data = schema_types.VectorField()
    element_type = field_info.get("element_type", None)

    field_data.dim = _get_dim(field_info)

    if element_type == DataType.FLOAT_VECTOR:
        attr_name = type_info.get_vector_attr(element_type)
        vector_values = getattr(field_data, attr_name).data
        if not obj:
            vector_values.extend([])
        for field_value in obj:
            f_value = field_value
            if isinstance(field_value, np.ndarray):
                if field_value.dtype not in ("float32", "float64"):
                    raise ParamError(
                        message="invalid input for float32 vector. Expected an np.ndarray with dtype=float32"
                    )
                f_value = field_value.tolist()
            vector_values.extend(f_value)

    elif type_info.is_byte_vector_type(element_type):
        attr_name = type_info.get_vector_attr(element_type)
        if element_type == DataType.BINARY_VECTOR:
            all_bytes = b"".join(fv if isinstance(fv, bytes) else bytes(fv) for fv in obj)
        else:
            all_bytes = b"".join(_convert_to_vector_bytes(fv, element_type) for fv in obj)
        setattr(field_data, attr_name, all_bytes)

    else:
        raise ParamError(
            message=f"Unsupported element type: {element_type} for Array of Vector field: {field_info.get('name')}"
        )
    return field_data


def entity_to_array_arr(entity_values: List[Any], field_info: Any):
    return convert_to_array_arr(entity_values, field_info)


_ROW_SCALAR_NORMALIZERS = {
    DataType.VARCHAR: lambda v, fi: convert_to_str_array(v, fi, CHECK_STR_ARRAY),
    DataType.GEOMETRY: lambda v, fi: convert_to_str_array(v, fi, CHECK_STR_ARRAY),
    DataType.JSON: lambda v, _: convert_to_json(v),
    DataType.ARRAY: convert_to_array,
}


def _get_protobuf_payload(field_data: schema_types.FieldData, dtype: Optional[DataType] = None):
    dtype = field_data.type if dtype is None else dtype
    slot = type_info.get_protobuf_slot(dtype)
    if slot.kind is type_info.ProtobufSlotKind.SCALAR:
        return getattr(field_data.scalars, slot.attr)
    return getattr(field_data.vectors, slot.attr)


def flush_vector_bytes(
    field_data: schema_types.FieldData, vector_bytes_cache: Dict[int, List[bytes]]
):
    """Flush the temporary byte list for bytes vector fields, merging all collected bytes.

    This function is used to optimize performance by avoiding O(n²) memory operations
    caused by using += operations in pack_field_value_to_field_data.
    Supports: INT8_VECTOR, BINARY_VECTOR, FLOAT16_VECTOR, BFLOAT16_VECTOR
    """
    field_id = id(field_data)
    bytes_list = vector_bytes_cache.pop(field_id, None)
    if not bytes_list:
        return

    attr_name = (
        type_info.get_vector_attr(field_data.type)
        if type_info.is_byte_vector_type(field_data.type)
        else None
    )
    if attr_name:
        setattr(field_data.vectors, attr_name, b"".join(bytes_list))


def _pack_scalar_row(
    dtype: DataType,
    value: Any,
    field_data: schema_types.FieldData,
    field_info: Any,
    field_name: str,
):
    try:
        payload = _get_protobuf_payload(field_data, dtype)
        if value is None:
            payload.data.extend([])
            return
        normalizer = _ROW_SCALAR_NORMALIZERS.get(dtype)
        payload.data.append(normalizer(value, field_info) if normalizer else value)
    except (TypeError, ValueError) as e:
        raise DataNotMatchException(
            message=_scalar_row_error_message(dtype, field_name, value, e)
        ) from e


def _pack_float_vector_row(
    value: Any,
    field_data: schema_types.FieldData,
    field_info: Any,
    field_name: str,
):
    try:
        payload = _get_protobuf_payload(field_data, DataType.FLOAT_VECTOR)
        if value is None:
            if field_data.vectors.dim == 0:
                field_data.vectors.dim = _get_dim(field_info)
            return

        f_value = value
        if isinstance(value, np.ndarray):
            if value.dtype not in ("float32", "float64"):
                raise ParamError(
                    message="invalid input for float32 vector. Expected an np.ndarray with dtype=float32"
                )
            f_value = value.tolist()

        field_data.vectors.dim = len(f_value)
        payload.data.extend(f_value)
    except (TypeError, ValueError) as e:
        raise DataNotMatchException(
            message=ExceptionsMessage.FieldDataInconsistent
            % (field_name, _error_label(DataType.FLOAT_VECTOR), type(value))
            + f" Detail: {e!s}"
        ) from e


def _normalize_byte_vector_row(dtype: DataType, value: Any) -> Optional[bytes]:
    if value is None:
        return None

    dtype = type_info.get_type_info(dtype).dtype

    if dtype == DataType.BINARY_VECTOR:
        # Preserve binary-vector input compatibility: bytes(123) is valid
        # Python but was rejected because scalar integers have no length.
        if not isinstance(value, Sized):
            message = f"object of type '{type(value).__name__}' has no len()"
            raise TypeError(message)
        return bytes(value)

    if not type_info.is_byte_vector_type(dtype):
        raise ParamError(message=f"Unsupported data type: {dtype}")

    return _convert_to_vector_bytes(value, dtype)


def _pack_byte_vector_row(
    dtype: DataType,
    value: Any,
    field_data: schema_types.FieldData,
    field_info: Any,
    field_name: str,
    vector_bytes_cache: Dict[int, List[bytes]],
):
    try:
        if dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR) and _is_fp32_vector_value(
            value
        ):
            _pack_fp32_vector_value(value, field_data)
            return

        payload = _normalize_byte_vector_row(dtype, value)
        if payload is None:
            if field_data.vectors.dim == 0:
                field_data.vectors.dim = _get_dim(field_info)
            return

        field_data.vectors.dim = _get_dim(field_info)
        vector_bytes_cache.setdefault(id(field_data), []).append(payload)
    except (TypeError, ValueError) as e:
        raise DataNotMatchException(
            message=ExceptionsMessage.FieldDataInconsistent
            % (field_name, _error_label(dtype), type(value))
            + f" Detail: {e!s}"
        ) from e


def _pack_sparse_vector_row(
    value: Any,
    field_data: schema_types.FieldData,
    field_name: str,
):
    try:
        payload = _get_protobuf_payload(field_data, DataType.SPARSE_FLOAT_VECTOR)
        if value is None:
            return

        if not SciPyHelper.is_scipy_sparse(value):
            value = [value]
        elif value.shape[0] != 1:
            raise ParamError(message="invalid input for sparse float vector: expect 1 row")
        if not entity_is_sparse_matrix(value):
            raise ParamError(message="invalid input for sparse float vector")
        payload.contents.append(sparse_rows_to_proto(value).contents[0])
    except (TypeError, ValueError) as e:
        raise DataNotMatchException(
            message=ExceptionsMessage.FieldDataInconsistent
            % (field_name, _error_label(DataType.SPARSE_FLOAT_VECTOR), type(value))
            + f" Detail: {e!s}"
        ) from e


def pack_field_value_to_field_data(
    field_value: Any,
    field_data: schema_types.FieldData,
    field_info: Any,
    vector_bytes_cache: Dict[int, List[bytes]],
):
    field_type = field_data.type
    field_name = field_info["name"]
    if type_info.is_scalar_type(field_type) or field_type == DataType.ARRAY:
        return _pack_scalar_row(field_type, field_value, field_data, field_info, field_name)
    if field_type == DataType.FLOAT_VECTOR:
        return _pack_float_vector_row(field_value, field_data, field_info, field_name)
    if field_type == DataType.SPARSE_FLOAT_VECTOR:
        return _pack_sparse_vector_row(field_value, field_data, field_name)
    if type_info.is_byte_vector_type(field_type):
        return _pack_byte_vector_row(
            field_type, field_value, field_data, field_info, field_name, vector_bytes_cache
        )
    raise ParamError(message=f"Unsupported data type: {field_type}")


# Don't change entity inside.
def entity_to_field_data(entity: Dict, field_info: Any, num_rows: int) -> schema_types.FieldData:
    entity_type = entity.get("type")
    field_name = entity.get("name")
    entity_values = entity.get("values")
    field_data = schema_types.FieldData(
        type=entity_type_to_dtype(entity_type),
        field_name=field_name,
    )

    valid_data = []
    if field_info.get("nullable", False) or field_info.get("default_value", None):
        if len(entity_values) == 0:
            valid_data = [False] * num_rows
        else:
            valid_data = [value is not None for value in entity_values]
            entity_values = [value for value in entity_values if value is not None]

    field_data.valid_data.extend(valid_data)

    try:
        if type_info.is_scalar_type(entity_type) or entity_type == DataType.ARRAY:
            attr_name = type_info.get_scalar_attr(entity_type)
            if entity_type in (DataType.VARCHAR, DataType.TIMESTAMPTZ, DataType.GEOMETRY):
                entity_values = entity_to_str_arr(entity_values, field_info, CHECK_STR_ARRAY)
            elif entity_type == DataType.JSON:
                entity_values = entity_to_json_arr(entity_values, field_info)
            elif entity_type == DataType.ARRAY:
                entity_values = entity_to_array_arr(entity_values, field_info)
            getattr(field_data.scalars, attr_name).data.extend(entity_values)
        elif entity_type == DataType.FLOAT_VECTOR:
            if len(entity_values) > 0:
                field_data.vectors.dim = len(entity_values[0])
            else:
                field_data.vectors.dim = _get_dim(field_info)
            all_floats = [f for vector in entity_values for f in vector]
            attr_name = type_info.get_vector_attr(entity_type)
            getattr(field_data.vectors, attr_name).data.extend(all_floats)
        elif type_info.is_byte_vector_type(entity_type):
            field_data.vectors.dim = _get_dim(field_info)
            attr_name = type_info.get_vector_attr(entity_type)
            setattr(field_data.vectors, attr_name, b"".join(entity_values))
        elif entity_type == DataType.SPARSE_FLOAT_VECTOR:
            entity_len = (
                entity_values.shape[0]
                if SciPyHelper.is_scipy_sparse(entity_values)
                else len(entity_values)
            )
            if entity_len > 0:
                attr_name = type_info.get_vector_attr(entity_type)
                getattr(field_data.vectors, attr_name).CopyFrom(sparse_rows_to_proto(entity_values))
        else:
            raise ParamError(message=f"Unsupported data type: {entity_type}")
    except (TypeError, ValueError) as e:
        raise DataNotMatchException(
            message=ExceptionsMessage.FieldDataInconsistent
            % (field_name, entity_type.name, type(entity_values[0]))
            + f" Detail: {e!s}"
        ) from e
    return field_data


def extract_dynamic_field_from_result(raw: Any):
    dynamic_field_name = None
    field_names = set()
    if raw.fields_data:
        for field_data in raw.fields_data:
            field_names.add(field_data.field_name)
            if field_data.is_dynamic:
                dynamic_field_name = field_data.field_name

    dynamic_fields = set()
    for name in raw.output_fields:
        if name == dynamic_field_name:
            dynamic_fields.clear()
            break
        if name not in field_names:
            dynamic_fields.add(name)
    return dynamic_field_name, dynamic_fields


def extract_array_rows(field_data: Any, entity_rows: List[Dict], row_count: int, has_valid: bool):
    if row_count == 0:
        return
    field_name = field_data.field_name
    valid_data = field_data.valid_data
    if has_valid:
        for i in range(row_count):
            entity_rows[i][field_name] = (
                field_data_extractors.decode_array_cell(field_data, i) if valid_data[i] else None
            )
    else:
        for i in range(row_count):
            entity_rows[i][field_name] = field_data_extractors.decode_array_cell(field_data, i)


def extract_row_data_from_fields_data_v2(
    field_data: Any,
    entity_rows: List[Dict],
) -> bool:
    row_count = len(entity_rows)
    has_valid = len(field_data.valid_data) > 0
    field_name = field_data.field_name
    valid_data = field_data.valid_data

    def assign_scalar(data: List[Any]) -> None:
        if has_valid:
            [
                entity_rows[i].__setitem__(field_name, None if not valid_data[i] else data[i])
                for i in range(row_count)
            ]
        else:
            [entity_rows[i].__setitem__(field_name, data[i]) for i in range(row_count)]

    if field_data.type == DataType.STRING:
        raise MilvusException(message="Not support string yet")

    try:
        scalar_attr = type_info.get_scalar_attr(field_data.type)
    except ParamError:
        scalar_attr = None
    if scalar_attr is not None and field_data.type not in (DataType.JSON, DataType.ARRAY):
        assign_scalar(getattr(field_data.scalars, scalar_attr).data)
        return False

    if field_data.type == DataType.JSON:
        return True

    if field_data.type == DataType.ARRAY:
        extract_array_rows(field_data, entity_rows, row_count, has_valid)
        return False
    return type_info.is_vector_type(field_data.type) or field_data.type in (
        DataType._ARRAY_OF_STRUCT,
        DataType._ARRAY_OF_VECTOR,
    )


def get_array_length(array_item: Any) -> int:
    """Get the length of an array field from its data."""
    return field_data_extractors.array_cell_length(array_item)


def get_array_value_at_index(array_item: Any, idx: int) -> Any:
    """Get the value at a specific index from an array field."""
    return field_data_extractors.decode_array_value(array_item, idx)


def _vector_array_element_count(vector_data: Any, element_type: DataType) -> int:
    try:
        return field_data_extractors.vector_array_length(vector_data, element_type)
    except ParamError:
        return 0


def _materialize_vector_array_value(element_type: DataType, value: Any) -> Any:
    if value is None:
        return None
    if element_type == DataType.FLOAT_VECTOR:
        return list(value)
    if element_type == DataType.BINARY_VECTOR:
        return [value]
    if not type_info.is_byte_vector_type(element_type):
        raise ParamError(message=f"Unimplemented type: {element_type} for vector array extraction")

    numpy_dtype = type_info.require_numpy_dtype(element_type)
    if element_type == DataType.BFLOAT16_VECTOR and not hasattr(np, "bfloat16"):
        numpy_dtype = "uint16"
    return list(np.frombuffer(value, dtype=numpy_dtype))


def extract_struct_array_from_column_data(struct_arrays: Any, row_idx: int) -> List[Dict[str, Any]]:
    """Convert column-format struct data back to array of structs format.

    Milvus stores struct arrays in column format where each field's data is stored separately.
    This function converts it back to row format for user consumption.

    For example, if the original one row of array of struct data was:
    [
        {"name": "Alice", "age": 30, "vector": [1.0, 2.0]},
        {"name": "Bob", "age": 25, "vector": [3.0, 4.0]}
    ]

    Milvus stores it as:
    - name field: ["Alice", "Bob"]
    - age field: [30, 25]
    - vector field: [[1.0, 2.0], [3.0, 4.0]]

    This function reconstructs the original array of struct format.

    Args:
        struct_arrays: The struct_arrays field containing column-format data with sub-fields
        row_idx: The row index to extract data for

    Returns:
        List of dictionaries representing the struct array in row format
    """
    if not struct_arrays or not hasattr(struct_arrays, "fields"):
        return []

    # Determine the number of struct elements by checking the first sub-field's data length
    # All sub-fields should have the same number of elements
    num_structs = 0
    for sub_field in struct_arrays.fields:
        if sub_field.type == DataType.ARRAY and row_idx < len(sub_field.scalars.array_data.data):
            array_item = sub_field.scalars.array_data.data[row_idx]
            num_structs = get_array_length(array_item)
            if num_structs > 0:
                break
        elif sub_field.type == DataType._ARRAY_OF_VECTOR:
            if (
                hasattr(sub_field, "vectors")
                and hasattr(sub_field.vectors, "vector_array")
                and row_idx < len(sub_field.vectors.vector_array.data)
            ):
                vector_data = sub_field.vectors.vector_array.data[row_idx]
                element_type = sub_field.vectors.vector_array.element_type
                num_structs = _vector_array_element_count(vector_data, element_type)

                if num_structs > 0:
                    break

    # Build array of struct objects by extracting data at each struct index
    struct_array = []
    for struct_idx in range(num_structs):
        struct_obj = {}

        for sub_field in struct_arrays.fields:
            sub_field_name = sub_field.field_name

            # Handle scalar array fields (VARCHAR, INT, FLOAT, etc.)
            if sub_field.type == DataType.ARRAY:
                if row_idx < len(sub_field.scalars.array_data.data):
                    array_item = sub_field.scalars.array_data.data[row_idx]
                    # Extract the value at struct_idx from the appropriate data type
                    struct_obj[sub_field_name] = get_array_value_at_index(array_item, struct_idx)

            elif sub_field.type == DataType._ARRAY_OF_VECTOR:
                if hasattr(sub_field, "vectors") and hasattr(sub_field.vectors, "vector_array"):
                    vector_array = sub_field.vectors.vector_array
                    if row_idx < len(vector_array.data):
                        vector_data = vector_array.data[row_idx]
                        element_type = vector_array.element_type
                        try:
                            struct_obj[sub_field_name] = _materialize_vector_array_value(
                                element_type,
                                field_data_extractors.decode_vector_array_value(
                                    vector_data, element_type, struct_idx
                                ),
                            )
                        except ParamError:
                            struct_obj[sub_field_name] = None

            # All struct sub-fields should be either ARRAY or ARRAY_OF_VECTOR
            # Any other type indicates a bug in the data conversion logic
            else:
                raise ParamError(
                    message=f"Unexpected field type {sub_field.type} for struct sub-field {sub_field_name}. "
                    f"Struct fields must be ARRAY or ARRAY_OF_VECTOR internally."
                )

        struct_array.append(struct_obj)

    return struct_array
