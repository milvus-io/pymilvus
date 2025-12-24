import json
import logging
import math
import struct
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

from .types import DataType
from .utils import (
    SciPyHelper,
    SparseMatrixInputType,
    SparseRowOutputType,
    sparse_parse_single_row,
)

logger = logging.getLogger(__name__)

CHECK_STR_ARRAY = True


def entity_is_sparse_matrix(entity: Any):
    if SciPyHelper.is_scipy_sparse(entity):
        return True
    try:

        def is_type_in_str(v: Any, t: Any):
            if not isinstance(v, str):
                return False
            try:
                t(v)
            except ValueError:
                return False
            return True

        def is_int_type(v: Any):
            return isinstance(v, (int, np.integer)) or is_type_in_str(v, int)

        def is_float_type(v: Any):
            return isinstance(v, (float, np.floating)) or is_type_in_str(v, float)

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
                if len(pair) != 2 or not is_int_type(pair[0]) or not is_float_type(pair[1]):
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
        data = b""
        for i, v in sorted(zip(indices, values), key=lambda x: x[0]):
            if not (0 <= i < 2**32 - 1):
                raise ParamError(
                    message=f"sparse vector index must be positive and less than 2^32-1: {i}"
                )
            if math.isnan(v):
                raise ParamError(message="sparse vector value must not be NaN")
            data += struct.pack("I", i)
            data += struct.pack("f", v)
        return data

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
        for _, row_data in enumerate(data):
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
    start = start or 0
    end = end or len(sfv.contents)
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
    if Config.EncodeProtocol.lower() != "utf-8".lower():
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
                items = list(current.items())
                for k, v in reversed(items):
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
    arr = []
    for obj in objs:
        if obj is None:
            raise ParamError(message=f"field ({field_info['name']}) expects a non-None input")
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
    if element_type == DataType.BOOL:
        field_data.bool_data.data.extend(obj)
        return field_data
    if element_type in (DataType.INT8, DataType.INT16, DataType.INT32):
        field_data.int_data.data.extend(obj)
        return field_data
    if element_type == DataType.INT64:
        field_data.long_data.data.extend(obj)
        return field_data
    if element_type == DataType.FLOAT:
        field_data.float_data.data.extend(obj)
        return field_data
    if element_type == DataType.DOUBLE:
        field_data.double_data.data.extend(obj)
        return field_data
    if element_type in (DataType.VARCHAR, DataType.STRING):
        field_data.string_data.data.extend(obj)
        return field_data
    raise ParamError(
        message=f"Unsupported element type: {element_type} for Array field: {field_info.get('name')}"
    )


def convert_to_array_of_vector(obj: List[Any], field_info: Any):
    # Create a single VectorField that contains all vectors flattened
    field_data = schema_types.VectorField()
    element_type = field_info.get("element_type", None)

    dim_value = field_info.get("params", {}).get("dim", 0)
    if isinstance(dim_value, str):
        dim_value = int(dim_value)
    field_data.dim = dim_value

    if element_type == DataType.FLOAT_VECTOR:
        if not obj:
            field_data.float_vector.data.extend([])
        for field_value in obj:
            f_value = field_value
            if isinstance(field_value, np.ndarray):
                if field_value.dtype not in ("float32", "float64"):
                    raise ParamError(
                        message="invalid input for float32 vector. Expected an np.ndarray with dtype=float32"
                    )
                f_value = field_value.tolist()
            field_data.float_vector.data.extend(f_value)

    else:
        # todo(SpadeA): other types are now not supported. When it's supported, make sure empty
        # array is handled correctly.
        raise ParamError(
            message=f"Unsupported element type: {element_type} for Array of Vector field: {field_info.get('name')}"
        )
    return field_data


def entity_to_array_arr(entity_values: List[Any], field_info: Any):
    return convert_to_array_arr(entity_values, field_info)


def pack_field_value_to_field_data(
    field_value: Any, field_data: schema_types.FieldData, field_info: Any
):
    field_type = field_data.type
    field_name = field_info["name"]
    if field_type == DataType.BOOL:
        try:
            if field_value is None:
                field_data.scalars.bool_data.data.extend([])
            else:
                field_data.scalars.bool_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "bool", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type in (DataType.INT8, DataType.INT16, DataType.INT32):
        try:
            # need to extend it, or cannot correctly identify field_data.scalars.int_data.data
            if field_value is None:
                field_data.scalars.int_data.data.extend([])
            else:
                field_data.scalars.int_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.INT64:
        try:
            if field_value is None:
                field_data.scalars.long_data.data.extend([])
            else:
                field_data.scalars.long_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int64", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.FLOAT:
        try:
            if field_value is None:
                field_data.scalars.float_data.data.extend([])
            else:
                field_data.scalars.float_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.DOUBLE:
        try:
            if field_value is None:
                field_data.scalars.double_data.data.extend([])
            else:
                field_data.scalars.double_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "double", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.TIMESTAMPTZ:
        try:
            if field_value is None:
                field_data.scalars.string_data.data.extend([])  # Timestamptz is passed as String
            else:
                field_data.scalars.string_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "string", type(field_value))
            ) from e
    elif field_type == DataType.FLOAT_VECTOR:
        try:
            if field_value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                f_value = field_value
                if isinstance(field_value, np.ndarray):
                    if field_value.dtype not in ("float32", "float64"):
                        raise ParamError(
                            message="invalid input for float32 vector. Expected an np.ndarray with dtype=float32"
                        )
                    f_value = field_value.tolist()

                field_data.vectors.dim = len(f_value)
                field_data.vectors.float_vector.data.extend(f_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.BINARY_VECTOR:
        try:
            if field_value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                field_data.vectors.dim = len(field_value) * 8
                field_data.vectors.binary_vector += bytes(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "binary_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.FLOAT16_VECTOR:
        try:
            if field_value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                if isinstance(field_value, bytes):
                    v_bytes = field_value
                elif isinstance(field_value, np.ndarray):
                    if field_value.dtype != "float16":
                        raise ParamError(
                            message="invalid input for float16 vector. Expected an np.ndarray with dtype=float16"
                        )
                    v_bytes = field_value.view(np.uint8).tobytes()
                else:
                    raise ParamError(
                        message="invalid input type for float16 vector. Expected an np.ndarray with dtype=float16"
                    )

                field_data.vectors.dim = len(v_bytes) // 2
                field_data.vectors.float16_vector += v_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float16_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.BFLOAT16_VECTOR:
        try:
            if field_value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                if isinstance(field_value, bytes):
                    v_bytes = field_value
                elif isinstance(field_value, np.ndarray):
                    if field_value.dtype != "bfloat16":
                        raise ParamError(
                            message="invalid input for bfloat16 vector. Expected an np.ndarray with dtype=bfloat16"
                        )
                    v_bytes = field_value.view(np.uint8).tobytes()
                else:
                    raise ParamError(
                        message="invalid input type for bfloat16 vector. Expected an np.ndarray with dtype=bfloat16"
                    )

                field_data.vectors.dim = len(v_bytes) // 2
                field_data.vectors.bfloat16_vector += v_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "bfloat16_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.SPARSE_FLOAT_VECTOR:
        try:
            if field_value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = 0
            else:
                if not SciPyHelper.is_scipy_sparse(field_value):
                    field_value = [field_value]
                elif field_value.shape[0] != 1:
                    raise ParamError(message="invalid input for sparse float vector: expect 1 row")
                if not entity_is_sparse_matrix(field_value):
                    raise ParamError(message="invalid input for sparse float vector")
                field_data.vectors.sparse_float_vector.contents.append(
                    sparse_rows_to_proto(field_value).contents[0]
                )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "sparse_float_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.INT8_VECTOR:
        try:
            if field_value is None:
                if field_data.vectors.dim == 0:
                    field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            else:
                if isinstance(field_value, np.ndarray):
                    if field_value.dtype != "int8":
                        raise ParamError(
                            message="invalid input for int8 vector. Expected an np.ndarray with dtype=int8"
                        )
                    i_bytes = field_value.view(np.int8).tobytes()
                else:
                    raise ParamError(
                        message="invalid input for int8 vector. Expected an np.ndarray with dtype=int8"
                    )

                field_data.vectors.dim = len(i_bytes)
                field_data.vectors.int8_vector += i_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int8_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.VARCHAR:
        try:
            if field_value is None:
                field_data.scalars.string_data.data.extend([])
            else:
                field_data.scalars.string_data.data.append(
                    convert_to_str_array(field_value, field_info, CHECK_STR_ARRAY)
                )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "varchar", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.GEOMETRY:
        try:
            if field_value is None:
                field_data.scalars.geometry_wkt_data.data.extend([])
            else:
                field_data.scalars.geometry_wkt_data.data.append(
                    convert_to_str_array(field_value, field_info, CHECK_STR_ARRAY)
                )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "geometry", type(field_value))
            ) from e
    elif field_type == DataType.JSON:
        try:
            if field_value is None:
                field_data.scalars.json_data.data.extend([])
            else:
                field_data.scalars.json_data.data.append(convert_to_json(field_value))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "json", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    elif field_type == DataType.ARRAY:
        try:
            if field_value is None:
                field_data.scalars.array_data.data.extend([])
            else:
                field_data.scalars.array_data.data.append(convert_to_array(field_value, field_info))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "array", type(field_value))
                + f" Detail: {e!s}"
            ) from e
    else:
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
        if entity_type == DataType.BOOL:
            field_data.scalars.bool_data.data.extend(entity_values)
        elif entity_type in (DataType.INT8, DataType.INT16, DataType.INT32):
            field_data.scalars.int_data.data.extend(entity_values)
        elif entity_type == DataType.INT64:
            field_data.scalars.long_data.data.extend(entity_values)
        elif entity_type == DataType.FLOAT:
            field_data.scalars.float_data.data.extend(entity_values)
        elif entity_type == DataType.DOUBLE:
            field_data.scalars.double_data.data.extend(entity_values)

        elif entity_type == DataType.FLOAT_VECTOR:
            if len(entity_values) > 0:
                field_data.vectors.dim = len(entity_values[0])
            else:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            all_floats = [f for vector in entity_values for f in vector]
            field_data.vectors.float_vector.data.extend(all_floats)
        elif entity_type == DataType.BINARY_VECTOR:
            if len(entity_values) > 0:
                field_data.vectors.dim = len(entity_values[0]) * 8
            else:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            field_data.vectors.binary_vector = b"".join(entity_values)
        elif entity_type == DataType.FLOAT16_VECTOR:
            if len(entity_values) > 0:
                field_data.vectors.dim = len(entity_values[0]) // 2
            else:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            field_data.vectors.float16_vector = b"".join(entity_values)
        elif entity_type == DataType.BFLOAT16_VECTOR:
            if len(entity_values) > 0:
                field_data.vectors.dim = len(entity_values[0]) // 2
            else:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            field_data.vectors.bfloat16_vector = b"".join(entity_values)
        elif entity_type == DataType.SPARSE_FLOAT_VECTOR:
            if len(entity_values) > 0:
                field_data.vectors.sparse_float_vector.CopyFrom(sparse_rows_to_proto(entity_values))
        elif entity_type == DataType.INT8_VECTOR:
            if len(entity_values) > 0:
                field_data.vectors.dim = len(entity_values[0])
            else:
                field_data.vectors.dim = field_info.get("params", {}).get("dim", 0)
            field_data.vectors.int8_vector = b"".join(entity_values)

        elif entity_type == DataType.VARCHAR:
            field_data.scalars.string_data.data.extend(
                entity_to_str_arr(entity_values, field_info, CHECK_STR_ARRAY)
            )
        elif entity_type == DataType.JSON:
            field_data.scalars.json_data.data.extend(entity_to_json_arr(entity_values, field_info))
        elif entity_type == DataType.ARRAY:
            field_data.scalars.array_data.data.extend(
                entity_to_array_arr(entity_values, field_info)
            )
        elif entity_type == DataType.GEOMETRY:
            field_data.scalars.geometry_wkt_data.data.extend(
                entity_to_str_arr(entity_values, field_info, CHECK_STR_ARRAY)
            )
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


def extract_array_row_data_with_validity(field_data: Any, entity_rows: List[Dict], row_count: int):
    field_name = field_data.field_name
    data = field_data.scalars.array_data.data
    element_type = field_data.scalars.array_data.element_type
    if element_type == DataType.INT64:
        [
            entity_rows[i].__setitem__(
                field_name, data[i].long_data.data if field_data.valid_data[i] else None
            )
            for i in range(row_count)
        ]
    elif element_type == DataType.BOOL:
        [
            entity_rows[i].__setitem__(
                field_name, data[i].bool_data.data if field_data.valid_data[i] else None
            )
            for i in range(row_count)
        ]
    elif element_type in (
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
    ):
        [
            entity_rows[i].__setitem__(
                field_name, data[i].int_data.data if field_data.valid_data[i] else None
            )
            for i in range(row_count)
        ]
    elif element_type == DataType.FLOAT:
        [
            entity_rows[i].__setitem__(
                field_name, data[i].float_data.data if field_data.valid_data[i] else None
            )
            for i in range(row_count)
        ]
    elif element_type == DataType.DOUBLE:
        [
            entity_rows[i].__setitem__(
                field_name, data[i].double_data.data if field_data.valid_data[i] else None
            )
            for i in range(row_count)
        ]
    elif element_type in (
        DataType.STRING,
        DataType.VARCHAR,
    ):
        [
            entity_rows[i].__setitem__(
                field_name, data[i].string_data.data if field_data.valid_data[i] else None
            )
            for i in range(row_count)
        ]
    else:
        raise MilvusException(message=f"Unsupported data type: {element_type}")


def extract_array_row_data_no_validity(field_data: Any, entity_rows: List[Dict], row_count: int):
    field_name = field_data.field_name
    data = field_data.scalars.array_data.data
    element_type = field_data.scalars.array_data.element_type
    if element_type == DataType.INT64:
        [entity_rows[i].__setitem__(field_name, data[i].long_data.data) for i in range(row_count)]
    elif element_type == DataType.BOOL:
        [entity_rows[i].__setitem__(field_name, data[i].bool_data.data) for i in range(row_count)]
    elif element_type in (
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
    ):
        [entity_rows[i].__setitem__(field_name, data[i].int_data.data) for i in range(row_count)]
    elif element_type == DataType.FLOAT:
        [entity_rows[i].__setitem__(field_name, data[i].float_data.data) for i in range(row_count)]
    elif element_type == DataType.DOUBLE:
        [entity_rows[i].__setitem__(field_name, data[i].double_data.data) for i in range(row_count)]
    elif element_type in (
        DataType.STRING,
        DataType.VARCHAR,
    ):
        [entity_rows[i].__setitem__(field_name, data[i].string_data.data) for i in range(row_count)]
    else:
        raise MilvusException(message=f"Unsupported data type: {element_type}")


def extract_array_row_data(field_data: Any, index: int):
    array = field_data.scalars.array_data.data[index]
    if field_data.scalars.array_data.element_type == DataType.INT64:
        return array.long_data.data
    if field_data.scalars.array_data.element_type == DataType.BOOL:
        return array.bool_data.data
    if field_data.scalars.array_data.element_type in (
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
    ):
        return array.int_data.data
    if field_data.scalars.array_data.element_type == DataType.FLOAT:
        return array.float_data.data
    if field_data.scalars.array_data.element_type == DataType.DOUBLE:
        return array.double_data.data
    if field_data.scalars.array_data.element_type in (
        DataType.STRING,
        DataType.VARCHAR,
    ):
        return array.string_data.data
    return None


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

    if field_data.type == DataType.BOOL:
        data = field_data.scalars.bool_data.data
        assign_scalar(data)
        return False

    if field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
        data = field_data.scalars.int_data.data
        assign_scalar(data)
        return False

    if field_data.type == DataType.INT64:
        data = field_data.scalars.long_data.data
        assign_scalar(data)
        return False

    if field_data.type == DataType.FLOAT:
        data = field_data.scalars.float_data.data
        assign_scalar(data)
        return False

    if field_data.type == DataType.DOUBLE:
        data = field_data.scalars.double_data.data
        assign_scalar(data)
        return False

    if field_data.type == DataType.TIMESTAMPTZ:
        data = field_data.scalars.string_data.data
        assign_scalar(data)
        return False

    if field_data.type == DataType.VARCHAR:
        data = field_data.scalars.string_data.data
        assign_scalar(data)
        return False

    if field_data.type == DataType.GEOMETRY:
        data = field_data.scalars.geometry_wkt_data.data
        assign_scalar(data)
        return False

    if field_data.type == DataType.JSON:
        return True

    if field_data.type == DataType.ARRAY:
        if has_valid:
            extract_array_row_data_with_validity(field_data, entity_rows, row_count)
        else:
            extract_array_row_data_no_validity(field_data, entity_rows, row_count)
        return False
    if field_data.type in (
        DataType.FLOAT_VECTOR,
        DataType.FLOAT16_VECTOR,
        DataType.BFLOAT16_VECTOR,
        DataType.BINARY_VECTOR,
        DataType.SPARSE_FLOAT_VECTOR,
        DataType.INT8_VECTOR,
    ):
        return True
    if field_data.type == DataType._ARRAY_OF_STRUCT:
        return True
    if field_data.type == DataType._ARRAY_OF_VECTOR:
        return True
    if field_data.type == DataType.STRING:
        raise MilvusException(message="Not support string yet")
    return False


def extract_vector_array_row_data(field_data: Any, index: int):
    array = field_data.vectors.vector_array.data[index]
    element_type = field_data.vectors.vector_array.element_type

    if element_type == DataType.FLOAT_VECTOR:
        return list(np.array(array.float_vector.data, dtype=np.float32))

    if element_type == DataType.FLOAT16_VECTOR:
        byte_data = array.float16_vector
        return list(np.frombuffer(byte_data, dtype=np.float16))

    if element_type == DataType.BFLOAT16_VECTOR:
        byte_data = array.bfloat16_vector
        return list(
            np.frombuffer(byte_data, dtype="bfloat16" if hasattr(np, "bfloat16") else np.uint16)
        )

    if element_type == DataType.INT8_VECTOR:
        byte_data = array.int8_vector
        return list(np.frombuffer(byte_data, dtype=np.int8))

    if element_type == DataType.BINARY_VECTOR:
        return [array.binary_vector]

    raise ParamError(message=f"Unimplemented type: {element_type} for vector array extraction")


# pylint: disable=R1702 (too-many-nested-blocks)
# pylint: disable=R0915 (too-many-statements)
def extract_row_data_from_fields_data(
    fields_data: Any,
    index: Any,
    dynamic_output_fields: Optional[List] = None,
):
    if not fields_data:
        return {}

    entity_row_data = {}
    dynamic_fields = dynamic_output_fields or set()

    # Cache prefix sums by field_data id for O(1) lookup (avoids O(n) per row)
    prefix_sum_cache: Dict[int, Any] = {}

    def get_physical_index(field_data: Any, logical_index: int) -> int:
        """Calculate physical index for nullable vectors with sparse storage.

        For nullable vectors, valid_data indicates which logical positions have valid data,
        and the actual data only contains valid values (sparse storage).
        Uses prefix sum for O(1) lookup instead of O(n) iteration.
        """
        field_id = id(field_data)
        if field_id not in prefix_sum_cache:
            if len(field_data.valid_data) == 0:
                prefix_sum_cache[field_id] = None
            else:
                prefix_sum_cache[field_id] = np.cumsum(
                    [0] + [1 if v else 0 for v in field_data.valid_data]
                )
        prefix_sum = prefix_sum_cache[field_id]
        if prefix_sum is None:
            return logical_index
        return int(prefix_sum[logical_index])

    def check_append(field_data: Any, row_data: Dict):
        if field_data.type == DataType.STRING:
            raise MilvusException(message="Not support string yet")

        if field_data.type == DataType.BOOL and len(field_data.scalars.bool_data.data) >= index:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            row_data[field_data.field_name] = field_data.scalars.bool_data.data[index]
            return

        if (
            field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32)
            and len(field_data.scalars.int_data.data) >= index
        ):
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            row_data[field_data.field_name] = field_data.scalars.int_data.data[index]
            return

        if field_data.type == DataType.INT64 and len(field_data.scalars.long_data.data) >= index:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            row_data[field_data.field_name] = field_data.scalars.long_data.data[index]
            return

        if field_data.type == DataType.FLOAT and len(field_data.scalars.float_data.data) >= index:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            row_data[field_data.field_name] = np.single(field_data.scalars.float_data.data[index])
            return

        if field_data.type == DataType.DOUBLE and len(field_data.scalars.double_data.data) >= index:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            row_data[field_data.field_name] = field_data.scalars.double_data.data[index]
            return

        if (
            field_data.type == DataType.VARCHAR
            and len(field_data.scalars.string_data.data) >= index
        ):
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            row_data[field_data.field_name] = field_data.scalars.string_data.data[index]
            return

        if (
            field_data.type == DataType.GEOMETRY
            and len(field_data.scalars.geometry_wkt_data.data) >= index
        ):
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                entity_row_data[field_data.field_name] = None
                return
            entity_row_data[field_data.field_name] = field_data.scalars.geometry_wkt_data.data[
                index
            ]
            return

        if field_data.type == DataType.JSON and len(field_data.scalars.json_data.data) >= index:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            try:
                json_dict = orjson.loads(field_data.scalars.json_data.data[index])
            except Exception as e:
                logger.error(
                    f"extract_row_data_from_fields_data::Failed to load JSON data: {e}, original data: {field_data.scalars.json_data.data[index]}"
                )
                raise

            if not field_data.is_dynamic:
                row_data[field_data.field_name] = json_dict
                return

            if not dynamic_fields:
                row_data.update(json_dict)
                return

            row_data.update({k: v for k, v in json_dict.items() if k in dynamic_fields})
            return
        if field_data.type == DataType.ARRAY and len(field_data.scalars.array_data.data) >= index:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
                return
            row_data[field_data.field_name] = extract_array_row_data(field_data, index)

        elif field_data.type == DataType.FLOAT_VECTOR:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
            else:
                dim = field_data.vectors.dim
                phys_idx = get_physical_index(field_data, index)
                if len(field_data.vectors.float_vector.data) >= (phys_idx + 1) * dim:
                    start_pos, end_pos = phys_idx * dim, (phys_idx + 1) * dim
                    # Here we use numpy.array to convert the float64 values to numpy.float32 values,
                    # and return a list of numpy.float32 to users
                    # By using numpy.array, performance improved by 60%
                    # for topk=16384 dim=1536 case.
                    arr = np.array(
                        field_data.vectors.float_vector.data[start_pos:end_pos], dtype=np.float32
                    )
                    row_data[field_data.field_name] = list(arr)
        elif field_data.type == DataType.BINARY_VECTOR:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
            else:
                dim = field_data.vectors.dim
                blen = dim // 8
                phys_idx = get_physical_index(field_data, index)
                if len(field_data.vectors.binary_vector) >= (phys_idx + 1) * blen:
                    start_pos, end_pos = phys_idx * blen, (phys_idx + 1) * blen
                    row_data[field_data.field_name] = [
                        field_data.vectors.binary_vector[start_pos:end_pos]
                    ]
        elif field_data.type == DataType.BFLOAT16_VECTOR:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
            else:
                dim = field_data.vectors.dim
                byte_per_row = dim * 2
                phys_idx = get_physical_index(field_data, index)
                if len(field_data.vectors.bfloat16_vector) >= (phys_idx + 1) * byte_per_row:
                    start_pos, end_pos = phys_idx * byte_per_row, (phys_idx + 1) * byte_per_row
                    row_data[field_data.field_name] = [
                        field_data.vectors.bfloat16_vector[start_pos:end_pos]
                    ]
        elif field_data.type == DataType.FLOAT16_VECTOR:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
            else:
                dim = field_data.vectors.dim
                byte_per_row = dim * 2
                phys_idx = get_physical_index(field_data, index)
                if len(field_data.vectors.float16_vector) >= (phys_idx + 1) * byte_per_row:
                    start_pos, end_pos = phys_idx * byte_per_row, (phys_idx + 1) * byte_per_row
                    row_data[field_data.field_name] = [
                        field_data.vectors.float16_vector[start_pos:end_pos]
                    ]
        elif field_data.type == DataType.SPARSE_FLOAT_VECTOR:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
            else:
                phys_idx = get_physical_index(field_data, index)
                row_data[field_data.field_name] = sparse_parse_single_row(
                    field_data.vectors.sparse_float_vector.contents[phys_idx]
                )
        elif field_data.type == DataType.INT8_VECTOR:
            if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
                row_data[field_data.field_name] = None
            else:
                dim = field_data.vectors.dim
                phys_idx = get_physical_index(field_data, index)
                if len(field_data.vectors.int8_vector) >= (phys_idx + 1) * dim:
                    start_pos, end_pos = phys_idx * dim, (phys_idx + 1) * dim
                    row_data[field_data.field_name] = [
                        field_data.vectors.int8_vector[start_pos:end_pos]
                    ]
        elif (
            field_data.type == DataType._ARRAY_OF_VECTOR
            and len(field_data.vectors.vector_array.data) >= index
        ):
            row_data[field_data.field_name] = extract_vector_array_row_data(field_data, index)
        elif field_data.type == DataType._ARRAY_OF_STRUCT:
            row_data[field_data.field_name] = {}
            for sub_field_data in field_data.struct_arrays.fields:
                check_append(sub_field_data, row_data[field_data.field_name])

    for field_data in fields_data:
        check_append(field_data, entity_row_data)

    return entity_row_data


def get_array_length(array_item: Any) -> int:
    """Get the length of an array field from its data."""
    if hasattr(array_item, "string_data") and hasattr(array_item.string_data, "data"):
        length = len(array_item.string_data.data)
        if length > 0:
            return length
    if hasattr(array_item, "int_data") and hasattr(array_item.int_data, "data"):
        length = len(array_item.int_data.data)
        if length > 0:
            return length
    if hasattr(array_item, "long_data") and hasattr(array_item.long_data, "data"):
        length = len(array_item.long_data.data)
        if length > 0:
            return length
    if hasattr(array_item, "float_data") and hasattr(array_item.float_data, "data"):
        length = len(array_item.float_data.data)
        if length > 0:
            return length
    if hasattr(array_item, "double_data") and hasattr(array_item.double_data, "data"):
        length = len(array_item.double_data.data)
        if length > 0:
            return length
    if hasattr(array_item, "bool_data") and hasattr(array_item.bool_data, "data"):
        length = len(array_item.bool_data.data)
        if length > 0:
            return length
    return 0


def get_array_value_at_index(array_item: Any, idx: int) -> Any:
    """Get the value at a specific index from an array field."""
    if (
        hasattr(array_item, "string_data")
        and hasattr(array_item.string_data, "data")
        and len(array_item.string_data.data) > idx
    ):
        return array_item.string_data.data[idx]
    if (
        hasattr(array_item, "int_data")
        and hasattr(array_item.int_data, "data")
        and len(array_item.int_data.data) > idx
    ):
        return array_item.int_data.data[idx]
    if (
        hasattr(array_item, "long_data")
        and hasattr(array_item.long_data, "data")
        and len(array_item.long_data.data) > idx
    ):
        return array_item.long_data.data[idx]
    if (
        hasattr(array_item, "float_data")
        and hasattr(array_item.float_data, "data")
        and len(array_item.float_data.data) > idx
    ):
        return array_item.float_data.data[idx]
    if (
        hasattr(array_item, "double_data")
        and hasattr(array_item.double_data, "data")
        and len(array_item.double_data.data) > idx
    ):
        return array_item.double_data.data[idx]
    if (
        hasattr(array_item, "bool_data")
        and hasattr(array_item.bool_data, "data")
        and len(array_item.bool_data.data) > idx
    ):
        return array_item.bool_data.data[idx]
    return None


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
                dim = vector_data.dim
                element_type = sub_field.vectors.vector_array.element_type

                if element_type == DataType.FLOAT_VECTOR:
                    num_structs = len(vector_data.float_vector.data) // dim
                elif element_type == DataType.FLOAT16_VECTOR:
                    num_structs = len(vector_data.float16_vector) // (dim * 2)
                elif element_type == DataType.BFLOAT16_VECTOR:
                    num_structs = len(vector_data.bfloat16_vector) // (dim * 2)
                elif element_type == DataType.INT8_VECTOR:
                    num_structs = len(vector_data.int8_vector) // dim
                elif element_type == DataType.BINARY_VECTOR:
                    num_structs = len(vector_data.binary_vector) // (dim // 8)
                else:
                    num_structs = 0

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
                        dim = vector_data.dim
                        element_type = vector_array.element_type

                        if element_type == DataType.FLOAT_VECTOR:
                            float_data = vector_data.float_vector.data
                            vec_start = struct_idx * dim
                            vec_end = vec_start + dim
                            if vec_end <= len(float_data):
                                struct_obj[sub_field_name] = list(float_data[vec_start:vec_end])
                            else:
                                struct_obj[sub_field_name] = None

                        elif element_type == DataType.FLOAT16_VECTOR:
                            byte_data = vector_data.float16_vector
                            bytes_per_vec = dim * 2
                            vec_start = struct_idx * bytes_per_vec
                            vec_end = vec_start + bytes_per_vec
                            if vec_end <= len(byte_data):
                                vec_bytes = byte_data[vec_start:vec_end]
                                struct_obj[sub_field_name] = list(
                                    np.frombuffer(vec_bytes, dtype=np.float16)
                                )
                            else:
                                struct_obj[sub_field_name] = None

                        elif element_type == DataType.BFLOAT16_VECTOR:
                            byte_data = vector_data.bfloat16_vector
                            bytes_per_vec = dim * 2
                            vec_start = struct_idx * bytes_per_vec
                            vec_end = vec_start + bytes_per_vec
                            if vec_end <= len(byte_data):
                                vec_bytes = byte_data[vec_start:vec_end]
                                dtype = "bfloat16" if hasattr(np, "bfloat16") else np.uint16
                                struct_obj[sub_field_name] = list(
                                    np.frombuffer(vec_bytes, dtype=dtype)
                                )
                            else:
                                struct_obj[sub_field_name] = None

                        elif element_type == DataType.INT8_VECTOR:
                            byte_data = vector_data.int8_vector
                            bytes_per_vec = dim
                            vec_start = struct_idx * bytes_per_vec
                            vec_end = vec_start + bytes_per_vec
                            if vec_end <= len(byte_data):
                                vec_bytes = byte_data[vec_start:vec_end]
                                struct_obj[sub_field_name] = list(
                                    np.frombuffer(vec_bytes, dtype=np.int8)
                                )
                            else:
                                struct_obj[sub_field_name] = None

                        elif element_type == DataType.BINARY_VECTOR:
                            byte_data = vector_data.binary_vector
                            bytes_per_vec = dim // 8
                            vec_start = struct_idx * bytes_per_vec
                            vec_end = vec_start + bytes_per_vec
                            if vec_end <= len(byte_data):
                                struct_obj[sub_field_name] = [byte_data[vec_start:vec_end]]
                            else:
                                struct_obj[sub_field_name] = None
                        else:
                            # Unsupported vector type, set to None
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
