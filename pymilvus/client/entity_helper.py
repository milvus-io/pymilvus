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

from .type_handlers import get_type_handler
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
        if isinstance(obj, dict):
            return {k: preprocess_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [preprocess_numpy_types(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

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

    return orjson.dumps(processed_obj)


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

    # Use type handler to pack data into ScalarField
    try:
        handler = get_type_handler(element_type)
        handler.pack_to_scalar_field(obj, field_data)
    except (ValueError, NotImplementedError):
        # Handler not found or not implemented
        raise ParamError(
            message=f"Unsupported element type: {element_type} for Array field: {field_info.get('name')}"
        ) from None
    else:
        return field_data


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

    # Use type handler for all types
    try:
        handler = get_type_handler(field_type)
        handler.pack_single_value(field_value, field_data, field_info)
    except ValueError as err:
        # Handler not found, fallback to old behavior
        raise ParamError(message=f"Unsupported data type: {field_type}") from err


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
        # Use type handler for all types
        try:
            handler = get_type_handler(entity_type)
            handler.pack_to_field_data(entity_values, field_data, field_info)
        except ValueError as err:
            # Handler not found, fallback to old behavior
            raise ParamError(message=f"Unsupported data type: {entity_type}") from err
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


def extract_array_row_data(field_data: Any, index: int):
    array = field_data.scalars.array_data.data[index]
    element_type = field_data.scalars.array_data.element_type

    # Use type handler to extract data from ScalarField
    try:
        handler = get_type_handler(element_type)
        return handler.extract_from_scalar_field(array)
    except (ValueError, NotImplementedError):
        # Handler not found or not implemented
        return None


def extract_row_data_from_fields_data_v2(
    field_data: Any,
    entity_rows: List[Dict],
) -> bool:
    """Extract row data from FieldData using unified handler interface."""
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

    # Use handler to determine if field is lazy and extract data
    try:
        handler = get_type_handler(field_data.type)
    except ValueError as err:
        # Handler not found, check special cases
        if field_data.type == DataType.STRING:
            raise MilvusException(message="Not support string yet") from err
        # Unknown type, assume lazy
        return True

    # Check if field should be lazy loaded
    if handler.is_lazy_field():
        return True

    # Scalar types: extract immediately using handler
    data = handler.get_raw_data(field_data)
    assign_scalar(data)
    return False


def extract_vector_array_row_data(field_data: Any, index: int):
    """Extract vector array row data using type handler."""
    array = field_data.vectors.vector_array.data[index]
    element_type = field_data.vectors.vector_array.element_type

    try:
        handler = get_type_handler(element_type)
    except ValueError as err:
        msg = f"Unimplemented type: {element_type} for vector array extraction"
        raise ParamError(message=msg) from err

    # Create a temporary FieldData-like object for extraction
    # Vector array elements have a different structure, so we need to adapt
    if element_type == DataType.FLOAT_VECTOR:
        # Create temporary FieldData with float_vector data
        temp_field_data = schema_types.FieldData(
            type=element_type,
            field_name="temp",
        )
        temp_field_data.vectors.dim = len(array.float_vector.data)
        temp_field_data.vectors.float_vector.data.extend(array.float_vector.data)
        row_data = {}
        handler.extract_from_field_data(temp_field_data, 0, row_data)
        return row_data.get("temp", [])

    if element_type in (
        DataType.FLOAT16_VECTOR,
        DataType.BFLOAT16_VECTOR,
        DataType.INT8_VECTOR,
        DataType.BINARY_VECTOR,
    ):
        # For bytes-based vectors, create temporary FieldData
        temp_field_data = schema_types.FieldData(
            type=element_type,
            field_name="temp",
        )
        # Get dimension from the array element
        dim = array.dim
        temp_field_data.vectors.dim = dim

        # Set the appropriate vector field
        if element_type == DataType.FLOAT16_VECTOR:
            temp_field_data.vectors.float16_vector = array.float16_vector
        elif element_type == DataType.BFLOAT16_VECTOR:
            temp_field_data.vectors.bfloat16_vector = array.bfloat16_vector
        elif element_type == DataType.INT8_VECTOR:
            temp_field_data.vectors.int8_vector = array.int8_vector
        elif element_type == DataType.BINARY_VECTOR:
            temp_field_data.vectors.binary_vector = array.binary_vector

        row_data = {}
        handler.extract_from_field_data(temp_field_data, 0, row_data)
        result = row_data.get("temp")
        # BINARY_VECTOR returns bytes, but original code returns [bytes]
        if element_type == DataType.BINARY_VECTOR and isinstance(result, bytes):
            return [result]
        return result if isinstance(result, list) else list(result) if result is not None else []

    msg = f"Unimplemented type: {element_type} for vector array extraction"
    raise ParamError(message=msg)


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

    def check_append(field_data: Any, row_data: Dict):
        if field_data.type == DataType.STRING:
            raise MilvusException(message="Not support string yet")

        # Use type handler for extraction
        try:
            handler = get_type_handler(field_data.type)
            context = {
                "dynamic_fields": dynamic_fields,  # For JSON handler
            }
            handler.extract_from_field_data(field_data, index, row_data, context)
            # Handle GEOMETRY special case (uses entity_row_data instead of row_data)
            if field_data.type == DataType.GEOMETRY and field_data.field_name in row_data:
                entity_row_data[field_data.field_name] = row_data.pop(field_data.field_name)
        except ValueError as err:
            # Handler not found - this should not happen as all types have handlers
            raise MilvusException(message=f"No handler for type: {field_data.type}") from err

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


def _extract_vector_from_array_element(
    vector_data: Any, element_type: DataType, struct_idx: int, dim: int
) -> Optional[List[Any]]:
    """Extract a single vector from vector array element using handler."""
    try:
        handler = get_type_handler(element_type)
    except ValueError:
        return None

    # Get bytes per vector for calculation
    if element_type == DataType.FLOAT_VECTOR:
        float_data = vector_data.float_vector.data
        vec_start = struct_idx * dim
        vec_end = vec_start + dim
        if vec_end <= len(float_data):
            return list(float_data[vec_start:vec_end])
        return None

    # For bytes-based vectors, use handler's get_bytes_per_vector
    bytes_per_vector = handler.get_bytes_per_vector(dim)
    if bytes_per_vector > 0:
        vec_start = struct_idx * bytes_per_vector
        vec_end = vec_start + bytes_per_vector

        # Get raw data based on type
        if element_type == DataType.FLOAT16_VECTOR:
            byte_data = vector_data.float16_vector
            if vec_end <= len(byte_data):
                vec_bytes = byte_data[vec_start:vec_end]
                return list(np.frombuffer(vec_bytes, dtype=np.float16))
        elif element_type == DataType.BFLOAT16_VECTOR:
            byte_data = vector_data.bfloat16_vector
            if vec_end <= len(byte_data):
                vec_bytes = byte_data[vec_start:vec_end]
                dtype = "bfloat16" if hasattr(np, "bfloat16") else np.uint16
                return list(np.frombuffer(vec_bytes, dtype=dtype))
        elif element_type == DataType.INT8_VECTOR:
            byte_data = vector_data.int8_vector
            if vec_end <= len(byte_data):
                vec_bytes = byte_data[vec_start:vec_end]
                return list(np.frombuffer(vec_bytes, dtype=np.int8))
        elif element_type == DataType.BINARY_VECTOR:
            byte_data = vector_data.binary_vector
            if vec_end <= len(byte_data):
                return [byte_data[vec_start:vec_end]]

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

                try:
                    handler = get_type_handler(element_type)
                    # Get raw data and calculate bytes per vector
                    if element_type == DataType.FLOAT_VECTOR:
                        raw_data = vector_data.float_vector.data
                        num_structs = len(raw_data) // dim
                    else:
                        # For bytes-based vectors, use handler's get_bytes_per_vector
                        bytes_per_vector = handler.get_bytes_per_vector(dim)
                        # Get raw data based on type
                        if element_type == DataType.FLOAT16_VECTOR:
                            raw_data = vector_data.float16_vector
                        elif element_type == DataType.BFLOAT16_VECTOR:
                            raw_data = vector_data.bfloat16_vector
                        elif element_type == DataType.INT8_VECTOR:
                            raw_data = vector_data.int8_vector
                        elif element_type == DataType.BINARY_VECTOR:
                            raw_data = vector_data.binary_vector
                        else:
                            raw_data = b""
                        num_structs = (
                            len(raw_data) // bytes_per_vector if bytes_per_vector > 0 else 0
                        )
                except ValueError:
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

                        # Use handler to extract vector
                        extracted_vector = _extract_vector_from_array_element(
                            vector_data, element_type, struct_idx, dim
                        )
                        struct_obj[sub_field_name] = extracted_vector

            # All struct sub-fields should be either ARRAY or ARRAY_OF_VECTOR
            # Any other type indicates a bug in the data conversion logic
            else:
                raise ParamError(
                    message=f"Unexpected field type {sub_field.type} for struct sub-field {sub_field_name}. "
                    f"Struct fields must be ARRAY or ARRAY_OF_VECTOR internally."
                )

        struct_array.append(struct_obj)

    return struct_array
