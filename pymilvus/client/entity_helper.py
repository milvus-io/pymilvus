import math
import struct
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import ujson

from pymilvus.exceptions import (
    DataNotMatchException,
    ExceptionsMessage,
    MilvusException,
    ParamError,
)
from pymilvus.grpc_gen import schema_pb2 as schema_types
from pymilvus.settings import Config

from .types import DataType
from .utils import SciPyHelper, SparseMatrixInputType, SparseRowOutputType

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
            pairs = item.items() if isinstance(item, dict) else item
            # each row must be a non-empty list of Tuple[int, float]
            if len(pairs) == 0:
                return False
            for pair in pairs:
                if len(pair) != 2 or not is_int_type(pair[0]) or not is_float_type(pair[1]):
                    return False
    except Exception:
        return False
    return True


# parses plain bytes to a sparse float vector(SparseRowOutputType)
def sparse_parse_single_row(data: bytes) -> SparseRowOutputType:
    if len(data) % 8 != 0:
        raise ParamError(message=f"The length of data must be a multiple of 8, got {len(data)}")

    return {
        struct.unpack("I", data[i : i + 4])[0]: struct.unpack("f", data[i + 4 : i + 8])[0]
        for i in range(0, len(data), 8)
    }


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
                dim = max(dim, indices[-1] + 1)
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
                    message=f"field ({field_info['name']}) expect string input, got: {type(s)}"
                )
            if len(s) > max_len:
                raise ParamError(
                    message=f"invalid input of field ({field_info['name']}), "
                    f"length of string exceeds max length. length: {len(s)}, max length: {max_len}"
                )
    return arr


def entity_to_str_arr(entity: Any, field_info: Any, check: bool = True):
    return convert_to_str_array(entity.get("values", []), field_info, check=check)


def convert_to_json(obj: object):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise DataNotMatchException(message=ExceptionsMessage.JSONKeyMustBeStr)
            if isinstance(v, np.ndarray):
                obj[k] = v.tolist()
    return ujson.dumps(obj, ensure_ascii=False).encode(Config.EncodeProtocol)


def convert_to_json_arr(objs: List[object]):
    arr = []
    for obj in objs:
        arr.append(convert_to_json(obj))
    return arr


def entity_to_json_arr(entity: Dict):
    return convert_to_json_arr(entity.get("values", []))


def convert_to_array_arr(objs: List[Any], field_info: Any):
    return [convert_to_array(obj, field_info) for obj in objs]


def convert_to_array(obj: List[Any], field_info: Any):
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
        message=f"UnSupported element type: {element_type} for Array field: {field_info.get('name')}"
    )


def entity_to_array_arr(entity: List[Any], field_info: Any):
    return convert_to_array_arr(entity.get("values", []), field_info)


def pack_field_value_to_field_data(
    field_value: Any, field_data: schema_types.FieldData, field_info: Any
):
    field_type = field_data.type
    field_name = field_info["name"]
    if field_type == DataType.BOOL:
        try:
            field_data.scalars.bool_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "bool", type(field_value))
            ) from e
    elif field_type in (DataType.INT8, DataType.INT16, DataType.INT32):
        try:
            field_data.scalars.int_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int", type(field_value))
            ) from e
    elif field_type == DataType.INT64:
        try:
            field_data.scalars.long_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int64", type(field_value))
            ) from e
    elif field_type == DataType.FLOAT:
        try:
            field_data.scalars.float_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float", type(field_value))
            ) from e
    elif field_type == DataType.DOUBLE:
        try:
            field_data.scalars.double_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "double", type(field_value))
            ) from e
    elif field_type == DataType.FLOAT_VECTOR:
        try:
            f_value = field_value
            if isinstance(field_value, np.ndarray):
                if field_value.dtype not in ("float32", "float64"):
                    raise ParamError(
                        message="invalid input for float32 vector, expect np.ndarray with dtype=float32"
                    )
                f_value = field_value.tolist()

            field_data.vectors.dim = len(f_value)
            field_data.vectors.float_vector.data.extend(f_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float_vector", type(field_value))
            ) from e
    elif field_type == DataType.BINARY_VECTOR:
        try:
            field_data.vectors.dim = len(field_value) * 8
            field_data.vectors.binary_vector += bytes(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "binary_vector", type(field_value))
            ) from e
    elif field_type == DataType.FLOAT16_VECTOR:
        try:
            if isinstance(field_value, bytes):
                v_bytes = field_value
            elif isinstance(field_value, np.ndarray):
                if field_value.dtype != "float16":
                    raise ParamError(
                        message="invalid input for float16 vector, expect np.ndarray with dtype=float16"
                    )
                v_bytes = field_value.view(np.uint8).tobytes()
            else:
                raise ParamError(
                    message="invalid input type for float16 vector, expect np.ndarray with dtype=float16"
                )

            field_data.vectors.dim = len(v_bytes) // 2
            field_data.vectors.float16_vector += v_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float16_vector", type(field_value))
            ) from e
    elif field_type == DataType.BFLOAT16_VECTOR:
        try:
            if isinstance(field_value, bytes):
                v_bytes = field_value
            elif isinstance(field_value, np.ndarray):
                if field_value.dtype != "bfloat16":
                    raise ParamError(
                        message="invalid input for bfloat16 vector, expect np.ndarray with dtype=bfloat16"
                    )
                v_bytes = field_value.view(np.uint8).tobytes()
            else:
                raise ParamError(
                    message="invalid input type for bfloat16 vector, expect np.ndarray with dtype=bfloat16"
                )

            field_data.vectors.dim = len(v_bytes) // 2
            field_data.vectors.bfloat16_vector += v_bytes
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "bfloat16_vector", type(field_value))
            ) from e
    elif field_type == DataType.SPARSE_FLOAT_VECTOR:
        try:
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
            ) from e
    elif field_type == DataType.VARCHAR:
        try:
            field_data.scalars.string_data.data.append(
                convert_to_str_array(field_value, field_info, CHECK_STR_ARRAY)
            )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "varchar", type(field_value))
            ) from e
    elif field_type == DataType.JSON:
        try:
            field_data.scalars.json_data.data.append(convert_to_json(field_value))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "json", type(field_value))
            ) from e
    elif field_type == DataType.ARRAY:
        try:
            field_data.scalars.array_data.data.append(convert_to_array(field_value, field_info))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "array", type(field_value))
            ) from e
    else:
        raise ParamError(message=f"UnSupported data type: {field_type}")


# TODO: refactor here.
def entity_to_field_data(entity: Any, field_info: Any):
    field_data = schema_types.FieldData()

    entity_type = entity.get("type")
    field_name = entity.get("name")
    field_data.field_name = field_name
    field_data.type = entity_type_to_dtype(entity_type)

    if entity_type == DataType.BOOL:
        try:
            field_data.scalars.bool_data.data.extend(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "bool", type(entity.get("values")[0]))
            ) from e
    elif entity_type in (DataType.INT8, DataType.INT16, DataType.INT32):
        try:
            field_data.scalars.int_data.data.extend(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.INT64:
        try:
            field_data.scalars.long_data.data.extend(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int64", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.FLOAT:
        try:
            field_data.scalars.float_data.data.extend(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.DOUBLE:
        try:
            field_data.scalars.double_data.data.extend(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "double", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.FLOAT_VECTOR:
        try:
            field_data.vectors.dim = len(entity.get("values")[0])
            all_floats = [f for vector in entity.get("values") for f in vector]
            field_data.vectors.float_vector.data.extend(all_floats)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float_vector", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.BINARY_VECTOR:
        try:
            field_data.vectors.dim = len(entity.get("values")[0]) * 8
            field_data.vectors.binary_vector = b"".join(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "binary_vector", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.FLOAT16_VECTOR:
        try:
            field_data.vectors.dim = len(entity.get("values")[0]) // 2
            field_data.vectors.float16_vector = b"".join(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float16_vector", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.BFLOAT16_VECTOR:
        try:
            field_data.vectors.dim = len(entity.get("values")[0]) // 2
            field_data.vectors.bfloat16_vector = b"".join(entity.get("values"))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "bfloat16_vector", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.VARCHAR:
        try:
            field_data.scalars.string_data.data.extend(
                entity_to_str_arr(entity, field_info, CHECK_STR_ARRAY)
            )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "varchar", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.JSON:
        try:
            field_data.scalars.json_data.data.extend(entity_to_json_arr(entity))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "json", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.ARRAY:
        try:
            field_data.scalars.array_data.data.extend(entity_to_array_arr(entity, field_info))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "array", type(entity.get("values")[0]))
            ) from e
    elif entity_type == DataType.SPARSE_FLOAT_VECTOR:
        try:
            field_data.vectors.sparse_float_vector.CopyFrom(
                sparse_rows_to_proto(entity.get("values"))
            )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "sparse_float_vector", type(entity.get("values")[0]))
            ) from e
    else:
        raise ParamError(message=f"UnSupported data type: {entity_type}")

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
    row = []
    if field_data.scalars.array_data.element_type == DataType.INT64:
        row.extend(array.long_data.data)
        return row
    if field_data.scalars.array_data.element_type == DataType.BOOL:
        row.extend(array.bool_data.data)
        return row
    if field_data.scalars.array_data.element_type in (
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
    ):
        row.extend(array.int_data.data)
        return row
    if field_data.scalars.array_data.element_type == DataType.FLOAT:
        row.extend(array.float_data.data)
        return row
    if field_data.scalars.array_data.element_type == DataType.DOUBLE:
        row.extend(array.double_data.data)
        return row
    if field_data.scalars.array_data.element_type in (
        DataType.STRING,
        DataType.VARCHAR,
    ):
        row.extend(array.string_data.data)
        return row
    return row


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

    def check_append(field_data: Any):
        if field_data.type == DataType.STRING:
            raise MilvusException(message="Not support string yet")

        if field_data.type == DataType.BOOL and len(field_data.scalars.bool_data.data) >= index:
            entity_row_data[field_data.field_name] = field_data.scalars.bool_data.data[index]
            return

        if (
            field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32)
            and len(field_data.scalars.int_data.data) >= index
        ):
            entity_row_data[field_data.field_name] = field_data.scalars.int_data.data[index]
            return

        if field_data.type == DataType.INT64 and len(field_data.scalars.long_data.data) >= index:
            entity_row_data[field_data.field_name] = field_data.scalars.long_data.data[index]
            return

        if field_data.type == DataType.FLOAT and len(field_data.scalars.float_data.data) >= index:
            entity_row_data[field_data.field_name] = np.single(
                field_data.scalars.float_data.data[index]
            )
            return

        if field_data.type == DataType.DOUBLE and len(field_data.scalars.double_data.data) >= index:
            entity_row_data[field_data.field_name] = field_data.scalars.double_data.data[index]
            return

        if (
            field_data.type == DataType.VARCHAR
            and len(field_data.scalars.string_data.data) >= index
        ):
            entity_row_data[field_data.field_name] = field_data.scalars.string_data.data[index]
            return

        if field_data.type == DataType.JSON and len(field_data.scalars.json_data.data) >= index:
            json_dict = ujson.loads(field_data.scalars.json_data.data[index])

            if not field_data.is_dynamic:
                entity_row_data[field_data.field_name] = json_dict
                return

            if not dynamic_fields:
                entity_row_data.update(json_dict)
                return

            entity_row_data.update({k: v for k, v in json_dict.items() if k in dynamic_fields})
            return
        if field_data.type == DataType.ARRAY and len(field_data.scalars.array_data.data) >= index:
            entity_row_data[field_data.field_name] = extract_array_row_data(field_data, index)

        if field_data.type == DataType.FLOAT_VECTOR:
            dim = field_data.vectors.dim
            if len(field_data.vectors.float_vector.data) >= index * dim:
                start_pos, end_pos = index * dim, (index + 1) * dim
                entity_row_data[field_data.field_name] = [
                    np.single(x) for x in field_data.vectors.float_vector.data[start_pos:end_pos]
                ]
        elif field_data.type == DataType.BINARY_VECTOR:
            dim = field_data.vectors.dim
            if len(field_data.vectors.binary_vector) >= index * (dim // 8):
                start_pos, end_pos = index * (dim // 8), (index + 1) * (dim // 8)
                entity_row_data[field_data.field_name] = [
                    field_data.vectors.binary_vector[start_pos:end_pos]
                ]
        elif field_data.type == DataType.BFLOAT16_VECTOR:
            dim = field_data.vectors.dim
            if len(field_data.vectors.bfloat16_vector) >= index * (dim * 2):
                start_pos, end_pos = index * (dim * 2), (index + 1) * (dim * 2)
                entity_row_data[field_data.field_name] = [
                    field_data.vectors.bfloat16_vector[start_pos:end_pos]
                ]
        elif field_data.type == DataType.FLOAT16_VECTOR:
            dim = field_data.vectors.dim
            if len(field_data.vectors.float16_vector) >= index * (dim * 2):
                start_pos, end_pos = index * (dim * 2), (index + 1) * (dim * 2)
                entity_row_data[field_data.field_name] = [
                    field_data.vectors.float16_vector[start_pos:end_pos]
                ]
        elif field_data.type == DataType.SPARSE_FLOAT_VECTOR:
            entity_row_data[field_data.field_name] = sparse_parse_single_row(
                field_data.vectors.sparse_float_vector.contents[index]
            )

    for field_data in fields_data:
        check_append(field_data)

    return entity_row_data
