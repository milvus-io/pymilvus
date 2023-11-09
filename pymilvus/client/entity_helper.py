from typing import Any, Dict, List, Optional

import numpy as np
import ujson

from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.grpc_gen import schema_pb2 as schema_types
from pymilvus.settings import Config

from .types import DataType

CHECK_STR_ARRAY = True


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


def check_str_arr(str_arr: Any, max_len: int):
    for s in str_arr:
        if not isinstance(s, str):
            raise ParamError(message=f"expect string input, got: {type(s)}")
        if len(s) > max_len:
            raise ParamError(
                message=f"invalid input, length of string exceeds max length. "
                f"length: {len(s)}, max length: {max_len}"
            )


def convert_to_str_array(orig_str_arr: Any, field_info: Any, check: bool = True):
    arr = []
    if Config.EncodeProtocol.lower() != "utf-8".lower():
        for s in orig_str_arr:
            arr.append(s.encode(Config.EncodeProtocol))
    else:
        arr = orig_str_arr
    max_len = int(get_max_len_of_var_char(field_info))
    if check:
        check_str_arr(arr, max_len)
    return arr


def entity_to_str_arr(entity: Any, field_info: Any, check: bool = True):
    return convert_to_str_array(entity.get("values", []), field_info, check=check)


def convert_to_json(obj: object):
    return ujson.dumps(obj, ensure_ascii=False).encode(Config.EncodeProtocol)


def convert_to_json_arr(objs: List[object]):
    arr = []
    for obj in objs:
        arr.append(ujson.dumps(obj, ensure_ascii=False).encode(Config.EncodeProtocol))
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


def pack_field_value_to_field_data(field_value: Any, field_data: Any, field_info: Any):
    field_type = field_data.type
    if field_type == DataType.BOOL:
        field_data.scalars.bool_data.data.append(field_value)
    elif field_type in (DataType.INT8, DataType.INT16, DataType.INT32):
        field_data.scalars.int_data.data.append(field_value)
    elif field_type == DataType.INT64:
        field_data.scalars.long_data.data.append(field_value)
    elif field_type == DataType.FLOAT:
        field_data.scalars.float_data.data.append(field_value)
    elif field_type == DataType.DOUBLE:
        field_data.scalars.double_data.data.append(field_value)
    elif field_type == DataType.FLOAT_VECTOR:
        field_data.vectors.dim = len(field_value)
        field_data.vectors.float_vector.data.extend(field_value)
    elif field_type == DataType.BINARY_VECTOR:
        field_data.vectors.dim = len(field_value) * 8
        field_data.vectors.binary_vector += bytes(field_value)
    elif field_type == DataType.VARCHAR:
        field_data.scalars.string_data.data.append(
            convert_to_str_array(field_value, field_info, CHECK_STR_ARRAY)
        )
    elif field_type == DataType.JSON:
        field_data.scalars.json_data.data.append(convert_to_json(field_value))
    elif field_type == DataType.ARRAY:
        field_data.scalars.array_data.data.append(convert_to_array(field_value, field_info))
    else:
        raise ParamError(message=f"UnSupported data type: {field_type}")


# TODO: refactor here.
def entity_to_field_data(entity: Any, field_info: Any):
    field_data = schema_types.FieldData()

    entity_type = entity.get("type")
    field_data.field_name = entity.get("name")
    field_data.type = entity_type_to_dtype(entity_type)

    if entity_type == DataType.BOOL:
        field_data.scalars.bool_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.INT8, DataType.INT16, DataType.INT32):
        field_data.scalars.int_data.data.extend(entity.get("values"))
    elif entity_type == DataType.INT64:
        field_data.scalars.long_data.data.extend(entity.get("values"))
    elif entity_type == DataType.FLOAT:
        field_data.scalars.float_data.data.extend(entity.get("values"))
    elif entity_type == DataType.DOUBLE:
        field_data.scalars.double_data.data.extend(entity.get("values"))
    elif entity_type == DataType.FLOAT_VECTOR:
        field_data.vectors.dim = len(entity.get("values")[0])
        all_floats = [f for vector in entity.get("values") for f in vector]
        field_data.vectors.float_vector.data.extend(all_floats)
    elif entity_type == DataType.BINARY_VECTOR:
        field_data.vectors.dim = len(entity.get("values")[0]) * 8
        field_data.vectors.binary_vector = b"".join(entity.get("values"))
    elif entity_type == DataType.VARCHAR:
        field_data.scalars.string_data.data.extend(
            entity_to_str_arr(entity, field_info, CHECK_STR_ARRAY)
        )
    elif entity_type == DataType.JSON:
        field_data.scalars.json_data.data.extend(entity_to_json_arr(entity))
    elif entity_type == DataType.ARRAY:
        field_data.scalars.array_data.data.extend(entity_to_array_arr(entity, field_info))
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

            tmp_dict = {k: v for k, v in json_dict.items() if k in dynamic_fields}
            entity_row_data.update(tmp_dict)
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

    for field_data in fields_data:
        check_append(field_data)

    return entity_row_data
