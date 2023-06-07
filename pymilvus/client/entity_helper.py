import ujson
import numpy as np

from ..grpc_gen import schema_pb2 as schema_types
from .types import DataType
from ..exceptions import ParamError, MilvusException
from ..settings import Config


def entity_type_to_dtype(entity_type):
    if isinstance(entity_type, int):
        return entity_type
    if isinstance(entity_type, str):
        # case sensitive
        return schema_types.DataType.Value(entity_type)
    raise ParamError(message=f"invalid entity type: {entity_type}")


def get_max_len_of_var_char(field_info) -> int:
    k = Config.MaxVarCharLengthKey
    v = Config.MaxVarCharLength
    return field_info.get("params", {}).get(k, v)


def check_str_arr(str_arr, max_len):
    for s in str_arr:
        if not isinstance(s, str):
            raise ParamError(message=f"expect string input, got: {type(s)}")
        if len(s) > max_len:
            raise ParamError(message=f"invalid input, length of string exceeds max length. length: {len(s)}, "
                                     f"max length: {max_len}")


def convert_to_str_array(orig_str_arr, field_info, check=True):
    arr = []
    if Config.EncodeProtocol.lower() != 'utf-8'.lower():
        for s in orig_str_arr:
            arr.append(s.encode(Config.EncodeProtocol))
    else:
        arr = orig_str_arr
    max_len = int(get_max_len_of_var_char(field_info))
    if check:
        check_str_arr(arr, max_len)
    return arr


def entity_to_str_arr(entity, field_info, check=True):
    return convert_to_str_array(entity.get("values", []), field_info, check=check)


def convert_to_json(obj):
    return ujson.dumps(obj).encode(Config.EncodeProtocol)


def convert_to_json_arr(objs):
    arr = []
    for obj in objs:
        arr.append(ujson.dumps(obj).encode(Config.EncodeProtocol))
    return arr


def entity_to_json_arr(entity):
    return convert_to_json_arr(entity.get("values", []))


def pack_field_value_to_field_data(field_value, field_data, field_info):
    field_type = field_data.type
    if field_type in (DataType.BOOL,):
        field_data.scalars.bool_data.data.append(field_value)
    elif field_type in (DataType.INT8,):
        field_data.scalars.int_data.data.append(field_value)
    elif field_type in (DataType.INT16,):
        field_data.scalars.int_data.data.append(field_value)
    elif field_type in (DataType.INT32,):
        field_data.scalars.int_data.data.append(field_value)
    elif field_type in (DataType.INT64,):
        field_data.scalars.long_data.data.append(field_value)
    elif field_type in (DataType.FLOAT,):
        field_data.scalars.float_data.data.append(field_value)
    elif field_type in (DataType.DOUBLE,):
        field_data.scalars.double_data.data.append(field_value)
    elif field_type in (DataType.FLOAT_VECTOR,):
        field_data.vectors.dim = len(field_value)
        field_data.vectors.float_vector.data.extend(field_value)
    elif field_type in (DataType.BINARY_VECTOR,):
        field_data.vectors.dim = len(field_value) * 8
        field_data.vectors.binary_vector += bytes(field_value)
    elif field_type in (DataType.VARCHAR,):
        field_data.scalars.string_data.data.append(
            convert_to_str_array(field_value, field_info, True))
    elif field_type in (DataType.JSON,):
        field_data.scalars.json_data.data.append(convert_to_json(field_value))
    else:
        raise ParamError(message=f"UnSupported data type: {field_type}")


# TODO: refactor here.
def entity_to_field_data(entity, field_info):
    field_data = schema_types.FieldData()

    entity_type = entity.get("type")
    field_data.field_name = entity.get("name")
    field_data.type = entity_type_to_dtype(entity_type)

    if entity_type in (DataType.BOOL,):
        field_data.scalars.bool_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.INT8,):
        field_data.scalars.int_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.INT16,):
        field_data.scalars.int_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.INT32,):
        field_data.scalars.int_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.INT64,):
        field_data.scalars.long_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.FLOAT,):
        field_data.scalars.float_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.DOUBLE,):
        field_data.scalars.double_data.data.extend(entity.get("values"))
    elif entity_type in (DataType.FLOAT_VECTOR,):
        field_data.vectors.dim = len(entity.get("values")[0])
        all_floats = [f for vector in entity.get("values") for f in vector]
        field_data.vectors.float_vector.data.extend(all_floats)
    elif entity_type in (DataType.BINARY_VECTOR,):
        field_data.vectors.dim = len(entity.get("values")[0]) * 8
        field_data.vectors.binary_vector = b''.join(entity.get("values"))
    elif entity_type in (DataType.VARCHAR,):
        field_data.scalars.string_data.data.extend(entity_to_str_arr(entity, field_info, True))
    elif entity_type in (DataType.JSON,):
        field_data.scalars.json_data.data.extend(entity_to_json_arr(entity))
    else:
        raise ParamError(message=f"UnSupported data type: {entity_type}")

    return field_data


def extract_dynamic_field_from_result(raw):
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


# pylint: disable=R1702 (too-many-nested-blocks)
def extract_row_data_from_fields_data(fields_data, index, dynamic_output_fields=None):
    if not fields_data:
        return {}

    entity_row_data = {}
    dynamic_fields = dynamic_output_fields or set()
    for field_data in fields_data:
        if field_data.type == DataType.BOOL:
            if len(field_data.scalars.bool_data.data) >= index:
                entity_row_data[field_data.field_name] = field_data.scalars.bool_data.data[index]
        elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
            if len(field_data.scalars.int_data.data) >= index:
                entity_row_data[field_data.field_name] = field_data.scalars.int_data.data[index]
        elif field_data.type == DataType.INT64:
            if len(field_data.scalars.long_data.data) >= index:
                entity_row_data[field_data.field_name] = field_data.scalars.long_data.data[index]
        elif field_data.type == DataType.FLOAT:
            if len(field_data.scalars.float_data.data) >= index:
                entity_row_data[field_data.field_name] = np.single(field_data.scalars.float_data.data[index])
        elif field_data.type == DataType.DOUBLE:
            if len(field_data.scalars.double_data.data) >= index:
                entity_row_data[field_data.field_name] = field_data.scalars.double_data.data[index]
        elif field_data.type == DataType.VARCHAR:
            if len(field_data.scalars.string_data.data) >= index:
                entity_row_data[field_data.field_name] = field_data.scalars.string_data.data[index]
        elif field_data.type == DataType.STRING:
            raise MilvusException(message="Not support string yet")
            # result[field_data.field_name] = field_data.scalars.string_data.data[index]
        elif field_data.type == DataType.JSON:
            if len(field_data.scalars.json_data.data) >= index:
                json_value = field_data.scalars.json_data.data[index]
                json_dict = ujson.loads(json_value)
                if field_data.is_dynamic:
                    if dynamic_fields:
                        for key in json_dict:
                            if key in dynamic_fields:
                                entity_row_data[key] = json_dict[key]
                    else:
                        entity_row_data.update(json_dict)
                    continue
                entity_row_data[field_data.field_name] = json_dict
        elif field_data.type == DataType.FLOAT_VECTOR:
            dim = field_data.vectors.dim
            if len(field_data.vectors.float_vector.data) >= index * dim:
                start_pos = index * dim
                end_pos = index * dim + dim
                entity_row_data[field_data.field_name] = [np.single(x) for x in
                                                          field_data.vectors.float_vector.data[
                                                          start_pos:end_pos]]
        elif field_data.type == DataType.BINARY_VECTOR:
            dim = field_data.vectors.dim
            if len(field_data.vectors.binary_vector) >= index * (dim // 8):
                start_pos = index * (dim // 8)
                end_pos = (index + 1) * (dim // 8)
                entity_row_data[field_data.field_name] = [
                    field_data.vectors.binary_vector[start_pos:end_pos]]

    return entity_row_data
