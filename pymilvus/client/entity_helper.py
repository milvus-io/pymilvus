from ..grpc_gen import schema_pb2 as schema_types
from .types import DataType
from ..exceptions import ParamError
from .configs import DefaultConfigs


def entity_type_to_dtype(entity_type):
    if isinstance(entity_type, int):
        return entity_type
    if isinstance(entity_type, str):
        # case sensitive
        return schema_types.DataType.Value(entity_type)
    raise ParamError(message=f"invalid entity type: {entity_type}")


def get_max_len_of_var_char(field_info) -> int:
    k = DefaultConfigs.MaxVarCharLengthKey
    v = DefaultConfigs.MaxVarCharLength
    return field_info.get("params", {}).get(k, v)


def check_str_arr(str_arr, max_len):
    for s in str_arr:
        if not isinstance(s, str):
            raise ParamError(message=f"expect string input, got: {type(s)}")
        if len(s) >= max_len:
            raise ParamError(message=f"invalid input, length of string exceeds max length. length: {len(s)}, max length: {max_len}")


def entity_to_str_arr(entity, field_info, check=True):
    arr = []
    if DefaultConfigs.EncodeProtocol.lower() != 'utf-8'.lower():
        for s in entity.get("values"):
            arr.append(s.encode(DefaultConfigs.EncodeProtocol))
    else:
        arr = entity.get("values")
    max_len = int(get_max_len_of_var_char(field_info))
    if check:
        check_str_arr(arr, max_len)
    return arr


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
    else:
        raise ParamError(message=f"UnSupported data type: {entity_type}")

    return field_data
