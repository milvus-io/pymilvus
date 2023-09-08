import datetime
from datetime import timedelta
from typing import Any, List, Optional, Union

from pymilvus.exceptions import MilvusException, ParamError

from .constants import LOGICAL_BITS, LOGICAL_BITS_MASK
from .types import DataType

MILVUS = "milvus"
ZILLIZ = "zilliz"

valid_index_types = [
    "GPU_IVF_FLAT",
    "GPU_IVF_PQ",
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_PQ",
    "HNSW",
    "BIN_FLAT",
    "BIN_IVF_FLAT",
    "DISKANN",
    "AUTOINDEX",
]

valid_index_params_keys = [
    "nlist",
    "m",
    "nbits",
    "M",
    "efConstruction",
    "PQM",
    "n_trees",
]

valid_binary_index_types = [
    "BIN_FLAT",
    "BIN_IVF_FLAT",
]

valid_binary_metric_types = [
    "JACCARD",
    "HAMMING",
    "TANIMOTO",
    "SUBSTRUCTURE",
    "SUPERSTRUCTURE",
]


def hybridts_to_unixtime(ts: int):
    physical = ts >> LOGICAL_BITS
    return physical / 1000.0


def mkts_from_hybridts(
    hybridts: int,
    milliseconds: Union[float] = 0.0,
    delta: Optional[timedelta] = None,
) -> int:
    if not isinstance(milliseconds, (int, float)):
        raise MilvusException(message="parameter milliseconds should be type of int or float")

    if isinstance(delta, datetime.timedelta):
        milliseconds += delta.microseconds / 1000.0
    elif delta is not None:
        raise MilvusException(message="parameter delta should be type of datetime.timedelta")

    if not isinstance(hybridts, int):
        raise MilvusException(message="parameter hybridts should be type of int")

    logical = hybridts & LOGICAL_BITS_MASK
    physical = hybridts >> LOGICAL_BITS

    return int((int(physical + milliseconds) << LOGICAL_BITS) + logical)


def mkts_from_unixtime(
    epoch: Union[float],
    milliseconds: Union[float] = 0.0,
    delta: Optional[timedelta] = None,
) -> int:
    if not isinstance(epoch, (int, float)):
        raise MilvusException(message="parameter epoch should be type of int or float")

    if not isinstance(milliseconds, (int, float)):
        raise MilvusException(message="parameter milliseconds should be type of int or float")

    if isinstance(delta, datetime.timedelta):
        milliseconds += delta.microseconds / 1000.0
    elif delta is not None:
        raise MilvusException(message="parameter delta should be type of datetime.timedelta")

    epoch += milliseconds / 1000.0
    int_msecs = int(epoch * 1000 // 1)
    return int(int_msecs << LOGICAL_BITS)


def mkts_from_datetime(
    d_time: datetime.datetime,
    milliseconds: Union[float] = 0.0,
    delta: Optional[timedelta] = None,
) -> int:
    if not isinstance(d_time, datetime.datetime):
        raise MilvusException(message="parameter d_time should be type of datetime.datetime")

    return mkts_from_unixtime(d_time.timestamp(), milliseconds=milliseconds, delta=delta)


def check_invalid_binary_vector(entities: List) -> bool:
    for entity in entities:
        if entity["type"] == DataType.BINARY_VECTOR:
            if not isinstance(entity["values"], list) and len(entity["values"]) == 0:
                return False

            dim = len(entity["values"][0]) * 8
            if dim == 0:
                return False

            for values in entity["values"]:
                if len(values) * 8 != dim:
                    return False
                if not isinstance(values, bytes):
                    return False
    return True


def len_of(field_data: Any) -> int:
    if field_data.HasField("scalars"):
        if field_data.scalars.HasField("bool_data"):
            return len(field_data.scalars.bool_data.data)

        if field_data.scalars.HasField("int_data"):
            return len(field_data.scalars.int_data.data)

        if field_data.scalars.HasField("long_data"):
            return len(field_data.scalars.long_data.data)

        if field_data.scalars.HasField("float_data"):
            return len(field_data.scalars.float_data.data)

        if field_data.scalars.HasField("double_data"):
            return len(field_data.scalars.double_data.data)

        if field_data.scalars.HasField("string_data"):
            return len(field_data.scalars.string_data.data)

        if field_data.scalars.HasField("bytes_data"):
            return len(field_data.scalars.bytes_data.data)

        if field_data.scalars.HasField("json_data"):
            return len(field_data.scalars.json_data.data)

        if field_data.scalars.HasField("array_data"):
            return len(field_data.scalars.array_data.data)

        raise MilvusException(message="Unsupported scalar type")

    if field_data.HasField("vectors"):
        dim = field_data.vectors.dim
        if field_data.vectors.HasField("float_vector"):
            total_len = len(field_data.vectors.float_vector.data)
            if total_len % dim != 0:
                raise MilvusException(
                    message=f"Invalid vector length: total_len={total_len}, dim={dim}"
                )
            return int(total_len / dim)

        total_len = len(field_data.vectors.binary_vector)
        return int(total_len / (dim / 8))

    raise MilvusException(message="Unknown data type")


def traverse_rows_info(fields_info: Any, entities: List):
    location, primary_key_loc, auto_id_loc = {}, None, None

    for i, field in enumerate(fields_info):
        is_auto_id = False
        is_dynamic = False

        if field.get("auto_id", False):
            auto_id_loc = i
            is_auto_id = True

        if field.get("is_primary", False):
            primary_key_loc = i

        field_name = field["name"]
        location[field_name] = i
        field_type = field["type"]

        if field.get("is_dynamic", False):
            is_dynamic = True

        for j, entity in enumerate(entities):
            if is_auto_id:
                if field_name in entity:
                    raise ParamError(
                        message=f"auto id enabled, {field_name} shouldn't in entities[{j}]"
                    )
                continue

            if is_dynamic and field_name in entity:
                raise ParamError(
                    message=f"dynamic field enabled, {field_name} shouldn't in entities[{j}]"
                )

            value = entity.get(field_name, None)
            if value is None:
                raise ParamError(message=f"Field {field_name} don't match in entities[{j}]")

            if field_type in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                field_dim = field["params"]["dim"]
                entity_dim = len(value) if field_type == DataType.FLOAT_VECTOR else len(value) * 8

                if entity_dim != field_dim:
                    raise ParamError(
                        message=f"Collection field dim is {field_dim}"
                        f", but entities field dim is {entity_dim}"
                    )

    # though impossible from sdk
    if primary_key_loc is None:
        raise ParamError(message="primary key not found")

    return location, primary_key_loc, auto_id_loc


def traverse_info(fields_info: Any, entities: List):
    location, primary_key_loc, auto_id_loc = {}, None, None
    for i, field in enumerate(fields_info):
        if field.get("is_primary", False):
            primary_key_loc = i

        if field.get("auto_id", False):
            auto_id_loc = i
            continue

        match_flag = False
        field_name = field["name"]
        field_type = field["type"]

        for entity in entities:
            entity_name, entity_type = entity["name"], entity["type"]

            if field_name == entity_name:
                if field_type != entity_type:
                    raise ParamError(
                        message=f"Collection field type is {field_type}"
                        f", but entities field type is {entity_type}"
                    )

                entity_dim, field_dim = 0, 0
                if entity_type in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                    field_dim = field["params"]["dim"]
                    entity_dim = len(entity["values"][0])

                if entity_type in [DataType.FLOAT_VECTOR] and entity_dim != field_dim:
                    raise ParamError(
                        message=f"Collection field dim is {field_dim}"
                        f", but entities field dim is {entity_dim}"
                    )

                if entity_type in [DataType.BINARY_VECTOR] and entity_dim * 8 != field_dim:
                    raise ParamError(
                        message=f"Collection field dim is {field_dim}"
                        f", but entities field dim is {entity_dim * 8}"
                    )

                location[field["name"]] = i
                match_flag = True
                break

        if not match_flag:
            raise ParamError(message=f"Field {field['name']} don't match in entities")

    return location, primary_key_loc, auto_id_loc


def get_server_type(host: str):
    if host is None or not isinstance(host, str):
        return MILVUS
    splits = host.split(".")
    len_of_splits = len(splits)
    if (
        len_of_splits >= 2
        and (
            splits[len_of_splits - 2].lower() == "zilliz"
            or splits[len_of_splits - 2].lower() == "zillizcloud"
        )
        and splits[len_of_splits - 1].lower() == "com"
    ):
        return ZILLIZ
    return MILVUS
