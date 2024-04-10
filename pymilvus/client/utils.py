import datetime
from datetime import timedelta
from typing import Any, List, Optional, Union

import ujson

from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.grpc_gen.common_pb2 import Status

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
    "GPU_CAGRA",
    "GPU_BRUTE_FORCE",
]

valid_index_params_keys = [
    "nlist",
    "m",
    "nbits",
    "M",
    "efConstruction",
    "PQM",
    "n_trees",
    "intermediate_graph_degree",
    "graph_degree",
    "build_algo",
    "cache_dataset_on_device",
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


def check_status(status: Status):
    if status.code != 0 or status.error_code != 0:
        raise MilvusException(status.code, status.reason, status.error_code)


def is_successful(status: Status):
    return status.code == 0 and status.error_code == 0


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
        if field_data.vectors.HasField("bfloat16_vector") or field_data.vectors.HasField(
            "float16_vector"
        ):
            total_len = (
                len(field_data.vectors.bfloat16_vector)
                if field_data.vectors.HasField("bfloat16_vector")
                else len(field_data.vectors.float16_vector)
            )
            data_wide_in_bytes = 2
            if total_len % (dim * data_wide_in_bytes) != 0:
                raise MilvusException(
                    message=f"Invalid bfloat16 or float16 vector length: total_len={total_len}, dim={dim}"
                )
            return int(total_len / (dim * data_wide_in_bytes))
        if field_data.vectors.HasField("sparse_float_vector"):
            return len(field_data.vectors.sparse_float_vector.contents)

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

    # though impossible from sdk
    if primary_key_loc is None:
        raise ParamError(message="primary key not found")

    return location, primary_key_loc, auto_id_loc


def traverse_info(fields_info: Any):
    location, primary_key_loc, auto_id_loc = {}, None, None
    for i, field in enumerate(fields_info):
        if field.get("is_primary", False):
            primary_key_loc = i

        if field.get("auto_id", False):
            auto_id_loc = i
            continue
        location[field["name"]] = i

    return location, primary_key_loc, auto_id_loc


def get_server_type(host: str):
    return ZILLIZ if (isinstance(host, str) and "zilliz" in host.lower()) else MILVUS


def dumps(v: Union[dict, str]) -> str:
    return ujson.dumps(v) if isinstance(v, dict) else str(v)
