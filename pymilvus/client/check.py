import sys
import datetime
from typing import Any, Union
from .exceptions import ParamError


def is_legal_host(host: Any) -> bool:
    return isinstance(host, str)


def is_legal_port(port: Any) -> bool:
    if isinstance(port, str):
        try:
            _port = int(port)
        except ValueError:
            return False
        else:
            return 0 <= _port < 65535

    return False


def is_legal_vector(array: Any) -> bool:
    if not array or \
            not isinstance(array, list) or \
            len(array) == 0:
        return False

    # for v in array:
    #     if not isinstance(v, float):
    #         return False

    return True


def is_legal_bin_vector(array: Any) -> bool:
    if not array or \
            not isinstance(array, bytes) or \
            len(array) == 0:
        return False

    return True


def is_legal_numpy_array(array: Any) -> bool:
    return not (array is None or array.size == 0)


# def is_legal_records(value):
#     param_error = ParamError(
#         'A vector must be a non-empty, 2-dimensional array and '
#         'must contain only elements with the float data type or the bytes data type.'
#     )
#
#     if isinstance(value, np.ndarray):
#         if not is_legal_numpy_array(value):
#             raise param_error
#
#         return True
#
#     if not isinstance(value, list) or len(value) == 0:
#         raise param_error
#
#     if isinstance(value[0], bytes):
#         check_func = is_legal_bin_vector
#     elif isinstance(value[0], list):
#         check_func = is_legal_vector
#     else:
#         raise param_error
#
#     _dim = len(value[0])
#     for record in value:
#         if not check_func(record):
#             raise param_error
#         if _dim != len(record):
#             raise ParamError('Whole vectors must have the same dimension')
#
#     return True


def int_or_str(item: Union[int, str]) -> str:
    if isinstance(item, int):
        return str(item)

    return item


def is_correct_date_str(param: str) -> bool:
    try:
        datetime.datetime.strptime(param, '%Y-%m-%d')
    except ValueError:
        return False

    return True


def is_legal_dimension(dim: Any) -> bool:
    return isinstance(dim, int)


def is_legal_index_size(index_size: Any) -> bool:
    return isinstance(index_size, int)


def is_legal_table_name(table_name: Any) -> bool:
    return table_name and isinstance(table_name, str)


def is_legal_field_name(field_name: Any) -> bool:
    return field_name and isinstance(field_name, str)


def is_legal_nlist(nlist: Any) -> bool:
    return not isinstance(nlist, bool) and isinstance(nlist, int)


def is_legal_topk(topk: Any) -> bool:
    return not isinstance(topk, bool) and isinstance(topk, int)


def is_legal_ids(ids: Any) -> bool:
    if not ids or not isinstance(ids, list):
        return False

    # TODO: Here check id valid value range may not match other SDK
    for i in ids:
        if not isinstance(i, (int, str)):
            return False
        try:
            i_ = int(i)
            if i_ < 0 or i_ > sys.maxsize:
                return False
        except Exception:
            return False

    return True


def is_legal_nprobe(nprobe: Any) -> bool:
    return isinstance(nprobe, int)


def is_legal_cmd(cmd: Any) -> bool:
    return cmd and isinstance(cmd, str)


def parser_range_date(date: Union[str, datetime.date]) -> str:
    if isinstance(date, datetime.date):
        return date.strftime('%Y-%m-%d')

    if isinstance(date, str):
        if not is_correct_date_str(date):
            raise ParamError('Date string should be YY-MM-DD format!')

        return date

    raise ParamError(
        'Date should be YY-MM-DD format string or datetime.date, '
        'or datetime.datetime object')


def is_legal_date_range(start: str, end: str) -> bool:
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    if (end_date - start_date).days < 0:
        return False

    return True


def is_legal_partition_name(tag: Any) -> bool:
    return tag is not None and isinstance(tag, str)


def is_legal_limit(limit: Any) -> bool:
    return isinstance(limit, int) and limit > 0


def is_legal_anns_field(field: Any) -> bool:
    return field and isinstance(field, str)


def is_legal_search_data(data: Any) -> bool:
    import numpy as np
    if not isinstance(data, (list, np.ndarray)):
        return False

    for vector in data:
        # list -> float vector
        # bytes -> byte vector
        if not isinstance(vector, (list, bytes, np.ndarray)):
            return False

    return True


def is_legal_output_fields(output_fields: Any) -> bool:
    if output_fields is None:
        return True

    if not isinstance(output_fields, list):
        return False

    for field in output_fields:
        if not is_legal_field_name(field):
            return False

    return True


def is_legal_partition_name_array(tag_array: Any) -> bool:
    if tag_array is None:
        return True

    if not isinstance(tag_array, list):
        return False

    for tag in tag_array:
        if not is_legal_partition_name(tag):
            return False

    return True


# https://milvus.io/cn/docs/v1.0.0/metric.md#floating
def is_legal_index_metric_type(index_type: str, metric_type: str) -> bool:
    if index_type not in ("FLAT",
                          "IVF_FLAT",
                          "IVF_SQ8",
                          # "IVF_SQ8_HYBRID",
                          "IVF_PQ",
                          "HNSW",
                          # "NSG",
                          "ANNOY",
                          "RHNSW_FLAT",
                          "RHNSW_PQ",
                          "RHNSW_SQ"):
        return False
    if metric_type not in ("L2", "IP"):
        return False
    return True


# https://milvus.io/cn/docs/v1.0.0/metric.md#binary
def is_legal_binary_index_metric_type(index_type: str, metric_type: str) -> bool:
    if index_type == "BIN_FLAT":
        if metric_type in ("JACCARD", "TANIMOTO", "HAMMING", "SUBSTRUCTURE", "SUPERSTRUCTURE"):
            return True
    elif index_type == "BIN_IVF_FLAT":
        if metric_type in ("JACCARD", "TANIMOTO", "HAMMING"):
            return True
    return False


def _raise_param_error(param_name: str, param_value: Any) -> None:
    raise ParamError(f"`{param_name}` value {param_value} is illegal")


def is_legal_round_decimal(round_decimal: Any) -> bool:
    return isinstance(round_decimal, int) and -2 < round_decimal < 7


def is_legal_travel_timestamp(ts: Any) -> bool:
    return isinstance(ts, int) and ts >= 0


def is_legal_guarantee_timestamp(ts: Any) -> bool:
    return isinstance(ts, int) and ts >= 0


def check_pass_param(*_args: Any, **kwargs: Any) -> None:  # pylint: disable=too-many-statements
    if kwargs is None:
        raise ParamError("Param should not be None")

    for key, value in kwargs.items():
        if key in ("collection_name",):
            if not is_legal_table_name(value):
                _raise_param_error(key, value)
        elif key == "field_name":
            if not is_legal_field_name(value):
                _raise_param_error(key, value)
        elif key == "dimension":
            if not is_legal_dimension(value):
                _raise_param_error(key, value)
        elif key == "index_file_size":
            if not is_legal_index_size(value):
                _raise_param_error(key, value)
        elif key in ("topk", "top_k"):
            if not is_legal_topk(value):
                _raise_param_error(key, value)
        elif key in ("ids",):
            if not is_legal_ids(value):
                _raise_param_error(key, value)
        elif key in ("nprobe",):
            if not is_legal_nprobe(value):
                _raise_param_error(key, value)
        elif key in ("nlist",):
            if not is_legal_nlist(value):
                _raise_param_error(key, value)
        elif key in ("cmd",):
            if not is_legal_cmd(value):
                _raise_param_error(key, value)
        elif key in ("partition_name",):
            if not is_legal_partition_name(value):
                _raise_param_error(key, value)
        elif key in ("partition_name_array",):
            if not is_legal_partition_name_array(value):
                _raise_param_error(key, value)
        elif key in ("limit",):
            if not is_legal_limit(value):
                _raise_param_error(key, value)
        elif key in ("anns_field",):
            if not is_legal_anns_field(value):
                _raise_param_error(key, value)
        elif key in ("search_data",):
            if not is_legal_search_data(value):
                _raise_param_error(key, value)
        elif key in ("output_fields",):
            if not is_legal_output_fields(value):
                _raise_param_error(key, value)
        elif key in ("round_decimal",):
            if not is_legal_round_decimal(value):
                _raise_param_error(key, value)
        elif key in ("travel_timestamp",):
            if not is_legal_travel_timestamp(value):
                _raise_param_error(key, value)
        elif key in ("guarantee_timestamp",):
            if not is_legal_guarantee_timestamp(value):
                _raise_param_error(key, value)
        # elif key in ("records",):
        #     if not is_legal_records(value):
        #         _raise_param_error(key, value)
        else:
            raise ParamError(f"unknown param `{key}`")
