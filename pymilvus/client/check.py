import sys
import datetime
from typing import Any, Union
from ..exceptions import ParamError
from ..grpc_gen import milvus_pb2 as milvus_types
from .utils import (
    valid_index_types,
    valid_binary_index_types,
    valid_index_params_keys,
)


def is_legal_address(addr: Any) -> bool:
    if not isinstance(addr, str):
        return False

    a = addr.split(":")
    if len(a) != 2:
        return False

    if not is_legal_host(a[0]) or not is_legal_port(a[1]):
        return False

    return True


def is_legal_host(host: Any) -> bool:
    if not isinstance(host, str):
        return False

    if len(host) == 0:
        return False

    return True


def is_legal_port(port: Any) -> bool:
    if isinstance(port, (str, int)):
        try:
            int(port)
        except ValueError:
            return False
        else:
            return True
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
            raise ParamError(message='Date string should be YY-MM-DD format!')

        return date

    raise ParamError(message='Date should be YY-MM-DD format string or datetime.date, '
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
                          "RHNSW_SQ",
                          "AUTOINDEX",
                          "DISKANN"):
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
    raise ParamError(message=f"`{param_name}` value {param_value} is illegal")


def is_legal_round_decimal(round_decimal: Any) -> bool:
    return isinstance(round_decimal, int) and -2 < round_decimal < 7


def is_legal_travel_timestamp(ts: Any) -> bool:
    return isinstance(ts, int) and ts >= 0


def is_legal_guarantee_timestamp(ts: Any) -> bool:
    return isinstance(ts, int) and ts >= 0


def is_legal_user(user) -> bool:
    return isinstance(user, str)


def is_legal_password(password) -> bool:
    return isinstance(password, str)


def is_legal_role_name(role_name: Any) -> bool:
    return role_name and isinstance(role_name, str)


def is_legal_operate_user_role_type(operate_user_role_type: Any) -> bool:
    return operate_user_role_type in \
        (milvus_types.OperateUserRoleType.AddUserToRole, milvus_types.OperateUserRoleType.RemoveUserFromRole)


def is_legal_include_user_info(include_user_info: Any) -> bool:
    return isinstance(include_user_info, bool)


def is_legal_include_role_info(include_role_info: Any) -> bool:
    return isinstance(include_role_info, bool)


def is_legal_object(object: Any) -> bool:
    return object and isinstance(object, str)


def is_legal_object_name(object_name: Any) -> bool:
    return object_name and isinstance(object_name, str)


def is_legal_privilege(privilege: Any) -> bool:
    return privilege and isinstance(privilege, str)


def is_legal_collection_properties(properties: Any) -> bool:
    return properties and isinstance(properties, dict)


def is_legal_operate_privilege_type(operate_privilege_type: Any) -> bool:
    return operate_privilege_type in \
            (milvus_types.OperatePrivilegeType.Grant, milvus_types.OperatePrivilegeType.Revoke)


def check_pass_param(*_args: Any, **kwargs: Any) -> None:  # pylint: disable=too-many-statements
    if kwargs is None:
        raise ParamError(message="Param should not be None")

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
        elif key in ("user",):
            if not is_legal_user(value):
                _raise_param_error(key, value)
        elif key in ("password",):
            if not is_legal_password(value):
                _raise_param_error(key, value)
        # elif key in ("records",):
        #     if not is_legal_records(value):
        #         _raise_param_error(key, value)
        elif key in ("role_name",):
            if not is_legal_role_name(value):
                _raise_param_error(key, value)
        elif key in ("operate_user_role_type",):
            if not is_legal_operate_user_role_type(value):
                _raise_param_error(key, value)
        elif key in ("include_user_info",):
            if not is_legal_include_user_info(value):
                _raise_param_error(key, value)
        elif key in ("include_role_info",):
            if not is_legal_include_role_info(value):
                _raise_param_error(key, value)
        elif key in ("object",):
            if not is_legal_object(value):
                _raise_param_error(key, value)
        elif key in ("object_name",):
            if not is_legal_object_name(value):
                _raise_param_error(key, value)
        elif key in ("privilege",):
            if not is_legal_privilege(value):
                _raise_param_error(key, value)
        elif key in ("operate_privilege_type",):
            if not is_legal_operate_privilege_type(value):
                _raise_param_error(key, value)
        elif key == "properties":
            if not is_legal_collection_properties(value):
                _raise_param_error(key, value)
        else:
            raise ParamError(message=f"unknown param `{key}`")


def check_index_params(params):
    params = params or {}
    if not isinstance(params, dict):
        raise ParamError(message="Params must be a dictionary type")
    # params preliminary validate
    if 'index_type' not in params:
        raise ParamError(message="Params must contains key: 'index_type'")
    if 'params' not in params:
        raise ParamError(message="Params must contains key: 'params'")
    if 'metric_type' not in params:
        raise ParamError(message="Params must contains key: 'metric_type'")
    if not isinstance(params['params'], dict):
        raise ParamError(message="Params['params'] must be a dictionary type")
    if params['index_type'] not in valid_index_types:
        raise ParamError(message=f"Invalid index_type: {params['index_type']}, which must be one of: {str(valid_index_types)}")
    for k in params['params'].keys():
        if k not in valid_index_params_keys:
            raise ParamError(message=f"Invalid params['params'].key: {k}")
    for v in params['params'].values():
        if not isinstance(v, int):
            raise ParamError(message=f"Invalid params['params'].value: {v}, which must be an integer")
    # filter invalid metric type
    if params['index_type'] in valid_binary_index_types:
        if not is_legal_binary_index_metric_type(params['index_type'], params['metric_type']):
            raise ParamError(message=f"Invalid metric_type: {params['metric_type']}, which does not match the index type: {params['index_type']}")
    else:
        if not is_legal_index_metric_type(params['index_type'], params['metric_type']):
            raise ParamError(message=f"Invalid metric_type: {params['metric_type']}, which does not match the index type: {params['index_type']}")
