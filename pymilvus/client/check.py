import datetime
import sys
from typing import Any, Callable, Union

from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import milvus_pb2 as milvus_types

from .singleton_utils import Singleton


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
    if not array or not isinstance(array, list) or len(array) == 0:
        return False

    return True


def is_legal_bin_vector(array: Any) -> bool:
    if not array or not isinstance(array, bytes) or len(array) == 0:
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
        datetime.datetime.strptime(param, "%Y-%m-%d")
    except ValueError:
        return False

    return True


def is_legal_dimension(dim: Any) -> bool:
    return isinstance(dim, int)


def is_legal_index_size(index_size: Any) -> bool:
    return isinstance(index_size, int)


def is_legal_table_name(table_name: Any) -> bool:
    return table_name and isinstance(table_name, str)


def is_legal_db_name(db_name: Any) -> bool:
    # you can connect to the default database "".
    return isinstance(db_name, str)


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
        return date.strftime("%Y-%m-%d")

    if isinstance(date, str):
        if not is_correct_date_str(date):
            raise ParamError(message="Date string should be YY-MM-DD format!")

        return date

    raise ParamError(
        message="Date should be YY-MM-DD format string or datetime.date, "
        "or datetime.datetime object"
    )


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
    return field is None or isinstance(field, str)


def is_legal_search_data(data: Any) -> bool:
    import numpy as np

    if not isinstance(data, (list, np.ndarray)):
        return False

    return all(isinstance(vector, (list, bytes, np.ndarray)) for vector in data)


def is_legal_output_fields(output_fields: Any) -> bool:
    if output_fields is None:
        return True

    if not isinstance(output_fields, list):
        return False

    return all(is_legal_field_name(field) for field in output_fields)


def is_legal_partition_name_array(tag_array: Any) -> bool:
    if tag_array is None:
        return True

    if not isinstance(tag_array, list):
        return False

    return all(is_legal_partition_name(tag) for tag in tag_array)


def is_legal_replica_number(replica_number: int) -> bool:
    return isinstance(replica_number, int)


# https://milvus.io/cn/docs/v1.0.0/metric.md#floating
def is_legal_index_metric_type(index_type: str, metric_type: str) -> bool:
    if index_type not in (
        "FLAT",
        "IVF_FLAT",
        "IVF_SQ8",
        "IVF_PQ",
        "HNSW",
        "AUTOINDEX",
        "DISKANN",
        "ANNOY",
        "RHNSW_FLAT",
        "RHNSW_PQ",
        "RHNSW_SQ",
    ):
        return False
    if metric_type not in ("L2", "IP"):
        return False
    return True


# https://milvus.io/cn/docs/v1.0.0/metric.md#binary
def is_legal_binary_index_metric_type(index_type: str, metric_type: str) -> bool:
    if index_type == "BIN_FLAT":
        if metric_type in ("JACCARD", "TANIMOTO", "HAMMING", "SUBSTRUCTURE", "SUPERSTRUCTURE"):
            return True
    elif index_type == "BIN_IVF_FLAT" and metric_type in ("JACCARD", "TANIMOTO", "HAMMING"):
        return True
    return False


def _raise_param_error(param_name: str, param_value: Any) -> None:
    raise ParamError(message=f"`{param_name}` value {param_value} is illegal")


def is_legal_round_decimal(round_decimal: Any) -> bool:
    return isinstance(round_decimal, int) and -2 < round_decimal < 7


def is_legal_travel_timestamp(ts: Any) -> bool:
    return isinstance(ts, int) and ts >= 0


def is_legal_guarantee_timestamp(ts: Any) -> bool:
    return ts is None or isinstance(ts, int) and ts >= 0


def is_legal_user(user: Any) -> bool:
    return isinstance(user, str)


def is_legal_password(password: Any) -> bool:
    return isinstance(password, str)


def is_legal_role_name(role_name: Any) -> bool:
    return role_name and isinstance(role_name, str)


def is_legal_operate_user_role_type(operate_user_role_type: Any) -> bool:
    return operate_user_role_type in (
        milvus_types.OperateUserRoleType.AddUserToRole,
        milvus_types.OperateUserRoleType.RemoveUserFromRole,
    )


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
    return operate_privilege_type in (
        milvus_types.OperatePrivilegeType.Grant,
        milvus_types.OperatePrivilegeType.Revoke,
    )


class ParamChecker(metaclass=Singleton):
    def __init__(self) -> None:
        self.check_dict = {
            "db_name": is_legal_db_name,
            "collection_name": is_legal_table_name,
            "field_name": is_legal_field_name,
            "dimension": is_legal_dimension,
            "index_file_size": is_legal_index_size,
            "topk": is_legal_topk,
            "ids": is_legal_ids,
            "nprobe": is_legal_nprobe,
            "nlist": is_legal_nlist,
            "cmd": is_legal_cmd,
            "partition_name": is_legal_partition_name,
            "partition_name_array": is_legal_partition_name_array,
            "limit": is_legal_limit,
            "anns_field": is_legal_anns_field,
            "search_data": is_legal_search_data,
            "output_fields": is_legal_output_fields,
            "round_decimal": is_legal_round_decimal,
            "travel_timestamp": is_legal_travel_timestamp,
            "guarantee_timestamp": is_legal_guarantee_timestamp,
            "user": is_legal_user,
            "password": is_legal_password,
            "role_name": is_legal_role_name,
            "operate_user_role_type": is_legal_operate_user_role_type,
            "include_user_info": is_legal_include_user_info,
            "include_role_info": is_legal_include_role_info,
            "object": is_legal_object,
            "object_name": is_legal_object_name,
            "privilege": is_legal_privilege,
            "operate_privilege_type": is_legal_operate_privilege_type,
            "properties": is_legal_collection_properties,
            "replica_number": is_legal_replica_number,
            "resource_group_name": is_legal_table_name,
        }

    def check(self, key: str, value: Callable):
        if key in self.check_dict:
            if not self.check_dict[key](value):
                _raise_param_error(key, value)
        else:
            raise ParamError(message=f"unknown param `{key}`")


def _get_param_checker():
    return ParamChecker()


def check_pass_param(*_args: Any, **kwargs: Any) -> None:  # pylint: disable=too-many-statements
    if kwargs is None:
        raise ParamError(message="Param should not be None")
    checker = _get_param_checker()
    for key, value in kwargs.items():
        checker.check(key, value)
