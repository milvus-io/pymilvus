import sys
import datetime
import numpy as np
from urllib.parse import urlparse

from .exceptions import ParamError
from .types import MetricType, IndexType


def is_legal_host(host):
    if not isinstance(host, str):
        return False

    return True


def is_legal_port(port):
    if isinstance(port, (str, int)):
        try:
            _port = int(port)
        except ValueError:
            return False
        else:
            if _port <= 0 or _port > 65535:
                return False

    return True


def is_legal_uri(uri):
    if uri is None:
        return True

    try:
        _uri = urlparse(uri)
        return _uri.scheme == 'tcp'
    except (AttributeError, ValueError, TypeError):
        return False


def is_legal_vector(array):
    if not array or \
            not isinstance(array, list) or \
            len(array) == 0:
        return False

    # for v in array:
    #     if not isinstance(v, float):
    #         return False

    return True


def is_legal_bin_vector(array):
    if not array or \
            not isinstance(array, bytes) or \
            len(array) == 0:
        return False

    return True


def is_legal_numpy_array(array):
    return False if array is None or array.size == 0 else True


def is_legal_records(value):
    param_error = ParamError('A vector must be a non-empty, 2-dimensional array and '
                             'must contain only elements with the float data type or the bytes data type.')

    if isinstance(value, np.ndarray):
        if not is_legal_numpy_array(value):
            raise param_error

        return True

    if not isinstance(value, list) or len(value) == 0:
        raise param_error

    if isinstance(value[0], bytes):
        check_func = is_legal_bin_vector
    elif isinstance(value[0], list):
        check_func = is_legal_vector
    else:
        raise param_error

    _dim = len(value[0])
    for record in value:
        if not check_func(record):
            raise param_error
        if _dim != len(record):
            raise ParamError('Whole vectors must have the same dimension')

    return True


def int_or_str(item):
    if isinstance(item, int):
        return str(item)

    return item


def is_correct_date_str(param):
    try:
        datetime.datetime.strptime(param, '%Y-%m-%d')
    except ValueError:
        raise ParamError('Incorrect data format, should be YYYY-MM-DD')

    return True


def is_legal_dimension(dim):
    return isinstance(dim, int)


def is_legal_index_size(index_size):
    return isinstance(index_size, int)


def is_legal_metric_type(metric_type):
    if isinstance(metric_type, (int, MetricType)):
        try:
            return MetricType(metric_type) != MetricType.INVALID
        except ValueError:
            return False

    return False


def is_legal_index_type(index_type):
    if isinstance(index_type, int):
        try:
            index_type = IndexType(index_type)
        except ValueError:
            return False

    if isinstance(index_type, IndexType):
        if index_type != IndexType.INVALID:
            return True

    return False


def is_legal_table_name(table_name):
    return isinstance(table_name, str) and len(table_name) > 0


def is_legal_nlist(nlist):
    return not isinstance(nlist, bool) and isinstance(nlist, int)


def is_legal_topk(topk):
    return not isinstance(topk, bool) and isinstance(topk, int)


def is_legal_ids(ids):
    if not isinstance(ids, list) or \
            len(ids) == 0:
        return False

    # TODO: Here check id valid value range may not match other SDK
    for i in ids:
        if not isinstance(i, (int, str)):
            return False
        try:
            i_ = int(i)
            if i_ < 0 or i_ > sys.maxsize:
                return False
        except:
            return False

    return True


def is_legal_nprobe(nprobe):
    return isinstance(nprobe, int)


def is_legal_cmd(cmd):
    return isinstance(cmd, str) and len(cmd) > 0


def parser_range_date(date):
    if isinstance(date, datetime.date):
        return date.strftime('%Y-%m-%d')

    if isinstance(date, str):
        if not is_correct_date_str(date):
            raise ParamError('Date string should be YY-MM-DD format!')

        return date

    raise ParamError(
        'Date should be YY-MM-DD format string or datetime.date, '
        'or datetime.datetime object')


def is_legal_date_range(start, end):
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    if (end_date - start_date).days < 0:
        return False

    return True


def is_legal_partition_tag(tag):
    return tag is not None and isinstance(tag, str)


def is_legal_partition_tag_array(tag_array):
    if tag_array is None:
        return True

    if not isinstance(tag_array, list):
        return False

    for tag in tag_array:
        if not is_legal_partition_tag(tag):
            return False

    return True


def _raise_param_error(param_name):
    raise ParamError("{} is illegal".format(param_name))


def check_pass_param(*args, **kwargs):
    if kwargs is None:
        raise ParamError("Param should not be None")

    for key, value in kwargs.items():
        if key in ("collection_name",):
            if not is_legal_table_name(value):
                _raise_param_error(key)
        elif key == "dimension":
            if not is_legal_dimension(value):
                _raise_param_error(key)
        elif key in ("index_type",):
            if not is_legal_index_type(value):
                _raise_param_error(key)
        elif key == "index_file_size":
            if not is_legal_index_size(value):
                _raise_param_error(key)
        elif key == "metric_type":
            if not is_legal_metric_type(value):
                _raise_param_error(key)
        elif key in ("topk", "top_k"):
            if not is_legal_topk(value):
                _raise_param_error(key)
        elif key in ("ids",):
            if not is_legal_ids(value):
                _raise_param_error(key)
        elif key in ("nprobe",):
            if not is_legal_nprobe(value):
                _raise_param_error(key)
        elif key in ("nlist",):
            if not is_legal_nlist(value):
                _raise_param_error(key)
        elif key in ("cmd",):
            if not is_legal_cmd(value):
                _raise_param_error(key)
        elif key in ("partition_tag",):
            if not is_legal_partition_tag(value):
                _raise_param_error(key)
        elif key in ("partition_tag_array",):
            if not is_legal_partition_tag_array(value):
                _raise_param_error(key)
        elif key in ("records",):
            if not is_legal_records(value):
                _raise_param_error(key)
        else:
            raise ParamError("unknown param `{}`".format(key))
