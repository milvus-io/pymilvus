import datetime
from .exceptions import ParamError
from .types import MetricType, IndexType


def is_legal_host(host):
    if not isinstance(host, str):
        return False

    return True


def is_legal_port(port):
    if isinstance(port, str):
        try:
            _port = int(port)
        except ValueError:
            return False
        else:
            if _port <= 0 or _port > 65535:
                return False

    return True


def is_legal_array(array):
    if not array or \
       not isinstance(array, list) or \
       len(array) <= 0 or \
       not isinstance(array[0], float):
        return False

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
    if isinstance(metric_type, int):
        try:
            _metric_type = MetricType(metric_type)
        except ValueError:
            return False

    return True


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
    return isinstance(nlist, int)


def is_legal_topk(topk):
    return isinstance(topk, int)


def is_legal_ids(ids):
    return isinstance(ids, list) and \
           len(ids) > 0 and \
           isinstance(ids[0], int)


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


def _raise_param_error(param_name):
    raise ParamError("{} is illegal".format(param_name))


def check_pass_param(*args, **kwargs):
    if kwargs is None:
        raise ParamError("Param should not be None")

    for key, value in kwargs.items():
        if key in ("table_name",):
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
