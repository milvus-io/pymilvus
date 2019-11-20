import datetime
import time
from .exceptions import ParamError
from .types import MetricType, IndexType

from ..grpc_gen import status_pb2


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


def is_legal_partition_name(name):
    return name is None or isinstance(name, str)


def is_legal_partition_tag(tag):
    return tag is None or isinstance(tag, str)


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
        elif key in ("partition_name",):
            if not is_legal_partition_name(value):
                _raise_param_error(key)
        elif key in ("partition_tag",):
            if not is_legal_partition_tag(value):
                _raise_param_error(key)
        elif key in ("partition_tag_array",):
            if not is_legal_partition_tag_array(value):
                _raise_param_error(key)
        else:
            raise ParamError("unknown param `{}`".format(key))

    def _do_merge(files_n_topk_results, topk, reverse=False, **kwargs):
        """
        merge query results
        """

        def _reduce(source_ids, ids, source_diss, diss, k, reverse):
            """

            """
            if source_diss[k - 1] <= diss[0]:
                return source_ids, source_diss
            if diss[k - 1] <= source_diss[0]:
                return ids, diss

            source_diss.extend(diss)
            diss_t = enumerate(source_diss)
            diss_m_rst = sorted(diss_t, key=lambda x: x[1], reverse=reverse)[:k]
            diss_m_out = [id_ for _, id_ in diss_m_rst]

            source_ids.extend(ids)
            id_m_out = [source_ids[i] for i, _ in diss_m_rst]

            return id_m_out, diss_m_out

        status = status_pb2.Status(error_code=status_pb2.SUCCESS,
                                   reason="Success")
        if not files_n_topk_results:
            return status, [], []

        merge_id_results = []
        merge_dis_results = []

        for files_collection in files_n_topk_results:
            if isinstance(files_collection, tuple):
                status, _ = files_collection
                return status, []

            row_num = files_collection.row_num
            ids = files_collection.ids
            diss = files_collection.distances  # distance collections
            # TODO: batch_len is equal to topk, may need to compare with topk
            batch_len = len(ids) // row_num

            for row_index in range(row_num):
                id_batch = ids[row_index * batch_len: (row_index + 1) * batch_len]
                dis_batch = diss[row_index * batch_len: (row_index + 1) * batch_len]

                if len(merge_id_results) < row_index:
                    raise ValueError("merge error")
                elif len(merge_id_results) == row_index:
                    # TODO: may bug here
                    merge_id_results.append(id_batch)
                    merge_dis_results.append(dis_batch)
                else:
                    merge_id_results[row_index], merge_dis_results[row_index] = \
                        _reduce(merge_id_results[row_index], id_batch,
                                merge_dis_results[row_index], dis_batch,
                                batch_len,
                                reverse)

        id_mrege_list = []
        dis_mrege_list = []

        for id_results, dis_results in zip(merge_id_results, merge_dis_results):
            id_mrege_list.extend(id_results)
            dis_mrege_list.extend(dis_results)

        return status, id_mrege_list, dis_mrege_list
