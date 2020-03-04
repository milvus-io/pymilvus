import copy
import functools

from . import __version__
from .types import IndexType, MetricType
from .check import check_pass_param
from .grpc_handler import GrpcHandler
from .http_handler import HttpHandler
from .exceptions import ParamError, NotConnectError


def check_connect(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        return f(self, *args, **kwargs)

    return wrapper


class Milvus:
    def __init__(self, host=None, port=None, handler="GRPC", **kwargs):
        if handler == "GRPC":
            self._handler = GrpcHandler(host=host, port=port, **kwargs)
        elif handler == "HTTP":
            self._handler = HttpHandler(host=host, port=port, **kwargs)
        else:
            raise ParamError("Unknown handler options, please use \'GRPC\' or \'HTTP\'")

    def __enter__(self):
        self._handler.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handler.__exit__(exc_type, exc_val, exc_tb)

    def set_hook(self, **kwargs):
        return self._handler.set_hook(**kwargs)

    @property
    def status(self):
        return self._handler.status

    @property
    def handler(self):
        if isinstance(self._handler, GrpcHandler):
            return "GRPC"

        if isinstance(self._handler, HttpHandler):
            return "HTTP"

        return "NULL"

    def connect(self, host=None, port=None, uri=None, timeout=1):
        return self._handler.connect(host, port, uri, timeout)

    def connected(self):
        return self._handler.connected()

    @check_connect
    def disconnect(self):
        return self._handler.disconnect()

    def client_version(self):
        """
                Provide client version

                :return:
                    version: Client version

                :rtype: (str)
                """
        return __version__

    @check_connect
    def server_status(self, timeout=10):
        return self._cmd("status", timeout)

    @check_connect
    def server_version(self, timeout=10):
        return self._cmd("version", timeout)

    @check_connect
    def _cmd(self, cmd, timeout=10):
        check_pass_param(cmd=cmd)

        return self._handler._cmd(cmd, timeout)

    @check_connect
    def create_table(self, param, timeout=10):
        if not isinstance(param, dict):
            raise ParamError('Param type incorrect, expect {} but get {} instead'
                             .format(type(dict), type(param)))

        table_param = copy.deepcopy(param)

        if 'table_name' not in table_param:
            raise ParamError('table_name is required')
        table_name = table_param["table_name"]
        table_param.pop('table_name')

        if 'dimension' not in table_param:
            raise ParamError('dimension is required')
        dim = table_param["dimension"]
        table_param.pop("dimension")

        index_file_size = table_param.get('index_file_size', 1024)
        table_param.pop('index_file_size', None)

        metric_type = table_param.get('metric_type', MetricType.L2)
        table_param.pop('metric_type', None)

        check_pass_param(table_name=table_name, dimension=dim, index_file_size=index_file_size, metric_type=metric_type)

        return self._handler.create_table(table_name, dim, index_file_size, metric_type, table_param, timeout)

    @check_connect
    def has_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.has_table(table_name, timeout)

    @check_connect
    def describe_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.describe_table(table_name, timeout)

    @check_connect
    def count_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.count_table(table_name, timeout)

    @check_connect
    def show_tables(self, timeout=10):
        return self._handler.show_tables(timeout)

    @check_connect
    def table_info(self, table_name, timeout=10):
        return self._handler.show_table_info(table_name, timeout)

    @check_connect
    def preload_table(self, table_name, timeout=None):
        check_pass_param(table_name=table_name)
        return self._handler.preload_table(table_name, timeout)

    @check_connect
    def drop_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.drop_table(table_name, timeout)

    @check_connect
    def insert(self, table_name, records, ids=None, partition_tag=None, params=None, timeout=-1):
        check_pass_param(table_name=table_name, records=records,
                         ids=ids, partition_tag=partition_tag)

        if ids is not None and len(records) != len(ids):
            raise ParamError("length of vectors do not match that of ids")

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.insert(table_name, records, ids, partition_tag, params, timeout)

    @check_connect
    def get_vector_by_id(self, table_name, vector_id):
        check_pass_param(table_name=table_name, ids=[vector_id])

        return self._handler.get_vector_by_id(table_name, vector_id)

    @check_connect
    def get_vector_ids(self, table_name, segment_name):
        check_pass_param(table_name=table_name)
        check_pass_param(table_name=segment_name)

        return self._handler.get_vector_ids(table_name, segment_name)

    @check_connect
    def create_index(self, table_name, index_type=None, params=None, timeout=-1):
        _index_type = IndexType.FLAT if index_type is None else index_type
        check_pass_param(table_name=table_name, index_type=_index_type)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.create_index(table_name, _index_type, params, timeout)

    @check_connect
    def describe_index(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.describe_index(table_name, timeout)

    @check_connect
    def drop_index(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.drop_index(table_name, timeout)

    @check_connect
    def create_partition(self, table_name, partition_tag, timeout=10):
        check_pass_param(table_name=table_name, partition_tag=partition_tag)

        return self._handler.create_partition(table_name, partition_tag, timeout)

    @check_connect
    def show_partitions(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.show_partitions(table_name, timeout)

    @check_connect
    def drop_partition(self, table_name, partition_tag, timeout=10):
        check_pass_param(table_name=table_name, partition_tag=partition_tag)
        return self._handler.drop_partition(table_name, partition_tag, timeout)

    @check_connect
    def search(self, table_name, top_k, query_records, partition_tags=None, params=None):
        check_pass_param(table_name=table_name, topk=top_k,
                         records=query_records, partition_tag_array=partition_tags)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.search(table_name, top_k, query_records, partition_tags, params)

    # def search_by_id(self, table_name, top_k, vector_id, partition_tags=None, params=None):
    #     if not isinstance(vector_id, int):
    #         raise ParamError("Vector id must be an integer")
    #
    #     partition_tags = partition_tags or list()
    #     check_pass_param(table_name=table_name, topk=top_k, partition_tag_array=partition_tags)
    #
    #     return self._handler.search_by_id(table_name, top_k, vector_id, partition_tags, params)

    @check_connect
    def search_in_files(self, table_name, file_ids, query_records, top_k, params=None):
        check_pass_param(table_name=table_name, topk=top_k, records=query_records)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.search_in_files(table_name, file_ids,
                                             query_records, top_k, params)

    @check_connect
    def delete_by_id(self, table_name, id_array, timeout=None):
        check_pass_param(table_name=table_name, ids=id_array)

        return self._handler.delete_by_id(table_name, id_array, timeout)

    @check_connect
    def flush(self, table_name_array=None):
        if table_name_array is None:
            return self._handler.flush(table_name_array)

        if not isinstance(table_name_array, list):
            raise ParamError("Table name array must be type of list")

        if len(table_name_array) <= 0:
            raise ParamError("Table name array is not allowed to be empty")

        for name in table_name_array:
            check_pass_param(table_name=name)

        return self._handler.flush(table_name_array)

    @check_connect
    def compact(self, table_name, timeout=None):
        check_pass_param(table_name=table_name)

        return self._handler.compact(table_name, timeout)

    @check_connect
    def set_config(self, parent_key, child_key, value):
        cmd_str = "set_config {}.{} {}".format(parent_key, child_key, value)
        return self._cmd(cmd_str)

    @check_connect
    def get_config(self, parent_key, child_key):
        cmd_str = "get_config {}.{}".format(parent_key, child_key)
        return self._cmd(cmd_str)

    # In old version of pymilvus, some methods are different from the new.
    # apply alternative method name for compatibility
    get_table_row_count = count_table
    delete_table = drop_table
    add_vectors = insert
    search_vectors = search
    search_vectors_in_files = search_in_files
