import copy

from . import __version__
from .types import IndexType, MetricType
from .check import check_pass_param, is_legal_array
from .grpc_handler import GrpcHandler
from .http_handler import HttpHandler
from .exceptions import ParamError


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
        elif isinstance(self._handler, HttpHandler):
            return "HTTP"
        else:
            return "NULL"

    def connect(self, host=None, port=None, uri=None, timeout=1):
        return self._handler.connect(host, port, uri, timeout)

    def connected(self):
        return self._handler.connected()

    def disconnect(self):
        return self._handler.disconnect()

    def client_version(self, timeout=10):
        """
                Provide client version

                :return:
                    version: Client version

                :rtype: (str)
                """
        return __version__

    def server_status(self, timeout=10):
        return self._cmd("status", timeout)

    def server_version(self, timeout=10):
        return self._cmd("version", timeout)

    def _cmd(self, cmd, timeout=10):
        check_pass_param(cmd=cmd)

        return self._handler._cmd(cmd, timeout)

    def create_table(self, param, timeout=10):
        if not isinstance(param, dict):
            raise ParamError('Param type incorrect, expect {} but get {} instead'
                             .format(type(dict), type(param)))

        table_param = copy.deepcopy(param)

        if 'index_file_size' not in param:
            table_param['index_file_size'] = 1024
        if 'metric_type' not in param:
            table_param['metric_type'] = MetricType.L2

        check_pass_param(**table_param)

        return self._handler.create_table(table_param, timeout)

    def has_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.has_table(table_name, timeout)

    def describe_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.describe_table(table_name, timeout)

    def count_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.count_table(table_name, timeout)

    def show_tables(self, timeout=10):
        return self._handler.show_tables(timeout)

    def preload_table(self, table_name, timeout=None):
        check_pass_param(table_name=table_name)
        return self._handler.preload_table(table_name, timeout)

    def drop_table(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.drop_table(table_name, timeout)

    def insert(self, table_name, records, ids=None, partition_tag=None, timeout=-1, **kwargs):
        check_pass_param(table_name=table_name, records=records, ids=ids, partition_tag=partition_tag)

        if ids is not None:
            check_pass_param(ids=ids)

            if len(records) != len(ids):
                raise ParamError("length of vectors do not match that of ids")

        return self._handler.insert(table_name, records, ids, partition_tag, timeout, **kwargs)

    def create_index(self, table_name, index=None, timeout=-1):
        index_default = {'index_type': IndexType.FLAT, 'nlist': 16384}
        if not index:
            _index = index_default
        elif not isinstance(index, dict):
            raise ParamError("param `index` should be a dictionary")
        else:
            _index = copy.deepcopy(index)
            if index.get('index_type', None) is None:
                _index.update({'index_type': IndexType.FLAT})
            if index.get('nlist', None) is None:
                _index.update({'nlist': 16384})

        check_pass_param(table_name=table_name, **_index)

        return self._handler.create_index(table_name, _index, timeout)

    def describe_index(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.describe_index(table_name, timeout)

    def drop_index(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.drop_index(table_name, timeout)

    def create_partition(self, table_name, partition_name, partition_tag, timeout=10):
        check_pass_param(table_name=table_name,
                         partition_name=partition_name,
                         partition_tag=partition_tag)

        return self._handler.create_partition(table_name, partition_name, partition_tag, timeout)

    def show_partitions(self, table_name, timeout=10):
        check_pass_param(table_name=table_name)
        return self._handler.show_partitions(table_name, timeout)

    def drop_partition(self, table_name, partition_tag, timeout=10):
        check_pass_param(table_name=table_name, partition_tag=partition_tag)
        return self._handler.drop_partition(table_name, partition_tag, timeout)

    def search(self, table_name, top_k, nprobe,
               query_records, query_ranges=None, partition_tags=None, **kwargs):
        check_pass_param(table_name=table_name, topk=top_k, records=query_records,
                         nprobe=nprobe, partition_tag_array=partition_tags)
        return self._handler.search(table_name, top_k, nprobe, query_records, query_ranges, partition_tags, **kwargs)

    def search_in_files(self, table_name, file_ids, query_records, top_k,
                        nprobe=16, query_ranges=None, **kwargs):
        check_pass_param(table_name=table_name, topk=top_k, nprobe=nprobe, records=query_records)
        return self._handler.search_in_files(table_name, file_ids, query_records, top_k, nprobe, query_ranges, **kwargs)

    get_table_row_count = count_table
    delete_table = drop_table
    add_vectors = insert
    search_vectors = search
    search_vectors_in_files = search_in_files
