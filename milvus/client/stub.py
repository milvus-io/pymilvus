from . import __version__
from .types import IndexType, MetricType
from .grpc_handler import GrpcHandler
from .http_handler import HttpHandler
from .exceptions import ParamError


class Milvus:
    def __init__(self, host=None, port=None, handler="HTTP", **kwargs):
        if handler == "GRPC":
            self._handler = GrpcHandler()
        elif handler == "HTTP":
            self._handler = HttpHandler()
        else:
            raise ParamError("Unknown handler options, please use \'GRPC\' or \'HTTP\'")

        self._handler._set_uri(host, port, **kwargs)

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
        return self._handler.server_status(timeout)

    def server_version(self, timeout=10):
        return self._handler.server_version(timeout)

    def _cmd(self, cmd, timeout=10):
        return self._handler._cmd(cmd)

    def create_table(self, param, timeout=10):
        return self._handler.create_table(param, timeout)

    def has_table(self, table_name, timeout=10):
        return self._handler.has_table(table_name, timeout)

    def describe_table(self, table_name, timeout=10):
        return self._handler.describe_table(table_name, timeout)

    def count_table(self, table_name, timeout=10):
        return self._handler.count_table(table_name, timeout)

    def show_tables(self, timeout=10):
        return self._handler.show_tables(timeout)

    def preload_table(self, table_name, timeout=None):
        return self._handler.preload_table(table_name, timeout)

    def drop_table(self, table_name, timeout=10):
        return self._handler.drop_table(table_name, timeout)

    def insert(self, table_name, records, ids=None, partition_tag=None, timeout=-1, **kwargs):
        return self._handler.insert(table_name, records, ids, partition_tag, timeout, **kwargs)

    def create_index(self, table_name, index=None, timeout=-1):
        index_default = {
            'index_type': IndexType.FLAT,
            'nlist': 16384
        }
        if not index:
            _index = index_default
        elif not isinstance(index, dict):
            raise ParamError("param `index` should be a dictionary")
        else:
            _index = index
            if index.get('index_type', None) is None:
                _index.update({'index_type': IndexType.FLAT})
            if index.get('nlist', None) is None:
                _index.update({'nlist': 16384})

        return self._handler.create_index(table_name, _index, timeout)

    def describe_index(self, table_name, timeout=10):
        return self._handler.describe_index(table_name, timeout)

    def drop_index(self, table_name, timeout=10):
        return self._handler.drop_index(table_name, timeout)

    def create_partition(self, table_name, partition_name, partition_tag, timeout=10):
        return self._handler.create_partition(table_name, partition_name, partition_tag, timeout)

    def show_partitions(self, table_name, timeout=10):
        return self._handler.show_partitions(table_name, timeout)

    def drop_partition(self, table_name, partition_tag, timeout=10):
        return self._handler.drop_partition(table_name, partition_tag, timeout)

    def search(self, table_name, top_k, nprobe,
               query_records, query_ranges=None, partition_tags=None, **kwargs):
        return self._handler.search(table_name, top_k, nprobe, query_records, query_ranges, partition_tags, **kwargs)

    def search_in_files(self, table_name, file_ids, query_records, top_k,
                        nprobe=16, query_ranges=None, **kwargs):
        return self._handler.search_in_files(table_name, file_ids, query_records, top_k, nprobe, query_ranges, **kwargs)
