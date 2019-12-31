from . import __version__
from .grpc_handler import GrpcHandler


class Milvus:
    def __init__(self, host=None, port=None, handler=GrpcHandler(), **kwargs):
        self._handler = handler
        self._handler._set_uri(host, port, **kwargs)

    @property
    def status(self):
        return self._handler.status

    def connect(self, host=None, port=None, **kwargs):
        return self._handler.connect(host, port, **kwargs)

    def connected(self):
        return self._handler.connected()

    def disconncet(self):
        return self._handler.disconnect()

    def client_version(self):
        """
                Provide client version

                :return:
                    version: Client version

                :rtype: (str)
                """
        return __version__

    def server_status(self):
        return self._handler.server_status()

    def _cmd(self, cmd):
        return self._handler._cmd(cmd)

    def create_table(self, param):
        return self._handler.create_table(param)

    def has_table(self, table_name):
        return self._handler.has_table(table_name)

    def describe_table(self, table_name, timeout=10):
        return self._handler.describe_table(table_name, timeout)

    def count_table(self, table_name, timeout=30):
        return self._handler.count_table(table_name, timeout)

    def show_tables(self):
        return self._handler.show_tables()

    def preload_table(self, table_name):
        return self._handler.preload_table(table_name)

    def drop_table(self, table_name):
        return self._handler.drop_table(table_name)

    def insert(self, table_name, records, ids=None, partition_tag=None, timeout=-1, **kwargs):
        return self._handler.insert(table_name, records, ids, partition_tag, timeout, **kwargs)

    def create_index(self, table_name, index=None, timeout=-1):
        return self._handler.create_index(table_name, index, timeout)

    def describe_index(self, table_name, timeout=10):
        return self._handler.describe_index(table_name, timeout)

    def drop_index(self, table_name, timeout=10):
        return self._handler.drop_index(table_name, timeout)

    def create_partition(self, table_name, partition_name, partition_tag, timeout=10):
        return self.create_partition(table_name, partition_name, partition_tag, timeout)

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
