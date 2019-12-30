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

