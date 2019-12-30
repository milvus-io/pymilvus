import requests as rq
from urllib.parse import urlparse

from .abstract import ConnectIntf
from .types import Status
from .exceptions import ParamError
from .check import is_legal_host, is_legal_port

from ..settings import DefaultConfig as config


class HttpHandler(ConnectIntf):

    def __init__(self, host, port, **kwargs):
        self._uri = kwargs.get("uri", "127.0.0.1:19121")

    def _set_uri(self, host, port, **kwargs):
        """
        Set server network address

        :raises: ParamError

        """
        if host is not None:
            _port = port if port is not None else config.GRPC_PORT
            _host = host
        elif port is None:
            try:
                _uri = kwargs.get("uri", None)
                _uri = urlparse(_uri) if _uri else urlparse(config.GRPC_URI)
                _host = _uri.hostname
                _port = _uri.port
            except (AttributeError, ValueError, TypeError) as e:
                raise ParamError("uri is illegal: {}".format(e))
        else:
            raise ParamError("Param is not complete. Please invoke as follow:\n"
                             "\t(host = ${HOST}, port = ${PORT})\n"
                             "\t(uri = ${URI})\n")

        if not is_legal_host(_host) or not is_legal_port(_port):
            raise ParamError("host or port is illeagl")

        self._uri = "{}:{}".format(str(_host), str(_port))

    def ping(self):
        response = rq.get(self._uri + "/state")
        try:
            js = response.json()
            return Status(js["code"], js["message"])
        except ValueError as e:
            pass

        return Status(Status.UNEXPECTED_ERROR, "Error occurred when parse response.")

    def connect(self, host, port, uri, timeout):
        return self.ping()

    def connected(self):
        return self.ping()

    def disconnect(self):
        pass

    def create_table(self, param, timeout):
        if isinstance(param, dict):
            response = rq.post(self._uri + "/tables", data=param)
        else:
            raise ParamError("Param is illegal")

    def has_table(self, table_name, timeout):
        url = self._uri + "/tables/" + table_name
        response = rq.get(url=url)
        if (200 == response.status_code):
            return Status()

        js = response.json()
        return Status(Status(js["code"], js["message"]))

    def delete_table(self, table_name, timeout):
        super().delete_table(table_name, timeout)

    def add_vectors(self, table_name, records, ids, timeout, **kwargs):
        super().add_vectors(table_name, records, ids, timeout, **kwargs)

    def search_vectors(self, table_name, top_k, nprobe, query_records, query_ranges, **kwargs):
        super().search_vectors(table_name, top_k, nprobe, query_records, query_ranges, **kwargs)

    def search_vectors_in_files(self, table_name, file_ids, query_records, top_k, nprobe, query_ranges, **kwargs):
        super().search_vectors_in_files(table_name, file_ids, query_records, top_k, nprobe, query_ranges, **kwargs)

    def describe_table(self, table_name, timeout):
        super().describe_table(table_name, timeout)

    def get_table_row_count(self, table_name, timeout):
        super().get_table_row_count(table_name, timeout)

    def show_tables(self, timeout):
        super().show_tables(timeout)

    def create_index(self, table_name, index, timeout):
        return super().create_index(table_name, index, timeout)

    def client_version(self):
        return super().client_version()

    def server_version(self, timeout):
        return super().server_version(timeout)

    def server_status(self, timeout):
        return super().server_status(timeout)

    def preload_table(self, table_name, timeout):
        super().preload_table(table_name, timeout)

    def describe_index(self, table_name, timeout):
        return super().describe_index(table_name, timeout)

    def drop_index(self, table_name, timeout):
        super().drop_index(table_name, timeout)
