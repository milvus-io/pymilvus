import logging
import pytest
import pymilvus
from unittest import mock

from pymilvus import connections
from pymilvus import DefaultConfig

LOGGER = logging.getLogger(__name__)


# TODO rewrite
@pytest.mark.xfail
class TestConnections:
    @pytest.fixture(scope="function")
    def c(self):
        return connections

    @pytest.fixture(scope="function")
    def configure_params(self):
        params = {
            "test": {"host": "localhost", "port": "19530"},
            "dev": {"host": "127.0.0.1", "port": "19530"},
        }
        return params

    @pytest.fixture(scope="function")
    def host(self):
        return "localhost"

    @pytest.fixture(scope="function")
    def port(self):
        return "19530"

    @pytest.fixture(scope="function")
    def params(self, host, port):
        d = {
            "host": host,
            "port": port,
        }
        return d

    def test_constructor(self, c):
        LOGGER.info(type(c))

    def test_add_connection(self, c, configure_params):
        c.add_connection(**configure_params)

    @pytest.fixture(scope="function")
    def invalid_params(self):
        params = {
            "invalid1": {"port": "19530"},
            "invalid2": {"host": bytes("127.0.0.1", "utf-8"), "port": "19530"},
            "invalid3": {"host": 0, "port": "19530"},
            "invalid4": {"host": -1, "port": "19530"},
            "invalid5": {"host": 1.0, "port": "19530"},
            "invalid6": {"host": "{}", "port": "19530"},
            "invalid7": {"host": "[a, b, c]", "port": "19530"},
        }
        return params

    def test_invalid_params_connect(self, c, invalid_params):
        with pytest.raises(Exception):
            for k, v in range(invalid_params):
                temp = {k: v}
                c.add_connection(**temp)

    def test_add_another_connection_after_connect(self, c):
        k = "test"
        p = {k: {"host": "localhost", "port": "19530"}}
        c.add_connection(**p)

        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            c.connect(k)

        with pytest.raises(Exception):

            ap = {k: {"host": "localhost", "port": "19531"}}
            c.add_connection(**ap)

    def test_remove_connection_without_no_connections(self, c):
        c.remove_connection("remove")

    def test_remove_connection(self, c, host, port):
        with mock.patch("pymilvus.client.grpc_handler.GrpcHandler.__init__", return_value=None):
            with mock.patch("pymilvus.Milvus.close", return_value=None):
                alias = "default"

                c.connect(alias, host=host, port=port)
                c.disconnect(alias)

                assert c.get_connection(alias) is None
                c.remove_connection(alias)

    def test_connect_without_param(self, c):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            alias = "default"
            c.connect(alias)
            conn_got = c.get_connection(alias)
            assert isinstance(conn_got, pymilvus.client.grpc_handler.GrpcHandler)

    def test_connect(self, c, params):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            alias = "default"
            c.connect(alias, **params)
            conn_got = c.get_connection(alias)
            assert isinstance(conn_got, pymilvus.client.grpc_handler.GrpcHandler)

    def test_get_connection_with_no_connections(self, c):
        assert c.get_connection("get") is None

    def test_get_connection(self, c, host, port):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            alias = "default"

            c.connect(alias, host=host, port=port)

            conn_got = c.get_connection(alias)
            assert isinstance(conn_got, pymilvus.client.grpc_handler.GrpcHandler)

    def test_get_connection_without_alias(self, c, host, port):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            alias = DefaultConfig.DEFAULT_USING

            c.connect(alias, host=host, port=port)

            conn_got = c.get_connection()
            assert isinstance(conn_got, pymilvus.client.grpc_handler.GrpcHandler)

    def test_get_connection_with_configure_without_add(self, c, configure_params):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            c.add_connection(**configure_params)
            for key, _ in configure_params.items():
                c.connect(key)
                conn = c.get_connection(key)
                assert isinstance(conn, pymilvus.client.grpc_handler.GrpcHandler)

    def test_get_connection_addr(self, c, host, port):
        alias = DefaultConfig.DEFAULT_USING

        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            c.connect(host=host, port=port)

        connection_addr = c.get_connection_addr(alias)
        assert connection_addr["host"] == host
        assert connection_addr["port"] == port

    def test_list_connections(self, c, host, port):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            c.connect()

        conns = c.list_connections()

        assert len(conns) == 1
