import logging
import pytest
import pymilvus
from unittest import mock

from pymilvus import connections
from pymilvus import DefaultConfig, MilvusException

LOGGER = logging.getLogger(__name__)


class TestAddConnection:
    @pytest.fixture(scope="function", params=[
        {"host": "localhost", "port": "19530"},
        {"host": "localhost", "port": "19531"},
        {"host": "localhost", "port": "19530", "random": "useless"},
    ])
    def addr(self, request):
        return request.param

    @pytest.fixture(scope="function", params=[
        {"host": None, "port": "19530"},
        {"host": 1, "port": "19531"},
        {"host": 1.0, "port": "19530", "random": "useless"},
    ])
    def invalid_host(self, request):
        return request.param

    @pytest.fixture(scope="function", params=[
        {"host": "localhost", "port": None},
        {"host": "localhost", "port": 1.0},
        {"host": "localhost", "port": b'19530', "random": "useless"},
    ])
    def invalid_port(self, request):
        return request.param

    @pytest.mark.skip("TODO: empty user in connection addr")
    def test_add_connection_no_error(self, addr):
        add_connection = connections.add_connection

        add_connection(test=addr)
        assert connections.get_connection_addr("test") == addr

        connections.remove_connection("test")

    def test_add_connection_no_error_with_user(self):
        add_connection = connections.add_connection

        addr = {"host": "localhost", "port": "19530", "user": "_user"}

        add_connection(test=addr)
        assert connections.get_connection_addr("test") == addr

        add_connection(default=addr)
        assert connections.get_connection_addr("default") == addr

        connections.remove_connection("test")
        connections.remove_connection("default")

    def test_add_connection_raise_NoHostPort(self, addr):
        add_connection = connections.add_connection

        host = addr.pop("host")
        port = addr.pop("port")
        LOGGER.info(f"Address: {addr}")
        with pytest.raises(MilvusException):
            add_connection(test=addr)

        addr["host"] = host
        LOGGER.info(f"Address: {addr}")
        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=addr)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "connection configuration must contain" in excinfo.value.message
        assert 0 == excinfo.value.code

        host = addr.pop("host")
        addr["port"] = port
        LOGGER.info(f"Address: {addr}")
        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=addr)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "connection configuration must contain" in excinfo.value.message
        assert 0 == excinfo.value.code

    def test_add_connection_raise_HostType(self, invalid_host):
        add_connection = connections.add_connection

        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=invalid_host)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Type of 'host' must be str." in excinfo.value.message
        assert 0 == excinfo.value.code

    def test_add_connection_raise_PortType(self, invalid_port):
        add_connection = connections.add_connection

        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=invalid_port)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Type of 'port' must be str" in excinfo.value.message
        assert 0 == excinfo.value.code


@pytest.mark.skip("to remove")
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
