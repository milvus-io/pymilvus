import logging
import pytest
import pymilvus
from unittest import mock

from pymilvus import connections
from pymilvus import DefaultConfig, MilvusException

LOGGER = logging.getLogger(__name__)

mock_prefix = "pymilvus.client.grpc_handler.GrpcHandler"


class TestConnect:
    """
        Connect to a connected alias will:
        - ignore and return if no configs are given
        - raise ConnectionConfigException if inconsistent configs are given

        Connect to an existing and not connected alias will:
        - connect with the existing alias config if no configs are given
        - connect with the providing configs if valid, and replace the old ones.

        Connect to a new alias will:
        - connect with the provieding configs if valid, store the new alias with these configs

    """
    @pytest.fixture(scope="function", params=[
        {},
        {"random": "not useful"},
    ])
    def no_host_port(self, request):
        return request.param

    @pytest.fixture(scope="function", params=[
        {"port": "19530"},
        {"host": "localhost"},
    ])
    def no_host_or_port(self, request):
        return request.param

    @pytest.fixture(scope="function", params=[
        {"uri": "https://127.0.0.1:19530"},
        {"uri": "tcp://127.0.0.1:19530"},
        {"uri": "http://127.0.0.1:19530"},
        {"uri": "http://example.com:80"},
    ])
    def uri(self, request):
        return request.param

    def test_connect_with_default_config(self):
        alias = "default"
        default_addr = {"address": "localhost:19530", "user": ""}

        assert connections.has_connection(alias) is False
        addr = connections.get_connection_addr(alias)

        assert addr == default_addr

        with mock.patch(f"{mock_prefix}.__init__", return_value=None):
            with mock.patch(f"{mock_prefix}._wait_for_channel_ready", return_value=None):
                connections.connect()

        assert connections.has_connection(alias) is True

        addr = connections.get_connection_addr(alias)
        assert addr == default_addr

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.disconnect(alias)

    def test_connect_new_alias_with_configs(self):
        alias = "exist"
        addr = {"address": "localhost:19530"}

        assert connections.has_connection(alias) is False
        a = connections.get_connection_addr(alias)
        assert a == {}

        with mock.patch(f"{mock_prefix}.__init__", return_value=None):
            with mock.patch(f"{mock_prefix}._wait_for_channel_ready", return_value=None):
                connections.connect(alias, **addr)

        assert connections.has_connection(alias) is True

        a = connections.get_connection_addr(alias)
        a.pop("user")
        assert a == addr

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_connect_new_alias_with_configs_NoHostOrPort(self, no_host_or_port):
        alias = "no_host_or_port"

        assert connections.has_connection(alias) is False
        a = connections.get_connection_addr(alias)
        assert a == {}

        with mock.patch(f"{mock_prefix}.__init__", return_value=None):
            with mock.patch(f"{mock_prefix}._wait_for_channel_ready", return_value=None):
                connections.connect(alias, **no_host_or_port)

        assert connections.has_connection(alias) is True
        assert connections.get_connection_addr(alias) == {"address": "localhost:19530", "user": ""}

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_connect_new_alias_with_no_config(self):
        alias = self.test_connect_new_alias_with_no_config.__name__

        assert connections.has_connection(alias) is False
        a = connections.get_connection_addr(alias)
        assert a == {}

        with pytest.raises(MilvusException) as excinfo:
            connections.connect(alias)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "You need to pass in the configuration" in excinfo.value.message
        assert -1 == excinfo.value.code

    def test_connect_with_uri(self, uri):
        alias = self.test_connect_with_uri.__name__

        with mock.patch(f"{mock_prefix}._setup_grpc_channel", return_value=None):
            with mock.patch(f"{mock_prefix}._wait_for_channel_ready", return_value=None):
                connections.connect(alias, **uri)

        addr = connections.get_connection_addr(alias)
        LOGGER.debug(addr)

        assert connections.has_connection(alias) is True

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_add_connection_then_connect(self, uri):
        alias = self.test_add_connection_then_connect.__name__

        connections.add_connection(**{alias: uri})
        addr1 = connections.get_connection_addr(alias)
        LOGGER.debug(f"addr1: {addr1}")

        with mock.patch(f"{mock_prefix}._setup_grpc_channel", return_value=None):
            with mock.patch(f"{mock_prefix}._wait_for_channel_ready", return_value=None):
                connections.connect(alias)

        addr2 = connections.get_connection_addr(alias)
        LOGGER.debug(f"addr2: {addr2}")

        assert addr1 == addr2

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)


class TestAddConnection:
    @pytest.fixture(scope="function", params=[
        {"host": "localhost", "port": "19530"},
        {"host": "localhost", "port": "19531"},
        {"host": "localhost", "port": "19530", "random": "useless"},
    ])
    def host_port(self, request):
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

    def test_add_connection_no_error(self, host_port):
        add_connection = connections.add_connection

        add_connection(test=host_port)
        assert connections.get_connection_addr("test").get("address") == f"{host_port['host']}:{host_port['port']}"

        connections.remove_connection("test")

    def test_add_connection_no_error_with_user(self):
        add_connection = connections.add_connection

        host_port = {"host": "localhost", "port": "19530", "user": "_user"}

        add_connection(test=host_port)

        config = connections.get_connection_addr("test")
        assert config.get("address") == f"{host_port['host']}:{host_port['port']}"
        assert config.get("user") == host_port['user']

        add_connection(default=host_port)
        config = connections.get_connection_addr("default")
        assert config.get("address") == f"{host_port['host']}:{host_port['port']}"
        assert config.get("user") == host_port['user']

        connections.remove_connection("test")
        connections.disconnect("default")

    def test_add_connection_raise_HostType(self, invalid_host):
        add_connection = connections.add_connection

        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=invalid_host)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Type of 'host' must be str." in excinfo.value.message
        assert -1 == excinfo.value.code

    def test_add_connection_raise_PortType(self, invalid_port):
        add_connection = connections.add_connection

        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=invalid_port)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Type of 'port' must be str" in excinfo.value.message
        assert -1 == excinfo.value.code

    @pytest.mark.parametrize("valid_addr", [
        {"address": "127.0.0.1:19530"},
        {"address": "example.com:19530"},
    ])
    def test_add_connection_address(self, valid_addr):
        alias = self.test_add_connection_address.__name__
        config = {alias: valid_addr}
        connections.add_connection(**config)

        addr = connections.get_connection_addr(alias)
        assert addr.get("address") == valid_addr.get("address")
        LOGGER.info(f"addr: {addr}")

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    @pytest.mark.parametrize("invalid_addr", [
        {"address": "127.0.0.1"},
        {"address": "19530"},
    ])
    def test_add_connection_address_invalid(self, invalid_addr):
        alias = self.test_add_connection_address_invalid.__name__
        config = {alias: invalid_addr}

        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(**config)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Illegal address" in excinfo.value.message
        assert -1 == excinfo.value.code

    @pytest.mark.parametrize("valid_uri", [
        {"uri": "http://127.0.0.1:19530"},
        {"uri": "http://localhost:19530"},
        {"uri": "http://example.com:80"},
        {"uri": "http://example.com"},
    ])
    def test_add_connection_uri(self, valid_uri):
        alias = self.test_add_connection_uri.__name__
        config = {alias: valid_uri}
        connections.add_connection(**config)

        addr = connections.get_connection_addr(alias)
        LOGGER.info(f"addr: {addr}")

        host, port = addr["address"].split(':')
        assert host in valid_uri['uri'] or host in DefaultConfig.DEFAULT_HOST
        assert port in valid_uri['uri'] or port in DefaultConfig.DEFAULT_PORT

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    @pytest.mark.parametrize("invalid_uri", [
        {"uri": "http://:19530"},
        {"uri": "localhost:19530"},
        {"uri": ":80"},
        {"uri": None},
        {"uri": -1},
    ])
    def test_add_connection_uri_invalid(self, invalid_uri):
        alias = self.test_add_connection_uri_invalid.__name__
        config = {alias: invalid_uri}

        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(**config)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Illegal uri" in excinfo.value.message
        assert -1 == excinfo.value.code


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
