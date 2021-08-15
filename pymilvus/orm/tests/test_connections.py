import copy
import logging
import pytest
import pymilvus
from unittest import mock

from pymilvus_orm import connections, Connections
from pymilvus_orm.default_config import DefaultConfig
from pymilvus_orm.exceptions import *

LOGGER = logging.getLogger(__name__)


class TestConnections:
    @pytest.fixture(scope="function")
    def c(self):
        return copy.deepcopy(Connections())

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
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            c.add_connection(**configure_params)

            for key, _ in configure_params.items():
                c.connect(key)

                conn = c.get_connection(key)

                assert isinstance(conn, pymilvus.Milvus)

                with pytest.raises(ConnectionConfigException):
                    c.add_connection(**{key: {"host": "192.168.1.1", "port": "13500"}})

                c.remove_connection(key)

    def test_remove_connection_without_no_connections(self, c):
        c.remove_connection("remove")

    def test_remove_connection(self, c, host, port):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
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
            assert isinstance(conn_got, pymilvus.Milvus)
            c.remove_connection(alias)

    def test_connect(self, c, params):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            alias = "default"
            c.connect(alias, **params)
            conn_got = c.get_connection(alias)
            assert isinstance(conn_got, pymilvus.Milvus)
            c.remove_connection(alias)

    def test_get_connection_without_no_connections(self, c):
        assert c.get_connection("get") is None

    def test_get_connection(self, c, host, port):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            alias = "default"

            c.connect(alias, host=host, port=port)

            conn_got = c.get_connection(alias)
            assert isinstance(conn_got, pymilvus.Milvus)

            c.remove_connection(alias)

    def test_get_connection_without_alias(self, c, host, port):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            alias = DefaultConfig.DEFAULT_USING

            c.connect(alias, host=host, port=port)

            conn_got = c.get_connection()
            assert isinstance(conn_got, pymilvus.Milvus)

            c.remove_connection(alias)

    def test_get_connection_with_configure_without_add(self, c, configure_params):
        with mock.patch("pymilvus.Milvus.__init__", return_value=None):
            c.add_connection(**configure_params)
            for key, _ in configure_params.items():
                c.connect(key)
                conn = c.get_connection(key)
                assert isinstance(conn, pymilvus.Milvus)
                c.remove_connection(key)

    def test_get_connection_addr(self, c, host, port):
        alias = DefaultConfig.DEFAULT_USING

        c.connect(alias, host=host, port=port)

        connection_addr = c.get_connection_addr(alias)

        assert connection_addr["host"] == host
        assert connection_addr["port"] == port
        c.remove_connection(alias)

    def test_list_connections(self, c, host, port):
        alias = DefaultConfig.DEFAULT_USING

        c.connect(alias, host=host, port=port)

        conns = c.list_connections()

        assert len(conns) == 1
        c.remove_connection(alias)
