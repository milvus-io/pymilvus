import logging
import pytest
import milvus
from unittest import mock

try:
    from pymilvus_orm import connections, Connections
    from pymilvus_orm.default_config import DefaultConfig
except ImportError:
    from os.path import dirname, abspath
    import sys

    sys.path.append(dirname(dirname(abspath(__file__))))

    from pymilvus_orm import connections, Connections
    from pymilvus_orm.default_config import DefaultConfig

LOGGER = logging.getLogger(__name__)


class TestConnections:
    @pytest.fixture(
        scope="function",
    )
    def c(self):
        return Connections()

    @pytest.fixture(
        scope="function",
    )
    def configure_params(self):
        params = {
            "default": {"host": "localhost", "port": "19530"},
            "dev": {"host": "localhost", "port": "19530"},
        }
        return params

    @pytest.fixture(
        scope="function",
    )
    def host(self):
        return "localhost"

    @pytest.fixture(
        scope="function",
    )
    def port(self):
        return "19530"

    @pytest.fixture(
        scope="function",
    )
    def params(self, host, port):
        d = {
            "host": host,
            "port": port,
        }
        return d

    def test_constructor(self, c):
        LOGGER.info(type(c))

    def test_configure(self, c, configure_params):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            c.configure(**configure_params)

            for key, _ in configure_params.items():
                conn = c.get_connection(key)

                assert isinstance(conn, milvus.Milvus)

                c.remove_connection(key)

    def test_add_connection(self, c, host, port):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            conn = milvus.Milvus(host, port)
            alias = "default"

            c.add_connection(alias, conn)

            conn_got = c.get_connection(alias)
            assert conn_got is conn

            c.remove_connection(alias)

    def test_remove_connection_without_no_connections(self, c):
        with pytest.raises(Exception):
            c.remove_connection("default")

    def test_remove_connection(self, c, host, port):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            conn = milvus.Milvus(host, port)
            alias = "default"

            c.add_connection(alias, conn)
            c.remove_connection(alias)

            with pytest.raises(Exception):
                c.get_connection(alias)

    def test_create_collection_without_param(self, c):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            alias = "default"
            c.create_connection(alias)
            conn_got = c.get_connection(alias)
            assert isinstance(conn_got, milvus.Milvus)
            c.remove_connection(alias)

    def test_create_collection(self, c, params):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            alias = "default"
            c.create_connection(alias, **params)
            conn_got = c.get_connection(alias)
            assert isinstance(conn_got, milvus.Milvus)
            c.remove_connection(alias)

    def test_get_collection_without_no_connections(self, c):
        with pytest.raises(Exception):
            c.get_connection("default")

    def test_get_collection(self, c, host, port):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            conn = milvus.Milvus(host, port)
            alias = "default"

            c.add_connection(alias, conn)

            conn_got = c.get_connection(alias)
            assert conn_got is conn

            c.remove_connection(alias)

    def test_get_collection_without_alias(self, c, host, port):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            conn = milvus.Milvus(host, port)
            alias = DefaultConfig.DEFAULT_USING

            c.add_connection(alias, conn)

            conn_got = c.get_connection()
            assert conn_got is conn

            c.remove_connection(alias)

    def test_get_collection_with_configure_without_add(self, c, configure_params):
        with mock.patch("milvus.Milvus.__init__", return_value=None):
            c.configure(**configure_params)
            for key, _ in configure_params.items():
                conn = c.get_connection(key)
                assert isinstance(conn, milvus.Milvus)
                c.remove_connection(key)
