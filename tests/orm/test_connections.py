import logging
import os
from unittest import mock

import pytest
from pymilvus import *
from pymilvus import DefaultConfig, MilvusException, connections
from pymilvus.exceptions import ConnectionConfigException, ErrorCode
from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClient
from pymilvus.milvus_client.milvus_client import MilvusClient
from pymilvus.orm.db import using_database

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
    - connect with the providing configs if valid, store the new alias with these configs

    """

    @pytest.fixture(
        scope="function",
        params=[
            {},
            {"random": "not useful"},
        ],
    )
    def no_host_port(self, request):
        return request.param

    @pytest.fixture(
        scope="function",
        params=[
            {"port": "19530"},
            {"host": "localhost"},
        ],
    )
    def no_host_or_port(self, request):
        return request.param

    @pytest.fixture(
        scope="function",
        params=[
            {"uri": "https://127.0.0.1:19530"},
            {"uri": "tcp://127.0.0.1:19530"},
            {"uri": "http://127.0.0.1:19530"},
            {"uri": "http://example.com:80"},
            {"uri": "http://example.com:80/database1"},
            {"uri": "https://127.0.0.1:19530/database2"},
            {"uri": "https://127.0.0.1/database3"},
            {"uri": "http://127.0.0.1/database4"},
        ],
    )
    def uri(self, request):
        return request.param

    def test_connect_with_default_config(self):
        alias = "default"
        # Reset the default connection config to its pristine state (from Connections.__init__)
        # to avoid pollution from previous tests that might have called connect() and added db_name.
        connections.remove_connection(alias)
        connections.add_connection(default={"address": "localhost:19530", "user": ""})

        default_addr = {"address": "localhost:19530", "user": "", "db_name": "default"}

        assert connections.has_connection(alias) is False
        addr = connections.get_connection_addr(alias)
        # Before connect, the default config from __init__ has no db_name
        assert addr == {"address": "localhost:19530", "user": ""}

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(keep_alive=False)

        assert connections.has_connection(alias) is True

        addr = connections.get_connection_addr(alias)
        # After connect, it has db_name: "default"
        assert addr == default_addr

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    @pytest.fixture(
        scope="function",
        params=[
            ("", {"address": "localhost:19530", "user": ""}),
            ("localhost", {"address": "localhost:19530", "user": ""}),
            ("localhost:19530", {"address": "localhost:19530", "user": ""}),
            ("abc@localhost", {"address": "localhost:19530", "user": "abc"}),
            ("milvus_host", {"address": "milvus_host:19530", "user": ""}),
            ("milvus_host:12012", {"address": "milvus_host:12012", "user": ""}),
            ("abc@milvus_host:12012", {"address": "milvus_host:12012", "user": "abc"}),
            ("abc@milvus_host", {"address": "milvus_host:19530", "user": "abc"}),
        ],
    )
    def test_connect_with_default_config_from_environment(self, env_result):
        os.environ[DefaultConfig.MILVUS_URI] = env_result[0]
        assert env_result[1] == connections._read_default_config_from_os_env()

        # use env
        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(keep_alive=False)

        expected = dict(env_result[1])
        expected["db_name"] = "default"
        assert expected == connections.get_connection_addr(DefaultConfig.MILVUS_CONN_ALIAS)

        # use param
        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(
                DefaultConfig.MILVUS_CONN_ALIAS, host="test_host", port="19999", keep_alive=False
            )

        curr_addr = connections.get_connection_addr(DefaultConfig.MILVUS_CONN_ALIAS)
        expected = dict(env_result[1])
        expected["db_name"] = "default"
        assert expected != curr_addr
        assert curr_addr == {"address": "test_host:19999", "user": "", "db_name": "default"}

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(DefaultConfig.MILVUS_CONN_ALIAS)

    def test_connect_new_alias_with_configs(self):
        alias = "exist"
        addr = {"address": "localhost:19530"}

        assert connections.has_connection(alias) is False
        a = connections.get_connection_addr(alias)
        assert a == {}

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, **addr, keep_alive=False)

        assert connections.has_connection(alias) is True

        a = connections.get_connection_addr(alias)
        a.pop("user")
        a.pop("db_name", None)
        assert a == addr

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_connect_new_alias_with_configs_NoHostOrPort(self, no_host_or_port):
        alias = "no_host_or_port"

        if connections.has_connection(alias):
            connections.remove_connection(alias)
        assert connections.has_connection(alias) is False
        a = connections.get_connection_addr(alias)
        assert a == {}

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, **no_host_or_port, keep_alive=False)

        assert connections.has_connection(alias) is True
        assert connections.get_connection_addr(alias) == {
            "address": "localhost:19530",
            "user": "",
            "db_name": "default",
        }

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_connect_new_alias_with_no_config(self):
        alias = self.test_connect_new_alias_with_no_config.__name__

        assert connections.has_connection(alias) is False
        a = connections.get_connection_addr(alias)
        assert a == {}

        with pytest.raises(MilvusException) as excinfo:
            connections.connect(alias, keep_alive=False)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "You need to pass in the configuration" in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR

    def test_connect_with_uri(self, uri):
        alias = self.test_connect_with_uri.__name__

        with mock.patch(f"{mock_prefix}._setup_grpc_channel", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, **uri, keep_alive=False)

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

        with mock.patch(f"{mock_prefix}._setup_grpc_channel", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, keep_alive=False)

        addr2 = connections.get_connection_addr(alias)
        LOGGER.debug(f"addr2: {addr2}")

        # addr1 is from add_connection (no db_name)
        # addr2 is from connect (has db_name: "default")
        addr2.pop("db_name", None)
        assert addr1 == addr2

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)


class TestAddConnection:
    @pytest.fixture(
        scope="function",
        params=[
            {"host": "localhost", "port": "19530"},
            {"host": "localhost", "port": "19531"},
            {"host": "localhost", "port": "19530", "random": "useless"},
        ],
    )
    def host_port(self, request):
        return request.param

    @pytest.fixture(
        scope="function",
        params=[
            {"host": None, "port": "19530"},
            {"host": 1, "port": "19531"},
            {"host": 1.0, "port": "19530", "random": "useless"},
        ],
    )
    def invalid_host(self, request):
        return request.param

    @pytest.fixture(
        scope="function",
        params=[
            {"host": "localhost", "port": None},
            {"host": "localhost", "port": 1.0},
            {"host": "localhost", "port": b"19530", "random": "useless"},
        ],
    )
    def invalid_port(self, request):
        return request.param

    def test_add_connection_no_error(self, host_port):
        add_connection = connections.add_connection

        add_connection(test=host_port)
        assert (
            connections.get_connection_addr("test").get("address")
            == f"{host_port['host']}:{host_port['port']}"
        )

        connections.remove_connection("test")

    def test_add_connection_no_error_with_user(self):
        add_connection = connections.add_connection

        host_port = {"host": "localhost", "port": "19530", "user": "_user"}

        add_connection(test=host_port)

        config = connections.get_connection_addr("test")
        assert config.get("address") == f"{host_port['host']}:{host_port['port']}"
        assert config.get("user") == host_port["user"]

        add_connection(default=host_port)
        config = connections.get_connection_addr("default")
        assert config.get("address") == f"{host_port['host']}:{host_port['port']}"
        assert config.get("user") == host_port["user"]

        with mock.patch(f"{mock_prefix}.close", side_effect=lambda: None):
            connections.remove_connection("test")
            connections.disconnect("default")

    def test_add_connection_raise_HostType(self, invalid_host):
        add_connection = connections.add_connection

        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=invalid_host)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Type of 'host' must be str." in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR

    def test_add_connection_raise_PortType(self, invalid_port):
        add_connection = connections.add_connection

        with pytest.raises(MilvusException) as excinfo:
            add_connection(test=invalid_port)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Type of 'port' must be str" in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR

    @pytest.mark.parametrize(
        "valid_addr",
        [
            {"address": "127.0.0.1:19530"},
            {"address": "example.com:19530"},
        ],
    )
    def test_add_connection_address(self, valid_addr):
        alias = self.test_add_connection_address.__name__
        config = {alias: valid_addr}
        connections.add_connection(**config)

        addr = connections.get_connection_addr(alias)
        assert addr.get("address") == valid_addr.get("address")
        LOGGER.info(f"addr: {addr}")

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    @pytest.mark.parametrize(
        "invalid_addr",
        [
            {"address": "127.0.0.1"},
            {"address": "19530"},
        ],
    )
    def test_add_connection_address_invalid(self, invalid_addr):
        alias = self.test_add_connection_address_invalid.__name__
        config = {alias: invalid_addr}

        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(**config)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Illegal address" in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR

    @pytest.mark.parametrize(
        "valid_uri",
        [
            {"uri": "http://127.0.0.1:19530"},
            {"uri": "http://localhost:19530"},
            {"uri": "http://example.com:80"},
            {"uri": "http://example.com"},
        ],
    )
    def test_add_connection_uri(self, valid_uri):
        alias = self.test_add_connection_uri.__name__
        config = {alias: valid_uri}
        connections.add_connection(**config)

        addr = connections.get_connection_addr(alias)
        LOGGER.info(f"addr: {addr}")

        host, port = addr["address"].split(":")
        assert host in valid_uri["uri"] or host in DefaultConfig.DEFAULT_HOST
        assert port in valid_uri["uri"] or port in DefaultConfig.DEFAULT_PORT
        LOGGER.info(f"host: {host}, port: {port}")

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    @pytest.mark.parametrize(
        "invalid_uri",
        [
            {"uri": "http://"},
            {"uri": None},
            {"uri": -1},
        ],
    )
    def test_add_connection_uri_invalid(self, invalid_uri):
        alias = self.test_add_connection_uri_invalid.__name__
        config = {alias: invalid_uri}

        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(**config)

        LOGGER.info(f"Exception info: {excinfo.value}")
        assert "Illegal uri" in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR


class TestIssues:
    def test_issue_1196(self):
        """
        >>> connections.connect(alias="default11", host="xxx.com", port=19541, user="root", password="12345", secure=True)
        >>> connections.add_connection(default={"host": "xxx.com", "port": 19541})
        >>> connections.connect("default", user="root", password="12345", secure=True)
        Traceback (most recent call last):
          File "/usr/local/lib/python3.8/dist-packages/pymilvus/client/grpc_handler.py", line 114, in _wait_for_channel_ready
            grpc.channel_ready_future(self._channel).result(timeout=3)
          File "/usr/local/lib/python3.8/dist-packages/grpc/_utilities.py", line 139, in result
            self._block(timeout)
          File "/usr/local/lib/python3.8/dist-packages/grpc/_utilities.py", line 85, in _block
            raise grpc.FutureTimeoutError()
        grpc.FutureTimeoutError
        """

        alias = self.test_issue_1196.__name__

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            config = {
                "alias": alias,
                "host": "localhost",
                "port": "19531",
                "user": "root",
                "password": 12345,
                "secure": True,
            }
            connections.connect(**config, keep_alive=False)
            config = connections.get_connection_addr(alias)
            assert config == {
                "address": "localhost:19531",
                "user": "root",
                "secure": True,
                "db_name": "default",
            }

            connections.add_connection(default={"host": "localhost", "port": 19531})
            config = connections.get_connection_addr("default")
            assert config == {"address": "localhost:19531", "user": ""}

            connections.connect(
                "default", user="root", password="12345", secure=True, keep_alive=False
            )

            config = connections.get_connection_addr("default")
            assert config == {
                "address": "localhost:19531",
                "user": "root",
                "secure": True,
                "db_name": "default",
            }

    @pytest.mark.parametrize(
        "uri, db_name, expected_db_name",
        [
            # Issue #2670: URI ending with slash should not overwrite explicit db_name
            ("http://localhost:19530/", "test_db", "test_db"),
            ("https://localhost:19530/", "production_db", "production_db"),
            ("tcp://localhost:19530/", "test_db", "test_db"),
            # Issue #2727: db_name passed in URI path should be used when no explicit db_name
            ("http://localhost:19530/test_db", "", "test_db"),
            ("http://localhost:19530/production_db", "", "production_db"),
            ("https://localhost:19530/test_db", "", "test_db"),
            # Mixed scenarios: explicit db_name takes precedence over URI path
            ("http://localhost:19530/uri_db", "explicit_db", "explicit_db"),
            ("http://localhost:19530", "test_db", "test_db"),
            ("http://localhost:19530", "", "default"),
            # Multiple path segments - only first should be used as db_name
            ("http://localhost:19530/db1/collection1", "", "db1"),
            ("http://localhost:19530/db1/collection1/", "", "db1"),
            # Empty path segments should be handled correctly
            ("http://localhost:19530//", "test_db", "test_db"),
            ("http://localhost:19530///", "test_db", "test_db"),
        ],
    )
    def test_issue_2670_2727(self, uri: str, db_name: str, expected_db_name: str):
        """
        Issue 2670:
        Test for db_name being overwritten with empty string, when the uri
        ends in a slash - e.g. http://localhost:19530/

        See: https://github.com/milvus-io/pymilvus/issues/2670

        Actual behaviour before fix: if a uri is passed ending with a slash,
            it will overwrite the db_name with an empty string.
        Expected and current behaviour: if db_name is passed explicitly,
            it should be used in the initialization of the GrpcHandler.

        Issue 2727:
        If db_name is passed as a path to the uri and not explicitly passed as an argument,
        it is not overwritten with an empty string.

        See: https://github.com/milvus-io/pymilvus/issues/2727

        Actual behaviour before fix: if db_name is passed as a path to the uri,
            it will overwrite the db_name with an empty string.
        Expected and current behaviour: if db_name is passed as a path to the uri,
            it should be used in the initialization of the GrpcHandler.
        """
        alias = f"test_2670_2727_{uri.replace('://', '_').replace('/', '_')}_{db_name}"

        with mock.patch(f"{mock_prefix}.__init__", return_value=None) as mock_init, mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            config = {"alias": alias, "uri": uri}
            # Always pass db_name parameter, even if it's an empty string
            if db_name or db_name == "":  # Pass both empty and non-empty strings
                config["db_name"] = db_name
            connections.connect(**config, keep_alive=False)

            # Verify that GrpcHandler was initialized with the correct db_name
            mock_init.assert_called_once()
            call_args = mock_init.call_args
            actual_db_name = call_args.kwargs.get("db_name", "default")

            assert actual_db_name == expected_db_name, (
                f"Expected db_name to be '{expected_db_name}', "
                f"but got '{actual_db_name}' for uri='{uri}' and db_name='{db_name}'"
            )

            # Clean up - mock the close method to avoid AttributeError
            with mock.patch(f"{mock_prefix}.close", return_value=None):
                connections.remove_connection(alias)


class TestUnbindWithDb:
    def test_connect_with_unbind_with_db_false(self):
        alias = "test_orm_alias"
        connections.remove_connection(alias)

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(
                alias,
                uri="http://localhost:19530",
                db_name="test_db",
                _unbind_with_db=False,
                keep_alive=False,
            )

        config = connections.get_connection_addr(alias)
        assert "db_name" in config
        assert config["db_name"] == "test_db"

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_connect_with_unbind_with_db_true(self):
        alias = "test_mc_alias"
        connections.remove_connection(alias)

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(
                alias,
                uri="http://localhost:19530",
                db_name="test_db",
                _unbind_with_db=True,
                keep_alive=False,
            )

        config = connections.get_connection_addr(alias)
        assert "db_name" not in config

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_update_db_name_with_bound_alias(self):
        alias = "test_bound_alias"
        connections.remove_connection(alias)

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ), mock.patch(f"{mock_prefix}.reset_db_name", return_value=None):
            connections.connect(
                alias,
                uri="http://localhost:19530",
                db_name="db1",
                _unbind_with_db=False,
                keep_alive=False,
            )

        config = connections.get_connection_addr(alias)
        assert config["db_name"] == "db1"

        connections._update_db_name(alias, "db2")
        config = connections.get_connection_addr(alias)
        assert config["db_name"] == "db2"

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_update_db_name_with_unbound_alias_raises_exception(self):
        alias = "test_unbound_alias"
        connections.remove_connection(alias)

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(
                alias,
                uri="http://localhost:19530",
                db_name="test_db",
                _unbind_with_db=True,
                keep_alive=False,
            )

        config = connections.get_connection_addr(alias)
        assert "db_name" not in config

        with pytest.raises(ConnectionConfigException) as exc_info:
            connections._update_db_name(alias, "new_db")
        assert "not bound with a database" in str(exc_info.value)
        assert "_unbind_with_db=True" in str(exc_info.value)

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_using_database_with_unbound_alias_raises_exception(self):
        alias = "test_unbound_alias_orm"
        connections.remove_connection(alias)

        with mock.patch(f"{mock_prefix}.__init__", return_value=None), mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(
                alias,
                uri="http://localhost:19530",
                db_name="test_db",
                _unbind_with_db=True,
                keep_alive=False,
            )

        with pytest.raises(ConnectionConfigException):
            using_database("new_db", using=alias)

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_milvus_client_creates_unbound_alias(self):
        alias = "test_mc_connection"
        connections.remove_connection(alias)

        mock_handler = mock.MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with mock.patch(
            "pymilvus.milvus_client._utils.connections.connect"
        ) as mock_connect, mock.patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ):
            _ = MilvusClient(uri="http://localhost:19530", db_name="test_db")
            mock_connect.assert_called_once()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs.get("_unbind_with_db") is True

    def test_async_milvus_client_creates_unbound_alias(self):
        alias = "test_amc_connection"
        connections.remove_connection(alias)

        mock_handler = mock.MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with mock.patch(
            "pymilvus.milvus_client._utils.connections.connect"
        ) as mock_connect, mock.patch(
            "pymilvus.milvus_client._utils.asyncio.get_running_loop"
        ) as mock_get_loop, mock.patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ):
            mock_get_loop.return_value = mock.MagicMock()
            _ = AsyncMilvusClient(uri="http://localhost:19530", db_name="test_db")
            mock_connect.assert_called_once()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs.get("_unbind_with_db") is True


class TestGlobalEndpointUriPreserved:
    def test_global_uri_passed_to_grpc_handler(self):
        """Global endpoint URI should be preserved in kwargs for GrpcHandler."""
        alias = "test_global_uri"
        connections.remove_connection(alias)
        global_uri = "https://glo-xxx.global-cluster.vectordb.zilliz.com"

        with mock.patch(f"{mock_prefix}.__init__", return_value=None) as mock_init, mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, uri=global_uri, token="test-token", keep_alive=False)

            _, call_kwargs = mock_init.call_args
            assert call_kwargs.get("uri") == global_uri

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)

    def test_regular_uri_not_passed_to_grpc_handler(self):
        """Regular (non-global) URI should NOT be preserved in kwargs."""
        alias = "test_regular_uri"
        connections.remove_connection(alias)
        regular_uri = "https://in01-xxx.zilliz.com"

        with mock.patch(f"{mock_prefix}.__init__", return_value=None) as mock_init, mock.patch(
            f"{mock_prefix}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, uri=regular_uri, token="test-token", keep_alive=False)

            _, call_kwargs = mock_init.call_args
            assert "uri" not in call_kwargs

        with mock.patch(f"{mock_prefix}.close", return_value=None):
            connections.remove_connection(alias)
