from unittest import mock
from urllib import parse

import pytest
from pymilvus import DefaultConfig, MilvusException, connections
from pymilvus.client.call_context import CallContext
from pymilvus.exceptions import ConnectionNotExistException, ErrorCode

from .conftest import GRPC_PREFIX


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

    def test_connect_with_default_config(self, mock_grpc_connect, mock_grpc_close):
        alias = "default"
        connections.remove_connection(alias)
        connections.add_connection(default={"address": "localhost:19530", "user": ""})

        assert connections.has_connection(alias) is False
        assert connections.get_connection_addr(alias) == {"address": "localhost:19530", "user": ""}

        connections.connect(keep_alive=False)

        assert connections.has_connection(alias) is True
        assert connections.get_connection_addr(alias) == {
            "address": "localhost:19530",
            "user": "",
            "db_name": "default",
        }

    def test_connect_new_alias_with_configs(self, mock_grpc_connect, mock_grpc_close):
        alias = "exist"
        addr = {"address": "localhost:19530"}

        assert connections.has_connection(alias) is False
        assert connections.get_connection_addr(alias) == {}

        connections.connect(alias, **addr, keep_alive=False)

        assert connections.has_connection(alias) is True
        a = connections.get_connection_addr(alias)
        a.pop("user")
        a.pop("db_name", None)
        assert a == addr

    @pytest.mark.parametrize(
        "partial_config",
        [
            {"port": "19530"},
            {"host": "localhost"},
        ],
    )
    def test_connect_new_alias_partial_config(
        self, partial_config, mock_grpc_connect, mock_grpc_close
    ):
        alias = "partial"
        assert connections.has_connection(alias) is False

        connections.connect(alias, **partial_config, keep_alive=False)

        assert connections.has_connection(alias) is True
        assert connections.get_connection_addr(alias) == {
            "address": "localhost:19530",
            "user": "",
            "db_name": "default",
        }

    def test_connect_new_alias_with_no_config(self):
        alias = "no_config_alias"

        assert connections.has_connection(alias) is False
        with pytest.raises(MilvusException) as excinfo:
            connections.connect(alias, keep_alive=False)

        assert "You need to pass in the configuration" in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR

    @pytest.mark.parametrize(
        "uri",
        [
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
    def test_connect_with_uri(self, uri, mock_grpc_close):
        alias = "uri_connect"

        with mock.patch(f"{GRPC_PREFIX}._setup_grpc_channel", return_value=None), mock.patch(
            f"{GRPC_PREFIX}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, **uri, keep_alive=False)

        assert connections.has_connection(alias) is True

    @pytest.mark.parametrize(
        "uri",
        [
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
    def test_add_connection_then_connect(self, uri, mock_grpc_close):
        alias = "add_then_connect"

        connections.add_connection(**{alias: uri})
        addr1 = connections.get_connection_addr(alias)

        with mock.patch(f"{GRPC_PREFIX}._setup_grpc_channel", return_value=None), mock.patch(
            f"{GRPC_PREFIX}._wait_for_channel_ready", return_value=None
        ):
            connections.connect(alias, keep_alive=False)

        addr2 = connections.get_connection_addr(alias)
        addr2.pop("db_name", None)
        assert addr1 == addr2


class TestConnectNoneCredentials:
    """Passing None for user/password/token should not produce 'None' string."""

    def test_connect_with_none_user(self, mock_grpc_connect, mock_grpc_close):
        alias = "none_user_test"
        connections.remove_connection(alias)
        connections.connect(
            alias,
            uri="http://localhost:19530",
            user=None,
            password="testpass",
            keep_alive=False,
        )
        addr = connections.get_connection_addr(alias)
        assert addr.get("user", "") != "None"
        assert addr.get("user", "") == ""

    def test_connect_with_none_token(self, mock_grpc_connect, mock_grpc_close):
        alias = "none_token_test"
        connections.remove_connection(alias)
        connections.connect(
            alias,
            uri="http://localhost:19530",
            user="testuser",
            password="testpass",
            token=None,
            keep_alive=False,
        )
        # Verify user is correct and not "None"
        addr = connections.get_connection_addr(alias)
        assert addr.get("user", "") == "testuser"


class TestAddConnection:
    @pytest.mark.parametrize(
        "host_port",
        [
            {"host": "localhost", "port": "19530"},
            {"host": "localhost", "port": "19531"},
            {"host": "localhost", "port": "19530", "random": "useless"},
        ],
    )
    def test_add_connection_no_error(self, host_port):
        connections.add_connection(test=host_port)
        assert (
            connections.get_connection_addr("test").get("address")
            == f"{host_port['host']}:{host_port['port']}"
        )

    def test_add_connection_no_error_with_user(self, mock_grpc_close):
        host_port = {"host": "localhost", "port": "19530", "user": "_user"}

        connections.add_connection(test=host_port)
        config = connections.get_connection_addr("test")
        assert config.get("address") == f"{host_port['host']}:{host_port['port']}"
        assert config.get("user") == host_port["user"]

        connections.add_connection(default=host_port)
        config = connections.get_connection_addr("default")
        assert config.get("address") == f"{host_port['host']}:{host_port['port']}"
        assert config.get("user") == host_port["user"]

    @pytest.mark.parametrize(
        "invalid_host",
        [
            {"host": None, "port": "19530"},
            {"host": 1, "port": "19531"},
            {"host": 1.0, "port": "19530", "random": "useless"},
        ],
    )
    def test_add_connection_raise_host_type(self, invalid_host):
        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(test=invalid_host)

        assert "Type of 'host' must be str." in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR

    @pytest.mark.parametrize(
        "invalid_port",
        [
            {"host": "localhost", "port": None},
            {"host": "localhost", "port": 1.0},
            {"host": "localhost", "port": b"19530", "random": "useless"},
        ],
    )
    def test_add_connection_raise_port_type(self, invalid_port):
        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(test=invalid_port)

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
        alias = "addr_test"
        connections.add_connection(**{alias: valid_addr})
        assert connections.get_connection_addr(alias).get("address") == valid_addr.get("address")

    @pytest.mark.parametrize(
        "invalid_addr",
        [
            {"address": "127.0.0.1"},
            {"address": "19530"},
        ],
    )
    def test_add_connection_address_invalid(self, invalid_addr):
        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(test=invalid_addr)

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
        alias = "uri_test"
        connections.add_connection(**{alias: valid_uri})
        addr = connections.get_connection_addr(alias)
        host, port = addr["address"].split(":")
        assert host in valid_uri["uri"] or host in DefaultConfig.DEFAULT_HOST
        assert port in valid_uri["uri"] or port in DefaultConfig.DEFAULT_PORT

    @pytest.mark.parametrize(
        "invalid_uri",
        [
            {"uri": "http://"},
            {"uri": None},
            {"uri": -1},
        ],
    )
    def test_add_connection_uri_invalid(self, invalid_uri):
        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(test=invalid_uri)

        assert "Illegal uri" in excinfo.value.message
        assert excinfo.value.code == ErrorCode.UNEXPECTED_ERROR


class TestIssues:
    def test_issue_1196(self, mock_grpc_connect, mock_grpc_close):
        """Verify connect with secure=True stores correct config (issue #1196)."""
        alias = "issue_1196"

        connections.connect(
            alias=alias,
            host="localhost",
            port="19531",
            user="root",
            password=12345,
            secure=True,
            keep_alive=False,
        )
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

        connections.connect("default", user="root", password="12345", secure=True, keep_alive=False)
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
            # Mixed: explicit db_name takes precedence over URI path
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
    def test_issue_2670_2727(self, uri, db_name, expected_db_name, mock_grpc_close):
        """Verify db_name from URI path vs explicit db_name (issues #2670, #2727)."""
        alias = f"test_2670_2727_{uri.replace('://', '_').replace('/', '_')}_{db_name}"

        with mock.patch(f"{GRPC_PREFIX}.__init__", return_value=None) as mock_init, mock.patch(
            f"{GRPC_PREFIX}._wait_for_channel_ready", return_value=None
        ):
            config = {"alias": alias, "uri": uri, "db_name": db_name}
            connections.connect(**config, keep_alive=False)

            mock_init.assert_called_once()
            actual_db_name = mock_init.call_args.kwargs.get("db_name", "default")

            assert actual_db_name == expected_db_name, (
                f"Expected db_name to be '{expected_db_name}', "
                f"but got '{actual_db_name}' for uri='{uri}' and db_name='{db_name}'"
            )


class TestParseAddressFromUri:
    """Cover exception paths in __parse_address_from_uri (lines 95-98, 116-117, 141)."""

    def test_uri_with_empty_netloc_raises(self):
        """URI with empty netloc (e.g. 'http://') should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(test={"uri": "http://"})
        assert "Illegal uri" in excinfo.value.message

    def test_uri_port_out_of_range_raises(self):
        """Port number >= 65535 should raise MilvusException or ValueError."""
        with pytest.raises((MilvusException, ValueError)):
            connections.add_connection(test={"uri": "http://localhost:99999"})

    def test_uri_non_string_raises(self):
        """Non-string URI should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(test={"uri": -1})
        assert "Illegal uri" in excinfo.value.message

    def test_uri_none_raises(self):
        """None URI should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(test={"uri": None})
        assert "Illegal uri" in excinfo.value.message


class TestGetFullAddress:
    """Cover __get_full_address paths (lines 213, 223)."""

    def test_unix_socket_uri(self, mock_grpc_connect, mock_grpc_close):
        """unix: URI should be returned as-is without parsing (line 213)."""
        alias = "unix_test"
        connections.connect(alias, uri="unix:/tmp/milvus.sock", keep_alive=False)
        config = connections.get_connection_addr(alias)
        assert config.get("address") == "unix:/tmp/milvus.sock"

    def test_add_connection_with_unix_socket_uri(self):
        """unix: URI through add_connection should be stored directly."""
        connections.add_connection(unix_alias={"uri": "unix:/var/run/milvus.sock"})
        config = connections.get_connection_addr("unix_alias")
        assert config.get("address") == "unix:/var/run/milvus.sock"


class TestAddConnectionWithExistingHandler:
    """Cover add_connection when handler already exists with a different address (line 185)."""

    def test_add_connection_existing_handler_different_address(
        self, mock_grpc_connect, mock_grpc_close
    ):
        """Should raise ConnectionConfigException if alias already has a handler with
        a different address."""
        alias = "handler_test"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        assert connections.has_connection(alias) is True

        with pytest.raises(MilvusException) as excinfo:
            connections.add_connection(**{alias: {"address": "otherhost:19531"}})
        assert "already creating connections" in excinfo.value.message

    def test_add_connection_existing_handler_same_address(self, mock_grpc_connect, mock_grpc_close):
        """Should succeed if alias already has a handler with the same address."""
        alias = "handler_same"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        # This should not raise
        connections.add_connection(**{alias: {"address": "localhost:19530"}})
        config = connections.get_connection_addr(alias)
        assert config.get("address") == "localhost:19530"


class TestDisconnect:
    """Cover disconnect and related methods (lines 236, 242-246, 249-250, 259)."""

    def test_disconnect_non_string_alias_raises(self):
        """disconnect with non-string alias should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections.disconnect(123)
        assert "Alias should be string" in excinfo.value.message

    def test_disconnect_unknown_alias_no_error(self):
        """disconnect with unknown alias should not raise."""
        connections.disconnect("nonexistent_alias")

    def test_disconnect_connected_alias(self, mock_grpc_connect, mock_grpc_close):
        """disconnect should remove handler and call close."""
        alias = "disc_test"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        assert connections.has_connection(alias) is True

        connections.disconnect(alias)
        assert connections.has_connection(alias) is False
        mock_grpc_close.assert_called()

    def test_remove_connection_non_string_alias_raises(self):
        """remove_connection with non-string alias should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections.remove_connection(42)
        assert "Alias should be string" in excinfo.value.message

    @pytest.mark.asyncio
    async def test_async_disconnect_non_string_alias_raises(self):
        """async_disconnect with non-string alias should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            await connections.async_disconnect(123)
        assert "Alias should be string" in excinfo.value.message

    @pytest.mark.asyncio
    async def test_async_disconnect_connected_alias(self, mock_grpc_connect):
        """async_disconnect should pop handler and call close."""
        alias = "async_disc"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        assert connections.has_connection(alias) is True

        handler = connections._alias_handlers[alias]
        with mock.patch.object(handler, "close", new_callable=mock.AsyncMock) as mock_close:
            await connections.async_disconnect(alias)
            mock_close.assert_awaited_once()
        assert connections.has_connection(alias) is False

    @pytest.mark.asyncio
    async def test_async_remove_connection(self, mock_grpc_connect):
        """async_remove_connection should remove handler and config."""
        alias = "async_remove"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        assert alias in connections._alias_config

        handler = connections._alias_handlers[alias]
        with mock.patch.object(handler, "close", new_callable=mock.AsyncMock):
            await connections.async_remove_connection(alias)
        assert connections.has_connection(alias) is False
        assert alias not in connections._alias_config


class TestConnectMilvusLite:
    """Cover the milvus-lite path in connect (lines 337-362)."""

    def test_uri_not_ending_with_db_raises(self):
        """URI that doesn't match schemes and doesn't end with .db should raise."""
        with pytest.raises(MilvusException) as excinfo:
            connections.connect("lite_test", uri="/tmp/data.txt")
        assert "illegal" in excinfo.value.message.lower()

    def test_parent_dir_not_exist_raises(self):
        """URI ending with .db but parent dir doesn't exist should raise."""
        with pytest.raises(MilvusException) as excinfo:
            connections.connect("lite_test", uri="/nonexistent_dir_xyz/test.db")
        assert "not exists" in excinfo.value.message or "not exist" in excinfo.value.message

    def test_milvus_lite_import_fails_raises(self, tmp_path):
        """When milvus_lite import fails, should raise ConnectionConfigException."""
        db_path = str(tmp_path / "test.db")
        with mock.patch.dict(
            "sys.modules", {"milvus_lite": None, "milvus_lite.server_manager": None}
        ):
            with pytest.raises(MilvusException) as excinfo:
                connections.connect("lite_test", uri=db_path)
            assert "milvus-lite" in excinfo.value.message.lower()

    def test_milvus_lite_server_returns_none_raises(self, tmp_path):
        """When server_manager returns None URI, should raise ConnectionConfigException."""
        db_path = str(tmp_path / "test.db")
        mock_manager = mock.MagicMock()
        mock_manager.start_and_get_uri.return_value = None
        mock_module = mock.MagicMock()
        mock_module.server_manager_instance = mock_manager

        with mock.patch.dict(
            "sys.modules",
            {
                "milvus_lite": mock.MagicMock(),
                "milvus_lite.server_manager": mock_module,
            },
        ):
            with pytest.raises(MilvusException) as excinfo:
                connections.connect("lite_test", uri=db_path)
            assert "Open local milvus failed" in excinfo.value.message

    def test_milvus_lite_success(self, tmp_path, mock_grpc_connect, mock_grpc_close):
        """When milvus_lite is available and returns a URI, connect should succeed."""
        db_path = str(tmp_path / "test.db")
        mock_manager = mock.MagicMock()
        mock_manager.start_and_get_uri.return_value = "http://127.0.0.1:19530"
        mock_module = mock.MagicMock()
        mock_module.server_manager_instance = mock_manager

        with mock.patch.dict(
            "sys.modules",
            {
                "milvus_lite": mock.MagicMock(),
                "milvus_lite.server_manager": mock_module,
            },
        ):
            connections.connect("lite_ok", uri=db_path, keep_alive=False)
        assert connections.has_connection("lite_ok") is True


class TestConnectEnvUri:
    """Cover connect when _env_uri is set (lines 442-453)."""

    def test_connect_uses_env_uri_when_no_params(self, mock_grpc_connect, mock_grpc_close):
        """When _env_uri is set and no config params given, should use env URI."""
        env_addr = "envhost:19530"
        env_parsed = parse.urlparse("http://envhost:19530")

        alias = "env_test"
        # Remove existing config for this alias so we fall through to env
        connections._alias_config.pop(alias, None)
        old_env = connections._env_uri
        try:
            connections._env_uri = (env_addr, env_parsed)
            connections.connect(alias, keep_alive=False)
            assert connections.has_connection(alias) is True
            config = connections.get_connection_addr(alias)
            assert config.get("address") == env_addr
        finally:
            connections._env_uri = old_env

    def test_connect_env_uri_https_sets_secure(self, mock_grpc_connect, mock_grpc_close):
        """When _env_uri has https scheme, secure should be set to True."""
        env_addr = "secure.example.com:443"
        env_parsed = parse.urlparse("https://secure.example.com:443")

        alias = "env_secure"
        connections._alias_config.pop(alias, None)
        old_env = connections._env_uri
        try:
            connections._env_uri = (env_addr, env_parsed)
            connections.connect(alias, keep_alive=False)
            config = connections.get_connection_addr(alias)
            assert config.get("secure") is True
        finally:
            connections._env_uri = old_env


class TestConnectAliasExistsNoParams:
    """Cover connect when alias is in config but no params given (lines 456-460)."""

    def test_connect_falls_through_to_cached_config(self, mock_grpc_connect, mock_grpc_close):
        """When alias exists in config and no URI/address/host/port given, use cached config."""
        alias = "cached"
        connections.add_connection(**{alias: {"address": "cached-host:19530"}})
        connections.connect(alias, keep_alive=False)
        assert connections.has_connection(alias) is True
        config = connections.get_connection_addr(alias)
        assert config.get("address") == "cached-host:19530"


class TestConnectNonStringAlias:
    """Cover connect with non-string alias (line 397)."""

    def test_connect_non_string_alias_raises(self):
        """connect with non-string alias should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections.connect(123)
        assert "Alias should be string" in excinfo.value.message


class TestConnectDifferentAddress:
    """Cover connect with existing handler but different address (line 415)."""

    def test_connect_existing_handler_different_address_raises(
        self, mock_grpc_connect, mock_grpc_close
    ):
        """Should raise ConnectionConfigException when connecting with a different address."""
        alias = "diff_addr"
        connections.connect(alias, address="host1:19530", keep_alive=False)
        assert connections.has_connection(alias) is True

        with pytest.raises(MilvusException) as excinfo:
            connections.connect(alias, address="host2:19531", keep_alive=False)
        assert "already creating connections" in excinfo.value.message


class TestListConnections:
    """Cover list_connections (line 476)."""

    def test_list_connections_returns_tuples(self, mock_grpc_connect, mock_grpc_close):
        """list_connections should return list of (alias, handler) tuples."""
        alias = "listed"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        result = connections.list_connections()
        assert isinstance(result, list)
        aliases = [name for name, _ in result]
        assert alias in aliases

        # Handler should be not None for connected alias
        for name, handler in result:
            if name == alias:
                assert handler is not None

    def test_list_connections_handler_is_none_for_unconfigured(self):
        """Aliases with config but no handler should return None handler."""
        connections.add_connection(no_handler={"address": "localhost:19530"})
        result = connections.list_connections()
        for name, handler in result:
            if name == "no_handler":
                assert handler is None


class TestUpdateDbName:
    """Cover _update_db_name (lines 539-560)."""

    def test_update_db_name_non_string_alias_raises(self):
        """Non-string alias should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections._update_db_name(123, "new_db")
        assert "Alias should be string" in excinfo.value.message

    def test_update_db_name_non_string_db_name_raises(self):
        """Non-string db_name should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections._update_db_name("default", 123)
        assert "db_name must be a string" in excinfo.value.message

    def test_update_db_name_alias_not_in_handlers_raises(self):
        """Alias not in handlers should raise ConnectionNotExistException."""
        with pytest.raises(ConnectionNotExistException) as excinfo:
            connections._update_db_name("no_such_handler", "new_db")
        assert "should create connection first" in excinfo.value.message

    def test_update_db_name_alias_not_in_config_raises(self, mock_grpc_connect, mock_grpc_close):
        """Alias in handlers but not in config should raise ConnectionConfigException."""
        alias = "handler_only"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        # Forcibly remove from config but keep in handlers
        connections._alias_config.pop(alias)

        with pytest.raises(MilvusException) as excinfo:
            connections._update_db_name(alias, "new_db")
        assert "not bound" in excinfo.value.message

    def test_update_db_name_unbound_alias_raises(self, mock_grpc_connect, mock_grpc_close):
        """Alias created with _unbind_with_db=True should raise ConnectionConfigException
        because config has no db_name key."""
        alias = "unbound"
        connections.connect(
            alias, address="localhost:19530", _unbind_with_db=True, keep_alive=False
        )
        # The config should exist but without db_name
        assert alias in connections._alias_config
        assert "db_name" not in connections._alias_config[alias]

        with pytest.raises(MilvusException) as excinfo:
            connections._update_db_name(alias, "new_db")
        assert "not bound with a database" in excinfo.value.message

    def test_update_db_name_success(self, mock_grpc_connect, mock_grpc_close):
        """Normal update should set db_name in config."""
        alias = "update_db"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        config = connections.get_connection_addr(alias)
        assert config.get("db_name") == "default"

        connections._update_db_name(alias, "my_new_db")
        config = connections.get_connection_addr(alias)
        assert config.get("db_name") == "my_new_db"


class TestFetchHandler:
    """Cover _fetch_handler (lines 567, 571)."""

    def test_fetch_handler_non_string_alias_raises(self):
        """Non-string alias should raise ConnectionConfigException."""
        with pytest.raises(MilvusException) as excinfo:
            connections._fetch_handler(123)
        assert "Alias should be string" in excinfo.value.message

    def test_fetch_handler_nonexistent_alias_raises(self):
        """Non-existent alias should raise ConnectionNotExistException."""
        with pytest.raises(ConnectionNotExistException) as excinfo:
            connections._fetch_handler("does_not_exist")
        assert "should create connection first" in excinfo.value.message

    def test_fetch_handler_success(self, mock_grpc_connect, mock_grpc_close):
        """Should return the handler for a connected alias."""
        alias = "fetch_test"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        handler = connections._fetch_handler(alias)
        assert handler is not None


class TestGenerateCallContext:
    """Cover _generate_call_context (lines 575-589)."""

    def test_generate_call_context_returns_call_context(self, mock_grpc_connect, mock_grpc_close):
        """Should return a CallContext with db_name from config."""
        alias = "ctx_test"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        ctx = connections._generate_call_context(alias)
        assert isinstance(ctx, CallContext)

    def test_generate_call_context_with_db_name(self, mock_grpc_connect, mock_grpc_close):
        """CallContext should contain the db_name from the alias config."""
        alias = "ctx_db"
        connections.connect(alias, address="localhost:19530", db_name="test_db", keep_alive=False)
        ctx = connections._generate_call_context(alias)
        assert isinstance(ctx, CallContext)
        assert ctx._db_name == "test_db"

    def test_generate_call_context_with_request_id(self, mock_grpc_connect, mock_grpc_close):
        """CallContext should contain client_request_id if passed."""
        alias = "ctx_req"
        connections.connect(alias, address="localhost:19530", keep_alive=False)
        ctx = connections._generate_call_context(alias, client_request_id="req-123")
        assert ctx._client_request_id == "req-123"

    def test_generate_call_context_unconfigured_alias(self):
        """For an alias not in config, db_name should default to empty string."""
        ctx = connections._generate_call_context("unknown_alias")
        assert ctx._db_name == ""


class TestConnectChannelReadyFailure:
    """Cover the exception path when _wait_for_channel_ready fails (lines 388-391)."""

    def test_connect_removes_connection_on_channel_failure(self, mock_grpc_close):
        """If _wait_for_channel_ready raises, the connection should be removed."""
        alias = "channel_fail"

        with mock.patch(f"{GRPC_PREFIX}.__init__", return_value=None), mock.patch(
            f"{GRPC_PREFIX}._wait_for_channel_ready",
            side_effect=Exception("channel error"),
        ):
            with pytest.raises(Exception, match="channel error"):
                connections.connect(alias, address="localhost:19530", keep_alive=False)

        assert connections.has_connection(alias) is False


class TestConnectKeepAlive:
    """Cover the keep_alive=True path (line 388)."""

    def test_connect_with_keep_alive_registers_reconnect_handler(self, mock_grpc_close):
        """When keep_alive=True, register_reconnect_handler should be called."""
        alias = "keep_alive_test"
        with mock.patch(f"{GRPC_PREFIX}.__init__", return_value=None), mock.patch(
            f"{GRPC_PREFIX}._wait_for_channel_ready", return_value=None
        ), mock.patch(f"{GRPC_PREFIX}.register_reconnect_handler") as mock_register:
            connections.connect(alias, address="localhost:19530", keep_alive=True)
            mock_register.assert_called_once()
