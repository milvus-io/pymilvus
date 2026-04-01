import asyncio
import hashlib
from unittest.mock import patch

import pytest
from pymilvus.exceptions import ConnectionConfigException
from pymilvus.milvus_client._utils import create_connection


class TestCreateConnectionAlias:
    def test_explicit_alias_reuses_existing(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            result = create_connection("http://localhost:19530", alias="my_alias")
            assert result == "my_alias"
            mock_conns.connect.assert_not_called()

    def test_explicit_alias_creates_new(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = False
            result = create_connection("http://localhost:19530", alias="my_alias")
            assert result == "my_alias"
            mock_conns.connect.assert_called_once()

    def test_explicit_alias_overrides_kwargs(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            result = create_connection("http://someother:19530", alias="override")
            assert result == "override"


class TestCreateConnectionSyncAlias:
    def test_basic_uri_alias(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            result = create_connection("http://localhost:19530")
            assert result == "http://localhost:19530"

    def test_user_included_in_alias(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            result = create_connection("http://localhost:19530", user="alice", password="pass")
            assert "http://localhost:19530" in result
            # alias now contains md5 hash of "alice:pass", not raw username
            md5 = hashlib.new("md5", usedforsecurity=False)
            md5.update(b"alice:pass")
            assert md5.hexdigest() in result

    def test_token_md5_included_in_alias(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            token = "secrettoken"
            md5 = hashlib.new("md5", usedforsecurity=False)
            md5.update(token.encode())
            expected_hash = md5.hexdigest()
            result = create_connection("http://localhost:19530", token=token)
            assert expected_hash in result

    def test_user_takes_priority_over_token(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            result = create_connection("http://localhost:19530", user="bob", password="pw", token="tok")
            # Should use md5 of "bob:pw", not token hash
            md5 = hashlib.new("md5", usedforsecurity=False)
            md5.update(b"bob:pw")
            assert md5.hexdigest() in result

    def test_different_passwords_produce_different_aliases(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            result1 = create_connection("http://localhost:19530", user="alice", password="pass1")
            result2 = create_connection("http://localhost:19530", user="alice", password="pass2")
            assert result1 != result2

    def test_no_async_prefix_in_sync_mode(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = True
            result = create_connection("http://localhost:19530")
            assert not result.startswith("async")


class TestCreateConnectionNewConnection:
    def test_creates_connection_when_not_exists(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = False
            result = create_connection("http://localhost:19530")
            mock_conns.connect.assert_called_once()
            assert isinstance(result, str)

    def test_connect_called_with_uri(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = False
            create_connection("http://localhost:19530", user="u", password="p", db_name="mydb")
            args, kwargs = mock_conns.connect.call_args
            assert kwargs["uri"] == "http://localhost:19530"
            assert kwargs["_async"] is False
            # db_name is the 4th positional argument
            assert args[3] == "mydb"

    def test_connect_called_with_token(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = False
            create_connection("http://localhost:19530", token="mytoken")
            args, _ = mock_conns.connect.call_args
            # token is the 5th positional argument to connections.connect
            assert args[4] == "mytoken"

    def test_returns_alias_after_creating(self):
        with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
            mock_conns.has_connection.return_value = False
            result = create_connection("http://host:19530")
            assert "http://host:19530" in result


class TestCreateConnectionAsync:
    def test_no_running_loop_raises(self):
        with patch("pymilvus.milvus_client._utils.connections"):
            with patch(
                "pymilvus.milvus_client._utils.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ):
                with pytest.raises(ConnectionConfigException):
                    create_connection("http://localhost:19530", use_async=True)

    def test_async_alias_starts_with_async(self):
        async def run():
            with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
                mock_conns.has_connection.return_value = True
                result = create_connection("http://localhost:19530", use_async=True)
                assert result.startswith("async-")
                assert "loop" in result
                return result

        asyncio.run(run())

    def test_async_alias_includes_loop_id(self):
        async def run():
            with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
                mock_conns.has_connection.return_value = True
                result1 = create_connection("http://localhost:19530", use_async=True)
                result2 = create_connection("http://localhost:19530", use_async=True)
                # Same event loop → same alias
                assert result1 == result2

        asyncio.run(run())

    def test_async_creates_connection_with_async_flag(self):
        async def run():
            with patch("pymilvus.milvus_client._utils.connections") as mock_conns:
                mock_conns.has_connection.return_value = False
                create_connection("http://localhost:19530", use_async=True)
                _, kwargs = mock_conns.connect.call_args
                assert kwargs["_async"] is True

        asyncio.run(run())
