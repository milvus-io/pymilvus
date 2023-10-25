# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import copy
import threading
from typing import Callable, Tuple, Union
from urllib import parse

from pymilvus.client.check import is_legal_address, is_legal_host, is_legal_port
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.utils import ZILLIZ, get_server_type
from pymilvus.exceptions import (
    ConnectionConfigException,
    ConnectionNotExistException,
    ExceptionsMessage,
)
from pymilvus.settings import Config

VIRTUAL_PORT = 443


def synchronized(func: Callable):
    """
    Decorator in order to achieve thread-safe singleton class.
    """
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


class SingleInstanceMetaClass(type):
    instance = None

    def __init__(cls, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.instance:
            return cls.instance
        cls.instance = cls.__new__(cls)
        cls.instance.__init__(*args, **kwargs)
        return cls.instance

    @synchronized
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class Connections(metaclass=SingleInstanceMetaClass):
    """Class for managing all connections of milvus.  Used as a singleton in this module."""

    def __init__(self) -> None:
        """Constructs a default milvus alias config

            default config will be read from env: MILVUS_URI and MILVUS_CONN_ALIAS
            with default value: default="localhost:19530"

         Read default connection config from environment variable: MILVUS_URI.
            Format is:
                [scheme://][<user>@<password>]host[:<port>]

                scheme is one of: http, https, or <empty>

        Examples:
            localhost
            localhost:19530
            test_user@localhost:19530
            http://test_userlocalhost:19530
            https://test_user:password@localhost:19530

        """
        self._alias = {}
        self._connected_alias = {}
        self._env_uri = None

        if Config.MILVUS_URI != "":
            address, parsed_uri = self.__parse_address_from_uri(Config.MILVUS_URI)
            self._env_uri = (address, parsed_uri)

            default_conn_config = {
                "user": parsed_uri.username if parsed_uri.username is not None else "",
                "address": address,
            }
        else:
            default_conn_config = {
                "user": "",
                "address": f"{Config.DEFAULT_HOST}:{Config.DEFAULT_PORT}",
            }

        self.add_connection(**{Config.MILVUS_CONN_ALIAS: default_conn_config})

    def __verify_host_port(self, host: str, port: Union[int, str]):
        if not is_legal_host(host):
            raise ConnectionConfigException(message=ExceptionsMessage.HostType)
        if not is_legal_port(port):
            raise ConnectionConfigException(message=ExceptionsMessage.PortType)
        if not 0 <= int(port) < 65535:
            msg = f"port number {port} out of range, valid range [0, 65535)"
            raise ConnectionConfigException(message=msg)

    def __parse_address_from_uri(self, uri: str) -> (str, parse.ParseResult):
        illegal_uri_msg = (
            "Illegal uri: [{}], expected form 'http[s]://[user:password@]example.com:12345'"
        )
        try:
            parsed_uri = parse.urlparse(uri)
        except Exception as e:
            raise ConnectionConfigException(
                message=f"{illegal_uri_msg.format(uri)}: <{type(e).__name__}, {e}>"
            ) from None

        if len(parsed_uri.netloc) == 0:
            raise ConnectionConfigException(message=f"{illegal_uri_msg.format(uri)}") from None

        host = parsed_uri.hostname if parsed_uri.hostname is not None else Config.DEFAULT_HOST
        port = parsed_uri.port if parsed_uri.port is not None else Config.DEFAULT_PORT
        addr = f"{host}:{port}"

        self.__verify_host_port(host, port)

        if not is_legal_address(addr):
            raise ConnectionConfigException(message=illegal_uri_msg.format(uri))

        return addr, parsed_uri

    def add_connection(self, **kwargs):
        """Configures a milvus connection.

        Addresses priority in kwargs: address, uri, host and port

        :param kwargs:
            * *address* (``str``) -- Optional. The actual address of Milvus instance.
                Example address: "localhost:19530"

            * *uri* (``str``) -- Optional. The uri of Milvus instance.
                Example uri: "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".

            * *host* (``str``) -- Optional. The host of Milvus instance.
                Default at "localhost", PyMilvus will fill in the default host
                if only port is provided.

            * *port* (``str/int``) -- Optional. The port of Milvus instance.
                Default at 19530, PyMilvus will fill in the default port if only host is provided.

        Example::

            connections.add_connection(
                default={"host": "localhost", "port": "19530"},
                dev1={"host": "localhost", "port": "19531"},
                dev2={"uri": "http://random.com/random"},
                dev3={"uri": "http://localhost:19530"},
                dev4={"uri": "tcp://localhost:19530"},
                dev5={"address": "localhost:19530"},
                prod={"uri": "http://random.random.random.com:19530"},
            )
        """
        for alias, config in kwargs.items():
            addr, _ = self.__get_full_address(
                config.get("address", ""),
                config.get("uri", ""),
                config.get("host", ""),
                config.get("port", ""),
            )

            if alias in self._connected_alias and self._alias[alias].get("address") != addr:
                raise ConnectionConfigException(message=ExceptionsMessage.ConnDiffConf % alias)

            alias_config = {
                "address": addr,
                "user": config.get("user", ""),
            }

            self._alias[alias] = alias_config

    def __get_full_address(
        self,
        address: str = "",
        uri: str = "",
        host: str = "",
        port: str = "",
    ) -> (str, parse.ParseResult):
        if address != "":
            if not is_legal_address(address):
                raise ConnectionConfigException(
                    message=f"Illegal address: {address}, should be in form 'localhost:19530'"
                )
            return address, None

        if uri != "":
            address, parsed = self.__parse_address_from_uri(uri)
            return address, parsed

        host = host if host != "" else Config.DEFAULT_HOST
        port = port if port != "" else Config.DEFAULT_PORT
        self.__verify_host_port(host, port)

        return f"{host}:{port}", None

    def disconnect(self, alias: str):
        """Disconnects connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        if alias in self._connected_alias:
            self._connected_alias.pop(alias).close()

    def remove_connection(self, alias: str):
        """Removes connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        self.disconnect(alias)
        self._alias.pop(alias, None)

    def connect(
        self,
        alias: str = Config.MILVUS_CONN_ALIAS,
        user: str = "",
        password: str = "",
        db_name: str = "default",
        token: str = "",
        **kwargs,
    ) -> None:
        """
        Constructs a milvus connection and register it under given alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :param kwargs:
            * *address* (``str``) -- Optional. The actual address of Milvus instance.
                Example address: "localhost:19530"
            * *uri* (``str``) -- Optional. The uri of Milvus instance.
                Example uri: "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".
            * *host* (``str``) -- Optional. The host of Milvus instance.
                Default at "localhost", PyMilvus will fill in the default host
                if only port is provided.
            * *port* (``str/int``) -- Optional. The port of Milvus instance.
                Default at 19530, PyMilvus will fill in the default port if only host is provided.
            * *secure* (``bool``) --
                Optional. Default is false. If set to true, tls will be enabled.
            * *user* (``str``) --
                Optional. Use which user to connect to Milvus instance. If user and password
                are provided, we will add related header in every RPC call.
            * *password* (``str``) --
                Optional and required when user is provided. The password corresponding to
                the user.
            * *token* (``str``) --
                Optional. Serving as the key for identification and authentication purposes.
                Whenever a token is furnished, we shall supplement the corresponding header
                to each RPC call.
            * *db_name* (``str``) --
                Optional. default database name of this connection
            * *client_key_path* (``str``) --
                Optional. If use tls two-way authentication, need to write the client.key path.
            * *client_pem_path* (``str``) --
                Optional. If use tls two-way authentication, need to write the client.pem path.
            * *ca_pem_path* (``str``) --
                Optional. If use tls two-way authentication, need to write the ca.pem path.
            * *server_pem_path* (``str``) --
                Optional. If use tls one-way authentication, need to write the server.pem path.
            * *server_name* (``str``) --
                Optional. If use tls, need to write the common name.

        :raises NotImplementedError: If handler in connection parameters is not GRPC.
        :raises ParamError: If pool in connection parameters is not supported.
        :raises Exception: If server specified in parameters is not ready, we cannot connect to
                           server.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
        """

        def connect_milvus(**kwargs):
            gh = GrpcHandler(**kwargs)

            t = kwargs.get("timeout")
            timeout = t if isinstance(t, (int, float)) else Config.MILVUS_CONN_TIMEOUT

            gh._wait_for_channel_ready(timeout=timeout)
            kwargs.pop("password")
            kwargs.pop("token", None)
            kwargs.pop("db_name", None)
            kwargs.pop("secure", None)
            kwargs.pop("db_name", "")

            self._connected_alias[alias] = gh
            self._alias[alias] = copy.deepcopy(kwargs)

        def with_config(config: Tuple) -> bool:
            return any(c != "" for c in config)

        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        # Set port if server type is zilliz cloud serverless
        uri = kwargs.get("uri")
        if uri is not None:
            server_type = get_server_type(uri)
            parsed_uri = parse.urlparse(uri)
            if server_type == ZILLIZ and parsed_uri.port is None:
                kwargs["uri"] = uri + ":" + str(VIRTUAL_PORT)

        config = (
            kwargs.pop("address", ""),
            kwargs.pop("uri", ""),
            kwargs.pop("host", ""),
            kwargs.pop("port", ""),
        )

        # Make sure passed in None doesnt break
        user, password, token = str(user) or "", str(password) or "", str(token) or ""

        # 1st Priority: connection from params
        if with_config(config):
            in_addr, parsed_uri = self.__get_full_address(*config)
            kwargs["address"] = in_addr

            if self.has_connection(alias) and self._alias[alias].get("address") != in_addr:
                raise ConnectionConfigException(message=ExceptionsMessage.ConnDiffConf % alias)

            # uri might take extra info
            if parsed_uri is not None:
                user = parsed_uri.username or user
                password = parsed_uri.password or password

                group = parsed_uri.path.split("/")
                db_name = group[1] if len(group) > 1 else db_name

                # Set secure=True if https scheme
                if parsed_uri.scheme == "https":
                    kwargs["secure"] = True

            connect_milvus(**kwargs, user=user, password=password, token=token, db_name=db_name)
            return

        # 2nd Priority, connection configs from env
        if self._env_uri is not None:
            addr, parsed_uri = self._env_uri
            kwargs["address"] = addr

            user = parsed_uri.username if parsed_uri.username is not None else ""
            password = parsed_uri.password if parsed_uri.password is not None else ""

            # Set secure=True if https scheme
            if parsed_uri.scheme == "https":
                kwargs["secure"] = True

            connect_milvus(**kwargs, user=user, password=password, db_name=db_name)
            return

        # 3rd Priority, connect to cached configs with provided user and password
        if alias in self._alias:
            connect_alias = dict(self._alias[alias].items())
            connect_alias["user"] = user
            connect_milvus(**connect_alias, password=password, db_name=db_name, **kwargs)
            return

        # No params, env, and cached configs for the alias
        raise ConnectionConfigException(message=ExceptionsMessage.ConnLackConf % alias)

    def list_connections(self) -> list:
        """List names of all connections.

        :return list:
            Names of all connections.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
        """
        return [(k, self._connected_alias.get(k, None)) for k in self._alias]

    def get_connection_addr(self, alias: str):
        """
        Retrieves connection configure by alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :return dict:
            The connection configure which of the name is alias.
            If alias does not exist, return empty dict.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
            >>> connections.get_connection_addr('test')
            {'host': 'localhost', 'port': '19530'}
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        return self._alias.get(alias, {})

    def has_connection(self, alias: str) -> bool:
        """Check if connection named alias exists.

        :param alias: The name of milvus connection
        :type  alias: str

        :return bool:
            if the connection of name alias exists.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
            >>> connections.get_connection_addr('test')
            {'host': 'localhost', 'port': '19530'}
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))
        return alias in self._connected_alias

    def _fetch_handler(self, alias: str = Config.MILVUS_CONN_ALIAS) -> GrpcHandler:
        """Retrieves a GrpcHandler by alias."""
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        conn = self._connected_alias.get(alias, None)
        if conn is None:
            raise ConnectionNotExistException(message=ExceptionsMessage.ConnectFirst)

        return conn


# Singleton Mode in Python
connections = Connections()
