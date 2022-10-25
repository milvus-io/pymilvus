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

import os
import copy
import re
import threading
from urllib import parse
from typing import Tuple

from ..client.check import is_legal_host, is_legal_port, is_legal_address
from ..client.grpc_handler import GrpcHandler

from .default_config import DefaultConfig, ENV_CONNECTION_CONF
from ..exceptions import ExceptionsMessage, ConnectionConfigException, ConnectionNotExistException


def synchronized(func):
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

    def __init__(cls, *args, **kwargs):
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
    """ Class for managing all connections of milvus.  Used as a singleton in this module.  """

    def __init__(self):
        """ Constructs a default milvus alias config

            default config will be read from env: MILVUS_DEFAULT_CONNECTION，
            or "localhost:19530"

        """
        self._alias = {}
        self._connected_alias = {}

        self.add_connection(default=self._read_default_config_from_os_env())

    def _read_default_config_from_os_env(self):
        """ Read default connection config from environment variable: MILVUS_DEFAULT_CONNECTION.
        Format is:
            [<user>@]host[:<port>]

            protocol is one of: http, https, tcp, or <empty>
        Examples::
            localhost
            localhost:19530
            test_user@localhost:19530
        """

        # no need to adjust http://xxx, https://xxx, tcp://xxxx,
        # because protocol is ignored
        # @see __generate_address

        conf = os.getenv(ENV_CONNECTION_CONF, "").strip()
        if not conf:
            conf = DefaultConfig.DEFAULT_HOST

        rex = re.compile(r"^(?:([^\s/\\:]+)@)?([^\s/\\:]+)(?::(\d{1,5}))?$")
        matched = rex.search(conf)

        if not matched:
            raise ConnectionConfigException(message=ExceptionsMessage.EnvConfigErr % (ENV_CONNECTION_CONF, conf))

        user, host, port = matched.groups()
        user = user or ""
        port = port or DefaultConfig.DEFAULT_PORT

        return {
            "user": user,
            "address": f"{host}:{port}"
        }

    def add_connection(self, **kwargs):
        """ Configures a milvus connection.

        Addresses priority in kwargs: address, uri, host and port

        :param kwargs:
            * *address* (``str``) -- Optional. The actual address of Milvus instance.
                Example address: "localhost:19530"

            * *uri* (``str``) -- Optional. The uri of Milvus instance.
                Example uri: "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".

            * *host* (``str``) -- Optional. The host of Milvus instance.
                Default at "localhost", PyMilvus will fill in the default host if only port is provided.

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
            addr = self.__get_full_address(
                config.get("address", ""),
                config.get("uri", ""),
                config.get("host", ""),
                config.get("port", ""))

            if alias in self._connected_alias:
                if self._alias[alias].get("address") != addr:
                    raise ConnectionConfigException(message=ExceptionsMessage.ConnDiffConf % alias)

            alias_config = {
                "address": addr,
                "user": config.get("user", ""),
            }

            self._alias[alias] = alias_config

    def __get_full_address(self, address: str = "", uri: str = "", host: str = "", port: str = "") -> str:
        if address != "":
            if not is_legal_address(address):
                raise ConnectionConfigException(message=f"Illegal address: {address}, should be in form 'localhost:19530'")
        else:
            address = self.__generate_address(uri, host, port)

        return address

    def __generate_address(self, uri: str, host: str, port: str) -> str:
        illegal_uri_msg = "Illegal uri: [{}], should be in form 'http://example.com' or 'tcp://6.6.6.6:12345'"
        if uri != "":
            try:
                parsed_uri = parse.urlparse(uri)
            except (Exception) as e:
                raise ConnectionConfigException(message=f"{illegal_uri_msg.format(uri)}: <{type(e).__name__}, {e}>") from None

            if len(parsed_uri.netloc) == 0:
                raise ConnectionConfigException(message=illegal_uri_msg.format(uri))

            addr = parsed_uri.netloc if ":" in parsed_uri.netloc else f"{parsed_uri.netloc}:{DefaultConfig.DEFAULT_PORT}"
            if not is_legal_address(addr):
                raise ConnectionConfigException(message=illegal_uri_msg.format(uri))
            return addr

        host = host if host != "" else DefaultConfig.DEFAULT_HOST
        port = port if port != "" else DefaultConfig.DEFAULT_PORT

        if not is_legal_host(host):
            raise ConnectionConfigException(message=ExceptionsMessage.HostType)
        if not is_legal_port(port):
            raise ConnectionConfigException(message=ExceptionsMessage.PortType)
        if not 0 <= int(port) < 65535:
            raise ConnectionConfigException(message=f"port number {port} out of range, valid range [0, 65535)")

        return f"{host}:{port}"

    def disconnect(self, alias: str):
        """ Disconnects connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        if alias in self._connected_alias:
            self._connected_alias.pop(alias).close()

    def remove_connection(self, alias: str):
        """ Removes connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        self.disconnect(alias)
        self._alias.pop(alias, None)

    def connect(self, alias=DefaultConfig.DEFAULT_USING, user="", password="", **kwargs):
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
                Default at "localhost", PyMilvus will fill in the default host if only port is provided.

            * *port* (``str/int``) -- Optional. The port of Milvus instance.
                Default at 19530, PyMilvus will fill in the default port if only host is provided.

            * *user* (``str``) --
                Optional. Use which user to connect to Milvus instance. If user and password
                are provided, we will add related header in every RPC call.
            * *password* (``str``) --
                Optional and required when user is provided. The password corresponding to
                the user.
            * *secure* (``bool``) --
                Optional. Default is false. If set to true, tls will be enabled.
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
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        def connect_milvus(**kwargs):
            gh = GrpcHandler(**kwargs)

            t = kwargs.get("timeout")
            timeout = t if isinstance(t, int) else DefaultConfig.DEFAULT_CONNECT_TIMEOUT

            gh._wait_for_channel_ready(timeout=timeout)
            kwargs.pop('password')
            kwargs.pop('secure', None)

            self._connected_alias[alias] = gh
            self._alias[alias] = copy.deepcopy(kwargs)

        def with_config(config: Tuple) -> bool:
            for c in config:
                if c != "":
                    return True

            return False

        config = (
            kwargs.pop("address", ""),
            kwargs.pop("uri", ""),
            kwargs.pop("host", ""),
            kwargs.pop("port", "")
        )

        if with_config(config):
            in_addr = self.__get_full_address(*config)
            kwargs["address"] = in_addr

            if self.has_connection(alias):
                if self._alias[alias].get("address") != in_addr:
                    raise ConnectionConfigException(message=ExceptionsMessage.ConnDiffConf % alias)

            connect_milvus(**kwargs, user=user, password=password)

        else:
            if alias not in self._alias:
                raise ConnectionConfigException(message=ExceptionsMessage.ConnLackConf % alias)

            connect_alias = dict(self._alias[alias].items())
            connect_alias["user"] = user
            connect_milvus(**connect_alias, password=password, **kwargs)

    def list_connections(self) -> list:
        """ List names of all connections.

        :return list:
            Names of all connections.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
            // TODO [('default', None), ('test', <pymilvus.client.grpc_handler.GrpcHandler object at 0x7f05003f3e80>)]
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
            [('default', None), ('test', <pymilvus.client.grpc_handler.GrpcHandler object at 0x7f4045335f10>)]
            >>> connections.get_connection_addr('test')
            {'host': 'localhost', 'port': '19530'}
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        return self._alias.get(alias, {})

    def has_connection(self, alias: str) -> bool:
        """ Check if connection named alias exists.

        :param alias: The name of milvus connection
        :type  alias: str

        :return bool:
            if the connection of name alias exists.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
            [('default', None), ('test', <pymilvus.client.grpc_handler.GrpcHandler object at 0x7f4045335f10>)]
            >>> connections.get_connection_addr('test')
            {'host': 'localhost', 'port': '19530'}
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))
        return alias in self._connected_alias

    def _fetch_handler(self, alias=DefaultConfig.DEFAULT_USING) -> GrpcHandler:
        """ Retrieves a GrpcHandler by alias. """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        conn = self._connected_alias.get(alias, None)
        if conn is None:
            raise ConnectionNotExistException(message=ExceptionsMessage.ConnectFirst)

        return conn


# Singleton Mode in Python
connections = Connections()
