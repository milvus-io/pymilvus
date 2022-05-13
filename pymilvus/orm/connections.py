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

from ..client.grpc_handler import GrpcHandler

from .default_config import DefaultConfig
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
    """
    Class for managing all connections of milvus.
    Used as a singleton in this module.
    """

    def __init__(self):
        """ Constructs a milvus connection  """
        self._alias = {"default": {"host": DefaultConfig.DEFAULT_HOST,
                                   "port": DefaultConfig.DEFAULT_PORT,
                                   "user": ""}}

        self._connected_alias = {}

    def add_connection(self, **kwargs):
        """ Configures a milvus connection.

        Example::

            This example creates two milvus connections named `default` and `dev`.
            connections.add_connection(
                default={"host": "localhost", "port": "19530"},
                dev={"host": "localhost", "port": "19531"},
            )

        """
        for k in kwargs:
            tmp_kwargs = copy.deepcopy(kwargs.get(k, {}))
            tmp_kwargs.pop("password", None)
            tmp_kwargs["user"] = tmp_kwargs.get("user", "")

            if k in self._connected_alias:
                if self._alias.get(k, None) != tmp_kwargs:
                    raise ConnectionConfigException(0, ExceptionsMessage.ConnDiffConf % k)
            if "host" not in tmp_kwargs or "port" not in tmp_kwargs:
                raise ConnectionConfigException(0, ExceptionsMessage.NoHostPort)

            if not isinstance(tmp_kwargs["host"], str):
                raise ConnectionConfigException(0, ExceptionsMessage.HostType)
            if not isinstance(tmp_kwargs["port"], (str, int)):
                raise ConnectionConfigException(0, ExceptionsMessage.PortType)

            self._alias[k] = tmp_kwargs

    def disconnect(self, alias: str):
        """ Disconnects connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        if alias in self._connected_alias:
            self._connected_alias.pop(alias).close()

    def remove_connection(self, alias: str):
        """ Removes connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        self.disconnect(alias)
        self._alias.pop(alias, None)

    def connect(self, alias=DefaultConfig.DEFAULT_USING, user="", password="", **kwargs):
        """
        Constructs a milvus connection and register it under given alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :param kwargs:
            * *host* (``str``) --
                Required. The host of Milvus instance.
            * *port* (``str/int``) --
                Required. The port of Milvus instance.
            * *user* (``str``) --
                Optional. Use which user to connect to Milvus instance. If user and password
                are provided, we will add related header in every RPC call.
            * *password* (``str``) --
                Optional and required when user is provided. The password corresponding to
                the user.
            * *secure* (``bool``) --
                Optional. Default is false. If set to true, tls will be enabled.
            * *client_key_path* (``str``) --
                Optional. If use a self-signed certificate, need to write the client.key path.
            * *client_pem_path* (``str``) --
                Optional. If use a self-signed certificate, need to write the client.pem path.
            * *ca_pem_path* (``str``) --
                Optional. If use a self-signed certificate, need to write the ca.pem path.
            * *server_name* (``str``) --
                Optional. If use a self-signed certificate, need to write the common name.

        :raises NotImplementedError: If handler in connection parameters is not GRPC.
        :raises ParamError: If pool in connection parameters is not supported.
        :raises Exception: If server specified in parameters is not ready, we cannot connect to
                           server.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        def connect_milvus(**kwargs):
            tmp_kwargs = copy.deepcopy(kwargs)
            tmp_host = tmp_kwargs.pop("host", None)
            tmp_port = tmp_kwargs.pop("port", None)
            gh = GrpcHandler(host=str(tmp_host), port=str(tmp_port), **tmp_kwargs)
            gh._wait_for_channel_ready()

            kwargs.pop('password')
            self._connected_alias[alias] = gh
            self._alias[alias] = copy.deepcopy(kwargs)
            return

        def dup_alias_consist(cached_kw, current_kw) -> bool:
            if ("host" in current_kw and cached_kw[alias].get("host") != current_kw.get("host")) or \
               ("port" in current_kw and cached_kw[alias].get("port") != current_kw.get("port")):
                return False
            return True

        kwargs["user"] = user
        if self.has_connection(alias):
            if not dup_alias_consist(self._alias, kwargs):
                raise ConnectionConfigException(0, ExceptionsMessage.ConnDiffConf % alias)
            return

        if alias in self._alias:
            if "host" not in kwargs and "port" not in kwargs:
                connect_milvus(**self._alias.get(alias), password=password)
                return

        if "host" not in kwargs and "port" not in kwargs:
            raise ConnectionConfigException(0, ExceptionsMessage.ConnLackConf % alias)

        if "host" not in kwargs or "port" not in kwargs:
            raise ConnectionConfigException(0, ExceptionsMessage.NoHostPort)

        connect_milvus(**kwargs, password=password)

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
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

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
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))
        return alias in self._connected_alias

    def _fetch_handler(self, alias=DefaultConfig.DEFAULT_USING) -> GrpcHandler:
        """ Retrieves a GrpcHandler by alias. """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        conn = self._connected_alias.get(alias, None)
        if conn is None:
            raise ConnectionNotExistException(0, ExceptionsMessage.ConnectFirst)

        return conn


# Singleton Mode in Python

connections = Connections()
