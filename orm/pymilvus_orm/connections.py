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

from pymilvus import Milvus

from .default_config import DefaultConfig
from .exceptions import ExceptionsMessage, ConnectionConfigException


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
        """
        Construct a Connections object.
        """
        self._kwargs = {"default": {"host": DefaultConfig.DEFAULT_HOST,
                                    "port": DefaultConfig.DEFAULT_PORT}}
        self._conns = {}

    def add_connection(self, **kwargs):
        """
        Configures a milvus connection.

        Example::

            connections.add_connection(
                default={"host": "localhost", "port": "19530"},
                dev={"host": "localhost", "port": "19531"},
            )

        The above example creates two milvus connections named default and dev.
        """
        for k in kwargs:
            if k in self._conns:
                if self._kwargs.get(k, None) != kwargs.get(k, None):
                    raise ConnectionConfigException(0, ExceptionsMessage.ConnDiffConf % k)
            if "host" not in kwargs.get(k, {}) or "port" not in kwargs.get(k, {}):
                raise ConnectionConfigException(0, ExceptionsMessage.NoHostPort)

            if not isinstance(kwargs.get(k)["host"], str):
                raise ConnectionConfigException(0, ExceptionsMessage.HostType)
            if not isinstance(kwargs.get(k)["port"], (str, int)):
                raise ConnectionConfigException(0, ExceptionsMessage.PortType)

            self._kwargs[k] = kwargs.get(k, None)

    def disconnect(self, alias: str):
        """
        Disconnects connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        if alias in self._conns:
            self._conns.pop(alias).close()

    def remove_connection(self, alias: str):
        """
        Removes connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        self.disconnect(alias)
        self._kwargs.pop(alias, None)

    def connect(self, alias=DefaultConfig.DEFAULT_USING, **kwargs) -> Milvus:
        """
        Constructs a milvus connection and register it under given alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :return Milvus:
            A milvus connection created by the passed parameters.

        :raises NotImplementedError: If handler in connection parameters is not GRPC.
        :raises ParamError: If pool in connection parameters is not supported.
        :raises Exception: If server specified in parameters is not ready, we cannot connect to
                           server.

        :example:
            >>> from pymilvus_orm import connections
            >>> connections.connect("test", host="localhost", port="19530")
            <pymilvus.client.stub.Milvus object at 0x7f4045335f10>
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        def connect_milvus(**kwargs):
            tmp_kwargs = copy.deepcopy(kwargs)
            tmp_host = tmp_kwargs.pop("host", None)
            tmp_port = tmp_kwargs.pop("port", None)
            handler = tmp_kwargs.pop("handler", DefaultConfig.DEFAULT_HANDLER)
            pool = tmp_kwargs.pop("pool", DefaultConfig.DEFAULT_POOL)
            return Milvus(tmp_host, tmp_port, handler, pool, **tmp_kwargs)
        if alias in self._conns:
            if len(kwargs) > 0 and self._kwargs[alias] != kwargs:
                raise ConnectionConfigException(0, ExceptionsMessage.ConnDiffConf % alias)
            return self._conns[alias]

        if alias in self._kwargs:
            if len(kwargs) > 0:
                if "host" not in kwargs or "port" not in kwargs:
                    raise ConnectionConfigException(0, ExceptionsMessage.NoHostPort)
                conn = connect_milvus(**kwargs)
                self._kwargs[alias] = copy.deepcopy(kwargs)
                self._conns[alias] = conn
                return conn
            conn = connect_milvus(**self._kwargs[alias])
            self._conns[alias] = conn
            return conn

        if len(kwargs) > 0:
            if "host" not in kwargs or "port" not in kwargs:
                raise ConnectionConfigException(0, ExceptionsMessage.NoHostPort)
            conn = connect_milvus(**kwargs)
            self._kwargs[alias] = copy.deepcopy(kwargs)
            self._conns[alias] = conn
            return conn
        raise ConnectionConfigException(0, ExceptionsMessage.ConnLackConf % alias)

    def get_connection(self, alias=DefaultConfig.DEFAULT_USING) -> Milvus:
        """
        Retrieves a milvus connection by alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :return Milvus:
            A milvus connection which of the name is alias.

        :raises KeyError: If there is no connection with alias.
        :raises NotImplementedError: If handler in connection parameters is not GRPC.
        :raises ParamError: If pool in connection parameters is not supported.
        :raises Exception: If server specific in parameters is not ready, we cannot connect to
                           server.
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        return self._conns.get(alias, None)

    def list_connections(self) -> list:
        """
        Lists all connections.

        :return list:
            Names of all connections.

        :example:
            >>> from pymilvus_orm import connections as conn
            >>> conn.connect("test", host="localhost", port="19530")
            <pymilvus.client.stub.Milvus object at 0x7f05003f3e80>
            >>> conn.list_connections()
            [('default', None), ('test', <pymilvus.client.stub.Milvus object at 0x7f05003f3e80>)]
        """
        return [(k, self._conns.get(k, None)) for k in self._kwargs]

    def get_connection_addr(self, alias: str):
        """
        Retrieves connection configure by alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :return dict:
            The connection configure which of the name is alias.
            If alias does not exist, return empty dict.

        :example:
            >>> from pymilvus_orm import connections
            >>> connections.connect("test", host="localhost", port="19530")
            <pymilvus.client.stub.Milvus object at 0x7f4045335f10>
            >>> connections.list_connections()
            [('default', None), ('test', <pymilvus.client.stub.Milvus object at 0x7f4045335f10>)]
            >>> connections.get_connection_addr('test')
            {'host': 'localhost', 'port': '19530'}
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(0, ExceptionsMessage.AliasType % type(alias))

        return self._kwargs.get(alias, {})


# Singleton Mode in Python

connections = Connections()
add_connection = connections.add_connection
list_connections = connections.list_connections
get_connection_addr = connections.get_connection_addr
remove_connection = connections.remove_connection
connect = connections.connect
get_connection = connections.get_connection
disconnect = connections.disconnect
