# Copyright (C) 2019-2020 Zilliz. All rights reserved.
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

from pymilvus import Milvus

from .default_config import DefaultConfig
from .exceptions import ParamError


class SingleInstanceMetaClass(type):
    def __init__(cls, name, bases, dic):
        cls.__single_instance = None
        super().__init__(name, bases, dic)

    def __call__(cls, *args, **kwargs):
        if cls.__single_instance:
            return cls.__single_instance
        single_obj = cls.__new__(cls)
        single_obj.__init__(*args, **kwargs)
        cls.__single_instance = single_obj
        return single_obj


class Connections(metaclass=SingleInstanceMetaClass):
    """
    Connections is a class which is used to manage all connections of milvus.
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
        Configure the milvus connections and then create milvus connections by the passed
        parameters.

        Example::

            connections.add_connection(
                default={"host": "localhost", "port": "19530"},
                dev={"host": "localhost", "port": "19531"},
            )

        This will create two milvus connections named default and dev.
        """
        for k in kwargs:
            if k in self._conns:
                if self._kwargs.get(k, None) != kwargs.get(k, None):
                    raise ParamError("alias of %r already creating connections, "
                                     "but the configure is not the same as passed in." % k)
            self._kwargs[k] = kwargs.get(k, None)

    def disconnect(self, alias):
        """
        Disconnect connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if alias in self._conns:
            conn = self._conns[alias]
            conn.close()
            self._conns.pop(alias, None)

    def remove_connection(self, alias):
        """
        Remove connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        self.disconnect(alias)
        self._kwargs.pop(alias, None)

    def connect(self, alias=DefaultConfig.DEFAULT_USING, **kwargs) -> Milvus:
        """
        Construct a milvus connection and register it under given alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :return Milvus:
            A milvus connection created by the passed parameters.

        :raises NotImplementedError: If handler in connection parameters is not GRPC.
        :raises ParamError: If pool in connection parameters is not supported.
        :raises Exception: If server specific in parameters is not ready, we cannot connect to
                           server.

        :example:
        >>> from pymilvus_orm import connections
        >>> connections.connect("test", host="localhost", port="19530")
        <milvus.client.stub.Milvus object at 0x7f4045335f10>
        """
        if alias in self._conns:
            if len(kwargs) > 0 and self._kwargs[alias] != kwargs:
                raise ParamError(f"The connection named {alias} already creating, "
                                 "but passed parameters don't match the configured parameters, "
                                 f"passed: {kwargs}, "
                                 f"configured: {self._kwargs[alias]}")
            return self._conns[alias]

        if alias in self._kwargs and len(kwargs) > 0:
            self._kwargs[alias] = copy.deepcopy(kwargs)

        if alias not in self._kwargs:
            if len(kwargs) > 0:
                self._kwargs[alias] = copy.deepcopy(kwargs)
            else:
                raise ParamError("You need to pass in the configuration "
                                 "of the connection named %r" % alias)

        host = self._kwargs[alias].get("host", None)
        port = self._kwargs[alias].get("port", None)
        handler = self._kwargs[alias].get("handler", DefaultConfig.DEFAULT_HANDLER)
        pool = self._kwargs[alias].get("pool", DefaultConfig.DEFAULT_POOL)

        kwargs.pop("host", None)
        kwargs.pop("port", None)
        kwargs.pop("handler", None)
        kwargs.pop("pool", None)

        conn = Milvus(host, port, handler, pool, **kwargs)
        self._conns[alias] = conn
        return conn

    def get_connection(self, alias=DefaultConfig.DEFAULT_USING) -> Milvus:
        """
        Retrieve a milvus connection by alias.

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
        return self._conns.get(alias, None)

    def list_connections(self) -> list:
        """
        List all connections.

        :return list:
            Names of all connections.

        :example:
        >>> from pymilvus_orm import connections as conn
        >>> conn.connect("test", host="localhost", port="19530")
        <milvus.client.stub.Milvus object at 0x7f05003f3e80>
        >>> conn.list_connections()
        [('default', None), ('test', <milvus.client.stub.Milvus object at 0x7f05003f3e80>)]
        """
        return [(k, self._conns.get(k, None)) for k in self._kwargs]

    def get_connection_addr(self, alias):
        """
        Get connection configure by alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :return dict:
            The connection configure which of the name is alias.
            If alias does not exist, return empty dict.

        :example:
        >>> from pymilvus_orm import connections
        >>> connections.connect("test", host="localhost", port="19530")
        <milvus.client.stub.Milvus object at 0x7f4045335f10>
        >>> connections.list_connections()
        [('default', None), ('test', <milvus.client.stub.Milvus object at 0x7f4045335f10>)]
        >>> connections.get_connection_addr('test')
        {'host': 'localhost', 'port': '19530'}
        """
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
