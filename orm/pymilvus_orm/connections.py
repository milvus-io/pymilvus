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

from milvus import Milvus

from .default_config import DefaultConfig
from .exceptions import ParamError


class Connections:
    """
    Connections is a class which is used to manage all connections of milvus.
    Used as a singleton in this module.
    """

    def __init__(self):
        """
        Construct a Connections object.
        """
        self._kwargs = {}
        self._conns = {}
        self._addrs = {}

    def configure(self, **kwargs):
        """
        Configure the milvus connections and then create milvus connections by the passed
        parameters.

        Example::

            connections.configure(
                default={"host": "localhost", "port": "19530"},
                dev={"host": "localhost", "port": "19531"},
            )

        This will create two milvus connections named default and dev.
        """
        for k in list(self._conns):
            # try and preserve existing client to keep the persistent connections alive
            if k in self._kwargs and kwargs.get(k, None) == self._kwargs[k]:
                continue
            self.remove_connection(alias=k)
        self._kwargs = kwargs
        for c in self._kwargs:
            self.create_connection(alias=c)

    def remove_connection(self, alias):
        """
        Remove connection from the registry. Raises ``KeyError`` if connection
        wasn't found.

        :param alias: The name of milvus connection
        :type alias: str

        :raises KeyError: If there is no connection with alias.
        """
        try:
            conn = self._conns[alias]
        except KeyError:
            raise KeyError("There is no connection with alias %r." % alias)
        errors = 0
        for d in (self._conns, self._kwargs, self._addrs):
            try:
                del d[alias]
            except KeyError:
                errors += 1

        if errors == 3:
            raise KeyError("There is no connection with alias %r." % alias)
        conn.close()

    def create_connection(self, alias=DefaultConfig.DEFAULT_USING, **kwargs) -> Milvus:
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
        >>> connections.create_connection("test", host="localhost", port="19530")
        <milvus.client.stub.Milvus object at 0x7f4045335f10>
        """
        if alias in self._conns:
            return self._conns[alias]

        if alias in self._kwargs and len(kwargs) > 0 and self._kwargs[alias] != kwargs:
            raise ParamError("passed parameters don't match the configured parameters, "
                             f"passed: {kwargs}, "
                             f"configured: {self._kwargs[alias]}")

        _using_parameters = kwargs
        if len(kwargs) <= 0 and alias in self._kwargs:
            _using_parameters = self._kwargs[alias]
        # else:
        #     self._kwargs[alias] = kwargs

        host = _using_parameters.pop("host", DefaultConfig.DEFAULT_HOST)
        port = _using_parameters.pop("port", DefaultConfig.DEFAULT_PORT)
        handler = _using_parameters.pop("handler", DefaultConfig.DEFAULT_HANDLER)
        pool = _using_parameters.pop("pool", DefaultConfig.DEFAULT_POOL)

        conn = Milvus(host, port, handler, pool, **_using_parameters)
        self._conns[alias] = conn
        self._addrs[alias] = {"host": host, "port": port}
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
        try:
            return self._conns[alias]
        except KeyError:
            raise KeyError("There is no connection with alias %r." % alias)

    def list_connections(self) -> list:
        """
        List all connections.

        :return list:
            Names of all connections.

        :example:
        >>> from pymilvus_orm import connections
        >>> connections.create_connection("test", host="localhost", port="19530")
        <milvus.client.stub.Milvus object at 0x7f4045335f10>
        >>> connections.list_connections()
        ['test']
        """

        all_alias = list(self._conns.keys())
        all_alias.extend(alias for alias in self._kwargs if alias not in all_alias)
        assert len(set(all_alias)) == len(all_alias)
        return all_alias

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
        >>> connections.create_connection("test", host="localhost", port="19530")
        <milvus.client.stub.Milvus object at 0x7f4045335f10>
        >>> connections.list_connections()
        ['test']
        >>> connections.get_connection_addr('test')
        {'host': 'localhost', 'port': '19530'}
        """
        return self._addrs.get(alias, {})


# Singleton Mode in Python

connections = Connections()
configure = connections.configure
list_connections = connections.list_connections
get_connection_addr = connections.get_connection_addr
remove_connection = connections.remove_connection
create_connection = connections.create_connection
get_connection = connections.get_connection
