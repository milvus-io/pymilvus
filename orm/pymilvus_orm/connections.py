# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

from milvus import Milvus
from .default_config import DefaultConfig


class Connections(object):
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

    def configure(self, **kwargs):
        """
        Configure the milvus connections and then create milvus connections by the passed parameters.

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
            del self._conns[k]
        self._kwargs = kwargs

    def add_connection(self, alias, conn):
        """
        Add a connection object, it will be passed through as-is.

        :param alias: The name of milvus connection
        :type alias: str

        :param conn: The milvus connection.
        :type conn: class `Milvus`
        """
        self._conns[alias] = conn

    def remove_connection(self, alias):
        """
        Remove connection from the registry. Raises ``KeyError`` if connection
        wasn't found.

        :param alias: The name of milvus connection
        :type alias: str
        """
        errors = 0
        for d in (self._conns, self._kwargs):
            try:
                del d[alias]
            except KeyError:
                errors += 1

        if errors == 2:
            raise KeyError("There is no connection with alias %r." % alias)

    def create_connection(self, alias=DefaultConfig.DEFAULT_USING, **kwargs) -> Milvus:
        """
        Construct a milvus connection and register it under given alias.

        :param alias: The name of milvus connection
        :type alias: str

        :return: A milvus connection created by the passed parameters.
        :rtype: class `Milvus`
        """
        host = kwargs.pop("host", DefaultConfig.DEFAULT_HOST)
        port = kwargs.pop("port", DefaultConfig.DEFAULT_PORT)
        handler = kwargs.pop("handler", DefaultConfig.DEFAULT_HANDLER)
        pool = kwargs.pop("pool", DefaultConfig.DEFAULT_POOL)

        conn = Milvus(host, port, handler, pool, **kwargs)
        self._conns[alias] = conn
        return conn

    def get_connection(self, alias=DefaultConfig.DEFAULT_USING) -> Milvus:
        """
        Retrieve a milvus connection by alias.

        :param alias: The name of milvus connection
        :type alias: str

        :return: A milvus connection which of the name is alias.
        :rtype: class `Milvus`
        """
        try:
            return self._conns[alias]
        except KeyError:
            pass

        # if not, try to create it
        try:
            return self.create_connection(alias, **self._kwargs[alias])
        except KeyError:
            # no connection and no kwargs to set one up
            raise KeyError("There is no connection with alias %r." % alias)


# Singleton Mode in Python

connections = Connections()
configure = connections.configure
add_connection = connections.add_connection
remove_connection = connections.remove_connection
create_connection = connections.create_connection
get_connection = connections.get_connection
