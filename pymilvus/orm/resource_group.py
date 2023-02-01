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
from .connections import connections


class ResourceGroup:
    """ ResourceGroup, an abstract of node group, which impl physical resource isolation. """

    def __init__(self, name: str, using="default", **kwargs):
        """ Constructs a resource group by name
            :param name: resource group name.
            :type  name: str
        """
        self._name = name
        self._using = using
        self._kwargs = kwargs

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    @property
    def name(self):
        return self._name

    def create(self):
        """ Create a resource group
            It will success whether or not the resource group exists.

        :example:
            >>> from pymilvus import connections, utility
            >>> from pymilvus import ResourceGroup
            >>> connections.connect()
            >>> rg = ResourceGroup(name)
            >>> rg.create()
            >>> rgs = utility.list_resource_groups()
            >>> print(f"resource groups in Milvus: {rgs}")
        """
        return self._get_connection().create_resource_group(self._name)

    def drop(self):
        """ Drop a resource group
            It will success if the resource group is existed and empty, otherwise fail.

        :example:
            >>> from pymilvus import connections, utility
            >>> from pymilvus import ResourceGroup
            >>> connections.connect()
            >>> rg = ResourceGroup(name)
            >>> rg.drop()
            >>> rgs = utility.list_resource_groups()
            >>> print(f"resource groups in Milvus: {rgs}")
        """
        return self._get_connection().drop_resource_group(self._name)

    def describe(self):
        """ Drop a resource group
            It will success if the resource group is existed and empty, otherwise fail.

        :example:
            >>> from pymilvus import connections, utility
            >>> from pymilvus import ResourceGroup
            >>> connections.connect()
            >>> rg = ResourceGroup(name)
            >>> rgInfo = rg.describe()
            >>> print(f"resource group info: {rgInfo}")
        """
        return self._get_connection().describe_resource_group(self._name)
