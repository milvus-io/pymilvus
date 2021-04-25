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

import copy


class Index(object):
    def __init__(self, collection, field_name, index_params, name="", **kwargs):
        """
        Create index on a specified column according to the index parameters.

        :param collection: The collection of index
        :type  collection: Collection

        :param name: The name of index
        :type  name: str

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param index_params: Indexing parameters.
        :type  index_params: dict
        """
        from .collection import Collection
        self._collection = collection
        self._name = name
        self._field_name = field_name
        self._index_params = index_params
        self._kwargs = kwargs

        conn = self._get_connection()
        index = conn.describe_index(self._collection.name, self._field_name)
        if index is None:
            conn.create_index(self._collection.name, self._field_name, self._index_params)
        else:
            if self._index_params != index["params"]:
                raise Exception("The index already exists, but the index params is not the same as the passed in")

    def _get_connection(self):
        return self._collection._get_connection()

    # read-only
    @property
    def name(self) -> str:
        """
        Return the index name.

        :return: The name of index
        :rtype:  str
        """
        return self._name

    # read-only
    @property
    def params(self) -> dict:
        """
        Return the index params.

        :return: Index parameters
        :rtype:  dict
        """
        return copy.deepcopy(self._index_params)

    # read-only
    @property
    def collection_name(self) -> str:
        """
        Return corresponding collection name.

        :return: Corresponding collection name.
        :rtype:  str
        """
        return self._collection.name

    @property
    def field_name(self) -> str:
        """
        Return corresponding column name.

        :return: Corresponding column name.
        :rtype:  str
        """
        return self._field_name

    def drop(self, **kwargs):
        """
        Drop index and its corresponding index files.
        """
        conn = self._get_connection()
        if conn.describe_index(self._collection.name, self._field_name) is None:
            raise Exception("Index doesn't exist")
        conn.drop_index(self._collection.name, self.field_name, **kwargs)
