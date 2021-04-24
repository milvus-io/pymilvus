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

from .prepare import Prepare


class Partition(object):

    def __init__(self, collection, name, description="", **kwargs):
        self._collection = collection
        self._name = name
        self._description = description
        self._kwargs = kwargs

        conn = self._get_connection()
        has = conn.has_partition(self._collection.name, self._name)
        if not has:
            conn.create_partition(self._collection.name, self._name)

    def _get_connection(self):
        return self._collection._get_connection()

    # read-only
    @property
    def description(self):
        """
        Return the description text.

        :return: Partition description text, return when operation is successful
        :rtype: str
        """
        return self._description

    # read-only
    @property
    def name(self):
        """
        Return the partition name.

        :return: Partition name, return when operation is successful
        :rtype: str
        """
        return self._name

    # read-only
    @property
    def is_empty(self):
        """
        Return whether the partition is empty

        :return: Whether the partition is empty
        :rtype: bool
        """
        conn = self._get_connection()
        return conn.has_partition(self._collection.name, self._name)

    # read-only
    @property
    def num_entities(self):
        """
        Return the number of entities.

        :return: Number of entities in this partition.
        :rtype: int
        """
        # TODO(yukun): Currently there is not way to get num_entities of a partition
        pass

    def drop(self, **kwargs):
        """
        Drop the partition, as well as its corresponding index files.
        """
        conn = self._get_connection()
        return conn.drop_partition(self._collection.name, self._name, **kwargs)

    def load(self, field_names=None, index_names=None, **kwargs):
        """
        Load the partition from disk to memory.

        :param field_names: The specified fields to load.
        :type  field_names: list[str]

        :param index_names: The specified indexes to load.
        :type  index_names: list[str]
        """
        # TODO(yukun): No field_names and index_names in load_partition api
        conn = self._get_connection()
        return conn.load_partitions(self._collection.name, [self._name])

    def release(self, **kwargs):
        """
        Release the partition from memory.
        """
        self._get_connection().release_partitions(self._collection.name, [self._name])

    def insert(self, data, **kwargs):
        """
        Insert data into partition.

        :param data: The specified data to insert, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple) object or pandas.DataFrame

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.
        """
        conn = self._get_connection()
        if isinstance(data, (list, tuple)):
            entities = Prepare.prepare_insert_data_for_list_or_tuple(data, self._collection.schema)
            timeout = kwargs.pop("timeout", None)
            return conn.insert(self._collection.name, entities, self._name, timeout=timeout, **kwargs)

    def search(self, data, anns_field, params, limit, expr=None, output_fields=None, **kwargs):
        """
        Vector similarity search with an optional boolean expression as filters.

        :param data: Data to search, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple, numpy.ndarray) object or pandas.DataFrame

        :param params: Search parameters
        :type  params: dict

        :param limit:
        :type  limit: int

        :param expr: Search expression
        :type  expr: str

        :param fields: The fields to return in the search result
        :type  fields: list[str]

        :return: Query result. QueryResult is iterable and is a 2d-array-like class, the first dimension is
                 the number of vectors to query (nq), the second dimension is the number of topk.
        :rtype: QueryResult
        """
        pass
