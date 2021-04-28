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
import json


class Partition(object):
    # TODO(yukun): Need a place to store the description
    def __init__(self, collection, name, description="", **kwargs):
        self._collection = collection
        self._name = name
        self._description = description
        self._kwargs = kwargs

        conn = self._get_connection()
        has = conn.has_partition(self._collection.name, self._name)
        if not has:
            conn.create_partition(self._collection.name, self._name)

    def __repr__(self):
        return json.dumps({'name': self.name, 'description': self.description, 'num_entities': self.num_entities})

    def _get_connection(self):
        return self._collection._get_connection()

    # read-only
    @property
    def description(self) -> str:
        """
        Return the description text.

        :return str: Partition description text, return when operation is successful
        """
        return self._description

    # read-only
    @property
    def name(self) -> str:
        """
        Return the partition name.

        :return str: Partition name, return when operation is successful
        """
        return self._name

    # read-only
    @property
    def is_empty(self) -> bool:
        """
        Return whether the partition is empty

        :return bool: Whether the partition is empty
        """
        return self.num_entities == 0

    # read-only
    @property
    def num_entities(self) -> int:
        """
        Return the number of entities.

        :return int: Number of entities in this partition.
        """
        # TODO: Need to add functions in pymilvus-distributed
        return 0
        # raise NotImplementedError

    def drop(self, **kwargs):
        """
        Drop the partition, as well as its corresponding index files.
        
        :raises PartitionNotExistException:
            When partitoin does not exist
        """
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name) is False:
            raise Exception("Partition doesn't exist")
        return conn.drop_partition(self._collection.name, self._name, **kwargs)

    def load(self, field_names=None, index_names=None, **kwargs):
        """
        Load the partition from disk to memory.

        :param field_names: The specified fields to load.
        :type  field_names: list[str]

        :param index_names: The specified indexes to load.
        :type  index_names: list[str]
        
        :raises InvalidArgumentException: 
            If argument is not valid

        """
        # TODO(yukun): If field_names is not None and not equal schema.field_names, raise Exception Not Supported,
        #  if index_names is not None, raise Exception Not Supported
        if field_names is not None and len(field_names) != len(self._collection.schema.fields):
            raise Exception("field_names should be not None or equal schema.field_names")
        if index_names is not None:
            raise Exception("index_names should be None")
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name):
            return conn.load_partitions(self._collection.name, [self._name])
        else:
            raise Exception("Partition doesn't exist")

    def release(self, **kwargs):
        """
        Release the partition from memory.

        :raises PartitionNotExistException:
            When partitoin does not exist
        """
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name):
            return conn.release_partitions(self._collection.name, [self._name])
        else:
            raise Exception("Partition doesn't exist")

    def insert(self, data, **kwargs):
        """
        Insert data into partition.

        :param data: The specified data to insert, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple) object or pandas.DataFrame

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises PartitionNotExistException:
            When partitoin does not exist
        """
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name) is False:
            raise Exception("Partition doesn't exist")
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

        :return QueryResult: Query result. QueryResult is iterable and is a 2d-array-like class, the first dimension is
                 the number of vectors to query (nq), the second dimension is the number of topk.
        """
        # TODO(DragonDriver): Vector similarity search with an optional boolean expression as filters
