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

import json

from .exceptions import CollectionNotExistException, PartitionNotExistException, ExceptionsMessage
from .prepare import Prepare
from .search import SearchResult
from .mutation import MutationResult
from .future import SearchFuture, MutationFuture


class Partition:
    # TODO(yukun): Need a place to store the description
    def __init__(self, collection, name, description="", **kwargs):
        from .collection import Collection
        if not isinstance(collection, Collection):
            raise CollectionNotExistException(0, ExceptionsMessage.CollectionType)
        self._collection = collection
        self._name = name
        self._description = description
        self._kwargs = kwargs

        conn = self._get_connection()
        if kwargs.get("construct_only", False):
            return
        has = conn.has_partition(self._collection.name, self._name)
        if not has:
            conn.create_partition(self._collection.name, self._name)

    def __repr__(self):
        return json.dumps({
            'name': self.name,
            'collection_name': self._collection.name,
            'description': self.description,
        })

    def _get_connection(self):
        return self._collection._get_connection()

    @property
    def description(self) -> str:
        """
        Return the description text.

        :return str: Partition description text, return when operation is successful

        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_description", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> partition.description
            'comedy films'
        """
        return self._description

    @property
    def name(self) -> str:
        """
        Return the partition name.

        :return str: Partition name, return when operation is successful
        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_name", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> partition.name
            'comedy'
        """
        return self._name

    @property
    def is_empty(self) -> bool:
        """
        Returns whether the partition is empty

        :return bool: Whether the partition is empty
        * True: The partition is empty.
        * False: The partition is not empty.

        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_is_empty", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> partition.is_empty
            True
        """
        return self.num_entities == 0

    @property
    def num_entities(self) -> int:
        """
        Return the number of entities.

        :return int: Number of entities in this partition.

        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_num_entities", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> data = [
            ...     [i for i in range(10)],
            ...     [[float(i) for i in range(2)] for _ in range(10)],
            ... ]
            >>> partition.insert(data)
            >>> partition.num_entities
            10
        """
        conn = self._get_connection()
        conn.flush([self._collection.name])
        status = conn.get_partition_stats(db_name="", collection_name=self._collection.name,
                                          partition_name=self._name)
        return status["row_count"]

    def drop(self, **kwargs):
        """
        Drop the partition, as well as its corresponding index files.

        :raises PartitionNotExistException:
            When partitoin does not exist

        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_drop", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> partition.drop()
        """
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name) is False:
            raise PartitionNotExistException(0, ExceptionsMessage.PartitionNotExist)
        return conn.drop_partition(self._collection.name, self._name, **kwargs)

    def load(self, **kwargs):
        """
        Load the partition from disk to memory.

        :raises InvalidArgumentException:
            If argument is not valid

        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_load", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> partition.load()
        """
        # TODO(yukun): If field_names is not None and not equal schema.field_names,
        #  raise Exception Not Supported,
        #  if index_names is not None, raise Exception Not Supported
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name):
            return conn.load_partitions(self._collection.name, [self._name], **kwargs)
        raise PartitionNotExistException(0, ExceptionsMessage.PartitionNotExist)

    def release(self, **kwargs):
        """
        Release the partition from memory.

        :raises PartitionNotExistException:
            When partitoin does not exist

        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_release", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> partition.load()
            >>> partition.release()
        """
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name):
            return conn.release_partitions(self._collection.name, [self._name], **kwargs)
        raise PartitionNotExistException(0, ExceptionsMessage.PartitionNotExist)

    def insert(self, data, **kwargs):
        """
        Insert data into partition.

        :param data: The specified data to insert, the dimension of data needs to align with column
                     number
        :type  data: list-like(list, tuple) object or pandas.DataFrame

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises PartitionNotExistException:
            When partitoin does not exist

        :example:
            >>> from pymilvus_orm import connections, Collection, Partition, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_partition_insert", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> data = [
            ...     [i for i in range(10)],
            ...     [[float(i) for i in range(2)] for _ in range(10)],
            ... ]
            >>> partition.insert(data)
            >>> partition.num_entities
            10
        """
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name) is False:
            raise PartitionNotExistException(0, ExceptionsMessage.PartitionNotExist)
        entities = Prepare.prepare_insert_data(data, self._collection.schema)
        timeout = kwargs.pop("timeout", None)
        res = conn.insert(self._collection.name, entities=entities, ids=None,
                          partition_name=self._name, timeout=timeout, orm=True, **kwargs)
        if kwargs.get("_async", False):
            return MutationFuture(res)
        return MutationResult(res)

    def search(self, data, anns_field, param, limit, expr=None, output_fields=None, timeout=None,
               **kwargs):
        """
        Vector similarity search with an optional boolean expression as filters.

        :param data: The vectors of search data, the length of data is number of query (nq), the
                     dim of every vector in data must be equal to vector field's of collection.
        :type  data: list[list[float]]
        :param anns_field: The vector field used to search of collection.
        :type  anns_field: str
        :param param: The parameters of search, such as nprobe, etc.
        :type  param: dict
        :param limit: The max number of returned record, we also called this parameter as topk.
        :type  limit: int
        :param expr: The boolean expression used to filter attribute.
        :type  expr: str
        :param output_fields: The fields to return in the search result, not supported now.
        :type  output_fields: list[str]
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a
              SearchFuture object; otherwise, method returns results from server directly.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only
              takes effect when _async is set to True.

        :return: SearchResult:
            SearchResult is iterable and is a 2d-array-like class, the first dimension is
            the number of vectors to query (nq), the second dimension is the number of limit(topk).
        :rtype: SearchResult

        :raises RpcError: If gRPC encounter an error.
        :raises ParamError: If parameters are invalid.
        :raises BaseException: If the return result from server is not ok.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> connections.connect()
            <pymilvus.client.stub.Milvus object at 0x7f8579002dc0>
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_search", schema)
            >>> partition = Partition(collection, "comedy", "comedy films")
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> partition.insert(data)
            >>> partition.num_entities
            10
            >>> partition.load()
            >>> # search
            >>> search_param = {
            ...     "data": [[1.0, 1.0]],
            ...     "anns_field": "films",
            ...     "param": {"metric_type": "L2"},
            ...     "limit": 2,
            ...     "expr": "film_id > 0",
            ... }
            >>> res = partition.search(**search_param)
            >>> assert len(res) == 1
            >>> hits = res[0]
            >>> assert len(hits) == 2
            >>> print(f"- Total hits: {len(hits)}, hits ids: {hits.ids} ")
            - Total hits: 2, hits ids: [8, 5]
            >>> print(f"- Top1 hit id: {hits[0].id}, distance: {hits[0].distance}, score: {hits[0].score} ")
            - Top1 hit id: 8, distance: 0.10143111646175385, score: 0.10143111646175385
        """
        conn = self._get_connection()
        res = conn.search_with_expression(self._collection.name, data, anns_field, param, limit,
                                          expr, [self._name], output_fields, timeout, **kwargs)
        if kwargs.get("_async", False):
            return SearchFuture(res)
        return SearchResult(res)

    def query(self, expr, output_fields=None, timeout=None):
        """
        Query with a set of criteria, and results in a list of records that match the query exactly.

        :param expr: The query expression
        :type  expr: str

        :param output_fields: A list of fields to return
        :type  output_fields: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :return: A list that contains all results
        :rtype: list

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        conn = self._get_connection()
        res = conn.query(self._collection.name, expr, output_fields, [self._name], timeout)
        return res
