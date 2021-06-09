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

import json

from .exceptions import CollectionNotExistException
from .prepare import Prepare
from .search import SearchResult
from .future import SearchResultFuture, InsertFuture


class Partition:
    # TODO(yukun): Need a place to store the description
    def __init__(self, collection, name, description="", **kwargs):
        from .collection import Collection
        if not isinstance(collection, Collection):
            raise CollectionNotExistException(0, "The type of collection must be "
                                                 "pymilvus_orm.Collection")
        self._collection = collection
        self._name = name
        self._description = description
        self._kwargs = kwargs

        conn = self._get_connection()
        has = conn.has_partition(self._collection.name, self._name)
        if not has:
            conn.create_partition(self._collection.name, self._name)

    def __repr__(self):
        return json.dumps({
            'name': self.name,
            'description': self.description,
            'num_entities': self.num_entities,
        })

    def _get_connection(self):
        return self._collection._get_connection()

    # read-only
    @property
    def description(self) -> str:
        """
        Return the description text.

        :return str: Partition description text, return when operation is successful

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.partition import Partition
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema("int64", DataType.INT64, "int64", is_parimary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", schema=schema, alias="default")
        >>> partition = Partition(collection, "test_partition", "test partition desc")
        >>> partition.description
        'test partition desc'
        """
        return self._description

    # read-only
    @property
    def name(self) -> str:
        """
        Return the partition name.

        :return str: Partition name, return when operation is successful
        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.partition import Partition
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema("int64", DataType.INT64, descrition="int64", is_parimary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", schema=schema, alias="default")
        >>> partition = Partition(collection, "test_partition", "test partition desc")
        >>> partition.name
        'test_partition'
        """
        return self._name

    # read-only
    @property
    def is_empty(self) -> bool:
        """
        Return whether the partition is empty

        :return bool: Whether the partition is empty
        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.partition import Partition
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema("int64", DataType.INT64, "int64", is_parimary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", schema=schema, alias="default")
        >>> partition = Partition(collection, "test_partition", "test partition desc")
        >>> partition.is_empty
        True
        """
        return self.num_entities == 0

    # read-only
    @property
    def num_entities(self) -> int:
        """
        Return the number of entities.

        :return int: Number of entities in this partition.
        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm import connections
        >>> from pymilvus_orm.types import DataType
        >>> connections.create_connection()
        <milvus.client.stub.Milvus object at 0x7f4d59da0be0>
        >>> field = FieldSchema("int64", DataType.INT64, is_primary=False, description="int64")
        >>> schema = CollectionSchema([field], description="collection schema has a int64 field")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> from pymilvus_orm.partition import Partition
        >>> partition = Partition(collection, "test_partition")
        >>> import random
        >>> data = [[random.randint(1,100) for _ in range(10)]]
        >>> partition.insert(data)
        >>> partition.num_entities
        10
        """
        conn = self._get_connection()
        status = conn.get_partition_stats(db_name="", collection_name=self._collection.name,
                                          partition_name=self._name)
        return status["row_count"]

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
        # TODO(yukun): If field_names is not None and not equal schema.field_names,
        #  raise Exception Not Supported,
        #  if index_names is not None, raise Exception Not Supported
        if field_names is not None and len(field_names) != len(self._collection.schema.fields):
            raise Exception("field_names should be not None or equal schema.field_names")
        if index_names is not None:
            raise Exception("index_names should be None")
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name):
            return conn.load_partitions(self._collection.name, [self._name])
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
        raise Exception("Partition doesn't exist")

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
        """
        conn = self._get_connection()
        if conn.has_partition(self._collection.name, self._name) is False:
            raise Exception("Partition doesn't exist")
        entities = Prepare.prepare_insert_data(data, self._collection.schema)
        timeout = kwargs.pop("timeout", None)
        res = conn.insert(self._collection.name, entities=entities, ids=None,
                          partition_name=self._name, timeout=timeout, orm=True, **kwargs)
        if kwargs.get("_async", False):
            return InsertFuture(res)
        return res

    def search(self, data, anns_field, params, limit, expr=None, output_fields=None, timeout=None,
               **kwargs):
        """
        Vector similarity search with an optional boolean expression as filters.

        :param data: The vectors of search data, the length of data is number of query (nq), the
                     dim of every vector in data must be equal to vector field's of collection.
        :type  data: list[list[float]]
        :param anns_field: The vector field used to search of collection.
        :type  anns_field: str
        :param params: The parameters of search, such as nprobe, etc.
        :type  params: dict
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
              SearchResultFuture object; otherwise, method returns results from server directly.
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
        """
        conn = self._get_connection()
        res = conn.search_with_expression(self._collection.name, data, anns_field, params, limit,
                                          expr, [self._name], output_fields, timeout, **kwargs)
        if kwargs.get("_async", False):
            return SearchResultFuture(res)
        return SearchResult(res)

    def get(self, ids, output_field=None, timeout=None):
        """
        Retrieve multiple entities by entityID. Returns a dict that the key is entityID and
        the value is entity. If entityID not found in the collection,
        it's value in the result will be None.

        :param ids: A list of entityID
        :type  ids: list[int]

        :param output_fields: A list of fields to return
        :type  output_fields: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :return: A dict that contains all results
        :rtype: dict

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        conn = self._get_connection()
        res = conn.get(self._collection.name, ids, output_field, [self._name], timeout)
        return res
