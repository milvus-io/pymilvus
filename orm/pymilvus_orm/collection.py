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

import pandas

from .connections import get_connection
from .schema import CollectionSchema, FieldSchema, parse_fields_from_data
from .prepare import Prepare
from .partition import Partition
from .index import Index
from .search import SearchResult
from .types import DataType
from .exceptions import *
from .future import SearchResultFuture, InsertFuture


class Collection(object):
    """
    This is a class corresponding to collection in milvus.
    """

    def __init__(self, name, data=None, schema=None, **kwargs):
        """
        Construct a collection by the name, schema and other parameters.
        Connection information is contained in kwargs.

        :param name: the name of collection
        :type name: str

        :param schema: the schema of collection
        :type schema: class `schema.CollectionSchema`

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", data=None, schema=schema, alias="default")
        >>> collection.name
        'test_collection'
        >>> collection.description
        'collection description'
        >>> collection.is_empty
        True
        >>> collection.num_entities
        0
        """
        self._name = name
        self._kwargs = kwargs
        conn = self._get_connection()
        has = conn.has_collection(self._name)
        if has:
            resp = conn.describe_collection(self._name)
            server_schema = CollectionSchema.construct_from_dict(resp)
            if schema is None:
                self._schema = server_schema
                if data is not None:
                    self.insert(data=data)
            else:
                if server_schema != schema:
                    raise SchemaNotReadyException(0, "The collection already exist, but the schema is not the same as "
                                                  "the passed in.")
                self._schema = schema
                if data is not None:
                    self.insert(data=data)

        else:
            if schema is None:
                if data is None:
                    raise SchemaNotReadyException(0, "Collection missing schema.")
                else:
                    if isinstance(data, pandas.DataFrame):
                        fields = parse_fields_from_data(data)
                        self._schema = CollectionSchema(fields=fields)
                        conn.create_collection(self._name, fields=self._schema.to_dict(), orm=True)
                        self.insert(data=data)
                    else:
                        raise SchemaNotReadyException(0, "Data of not pandas.DataFrame type should be passed into the "
                                                      "schema.")
            else:
                if isinstance(schema, CollectionSchema):
                    conn.create_collection(self._name, fields=schema.to_dict(), orm=True)
                    self._schema = schema
                    if data is not None:
                        self.insert(data=data)
                else:
                    raise SchemaNotReadyException(0, "schema type must be schema.CollectionSchema.")

    def _get_using(self):
        return self._kwargs.get("_using", "default")

    def _get_connection(self):
        return get_connection(self._get_using())

    def _check_insert_data_schema(self, data):
        """
        Check whether the data type matches the schema.
        """
        if self._schema is None:
            return False
        infer_fields = parse_fields_from_data(data)

        if len(infer_fields) != len(self._schema):
            raise DataTypeNotMatchException(0, "Column cnt not match with schema")

        for x, y in zip(infer_fields, self._schema.fields):
            if x.dtype != y.dtype:
                return False
            # todo check dim
        return True

    def _check_schema(self):
        if self._schema is None:
            raise SchemaNotReadyException(0, "Schema is not ready")

    def _get_vector_field(self) -> str:
        for field in self._schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR or field.dtype == DataType.BINARY_VECTOR:
                return field.name

    @property
    def schema(self) -> CollectionSchema:
        """
        Return the schema of collection.

        :return schema.CollectionSchema:
            Schema of collection.
        """
        return self._schema

    @property
    def description(self) -> str:
        """
        Return the description text about the collection.

        :return str:
            Collection description text, return when operation is successful.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="test get description")
        >>> collection = Collection(name="test_collection", schema=schema, alias="default")
        >>> collection.description
        'test get description'
        """

        return self._schema.description

    @property
    def name(self) -> str:
        """
        Return the collection name.

        :return str:
            Collection name, return when operation is successful.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="test get collection name")
        >>> collection = Collection(name="test_collection", schema=schema, alias="default")
        >>> collection.name
        'test_collection'
        """
        return self._name

    # read-only
    @property
    def is_empty(self) -> bool:
        """
        Return whether the collection is empty.
        This method need to call `num_entities <#pymilvus_orm.Collection.num_entities>`_.

        :return bool:
            Whether the collection is empty.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="test collection is empty")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> collection.is_empty
        True
        >>> data = [[1,2,3,4]]
        >>> collection.insert(data)
        [424769928069057860, 424769928069057861, 424769928069057862, 424769928069057863]
        >>> collection.is_empty
        False
        """
        return self.num_entities == 0

    # read-only
    @property
    def num_entities(self) -> int:
        """
        Return the number of entities.

        :return int:
            Number of entities in this collection.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="get collection entities num")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> collection.num_entities
        0
        >>> data = [[1,2,3,4]]
        >>> collection.insert(data)
        [424769928069057860, 424769928069057861, 424769928069057862, 424769928069057863]
        >>> collection.num_entities
        4
        """
        conn = self._get_connection()
        status = conn.get_collection_stats(db_name="", collection_name=self._name)
        return status["row_count"]

    @property
    def primary_field(self) -> FieldSchema:
        """
        Return the primary field of collection.

        :return schema.FieldSchema:
            The primary field of collection.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=True)
        >>> schema = CollectionSchema(fields=[field], description="get collection entities num")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> collection.primary_field
        <pymilvus_orm.schema.FieldSchema object at 0x7f64f6a3cc40>
        >>> collection.primary_field.name
        'int64'
        """
        return self._schema.primary_field

    def drop(self, **kwargs):
        """
        Drop the collection, as well as its corresponding index files.

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="drop collection")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import pandas as pd
        >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))
        >>> data = pd.DataFrame(data={"int64" : int64_series})
        >>> collection.insert(data=data)
        >>> collection.num_entities
        >>> collection.drop()
        >>> from pymilvus_orm import utility
        >>> utility.has_collection("test_collection")
        False
        """
        conn = self._get_connection()
        indexes = self.indexes
        for index in indexes:
            index.drop(**kwargs)
        conn.drop_collection(self._name, timeout=kwargs.get("timeout", None))

    def load(self, field_names=None, index_names=None, partition_names=None, **kwargs):
        """
        Load the collection from disk to memory.

        :param field_names: The specified fields to load.
        :type  field_names: list[str]

        :param index_names: The specified indexes to load.
        :type  index_names: list[str]

        :param partition_names: The specified partitions to load.
        :type partition_names: list[str]

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises ParamError: If parameters are invalid.
        :raises BaseException: If fields, index or partition doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm import connections
        >>> from pymilvus_orm.types import DataType
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=False, description="int64")
        >>> schema = CollectionSchema(fields=[field], description="collection schema has a int64 field")
        >>> connections.create_connection()
        <milvus.client.stub.Milvus object at 0x7f8579002dc0>
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import pandas as pd
        >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))
        >>> data = pd.DataFrame(data={"int64" : int64_series})
        >>> collection.insert(data)
        >>> collection.load() # load collection to memory
        >>> assert not collection.is_empty
        >>> assert collection.num_entities == 10
        """
        conn = self._get_connection()
        conn.load_collection(self._name, timeout=kwargs.get("timeout", None))

    def release(self, **kwargs):
        """
        Release the collection from memory.

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If collection hasn't been loaded.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm import connections
        >>> from pymilvus_orm.types import DataType
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=False, description="int64")
        >>> schema = CollectionSchema(fields=[field], description="collection schema has a int64 field")
        >>> connections.create_connection()
        <milvus.client.stub.Milvus object at 0x7f8579002dc0>
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import pandas as pd
        >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))
        >>> data = pd.DataFrame(data={"int64" : int64_series})
        >>> collection.insert(data)
        >>> collection.load()   # load collection to memory
        >>> assert not collection.is_empty
        >>> assert collection.num_entities == 10
        >>> collection.release()    # release the collection from memory
        """
        conn = self._get_connection()
        conn.release_collection(self._name, timeout=kwargs.get("timeout", None))

    def insert(self, data, partition_name=None, **kwargs):
        """
        Insert data into collection.

        :param data: The specified data to insert, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple) object or pandas.DataFrame
        :param partition_name: The partition name which the data will be inserted to, if partition name is
                               not passed, then the data will be inserted to "_default" partition
        :type partition_name: str

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises ParamError: If parameters are invalid.
        :raises BaseException: If partition doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm import connections
        >>> from pymilvus_orm.types import DataType
        >>> connections.create_connection()
        <milvus.client.stub.Milvus object at 0x7f8579002dc0>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=False, description="int64")
        >>> schema = CollectionSchema(fields=[field], description="collection schema has a int64 field")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import random
        >>> data = [[random.randint(1, 100) for _ in range(10)]]
        >>> collection.insert(data)
        >>> collection.load()
        >>> assert not collection.is_empty
        >>> assert collection.num_entities == 10
        """
        if not self._check_insert_data_schema(data):
            raise SchemaNotReadyException("The types of schema and data do not match.")
        conn = self._get_connection()
        entities, ids = Prepare.prepare_insert_data(data, self._schema)
        timeout = kwargs.pop("timeout", None)
        res = conn.insert(collection_name=self._name, entities=entities, ids=ids, partition_tag=partition_name,
                          timeout=timeout, orm=True, **kwargs)
        if kwargs.get("_async", False):
            return InsertFuture(res)
        return res

    def search(self, data, anns_field, param, limit, expression, partition_names=None, output_fields=None, timeout=None,
               **kwargs):
        """
        Vector similarity search with an optional boolean expression as filters.

        :param data: The vectors of search data, the length of data is number of query (nq), the dim of every vector in
                     data must be equal to vector field's of collection.
        :type  data: list[list[float]]
        :param anns_field: The vector field used to search of collection.
        :type  anns_field: str
        :param param: The parameters of search, such as nprobe, etc.
        :type  param: dict
        :param limit: The max number of returned record, we also called this parameter as topk.
        :type  limit: int
        :param expression: The boolean expression used to filter attribute.
        :type  expression: str
        :param partition_names: The names of partitions to search.
        :type  partition_names: list[str]
        :param output_fields: The fields to return in the search result, not supported now.
        :type  output_fields: list[str]
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a SearchResultFuture object;
              otherwise, method returns results from server directly.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: SearchResult:
            SearchResult is iterable and is a 2d-array-like class, the first dimension is
            the number of vectors to query (nq), the second dimension is the number of limit(topk).
        :rtype: SearchResult

        :raises RpcError: If gRPC encounter an error.
        :raises ParamError: If parameters are invalid.
        :raises BaseException: If the return result from server is not ok.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm import connections
        >>> from pymilvus_orm.types import DataType
        >>> connections.create_connection()
        <milvus.client.stub.Milvus object at 0x7f8579002dc0>
        >>> dim = 128
        >>> year_field = FieldSchema(name="year", dtype=DataType.INT64, is_primary=False, description="year")
        >>> embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        >>> schema = CollectionSchema(fields=[year_field, embedding_field])
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import random
        >>> nb = 3000
        >>> nq = 10
        >>> limit = 10
        >>> years = [i for i in range(nb)]
        >>> embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
        >>> collection.insert([years, embeddings])
        >>> collection.load()
        >>> search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        >>> res = collection.search(embeddings[:10], "embedding", search_params, limit, "year > 20")
        >>> assert len(res) == nq
        >>> for hits in res:
        >>>     assert len(hits) == limit
        >>> hits = res[0]
        >>> assert len(hits.ids) == limit
        >>> top1 = hits[0]
        >>> print(top1.id)
        >>> print(top1.distance)
        >>> print(top1.score)
        """
        conn = self._get_connection()
        res = conn.search_with_expression(self._name, data, anns_field, param, limit, expression, partition_names,
                                          output_fields, timeout, **kwargs)
        if kwargs.get("_async", False):
            return SearchResultFuture(res)
        return SearchResult(res)

    @property
    def partitions(self) -> list:
        """
        Return all partitions of the collection.

        :return list[Partition]:
            List of Partition object, return when operation is successful.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", data=None, schema=schema, alias="default")
        >>> collection.partitions
        [{"name": "_default", "description": "", "num_entities": 0}]
        """
        conn = self._get_connection()
        partition_strs = conn.list_partitions(self._name)
        partitions = []
        for partition in partition_strs:
            partitions.append(Partition(self, partition))
        return partitions

    def partition(self, partition_name) -> Partition:
        """
        Return the partition corresponding to name. Return None if not existed.

        :param partition_name: The name of the partition to get.
        :type  partition_name: str

        :return Partition:
            Partition object corresponding to partition_name.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If partition doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", data=None, schema=schema, alias="default")
        >>> collection.partition("partition")

        >>> collection.partition("_default")
        {"name": "_default", "description": "", "num_entities": 0}
        """
        if self.has_partition(partition_name) is False:
            return None
        return Partition(self, partition_name)

    def create_partition(self, partition_name, description=""):
        """
        Create the partition corresponding to name if not existed.

        :param partition_name: The name of the partition to create.
        :type  partition_name: str

        :param description: The description of the partition corresponding to name.
        :type description: str

        :return Partition:
            Partition object corresponding to partition_name.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If partition doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", data=None, schema=schema, alias="default")
        >>> collection.create_partition(partition_name="partition", description="test partition")
        {"name": "partition", "description": "", "num_entities": 0}
        >>> collection.partition("partition")
        {"name": "partition", "description": "", "num_entities": 0}
        """
        from .partition import Partition
        if self.has_partition(partition_name) is True:
            raise Exception("Partition already exist.")
        return Partition(self, partition_name)

    def has_partition(self, partition_name) -> bool:
        """
        Checks if a specified partition exists.

        :param partition_name: The name of the partition to check
        :type  partition_name: str

        :return bool:
            Whether a specified partition exists.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", data=None, schema=schema, alias="default")
        >>> collection.create_partition(partition_name="partition", description="test partition")
        {"name": "partition", "description": "", "num_entities": 0}
        >>> collection.partition("partition")
        {"name": "partition", "description": "", "num_entities": 0}
        >>> collection.has_partition("partition")
        True
        >>> collection.has_partition("partition2")
        False
        """
        conn = self._get_connection()
        return conn.has_partition(self._name, partition_name)

    def drop_partition(self, partition_name, **kwargs):
        """
        Drop the partition and its corresponding index files.

        :param partition_name: The name of the partition to drop.
        :type  partition_name: str

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If partition doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", data=None, schema=schema, alias="default")
        >>> collection.create_partition(partition_name="partition", description="test partition")
        {"name": "partition", "description": "", "num_entities": 0}
        >>> collection.partition("partition")
        {"name": "partition", "description": "", "num_entities": 0}
        >>> collection.has_partition("partition")
        True
        >>> collection.drop_partition("partition")
        >>> collection.has_partition("partition")
        False
        """
        if self.has_partition(partition_name) is False:
            raise Exception("Partition doesn't exist")
        conn = self._get_connection()
        return conn.drop_partition(self._name, partition_name, timeout=kwargs.get("timeout", None))

    @property
    def indexes(self) -> list:
        """
        Return all indexes of the collection.

        :return list[Index]:
            List of Index object, return when operation is successful.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_primary=False)
        >>> schema = CollectionSchema(fields=[field], description="collection description")
        >>> collection = Collection(name="test_collection", data=None, schema=schema, alias="default")
        >>> collection.indexes
        []
        """
        conn = self._get_connection()
        indexes = []
        tmp_index = conn.describe_index(self._name)
        if tmp_index is not None:
            indexes.append(Index(self, tmp_index['field_name'], tmp_index))
        return indexes

    def index(self, index_name="") -> Index:
        """
        Return the index corresponding to name.

        :param index_name: The name of the index to create.
        :type  index_name: str

        :return Index:
            Index object corresponding to index_name.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If index doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> year_field = FieldSchema(name="year", dtype=DataType.INT64, is_primary=False, description="year")
        >>> embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        >>> schema = CollectionSchema(fields=[year_field, embedding_field])
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
        >>> collection.create_index("embedding", default_index)
        Status(code=0, message='')
        >>> collection.indexes
        [<pymilvus_orm.index.Index object at 0x7f4435587e20>]
        >>> collection.index()
        <pymilvus_orm.index.Index object at 0x7f44355a1460>
        """
        conn = self._get_connection()
        tmp_index = conn.describe_index(self._name)
        if tmp_index is not None:
            field_name = tmp_index.pop("field_name", None)
            return Index(self, field_name, tmp_index)

    def create_index(self, field_name, index_params, index_name="", **kwargs) -> Index:
        """
        Create index on a specified column according to the index parameters. Return Index Object.

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param index_params: Indexing parameters.
        :type  index_params: dict

        :param index_name: The name of the index to create.
        :type  index_name: str

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises ParamError: If index parameters are invalid.
        :raises BaseException: If field doesn't exist.
        :raises BaseException: If index has been created.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> year_field = FieldSchema(name="year", dtype=DataType.INT64, is_primary=False, description="year")
        >>> embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        >>> schema = CollectionSchema(fields=[year_field, embedding_field])
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
        >>> collection.create_index("embedding", default_index)
        Status(code=0, message='')
        >>> collection.indexes
        [<pymilvus_orm.index.Index object at 0x7f4435587e20>]
        >>> collection.index()
        <pymilvus_orm.index.Index object at 0x7f44355a1460>
        """
        conn = self._get_connection()
        return conn.create_index(self._name, field_name, index_params, timeout=kwargs.get("timeout", None),
                                 **kwargs)

    def has_index(self, index_name="") -> bool:
        """
        Checks whether a specified index exists.

        :param index_name: The name of the index to check.
        :type  index_name: str

        :return bool:
            If specified index exists.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7f9a190ca898>
        >>> year_field = FieldSchema(name="year", dtype=DataType.INT64, is_primary=False, description="year")
        >>> embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        >>> schema = CollectionSchema(fields=[year_field, embedding_field])
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
        >>> collection.create_index("embedding", default_index)
        Status(code=0, message='')
        >>> collection.indexes
        [<pymilvus_orm.index.Index object at 0x7f4435587e20>]
        >>> collection.index()
        <pymilvus_orm.index.Index object at 0x7f44355a1460>
        >>> collection.has_index()
        True
        """
        conn = self._get_connection()
        # TODO(yukun): Need field name, but provide index name
        if conn.describe_index(self._name, "") is None:
            return False
        return True

    def drop_index(self, index_name="", **kwargs):
        """
        Drop index and its corresponding index files.

        :param index_name: The name of the partition to drop.
        :type  index_name: str

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. When timeout
              is set to None, client waits until server response or error occur.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If index has been created.

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> from pymilvus_orm.types import DataType
        >>> from pymilvus_orm import connections
        >>> connections.create_connection(alias="default")
        <milvus.client.stub.Milvus object at 0x7feaddc9cb80>
        >>> year_field = FieldSchema(name="year", dtype=DataType.INT64, is_primary=False, description="year")
        >>> embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        >>> schema = CollectionSchema(fields=[year_field, embedding_field])
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
        >>> collection.create_index("embedding", default_index)
        Status(code=0, message='')
        >>> collection.has_index()
        True
        >>> collection.drop_index()
        >>> collection.has_index()
        False
        """
        if self.has_index(index_name) is False:
            raise Exception("Index doesn't exist")
        conn = self._get_connection()
        tmp_index = conn.describe_index(self._name, "")
        if tmp_index is not None:
            index = Index(self, tmp_index['field_name'], tmp_index, index_name)
            index.drop()
