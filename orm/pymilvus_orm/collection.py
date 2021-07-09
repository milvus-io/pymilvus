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
import copy
import json

import pandas

from .connections import get_connection
from .schema import (
    CollectionSchema,
    FieldSchema,
    parse_fields_from_data,
    infer_dtype_bydata,
)
from .prepare import Prepare
from .partition import Partition
from .index import Index
from .search import SearchResult
from .mutation import MutationResult
from .types import DataType
from .exceptions import (
    SchemaNotReadyException,
    DataTypeNotMatchException,
    DataNotMatchException,
    ConnectionNotExistException,
    PartitionAlreadyExistException,
    PartitionNotExistException,
    IndexNotExistException,
    AutoIDException,
    ExceptionsMessage,
)
from .future import SearchFuture, MutationFuture


def _check_schema(schema):
    if schema is None:
        raise SchemaNotReadyException(0, ExceptionsMessage.NoSchema)
    if len(schema.fields) < 1:
        raise SchemaNotReadyException(0, ExceptionsMessage.EmptySchema)
    vector_fields = []
    for field in schema.fields:
        if field.dtype == DataType.FLOAT_VECTOR or field.dtype == DataType.BINARY_VECTOR:
            vector_fields.append(field.name)
    if len(vector_fields) < 1:
        raise SchemaNotReadyException(0, ExceptionsMessage.NoVector)


def _check_data_schema(fields, data):
    if isinstance(data, pandas.DataFrame):
        for i, field in enumerate(fields):
            for j, _ in enumerate(data[field.name]):
                tmp_type = infer_dtype_bydata(data[field.name][j])
                if tmp_type != field.dtype:
                    raise DataNotMatchException(0, ExceptionsMessage.DataTypeInconsistent)
    else:
        for i, field in enumerate(fields):
            for j, _ in enumerate(data[i]):
                tmp_type = infer_dtype_bydata(data[i][j])
                if tmp_type != field.dtype:
                    raise DataNotMatchException(0, ExceptionsMessage.DataTypeInconsistent)


class Collection:
    """
    This is a class corresponding to collection in milvus.

        """

    def __init__(self, name, schema=None, **kwargs):
        """
        Constructs a collection by name, schema and other parameters.
        Connection information is contained in kwargs.

        :param name: the name of collection
        :type name: str

        :param schema: the schema of collection
        :type schema: class `schema.CollectionSchema`

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            <pymilvus.client.stub.Milvus object at 0x7f9a190ca898>
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> description="This is a new collection description."
            >>> schema = CollectionSchema(fields=fields, description=description)
            >>> collection = Collection(name="test_collection_init", schema=schema)
            >>> collection.name
            'test_collection_init'
            >>> collection.description
            'This is a new collection description.'
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
            else:
                if not isinstance(schema, CollectionSchema):
                    raise SchemaNotReadyException(0, ExceptionsMessage.SchemaType)
                if server_schema != schema:
                    raise SchemaNotReadyException(0, ExceptionsMessage.SchemaInconsistent)
                self._schema = schema

        else:
            if schema is None:
                raise SchemaNotReadyException(0, ExceptionsMessage.NoSchema)
            if isinstance(schema, CollectionSchema):
                _check_schema(schema)
                conn.create_collection(self._name, fields=schema.to_dict())
                self._schema = schema
            else:
                raise SchemaNotReadyException(0, ExceptionsMessage.SchemaType)

    def __repr__(self):
        return json.dumps({
            'name': self.name,
            'schema': self._schema.to_dict(),
            'partitions': [json.loads(p.__repr__()) for p in self.partitions],
            'description': self.description,
        })

    def _get_using(self):
        return self._kwargs.get("_using", "default")

    def _get_connection(self):
        conn = get_connection(self._get_using())
        if conn is None:
            raise ConnectionNotExistException(0, ExceptionsMessage.ConnectFirst)
        return conn

    def _check_insert_data_schema(self, data):
        """
        Checks whether the data type matches the schema.
        """
        if self._schema is None:
            return False
        if self._schema.auto_id:
            if isinstance(data, pandas.DataFrame):
                if self._schema.primary_field.name in data:
                    if not data[self._schema.primary_field.name].isnull().all():
                        raise DataNotMatchException(0, ExceptionsMessage.AutoIDWithData)
                    data = data.drop(self._schema.primary_field.name, axis=1)

        infer_fields = parse_fields_from_data(data)
        tmp_fields = copy.deepcopy(self._schema.fields)

        for i, field in enumerate(self._schema.fields):
            if field.is_primary and field.auto_id:
                tmp_fields.pop(i)

        if len(infer_fields) != len(tmp_fields):
            raise DataTypeNotMatchException(0, ExceptionsMessage.FieldsNumInconsistent)

        _check_data_schema(infer_fields, data)

        for x, y in zip(infer_fields, tmp_fields):
            if x.dtype != y.dtype:
                return False
            if isinstance(data, pandas.DataFrame):
                if x.name != y.name:
                    return False
            # todo check dim
        return True

    def _check_schema(self):
        if self._schema is None:
            raise SchemaNotReadyException(0, ExceptionsMessage.NoSchema)

    def _get_vector_field(self) -> str:
        for field in self._schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR or field.dtype == DataType.BINARY_VECTOR:
                return field.name
        raise SchemaNotReadyException(0, ExceptionsMessage.NoVector)

    @classmethod
    def construct_from_dataframe(cls, name, dataframe, **kwargs):
        if dataframe is None:
            raise SchemaNotReadyException(0, ExceptionsMessage.NoneDataFrame)
        if not isinstance(dataframe, pandas.DataFrame):
            raise SchemaNotReadyException(0, ExceptionsMessage.DataFrameType)
        primary_field = kwargs.pop("primary_field", None)
        if primary_field is None:
            raise SchemaNotReadyException(0, ExceptionsMessage.NoPrimaryKey)
        pk_index = -1
        for i, field in enumerate(dataframe):
            if field == primary_field:
                pk_index = i
        if pk_index == -1:
            raise SchemaNotReadyException(0, ExceptionsMessage.PrimaryKeyNotExist)
        if "auto_id" in kwargs:
            if not isinstance(kwargs.get("auto_id", None), bool):
                raise AutoIDException(0, ExceptionsMessage.AutoIDType)
        auto_id = kwargs.pop("auto_id", False)
        if auto_id:
            if dataframe[primary_field].isnull().all():
                dataframe = dataframe.drop(primary_field, axis=1)
            else:
                raise SchemaNotReadyException(0, ExceptionsMessage.AutoIDWithData)

        fields = parse_fields_from_data(dataframe)
        _check_data_schema(fields, dataframe)
        if auto_id:
            fields.insert(pk_index, FieldSchema(name=primary_field, dtype=DataType.INT64, is_primary=True, auto_id=True,
                                                **kwargs))
        else:
            for field in fields:
                if field.name == primary_field:
                    field.is_primary = True
                    field.auto_id = False

        schema = CollectionSchema(fields=fields)
        _check_schema(schema)
        collection = cls(name, schema, **kwargs)
        res = collection.insert(data=dataframe)
        return collection, res

    @property
    def schema(self) -> CollectionSchema:
        """
        Returns the schema of the collection.

        :return schema.CollectionSchema:
            Schema of the collection.
        """
        return self._schema

    @property
    def description(self) -> str:
        """
        Returns a text description of the collection.

        :return str:
            Collection description text, returned when the operation succeeds.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> description="This is an example text description."
            >>> schema = CollectionSchema(fields=fields, description=description)
            >>> collection = Collection(name="test_collection_description", schema=schema)
            >>> collection.description
            'This is an example text description.'
        """

        return self._schema.description

    @property
    def name(self) -> str:
        """
        Returns the collection name.

        :return str:
            The collection name, returned when the operation succeeds.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> schema = CollectionSchema(fields)
            >>> collection = Collection("test_collection_name", schema)
            >>> collection.name
            'test_collection_name'
        """
        return self._name

    @property
    def is_empty(self) -> bool:
        """
        Whether the collection is empty.
        This method need to call `num_entities <#pymilvus_orm.Collection.num_entities>`_.

        :return bool:
            * True: The collection is empty.
            * False: The collection is  gfghnot empty.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_is_empty", schema)
            >>> collection.is_empty
            True
            >>> collection.insert([[1], [[1.0, 2.0]]])
            <pymilvus_orm.search.MutationResult object at 0x7fabaf3e5d50>
            >>> collection.is_empty
            False
        """
        return self.num_entities == 0

    # read-only
    @property
    def num_entities(self) -> int:
        """
        Returns the number of entities in the collection.

        :return int:
            Number of entities in the collection.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_num_entities", schema)
            >>> collection.num_entities
            0
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.num_entities
            2
            """
        conn = self._get_connection()
        conn.flush([self._name])
        status = conn.get_collection_stats(db_name="", collection_name=self._name)
        return status["row_count"]

    @property
    def primary_field(self) -> FieldSchema:
        """
        Returns the primary field of the collection.

        :return schema.FieldSchema:
            The primary field of the collection.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("film_length", DataType.INT64, description="length in miniute"),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_primary_field", schema)
            >>> collection.primary_field.name
            'film_id'
        """
        return self._schema.primary_field

    def drop(self, **kwargs):
        """
        Drops the collection together with its index files.

        :param kwargs:
            * *timeout* (``float``) --
                An optional duration of time in seconds to allow for the RPC.
                If timeout is set to None,
                the client keeps waiting until the server responds or an error occurs.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_drop", schema)
            >>> utility.has_collection("test_collection_drop")
            True
            >>> collection.drop()
            >>> utility.has_collection("test_collection_drop")
            False
        """
        conn = self._get_connection()
        indexes = self.indexes
        for index in indexes:
            index.drop(**kwargs)
        conn.drop_collection(self._name, timeout=kwargs.get("timeout", None))

    def load(self, partition_names=None, **kwargs):
        """
        Loads the collection from disk to memory.

        :param partition_names: The specified partitions to load.
        :type partition_names: list[str]

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or error occurs.

        :raises CollectionNotExistException: If the collection does not exist.
        :raises ParamError: If the parameters are invalid.
        :raises BaseException: If the specified field, index or partition does not exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_load", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            <pymilvus_orm.search.MutationResult object at 0x7fabaf3e5d50>
            >>> collection.load()
            >>> collection.num_entities
            2
        """
        conn = self._get_connection()
        if partition_names is not None:
            conn.load_partitions(self._name, partition_names, timeout=kwargs.get("timeout", None))
        else:
            conn.load_collection(self._name, timeout=kwargs.get("timeout", None))

    def release(self, **kwargs):
        """
        Releases the collection from memory.

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or an error occurs.

        :raises CollectionNotExistException: If collection does not exist.
        :raises BaseException: If collection has not been loaded to memory.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_release", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            <pymilvus_orm.search.MutationResult object at 0x7fabaf3e5d50>
            >>> collection.load()
            >>> collection.num_entities
            2
            >>> collection.release()    # release the collection from memory
        """
        conn = self._get_connection()
        conn.release_collection(self._name, timeout=kwargs.get("timeout", None))

    def insert(self, data, partition_name=None, **kwargs):
        """
        Insert data into the collection.

        :param data: The specified data to insert, the dimension of data needs to align with column
                     number
        :type  data: list-like(list, tuple) object or pandas.DataFrame
        :param partition_name: The partition name which the data will be inserted to, if partition
                               name is not passed, then the data will be inserted to "_default"
                               partition
        :type partition_name: str

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or an error occurs.

        :raises CollectionNotExistException: If the specified collection does not exist.
        :raises ParamError: If input parameters are invalid.
        :raises BaseException: If the specified partition does not exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> connections.connect()
            <pymilvus.client.stub.Milvus object at 0x7f8579002dc0>
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_insert", schema)
            >>> data = [
            ...     [random.randint(1, 100) for _ in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> collection.num_entities
            10
        """
        if data is None:
            return MutationResult(data)
        if not self._check_insert_data_schema(data):
            raise SchemaNotReadyException(0, ExceptionsMessage.TypeOfDataAndSchemaInconsistent)
        conn = self._get_connection()
        entities = Prepare.prepare_insert_data(data, self._schema)
        timeout = kwargs.pop("timeout", None)
        res = conn.insert(collection_name=self._name, entities=entities, ids=None,
                          partition_name=partition_name, timeout=timeout, **kwargs)
        if kwargs.get("_async", False):
            return MutationFuture(res)
        return MutationResult(res)

    def search(self, data, anns_field, param, limit, expr=None, partition_names=None,
               output_fields=None, timeout=None, **kwargs):
        """
        Conducts a vector similarity search with an optional boolean expression as filter.

        :param data: The vectors of search data, the length of data is number of query (nq), the
                     dim of every vector in data must be equal to vector field's of collection.
        :type  data: list[list[float]]
        :param anns_field: The vector field used to search of collection.
        :type  anns_field: str
        :param param: The parameters of search, such as ``nprobe``.
        :type  param: dict
        :param limit: The max number of returned record, also known as ``topk``.
        :type  limit: int
        :param expr: The boolean expression used to filter attribute.
        :type  expr: str
        :param partition_names: The names of partitions to search.
        :type  partition_names: list[str]
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
              The callback function which is invoked after server response successfully.
              It functions only if _async is set to True.

        :return: SearchResult:
            SearchResult is iterable and is a 2d-array-like class, the first dimension is
            the number of vectors to query (nq), the second dimension is the number of limit(topk).
        :rtype: SearchResult

        :raises RpcError: If gRPC encounter an error.
        :raises ParamError: If parameters are invalid.
        :raises DataTypeNotMatchException: If wrong type of param is passed.
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
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> collection.num_entities
            10
            >>> collection.load()
            >>> # search
            >>> search_param = {
            ...     "data": [[1.0, 1.0]],
            ...     "anns_field": "films",
            ...     "param": {"metric_type": "L2"},
            ...     "limit": 2,
            ...     "expr": "film_id > 0",
            ... }
            >>> res = collection.search(**search_param)
            >>> assert len(res) == 1
            >>> hits = res[0]
            >>> assert len(hits) == 2
            >>> print(f"- Total hits: {len(hits)}, hits ids: {hits.ids} ")
            - Total hits: 2, hits ids: [8, 5]
            >>> print(f"- Top1 hit id: {hits[0].id}, distance: {hits[0].distance}, score: {hits[0].score} ")
            - Top1 hit id: 8, distance: 0.10143111646175385, score: 0.10143111646175385
        """
        if expr is not None and not isinstance(expr, str):
            raise DataTypeNotMatchException(0, ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        res = conn.search_with_expression(self._name, data, anns_field, param, limit, expr,
                                          partition_names, output_fields, timeout, **kwargs)
        if kwargs.get("_async", False):
            return SearchFuture(res)
        return SearchResult(res)

    def query(self, expr, output_fields=None, partition_names=None, timeout=None):
        """
        Query with a set of criteria, and results in a list of records that match the query exactly.

        :param expr: The query expression
        :type  expr: str

        :param output_fields: A list of fields to return
        :type  output_fields: list[str]

        :param partition_names: Name of partitions that contain entities
        :type  partition_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :return: A list that contains all results
        :rtype: list

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            DataTypeNotMatchException: If wrong type of param is passed
            BaseException: If the return result from server is not ok
        """
        if not isinstance(expr, str):
            raise DataTypeNotMatchException(0, ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        res = conn.query(self._name, expr, output_fields, partition_names, timeout)
        return res

    @property
    def partitions(self) -> list:
        """
        Return all partitions of the collection.

        :return list[Partition]:
            List of Partition object, return when operation is successful.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            <pymilvus.client.stub.Milvus object at 0x7f8579002dc0>
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_partitions", schema)
            >>> collection.partitions
            [{"name": "_default", "description": "", "num_entities": 0}]
        """
        conn = self._get_connection()
        partition_strs = conn.list_partitions(self._name)
        partitions = []
        for partition in partition_strs:
            partitions.append(Partition(self, partition, construct_only=True))
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
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            <pymilvus.client.stub.Milvus object at 0x7f8579002dc0>
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_partition", schema)
            >>> collection.partition("_default")
            {"name": "_default", "description": "", "num_entities": 0}
            >>> collection.partition("partition")

        """
        if self.has_partition(partition_name) is False:
            return None
        return Partition(self, partition_name, construct_only=True)

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
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_create_partition", schema)
            >>> collection.create_partition("comedy", description="comedy films")
            {"name": "comedy", "description": "comedy films", "num_entities": 0}
            >>> collection.partition("comedy")
            {"name": "partition", "description": "comedy films", "num_entities": 0}
        """
        if self.has_partition(partition_name) is True:
            raise PartitionAlreadyExistException(0, ExceptionsMessage.PartitionAlreadyExist)
        return Partition(self, partition_name, description=description)

    def has_partition(self, partition_name) -> bool:
        """
        Checks if a specified partition exists.

        :param partition_name: The name of the partition to check
        :type  partition_name: str

        :return bool:
            Whether a specified partition exists.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_has_partition", schema)
            >>> collection.create_partition("comedy", description="comedy films")
            {"name": "comedy", "description": "comedy films", "num_entities": 0}
            >>> collection.has_partition("comedy")
            True
            >>> collection.has_partition("science_fiction")
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
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or an error occurs.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If partition doesn't exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_drop_partition", schema)
            >>> collection.create_partition("comedy", description="comedy films")
            {"name": "comedy", "description": "comedy films", "num_entities": 0}
            >>> collection.has_partition("comedy")
            True
            >>> collection.drop_partition("comedy")
            >>> collection.has_partition("comedy")
            False
        """
        if self.has_partition(partition_name) is False:
            raise PartitionNotExistException(0, ExceptionsMessage.PartitionNotExist)
        conn = self._get_connection()
        return conn.drop_partition(self._name, partition_name, timeout=kwargs.get("timeout", None))

    @property
    def indexes(self) -> list:
        """
        Returns all indexes of the collection.

        :return list[Index]:
            List of Index objects, returned when this operation is successful.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_indexes", schema)
            >>> collection.indexes
            []
        """
        conn = self._get_connection()
        indexes = []
        tmp_index = conn.describe_index(self._name)
        if tmp_index is not None:
            field_name = tmp_index.pop("field_name", None)
            indexes.append(Index(self, field_name, tmp_index, construct_only=True))
        return indexes

    def index(self) -> Index:
        """
        Fetches the index object of the of the specified name.

        :return Index:
            Index object corresponding to index_name.

        :raises CollectionNotExistException: If the collection does not exist.
        :raises BaseException: If the specified index does not exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_index", schema)
            >>> index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index)
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
            return Index(self, field_name, tmp_index, construct_only=True)
        raise IndexNotExistException(0, ExceptionsMessage.IndexNotExist)

    def create_index(self, field_name, index_params, **kwargs) -> Index:
        """
        Creates index for a specified field. Return Index Object.

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param index_params: The indexing parameters.
        :type  index_params: dict

        :raises CollectionNotExistException: If the collection does not exist.
        :raises ParamError: If the index parameters are invalid.
        :raises BaseException: If field does not exist.
        :raises BaseException: If the index has been created.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_create_index", schema)
            >>> index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index)
            Status(code=0, message='')
            >>> collection.index()
            <pymilvus_orm.index.Index object at 0x7f44355a1460>
        """
        conn = self._get_connection()
        return conn.create_index(self._name, field_name, index_params,
                                 timeout=kwargs.pop("timeout", None), **kwargs)

    def has_index(self) -> bool:
        """
        Checks whether a specified index exists.

        :return bool:
            Whether the specified index exists.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_has_index", schema)
            >>> index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index)
            >>> collection.has_index()
            True
        """
        conn = self._get_connection()
        # TODO(yukun): Need field name, but provide index name
        if conn.describe_index(self._name, "") is None:
            return False
        return True

    def drop_index(self, **kwargs):
        """
        Drop index and its corresponding index files.

        :param kwargs:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or an error occurs.
              Optional. A duration of time in seconds.

        :raises CollectionNotExistException: If the collection does not exist.
        :raises BaseException: If the index does not exist or has been dropped.

        :example:
            >>> from pymilvus_orm import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_has_index", schema)
            >>> index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index)
            >>> collection.has_index()
            True
            >>> collection.drop_index()
            >>> collection.has_index()
            False
        """
        if self.has_index() is False:
            raise IndexNotExistException(0, ExceptionsMessage.IndexNotExist)
        conn = self._get_connection()
        tmp_index = conn.describe_index(self._name, "")
        if tmp_index is not None:
            index = Index(self, tmp_index['field_name'], tmp_index, construct_only=True)
            index.drop(**kwargs)
