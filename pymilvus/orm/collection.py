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
from typing import List
import pandas

from .connections import connections
from .schema import (
    CollectionSchema,
    FieldSchema,
    parse_fields_from_data,
    check_insert_data_schema,
    check_schema,
)
from .prepare import Prepare
from .partition import Partition
from .index import Index
from .search import SearchResult
from .mutation import MutationResult
from .types import DataType
from ..exceptions import (
    SchemaNotReadyException,
    DataTypeNotMatchException,
    PartitionAlreadyExistException,
    PartitionNotExistException,
    IndexNotExistException,
    AutoIDException,
    ExceptionsMessage,
)
from .future import SearchFuture, MutationFuture
from .utility import _get_connection
from .default_config import DefaultConfig
from ..client.types import CompactionState, CompactionPlans, Replica
from ..client.types import get_consistency_level, cmp_consistency_level
from ..client.constants import DEFAULT_CONSISTENCY_LEVEL
from ..client.configs import DefaultConfigs



class Collection:
    """ This is a class corresponding to collection in milvus. """

    def __init__(self, name, schema=None, using="default", shards_num=2, **kwargs):
        """
        Constructs a collection by name, schema and other parameters.
        Connection information is contained in kwargs.

        :param name: the name of collection
        :type name: str

        :param schema: the schema of collection
        :type schema: class `schema.CollectionSchema`

        :param using: Milvus link of create collection
        :type using: str

        :param shards_num: How wide to scale collection. Corresponds to how many active datanodes
                        can be used on insert.
        :type shards_num: int

        :param kwargs:
            * *consistency_level* (``str/int``) --
            Which consistency level to use when searching in the collection. For details, see
            https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md.
            Options of consistency level: Strong, Bounded, Eventually, Session, Customized.
            Note: this parameter can be overwritten by the same parameter specified in search.

            * *properties* (``dictionary``) --

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        self._using = using
        self._shards_num = shards_num
        self._kwargs = kwargs
        conn = self._get_connection()

        has = conn.has_collection(self._name, **kwargs)
        if has:
            resp = conn.describe_collection(self._name, **kwargs)
            s_consistency_level = resp.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
            arg_consistency_level = kwargs.get("consistency_level", s_consistency_level)
            if not cmp_consistency_level(s_consistency_level, arg_consistency_level):
                raise SchemaNotReadyException(message=ExceptionsMessage.ConsistencyLevelInconsistent)
            server_schema = CollectionSchema.construct_from_dict(resp)
            self._consistency_level = s_consistency_level
            if schema is None:
                self._schema = server_schema
            else:
                if not isinstance(schema, CollectionSchema):
                    raise SchemaNotReadyException(message=ExceptionsMessage.SchemaType)
                if server_schema != schema:
                    raise SchemaNotReadyException(message=ExceptionsMessage.SchemaInconsistent)
                self._schema = schema

        else:
            if schema is None:
                raise SchemaNotReadyException(message=ExceptionsMessage.CollectionNotExistNoSchema % name)
            if isinstance(schema, CollectionSchema):
                check_schema(schema)
                consistency_level = get_consistency_level(kwargs.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL))
                conn.create_collection(self._name, schema, shards_num=self._shards_num, **kwargs)
                self._schema = schema
                self._consistency_level = consistency_level
            else:
                raise SchemaNotReadyException(message=ExceptionsMessage.SchemaType)

    def __repr__(self):
        _dict = {
            'name': self.name,
            'partitions': self.partitions,
            'description': self.description,
            'schema': self._schema,
        }
        r = ["<Collection>:\n-------------\n"]
        s = "<{}>: {}\n"
        for k, v in _dict.items():
            r.append(s.format(k, v))
        return "".join(r)

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    @classmethod
    def construct_from_dataframe(cls, name, dataframe, **kwargs):
        if dataframe is None:
            raise SchemaNotReadyException(message=ExceptionsMessage.NoneDataFrame)
        if not isinstance(dataframe, pandas.DataFrame):
            raise SchemaNotReadyException(message=ExceptionsMessage.DataFrameType)
        primary_field = kwargs.pop("primary_field", None)
        if primary_field is None:
            raise SchemaNotReadyException(message=ExceptionsMessage.NoPrimaryKey)
        pk_index = -1
        for i, field in enumerate(dataframe):
            if field == primary_field:
                pk_index = i
        if pk_index == -1:
            raise SchemaNotReadyException(message=ExceptionsMessage.PrimaryKeyNotExist)
        if "auto_id" in kwargs:
            if not isinstance(kwargs.get("auto_id", None), bool):
                raise AutoIDException(message=ExceptionsMessage.AutoIDType)
        auto_id = kwargs.pop("auto_id", False)
        if auto_id:
            if dataframe[primary_field].isnull().all():
                dataframe = dataframe.drop(primary_field, axis=1)
            else:
                raise SchemaNotReadyException(message=ExceptionsMessage.AutoIDWithData)

        using = kwargs.get("using", DefaultConfig.DEFAULT_USING)
        conn = _get_connection(using)
        if conn.has_collection(name, **kwargs):
            resp = conn.describe_collection(name, **kwargs)
            server_schema = CollectionSchema.construct_from_dict(resp)
            schema = server_schema
        else:
            fields_schema = parse_fields_from_data(dataframe)
            if auto_id:
                fields_schema.insert(pk_index,
                                     FieldSchema(name=primary_field, dtype=DataType.INT64, is_primary=True,
                                                 auto_id=True,
                                                 **kwargs))

            for field in fields_schema:
                if auto_id is False and field.name == primary_field:
                    field.is_primary = True
                    field.auto_id = False
                if field.dtype == DataType.VARCHAR:
                    field.params[DefaultConfigs.MaxVarCharLengthKey] = int(DefaultConfigs.MaxVarCharLength)
            schema = CollectionSchema(fields=fields_schema)

        check_schema(schema)
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
    def aliases(self, **kwargs) -> list:
        """
        Returns the aliases of the collection.

        :return list[str]:
            Aliases of the collection.
        """
        conn = self._get_connection()
        resp = conn.describe_collection(self._name, **kwargs)
        aliases = resp["aliases"]
        return aliases

    @property
    def description(self) -> str:
        """
        Returns a text description of the collection.

        :return str:
            Collection description text, returned when the operation succeeds.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        This method need to call `num_entities <#pymilvus.Collection.num_entities>`_.

        :return bool:
            * True: The collection is empty.
            * False: The collection is  gfghnot empty.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_is_empty", schema)
            >>> collection.is_empty
            True
            >>> collection.insert([[1], [[1.0, 2.0]]])
            <pymilvus.search.MutationResult object at 0x7fabaf3e5d50>
            >>> collection.is_empty
            False
        """
        return self.num_entities == 0

    # read-only
    @property
    def num_entities(self, **kwargs) -> int:
        """
        Returns the number of entities in the collection.

        :return int:
            Number of entities in the collection.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        stats = conn.get_collection_stats(collection_name=self._name, **kwargs)
        result = {stat.key: stat.value for stat in stats}
        result["row_count"] = int(result["row_count"])
        return result["row_count"]

    @property
    def primary_field(self) -> FieldSchema:
        """
        Returns the primary field of the collection.

        :return schema.FieldSchema:
            The primary field of the collection.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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

    def flush(self, timeout=None, **kwargs):
        """ Flush """
        conn = self._get_connection()
        conn.flush([self.name], timeout=timeout, **kwargs)

    def drop(self, timeout=None, **kwargs):
        """
        Drops the collection together with its index files.

        :param timeout:
            * *timeout* (``float``) --
                An optional duration of time in seconds to allow for the RPC.
                If timeout is set to None,
                the client keeps waiting until the server responds or an error occurs.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        conn.drop_collection(self._name, timeout=timeout, **kwargs)

    def set_properties(self, properties, timeout=None, **kwargs):
        conn = self._get_connection()
        conn.alter_collection(self.name, properties=properties, timeout=timeout)

    def load(self, partition_names=None, replica_number=1, timeout=None, **kwargs):
        """ Load the collection from disk to memory.

        :param partition_names: The specified partitions to load.
        :type partition_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. If timeout
            is set to None, the client keeps waiting until the server responds or error occurs.
        :type timeout: float

        :param kwargs:
            * *_async* (``bool``) -- Indicate if invoke asynchronously.

        :raises CollectionNotExistException: If the collection does not exist.
        :raises ParamError: If the parameters are invalid.
        :raises BaseException: If the specified field, index or partition does not exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_load", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.load()
        """
        conn = self._get_connection()
        if partition_names is not None:
            conn.load_partitions(self._name, partition_names, replica_number=replica_number, timeout=timeout, **kwargs)
        else:
            conn.load_collection(self._name, replica_number=replica_number, timeout=timeout, **kwargs)

    def release(self, timeout=None, **kwargs):
        """
        Releases the collection from memory.

        :param timeout:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or an error occurs.

        :raises CollectionNotExistException: If collection does not exist.
        :raises BaseException: If collection has not been loaded to memory.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_release", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.load()
            >>> collection.release()
        """
        conn = self._get_connection()
        conn.release_collection(self._name, timeout=timeout, **kwargs)

    def insert(self, data: [List, pandas.DataFrame], partition_name: str=None, timeout=None, **kwargs) -> MutationResult:
        """ Insert data into the collection.

        Args:
            data (list, tuple, pandas.DataFrame): The specified data to insert
            partition_name (str): The partition name which the data will be inserted to,
                if partition name is not passed, then the data will be inserted to "_default" partition
            timeout (float, optional): A duration of time in seconds to allow for the RPC. Defaults to None.
                If timeout is set to None, the client keeps waiting until the server responds or an error occurs.
        Returns:
            MutationResult: contains 2 properties `insert_count`, and, `primary_keys`
                `insert_count`: how may entites have been inserted into Milvus,
                `primary_keys`: list of primary keys of the inserted entities
        Raises:
            CollectionNotExistException: If the specified collection does not exist.
            ParamError: If input parameters are invalid.
            MilvusException: If the specified partition does not exist.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_insert", schema)
            >>> data = [
            ...     [random.randint(1, 100) for _ in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> res = collection.insert(data)
            >>> res.insert_count
            10
        """
        if data is None:
            return MutationResult(data)
        check_insert_data_schema(self._schema, data)

        conn = self._get_connection()
        entities = Prepare.prepare_insert_data(data, self._schema)
        schema_dict = self._schema.to_dict()
        schema_dict["consistency_level"] = self._consistency_level
        res = conn.batch_insert(self._name, entities, partition_name, timeout=timeout, schema=schema_dict, **kwargs)

        if kwargs.get("_async", False):
            return MutationFuture(res)
        return MutationResult(res)

    def delete(self, expr, partition_name=None, timeout=None, **kwargs):
        """
        Delete entities with an expression condition.
        And return results to show how many entities will be deleted.

        :param expr: The expression to specify entities to be deleted
        :type  expr: str

        :param partition_name: Name of partitions that contain entities
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :return: A MutationResult object contains a property named `delete_count` represents how many
                 entities will be deleted.
        :rtype: MutationResult

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises BaseException: If the return result from server is not ok

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("film_date", DataType.INT64),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_query", schema)
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [i + 2000 for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> collection.num_entities

            >>> expr = "film_id in [ 0, 1 ]"
            >>> res = collection.delete(expr)
            >>> assert len(res) == 2
            >>> print(f"- Deleted entities: {res}")
            - Delete results: [0, 1]
        """

        conn = self._get_connection()
        res = conn.delete(self._name, expr, partition_name, timeout=timeout, **kwargs)
        if kwargs.get("_async", False):
            return MutationFuture(res)
        return MutationResult(res)

    def search(self, data, anns_field, param, limit, expr=None, partition_names=None,
               output_fields=None, timeout=None, round_decimal=-1, **kwargs):
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
        :param round_decimal: The specified number of decimal places of returned distance
        :type  round_decimal: int
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a
              SearchFuture object; otherwise, method returns results from server directly.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully.
              It functions only if _async is set to True.
            * *consistency_level* (``str/int``) --
              Which consistency level to use when searching in the collection. See details in
              https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md.
              Options of consistency level: Strong, Bounded, Eventually, Session, Customized.
              Note: this parameter will overwrite the same parameter specified when user created the collection,
              if no consistency level was specified, search will use the consistency level when you create the
              collection.
            * *guarantee_timestamp* (``int``) --
              This function instructs Milvus to see all operations performed before a provided timestamp. If no
              such timestamp is provided, then Milvus will search all operations performed to date.
              Note: only used in Customized consistency level.
            * *graceful_time* (``int``) --
              Only used in bounded consistency level. If graceful_time is set, PyMilvus will use current timestamp minus
              the graceful_time as the `guarantee_timestamp`. This option is 5s by default if not set.
            * *travel_timestamp* (``int``) --
              Users can specify a timestamp in a search to get results based on a data view
              at a specified point in time.

        :return: SearchResult: SearchResult is iterable and is a 2d-array-like class, the first dimension is
            the number of vectors to query (nq), the second dimension is the number of limit(topk).
        :rtype: SearchResult

        :raises RpcError: If gRPC encounter an error.
        :raises ParamError: If parameters are invalid.
        :raises DataTypeNotMatchException: If wrong type of param is passed.
        :raises BaseException: If the return result from server is not ok.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        schema_dict = self._schema.to_dict()
        schema_dict["consistency_level"] = self._consistency_level
        res = conn.search(self._name, data, anns_field, param, limit, expr,
                          partition_names, output_fields, round_decimal, timeout=timeout, schema=schema_dict, **kwargs)
        if kwargs.get("_async", False):
            return SearchFuture(res)
        return SearchResult(res)

    def query(self, expr, output_fields=None, partition_names=None, timeout=None, **kwargs):
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

        :param kwargs:
            * *consistency_level* (``str/int``) --
              Which consistency level to use during a query on the collection. For details, see
              https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md.
              Options of consistency level: Strong, Bounded, Eventually, Session, Customized.
              Note: this parameter will overwrite the same parameter specified when user created the collection,
              if no consistency level was specified, query will use the consistency level when you create the
              collection.
            * *guarantee_timestamp* (``int``) --
              This function instructs Milvus to see all operations performed before a provided timestamp. If no
              such timestamp is specified, Milvus queries all operations performed to date.
              Note: only used in Customized consistency level.
            * *graceful_time* (``int``) --
              Only used in bounded consistency level. If graceful_time is set, PyMilvus will use current timestamp minus
              the graceful_time as the `guarantee_timestamp`. This option is 5s by default if not set.
            * *travel_timestamp* (``int``) --
              Users can specify a timestamp in a search to get results based on a data view
              at a specified point in time.
            * *offset* (``int``) --
              Combined with limit to enable pagination
            * *limit* (``int``) --
              Combined with limit to enable pagination

        :return: A list that contains all results
        :rtype: list

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises DataTypeNotMatchException: If wrong type of param is passed
        :raises BaseException: If the return result from server is not ok

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> connections.connect()
            <pymilvus.client.stub.Milvus object at 0x7f8579002dc0>
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("film_date", DataType.INT64),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_query", schema)
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [i + 2000 for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> collection.num_entities
            10
            >>> collection.load()
            >>> # query
            >>> expr = "film_id in [ 0, 1 ]"
            >>> res = collection.query(expr, output_fields=["film_date"])
            >>> assert len(res) == 2
            >>> print(f"- Query results: {res}")
            - Query results: [{'film_id': 0, 'film_date': 2000}, {'film_id': 1, 'film_date': 2001}]
        """
        if not isinstance(expr, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        schema = self._schema.to_dict()
        schema["consistency_level"] = self._consistency_level
        res = conn.query(self._name, expr, output_fields, partition_names, timeout=timeout, schema=schema, **kwargs)
        return res

    @property
    def partitions(self, **kwargs) -> list:
        """
        Return all partitions of the collection.

        :return list[Partition]:
            List of Partition object, return when operation is successful.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        partition_strs = conn.list_partitions(self._name, **kwargs)
        partitions = []
        for partition in partition_strs:
            partitions.append(Partition(self, partition, construct_only=True))
        return partitions

    def partition(self, partition_name, **kwargs) -> Partition:
        """
        Return the partition corresponding to name. Return None if not existed.

        :param partition_name: The name of the partition to get.
        :type  partition_name: str

        :return Partition:
            Partition object corresponding to partition_name.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If partition doesn't exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        if self.has_partition(partition_name, **kwargs) is False:
            return None
        return Partition(self, partition_name, construct_only=True, **kwargs)

    def create_partition(self, partition_name, description="", **kwargs):
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
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_create_partition", schema)
            >>> collection.create_partition("comedy", description="comedy films")
            {"name": "comedy", "collection_name": "test_collection_create_partition", "description": ""}
            >>> collection.partition("comedy")
            {"name": "comedy", "collection_name": "test_collection_create_partition", "description": ""}
        """
        if self.has_partition(partition_name, **kwargs) is True:
            raise PartitionAlreadyExistException(message=ExceptionsMessage.PartitionAlreadyExist)
        return Partition(self, partition_name, description=description, **kwargs)

    def has_partition(self, partition_name, timeout=None, **kwargs) -> bool:
        """
        Checks if a specified partition exists.

        :param partition_name: The name of the partition to check
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :return bool:
            Whether a specified partition exists.

        :raises CollectionNotExistException: If collection doesn't exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        return conn.has_partition(self._name, partition_name, timeout=timeout, **kwargs)

    def drop_partition(self, partition_name, timeout=None, **kwargs):
        """
        Drop the partition and its corresponding index files.

        :param partition_name: The name of the partition to drop.
        :type  partition_name: str

        :param timeout:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or an error occurs.

        :raises CollectionNotExistException: If collection doesn't exist.
        :raises BaseException: If partition doesn't exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        if self.has_partition(partition_name, **kwargs) is False:
            raise PartitionNotExistException(message=ExceptionsMessage.PartitionNotExist)
        conn = self._get_connection()
        return conn.drop_partition(self._name, partition_name, timeout=timeout, **kwargs)

    @property
    def indexes(self, **kwargs) -> list:
        """
        Returns all indexes of the collection.

        :return list[Index]:
            List of Index objects, returned when this operation is successful.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        tmp_index = conn.list_indexes(self._name, **kwargs)
        for index in tmp_index:
            if index is not None:
                info_dict = {kv.key: kv.value for kv in index.params}
                if info_dict.get("params", None):
                    info_dict["params"] = json.loads(info_dict["params"])

                index_info = Index(self, index.field_name, info_dict, index_name=index.index_name, construct_only=True)
                indexes.append(index_info)
        return indexes

    def index(self, **kwargs) -> Index:
        """
        Fetches the index object of the of the specified name.

        :param kwargs:
            * *index_name* (``str``) --
              The name of index. If no index is specified, the default index name is used.

        :return Index:
            Index object corresponding to index_name.

        :raises CollectionNotExistException: If the collection does not exist.
        :raises BaseException: If the specified index does not exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
            [<pymilvus.index.Index object at 0x7f4435587e20>]
            >>> collection.index()
            <pymilvus.index.Index object at 0x7f44355a1460>
        """
        copy_kwargs = copy.deepcopy(kwargs)
        index_name = copy_kwargs.pop("index_name", DefaultConfigs.IndexName)
        conn = self._get_connection()
        tmp_index = conn.describe_index(self._name, index_name, **copy_kwargs)
        if tmp_index is not None:
            field_name = tmp_index.pop("field_name", None)
            index_name = tmp_index.pop("index_name", index_name)
            return Index(self, field_name, tmp_index, construct_only=True, index_name=index_name)
        raise IndexNotExistException(message=ExceptionsMessage.IndexNotExist)

    def create_index(self, field_name, index_params={}, timeout=None, **kwargs) -> Index:
        """
        Creates index for a specified field. Return Index Object.

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param index_params: The indexing parameters.
        :type  index_params: dict

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a IndexFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.
            * *index_name* (``str``) --
              The name of index which will be created. Then you can use the index name to check the state of index.
              If no index name is specified, the default index name is used.

        :raises CollectionNotExistException: If the collection does not exist.
        :raises ParamError: If the index parameters are invalid.
        :raises BaseException: If field does not exist.
        :raises BaseException: If the index has been created.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
            <pymilvus.index.Index object at 0x7f44355a1460>
        """
        conn = self._get_connection()
        return conn.create_index(self._name, field_name, index_params,
                                 timeout=timeout, **kwargs)

    def has_index(self, timeout=None, **kwargs) -> bool:
        """
        Checks whether a specified index exists.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :param kwargs:
            * *index_name* (``str``) --
              The name of index. If no index is specified, the default index name is used.

        :return bool:
            Whether the specified index exists.

        :raises CollectionNotExistException: If the collection does not exist.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        copy_kwargs = copy.deepcopy(kwargs)
        index_name = copy_kwargs.pop("index_name", DefaultConfigs.IndexName)
        # TODO(yukun): Need field name, but provide index name
        if conn.describe_index(self._name, index_name, timeout=timeout, **copy_kwargs) is None:
            return False
        return True

    def drop_index(self, timeout=None, **kwargs):
        """
        Drop index and its corresponding index files.

        :param timeout:
            * *timeout* (``float``) --
              An optional duration of time in seconds to allow for the RPC. If timeout
              is set to None, the client keeps waiting until the server responds or an error occurs.
              Optional. A duration of time in seconds.

        :param kwargs:
            * *index_name* (``str``) --
              The name of index. If no index is specified, the default index name is used.

        :raises CollectionNotExistException: If the collection does not exist.
        :raises BaseException: If the index does not exist or has been dropped.

        :example:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
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
        copy_kwargs = copy.deepcopy(kwargs)
        index_name = copy_kwargs.pop("index_name", DefaultConfigs.IndexName)
        conn = self._get_connection()
        tmp_index = conn.describe_index(self._name, index_name, timeout=timeout, **copy_kwargs)
        if tmp_index is not None:
            index = Index(self, tmp_index['field_name'], tmp_index, construct_only=True, index_name=index_name)
            index.drop(timeout=timeout, **kwargs)

    def compact(self, timeout=None, **kwargs):
        """
        Compact merge the small segments in a collection

        :param collection_name: The name of the collection.
        :type  collection_name: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :raises BaseException: If the collection does not exist.

        :example:
        """
        conn = self._get_connection()
        self.compaction_id = conn.compact(self._name, timeout=timeout, **kwargs)

    def get_compaction_state(self, timeout=None, **kwargs) -> CompactionState:
        """
        get_compaction_state returns the current collection's compaction state

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :raises BaseException: If the collection does not exist.

        :example:
        """
        conn = self._get_connection()
        return conn.get_compaction_state(self.compaction_id, timeout=timeout, **kwargs)

    def wait_for_compaction_completed(self, timeout=None, **kwargs) -> CompactionState:
        """
        Block until the current collection's compaction completed

        :param timeout: The timeout for this method, unit: second
                        when timeout is set to None, client waits until compaction completed or error occur
        :type  timeout: float

        :raises BaseException: If the time is up and the compression has not been completed

        :example:
        """
        conn = self._get_connection()
        return conn.wait_for_compaction_completed(self.compaction_id, timeout=timeout, **kwargs)

    def get_compaction_plans(self, timeout=None, **kwargs) -> CompactionPlans:
        """
        get_compaction_state returns the current collection's compaction state

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :raises BaseException: If the collection does not exist.

        :example:
        """
        conn = self._get_connection()
        return conn.get_compaction_plans(self.compaction_id, timeout=timeout, **kwargs)

    def get_replicas(self, timeout=None, **kwargs) -> Replica:
        """get_replicas returns the current collection's replica information

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :raises BaseException: If the collection does not exist.

        :example:
        """
        conn = self._get_connection()
        return conn.get_replicas(self.name, timeout=timeout, **kwargs)

    def describe(self):
        conn = self._get_connection()
        return conn.describe_collection(self.name)
