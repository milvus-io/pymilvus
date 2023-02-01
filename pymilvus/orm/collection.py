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
from ..client.types import CompactionState, CompactionPlans, Replica, get_consistency_level, cmp_consistency_level
from ..client.constants import DEFAULT_CONSISTENCY_LEVEL
from ..client.configs import DefaultConfigs



class Collection:
    def __init__(self, name: str, schema: CollectionSchema=None, using: str="default", shards_num: int=2, **kwargs):
        """ Constructs a collection by name, schema and other parameters.

        Args:
            name (``str``): the name of collection
            schema (``CollectionSchema``, optional): the schema of collection, defaults to None.
            using (``str``, optional): Milvus connection alias name, defaults to 'default'.
            shards_num (``int``, optional): how many shards will the insert data be divided, defaults to 2.
            **kwargs (``dict``):

                * *consistency_level* (``int/ str``)
                    Which consistency level to use when searching in the collection.
                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: this parameter can be overwritten by the same parameter specified in search.

                * *properties* (``dict``, optional)
                    Collection properties.

                * *timeout* (``float``)
                    An optional duration of time in seconds to allow for the RPCs.
                    If timeout is not set, the client keeps waiting until the server responds or an error occurs.

        Raises:
            SchemaNotReadyException: if the schema is wrong.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> schema = CollectionSchema(fields=fields)
            >>> properties = {"collection.ttl.seconds": 1800}
            >>> collection = Collection(name="test_collection_init", schema=schema, properties=properties)
            >>> collection.name
            'test_collection_init'
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

        self._schema_dict = self._schema.to_dict()
        self._schema_dict["consistency_level"] = self._consistency_level

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
        """CollectionSchema: schema of the collection. """
        return self._schema

    @property
    def aliases(self, **kwargs) -> list:
        """List[str]: all the aliases of the collection. """
        conn = self._get_connection()
        resp = conn.describe_collection(self._name, **kwargs)
        aliases = resp["aliases"]
        return aliases

    @property
    def description(self) -> str:
        """str: a text description of the collection. """
        return self._schema.description

    @property
    def name(self) -> str:
        """str: the name of the collection. """
        return self._name

    @property
    def is_empty(self) -> bool:
        """bool: whether the collection is empty or not."""
        return self.num_entities == 0

    @property
    def num_entities(self, **kwargs) -> int:
        """int: The number of entities in the collection, not real time.

        Examples:
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
            0
            >>> collection.flush()
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
        """FieldSchema: the primary field of the collection."""
        return self._schema.primary_field

    def flush(self, timeout=None, **kwargs):
        """ Seal all segments in the collection. Inserts after flushing will be written into
            new segments. Only sealed segments can be indexed.

        Args:
            timeout (float): an optional duration of time in seconds to allow for the RPCs.
                If timeout is not set, the client keeps waiting until the server responds or an error occurs.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> schema = CollectionSchema(fields=fields)
            >>> collection = Collection(name="test_collection_flush", schema=schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.flush()
            >>> collection.num_entities
            2
        """
        conn = self._get_connection()
        conn.flush([self.name], timeout=timeout, **kwargs)

    def drop(self, timeout=None, **kwargs):
        """ Drops the collection. The same as `utility.drop_collection()`

        Args:
            timeout (float, optional): an optional duration of time in seconds to allow for the RPCs.
                If timeout is not set, the client keeps waiting until the server responds or an error occurs.

        Examples:
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
        """ Set properties for the collection

        Args:
            properties (``dict``): collection properties.
                 only support collection TTL with key `collection.ttl.seconds`
            timeout (``float``, optional): an optional duration of time in seconds to allow for the RPCs.
                If timeout is not set, the client keeps waiting until the server responds or an error occurs.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> fields = [
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
            ... ]
            >>> schema = CollectionSchema(fields=fields)
            >>> collection = Collection("test_set_properties", schema)
            >>> collection.set_properties({"collection.ttl.seconds": 60})
        """
        conn = self._get_connection()
        conn.alter_collection(self.name, properties=properties, timeout=timeout)

    def load(self, partition_names=None, replica_number=1, timeout=None, **kwargs):
        """ Load the data into memory.

        Args:
            partition_names (``List[str]``): The specified partitions to load.
            replica_number (``int``, optional): The replica number to load, defaults to 1.
            timeout (``float``, optional): an optional duration of time in seconds to allow for the RPCs.
                If timeout is not set, the client keeps waiting until the server responds or an error occurs.
            **kwargs (``dict``, optional):

                * *_async*(``bool``)
                    Indicate if invoke asynchronously.

                * *_refresh*(``bool``)
                    Whether to enable refresh mode(renew the segment list of this collection before loading).
                * *_resource_groups(``List[str]``)
                    Specify resource groups which can be used during loading.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_load", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.create_index("films", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
            >>> collection.load()
        """
        conn = self._get_connection()
        if partition_names is not None:
            conn.load_partitions(self._name, partition_names, replica_number=replica_number, timeout=timeout, **kwargs)
        else:
            conn.load_collection(self._name, replica_number=replica_number, timeout=timeout, **kwargs)

    def release(self, timeout=None, **kwargs):
        """ Releases the collection data from memory.

        Args:
            timeout (``float``, optional): an optional duration of time in seconds to allow for the RPCs.
                If timeout is not set, the client keeps waiting until the server responds or an error occurs.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_release", schema)
            >>> collection.insert([[1, 2], [[1.0, 2.0], [3.0, 4.0]]])
            >>> collection.create_index("films", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
            >>> collection.load()
            >>> collection.release()
        """
        conn = self._get_connection()
        conn.release_collection(self._name, timeout=timeout, **kwargs)

    def insert(self, data: [List, pandas.DataFrame], partition_name: str=None, timeout=None, **kwargs) -> MutationResult:
        """ Insert data into the collection.

        Args:
            data (``list/tuple/pandas.DataFrame``): The specified data to insert
            partition_name (``str``): The partition name which the data will be inserted to,
                if partition name is not passed, then the data will be inserted to "_default" partition
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC. Defaults to None.
                If timeout is set to None, the client keeps waiting until the server responds or an error occurs.
        Returns:
            MutationResult: contains 2 properties `insert_count`, and, `primary_keys`
                `insert_count`: how may entites have been inserted into Milvus,
                `primary_keys`: list of primary keys of the inserted entities
        Raises:
            MilvusException: If anything goes wrong.

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
        entities = Prepare.prepare_insert_data(data, self._schema)

        conn = self._get_connection()
        res = conn.batch_insert(self._name, entities, partition_name,
                                timeout=timeout, schema=self._schema_dict, **kwargs)

        if kwargs.get("_async", False):
            return MutationFuture(res)
        return MutationResult(res)

    def delete(self, expr, partition_name=None, timeout=None, **kwargs):
        """ Delete entities with an expression condition.

        Args:
            expr (``str``): The specified data to insert.
            partition_names (``List[str]``): Name of partitions to delete entities.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC. Defaults to None.
                If timeout is set to None, the client keeps waiting until the server responds or an error occurs.

        Returns:
            MutationResult: contains `delete_count` properties represents how many entities might be deleted.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("film_date", DataType.INT64),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2),
            ... ])
            >>> collection = Collection("test_collection_delete", schema)
            >>> # insert
            >>> data = [
            ...     [i for i in range(10)],
            ...     [i + 2000 for i in range(10)],
            ...     [[random.random() for _ in range(2)] for _ in range(10)],
            ... ]
            >>> collection.insert(data)
            >>> res = collection.delete("film_id in [ 0, 1 ]")
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
        """ Conducts a vector similarity search with an optional boolean expression as filter.

        Args:
            data (``List[List[float]]``): The vectors of search data.
                the length of data is number of query (nq), and the dim of every vector in data must be equal to
                the vector field's of collection.
            anns_field (``str``): The name of the vector field used to search of collection.
            param (``dict[str, Any]``):

                The parameters of search. The followings are valid keys of param.

                * *nprobe*, *ef*, *search_k*, etc
                    Corresponding search params for a certain index.

                * *metric_type* (``str``)
                    similar metricy types, the value must be of type str.

                * *offset* (``int``, optional)
                    offset for pagination.

                * *limit* (``int``, optional)
                    limit for the search results and pagination.

                example for param::

                    {
                        "nprobe": 128,
                        "metric_type": "L2",
                        "offset": 10,
                        "limit": 10,
                    }

            limit (``int``): The max number of returned record, also known as `topk`.
            expr (``str``): The boolean expression used to filter attribute. Default to None.

                example for expr::

                    "id_field >= 0", "id_field in [1, 2, 3, 4]"

            partition_names (``List[str]``, optional): The names of partitions to search on. Default to None.
            output_fields (``List[str]``, optional):
                The name of fields to return in the search result.  Can only get scalar fields.
            round_decimal (``int``, optional): The specified number of decimal places of returned distance.
                Defaults to -1 means no round to returned distance.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC. Defaults to None.
                If timeout is set to None, the client keeps waiting until the server responds or an error occurs.
            **kwargs (``dict``): Optional search params

                *  *_async* (``bool``, optional)
                    Indicate if invoke asynchronously.
                    Returns a SearchFuture if True, else returns results from server directly.

                * *_callback* (``function``, optional)
                    The callback function which is invoked after server response successfully.
                    It functions only if _async is set to True.

                * *consistency_level* (``str/int``, optional)
                    Which consistency level to use when searching in the collection.

                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: this parameter will overwrite the same parameter specified when user created the collection,
                    if no consistency level was specified, search will use the consistency level when you create the
                    collection.

                * *guarantee_timestamp* (``int``, optional)
                    Instructs Milvus to see all operations performed before this timestamp.
                    By default Milvus will search all operations performed to date.

                    Note: only valid in Customized consistency level.

                * *graceful_time* (``int``, optional)
                    Search will use the (current_timestamp - the graceful_time) as the
                    `guarantee_timestamp`. By default with 5s.

                    Note: only valid in Bounded consistency level

                * *travel_timestamp* (``int``, optional)
                    A specific timestamp to get results based on a data view at.

        Returns:
            SearchResult:
                Returns ``SearchResult`` if `_async` is False , otherwise ``SearchFuture``

        .. _Metric type documentations:
            https://milvus.io/docs/v2.2.x/metric.md
        .. _Index documentations:
            https://milvus.io/docs/v2.2.x/index.md
        .. _How guarantee ts works:
            https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md

        Raises:
            MilvusException: If anything goes wrong

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> import random
            >>> connections.connect()
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
            >>> collection.create_index("films", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
            >>> collection.load()
            >>> # search
            >>> search_param = {
            ...     "data": [[1.0, 1.0]],
            ...     "anns_field": "films",
            ...     "param": {"metric_type": "L2", "offset": 1},
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
        res = conn.search(self._name, data, anns_field, param, limit, expr,
                          partition_names, output_fields, round_decimal, timeout=timeout,
                          schema=self._schema_dict, **kwargs)
        if kwargs.get("_async", False):
            return SearchFuture(res)
        return SearchResult(res)

    def query(self, expr, output_fields=None, partition_names=None, timeout=None, **kwargs):
        """ Query with expressions

        Args:
            expr (``str``): The query expression.
            output_fields(``List[str]``): A list of field names to return. Defaults to None.
            partition_names: (``List[str]``, optional): A list of partition names to query in. Defaults to None.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC. Defaults to None.
                If timeout is set to None, the client keeps waiting until the server responds or an error occurs.
            **kwargs (``dict``, optional):

                * *consistency_level* (``str/int``, optional)
                    Which consistency level to use when searching in the collection.

                    Options of consistency level: Strong, Bounded, Eventually, Session, Customized.

                    Note: this parameter will overwrite the same parameter specified when user created the collection,
                    if no consistency level was specified, search will use the consistency level when you create the
                    collection.

                * *guarantee_timestamp* (``int``, optional)
                    Instructs Milvus to see all operations performed before this timestamp.
                    By default Milvus will search all operations performed to date.

                    Note: only valid in Customized consistency level.

                * *graceful_time* (``int``, optional)
                    Search will use the (current_timestamp - the graceful_time) as the
                    `guarantee_timestamp`. By default with 5s.

                    Note: only valid in Bounded consistency level

                * *travel_timestamp* (``int``, optional)
                    A specific timestamp to get results based on a data view at.

                * *offset* (``int``)
                    Combined with limit to enable pagination

                * *limit* (``int``)
                    Combined with limit to enable pagination

        Returns:
            List, contains all results

        Raises:
            MilvusException: If anything goes wrong

        Examples:
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
            >>> collection.create_index("films", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
            >>> collection.load()
            >>> # query
            >>> expr = "film_id <= 1"
            >>> res = collection.query(expr, output_fields=["film_date"], offset=1, limit=1)
            >>> assert len(res) == 1
            >>> print(f"- Query results: {res}")
            - Query results: [{'film_id': 1, 'film_date': 2001}]
        """
        if not isinstance(expr, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        res = conn.query(self._name, expr, output_fields, partition_names,
                         timeout=timeout, schema=self._schema_dict, **kwargs)
        return res

    @property
    def partitions(self, **kwargs) -> List[Partition]:
        """ List[Partition]: List of Partition object.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
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
        """ Get the existing partition object according to name. Return None if not existed.

        Args:
            partition_name (``str``): The name of the partition to get.

        Returns:
            Partition: Partition object corresponding to partition_name.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_partition", schema)
            >>> collection.partition("_default")
            {"name": "_default", "description": "", "num_entities": 0}
        """
        if self.has_partition(partition_name, **kwargs) is False:
            return None
        return Partition(self, partition_name, construct_only=True, **kwargs)

    def create_partition(self, partition_name, description="", **kwargs) -> Partition:
        """ Create a new partition corresponding to name if not existed.

        Args:
            partition_name (``str``): The name of the partition to create.
            description (``str``, optional): The description of this partition.

        Returns:
            Partition: Partition object corresponding to partition_name.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
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
        """ Checks if a specified partition exists.

        Args:
            partition_name (``str``): The name of the partition to check.
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.

        Returns:
            bool: True if exists, otherwise false.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
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
        """ Drop the partition in this collection.

        Args:
            partition_name (``str``): The name of the partition to drop.
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.

        Raises:
            PartitionNotExistException: If the partition doesn't exists.
            MilvusException: If anything goes wrong.

        Examples:
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
    def indexes(self, **kwargs) -> List[Index]:
        """List[Index]: list of indexes of this collection.

        Examples:
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
        """Get the index object of index name.

        Args:
            **kwargs (``dict``):
                * *index_name* (``str``)
                    The name of index. If no index is specified, the default index name is used.

        Returns:
            Index: Index object corresponding to index_name.

        Raises:
            IndexNotExistException: If the index doesn't exists.

        Examples:
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

    def create_index(self, field_name, index_params={}, timeout=None, **kwargs):
        """Creates index for a specified field, with a index name.

        Args:
            field_name (``str``): The name of the field to create index
            index_params (``dict``): The parameters to index
                * *index_type* (``str``)
                    "index_type" as the key, example values: "FLAT", "IVF_FLAT", etc.

                * *metric_type* (``str``)
                    "metric_type" as the key, examples values: "L2", "IP", "JACCARD".

                * *params* (``dict``)
                    "params" as the key, corresponding index params.

            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.
            index_name (``str``): The name of index which will be created, must be unique.
                If no index name is specified, the default index name will be used.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            >>> connections.connect()
            >>> schema = CollectionSchema([
            ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
            ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
            ... ])
            >>> collection = Collection("test_collection_create_index", schema)
            >>> index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
            >>> collection.create_index("films", index_params, index_name="idx")
            Status(code=0, message='')
        """
        conn = self._get_connection()
        return conn.create_index(self._name, field_name, index_params, timeout=timeout, **kwargs)

    def has_index(self, timeout=None, **kwargs) -> bool:
        """ Check whether a specified index exists.

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.

            **kwargs (``dict``):
                * *index_name* (``str``)
                  The name of index. If no index is specified, the default index name will be used.

        Returns:
            bool: Whether the specified index exists.

        Examples:
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
        if conn.describe_index(self._name, index_name, timeout=timeout, **copy_kwargs) is None:
            return False
        return True

    def drop_index(self, timeout=None, **kwargs):
        """ Drop index and its corresponding index files.
        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.

            **kwargs (``dict``):
                * *index_name* (``str``)
                  The name of index. If no index is specified, the default index name will be used.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
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
        """ Compact merge the small segments in a collection

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.

        Raises:
            MilvusException: If anything goes wrong.
        """
        conn = self._get_connection()
        self.compaction_id = conn.compact(self._name, timeout=timeout, **kwargs)

    def get_compaction_state(self, timeout=None, **kwargs) -> CompactionState:
        """ Get the current compaction state

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.

        Raises:
            MilvusException: If anything goes wrong.
        """
        conn = self._get_connection()
        return conn.get_compaction_state(self.compaction_id, timeout=timeout, **kwargs)

    def wait_for_compaction_completed(self, timeout=None, **kwargs) -> CompactionState:
        """ Block until the current collection's compaction completed

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.

        Raises:
            MilvusException: If anything goes wrong.
        """
        conn = self._get_connection()
        return conn.wait_for_compaction_completed(self.compaction_id, timeout=timeout, **kwargs)

    def get_compaction_plans(self, timeout=None, **kwargs) -> CompactionPlans:
        """Get the current compaction plans

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.
        Returns:
            CompactionPlans: All the plans' states of this compaction.
        """
        conn = self._get_connection()
        return conn.get_compaction_plans(self.compaction_id, timeout=timeout, **kwargs)

    def get_replicas(self, timeout=None, **kwargs) -> Replica:
        """Get the current loaded replica information

        Args:
            timeout (``float``, optional): An optional duration of time in seconds to allow for the RPC. When timeout
                is set to None, client waits until server response or error occur.
        Returns:
            Replica: All the replica information.
        """
        conn = self._get_connection()
        return conn.get_replicas(self.name, timeout=timeout, **kwargs)

    def describe(self, timeout=None):
        conn = self._get_connection()
        return conn.describe_collection(self.name, timeout=timeout)
