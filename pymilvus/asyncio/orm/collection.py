import asyncio
import typing

import pandas

from ...orm.collection import (
    AbstractCollection, DataTypeNotMatchException, ExceptionsMessage, 
    SearchResult, CollectionSchema,
    DEFAULT_CONSISTENCY_LEVEL,
    cmp_consistency_level,
    SchemaNotReadyException,
    check_schema,
    get_consistency_level,
    MutationResult,
    check_insert_data_schema,
    Prepare,
)
from ..client.grpc_handler import GrpcHandler as AsyncGrpcHandler
from .connections import connections, Connections as AsyncConnections


class Collection(AbstractCollection[AsyncConnections]):
    connections = connections

    def _init(self):
        self._ready = asyncio.create_task(self._async_init())

    # DEBUG
    def __getattr__(self, attr):
        if attr in ('_schema', '_schema_dict'):
            raise AssertionError(f"await self._ready before accessing self.{attr}")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    # DEBUG
    def _get_connection(self):
        ret = super()._get_connection()
        assert isinstance(ret, AsyncGrpcHandler)
        return ret

    async def _async_init(self):
        schema = self._init_schema
        kwargs = self._kwargs
        conn = self._get_connection()

        has = await conn.has_collection(self._name, **kwargs)
        if has:
            resp = await conn.describe_collection(self._name, **kwargs)
            consistency_level = resp.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
            arg_consistency_level = kwargs.get("consistency_level", consistency_level)
            if not cmp_consistency_level(consistency_level, arg_consistency_level):
                raise SchemaNotReadyException(message=ExceptionsMessage.ConsistencyLevelInconsistent)
            server_schema = CollectionSchema.construct_from_dict(resp)
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
                raise SchemaNotReadyException(message=ExceptionsMessage.CollectionNotExistNoSchema % self._name)
            if isinstance(schema, CollectionSchema):
                check_schema(schema)
                consistency_level = get_consistency_level(kwargs.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL))
                await conn.create_collection(self._name, schema, shards_num=self._shards_num, **kwargs)
                self._schema = schema
            else:
                raise SchemaNotReadyException(message=ExceptionsMessage.SchemaType)

        self._schema_dict = self._schema.to_dict()
        self._schema_dict["consistency_level"] = consistency_level

    async def load(self, partition_names=None, replica_number=1, timeout=None, **kwargs):
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
            await conn.load_partitions(
                self._name, partition_names, replica_number=replica_number, timeout=timeout, **kwargs,
            )
        else:
            await conn.load_collection(
                self._name, replica_number=replica_number, timeout=timeout, **kwargs,
            )


    async def insert(
        self,
        data: typing.Union[typing.List, pandas.DataFrame],
        partition_name: str = None, timeout=None, **kwargs
    ) -> MutationResult:
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
        await self._ready
        if data is None:
            return MutationResult(data)
        check_insert_data_schema(self._schema, data)
        entities = Prepare.prepare_insert_data(data, self._schema)

        conn = self._get_connection()

        res = await conn.batch_insert(
            self._name, entities, partition_name, timeout=timeout, schema=self._schema_dict, **kwargs,
        )

        return MutationResult(res)

    async def delete(self, expr, partition_name=None, timeout=None, **kwargs):
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
        res = await conn.delete(self._name, expr, partition_name, timeout=timeout, **kwargs)
        return MutationResult(res)

    async def search(self, data, anns_field, param, limit, expr=None, partition_names=None,
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
                Returns ``SearchResult``

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
        await self._ready
        if expr is not None and not isinstance(expr, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))

        conn = self._get_connection()
        res = await conn.search(
            self._name, data, anns_field, param, limit, expr,
            partition_names, output_fields, round_decimal, timeout=timeout,
            schema=self._schema_dict, **kwargs)
        return SearchResult(res)

    async def create_index(self, field_name, index_params={}, timeout=None, **kwargs):
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
        return await conn.create_index(self._name, field_name, index_params, timeout=timeout, **kwargs)

