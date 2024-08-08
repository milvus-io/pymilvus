from urllib import parse

from pymilvus.decorators import deprecated
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.settings import Config
from typing import Mapping

from .check import is_legal_host, is_legal_port
from .grpc_handler import GrpcHandler
from .types import (
    BulkInsertState,
    CompactionPlans,
    CompactionState,
    Replica,
    ResourceGroupInfo,
    ResourceGroupConfig,
)


class Milvus:
    @deprecated
    def __init__(
        self, host=None, port=Config.GRPC_PORT, uri=Config.GRPC_URI, channel=None, **kwargs
    ) -> None:
        self.address = self.__get_address(host, port, uri)
        self._handler = GrpcHandler(address=self.address, channel=channel, **kwargs)

        if kwargs.get("pre_ping", False) is True:
            self._handler._wait_for_channel_ready()

    def __get_address(self, host=None, port=Config.GRPC_PORT, uri=Config.GRPC_URI):
        if host is None and uri is None:
            raise ParamError(message="Host and uri cannot both be None")

        if host is None:
            try:
                parsed_uri = parse.urlparse(uri, "tcp")
            except Exception as e:
                raise ParamError(message=f"Illegal uri [{uri}]: {e}") from e

            host, port = parsed_uri.hostname, parsed_uri.port

        host, port = str(host), str(port)
        if not (is_legal_host(host) and is_legal_port(port)):
            raise ParamError(message=f"Illegal host [{host}] or port [{port}]")

        return f"{host}:{port}"

    def _connection(self):
        return self.handler

    @property
    def name(self):
        return self._name

    @property
    def handler(self):
        return self._handler

    def get_server_type(self):
        return self._handler.get_server_type()

    def reset_password(self, user, old_password, new_password):
        self._handler.reset_password(user, old_password, new_password)

    def close(self):
        if self._handler is None:
            raise MilvusException(message="Closing on closed handler")
        self.handler.close()
        self._handler = None

    def create_collection(self, collection_name, fields, timeout=None, **kwargs):
        """Creates a collection.

        :param collection_name: The name of the collection. A collection name can only include
        numbers, letters, and underscores, and must not begin with a number.
        :type  collection_name: str

        :param fields: Field parameters.
        :type  fields: dict

            ` {"fields": [
                    {"field": "A", "type": DataType.INT32}
                    {"field": "B", "type": DataType.INT64},
                    {"field": "C", "type": DataType.FLOAT},
                    {"field": "Vec", "type": DataType.FLOAT_VECTOR,
                     "params": {"dim": 128}}
                ],
            "auto_id": True}`

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param kwargs:
            * *num_shards* (``int``) --
            How wide to scale collection. Corresponds to how many active datanodes can be used on insert.
            * *shards_num* (``int``, deprecated) --
            How wide to scale collection. Corresponds to how many active datanodes can be used on insert.
            * *consistency_level* (``str/int``) --
            Which consistency level to use when searching in the collection. For details, see
            https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md.
            Note: this parameter can be overwritten by the same parameter specified in search.
            * *properties* (``dict``) --


        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.create_collection(collection_name, fields, timeout=timeout, **kwargs)

    def drop_collection(self, collection_name, timeout=None):
        """
        Delete a specified collection.

        :param collection_name: The name of the collection to delete.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.drop_collection(collection_name, timeout=timeout)

    def has_collection(self, collection_name, timeout=None):
        """
        Checks whether a specified collection exists.

        :param collection_name: The name of the collection to check.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: If specified collection exists
        :rtype: bool

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.has_collection(collection_name, timeout=timeout)

    def describe_collection(self, collection_name, timeout=None):
        """
        Returns the schema of specified collection.
        Example: {'collection_name': 'create_collection_eXgbpOtn', 'auto_id': True, 'description': '',
                 'fields': [{'field_id': 100, 'name': 'INT32', 'description': '', 'type': 4, 'params': {},
                 {'field_id': 101, 'name': 'FLOAT_VECTOR', 'description': '', 'type': 101,
                 'params': {'dim': '128'}}]}

        :param collection_name: The name of the collection to describe.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: The schema of collection to describe.
        :rtype: dict

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.describe_collection(collection_name, timeout=timeout)

    def load_collection(self, collection_name, replica_number=1, timeout=None, **kwargs):
        """
        Loads a specified collection from disk to memory.

        :param collection_name: The name of the collection to load.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param replica_number: Number of replication in memory to load
        :type replica_number: int

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.load_collection(
                collection_name, replica_number, timeout=timeout, **kwargs
            )

    def release_collection(self, collection_name, timeout=None):
        """
        Clear collection data from memory.

        :param collection_name: The name of collection to release.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.release_collection(collection_name=collection_name, timeout=timeout)

    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        """
        Returns collection statistics information.
        Example: {"row_count": 10}

        :param collection_name: The name of collection.
        :type  collection_name: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: statistics information
        :rtype: dict

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            stats = handler.get_collection_stats(collection_name, timeout=timeout, **kwargs)
            result = {stat.key: stat.value for stat in stats}
            result["row_count"] = int(result["row_count"])
            return result

    def list_collections(self, timeout=None):
        """
        Returns a list of all collection names.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: List of collection names, return when operation is successful
        :rtype: list[str]

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.list_collections(timeout=timeout)

    def create_partition(self, collection_name, partition_name, timeout=None):
        """
        Creates a partition in a specified collection. You only need to import the
        parameters of partition_name to create a partition. A collection cannot hold
        partitions of the same tag, whilst you can insert the same tag in different collections.

        :param collection_name: The name of the collection to create partitions in.
        :type  collection_name: str

        :param partition_name: The tag name of the partition to create.
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.create_partition(collection_name, partition_name, timeout=timeout)

    def drop_partition(self, collection_name, partition_name, timeout=None):
        """
        Deletes the specified partition in a collection. Note that the default partition
        '_default' is not permitted to delete. When a partition deleted, all data stored in it
        will be deleted.

        :param collection_name: The name of the collection to delete partitions from.
        :type  collection_name: str

        :param partition_name: The tag name of the partition to delete.
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.drop_partition(collection_name, partition_name, timeout=timeout)

    def has_partition(self, collection_name, partition_name, timeout=None):
        """
        Checks if a specified partition exists in a collection.

        :param collection_name: The name of the collection to find the partition in.
        :type  collection_name: str

        :param partition_name: The tag name of the partition to check
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: Whether a specified partition exists in a collection.
        :rtype: bool

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.has_partition(collection_name, partition_name, timeout=timeout)

    def load_partitions(self, collection_name, partition_names, replica_number=1, timeout=None):
        """
        Load specified partitions from disk to memory.

        :param collection_name: The collection name which partitions belong to.
        :type  collection_name: str

        :param partition_names: The specified partitions to load.
        :type  partition_names: list[str]

        :param replica_number: The replication numbers to load.
        :type  replica_number: int

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.load_partitions(
                collection_name=collection_name,
                partition_names=partition_names,
                replica_number=replica_number,
                timeout=timeout,
            )

    def release_partitions(self, collection_name, partition_names, timeout=None):
        """
        Clear partitions data from memory.

        :param collection_name: The collection name which partitions belong to.
        :type  collection_name: str

        :param partition_names: The specified partition to release.
        :type  partition_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.release_partitions(
                collection_name=collection_name, partition_names=partition_names, timeout=timeout
            )

    def list_partitions(self, collection_name, timeout=None):
        """
        Returns a list of all partition tags in a specified collection.

        :param collection_name: The name of the collection to retrieve partition tags from.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: A list of all partition tags in specified collection.
        :rtype: list[str]

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.list_partitions(collection_name, timeout=timeout)

    def get_partition_stats(self, collection_name, partition_name, timeout=None, **kwargs):
        """
        Returns partition statistics information.
        Example: {"row_count": 10}

        :param collection_name: The name of collection.
        :type  collection_name: str.

        :param partition_name: The name of partition.
        :type  partition_name: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: statistics information
        :rtype: dict

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            stats = handler.get_partition_stats(
                collection_name, partition_name, timeout=timeout, **kwargs
            )
            result = {stat.key: stat.value for stat in stats}
            result["row_count"] = int(result["row_count"])
            return result

    def create_alias(self, collection_name, alias, timeout=None, **kwargs):
        """
        Specify alias for a collection.
        Alias cannot be duplicated, you can't assign same alias to different collections.
        But you can specify multiple aliases for a collection, for example:
            before create_alias("collection_1", "bob"):
                collection_1's aliases = ["tom"]
            after create_alias("collection_1", "bob"):
                collection_1's aliases = ["tom", "bob"]

        :param collection_name: The name of collection.
        :type  collection_name: str.

        :param alias: The alias of the collection.
        :type  alias: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.create_alias(collection_name, alias, timeout=timeout, **kwargs)

    def drop_alias(self, alias, timeout=None, **kwargs):
        """
        Delete an alias.
        This api no need to specify collection name because the milvus server knows which collection it belongs.
        For example:
            before drop_alias("bob"):
                collection_1's aliases = ["tom", "bob"]
            after drop_alias("bob"):
                collection_1's aliases = ["tom"]

        :param alias: The alias to be deleted.
        :type  alias: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.drop_alias(alias, timeout=timeout, **kwargs)

    def alter_alias(self, collection_name, alias, timeout=None, **kwargs):
        """
        Change alias of a collection to another collection. If the alias doesn't exist, the api will return error.
        Alias cannot be duplicated, you can't assign same alias to different collections.
        This api can change alias owner collection, for example:
            before alter_alias("collection_2", "bob"):
                collection_1's aliases = ["bob"]
                collection_2's aliases = []
            after alter_alias("collection_2", "bob"):
                collection_1's aliases = []
                collection_2's aliases = ["bob"]

        :param collection_name: The name of collection.
        :type  collection_name: str.

        :param alias: The new alias of the collection.
        :type  alias: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.alter_alias(collection_name, alias, timeout=timeout, **kwargs)

    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        """
        Creates an index for a field in a specified collection. Milvus does not support creating multiple
        indexes for a field. In a scenario where the field already has an index, if you create another one,
        the server will replace the existing index files with the new ones.

        Note that you need to call load_collection() or load_partitions() to make the new index take effect
        on searching tasks.

        :param collection_name: The name of the collection to create field indexes.
        :type  collection_name: str

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param params: Indexing parameters.
        :type  params: dict
            There are examples of supported indexes:

            IVF_FLAT:
                ` {
                    "metric_type":"L2",
                    "index_type": "IVF_FLAT",
                    "params":{"nlist": 1024}
                }`

            IVF_PQ:
                `{
                    "metric_type": "L2",
                    "index_type": "IVF_PQ",
                    "params": {"nlist": 1024, "m": 8, "nbits": 8}
                }`

            IVF_SQ8:
                `{
                    "metric_type": "L2",
                    "index_type": "IVF_SQ8",
                    "params": {"nlist": 1024}
                }`

            BIN_IVF_FLAT:
                `{
                    "metric_type": "JACCARD",
                    "index_type": "BIN_IVF_FLAT",
                    "params": {"nlist": 1024}
                }`

            HNSW:
                `{
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {"M": 48, "efConstruction": 50}
                }`

            RHNSW_FLAT:
                `{
                    "metric_type": "L2",
                    "index_type": "RHNSW_FLAT",
                    "params": {"M": 48, "efConstruction": 50}
                }`

            RHNSW_PQ:
                `{
                    "metric_type": "L2",
                    "index_type": "RHNSW_PQ",
                    "params": {"M": 48, "efConstruction": 50, "PQM": 8}
                }`

            RHNSW_SQ:
                `{
                    "metric_type": "L2",
                    "index_type": "RHNSW_SQ",
                    "params": {"M": 48, "efConstruction": 50}
                }`

            ANNOY:
                `{
                    "metric_type": "L2",
                    "index_type": "ANNOY",
                    "params": {"n_trees": 8}
                }`

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
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
              If no index name is specified, default index name is used.

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.create_index(
                collection_name, field_name, params, timeout=timeout, **kwargs
            )

    def drop_index(self, collection_name, field_name, timeout=None):
        """
        Removes the index of a field in a specified collection.

        :param collection_name: The name of the collection to remove the field index from.
        :type  collection_name: str

        :param field_name: The name of the field to remove the index of.
        :type  field_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.drop_index(
                collection_name=collection_name,
                field_name=field_name,
                index_name="",
                timeout=timeout,
            )

    def describe_index(self, collection_name, index_name="", timeout=None):
        """
        Returns the schema of index built on specified field.
        Example: {'index_type': 'FLAT', 'metric_type': 'L2', 'params': {'nlist': 128}}

        :param collection_name: The name of the collection which field belong to.
        :type  collection_name: str

        :param field_name: The name of field to describe.
        :type  field_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: the schema of index built on specified field.
        :rtype: dict

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.describe_index(collection_name, index_name, timeout=timeout)

    def insert(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        """
        Inserts entities in a specified collection.

        :param collection_name: The name of the collection to insert entities in.
        :type  collection_name: str.

        :param entities: The entities to insert.
        :type  entities: list

        :param partition_name: The name of the partition to insert entities in. The default value is
         None. The server stores entities in the “_default” partition by default.
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a MutationFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: list of ids of the inserted vectors.
        :rtype: list[int]

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.batch_insert(
                collection_name, entities, partition_name, timeout=timeout, **kwargs
            )

    def delete(self, collection_name, expr, partition_name=None, timeout=None, **kwargs):
        """
        Delete entities with an expression condition.
        And return results to show which primary key is deleted successfully

        :param collection_name: Name of the collection to delete entities from
        :type  collection_name: str

        :param expr: The expression to specify entities to be deleted
        :type  expr: str

        :param partition_name: Name of partitions that contain entities
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :return: list of ids of the deleted vectors.
        :rtype: list

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.delete(collection_name, expr, partition_name, timeout=timeout, **kwargs)

    def flush(self, collection_names=None, timeout=None, **kwargs):
        """
        Internally, Milvus organizes data into segments, and indexes are built in a per-segment manner.
        By default, a segment will be sealed if it grows large enough (according to segment size configuration).
        If any index is specified on certain field, the index-creating task will be triggered automatically
        when a segment is sealed.

        The flush() call will seal all the growing segments immediately of the given collection,
        and force trigger the index-creating tasks.

        :param collection_names: The name of collection to flush.
        :type  collection_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a FlushFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: None
        :rtype: NoneType

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.flush(collection_names, timeout=timeout, **kwargs)

    def search(
        self,
        collection_name,
        data,
        anns_field,
        param,
        limit,
        expression=None,
        partition_names=None,
        output_fields=None,
        timeout=None,
        round_decimal=-1,
        **kwargs,
    ):
        """
        Searches a collection based on the given expression and returns query results.

        :param collection_name: The name of the collection to search.
        :type  collection_name: str
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
        :param round_decimal: The specified number of decimal places of returned distance
        :type  round_decimal: int
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a SearchFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.
            * *consistency_level* (``str/int``) --
              Which consistency level to use when searching in the collection. For details, see
              https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md.
              Note: this parameter will overwrite the same parameter user specified when creating the collection,
              if no consistency level was specified, search will use the collection consistency level.
            * *guarantee_timestamp* (``int``) --
              This function instructs Milvus to see all operations performed before a provided timestamp. If no
              such timestamp is provided, then Milvus will search all operations performed to date.
              Note: only used in Customized consistency level.
            * *graceful_time* (``int``) --
              Only used in bounded consistency level. If graceful_time is set, PyMilvus will use current timestamp minus
              the graceful_time as the `guarantee_timestamp`. This option is 5s by default if not set.

        :return: Query result. QueryResult is iterable and is a 2d-array-like class, the first dimension is
                 the number of vectors to query (nq), the second dimension is the number of limit(topk).
        :rtype: QueryResult

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.search(
                collection_name,
                data,
                anns_field,
                param,
                limit,
                expression,
                partition_names,
                output_fields,
                round_decimal=round_decimal,
                timeout=timeout,
                **kwargs,
            )

    def get_query_segment_info(self, collection_name, timeout=None, **kwargs):
        """
        Notifies Proxy to return segments information from query nodes.

        :param collection_name: The name of the collection to get segments info.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: QuerySegmentInfo:
            QuerySegmentInfo is the growing segments's information in query cluster.
        :rtype: QuerySegmentInfo
        """
        with self._connection() as handler:
            return handler.get_query_segment_info(collection_name, timeout=timeout, **kwargs)

    def load_collection_progress(self, collection_name, timeout=None):
        """{
            'loading_progress': '100%',
            'num_loaded_partitions': 3,
            'not_loaded_partitions': [],
        }
        """
        with self._connection() as handler:
            return handler.load_collection_progress(collection_name, timeout=timeout)

    def load_partitions_progress(self, collection_name, partition_names, timeout=None):
        """{
            'loading_progress': '100%',
            'num_loaded_partitions': 3,
            'not_loaded_partitions': [],
        }
        """
        with self._connection() as handler:
            return handler.load_partitions_progress(
                collection_name, partition_names, timeout=timeout
            )

    def wait_for_loading_collection_complete(self, collection_name, timeout=None):
        with self._connection() as handler:
            return handler.wait_for_loading_collection(collection_name, timeout=timeout)

    def wait_for_loading_partitions_complete(self, collection_name, partition_names, timeout=None):
        with self._connection() as handler:
            return handler.wait_for_loading_partitions(
                collection_name, partition_names, timeout=timeout
            )

    def get_index_build_progress(self, collection_name, index_name, timeout=None):
        with self._connection() as handler:
            return handler.get_index_build_progress(collection_name, index_name, timeout=timeout)

    def wait_for_creating_index(self, collection_name, index_name, timeout=None):
        with self._connection() as handler:
            return handler.wait_for_creating_index(collection_name, index_name, timeout=timeout)

    def dummy(self, request_type, timeout=None):
        with self._connection() as handler:
            return handler.dummy(request_type, timeout=timeout)

    def query(
        self,
        collection_name,
        expr,
        output_fields=None,
        partition_names=None,
        timeout=None,
        **kwargs,
    ):
        """
        Query with a set of criteria, and results in a list of records that match the query exactly.

        :param collection_name: Name of the collection to retrieve entities from
        :type  collection_name: str

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

        :param kwargs:
            * *consistency_level* (``str/int``) --
              Which consistency level to use during a query on the collection. For details, see
              https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md.
              Note: this parameter will overwrite the same parameter user specified when creating the collection,
              if no consistency level was specified, query will use the collection consistency level.
            * *guarantee_timestamp* (``int``) --
              This function instructs Milvus to see all operations performed before a provided timestamp. If no
              such timestamp is specified, Milvus queries all operations performed to date.
              Note: only used in Customized consistency level.
            * *graceful_time* (``int``) --
              Only used in bounded consistency level. If graceful_time is set, PyMilvus will use current timestamp minus
              the graceful_time as the `guarantee_timestamp`. This option is 5s by default if not set.

        :raises RpcError: If gRPC encounter an error
        :raises ParamError: If parameters are invalid
        :raises MilvusException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.query(
                collection_name, expr, output_fields, partition_names, timeout=timeout, **kwargs
            )

    def load_balance(
        self,
        collection_name: str,
        src_node_id,
        dst_node_ids,
        sealed_segment_ids,
        timeout=None,
        **kwargs,
    ):
        """
        Do load balancing operation from source query node to destination query node.
        :param collection_name: The collection to balance.
        :type  collection_name: str

        :param src_node_id: The source query node id to balance.
        :type  src_node_id: int

        :param dst_node_ids: The destination query node ids to balance.
        :type  dst_node_ids: list[int]

        :param sealed_segment_ids: Sealed segment ids to balance.
        :type  sealed_segment_ids: list[int]

        :param timeout: The timeout for this method, unit: second
        :type  timeout: int

        :raises MilvusException: If query nodes not exist.
        :raises MilvusException: If sealed segments not exist.
        """
        with self._connection() as handler:
            return handler.load_balance(
                collection_name,
                src_node_id,
                dst_node_ids,
                sealed_segment_ids,
                timeout=timeout,
                **kwargs,
            )

    def compact(self, collection_name, is_clustering=False, timeout=None, **kwargs) -> int:
        """
        Do compaction for the collection.

        :param collection_name: The collection name to compact
        :type  collection_name: str

        :param is_clustering: trigger clustering compaction
        :type  is_clustering: bool

        :param timeout: The timeout for this method, unit: second
        :type  timeout: int

        :return: the compaction ID
        :rtype: int

        :raises MilvusException: If collection name not exist.
        """
        with self._connection() as handler:
            return handler.compact(
                collection_name, is_clustering=is_clustering, timeout=timeout, **kwargs
            )

    def get_compaction_state(
        self, compaction_id: int, is_clustering=False, timeout=None, **kwargs
    ) -> CompactionState:
        """
        Get compaction states of a targeted compaction id

        :param compaction_id: the id returned by compact
        :type  compaction_id: int

        :param is_clustering: get clustering compaction
        :type  is_clustering: bool

        :param timeout: The timeout for this method, unit: second
        :type  timeout: int

        :return: the state of the compaction
        :rtype: CompactionState

        :raises MilvusException: If compaction_id doesn't exist.
        """

        with self._connection() as handler:
            return handler.get_compaction_state(
                compaction_id, is_clustering=is_clustering, timeout=timeout, **kwargs
            )

    def wait_for_compaction_completed(
        self, compaction_id: int, timeout=None, **kwargs
    ) -> CompactionState:
        with self._connection() as handler:
            return handler.wait_for_compaction_completed(compaction_id, timeout=timeout, **kwargs)

    def get_compaction_plans(self, compaction_id: int, timeout=None, **kwargs) -> CompactionPlans:
        """
        Get compaction states of a targeted compaction id

        :param compaction_id: the id returned by compact
        :type  compaction_id: int

        :param timeout: The timeout for this method, unit: second
        :type  timeout: int

        :return: the state of the compaction
        :rtype: CompactionState

        :raises MilvusException: If compaction_id doesn't exist.
        """
        with self._connection() as handler:
            return handler.get_compaction_plans(compaction_id, timeout=timeout, **kwargs)

    def get_replicas(self, collection_name: str, timeout=None, **kwargs) -> Replica:
        """Get replica infos of a collection

        :param collection_name: the name of the collection
        :type  collection_name: str

        :param timeout: The timeout for this method, unit: second
        :type  timeout: int

        :return: the replica info
        :rtype: Replica

        :raises MilvusException: If collection_name doesn't exist.
        """
        with self._connection() as handler:
            return handler.get_replicas(collection_name, timeout=timeout, **kwargs)

    def do_bulk_insert(
        self, collection_name: str, partition_name: str, files: list, timeout=None, **kwargs
    ) -> int:
        """do_bulk_insert inserts entities through files, currently supports row-based json file.
        User need to create the json file with a specified json format which is described in the official user guide.
        Let's say a collection has two fields: "id" and "vec"(dimension=8), the row-based json format is:
          {"rows": [
              {"id": "0", "vec": [0.190, 0.046, 0.143, 0.972, 0.592, 0.238, 0.266, 0.995]},
              {"id": "1", "vec": [0.149, 0.586, 0.012, 0.673, 0.588, 0.917, 0.949, 0.944]},
              ......
            ]
          }
        The json file must be uploaded to root path of MinIO/S3 storage which is accessed by milvus server.
        For example:
            the milvus.yml specify the MinIO/S3 storage bucketName as "a-bucket", user can upload his json file
             to a-bucket/xxx.json, then call do_bulk_insert(files=["a-bucket/xxx.json"])

        :param collection_name: the name of the collection
        :type  collection_name: str

        :param partition_name: the name of the partition
        :type  partition_name: str

        :param files: related path of the file to be imported. for row-based json file, only allow
                      one file each invocation.
        :type  files: list[str]

        :param timeout: The timeout for this method, unit: second
        :type  timeout: int

        :param kwargs: other infos

        :return: id of the task
        :rtype:  int

        :raises BaseException: If collection_name doesn't exist.
        :raises BaseException: If the files input is illegal.
        """
        with self._connection() as handler:
            return handler.do_bulk_insert(
                collection_name, partition_name, files, timeout=timeout, **kwargs
            )

    def get_bulk_insert_state(self, task_id, timeout=None, **kwargs) -> BulkInsertState:
        """get_bulk_insert_state returns state of a certain task_id

        :param task_id: the task id returned by bulk_insert
        :type  task_id: int

        :return: BulkInsertState
        :rtype:  BulkInsertState
        """
        with self._connection() as handler:
            return handler.get_bulk_insert_state(task_id, timeout=timeout, **kwargs)

    def list_bulk_insert_tasks(self, timeout=None, **kwargs) -> list:
        """list_bulk_insert_tasks lists all bulk load tasks

        :param limit: maximum number of tasks returned, list all tasks if the value is 0, else return the latest tasks
        :type  limit: int

        :param collection_name: target collection name, list all tasks if the name is empty
        :type  collection_name: str

        :return: list[BulkInsertState]
        :rtype:  list[BulkInsertState]

        """
        with self._connection() as handler:
            return handler.list_bulk_insert_tasks(timeout=timeout, **kwargs)

    def create_user(self, user, password, timeout=None, **kwargs):
        """Create a user using the given user and password.
        :param user: the user name.
        :type  user: str
        :param password: the password.
        :type  password: str
        :param timeout: The timeout for this method, unit: second
        :type  timeout: int
        """
        with self._connection() as handler:
            handler.create_user(user, password, timeout=timeout, **kwargs)

    def update_password(self, user, old_password, new_password, timeout=None, **kwargs):
        """
            Update the user password using the given user and password.
            You must provide the original password to check if the operation is valid.
            Note: after this operation, PyMilvus won't change the related header of this connection.
            So if you update credential for this connection, the connection may be invalid.

        :param user: the user name.
        :type  user: str
        :param old_password: the original password.
        :type  old_password: str
        :param new_password: the newly password of this user.
        :type  new_password: str
        """
        with self._connection() as handler:
            handler.update_password(user, old_password, new_password, timeout=timeout, **kwargs)

    def delete_user(self, user, timeout=None, **kwargs):
        """Delete user corresponding to the username.
        :param user: the user name.
        :type  user: str
        :param timeout: The timeout for this method, unit: second
        :type  timeout: int
        """
        with self._connection() as handler:
            handler.delete_user(user, timeout=timeout, **kwargs)

    def list_usernames(self, timeout=None, **kwargs):
        """List all usernames.
        :param timeout: The timeout for this method, unit: second
        :type  timeout: int
        :return list of str:
            The usernames in Milvus instances.
        """
        with self._connection() as handler:
            return handler.list_usernames(timeout=timeout, **kwargs)

    def create_role(self, role_name, timeout=None, **kwargs):
        """Create Role
        :param role_name: the role name.
        :type  role_name: str
        """
        with self._connection() as handler:
            handler.create_role(role_name, timeout=timeout, **kwargs)

    def drop_role(self, role_name, timeout=None, **kwargs):
        """Drop Role
        :param role_name: role name.
        :type  role_name: str
        """
        with self._connection() as handler:
            handler.drop_role(role_name, timeout=timeout, **kwargs)

    def add_user_to_role(self, username, role_name, timeout=None, **kwargs):
        """Add User To Role
        :param username: user name.
        :type  username: str
        :param role_name: role name.
        :type  role_name: str
        """
        with self._connection() as handler:
            handler.add_user_to_role(username, role_name, timeout=timeout, **kwargs)

    def remove_user_from_role(self, username, role_name, timeout=None, **kwargs):
        """Remove User From Role
        :param username: user name.
        :type  username: str
        :param role_name: role name.
        :type  role_name: str
        """
        with self._connection() as handler:
            handler.remove_user_from_role(username, role_name, timeout=timeout, **kwargs)

    def select_one_role(self, role_name, include_user_info, timeout=None, **kwargs):
        """Select One Role Info
        :param role_name: role name.
        :type  role_name: str
        :param include_user_info: whether to obtain the user information associated with the role
        :type  include_user_info: bool
        """
        with self._connection() as handler:
            handler.select_one_role(role_name, include_user_info, timeout=timeout, **kwargs)

    def select_all_role(self, include_user_info, timeout=None, **kwargs):
        """Select All Role Info
        :param include_user_info: whether to obtain the user information associated with roles
        :type  include_user_info: bool
        """
        with self._connection() as handler:
            handler.select_all_role(include_user_info, timeout=timeout, **kwargs)

    def select_one_user(self, username, include_role_info, timeout=None, **kwargs):
        """Select One User Info
        :param username: user name.
        :type  username: str
        :param include_role_info: whether to obtain the role information associated with the user
        :type  include_role_info: bool
        """
        with self._connection() as handler:
            handler.select_one_user(username, include_role_info, timeout=timeout, **kwargs)

    def select_all_user(self, include_role_info, timeout=None, **kwargs):
        """Select All User Info
        :param include_role_info: whether to obtain the role information associated with users
        :type  include_role_info: bool
        """
        with self._connection() as handler:
            handler.select_all_role(include_role_info, timeout=timeout, **kwargs)

    def grant_privilege(self, role_name, object, object_name, privilege, timeout=None, **kwargs):
        """Grant Privilege
        :param role_name: role name.
        :type  role_name: str
        :param object: object that will be granted the privilege.
        :type  object: str
        :param object_name: identifies a specific resource name.
        :type  object_name: str
        :param privilege: privilege name.
        :type  privilege: str
        """
        with self._connection() as handler:
            handler.grant_privilege(
                role_name, object, object_name, privilege, timeout=timeout, **kwargs
            )

    def revoke_privilege(self, role_name, object, object_name, privilege, timeout=None, **kwargs):
        """Revoke Privilege
        :param role_name: role name.
        :type  role_name: str
        :param object: object that will be granted the privilege.
        :type  object: str
        :param object_name: identifies a specific resource name.
        :type  object_name: str
        :param privilege: privilege name.
        :type  privilege: str
        """
        with self._connection() as handler:
            handler.revoke_privilege(
                role_name, object, object_name, privilege, timeout=timeout, **kwargs
            )

    def select_grant_for_one_role(self, role_name, timeout=None, **kwargs):
        """Select the grant info about the role
        :param role_name: role name.
        :type  role_name: str
        """
        with self._connection() as handler:
            handler.select_grant_for_one_role(role_name, timeout=timeout, **kwargs)

    def select_grant_for_role_and_object(
        self, role_name, object, object_name, timeout=None, **kwargs
    ):
        """Select the grant info about the role and specific object
        :param role_name: role name.
        :type  role_name: str
        :param object: object that will be selected the privilege info.
        :type  object: str
        :param object_name: identifies a specific resource name.
        :type  object_name: str
        """
        with self._connection() as handler:
            handler.select_grant_for_role_and_object(
                role_name, object, object_name, timeout=timeout, **kwargs
            )

    def get_version(self, timeout=None, **kwargs):
        with self._connection() as handler:
            handler.get_version(timeout=timeout, **kwargs)

    def create_resource_group(self, name, timeout=None, **kwargs):
        """create resource group with specific name

        :param name: resource group name
        :type name: str
        """
        with self._connection() as handler:
            handler.create_resource_group(name, timeout=timeout, **kwargs)

    def update_resource_groups(
        self, configs: Mapping[str, ResourceGroupConfig], timeout=None, **kwargs
    ):
        """update resource groups with specific configs

        :param configs: resource group configs
        :type name: Mapping
        """
        with self._connection() as handler:
            handler.update_resource_groups(configs=configs, timeout=timeout, **kwargs)

    def drop_resource_group(self, name, timeout=None, **kwargs):
        """drop resource group with specific name

        :param name: resource group name
        :type name: str
        """
        with self._connection() as handler:
            handler.drop_resource_group(name, timeout=timeout, **kwargs)

    def list_resource_groups(self, timeout=None, **kwargs):
        """list all resource group names"""
        with self._connection() as handler:
            handler.list_resource_groups(timeout=timeout, **kwargs)

    def describe_resource_group(self, name, timeout=None, **kwargs) -> ResourceGroupInfo:
        """describe resource group with specific name

        :param name: resource group info
        :type name: str
        :return: resource group info
        :rtype: ResourceGroupInfo
        """
        with self._connection() as handler:
            handler.describe_resource_group(name, timeout=timeout, **kwargs)

    def transfer_node(self, source, target, num_node, timeout=None, **kwargs):
        """transfer num_node from source resource group to target resource_group

        :param source: source resource group name
        :type source: str
        :param target: target resource group name
        :type target: str
        :param num_node: transfer node num
        :type num_node: int
        """
        with self._connection() as handler:
            handler.transfer_node(source, target, num_node, timeout=timeout, **kwargs)

    def transfer_replica(
        self, source, target, collection_name, num_replica, timeout=None, **kwargs
    ):
        """transfer num_replica from source resource group to target resource group

        :param source: source resource group name
        :type source: str
        :param target: target resource group name
        :type target: str
        :param collection_name: collection name which replica belong to
        :type collection_name: str
        :param num_replica: transfer replica num
        :type num_replica: int
        """
        with self._connection() as handler:
            handler.transfer_replica(
                source, target, collection_name, num_replica, timeout=timeout, **kwargs
            )
