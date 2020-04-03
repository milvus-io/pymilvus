import copy
import functools

from . import __version__
from .types import IndexType, MetricType
from .check import check_pass_param
from .grpc_handler import GrpcHandler
from .http_handler import HttpHandler
from .exceptions import ParamError, NotConnectError


def check_connect(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        return f(self, *args, **kwargs)

    return wrapper


class Milvus:
    def __init__(self, host=None, port=None, handler="GRPC", **kwargs):
        if handler == "GRPC":
            self._handler = GrpcHandler(host=host, port=port, **kwargs)
        elif handler == "HTTP":
            self._handler = HttpHandler(host=host, port=port, **kwargs)
        else:
            raise ParamError("Unknown handler options, please use \'GRPC\' or \'HTTP\'")

    def __enter__(self):
        self._handler.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handler.__exit__(exc_type, exc_val, exc_tb)

    def set_hook(self, **kwargs):
        return self._handler.set_hook(**kwargs)

    @property
    def status(self):
        return self._handler.status

    @property
    def handler(self):
        if isinstance(self._handler, GrpcHandler):
            return "GRPC"

        if isinstance(self._handler, HttpHandler):
            return "HTTP"

        return "NULL"

    def connect(self, host=None, port=None, uri=None, timeout=1):
        """
        Connects to the Milvus server.

        :type  host: str
        :type  port: str
        :type  uri: str
        :type  timeout: float
        :param host: (Optional) Ip address of the Milvus server. The default is `127.0.0.1`.
        :param port: (Optional) Port of the Milvus server. The default is `19530`.
        :param uri: (Optional) URI of the Milvus server. Only TCP is supported. The default is `tcp://127.0.0.1:19530`.

        :param timeout: (Optional) Connection timeout in milliseconds. The default is 3000.

        :return: Indicates whether the connection is successful.
        :rtype: Status
        
        :raises: NotConnectError
        """
        return self._handler.connect(host, port, uri, timeout)

    def connected(self):
        """
        Checks whether the client is connected to the Milvus server.

        :return: Whether the client is connected to the Milvus server.
        :rtype: bool
        """
        return self._handler.connected()

    @check_connect
    def disconnect(self):
        """
        Disconnects from the Milvus server.

        :return: Whether the client is disconnected from the Milvus server.
        :rtype: Status
        """
        return self._handler.disconnect()

    def client_version(self):
        """
        Returns the version of the client.

        :return: Version of the client.

        :rtype: (str)
        """
        return __version__

    @check_connect
    def server_status(self, timeout=10):
        """
        Returns the status of the Milvus server.

        :return:
            Status: Whether the operation is successful.

            str : Status of the Milvus server.

        :rtype: (Status, str)
        """
        return self._cmd("status", timeout)

    @check_connect
    def server_version(self, timeout=10):
        """
       Returns the version of the Milvus server.

       :return:
           Status: Whether the operation is successful.

           str : Version of the Milvus server.

       :rtype: (Status, str)
       """

        return self._cmd("version", timeout)

    @check_connect
    def _cmd(self, cmd, timeout=10):
        check_pass_param(cmd=cmd)

        return self._handler._cmd(cmd, timeout)

    @check_connect
    def create_collection(self, param, timeout=10):
        """
        Creates a collection.

        :type  param: dict
        :param param: Information needed to create a collection.

                `param={'collection_name': 'name',
                                'dimension': 16,
                                'index_file_size': 1024 (default)，
                                'metric_type': Metric_type.L2 (default)
                                }`

        :param timeout: Timeout in seconds.
        :type  timeout: double

        :return: Whether the operation is successful.
        :rtype: Status
        """
        if not isinstance(param, dict):
            raise ParamError('Param type incorrect, expect {} but get {} instead'
                             .format(type(dict), type(param)))

        collection_param = copy.deepcopy(param)

        if 'collection_name' not in collection_param:
            raise ParamError('collection_name is required')
        collection_name = collection_param["collection_name"]
        collection_param.pop('collection_name')

        if 'dimension' not in collection_param:
            raise ParamError('dimension is required')
        dim = collection_param["dimension"]
        collection_param.pop("dimension")

        index_file_size = collection_param.get('index_file_size', 1024)
        collection_param.pop('index_file_size', None)

        metric_type = collection_param.get('metric_type', MetricType.L2)
        collection_param.pop('metric_type', None)

        check_pass_param(collection_name=collection_name, dimension=dim, index_file_size=index_file_size,
                         metric_type=metric_type)

        return self._handler.create_table(collection_name, dim, index_file_size, metric_type, collection_param, timeout)

    @check_connect
    def has_collection(self, collection_name, timeout=10):
        """

        Checks whether a collection exists.

        :param collection_name: Name of the collection to check.
        :type  collection_name: str
        :param timeout: Timeout in seconds.
        :type  timeout: int

        :return:
            Status: indicate whether the operation is successful.
            bool if given collection_name exists

        """
        check_pass_param(collection_name=collection_name)
        return self._handler.has_table(collection_name, timeout)

    @check_connect
    def describe_collection(self, collection_name, timeout=10):
        """
        Returns information of a collection.

        :type  collection_name: str
        :param collection_name: Name of the collection to describe.

        :returns: (Status, table_schema)
            Status: indicate if query is successful
            table_schema: return when operation is successful
        :rtype: (Status, TableSchema)
        """
        check_pass_param(collection_name=collection_name)
        return self._handler.describe_table(collection_name, timeout)

    @check_connect
    def count_collection(self, collection_name, timeout=10):
        """
        Returns the number of vectors in a collection.

        :type  collection_name: str
        :param collection_name: target table name.

        :returns:
            Status: indicate if operation is successful

            res: int, table row count
        """
        check_pass_param(collection_name=collection_name)
        return self._handler.count_table(collection_name, timeout)

    @check_connect
    def show_collections(self, timeout=10):
        """
        Returns information of all collections.

        :return:
            Status: indicate if this operation is successful

            collections: list of table names, return when operation
                    is successful
        :rtype:
            (Status, list[str])
        """
        return self._handler.show_tables(timeout)

    @check_connect
    def collection_info(self, collection_name, timeout=10):
        return self._handler.show_table_info(collection_name, timeout)

    @check_connect
    def preload_collection(self, collection_name, timeout=None):
        """
        Loads a collection for caching.

        :type collection_name: str
        :param collection_name: table to preload

        :returns:
            Status:  indicate if invoke is successful
        """
        check_pass_param(collection_name=collection_name)
        return self._handler.preload_table(collection_name, timeout)

    @check_connect
    def drop_collection(self, collection_name, timeout=10):
        """
        Deletes a collection by name.

        :type  collection_name: str
        :param collection_name: Name of the table being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        return self._handler.drop_table(collection_name, timeout)

    @check_connect
    def insert(self, collection_name, records, ids=None, partition_tag=None, params=None, **kwargs):
        """
        Insert vectors to a collection.

        :param ids: list of id
        :type  ids: list[int]

        :type  collection_name: str
        :param collection_name: Name of the collection to insert vectors to.

        :type  records: list[list[float]]

                `example records: [[1.2345],[1.2345]]`

                `OR using Prepare.records`

        :param records: List of vectors to insert.

        :type partition_tag: str or None.
            If partition_tag is None, vectors will be inserted to the collection rather than partitions.

        :param partition_tag: Tag of a partition.

        :type  timeout: int
        :param timeout: Time to wait for server response before timeout.

        :returns:
            Status: Whether vectors are inserted successfully.
            ids: IDs of the inserted vectors.
        :rtype: (Status, list(int))
        """
        if kwargs.get("insert_param", None) is not None:
            return self._handler.insert(None, None, **kwargs)

        check_pass_param(collection_name=collection_name, records=records,
                         ids=ids, partition_tag=partition_tag)

        if ids is not None and len(records) != len(ids):
            raise ParamError("length of vectors do not match that of ids")

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.insert(collection_name, records, ids, partition_tag, params, None, **kwargs)

    @check_connect
    def get_vector_by_id(self, collection_name, vector_id, timeout=None):
        check_pass_param(collection_name=collection_name, ids=[vector_id])

        return self._handler.get_vector_by_id(collection_name, vector_id, timeout=timeout)

    @check_connect
    def get_vector_ids(self, collection_name, segment_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        check_pass_param(collection_name=segment_name)

        return self._handler.get_vector_ids(collection_name, segment_name, timeout)

    @check_connect
    def create_index(self, collection_name, index_type=None, params=None, timeout=None):
        """
        Creates index for a collection.

        :param collection_name: Collection used to create index.
        :type collection_name: str
        :param index: index params
        :type index: dict

            index_param can be None

            `example (default) param={'index_type': IndexType.FLAT,
                            'nlist': 16384}`

        :param timeout: grpc request timeout.

            if `timeout` = -1, method invoke a synchronous call, waiting util grpc response
            else method invoke a asynchronous call, timeout work here

        :type  timeout: int

        :return: Whether the operation is successful.
        """
        _index_type = IndexType.FLAT if index_type is None else index_type
        check_pass_param(collection_name=collection_name, index_type=_index_type)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.create_index(collection_name, _index_type, params, timeout)

    @check_connect
    def describe_index(self, collection_name, timeout=10):
        """
        Show index information of a collection.

        :type collection_name: str
        :param collection_name: table name been queried

        :returns:
            Status:  Whether the operation is successful.
            IndexSchema:

        """
        check_pass_param(collection_name=collection_name)
        return self._handler.describe_index(collection_name, timeout)

    @check_connect
    def drop_index(self, collection_name, timeout=10):
        """
        Removes an index.

        :param collection_name: target collection name.
        :type collection_name: str

        :return:
            Status: Whether the operation is successful.

        ：:rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        return self._handler.drop_index(collection_name, timeout)

    @check_connect
    def create_partition(self, collection_name, partition_tag, timeout=10):
        """
        create a partition for a collection. 

        :param collection_name: Name of the collection.
        :type  collection_name: str

        :param partition_name: Name of the partition.
        :type  partition_name: str

        :param partition_tag: Name of the partition tag.
        :type  partition_tag: str

        :param timeout: time waiting for response.
        :type  timeout: int

       :return:
            Status: Whether the operation is successful.

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)

        return self._handler.create_partition(collection_name, partition_tag, timeout)

    @check_connect
    def show_partitions(self, collection_name, timeout=10):
        """
        Show all partitions in a collection.

        :param collection_name: target table name.
        :type  collection_name: str

        :param timeout: time waiting for response.
        :type  timeout: int

        :return:
            Status: Whether the operation is successful.
            partition_list:

        """
        check_pass_param(collection_name=collection_name)
        return self._handler.show_partitions(collection_name, timeout)

    @check_connect
    def drop_partition(self, collection_name, partition_tag, timeout=10):
        """
        Deletes a partition in a collection.

        :param collection_name: Collection name.
        :type  collection_name: str

        :param partition_tag: Partition name.
        :type  partition_tag: str

        :param timeout: time waiting for response.
        :type  timeout: int

        :return:
            Status: Whether the operation is successful.

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        return self._handler.drop_partition(collection_name, partition_tag, timeout)

    @check_connect
    def search(self, collection_name, top_k, query_records, partition_tags=None, params=None):
        """
        Search vectors in a collection.

        :param collection_name: Name of the collection.
        :type  collection_name: str

        :param top_k: number of vertors which is most similar with query vectors
        :type  top_k: int

        :param nprobe: cell number of probe
        :type  nprobe: int

        :param query_records: vectors to query
        :type  query_records: list[list[float32]]

        :param partition_tags: tags to search
        :type  partition_tags: list

        :return
            Status: Whether the operation is successful.
            result: query result

        :rtype: (Status, TopKQueryResult)

        """
        check_pass_param(collection_name=collection_name, topk=top_k,
                         records=query_records, partition_tag_array=partition_tags)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.search(collection_name, top_k, query_records, partition_tags, params)

    @check_connect
    def search_in_files(self, collection_name, file_ids, query_records, top_k, params=None):
        """
        Searches for vectors in specific files of a collection.

        The Milvus server stores vector data into multiple files. Searching for vectors in specific files is a
        method used in Mishards. Obtain more detail about Mishards, see
        <a href="https://github.com/milvus-io/milvus/tree/master/shards">

        :type  collection_name: str
        :param collection_name: table name been queried

        :type  file_ids: list[str] or list[int]
        :param file_ids: Specified files id array

        :type  query_records: list[list[float]]
        :param query_records: all vectors going to be queried

        :param query_ranges: Optional ranges for conditional search.

            If not specified, search in the whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            results: query result

        :rtype: (Status, TopKQueryResult)
        """
        check_pass_param(collection_name=collection_name, topk=top_k, records=query_records)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.search_in_files(collection_name, file_ids,
                                             query_records, top_k, params)

    @check_connect
    def delete_by_id(self, collection_name, id_array, timeout=None):
        """
        Deletes vectors in a collection by vector ID.

        """
        check_pass_param(collection_name=collection_name, ids=id_array)

        return self._handler.delete_by_id(collection_name, id_array, timeout)

    @check_connect
    def flush(self, collection_name_array=None):
        """
        Flushes vector data in one collection or multiple collections to disk.

        :type  collection_name_array: list
        :param collection_name: Name of one or multiple collections to flush.

        """
        if collection_name_array in (None, []):
            return self._handler.flush([])

        if not isinstance(collection_name_array, list):
            raise ParamError("Collection name array must be type of list")

        if len(collection_name_array) <= 0:
            raise ParamError("Collection name array is not allowed to be empty")

        for name in collection_name_array:
            check_pass_param(collection_name=name)

        return self._handler.flush(collection_name_array)

    @check_connect
    def compact(self, collection_name, timeout=None):
        """
        Compacts segments in a collection. This function is recommended after deleting vectors.

        :type  collection_name: str
        :param collection_name: Name of the collections to compact.

        """
        check_pass_param(collection_name=collection_name)

        return self._handler.compact(collection_name, timeout)

    @check_connect
    def get_config(self, parent_key, child_key):
        """
        Gets Milvus configurations.

        """
        cmd = "get_config {}.{}".format(parent_key, child_key)

        return self._cmd(cmd)

    @check_connect
    def set_config(self, parent_key, child_key, value):
        """
        Sets Milvus configurations.

        """
        cmd = "set_config {}.{} {}".format(parent_key, child_key, value)

        return self._cmd(cmd)

    # In old version of pymilvus, some methods are different from the new.
    # apply alternative method name for compatibility

    # get_collection_row_count = count_collection
    # delete_collection = drop_collection
    add_vectors = insert
    search_vectors = search
    search_vectors_in_files = search_in_files
