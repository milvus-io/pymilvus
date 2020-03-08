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
        return self._handler.connect(host, port, uri, timeout)

    def connected(self):
        return self._handler.connected()

    @check_connect
    def disconnect(self):
        return self._handler.disconnect()

    def client_version(self):
        """
                Provide client version

                :return:
                    version: Client version

                :rtype: (str)
                """
        return __version__

    @check_connect
    def server_status(self, timeout=10):
        return self._cmd("status", timeout)

    @check_connect
    def server_version(self, timeout=10):
        return self._cmd("version", timeout)

    @check_connect
    def _cmd(self, cmd, timeout=10):
        check_pass_param(cmd=cmd)

        return self._handler._cmd(cmd, timeout)

    @check_connect
    def create_collection(self, param, timeout=10):
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

        check_pass_param(collection_name=collection_name, dimension=dim, index_file_size=index_file_size, metric_type=metric_type)

        return self._handler.create_table(collection_name, dim, index_file_size, metric_type, collection_param, timeout)

    @check_connect
    def has_collection(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        return self._handler.has_table(collection_name, timeout)

    @check_connect
    def describe_collection(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        return self._handler.describe_table(collection_name, timeout)

    @check_connect
    def count_collection(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        return self._handler.count_table(collection_name, timeout)

    @check_connect
    def show_collections(self, timeout=10):
        return self._handler.show_tables(timeout)

    @check_connect
    def collection_info(self, collection_name, timeout=10):
        return self._handler.show_table_info(collection_name, timeout)

    @check_connect
    def preload_collection(self, collection_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        return self._handler.preload_table(collection_name, timeout)

    @check_connect
    def drop_collection(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        return self._handler.drop_table(collection_name, timeout)

    @check_connect
    def insert(self, collection_name, records, ids=None, partition_tag=None, params=None, timeout=-1):
        check_pass_param(collection_name=collection_name, records=records,
                         ids=ids, partition_tag=partition_tag)

        if ids is not None and len(records) != len(ids):
            raise ParamError("length of vectors do not match that of ids")

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.insert(collection_name, records, ids, partition_tag, params, timeout)

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
        _index_type = IndexType.FLAT if index_type is None else index_type
        check_pass_param(collection_name=collection_name, index_type=_index_type)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.create_index(collection_name, _index_type, params, timeout)

    @check_connect
    def describe_index(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        return self._handler.describe_index(collection_name, timeout)

    @check_connect
    def drop_index(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        return self._handler.drop_index(collection_name, timeout)

    @check_connect
    def create_partition(self, collection_name, partition_tag, timeout=10):
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)

        return self._handler.create_partition(collection_name, partition_tag, timeout)

    @check_connect
    def show_partitions(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        return self._handler.show_partitions(collection_name, timeout)

    @check_connect
    def drop_partition(self, collection_name, partition_tag, timeout=10):
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        return self._handler.drop_partition(collection_name, partition_tag, timeout)

    @check_connect
    def search(self, collection_name, top_k, query_records, partition_tags=None, params=None):
        check_pass_param(collection_name=collection_name, topk=top_k,
                         records=query_records, partition_tag_array=partition_tags)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.search(collection_name, top_k, query_records, partition_tags, params)

    @check_connect
    def search_in_files(self, collection_name, file_ids, query_records, top_k, params=None):
        check_pass_param(collection_name=collection_name, topk=top_k, records=query_records)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        return self._handler.search_in_files(collection_name, file_ids,
                                             query_records, top_k, params)

    @check_connect
    def delete_by_id(self, collection_name, id_array, timeout=None):
        check_pass_param(collection_name=collection_name, ids=id_array)

        return self._handler.delete_by_id(collection_name, id_array, timeout)

    @check_connect
    def flush(self, collection_name_array=None):
        if collection_name_array is None:
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
        check_pass_param(collection_name=collection_name)

        return self._handler.compact(collection_name, timeout)

    @check_connect
    def get_config(self, parent_key, child_key):
        cmd = "get_config {}.{}".format(parent_key, child_key)

        return self._cmd(cmd)

    @check_connect
    def set_config(self, parent_key, child_key, value):
        cmd = "set_config {}.{} {}".format(parent_key, child_key, value)

        return self._cmd(cmd)

    # In old version of pymilvus, some methods are different from the new.
    # apply alternative method name for compatibility

    # get_collection_row_count = count_collection
    # delete_collection = drop_collection
    add_vectors = insert
    search_vectors = search
    search_vectors_in_files = search_in_files
