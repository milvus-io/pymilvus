from urllib.parse import urlparse
import sys
import logging
import collections
import uuid
import os

import grpc
from grpc._cython import cygrpc

from ..grpc_gen import milvus_pb2_grpc
from ..grpc_gen import milvus_pb2 as grpc_types
from .abstract import (
    ConnectIntf,
    TableSchema,
    IndexParam,
    PartitionParam
)
from .prepare import Prepare
from .types import IndexType, MetricType, Status
from .utils import (
    int_or_str,
    is_legal_host,
    is_legal_port,
)

from .hooks import BaseaSearchHook
from .client_hooks import (
    SearchHook
)

from .exceptions import ParamError, NotConnectError
from ..settings import DefaultConfig as config
from . import __version__

LOGGER = logging.getLogger(__name__)


class _ClientCallDetails(
    collections.namedtuple(
        '_ClientCallDetails',
        ('method', 'timeout', 'metadata', 'credentials', 'wait_for_ready')),
    grpc.ClientCallDetails):
    pass


class RequestIDClientInterceptor(grpc.UnaryUnaryClientInterceptor):

    def __generate_request_id(self):
        return str(uuid.uuid1()) + "-" + str(os.getpid())
        # return uuid.uuid1() + sys.os.

    def intercept_unary_unary(self, continuation, client_call_details, request):
        rid = self.__generate_request_id()
        LOGGER.info("Sending RPC request, Method: {}, Request ID: {}.".format(client_call_details.method, rid))

        # Add request into client call details, aka, metadata.
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        metadata.append(("request_id", rid))

        client_call_details = _ClientCallDetails(
            client_call_details.method, client_call_details.timeout, metadata,
            client_call_details.credentials, client_call_details.wait_for_ready)
        return continuation(client_call_details, request)


class GrpcMilvus(ConnectIntf):

    def __init__(self):
        self._channel = None
        self._stub = None
        self._uri = None
        self.status = None

        # hook
        self._search_hook = SearchHook()
        self._search_file_hook = SearchHook()

    def __str__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items() if not key.startswith('_')]
        return '<Milvus: {}>'.format(', '.join(attr_list))

    # def __del__(self):
    #     if self.connected():
    #         self.disconnect()

    #     if self._channel:
    #         del self._channel

    #     if self._stub:
    #         del self._stub

    # def __del__(self):

    def _set_uri(self, host=None, port=None, uri=None):
        """
        Set server network address

        """
        if host is not None:
            _port = port if port is not None else config.GRPC_PORT
            _host = host
        elif port is None:
            try:
                config_uri = urlparse(config.GRPC_URI)
                _uri = urlparse(uri) if uri else config_uri

                if _uri.scheme != 'tcp':
                    raise ParamError(
                        'Invalid parameter uri: `{}`. Scheme `{}` '
                        'is not supported'.format(_uri, _uri.scheme))

                _host = _uri.hostname
                _port = _uri.port
            except Exception:
                raise ParamError("`{}` is illegal".format(uri))
        else:
            raise ParamError("Param is not complete. Please invoke as follow:\n"
                             "\t(host = ${HOST}, port = ${PORT})\n"
                             "\t(uri = ${URI})\n")

        if not is_legal_host(_host) or not is_legal_port(_port):
            raise ParamError("host or port is illegal")

        self._uri = "{}:{}".format(str(_host), str(_port))

    def _set_channel(self, host=None, port=None, uri=None):
        """
        set grpc channel
        """
        self._set_uri(host, port, uri)

        # set transport unlimited
        self._channel = grpc.insecure_channel(
            self._uri,
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)]
        )

    def _set_hook(self, **kwargs):
        _search_hook = kwargs.get('search', None)
        if _search_hook:
            if not isinstance(_search_hook, BaseaSearchHook):
                raise ParamError("search hook must be a subclass of `BaseSearchHook`")

            self._search_hook = _search_hook

        _search_file_hook = kwargs.get('search_in_file', None)
        if _search_file_hook:
            if not isinstance(_search_file_hook, BaseaSearchHook):
                raise ParamError("search hook must be a subclass of `BaseSearchHook`")

            self._search_file_hook = _search_file_hook

    @property
    def server_address(self):
        """
        Server network address
        """
        return self._uri

    def connect(self, host=None, port=None, uri=None, timeout=3):
        """
        Connect method should be called before any operations.
        Server will be connected after connect return OK

        :type  host: str
        :type  port: str
        :type  uri: str
        :type  timeout: float
        :param host: (Optional) host of the server, default host is 127.0.0.1
        :param port: (Optional) port of the server, default port is 19530
        :param uri: (Optional) only support tcp proto now, default uri is

                `tcp://127.0.0.1:19530`

        :param timeout: (Optional) connection timeout, default timeout is 3000ms

        :return: Status, indicate if connect is successful
        :rtype: Status
        """
        if not self._channel:
            self._set_channel(host, port, uri)

        elif self.connected():
            return Status(message="You have already connected!", code=Status.CONNECT_FAILED)

        try:
            # check if server is ready
            grpc.channel_ready_future(self._channel).result(timeout=timeout)
            _intercept_channel = grpc.intercept_channel(self._channel, RequestIDClientInterceptor())
            self._channel = _intercept_channel
        except grpc.FutureTimeoutError:
            del self._channel
            self._channel = None
            raise NotConnectError('Fail connecting to server on {}. Timeout'.format(self._uri))
        except grpc.RpcError as e:
            del self._channel
            self._channel = None
            raise NotConnectError("Connect error: <{}>".format(e))
        # Unexpected error
        except Exception as e:
            raise NotConnectError("Error occurred when trying to connect server:\n"
                                  "\t<{}>".format(str(e)))

        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._channel)
        self.status = Status()
        return self.status

    def connected(self):
        """
        Check if client is connected to the server

        :return: if client is connected
        :rtype: bool
        """
        if not self._stub or not self._channel:
            return False
        try:
            grpc.channel_ready_future(self._channel).result(timeout=2)
            return True
        except (grpc.FutureTimeoutError, grpc.RpcError):
            return False

    def disconnect(self):
        """
        Disconnect with the server and distroy the channel

        :return: Status, indicate if disconnect is successful
        :rtype: Status
        """
        # After closeing, a exception stack trace is printed from a background thread and
        # no exception is thrown in the main thread, issue is under test and not done yet
        # checkout https://github.com/grpc/grpc/issues/18995
        # Also checkout Properly Specify Channel.close Behavior in Python:
        # https://github.com/grpc/grpc/issues/19235
        if not self.connected():
            raise NotConnectError('Please connect to the server first!')

        # closing channel by calling interface close() will result in grpc interval error
        del self._channel

        # try:
        #     self._channel.close()
        # except Exception as e:
        #     LOGGER.error(e)
        #     return Status(code=Status.CONNECT_FAILED, message='Disconnection failed')

        self.status = None
        self._channel = None
        self._stub = None

        return Status(message='Disconnect successfully')

    def client_version(self):
        """
        Provide client version

        :return:
            version: Client version

        :rtype: (str)
        """
        return __version__

    def server_version(self, timeout=10):
        """
        Provide server version

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        return self._cmd(cmd='version', timeout=timeout)

    def server_status(self, timeout=10):
        """
        Provide server status

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        return self._cmd(cmd='OK', timeout=timeout)

    def _cmd(self, cmd, timeout=10):
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        cmd = Prepare.cmd(cmd)
        try:
            response = self._stub.Cmd.future(cmd).result(timeout=timeout)
            if response.status.error_code == 0:
                return Status(message='Success!'), response.string_reply

            return Status(code=response.status.error_code, message=response.status.reason), None
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details())), None

    def create_table(self, param, timeout=10):
        """
        Create table

        :type  param: dict or TableSchema
        :param param: Provide table information to be created

                `example param={'table_name': 'name',
                                'dimension': 16,
                                'index_file_size': 1024 (default)，
                                'metric_type': Metric_type.L2 (default)
                                }`

                `OR using Prepare.table_schema to create param`

        :param timeout: timeout, The unit is seconds
        :type  timeout: double

        :return: Status, indicate if operation is successful
        :rtype: Status
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_schema = Prepare.table_schema(param)

        try:
            status = self._stub.CreateTable.future(table_schema).result(timeout=timeout)
            if status.error_code == 0:
                return Status(message='Create table successfully!')

            LOGGER.error(status)
            return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred: {}'.format(e.details()))

    def has_table(self, table_name, timeout=10):
        """

        This method is used to test table existence.

        :param table_name: table name is going to be tested.
        :type  table_name: str
        :param timeout: time waiting for server response
        :type  timeout: int

        :return:
            Status: indicate if vectors inserted successfully
            bool if given table_name exists

        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)

        try:
            reply = self._stub.HasTable.future(table_name).result(timeout=timeout)
            if reply.status.error_code == 0:
                return Status(), reply.bool_reply

            return Status(code=reply.status.error_code,
                          message=reply.status.reason), False
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(code=Status.UNEXPECTED_ERROR, message="request timeout"), False
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(code=e.code(), message=e.details()), False

    def describe_table(self, table_name, timeout=10):
        """
        Show table information

        :type  table_name: str
        :param table_name: which table to be shown

        :returns: (Status, table_schema)
            Status: indicate if query is successful
            table_schema: return when operation is successful
        :rtype: (Status, TableSchema)
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)

        try:
            response = self._stub.DescribeTable.future(table_name).result(timeout=timeout)

            if response.status.error_code == 0:
                table = TableSchema(
                    table_name=response.table_name,
                    dimension=response.dimension,
                    index_file_size=response.index_file_size,
                    metric_type=MetricType(response.metric_type)
                )

                return Status(message='Describe table successfully!'), table

            LOGGER.error(response.status)
            return Status(code=response.status.error_code, message=response.status.reason), None

        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details())), None

    def count_table(self, table_name, timeout=30):
        """
        obtain vector number in table

        :type  table_name: str
        :param table_name: target table name.

        :returns:
            Status: indicate if operation is successful

            res: int, table row count
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)

        try:
            trc = self._stub.CountTable.future(table_name).result(timeout=timeout)
            if trc.status.error_code == 0:
                return Status(message='Success!'), trc.table_row_count

            return Status(code=trc.status.error_code, message=trc.status.reason), None
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details())), None

    def show_tables(self, timeout=10):
        """
        Show all tables information in database

        :return:
            Status: indicate if this operation is successful

            tables: list of table names, return when operation
                    is successful
        :rtype:
            (Status, list[str])
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        cmd = Prepare.cmd('show_tables')
        try:
            response = self._stub.ShowTables.future(cmd).result(timeout=timeout)
            if response.status.error_code == 0:
                return Status(message='Show tables successfully!'), \
                       [name for name in response.table_names if len(name) > 0]
            return Status(response.status.error_code, message=response.status.reason), []
        except grpc.FutureTimeoutError:
            return Status(Status.UNEXPECTED_ERROR, message="Request timeout"), []
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details())), []

    def preload_table(self, table_name, timeout=300):
        """
        Load table to cache in advance

        :type table_name: str
        :param table_name: table to preload

        :returns:
            Status:  indicate if invoke is successful
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)

        try:
            status = self._stub.PreloadTable.future(table_name).result(timeout=timeout)
            return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except grpc.RpcError as e:
            return Status(code=e.code(), message='Error occurred. {}'.format(e.details()))

    def drop_table(self, table_name, timeout=20):
        """
        Delete table with table_name

        :type  table_name: str
        :param table_name: Name of the table being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)

        try:
            status = self._stub.DropTable.future(table_name).result(timeout=timeout)
            if status.error_code == 0:
                return Status(message='Delete table successfully!')
            return Status(code=status.error_code, message=status.reason)

        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred: {}'.format(e.details()))

    def insert(self, table_name, records, ids=None, partition_tag=None, timeout=-1, **kwargs):
        """
        Add vectors to table

        :param ids: list of id
        :type  ids: list[int]

        :type  table_name: str
        :param table_name: table name been inserted

        :type  records: list[list[float]]

                `example records: [[1.2345],[1.2345]]`

                `OR using Prepare.records`

        :param records: list of vectors been inserted

        :type  timeout: int
        :param timeout: time waiting for server response

        :returns:
            Status: indicate if vectors inserted successfully
            ids: list of id, after inserted every vector is given a id
        :rtype: (Status, list(int))
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        insert_param = kwargs.get('insert_param', None)

        if not insert_param:
            insert_param = Prepare.insert_param(table_name, records, partition_tag, ids)
        else:
            if not isinstance(insert_param, grpc_types.InsertParam):
                raise ParamError("The value of key 'insert_param' is invalid")

        try:
            if timeout == -1:
                vector_ids = self._stub.Insert(insert_param)
            else:
                vector_ids = self._stub.Insert.future(insert_param).result(timeout=timeout)

            if vector_ids.status.error_code == 0:
                ids = list(vector_ids.vector_id_array)
                return Status(message='Add vectors successfully!'), ids

            return Status(code=vector_ids.status.error_code, message=vector_ids.status.reason), []
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details())), []
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(code=Status.UNEXPECTED_ERROR, message="Request timeout"), []

    def create_index(self, table_name, index=None, timeout=-1):
        """
        build vectors of specific table and create vector index

        :param table_name: table used to crete index.
        :type table_name: str
        :param index: index params
        :type index: dict

            index_param can be None

            `example (default) param={'index_type': IndexType.FLAT,
                            'nlist': 16384}`

        :param timeout: grpc request timeout.

            if `timeout` = -1, method invoke a synchronous call, waiting util grpc response
            else method invoke a asynchronous call, timeout work here

        :type  timeout: int

        :return: Status, indicate if operation is successful
        """

        index_default = {
            'index_type': IndexType.FLAT,
            'nlist': 16384
        }
        if not index:
            _index = index_default
        elif not isinstance(index, dict):
            raise ParamError("param `index` should be a dictionary")
        else:
            _index = index
            if index.get('index_type', None) is None:
                _index.update({'index_type': IndexType.FLAT})
            if index.get('nlist', None) is None:
                _index.update({'nlist': 16384})
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        index_param = Prepare.index_param(table_name, _index)
        try:
            if timeout == -1:
                status = self._stub.CreateIndex(index_param)
            elif timeout < 0:
                raise ParamError("Param `timeout` should be a positive number or -1")
            else:
                try:
                    status = self._stub.CreateIndex.future(index_param).result(timeout=timeout)
                except grpc.FutureTimeoutError as e:
                    LOGGER.error(e)
                    return Status(Status.UNEXPECTED_ERROR, message='Request timeout')

            if status.error_code == 0:
                return Status(message='Build index successfully!')

            return Status(code=status.error_code, message=status.reason)
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details()))

    def describe_index(self, table_name, timeout=10):
        """
        Show index information of designated table

        :type table_name: str
        :param table_name: table name been queried

        :returns:
            Status:  indicate if query is successful
            IndexSchema:

        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)

        try:
            index_param = self._stub.DescribeIndex.future(table_name).result(timeout=timeout)

            status = index_param.status

            if status.error_code == 0:
                return Status(message="Successfully"), \
                       IndexParam(index_param.table_name,
                                  index_param.index.index_type,
                                  index_param.index.nlist)

            return Status(code=status.error_code, message=status.reason), None
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(code=Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details())), None

    def drop_index(self, table_name, timeout=10):
        """
        drop index from index file

        :param table_name: target table name.
        :type table_name: str

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)
        try:
            status = self._stub.DropIndex.future(table_name).result(timeout=timeout)
            return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details()))

    def create_partition(self, table_name, partition_name, partition_tag):
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        # TODO: prepare request
        request = Prepare.partition_param(table_name, partition_name, partition_tag)

        try:
            response = self._stub.CreatePartition(request)
            return Status(code=response.error_code, message=response.reason)
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details()))

    def show_partitions(self, table_name):
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        # TODO: prepare request
        request = Prepare.table_name(table_name)

        try:
            response = self._stub.ShowPartitions(request)
            status = response.status
            if status.error_code == 0:

                partition_list = []
                for partition in response.partition_array:
                    partition_param = PartitionParam(
                        partition.table_name,
                        partition.partition_name,
                        partition.tag
                    )
                    partition_list.append(partition_param
                                          )
                # TODO: return patririons list
                return Status(), partition_list

            return Status(), []
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(), []

    def drop_partition(self, table_name, partition_tag):
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        # TODO: prepare request
        request = Prepare.partition_param(table_name=table_name, partition_name=None, tag=partition_tag)

        try:
            response = self._stub.DropPartition(request)

            return Status(code=response.error_code, message=response.reason)
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(code=1, message="")

    def search(self, table_name, top_k, nprobe, query_records, query_ranges=None, partition_tags=None, **kwargs):
        """
        Search similar vectors in designated table

        :param table_name: target table name
        :type  table_name: str
        :param top_k: number of vertors which is most similar with query vectors
        :type  top_k: int
        :param nprobe: cell number of probe
        :type  nprobe: int
        :param query_records: vectors to query
        :type  query_records: list[list[float32]]
        :param query_ranges: query data range

        :return
            Status: indicate if search successfully
            result: query result

        :rtype: (Status, TopKQueryResult)

        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        request = Prepare.search_param(
            table_name, top_k, nprobe, query_records, query_ranges, partition_tags
        )

        try:
            self._search_hook.pre_search()
            response = self._stub.Search(request)
            self._search_hook.aft_search()

            if self._search_hook.on_response():
                return response

            if response.status.error_code != 0:
                return Status(code=response.status.error_code,
                              message=response.status.reason), []

            resutls = self._search_hook.handle_response(response)
            return Status(message='Search vectors successfully!'), resutls

        except grpc.RpcError as e:
            LOGGER.error(e)
            status = Status(code=e.code(), message='Error occurred: {}'.format(e.details()))
            return status, []

    def search_in_files(self, table_name, file_ids, query_records, top_k,
                        nprobe=16, query_ranges=None, **kwargs):
        """
        Query vectors in a table, in specified files

        :type  nprobe: int
        :param nprobe:

        :type  table_name: str
        :param table_name: table name been queried

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

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        file_ids = list(map(int_or_str, file_ids))

        infos = Prepare.search_vector_in_files_param(
            table_name, query_records, query_ranges, top_k, nprobe, file_ids
        )

        try:
            self._search_file_hook.pre_search()
            response = self._stub.SearchInFiles(infos)
            self._search_file_hook.aft_search()

            if self._search_file_hook.on_response():
                return response

            if response.status.error_code != 0:
                return Status(code=response.status.error_code,
                              message=response.status.reason), []

            return Status(message='Search vectors successfully!'), \
                   self._search_file_hook.handle_response(response)
        except grpc.RpcError as e:
            LOGGER.error(e)
            status = Status(code=e.code(), message='Error occurred. {}'.format(e.details()))
            return status, []

    def __delete_vectors_by_range(self, table_name, start_date=None, end_date=None, timeout=10):
        """
        Delete vectors by range. The data range contains start_time but not end_time
        This method is deprecated, not recommended for users

        :type  table_name: str
        :param table_name: str, date, datetime

        :type  start_date: str, date, datetime
        :param start_date:

        :type  end_date: str, date, datetime
        :param end_date:

        :return:
            Status:  indicate if invoke is successful
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        delete_range = Prepare.delete_param(table_name, start_date, end_date)

        try:
            status = self._stub.DeleteByRange.future(delete_range).result(timeout=timeout)
            return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='Error occurred. {}'.format(e.details()))

    # In old version of pymilvus, some methods are different from the new.
    # apply alternative method name for compatibility
    get_table_row_count = count_table
    delete_table = drop_table
    add_vectors = insert
    search_vectors = search
    search_vectors_in_files = search_in_files
