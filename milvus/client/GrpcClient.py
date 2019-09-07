"""
This is a client for milvus of gRPC
"""
import time
import grpc
import logging

from .Abstract import (
    ConnectIntf,
    TableSchema,
    Range,
    QueryResult,
    TopKQueryResult,
    IndexParam
)

from .types import IndexType, MetricType, Status
from .utils import *
from .Exceptions import *
from ..settings import DefaultConfig as config

from ..grpc_gen import milvus_pb2_grpc, status_pb2
from ..grpc_gen import milvus_pb2 as grpc_types
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
from . import __version__


class Prepare(object):

    @classmethod
    def table_name(cls, table_name):

        check_pass_param(table_name=table_name)

        status = status_pb2.Status(error_code=0, reason='Client')
        return grpc_types.TableName(status=status, table_name=table_name)

    @classmethod
    def table_schema(cls, param):
        """
        :type param: dict
        :param param: (Required)

            `example param={'table_name': 'name',
                            'dimension': 16,
                            'index_type': IndexType.FLAT
                            }`

        :return: ttypes.TableSchema object
        """
        if isinstance(param, grpc_types.TableSchema):
            return param

        if not isinstance(param, dict):
            raise ParamError('Param type incorrect, expect {} but get {} instead '.format(
                type(dict), type(param)
            ))

        if 'index_file_size' not in param:
            param['index_file_size'] = 1024
        if 'metric_type' not in param:
            param['metric_type'] = MetricType.L2

        _param = {
            'table_name': param['table_name'],
            'dimension': param['dimension'],
            'index_file_size': param['index_file_size'],
            'metric_type': param['metric_type']
        }

        check_pass_param(**_param)

        table_name = Prepare.table_name(_param["table_name"])
        return grpc_types.TableSchema(table_name=table_name,
                                      dimension=_param["dimension"],
                                      index_file_size=_param["index_file_size"],
                                      metric_type=_param["metric_type"])

    @classmethod
    def range(cls, start_date, end_date):
        """
        Parser a 'yyyy-mm-dd' like str or date/datetime object to Range object

            `Range: (start_date, end_date]`

            `start_date : '2019-05-25'`

        :param start_date: start date
        :type  start_date: str, date, datetime
        :param end_date: end date
        :type  end_date: str, date, datetime

        :return: Range object
        """
        temp = Range(start_date, end_date)

        return grpc_types.Range(start_value=temp.start_date,
                                end_value=temp.end_date)

    @classmethod
    def ranges(cls, ranges):
        """
        prepare query_ranges

        :param ranges: prepare query_ranges
        :type  ranges: [[str, str], (str,str)], iterable

            `Example: [[start, end]], ((start, end), (start, end)), or
                    [(start, end)]`

        :return: list[Range]
        """
        res = []
        for _range in ranges:
            if not isinstance(_range, grpc_types.Range):
                res.append(Prepare.range(_range[0], _range[1]))
            else:
                res.append(_range)
        return res

    @classmethod
    def insert_param(cls, table_name, vectors, ids=None):

        check_pass_param(table_name=table_name)

        if ids is None:
            _param = grpc_types.InsertParam(table_name=table_name)
        else:
            check_pass_param(ids=ids)
            _param = grpc_types.InsertParam(table_name=table_name, row_id_array=ids)

        for vector in vectors:
            if is_legal_array(vector):
                _param.row_record_array.add(vector_data=vector)
            else:
                raise ParamError('Vectors should be 2-dim array!')

        return _param

    @classmethod
    def index(cls, index_type, nlist):
        """

        :type index_type: IndexType
        :param index_type: index type

        :type  nlist:
        :param nlist:

        :return:
        """
        check_pass_param(index_type=index_type, nlist=nlist)

        return grpc_types.Index(index_type=index_type, nlist=nlist)

    @classmethod
    def index_param(cls, table_name, index_param):

        if not isinstance(index_param, dict):
            raise ParamError('Param type incorrect, expect {} but get {} instead '.format(
                type(dict), type(index_param)
            ))

        check_pass_param(table_name=table_name, **index_param)

        _table_name = Prepare.table_name(table_name)
        _index = Prepare.index(**index_param)

        return grpc_types.IndexParam(table_name=_table_name, index=_index)

    @classmethod
    def vector_ids(cls, ids):
        _status = status_pb2.Status(error_code=0, reason="OK")

        if not is_legal_array(ids):
            raise ParamError('Ids must be a list')

        return grpc_types.VectorIds(status=_status, vector_id_array=ids)

    @classmethod
    def search_param(cls, table_name, query_records, query_ranges, topk, nprobe):
        query_ranges = Prepare.ranges(query_ranges) if query_ranges else None

        check_pass_param(table_name=table_name, topk=topk, nprobe=nprobe)

        search_param = grpc_types.SearchParam(
            table_name=table_name,
            query_range_array=query_ranges,
            topk=topk,
            nprobe=nprobe
        )

        for vector in query_records:
            if is_legal_array(vector):
                search_param.query_record_array.add(vector_data=vector)
            else:
                raise ParamError('Vectors should be 2-dim array!')

        return search_param

    @classmethod
    def search_vector_in_files_param(cls, table_name, query_records, query_ranges, topk, nprobe, ids):
        _search_param = Prepare.search_param(table_name, query_records, query_ranges, topk, nprobe)

        # check_pass_param(ids=ids)

        return grpc_types.SearchInFilesParam(
            file_id_array=ids,
            search_param=_search_param
        )

    @classmethod
    def cmd(cls, cmd):
        check_pass_param(cmd=cmd)

        return grpc_types.Command(cmd=cmd)

    @classmethod
    def delete_param(cls, table_name, start_date, end_date):

        range = Prepare.range(start_date, end_date)

        check_pass_param(table_name=table_name)

        return grpc_types.DeleteByRangeParam(range=range, table_name=table_name)


class GrpcMilvus(ConnectIntf):
    def __init__(self):
        self._channel = None
        self._stub = None
        self._uri = None
        self.server_address = None
        self.status = None

    def __str__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items() if not key.startswith('_')]
        return '<Milvus: {}>'.format(', '.join(L))

    def set_channel(self, host=None, port=None, uri=None):

        config_uri = urlparse(config.GRPC_URI)

        _uri = urlparse(uri) if uri else config_uri

        if not host:
            if _uri.scheme == 'tcp':
                host = _uri.hostname or 'localhost'
                port = _uri.port or '19530'
            else:
                if uri:
                    raise RuntimeError(
                        'Invalid parameter uri: {}'.format(uri)
                    )
                raise RuntimeError(
                    'Invalid configuration for GRPC_URI: {}'.format(
                        config.GRPC_URI)
                )
        else:
            host = host
            port = port or '19530'

        self._uri = str(host) + ':' + str(port)
        self.server_address = self._uri
        self._channel = grpc.insecure_channel(self._uri)

    def connect(self, host=None, port=None, uri=None, timeout=5):
        """
        Connect method should be called before any operations.
        Server will be connected after connect return OK

        :type  host: str
        :type  port: str
        :type  uri: str
        :type  timeout: int
        :param host: (Optional) host of the server, default host is 127.0.0.1
        :param port: (Optional) port of the server, default port is 19530
        :param uri: (Optional) only support tcp proto now, default uri is

                `tcp://127.0.0.1:19530`

        :param timeout: (Optional) connection timeout, default timeout is 3000ms

        :return: Status, indicate if connect is successful
        :rtype: Status
        """
        if self._channel is None:
            self.set_channel(host, port, uri)

        elif self.connected():
            return Status(message="You have already connected!", code=Status.CONNECT_FAILED)

        try:
            grpc.channel_ready_future(self._channel).result(timeout=timeout)
        except grpc.FutureTimeoutError as e:
            raise NotConnectError('Fail connecting to server on {}'.format(self._uri))
        else:
            self._stub = milvus_pb2_grpc.MilvusServiceStub(self._channel)
            self.status = Status(message='Successfully connected!')
            return self.status

    def connected(self):
        """
        Check if client is connected to the server

        :return: if client is connected
        :rtype bool
        """
        if not self._stub or not self.status or not self._channel:
            return False

        try:
            grpc.channel_ready_future(self._channel).result(timeout=2)
        except grpc.FutureTimeoutError:
            return False
        else:
            return True

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
            raise DisconnectNotConnectedClientError('Please connect to the server first!')

        try:
            del self._channel
            # self._channel.close()
        except Exception as e:
            LOGGER.error(e)
            return Status(code=Status.CONNECT_FAILED, message='Disconnection failed')

        self.status = None
        self._channel = None
        self._stub = None

        return Status(message='Disconnect successfully')

    def create_table(self, param, timeout=10):
        """Create table

        :type  param: dict or TableSchema
        :param param: Provide table information to be created

                `example param={'table_name': 'name',
                                'dimension': 16,
                                'index_file_size': 1024 (default)，
                                'metric_type': Metric_type.L2 (default)
                                }`

                `OR using Prepare.table_schema to create param`

        :type  metric_type:
        :param metric_type:

        :type  timeout:
        :param timeout:

        :return: Status, indicate if operation is successful
        :rtype: Status
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        if not isinstance(param, dict):
            raise ParamError("`param` should be a dict")

        if 'index_file_size' not in param:
            param['index_file_size'] = 1024
        if 'metric_type' not in param:
            param['metric_type'] = MetricType.L2

        table_schema = Prepare.table_schema(param)

        try:
            status = self._stub.CreateTable.future(table_schema).result(timeout=timeout)
            if status.error_code == 0:
                return Status(message='Create table successfully!')
            elif status.error_code == status_pb2.META_FAILED:
                LOGGER.error("Table {} already exists".format(param['table_name']))
                return Status(code=status.error_code, message=status.reason)
            else:
                LOGGER.error(status)
                return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='request timeout')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error')

    def has_table(self, table_name, timeout=10):
        """

        This method is used to test table existence.

        :param table_name: table name is going to be tested.
        :type table_name: str

        :return:
            bool, if given table_name exists

        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_name = Prepare.table_name(table_name)

        try:
            reply = self._stub.HasTable.future(table_name).result(timeout=timeout)
            if reply.status.error_code == 0:
                return reply.bool_reply
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return False
        except grpc.RpcError as e:
            LOGGER.error(e)
            return False

    def delete_table(self, table_name, timeout=20):
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
            else:
                return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='grpc transport timeout')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error')

    def create_index(self, table_name, index=None, timeout=-1):
        """
        :param table_name: table used to build index.
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
        if index is None:
            index = {
                'index_type': IndexType.FLAT,
                'nlist': 16384
            }
        elif not isinstance(index, dict):
            raise TypeError("param `index` should be a dictionary")

        index = {
            'index_type': index['index_type'] if 'index_type' in index else IndexType.FLAT,
            'nlist': index['nlist'] if 'nlist' in index else 16384
        }

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        index_param = Prepare.index_param(table_name, index)
        try:
            if timeout == -1:
                status = self._stub.CreateIndex(index_param)
            elif timeout < 0:
                raise ValueError("Param `timeout` should be a positive number or -1")
            else:
                status = self._stub.CreateIndex.future(index_param).result(timeout=timeout)
            if status.error_code == 0:
                return Status(message='Build index successfully!')
            else:
                return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error')

    def add_vectors(self, table_name, records, ids=None, timeout=180, *args, **kwargs):
        """
        Add vectors to table

        This function allows to pass in arguments which is type of `milvus_ob2.InsertParam`
        to avoid serializing and deserializing repeatedly, as follows:

            `obj.add_vectors(None, None, insert_param=param)`

        `obj` is a milvus object, param is an object which is type of `milvus_ob2.InsertParam`

        :param ids:

        :type  table_name: str
        :param table_name: table name been inserted

        :type  records: list[list[float]]

                `example records: [[1.2345],[1.2345]]`

                `OR using Prepare.records`

        :param records: list of vectors been inserted

        :returns:
            Status: indicate if vectors inserted successfully
            ids: list of id, after inserted every vector is given a id
        :rtype: (Status, list(int))
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        insert_param = kwargs.get('insert_param', None)

        if not insert_param:
            insert_param = Prepare.insert_param(table_name, records, ids)
        else:
            if not isinstance(insert_param, grpc_types.InsertParam):
                raise ParamError("The value of key 'insert_param' must be type of milvus_pb2.InsertParam")

        try:
            vector_ids = self._stub.Insert.future(insert_param).result(timeout=timeout)

            if vector_ids.status.error_code == 0:
                ids = list(vector_ids.vector_id_array)
                return Status(message='Add vectors successfully!'), ids
            else:
                return Status(code=vector_ids.status.error_code, message=vector_ids.status.reason), []
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), []
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(code=Status.UNEXPECTED_ERROR, message="request timeout"), []

    def search_vectors(self, table_name, top_k, nprobe, query_records, query_ranges=None):
        """
        Query vectors in a table

        :param query_ranges: (Optional) ranges for conditional search.
            If not specified, search in the whole table
        :type  query_ranges: list[(date, date)]

                `date` supports:
                   a. date-like-str, e.g. '2019-01-01'
                   b. datetime.date object, e.g. datetime.date(2019, 1, 1)
                   c. datetime.datetime object, e.g. datetime.datetime.now()

                example query_ranges:

                `query_ranges = [('2019-05-10', '2019-05-10'),(..., ...), ...]`

        :param table_name: table name been queried
        :type  table_name: str
        :param query_records: all vectors going to be queried

                `Using Prepare.records generate query_records`

        :type  query_records: list[list[float]] or list[RowRecord]
        :param top_k: int, how many similar vectors will be searched
        :type  top_k: int
        :param nprobe: cell num of probing
        :type nprobe: int

        :returns: (Status, res)

            Status:  indicate if query is successful
            res: TopKQueryResult, return when operation is successful

        :rtype: (Status, TopKQueryResult[QueryResult])
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        infos = Prepare.search_param(
            table_name, query_records, query_ranges, top_k, nprobe
        )

        results = TopKQueryResult()

        try:
            response = self._stub.Search(infos)
            if response.status.error_code != 0:
                return Status(code=response.status.error_code, message='grpc transport error'), []

            t0 = time.time()
            for topk_query_result in list(response.topk_query_result):
                results.append(
                    [QueryResult(id=qr.id, distance=qr.distance) for qr in list(topk_query_result.query_result_arrays)])

            results.clear()

            t1 = time.time()
            for topk_query_result in response.topk_query_result:
                results.append([QueryResult(id=result.id, distance=result.distance) for result in
                                topk_query_result.query_result_arrays])
            t2 = time.time()

            print("\n\t[1]: {:.4f} s\n\t[2]: {:.4f} s".format(t1 - t0, t2 - t1))

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                status = Status(code=Status.ILLEGAL_ARGUMENT, message=e.details())
            else:
                status = Status(code=e.code(), message='grpc transport error')

            return status, []

        return Status(message='Search vectors successfully!'), results

    def search_vectors_grpc(self, table_name, top_k, nprobe, query_records, query_ranges=None):
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        infos = Prepare.search_param(
            table_name, query_records, query_ranges, top_k, nprobe
        )

        results = TopKQueryResult()

        try:
            response = self._stub.Search(infos)
            if response.status.error_code != 0:
                return Status(code=response.status.error_code, message='grpc transport error'), []
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                status = Status(code=Status.ILLEGAL_ARGUMENT, message=e.details())
            else:
                status = Status(code=e.code(), message='grpc transport error')

            return status, []

        return Status(message='Search vectors successfully!'), response

    def search_vectors_in_files(self, table_name, file_ids, query_records, top_k, nprobe=16, query_ranges=None):
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
            query_results: list[TopKQueryResult]

        :rtype: (Status, TopKQueryResult[QueryResult])
        """

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        file_ids = list(map(int_or_str, file_ids))

        infos = Prepare.search_vector_in_files_param(
            table_name, query_records, query_ranges, top_k, nprobe, file_ids
        )

        results = TopKQueryResult()
        try:
            response = self._stub.SearchInFiles(infos)
            if response.status.error_code != 0:
                return Status(code=response.status.error_code, message='grpc transport error'), []

            for topk_query_result in list(response.topk_query_result):
                results.append(
                    [QueryResult(id=qr.id, distance=qr.distance) for qr in list(topk_query_result.query_result_arrays)])

            return Status(message='Search vectors successfully!'), results
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                status = Status(code=Status.ILLEGAL_ARGUMENT, message=e.details())
            else:
                status = Status(code=e.code(), message='grpc transport error')

            return status, []

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
            ts = self._stub.DescribeTable.future(table_name).result(timeout=timeout)

            if ts.table_name.status.error_code == 0:

                table = TableSchema(
                    table_name=ts.table_name.table_name,
                    dimension=ts.dimension,
                    index_file_size=ts.index_file_size,
                    metric_type=ts.metric_type
                )

                return Status(message='Describe table successfully!'), table
            else:
                LOGGER.error(ts.table_name.status)
                return Status(code=ts.table_name.status.error_code,
                              message=ts.table_name.status.reason), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), None

    def show_tables(self):
        """
        Show all tables in database

        :return:
            Status: indicate if this operation is successful

            tables: list of table names, return when operation
                    is successful
        :rtype:
            (Status, list[str])
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        # TODO how to status errors
        cmd = Prepare.cmd('balala')
        try:
            results = [table.table_name for table in self._stub.ShowTables(cmd)]
            return Status(message='Show tables successfully!'), results
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), []

    def get_table_row_count(self, table_name, timeout=10):
        """
        Get table row count

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
                return Status(message='Get table row count successfully!'), trc.table_row_count
            else:
                return Status(code=trc.status.error_code, message=trc.status.reason), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), []

    def client_version(self):
        """
        Provide client version

        :return:
            Status: indicate if operation is successful

            str : Client version

        :rtype: (Status, str)
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
        return self.cmd(cmd='version', timeout=timeout)

    def server_status(self, timeout=10):
        """
        Provide server status

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        return self.cmd(cmd='OK', timeout=timeout)

    def cmd(self, cmd, timeout=10):

        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        cmd = Prepare.cmd(cmd)
        try:
            ss = self._stub.Cmd.future(cmd).result(timeout=timeout)
            if ss.status.error_code == 0:
                return Status(message='Success!'), ss.string_reply
            else:
                return Status(code=ss.status.error_code, message=ss.status.reason), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), None

    def delete_vectors_by_range(self, table_name, start_date=None, end_date=None):
        """
        Delete vectors by range

        :type  table_name: str
        :param table_name:

        :type  start_date: str
        :param start_date:

        :type  end_date:
        :param end_date:

        :return:
        """
        delete_range = Prepare.delete_param(table_name, start_date, end_date)

        try:
            status = self._stub.DeleteByRange(delete_range)
            return Status(code=status.error_code, message=status.reason)
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error')

    def preload_table(self, table_name):
        """
        Load table to cache in advance

        :type table_name: str
        :param table_name: table to preload

        :returns:
            Status:  indicate if query is successful
        """
        table_name = Prepare.table_name(table_name)

        try:
            status = self._stub.PreloadTable(table_name)
            return Status(code=status.error_code, message=status.reason)
        except grpc.RpcError as e:
            return Status(code=e.code(), message='grpc transport error')

    def describe_index(self, table_name, timeout=10):
        """
        Show index information of designated table

        :type table_name: str
        :param table_name: table name been queried

        :returns:
            Status:  indicate if query is successful
            IndexSchema:
            
        """
        table_name = Prepare.table_name(table_name)

        try:
            index_param = self._stub.DescribeIndex.future(table_name).result(timeout=timeout)

            status = index_param.table_name.status

            if status.error_code == 0:
                return Status(message="Successfully"), \
                       IndexParam(index_param.table_name.table_name,
                                  index_param.index.index_type,
                                  index_param.index.nlist)
            else:
                return Status(code=status.error_code, message=status.reason), None
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(code=Status.UNEXPECTED_ERROR, message='grpc request timeout'), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error{}'.format(e.details())), None

    def drop_index(self, table_name, timeout=10):

        table_name = Prepare.table_name(table_name)

        try:
            status = self._stub.DropIndex.future(table_name).result(timeout=timeout)
            return Status(code=status.error_code, message=status.reason)
        except grpc.FutureTimeoutError as e:
            LOGGER.error(e)
            return Status(Status.UNEXPECTED_ERROR, message='grpc request timeout')
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error{}'.format(e.details()))
