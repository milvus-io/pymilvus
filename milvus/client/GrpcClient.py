"""
This is a client for milvus of gRPC
"""
__version__ = '0.1.25'


import grpc
import logging
import concurrent.futures

from .Abstract import (
    ConnectIntf,
    IndexType,
    TableSchema,
    Range,
    RowRecord,
    QueryResult,
    TopKQueryResult
)
from .utils import *
from .Status import Status
from .Exceptions import *
from milvus.settings import DefaultConfig as config

from milvus.grpc_gen import milvus_pb2_grpc, status_pb2_grpc, status_pb2
from milvus.grpc_gen import milvus_pb2 as grpc_types 
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)


class Prepare(object):

    @classmethod
    def table_name(cls, table_name):
        if not isinstance(table_name, grpc_types.TableName):
            status = status_pb2.Status(error_code=0, reason='Client')
            return grpc_types.TableName(status=status, table_name=table_name)
        else:
            return table_name


    @classmethod
    def table_schema(cls, param):
        """
        :type param: dict
        :param param: (Required)

            `example param={'table_name': 'name',
                            'dimension': 16,
                            'index_type': IndexType.FLAT,
                            'store_raw_vector': False}`

        :return: ttypes.TableSchema object
        """
        if not isinstance(param, grpc_types.TableSchema):
            if isinstance(param, dict):

                temp = TableSchema(**param)

            else:
                raise ParamError('Param type incorrect, expect {} but get {} instead '.format(
                    type(dict), type(param)
                ))

        else:
            return param

        table_name = Prepare.table_name(temp.table_name)
        return grpc_types.TableSchema(table_name=table_name,
                                      index_type=temp.index_type,
                                      dimension=temp.dimension,
                                      store_raw_vector=temp.store_raw_vector)

    @classmethod
    def range(cls, start_date, end_date):
        """
        Parser a 'yyyy-mm-dd' like str or date/datetime object to Range object

            `Range: [start_date, end_date)`

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

    # TODO param handlers
    @classmethod
    def row_record(cls, vector_data):
        """
        Transfer a float binary str to RowRecord and return

        :type vector_data: list, list of float
        :param vector_data: (Required) vector data to store

        :return: RowRecord object

        """
        try:
            temp = vector_data if isinstance(vector_data, grpc_types.RowRecord) \
                else grpc_types.RowRecord(vector_data=vector_data)
        except TypeError as e:
            raise ParamError(str(e))
        return temp

    @classmethod
    def records(cls, vectors):
        """
        Parser 2-dim array to list of RowRecords

        :param vectors: 2-dim array, lists of vectors
        :type vectors: list[list[float]]

        :return: binary vectors
        """
        print('Start ...')
        if is_legal_array(vectors):
            return [Prepare.row_record(vector) for vector in vectors]
        else:
            raise ParamError('Vectors should be 2-dim array!')

    @classmethod
    def insert_infos(cls, table_name, vectors):
        infos = grpc_types.InsertInfos()

        infos.table_name = table_name

        for vector in vectors:
            if is_legal_array(vector):
                infos.row_record_array.add(vector_data=vector)
            else:
                raise ParamError('Vectors should be 2-dim array!')

        return infos

    @classmethod
    def search_vector_infos(cls, table_name, query_records, query_ranges, topk):
        query_records = Prepare.records(query_records)
        query_ranges = Prepare.ranges(query_ranges) if query_ranges else None

        return grpc_types.SearchVectorInfos(
                table_name=table_name,
                query_record_array=query_records,
                query_range_array=query_ranges,
                topk=topk
        )

    @classmethod
    def search_vector_in_files_infos(cls, table_name, query_records, topk, ids):
        search_infos = Prepare.search_vector_infos(table_name, query_records, None, topk)
        return grpc_types.SearchVectorInFilesInfos(
                    file_id_array=ids,
                    search_vector_infos=search_infos
        )


def on_connectivity_change(value):
    if value != grpc.ChannelConnectivity.READY:
        LOGGER.info("Connection changed: %s" % value)
    return

class GrpcMilvus(ConnectIntf):
    def __init__(self):
        self._channel = None
        self._stub = None
        self._uri = None
        self.server_address = None
        self.status = None

    def __str__(self):
        return '<Milvus: {}>'.format(self.status)

    def set_channel(self, host=None, port=None, uri=None):

        config_uri = urlparse(config.GRPC_URI)

        _uri = urlparse(uri) if uri else config_uri

        if not host:
            if _uri.scheme == 'tcp':
                host = _uri.hostname
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
        self._channel.subscribe(on_connectivity_change, try_to_connect=True)


    def connect(self, host=None, port=None, uri=None, timeout=3000):
        """
        Connect method should be called before any operations.
        Server will be connected after connect return OK

        :type  host: str
        :type  port: str
        :type  uri: str
        :type  timeout: str
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
            grpc.channel_ready_future(self._channel).result(timeout=timeout//1000)
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
            self._channel.close()
        except Exception as e:
            LOGGER.error(e)
            return Status(code=Status.CONNECT_FAILED, message='Disconnection failed')

        self.status = None
        self._channel = None
        self._stub = None

        return Status(message='Disconnect successfully')

    def create_table(self, param, timeout=None):
        """Create table

        :type  param: dict or TableSchema
        :param param: Provide table information to be created

                `example param={'table_name': 'name',
                                'dimension': 16,
                                'index_type': IndexType.FLAT,
                                'store_raw_vector': False}`

                `OR using Prepare.table_schema to create param`

        :return: Status, indicate if operation is successful
        :rtype: Status
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        table_schema = Prepare.table_schema(param)
        try:
            status = self._stub.CreateTable(table_schema)
            if status.error_code == 0:
                return Status(message='Create table successfully!')
            else:
                LOGGER.error(status)
                return Status(code=status.error_code, message=status.reason)
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error')


    def has_table(self, table_name):
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
            reply = self._stub.HasTable(table_name)
            if reply.status.error_code == 0:
                return reply.bool_reply
        except grpc.RpcError as e:
            LOGGER.error(e)
            return False

    def delete_table(self, table_name):
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
        status = self._stub.DropTable(table_name)

        try:
            if status.error_code == 0:
                return Status(message='Delete table successfully!')
            else:
                return Status(code=status.error_code, message=status.reason)
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error')

    def build_index(self, table_name):
        """
        Build index by table name

        This method is used to build index by table in sync mode.

        :param table_name: table is going to be built index.
        :type  table_name: str

        :return: Status, indicate if operation is successful
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        try:
            table_name = Prepare.table_name(table_name)
            status = self._stub.BuildIndex(table_name)
            if status.error_code == 0:
                return Status(message='Build index successfully!')
            else:
                return Status(code=status.error_code, message=status.reason)
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error')

    def add_vectors(self, table_name, records):
        """
        Add vectors to table

        :type  table_name: str
        :type  records: list[list[float]]

                `example records: [[1.2345],[1.2345]]`

                `OR using Prepare.records`

        :param table_name: table name been inserted
        :param records: list of vectors been inserted

        :returns:
            Status: indicate if vectors inserted successfully
            ids: list of id, after inserted every vector is given a id
        :rtype: (Status, list(int))
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        insert_infos = Prepare.insert_infos(table_name, records)
        
        try:
            vector_ids = self._stub.InsertVector(insert_infos)
            if vector_ids.status.error_code == 0:
                ids = list(vector_ids.vector_id_array)
                return Status(message='Add vectors successfully!'), ids
            else:
                return Status(code=vector_ids.status.error_code, message=vector_ids.status.reason), []
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), []


    def search_vectors(self, table_name, top_k, query_records, query_ranges=None):
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

        :returns: (Status, res)

            Status:  indicate if query is successful
            res: TopKQueryResult, return when operation is successful

        :rtype: (Status, TopKQueryResult[QueryResult])
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        if not isinstance(top_k, int) or top_k <= 0:
            raise ParamError('Param top_k should be larger than 0!')

        infos = Prepare.search_vector_infos(
                table_name, query_records, query_ranges, top_k
        )

        results = TopKQueryResult()
        try:
            for topks in self._stub.SearchVector(infos):
                results.append([QueryResult(id=qr.id, distance=qr.distance) for qr in topks.query_result_arrays])
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                status = Status(code=Status.ILLEGAL_ARGUMENT, message=e.details())
            else:
                status = Status(code=e.code(), message='grpc transport error')

            return status, []



        return Status(message='Search vectors successfully!'), results

    def search_vectors_in_files(self, table_name, file_ids, query_records, top_k, query_ranges=None):
        """
        Query vectors in a table, in specified files
        
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

        if not isinstance(top_k, int) or top_k <= 0:
            raise ParamError('Param top_k should be larger than 0!')

        file_ids = list(map(int_or_str, file_ids))

        infos = Prepare.search_vector_in_files_infos(
                table_name, query_records, top_k, file_ids
        )

        results = TopKQueryResult()
        try:
            for topks in self._stub.SearchVectorInFiles(infos):
                if topks.status.error_code == 0:
                    results.append([QueryResult(id=qr.id, distance=qr.distance) for qr in topks.query_result_arrays])
                else:
                    return Status(code=topks.status.error_code, message=topks.status.reason), []

            return Status(message='Search vectors successfully!'), results
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                status = Status(code=Status.ILLEGAL_ARGUMENT, message=e.details())
            else:
                status = Status(code=e.code(), message='grpc transport error')

            return status, []


    def describe_table(self, table_name):
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
            ts = self._stub.DescribeTable(table_name)

            if ts.table_name.status.error_code == 0:

                table = TableSchema(
                    table_name=ts.table_name.table_name,
                    dimension=ts.dimension,
                    index_type=IndexType(ts.index_type),
                    store_raw_vector=ts.store_raw_vector
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
        cmd  = grpc_types.Command(cmd='balala')
        try:
            results = [table.table_name for table in self._stub.ShowTables(cmd)]    
            return Status(message='Show tables successfully!'), results
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), []

    def get_table_row_count(self, table_name):
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
            trc = self._stub.GetTableRowCount(table_name)
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

    def server_version(self):
        """
        Provide server version

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        cmd  = grpc_types.Command(cmd='version')
        try:
            ss = self._stub.Ping(cmd)
            if ss.status.error_code == 0:
                return Status(message='Success!'), ss.info
            else:
                return Status(code=ss.status.error_code, message=ss.status.reason), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), None

    def server_status(self, cmd=None):
        """
        Provide server status. When cmd !='version', provide 'OK'

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        if not self.connected():
            raise NotConnectError('Please connect to the server first')

        if not cmd:
            cmd = 'OK'
        elif cmd == 'version':
            status, version = self.server_version()
            return status, version
        
        try:
            cmd = grpc_types.Command(cmd=cmd)
            ss = self._stub.Ping(cmd)
            if ss.status.error_code == 0:
                return Status(message='Success!'), ss.info
            else:
                return Status(code=ss.status.error_code, message=ss.status.reason), None
        except grpc.RpcError as e:
            LOGGER.error(e)
            return Status(e.code(), message='grpc transport error'), None
