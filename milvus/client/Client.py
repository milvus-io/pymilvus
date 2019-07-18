import logging
import sys
import struct

from thrift.transport import TSocket
from thrift.transport import TTransport, TZlibTransport
from thrift.protocol import TBinaryProtocol, TCompactProtocol, TJSONProtocol
from threading import Lock
from milvus.thrift import MilvusService
from milvus.thrift import ttypes
from milvus.client.Abstract import (
    ConnectIntf,
    TableSchema,
    QueryResult,
    TopKQueryResult,
    Range,
    RowRecord,
    IndexType
)

from milvus.client.Status import Status
from milvus.client.Exceptions import (
    RepeatingConnectError,
    DisconnectNotConnectedClientError,
    NotConnectError,
    ParamError,
)
from milvus.settings import DefaultConfig as config
from milvus.client.utils import *

if sys.version_info[0] > 2:
    from urllib.parse import urlparse
else:
    from urlparse import urlparse

LOGGER = logging.getLogger(__name__)

__version__ = '0.1.24'
__NAME__ = 'pymilvus'


class Protocol:
    JSON = 'JSON'
    BINARY = 'BINARY'
    COMPACT = 'COMPACT'


class Prepare(object):

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
        if not isinstance(param, ttypes.TableSchema):
            if isinstance(param, dict):
                temp = TableSchema(**param)

            else:
                raise ParamError('Param type incorrect, expect {} but get {} instead '.format(
                    type(dict), type(param)
                ))

        else:
            return param

        return ttypes.TableSchema(table_name=temp.table_name,
                                  dimension=temp.dimension,
                                  index_type=temp.index_type,
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

        return ttypes.Range(start_value=temp.start_date,
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
            if not isinstance(_range, ttypes.Range):
                res.append(Prepare.range(_range[0], _range[1]))
            else:
                res.append(_range)
        return res

    @classmethod
    def row_record(cls, vector_data):
        """
        Transfer a float binary str to RowRecord and return

        :type vector_data: list, list of float
        :param vector_data: (Required) vector data to store

        :return: RowRecord object

        """
        temp = vector_data if isinstance(vector_data, ttypes.RowRecord) \
            else RowRecord(vector_data)
        return ttypes.RowRecord(vector_data=temp.vector_data)

    @classmethod
    def records(cls, vectors):
        """
        Parser 2-dim array to list of RowRecords

        :param vectors: 2-dim array, lists of vectors
        :type vectors: list[list[float]]

        :return: binary vectors
        """
        if is_legal_array(vectors):
            return [Prepare.row_record(vector) for vector in vectors]
        else:
            raise ParamError('Vectors should be 2-dim array!')


class Milvus(ConnectIntf):
    """
    The Milvus object is used to connect and communicate with the server
    """

    def __init__(self):
        self.status = None
        self._transport = None
        self._tt = None
        self._client = None
        self.mutex = Lock()

    def __str__(self):
        return '<Milvus: {}>'.format(self.status)

    def set_client(self, host=None, port=None, uri=None):

        config_uri = urlparse(config.THRIFTCLIENT_TRANSPORT)

        _uri = urlparse(uri) if uri else config_uri

        if not host:
            if _uri.scheme == 'tcp':
                host = _uri.hostname
                port = _uri.port or 19530
            else:
                if uri:
                    raise RuntimeError(
                        'Invalid parameter uri: {}'.format(uri)
                    )
                raise RuntimeError(
                    'Invalid configuration for THRIFTCLIENT_TRANSPORT: {transport}'.format(
                        transport=config.THRIFTCLIENT_TRANSPORT)
                )
        else:
            host = host
            port = port or 19530

        self._transport = TSocket.TSocket(host, port)

        if config.THRIFTCLIENT_BUFFERED:
            self._tt = TTransport.TBufferedTransport(self._transport)
        if config.THRIFTCLIENT_ZLIB:
            self._tt = TZlibTransport.TZlibTransport(self._transport)
        if config.THRIFTCLIENT_FRAMED:
            self._tt = TTransport.TFramedTransport(self._transport)

        if config.THRIFTCLIENT_PROTOCOL == Protocol.BINARY:
            protocol = TBinaryProtocol.TBinaryProtocol(self._tt)

        elif config.THRIFTCLIENT_PROTOCOL == Protocol.COMPACT:
            protocol = TCompactProtocol.TCompactProtocol(self._tt)

        elif config.THRIFTCLIENT_PROTOCOL == Protocol.JSON:
            protocol = TJSONProtocol.TJSONProtocol(self._tt)

        else:
            raise RuntimeError(
                "invalid configuration for THRIFTCLIENT_PROTOCOL: {protocol}"
                .format(protocol=config.THRIFTCLIENT_PROTOCOL)
            )

        self._client = MilvusService.Client(protocol)


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

        :param timeout: (Optional) connection timeout, default timeout is 3s

        :return: Status, indicate if connect is successful
        :rtype: Status
        """
        if self.status and self.status == Status.SUCCESS:
            return Status(message="You have already connected!", code=Status.CONNECT_FAILED)

        if self._client is None:
            self.set_client(host, port, uri)

        try:
            self._transport.setTimeout(timeout)
            self._tt.open()
            self.status = Status(Status.SUCCESS, 'Connected')
            self._transport.setTimeout(None)
            return self.status

        except TTransport.TTransportException as e:
            self.status = Status(code=e.type, message=e.message)
            LOGGER.error(e)
            raise NotConnectError('Connection failed')

    def connected(self):
        """
        Check if client is connected to the server

        :return: if client is connected
        :rtype bool
        """
        if self._tt is None or not self._tt.isOpen():
            return False
        return True

    def disconnect(self):
        """
        Disconnect the client

        :return: Status, indicate if disconnect is successful
        :rtype: Status
        """

        if not self.connected():
            raise DisconnectNotConnectedClientError('Disconnect not connected client!')

        try:
            self._tt.close()
            self.status = None

        except TTransport.TTransportException as e:
            LOGGER.error(e)
            return Status(code=e.type, message=e.message)
        return Status(Status.SUCCESS, 'Disconnect successfully!')

    def create_table(self, param):
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
        with self.mutex:
            return self._create_table(param)

    def delete_table(self, table_name):
        """
        Delete table with table_name

        :type  table_name: str
        :param table_name: Name of the table being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """
        with self.mutex:
            return self._delete_table(table_name)

    def build_index(self, table_name):
        """
        Build index by table name

        This method is used to build index by table in sync mode.

        :param table_name: table is going to be built index.
        :type  table_name: str

        :return: Status, indicate if operation is successful
        """
        with self.mutex:
            return self._build_index(table_name)

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
        :rtype: (Status, list(str))
        """
        with self.mutex:
            return self._add_vectors(table_name, records)

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
        with self.mutex:
            return self._search_vectors_bin(table_name, top_k, query_records, query_ranges)

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

        :type  query_ranges: list[(date, date)]

            `date` supports:
               a. date-like-str, e.g. '2019-01-01'
               b. datetime.date object, e.g. datetime.date(2019, 1, 1)
               c. datetime.datetime object, e.g. datetime.datetime.now()

            example `query_ranges`:

                `query_ranges = [('2019-05-10', '2019-05-10'),(..., ...), ...]`

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]

        :rtype: (Status, TopKQueryResult[QueryResult])
        """
        with self.mutex:
            return self._search_vectors_in_files(table_name, file_ids, query_records, top_k, query_ranges)

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
        with self.mutex:
            return self._describe_table(table_name)

    def has_table(self, table_name):
        """

        This method is used to test table existence.

        :param table_name: table name is going to be tested.
        :type table_name: str

        :return:
            has_table: bool, if given table_name exists


        """
        with self.mutex:
            return self._has_table(table_name)

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
        with self.mutex:
            return self._show_tables()

    def get_table_row_count(self, table_name):
        """
        Get table row count

        :type  table_name: str
        :param table_name: target table name.

        :returns:
            Status: indicate if operation is successful

            res: int, table row count
        """
        with self.mutex:
            return self._get_table_row_count(table_name)

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
        with self.mutex:
            return self._server_version()

    def server_status(self, cmd=None):
        """
        Provide server status. When cmd !='version', provide 'OK'

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        with self.mutex:
            return self._server_status(cmd)

    def _create_table(self, param):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        param = Prepare.table_schema(param)

        try:
            self._client.CreateTable(param)
            return Status(message='Table {} created!'.format(param.table_name))
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason)

    def _delete_table(self, table_name):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        try:
            self._client.DeleteTable(table_name)
            return Status(message='Table {} deleted!'.format(table_name))
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason)

    def _build_index(self, table_name):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        try:
            self._client.BuildIndex(table_name)
            return Status(message='Success!'.format(table_name))
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason)

    def _add_vectors(self, table_name, records):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        records = Prepare.records(records)
        ids = []

        try:
            ids = self._client.AddVector(table_name=table_name, record_array=records)
            return Status(message='Vectors added successfully!'), ids
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), ids

    def _search_vectors_bin(self, table_name, top_k, query_records, query_ranges=None):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        query_records = Prepare.records(query_records)

        if not isinstance(top_k, int) or top_k <= 0:
            raise ParamError('Param top_k should be larger than 0!')

        if query_ranges:
            query_ranges = Prepare.ranges(query_ranges)

        res = TopKQueryResult()
        try:
            results = self._client.SearchVector2(
                table_name=table_name,
                query_record_array=query_records,
                query_range_array=query_ranges,
                topk=top_k)

            # deserialize bin array
            for topks in results:
                ids = topks.id_array
                distances = topks.distance_array
                count = len(ids) // 8
                assert count == len(distances) // 8

                ids = struct.unpack(str(count) + 'l', ids)
                distances = struct.unpack(str(count) + 'd', distances)

                qr = [QueryResult(ids[i], distances[i]) for i in range(count)]
                assert len(qr) == count

                res.append(qr)
            return Status(Status.SUCCESS, message='Search Vectors successfully!'), res
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), res



    def _search_vectors(self, table_name, top_k, query_records, query_ranges=None):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        query_records = Prepare.records(query_records)

        if not isinstance(top_k, int) or top_k <= 0:
            raise ParamError('Param top_k should be larger than 0!')

        if query_ranges:
            query_ranges = Prepare.ranges(query_ranges)

        res = TopKQueryResult()
        try:
            top_k_query_results = self._client.SearchVector(
                table_name=table_name,
                query_record_array=query_records,
                query_range_array=query_ranges,
                topk=top_k)

            for topk in top_k_query_results:
                res.append([QueryResult(id=qr.id, distance=qr.distance) for qr in topk.query_result_arrays])
            return Status(Status.SUCCESS, message='Search Vectors successfully!'), res
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), res

    def _search_vectors_in_files(self, table_name, file_ids, query_records, top_k, query_ranges=None):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        query_records = Prepare.records(query_records)

        if not isinstance(top_k, int) or top_k <= 0:
            raise ParamError('Param top_k should be larger than 0!')

        res = TopKQueryResult()
        file_ids = list(map(int_or_str, file_ids))
        try:
            top_k_query_results = self._client.SearchVectorInFiles(
                table_name=table_name,
                file_id_array=file_ids,
                query_record_array=query_records,
                query_range_array=query_ranges,
                topk=top_k)

            for topk in top_k_query_results:
                res.append([QueryResult(id=qr.id, distance=qr.distance) for qr in topk.query_result_arrays])
            return Status(Status.SUCCESS, message='Search vectors in files successfully!'), res
        except TTransport.TTransportException:
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), res

    def _describe_table(self, table_name):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        table = ''
        try:
            table = self._client.DescribeTable(table_name)
            return Status(message='Describe table successfully!'), table
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            if e.code == 3:
                LOGGER.info(e)
            else:
                LOGGER.error(e)
            return Status(code=e.code, message=e.reason), table

    def _has_table(self, table_name):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        has_table = False

        try:
            has_table = self._client.HasTable(table_name)
            return has_table
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            return NotConnectError('Please Connect to the server first!')
        # except ttypes.Exception as e:
        # LOGGER.error(e)

    def _show_tables(self):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        tables = []

        try:
            tables = self._client.ShowTables()
            return Status(message='Show tables successfully!'), tables
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), tables

    def _get_table_row_count(self, table_name):
        if not self.connected():
            raise NotConnectError('Please Connect to the server first!')

        count = 0

        try:
            count = self._client.GetTableRowCount(table_name)
            return Status(message='Get table row counts successfully'), count
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), count

    def _server_version(self):
        if not self.connected():
            raise NotConnectError('You have to connect first')
        server_version = ''
        try:
            server_version = self._client.Ping('version')
            return Status(message='Get version of server successfully'), server_version
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), server_version

    def _server_status(self, cmd=None):
        if not self.connected():
            raise NotConnectError('You have to connect first')

        result = 'OK'
        status = Status(message='Get status of server successfully')
        if cmd and cmd == 'version':
            status, result = self._server_version()

        return status, result
