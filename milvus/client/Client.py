import logging
import sys
import struct
import datetime

from thrift.transport import TSocket
from thrift.transport import TTransport, TZlibTransport
from thrift.protocol import TBinaryProtocol, TCompactProtocol, TJSONProtocol

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
    ParamError
)
from milvus.settings import DefaultConfig as config

if sys.version_info[0] > 2:
    from urllib.parse import urlparse
else:
    from urlparse import urlparse

LOGGER = logging.getLogger(__name__)

__version__ = '0.1.12'
__NAME__ = 'pymilvus'


class Protocol:
    JSON = 'JSON'
    BINARY = 'BINARY'
    COMPACT = 'COMPACT'


def is_correct_date_str(param):
    try:
        datetime.datetime.strptime(param, '%Y-%m-%d')
    except ValueError:
        LOGGER.error('Incorrect data format, should be YYYY-MM-DD')
        return False
    return True


def _legal_dimension(dim):
    if not isinstance(dim, int) or dim <=0 or dim > 10000:
        return False
    return True


class Prepare(object):

    @classmethod
    def table_schema(cls,
                     table_name,
                     dimension,
                     index_type=IndexType.INVALID,
                     store_raw_vector=False):
        """
        :type table_name: str
        :type dimension: int
        :type index_type: IndexType
        :type store_raw_vector: bool
        :param table_name: (Required) name of table
        :param dimension: (Required) dimension of the table
        :param index_type: (Required) index type, default = IndexType.INVALID
        :param store_raw_vector: (Optional) default = False

        :return: TableSchema object
        """
        if not isinstance(table_name, str) or not _legal_dimension(dimension) \
                or not isinstance(index_type, IndexType) or not isinstance(store_raw_vector, bool):
            raise ParamError('Param incorrect!')

        temp = TableSchema(table_name, dimension, index_type, store_raw_vector)

        return ttypes.TableSchema(table_name=temp.table_name,
                                  dimension=dimension,
                                  index_type=index_type,
                                  store_raw_vector=store_raw_vector)

    @classmethod
    def range(cls, start_date, end_date):
        """
        Parser a 'yyyy-mm-dd' like str to Range object

            `Range: [start_date, end_date)`

            `start_date : '2019-05-25'`

        :param start_date: start date
        :type  start_date: str
        :param end_date: end date
        :type  end_date: str

        :return: Range object
        """
        if is_correct_date_str(start_date) and is_correct_date_str(end_date):
            return ttypes.Range(start_value=start_date, end_value=end_date)
        else:
            raise ParamError('Range param start_date and end_date should be YYYY-MM-DD form')

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
            res.append(Prepare.range(_range[0], _range[1]))
        return res

    @classmethod
    def row_record(cls, vector_data):
        """
        Transfer a float binary str to RowRecord and return

        :type vector_data: list, list of float
        :param vector_data: (Required) vector data to store

        :return: RowRecord object

        """
        vector = struct.pack(str(len(vector_data)) + 'd', *vector_data)
        temp = RowRecord(vector)
        return ttypes.RowRecord(vector_data=temp.vector_data)

    @classmethod
    def records(cls, vectors):
        """
        Parser 2-dim array to list of RowRecords

        :param vectors: 2-dim array, lists of vectors
        :type vectors: list[list[float]]

        :return: binary vectors
        """
        return [Prepare.row_record(vector) for vector in vectors]


class Milvus(ConnectIntf):
    """
    The Milvus object is used to connect and communicate with the server
    """

    def __init__(self):
        self.status = None
        self._transport = None
        self._client = None

    def __repr__(self):
        return '{}'.format(self.status)

    def connect(self, host=None, port=None, uri=None):
        """
        Connect method should be called before any operations.
        Server will be connected after connect return OK

        :type  host: str
        :type  port: str
        :type  uri: str
        :param host: (Optional) host of the server, default host is 127.0.0.1
        :param port: (Optional) port of the server, default port is 9090
        :param uri: (Optional) only support tcp proto now, default uri is

                `tcp://127.0.0.1:9090`

        :return: Status, indicate if connect is successful
        :rtype: Status
        """
        if self.status and self.status == Status.SUCCESS:
            raise RepeatingConnectError("You have already connected!")

        config_uri = urlparse(config.THRIFTCLIENT_TRANSPORT)

        _uri = urlparse(uri) if uri else config_uri

        if not host:
            if _uri.scheme == 'tcp':
                host = _uri.hostname
                port = _uri.port or 9090
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
            port = port or 9090

        self._transport = TSocket.TSocket(host, port)
        self._transport.setTimeout(240)

        if config.THRIFTCLIENT_BUFFERED:
            self._transport = TTransport.TBufferedTransport(self._transport)
        if config.THRIFTCLIENT_ZLIB:
            self._transport = TZlibTransport.TZlibTransport(self._transport)
        if config.THRIFTCLIENT_FRAMED:
            self._transport = TTransport.TFramedTransport(self._transport)

        if config.THRIFTCLIENT_PROTOCOL == Protocol.BINARY:
            protocol = TBinaryProtocol.TBinaryProtocol(self._transport)

        elif config.THRIFTCLIENT_PROTOCOL == Protocol.COMPACT:
            protocol = TCompactProtocol.TCompactProtocol(self._transport)

        elif config.THRIFTCLIENT_PROTOCOL == Protocol.JSON:
            protocol = TJSONProtocol.TJSONProtocol(self._transport)

        else:
            raise RuntimeError(
                "invalid configuration for THRIFTCLIENT_PROTOCOL: {protocol}"
                    .format(protocol=config.THRIFTCLIENT_PROTOCOL)
            )

        self._client = MilvusService.Client(protocol)

        try:
            self._transport.open()
            self.status = Status(Status.SUCCESS, 'Connected')
            return self.status

        except TTransport.TTransportException as e:
            self.status = Status(code=e.type, message=e.message)
            LOGGER.error(e)
            raise NotConnectError('Connection failed')

    @property
    def connected(self):
        """
        Check if client is connected to the server

        :return: if client is connected
        :rtype bool
        """
        if not self:
            return False
        elif self.status and self.status.OK():
            return True
        return False

    def disconnect(self):
        """
        Disconnect the client

        :return: Status, indicate if disconnect is successful
        :rtype: Status
        """

        if not self._transport or not self.connected:
            raise DisconnectNotConnectedClientError('Disconnect not connected client!')

        try:
            self._transport.close()
            self.status = None

        except TTransport.TTransportException as e:
            LOGGER.error(e)
            return Status(code=e.type, message=e.message)
        return Status(Status.SUCCESS, 'Disconnect successfully!')

    def create_table(self, param):
        """Create table
        # TODO param dict

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
        if not self.connected:
            raise NotConnectError('Please Connect to the server first!')

        if not isinstance(param, ttypes.TableSchema):
            if isinstance(param, dict):
                param = Prepare.table_schema(**param)
            else:
                raise ParamError('Param incorrect!')

        try:
            self._client.CreateTable(param)
            return Status(message='Table {} created!'.format(param.table_name))
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason)

    def delete_table(self, table_name):
        """
        Delete table with table_name

        :type  table_name: str
        :param table_name: Name of the table being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """
        if not self.connected:
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

    def add_vectors(self, table_name, records):
        """
        Add vectors to table

        :type  table_name: str
        :type  records: list[list[float]] or list[RowRecord]

                `example records: [[1.2345],[1.2345]]`

                `OR using Prepare.records`

        :param table_name: table name been inserted
        :param records: list of vectors been inserted

        :returns:
            Status: indicate if vectors inserted successfully

            ids: list of id, after inserted every vector is given a id
        :rtype: (Status, list(str))
        """
        if not self.connected:
            raise NotConnectError('Please Connect to the server first!')

        if not isinstance(records[0], ttypes.RowRecord):
            if not records or not records[:1]:
                raise ParamError('Records empty!')
            if isinstance(records, list) and isinstance(records[0], list):
                records = Prepare.records(records)
            else:
                raise ParamError('Records param incorrect!')

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

    def search_vectors(self, table_name, top_k, query_records, query_ranges=None):
        """
        Query vectors in a table

        :param query_ranges: (Optional) ranges for conditional search.
            If not specified, search whole table
        :type  query_ranges: list[(str, str)]

                `date` supports date-like-str, e.g. '2019-01-01'

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
        if not self.connected:
            raise NotConnectError('Please Connect to the server first!')

        if not isinstance(query_records[0], ttypes.RowRecord):
            if not query_records or not query_records[:1]:
                raise ParamError('query_records empty!')
            if isinstance(query_records, list) and isinstance(query_records[0], list):
                query_records = Prepare.records(query_records)
            else:
                raise ParamError('query_records param incorrect!')

        if not isinstance(top_k, int) or top_k <= 0 or top_k > 10000:
            raise ParamError('Param top_k should be integer between (0, 10000]!')

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
                res.append([QueryResult(id=qr.id, score=qr.score) for qr in topk.query_result_arrays])
            return Status(Status.SUCCESS, message='Search Vectors successfully!'), res
        except TTransport.TTransportException as e:
            LOGGER.error(e)
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), res

    def search_vectors_in_files(self, table_name, file_ids, query_records, top_k, query_ranges=None):
        """
        Query vectors in a table, in specified files

        :type  table_name: str
        :param table_name: table name been queried

        :type  file_ids: list[str] or list[int]
        :param file_ids: Specified files id array

        :type  query_records: list[list[float]]
        :param query_records: all vectors going to be queried

        :type  query_ranges: list[Range]
        :param query_ranges: Optional ranges for conditional search.
            If not specified, search whole table

                `Range can be generated by Prepare.range`

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        :rtype: (Status, TopKQueryResult[QueryResult])
        """
        if not self.connected:
            raise NotConnectError('Please Connect to the server first!')

        # TODO query_ranges
        if not isinstance(query_records[0], ttypes.RowRecord):
            if not query_records or not query_records[:1]:
                raise ParamError('query_records empty!')
            if isinstance(query_records, list) and isinstance(query_records[0], list):
                query_records = Prepare.records(query_records)
            else:
                raise ParamError('query_records param incorrect!')

        if not isinstance(top_k, int) or top_k <= 0 or top_k > 10000:
            raise ParamError('Param top_k should be integer between (0, 10000]!')

        res = TopKQueryResult()
        file_ids = [str(item) for item in file_ids if isinstance(item, int)]
        try:
            top_k_query_results = self._client.SearchVectorInFiles(
                table_name=table_name,
                file_id_array=file_ids,
                query_record_array=query_records,
                query_range_array=query_ranges,
                topk=top_k)

            for topk in top_k_query_results:
                res.append([QueryResult(id=qr.id, score=qr.score) for qr in topk.query_result_arrays])
            return Status(Status.SUCCESS, message='Search vectors in files successfully!'), res
        except TTransport.TTransportException as e:
            raise NotConnectError('Please Connect to the server first')
        except ttypes.Exception as e:
            LOGGER.error(e)
            return Status(code=e.code, message=e.reason), res

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
        if not self.connected:
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

    # def has_table(self, table_name):
    #     """
    #
    #     This method is used to test table existence.
    #     Should be implemented
    #
    #     :param table_name: table name is going to be tested.
    #     :type table_name: str
    #
    #     :return:
    #         has_table: bool, if given table_name exists
    #
    #     """
    #     status, has_table = self._has_table(table_name)
    #
    #     if status.OK():
    #         return has_table

    def has_table(self, table_name):
        """

        This method is used to test table existence.

        :param table_name: table name is going to be tested.
        :type table_name: str

        :return:
            has_table: bool, if given table_name exists


        """
        if not self.connected:
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
        if not self.connected:
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

    def get_table_row_count(self, table_name):
        """
        Get table row count

        :type  table_name: str
        :param table_name: target table name.

        :returns:
            Status: indicate if operation is successful

            res: int, table row count
        """
        if not self.connected:
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
        if not self.connected:
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

    def server_status(self, cmd=None):
        """
        Provide server status. When cmd !='version', provide 'OK'

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        if not self.connected:
            raise NotConnectError('You have to connect first')

        result = 'OK'
        status = Status(message='Get status of server successfully')
        if cmd and cmd == 'version':
            status, result = self.server_version()

        return status, result
