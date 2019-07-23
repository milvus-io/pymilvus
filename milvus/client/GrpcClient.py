"""
This is a client for milvus of gRPC
"""
__version__ = '0.1.25'


import grpc

from .Abstract import (
    ConnectIntf,
    IndexType,
    TableSchema,
    Range,
    RowRecord,
    QueryResult,
    TopKQueryResult
)
from .Status import Status
from .Exceptions import *
from milvus.settings import DefaultConfig as config

from milvus.grpc_gen import milvus_pb2_grpc, status_pb2_grpc, status_pb2
from milvus.grpc_gen import milvus_pb2 as grpc_types 


class Prepare(object):

    @classmethod
    def table_name(cls, table_name):
        if not isinstance(table_name, grpc_types.TableName):
            status = grpc_types.status__pb2.Status(error_code=0, reason='')
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

    # TODO
    @classmethod
    def row_record(cls, vector_data):
        """
        Transfer a float binary str to RowRecord and return

        :type vector_data: list, list of float
        :param vector_data: (Required) vector data to store

        :return: RowRecord object

        """
        temp = vector_data if isinstance(vector_data, grpc_types.RowRecord) \
            else grpc_types.RowRecord().vector_data.extend(vector_data)
        return temp

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

    @classmethod
    def insert_infos(cls, table_name, vectors):

        records = Prepare.records(vectors)

        return grpc_types.SearchVectorInfos(
                table_name=table_name,
                row_record_array=records
        )

    @classmethod
    def search_vector_infos(cls, table_name, query_records, query_ranges, topk):
        query_records = Prepare.records(query_records)
        query_ranges = Prepare.ranges(query_ranges)

        # TODO test query range default None
        return grpc_types.SearchVectorInfos(
                table_name=table_name,
                query_record_array=query_records,
                query_range_array=query_ranges,
                topk=topk
        )


class GrpcMilvus(ConnectIntf):
    def __init__(self):
        self._channel = None
        self._stub = None
        self._uri = None
        self.status = None

    def __str__(self):
        return '<Milvus: {}>'.format(self.status)

    def set_uri(self, host=None, port=None, uri=None):

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

        self._uri = host + ':' + port



    # TODO timeout, connected, disconnected
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
        if self._uri is None:
            self.set_uri(host, port, uri)

        self.channel = grpc.insecure_channel(target)
        self.stub = milvus_pb2_grpc.MilvusServiceStub(self.channel)
        
        
    def connected(self):
        pass

    def disconnect(self):
        pass

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
        table_schema = Prepare.table_schema(param)
        # TODO status and exceptions
        status = self._stub.CreateTable(table_schema)
        if status.error_code == 0:
            return Status(message='Create table successfully!')

    def has_table(self, table_name):
        """

        This method is used to test table existence.

        :param table_name: table name is going to be tested.
        :type table_name: str

        :return:
            bool, if given table_name exists

        """
        table_name = Prepare.table_name(table_name)
        reply = self._stub.HasTable(table_name)
        if reply.status.error_code == 0:
            return reply.bool_reply
        pass

    def delete_table(self, table_name):
        """
        Delete table with table_name

        :type  table_name: str
        :param table_name: Name of the table being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """
        table_name = Prepare.table_name(table_name)
        status = self._stub.DropTable(table_name)
        if status.error_code == 0:
            return Status(message='Delete table successfully!')
        pass

    def build_index(self, table_name):
        """
        Build index by table name

        This method is used to build index by table in sync mode.

        :param table_name: table is going to be built index.
        :type  table_name: str

        :return: Status, indicate if operation is successful
        """
        table_name = Prepare.table_name(table_name)
        status = self._stub.BuildIndex(table_name)
        if status.error_code == 0:
            return Status(message='Build index successfully!')
        pass

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
        insert_infos = Prepare.insert_infos(table_name, records)
        
        vertor_ids = self._stub.InsertVector(insert_infos)
        if vertor_ids.status.error_code == 0:
            return vertor_ids.vertor_id_array

        pass

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
        infos = Prepare.search_vector_infos(
                table_name, query_records, query_ranges, top_k
                )

        #result = self._stub.SearchVector(infos)

        # TODO
        for k in range self._stub.SearchVector(infos):

    # TODO
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
        pass

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
        pass


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
        pass

    def get_table_row_count(self, table_name):
        """
        Get table row count

        :type  table_name: str
        :param table_name: target table name.

        :returns:
            Status: indicate if operation is successful

            res: int, table row count
        """
        pass

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
        pass

    def server_status(self, cmd=None):
        """
        Provide server status. When cmd !='version', provide 'OK'

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        pass

