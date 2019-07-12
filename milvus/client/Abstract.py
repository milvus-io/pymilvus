from enum import IntEnum
import struct
from .utils import *

from milvus.client.Exceptions import (
    ParamError
)


class IndexType(IntEnum):
    INVALID = 0
    FLAT = 1
    IVFLAT = 2
    IVF_SQ8 = 3


class TableSchema(object):
    """
    Table Schema

    :type  table_name: str
    :param table_name: (Required) name of table

    :type  index_type: IndexType
    :param index_type: (Required) index type, default = IndexType.INVALID

        `IndexType`: 0-invalid, 1-flat, 2-ivflat

    :type  dimension: int64
    :param dimension: (Required) dimension of vector

    :type  store_raw_vector: bool
    :param store_raw_vector: (Optional) default = False

    """

    def __init__(self, table_name,
                 dimension=0,
                 index_type=IndexType.INVALID,
                 store_raw_vector=False):

        # TODO may raise UnicodeEncodeError
        if table_name is None:
            raise ParamError('Table name can\'t be None')
        table_name = str(table_name) if not isinstance(table_name, str) else table_name
        if not legal_dimension(dimension):
            raise ParamError('Illegal dimension, effective range: (0 , 16384]')
        if not isinstance(index_type, IndexType) or index_type == IndexType.INVALID:
            raise ParamError('Illegal index_type, should be IndexType but not IndexType.INVALID')
        if not isinstance(store_raw_vector, bool):
            raise ParamError('Illegal store_raw_vector, should be bool')

        self.table_name = table_name
        self.index_type = index_type
        self.dimension = dimension
        self.store_raw_vector = store_raw_vector

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))


class Range(object):
    """
    Range information

    :type  start_date: str, date or datetime

        `str should be YY-MM-DD format, e.g. "2019-07-01"`

    :param start_date: Range start date

    :type  end_date: str, date or datetime

        `str should be YY-MM-DD format, e.g. "2019-07-01"`

    :param end_date: Range end date

    """

    def __init__(self, start_date, end_date):
        start_date = parser_range_date(start_date)
        end_date = parser_range_date(end_date)
        if is_legal_date_range(start_date, end_date):
            self.start_date = start_date
            self.end_date = end_date
        else:
            raise ParamError("The start-date should be smaller"
                             " than or equal to end-date!")


class RowRecord(object):
    """
    Record inserted

    :type  vector_data: binary str
    :param vector_data: (Required) a vector

    """

    def __init__(self, vector_data):
        if isinstance(vector_data, list) and len(vector_data) > 0 \
                and isinstance(vector_data[0], float):
            self.vector_data = struct.pack(str(len(vector_data)) + 'd', *vector_data)
        else:
            raise ParamError('Illegal vector! Vector should be non-empty list of float.\n {}'
                             .format(vector_data))


class QueryResult(object):
    """
    Query result

    :type  id: int64
    :param id: id of the vector

    :type  distance: float
    :param distance: Vector similarity 0 <= distance <= 100

    """

    def __init__(self, id, distance):
        self.id = id
        self.distance = distance

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))


class TopKQueryResult(list):
    """
    TopK query results, list of QueryResult
    """
    pass


def _abstract():
    raise NotImplementedError('You need to override this function')


class ConnectIntf(object):
    """SDK client abstract class

    Connection is a abstract class

    """

    def connect(self, host=None, port=None, uri=None):
        """
        Connect method should be called before any operations
        Server will be connected after connect return OK
        Should be implemented

        :type  host: str
        :param host: host

        :type  port: str
        :param port: port

        :type  uri: str
        :param uri: (Optional) uri

        :return: Status,  indicate if connect is successful
        """
        _abstract()

    def connected(self):
        """
        connected, connection status
        Should be implemented

        :return: Status,  indicate if connect is successful
        """
        _abstract()

    def disconnect(self):
        """
        Disconnect, server will be disconnected after disconnect return SUCCESS
        Should be implemented

        :return: Status,  indicate if connect is successful
        """
        _abstract()

    def create_table(self, param):
        """
        Create table
        Should be implemented

        :type  param: TableSchema
        :param param: provide table information to be created

        :return: Status, indicate if connect is successful
        """
        _abstract()

    def has_table(self, table_name):
        """

        This method is used to test table existence.
        Should be implemented

        :param table_name: table name is going to be tested.
        :type table_name: str

        :return:
            has_table: bool, if given table_name exists

        """
        _abstract()

    def build_index(self, table_name):
        """
        Build index by table name

        This method is used to build index by table in sync mode.

        :param table_name: table is going to be built index.
        :type  table_name: str

        :return: Status, indicate if operation is successful
        """
        _abstract()

    def delete_table(self, table_name):
        """
        Delete table
        Should be implemented

        :type  table_name: str
        :param table_name: table_name of the deleting table

        :return: Status, indicate if connect is successful
        """
        _abstract()

    def add_vectors(self, table_name, records):
        """
        Add vectors to table
        Should be implemented

        :type  table_name: str
        :param table_name: table name been inserted

        :type  records: list[RowRecord]
        :param records: list of vectors been inserted

        :returns:
            Status : indicate if vectors inserted successfully
            ids :list of id, after inserted every vector is given a id
        """
        _abstract()

    def search_vectors(self, table_name, query_records, query_ranges, top_k):
        """
        Query vectors in a table
        Should be implemented

        :type  table_name: str
        :param table_name: table name been queried

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  query_ranges: list[Range]
        :param query_ranges: Optional ranges for conditional search.
            If not specified, search whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        _abstract()

    def search_vectors_in_files(self, table_name, file_ids, query_records, query_ranges, top_k):
        """
        Query vectors in a table, query vector in specified files
        Should be implemented

        :type  table_name: str
        :param table_name: table name been queried

        :type  file_ids: list[str]
        :param file_ids: Specified files id array

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  query_ranges: list[Range]
        :param query_ranges: Optional ranges for conditional search.
            If not specified, search whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        _abstract()

    def describe_table(self, table_name):
        """
        Show table information
        Should be implemented

        :type  table_name: str
        :param table_name: which table to be shown

        :returns:
            Status: indicate if query is successful
            table_schema: TableSchema, given when operation is successful
        """
        _abstract()

    def get_table_row_count(self, table_name):
        """
        Get table row count
        Should be implemented

        :type  table_name, str
        :param table_name, target table name.

        :returns:
            Status: indicate if operation is successful
            count: int, table row count
        """
        _abstract()

    def show_tables(self):
        """
        Show all tables in database
        should be implemented

        :return:
            Status: indicate if this operation is successful
            tables: list[str], list of table names
        """
        _abstract()

    def client_version(self):
        """
        Provide client version
        should be implemented

        :return:
            Status: indicate if operation is successful

            str : Client version

        :rtype: (Status, str)
        """
        _abstract()


    def server_version(self):
        """
        Provide server version
        should be implemented

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        _abstract()

    def server_status(self, cmd):
        """
        Provide server status. When cmd !='version', provide 'OK'
        should be implemented

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        _abstract()
