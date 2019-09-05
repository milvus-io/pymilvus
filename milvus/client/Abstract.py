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
    MIX_NSG = 4

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
        return self._name_


class MetricType(IntEnum):
    L2 = 1
    IP = 2

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
        return self._name_


class TableSchema(object):
    """
    Table Schema

    :type  table_name: str
    :param table_name: (Required) name of table

        `IndexType`: 0-invalid, 1-flat, 2-ivflat, 3-IVF_SQ8, 4-MIX_NSG

    :type  dimension: int64
    :param dimension: (Required) dimension of vector

    :type  index_file_size:
    :param index_file_size:

    :type  metric_type:
    :param metric_type:

    """

    def __init__(self, table_name, dimension, index_file_size, metric_type):

        # TODO may raise UnicodeEncodeError
        if table_name is None:
            raise ParamError('Table name can\'t be None')
        table_name = str(table_name) if not isinstance(table_name, str) else table_name
        if not legal_dimension(dimension):
            raise ParamError('Illegal dimension, effective range: (0 , 16384]')
        if isinstance(metric_type, int):
            metric_type = MetricType(metric_type)
        if not isinstance(metric_type, MetricType):
            raise ParamError('Illegal metric_type, should be MetricType')

        self.table_name = table_name
        self.dimension = dimension
        self.index_file_size = index_file_size
        self.metric_type = metric_type

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

    def __repr__(self):

        if self.__len__() > 10:
            middle = ''
            for topk in self[:3]:
                middle = middle + " [ %s" % ",\n   ".join(map(str, topk[:3]))
                middle += ",\n   ..."
                middle += "\n   %s ]\n\n" % str(topk[-1])

            spaces = """        ......
        ......"""

            ahead = "[\n%s%s\n]" % (middle, spaces)
            return ahead
        else:
            return "[\n%s\n]" % ",\n".join(map(str, self))


class IndexParam(object):
    """
    Index Param

    :type  table_name: str
    :param table_name: (Required) name of table

    :type  index_type: IndexType
    :param index_type: (Required) index type, default = IndexType.INVALID

        `IndexType`: 0-invalid, 1-flat, 2-ivflat, 3-IVF_SQ8, 4-MIX_NSG

    :type  nlist: int64
    :param nlist: (Required) num of cell

    :type  metric_type: int32
    :param metric_type: ???
    """

    def __init__(self, table_name, index_type, nlist):

        if table_name is None:
            raise ParamError('Table name can\'t be None')
        table_name = str(table_name) if not isinstance(table_name, str) else table_name

        if isinstance(index_type, int):
            index_type = IndexType(index_type)
        if not isinstance(index_type, IndexType) or index_type == IndexType.INVALID:
            raise ParamError('Illegal index_type, should be IndexType but not IndexType.INVALID')

        self._table_name = table_name
        self._index_type = index_type
        self._nlist = nlist

    def __str__(self):
        L = ['%s=%r' % (key.lstrip('_'), value)
             for key, value in self.__dict__.items()]
        return '(%s)' % (', '.join(L))

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))


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

    def delete_vectors_by_range(self, start_time, end_time):
        """

        :param start_time:
        :param end_time:
        :return:
        """

        _abstract()

    def preload_table(self, table_name):
        """
        load table to cache in advance
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :return:
        """

        _abstract()

    def describe_index(self, table_name):
        """
        Show index information
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :return:
        """

        _abstract()

    def drop_index(self, table_name):
        """
        Show index information
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :return:
        """

        _abstract()
