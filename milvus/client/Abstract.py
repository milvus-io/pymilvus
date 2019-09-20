from .utils import *

from milvus.client.Exceptions import (
    ParamError
)

from .types import MetricType, IndexType


class TableSchema(object):
    """
    Table Schema

    :type  table_name: str
    :param table_name: (Required) name of table

        `IndexType`: 0-invalid, 1-flat, 2-ivflat, 3-IVF_SQ8, 4-MIX_NSG

    :type  dimension: int64
    :param dimension: (Required) dimension of vector

    :type  index_file_size: int64
    :param index_file_size: (Optional) max size of files which store index

    :type  metric_type: MetricType
    :param metric_type: (Optional) vectors metric type

        `MetricType`: 1-L2, 2-IP

    """

    def __init__(self, table_name, dimension, index_file_size, metric_type):

        # TODO may raise UnicodeEncodeError
        if table_name is None:
            raise ParamError('Table name can\'t be None')
        table_name = str(table_name) if not isinstance(table_name, str) else table_name
        if not is_legal_dimension(dimension):
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


class TopKQueryResult(object):
    """
    TopK query results, 2-D array of query result
    """

    def __init__(self, raw_source):
        self._raw = raw_source
        self._array = self._create_array(self._raw)

    def _create_array(self, raw):

        array = []
        for topk_result in raw.topk_query_result:
            topk_result_list = []
            topk_result_list.extend(topk_result.query_result_arrays)
            array.append(topk_result_list)

        return array

    @property
    def shape(self):
        row = len(self._array)

        if row == 0:
            column = 0
        else:
            column = len(self._array[0])

        return row, column

    def __getitem__(self, item):
        return self._array.__getitem__(item)

    def __iter__(self):
        return self._array.__iter__()

    def __len__(self):
        return self._array.__len__()

    def __repr__(self):
        """

        :return:
        """

        lam = lambda x: "(id:{}, distance:{})".format(x.id, x.distance)

        if self.__len__() > 10:
            middle = ''

            for topk in self[:3]:
                middle = middle + " [ %s" % ",\n   ".join(map(lam, topk[:3]))
                middle += ",\n   ..."
                middle += "\n   %s ]\n\n" % lam(topk[-1])

            spaces = """        ......
        ......"""

            ahead = "[\n%s%s\n]" % (middle, spaces)
            return ahead
        else:
            str_out_list = []
            for i in range(self.__len__()):
                str_out_list.append("[\n%s\n]" % ",\n".join(map(lam, self[i])))

            return "[\n%s\n]" % ",\n".join(str_out_list)


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

    def create_index(self, table_name, index):
        """
        Create specified index in a table
        should be implemented

        :type  table_name: str
        :param table_name: table name

         :type index: dict
        :param index: index information dict

            example: index = {
                "index_type": IndexType.FLAT,
                "nlist": 18384
            }

        :return:
            Status: indicate if this operation is successful

        :rtype: Status
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

    def server_status(self):
        """
        Provide server status. When cmd !='version', provide 'OK'
        should be implemented

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        _abstract()

    def delete_vectors_by_range(self, table_name, start_time, end_time):
        """
        delete vector by date range. The data range contains start_time but not end_time
        should be implemented

        :param table_name: table name
        :type  table_name: str

        :param start_time: range start time
        :type  start_time: str, date, datetime

        :param end_time: range end time(not contains in range)
        :type  end_time: str, date, datetime

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
            Status: indicate if operation is successful

        ：:rtype: Status
        """

        _abstract()

    def describe_index(self, table_name):
        """
        Show index information
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :return:
            Status: indicate if operation is successful

            TableSchema: table detail information

        :rtype: (Status, TableSchema)
        """

        _abstract()

    def drop_index(self, table_name):
        """
        Show index information
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """

        _abstract()
