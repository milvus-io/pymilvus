from ..grpc_gen import milvus_pb2 as grpc_types
from ..grpc_gen import status_pb2
from .abstract import Range


class Prepare:

    @classmethod
    def table_name(cls, table_name):

        return grpc_types.TableName(table_name=table_name)

    @classmethod
    def table_schema(cls, param):
        """
        :type param: dict
        :param param: (Required)

            `example param={'table_name': 'name',
                            'dimension': 16,
                            'index_file_size': 1024
                            'metric_type': MetricType.L2
                            }`

        :return: ttypes.TableSchema object
        """

        return grpc_types.TableSchema(status=status_pb2.Status(error_code=0, reason='Client'),
                                      table_name=param["table_name"],
                                      dimension=param["dimension"],
                                      index_file_size=param["index_file_size"],
                                      metric_type=param["metric_type"])

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
    def insert_param(cls, table_name, vectors, partition_tag, ids=None):

        if ids is None:
            _param = grpc_types.InsertParam(table_name=table_name, partition_tag=partition_tag)
        else:
            _param = grpc_types.InsertParam(
                table_name=table_name,
                row_id_array=ids,
                partition_tag=partition_tag)

        for vector in vectors:
            if isinstance(vector, bytes):
                _param.row_record_array.add(binary_data=vector)
            else:
                _param.row_record_array.add(float_data=vector)

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

        return grpc_types.Index(index_type=index_type, nlist=nlist)

    @classmethod
    def index_param(cls, table_name, index_param):

        _index = Prepare.index(**index_param)

        return grpc_types.IndexParam(status=status_pb2.Status(error_code=0, reason='Client'),
                                     table_name=table_name,
                                     index=_index)

    @classmethod
    def search_param(cls, table_name, topk, nprobe, query_records, query_ranges, partitions):
        query_ranges = Prepare.ranges(query_ranges) if query_ranges else None

        search_param = grpc_types.SearchParam(
            table_name=table_name,
            query_range_array=query_ranges,
            topk=topk,
            nprobe=nprobe,
            partition_tag_array=partitions
        )

        for vector in query_records:
            if isinstance(vector, bytes):
                search_param.query_record_array.add(binary_data=vector)
            else:
                search_param.query_record_array.add(float_data=vector)

        return search_param

    @classmethod
    def search_by_id_param(cls, table_name, top_k, nprobe, id_, partition_tag_array):
        return grpc_types.SearchByIDParam(
            table_name=table_name, id=id_,
            topk=top_k, nprobe=nprobe,
            partition_tag_array=partition_tag_array
        )

    @classmethod
    def search_vector_in_files_param(cls, table_name, query_records,
                                     query_ranges, topk, nprobe, ids):
        _search_param = Prepare.search_param(table_name, topk, nprobe, query_records,
                                             query_ranges, partitions=[])

        return grpc_types.SearchInFilesParam(
            file_id_array=ids,
            search_param=_search_param
        )

    @classmethod
    def cmd(cls, cmd):

        return grpc_types.Command(cmd=cmd)

    @classmethod
    def delete_param(cls, table_name, start_date, end_date):

        range_ = Prepare.range(start_date, end_date)

        return grpc_types.DeleteByDateParam(range=range_, table_name=table_name)

    @classmethod
    def partition_param(cls, table_name, partition_name, tag):

        return grpc_types.PartitionParam(table_name=table_name,
                                         partition_name=partition_name, tag=tag)

    @classmethod
    def delete_by_id_param(cls, table_name, id_array):

        return grpc_types.DeleteByIDParam(table_name=table_name, id_array=id_array)

    @classmethod
    def flush_param(cls, table_names):

        return grpc_types.FlushParam(table_name_array=table_names)

    @classmethod
    def compact_param(cls, table_name):
        return grpc_types.TableName(table_name=table_name)
