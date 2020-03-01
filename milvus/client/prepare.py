import copy

from ..grpc_gen import milvus_pb2 as grpc_types
from ..grpc_gen import status_pb2


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

        table_param = copy.deepcopy(param)

        table_name = table_param["table_name"]
        table_param.pop("table_name")

        dimension = table_param["dimension"]
        table_param.pop("dimension")

        index_file_size = table_param["index_file_size"]
        table_param.pop("index_file_size")

        metric_type = table_param["metric_type"]
        table_param.pop("metric_type")

        _param = grpc_types.TableSchema(status=status_pb2.Status(error_code=0, reason='Client'),
                                        table_name=table_name,
                                        dimension=dimension,
                                        index_file_size=index_file_size,
                                        metric_type=metric_type)

        for k, v in table_param.items():
            _param.extra_params.add(key=k, value=v)

    @classmethod
    def insert_param(cls, table_name, vectors, partition_tag, ids=None, **kwargs):

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

        for k, v in kwargs.items():
            _param.extra_params.add(key=k, value=v)

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
    def search_param(cls, table_name, topk, query_records, partitions, **kwargs):

        search_param = grpc_types.SearchParam(
            table_name=table_name,
            topk=topk,
            partition_tag_array=partitions
        )

        for vector in query_records:
            if isinstance(vector, bytes):
                search_param.query_record_array.add(binary_data=vector)
            else:
                search_param.query_record_array.add(float_data=vector)

        for k, v in kwargs.items():
            search_param.extra_params.add(key=k, value=v)

        return search_param

    @classmethod
    def search_by_id_param(cls, table_name, top_k, id_, partition_tag_array, **kwargs):
        _param = grpc_types.SearchByIDParam(
            table_name=table_name, id=id_, topk=top_k,
            partition_tag_array=partition_tag_array
        )

        for k, v in kwargs.items():
            _param.extra_params.add(key=k, value=v)

    @classmethod
    def search_vector_in_files_param(cls, table_name, query_records, topk, ids, **kwargs):
        _search_param = Prepare.search_param(table_name, topk, query_records,
                                             partitions=[], **kwargs)

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
    def partition_param(cls, table_name, tag):

        return grpc_types.PartitionParam(table_name=table_name, tag=tag)

    @classmethod
    def delete_by_id_param(cls, table_name, id_array):

        return grpc_types.DeleteByIDParam(table_name=table_name, id_array=id_array)

    @classmethod
    def flush_param(cls, table_names):

        return grpc_types.FlushParam(table_name_array=table_names)

    @classmethod
    def compact_param(cls, table_name):
        return grpc_types.TableName(table_name=table_name)
