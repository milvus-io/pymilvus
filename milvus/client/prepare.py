import copy

import ujson

from ..grpc_gen import milvus_pb2 as grpc_types
from ..grpc_gen import status_pb2


class Prepare:

    @classmethod
    def table_name(cls, table_name):

        return grpc_types.CollectionName(collection_name=table_name)

    @classmethod
    def table_schema(cls, table_name, dimension, index_file_size, metric_type, param):
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

        _param = grpc_types.CollectionSchema(status=status_pb2.Status(error_code=0, reason='Client'),
                                        collection_name=table_name,
                                        dimension=dimension,
                                        index_file_size=index_file_size,
                                        metric_type=metric_type)

        if param:
            param_str = ujson.dumps(param)
            _param.extra_params.add(key="params", value=param_str)

        return _param

    @classmethod
    def insert_param(cls, table_name, vectors, partition_tag, ids=None, params=None, **kwargs):
        if ids is None:
            _param = grpc_types.InsertParam(collection_name=table_name, partition_tag=partition_tag)
        else:
            _param = grpc_types.InsertParam(
                collection_name=table_name,
                row_id_array=ids,
                partition_tag=partition_tag)

        for vector in vectors:
            if isinstance(vector, bytes):
                _param.row_record_array.add(binary_data=vector)
            else:
                _param.row_record_array.add(float_data=vector)

        params = params or dict()
        params_str = ujson.dumps(params)
        _param.extra_params.add(key="params", value=params_str)

        return _param

    @classmethod
    def index_param(cls, table_name, index_type, params):

        _param = grpc_types.IndexParam(status=status_pb2.Status(error_code=0, reason='Client'),
                                       collection_name=table_name,
                                       index_type=index_type)
        params = params or dict()
        params_str = ujson.dumps(params)
        _param.extra_params.add(key="params", value=params_str)

        return _param

    @classmethod
    def search_param(cls, table_name, topk, query_records, partitions, params):

        search_param = grpc_types.SearchParam(
            collection_name=table_name,
            topk=topk,
            partition_tag_array=partitions
        )

        for vector in query_records:
            if isinstance(vector, bytes):
                search_param.query_record_array.add(binary_data=vector)
            else:
                search_param.query_record_array.add(float_data=vector)

        params = params or dict()
        params_str = ujson.dumps(params)
        search_param.extra_params.add(key="params", value=params_str)

        return search_param

    @classmethod
    def search_by_id_param(cls, table_name, top_k, id_, partition_tag_array, params):
        _param = grpc_types.SearchByIDParam(
            collection_name=table_name, id=id_, topk=top_k,
            partition_tag_array=partition_tag_array
        )

        params = params or dict()
        params_str = ujson.dumps(params)
        _param.extra_params.add(key="params", value=params_str)

        return _param

    @classmethod
    def search_vector_in_files_param(cls, table_name, query_records, topk, ids, params):
        _search_param = Prepare.search_param(table_name, topk, query_records,
                                             partitions=[], params=params)

        return grpc_types.SearchInFilesParam(
            file_id_array=ids,
            search_param=_search_param
        )

    @classmethod
    def cmd(cls, cmd):

        return grpc_types.Command(cmd=cmd)

    @classmethod
    def partition_param(cls, table_name, tag):

        return grpc_types.PartitionParam(collection_name=table_name, tag=tag)

    @classmethod
    def delete_by_id_param(cls, table_name, id_array):

        return grpc_types.DeleteByIDParam(collection_name=table_name, id_array=id_array)

    @classmethod
    def flush_param(cls, table_names):

        return grpc_types.FlushParam(collection_name_array=table_names)

    @classmethod
    def compact_param(cls, table_name):
        return grpc_types.CollectionName(collection_name=table_name)
