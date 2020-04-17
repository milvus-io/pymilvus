import copy
import struct
import ujson

from ..grpc_gen import milvus_pb2 as grpc_types
from ..grpc_gen import status_pb2


class Prepare:

    @classmethod
    def collection_name(cls, collection_name):

        return grpc_types.CollectionName(collection_name=collection_name)

    @classmethod
    def collection_schema(cls, collection_name, dimension, index_file_size, metric_type, param):
        """
        :type param: dict
        :param param: (Required)

            `example param={'collection_name': 'name',
                            'dimension': 16,
                            'index_file_size': 1024
                            'metric_type': MetricType.L2
                            }`

        :return: ttypes.TableSchema object
        """

        _param = grpc_types.CollectionSchema(status=status_pb2.Status(error_code=0, reason='Client'),
                                             collection_name=collection_name,
                                             dimension=dimension,
                                             index_file_size=index_file_size,
                                             metric_type=metric_type)

        if param:
            param_str = ujson.dumps(param)
            _param.extra_params.add(key="params", value=param_str)

        return _param

    @classmethod
    def collection_hybrid_schema(cls, collection_name, field, param):
        _param = grpc_types.Mapping(
            collection_name=collection_name
        )

        for k, v in field.items():
            if "data_type" in v:
                ft = grpc_types.FieldType(data_type=int(v["data_type"]))
                # ft.data_type = int(v["data_type"])
            elif "dimension" in v:
                ft = grpc_types.FieldType(vector_param=grpc_types.VectorFieldParam(dimension=v["dimension"]))
                # ft.vector_param = grpc_types.VectorFieldParam(dimension=v["dimension"])
            else:
                raise ValueError("Collection field not support {}".format(v))
            _param.fields.add(name=k, type=ft)

        if param:
            param_str = ujson.dumps(param)
            _param.extra_params.add(key="params", value=param_str)

        return _param

    @classmethod
    def insert_param(cls, collection_name, vectors, partition_tag, ids=None, params=None, **kwargs):
        if ids is None:
            _param = grpc_types.InsertParam(collection_name=collection_name, partition_tag=partition_tag)
        else:
            _param = grpc_types.InsertParam(
                collection_name=collection_name,
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
    def insert_hybrid_param(cls, collection_name, tag, entities, vector_entities, ids=None, params=None):
        entity = grpc_types.HEntity()

        _len = -1
        for v in entities.values():
            if not isinstance(v, list):
                raise ValueError("Field values must be a list")
            if _len == -1:
                _len = len(v)
            else:
                if len(v) != _len:
                    raise ValueError("Length is not equal")

        item_bytes = bytearray()
        for k, v in entities.items():
            entity.field_names.append(k)
            if isinstance(v, list):
                if isinstance(v[0], int):
                    item_bytes += struct.pack(str(len(v)) + 'q', *v)
                elif isinstance(v[0], float):
                    item_bytes += struct.pack(str(len(v)) + 'd', *v)
                else:
                    raise ValueError("Field item must be int or float")
            else:
                raise ValueError("Field values must be a list")
        entity.attr_records = bytes(item_bytes)
        entity.row_num=_len
        # vectors
        # entity.field_names.append(vector_field)
        for kv, vv in vector_entities.items():
            entity.field_names.append(kv)
            vector_field = grpc_types.VectorFieldValue()
            for vector in vv:
                if isinstance(vector, bytes):
                    vector_field.value.append(grpc_types.RowRecord(binary_data=vector))
                else:
                    vector_field.value.append(grpc_types.RowRecord(float_data=vector))
            entity.result_values.append(grpc_types.FieldValue(vector_value=vector_field))

        h_param = grpc_types.HInsertParam(
            collection_name=collection_name,
            partition_tag=tag,
            entities=entity
        )

        if ids:
            h_param.entity_id_array[:] = ids
        params = params or dict()
        params_str = ujson.dumps(params)
        h_param.extra_params.add(key="params", value=params_str)
        return h_param

    @classmethod
    def index_param(cls, collection_name, index_type, params):

        _param = grpc_types.IndexParam(status=status_pb2.Status(error_code=0, reason='Client'),
                                       collection_name=collection_name,
                                       index_type=index_type)
        params = params or dict()
        params_str = ujson.dumps(params)
        _param.extra_params.add(key="params", value=params_str)

        return _param

    @classmethod
    def search_param(cls, collection_name, topk, query_records, partitions, params):

        search_param = grpc_types.SearchParam(
            collection_name=collection_name,
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
    def search_by_id_param(cls, collection_name, top_k, id_, partition_tag_array, params):
        _param = grpc_types.SearchByIDParam(
            collection_name=collection_name, id=id_, topk=top_k,
            partition_tag_array=partition_tag_array
        )

        params = params or dict()
        params_str = ujson.dumps(params)
        _param.extra_params.add(key="params", value=params_str)

        return _param

    @classmethod
    def search_vector_in_files_param(cls, collection_name, query_records, topk, ids, params):
        _search_param = Prepare.search_param(collection_name, topk, query_records,
                                             partitions=None, params=params)

        return grpc_types.SearchInFilesParam(
            file_id_array=ids,
            search_param=_search_param
        )

    @classmethod
    def cmd(cls, cmd):

        return grpc_types.Command(cmd=cmd)

    @classmethod
    def partition_param(cls, collection_name, tag):

        return grpc_types.PartitionParam(collection_name=collection_name, tag=tag)

    @classmethod
    def delete_by_id_param(cls, collection_name, id_array):

        return grpc_types.DeleteByIDParam(collection_name=collection_name, id_array=id_array)

    @classmethod
    def flush_param(cls, collection_names):

        return grpc_types.FlushParam(collection_name_array=collection_names)

    @classmethod
    def compact_param(cls, collection_name):
        return grpc_types.CollectionName(collection_name=collection_name)
