import abc
import copy
import struct
import ujson

from .exceptions import ParamError

from ..grpc_gen import milvus_pb2 as grpc_types
from ..grpc_gen import status_pb2

from .types import RangeType, DataType

BoolOccurMap = {
    "must": grpc_types.MUST,
    "must_not": grpc_types.MUST_NOT,
    "should": grpc_types.SHOULD
}

RangeOperatorMap = {
    RangeType.LT: grpc_types.LT,
    RangeType.LTE: grpc_types.LTE,
    RangeType.EQ: grpc_types.EQ,
    RangeType.GT: grpc_types.GT,
    RangeType.GTE: grpc_types.GTE,
    RangeType.NE: grpc_types.NE
}


class Prepare:

    @classmethod
    def collection_name(cls, collection_name):

        return grpc_types.CollectionName(collection_name=collection_name)

    @classmethod
    def collection_schema(cls, collection_name, fields):
        """
        :type param: dict
        :param param: (Required)

            ` {"fields": [
                    {"field": "A", "type": DataType.INT64, "index": {"name":"", "type":"", "params": {..}}}
                    {"field": "B", "type": DataType.INT64},
                    {"field": "C", "type": DataType.INT64},
                    {"field": "Vec", "type": DataType.BINARY_VECTOR,
                     "params": {"metric_type": MetricType.L2, "dimension": 128}}
                ],
            "segment_size": 100}`

        :return: ttypes.TableSchema object
        """

        if not isinstance(fields, dict):
            raise ParamError("Param fields must be a dict")

        if "fields" not in fields:
            raise ParamError("Param fields must contains key 'fields'")

        schema = grpc_types.Mapping(collection_name=collection_name)

        extra_params = dict()
        for fk, fv in fields.items():
            if fk != "fields":
                extra_params[fk] = fv
                continue
            for field in fv:
                field_param = grpc_types.FieldParam()
                field_param.name = field["field"]

                ftype = field["type"]
                if not isinstance(ftype, (int, DataType)):
                    raise ParamError("'type' type is invalid, use DataType")

                field_param.type = int(DataType(ftype))
                if "index" in field:
                    index_field = copy.deepcopy(field["index"])
                    index_name = index_field["index_name"]
                    index_field.pop("index_name")
                    field_param.index_params.add(key=index_name, value=ujson.dumps(index_field))

                if "params" in field:
                    field_param.extra_params.add(key="params", value=ujson.dumps(field["params"]))

                schema.fields.append(field_param)

        schema.extra_params.add(key="params", value=ujson.dumps(extra_params))
        return schema

    @classmethod
    def reload_param(cls, collection_name, segment_ids):
        return grpc_types.ReLoadSegmentsParam(collection_name=collection_name, segment_id_array=segment_ids)

    @classmethod
    def insert_param(cls, collection_name, entities, partition_tag, ids=None, params=None, **kwargs):
        if ids is None:
            _param = grpc_types.InsertParam(collection_name=collection_name, partition_tag=partition_tag)
        else:
            _param = grpc_types.InsertParam(
                collection_name=collection_name,
                entity_id_array=ids,
                partition_tag=partition_tag)

        for entity in entities:
            type = entity.get("type", None)
            if type is None:
                raise ParamError("param entities must contain type")

            if not isinstance(type, DataType):
                raise ParamError("Param type must be type of DataType")

            values = entity.get("values", None)
            if values is None:
                raise ParamError("Param entities must contain values")

            field_param = grpc_types.FieldValue(field_name=entity["field"])
            if type in (DataType.INT32,):
            # if type in (DataType.INT8, DataType.INT16, DataType.INT32,):
                field_param.attr_record.CopyFrom(grpc_types.AttrRecord(int32_value=values))
            elif type in (DataType.INT64, ):
                field_param.attr_record.CopyFrom(grpc_types.AttrRecord(int64_value=values))
            elif type in (DataType.FLOAT, ):
                field_param.attr_record.CopyFrom(grpc_types.AttrRecord(float_value=values))
            elif type in (DataType.DOUBLE, ):
                field_param.attr_record.CopyFrom(grpc_types.AttrRecord(double_value=values))
            elif type in (DataType.FLOAT_VECTOR,):
                records = grpc_types.VectorRecord()
                for vector in values:
                    records.records.add(float_data=vector)
                field_param.vector_record.CopyFrom(records)
            elif type in (DataType.BINARY_VECTOR,):
                records = grpc_types.VectorRecord()
                for vector in values:
                    records.records.add(binary_data=vector)
                field_param.vector_record.CopyFrom(records)
            else:
                raise ParamError("Unknown data type.")

            _param.fields.append(field_param)

        params = params or dict()
        _param.extra_params.add(key="params", value=ujson.dumps(params))

        return _param

    @classmethod
    def get_entity_by_id_param(cls, collection_name, ids, fields):
        return grpc_types.EntityIdentity(collection_name=collection_name, id_array=ids, field_names=fields)

    @classmethod
    def index_param(cls, collection_name, field_name, params):
        _param = grpc_types.IndexParam(collection_name=collection_name, field_name=field_name)
        if params:
            for k, v in params.items():
                val = v if isinstance(v, str) else ujson.dumps(v)
                _param.extra_params.add(key=k, value=val)
            # _param.extra_params.add(key="params", value=ujson.dumps(params))
            # _param.extra_params.add(key="index_type", value=params["index_type"])
            # _param.extra_params.add(key="params", value=ujson.dumps(params["params"]))

        return _param

    @classmethod
    def search_param(cls, collection_name, query_entities, partition_tags=None, fields=None, **kwargs):
        if not isinstance(query_entities, (dict,)):
            raise ParamError("Invalid query format. 'query_entities' must be a dict")

        duplicated_entities = copy.deepcopy(query_entities)

        search_param = grpc_types.SearchParam(
            collection_name=collection_name,
            partition_tag_array=partition_tags
        )

        vector_placeholders = dict()

        def extract_vectors_param(param, placeholders):
            if not isinstance(param, (dict, list)):
                return

            if isinstance(param, dict):
                if "vector" in param:
                    # TODO: Here may not replace ph
                    ph = "place_holder_" + str(len(placeholders))
                    placeholders[ph] = param["vector"]
                    param["vector"] = ph
                    return
                else:
                    for k, v in param.items():
                        extract_vectors_param(v, placeholders)

            if isinstance(param, list):
                for item in param:
                    extract_vectors_param(item, placeholders)

        extract_vectors_param(duplicated_entities, vector_placeholders)
        search_param.dsl = ujson.dumps(duplicated_entities)

        for pk, pv in vector_placeholders.items():

            vector_param = grpc_types.VectorParam()
            for ppk, ppv in pv.items():
                if "query" not in ppv:
                    raise ParamError("param vector must contain 'query'")
                query = ppv["query"]
                for vec in query:
                    if isinstance(vec, bytes):
                        vector_param.row_record.records.add(binary_data=vec)
                    else:
                        vector_param.row_record.records.add(float_data=vec)

                ppv.pop("query")
                vector_param.json = ujson.dumps({pk: pv})
                # pv["query"] = query
                search_param.vector_param.append(vector_param)
                break

        field_list = fields or list()
        params = {"fields": field_list}

        search_param.extra_params.add(key="params", value=ujson.dumps(params))

        return search_param

    @classmethod
    def search_by_ids_param(cls, collection_name, ids, top_k, partition_tag_array, params):
        _param = grpc_types.SearchByIDParam(
            collection_name=collection_name, id_array=ids, topk=top_k,
        )
        if partition_tag_array:
            _param.partition_tag_array[:] = partition_tag_array

        params = params or dict()
        params_str = ujson.dumps(params)
        _param.extra_params.add(key="params", value=params_str)

        return _param

    @classmethod
    def search_vector_in_files_param(cls, collection_name, segment_ids, query_entities, fields, params):
        _search_param = Prepare.search_param(collection_name, query_entities, None, fields, params)

        return grpc_types.SearchInFilesParam(
            file_id_array=segment_ids,
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
    def compact_param(cls, collection_name, threshold):
        return grpc_types.CompactParam(collection_name=collection_name, threshold=threshold)
