import copy
import struct
import ujson

from .exceptions import ParamError

from ..grpc_gen import milvus_pb2 as grpc_types
from ..grpc_gen import status_pb2

from .types import RangeType

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
    def collection_hybrid_schema(cls, collection_name, fields):
        _param = grpc_types.Mapping(
            collection_name=collection_name
        )

        for field in fields:
            if "data_type" in field:
                ft = grpc_types.FieldType(data_type=int(field["data_type"]))
                # ft.data_type = int(v["data_type"])
            elif "dimension" in field:
                ft = grpc_types.FieldType(vector_param=grpc_types.VectorFieldParam(dimension=field["dimension"]))
                # ft.vector_param = grpc_types.VectorFieldParam(dimension=v["dimension"])
            else:
                raise ValueError("Collection field not support {}".format(field))
            field_param = grpc_types.FieldParam(name=field["field_name"], type=ft)
            extra_params = field.get("extra_params", None)
            if extra_params:
                u = ujson.dumps(extra_params)
                field_param.extra_params.add(key="params", value=u)
            _param.fields.append(field_param)

        return _param

    @classmethod
    def reload_param(cls, collection_name, segment_ids):
        return grpc_types.ReLoadSegmentsParam(collection_name=collection_name, segment_id_array=segment_ids)

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
        entity_param = grpc_types.HEntity()

        _len = -1
        for entity in entities:
            values = entity["field_values"]
            if not isinstance(values, list):
                raise ValueError("Field values must be a list")
            if _len == -1:
                _len = len(values)
            else:
                if len(values) != _len:
                    raise ValueError("Length is not equal")

        for entity in entities:
            entity_param.field_names.append(entity["field_name"])
            values = entity["field_values"]
            if isinstance(values, list):
                if isinstance(values[0], int):
                    entity_param.attr_data.add(int_value=values)
                elif isinstance(values[0], float):
                    entity_param.attr_data.add(double_value=values)
                else:
                    raise ValueError("Field item must be int or float")
            else:
                raise ValueError("Field values must be a list")
        # entity_param.attr_records = bytes(item_bytes)
        entity_param.row_num = _len
        # vectors
        # entity.field_names.append(vector_field)
        for vector_entity in vector_entities:
            entity_param.field_names.append(vector_entity["field_name"])
            vector_field = grpc_types.VectorFieldRecord()
            vectors = vector_entity["field_values"]
            for vector in vectors:
                if isinstance(vector, bytes):
                    vector_field.value.add(binary_data=vector)
                else:
                    vector_field.value.add(float_data=vector)
            entity_param.vector_data.append(vector_field)

        h_param = grpc_types.HInsertParam(
            collection_name=collection_name,
            partition_tag=tag,
            entity=entity_param
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
    def search_hybrid_pb_param(cls, collection_name, query_entities, partition_tags, params):

        def term_query(node):
            if len(node) > 1:
                raise Exception()
            for k, v in node.items():
                vs = v.get("values", None)
                if not vs:
                    raise ValueError("Key values is missing")

                _term_param = grpc_types.TermQuery(field_name=k,
                                                   value_num=len(v["values"]),
                                                   # boost=node["boost"]
                                                   )
                if isinstance(vs, list):
                    if isinstance(vs[0], int):
                        _term_param.int_value[:] = vs
                    elif isinstance(vs[0], float):
                        _term_param.double_value[:] = vs
                    else:
                        raise ValueError("Field item must be int or float")
                else:
                    raise ValueError("Field values must be a list")
                return _term_param

        def range_query(node):
            if len(node) > 1:
                raise Exception("Item size > 1")
            for name, query in node.items():
                _range_param = grpc_types.RangeQuery(field_name=name,
                                                     # boost=node["boost"]
                                                     )
                for k, v in query["ranges"].items():
                    ope = RangeOperatorMap[k]
                    _range_param.operand.add(operator=ope, operand=str(v))

                return _range_param

        def vector_query(node):
            if len(node) > 1:
                raise Exception("Item size > 1")
            for name, query in node.items():
                _vector_param = grpc_types.VectorQuery(field_name=name,
                                                       # query_boost=node["boost"],
                                                       topk=query["topk"]
                                                       )
                for vector in query["query"]:
                    if isinstance(vector, bytes):
                        _vector_param.records.add(binary_data=vector)
                    else:
                        _vector_param.records.add(float_data=vector)

                _extra_param = query.get("params", None)

                _extra_param = _extra_param or dict()
                params_str = ujson.dumps(_extra_param)
                _vector_param.extra_params.add(key="params", value=params_str)
                return _vector_param

        def gene_node(key, node):
            if isinstance(node, list):
                bqr = grpc_types.BooleanQuery(occur=BoolOccurMap[key])
                for query in node:
                    if "term" in query:
                        # bqr.general_query.append(grpc_types.GeneralQuery(term_query=term_query(query["term"])))
                        bqr.general_query.add(term_query=term_query(query["term"]))
                    elif "range" in query:
                        # bqr.general_query.append(grpc_types.GeneralQuery(range_query=range_query(query["range"])))
                        bqr.general_query.add(range_query=range_query(query["range"]))
                    elif "vector" in query:
                        # bqr.general_query.append(grpc_types.GeneralQuery(vector_query=vector_query(query["vector"])))
                        bqr.general_query.add(vector_query=vector_query(query["vector"]))
                    else:
                        raise ValueError("Unknown ")

                return grpc_types.GeneralQuery(boolean_query=bqr)

            keys = node.keys()
            sq = {"must", "must_not", "should"}
            if len(keys) + len(sq) > len(set(keys) | sq):
                gqs = list()
                for k, v in node.items():
                    gq = gene_node(k, v)
                    gqs.append(gq)
                if len(gqs) == 1:
                    return gqs[0]

                bq0 = grpc_types.BooleanQuery(occur=grpc_types.INVALID)
                for g in gqs:
                    bq0.general_query.append(g)
                return grpc_types.GeneralQuery(boolean_query=bq0)

            # bqr = grpc_types.BooleanQuery(occur=BoolOccurMap[key])
            # for k, v in node.items():
            #     field_name = node["field_name"]
            #     if k == "term":
            #         bqr.general_query.append(grpc_types.GeneralQuery(term_query=term_query(v)))
            #     elif k == "range":
            #         bqr.general_query.append(grpc_types.GeneralQuery(range_query=range_query(v)))
            #     elif k == "vector":
            #         bqr.general_query.append(grpc_types.GeneralQuery(vector_query=vector_query(v)))
            #     else:
            #         raise ValueError("Unknown ")
            #
            # return grpc_types.GeneralQuery(boolean_query=bqr)

            # if len(node) == 1:
            #     for k, v in node.items():
            #         if k in ("must", "must_not", "should"):
            #             bq = grpc_types.BooleanQuery(occur=BoolOccurMap[k])
            #
            # for k, v in node.items():
            #     if k in ("must", "must_not", "should"):
            #         len(node) == 1:
            #         vqq = grpc_types.BooleanQuery(occur=grpc_types.INVALID)
            #     if k in ("must", "must_not", "should"):
            #         bq = grpc_types.BooleanQuery(occur=BoolOccurMap[k])
            #         vqq.general_query.append(grpc_types.GeneralQuery(boolean_query=bq))

        _param = grpc_types.HSearchParamPB(
            collection_name=collection_name,
            partition_tag_array=partition_tags
        )

        _param.general_query.CopyFrom(gene_node(None, query_entities["bool"]))

        for k, v in query_entities.items():
            if k == "bool":
                continue
            _param.extra_params.add(key=k, value=ujson.dumps(v))

        # import pdb;pdb.set_trace()
        # grpc_types.GeneralQuery(boolean_query=bool_node(query_entities))
        # params = params or dict()
        # params_str = ujson.dumps(params)
        # _param.extra_params.add(key="params", value=params_str)

        return _param

    @classmethod
    def search_hybrid_param(cls, collection_name, vector_params, dsl, partition_tags, params):
        # def replace_range_item(d):
        #     if not isinstance(d, dict):
        #         return
        #
        #     if "range" not in d:
        #         for ki, vi in d.itmes():
        #             replace_range_item(vi)
        #     else:
        #         range = d["range"]
        #         for ki, vi in range.itmes():
        #             ranges = vi["values"]
        #             for kii, vii in ranges.items():
        #                 ranges.pop(kii)
        #                 ranges[int(kii)] = vii
        #         return

        # dsl_out = copy.deepcopy(dsl)
        # replace_range_item(dsl_out)

        dsl_str = dsl if isinstance(dsl, str) else ujson.dumps(dsl)
        hybrid_param = grpc_types.HSearchParam(collection_name=collection_name,
                                               partition_tag_array=partition_tags or [],
                                               dsl=dsl_str)

        for v_p in vector_params:
            if "vector" not in v_p:
                raise ParamError("Vector param must contains key \'vector\'")
            # TODO: may need to copy vector_params
            query_vectors = v_p.pop("vector")
            json_ = ujson.dumps(v_p)

            vector_param = grpc_types.VectorParam(json=json_)
            for vector in query_vectors:
                if isinstance(vector, bytes):
                    vector_param.row_record.add(binary_data=vector)
                else:
                    vector_param.row_record.add(float_data=vector)
            hybrid_param.vector_param.append(vector_param)

        _params = params or dict()
        for k, v in _params.items():
            hybrid_param.extra_params.add(key=k, value=ujson.dumps(v))

        return hybrid_param

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
