import abc
import copy
import struct
import ujson
import mmh3

from .exceptions import ParamError

from ..grpc_gen import milvus_pb2 as grpc_types

# for milvus-distributed
from ..grpc_gen import common_pb2 as common_types
from ..grpc_gen import schema_pb2 as schema_types
from ..grpc_gen import milvus_pb2 as milvus_types

from . import blob

from .types import RangeType, DataType, MetricType, IndexType, PlaceholderType, DeployMode


class Prepare:

    @classmethod
    def create_collection_request(cls, collection_name, fields, **kwargs):
        """
        :type param: dict
        :param param: (Required)

            ` {"fields": [
                    {"field": "A", "type": DataType.INT32}
                    {"field": "B", "type": DataType.INT64, "auto_id": True},
                    {"field": "C", "type": DataType.FLOAT},
                    {"field": "Vec", "type": DataType.FLOAT_VECTOR,
                     "params": {"dim": 128}}
                ]
            }`

        :return: ttypes.TableSchema object
        """

        if not isinstance(fields, dict):
            raise ParamError("Param fields must be a dict")

        if "fields" not in fields:
            raise ParamError("Param fields must contains key 'fields'")

        schema = schema_types.CollectionSchema(name=collection_name)

        # auto_id = fields.get('auto_id', True)
        schema.autoID = False

        primary_field = None
        auto_id_field = None
        for fk, fv in fields.items():
            if fk != 'fields':
                if fk == 'description':
                    schema.description = fv
                continue

            for field in fv:
                field_schema = schema_types.FieldSchema()

                field_name = field.get('name')
                if field_name is None:
                    raise ParamError("You should specify the name of field!")
                field_schema.name = field_name

                data_type = field.get('type')
                if data_type is None:
                    raise ParamError("You should specify the data type of field!")
                if not isinstance(data_type, (int, DataType)):
                    raise ParamError("Field type must be of DataType!")
                field_schema.data_type = data_type

                field_schema.description = field.get('description', "")

                is_primary = field.get("is_primary", False)
                auto_id = field.get('auto_id', False)

                if is_primary and primary_field:
                    raise ParamError("A collection should only have a primary field")

                if auto_id and auto_id_field:
                    if DataType(data_type) != DataType.INT64:
                        raise ParamError("int64 is the only supported type of automatic generated id")
                    raise ParamError("A collection should only have a field whose id is automatically generated")

                if is_primary:
                    if DataType(data_type) != DataType.INT64:
                        raise ParamError("int64 is the only supported type of primary key")
                    primary_field = field_name

                if auto_id:
                    auto_id_field = field_name

                field_schema.is_primary_key = is_primary
                field_schema.autoID = auto_id

                type_params = field.get('params')
                if isinstance(type_params, dict):
                    for tk, tv in type_params.items():
                        if tk == "dim":
                            try:
                                int(tv)
                            except (TypeError, ValueError):
                                raise ParamError("invalid dim: " + str(tv)) from None
                        kv_pair = common_types.KeyValuePair(key=str(tk), value=str(tv))
                        field_schema.type_params.append(kv_pair)

                # No longer supported
                # index_params = field.get('indexes')
                # if isinstance(index_params, list):
                #     for index_param in index_params:
                #         if not isinstance(index_param, dict):
                #             raise ParamError("Every index param must be of dict type!")
                #         for ik, iv in index_param.items():
                #             if ik == "metric_type":
                #                 if not isinstance(iv, MetricType) and not isinstance(iv, str):
                #                     raise ParamError("metric_type must be of Milvus.MetricType or str!")
                #             kv_pair = common_types.KeyValuePair(key=str(ik), value=str(iv))
                #             field_schema.index_params.append(kv_pair)

                schema.fields.append(field_schema)

        return milvus_types.CreateCollectionRequest(collection_name=collection_name,
                                                    schema=bytes(schema.SerializeToString()))

    @classmethod
    def drop_collection_request(cls, collection_name):
        return milvus_types.DropCollectionRequest(collection_name=collection_name)

    @classmethod
    def has_collection_request(cls, collection_name):
        return milvus_types.HasCollectionRequest(collection_name=collection_name)

    @classmethod
    def describe_collection_request(cls, collection_name):
        return milvus_types.DescribeCollectionRequest(collection_name=collection_name)

    @classmethod
    def collection_stats_request(cls, collection_name):
        return milvus_types.CollectionStatsRequest(collection_name=collection_name)

    @classmethod
    def show_collections_request(cls):
        return milvus_types.ShowCollectionsRequest()

    @classmethod
    def create_partition_request(cls, collection_name, partition_name):
        return milvus_types.CreatePartitionRequest(collection_name=collection_name, partition_name=partition_name)

    @classmethod
    def drop_partition_request(cls, collection_name, partition_name):
        return milvus_types.DropPartitionRequest(collection_name=collection_name, partition_name=partition_name)

    @classmethod
    def has_partition_request(cls, collection_name, partition_name):
        return milvus_types.HasPartitionRequest(collection_name=collection_name, partition_name=partition_name)

    @classmethod
    def partition_stats_request(cls, collection_name, partition_name):
        return milvus_types.PartitionStatsRequest(collection_name=collection_name, partition_name=partition_name)

    @classmethod
    def show_partitions_request(cls, collection_name):
        return milvus_types.ShowPartitionsRequest(collection_name=collection_name)

    @classmethod
    def empty(cls):
        raise DeprecationWarning("no empty request later")
        # return common_types.Empty()

    @classmethod
    def register_link_request(cls):
        return milvus_types.RegisterLinkRequest()

    @classmethod
    def partition_name(cls, collection_name, partition_name):
        if not isinstance(collection_name, str):
            raise ParamError("collection_name must be of str type")
        if not isinstance(partition_name, str):
            raise ParamError("partition_name must be of str type")
        return milvus_types.PartitionName(collection_name=collection_name,
                                          tag=partition_name)

    @classmethod
    def bulk_insert_param(cls, collection_name, entities, partition_name, fields_info=None, **kwargs):
        default_partition_name = "_default"  # should here?
        tag = partition_name or default_partition_name
        insert_request = milvus_types.InsertRequest(collection_name=collection_name, partition_name=tag)

        for entity in entities:
            if not entity.get("name", None) or not entity.get("values", None) or not entity.get("type", None):
                raise ParamError("Missing param in entities, a field must have type, name and values")

        fields_name = list()
        fields_type = list()
        fields_len = len(entities)
        for i in range(fields_len):
            fields_name.append(entities[i]["name"])

        if not fields_info:
            raise ParamError("Missing collection meta to validate entities")

        location = dict()
        primary_key_loc = None
        auto_id_loc = None
        for i, field in enumerate(fields_info):
            if field.get("is_primary", False):
                primary_key_loc = i

            if field.get("auto_id", False):
                auto_id_loc = i
                continue

            match_flag = False
            field_name = field["name"]
            field_type = field["type"]

            for j in range(fields_len):
                entity_name = entities[j]["name"]
                entity_type = entities[j]["type"]

                if field_name == entity_name:
                    if field_type != entity_type:
                        raise ParamError(f"Collection field type is {field_type}"
                                         f", but entities field type is {entity_type}")

                    entity_dim = 0
                    field_dim = 0
                    if entity_type in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                        field_dim = field["params"]["dim"]
                        entity_dim = len(entities[j]["values"][0])

                    if entity_type in [DataType.FLOAT_VECTOR, ] and entity_dim != field_dim:
                        raise ParamError(f"Collection field dim is {field_dim}"
                                         f", but entities field dim is {entity_dim}")

                    if entity_type in [DataType.BINARY_VECTOR, ] and entity_dim * 8 != field_dim:
                        raise ParamError(f"Collection field dim is {field_dim}"
                                         f", but entities field dim is {entity_dim * 8}")

                    location[field["name"]] = j
                    fields_type.append(entities[j]["type"])
                    match_flag = True
                    break

            if not match_flag:
                raise ParamError("Field {} don't match in entities".format(field["name"]))

        # though impossible from sdk
        if primary_key_loc is None:
            raise ParamError("primary key not found")

        if auto_id_loc is None and len(entities) != len(fields_info):
            raise ParamError(f"number of fields: {len(fields_info)}, number of entities: {len(entities)}")

        if auto_id_loc is not None and len(entities) + 1 != len(fields_info):
            raise ParamError(f"number of fields: {len(fields_info)}, number of entities: {len(entities)}")

        row_num = 0

        for entity in entities:
            field_data = schema_types.FieldData()
            if entity.get("type") in (DataType.BOOL,):
                field_data.type = schema_types.DataType.Value("Bool")
                field_data.field_name = entity.get("name")
                field_data.scalars.bool_data.data.extend(entity.get("values"))
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
            elif entity.get("type") in (DataType.INT8,):
                field_data.type = schema_types.DataType.Value("Int8")
                field_data.field_name = entity.get("name")
                field_data.scalars.int_data.data.extend(entity.get("values"))
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
            elif entity.get("type") in (DataType.INT16,):
                field_data.type = schema_types.DataType.Value("Int16")
                field_data.field_name = entity.get("name")
                field_data.scalars.int_data.data.extend(entity.get("values"))
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
            elif entity.get("type") in (DataType.INT32,):
                field_data.type = schema_types.DataType.Value("Int32")
                field_data.field_name = entity.get("name")
                field_data.scalars.int_data.data.extend(entity.get("values"))
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
            elif entity.get("type") in (DataType.INT64,):
                field_data.type = schema_types.DataType.Value("Int64")
                field_data.field_name = entity.get("name")
                field_data.scalars.long_data.data.extend(entity.get("values"))
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
            elif entity.get("type") in (DataType.FLOAT,):
                field_data.type = schema_types.DataType.Value("Float")
                field_data.field_name = entity.get("name")
                field_data.scalars.float_data.data.extend(entity.get("values"))
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
            elif entity.get("type") in (DataType.DOUBLE,):
                field_data.type = schema_types.DataType.Value("Double")
                field_data.field_name = entity.get("name")
                field_data.scalars.double_data.data.extend(entity.get("values"))
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
            elif entity.get("type") in (DataType.FLOAT_VECTOR,):
                field_data.type = schema_types.DataType.Value("FloatVector")
                field_data.field_name = entity.get("name")
                field_data.vectors.dim = len(entity.get("values")[0])
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
                all_floats = [f for vector in entity.get("values") for f in vector]
                field_data.vectors.float_vector.data.extend(all_floats)
            elif entity.get("type") in (DataType.BINARY_VECTOR,):
                field_data.type = schema_types.DataType.Value("BinaryVector")
                field_data.field_name = entity.get("name")
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError("row num of all fields is not different")
                row_num = len(entity.get("values"))
                field_data.vectors.dim = len(entity.get("values")[0]) * 8
                for vector in entity.get("values"):
                    field_data.vectors.binary_vector += vector
            else:
                raise ParamError("UnSupported data type")

            insert_request.fields_data.append(field_data)

        insert_request.num_rows = row_num

        # generate hash keys, TODO: better hash function
        if not fields_info[primary_key_loc].get("auto_id", False):
            field_name = fields_info[primary_key_loc].get("name")
            entity_loc = location[field_name]
            hash_keys = [abs(mmh3.hash(str(e))) for e in entities[entity_loc].get("values")]
            insert_request.hash_keys.extend(hash_keys)

        return insert_request

    @classmethod
    def _prepare_placeholders(cls, vectors, nq, max_nq_per_batch, tag, pl_type, is_binary, dimension=0):
        pls = []
        for begin in range(0, nq, max_nq_per_batch):
            end = min(begin + max_nq_per_batch, nq)
            pl = milvus_types.PlaceholderValue(tag=tag)
            pl.type = pl_type
            for i in range(begin, end):
                if is_binary:
                    if len(vectors[i]) * 8 != dimension:
                        raise ParamError("The dimension of query entities is different from schema")
                    pl.values.append(blob.vectorBinaryToBytes(vectors[i]))
                else:
                    if len(vectors[i]) != dimension:
                        raise ParamError("The dimension of query entities is different from schema")
                    pl.values.append(blob.vectorFloatToBytes(vectors[i]))
            pls.append(pl)
        return pls

    @classmethod
    def divide_search_request(cls, collection_name, query_entities, partition_names=None, fields=None, **kwargs):
        schema = kwargs.get("schema", None)
        fields_schema = schema.get("fields", None)  # list
        fields_name_locs = {fields_schema[loc]["name"]: loc
                            for loc in range(len(fields_schema))}

        if not isinstance(query_entities, (dict,)):
            raise ParamError("Invalid query format. 'query_entities' must be a dict")

        duplicated_entities = copy.deepcopy(query_entities)
        vector_placeholders = dict()
        vector_names = dict()

        meta = {}   # TODO: ugly here, find a better method
        def extract_vectors_param(param, placeholders, meta=None, names=None):
            if not isinstance(param, (dict, list)):
                return

            if isinstance(param, dict):
                if "vector" in param:
                    # TODO: Here may not replace ph
                    ph = "$" + str(len(placeholders))

                    for pk, pv in param["vector"].items():
                        if "query" not in pv:
                            raise ParamError("param vector must contain 'query'")
                        if "topk" not in pv:
                            raise ParamError("dsl must contain 'topk'")
                        topk = pv["topk"]
                        if not isinstance(topk, (int, str)):
                            raise ParamError("topk must be int or str")
                        try:
                            topk = int(topk)
                        except Exception:
                            raise ParamError("topk is not illegal") from None
                        if topk < 0:
                            raise ParamError("topk must be greater than zero")
                        meta["topk"] = topk
                        placeholders[ph] = pv["query"]
                        names[ph] = pk
                        param["vector"][pk]["query"] = ph

                    return
                else:
                    for _, v in param.items():
                        extract_vectors_param(v, placeholders, meta, names)

            if isinstance(param, list):
                for item in param:
                    extract_vectors_param(item, placeholders, meta, names)

        extract_vectors_param(duplicated_entities, vector_placeholders, meta, vector_names)

        if len(vector_placeholders) > 1:
            raise ParamError("query on two vector field is not supported now!")

        requests = []
        factor = 10
        topk = meta.get("topk", 100)    # TODO: ugly here, find a better method
        for tag, vectors in vector_placeholders.items():
            if len(vectors) <= 0:
                continue

            if isinstance(vectors[0], bytes):
                bytes_per_vector = len(vectors[0])
            else:
                bytes_per_vector = len(vectors[0]) * 4

            nq = len(vectors)
            max_nq_per_batch = (5 * 1024 * 1024) / (bytes_per_vector * topk * factor)
            max_nq_per_batch = int(max_nq_per_batch)
            if max_nq_per_batch <= 0:
                raise ParamError(f"topk {topk} is too large!")

            if isinstance(vectors[0], bytes):
                is_binary = True
                pl_type = PlaceholderType.BinaryVector
            else:
                is_binary = False
                pl_type = PlaceholderType.FloatVector

            fname = vector_names[tag]
            if fname not in fields_name_locs:
                raise ParamError(f"Field {fname} doesn't exist in schema")
            dimension = int(fields_schema[fields_name_locs[fname]]["params"].get("dim", 0))
            pls = cls._prepare_placeholders(vectors, nq, max_nq_per_batch, tag, pl_type, is_binary, dimension)

            for pl in pls:
                plg = milvus_types.PlaceholderGroup()
                plg.placeholders.append(pl)
                plg_str = milvus_types.PlaceholderGroup.SerializeToString(plg)
                request = milvus_types.SearchRequest(
                    collection_name=collection_name,
                    partition_names=partition_names,
                    output_fields=fields,
                )
                request.dsl = ujson.dumps(duplicated_entities)
                request.placeholder_group = plg_str
                requests.append(request)

            return requests

    @classmethod
    def search_request(cls, collection_name, query_entities, partition_names=None, fields=None, **kwargs):
        schema = kwargs.get("schema", None)
        fields_schema = schema.get("fields", None)  # list
        fields_name_locs = {fields_schema[loc]["name"]: loc
                            for loc in range(len(fields_schema))}

        if not isinstance(query_entities, (dict,)):
            raise ParamError("Invalid query format. 'query_entities' must be a dict")

        if fields is not None and not isinstance(fields, (list, )):
            raise ParamError("Invalid query format. 'fields' must be a list")

        request = milvus_types.SearchRequest(
            collection_name=collection_name,
            partition_names=partition_names,
            output_fields=fields,
        )

        duplicated_entities = copy.deepcopy(query_entities)
        vector_placeholders = dict()
        vector_names = dict()

        def extract_vectors_param(param, placeholders, names):
            if not isinstance(param, (dict, list)):
                return

            if isinstance(param, dict):
                if "vector" in param:
                    # TODO: Here may not replace ph
                    ph = "$" + str(len(placeholders))

                    for pk, pv in param["vector"].items():
                        if "query" not in pv:
                            raise ParamError("param vector must contain 'query'")
                        placeholders[ph] = pv["query"]
                        names[ph] = pk
                        param["vector"][pk]["query"] = ph

                    return
                else:
                    for _, v in param.items():
                        extract_vectors_param(v, placeholders, names)

            if isinstance(param, list):
                for item in param:
                    extract_vectors_param(item, placeholders, names)

        extract_vectors_param(duplicated_entities, vector_placeholders, vector_names)
        request.dsl = ujson.dumps(duplicated_entities)

        plg = milvus_types.PlaceholderGroup()
        for tag, vectors in vector_placeholders.items():
            if len(vectors) <= 0:
                continue
            pl = milvus_types.PlaceholderValue(tag=tag)

            fname = vector_names[tag]
            if fname not in fields_name_locs:
                raise ParamError(f"Field {fname} doesn't exist in schema")
            dimension = int(fields_schema[fields_name_locs[fname]]["params"].get("dim", 0))

            if isinstance(vectors[0], bytes):
                pl.type = PlaceholderType.BinaryVector
                for vector in vectors:
                    if dimension != len(vector) * 8:
                        raise ParamError("The dimension of query vector is different from schema")
                    pl.values.append(blob.vectorBinaryToBytes(vector))
            else:
                pl.type = PlaceholderType.FloatVector
                for vector in vectors:
                    if dimension != len(vector):
                        raise ParamError("The dimension of query vector is different from schema")
                    pl.values.append(blob.vectorFloatToBytes(vector))
            # vector_values_bytes = service_msg_types.VectorValues.SerializeToString(vector_values)

            plg.placeholders.append(pl)
        plg_str = milvus_types.PlaceholderGroup.SerializeToString(plg)
        request.placeholder_group = plg_str

        return request

    @classmethod
    def search_requests_with_expr(cls, collection_name, data, anns_field, param, limit, expr=None, partition_names=None, output_fields=None, **kwargs):
        schema = kwargs.get("schema", None)
        fields_schema = schema.get("fields", None)  # list
        fields_name_locs = {fields_schema[loc]["name"]: loc
                            for loc in range(len(fields_schema))}

        requests = []

        if len(data) <= 0:
            return requests

        if isinstance(data[0], bytes):
            bytes_per_vector = len(data[0])
            is_binary = True
            pl_type = PlaceholderType.BinaryVector
        else:
            bytes_per_vector = len(data[0]) * 4
            is_binary = False
            pl_type = PlaceholderType.FloatVector

        nq = len(data)
        max_nq_per_batch = nq
        if kwargs.get("_deploy_mode", DeployMode.Distributed):
            factor = 10
            max_nq_per_batch = (5 * 1024 * 1024) / (bytes_per_vector * limit * factor)
            max_nq_per_batch = int(max_nq_per_batch)
            if max_nq_per_batch <= 0:
                raise ParamError(f"limit {limit} is too large!")

        tag = "$0"
        if anns_field not in fields_name_locs:
            raise ParamError(f"Field {anns_field} doesn't exist in schema")
        dimension = int(fields_schema[fields_name_locs[anns_field]]["params"].get("dim", 0))
        pls = cls._prepare_placeholders(data, nq, max_nq_per_batch, tag, pl_type, is_binary, dimension)

        param_copy = copy.deepcopy(param)
        metric_type = param_copy.pop("metric_type", "L2")
        params = param_copy.pop("params", {})
        if not isinstance(params, dict):
            raise ParamError("Search params must be a dict")
        search_params = {"anns_field": anns_field, "topk": limit, "metric_type": metric_type, "params": params}

        def dump(v):
            if isinstance(v, dict):
                return ujson.dumps(v)
            return str(v)

        for pl in pls:
            plg = milvus_types.PlaceholderGroup()
            plg.placeholders.append(pl)
            plg_str = milvus_types.PlaceholderGroup.SerializeToString(plg)
            request = milvus_types.SearchRequest(
                collection_name=collection_name,
                partition_names=partition_names,
                output_fields=output_fields,
            )
            request.placeholder_group = plg_str

            request.dsl_type = common_types.DslType.BoolExprV1
            if expr is not None:
                request.dsl = expr
            request.search_params.extend([common_types.KeyValuePair(key=str(key), value=dump(value))
                                          for key, value in search_params.items()])

            requests.append(request)

        return requests

    @classmethod
    def create_index__request(cls, collection_name, field_name, params):
        index_params = milvus_types.CreateIndexRequest(collection_name=collection_name, field_name=field_name)

        # index_params.collection_name = collection_name
        # index_params.field_name = field_name

        def dump(tv):
            if isinstance(tv, dict):
                import json
                return json.dumps(tv)
            return str(tv)

        if isinstance(params, dict):
            for tk, tv in params.items():
                if tk == "dim":
                    if not tv or not isinstance(tv, int):
                        raise ParamError("dim must be of int!")
                kv_pair = common_types.KeyValuePair(key=str(tk), value=dump(tv))
                index_params.extra_params.append(kv_pair)

        return index_params

    @classmethod
    def describe_index_request(cls, collection_name, index_name):
        return milvus_types.DescribeIndexRequest(collection_name=collection_name, index_name=index_name)

    @classmethod
    def describe_index_progress_request(cls, collection_name, field_name):
        return milvus_types.DescribeIndexProgressRequest(collection_name=collection_name, field_name=field_name)

    @classmethod
    def get_index_build_progress(cls, collection_name, field_name):
        return milvus_types.GetIndexBuildProgressRequest(collection_name=collection_name, field_name=field_name)

    @classmethod
    def get_index_state_request(cls, collection_name, field_name):
        return milvus_types.GetIndexStateRequest(collection_name=collection_name, field_name=field_name)

    @classmethod
    def load_collection(cls, db_name, collection_name):
        return milvus_types.LoadCollectionRequest(db_name=db_name, collection_name=collection_name)

    @classmethod
    def release_collection(cls, db_name, collection_name):
        return milvus_types.ReleaseCollectionRequest(db_name=db_name, collection_name=collection_name)

    @classmethod
    def load_partitions(cls, db_name, collection_name, partition_names):
        return milvus_types.LoadPartitionsRequest(db_name=db_name, collection_name=collection_name,
                                                  partition_names=partition_names)

    @classmethod
    def release_partitions(cls, db_name, collection_name, partition_names):
        return milvus_types.ReleasePartitionsRequest(db_name=db_name, collection_name=collection_name,
                                                    partition_names=partition_names)

    @classmethod
    def get_collection_stats_request(cls, collection_name):
        return milvus_types.GetCollectionStatisticsRequest(collection_name=collection_name)

    @classmethod
    def get_persistent_segment_info_request(cls, collection_name):
        return milvus_types.GetPersistentSegmentInfoRequest(collectionName=collection_name)

    @classmethod
    def get_query_segment_info_request(cls, collection_name):
        return milvus_types.GetQuerySegmentInfoRequest(collectionName=collection_name)

    @classmethod
    def flush_param(cls, collection_names):
        return milvus_types.FlushRequest(collection_names=collection_names)

    @classmethod
    def drop_index_request(cls, collection_name, field_name, index_name):
        return milvus_types.DropIndexRequest(db_name="", collection_name=collection_name, field_name=field_name,
                                             index_name=index_name)

    @classmethod
    def get_partition_stats_request(cls, collection_name, partition_name):
        return milvus_types.GetPartitionStatisticsRequest(db_name="", collection_name=collection_name,
                                                          partition_name=partition_name)

    @classmethod
    def dummy_request(cls, request_type):
        return milvus_types.DummyRequest(request_type=request_type)

    @classmethod
    def retrieve_request(cls, collection_name, ids, output_fields, partition_names):
        ids = schema_types.IDs(int_id=schema_types.LongArray(data=ids))
        return milvus_types.RetrieveRequest(db_name="",
                                            collection_name=collection_name,
                                            ids=ids,
                                            output_fields=output_fields,
                                            partition_names=partition_names)

    @classmethod
    def query_request(cls, collection_name, expr, output_fields, partition_names):
        return milvus_types.QueryRequest(db_name="",
                                         collection_name=collection_name,
                                         expr=expr,
                                         output_fields=output_fields,
                                         partition_names=partition_names
                                         )

    @classmethod
    def calc_distance_request(cls,  vectors_left, vectors_right, params):
        if vectors_left == None or not isinstance(vectors_left, dict):
            raise ParamError("Left vectors array is invalid")

        if vectors_right == None or not isinstance(vectors_right, dict):
            raise ParamError("Right vectors array is invalid")

        if params == None or not isinstance(params, dict):
            params = {"metric": "L2"}
        if "metric" not in params.keys() or len(params["metric"]) == 0:
            params["metric"] = "L2"

        request = milvus_types.CalcDistanceRequest()
        request.params.extend([common_types.KeyValuePair(key=str(key), value=str(value))
                                      for key, value in params.items()])

        _TYPE_IDS = "ids"
        _TYPE_FLOAT = "float_vectors"
        _TYPE_BIN = "bin_vectors"
        def extract_vectors(vectors, request_op):
            calc_type = ""
            dimension = 0
            if _TYPE_IDS in vectors.keys():
                if "collection" not in vectors.keys():
                    raise ParamError("Collection name not specified")
                if "field" not in vectors.keys():
                    raise ParamError("Vector field name not specified")
                ids = vectors.get(_TYPE_IDS)
                if (not isinstance(ids, list)) or len(ids) == 0:
                    raise ParamError("Vector id array is empty or not a list")

                calc_type  = _TYPE_IDS
                if isinstance(ids[0], str):
                    request_op.id_array.id_array.str_id.data.extend(ids)
                else:
                    request_op.id_array.id_array.int_id.data.extend(ids)
                request_op.id_array.collection_name = vectors["collection"]
                request_op.id_array.field_name = vectors["field"]
                if "partitions" in vectors.keys():
                    request_op.partition_names.extend(vectors.get("partitions"))
            elif _TYPE_FLOAT in vectors.keys():
                float_array = vectors.get(_TYPE_FLOAT)
                if (not isinstance(float_array, list)) or len(float_array) == 0:
                    raise ParamError("Float vector array is empty or not a list")
                calc_type = _TYPE_FLOAT
                all_floats = [f for vector in float_array for f in vector]
                request_op.data_array.dim = len(float_array[0])
                dimension = request_op.data_array.dim
                request_op.data_array.float_vector.data.extend(all_floats)
            elif _TYPE_BIN in vectors.keys():
                bin_array = vectors.get(_TYPE_BIN)
                if (not isinstance(bin_array, list)) or len(bin_array) == 0:
                    raise ParamError("Binary vector array is empty or not a list")
                calc_type = _TYPE_BIN
                request_op.data_array.dim = len(bin_array[0]) * 8
                if "dim" in params.keys():
                    request_op.data_array.dim = params["dim"]
                dimension = request_op.data_array.dim
                for bin in bin_array:
                    request_op.data_array.binary_vector += bin
            return calc_type, dimension

        type_left, dim_left = extract_vectors(vectors_left, request.op_left)
        type_right, dim_right = extract_vectors(vectors_right, request.op_right)

        if (type_left == _TYPE_FLOAT and type_right == _TYPE_BIN) or (type_left == _TYPE_BIN and type_right == _TYPE_FLOAT):
            raise ParamError("Cannot calculate distance between float vectors and binary vectors")

        if (type_left != _TYPE_IDS and type_right != _TYPE_IDS) and dim_left != dim_right:
            raise ParamError("Cannot calculate distance between vectors with different dimension")

        return request