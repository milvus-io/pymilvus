import copy
import ujson
import base64

from . import blob
from .configs import DefaultConfigs
from ..exceptions import ParamError, DataNotMatchException, ExceptionsMessage
from .check import check_pass_param
from .types import DataType, PlaceholderType
from .types import get_consistency_level
from .constants import DEFAULT_CONSISTENCY_LEVEL
from . import entity_helper

from ..grpc_gen import common_pb2 as common_types
from ..grpc_gen import schema_pb2 as schema_types
from ..grpc_gen import milvus_pb2 as milvus_types


class Prepare:

    @classmethod
    def create_collection_request(cls, collection_name, fields, shards_num=2, **kwargs):
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
                    if DataType(data_type) not in [DataType.INT64, DataType.VARCHAR]:
                        raise ParamError("int64 and varChar are the only supported types of primary key")
                    primary_field = field_name

                if auto_id:
                    auto_id_field = field_name

                field_schema.is_primary_key = is_primary
                field_schema.autoID = auto_id

                type_params = field.get('params')
                if isinstance(type_params, dict):
                    for tk, tv in type_params.items():
                        if tk in ["dim",]:
                            try:
                                int(tv)
                            except (TypeError, ValueError):
                                raise ParamError(f"invalid {tk}: {tv}") from None
                        if tk in [DefaultConfigs.MaxVarCharLengthKey,]:
                            try:
                                max_len = int(tv)
                                if max_len > DefaultConfigs.MaxVarCharLength:
                                    raise ParamError(f"{tk} {max_len} exceeds {DefaultConfigs.MaxVarCharLength}")
                            except (TypeError, ValueError):
                                raise ParamError(f"invalid {tk}: {tv}") from None
                            except ParamError as e:
                                raise e from None
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

        consistency_level = get_consistency_level(kwargs.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL))
        return milvus_types.CreateCollectionRequest(collection_name=collection_name,
                                                    schema=bytes(schema.SerializeToString()), shards_num=shards_num,
                                                    consistency_level=consistency_level)

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
    def show_collections_request(cls, collection_names=None):
        req = milvus_types.ShowCollectionsRequest()
        if collection_names:
            if not isinstance(collection_names, (list,)):
                raise ParamError(f"collection_names must be a list of strings, but got: {collection_names}")
            for collection_name in collection_names:
                check_pass_param(collection_name=collection_name)
            req.collection_names.extend(collection_names)
            req.type = milvus_types.ShowType.InMemory
        return req

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
    def show_partitions_request(cls, collection_name, partition_names=None, type_in_memory=False):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        req = milvus_types.ShowPartitionsRequest(collection_name=collection_name)
        if partition_names:
            if not isinstance(partition_names, (list,)):
                raise ParamError(f"partition_names must be a list of strings, but got: {partition_names}")
            for partition_name in partition_names:
                check_pass_param(partition_name=partition_name)
            req.partition_names.extend(partition_names)
        if type_in_memory is False:
            req.type = milvus_types.ShowType.All
        else:
            req.type = milvus_types.ShowType.InMemory
        return req

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
        try:
            for entity in entities:
                if row_num != 0 and row_num != len(entity.get("values")):
                    raise ParamError(f"row num misaligned.")
                row_num = len(entity.get("values"))
                field_data = entity_helper.entity_to_field_data(entity, fields_info[location[entity.get("name")]])
                insert_request.fields_data.append(field_data)
        except (TypeError, ValueError):
            raise DataNotMatchException(0, ExceptionsMessage.DataTypeInconsistent)

        insert_request.num_rows = row_num

        # insert_request.hash_keys won't be filled in client.
        # It will be filled in proxy.

        return insert_request

    @classmethod
    def delete_request(cls, collection_name, partition_name, expr):
        def check_str(instr, prefix):
            if instr is None:
                raise ParamError(prefix + " cannot be None")
            if not isinstance(instr, str):
                msg = prefix + " value {} is illegal"
                raise ParamError(msg.format(instr))
            if instr == "":
                raise ParamError(prefix + " cannot be empty")

        check_str(collection_name, "collection_name")
        if partition_name is not None and partition_name != "":
            check_str(partition_name, "partition_name")
        check_str(expr, "expr")

        request = milvus_types.DeleteRequest(collection_name=collection_name, expr=expr, partition_name=partition_name)
        return request

    @classmethod
    def _prepare_placeholders(cls, vectors, nq, tag, pl_type, is_binary, dimension=0):
        pl = common_types.PlaceholderValue(tag=tag)
        pl.type = pl_type
        for i in range(0, nq):
            if is_binary:
                if len(vectors[i]) * 8 != dimension:
                    raise ParamError("The dimension of query entities is different from schema")
                pl.values.append(blob.vectorBinaryToBytes(vectors[i]))
            else:
                if len(vectors[i]) != dimension:
                    raise ParamError("The dimension of query entities is different from schema")
                pl.values.append(blob.vectorFloatToBytes(vectors[i]))
        return pl


    @classmethod
    def search_request(cls, collection_name, query_entities, partition_names=None, fields=None, round_decimal=-1,
                       **kwargs):
        schema = kwargs.get("schema", None)
        fields_schema = schema.get("fields", None)  # list
        fields_name_locs = {fields_schema[loc]["name"]: loc
                            for loc in range(len(fields_schema))}

        if not isinstance(query_entities, (dict,)):
            raise ParamError("Invalid query format. 'query_entities' must be a dict")

        if fields is not None and not isinstance(fields, (list,)):
            raise ParamError("Invalid query format. 'fields' must be a list")

        request = milvus_types.SearchRequest(
            collection_name=collection_name,
            partition_names=partition_names,
            output_fields=fields,
            guarantee_timestamp=kwargs.get("guarantee_timestamp", 0),
        )

        duplicated_entities = copy.deepcopy(query_entities)
        vector_placeholders = dict()
        vector_names = dict()

        def extract_vectors_param(param, placeholders, names, round_decimal):
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
                        param["vector"][pk]["round_decimal"] = round_decimal

                    return
                else:
                    for _, v in param.items():
                        extract_vectors_param(v, placeholders, names, round_decimal)

            if isinstance(param, list):
                for item in param:
                    extract_vectors_param(item, placeholders, names, round_decimal)

        extract_vectors_param(duplicated_entities, vector_placeholders, vector_names, round_decimal)
        request.dsl = ujson.dumps(duplicated_entities)

        plg = common_types.PlaceholderGroup()
        for tag, vectors in vector_placeholders.items():
            if len(vectors) <= 0:
                continue
            pl = common_types.PlaceholderValue(tag=tag)

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
        plg_str = common_types.PlaceholderGroup.SerializeToString(plg)
        request.placeholder_group = plg_str

        return request

    @classmethod
    def search_requests_with_expr(cls, collection_name, data, anns_field, param, limit, expr=None, partition_names=None,
                                  output_fields=None, round_decimal=-1, **kwargs):
        # TODO Move this impl into server side
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

        tag = "$0"
        if anns_field not in fields_name_locs:
            raise ParamError(f"Field {anns_field} doesn't exist in schema")
        dimension = int(fields_schema[fields_name_locs[anns_field]]["params"].get("dim", 0))

        param_copy = copy.deepcopy(param)
        metric_type = param_copy.pop("metric_type", "L2")
        params = param_copy.pop("params", {})
        if not isinstance(params, dict):
            raise ParamError("Search params must be a dict")
        search_params = {"anns_field": anns_field, "topk": limit, "metric_type": metric_type, "params": params,
                         "round_decimal": round_decimal}

        def dump(v):
            if isinstance(v, dict):
                return ujson.dumps(v)
            return str(v)

        nq = len(data)
        pl = cls._prepare_placeholders(data, nq, tag, pl_type, is_binary, dimension)
        plg = common_types.PlaceholderGroup()
        plg.placeholders.append(pl)
        plg_str = common_types.PlaceholderGroup.SerializeToString(plg)
        request = milvus_types.SearchRequest(
            collection_name=collection_name,
            partition_names=partition_names,
            output_fields=output_fields,
            guarantee_timestamp=kwargs.get("guarantee_timestamp", 0),
            travel_timestamp=kwargs.get("travel_timestamp", 0),
            nq=nq,
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
    def create_alias_request(cls, collection_name, alias):
        return milvus_types.CreateAliasRequest(collection_name=collection_name, alias=alias)

    @classmethod
    def drop_alias_request(cls, alias):
        return milvus_types.DropAliasRequest(alias=alias)

    @classmethod
    def alter_alias_request(cls, collection_name, alias):
        return milvus_types.AlterAliasRequest(collection_name=collection_name, alias=alias)

    @classmethod
    def create_index__request(cls, collection_name, field_name, params, **kwargs):
        index_params = milvus_types.CreateIndexRequest(collection_name=collection_name, field_name=field_name,
                                                       index_name=kwargs.get("index_name", DefaultConfigs.IndexName))

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
    def get_index_build_progress(cls, collection_name: str, index_name: str):
        return milvus_types.GetIndexBuildProgressRequest(collection_name=collection_name, index_name=index_name)

    @classmethod
    def get_index_state_request(cls, collection_name: str, index_name: str):
        return milvus_types.GetIndexStateRequest(collection_name=collection_name, index_name=index_name)

    @classmethod
    def load_collection(cls, db_name, collection_name, replica_number):
        return milvus_types.LoadCollectionRequest(db_name=db_name, collection_name=collection_name,
                                                  replica_number=replica_number)

    @classmethod
    def release_collection(cls, db_name, collection_name):
        return milvus_types.ReleaseCollectionRequest(db_name=db_name, collection_name=collection_name)

    @classmethod
    def load_partitions(cls, db_name, collection_name, partition_names, replica_number):
        return milvus_types.LoadPartitionsRequest(db_name=db_name, collection_name=collection_name,
                                                  partition_names=partition_names,
                                                  replica_number=replica_number)

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
    def get_flush_state_request(cls, segment_ids):
        return milvus_types.GetFlushStateRequest(segmentIDs=segment_ids)

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
    def query_request(cls, collection_name, expr, output_fields, partition_names, guarantee_timestamp,
                      travel_timestamp):
        return milvus_types.QueryRequest(db_name="",
                                         collection_name=collection_name,
                                         expr=expr,
                                         output_fields=output_fields,
                                         partition_names=partition_names,
                                         guarantee_timestamp=guarantee_timestamp,
                                         travel_timestamp=travel_timestamp,
                                         )

    @classmethod
    def calc_distance_request(cls, vectors_left, vectors_right, params):
        if vectors_left is None or not isinstance(vectors_left, dict):
            raise ParamError("vectors_left value {} is illegal".format(vectors_left))

        if vectors_right is None or not isinstance(vectors_right, dict):
            raise ParamError("vectors_right value {} is illegal".format(vectors_right))

        def precheck_params(p):
            ret = p or {"metric": "L2"}
            if "metric" not in ret.keys():
                ret["metric"] = "L2"

            if (not isinstance(ret, dict)) or (not isinstance(ret["metric"], str)) or len(ret["metric"]) == 0:
                raise ParamError("params value {} is illegal".format(p))
            return ret

        precheck_params(params)

        request = milvus_types.CalcDistanceRequest()
        request.params.extend([common_types.KeyValuePair(key=str(key), value=str(value))
                               for key, value in params.items()])

        _TYPE_IDS = "ids"
        _TYPE_FLOAT = "float_vectors"
        _TYPE_BIN = "bin_vectors"

        def extract_vectors(vectors, request_op, is_left):
            prefix = "vectors_right"
            if is_left:
                prefix = "vectors_left"
            dimension = 0
            if _TYPE_IDS in vectors.keys():
                if "collection" not in vectors.keys():
                    raise ParamError("Collection name not specified")
                if "field" not in vectors.keys():
                    raise ParamError("Vector field name not specified")
                ids = vectors.get(_TYPE_IDS)
                if (not isinstance(ids, list)) or len(ids) == 0:
                    raise ParamError("Vector id array is empty or not a list")

                calc_type = _TYPE_IDS
                if isinstance(ids[0], str):
                    request_op.id_array.id_array.str_id.data.extend(ids)
                else:
                    request_op.id_array.id_array.int_id.data.extend(ids)
                request_op.id_array.collection_name = vectors["collection"]
                request_op.id_array.field_name = vectors["field"]
                if "partition" in vectors.keys():
                    request_op.id_array.partition_names.append(vectors.get("partition"))
            elif _TYPE_FLOAT in vectors.keys():
                float_array = vectors.get(_TYPE_FLOAT)
                if (not isinstance(float_array, list)) or len(float_array) == 0:
                    msg = prefix + " value {} is illegal"
                    raise ParamError(msg.format(vectors))
                calc_type = _TYPE_FLOAT
                all_floats = [f for vector in float_array for f in vector]
                request_op.data_array.dim = len(float_array[0])
                dimension = request_op.data_array.dim
                request_op.data_array.float_vector.data.extend(all_floats)
            elif _TYPE_BIN in vectors.keys():
                bin_array = vectors.get(_TYPE_BIN)
                if (not isinstance(bin_array, list)) or len(bin_array) == 0:
                    msg = prefix + " value {} is illegal"
                    raise ParamError(msg.format(vectors))
                calc_type = _TYPE_BIN
                request_op.data_array.dim = len(bin_array[0]) * 8
                if "dim" in params.keys():
                    request_op.data_array.dim = params["dim"]
                dimension = request_op.data_array.dim
                for bin in bin_array:
                    request_op.data_array.binary_vector += bin
            else:
                msg = prefix + " value {} is illegal"
                raise ParamError(msg.format(vectors))
            return calc_type, dimension

        type_left, dim_left = extract_vectors(vectors_left, request.op_left, True)
        type_right, dim_right = extract_vectors(vectors_right, request.op_right, False)

        if (type_left == _TYPE_FLOAT and type_right == _TYPE_BIN) or (
                type_left == _TYPE_BIN and type_right == _TYPE_FLOAT):
            raise ParamError("Cannot calculate distance between float vectors and binary vectors")

        if (type_left != _TYPE_IDS and type_right != _TYPE_IDS) and dim_left != dim_right:
            raise ParamError("Cannot calculate distance between vectors with different dimension")

        def postcheck_params(vtype, p):
            metrics_f = ["L2", "IP"]
            metrics_b = ["HAMMING", "TANIMOTO"]
            m = p["metric"].upper()
            if vtype == _TYPE_FLOAT and (m not in metrics_f):
                msg = "{} metric type is invalid for float vector"
                raise ParamError(msg.format(p["metric"]))
            if vtype == _TYPE_BIN and (m not in metrics_b):
                msg = "{} metric type is invalid for binary vector"
                raise ParamError(msg.format(p["metric"]))

        postcheck_params(type_left, params)
        postcheck_params(type_right, params)

        return request

    @classmethod
    def load_balance_request(cls, collection_name, src_node_id, dst_node_ids, sealed_segment_ids):
        request = milvus_types.LoadBalanceRequest(
            collectionName=collection_name,
            src_nodeID=src_node_id,
            dst_nodeIDs=dst_node_ids,
            sealed_segmentIDs=sealed_segment_ids,
        )
        return request

    @classmethod
    def manual_compaction(cls, collection_id, timetravel):
        if collection_id is None or not isinstance(collection_id, int):
            raise ParamError(f"collection_id value {collection_id} is illegal")

        if timetravel is None or not isinstance(timetravel, int):
            raise ParamError(f"timetravel value {timetravel} is illegal")

        request = milvus_types.ManualCompactionRequest()
        request.collectionID = collection_id
        request.timetravel = timetravel

        return request

    @classmethod
    def get_compaction_state(cls, compaction_id: int):
        if compaction_id is None or not isinstance(compaction_id, int):
            raise ParamError(f"compaction_id value {compaction_id} is illegal")

        request = milvus_types.GetCompactionStateRequest()
        request.compactionID = compaction_id
        return request

    @classmethod
    def get_compaction_state_with_plans(cls, compaction_id: int):
        if compaction_id is None or not isinstance(compaction_id, int):
            raise ParamError(f"compaction_id value {compaction_id} is illegal")

        request = milvus_types.GetCompactionPlansRequest()
        request.compactionID = compaction_id
        return request

    @classmethod
    def get_replicas(cls, collection_id: int):
        if collection_id is None or not isinstance(collection_id, int):
            raise ParamError(f"collection_id value {collection_id} is illegal")

        request = milvus_types.GetReplicasRequest(
            collectionID=collection_id,
            with_shard_nodes=True,
        )
        return request

    @classmethod
    def bulk_load(cls, collection_name: str, partition_name: str, is_row_based: bool, files: list, **kwargs):
        channel_names = kwargs.get("channel_names", None)
        req = milvus_types.ImportRequest(
            collection_name=collection_name,
            partition_name=partition_name,
            row_based=is_row_based,
            files=files,
        )
        if channel_names is not None:
            req.channel_names.extend(channel_names)

        for k, v in kwargs.items():
            if k in ("bucket",):
                kv_pair = common_types.KeyValuePair(key=str(k), value=str(v))
                req.options.append(kv_pair)

        return req

    @classmethod
    def get_bulk_load_state(cls, task_id):
        req = milvus_types.GetImportStateRequest(task=task_id)
        return req

    @classmethod
    def list_bulk_load_tasks(cls):
        return milvus_types.ListImportTasksRequest()

    @classmethod
    def create_credential_request(cls, user, password):
        check_pass_param(user=user, password=password)
        return milvus_types.CreateCredentialRequest(username=user, password=base64.b64encode(password.encode('utf-8')))

    @classmethod
    def update_credential_request(cls, user, old_password, new_password):
        check_pass_param(user=user)
        check_pass_param(password=old_password)
        check_pass_param(password=new_password)
        return milvus_types.UpdateCredentialRequest(username=user,
                                                    oldPassword=base64.b64encode(old_password.encode('utf-8')),
                                                    newPassword=base64.b64encode(new_password.encode('utf-8')),
                                                    )

    @classmethod
    def delete_credential_request(cls, user):
        if not isinstance(user, str):
            raise ParamError(f"invalid user {user}")
        return milvus_types.DeleteCredentialRequest(username=user)

    @classmethod
    def list_credential_request(cls):
        return milvus_types.ListCredUsersRequest()
