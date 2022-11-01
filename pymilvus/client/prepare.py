import copy
import base64
from typing import Dict, Iterable, Union

import ujson

from . import blob
from . import entity_helper
from .check import check_pass_param, is_legal_collection_properties
from .types import DataType, PlaceholderType, get_consistency_level
from .constants import DEFAULT_CONSISTENCY_LEVEL
from ..exceptions import ParamError, DataNotMatchException, ExceptionsMessage
from ..orm.schema import CollectionSchema

from ..grpc_gen import common_pb2 as common_types
from ..grpc_gen import schema_pb2 as schema_types
from ..grpc_gen import milvus_pb2 as milvus_types


class Prepare:
    @classmethod
    def create_collection_request(cls, collection_name: str, fields: Union[Dict[str, Iterable], CollectionSchema],
                                  shards_num=2, **kwargs) -> milvus_types.CreateCollectionRequest:
        """
        :type fields: Union(Dict[str, Iterable], CollectionSchema)
        :param fields: (Required)

            `{"fields": [
                    {"name": "A", "type": DataType.INT32}
                    {"name": "B", "type": DataType.INT64, "auto_id": True, "is_primary": True},
                    {"name": "C", "type": DataType.FLOAT},
                    {"name": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}}]
            }`

        :return: milvus_types.CreateCollectionRequest
        """
        if isinstance(fields, CollectionSchema):
            schema = cls.get_schema_from_collection_schema(collection_name, fields, shards_num, **kwargs)
        else:
            schema = cls.get_schema(collection_name, fields, shards_num, **kwargs)

        consistency_level = get_consistency_level(kwargs.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL))

        req = milvus_types.CreateCollectionRequest(collection_name=collection_name,
                                                    schema=bytes(schema.SerializeToString()),
                                                    shards_num=shards_num,
                                                    consistency_level=consistency_level)

        properties = kwargs.get("properties")
        if is_legal_collection_properties(properties):
            properties = [common_types.KeyValuePair(key=str(k), value=str(v)) for k, v in properties.items()]
            req.properties.extend(properties)

        return req

    @classmethod
    def get_schema_from_collection_schema(cls, collection_name: str, fields: CollectionSchema, shards_num=2, **kwargs) -> milvus_types.CreateCollectionRequest:
        coll_description = fields.description
        if not isinstance(coll_description, (str, bytes)):
            raise ParamError(message=f"description [{coll_description}] has type {type(coll_description).__name__},  but expected one of: bytes, str")

        schema = schema_types.CollectionSchema(name=collection_name,
                                               autoID=fields.auto_id,
                                               description=coll_description)
        for f in fields.fields:
            field_schema = schema_types.FieldSchema(name=f.name,
                                                    data_type=f.dtype,
                                                    description=f.description,
                                                    is_primary_key=f.is_primary,
                                                    autoID=f.auto_id)
            for k, v in f.params.items():
                kv_pair = common_types.KeyValuePair(key=str(k), value=str(v))
                field_schema.type_params.append(kv_pair)

            schema.fields.append(field_schema)
        return schema

    @classmethod
    def get_schema(cls, collection_name: str, fields: Dict[str, Iterable], shards_num=2, **kwargs) -> schema_types.CollectionSchema:
        if not isinstance(fields, dict):
            raise ParamError(message="Param fields must be a dict")

        all_fields = fields.get("fields")
        if all_fields is None:
            raise ParamError(message="Param fields must contain key 'fields'")
        if len(all_fields) == 0:
            raise ParamError(message="Param fields value cannot be empty")

        schema = schema_types.CollectionSchema(name=collection_name,
                                               autoID=False,
                                               description=fields.get('description', ''))

        primary_field = None
        auto_id_field = None
        for field in all_fields:
            field_name = field.get('name')
            if field_name is None:
                raise ParamError(message="You should specify the name of field!")

            data_type = field.get('type')
            if data_type is None:
                raise ParamError(message="You should specify the data type of field!")
            if not isinstance(data_type, (int, DataType)):
                raise ParamError(message="Field type must be of DataType!")

            is_primary = field.get("is_primary", False)
            if not isinstance(is_primary, bool):
                raise ParamError(message="is_primary must be boolean")
            if is_primary:
                if primary_field is not None:
                    raise ParamError(message="A collection should only have one primary field")
                if DataType(data_type) not in [DataType.INT64, DataType.VARCHAR]:
                    raise ParamError(message="int64 and varChar are the only supported types of primary key")
                primary_field = field_name

            auto_id = field.get('auto_id', False)
            if not isinstance(auto_id, bool):
                raise ParamError(message="auto_id must be boolean")
            if auto_id:
                if auto_id_field is not None:
                    raise ParamError(message="A collection should only have one autoID field")
                if DataType(data_type) != DataType.INT64:
                    raise ParamError(message="int64 is the only supported type of automatic generated id")
                auto_id_field = field_name

            field_schema = schema_types.FieldSchema(name=field_name,
                                                    data_type=data_type,
                                                    description=field.get('description', ''),
                                                    is_primary_key=is_primary,
                                                    autoID=auto_id)

            type_params = field.get('params', {})
            if not isinstance(type_params, dict):
                raise ParamError(message="params should be dictionary type")
            kvs = [common_types.KeyValuePair(key=str(k), value=str(v)) for k, v in type_params.items()]
            field_schema.type_params.extend(kvs)

            schema.fields.append(field_schema)
        return schema

    @classmethod
    def drop_collection_request(cls, collection_name):
        return milvus_types.DropCollectionRequest(collection_name=collection_name)

    @classmethod
    # TODO remove
    def has_collection_request(cls, collection_name):
        return milvus_types.HasCollectionRequest(collection_name=collection_name)

    @classmethod
    def describe_collection_request(cls, collection_name):
        return milvus_types.DescribeCollectionRequest(collection_name=collection_name)

    @classmethod
    def alter_collection_request(cls, collection_name, properties):
        kvs = []
        for k in properties:
            kv = common_types.KeyValuePair(key=k, value=str(properties[k]))
            kvs.append(kv)

        return milvus_types.AlterCollectionRequest(collection_name=collection_name, properties=kvs)

    @classmethod
    def collection_stats_request(cls, collection_name):
        return milvus_types.CollectionStatsRequest(collection_name=collection_name)

    @classmethod
    def show_collections_request(cls, collection_names=None):
        req = milvus_types.ShowCollectionsRequest()
        if collection_names:
            if not isinstance(collection_names, (list,)):
                raise ParamError(message=f"collection_names must be a list of strings, but got: {collection_names}")
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
                raise ParamError(message=f"partition_names must be a list of strings, but got: {partition_names}")
            for partition_name in partition_names:
                check_pass_param(partition_name=partition_name)
            req.partition_names.extend(partition_names)
        if type_in_memory is False:
            req.type = milvus_types.ShowType.All
        else:
            req.type = milvus_types.ShowType.InMemory
        return req

    @classmethod
    def get_loading_progress(cls, collection_name, partition_names=None):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        req = milvus_types.GetLoadingProgressRequest(collection_name=collection_name)
        if partition_names:
            req.partition_names.extend(partition_names)
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
            raise ParamError(message="collection_name must be of str type")
        if not isinstance(partition_name, str):
            raise ParamError(message="partition_name must be of str type")
        return milvus_types.PartitionName(collection_name=collection_name,
                                          tag=partition_name)

    @classmethod
    def batch_insert_param(cls, collection_name, entities, partition_name, fields_info=None, **kwargs):
        # insert_request.hash_keys won't be filled in client. It will be filled in proxy.

        tag = partition_name or "_default" # should here?
        insert_request = milvus_types.InsertRequest(collection_name=collection_name, partition_name=tag)

        for entity in entities:
            if not entity.get("name", None) or not entity.get("values", None) or not entity.get("type", None):
                raise ParamError(message="Missing param in entities, a field must have type, name and values")
        if not fields_info:
            raise ParamError(message="Missing collection meta to validate entities")

        location, primary_key_loc, auto_id_loc = {}, None, None
        for i, field in enumerate(fields_info):
            if field.get("is_primary", False):
                primary_key_loc = i

            if field.get("auto_id", False):
                auto_id_loc = i
                continue

            match_flag = False
            field_name = field["name"]
            field_type = field["type"]

            for j, entity in enumerate(entities):
                entity_name, entity_type = entity["name"], entity["type"]

                if field_name == entity_name:
                    if field_type != entity_type:
                        raise ParamError(message=f"Collection field type is {field_type}"
                                         f", but entities field type is {entity_type}")

                    entity_dim, field_dim = 0, 0
                    if entity_type in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                        field_dim = field["params"]["dim"]
                        entity_dim = len(entity["values"][0])

                    if entity_type in [DataType.FLOAT_VECTOR, ] and entity_dim != field_dim:
                        raise ParamError(message=f"Collection field dim is {field_dim}"
                                         f", but entities field dim is {entity_dim}")

                    if entity_type in [DataType.BINARY_VECTOR, ] and entity_dim * 8 != field_dim:
                        raise ParamError(message=f"Collection field dim is {field_dim}"
                                         f", but entities field dim is {entity_dim * 8}")

                    location[field["name"]] = j
                    match_flag = True
                    break

            if not match_flag:
                raise ParamError(message=f"Field {field['name']} don't match in entities")

        # though impossible from sdk
        if primary_key_loc is None:
            raise ParamError(message="primary key not found")

        if auto_id_loc is None and len(entities) != len(fields_info):
            raise ParamError(message=f"number of fields: {len(fields_info)}, number of entities: {len(entities)}")

        if auto_id_loc is not None and len(entities) + 1 != len(fields_info):
            raise ParamError(message=f"number of fields: {len(fields_info)}, number of entities: {len(entities)}")

        row_num = 0
        try:
            for entity in entities:
                current = len(entity.get("values"))
                if row_num not in (0, current):
                    raise ParamError(message="row num misaligned current[{current}]!= previous[{row_num}]")
                row_num = current
                field_data = entity_helper.entity_to_field_data(entity, fields_info[location[entity.get("name")]])
                insert_request.fields_data.append(field_data)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(message=ExceptionsMessage.DataTypeInconsistent) from e

        insert_request.num_rows = row_num

        return insert_request

    @classmethod
    def delete_request(cls, collection_name, partition_name, expr):
        def check_str(instr, prefix):
            if instr is None:
                raise ParamError(message=f"{prefix} cannot be None")
            if not isinstance(instr, str):
                raise ParamError(message=f"{prefix} value {instr} is illegal")
            if instr == "":
                raise ParamError(message=f"{prefix} cannot be empty")

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
                    raise ParamError(message=f"The dimension of query entities[{vectors[i]*8}] is different from schema [{dimension}]")
                pl.values.append(blob.vectorBinaryToBytes(vectors[i]))
            else:
                if len(vectors[i]) != dimension:
                    raise ParamError(message=f"The dimension of query entities[{vectors[i]*8}] is different from schema [{dimension}]")
                pl.values.append(blob.vectorFloatToBytes(vectors[i]))
        return pl

    @classmethod
    def search_request(cls, collection_name, query_entities, partition_names=None, fields=None, round_decimal=-1, **kwargs):
        schema = kwargs.get("schema", None)
        fields_schema = schema.get("fields", None)  # list
        fields_name_locs = {fields_schema[loc]["name"]: loc
                            for loc in range(len(fields_schema))}

        if not isinstance(query_entities, (dict,)):
            raise ParamError(message="Invalid query format. 'query_entities' must be a dict")

        if fields is not None and not isinstance(fields, (list,)):
            raise ParamError(message="Invalid query format. 'fields' must be a list")

        request = milvus_types.SearchRequest(
            collection_name=collection_name,
            partition_names=partition_names,
            output_fields=fields,
            guarantee_timestamp=kwargs.get("guarantee_timestamp", 0),
        )

        duplicated_entities = copy.deepcopy(query_entities)
        vector_placeholders = {}
        vector_names = {}

        def extract_vectors_param(param, placeholders, names, round_decimal):
            if not isinstance(param, (dict, list)):
                return

            if isinstance(param, dict):
                if "vector" in param:
                    # TODO: Here may not replace ph
                    ph = "$" + str(len(placeholders))

                    for pk, pv in param["vector"].items():
                        if "query" not in pv:
                            raise ParamError(message="param vector must contain 'query'")
                        placeholders[ph] = pv["query"]
                        names[ph] = pk
                        param["vector"][pk]["query"] = ph
                        param["vector"][pk]["round_decimal"] = round_decimal
                    return

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
                raise ParamError(message=f"Field {fname} doesn't exist in schema")
            dimension = int(fields_schema[fields_name_locs[fname]]["params"].get("dim", 0))

            if isinstance(vectors[0], bytes):
                pl.type = PlaceholderType.BinaryVector
                for vector in vectors:
                    if dimension != len(vector) * 8:
                        raise ParamError(message="The dimension of query vector is different from schema")
                    pl.values.append(blob.vectorBinaryToBytes(vector))
            else:
                pl.type = PlaceholderType.FloatVector
                for vector in vectors:
                    if dimension != len(vector):
                        raise ParamError(message="The dimension of query vector is different from schema")
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
        fields_name_locs = {fields_schema[loc]["name"]: loc for loc in range(len(fields_schema))}

        requests = []
        if len(data) <= 0:
            return requests

        if isinstance(data[0], bytes):
            is_binary = True
            pl_type = PlaceholderType.BinaryVector
        else:
            is_binary = False
            pl_type = PlaceholderType.FloatVector

        if anns_field not in fields_name_locs:
            raise ParamError(message=f"Field {anns_field} doesn't exist in schema")
        dimension = int(fields_schema[fields_name_locs[anns_field]]["params"].get("dim", 0))

        params = param.get("params", {})
        if not isinstance(params, dict):
            raise ParamError(message=f"Search params must be a dict, got {type(params)}")
        search_params = {
            "anns_field": anns_field,
            "topk": limit,
            "metric_type": param.get("metric_type", "L2"),
            "params": params,
            "round_decimal": round_decimal,
            "offset": param.get("offset", 0),
        }

        def dump(v):
            if isinstance(v, dict):
                return ujson.dumps(v)
            return str(v)

        nq = len(data)
        tag = "$0"
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
    def create_index_request(cls, collection_name, field_name, params, **kwargs):
        index_params = milvus_types.CreateIndexRequest(collection_name=collection_name, field_name=field_name,
                                                       index_name=kwargs.get("index_name", ""))

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
                        raise ParamError(message="dim must be of int!")
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
    def query_request(cls, collection_name, expr, output_fields, partition_names, **kwargs):
        req = milvus_types.QueryRequest(db_name="",
                                        collection_name=collection_name,
                                        expr=expr,
                                        output_fields=output_fields,
                                        partition_names=partition_names,
                                        guarantee_timestamp=kwargs.get("guarantee_timestamp", 0),
                                        travel_timestamp=kwargs.get("travel_timestamp", 0),
                                        )

        limit = kwargs.get("limit", None)
        if limit is not None:
            req.query_params.append(common_types.KeyValuePair(key="limit", value=str(limit)))

        offset = kwargs.get("offset", None)
        if offset is not None:
            req.query_params.append(common_types.KeyValuePair(key="offset", value=str(offset)))

        return req

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
            raise ParamError(message=f"collection_id value {collection_id} is illegal")

        if timetravel is None or not isinstance(timetravel, int):
            raise ParamError(message=f"timetravel value {timetravel} is illegal")

        request = milvus_types.ManualCompactionRequest()
        request.collectionID = collection_id
        request.timetravel = timetravel

        return request

    @classmethod
    def get_compaction_state(cls, compaction_id: int):
        if compaction_id is None or not isinstance(compaction_id, int):
            raise ParamError(message=f"compaction_id value {compaction_id} is illegal")

        request = milvus_types.GetCompactionStateRequest()
        request.compactionID = compaction_id
        return request

    @classmethod
    def get_compaction_state_with_plans(cls, compaction_id: int):
        if compaction_id is None or not isinstance(compaction_id, int):
            raise ParamError(message=f"compaction_id value {compaction_id} is illegal")

        request = milvus_types.GetCompactionPlansRequest()
        request.compactionID = compaction_id
        return request

    @classmethod
    def get_replicas(cls, collection_id: int):
        if collection_id is None or not isinstance(collection_id, int):
            raise ParamError(message=f"collection_id value {collection_id} is illegal")

        request = milvus_types.GetReplicasRequest(
            collectionID=collection_id,
            with_shard_nodes=True,
        )
        return request

    @classmethod
    def do_bulk_insert(cls, collection_name: str, partition_name: str, files: list, **kwargs):
        channel_names = kwargs.get("channel_names", None)
        req = milvus_types.ImportRequest(
            collection_name=collection_name,
            partition_name=partition_name,
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
    def get_bulk_insert_state(cls, task_id):
        if task_id is None or not isinstance(task_id, int):
            raise ParamError(f"task_id value {task_id} is not an integer")

        req = milvus_types.GetImportStateRequest(task=task_id)
        return req

    @classmethod
    def list_bulk_insert_tasks(cls, limit, collection_name):
        if limit is None or not isinstance(limit, int):
            raise ParamError(f"limit value {limit} is not an integer")

        request = milvus_types.ListImportTasksRequest(
            collection_name=collection_name,
            limit=limit,
        )
        return request

    @classmethod
    def create_user_request(cls, user, password):
        check_pass_param(user=user, password=password)
        return milvus_types.CreateCredentialRequest(username=user, password=base64.b64encode(password.encode('utf-8')))

    @classmethod
    def update_password_request(cls, user, old_password, new_password):
        check_pass_param(user=user)
        check_pass_param(password=old_password)
        check_pass_param(password=new_password)
        return milvus_types.UpdateCredentialRequest(username=user,
                                                    oldPassword=base64.b64encode(old_password.encode('utf-8')),
                                                    newPassword=base64.b64encode(new_password.encode('utf-8')),
                                                    )

    @classmethod
    def delete_user_request(cls, user):
        if not isinstance(user, str):
            raise ParamError(message=f"invalid user {user}")
        return milvus_types.DeleteCredentialRequest(username=user)

    @classmethod
    def list_usernames_request(cls):
        return milvus_types.ListCredUsersRequest()

    @classmethod
    def create_role_request(cls, role_name):
        check_pass_param(role_name=role_name)
        return milvus_types.CreateRoleRequest(entity=milvus_types.RoleEntity(name=role_name))

    @classmethod
    def drop_role_request(cls, role_name):
        check_pass_param(role_name=role_name)
        return milvus_types.DropRoleRequest(role_name=role_name)

    @classmethod
    def operate_user_role_request(cls, username, role_name, operate_user_role_type):
        check_pass_param(user=username)
        check_pass_param(role_name=role_name)
        check_pass_param(operate_user_role_type=operate_user_role_type)
        return milvus_types.OperateUserRoleRequest(username=username, role_name=role_name, type=operate_user_role_type)

    @classmethod
    def select_role_request(cls, role_name, include_user_info):
        if role_name:
            check_pass_param(role_name=role_name)
        check_pass_param(include_user_info=include_user_info)
        return milvus_types.SelectRoleRequest(role=milvus_types.RoleEntity(name=role_name) if role_name else None,
                                              include_user_info=include_user_info)

    @classmethod
    def select_user_request(cls, username, include_role_info):
        if username:
            check_pass_param(user=username)
        check_pass_param(include_role_info=include_role_info)
        return milvus_types.SelectUserRequest(user=milvus_types.UserEntity(name=username) if username else None,
                                              include_role_info=include_role_info)

    @classmethod
    def operate_privilege_request(cls, role_name, object, object_name, privilege, operate_privilege_type):
        check_pass_param(role_name=role_name)
        check_pass_param(object=object)
        check_pass_param(object_name=object_name)
        check_pass_param(privilege=privilege)
        check_pass_param(operate_privilege_type=operate_privilege_type)
        return milvus_types.OperatePrivilegeRequest(
            entity=milvus_types.GrantEntity(role=milvus_types.RoleEntity(name=role_name),
                                            object=milvus_types.ObjectEntity(name=object),
                                            object_name=object_name,
                                            grantor=milvus_types.GrantorEntity(
                                                privilege=milvus_types.PrivilegeEntity(name=privilege))),
            type=operate_privilege_type)

    @classmethod
    def select_grant_request(cls, role_name, object, object_name):
        check_pass_param(role_name=role_name)
        if object:
            check_pass_param(object=object)
        if object_name:
            check_pass_param(object_name=object_name)
        return milvus_types.SelectGrantRequest(
            entity=milvus_types.GrantEntity(role=milvus_types.RoleEntity(name=role_name),
                                            object=milvus_types.ObjectEntity(name=object) if object else None,
                                            object_name=object_name if object_name else None))
