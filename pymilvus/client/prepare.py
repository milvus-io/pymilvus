import base64
import datetime
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import orjson

from pymilvus.exceptions import DataNotMatchException, ExceptionsMessage, ParamError
from pymilvus.grpc_gen import common_pb2
from pymilvus.grpc_gen import common_pb2 as common_types
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.grpc_gen import schema_pb2 as schema_types
from pymilvus.orm.schema import (
    CollectionSchema,
    FieldSchema,
    Function,
    FunctionScore,
    Highlighter,
    isVectorDataType,
)
from pymilvus.orm.types import infer_dtype_by_scalar_data
from pymilvus.settings import Config

from . import __version__, blob, check, entity_helper, ts_utils, utils
from .abstract import BaseRanker
from .check import check_pass_param, is_legal_collection_properties
from .constants import (
    COLLECTION_ID,
    DEFAULT_CONSISTENCY_LEVEL,
    DYNAMIC_FIELD_NAME,
    GROUP_BY_FIELD,
    GROUP_SIZE,
    HINTS,
    IS_EMBEDDING_LIST,
    ITER_SEARCH_BATCH_SIZE_KEY,
    ITER_SEARCH_ID_KEY,
    ITER_SEARCH_LAST_BOUND_KEY,
    ITER_SEARCH_V2_KEY,
    ITERATOR_FIELD,
    JSON_PATH,
    JSON_TYPE,
    PAGE_RETAIN_ORDER_FIELD,
    RANK_GROUP_SCORER,
    REDUCE_STOP_FOR_BEST,
    STRICT_CAST,
    STRICT_GROUP_SIZE,
)
from .entity_helper import convert_to_array, convert_to_array_of_vector
from .types import (
    DataType,
    PlaceholderType,
    ResourceGroupConfig,
    get_consistency_level,
)
from .utils import get_params, traverse_info, traverse_upsert_info


class Prepare:
    @classmethod
    def create_collection_request(
        cls,
        collection_name: str,
        fields: Union[Dict[str, Iterable], CollectionSchema],
        **kwargs,
    ) -> milvus_types.CreateCollectionRequest:
        """
        Args:
            fields (Union(Dict[str, Iterable], CollectionSchema)).

                {"fields": [
                        {"name": "A", "type": DataType.INT32}
                        {"name": "B", "type": DataType.INT64, "auto_id": True, "is_primary": True},
                        {"name": "C", "type": DataType.FLOAT},
                        {"name": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}}]
                }

        Returns:
            milvus_types.CreateCollectionRequest
        """

        if isinstance(fields, CollectionSchema):
            schema = cls.get_schema_from_collection_schema(collection_name, fields)
        else:
            schema = cls.get_schema(collection_name, fields, **kwargs)

        consistency_level = get_consistency_level(
            kwargs.get("consistency_level", DEFAULT_CONSISTENCY_LEVEL)
        )

        req = milvus_types.CreateCollectionRequest(
            collection_name=collection_name,
            schema=bytes(schema.SerializeToString()),
            consistency_level=consistency_level,
        )

        properties = kwargs.get("properties")
        if is_legal_collection_properties(properties):
            properties = [
                common_types.KeyValuePair(key=str(k), value=str(v)) for k, v in properties.items()
            ]
            req.properties.extend(properties)

        same_key = set(kwargs.keys()).intersection({"num_shards", "shards_num"})
        if len(same_key) > 0:
            if len(same_key) > 1:
                msg = "got both num_shards and shards_num in kwargs, expected only one of them"
                raise ParamError(message=msg)

            num_shards = kwargs[next(iter(same_key))]
            if not isinstance(num_shards, int):
                msg = f"invalid num_shards type, got {type(num_shards)}, expected int"
                raise ParamError(message=msg)
            req.shards_num = num_shards

        num_partitions = kwargs.get("num_partitions")
        if num_partitions is not None:
            if not isinstance(num_partitions, int) or isinstance(num_partitions, bool):
                msg = f"invalid num_partitions type, got {type(num_partitions)}, expected int"
                raise ParamError(message=msg)
            if num_partitions < 1:
                msg = f"The specified num_partitions should be greater than or equal to 1, got {num_partitions}"
                raise ParamError(message=msg)
            req.num_partitions = num_partitions

        return req

    @classmethod
    def get_schema_from_collection_schema(
        cls,
        collection_name: str,
        fields: CollectionSchema,
    ) -> schema_types.CollectionSchema:
        coll_description = fields.description
        if not isinstance(coll_description, (str, bytes)):
            msg = (
                f"description [{coll_description}] has type {type(coll_description).__name__}, "
                "but expected one of: bytes, str"
            )
            raise ParamError(message=msg)

        schema = schema_types.CollectionSchema(
            name=collection_name,
            autoID=fields.auto_id,
            description=coll_description,
            enable_dynamic_field=fields.enable_dynamic_field,
            enable_namespace=fields.enable_namespace,
        )
        for f in fields.fields:
            field_schema = schema_types.FieldSchema(
                name=f.name,
                data_type=f.dtype,
                description=f.description,
                is_primary_key=f.is_primary,
                default_value=f.default_value,
                nullable=f.nullable,
                autoID=f.auto_id,
                is_partition_key=f.is_partition_key,
                is_dynamic=f.is_dynamic,
                element_type=f.element_type,
                is_clustering_key=f.is_clustering_key,
                is_function_output=f.is_function_output,
            )
            for k, v in f.params.items():
                kv_pair = common_types.KeyValuePair(
                    key=str(k) if k != "mmap_enabled" else "mmap.enabled",
                    value=(
                        orjson.dumps(v).decode(Config.EncodeProtocol)
                        if not isinstance(v, str)
                        else str(v)
                    ),
                )
                field_schema.type_params.append(kv_pair)

            schema.fields.append(field_schema)

        for struct in fields.struct_fields:
            # Validate that max_capacity is set
            if struct.max_capacity is None:
                raise ParamError(message=f"max_capacity not set for struct field: {struct.name}")

            struct_schema = schema_types.StructArrayFieldSchema(
                name=struct.name,
                fields=[],
                description=struct.description,
            )

            if struct.params:
                for k, v in struct.params.items():
                    kv_pair = common_types.KeyValuePair(
                        key=str(k) if k != "mmap_enabled" else "mmap.enabled",
                        value=(
                            orjson.dumps(v).decode(Config.EncodeProtocol)
                            if not isinstance(v, str)
                            else str(v)
                        ),
                    )
                    struct_schema.type_params.append(kv_pair)

            for f in struct.fields:
                # Convert struct field types to backend representation
                # As struct itself only support array type, so all it's fields are array type
                # internally
                # So we need to convert the fields to array types
                actual_dtype = f.dtype
                actual_element_type = None

                # Convert to appropriate array type
                if isVectorDataType(f.dtype):
                    actual_dtype = DataType._ARRAY_OF_VECTOR
                    actual_element_type = f.dtype
                else:
                    actual_dtype = DataType.ARRAY
                    actual_element_type = f.dtype

                field_schema = schema_types.FieldSchema(
                    name=f.name,
                    data_type=actual_dtype,
                    description=f.description,
                    is_primary_key=f.is_primary,
                    default_value=f.default_value,
                    nullable=f.nullable,
                    autoID=f.auto_id,
                    is_partition_key=f.is_partition_key,
                    is_dynamic=f.is_dynamic,
                    element_type=actual_element_type,
                    is_clustering_key=f.is_clustering_key,
                    is_function_output=f.is_function_output,
                )

                # Copy field params and add max_capacity from struct_schema
                field_params = dict(f.params) if f.params else {}
                # max_capacity is required for struct fields
                field_params["max_capacity"] = struct.max_capacity

                for k, v in field_params.items():
                    kv_pair = common_types.KeyValuePair(
                        key=str(k) if k != "mmap_enabled" else "mmap.enabled", value=json.dumps(v)
                    )
                    field_schema.type_params.append(kv_pair)
                struct_schema.fields.append(field_schema)

            schema.struct_array_fields.append(struct_schema)

        for f in fields.functions:
            function_schema = cls.convert_function_to_function_schema(f)
            schema.functions.append(function_schema)

        return schema

    @staticmethod
    def get_field_schema(
        field: Dict,
        primary_field: Optional[str] = None,
        auto_id_field: Optional[str] = None,
    ) -> (schema_types.FieldSchema, Optional[str], Optional[str]):
        field_name = field.get("name")
        if field_name is None:
            raise ParamError(message="You should specify the name of field!")

        data_type = field.get("type")
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
                msg = "int64 and varChar are the only supported types of primary key"
                raise ParamError(message=msg)
            primary_field = field_name

        nullable = field.get("nullable", False)
        if not isinstance(nullable, bool):
            raise ParamError(message="nullable must be boolean")

        auto_id = field.get("auto_id", False)
        if not isinstance(auto_id, bool):
            raise ParamError(message="auto_id must be boolean")
        if auto_id:
            if auto_id_field is not None:
                raise ParamError(message="A collection should only have one autoID field")
            if DataType(data_type) != DataType.INT64:
                msg = "int64 is the only supported type of automatic generated id"
                raise ParamError(message=msg)
            auto_id_field = field_name

        field_schema = schema_types.FieldSchema(
            name=field_name,
            data_type=data_type,
            description=field.get("description", ""),
            is_primary_key=is_primary,
            autoID=auto_id,
            is_partition_key=field.get("is_partition_key", False),
            is_clustering_key=field.get("is_clustering_key", False),
            nullable=nullable,
            default_value=field.get("default_value"),
            element_type=field.get("element_type"),
        )

        type_params = field.get("params", {})
        if not isinstance(type_params, dict):
            raise ParamError(message="params should be dictionary type")
        kvs = [
            common_types.KeyValuePair(
                key=str(k) if k != "mmap_enabled" else "mmap.enabled",
                value=str(v),
            )
            for k, v in type_params.items()
        ]
        field_schema.type_params.extend(kvs)

        return field_schema, primary_field, auto_id_field

    @classmethod
    def get_schema(
        cls,
        collection_name: str,
        fields: Dict[str, Iterable],
        **kwargs,
    ) -> schema_types.CollectionSchema:
        if not isinstance(fields, dict):
            raise ParamError(message="Param fields must be a dict")

        all_fields = fields.get("fields")
        if all_fields is None:
            raise ParamError(message="Param fields must contain key 'fields'")
        if len(all_fields) == 0:
            raise ParamError(message="Param fields value cannot be empty")

        enable_dynamic_field = kwargs.get("enable_dynamic_field", False)
        if "enable_dynamic_field" in fields:
            enable_dynamic_field = fields["enable_dynamic_field"]

        enable_namespace = kwargs.get("enable_namespace", False)
        if "enable_namespace" in fields:
            enable_namespace = fields["enable_namespace"]

        schema = schema_types.CollectionSchema(
            name=collection_name,
            autoID=False,
            description=fields.get("description", ""),
            enable_dynamic_field=enable_dynamic_field,
            enable_namespace=enable_namespace,
        )

        primary_field, auto_id_field = None, None
        for field in all_fields:
            (field_schema, primary_field, auto_id_field) = cls.get_field_schema(
                field, primary_field, auto_id_field
            )
            schema.fields.append(field_schema)
        return schema

    @classmethod
    def drop_collection_request(cls, collection_name: str) -> milvus_types.DropCollectionRequest:
        return milvus_types.DropCollectionRequest(collection_name=collection_name)

    @classmethod
    def drop_collection_function_request(
        cls, collection_name: str, function_name: str
    ) -> milvus_types.DropCollectionFunctionRequest:
        return milvus_types.DropCollectionFunctionRequest(
            collection_name=collection_name, function_name=function_name
        )

    @classmethod
    def add_collection_function_request(
        cls, collection_name: str, f: Function
    ) -> milvus_types.AddCollectionFunctionRequest:
        function_schema = cls.convert_function_to_function_schema(f)
        return milvus_types.AddCollectionFunctionRequest(
            collection_name=collection_name, functionSchema=function_schema
        )

    @classmethod
    def alter_collection_function_request(
        cls, collection_name: str, function_name: str, f: Function
    ) -> milvus_types.AlterCollectionFunctionRequest:
        function_schema = cls.convert_function_to_function_schema(f)
        return milvus_types.AlterCollectionFunctionRequest(
            collection_name=collection_name,
            function_name=function_name,
            functionSchema=function_schema,
        )

    @classmethod
    def add_collection_field_request(
        cls,
        collection_name: str,
        field_schema: FieldSchema,
    ) -> milvus_types.AddCollectionFieldRequest:
        (field_schema_proto, _, _) = cls.get_field_schema(field=field_schema.to_dict())
        return milvus_types.AddCollectionFieldRequest(
            collection_name=collection_name,
            schema=bytes(field_schema_proto.SerializeToString()),
        )

    @classmethod
    def describe_collection_request(
        cls,
        collection_name: str,
    ) -> milvus_types.DescribeCollectionRequest:
        return milvus_types.DescribeCollectionRequest(collection_name=collection_name)

    @classmethod
    def alter_collection_request(
        cls,
        collection_name: str,
        properties: Optional[Dict] = None,
        delete_keys: Optional[List[str]] = None,
    ) -> milvus_types.AlterCollectionRequest:
        kvs = []
        if properties:
            kvs = [common_types.KeyValuePair(key=k, value=str(v)) for k, v in properties.items()]

        return milvus_types.AlterCollectionRequest(
            collection_name=collection_name, properties=kvs, delete_keys=delete_keys
        )

    @classmethod
    def alter_collection_field_request(
        cls, collection_name: str, field_name: str, field_param: Dict
    ) -> milvus_types.AlterCollectionFieldRequest:
        kvs = []
        if field_param:
            kvs = [common_types.KeyValuePair(key=k, value=str(v)) for k, v in field_param.items()]
        return milvus_types.AlterCollectionFieldRequest(
            collection_name=collection_name, field_name=field_name, properties=kvs
        )

    @classmethod
    def collection_stats_request(cls, collection_name: str):
        return milvus_types.CollectionStatsRequest(collection_name=collection_name)

    @classmethod
    def show_collections_request(cls, collection_names: Optional[List[str]] = None):
        req = milvus_types.ShowCollectionsRequest()
        if collection_names:
            if not isinstance(collection_names, (list,)):
                msg = f"collection_names must be a list of strings, but got: {collection_names}"
                raise ParamError(message=msg)
            for collection_name in collection_names:
                check_pass_param(collection_name=collection_name)
            req.collection_names.extend(collection_names)
            req.type = milvus_types.ShowType.InMemory
        return req

    @classmethod
    def rename_collections_request(cls, old_name: str, new_name: str, new_db_name: str):
        return milvus_types.RenameCollectionRequest(
            oldName=old_name, newName=new_name, newDBName=new_db_name
        )

    @classmethod
    def create_partition_request(cls, collection_name: str, partition_name: str):
        return milvus_types.CreatePartitionRequest(
            collection_name=collection_name, partition_name=partition_name
        )

    @classmethod
    def drop_partition_request(cls, collection_name: str, partition_name: str):
        return milvus_types.DropPartitionRequest(
            collection_name=collection_name, partition_name=partition_name
        )

    @classmethod
    def has_partition_request(cls, collection_name: str, partition_name: str):
        return milvus_types.HasPartitionRequest(
            collection_name=collection_name, partition_name=partition_name
        )

    @classmethod
    def partition_stats_request(cls, collection_name: str, partition_name: str):
        return milvus_types.PartitionStatsRequest(
            collection_name=collection_name, partition_name=partition_name
        )

    @classmethod
    def show_partitions_request(
        cls,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        type_in_memory: bool = False,
    ):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        req = milvus_types.ShowPartitionsRequest(collection_name=collection_name)
        if partition_names:
            if not isinstance(partition_names, (list,)):
                msg = f"partition_names must be a list of strings, but got: {partition_names}"
                raise ParamError(message=msg)
            for partition_name in partition_names:
                check_pass_param(partition_name=partition_name)
            req.partition_names.extend(partition_names)
        if type_in_memory is False:
            req.type = milvus_types.ShowType.All
        else:
            req.type = milvus_types.ShowType.InMemory
        return req

    @classmethod
    def get_loading_progress(
        cls, collection_name: str, partition_names: Optional[List[str]] = None
    ):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        req = milvus_types.GetLoadingProgressRequest(collection_name=collection_name)
        if partition_names:
            req.partition_names.extend(partition_names)
        return req

    @classmethod
    def get_load_state(cls, collection_name: str, partition_names: Optional[List[str]] = None):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        req = milvus_types.GetLoadStateRequest(collection_name=collection_name)
        if partition_names:
            req.partition_names.extend(partition_names)
        return req

    @classmethod
    def empty(cls):
        msg = "no empty request later"
        raise DeprecationWarning(msg)

    @classmethod
    def register_link_request(cls):
        return milvus_types.RegisterLinkRequest()

    @classmethod
    def partition_name(cls, collection_name: str, partition_name: str):
        if not isinstance(collection_name, str):
            raise ParamError(message="collection_name must be of str type")
        if not isinstance(partition_name, str):
            raise ParamError(message="partition_name must be of str type")
        return milvus_types.PartitionName(collection_name=collection_name, tag=partition_name)

    @staticmethod
    def _is_input_field(field: Dict, is_upsert: bool):
        return (not field.get("auto_id", False) or is_upsert) and not field.get(
            "is_function_output", False
        )

    @staticmethod
    def _function_output_field_names(fields_info: List[Dict]):
        return [field["name"] for field in fields_info if field.get("is_function_output", False)]

    @staticmethod
    def _num_input_fields(fields_info: List[Dict], is_upsert: bool):
        return len([field for field in fields_info if Prepare._is_input_field(field, is_upsert)])

    @staticmethod
    def _process_struct_field(
        field_name: str,
        values: Any,
        struct_info: Dict,
        struct_sub_field_info: Dict,
        struct_sub_fields_data: Dict,
    ):
        """Process a single struct field's data.

        Args:
            field_name: Name of the struct field
            values: List of struct values
            struct_info: Info about the struct field
            struct_sub_field_info: Two-level dict [struct_name][field_name] -> field info
            struct_sub_fields_data: Two-level dict [struct_name][field_name] -> FieldData
        """
        # Convert numpy ndarray to list if needed
        if isinstance(values, np.ndarray):
            values = values.tolist()

        if not isinstance(values, list):
            msg = f"Field '{field_name}': Expected list, got {type(values).__name__}"
            raise TypeError(msg)

        # Get expected fields for this specific struct
        expected_fields = {field["name"] for field in struct_info["fields"]}

        # Handle empty array - create empty data structures
        if not values:
            # Get relevant field info and data for this struct
            relevant_field_info = struct_sub_field_info[field_name]
            relevant_fields_data = struct_sub_fields_data[field_name]
            Prepare._add_empty_struct_data(relevant_field_info, relevant_fields_data)
            return

        # Validate and collect values
        field_values = Prepare._validate_and_collect_struct_values(
            values, expected_fields, field_name
        )

        # Process collected values using the struct-specific sub-dictionaries
        relevant_field_info = struct_sub_field_info[field_name]
        relevant_fields_data = struct_sub_fields_data[field_name]
        Prepare._process_struct_values(field_values, relevant_field_info, relevant_fields_data)

    @staticmethod
    def _add_empty_struct_data(struct_field_info: Dict, struct_sub_fields_data: Dict):
        """Add empty data for struct fields."""
        for field_name, field_info in struct_field_info.items():
            field_data = struct_sub_fields_data[field_name]

            if field_info["type"] == DataType.ARRAY:
                field_data.scalars.array_data.data.append(convert_to_array([], field_info))
            elif field_info["type"] == DataType._ARRAY_OF_VECTOR:
                field_data.vectors.vector_array.dim = Prepare._get_dim_value(field_info)
                field_data.vectors.vector_array.data.append(
                    convert_to_array_of_vector([], field_info)
                )

    @staticmethod
    def _validate_and_collect_struct_values(
        values: List, expected_fields: set, struct_field_name: str = ""
    ) -> Dict[str, List]:
        """Validate struct items and collect field values."""
        field_values = {field: [] for field in expected_fields}
        field_prefix = f"Field '{struct_field_name}': " if struct_field_name else ""

        for idx, struct_item in enumerate(values):
            if not isinstance(struct_item, dict):
                msg = f"{field_prefix}Element at index {idx} must be dict, got {type(struct_item).__name__}"
                raise TypeError(msg)

            # Validate fields
            actual_fields = set(struct_item.keys())
            missing_fields = expected_fields - actual_fields
            extra_fields = actual_fields - expected_fields

            if missing_fields:
                msg = f"{field_prefix}Element at index {idx} missing required fields: {missing_fields}"
                raise ValueError(msg)
            if extra_fields:
                msg = f"{field_prefix}Element at index {idx} has unexpected fields: {extra_fields}"
                raise ValueError(msg)

            # Collect values
            for field_name in expected_fields:
                value = struct_item[field_name]
                if value is None:
                    msg = f"{field_prefix}Field '{field_name}' in element at index {idx} cannot be None"
                    raise ValueError(msg)
                field_values[field_name].append(value)

        return field_values

    @staticmethod
    def _process_struct_values(
        field_values: Dict[str, List], struct_field_info: Dict, struct_sub_fields_data: Dict
    ):
        """Process collected struct field values."""
        for field_name, values in field_values.items():
            field_data = struct_sub_fields_data[field_name]
            field_info = struct_field_info[field_name]

            if field_info["type"] == DataType.ARRAY:
                field_data.scalars.array_data.data.append(convert_to_array(values, field_info))
            elif field_info["type"] == DataType._ARRAY_OF_VECTOR:
                field_data.vectors.vector_array.dim = Prepare._get_dim_value(field_info)
                field_data.vectors.vector_array.data.append(
                    convert_to_array_of_vector(values, field_info)
                )
            else:
                raise ParamError(message=f"Unsupported data type: {field_info['type']}")

    @staticmethod
    def _get_dim_value(field_info: Dict) -> int:
        """Extract dimension value from field info."""
        dim_value = field_info.get("params", {}).get("dim", 0)
        return int(dim_value) if isinstance(dim_value, str) else dim_value

    @staticmethod
    def _setup_struct_data_structures(struct_fields_info: Optional[List[Dict]]):
        """Setup common data structures for struct field processing.

        Returns:
            Tuple containing:
            - struct_fields_data: Dict of FieldData for struct fields
            - struct_info_map: Dict mapping struct field names to their info
            - struct_sub_fields_data: Two-level Dict of FieldData for
                sub-fields [struct_name][field_name]
            - struct_sub_field_info: Two-level Dict mapping sub-field names
                to their info [struct_name][field_name]
            - input_struct_field_info: List of struct fields info
        """
        struct_fields_data = {}
        struct_info_map = {}
        struct_sub_fields_data = {}
        struct_sub_field_info = {}
        input_struct_field_info = []

        if struct_fields_info:
            struct_fields_data = {
                field["name"]: schema_types.FieldData(field_name=field["name"], type=field["type"])
                for field in struct_fields_info
            }
            input_struct_field_info = list(struct_fields_info)
            struct_info_map = {struct["name"]: struct for struct in struct_fields_info}

            # Use two-level maps to avoid overwrite when different structs have fields
            # with same name
            # First level: struct name, Second level: field name
            for struct_field_info in struct_fields_info:
                struct_name = struct_field_info["name"]
                struct_sub_fields_data[struct_name] = {}
                struct_sub_field_info[struct_name] = {}

                for field in struct_field_info["fields"]:
                    field_name = field["name"]
                    field_data = schema_types.FieldData(field_name=field_name, type=field["type"])
                    # Set dim for ARRAY_OF_VECTOR types
                    if field["type"] == DataType._ARRAY_OF_VECTOR:
                        field_data.vectors.dim = Prepare._get_dim_value(field)
                    struct_sub_fields_data[struct_name][field_name] = field_data
                    struct_sub_field_info[struct_name][field_name] = field

        return (
            struct_fields_data,
            struct_info_map,
            struct_sub_fields_data,
            struct_sub_field_info,
            input_struct_field_info,
        )

    @staticmethod
    def _parse_row_request(
        request: Union[milvus_types.InsertRequest, milvus_types.UpsertRequest],
        fields_info: List[Dict],
        struct_fields_info: List[Dict],
        enable_dynamic: bool,
        entities: List,
    ):
        input_fields_info = [
            field for field in fields_info if Prepare._is_input_field(field, is_upsert=False)
        ]
        # check if pk exists in entities
        primary_field_info = next(
            (field for field in fields_info if field.get("is_primary", False)), None
        )
        if (
            primary_field_info
            and primary_field_info.get("auto_id", False)
            and entities
            and primary_field_info["name"] in entities[0]
        ):
            input_fields_info.append(primary_field_info)

        function_output_field_names = Prepare._function_output_field_names(fields_info)
        fields_data = {
            field["name"]: schema_types.FieldData(field_name=field["name"], type=field["type"])
            for field in input_fields_info
        }
        field_info_map = {field["name"]: field for field in input_fields_info}

        (
            struct_fields_data,
            struct_info_map,
            struct_sub_fields_data,
            struct_sub_field_info,
            input_struct_field_info,
        ) = Prepare._setup_struct_data_structures(struct_fields_info)

        if enable_dynamic:
            d_field = schema_types.FieldData(
                field_name=DYNAMIC_FIELD_NAME, is_dynamic=True, type=DataType.JSON
            )
            fields_data[d_field.field_name] = d_field
            field_info_map[d_field.field_name] = d_field

        try:
            for entity in entities:
                if not isinstance(entity, Dict):
                    msg = f"expected Dict, got '{type(entity).__name__}'"
                    raise TypeError(msg)
                for k, v in entity.items():
                    if k not in fields_data and k not in struct_fields_data:
                        if k in function_output_field_names:
                            raise DataNotMatchException(
                                message=ExceptionsMessage.InsertUnexpectedFunctionOutputField % k
                            )

                        if not enable_dynamic:
                            raise DataNotMatchException(
                                message=ExceptionsMessage.InsertUnexpectedField % k
                            )

                    if k in fields_data:
                        field_info, field_data = field_info_map[k], fields_data[k]
                        if field_info.get("nullable", False) or field_info.get(
                            "default_value", None
                        ):
                            field_data.valid_data.append(v is not None)
                        entity_helper.pack_field_value_to_field_data(v, field_data, field_info)
                    elif k in struct_fields_data:
                        # Array of structs format
                        try:
                            Prepare._process_struct_field(
                                k,
                                v,
                                struct_info_map[k],
                                struct_sub_field_info,
                                struct_sub_fields_data,
                            )
                        except (TypeError, ValueError) as e:
                            raise DataNotMatchException(
                                message=f"{ExceptionsMessage.FieldDataInconsistent % (k, 'struct array', type(v))} Detail: {e!s}"
                            ) from e

                for field in input_fields_info:
                    key = field["name"]
                    if key in entity:
                        continue

                    field_info, field_data = field_info_map[key], fields_data[key]
                    if field_info.get("nullable", False) or field_info.get("default_value", None):
                        field_data.valid_data.append(False)
                        entity_helper.pack_field_value_to_field_data(None, field_data, field_info)
                    else:
                        raise DataNotMatchException(
                            message=ExceptionsMessage.InsertMissedField % key
                        )
                json_dict = {
                    k: v
                    for k, v in entity.items()
                    if k not in fields_data and k not in struct_fields_data and enable_dynamic
                }

                if enable_dynamic:
                    json_value = entity_helper.convert_to_json(json_dict)
                    d_field.scalars.json_data.data.append(json_value)

        except (TypeError, ValueError) as e:
            raise DataNotMatchException(message=ExceptionsMessage.DataTypeInconsistent) from e

        # reconstruct the struct array fields data
        for struct in input_struct_field_info:
            struct_name = struct["name"]
            struct_field_data = struct_fields_data[struct_name]
            for field_info in struct["fields"]:
                # Use two-level map to get the correct sub-field data
                field_name = field_info["name"]
                struct_field_data.struct_arrays.fields.append(
                    struct_sub_fields_data[struct_name][field_name]
                )

        request.fields_data.extend(fields_data.values())
        request.fields_data.extend(struct_fields_data.values())

        expected_num_input_fields = (
            len(input_fields_info) + len(input_struct_field_info) + (1 if enable_dynamic else 0)
        )

        if len(request.fields_data) != expected_num_input_fields:
            msg = f"{ExceptionsMessage.FieldsNumInconsistent}, expected {expected_num_input_fields} fields, got {len(request.fields_data)}"
            raise ParamError(message=msg)

        return request

    @staticmethod
    def _parse_upsert_row_request(
        request: Union[milvus_types.InsertRequest, milvus_types.UpsertRequest],
        fields_info: List[Dict],
        struct_fields_info: List[Dict],
        enable_dynamic: bool,
        entities: List,
        partial_update: bool = False,
    ):
        # For partial update, struct fields are not supported
        if partial_update and struct_fields_info:
            raise ParamError(message="Struct fields are not supported in partial update")

        input_fields_info = [
            field for field in fields_info if Prepare._is_input_field(field, is_upsert=True)
        ]
        function_output_field_names = Prepare._function_output_field_names(fields_info)
        fields_data = {
            field["name"]: schema_types.FieldData(field_name=field["name"], type=field["type"])
            for field in input_fields_info
        }
        field_info_map = {field["name"]: field for field in input_fields_info}
        field_len = {field["name"]: 0 for field in input_fields_info}

        # Use common struct data setup (only if not partial update)
        if partial_update:
            struct_fields_data = {}
            struct_info_map = {}
            struct_sub_fields_data = {}
            struct_sub_field_info = {}
            input_struct_field_info = []
        else:
            (
                struct_fields_data,
                struct_info_map,
                struct_sub_fields_data,
                struct_sub_field_info,
                input_struct_field_info,
            ) = Prepare._setup_struct_data_structures(struct_fields_info)

        if enable_dynamic:
            d_field = schema_types.FieldData(
                field_name=DYNAMIC_FIELD_NAME, is_dynamic=True, type=DataType.JSON
            )
            fields_data[d_field.field_name] = d_field
            field_info_map[d_field.field_name] = d_field
            field_len[DYNAMIC_FIELD_NAME] = 0

        try:
            for entity in entities:
                if not isinstance(entity, Dict):
                    msg = f"expected Dict, got '{type(entity).__name__}'"
                    raise TypeError(msg)
                for k, v in entity.items():
                    if k not in fields_data and k not in struct_fields_data:
                        if k in function_output_field_names:
                            raise DataNotMatchException(
                                message=ExceptionsMessage.InsertUnexpectedFunctionOutputField % k
                            )

                        if not enable_dynamic:
                            raise DataNotMatchException(
                                message=ExceptionsMessage.InsertUnexpectedField % k
                            )

                    if k in fields_data:
                        field_info, field_data = field_info_map[k], fields_data[k]
                        if field_info.get("nullable", False) or field_info.get(
                            "default_value", None
                        ):
                            field_data.valid_data.append(v is not None)
                        entity_helper.pack_field_value_to_field_data(v, field_data, field_info)
                        field_len[k] += 1
                    elif k in struct_fields_data:
                        # Handle struct field (array of structs)
                        try:
                            Prepare._process_struct_field(
                                k,
                                v,
                                struct_info_map[k],
                                struct_sub_field_info,
                                struct_sub_fields_data,
                            )
                        except (TypeError, ValueError) as e:
                            raise DataNotMatchException(
                                message=f"{ExceptionsMessage.FieldDataInconsistent % (k, 'struct array', type(v))} Detail: {e!s}"
                            ) from e
                for field in input_fields_info:
                    key = field["name"]
                    if key in entity:
                        continue

                    # Skip missing field validation for partial updates
                    # Also skip set null value or default value for partial updates,
                    # in case of field is updated to null
                    if partial_update:
                        continue
                    field_info, field_data = field_info_map[key], fields_data[key]
                    if field_info.get("nullable", False) or field_info.get("default_value", None):
                        field_data.valid_data.append(False)
                        field_len[key] += 1
                        entity_helper.pack_field_value_to_field_data(None, field_data, field_info)
                    else:
                        raise DataNotMatchException(
                            message=ExceptionsMessage.InsertMissedField % key
                        )
                json_dict = {
                    k: v
                    for k, v in entity.items()
                    if k not in fields_data and k not in struct_fields_data and enable_dynamic
                }

                if enable_dynamic:
                    json_value = entity_helper.convert_to_json(json_dict)
                    d_field.scalars.json_data.data.append(json_value)
                    field_len[DYNAMIC_FIELD_NAME] += 1

        except (TypeError, ValueError) as e:
            raise DataNotMatchException(message=ExceptionsMessage.DataTypeInconsistent) from e

        if partial_update:
            # cause partial_update won't set null for missing fields,
            # so the field_len must be the same
            row_counts = {v for v in field_len.values() if v > 0}
            if len(row_counts) > 1:
                counts = list(row_counts)
                raise DataNotMatchException(
                    message=ExceptionsMessage.InsertFieldsLenInconsistent % (counts[0], counts[1])
                )

        fields_data = {k: v for k, v in fields_data.items() if field_len[k] > 0}
        request.fields_data.extend(fields_data.values())

        if struct_fields_data:
            # reconstruct the struct array fields data (same as in insert)
            for struct in input_struct_field_info:
                struct_name = struct["name"]
                struct_field_data = struct_fields_data[struct_name]
                for field_info in struct["fields"]:
                    # Use two-level map to get the correct sub-field data
                    field_name = field_info["name"]
                    struct_field_data.struct_arrays.fields.append(
                        struct_sub_fields_data[struct_name][field_name]
                    )
            request.fields_data.extend(struct_fields_data.values())

        for _, field in enumerate(input_fields_info):
            is_dynamic = False
            field_name = field["name"]

            if field.get("is_dynamic", False):
                is_dynamic = True

            for j, entity in enumerate(entities):
                if is_dynamic and field_name in entity:
                    raise ParamError(
                        message=f"dynamic field enabled, {field_name} shouldn't in entities[{j}]"
                    )

        # Include struct fields in expected count (if not partial update)
        struct_field_count = len(input_struct_field_info) if not partial_update else 0
        expected_num_input_fields = (
            len(input_fields_info) + struct_field_count + (1 if enable_dynamic else 0)
        )

        if not partial_update and len(request.fields_data) != expected_num_input_fields:
            msg = f"{ExceptionsMessage.FieldsNumInconsistent}, expected {expected_num_input_fields} fields, got {len(request.fields_data)}"
            raise ParamError(message=msg)

        return request

    @classmethod
    def row_insert_param(
        cls,
        collection_name: str,
        entities: List,
        partition_name: str,
        fields_info: Dict,
        struct_fields_info: Optional[Dict] = None,
        schema_timestamp: int = 0,
        enable_dynamic: bool = False,
        namespace: Optional[str] = None,
    ):
        if not fields_info:
            raise ParamError(message="Missing collection meta to validate entities")

        # insert_request.hash_keys won't be filled in client.
        p_name = partition_name if isinstance(partition_name, str) else ""
        request = milvus_types.InsertRequest(
            collection_name=collection_name,
            partition_name=p_name,
            num_rows=len(entities),
            schema_timestamp=schema_timestamp,
            namespace=namespace,
        )

        return cls._parse_row_request(
            request, fields_info, struct_fields_info, enable_dynamic, entities
        )

    @classmethod
    def row_upsert_param(
        cls,
        collection_name: str,
        entities: List,
        partition_name: str,
        fields_info: Any,
        struct_fields_info: Any = None,
        enable_dynamic: bool = False,
        schema_timestamp: int = 0,
        partial_update: bool = False,
    ):
        if not fields_info:
            raise ParamError(message="Missing collection meta to validate entities")

        # upsert_request.hash_keys won't be filled in client.
        p_name = partition_name if isinstance(partition_name, str) else ""
        request = milvus_types.UpsertRequest(
            collection_name=collection_name,
            partition_name=p_name,
            num_rows=len(entities),
            schema_timestamp=schema_timestamp,
            partial_update=partial_update,
        )

        return cls._parse_upsert_row_request(
            request, fields_info, struct_fields_info, enable_dynamic, entities, partial_update
        )

    @staticmethod
    def _pre_insert_batch_check(
        entities: List,
        fields_info: Any,
    ):
        for entity in entities:
            if (
                entity.get("name") is None
                or entity.get("values") is None
                or entity.get("type") is None
            ):
                raise ParamError(
                    message="Missing param in entities, a field must have type, name and values"
                )
        if not fields_info:
            raise ParamError(message="Missing collection meta to validate entities")

        location, primary_key_loc, _ = traverse_info(fields_info)

        # though impossible from sdk
        if primary_key_loc is None:
            raise ParamError(message="primary key not found")

        expected_num_input_fields = Prepare._num_input_fields(fields_info, is_upsert=False)

        if len(entities) != expected_num_input_fields:
            msg = f"expected number of fields: {expected_num_input_fields}, actual number of fields in entities: {len(entities)}"
            raise ParamError(message=msg)

        return location

    @staticmethod
    def _pre_upsert_batch_check(
        entities: List,
        fields_info: Any,
        partial_update: bool = False,
    ):
        for entity in entities:
            if (
                entity.get("name") is None
                or entity.get("values") is None
                or entity.get("type") is None
            ):
                raise ParamError(
                    message="Missing param in entities, a field must have type, name and values"
                )
        if not fields_info:
            raise ParamError(message="Missing collection meta to validate entities")

        location, primary_key_loc = traverse_upsert_info(fields_info)

        # though impossible from sdk
        if primary_key_loc is None:
            raise ParamError(message="primary key not found")

        # Skip field count validation for partial updates
        if not partial_update:
            expected_num_input_fields = Prepare._num_input_fields(fields_info, is_upsert=True)

            if len(entities) != expected_num_input_fields:
                msg = f"expected number of fields: {expected_num_input_fields}, actual number of fields in entities: {len(entities)}"
                raise ParamError(message=msg)

        return location

    @staticmethod
    def _parse_batch_request(
        request: Union[milvus_types.InsertRequest, milvus_types.UpsertRequest],
        entities: List,
        fields_info: Any,
        location: Dict,
    ):
        pre_field_size = 0
        try:
            for entity in entities:
                latest_field_size = entity_helper.get_input_num_rows(entity.get("values"))
                if latest_field_size != 0:
                    if pre_field_size not in (0, latest_field_size):
                        raise ParamError(
                            message=(
                                f"Field data size misaligned for field [{entity.get('name')}] ",
                                f"got size=[{latest_field_size}] ",
                                f"alignment size=[{pre_field_size}]",
                            )
                        )
                    pre_field_size = latest_field_size
            if pre_field_size == 0:
                raise ParamError(message=ExceptionsMessage.NumberRowsInvalid)
            request.num_rows = pre_field_size
            for entity in entities:
                field_name = entity.get("name")
                field_data = entity_helper.entity_to_field_data(
                    entity, fields_info[location[field_name]], request.num_rows
                )
                request.fields_data.append(field_data)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(message=ExceptionsMessage.DataTypeInconsistent) from e

        if pre_field_size == 0:
            raise ParamError(message=ExceptionsMessage.NumberRowsInvalid)
        request.num_rows = pre_field_size
        return request

    @classmethod
    def batch_insert_param(
        cls,
        collection_name: str,
        entities: List,
        partition_name: str,
        fields_info: Any,
    ):
        location = cls._pre_insert_batch_check(entities, fields_info)
        tag = partition_name if isinstance(partition_name, str) else ""
        request = milvus_types.InsertRequest(collection_name=collection_name, partition_name=tag)

        return cls._parse_batch_request(
            request,
            entities,
            fields_info,
            location,
        )

    @classmethod
    def batch_upsert_param(
        cls,
        collection_name: str,
        entities: List,
        partition_name: str,
        fields_info: Any,
        partial_update: bool = False,
    ):
        location = cls._pre_upsert_batch_check(entities, fields_info, partial_update)
        tag = partition_name if isinstance(partition_name, str) else ""
        request = milvus_types.UpsertRequest(
            collection_name=collection_name,
            partition_name=tag,
            partial_update=partial_update,
        )

        return cls._parse_batch_request(request, entities, fields_info, location)

    @classmethod
    def delete_request(
        cls,
        collection_name: str,
        filter: str,
        partition_name: Optional[str] = None,
        consistency_level: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        check.validate_strs(
            collection_name=collection_name,
            filter=filter,
        )
        check.validate_nullable_strs(partition_name=partition_name)

        return milvus_types.DeleteRequest(
            collection_name=collection_name,
            partition_name=partition_name,
            expr=filter,
            consistency_level=get_consistency_level(consistency_level),
            expr_template_values=cls.prepare_expression_template(kwargs.get("expr_params", {})),
        )

    @classmethod
    def _prepare_placeholder_str(cls, data: Any, is_embedding_list: bool = False):
        # sparse vector
        if entity_helper.entity_is_sparse_matrix(data):
            pl_type = PlaceholderType.SparseFloatVector
            pl_values = entity_helper.sparse_rows_to_proto(data).contents

        elif isinstance(data[0], np.ndarray):
            dtype = data[0].dtype

            if dtype == "bfloat16":
                pl_type = (
                    PlaceholderType.BFLOAT16_VECTOR
                    if not is_embedding_list
                    else PlaceholderType.EmbListBFloat16Vector
                )
                pl_values = (array.tobytes() for array in data)
            elif dtype == "float16":
                pl_type = (
                    PlaceholderType.FLOAT16_VECTOR
                    if not is_embedding_list
                    else PlaceholderType.EmbListFloat16Vector
                )
                pl_values = (array.tobytes() for array in data)
            elif dtype in ("float32", "float64"):
                pl_type = (
                    PlaceholderType.FloatVector
                    if not is_embedding_list
                    else PlaceholderType.EmbListFloatVector
                )
                pl_values = (blob.vector_float_to_bytes(entity) for entity in data)
            elif dtype == "int8":
                pl_type = (
                    PlaceholderType.Int8Vector
                    if not is_embedding_list
                    else PlaceholderType.EmbListInt8Vector
                )
                pl_values = (array.tobytes() for array in data)

            elif dtype == "byte":
                pl_type = PlaceholderType.BinaryVector
                pl_values = data

            else:
                err_msg = f"unsupported data type: {dtype}"
                raise ParamError(message=err_msg)

        elif isinstance(data[0], bytes):
            pl_type = PlaceholderType.BinaryVector
            pl_values = data  # data is already a list of bytes

        elif isinstance(data[0], str):
            pl_type = PlaceholderType.VARCHAR
            pl_values = (value.encode("utf-8") for value in data)

        else:
            pl_type = PlaceholderType.FloatVector
            pl_values = (blob.vector_float_to_bytes(entity) for entity in data)

        pl = common_types.PlaceholderValue(tag="$0", type=pl_type, values=pl_values)
        return common_types.PlaceholderGroup.SerializeToString(
            common_types.PlaceholderGroup(placeholders=[pl])
        )

    @classmethod
    def prepare_expression_template(cls, values: Dict) -> Any:
        def all_elements_same_type(lst: List):
            return all(isinstance(item, type(lst[0])) for item in lst)

        def add_array_data(v: List) -> schema_types.TemplateArrayValue:
            data = schema_types.TemplateArrayValue()
            if len(v) == 0:
                return data
            element_type = (
                infer_dtype_by_scalar_data(v[0]) if all_elements_same_type(v) else schema_types.JSON
            )
            if element_type in (schema_types.Bool,):
                data.bool_data.data.extend(v)
                return data
            if element_type in (
                schema_types.Int8,
                schema_types.Int16,
                schema_types.Int32,
                schema_types.Int64,
            ):
                data.long_data.data.extend(v)
                return data
            if element_type in (schema_types.Float, schema_types.Double):
                data.double_data.data.extend(v)
                return data
            if element_type in (schema_types.VarChar, schema_types.String):
                data.string_data.data.extend(v)
                return data
            if element_type in (schema_types.Array,):
                for e in v:
                    data.array_data.data.append(add_array_data(e))
                return data
            if element_type in (schema_types.JSON,):
                for e in v:
                    data.json_data.data.append(entity_helper.convert_to_json(e))
                return data
            raise ParamError(message=f"Unsupported element type: {element_type}")

        def add_data(v: Any) -> schema_types.TemplateValue:
            dtype = infer_dtype_by_scalar_data(v)
            data = schema_types.TemplateValue()
            if dtype in (schema_types.Bool,):
                data.bool_val = v
                return data
            if dtype in (
                schema_types.Int8,
                schema_types.Int16,
                schema_types.Int32,
                schema_types.Int64,
            ):
                data.int64_val = v
                return data
            if dtype in (schema_types.Float, schema_types.Double):
                data.float_val = v
                return data
            if dtype in (schema_types.VarChar, schema_types.String):
                data.string_val = v
                return data
            if dtype in (schema_types.Array,):
                data.array_val.CopyFrom(add_array_data(v))
                return data
            raise ParamError(message=f"Unsupported element type: {dtype}")

        expression_template_values = {}
        for k, v in values.items():
            expression_template_values[k] = add_data(v)
        return expression_template_values

    @classmethod
    def search_requests_with_expr(
        cls,
        collection_name: str,
        anns_field: str,
        param: Dict,
        limit: int,
        data: Optional[Union[List[List[float]], utils.SparseMatrixInputType]] = None,
        ids: Optional[Union[List[int], List[str], str, int]] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        round_decimal: int = -1,
        ranker: Optional[Union[Function, FunctionScore]] = None,
        highlighter: Optional[Highlighter] = None,
        **kwargs,
    ) -> milvus_types.SearchRequest:
        use_default_consistency = ts_utils.construct_guarantee_ts(collection_name, kwargs)

        ignore_growing = param.get("ignore_growing", False) or kwargs.get("ignore_growing", False)
        params = param.get("params", {})
        if not isinstance(params, dict):
            raise ParamError(message=f"Search params must be a dict, got {type(params)}")

        if PAGE_RETAIN_ORDER_FIELD in kwargs and PAGE_RETAIN_ORDER_FIELD in param:
            raise ParamError(
                message="Provide page_retain_order both in kwargs and param, expect just one"
            )
        page_retain_order = kwargs.get(PAGE_RETAIN_ORDER_FIELD) or param.get(
            PAGE_RETAIN_ORDER_FIELD
        )
        if page_retain_order is not None:
            if not isinstance(page_retain_order, bool):
                raise ParamError(
                    message=f"wrong type for page_retain_order, expect bool, got {type(page_retain_order)}"
                )
            params[PAGE_RETAIN_ORDER_FIELD] = page_retain_order

        search_params = {
            "topk": limit,
            "round_decimal": round_decimal,
            "ignore_growing": ignore_growing,
        }

        # parse offset
        if "offset" in kwargs and "offset" in param:
            raise ParamError(message="Provide offset both in kwargs and param, expect just one")

        offset = kwargs.get("offset") or param.get("offset")
        if offset is not None:
            if not isinstance(offset, int):
                raise ParamError(message=f"wrong type for offset, expect int, got {type(offset)}")
            search_params["offset"] = offset

        is_iterator = kwargs.get(ITERATOR_FIELD)
        if is_iterator is not None:
            search_params[ITERATOR_FIELD] = is_iterator

        collection_id = kwargs.get(COLLECTION_ID)
        if collection_id is not None:
            search_params[COLLECTION_ID] = str(collection_id)

        is_search_iter_v2 = kwargs.get(ITER_SEARCH_V2_KEY)
        if is_search_iter_v2 is not None:
            search_params[ITER_SEARCH_V2_KEY] = is_search_iter_v2

        search_iter_batch_size = kwargs.get(ITER_SEARCH_BATCH_SIZE_KEY)
        if search_iter_batch_size is not None:
            search_params[ITER_SEARCH_BATCH_SIZE_KEY] = search_iter_batch_size

        search_iter_last_bound = kwargs.get(ITER_SEARCH_LAST_BOUND_KEY)
        if search_iter_last_bound is not None:
            search_params[ITER_SEARCH_LAST_BOUND_KEY] = search_iter_last_bound

        search_iter_id = kwargs.get(ITER_SEARCH_ID_KEY)
        if search_iter_id is not None:
            search_params[ITER_SEARCH_ID_KEY] = search_iter_id

        group_by_field = kwargs.get(GROUP_BY_FIELD)
        if group_by_field is not None:
            search_params[GROUP_BY_FIELD] = group_by_field

        group_size = kwargs.get(GROUP_SIZE)
        if group_size is not None:
            search_params[GROUP_SIZE] = group_size

        strict_group_size = kwargs.get(STRICT_GROUP_SIZE)
        if strict_group_size is not None:
            search_params[STRICT_GROUP_SIZE] = strict_group_size

        json_path = kwargs.get(JSON_PATH)
        if json_path is not None:
            search_params[JSON_PATH] = json_path

        json_type = kwargs.get(JSON_TYPE)
        if json_type is not None:
            if json_type == DataType.INT8:
                search_params[JSON_TYPE] = "Int8"
            elif json_type == DataType.INT16:
                search_params[JSON_TYPE] = "Int16"
            elif json_type == DataType.INT32:
                search_params[JSON_TYPE] = "Int32"
            elif json_type == DataType.INT64:
                search_params[JSON_TYPE] = "Int64"
            elif json_type == DataType.BOOL:
                search_params[JSON_TYPE] = "Bool"
            elif json_type in (DataType.VARCHAR, DataType.STRING):
                search_params[JSON_TYPE] = "VarChar"
            else:
                raise ParamError(message=f"Unsupported json cast type: {json_type}")

        strict_cast = kwargs.get(STRICT_CAST)
        if strict_cast is not None:
            search_params[STRICT_CAST] = strict_cast

        if param.get("metric_type") is not None:
            search_params["metric_type"] = param["metric_type"]

        if anns_field:
            search_params["anns_field"] = anns_field

        if param.get(HINTS) is not None:
            search_params[HINTS] = param[HINTS]

        if param.get("analyzer_name") is not None:
            search_params["analyzer_name"] = param["analyzer_name"]

        if kwargs.get("timezone") is not None:
            search_params["timezone"] = kwargs["timezone"]

        if kwargs.get("time_fields") is not None:
            search_params["time_fields"] = kwargs["time_fields"]

        search_params["params"] = get_params(param)

        req_params = [
            common_types.KeyValuePair(key=str(key), value=utils.dumps(value))
            for key, value in search_params.items()
        ]

        expr_params = kwargs.get("expr_params")
        request_kwargs = {
            "collection_name": collection_name,
            "partition_names": partition_names,
            "output_fields": output_fields,
            "guarantee_timestamp": kwargs.get("guarantee_timestamp", 0),
            "use_default_consistency": use_default_consistency,
            "consistency_level": kwargs.get("consistency_level", 0),
            "dsl_type": common_types.DslType.BoolExprV1,
            "search_params": req_params,
            "expr_template_values": cls.prepare_expression_template(
                {} if expr_params is None else expr_params
            ),
            "namespace": kwargs.get("namespace"),
        }

        is_embedding_list = kwargs.get(IS_EMBEDDING_LIST, False)
        if data is not None:
            request_kwargs.update(
                nq=entity_helper.get_input_num_rows(data),
                placeholder_group=cls._prepare_placeholder_str(data, is_embedding_list),
            )
        elif ids is not None:
            request_kwargs.update(
                nq=len(ids),
                ids=cls._build_ids_proto(ids),
            )
        else:
            err_msg = "Either data or ids must be provided"
            raise ValueError(err_msg)

        request = milvus_types.SearchRequest(**request_kwargs)

        if expr is not None:
            request.dsl = expr

        if isinstance(ranker, Function):
            request.function_score.CopyFrom(Prepare.ranker_to_function_score(ranker))
        elif isinstance(ranker, FunctionScore):
            request.function_score.CopyFrom(Prepare.function_score_schema(ranker))
        elif ranker is not None:
            raise ParamError(message="The search ranker must be a Function or FunctionScore.")

        if highlighter is not None:
            request.highlighter.CopyFrom(Prepare.highlighter_schema(highlighter))

        return request

    @staticmethod
    def _build_ids_proto(ids: List[Union[int, np.integer, str]]) -> schema_types.IDs:
        if not ids:
            raise ParamError(message="ids must not be empty")

        first = ids[0]

        if isinstance(first, (bool, np.bool_)):
            raise ParamError(message="ids must not contain boolean values")

        if isinstance(first, (int, np.integer)):
            return schema_types.IDs(
                int_id=schema_types.LongArray(data=[int(value) for value in ids])
            )

        if isinstance(first, str):
            return schema_types.IDs(str_id=schema_types.StringArray(data=list(ids)))

        raise ParamError(message=f"Unsupported id type: {type(first)}")

    @classmethod
    def hybrid_search_request_with_ranker(
        cls,
        collection_name: str,
        reqs: List,
        rerank: Union[BaseRanker, Function],
        limit: int,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        round_decimal: int = -1,
        **kwargs,
    ) -> milvus_types.HybridSearchRequest:
        use_default_consistency = ts_utils.construct_guarantee_ts(collection_name, kwargs)
        if rerank is not None and not isinstance(rerank, (Function, BaseRanker)):
            raise ParamError(message="The hybrid search rerank must be a Function or a Ranker.")
        rerank_param = {}
        if isinstance(rerank, BaseRanker):
            rerank_param = rerank.dict()
        rerank_param["limit"] = limit
        rerank_param["round_decimal"] = round_decimal
        rerank_param["offset"] = kwargs.get("offset", 0)

        request = milvus_types.HybridSearchRequest(
            collection_name=collection_name,
            partition_names=partition_names,
            requests=reqs,
            output_fields=output_fields,
            guarantee_timestamp=kwargs.get("guarantee_timestamp", 0),
            use_default_consistency=use_default_consistency,
            consistency_level=kwargs.get("consistency_level", 0),
            namespace=kwargs.get("namespace"),
        )

        request.rank_params.extend(
            [
                common_types.KeyValuePair(key=str(key), value=utils.dumps(value))
                for key, value in rerank_param.items()
            ]
        )

        if kwargs.get(RANK_GROUP_SCORER) is not None:
            request.rank_params.extend(
                [
                    common_types.KeyValuePair(
                        key=RANK_GROUP_SCORER, value=kwargs.get(RANK_GROUP_SCORER)
                    )
                ]
            )

        if kwargs.get(GROUP_BY_FIELD) is not None:
            request.rank_params.extend(
                [
                    common_types.KeyValuePair(
                        key=GROUP_BY_FIELD, value=utils.dumps(kwargs.get(GROUP_BY_FIELD))
                    )
                ]
            )

        if kwargs.get(GROUP_SIZE) is not None:
            request.rank_params.extend(
                [
                    common_types.KeyValuePair(
                        key=GROUP_SIZE, value=utils.dumps(kwargs.get(GROUP_SIZE))
                    )
                ]
            )

        if kwargs.get(STRICT_GROUP_SIZE) is not None:
            request.rank_params.extend(
                [
                    common_types.KeyValuePair(
                        key=STRICT_GROUP_SIZE, value=utils.dumps(kwargs.get(STRICT_GROUP_SIZE))
                    )
                ]
            )

        if isinstance(rerank, Function):
            request.function_score.CopyFrom(Prepare.ranker_to_function_score(rerank))
        return request

    @staticmethod
    def common_kv_value(v: Any) -> str:
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)

    @staticmethod
    def highlighter_schema(highlighter: Highlighter) -> common_types.Highlighter:
        return common_types.Highlighter(
            type=highlighter.type,
            params=[
                common_types.KeyValuePair(key=str(k), value=Prepare.common_kv_value(v))
                for k, v in highlighter.params.items()
            ],
        )

    @staticmethod
    def function_score_schema(function_score: FunctionScore) -> schema_types.FunctionScore:
        functions = [
            schema_types.FunctionSchema(
                name=ranker.name,
                type=ranker.type,
                description=ranker.description,
                input_field_names=ranker.input_field_names,
                params=(
                    [
                        common_types.KeyValuePair(key=str(k), value=Prepare.common_kv_value(v))
                        for k, v in ranker.params.items()
                    ]
                    if ranker.params
                    else []
                ),
            )
            for ranker in function_score.functions
        ]

        return schema_types.FunctionScore(
            functions=functions,
            params=(
                [
                    common_types.KeyValuePair(key=str(k), value=Prepare.common_kv_value(v))
                    for k, v in function_score.params.items()
                ]
                if function_score.params
                else []
            ),
        )

    @staticmethod
    def ranker_to_function_score(ranker: Function) -> schema_types.FunctionScore:
        function_score = schema_types.FunctionScore(
            functions=[
                schema_types.FunctionSchema(
                    name=ranker.name,
                    type=ranker.type,
                    description=ranker.description,
                    input_field_names=ranker.input_field_names,
                )
            ],
        )
        for k, v in ranker.params.items():
            if isinstance(v, (dict, list)):
                kv_pair = common_types.KeyValuePair(key=str(k), value=json.dumps(v))
            else:
                kv_pair = common_types.KeyValuePair(key=str(k), value=str(v))
            function_score.functions[0].params.append(kv_pair)
        return function_score

    @classmethod
    def create_alias_request(cls, collection_name: str, alias: str):
        return milvus_types.CreateAliasRequest(collection_name=collection_name, alias=alias)

    @classmethod
    def drop_alias_request(cls, alias: str):
        return milvus_types.DropAliasRequest(alias=alias)

    @classmethod
    def alter_alias_request(cls, collection_name: str, alias: str):
        return milvus_types.AlterAliasRequest(collection_name=collection_name, alias=alias)

    @classmethod
    def describe_alias_request(cls, alias: str):
        return milvus_types.DescribeAliasRequest(alias=alias)

    @classmethod
    def list_aliases_request(cls, collection_name: str, db_name: str = ""):
        return milvus_types.ListAliasesRequest(collection_name=collection_name, db_name=db_name)

    @classmethod
    def create_index_request(cls, collection_name: str, field_name: str, params: Dict, **kwargs):
        index_params = milvus_types.CreateIndexRequest(
            collection_name=collection_name,
            field_name=field_name,
            index_name=kwargs.get("index_name", ""),
        )

        if isinstance(params, dict):
            for tk, tv in params.items():
                if tk == "dim" and (not tv or not isinstance(tv, int)):
                    raise ParamError(message="dim must be of int!")
                if tv is not None:
                    kv_pair = common_types.KeyValuePair(key=str(tk), value=utils.dumps(tv))
                    index_params.extra_params.append(kv_pair)

        return index_params

    @classmethod
    def alter_index_properties_request(
        cls, collection_name: str, index_name: str, properties: dict
    ):
        params = []
        for k, v in properties.items():
            params.append(common_types.KeyValuePair(key=str(k), value=utils.dumps(v)))
        return milvus_types.AlterIndexRequest(
            collection_name=collection_name, index_name=index_name, extra_params=params
        )

    @classmethod
    def drop_index_properties_request(
        cls, collection_name: str, index_name: str, delete_keys: List[str]
    ):
        return milvus_types.AlterIndexRequest(
            collection_name=collection_name, index_name=index_name, delete_keys=delete_keys
        )

    @classmethod
    def describe_index_request(
        cls, collection_name: str, index_name: str, timestamp: Optional[int] = None
    ):
        return milvus_types.DescribeIndexRequest(
            collection_name=collection_name, index_name=index_name, timestamp=timestamp
        )

    @classmethod
    def get_index_build_progress(cls, collection_name: str, index_name: str):
        return milvus_types.GetIndexBuildProgressRequest(
            collection_name=collection_name, index_name=index_name
        )

    @classmethod
    def get_index_state_request(cls, collection_name: str, index_name: str):
        return milvus_types.GetIndexStateRequest(
            collection_name=collection_name, index_name=index_name
        )

    @classmethod
    def load_collection(
        cls,
        collection_name: str,
        replica_number: Optional[int] = None,
        **kwargs,
    ):
        check_pass_param(collection_name=collection_name)
        req = milvus_types.LoadCollectionRequest(
            collection_name=collection_name,
        )

        if replica_number:
            check_pass_param(replica_number=replica_number)
            req.replica_number = replica_number

        # Keep underscore key for backward compatibility
        if "refresh" in kwargs or "_refresh" in kwargs:
            refresh = kwargs.get("refresh", kwargs.get("_refresh", False))
            req.refresh = refresh

        if "resource_groups" in kwargs or "_resource_groups" in kwargs:
            resource_groups = kwargs.get("resource_groups", kwargs.get("_resource_groups"))
            req.resource_groups.extend(resource_groups)

        if "load_fields" in kwargs or "_load_fields" in kwargs:
            load_fields = kwargs.get("load_fields", kwargs.get("_load_fields"))
            req.load_fields.extend(load_fields)

        if "skip_load_dynamic_field" in kwargs or "_skip_load_dynamic_field" in kwargs:
            skip_load_dynamic_field = kwargs.get(
                "skip_load_dynamic_field", kwargs.get("_skip_load_dynamic_field", False)
            )
            req.skip_load_dynamic_field = skip_load_dynamic_field

        if "priority" in kwargs:
            priority = kwargs.get("priority")
            req.load_params["load_priority"] = priority

        return req

    @classmethod
    def release_collection(cls, db_name: str, collection_name: str):
        return milvus_types.ReleaseCollectionRequest(
            db_name=db_name, collection_name=collection_name
        )

    @classmethod
    def load_partitions(
        cls,
        collection_name: str,
        partition_names: List[str],
        replica_number: Optional[int] = None,
        **kwargs,
    ):
        check_pass_param(collection_name=collection_name)
        req = milvus_types.LoadPartitionsRequest(
            collection_name=collection_name,
        )

        if partition_names:
            check_pass_param(partition_name_array=partition_names)
            req.partition_names.extend(partition_names)

        if replica_number:
            check_pass_param(replica_number=replica_number)
            req.replica_number = replica_number

        # Keep underscore key for backward compatibility
        if "refresh" in kwargs or "_refresh" in kwargs:
            refresh = kwargs.get("refresh", kwargs.get("_refresh", False))
            req.refresh = refresh

        if "resource_groups" in kwargs or "_resource_groups" in kwargs:
            resource_groups = kwargs.get("resource_groups", kwargs.get("_resource_groups"))
            req.resource_groups.extend(resource_groups)

        if "load_fields" in kwargs or "_load_fields" in kwargs:
            load_fields = kwargs.get("load_fields", kwargs.get("_load_fields"))
            req.load_fields.extend(load_fields)

        if "skip_load_dynamic_field" in kwargs or "_skip_load_dynamic_field" in kwargs:
            skip_load_dynamic_field = kwargs.get(
                "skip_load_dynamic_field", kwargs.get("_skip_load_dynamic_field", False)
            )
            req.skip_load_dynamic_field = skip_load_dynamic_field

        if "priority" in kwargs:
            priority = kwargs.get("priority")
            req.load_params["load_priority"] = priority
        return req

    @classmethod
    def release_partitions(cls, db_name: str, collection_name: str, partition_names: List[str]):
        return milvus_types.ReleasePartitionsRequest(
            db_name=db_name, collection_name=collection_name, partition_names=partition_names
        )

    @classmethod
    def get_collection_stats_request(cls, collection_name: str):
        return milvus_types.GetCollectionStatisticsRequest(collection_name=collection_name)

    @classmethod
    def get_persistent_segment_info_request(cls, collection_name: str):
        return milvus_types.GetPersistentSegmentInfoRequest(collectionName=collection_name)

    @classmethod
    def get_flush_state_request(cls, segment_ids: List[int], collection_name: str, flush_ts: int):
        return milvus_types.GetFlushStateRequest(
            segmentIDs=segment_ids, collection_name=collection_name, flush_ts=flush_ts
        )

    @classmethod
    def get_query_segment_info_request(cls, collection_name: str):
        return milvus_types.GetQuerySegmentInfoRequest(collectionName=collection_name)

    @classmethod
    def flush_param(cls, collection_names: List[str]):
        return milvus_types.FlushRequest(collection_names=collection_names)

    @classmethod
    def drop_index_request(cls, collection_name: str, field_name: str, index_name: str):
        return milvus_types.DropIndexRequest(
            db_name="",
            collection_name=collection_name,
            field_name=field_name,
            index_name=index_name,
        )

    @classmethod
    def get_partition_stats_request(cls, collection_name: str, partition_name: str):
        return milvus_types.GetPartitionStatisticsRequest(
            db_name="", collection_name=collection_name, partition_name=partition_name
        )

    @classmethod
    def dummy_request(cls, request_type: Any):
        return milvus_types.DummyRequest(request_type=request_type)

    @classmethod
    def retrieve_request(
        cls,
        collection_name: str,
        ids: List[str],
        output_fields: List[str],
        partition_names: List[str],
    ):
        ids = schema_types.IDs(int_id=schema_types.LongArray(data=ids))
        return milvus_types.RetrieveRequest(
            db_name="",
            collection_name=collection_name,
            ids=ids,
            output_fields=output_fields,
            partition_names=partition_names,
        )

    @classmethod
    def query_request(
        cls,
        collection_name: str,
        expr: str,
        output_fields: List[str],
        partition_names: List[str],
        **kwargs,
    ):
        use_default_consistency = ts_utils.construct_guarantee_ts(collection_name, kwargs)
        req = milvus_types.QueryRequest(
            db_name="",
            collection_name=collection_name,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            guarantee_timestamp=kwargs.get("guarantee_timestamp", 0),
            use_default_consistency=use_default_consistency,
            consistency_level=kwargs.get("consistency_level", 0),
            expr_template_values=cls.prepare_expression_template(kwargs.get("expr_params", {})),
            namespace=kwargs.get("namespace"),
        )
        collection_id = kwargs.get(COLLECTION_ID)
        if collection_id is not None:
            req.query_params.append(
                common_types.KeyValuePair(key=COLLECTION_ID, value=str(collection_id))
            )

        limit = kwargs.get("limit")
        if limit is not None:
            req.query_params.append(common_types.KeyValuePair(key="limit", value=str(limit)))

        offset = kwargs.get("offset")
        if offset is not None:
            req.query_params.append(common_types.KeyValuePair(key="offset", value=str(offset)))

        timezone = kwargs.get("timezone")
        if timezone is not None:
            req.query_params.append(common_types.KeyValuePair(key="timezone", value=timezone))

        timefileds = kwargs.get("time_fields")
        if timefileds is not None:
            req.query_params.append(common_types.KeyValuePair(key="time_fields", value=timefileds))

        ignore_growing = kwargs.get("ignore_growing", False)
        stop_reduce_for_best = kwargs.get(REDUCE_STOP_FOR_BEST, False)
        is_iterator = kwargs.get(ITERATOR_FIELD)
        if is_iterator is not None:
            req.query_params.append(
                common_types.KeyValuePair(key=ITERATOR_FIELD, value=is_iterator)
            )

        req.query_params.append(
            common_types.KeyValuePair(key="ignore_growing", value=str(ignore_growing))
        )
        req.query_params.append(
            common_types.KeyValuePair(key=REDUCE_STOP_FOR_BEST, value=str(stop_reduce_for_best))
        )
        return req

    @classmethod
    def load_balance_request(
        cls,
        collection_name: str,
        src_node_id: int,
        dst_node_ids: List[int],
        sealed_segment_ids: List[int],
    ):
        return milvus_types.LoadBalanceRequest(
            collectionName=collection_name,
            src_nodeID=src_node_id,
            dst_nodeIDs=dst_node_ids,
            sealed_segmentIDs=sealed_segment_ids,
        )

    @classmethod
    def manual_compaction(
        cls,
        collection_name: str,
        is_clustering: bool,
        is_l0: bool,
        collection_id: Optional[int] = None,
    ):
        if is_clustering is None or not isinstance(is_clustering, bool):
            raise ParamError(message=f"is_clustering value {is_clustering} is illegal")
        if is_l0 is None or not isinstance(is_l0, bool):
            raise ParamError(message=f"is_l0 value {is_l0} is illegal")

        request = milvus_types.ManualCompactionRequest()
        if collection_id is not None:
            request.collectionID = collection_id
        request.collection_name = collection_name
        request.majorCompaction = is_clustering
        request.l0Compaction = is_l0
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

        return milvus_types.GetReplicasRequest(
            collectionID=collection_id,
            with_shard_nodes=True,
        )

    @classmethod
    def do_bulk_insert(cls, collection_name: str, partition_name: str, files: list, **kwargs):
        channel_names = kwargs.get("channel_names")
        req = milvus_types.ImportRequest(
            collection_name=collection_name,
            partition_name=partition_name,
            files=files,
        )
        if channel_names is not None:
            req.channel_names.extend(channel_names)

        for k, v in kwargs.items():
            if k in ("bucket", "backup", "sep", "nullkey"):
                kv_pair = common_types.KeyValuePair(key=str(k), value=str(v))
                req.options.append(kv_pair)

        return req

    @classmethod
    def get_bulk_insert_state(cls, task_id: int):
        if task_id is None or not isinstance(task_id, int):
            msg = f"task_id value {task_id} is not an integer"
            raise ParamError(message=msg)

        return milvus_types.GetImportStateRequest(task=task_id)

    @classmethod
    def list_bulk_insert_tasks(cls, limit: int, collection_name: str):
        if limit is None or not isinstance(limit, int):
            msg = f"limit value {limit} is not an integer"
            raise ParamError(message=msg)

        return milvus_types.ListImportTasksRequest(
            collection_name=collection_name,
            limit=limit,
        )

    @classmethod
    def create_user_request(cls, user: str, password: str):
        check_pass_param(user=user, password=password)
        return milvus_types.CreateCredentialRequest(
            username=user, password=base64.b64encode(password.encode("utf-8"))
        )

    @classmethod
    def update_password_request(cls, user: str, old_password: str, new_password: str):
        check_pass_param(user=user)
        check_pass_param(password=old_password)
        check_pass_param(password=new_password)
        return milvus_types.UpdateCredentialRequest(
            username=user,
            oldPassword=base64.b64encode(old_password.encode("utf-8")),
            newPassword=base64.b64encode(new_password.encode("utf-8")),
        )

    @classmethod
    def delete_user_request(cls, user: str):
        if not isinstance(user, str):
            raise ParamError(message=f"invalid user {user}")
        return milvus_types.DeleteCredentialRequest(username=user)

    @classmethod
    def list_usernames_request(cls):
        return milvus_types.ListCredUsersRequest()

    @classmethod
    def create_role_request(cls, role_name: str):
        check_pass_param(role_name=role_name)
        return milvus_types.CreateRoleRequest(entity=milvus_types.RoleEntity(name=role_name))

    @classmethod
    def drop_role_request(cls, role_name: str, force_drop: bool = False):
        check_pass_param(role_name=role_name)
        return milvus_types.DropRoleRequest(role_name=role_name, force_drop=force_drop)

    @classmethod
    def operate_user_role_request(cls, username: str, role_name: str, operate_user_role_type: Any):
        check_pass_param(user=username)
        check_pass_param(role_name=role_name)
        check_pass_param(operate_user_role_type=operate_user_role_type)
        return milvus_types.OperateUserRoleRequest(
            username=username, role_name=role_name, type=operate_user_role_type
        )

    @classmethod
    def select_role_request(cls, role_name: str, include_user_info: bool):
        if role_name:
            check_pass_param(role_name=role_name)
        check_pass_param(include_user_info=include_user_info)
        return milvus_types.SelectRoleRequest(
            role=milvus_types.RoleEntity(name=role_name) if role_name else None,
            include_user_info=include_user_info,
        )

    @classmethod
    def select_user_request(cls, username: str, include_role_info: bool):
        if username:
            check_pass_param(user=username)
        check_pass_param(include_role_info=include_role_info)
        return milvus_types.SelectUserRequest(
            user=milvus_types.UserEntity(name=username) if username else None,
            include_role_info=include_role_info,
        )

    @classmethod
    def operate_privilege_request(
        cls,
        role_name: str,
        object: Any,
        object_name: str,
        privilege: str,
        db_name: str,
        operate_privilege_type: Any,
    ):
        check_pass_param(role_name=role_name)
        check_pass_param(object=object)
        check_pass_param(object_name=object_name)
        check_pass_param(privilege=privilege)
        check_pass_param(operate_privilege_type=operate_privilege_type)
        return milvus_types.OperatePrivilegeRequest(
            entity=milvus_types.GrantEntity(
                role=milvus_types.RoleEntity(name=role_name),
                object=milvus_types.ObjectEntity(name=object),
                object_name=object_name,
                db_name=db_name,
                grantor=milvus_types.GrantorEntity(
                    privilege=milvus_types.PrivilegeEntity(name=privilege)
                ),
            ),
            type=operate_privilege_type,
        )

    @classmethod
    def operate_privilege_v2_request(
        cls,
        role_name: str,
        privilege: str,
        operate_privilege_type: Any,
        db_name: str,
        collection_name: str,
    ):
        check_pass_param(
            role_name=role_name,
            privilege=privilege,
            collection_name=collection_name,
            operate_privilege_type=operate_privilege_type,
        )
        if db_name:
            check_pass_param(db_name=db_name)
        return milvus_types.OperatePrivilegeV2Request(
            role=milvus_types.RoleEntity(name=role_name),
            grantor=milvus_types.GrantorEntity(
                privilege=milvus_types.PrivilegeEntity(name=privilege)
            ),
            type=operate_privilege_type,
            db_name=db_name,
            collection_name=collection_name,
        )

    @classmethod
    def select_grant_request(cls, role_name: str, object: str, object_name: str, db_name: str):
        check_pass_param(role_name=role_name)
        if object:
            check_pass_param(object=object)
        if object_name:
            check_pass_param(object_name=object_name)
        return milvus_types.SelectGrantRequest(
            entity=milvus_types.GrantEntity(
                role=milvus_types.RoleEntity(name=role_name),
                object=milvus_types.ObjectEntity(name=object) if object else None,
                object_name=object_name if object_name else None,
                db_name=db_name,
            ),
        )

    @classmethod
    def get_server_version(cls):
        return milvus_types.GetVersionRequest()

    @classmethod
    def create_resource_group(cls, name: str, **kwargs):
        check_pass_param(resource_group_name=name)
        return milvus_types.CreateResourceGroupRequest(
            resource_group=name,
            config=kwargs.get("config"),
        )

    @classmethod
    def update_resource_groups(cls, configs: Mapping[str, ResourceGroupConfig]):
        return milvus_types.UpdateResourceGroupsRequest(
            resource_groups=configs,
        )

    @classmethod
    def drop_resource_group(cls, name: str):
        check_pass_param(resource_group_name=name)
        return milvus_types.DropResourceGroupRequest(resource_group=name)

    @classmethod
    def list_resource_groups(cls):
        return milvus_types.ListResourceGroupsRequest()

    @classmethod
    def describe_resource_group(cls, name: str):
        check_pass_param(resource_group_name=name)
        return milvus_types.DescribeResourceGroupRequest(resource_group=name)

    @classmethod
    def transfer_node(cls, source: str, target: str, num_node: int):
        check_pass_param(resource_group_name=source)
        check_pass_param(resource_group_name=target)
        return milvus_types.TransferNodeRequest(
            source_resource_group=source, target_resource_group=target, num_node=num_node
        )

    @classmethod
    def transfer_replica(cls, source: str, target: str, collection_name: str, num_replica: int):
        check_pass_param(resource_group_name=source)
        check_pass_param(resource_group_name=target)
        return milvus_types.TransferReplicaRequest(
            source_resource_group=source,
            target_resource_group=target,
            collection_name=collection_name,
            num_replica=num_replica,
        )

    @classmethod
    def flush_all_request(cls, db_name: str):
        return milvus_types.FlushAllRequest(db_name=db_name)

    @classmethod
    def get_flush_all_state_request(cls, flush_all_ts: int, db_name: str):
        return milvus_types.GetFlushAllStateRequest(flush_all_ts=flush_all_ts, db_name=db_name)

    @classmethod
    def register_request(cls, user: str, host: str, **kwargs):
        reserved = {}
        for k, v in kwargs.items():
            reserved[k] = v
        now = datetime.datetime.now()
        this = common_types.ClientInfo(
            sdk_type="Python",
            sdk_version=__version__,
            local_time=now.__str__(),
            reserved=reserved,
        )
        if user is not None:
            this.user = user
        if host is not None:
            this.host = host
        return milvus_types.ConnectRequest(
            client_info=this,
        )

    @classmethod
    def create_database_req(cls, db_name: str, properties: Optional[dict] = None):
        req = milvus_types.CreateDatabaseRequest(db_name=db_name)

        if is_legal_collection_properties(properties):
            properties = [
                common_types.KeyValuePair(key=str(k), value=str(v)) for k, v in properties.items()
            ]
            req.properties.extend(properties)
        return req

    @classmethod
    def drop_database_req(cls, db_name: str):
        check_pass_param(db_name=db_name)
        return milvus_types.DropDatabaseRequest(db_name=db_name)

    @classmethod
    def list_database_req(cls):
        return milvus_types.ListDatabasesRequest()

    @classmethod
    def alter_database_properties_req(cls, db_name: str, properties: Dict):
        check_pass_param(db_name=db_name)
        kvs = [common_types.KeyValuePair(key=k, value=str(v)) for k, v in properties.items()]
        return milvus_types.AlterDatabaseRequest(db_name=db_name, properties=kvs)

    @classmethod
    def drop_database_properties_req(cls, db_name: str, property_keys: List[str]):
        check_pass_param(db_name=db_name)
        return milvus_types.AlterDatabaseRequest(db_name=db_name, delete_keys=property_keys)

    @classmethod
    def describe_database_req(cls, db_name: str):
        check_pass_param(db_name=db_name)
        return milvus_types.DescribeDatabaseRequest(db_name=db_name)

    @classmethod
    def create_privilege_group_req(cls, privilege_group: str):
        check_pass_param(privilege_group=privilege_group)
        return milvus_types.CreatePrivilegeGroupRequest(group_name=privilege_group)

    @classmethod
    def drop_privilege_group_req(cls, privilege_group: str):
        check_pass_param(privilege_group=privilege_group)
        return milvus_types.DropPrivilegeGroupRequest(group_name=privilege_group)

    @classmethod
    def list_privilege_groups_req(cls):
        return milvus_types.ListPrivilegeGroupsRequest()

    @classmethod
    def operate_privilege_group_req(
        cls, privilege_group: str, privileges: List[str], operate_privilege_group_type: Any
    ):
        check_pass_param(privilege_group=privilege_group)
        check_pass_param(privileges=privileges)
        check_pass_param(operate_privilege_group_type=operate_privilege_group_type)
        return milvus_types.OperatePrivilegeGroupRequest(
            group_name=privilege_group,
            privileges=[milvus_types.PrivilegeEntity(name=p) for p in privileges],
            type=operate_privilege_group_type,
        )

    @classmethod
    def run_analyzer(
        cls,
        texts: Union[str, List[str]],
        analyzer_params: Optional[Union[str, Dict]] = None,
        with_hash: bool = False,
        with_detail: bool = False,
        collection_name: Optional[str] = None,
        field_name: Optional[str] = None,
        analyzer_names: Optional[Union[str, List[str]]] = None,
    ):
        req = milvus_types.RunAnalyzerRequest(with_hash=with_hash, with_detail=with_detail)
        if isinstance(texts, str):
            req.placeholder.append(texts.encode("utf-8"))
        else:
            req.placeholder.extend([text.encode("utf-8") for text in texts])

        if analyzer_params is not None:
            if isinstance(analyzer_params, dict):
                req.analyzer_params = orjson.dumps(analyzer_params).decode(Config.EncodeProtocol)
            else:
                req.analyzer_params = analyzer_params

        if collection_name is not None:
            req.collection_name = collection_name

        if field_name is not None:
            req.field_name = field_name

        if analyzer_names is not None:
            if isinstance(analyzer_names, str):
                req.analyzer_names.extend([analyzer_names])
            else:
                req.analyzer_names.extend(analyzer_names)
        return req

    @classmethod
    def update_replicate_configuration_request(
        cls,
        clusters: Optional[List[Dict]] = None,
        cross_cluster_topology: Optional[List[Dict]] = None,
    ):
        # Validate input parameters
        if clusters is None and cross_cluster_topology is None:
            msg = "Either 'clusters' or 'cross_cluster_topology' must be provided"
            raise ParamError(message=msg)

        # Build ReplicateConfiguration from simplified parameters
        replicate_configuration = common_pb2.ReplicateConfiguration()

        # Add clusters
        if clusters is not None:
            for cluster_config in clusters:
                cluster = common_pb2.MilvusCluster()

                if "cluster_id" not in cluster_config:
                    msg = "cluster_id is required for each cluster"
                    raise ParamError(message=msg)
                cluster.cluster_id = cluster_config["cluster_id"]

                if "connection_param" not in cluster_config:
                    msg = "connection_param is required for each cluster"
                    raise ParamError(message=msg)
                conn_param = cluster_config["connection_param"]
                if "uri" not in conn_param:
                    msg = "uri is required in connection_param"
                    raise ParamError(message=msg)

                cluster.connection_param.uri = conn_param["uri"]
                cluster.connection_param.token = conn_param.get("token", "")

                if "pchannels" in cluster_config:
                    cluster.pchannels.extend(cluster_config["pchannels"])

                replicate_configuration.clusters.append(cluster)

        # Add cross-cluster topology
        if cross_cluster_topology is not None:
            for topology_config in cross_cluster_topology:
                topology = common_pb2.CrossClusterTopology()

                if "source_cluster_id" not in topology_config:
                    msg = "source_cluster_id is required for each topology"
                    raise ParamError(message=msg)
                topology.source_cluster_id = topology_config["source_cluster_id"]

                if "target_cluster_id" not in topology_config:
                    msg = "target_cluster_id is required for each topology"
                    raise ParamError(message=msg)
                topology.target_cluster_id = topology_config["target_cluster_id"]

                replicate_configuration.cross_cluster_topology.append(topology)

        return milvus_types.UpdateReplicateConfigurationRequest(
            replicate_configuration=replicate_configuration
        )

    @staticmethod
    def convert_function_to_function_schema(f: Function) -> schema_types.FunctionSchema:
        function_schema = schema_types.FunctionSchema(
            name=f.name,
            description=f.description,
            type=f.type,
            input_field_names=f.input_field_names,
            output_field_names=f.output_field_names,
        )
        for k, v in f.params.items():
            kv_pair = common_types.KeyValuePair(key=str(k), value=str(v))
            function_schema.params.append(kv_pair)
        return function_schema
