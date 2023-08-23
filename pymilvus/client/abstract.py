import abc
from typing import Any, Dict, List

from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import schema_pb2
from pymilvus.settings import Config

from . import entity_helper
from .constants import DEFAULT_CONSISTENCY_LEVEL
from .types import DataType


class LoopBase:
    def __init__(self):
        self.__index = 0

    def __iter__(self):
        return self

    def __getitem__(self, item: Any):
        if isinstance(item, slice):
            _start = item.start or 0
            _end = min(item.stop, self.__len__()) if item.stop else self.__len__()
            _step = item.step or 1

            return [self.get__item(i) for i in range(_start, _end, _step)]

        if item >= self.__len__():
            msg = "Index out of range"
            raise IndexError(msg)

        return self.get__item(item)

    def __next__(self):
        while self.__index < self.__len__():
            self.__index += 1
            return self.__getitem__(self.__index - 1)

        # iterate stop, raise Exception
        self.__index = 0
        raise StopIteration

    def __str__(self):
        return str(list(map(str, self.__getitem__(slice(0, 10)))))

    @abc.abstractmethod
    def get__item(self, item: Any):
        raise NotImplementedError


class LoopCache:
    def __init__(self):
        self._array = []

    def fill(self, index: int, obj: Any):
        if len(self._array) + 1 < index:
            pass


class FieldSchema:
    def __init__(self, raw: Any):
        self._raw = raw

        #
        self.field_id = 0
        self.name = None
        self.is_primary = False
        self.description = None
        self.auto_id = False
        self.type = DataType.UNKNOWN
        self.indexes = []
        self.params = {}
        self.is_partition_key = False
        self.is_dynamic = False

        ##
        self.__pack(self._raw)

    def __pack(self, raw: Any):
        self.field_id = raw.fieldID
        self.name = raw.name
        self.is_primary = raw.is_primary_key
        self.description = raw.description
        self.auto_id = raw.autoID
        self.type = raw.data_type
        self.is_partition_key = raw.is_partition_key
        try:
            self.is_dynamic = raw.is_dynamic
        except Exception:
            self.is_dynamic = False

        for type_param in raw.type_params:
            if type_param.key == "params":
                import json

                self.params[type_param.key] = json.loads(type_param.value)
            else:
                self.params[type_param.key] = type_param.value
                if type_param.key in ["dim"]:
                    self.params[type_param.key] = int(type_param.value)
                if (
                    type_param.key in [Config.MaxVarCharLengthKey]
                    and raw.data_type == DataType.VARCHAR
                ):
                    self.params[type_param.key] = int(type_param.value)

        index_dict = {}
        for index_param in raw.index_params:
            if index_param.key == "params":
                import json

                index_dict[index_param.key] = json.loads(index_param.value)
            else:
                index_dict[index_param.key] = index_param.value

        self.indexes.extend([index_dict])

    def dict(self):
        _dict = {
            "field_id": self.field_id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "params": self.params or {},
        }

        if self.is_partition_key:
            _dict["is_partition_key"] = True
        if self.is_dynamic:
            _dict["is_dynamic"] = True
        if self.auto_id:
            _dict["auto_id"] = True
        if self.is_primary:
            _dict["is_primary"] = self.is_primary
        return _dict


class CollectionSchema:
    def __init__(self, raw: Any):
        self._raw = raw

        #
        self.collection_name = None
        self.description = None
        self.params = {}
        self.fields = []
        self.statistics = {}
        self.auto_id = False  # auto_id is not in collection level any more later
        self.aliases = []
        self.collection_id = 0
        self.consistency_level = DEFAULT_CONSISTENCY_LEVEL  # by default
        self.properties = {}
        self.num_shards = 0
        self.num_partitions = 0
        self.enable_dynamic_field = False

        #
        if self._raw:
            self.__pack(self._raw)

    def __pack(self, raw: Any):
        self.collection_name = raw.schema.name
        self.description = raw.schema.description
        self.aliases = raw.aliases
        self.collection_id = raw.collectionID
        self.num_shards = raw.shards_num
        self.num_partitions = raw.num_partitions

        # keep compatible with older Milvus
        try:
            self.consistency_level = raw.consistency_level
        except Exception:
            self.consistency_level = DEFAULT_CONSISTENCY_LEVEL

        try:
            self.enable_dynamic_field = raw.schema.enable_dynamic_field
        except Exception:
            self.enable_dynamic_field = False

        # TODO: extra_params here
        # for kv in raw.extra_params:

        self.fields = [FieldSchema(f) for f in raw.schema.fields]

        # for s in raw.statistics:

        self.properties = raw.properties

    @classmethod
    def _rewrite_schema_dict(cls, schema_dict: Dict):
        fields = schema_dict.get("fields", [])
        if not fields:
            return

        for field_dict in fields:
            if field_dict.get("auto_id", None) is not None:
                schema_dict["auto_id"] = field_dict["auto_id"]
                return

    def dict(self):
        if not self._raw:
            return {}
        _dict = {
            "collection_name": self.collection_name,
            "auto_id": self.auto_id,
            "num_shards": self.num_shards,
            "description": self.description,
            "fields": [f.dict() for f in self.fields],
            "aliases": self.aliases,
            "collection_id": self.collection_id,
            "consistency_level": self.consistency_level,
            "properties": self.properties,
            "num_partitions": self.num_partitions,
        }
        if self.enable_dynamic_field:
            _dict["enable_dynamic_field"] = self.enable_dynamic_field
        self._rewrite_schema_dict(_dict)
        return _dict

    def __str__(self):
        return self.dict().__str__()


class Entity:
    def __init__(self, entity_id: int, entity_row_data: Any, entity_score: float):
        self._id = entity_id
        self._row_data = entity_row_data
        self._score = entity_score
        self._distance = entity_score

    def __str__(self):
        return f"id: {self._id}, distance: {self._distance}, entity: {self._row_data}"

    def __getattr__(self, item: Any):
        return self.value_of_field(item)

    @property
    def id(self):
        return self._id

    @property
    def fields(self):
        return [k for k, v in self._row_data.items()]

    def get(self, field: Any):
        return self.value_of_field(field)

    def value_of_field(self, field: Any):
        if field not in self._row_data:
            raise MilvusException(message=f"Field {field} is not in return entity")
        return self._row_data[field]

    def type_of_field(self, field: Any):
        msg = "TODO: support field in Hits"
        raise NotImplementedError(msg)

    def to_dict(self):
        return {"id": self._id, "distance": self._distance, "entity": self._row_data}


class Hit:
    def __init__(self, entity_id: int, entity_row_data: Any, entity_score: float):
        self._id = entity_id
        self._row_data = entity_row_data
        self._score = entity_score
        self._distance = entity_score

    def __str__(self):
        return str(self.entity)

    __repr__ = __str__

    @property
    def entity(self):
        return Entity(self._id, self._row_data, self._score)

    @property
    def id(self):
        return self._id

    @property
    def distance(self):
        return self._distance

    @property
    def score(self):
        return self._score

    def to_dict(self):
        return self.entity.to_dict()


class Hits(LoopBase):
    def __init__(self, raw: Any, round_decimal: int = -1):
        super().__init__()
        self._raw = raw
        if round_decimal != -1:
            self._distances = [round(x, round_decimal) for x in self._raw.scores]
        else:
            self._distances = self._raw.scores

        self._dynamic_field_name = None
        self._dynamic_fields = set()
        (
            self._dynamic_field_name,
            self._dynamic_fields,
        ) = entity_helper.extract_dynamic_field_from_result(self._raw)

    def __len__(self):
        if self._raw.ids.HasField("int_id"):
            return len(self._raw.ids.int_id.data)
        if self._raw.ids.HasField("str_id"):
            return len(self._raw.ids.str_id.data)
        return 0

    def get__item(self, item: Any):
        if self._raw.ids.HasField("int_id"):
            entity_id = self._raw.ids.int_id.data[item]
        elif self._raw.ids.HasField("str_id"):
            entity_id = self._raw.ids.str_id.data[item]
        else:
            raise MilvusException(message="Unsupported ids type")

        entity_row_data = entity_helper.extract_row_data_from_fields_data(
            self._raw.fields_data, item, self._dynamic_fields
        )
        entity_score = self._distances[item]
        return Hit(entity_id, entity_row_data, entity_score)

    @property
    def ids(self):
        if self._raw.ids.HasField("int_id"):
            return self._raw.ids.int_id.data
        if self._raw.ids.HasField("str_id"):
            return self._raw.ids.str_id.data
        return []

    @property
    def distances(self):
        return self._distances


class MutationResult:
    def __init__(self, raw: Any):
        self._raw = raw
        self._primary_keys = []
        self._insert_cnt = 0
        self._delete_cnt = 0
        self._upsert_cnt = 0
        self._timestamp = 0
        self._succ_index = []
        self._err_index = []

        self._pack(raw)

    @property
    def primary_keys(self):
        return self._primary_keys

    @property
    def insert_count(self):
        return self._insert_cnt

    @property
    def delete_count(self):
        return self._delete_cnt

    @property
    def upsert_count(self):
        return self._upsert_cnt

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def succ_count(self):
        return len(self._succ_index)

    @property
    def err_count(self):
        return len(self._err_index)

    @property
    def succ_index(self):
        return self._succ_index

    @property
    def err_index(self):
        return self._err_index

    def __str__(self):
        return (
            f"(insert count: {self._insert_cnt}, delete count: {self._delete_cnt}, upsert count: {self._upsert_cnt}, "
            f"timestamp: {self._timestamp}, success count: {self.succ_count}, err count: {self.err_count})"
        )

    __repr__ = __str__

    # TODO
    # def error_code(self):
    #     pass
    #
    # def error_reason(self):
    #     pass

    def _pack(self, raw: Any):
        which = raw.IDs.WhichOneof("id_field")
        if which == "int_id":
            self._primary_keys = raw.IDs.int_id.data
        elif which == "str_id":
            self._primary_keys = raw.IDs.str_id.data

        self._insert_cnt = raw.insert_cnt
        self._delete_cnt = raw.delete_cnt
        self._upsert_cnt = raw.upsert_cnt
        self._timestamp = raw.timestamp
        self._succ_index = raw.succ_index
        self._err_index = raw.err_index


class QueryResult(LoopBase):
    def __init__(self, raw: Any):
        super().__init__()
        self._raw = raw
        self._pack(raw.hits)

    def __len__(self):
        return self._nq

    def _pack(self, raw: Any):
        self._nq = raw.results.num_queries
        self._topk = raw.results.top_k
        self._hits = []
        offset = 0
        for i in range(self._nq):
            hit = schema_pb2.SearchResultData()
            start_pos = offset
            end_pos = offset + raw.results.topks[i]
            hit.scores.append(raw.results.scores[start_pos:end_pos])
            if raw.results.ids.HasField("int_id"):
                hit.ids.append(raw.results.ids.int_id.data[start_pos:end_pos])
            elif raw.results.ids.HasField("str_id"):
                hit.ids.append(raw.results.ids.str_id.data[start_pos:end_pos])
            for field_data in raw.result.fields_data:
                field = schema_pb2.FieldData()
                field.type = field_data.type
                field.field_name = field_data.field_name
                if field_data.type == DataType.BOOL:
                    field.scalars.bool_data.data.extend(
                        field_data.scalars.bool_data.data[start_pos:end_pos]
                    )
                elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                    field.scalars.int_data.data.extend(
                        field_data.scalars.int_data.data[start_pos:end_pos]
                    )
                elif field_data.type == DataType.INT64:
                    field.scalars.long_data.data.extend(
                        field_data.scalars.long_data.data[start_pos:end_pos]
                    )
                elif field_data.type == DataType.FLOAT:
                    field.scalars.float_data.data.extend(
                        field_data.scalars.float_data.data[start_pos:end_pos]
                    )
                elif field_data.type == DataType.DOUBLE:
                    field.scalars.double_data.data.extend(
                        field_data.scalars.double_data.data[start_pos:end_pos]
                    )
                elif field_data.type == DataType.VARCHAR:
                    field.scalars.string_data.data.extend(
                        field_data.scalars.string_data.data[start_pos:end_pos]
                    )
                elif field_data.type == DataType.STRING:
                    raise MilvusException(message="Not support string yet")
                elif field_data.type == DataType.JSON:
                    field.scalars.json_data.data.extend(
                        field_data.scalars.json_data.data[start_pos:end_pos]
                    )
                elif field_data.type == DataType.FLOAT_VECTOR:
                    dim = field.vectors.dim
                    field.vectors.dim = dim
                    field.vectors.float_vector.data.extend(
                        field_data.vectors.float_data.data[start_pos * dim : end_pos * dim]
                    )
                elif field_data.type == DataType.BINARY_VECTOR:
                    dim = field_data.vectors.dim
                    field.vectors.dim = dim
                    field.vectors.binary_vector += field_data.vectors.binary_vector[
                        start_pos * (dim // 8) : end_pos * (dim // 8)
                    ]
                hit.fields_data.append(field)
            self._hits.append(hit)
            offset += raw.results.topks[i]

    def get__item(self, item: Any):
        return Hits(self._hits[item])


class ChunkedQueryResult(LoopBase):
    def __init__(self, raw_list: List, round_decimal: int = -1):
        super().__init__()
        self._raw_list = raw_list
        self._nq = 0
        self.round_decimal = round_decimal

        self._pack(self._raw_list)

    def __len__(self):
        return self._nq

    def _pack(self, raw_list: List):
        self._hits = []
        for raw in raw_list:
            nq = raw.results.num_queries
            self._nq += nq
            self._topk = raw.results.top_k
            offset = 0

            for i in range(nq):
                hit = schema_pb2.SearchResultData()
                start_pos = offset
                end_pos = offset + raw.results.topks[i]
                hit.scores.extend(raw.results.scores[start_pos:end_pos])
                if raw.results.ids.HasField("int_id"):
                    hit.ids.int_id.data.extend(raw.results.ids.int_id.data[start_pos:end_pos])
                elif raw.results.ids.HasField("str_id"):
                    hit.ids.str_id.data.extend(raw.results.ids.str_id.data[start_pos:end_pos])
                hit.output_fields.extend(raw.results.output_fields)
                for field_data in raw.results.fields_data:
                    field = schema_pb2.FieldData()
                    field.type = field_data.type
                    field.field_name = field_data.field_name
                    field.is_dynamic = field_data.is_dynamic
                    if field_data.type == DataType.BOOL:
                        field.scalars.bool_data.data.extend(
                            field_data.scalars.bool_data.data[start_pos:end_pos]
                        )
                    elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                        field.scalars.int_data.data.extend(
                            field_data.scalars.int_data.data[start_pos:end_pos]
                        )
                    elif field_data.type == DataType.INT64:
                        field.scalars.long_data.data.extend(
                            field_data.scalars.long_data.data[start_pos:end_pos]
                        )
                    elif field_data.type == DataType.FLOAT:
                        field.scalars.float_data.data.extend(
                            field_data.scalars.float_data.data[start_pos:end_pos]
                        )
                    elif field_data.type == DataType.DOUBLE:
                        field.scalars.double_data.data.extend(
                            field_data.scalars.double_data.data[start_pos:end_pos]
                        )
                    elif field_data.type == DataType.VARCHAR:
                        field.scalars.string_data.data.extend(
                            field_data.scalars.string_data.data[start_pos:end_pos]
                        )
                    elif field_data.type == DataType.STRING:
                        raise MilvusException(message="Not support string yet")
                    elif field_data.type == DataType.JSON:
                        field.scalars.json_data.data.extend(
                            field_data.scalars.json_data.data[start_pos:end_pos]
                        )
                    elif field_data.type == DataType.FLOAT_VECTOR:
                        dim = field_data.vectors.dim
                        field.vectors.dim = dim
                        field.vectors.float_vector.data.extend(
                            field_data.vectors.float_vector.data[start_pos * dim : end_pos * dim]
                        )
                    elif field_data.type == DataType.BINARY_VECTOR:
                        dim = field_data.vectors.dim
                        field.vectors.dim = dim
                        field.vectors.binary_vector += field_data.vectors.binary_vector[
                            start_pos * (dim // 8) : end_pos * (dim // 8)
                        ]
                    hit.fields_data.append(field)
                self._hits.append(hit)
                offset += raw.results.topks[i]

    def get__item(self, item: Any):
        return Hits(self._hits[item], self.round_decimal)


def _abstract():
    msg = "You need to override this function"
    raise NotImplementedError(msg)
