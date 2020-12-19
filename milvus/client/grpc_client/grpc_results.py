import abc
import ujson

from ..exceptions import ParamError

from ..types import DataType


class LoopBase:
    def __init__(self):
        self.__index = 0

    def __iter__(self):
        return self

    def __getitem__(self, item):
        if isinstance(item, slice):
            _start = item.start or 0
            _end = min(item.stop, self.__len__()) if item.stop else self.__len__()
            _step = item.step or 1

            elements = []
            for i in range(_start, _end, _step):
                elements.append(self.get__item(i))
            return elements

        if item >= self.__len__():
            raise IndexError("Index out of range")

        return self.get__item(item)

    def __next__(self):
        while self.__index < self.__len__():
            self.__index += 1
            return self.__getitem__(self.__index - 1)

        # iterate stop, raise Exception
        self.__index = 0
        raise StopIteration()

    @abc.abstractmethod
    def get__item(self, item):
        raise NotImplementedError()


class LoopCache:
    def __init__(self):
        self._array = []

    def fill(self, index, obj):
        if len(self._array) + 1 < index:
            pass


class FieldSchema:
    def __init__(self, raw):
        self._raw = raw

        #
        self.name = None
        self.type = DataType.UNKNOWN
        self.indexes = list()
        self.params = dict()

        ##
        self.__pack(self._raw)

    def __pack(self, raw):
        self.name = raw.name
        self.type = DataType(int(raw.type))

        for kv in raw.extra_params:
            if kv.key == "params":
                self.params = ujson.loads(kv.value)

        index_dict = dict()
        for ikv in raw.index_params:
            if ikv.key != "params":
                index_dict[ikv.key] = ikv.value
            else:
                index_dict[ikv.key] = ujson.loads(ikv.value)

        self.indexes.append(index_dict)

    def dict(self):
        _dict = dict()
        _dict["name"] = self.name
        _dict["type"] = self.type
        _dict["params"] = self.params or dict()
        _dict["indexes"] = self.indexes
        return _dict


class CollectionSchema:
    def __init__(self, raw):
        self._raw = raw

        #
        self.collection_name = None
        self.params = dict()
        self.fields = list()

        #
        self.__pack(self._raw)

    def __pack(self, raw):
        self.collection_name = raw.collection_name
        self.params = dict()
        for kv in raw.extra_params:
            par = ujson.loads(kv.value)
            self.params.update(par)
            # self.params[kv.key] = kv.value

        for f in raw.fields:
            self.fields.append(FieldSchema(f))

    def dict(self):
        _dict = dict()
        _dict["fields"] = [f.dict() for f in self.fields]
        for k, v in self.params.items():
            if isinstance(v, DataType):
                _dict[k] = v.value
            else:
                _dict[k] = v

        return _dict


class Entity:
    def __init__(self, entity_id, entity_fatal_offset, data_types, field_names, raw):
        self._id = entity_id
        self._fatal_id = entity_fatal_offset
        self._types = data_types
        self._field_names = field_names
        self._raw = raw

    def __str__(self):
        str_ = "(\tid: {} \n\tname\t\tvalue\n".format(self._id)
        for name in self._field_names:
            value = self.value_of_field(name)
            str_ += "\t{}\t\t{}".format(name, value)

        return str_

    def __getattr__(self, item):
        return self.value_of_field(item)

    @property
    def id(self):
        return self._id

    @property
    def fields(self):
        return self._field_names

    def get(self, field):
        return self.value_of_field(field)

    def value_of_field(self, field):
        if field not in self._field_names or not self._raw:
            raise ValueError("entity not contain field {}".format(field))

        i = self._field_names.index(field)
        type_ = self._types[i]

        for fd in self._raw.fields:
            if fd.field_name != field or type_ != DataType(int(fd.type)):
                continue

            if type_ in (DataType.INT32,):
                return fd.attr_record.int32_value[self._fatal_id]
            if type_ in (DataType.INT64,):
                return fd.attr_record.int64_value[self._fatal_id]
            if type_ in (DataType.FLOAT,):
                return fd.attr_record.float_value[self._fatal_id]
            if type_ in (DataType.DOUBLE,):
                return fd.attr_record.double_value[self._fatal_id]
            if type_ in (DataType.FLOAT_VECTOR,):
                return list(fd.vector_record.records[self._fatal_id].float_data)
            if type_ in (DataType.BINARY_VECTOR,):
                return bytes(fd.vector_record.records[self._fatal_id].binary_data)

            raise ParamError("Unknown field type {}".format(type_))

        raise IndexError("Unknown field {}".format(field))

    def type_of_field(self, field):
        if field not in self._field_names:
            raise ValueError("entity not contain field {}".format(field))

        i = self._field_names.index(field)
        return self._types[i]


class Entities(LoopBase):
    def __init__(self, raw):
        super().__init__()
        self._raw = raw

        self._flat_ids = list()
        self._valid_raw = list()
        self._field_types = list()
        self._field_names = list()

        #
        self._extract(self._raw)

    def __len__(self):
        return len(self._flat_ids)

    def _extract(self, raw):
        self._flat_ids = list(raw.ids)
        self._valid_raw = list(raw.valid_row)
        for field in raw.fields:
            self._field_types.append(DataType(int(field.type)))
            self._field_names.append(field.field_name)

    def get__item(self, item):
        # if
        if not self._valid_raw:
            return Entity(self._flat_ids[item], -1, [], [], None)

        if not self._valid_raw[item]:
            return None

        if self._flat_ids[item] == -1:
            return Entity(-1, -1, [], [], None)

        fatal_item = sum([1 for v in self._valid_raw[:item] if v])

        return Entity(self._flat_ids[item], fatal_item,
                      self._field_types, self._field_names, self._raw)

    @property
    def ids(self):
        return self._flat_ids

    def dict(self):
        entity_list = list()
        for field in self._raw.fields:
            type_ = DataType(int(field.type))
            if type_ in (DataType.INT32,):
                values = list(field.attr_record.int32_value)
            elif type_ in (DataType.INT64,):
                values = list(field.attr_record.int64_value)
            elif type_ in (DataType.FLOAT,):
                values = list(field.attr_record.float_value)
            elif type_ in (DataType.DOUBLE,):
                values = list(field.attr_record.double_value)
            elif type_ in (DataType.FLOAT_VECTOR,):
                values = [list(record.float_data) for record in field.vector_record.records]
            elif type_ in (DataType.BINARY_VECTOR,):
                values = [bytes(record.binary_data) for record in field.vector_record.records]
            else:
                raise ParamError("Unknown field type {}".format(type_))

            entity_list.append({"field": field.field_name, "values": values, "type": type_})

        return entity_list


class ItemQueryResult:
    def __init__(self, entity, distance, score):
        self._entity = entity
        self._dis = distance
        self._score = score

    def __str__(self):
        return "(distance: {}, score: {}, entity: {})".format(self._dis, self._score, self._entity)

    @property
    def entity(self):
        return self._entity

    @property
    def id(self):
        return self._entity.id

    @property
    def distance(self):
        return self._dis

    @property
    def score(self):
        return self._score


class RawQueryResult(LoopBase):
    def __init__(self, entity_list, distance_list, score_list):
        super().__init__()
        self._entities = entity_list
        self._distances = distance_list
        self._scores = score_list

    def __len__(self):
        return sum([1 for e in self._entities if e and e.id != -1])

    def get__item(self, item):
        score = self._scores[item] if self._scores else None
        return ItemQueryResult(self._entities[item], self._distances[item], score)

    @property
    def ids(self):
        return [e.id for e in self._entities]

    @property
    def distances(self):
        return self._distances


class ItemQueryResult2:
    def __init__(self, row_index, column_index, result, shape):
        self._r_index = row_index
        self._c_index = column_index
        self._result = result
        self._shape = shape
        self._location = row_index * shape[1] + column_index

    def __str__(self):
        return "(distance: {}, score: {}, entity: {})"\
            .format(self.distance, self.score, self.entity)

    @property
    def entity(self):
        return self._result.entities[self._location]

    @property
    def id(self):
        return self._result.raw.entities.ids[self._location]

    @property
    def distance(self):
        return self._result.raw.distances[self._location]

    @property
    def score(self):
        return self._result.raw.scores[self._location]


class RawQueryResult2(LoopBase):
    def __init__(self, index, result, raw_count, column_count):
        super().__init__()
        self._index = index
        self._result = result
        self._shape = (raw_count, column_count)
        self._begin = self._index * self._shape[1]
        self._end = self._begin + self._shape[1]
        self._distances = result._distances[self._begin: self._end]

        self._len = -1

    def __len__(self):
        if self._len == -1:
            self._len = len(self.ids)

        return self._len

    def get__item(self, item):
        return ItemQueryResult2(self._index, item, self._result, self._shape)

    @property
    def _entities(self):
        return self._result.entities[self._begin: self._end]

    @property
    def ids(self):
        start = self._index * self._shape[1]
        return list([id_ for id_ in
                     self._result.entities.ids[start: start + self._shape[1]] if id_ > -1])

    @property
    def distances(self):
        start = self._index * self._shape[1]
        return list(self._result.raw.distances[start, start + self._shape[1]])


class QueryResult(LoopBase):
    def __init__(self, raw):
        super().__init__()
        self._raw = raw
        self._nq = self._raw.row_num
        self._topk = len(self._raw.distances) // self._nq if self._nq > 0 else 0
        self._entities = Entities(self._raw.entities)
        self._distances = list(raw.distances)

    def __len__(self):
        return self._nq

    def __len(self):
        return self._nq

    def get__item(self, item):
        dis_len = len(self._raw.distances)
        topk = dis_len // self._nq if self._nq > 0 else 0

        return RawQueryResult2(item, self, self._nq, topk)

    @property
    def raw(self):
        return self._raw

    @property
    def entities(self):
        return self._entities
