import abc
import ujson

from ..client.exceptions import ParamError

from .check import check_pass_param

from .types import IndexType, DataType


class LoopBase(object):
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

        for ikv in raw.index_params:
            index_dict = ujson.loads(ikv.value)
            index_dict["index_name"] = ikv.key

            self.indexes.append(index_dict)

    def dict(self):
        _dict = dict()
        _dict["field"] = self.name
        _dict["type"] = self.type
        _dict["params"] = self.params
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
            _dict[k] = v

        return _dict


class Entity:
    def __init__(self, entity_id, data_types, field_names, field_values):
        self._id = entity_id
        self._types = data_types
        self._field_names = field_names
        self._field_values = field_values

    def __str__(self):
        str_ = "(\tid: {} \n\tname\t\tvalue"
        for name, value in zip(self._field_names, self._field_values):
            str_ += "\t{}\t\t{}\n".format(name, value)

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
        if field not in self._field_names:
            raise ValueError("entity not contain field {}".format(field))

        i = self._field_names.index(field)
        return self._field_values[i]

    def type_of_field(self, field):
        if field not in self._field_names:
            raise ValueError("entity not contain field {}".format(field))

        i = self._field_names.index(field)
        return self._types[i]


class Entities(LoopBase):
    def __init__(self, raw):
        super().__init__()
        self._raw = raw
        self._ids = list(self._raw.ids)
        self._valid_raw = []

    def __len__(self):
        return len(self._raw.ids)

    def get__item(self, item):
        if not self._ids:
            self._ids = list(self._raw.ids)

        if not self._valid_raw:
            self._valid_raw = list(self._raw.valid_row)

        # if
        if not self._valid_raw:
            return Entity(self._ids[item], [], [], [])

        if not self._valid_raw[item]:
            return None

        fatal_item = sum([1 for v in self._valid_raw[:item] if v])

        field_types = list()
        field_names = list()
        field_values = list()
        for field in self._raw.fields:
            type = DataType(int(field.type))
            if type in (DataType.INT32,):
            # if type in (DataType.INT8, DataType.INT16, DataType.INT32):
                slice_value = field.attr_record.int32_value[fatal_item]
            elif type in (DataType.INT64,):
                slice_value = field.attr_record.int64_value[fatal_item]
            elif type in (DataType.FLOAT,):
                slice_value = field.attr_record.float_value[fatal_item]
            elif type in (DataType.DOUBLE,):
                slice_value = field.attr_record.double_value[fatal_item]
            elif type in (DataType.FLOAT_VECTOR,):
                slice_value = list(field.vector_record.records[item].float_data)
            elif type in (DataType.BINARY_VECTOR,):
                slice_value = bytes(field.vector_record.records[item].binary_data)
            else:
                raise ParamError("Unknown field type {}".format(type))

            field_types.append(type)
            field_names.append(field.field_name)
            field_values.append(slice_value)

        # values = [fv[item] for fv in field_values]
        return Entity(self._ids[item], field_types, field_names, field_values)

    @property
    def ids(self):
        return self._ids

    def dict(self):
        entity_list = list()
        for field in self._raw.fields:
            type = DataType(int(field.type))
            if type in (DataType.INT32,):
            # if type in (DataType.INT8, DataType.INT16, DataType.INT32):
                values = list(field.attr_record.int32_value)
            elif type in (DataType.INT64,):
                values = list(field.attr_record.int64_value)
            elif type in (DataType.FLOAT,):
                values = list(field.attr_record.float_value)
            elif type in (DataType.DOUBLE,):
                values = list(field.attr_record.double_value)
            elif type in (DataType.FLOAT_VECTOR,):
                # values = list(field.vector_record.records[item].float_data)
                values = [list(record.float_data) for record in field.vector_record.records]
            elif type in (DataType.BINARY_VECTOR,):
                values = [bytes(record.binary_data) for record in field.vector_record.records]
            else:
                raise ParamError("Unknown field type {}".format(type))

            entity_list.append({"field": field.field_name, "values": values, "type": type})

        return entity_list


class ItemQueryResult:
    def __init__(self, entity, distance, score):
        self._entity = entity
        self._dis = distance
        self._score = score

    def __str__(self):
        str_ = "(distance: {}, score: {}, entity: {})".format(self._dis, self._score, self._entity)
        return str_

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
        return len(self._entities)

    def get__item(self, item):
        score = self._scores[item] if self._scores else None
        return ItemQueryResult(self._entities[item], self._distances[item], score)

    @property
    def ids(self):
        return [e.id for e in self._entities]

    @property
    def distances(self):
        return self._distances


class QueryResult(LoopBase):
    def __init__(self, raw):
        super().__init__()
        self._raw = raw
        self._nq = self._raw.row_num
        self._topk = len(self._raw.distances) // self._nq if self._nq > 0 else 0
        self._entities = Entities(self._raw.entities)

    def __len__(self):
        return self._nq

    def __len(self):
        return self._nq

    def _pack(self, raw):
        pass

    def get__item(self, item):
        dis_len = len(self._raw.distances)
        topk = dis_len // self._nq if self._nq > 0 else 0
        start = item * topk
        end = (item + 1) * topk
        slice_score = list(self._raw.scores)[start: end] if dis_len > 0 else []
        slice_distances = list(self._raw.distances)[start: end] if dis_len > 0 else []

        slice_entity = self._entities[start: end]

        return RawQueryResult(slice_entity, slice_distances, slice_score)


class IndexParam:
    """
    Index Param

    :type  table_name: str
    :param table_name: (Required) name of table

    :type  index_type: IndexType
    :param index_type: (Required) index type, default = IndexType.INVALID

        `IndexType`: 0-invalid, 1-flat, 2-ivflat, 3-IVF_SQ8, 4-MIX_NSG

    :type  nlist: int64
    :param nlist: (Required) num of cell

    """

    def __init__(self, collection_name, index_type, params):

        if collection_name is None:
            raise ParamError('Collection name can\'t be None')
        collection_name = str(collection_name) if not isinstance(collection_name, str) else collection_name

        if isinstance(index_type, int):
            index_type = IndexType(index_type)
        if not isinstance(index_type, IndexType) or index_type == IndexType.INVALID:
            raise ParamError('Illegal index_type, should be IndexType but not IndexType.INVALID')

        self._collection_name = collection_name
        self._index_type = index_type

        # if isinstance(params, RepeatedCompositeContainer):
        #     self._params = ujson.loads(params[0].value)
        # else:
        self._params = ujson.loads(params)

    def __str__(self):
        attr_list = ['%s=%r' % (key.lstrip('_'), value)
                     for key, value in self.__dict__.items()]
        return '(%s)' % (', '.join(attr_list))

    def __repr__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    @property
    def collection_name(self):
        return self._collection_name

    @property
    def index_type(self):
        return self._index_type

    @property
    def params(self):
        return self._params


class PartitionParam:

    def __init__(self, collection_name, tag):
        self.collection_name = collection_name
        self.tag = tag

    def __repr__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '(%s)' % (', '.join(attr_list))


class SegmentStat:
    def __init__(self, res):
        self.segment_name = res.segment_name
        self.count = res.row_count
        self.index_name = res.index_name
        self.data_size = res.data_size

    def __str__(self):
        attr_list = ['%s: %r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __repr__(self):
        return self.__str__()


class PartitionStat:
    def __init__(self, res):
        self.tag = res.tag
        self.count = res.total_row_count
        self.segments_stat = [SegmentStat(s) for s in list(res.segments_stat)]

    def __str__(self):
        attr_list = ['%s: %r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __repr__(self):
        return self.__str__()


class CollectionInfo:
    def __init__(self, res):
        self.count = res.total_row_count
        self.partitions_stat = [PartitionStat(p) for p in list(res.partitions_stat)]

    def __str__(self):
        attr_list = ['%s: %r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __repr__(self):
        return self.__str__()


class HSegmentStat:
    def __init__(self, res):
        self.segment_name = res["segment_name"]
        self.count = res["count"]
        self.index_name = res["index"]
        self.data_size = res["size"]

    def __str__(self):
        attr_list = ['%s: %r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __repr__(self):
        return self.__str__()


class HPartitionStat:
    def __init__(self, res):
        self.tag = res["partition_tag"]
        self.count = res["count"]
        self.segments_stat = [HSegmentStat(s) for s in res["segments_stat"]]

    def __str__(self):
        attr_list = ['%s: %r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __repr__(self):
        return self.__str__()


class HCollectionInfo:
    def __init__(self, res):
        self.count = res["count"]
        self.partitions_stat = [HPartitionStat(p) for p in res["partitions_stat"]]

    def __str__(self):
        attr_list = ['%s: %r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __repr__(self):
        return self.__str__()


def _abstract():
    raise NotImplementedError('You need to override this function')


class ConnectIntf:
    """SDK client abstract class

    Connection is a abstract class

    """

    def connect(self, host, port, uri, timeout):
        """
        Connect method should be called before any operations
        Server will be connected after connect return OK
        Should be implemented

        :type  host: str
        :param host: host

        :type  port: str
        :param port: port

        :type  uri: str
        :param uri: (Optional) uri

        :type  timeout: int
        :param timeout:

        :return: Status,  indicate if connect is successful
        """
        _abstract()

    def connected(self):
        """
        connected, connection status
        Should be implemented

        :return: Status,  indicate if connect is successful
        """
        _abstract()

    def disconnect(self):
        """
        Disconnect, server will be disconnected after disconnect return SUCCESS
        Should be implemented

        :return: Status,  indicate if connect is successful
        """
        _abstract()

    def create_table(self, param, timeout):
        """
        Create table
        Should be implemented

        :type  param: TableSchema
        :param param: provide table information to be created

        :type  timeout: int
        :param timeout:

        :return: Status, indicate if connect is successful
        """
        _abstract()

    def has_table(self, table_name, timeout):
        """

        This method is used to test table existence.
        Should be implemented

        :type table_name: str
        :param table_name: table name is going to be tested.

        :type  timeout: int
        :param timeout:

        :return:
            has_table: bool, if given table_name exists

        """
        _abstract()

    def delete_table(self, table_name, timeout):
        """
        Delete table
        Should be implemented

        :type  table_name: str
        :param table_name: table_name of the deleting table

        :type  timeout: int
        :param timeout:

        :return: Status, indicate if connect is successful
        """
        _abstract()

    def add_vectors(self, table_name, records, ids, timeout, **kwargs):
        """
        Add vectors to table
        Should be implemented

        :type  table_name: str
        :param table_name: table name been inserted

        :type  records: list[RowRecord]
        :param records: list of vectors been inserted

        :type  ids: list[int]
        :param ids: list of ids

        :type  timeout: int
        :param timeout:

        :returns:
            Status : indicate if vectors inserted successfully
            ids :list of id, after inserted every vector is given a id
        """
        _abstract()

    def search_vectors(self, table_name, top_k, nprobe, query_records, query_ranges, **kwargs):
        """
        Query vectors in a table
        Should be implemented

        :type  table_name: str
        :param table_name: table name been queried

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  query_ranges: list[Range]
        :param query_ranges: Optional ranges for conditional search.
            If not specified, search whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        _abstract()

    def search_vectors_in_files(self, table_name, file_ids, query_records,
                                top_k, nprobe, query_ranges, **kwargs):
        """
        Query vectors in a table, query vector in specified files
        Should be implemented

        :type  table_name: str
        :param table_name: table name been queried

        :type  file_ids: list[str]
        :param file_ids: Specified files id array

        :type  query_records: list[RowRecord]
        :param query_records: all vectors going to be queried

        :type  query_ranges: list[Range]
        :param query_ranges: Optional ranges for conditional search.
            If not specified, search whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            query_results: list[TopKQueryResult]
        """
        _abstract()

    def describe_table(self, table_name, timeout):
        """
        Show table information
        Should be implemented

        :type  table_name: str
        :param table_name: which table to be shown

        :type  timeout: int
        :param timeout:

        :returns:
            Status: indicate if query is successful
            table_schema: TableSchema, given when operation is successful
        """
        _abstract()

    def get_table_row_count(self, table_name, timeout):
        """
        Get table row count
        Should be implemented

        :type  table_name, str
        :param table_name, target table name.

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :returns:
            Status: indicate if operation is successful
            count: int, table row count
        """
        _abstract()

    def show_tables(self, timeout):
        """
        Show all tables in database
        should be implemented

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if this operation is successful
            tables: list[str], list of table names
        """
        _abstract()

    def create_index(self, table_name, index, timeout):
        """
        Create specified index in a table
        should be implemented

        :type  table_name: str
        :param table_name: table name

         :type index: dict
        :param index: index information dict

            example: index = {
                "index_type": IndexType.FLAT,
                "nlist": 18384
            }

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if this operation is successful

        :rtype: Status
        """
        _abstract()

    def server_version(self, timeout):
        """
        Provide server version
        should be implemented

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        _abstract()

    def server_status(self, timeout):
        """
        Provide server status. When cmd !='version', provide 'OK'
        should be implemented

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        _abstract()

    def preload_table(self, table_name, timeout):
        """
        load table to memory cache in advance
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """

        _abstract()

    def describe_index(self, table_name, timeout):
        """
        Show index information
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

            TableSchema: table detail information

        :rtype: (Status, TableSchema)
        """

        _abstract()

    def drop_index(self, table_name, timeout):
        """
        Show index information
        should be implemented

        :param table_name: target table name.
        :type table_name: str

        :type  timeout: int
        :param timeout: how many similar vectors will be searched

        :return:
            Status: indicate if operation is successful

        ：:rtype: Status
        """

        _abstract()
