import abc
import ujson

from ..client.exceptions import ParamError

from .check import check_pass_param

from .types import DataType
from . import blob

from ..grpc_gen import milvus_pb2
from ..grpc_gen import schema_pb2


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


class LoopCache(object):
    def __init__(self):
        self._array = []

    def fill(self, index, obj):
        if len(self._array) + 1 < index:
            pass


class FieldSchema:
    def __init__(self, raw):
        self._raw = raw

        #
        self.field_id = 0
        self.name = None
        self.is_primary = False
        self.description = None
        self.auto_id = False
        self.type = DataType.UNKNOWN
        self.indexes = list()
        self.params = dict()

        ##
        self.__pack(self._raw)

    def __pack(self, raw):
        self.field_id = raw.fieldID
        self.name = raw.name
        self.is_primary = raw.is_primary_key
        self.description = raw.description
        self.auto_id = raw.autoID
        self.type = raw.data_type
        # self.type = DataType(int(raw.type))

        for type_param in raw.type_params:
            if type_param.key == "params":
                import json
                self.params[type_param.key] = json.loads(type_param.value)
            else:
                self.params[type_param.key] = type_param.value
                if "dim" == type_param.key:
                    self.params[type_param.key] = int(type_param.value)

        index_dict = dict()
        for index_param in raw.index_params:
            if index_param.key == "params":
                import json
                index_dict[index_param.key] = json.loads(index_param.value)
            else:
                index_dict[index_param.key] = index_param.value

        self.indexes.extend([index_dict])

    def dict(self):
        _dict = dict()
        _dict["field_id"] = self.field_id
        _dict["name"] = self.name
        _dict["description"] = self.description
        _dict["type"] = self.type
        _dict["params"] = self.params or dict()
        _dict["is_primary"] = self.is_primary
        _dict["auto_id"] = self.auto_id
        return _dict


class CollectionSchema:
    def __init__(self, raw):
        self._raw = raw

        #
        self.collection_name = None
        self.description = None
        self.params = dict()
        self.fields = list()
        self.statistics = dict()
        self.auto_id = False  # auto_id is not in collection level any more later

        #
        if self._raw:
            self.__pack(self._raw)

    def __pack(self, raw):
        self.collection_name = raw.schema.name
        self.description = raw.schema.description
        # self.params = dict()
        # TODO: extra_params here
        # for kv in raw.extra_params:
        #     par = ujson.loads(kv.value)
        #     self.params.update(par)
        #     # self.params[kv.key] = kv.value

        for f in raw.schema.fields:
            self.fields.append(FieldSchema(f))

        # for s in raw.statistics:
        #     self.statistics[s.key] = s.value

    def dict(self):
        if not self._raw:
            return dict()
        _dict = dict()
        _dict["collection_name"] = self.collection_name
        _dict["auto_id"] = self.auto_id
        _dict["description"] = self.description
        _dict["fields"] = [f.dict() for f in self.fields]
        # for k, v in self.params.items():
        #     if isinstance(v, DataType):
        #         _dict[k] = v.value
        #     else:
        #         _dict[k] = v

        return _dict


class Entity:
    def __init__(self, entity_id, entity_row_data, entity_score):
        self._id = entity_id
        self._row_data = entity_row_data
        self._score = entity_score
        self._distance = entity_score

    def __str__(self):
        str_ = 'id: {}, distance: {}, entity: {},'.format(self._id, self._distance, self._row_data)
        return str_

    def __getattr__(self, item):
        return self.value_of_field(item)

    @property
    def id(self):
        return self._id

    @property
    def fields(self):
        fields = []
        for k, v in self._row_data.items():
            fields.append(k)
        return fields

    def get(self, field):
        return self.value_of_field(field)

    def value_of_field(self, field):
        if field in self._row_data:
            return self._row_data[field]
        else:
            raise BaseException(0, "Field {} is not in return entity".format(field))

    def type_of_field(self, field):
        raise NotImplementedError('TODO: support field in Hits')


class Hit:
    def __init__(self, entity_id, entity_row_data, entity_score):
        self._id = entity_id
        self._row_data = entity_row_data
        self._score = entity_score
        self._distance = entity_score

    def __str__(self):
        return "(distance: {}, score: {}, id: {})".format(self._distance, self._score, self._id)

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


class Hits(LoopBase):
    def __init__(self, raw, auto_id):
        super().__init__()
        self._raw = raw
        self._auto_id = auto_id
        self._distances = self._raw.scores
        self._entities = []
        self._pack(self._raw)

    def _pack(self, raw):
        self._entities = [item for item in self]

    def __len__(self):
        return len(self._raw.ids.int_id.data)

    def get__item(self, item):
        entity_id = self._raw.ids.int_id.data[item]
        entity_row_data = dict()
        if self._raw.fields_data:
            for field_data in self._raw.fields_data:
                if field_data.type == DataType.BOOL:
                    raise BaseException(0, "Not support bool yet")
                    # result[field_data.name] = field_data.field.scalars.data.bool_data[index]
                elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                    if len(field_data.scalars.int_data.data) >= item:
                        entity_row_data[field_data.field_name] = field_data.scalars.int_data.data[item]
                elif field_data.type == DataType.INT64:
                    if len(field_data.scalars.long_data.data) >= item:
                        entity_row_data[field_data.field_name] = field_data.scalars.long_data.data[item]
                elif field_data.type == DataType.FLOAT:
                    if len(field_data.scalars.float_data.data) >= item:
                        entity_row_data[field_data.field_name] = round(field_data.scalars.float_data.data[item], 6)
                elif field_data.type == DataType.DOUBLE:
                    if len(field_data.scalars.double_data.data) >= item:
                        entity_row_data[field_data.field_name] = field_data.scalars.double_data.data[item]
                elif field_data.type == DataType.STRING:
                    raise BaseException(0, "Not support string yet")
                    # result[field_data.field_name] = field_data.scalars.string_data.data[index]
                elif field_data.type == DataType.FLOAT_VECTOR:
                    dim = field_data.vectors.dim
                    if len(field_data.vectors.float_vector.data) >= item * dim:
                        start_pos = item * dim
                        end_pos = item * dim + dim
                        entity_row_data[field_data.field_name] = [round(x, 6) for x in
                                                                  field_data.vectors.float_vector.data[
                                                                  start_pos:end_pos]]
                elif field_data.type == DataType.BINARY_VECTOR:
                    dim = field_data.vectors.dim
                    if len(field_data.vectors.binary_vector.data) >= item * (dim / 8):
                        start_pos = item * (dim / 8)
                        end_pos = (item + 1) * (dim / 8)
                        entity_row_data[field_data.field_name] = [
                            field_data.vectors.binary_vector.data[start_pos:end_pos]]
        entity_score = self._raw.scores[item]
        return Hit(entity_id, entity_row_data, entity_score)

    @property
    def ids(self):
        return self._raw.ids.int_id.data

    @property
    def distances(self):
        return self._raw.scores


class MutationResult:
    def __init__(self, raw):
        self._raw = raw
        self._primary_keys = list()
        self._insert_cnt = 0
        self._delete_cnt = 0
        self._upsert_cnt = 0
        self._timestamp = 0
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

    # TODO
    # def error_code(self):
    #     pass
    #
    # def error_reason(self):
    #     pass

    def _pack(self, raw):
        # self._primary_keys = getattr(raw.IDs, raw.IDs.WhichOneof('id_field')).value.data
        which = raw.IDs.WhichOneof("id_field")
        if which == "int_id":
            self._primary_keys = raw.IDs.int_id.data
        elif which == "str_id":
            self._primary_keys = raw.IDs.str_id.data

        self._insert_cnt = raw.insert_cnt
        self._delete_cnt = raw.delete_cnt
        self._upsert_cnt = raw.upsert_cnt
        self._timestamp = raw.timestamp


class QueryResult(LoopBase):
    def __init__(self, raw, auto_id=True):
        super().__init__()
        self._raw = raw
        self._auto_id = auto_id
        self._pack(raw.hits)

    def __len__(self):
        return self._nq

    def __len(self):
        return self._nq

    def _pack(self, raw):
        self._nq = raw.results.num_queries
        self._topk = raw.results.top_k
        self._hits = []
        offset = 0
        for i in range(self._nq):
            hit = schema_pb2.SearchResultData()
            start_pos = offset
            end_pos = offset + raw.results.topks[i]
            hit.scores.append(raw.results.scores[start_pos: end_pos])
            hit.ids.append(raw.results.ids.int_id.data[start_pos: end_pos])
            for field_data in raw.result.fields_data:
                field = schema_pb2.FieldData()
                field.type = field_data.type
                field.field_name = field_data.field_name
                if field_data.type == DataType.BOOL:
                    raise BaseException(0, "Not support bool yet")
                    # result[field_data.name] = field_data.field.scalars.data.bool_data[index]
                elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                    field.scalars.int_data.data.extend(field_data.scalars.int_data.data[start_pos: end_pos])
                elif field_data.type == DataType.INT64:
                    field.scalars.long_data.data.extend(field_data.scalars.long_data.data[start_pos: end_pos])
                elif field_data.type == DataType.FLOAT:
                    field.scalars.float_data.data.extend(field_data.scalars.float_data.data[start_pos: end_pos])
                elif field_data.type == DataType.DOUBLE:
                    field.scalars.double_data.data.extend(field_data.scalars.double_data.data[start_pos: end_pos])
                elif field_data.type == DataType.STRING:
                    raise BaseException(0, "Not support string yet")
                    # result[field_data.field_name] = field_data.scalars.string_data.data[index]
                elif field_data.type == DataType.FLOAT_VECTOR:
                    dim = field.vectors.dim
                    field.vectors.dim = dim
                    field.vectors.float_vector.data.extend(
                        field_data.vectors.float_data.data[start_pos * dim: end_pos * dim])
                elif field_data.type == DataType.BINARY_VECTOR:
                    dim = field_data.vectors.dim
                    field.vectors.dim = dim
                    field.vectors.binary_vector.data.extend(field_data.vectors.binary_vector.data[
                                                            start_pos * (dim / 8): end_pos * (dim / 8)])
                hit.fields_data.append(field)
            self._hits.append(hit)
            offset += raw.results.topks[i]

    def get__item(self, item):
        return Hits(self._hits[item], self._auto_id)


class ChunkedQueryResult(LoopBase):
    def __init__(self, raw_list, auto_id=True):
        super().__init__()
        self._raw_list = raw_list
        self._auto_id = auto_id
        self._nq = 0

        self._pack(self._raw_list)

    def __len__(self):
        return self._nq

    def __len(self):
        return self._nq

    def _pack(self, raw_list):
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
                hit.scores.extend(raw.results.scores[start_pos: end_pos])
                hit.ids.int_id.data.extend(raw.results.ids.int_id.data[start_pos: end_pos])
                for field_data in raw.results.fields_data:
                    field = schema_pb2.FieldData()
                    field.type = field_data.type
                    field.field_name = field_data.field_name
                    if field_data.type == DataType.BOOL:
                        raise BaseException(0, "Not support bool yet")
                        # result[field_data.name] = field_data.field.scalars.data.bool_data[index]
                    elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                        field.scalars.int_data.data.extend(field_data.scalars.int_data.data[start_pos: end_pos])
                    elif field_data.type == DataType.INT64:
                        field.scalars.long_data.data.extend(field_data.scalars.long_data.data[start_pos: end_pos])
                    elif field_data.type == DataType.FLOAT:
                        field.scalars.float_data.data.extend(field_data.scalars.float_data.data[start_pos: end_pos])
                    elif field_data.type == DataType.DOUBLE:
                        field.scalars.double_data.data.extend(field_data.scalars.double_data.data[start_pos: end_pos])
                    elif field_data.type == DataType.STRING:
                        raise BaseException(0, "Not support string yet")
                        # result[field_data.field_name] = field_data.scalars.string_data.data[index]
                    elif field_data.type == DataType.FLOAT_VECTOR:
                        dim = field_data.vectors.dim
                        field.vectors.dim = dim
                        field.vectors.float_vector.data.extend(field_data.vectors.float_vector.data[
                                                               start_pos * dim: end_pos * dim])
                    elif field_data.type == DataType.BINARY_VECTOR:
                        dim = field_data.vectors.dim
                        field.vectors.dim = dim
                        field.vectors.binary_vector.data.extend(field_data.vectors.binary_vector.data[
                                                                start_pos * (dim / 8): end_pos * (dim / 8)])
                    hit.fields_data.append(field)
                self._hits.append(hit)
                offset += raw.results.topks[i]

    def get__item(self, item):
        return Hits(self._hits[item], self._auto_id)


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

    def load_collection(self, collection_name, timeout):
        _abstract()

    def release_collection(self, collection_name, timeout):
        _abstract()

    def load_partitions(self, collection_name, timeout):
        _abstract()

    def release_partitions(self, collection_name, timeout):
        _abstract()
