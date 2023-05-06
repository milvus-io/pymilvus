from .constants import OFFSET, LIMIT, ID, FIELDS, DISTANCE, RANGE_FILTER, RADIUS, PARAMS
from .types import DataType
from ..exceptions import (
    MilvusException,
)


class QueryIterator:

    def __init__(self, connection, collection_name, expr, output_fields=None, partition_names=None, schema=None,
                 timeout=None, **kwargs):
        self._conn = connection
        self._collection_name = collection_name
        self._expr = expr
        self._output_fields = output_fields
        self._partition_names = partition_names
        self._schema = schema
        self._timeout = timeout
        self._kwargs = kwargs
        self.__setup__pk_is_str()
        self.__seek()

    def __seek(self):
        if self._kwargs.get(OFFSET, 0) == 0:
            self._next_id = None
            return

        first_cursor_kwargs = self._kwargs.copy()
        first_cursor_kwargs[OFFSET] = 0
        # offset may be too large
        first_cursor_kwargs[LIMIT] = self._kwargs[OFFSET]

        res = self._conn.query(self._collection_name, self._expr, self._output_fields, self._partition_names,
                               timeout=self._timeout, **first_cursor_kwargs)
        self.__update_cursor(res)
        self._kwargs[OFFSET] = 0

    def next(self):
        current_expr = self.__setup_next_expr()
        res = self._conn.query(self._collection_name, current_expr, self._output_fields, self._partition_names,
                               timeout=self._timeout, **self._kwargs)
        self.__update_cursor(res)
        return res

    def __setup__pk_is_str(self):
        fields = self._schema[FIELDS]
        for field in fields:
            if field['is_primary']:
                if field['type'] == DataType.VARCHAR:
                    self._pk_str = True
                else:
                    self._pk_str = False
                break

    def __setup_next_expr(self):
        current_expr = self._expr
        if self._next_id is None:
            return current_expr
        if self._next_id is not None:
            if self._pk_str:
                current_expr = self._expr + f" and id > \"{self._next_id}\""
            else:
                current_expr = self._expr + f" and id > {self._next_id}"
        return current_expr

    def __update_cursor(self, res):
        if len(res) == 0:
            return
        self._next_id = res[-1][ID]

    def close(self):
        # do nothing for the moment, if iterator freeze snapshot on milvus server side
        # in the future, close function need to release the distribution snapshot accordingly
        return


class SearchIterator:

    def __init__(self, connection, collection_name, data, ann_field, param, limit, expr=None, partition_names=None,
                 output_fields=None, timeout=None, round_decimal=-1, schema=None, **kwargs):
        if len(data) > 1:
            raise MilvusException("Not support multiple vector iterator at present")
        self._conn = connection
        self._collection_name = collection_name
        self._data = data
        self._ann_field = ann_field
        self._param = param
        self._limit = limit
        self._expr = expr
        self._output_fields = output_fields
        self._partition_names = partition_names
        self._timeout = timeout
        self._round_decimal = round_decimal
        self._kwargs = kwargs
        self._distance_cursor = [0.0]
        self._last_id = [None]
        self._schema = schema
        self.__check_radius()
        self.__seek()
        self.__setup__pk_is_str()
        return

    def __setup__pk_is_str(self):
        fields = self._schema[FIELDS]
        for field in fields:
            if field['is_primary']:
                if field['type'] == DataType.VARCHAR:
                    self._pk_str = True
                else:
                    self._pk_str = False
                break

    def __check_radius(self):
        if self._param[PARAMS][RADIUS] is None:
            raise MilvusException(message="must provide radius parameter when using search iterator")

    def __seek(self):
        if self._kwargs.get(OFFSET, 0) != 0:
            raise MilvusException("Not support offset when searching iteration")
        '''
        if self._kwargs.get(OFFSET, 0) == 0:
            return

        first_cursor_kwargs = self._kwargs.copy()
        first_cursor_kwargs[OFFSET] = 0
        # offset may be too large
        first_cursor_kwargs[LIMIT] = self._kwargs[OFFSET]
        res = self._conn.search(self._collection_name, self._data, self._ann_field, self._param,
                                first_cursor_kwargs[LIMIT], self._expr, self._partition_names,
                                self._output_fields, self._round_decimal, timeout=self._timeout,
                                schema=self._schema, **self._kwargs)
        self.__update_cursor(res)
        self._kwargs[OFFSET] = 0
        '''

    def __update_cursor(self, res):
        if len(res[0]) == 0:
            return
        last_hit = res[0][-1]
        self._distance_cursor[0] = last_hit.distance
        self._last_id[0] = last_hit.id

    def next(self):
        next_params = self.__next_params()
        next_expr = self.__filtered_duplicated_result_expr(self._expr)
        res = self._conn.search(self._collection_name, self._data, self._ann_field, next_params,
                                self._limit, next_expr, self._partition_names,
                                self._output_fields, self._round_decimal, timeout=self._timeout,
                                schema=self._schema, **self._kwargs)
        self.__update_cursor(res)
        return res

    # at present, the range_filter parameter means 'larger/less and equal', so there would always
    # be one repeated result in every page, we need to refine and remove that result before returning
    def __filtered_duplicated_result_expr(self, expr):
        if self._last_id[0] is None:
            return expr
        filter_expr = None
        if self._pk_str:
            filter_expr = f"id != \"{self._last_id[0]}\""
        else:
            filter_expr = f"id != {self._last_id[0]}"
        if expr is not None:
            return expr + filter_expr
        else:
            return filter_expr

    def __next_params(self):
        next_params = self._param.copy()
        next_params[PARAMS][RANGE_FILTER] = self._distance_cursor[0]
        return next_params

    def close(self):
        pass
