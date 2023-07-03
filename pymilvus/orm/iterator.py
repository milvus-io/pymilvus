from typing import Any, Dict, List, Optional, TypeVar

from pymilvus.exceptions import MilvusException

from .connections import Connections
from .constants import (
    CALC_DIST_COSINE,
    CALC_DIST_HAMMING,
    CALC_DIST_IP,
    CALC_DIST_JACCARD,
    CALC_DIST_L2,
    CALC_DIST_TANIMOTO,
    DEFAULT_MAX_HAMMING_DISTANCE,
    DEFAULT_MAX_JACCARD_DISTANCE,
    DEFAULT_MAX_L2_DISTANCE,
    DEFAULT_MAX_TANIMOTO_DISTANCE,
    DEFAULT_MIN_COSINE_DISTANCE,
    DEFAULT_MIN_IP_DISTANCE,
    FIELDS,
    ITERATION_EXTENSION_REDUCE_RATE,
    LIMIT,
    MAX_FILTERED_IDS_COUNT_ITERATION,
    METRIC_TYPE,
    OFFSET,
    PARAMS,
    RADIUS,
    RANGE_FILTER,
)
from .schema import CollectionSchema
from .types import DataType

QueryIterator = TypeVar("QueryIterator")
SearchIterator = TypeVar("SearchIterator")


class QueryIterator:
    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        schema: Optional[CollectionSchema] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> QueryIterator:
        self._conn = connection
        self._collection_name = collection_name
        self._expr = expr
        self._output_fields = output_fields
        self._partition_names = partition_names
        self._schema = schema
        self._timeout = timeout
        self._kwargs = kwargs
        self.__setup__pk_prop()
        self.__seek()
        self._cache_id_in_use = NO_CACHE_ID

    def __seek(self):
        self._cache_id_in_use = NO_CACHE_ID
        if self._kwargs.get(OFFSET, 0) == 0:
            self._next_id = None
            return

        first_cursor_kwargs = self._kwargs.copy()
        first_cursor_kwargs[OFFSET] = 0
        # offset may be too large, needed to seek in multiple times
        first_cursor_kwargs[LIMIT] = self._kwargs[OFFSET]
        first_cursor_kwargs[ITERATION_EXTENSION_REDUCE_RATE] = 0

        res = self._conn.query(
            collection_name=self._collection_name,
            expr=self._expr,
            output_field=self._output_fields,
            partition_name=self._partition_names,
            timeout=self._timeout,
            **first_cursor_kwargs,
        )
        self.__update_cursor(res)
        self._kwargs[OFFSET] = 0

    def __maybe_cache(self, result: List):
        if len(result) < 2 * self._kwargs[LIMIT]:
            return
        start = self._kwargs[LIMIT]
        cache_result = result[start:]
        cache_id = iterator_cache.cache(cache_result, NO_CACHE_ID)
        self._cache_id_in_use = cache_id

    def __is_res_sufficient(self, res: List):
        return res is not None and len(res) >= self._kwargs[LIMIT]

    def next(self):
        cached_res = iterator_cache.fetch_cache(self._cache_id_in_use)
        ret = None
        if self.__is_res_sufficient(cached_res):
            ret = cached_res[0 : self._kwargs[LIMIT]]
            res_to_cache = cached_res[self._kwargs[LIMIT] :]
            iterator_cache.cache(res_to_cache, self._cache_id_in_use)
        else:
            iterator_cache.release_cache(self._cache_id_in_use)
            current_expr = self.__setup_next_expr()
            res = self._conn.query(
                collection_name=self._collection_name,
                expr=current_expr,
                output_fields=self._output_fields,
                partition_names=self._partition_names,
                timeout=self._timeout,
                **self._kwargs,
            )
            self.__maybe_cache(res)
            ret = res[0 : min(self._kwargs[LIMIT], len(res))]
        self.__update_cursor(ret)
        return ret

    def __setup__pk_prop(self):
        fields = self._schema[FIELDS]
        for field in fields:
            if field["is_primary"]:
                if field["type"] == DataType.VARCHAR:
                    self._pk_str = True
                else:
                    self._pk_str = False
                self._pk_field_name = field["name"]
                break
        if self._pk_field_name is None or self._pk_field_name == "":
            raise MilvusException(message="schema must contain pk field, broke")

    def __setup_next_expr(self) -> None:
        current_expr = self._expr
        if self._next_id is None:
            return current_expr
        filtered_pk_str = ""
        if self._pk_str:
            filtered_pk_str = f'{self._pk_field_name} > "{self._next_id}"'
        else:
            filtered_pk_str = f"{self._pk_field_name} > {self._next_id}"
        if current_expr is None or len(current_expr) == 0:
            return filtered_pk_str
        return current_expr + " and " + filtered_pk_str

    def __update_cursor(self, res: List) -> None:
        if len(res) == 0:
            return
        self._next_id = res[-1][self._pk_field_name]

    def close(self) -> None:
        # release cache in use
        iterator_cache.release_cache(self._cache_id_in_use)


def default_radius(metrics: str):
    if metrics is CALC_DIST_L2:
        return DEFAULT_MAX_L2_DISTANCE
    if metrics is CALC_DIST_IP:
        return DEFAULT_MIN_IP_DISTANCE
    if metrics is CALC_DIST_HAMMING:
        return DEFAULT_MAX_HAMMING_DISTANCE
    if metrics is CALC_DIST_TANIMOTO:
        return DEFAULT_MAX_TANIMOTO_DISTANCE
    if metrics is CALC_DIST_JACCARD:
        return DEFAULT_MAX_JACCARD_DISTANCE
    if metrics is CALC_DIST_COSINE:
        return DEFAULT_MIN_COSINE_DISTANCE
    raise MilvusException(message="unknown metrics type for search iteration")


class SearchIterator:
    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        data: List,
        ann_field: str,
        param: Dict,
        limit: int,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        round_decimal: int = -1,
        schema: Optional[CollectionSchema] = None,
        **kwargs,
    ) -> SearchIterator:
        if len(data) > 1:
            raise MilvusException(message="Not support multiple vector iterator at present")
        self._conn = connection
        self._iterator_params = {
            "collection_name": collection_name,
            "data": data,
            "ann_field": ann_field,
            "limit": limit,
            "output_fields": output_fields,
            "partition_names": partition_names,
            "timeout": timeout,
            "round_decimal": round_decimal,
        }
        self._expr = expr
        self._param = param
        self._kwargs = kwargs
        self._distance_cursor = [None]
        self._filtered_ids = []
        self._filtered_distance = None
        self._schema = schema
        self.__check_metrics()
        self.__check_radius()
        self.__seek()
        self.__setup__pk_prop()

    def __setup__pk_prop(self):
        fields = self._schema[FIELDS]
        for field in fields:
            if field["is_primary"]:
                if field["type"] == DataType.VARCHAR:
                    self._pk_str = True
                else:
                    self._pk_str = False
                self._pk_field_name = field["name"]
                break
        if self._pk_field_name is None or self._pk_field_name == "":
            raise MilvusException(message="schema must contain pk field, broke")

    def __check_metrics(self):
        if self._param[METRIC_TYPE] is None or self._param[METRIC_TYPE] == "":
            raise MilvusException(message="must specify metrics type for search iterator")

    def __check_radius(self):
        if PARAMS not in self._param:
            self._param[PARAMS] = {"radius": default_radius(self._param[METRIC_TYPE])}
        elif RADIUS not in self._param[PARAMS]:
            self._param[PARAMS][RADIUS] = default_radius(self._param[METRIC_TYPE])

    def __seek(self):
        if self._kwargs.get(OFFSET, 0) != 0:
            raise MilvusException(message="Not support offset when searching iteration")

    def __update_cursor(self, res: Any):
        if len(res[0]) == 0:
            return
        last_hit = res[0][-1]
        if last_hit is None:
            return
        self._distance_cursor[0] = last_hit.distance
        if self._distance_cursor[0] != self._filtered_distance:
            self._filtered_ids = []  # distance has changed, clear filter_ids array
            self._filtered_distance = self._distance_cursor[0]  # renew the distance for filtering
        for hit in res[0]:
            if hit.distance == last_hit.distance:
                self._filtered_ids.append(hit.id)
        if len(self._filtered_ids) > MAX_FILTERED_IDS_COUNT_ITERATION:
            raise MilvusException(
                message=f"filtered ids length has accumulated to more than "
                f"{MAX_FILTERED_IDS_COUNT_ITERATION!s}, "
                f"there is a danger of overly memory consumption"
            )

    def next(self):
        next_params = self.__next_params()
        next_expr = self.__filtered_duplicated_result_expr(self._expr)
        res = self._conn.search(
            self._iterator_params["collection_name"],
            self._iterator_params["data"],
            self._iterator_params["ann_field"],
            next_params,
            self._iterator_params["limit"],
            next_expr,
            self._iterator_params["partition_names"],
            self._iterator_params["output_fields"],
            self._iterator_params["round_decimal"],
            timeout=self._iterator_params["timeout"],
            schema=self._schema,
            **self._kwargs,
        )
        self.__update_cursor(res)
        return res

    # at present, the range_filter parameter means 'larger/less and equal',
    # so there would be vectors with same distances returned multiple times in different pages
    # we need to refine and remove these results before returning
    def __filtered_duplicated_result_expr(self, expr: str):
        if len(self._filtered_ids) == 0:
            return expr

        filtered_ids_str = ""
        for filtered_id in self._filtered_ids:
            if self._pk_str:
                filtered_ids_str += f'"{filtered_id}",'
            else:
                filtered_ids_str += f"{filtered_id},"
        filtered_ids_str = filtered_ids_str[0:-1]

        if len(filtered_ids_str) > 0:
            if expr is not None and len(expr) > 0:
                filter_expr = f" and {self._pk_field_name} not in [{filtered_ids_str}]"
                return expr + filter_expr
            return f"{self._pk_field_name} not in [{filtered_ids_str}]"
        return expr

    def __next_params(self):
        next_params = self._param.copy()
        if self._distance_cursor[0] is not None:
            next_params[PARAMS][RANGE_FILTER] = self._distance_cursor[0]
        return next_params

    def close(self):
        pass


class IteratorCache:
    def __init__(self) -> None:
        self._cache_id = 0
        self._cache_map = {}

    def cache(self, result: Any, cache_id: int):
        if cache_id == NO_CACHE_ID:
            self._cache_id += 1
            cache_id = self._cache_id
        self._cache_map[cache_id] = result
        return cache_id

    def fetch_cache(self, cache_id: int):
        return self._cache_map.get(cache_id, None)

    def release_cache(self, cache_id: int):
        if self._cache_map.get(cache_id, None) is not None:
            self._cache_map.pop(cache_id)


NO_CACHE_ID = -1
# Singleton Mode in Python
iterator_cache = IteratorCache()
