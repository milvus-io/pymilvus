import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, TypeVar, Union

from pymilvus.client import entity_helper, utils
from pymilvus.client.abstract import Hits, LoopBase
from pymilvus.exceptions import (
    MilvusException,
    ParamError,
)

from .connections import Connections
from .constants import (
    BATCH_SIZE,
    CALC_DIST_COSINE,
    CALC_DIST_HAMMING,
    CALC_DIST_IP,
    CALC_DIST_JACCARD,
    CALC_DIST_L2,
    CALC_DIST_TANIMOTO,
    DEFAULT_SEARCH_EXTENSION_RATE,
    EF,
    FIELDS,
    INT64_MAX,
    IS_PRIMARY,
    ITERATOR_FIELD,
    MAX_BATCH_SIZE,
    MAX_FILTERED_IDS_COUNT_ITERATION,
    MAX_TRY_TIME,
    METRIC_TYPE,
    MILVUS_LIMIT,
    OFFSET,
    PARAMS,
    RADIUS,
    RANGE_FILTER,
    REDUCE_STOP_FOR_BEST,
    UNLIMITED,
)
from .schema import CollectionSchema
from .types import DataType

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.ERROR)
QueryIterator = TypeVar("QueryIterator")
SearchIterator = TypeVar("SearchIterator")


def extend_batch_size(batch_size: int, next_param: dict, to_extend_batch_size: bool) -> int:
    extend_rate = 1
    if to_extend_batch_size:
        extend_rate = DEFAULT_SEARCH_EXTENSION_RATE
    if EF in next_param[PARAMS]:
        real_batch = min(MAX_BATCH_SIZE, batch_size * extend_rate, next_param[PARAMS][EF])
        next_param[PARAMS][EF] = min(real_batch, next_param[PARAMS][EF])
        return real_batch
    return min(MAX_BATCH_SIZE, batch_size * extend_rate)


class QueryIterator:
    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        batch_size: Optional[int] = 1000,
        limit: Optional[int] = -1,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        schema: Optional[CollectionSchema] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> QueryIterator:
        self._conn = connection
        self._collection_name = collection_name
        self._output_fields = output_fields
        self._partition_names = partition_names
        self._schema = schema
        self._timeout = timeout
        self._kwargs = kwargs
        self.__set_up_iteration_states()
        self.__check_set_batch_size(batch_size)
        self._limit = limit
        self.__check_set_reduce_stop_for_best()
        self._returned_count = 0
        self.__setup__pk_prop()
        self.__set_up_expr(expr)
        self.__seek()
        self._cache_id_in_use = NO_CACHE_ID

    def __set_up_iteration_states(self):
        self._kwargs[ITERATOR_FIELD] = "True"

    def __check_set_reduce_stop_for_best(self):
        if self._kwargs.get(REDUCE_STOP_FOR_BEST, True):
            self._kwargs[REDUCE_STOP_FOR_BEST] = "True"
        else:
            self._kwargs[REDUCE_STOP_FOR_BEST] = "False"

    def __check_set_batch_size(self, batch_size: int):
        if batch_size < 0:
            raise ParamError(message="batch size cannot be less than zero")
        if batch_size > MAX_BATCH_SIZE:
            raise ParamError(message=f"batch size cannot be larger than {MAX_BATCH_SIZE}")
        self._kwargs[BATCH_SIZE] = batch_size
        self._kwargs[MILVUS_LIMIT] = batch_size

    # rely on pk prop, so this method should be called after __set_up_expr
    def __set_up_expr(self, expr: str):
        if expr is not None:
            self._expr = expr
        elif self._pk_str:
            self._expr = self._pk_field_name + ' != ""'
        else:
            self._expr = self._pk_field_name + " < " + str(INT64_MAX)

    def __seek(self):
        self._cache_id_in_use = NO_CACHE_ID
        if self._kwargs.get(OFFSET, 0) == 0:
            self._next_id = None
            return

        first_cursor_kwargs = self._kwargs.copy()
        first_cursor_kwargs[OFFSET] = 0
        # offset may be too large, needed to seek in multiple times
        first_cursor_kwargs[MILVUS_LIMIT] = self._kwargs[OFFSET]

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
        if len(result) < 2 * self._kwargs[BATCH_SIZE]:
            return
        start = self._kwargs[BATCH_SIZE]
        cache_result = result[start:]
        cache_id = iterator_cache.cache(cache_result, NO_CACHE_ID)
        self._cache_id_in_use = cache_id

    def __is_res_sufficient(self, res: List):
        return res is not None and len(res) >= self._kwargs[BATCH_SIZE]

    def next(self):
        cached_res = iterator_cache.fetch_cache(self._cache_id_in_use)
        ret = None
        if self.__is_res_sufficient(cached_res):
            ret = cached_res[0 : self._kwargs[BATCH_SIZE]]
            res_to_cache = cached_res[self._kwargs[BATCH_SIZE] :]
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
            ret = res[0 : min(self._kwargs[BATCH_SIZE], len(res))]

        ret = self.__check_reached_limit(ret)
        self.__update_cursor(ret)
        self._returned_count += len(ret)
        return ret

    def __check_reached_limit(self, ret: List):
        if self._limit == UNLIMITED:
            return ret
        left_count = self._limit - self._returned_count
        if left_count >= len(ret):
            return ret
        # has exceeded the limit, cut off the result and return
        return ret[0:left_count]

    def __setup__pk_prop(self):
        fields = self._schema[FIELDS]
        for field in fields:
            if field.get(IS_PRIMARY):
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
        return "(" + current_expr + ")" + " and " + filtered_pk_str

    def __update_cursor(self, res: List) -> None:
        if len(res) == 0:
            return
        self._next_id = res[-1][self._pk_field_name]

    def close(self) -> None:
        # release cache in use
        iterator_cache.release_cache(self._cache_id_in_use)


def metrics_positive_related(metrics: str) -> bool:
    if metrics in [CALC_DIST_L2, CALC_DIST_JACCARD, CALC_DIST_HAMMING, CALC_DIST_TANIMOTO]:
        return True
    if metrics in [CALC_DIST_IP, CALC_DIST_COSINE]:
        return False
    raise MilvusException(message=f"unsupported metrics type for search iteration: {metrics}")


class SearchPage(LoopBase):
    """Since we only support nq=1 in search iteration, so search iteration response
    should be different from raw response of search operation"""

    def __init__(self, res: Hits):
        super().__init__()
        self._results = []
        if res is not None:
            self._results.append(res)

    def get_res(self):
        return self._results

    def __len__(self):
        length = 0
        for res in self._results:
            length += len(res)
        return length

    def get__item(self, idx: Any):
        if len(self._results) == 0:
            return None
        if idx >= self.__len__():
            msg = "Index out of range"
            raise IndexError(msg)
        index = 0
        ret = None
        for res in self._results:
            if index + len(res) <= idx:
                index += len(res)
            else:
                ret = res[idx - index]
                break
        return ret

    def merge(self, others: List[Hits]):
        if others is not None:
            for other in others:
                self._results.append(other)

    def ids(self):
        ids = []
        for res in self._results:
            for hit in res:
                ids.append(hit.id)
        return ids

    def distances(self):
        distances = []
        for res in self._results:
            for hit in res:
                distances.append(hit.distance)
        return distances


class SearchIterator:
    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        data: Union[List, utils.SparseMatrixInputType],
        ann_field: str,
        param: Dict,
        batch_size: Optional[int] = 1000,
        limit: Optional[int] = UNLIMITED,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        round_decimal: int = -1,
        schema: Optional[CollectionSchema] = None,
        **kwargs,
    ) -> SearchIterator:
        rows = entity_helper.get_input_num_rows(data)
        if rows > 1:
            raise ParamError(
                message="Not support search iteration over multiple vectors at present"
            )
        if rows == 0:
            raise ParamError(message="vector_data for search cannot be empty")
        self._conn = connection
        self._iterator_params = {
            "collection_name": collection_name,
            "data": data,
            "ann_field": ann_field,
            BATCH_SIZE: batch_size,
            "output_fields": output_fields,
            "partition_names": partition_names,
            "timeout": timeout,
            "round_decimal": round_decimal,
        }
        self._expr = expr
        self.__check_set_params(param)
        self.__check_for_special_index_param()
        self._kwargs = kwargs
        self.__set_up_iteration_states()
        self._filtered_ids = []
        self._filtered_distance = None
        self._schema = schema
        self._limit = limit
        self._returned_count = 0
        self.__check_metrics()
        self.__check_offset()
        self.__check_rm_range_search_parameters()
        self.__setup__pk_prop()
        self.__init_search_iterator()

    def __init_search_iterator(self):
        init_page = self.__execute_next_search(self._param, self._expr, False)
        if len(init_page) == 0:
            message = (
                "Cannot init search iterator because init page contains no matched rows, "
                "please check the radius and range_filter set up by searchParams"
            )
            LOGGER.error(message)
            self._cache_id = NO_CACHE_ID
            self._init_success = False
            return
        self._cache_id = iterator_cache.cache(init_page, NO_CACHE_ID)
        self.__set_up_range_parameters(init_page)
        self.__update_filtered_ids(init_page)
        self._init_success = True

    def __set_up_iteration_states(self):
        self._kwargs[ITERATOR_FIELD] = "True"

    def __update_width(self, page: SearchPage):
        first_hit, last_hit = page[0], page[-1]
        if metrics_positive_related(self._param[METRIC_TYPE]):
            self._width = last_hit.distance - first_hit.distance
        else:
            self._width = first_hit.distance - last_hit.distance
        if self._width == 0.0:
            self._width = 0.05
            # enable a minimum value for width to avoid radius and range_filter equal error

    def __set_up_range_parameters(self, page: SearchPage):
        self.__update_width(page)
        self._tail_band = page[-1].distance
        LOGGER.debug(
            f"set up init parameter for searchIterator width:{self._width} tail_band:{self._tail_band}"
        )

    def __check_reached_limit(self) -> bool:
        if self._limit == UNLIMITED or self._returned_count < self._limit:
            return False
        LOGGER.debug(
            f"reached search limit:{self._limit}, returned_count:{self._returned_count}, directly return"
        )
        return True

    def __check_set_params(self, param: Dict):
        if param is None:
            self._param = {}
        else:
            self._param = deepcopy(param)
        if PARAMS not in self._param:
            self._param[PARAMS] = {}

    def __check_for_special_index_param(self):
        if (
            EF in self._param[PARAMS]
            and self._param[PARAMS][EF] < self._iterator_params[BATCH_SIZE]
        ):
            raise MilvusException(
                message="When using hnsw index, provided ef must be larger than or equal to batch size"
            )

    def __setup__pk_prop(self):
        fields = self._schema[FIELDS]
        for field in fields:
            if field.get(IS_PRIMARY):
                if field["type"] == DataType.VARCHAR:
                    self._pk_str = True
                else:
                    self._pk_str = False
                self._pk_field_name = field["name"]
                break
        if self._pk_field_name is None or self._pk_field_name == "":
            raise ParamError(message="schema must contain pk field, broke")

    def __check_metrics(self):
        if self._param[METRIC_TYPE] is None or self._param[METRIC_TYPE] == "":
            raise ParamError(message="must specify metrics type for search iterator")

    """we use search && range search to implement search iterator,
    so range search parameters are disabled to clients"""

    def __check_rm_range_search_parameters(self):
        if (
            (PARAMS in self._param)
            and (RADIUS in self._param[PARAMS])
            and (RANGE_FILTER in self._param[PARAMS])
        ):
            radius = self._param[PARAMS][RADIUS]
            range_filter = self._param[PARAMS][RANGE_FILTER]
            if metrics_positive_related(self._param[METRIC_TYPE]) and radius <= range_filter:
                raise MilvusException(
                    message=f"for metrics:{self._param[METRIC_TYPE]}, radius must be "
                    f"larger than range_filter, please adjust your parameter"
                )
            if not metrics_positive_related(self._param[METRIC_TYPE]) and radius >= range_filter:
                raise MilvusException(
                    message=f"for metrics:{self._param[METRIC_TYPE]}, radius must be "
                    f"smalled than range_filter, please adjust your parameter"
                )

    def __check_offset(self):
        if self._kwargs.get(OFFSET, 0) != 0:
            raise ParamError(message="Not support offset when searching iteration")

    def __update_filtered_ids(self, res: SearchPage):
        if len(res) == 0:
            return
        last_hit = res[-1]
        if last_hit is None:
            return
        if last_hit.distance != self._filtered_distance:
            self._filtered_ids = []  # distance has changed, clear filter_ids array
            self._filtered_distance = last_hit.distance  # renew the distance for filtering
        for hit in res:
            if hit.distance == last_hit.distance:
                self._filtered_ids.append(hit.id)
        if len(self._filtered_ids) > MAX_FILTERED_IDS_COUNT_ITERATION:
            raise MilvusException(
                message=f"filtered ids length has accumulated to more than "
                f"{MAX_FILTERED_IDS_COUNT_ITERATION!s}, "
                f"there is a danger of overly memory consumption"
            )

    def __is_cache_enough(self, count: int) -> bool:
        cached_page = iterator_cache.fetch_cache(self._cache_id)
        return cached_page is not None and len(cached_page) >= count

    def __extract_page_from_cache(self, count: int) -> SearchPage:
        cached_page = iterator_cache.fetch_cache(self._cache_id)
        if cached_page is None or len(cached_page) < count:
            raise ParamError(
                message=f"Wrong, try to extract {count} result from cache, "
                f"more than {len(cached_page)} there must be sth wrong with code"
            )

        ret_page_res = cached_page[0:count]
        ret_page = SearchPage(ret_page_res)
        left_cache_page = SearchPage(cached_page[count:])
        iterator_cache.cache(left_cache_page, self._cache_id)
        return ret_page

    def __push_new_page_to_cache(self, page: SearchPage) -> int:
        if page is None:
            raise ParamError(message="Cannot push None page into cache")
        cached_page: SearchPage = iterator_cache.fetch_cache(self._cache_id)
        if cached_page is None:
            iterator_cache.cache(page, self._cache_id)
            cached_page = page
        else:
            cached_page.merge(page.get_res())
        return len(cached_page)

    def next(self):
        # 0. check reached limit
        if not self._init_success or self.__check_reached_limit():
            return SearchPage(None)
        ret_len = self._iterator_params[BATCH_SIZE]
        if self._limit is not UNLIMITED:
            left_len = self._limit - self._returned_count
            ret_len = min(left_len, ret_len)

        # 1. if cached page is sufficient, directly return
        if self.__is_cache_enough(ret_len):
            ret_page = self.__extract_page_from_cache(ret_len)
            self._returned_count += len(ret_page)
            return ret_page

        # 2. if cached page not enough, try to fill the result by probing with constant width
        # until finish filling or exceeding max trial time: 10
        new_page = self.__try_search_fill()
        cached_page_len = self.__push_new_page_to_cache(new_page)
        ret_len = min(cached_page_len, ret_len)
        ret_page = self.__extract_page_from_cache(ret_len)
        if len(ret_page) == self._iterator_params[BATCH_SIZE]:
            self.__update_width(ret_page)

        # 3. update filter ids to avoid returning result repeatedly
        self._returned_count += ret_len
        return ret_page

    def __try_search_fill(self) -> SearchPage:
        final_page = SearchPage(None)
        try_time = 0
        coefficient = 1
        while True:
            next_params = self.__next_params(coefficient)
            next_expr = self.__filtered_duplicated_result_expr(self._expr)
            new_page = self.__execute_next_search(next_params, next_expr, True)
            self.__update_filtered_ids(new_page)
            try_time += 1
            if len(new_page) > 0:
                final_page.merge(new_page.get_res())
                self._tail_band = new_page[-1].distance
            if len(final_page) >= self._iterator_params[BATCH_SIZE]:
                break
            if try_time > MAX_TRY_TIME:
                LOGGER.warning(f"Search probe exceed max try times:{MAX_TRY_TIME} directly break")
                break
            # if there's a ring containing no vectors matched, then we need to extend
            # the ring continually to avoid empty ring problem
            coefficient += 1
        return final_page

    def __execute_next_search(
        self, next_params: dict, next_expr: str, to_extend_batch: bool
    ) -> SearchPage:
        res = self._conn.search(
            self._iterator_params["collection_name"],
            self._iterator_params["data"],
            self._iterator_params["ann_field"],
            next_params,
            extend_batch_size(self._iterator_params[BATCH_SIZE], next_params, to_extend_batch),
            next_expr,
            self._iterator_params["partition_names"],
            self._iterator_params["output_fields"],
            self._iterator_params["round_decimal"],
            timeout=self._iterator_params["timeout"],
            schema=self._schema,
            **self._kwargs,
        )
        return SearchPage(res[0])

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
                return "(" + expr + ")" + filter_expr
            return f"{self._pk_field_name} not in [{filtered_ids_str}]"
        return expr

    def __next_params(self, coefficient: int):
        coefficient = max(1, coefficient)
        next_params = deepcopy(self._param)
        if metrics_positive_related(self._param[METRIC_TYPE]):
            next_radius = self._tail_band + self._width * coefficient
            if RADIUS in self._param[PARAMS] and next_radius > self._param[PARAMS][RADIUS]:
                next_params[PARAMS][RADIUS] = self._param[PARAMS][RADIUS]
            else:
                next_params[PARAMS][RADIUS] = next_radius
        else:
            next_radius = self._tail_band - self._width * coefficient
            if RADIUS in self._param[PARAMS] and next_radius < self._param[PARAMS][RADIUS]:
                next_params[PARAMS][RADIUS] = self._param[PARAMS][RADIUS]
            else:
                next_params[PARAMS][RADIUS] = next_radius
        next_params[PARAMS][RANGE_FILTER] = self._tail_band
        LOGGER.debug(
            f"next round search iteration radius:{next_params[PARAMS][RADIUS]},"
            f"range_filter:{next_params[PARAMS][RANGE_FILTER]},"
            f"coefficient:{coefficient}"
        )

        return next_params

    def close(self):
        iterator_cache.release_cache(self._cache_id)


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
