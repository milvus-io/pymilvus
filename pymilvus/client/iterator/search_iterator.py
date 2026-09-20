from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

from pymilvus.client import entity_helper, utils
from pymilvus.client.abstract import LoopBase
from pymilvus.client.constants import (
    BATCH_SIZE,
    CALC_DIST_BM25,
    CALC_DIST_COSINE,
    CALC_DIST_HAMMING,
    CALC_DIST_IP,
    CALC_DIST_JACCARD,
    CALC_DIST_L2,
    CALC_DIST_TANIMOTO,
    COLLECTION_ID,
    DEFAULT_SEARCH_EXTENSION_RATE,
    EF,
    FIELDS,
    GUARANTEE_TIMESTAMP,
    IS_PRIMARY,
    ITER_SEARCH_BATCH_SIZE_KEY,
    ITER_SEARCH_ID_KEY,
    ITER_SEARCH_LAST_BOUND_KEY,
    ITER_SEARCH_V2_KEY,
    ITERATOR_FIELD,
    MAX_BATCH_SIZE,
    MAX_FILTERED_IDS_COUNT_ITERATION,
    MAX_TRY_TIME,
    METRIC_TYPE,
    OFFSET,
    PARAMS,
    RADIUS,
    RANGE_FILTER,
    UNLIMITED,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import (
    ExceptionsMessage,
    MilvusException,
    ParamError,
    ServerVersionIncompatibleException,
)

from .query_iterator import NO_CACHE_ID, fall_back_to_latest_session_ts, iterator_cache

if TYPE_CHECKING:
    from pymilvus.client.call_context import CallContext
    from pymilvus.client.search_result import Hit, Hits

log = logging.getLogger(__name__)


class SearchIteratorHandler(Protocol):
    def describe_collection(self, collection_name: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def search(self, collection_name: str, **kwargs: Any) -> Any: ...


class SearchIteratorV2Handler(Protocol):
    def describe_collection(self, collection_name: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def search(self, **kwargs: Any) -> Any: ...


def extend_batch_size(batch_size: int, next_param: dict, to_extend_batch_size: bool) -> int:
    extend_rate = 1
    if to_extend_batch_size:
        extend_rate = DEFAULT_SEARCH_EXTENSION_RATE
    if EF in next_param[PARAMS]:
        return min(MAX_BATCH_SIZE, batch_size * extend_rate, next_param[PARAMS][EF])
    return min(MAX_BATCH_SIZE, batch_size * extend_rate)


def check_set_flag(obj: Any, flag_name: str, kwargs: dict[str, Any], key: str):
    setattr(obj, flag_name, kwargs.get(key, False))


def metrics_positive_related(metrics: str) -> bool:
    if metrics in [CALC_DIST_L2, CALC_DIST_JACCARD, CALC_DIST_HAMMING, CALC_DIST_TANIMOTO]:
        return True
    if metrics in [CALC_DIST_IP, CALC_DIST_COSINE, CALC_DIST_BM25]:
        return False
    raise MilvusException(message=f"unsupported metrics type for search iteration: {metrics}")


class SearchPage(LoopBase):
    """Since we only support nq=1 in search iteration, so search iteration response
    should be different from raw response of search operation"""

    def __init__(self, res: Hits, session_ts: int | None = 0):
        super().__init__()
        self._session_ts = session_ts
        self._results = []
        if res is not None:
            self._results.append(res)

    def get_session_ts(self):
        return self._session_ts

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

    def merge(self, others: list[Hits]):
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
        handler: SearchIteratorHandler,
        context: CallContext | None,
        collection_name: str,
        data: list | utils.SparseMatrixInputType,
        ann_field: str,
        param: dict,
        batch_size: int | None = 1000,
        limit: int | None = UNLIMITED,
        expr: str | None = None,
        partition_names: list[str] | None = None,
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        round_decimal: int = -1,
        schema: Mapping[str, Any] | None = None,
        rpc_options: Mapping[str, Any] | None = None,
    ) -> SearchIterator:
        rows = entity_helper.get_input_num_rows(data)
        if rows > 1:
            raise ParamError(
                message="Not support search iteration over multiple vectors at present"
            )
        if rows == 0:
            raise ParamError(message="vector_data for search cannot be empty")
        self._handler = handler
        self._context = context
        self._rpc_options = dict(rpc_options or {})
        self._rpc_options[ITERATOR_FIELD] = "True"
        self._search_options = self._rpc_options.copy()
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
        self._collection_name = collection_name
        self._expr = expr
        self.__check_set_params(param)
        self.__check_for_special_index_param()
        self.__set_up_collection_id()
        self._search_options[COLLECTION_ID] = self._collection_id
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

    def __set_up_collection_id(self):
        res = self._handler.describe_collection(
            self._collection_name,
            context=self._context,
            **self._rpc_options,
        )
        self._collection_id = res[COLLECTION_ID]

    def __init_search_iterator(self):
        init_page = self.__execute_next_search(self._param, self._expr, False)
        self._session_ts = init_page.get_session_ts()
        if self._session_ts <= 0:
            log.warning("failed to set up mvccTs from milvus server, use client-side ts instead")
            self._session_ts = fall_back_to_latest_session_ts()
        self._search_options[GUARANTEE_TIMESTAMP] = self._session_ts
        if len(init_page) == 0:
            message = (
                "Cannot init search iterator because init page contains no matched rows, "
                "please check the radius and range_filter set up by searchParams"
            )
            log.error(message)
            self._cache_id = NO_CACHE_ID
            self._init_success = False
            return
        self._cache_id = iterator_cache.cache(init_page, NO_CACHE_ID)
        self.__set_up_range_parameters(init_page)
        self.__update_filtered_ids(init_page)
        self._init_success = True

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
        log.debug(
            f"set up init parameter for searchIterator width:{self._width} tail_band:{self._tail_band}"
        )

    def __check_reached_limit(self) -> bool:
        if self._limit == UNLIMITED or self._returned_count < self._limit:
            return False
        log.debug(
            f"reached search limit:{self._limit}, returned_count:{self._returned_count}, directly return"
        )
        return True

    def __check_set_params(self, param: dict):
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
                    f"smaller than range_filter, please adjust your parameter"
                )

    def __check_offset(self):
        if self._search_options.get(OFFSET, 0) != 0:
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
                log.warning(f"Search probe exceed max try times:{MAX_TRY_TIME} directly break")
                break
            # if there's a ring containing no vectors matched, then we need to extend
            # the ring continually to avoid empty ring problem
            coefficient += 1
        return final_page

    def __execute_next_search(
        self, next_params: dict, next_expr: str, to_extend_batch: bool
    ) -> SearchPage:
        log.debug(f"search_iterator_next_expr:{next_expr}, next_params:{next_params}")
        res = self._handler.search(
            collection_name=self._iterator_params["collection_name"],
            anns_field=self._iterator_params["ann_field"],
            param=next_params,
            limit=extend_batch_size(
                self._iterator_params[BATCH_SIZE], next_params, to_extend_batch
            ),
            data=self._iterator_params["data"],
            expression=next_expr,
            partition_names=self._iterator_params["partition_names"],
            output_fields=self._iterator_params["output_fields"],
            round_decimal=self._iterator_params["round_decimal"],
            timeout=self._iterator_params["timeout"],
            schema=self._schema,
            context=self._context,
            **self._search_options,
        )
        return SearchPage(res[0], res.get_session_ts())

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
        log.debug(
            f"next round search iteration radius:{next_params[PARAMS][RADIUS]},"
            f"range_filter:{next_params[PARAMS][RANGE_FILTER]},"
            f"coefficient:{coefficient}"
        )

        return next_params

    def close(self):
        iterator_cache.release_cache(self._cache_id)


class SearchIteratorV2:
    # for compatibility, track the number of total results left
    _left_res_cnt = None

    def __init__(
        self,
        handler: SearchIteratorV2Handler,
        context: CallContext | None,
        collection_name: str,
        data: list | utils.SparseMatrixInputType,
        batch_size: int = 1000,
        limit: int | None = UNLIMITED,
        filter: str | None = None,
        output_fields: list[str] | None = None,
        search_params: dict | None = None,
        timeout: float | None = None,
        partition_names: list[str] | None = None,
        anns_field: str | None = None,
        round_decimal: int | None = -1,
        external_filter_func: Callable[[Hits], Hits | list[Hit]] | None = None,
        rpc_options: Mapping[str, Any] | None = None,
    ):
        self._rpc_options = dict(rpc_options or {})
        self._check_params(batch_size, data, self._rpc_options)

        # for compatibility, support limit, deprecate in future
        if limit != UNLIMITED:
            self._left_res_cnt = limit

        self._handler = handler
        self._context = context
        self._set_up_collection_id(collection_name)
        search_options = self._rpc_options.copy()
        search_options[COLLECTION_ID] = self._collection_id
        self._params = {
            "collection_name": collection_name,
            "data": data,
            "anns_field": anns_field,
            "param": deepcopy(search_params),
            "limit": batch_size,
            "expression": filter,
            "partition_names": partition_names,
            "output_fields": output_fields,
            "timeout": timeout,
            "round_decimal": round_decimal,
            ITERATOR_FIELD: True,
            ITER_SEARCH_V2_KEY: True,
            ITER_SEARCH_BATCH_SIZE_KEY: batch_size,
            GUARANTEE_TIMESTAMP: 0,
            **search_options,
        }
        self._external_filter_func = external_filter_func
        self._cache = []
        self._batch_size = batch_size
        self._probe_for_compability(self._params)

    def _set_up_collection_id(self, collection_name: str):
        res = self._handler.describe_collection(
            collection_name,
            context=self._context,
            **self._rpc_options,
        )
        self._collection_id = res[COLLECTION_ID]

    def _check_token_exists(self, token: str | None):
        if token is None or token == "":
            raise ServerVersionIncompatibleException(
                message=ExceptionsMessage.SearchIteratorV2FallbackWarning
            )

    # this detects whether the server supports search_iterator_v2 and is for compatibility only
    # if the server holds iterator states, this implementation needs to be reconsidered
    def _probe_for_compability(self, params: dict):
        dummy_params = deepcopy(params)
        dummy_batch_size = 1
        dummy_params["limit"] = dummy_batch_size
        dummy_params[ITER_SEARCH_BATCH_SIZE_KEY] = dummy_batch_size
        probe_result = self._handler.search(context=self._context, **dummy_params)
        iter_info = probe_result.get_search_iterator_v2_results_info()
        self._check_token_exists(iter_info.token)
        # Pin GUARANTEE_TIMESTAMP from probe call's session_ts so that all subsequent
        # next() calls (including the very first) see a consistent MVCC snapshot.
        # Without this, the first next() runs with GUARANTEE_TIMESTAMP=0 (latest available),
        # which means a segment reload triggered by add_collection_field can shift distances
        # by 1 ULP, causing last_bound items to pass the dist > last_bound filter again
        # and produce duplicate PKs. See: https://github.com/milvus-io/pymilvus/issues/3421
        if params[GUARANTEE_TIMESTAMP] <= 0:
            session_ts = probe_result.get_session_ts()
            if session_ts > 0:
                params[GUARANTEE_TIMESTAMP] = session_ts
            else:
                log.warning("failed to set up mvccTs from probe call, use client-side ts instead")
                params[GUARANTEE_TIMESTAMP] = fall_back_to_latest_session_ts()

    # internal next function, do not use this outside of this class
    def _next(self):
        res = self._handler.search(context=self._context, **self._params)
        iter_info = res.get_search_iterator_v2_results_info()
        self._check_token_exists(iter_info.token)
        self._params[ITER_SEARCH_LAST_BOUND_KEY] = iter_info.last_bound

        # patch token and guarantee timestamp for the first next() call
        if ITER_SEARCH_ID_KEY not in self._params:
            # the token should not change during the lifetime of the iterator
            self._params[ITER_SEARCH_ID_KEY] = iter_info.token
        if self._params[GUARANTEE_TIMESTAMP] <= 0:
            if res.get_session_ts() > 0:
                self._params[GUARANTEE_TIMESTAMP] = res.get_session_ts()
            else:
                log.warning(
                    "failed to set up mvccTs from milvus server, use client-side ts instead"
                )
                self._params[GUARANTEE_TIMESTAMP] = fall_back_to_latest_session_ts()
        return res

    def next(self):
        if self._left_res_cnt is not None and self._left_res_cnt <= 0:
            return None

        if self._external_filter_func is None:
            # return SearchPage for compability
            return self._wrap_return_res(self._next()[0])
        # the length of the results should be `batch_size` if no limit is set,
        # otherwise it should be the number of results left if less than `batch_size`
        target_len = (
            self._batch_size
            if self._left_res_cnt is None
            else min(self._batch_size, self._left_res_cnt)
        )
        while True:
            hits = self._next()[0]

            # no more results from server
            if len(hits) == 0:
                break

            # apply external filter
            if self._external_filter_func is not None:
                hits = self._external_filter_func(hits)

            self._cache.extend(hits)
            if len(self._cache) >= target_len:
                break

        # if the number of elements in cache is less than or equal to target_len,
        #   return all results we could possibly return
        # if the number of elements in cache is more than target_len,
        #   return target_len results and keep the rest for next call
        ret = self._cache[:target_len]
        del self._cache[:target_len]
        # return SearchPage for compability
        return self._wrap_return_res(ret)

    def close(self):
        pass

    def _check_params(
        self,
        batch_size: int,
        data: list | utils.SparseMatrixInputType,
        rpc_options: Mapping[str, Any],
    ):
        # metric_type can be empty, deduced at server side
        # anns_field can be empty, deduced at server side

        # check batch size
        if batch_size < 0:
            raise ParamError(message="batch size cannot be less than zero")
        if batch_size > MAX_BATCH_SIZE:
            raise ParamError(message=f"batch size cannot be larger than {MAX_BATCH_SIZE}")

        # check offset
        if rpc_options.get(OFFSET, 0) != 0:
            raise ParamError(message="Offset is not supported for search_iterator_v2")

        # check num queries, heavy to check at server side
        rows = entity_helper.get_input_num_rows(data)
        if rows > 1:
            raise ParamError(
                message="search_iterator_v2 does not support processing multiple vectors simultaneously"
            )
        if rows == 0:
            raise ParamError(message="The vector data for search cannot be empty")

    def _wrap_return_res(self, res: Hits) -> SearchPage:
        if len(res) == 0:
            return SearchPage(None)

        if self._left_res_cnt is None:
            return SearchPage(res)

        # When we have a limit, ensure we don't return more results than requested
        cur_len = len(res)
        if cur_len > self._left_res_cnt:
            res = res[: self._left_res_cnt]
        self._left_res_cnt -= cur_len
        return SearchPage(res)
