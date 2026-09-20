from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

from pymilvus.client.constants import (
    BATCH_SIZE,
    COLLECTION_ID,
    FIELDS,
    GUARANTEE_TIMESTAMP,
    INT64_MAX,
    IS_PRIMARY,
    ITERATOR_FIELD,
    ITERATOR_SESSION_CP_FILE,
    ITERATOR_SESSION_TS_FIELD,
    MAX_BATCH_SIZE,
    MILVUS_LIMIT,
    OFFSET,
    QUERY_ITER_LAST_ELEMENT_OFFSET,
    QUERY_ITER_LAST_PK,
    REDUCE_STOP_FOR_BEST,
    UNLIMITED,
)
from pymilvus.client.types import DataType
from pymilvus.client.utils import mkts_from_datetime
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.grpc_gen import milvus_pb2 as milvus_types

if TYPE_CHECKING:
    from pymilvus.client.call_context import CallContext

log = logging.getLogger(__name__)


class QueryIteratorHandler(Protocol):
    def describe_collection(self, collection_name: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def query(self, collection_name: str, **kwargs: Any) -> Any: ...


class QueryIteratorCursor:
    def __init__(
        self,
        session_ts: int,
        int_pk: int | None = None,
        str_pk: str | None = None,
        last_element_offset: int | None = None,
    ):
        self.session_ts = session_ts
        self.int_pk = int_pk
        self.str_pk = str_pk
        self.last_element_offset = last_element_offset

    def to_proto(self) -> milvus_types.QueryCursor:
        cursor = milvus_types.QueryCursor()
        cursor.session_ts = self.session_ts
        if self.str_pk is not None:
            cursor.str_pk = self.str_pk
        elif self.int_pk is not None:
            cursor.int_pk = self.int_pk
        return cursor


def fall_back_to_latest_session_ts():
    d = datetime.datetime.now()
    return mkts_from_datetime(d, milliseconds=1000.0)


def assert_info(condition: bool, message: str):
    if not condition:
        raise MilvusException(message)


def io_operation(io_func: Callable[[Any], None], message: str):
    try:
        io_func()
    except OSError as ose:
        raise MilvusException(message=message) from ose


class QueryIterator:
    def __init__(
        self,
        handler: QueryIteratorHandler,
        context: CallContext | None,
        collection_name: str,
        batch_size: int | None = 1000,
        limit: int | None = -1,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        schema: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        rpc_options: Mapping[str, Any] | None = None,
    ) -> QueryIterator:
        self._handler = handler
        self._context = context
        self._rpc_options = dict(rpc_options or {})
        self._query_options = self._rpc_options.copy()
        self._collection_name = collection_name
        self.__set_up_collection_id()
        self._output_fields = output_fields
        self._partition_names = partition_names
        self._schema = schema
        self._timeout = timeout
        self._session_ts = 0
        self._query_options[ITERATOR_FIELD] = "True"
        self._query_options[COLLECTION_ID] = self._collection_id
        self.__check_set_batch_size(batch_size)
        self._limit = limit
        self.__check_set_reduce_stop_for_best()
        self._returned_count = 0
        self.__setup__pk_prop()
        self.__set_up_expr(expr)
        self._next_id = None
        self._next_element_offset = None
        self._is_element_filter_iterator = self.__is_element_filter_expr(self._expr)
        self._cache_id_in_use = NO_CACHE_ID
        self._cp_file_handler = None
        self.__set_up_ts_cp()
        self.__seek_to_offset()

    def __set_up_collection_id(self):
        res = self._handler.describe_collection(
            self._collection_name,
            context=self._context,
            **self._rpc_options,
        )
        self._collection_id = res[COLLECTION_ID]

    def __seek_to_offset(self):
        # read pk cursor from cp file, no need to seek offset
        if self._next_id is not None:
            return
        offset = self._query_options.get(OFFSET, 0)
        if offset > 0:
            seek_params = self._query_options.copy()
            seek_params[OFFSET] = 0
            seek_params[ITERATOR_FIELD] = "False"
            seek_params[REDUCE_STOP_FOR_BEST] = "False"
            start_time = time.time()

            def seek_offset_by_batch(batch: int, expr: str) -> int:
                seek_params[MILVUS_LIMIT] = batch
                query_params = seek_params.copy()
                query_params.pop(QUERY_ITER_LAST_PK, None)
                query_params.pop(QUERY_ITER_LAST_ELEMENT_OFFSET, None)
                if self._has_element_cursor():
                    query_params[QUERY_ITER_LAST_PK] = self._next_id
                    query_params[QUERY_ITER_LAST_ELEMENT_OFFSET] = self._next_element_offset
                res = self._handler.query(
                    collection_name=self._collection_name,
                    expr=expr,
                    output_fields=[],
                    partition_names=self._partition_names,
                    timeout=self._timeout,
                    context=self._context,
                    **query_params,
                )
                self.__update_cursor(res)
                return len(res)

            while offset > 0:
                batch_size = min(MAX_BATCH_SIZE, offset)
                next_expr = self.__setup_next_expr()
                seeked_count = seek_offset_by_batch(batch_size, next_expr)
                log.debug(
                    f"seeked offset, seek_expr:{next_expr} batch_size:{batch_size} seeked_count:{seeked_count}"
                )
                if seeked_count == 0:
                    log.info(
                        "seek offset has drained all matched results for query iterator, break"
                    )
                    break
                offset -= seeked_count
            self._query_options[OFFSET] = 0
            seek_offset_duration = time.time() - start_time
            log.info(
                f"Finish seek offset for query iterator, offset:{offset}, current_pk_cursor:{self._next_id}, "
                f"duration:{seek_offset_duration}"
            )

    def __init_cp_file_handler(self) -> bool:
        mode = "w"
        if self._cp_file_path.exists():
            mode = "r+"
        try:
            self._cp_file_handler = self._cp_file_path.open(mode)
        except OSError as ose:
            raise MilvusException(
                message=f"Failed to open cp file for iterator:{self._cp_file_path_str}"
            ) from ose
        return mode == "r+"

    def __save_mvcc_ts(self):
        assert_info(
            self._cp_file_handler is not None,
            "Must init cp file handler before saving session_ts",
        )
        self._cp_file_handler.writelines(str(self._session_ts) + "\n")

    def __save_pk_cursor(self):
        if self._need_save_cp and self._next_id is not None:
            if not self._cp_file_path.exists():
                self._cp_file_handler.close()
                self._cp_file_handler = self._cp_file_path.open("w")
                self._buffer_cursor_lines_number = 0
                self.__save_mvcc_ts()
                log.warning(
                    "iterator cp file is not existed any more, recreate for iteration, "
                    "do not remove this file manually!"
                )
            if self._buffer_cursor_lines_number >= 100:
                self._cp_file_handler.seek(0)
                self._cp_file_handler.truncate()
                log.info(
                    "cursor lines in cp file has exceeded 100 lines, truncate the file and rewrite"
                )
                self._buffer_cursor_lines_number = 0
                self.__save_mvcc_ts()
            self._cp_file_handler.writelines(self.__dump_cursor_line() + "\n")
            self._cp_file_handler.flush()
            self._buffer_cursor_lines_number += 1

    def __dump_cursor_line(self) -> str:
        if self._has_element_cursor():
            return json.dumps(
                {
                    "pk": self._next_id,
                    "last_element_offset": self._next_element_offset,
                },
                separators=(",", ":"),
            )
        return str(self._next_id)

    def __restore_cursor_line(self, line: str) -> None:
        cursor_line = line.strip()
        try:
            cursor = json.loads(cursor_line)
        except json.JSONDecodeError:
            self._next_id = cursor_line
            self._next_element_offset = None
            return

        if isinstance(cursor, dict) and "pk" in cursor:
            self._next_id = cursor["pk"]
            self._next_element_offset = cursor.get("last_element_offset")
            return

        self._next_id = cursor_line
        self._next_element_offset = None

    def __check_set_reduce_stop_for_best(self):
        if self._query_options.get(REDUCE_STOP_FOR_BEST, True):
            self._query_options[REDUCE_STOP_FOR_BEST] = "True"
        else:
            self._query_options[REDUCE_STOP_FOR_BEST] = "False"

    def __check_set_batch_size(self, batch_size: int):
        if batch_size < 0:
            raise ParamError(message="batch size cannot be less than zero")
        if batch_size > MAX_BATCH_SIZE:
            raise ParamError(message=f"batch size cannot be larger than {MAX_BATCH_SIZE}")
        self._query_options[BATCH_SIZE] = batch_size
        self._query_options[MILVUS_LIMIT] = batch_size

    # rely on pk prop, so this method should be called after __setup__pk_prop
    def __set_up_expr(self, expr: str):
        if expr is not None:
            self._expr = expr
        elif self._pk_str:
            self._expr = self._pk_field_name + ' != ""'
        else:
            self._expr = self._pk_field_name + " < " + str(INT64_MAX)

    @staticmethod
    def __is_element_filter_expr(expr: str) -> bool:
        return expr is not None and "element_filter" in expr.lower()

    def __setup_ts_by_request(self):
        init_ts_kwargs = self._query_options.copy()
        init_ts_kwargs[OFFSET] = 0
        init_ts_kwargs[MILVUS_LIMIT] = 1
        # just to set up mvccTs for iterator, no need correct limit
        res = self._handler.query(
            collection_name=self._collection_name,
            expr=self._expr,
            output_fields=[],
            partition_names=[],
            timeout=self._timeout,
            context=self._context,
            **init_ts_kwargs,
        )
        if res is None:
            raise MilvusException(
                message="failed to connect to milvus for setting up "
                "mvccTs, check milvus servers' status"
            )
        if res.extra is not None:
            self._session_ts = res.extra.get(ITERATOR_SESSION_TS_FIELD, 0)
        if self._session_ts <= 0:
            log.warning("failed to get mvccTs from milvus server, use client-side ts instead")
            self._session_ts = fall_back_to_latest_session_ts()
        self._query_options[GUARANTEE_TIMESTAMP] = self._session_ts

    def __set_up_ts_cp(self):
        self._buffer_cursor_lines_number = 0
        self._cp_file_path_str = self._query_options.get(ITERATOR_SESSION_CP_FILE, None)
        self._cp_file_path = None
        # no input cp_file, set up mvccTs by query request
        if self._cp_file_path_str is None:
            self._need_save_cp = False
            self.__setup_ts_by_request()
        else:
            self._need_save_cp = True
            self._cp_file_path = Path(self._cp_file_path_str)
            if not self.__init_cp_file_handler():
                # input cp file is empty, set up mvccTs by query request
                self.__setup_ts_by_request()
                io_operation(self.__save_mvcc_ts, "Failed to save mvcc ts")
            else:
                try:
                    # input cp file is not emtpy, init mvccTs by reading cp file
                    lines = self._cp_file_handler.readlines()
                    line_count = len(lines)
                    if line_count < 2:
                        raise ParamError(
                            message=f"input cp file:{self._cp_file_path_str} should contain "
                            f"at least two lines, but only:{line_count} lines"
                        )
                    self._session_ts = int(lines[0])
                    self._query_options[GUARANTEE_TIMESTAMP] = self._session_ts
                    if line_count > 1:
                        self._buffer_cursor_lines_number = line_count - 1
                        self.__restore_cursor_line(lines[self._buffer_cursor_lines_number])
                except OSError as ose:
                    raise MilvusException(
                        message=f"Failed to read cp info from file:{self._cp_file_path_str}"
                    ) from ose
                except ValueError as e:
                    raise ParamError(message=f"cannot parse input cp session_ts:{lines[0]}") from e

    def __maybe_cache(self, result: list):
        if len(result) < 2 * self._query_options[BATCH_SIZE]:
            return
        start = self._query_options[BATCH_SIZE]
        cache_result = result[start:]
        cache_id = iterator_cache.cache(cache_result, NO_CACHE_ID)
        self._cache_id_in_use = cache_id

    def __is_res_sufficient(self, res: list):
        return res is not None and len(res) >= self._query_options[BATCH_SIZE]

    def get_cursor(self) -> QueryIteratorCursor:
        if self._pk_str:
            return QueryIteratorCursor(
                session_ts=self._session_ts,
                str_pk=str(self._next_id),
                last_element_offset=self._next_element_offset,
            )
        return QueryIteratorCursor(
            session_ts=self._session_ts,
            int_pk=None if self._next_id is None else int(self._next_id),
            last_element_offset=self._next_element_offset,
        )

    def next(self):
        cached_res = iterator_cache.fetch_cache(self._cache_id_in_use)
        ret = None
        if self.__is_res_sufficient(cached_res):
            ret = cached_res[0 : self._query_options[BATCH_SIZE]]
            res_to_cache = cached_res[self._query_options[BATCH_SIZE] :]
            iterator_cache.cache(res_to_cache, self._cache_id_in_use)
        else:
            iterator_cache.release_cache(self._cache_id_in_use)
            current_expr = self.__setup_next_expr()
            log.debug(f"query_iterator_next_expr:{current_expr}")
            query_params = self.__setup_query_params()
            res = self._handler.query(
                collection_name=self._collection_name,
                expr=current_expr,
                output_fields=self._output_fields,
                partition_names=self._partition_names,
                timeout=self._timeout,
                context=self._context,
                **query_params,
            )
            self.__maybe_cache(res)
            ret = res[0 : min(self._query_options[BATCH_SIZE], len(res))]

        ret = self.__check_reached_limit(ret)
        self.__update_cursor(ret)
        io_operation(self.__save_pk_cursor, "failed to save pk cursor")
        self._returned_count += len(ret)
        return ret

    def __check_reached_limit(self, ret: list):
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

    def __setup_next_expr(self) -> str:
        current_expr = self._expr
        if self._next_id is None:
            return current_expr
        filtered_pk_str = ""
        pk_op = ">=" if self._has_element_cursor() else ">"
        if self._pk_str:
            filtered_pk_str = f'{self._pk_field_name} {pk_op} "{self._next_id}"'
        else:
            filtered_pk_str = f"{self._pk_field_name} {pk_op} {self._next_id}"
        if current_expr is None or len(current_expr) == 0:
            return filtered_pk_str
        # Put the PK cursor on the LEFT of AND so that operators with strict
        # position constraints (e.g. `element_filter()` on struct array, which
        # must be the right-most operand of a logical AND) keep their position
        # across paginated calls.
        return filtered_pk_str + " and (" + current_expr + ")"

    def __setup_query_params(self) -> dict[str, Any]:
        params = self._query_options.copy()
        params.pop(QUERY_ITER_LAST_PK, None)
        params.pop(QUERY_ITER_LAST_ELEMENT_OFFSET, None)
        if self._has_element_cursor():
            params[QUERY_ITER_LAST_PK] = self._next_id
            params[QUERY_ITER_LAST_ELEMENT_OFFSET] = self._next_element_offset
        return params

    def _has_element_cursor(self) -> bool:
        return self._is_element_filter_iterator and self._next_element_offset is not None

    def __update_cursor(self, res: list) -> None:
        if len(res) == 0:
            return
        self._next_id = res[-1][self._pk_field_name]
        self._next_element_offset = res[-1].get(OFFSET)

    def close(self) -> None:
        # release cache in use
        iterator_cache.release_cache(self._cache_id_in_use)
        if self._cp_file_handler is not None:

            def inner_close():
                self._cp_file_handler.close()
                self._cp_file_path.unlink()
                log.info(f"removed cp file:{self._cp_file_path_str} for query iterator")

            io_operation(
                inner_close, f"failed to clear cp file:{self._cp_file_path_str} for query iterator"
            )


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
