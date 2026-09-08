import abc
import inspect
import threading
from contextlib import suppress
from typing import Any, Callable, Optional

import grpc

from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import milvus_pb2

from .abstract import MutationResult
from .search_result import SearchResult
from .types import Status
from .utils import check_status


def _build_none_response_exception(future: Any) -> MilvusException:
    """Build a MilvusException when a gRPC future returns None as the response message.

    This is a gRPC Python edge-case (race condition between client-side cancellation
    and the server response arriving) rather than a normal timeout flow.  Checking
    ``future.code()`` lets us surface the real gRPC status — in particular
    DEADLINE_EXCEEDED — so callers get an actionable error instead of an opaque
    AttributeError on ``None.status``.
    """
    try:
        code = future.code()
        details = future.details() or ""
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return MilvusException(message=f"gRPC call timed out (DEADLINE_EXCEEDED): {details}")
        return MilvusException(
            message=f"Received None response from server (code={code.name}, details={details})"
        )
    except Exception:
        return MilvusException(message="Received None response from server")


# TODO: remove this to a common util
def _parameter_is_empty(func: Callable):
    sig = inspect.signature(func)
    # todo: add more check to parameter, such as `default parameter`,
    #  `positional-only`, `positional-or-keyword`, `keyword-only`, `var-positional`, `var-keyword`
    # if len(params) == 0:
    # for param in params.values():
    #     if (param.kind == inspect.Parameter.POSITIONAL_ONLY or
    #             param.default == inspect._empty:
    return len(sig.parameters) == 0


class AbstractFuture:
    @abc.abstractmethod
    def result(self, **kwargs):
        """Return deserialized result.

        It's a synchronous interface. It will wait executing until
        server respond or timeout occur(if specified).

        This API is thread-safe.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cancel(self):
        """Cancle gRPC future.

        This API is thread-safe.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def done(self):
        """Wait for request done.

        This API is thread-safe.
        """
        raise NotImplementedError


class Future(AbstractFuture):
    def __init__(
        self,
        future: Any,
        done_callback: Optional[Callable] = None,
        pre_exception: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        self._future = future
        self._done_cb_list = []
        self.add_callback(done_callback)
        self._condition = threading.Condition()
        self._canceled = False
        self._done = False
        self._response = None
        self._results = None
        self._exception = pre_exception
        self._callback_called = False  # callback function should be called only once
        self._processed_callback_list = []
        self._processed = False
        self._processed_error = None
        self._kwargs = kwargs

    def add_callback(self, func: Callable):
        self._done_cb_list.append(func)

    def _add_processed_callback(self, func: Callable[[Optional[BaseException]], None]) -> None:
        """Run an internal callback after response parsing finishes.

        Unlike the underlying gRPC future's done callback, this completion point includes
        ``on_response()`` and the wrapper's result processing. It is intentionally separate
        from user callbacks, whose signature is based on the parsed result.
        """
        with self._condition:
            if not self._processed:
                self._processed_callback_list.append(func)
                return
            error = self._processed_error
        func(error)

    def _mark_processed_locked(
        self, error: Optional[BaseException]
    ) -> list[Callable[[Optional[BaseException]], None]]:
        """Atomically fix the first response-processing outcome.

        The caller must hold ``_condition`` through parsing and user callbacks, so a
        concurrent ``result()`` or ``done()`` cannot publish success between a failing
        processing step and this state transition.
        """
        if self._processed:
            return []
        self._processed = True
        self._processed_error = error
        callbacks = self._processed_callback_list
        self._processed_callback_list = []
        return callbacks

    @staticmethod
    def _dispatch_processed_callbacks(
        callbacks: list[Callable[[Optional[BaseException]], None]],
        error: Optional[BaseException],
    ) -> None:
        for callback in callbacks:
            with suppress(BaseException):
                callback(error)

    def __del__(self) -> None:
        self._future = None

    @abc.abstractmethod
    def on_response(self, response: Callable):
        """Parse response from gRPC server and return results."""
        raise NotImplementedError

    def _callback(self):
        if not self._callback_called:
            for cb in self._done_cb_list:
                if cb:
                    # necessary to check parameter signature of cb?
                    if isinstance(self._results, tuple):
                        cb(*self._results)
                    elif _parameter_is_empty(cb):
                        cb()
                    elif self._results is not None:
                        cb(self._results)
                    else:
                        raise MilvusException(message="callback function is not legal!")
        self._callback_called = True

    def result(self, **kwargs):
        processed_callbacks = []
        processed_error = None
        try:
            with self._condition:
                try:
                    self.exception()
                    # future not finished. wait callback being called.
                    to = kwargs.get("timeout")
                    if to is None:
                        to = self._kwargs.get("timeout", None)

                    if self._future and self._results is None:
                        try:
                            self._response = self._future.result(timeout=to)
                        except Exception as e:
                            raise MilvusException(message=str(e)) from e
                        if self._response is None:
                            raise _build_none_response_exception(self._future)
                        self._results = self.on_response(self._response)

                        self._callback()

                    self._done = True
                    self._condition.notify_all()

                    self.exception()
                    if kwargs.get("raw", False) is True:
                        # just return response object received from gRPC
                        result = self._response
                    elif self._results is not None:
                        result = self._results
                    else:
                        result = self.on_response(self._response)
                except BaseException as exc:
                    processed_error = exc
                    processed_callbacks = self._mark_processed_locked(exc)
                    raise
                else:
                    processed_callbacks = self._mark_processed_locked(None)
                    return result
        finally:
            self._dispatch_processed_callbacks(processed_callbacks, processed_error)

    def cancel(self):
        with self._condition:
            if self._future:
                self._future.cancel()
            self._condition.notify_all()

    def is_done(self):
        return self._done

    def done(self):
        processed_callbacks = []
        processed_error = None
        try:
            with self._condition:
                if self._processed:
                    self._done = True
                    self._condition.notify_all()
                    return
                try:
                    if self._future and self._results is None:
                        try:
                            self._response = self._future.result()
                            if self._response is None:
                                self._exception = _build_none_response_exception(self._future)
                            else:
                                self._results = self.on_response(self._response)
                                self._callback()  # https://github.com/milvus-io/milvus/issues/6160
                        except Exception as e:
                            self._exception = e

                    self._done = True
                    self._condition.notify_all()
                    processed_error = self._exception
                    processed_callbacks = self._mark_processed_locked(processed_error)
                except BaseException as exc:
                    processed_error = exc
                    processed_callbacks = self._mark_processed_locked(exc)
                    raise
        finally:
            self._dispatch_processed_callbacks(processed_callbacks, processed_error)

    def exception(self):
        if self._exception:
            raise self._exception
        if self._future:
            self._future.exception()


class SearchFuture(Future):
    def on_response(self, response: milvus_pb2.SearchResults):
        if response is None:
            raise MilvusException(message="Received None response from server during search")
        check_status(response.status)
        return SearchResult(response.results, status=response.status)


class MutationFuture(Future):
    def on_response(self, response: Any):
        check_status(response.status)
        return MutationResult(response)


class CreateIndexFuture(Future):
    def on_response(self, response: Any):
        check_status(response)
        return Status(response.code, response.reason)


class CreateFlatIndexFuture(AbstractFuture):
    def __init__(
        self,
        res: Any,
        done_callback: Optional[Callable] = None,
        pre_exception: Optional[Callable] = None,
    ) -> None:
        self._results = res
        self._done_cb_list = []
        self.add_callback(done_callback)
        self._condition = threading.Condition()
        self._exception = pre_exception

    def add_callback(self, func: Callable):
        self._done_cb_list.append(func)

    def __del__(self) -> None:
        self._results = None

    def on_response(self, response: Any):
        pass

    def result(self):
        self.exception()
        with self._condition:
            for cb in self._done_cb_list:
                if cb:
                    # necessary to check parameter signature of cb?
                    if isinstance(self._results, tuple):
                        cb(*self._results)
                    elif _parameter_is_empty(cb):
                        cb()
                    elif self._results is not None:
                        cb(self._results)
                    else:
                        raise MilvusException(message="callback function is not legal!")
            return self._results

    def cancel(self):
        with self._condition:
            self._condition.notify_all()

    def is_done(self):
        return True

    def done(self):
        with self._condition:
            self._condition.notify_all()

    def exception(self):
        if self._exception:
            raise self._exception


class FlushFuture(Future):
    def on_response(self, response: Any):
        check_status(response.status)
