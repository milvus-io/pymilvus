import abc
import threading

from .abstract import QueryResult
from .exceptions import BaseException
from .types import Status


class AbstractFuture:
    @abc.abstractmethod
    def result(self, **kwargs):
        '''Return deserialized result.

        It's a synchronous interface. It will wait executing until
        server respond or timeout occur(if specified).

        This API is thread-safe.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def cancel(self):
        '''Cancle gRPC future.

        This API is thread-safe.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def done(self):
        '''Wait for request done.

        This API is thread-safe.
        '''
        raise NotImplementedError()


class Future(AbstractFuture):
    def __init__(self, future, done_callback=None):
        self._future = future
        self._done_cb = done_callback
        self._condition = threading.Condition()
        self._canceled = False
        self._done = False
        self._response = None
        self._results = None
        self._exception = None

        self.__init()

    def __del__(self):
        self._future = None

    @abc.abstractmethod
    def on_response(self, response):
        ''' Parse response from gRPC server and return results.
        '''
        raise NotImplementedError()

    def __init(self):
        ''' Register request done callback of gRPC future
        Callback function can be executed in individual subthread of gRPC, so
        there need to notify main thread when callback function finished.
        '''

        def async_done_callback(future):
            with self._condition:
                # delete gRCP future manually
                # self._future.__del__()
                # self._future = None

                # If user specify done callback function, execute it.
                try:
                    self._response = future.result()
                    self._results = self.on_response(self._response)
                    if self._done_cb:
                        if isinstance(self._results, tuple):
                            self._done_cb(*self._results)
                        elif self._results is not None:
                            self._done_cb(self._results)
                        else:
                            self._done_cb()
                except Exception as e:
                    self._exception = e
                finally:
                    self._done = True
                    self._condition.notify_all()

        self._future.add_done_callback(async_done_callback)

    def result(self, **kwargs):
        self.exception()
        with self._condition:
            # future not finished. wait callback being called.
            to = kwargs.get("timeout", None)
            if not self._future.done() or not self._response:
                self._response = self._future.result(timeout=to)
            # if not self._done and not self._canceled:
            #     to = kwargs.get("timeout", None)
            #     self._condition.wait(to)
            #
            #     if not self._done and not self._canceled:
            #         self._condition.notify_all()
                    # raise FutureTimeoutError("Wait timeout")

            self._condition.notify_all()

        self.exception()
        if kwargs.get("raw", False) is True:
            # just return response object received from gRPC
            return self._response

        if self._results:
            return self._results
        else:
            return self.on_response(self._response)

    def cancel(self):
        with self._condition:
            self._future.cancel()
            # if not self._canceled or self._done:
            #     self._future.cancel()
            #     self._canceled = True
            self._condition.notify_all()

    def is_done(self):
        return self._done

    def done(self):
        # self.exception()
        with self._condition:
            if self._future and not self._future.done():
                try:
                    self._future.result()
                except Exception as e:
                    self._exception = e

            self._condition.notify_all()

    def exception(self):
        if not self._future.done():
            self._future.exception()
        if self._exception:
            raise self._exception


class SearchFuture(Future):

    def on_response(self, response):
        if response.status.error_code == 0:
            return QueryResult(response)

        status = response.status
        raise BaseException(status.error_code, status.reason)


class InsertFuture(Future):
    def on_response(self, response):
        status = response.status
        if status.error_code == 0:
            return list(response.entity_id_array)

        status = response.status
        raise BaseException(status.error_code, status.reason)


class CreateIndexFuture(Future):
    def on_response(self, response):
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

        return Status(response.error_code, response.reason)


class CompactFuture(Future):
    def on_response(self, response):
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

        return Status(response.error_code, response.reason)


class FlushFuture(Future):
    def on_response(self, response):
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)
