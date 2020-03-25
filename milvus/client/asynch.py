import abc
import threading

from .abstract import TopKQueryResult
from .types import Status


class Future:
    @abc.abstractmethod
    def result(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def done(self):
        raise NotImplementedError()


class SearchFuture(Future):

    def __init__(self, future, done_callback=None):
        self._future = future
        self._done_cb = done_callback
        self._condition = threading.Condition()
        self._done = False
        self._response = None

        self.__init()

    def __parse_response(self):
        if self._response.status.error_code == 0:
            return Status(message='Add vectors successfully!'), TopKQueryResult(self._response)

        return Status(code=self._response.status.error_code, message=self._response.status.reason), None

    def __init(self):
        def async_done_callback(future):
            with self._condition:
                # self._condition.acquire()
                self._response = future.result()
                self._done_cb and self._done_cb(*self.__parse_response())
                self._done = True
                self._condition.notify_all()
                # self._condition.release()

        self._future.add_done_callback(async_done_callback)

    def result(self, **kwargs):
        # self._condition.acquire()
        with self._condition:
            if not self._done:
                to = kwargs.get("timeout", None)
                self._condition.wait(to)

            self._condition.notify_all()
        # self._condition.release()

        if kwargs.get("raw", False) is True:
            return self._response

        return self.__parse_response()

    def cancel(self):
        with self._condition:
            if not self._done:
                self._future.cancel()

    def done(self):
        with self._condition:
            if not self._done:
                self._condition.wait()

            self._condition.notify_all()
        # self._condition.release()
