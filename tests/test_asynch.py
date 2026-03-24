from unittest.mock import MagicMock, patch

import grpc
import pytest
from pymilvus.client.asynch import (
    CreateFlatIndexFuture,
    CreateIndexFuture,
    Future,
    MutationFuture,
    SearchFuture,
    _parameter_is_empty,
)
from pymilvus.client.types import Status
from pymilvus.exceptions import MilvusException


class SimpleFuture(Future):
    """Concrete Future for testing common Future methods."""

    def on_response(self, response):
        return response


class TestParameterIsEmpty:
    def test_empty_function(self):
        def f():
            pass

        assert _parameter_is_empty(f) is True

    def test_non_empty_function(self):
        def f(x):
            pass

        assert _parameter_is_empty(f) is False

    def test_lambda_with_arg(self):
        assert _parameter_is_empty(lambda x: x) is False

    def test_lambda_empty(self):
        assert _parameter_is_empty(lambda: None) is True


@pytest.fixture
def mock_future():
    mf = MagicMock()
    mf.result.return_value = "mock_response"
    mf.exception.return_value = None
    return mf


class TestFutureCallback:
    def test_callback_with_tuple_results(self, mock_future):
        received = []

        def cb(a, b):
            received.extend([a, b])

        f = SimpleFuture(mock_future)
        f._results = ("val1", "val2")
        f._callback()
        # Since there's no callback initially, manually add and call again
        received.clear()
        f._callback_called = False
        f._done_cb_list = [cb]
        f._callback()
        assert received == ["val1", "val2"]

    def test_callback_with_zero_arg_function(self, mock_future):
        called = []

        def zero_arg_cb():
            called.append(True)

        f = SimpleFuture(mock_future)
        f._results = "some_result"
        f._done_cb_list = [zero_arg_cb]
        f._callback()
        assert called == [True]

    def test_callback_with_single_result(self, mock_future):
        received = []

        def cb(r):
            received.append(r)

        f = SimpleFuture(mock_future)
        f._results = "result_value"
        f._done_cb_list = [cb]
        f._callback()
        assert received == ["result_value"]

    def test_callback_raises_when_results_none(self, mock_future):
        def cb(r):
            pass

        f = SimpleFuture(mock_future)
        f._results = None
        f._done_cb_list = [cb]
        with pytest.raises(MilvusException):
            f._callback()

    def test_callback_called_only_once(self, mock_future):
        call_count = [0]

        def cb():
            call_count[0] += 1

        f = SimpleFuture(mock_future)
        f._results = "r"
        f._done_cb_list = [cb]
        f._callback()
        f._callback()
        assert call_count[0] == 1

    def test_none_callback_skipped(self, mock_future):
        f = SimpleFuture(mock_future)
        f._results = "r"
        f._done_cb_list = [None]
        f._callback()  # Should not raise

    def test_add_callback(self, mock_future):
        f = SimpleFuture(mock_future)
        received = []

        def _cb(r):
            received.append(r)

        f.add_callback(_cb)
        f._results = "data"
        f._callback()
        assert received == ["data"]


class TestFutureResult:
    def test_result_calls_on_response(self, mock_future):
        f = SimpleFuture(mock_future)
        result = f.result()
        assert result == "mock_response"

    def test_result_raises_milvus_exception_on_future_error(self, mock_future):
        mock_future.result.side_effect = Exception("grpc error")
        f = SimpleFuture(mock_future)
        with pytest.raises(MilvusException):
            f.result()

    def test_result_returns_cached_results(self, mock_future):
        f = SimpleFuture(mock_future)
        f._results = "cached"
        f._future = None
        result = f.result()
        assert result == "cached"

    def test_result_with_timeout(self, mock_future):
        f = SimpleFuture(mock_future, timeout=5.0)
        result = f.result(timeout=5.0)
        assert result == "mock_response"

    def test_result_returns_raw_response(self, mock_future):
        mock_future.result.return_value = "raw_response"
        f = SimpleFuture(mock_future)
        result = f.result(raw=True)
        assert result == "raw_response"

    def test_result_pre_exception_raises(self, mock_future):
        pre_exc = Exception("pre-init error")
        f = SimpleFuture(mock_future, pre_exception=pre_exc)
        with pytest.raises(Exception, match="pre-init error"):
            f.result()


class TestFutureDone:
    def test_done_processes_future(self, mock_future):
        mock_future.result.return_value = "response"
        f = SimpleFuture(mock_future)
        f.done()
        assert f._done is True

    def test_done_captures_exception(self, mock_future):
        mock_future.result.side_effect = Exception("done error")
        f = SimpleFuture(mock_future)
        f.done()
        assert f._exception is not None
        assert f._done is True

    def test_done_skips_if_results_already_set(self, mock_future):
        f = SimpleFuture(mock_future)
        f._results = "already_done"
        f.done()
        mock_future.result.assert_not_called()

    def test_is_done(self, mock_future):
        f = SimpleFuture(mock_future)
        assert f.is_done() is False
        f.done()
        assert f.is_done() is True


class TestFutureCancel:
    def test_cancel_calls_future_cancel(self, mock_future):
        f = SimpleFuture(mock_future)
        f.cancel()
        mock_future.cancel.assert_called_once()

    def test_cancel_without_future(self):
        f = SimpleFuture(None)
        f.cancel()  # Should not raise


class TestFutureException:
    def test_exception_raises_stored_exception(self, mock_future):
        exc = Exception("stored error")
        f = SimpleFuture(mock_future)
        f._exception = exc
        with pytest.raises(Exception, match="stored error"):
            f.exception()

    def test_exception_checks_future_exception(self, mock_future):
        f = SimpleFuture(mock_future)
        f.exception()
        mock_future.exception.assert_called_once()

    def test_exception_no_future(self):
        f = SimpleFuture(None)
        f.exception()  # Should not raise when no future and no stored exception


class TestCreateFlatIndexFuture:
    def test_result_with_zero_arg_callback(self):
        called = []

        def cb():
            called.append(True)

        f = CreateFlatIndexFuture(res="result_val", done_callback=cb)
        result = f.result()
        assert result == "result_val"
        assert called == [True]

    def test_result_with_single_arg_callback(self):
        received = []

        def _cb(r):
            received.append(r)

        f = CreateFlatIndexFuture(res="result_val", done_callback=_cb)
        f.result()
        assert received == ["result_val"]

    def test_result_with_tuple_result(self):
        received = []
        f = CreateFlatIndexFuture(
            res=("a", "b"), done_callback=lambda a, b: received.extend([a, b])
        )
        f.result()
        assert received == ["a", "b"]

    def test_result_with_none_callback(self):
        f = CreateFlatIndexFuture(res="result_val")
        result = f.result()
        assert result == "result_val"

    def test_result_raises_pre_exception(self):
        pre_exc = Exception("pre error")
        f = CreateFlatIndexFuture(res="something", pre_exception=pre_exc)
        with pytest.raises(Exception, match="pre error"):
            f.result()

    def test_result_raises_when_results_none_and_callback(self):
        def cb(r):
            pass

        f = CreateFlatIndexFuture(res=None, done_callback=cb)
        with pytest.raises(MilvusException):
            f.result()

    def test_cancel(self):
        f = CreateFlatIndexFuture(res="result_val")
        f.cancel()  # Should not raise

    def test_done(self):
        f = CreateFlatIndexFuture(res="result_val")
        f.done()  # Should not raise

    def test_is_done_always_true(self):
        f = CreateFlatIndexFuture(res="result_val")
        assert f.is_done() is True

    def test_on_response_returns_none(self):
        f = CreateFlatIndexFuture(res="r")
        assert f.on_response("anything") is None

    def test_del(self):
        f = CreateFlatIndexFuture(res="result_val")
        f.__del__()
        assert f._results is None


class TestSearchFutureOnResponse:
    def test_on_response_calls_check_status(self, mock_future):
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.reason = ""
        mock_response.results = MagicMock()

        with patch("pymilvus.client.asynch.check_status") as mock_check, patch(
            "pymilvus.client.asynch.SearchResult"
        ) as mock_sr:
            mock_sr.return_value = "search_result"
            f = SearchFuture(mock_future)
            result = f.on_response(mock_response)
            mock_check.assert_called_once_with(mock_response.status)
            assert result == "search_result"

    def test_on_response_none_raises_milvus_exception(self, mock_future):
        f = SearchFuture(mock_future)
        with pytest.raises(
            MilvusException, match="Received None response from server during search"
        ):
            f.on_response(None)

    def test_result_returns_empty_search_result(self, mock_future):
        """Empty SearchResult (0-hit, falsy list) must be returned as-is, not re-processed."""
        empty_result = []  # falsy, simulates empty SearchResult
        mock_future.result.return_value = MagicMock()
        mock_future.exception.return_value = None

        f = SearchFuture(mock_future)
        f._results = empty_result  # pre-set to simulate already-processed empty result
        f._future = None
        result = f.result()
        assert result is empty_result

    def test_result_none_response_deadline_exceeded_raises_timeout(self, mock_future):
        """result() with None gRPC response and DEADLINE_EXCEEDED code raises timeout error."""
        mock_future.result.return_value = None
        mock_future.exception.return_value = None
        mock_future.code.return_value = grpc.StatusCode.DEADLINE_EXCEEDED
        mock_future.details.return_value = "Deadline Exceeded"

        f = SearchFuture(mock_future)
        with pytest.raises(MilvusException, match="gRPC call timed out"):
            f.result()

    def test_result_none_response_other_code_includes_code_name(self, mock_future):
        """result() with None gRPC response and non-timeout code includes code name in error."""
        mock_future.result.return_value = None
        mock_future.exception.return_value = None
        mock_future.code.return_value = grpc.StatusCode.UNAVAILABLE
        mock_future.details.return_value = "connection reset"

        f = SearchFuture(mock_future)
        with pytest.raises(MilvusException, match="UNAVAILABLE"):
            f.result()

    def test_done_none_response_deadline_exceeded_stores_timeout_exception(self, mock_future):
        """done() with None gRPC response and DEADLINE_EXCEEDED stores a timeout MilvusException."""
        mock_future.result.return_value = None
        mock_future.code.return_value = grpc.StatusCode.DEADLINE_EXCEEDED
        mock_future.details.return_value = "Deadline Exceeded"

        f = SearchFuture(mock_future)
        f.done()

        assert f._exception is not None
        assert isinstance(f._exception, MilvusException)
        assert "timed out" in str(f._exception)


class TestMutationFutureOnResponse:
    def test_on_response_calls_check_status(self, mock_future):
        mock_response = MagicMock()
        mock_response.status.code = 0

        with patch("pymilvus.client.asynch.check_status") as mock_check, patch(
            "pymilvus.client.asynch.MutationResult"
        ) as mock_mr:
            mock_mr.return_value = "mutation_result"
            f = MutationFuture(mock_future)
            result = f.on_response(mock_response)
            mock_check.assert_called_once_with(mock_response.status)
            assert result == "mutation_result"


class TestCreateIndexFutureOnResponse:
    def test_on_response_returns_status(self, mock_future):
        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.reason = "ok"

        with patch("pymilvus.client.asynch.check_status") as mock_check:
            f = CreateIndexFuture(mock_future)
            result = f.on_response(mock_response)
            mock_check.assert_called_once_with(mock_response)
            assert isinstance(result, Status)
            assert result.code == 0
