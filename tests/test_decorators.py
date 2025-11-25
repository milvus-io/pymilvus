import time
from unittest.mock import patch

import grpc
import pytest
from pymilvus.decorators import IGNORE_RETRY_CODES, error_handler, retry_on_rpc_failure
from pymilvus.exceptions import ErrorCode, MilvusException
from pymilvus.grpc_gen import common_pb2


class TestDecorators:
    def mock_failure(self, code: grpc.StatusCode):
        if code in IGNORE_RETRY_CODES:
            raise MockUnRetriableError(code)
        if code == MockUnavailableError().code():
            raise MockUnavailableError
        if code == MockDeadlineExceededError().code():
            raise MockDeadlineExceededError

    def mock_milvus_exception(self, code: ErrorCode):
        if code == ErrorCode.FORCE_DENY:
            raise MockForceDenyError
        if code == ErrorCode.RATE_LIMIT:
            raise MockRateLimitError
        raise MilvusException(ErrorCode.UNEXPECTED_ERROR, "unexpected error")

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_unavailable(self, times):
        self.count_test_retry_decorators_Unavailable = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code):
            self.count_test_retry_decorators_Unavailable += 1
            self.mock_failure(code)

        with pytest.raises(MilvusException, match="unavailable"):
            test_api(self, grpc.StatusCode.UNAVAILABLE)

        # the first execute + retry times
        assert self.count_test_retry_decorators_Unavailable == times + 1

    def test_retry_decorators_timeout(self):
        self.count_test_retry_decorators_timeout = 0

        @retry_on_rpc_failure()
        def test_api(self, code, timeout=None):
            self.count_test_retry_decorators_timeout += 1
            time.sleep(1)
            self.mock_failure(code)

        with pytest.raises(MilvusException):
            test_api(self, grpc.StatusCode.UNAVAILABLE, timeout=1)

        assert self.count_test_retry_decorators_timeout == 1

    @pytest.mark.skip("Do not open this unless you have loads of time, get some coffee and wait")
    def test_retry_decorators_default_behaviour(self):
        self.test_retry_decorators_default_retry_times = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            self.test_retry_decorators_default_retry_times += 1
            self.mock_failure(code)

        with pytest.raises(MilvusException):
            test_api(self, grpc.StatusCode.UNAVAILABLE)

        assert self.test_retry_decorators_default_retry_times == 7 + 1

    def test_retry_decorators_force_deny(self):
        self.execute_times = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            self.execute_times += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException, match="force deny"):
            test_api(self, ErrorCode.FORCE_DENY)

        # the first execute + 0 retry times
        assert self.execute_times == 1

    def test_retry_decorators_set_retry_times(self):
        self.count_retry_times = 0

        @retry_on_rpc_failure()
        def test_api(self, code, retry_on_rate_limit, **kwargs):
            self.count_retry_times += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException):
            test_api(self, ErrorCode.RATE_LIMIT, retry_on_rate_limit=True, retry_times=3)

        # the first execute + 0 retry times
        assert self.count_retry_times == 3 + 1

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_rate_limit_without_retry(self, times):
        self.count_test_retry_decorators_force_deny = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code, retry_on_rate_limit):
            self.count_test_retry_decorators_force_deny += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException, match="rate limit"):
            test_api(self, ErrorCode.RATE_LIMIT, retry_on_rate_limit=False)

        # the first execute + 0 retry times
        assert self.count_test_retry_decorators_force_deny == 1

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_rate_limit_with_retry(self, times):
        self.count_test_retry_decorators_force_deny = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code, retry_on_rate_limit):
            self.count_test_retry_decorators_force_deny += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException, match="rate limit"):
            test_api(self, ErrorCode.RATE_LIMIT, retry_on_rate_limit=True)

        # the first execute + retry times
        assert self.count_test_retry_decorators_force_deny == times + 1

    @pytest.mark.parametrize("code", IGNORE_RETRY_CODES)
    def test_donot_retry_codes(self, code):
        self.count_test_donot_retry = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            self.count_test_donot_retry += 1
            self.mock_failure(code)

        with pytest.raises(grpc.RpcError):
            test_api(self, code)

        # no retry
        assert self.count_test_donot_retry == 1


class MockUnavailableError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "details of unavailable"


class MockDeadlineExceededError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "details of deadline exceeded"


class MockForceDenyError(MilvusException):
    def __init__(self, code=ErrorCode.FORCE_DENY, message="force deny"):
        super(MilvusException, self).__init__(message)
        self._code = code
        self._message = message
        self._compatible_code = common_pb2.ForceDeny

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message

    @property
    def compatible_code(self):
        return self._compatible_code


class MockRateLimitError(MilvusException):
    def __init__(self, code=ErrorCode.RATE_LIMIT, message="rate limit"):
        super(MilvusException, self).__init__(message)
        self._code = code
        self._message = message
        self._compatible_code = common_pb2.RateLimit

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message

    @property
    def compatible_code(self):
        return self._compatible_code

class MockUnRetriableError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, message="unretriable error"):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "details of unretriable error"


class TestErrorHandlerTraceback:
    """Test the traceback functionality in error_handler decorator."""

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_includes_traceback_for_milvus_exception(self, mock_logger):
        """Test that error_handler logs traceback for MilvusException."""

        @error_handler(func_name="test_func")
        def func_that_raises_milvus_exception():
            def inner_func():
                raise MilvusException(ErrorCode.UNEXPECTED_ERROR, "test error")

            inner_func()

        with pytest.raises(MilvusException):
            func_that_raises_milvus_exception()

        # Verify LOGGER.error was called
        assert mock_logger.error.called
        # Get the logged message
        log_message = mock_logger.error.call_args[0][0]
        # Check that traceback information is in the log message
        assert "Traceback:" in log_message
        assert "inner_func" in log_message
        assert "test_func" in log_message

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_includes_traceback_for_grpc_error(self, mock_logger):
        """Test that error_handler logs traceback for grpc.RpcError."""

        @error_handler(func_name="test_grpc_func")
        def func_that_raises_grpc_error():
            def inner_func():
                raise MockUnavailableError()

            inner_func()

        with pytest.raises(grpc.RpcError):
            func_that_raises_grpc_error()

        # Verify LOGGER.error was called
        assert mock_logger.error.called
        # Get the logged message
        log_message = mock_logger.error.call_args[0][0]
        # Check that traceback information is in the log message
        assert "Traceback:" in log_message
        assert "inner_func" in log_message
        assert "test_grpc_func" in log_message

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_includes_traceback_for_generic_exception(self, mock_logger):
        """Test that error_handler logs traceback for generic Exception."""

        @error_handler(func_name="test_generic_func")
        def func_that_raises_generic_exception():
            def inner_func():
                raise ValueError("test generic error")

            inner_func()

        with pytest.raises(MilvusException):
            func_that_raises_generic_exception()

        # Verify LOGGER.error was called
        assert mock_logger.error.called
        # Get the logged message
        log_message = mock_logger.error.call_args[0][0]
        # Check that traceback information is in the log message
        assert "Traceback:" in log_message
        assert "inner_func" in log_message
        assert "ValueError" in log_message

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_traceback_shows_call_stack(self, mock_logger):
        """Test that traceback shows the complete call stack."""

        @error_handler(func_name="outer_func")
        def outer_function():
            def middle_function():
                def inner_function():
                    raise MilvusException(ErrorCode.UNEXPECTED_ERROR, "deep error")

                inner_function()

            middle_function()

        with pytest.raises(MilvusException):
            outer_function()

        # Verify LOGGER.error was called
        assert mock_logger.error.called
        # Get the logged message
        log_message = mock_logger.error.call_args[0][0]
        # Verify the complete call stack is present
        assert "Traceback:" in log_message
        assert "outer_function" in log_message
        assert "middle_function" in log_message
        assert "inner_function" in log_message

    @pytest.mark.asyncio
    @patch("pymilvus.decorators.LOGGER")
    async def test_async_error_handler_includes_traceback(self, mock_logger):
        """Test that async error_handler logs traceback."""

        @error_handler(func_name="test_async_func")
        async def async_func_that_raises():
            def inner_func():
                raise MilvusException(ErrorCode.UNEXPECTED_ERROR, "async test error")

            inner_func()

        with pytest.raises(MilvusException):
            await async_func_that_raises()

        # Verify LOGGER.error was called
        assert mock_logger.error.called
        # Get the logged message
        log_message = mock_logger.error.call_args[0][0]
        # Check that traceback information is in the log message
        assert "Traceback:" in log_message
        assert "inner_func" in log_message
        assert "test_async_func" in log_message
