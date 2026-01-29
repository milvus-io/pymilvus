import time
from unittest.mock import MagicMock, patch

import grpc
import pytest
from pymilvus.decorators import (
    IGNORE_RETRY_CODES,
    deprecated,
    error_handler,
    ignore_unimplemented,
    retry_on_rpc_failure,
    retry_on_schema_mismatch,
    tracing_request,
    upgrade_reminder,
)
from pymilvus.exceptions import (
    DataNotMatchException,
    ErrorCode,
    MilvusException,
    ParamError,
    SchemaMismatchRetryableException,
)
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
                raise MockUnavailableError

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


class MockSelfForTracing:
    """Mock self object that provides set_onetime_loglevel for tracing_request decorator."""

    def set_onetime_loglevel(self, level):
        pass


class TestRetryDecoratorEdgeCases:
    """Tests for retry decorator edge cases and configuration."""

    def test_decorator_preserves_function_metadata(self):
        """Test that retry_on_rpc_failure preserves function name and docstring."""

        @retry_on_rpc_failure()
        def my_documented_function(self_arg):
            """This is a docstring."""

        assert my_documented_function.__name__ == "my_documented_function"
        # The decorator chain means the docstring is preserved
        assert "docstring" in my_documented_function.__doc__

    def test_decorator_with_custom_initial_back_off(self):
        """Test retry decorator with custom initial_back_off parameter."""
        mock_self = MockSelfForTracing()
        call_times = []

        @retry_on_rpc_failure(retry_times=2, initial_back_off=0.001, max_back_off=0.01)
        def failing_func(self_arg):
            call_times.append(time.time())
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            failing_func(mock_self)

        # Should have 3 calls (1 initial + 2 retries)
        assert len(call_times) == 3

    def test_decorator_with_custom_max_back_off(self):
        """Test retry decorator respects max_back_off parameter."""
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=3, initial_back_off=0.001, max_back_off=0.002)
        def failing_func(self_arg):
            nonlocal call_count
            call_count += 1
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            failing_func(mock_self)

        assert call_count == 4  # 1 initial + 3 retries

    def test_decorator_with_zero_retry_times(self):
        """Test that retry_times=0 means no retries."""
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=0)
        def failing_func(self_arg):
            nonlocal call_count
            call_count += 1
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            failing_func(mock_self)

        assert call_count == 1  # Only the initial call

    def test_successful_function_returns_value(self):
        """Test that a successful function returns its value without retry."""
        mock_self = MockSelfForTracing()

        @retry_on_rpc_failure()
        def successful_func(self_arg):
            return "success"

        result = successful_func(mock_self)
        assert result == "success"

    def test_function_succeeds_after_initial_failures(self):
        """Test function that succeeds after initial failures."""
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=5, initial_back_off=0.001)
        def sometimes_failing_func(self_arg):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise MockUnavailableError
            return "eventual_success"

        result = sometimes_failing_func(mock_self)
        assert result == "eventual_success"
        assert call_count == 3

    def test_deadline_exceeded_not_retried(self):
        """Test that DEADLINE_EXCEEDED errors are not retried."""
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=5)
        def func_deadline_exceeded(self_arg):
            nonlocal call_count
            call_count += 1
            raise MockDeadlineExceededError

        with pytest.raises(grpc.RpcError):
            func_deadline_exceeded(mock_self)

        assert call_count == 1  # No retry for deadline exceeded

    def test_back_off_multiplier_effect(self):
        """Test that back_off_multiplier increases wait time."""
        mock_self = MockSelfForTracing()
        call_times = []

        # Use larger initial_back_off (0.05s) to avoid Windows timer resolution issues.
        # Windows has ~15.6ms minimum sleep resolution, so small values get rounded up.
        @retry_on_rpc_failure(
            retry_times=2, initial_back_off=0.05, max_back_off=1, back_off_multiplier=3
        )
        def failing_func(self_arg):
            call_times.append(time.time())
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            failing_func(mock_self)

        assert len(call_times) == 3
        # Second interval should be roughly 3x the first (due to multiplier)
        # First sleep: 0.05s, Second sleep: 0.15s
        first_interval = call_times[1] - call_times[0]
        second_interval = call_times[2] - call_times[1]
        # Allow some tolerance for timing (expect ~3x ratio, assert > 2x)
        assert second_interval > first_interval * 2


class TestDeprecatedDecorator:
    """Tests for the deprecated decorator."""

    @patch("pymilvus.decorators.LOGGER")
    def test_deprecated_logs_warning(self, mock_logger):
        """Test that deprecated decorator logs a warning."""

        @deprecated
        def old_function():
            return "result"

        result = old_function()
        assert result == "result"
        assert mock_logger.warning.called

    def test_deprecated_preserves_function_metadata(self):
        """Test that deprecated decorator preserves function name."""

        @deprecated
        def my_old_function():
            """Old function docstring."""

        assert my_old_function.__name__ == "my_old_function"

    @patch("pymilvus.decorators.LOGGER")
    def test_deprecated_passes_arguments(self, mock_logger):
        """Test that deprecated decorator passes arguments correctly."""

        @deprecated
        def old_function_with_args(a, b, c=None):
            return a + b + (c or 0)

        result = old_function_with_args(1, 2, c=3)
        assert result == 6


class TestTracingRequestDecorator:
    """Tests for the tracing_request decorator."""

    def test_tracing_request_calls_function(self):
        """Test that tracing_request decorator calls the wrapped function."""
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg):
            return "result"

        result = test_func(mock_self)
        assert result == "result"

    def test_tracing_request_with_log_level(self):
        """Test that tracing_request sets log level when provided."""
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg, **kwargs):
            return "result"

        result = test_func(mock_self, log_level="DEBUG")
        assert result == "result"
        mock_self.set_onetime_loglevel.assert_called_once_with("DEBUG")

    def test_tracing_request_with_log_level_hyphen(self):
        """Test that tracing_request supports log-level with hyphen."""
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg, **kwargs):
            return "result"

        result = test_func(mock_self, **{"log-level": "INFO"})
        assert result == "result"
        mock_self.set_onetime_loglevel.assert_called_once_with("INFO")

    def test_tracing_request_without_log_level(self):
        """Test that tracing_request does not call set_onetime_loglevel without log level."""
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg, **kwargs):
            return "result"

        result = test_func(mock_self)
        assert result == "result"
        mock_self.set_onetime_loglevel.assert_not_called()

    def test_tracing_request_preserves_function_metadata(self):
        """Test that tracing_request preserves function metadata."""

        @tracing_request()
        def my_traced_function(self_arg):
            """Traced function docstring."""

        assert my_traced_function.__name__ == "my_traced_function"

    @pytest.mark.asyncio
    async def test_async_tracing_request_with_log_level(self):
        """Test async tracing_request sets log level when provided."""
        mock_self = MagicMock()

        @tracing_request()
        async def async_test_func(self_arg, **kwargs):
            return "async_result"

        result = await async_test_func(mock_self, log_level="WARNING")
        assert result == "async_result"
        mock_self.set_onetime_loglevel.assert_called_once_with("WARNING")


class TestIgnoreUnimplementedDecorator:
    """Tests for the ignore_unimplemented decorator."""

    def test_ignore_unimplemented_returns_value_on_success(self):
        """Test that successful function returns its value."""

        @ignore_unimplemented(default_return_value="default")
        def successful_func():
            return "actual_value"

        result = successful_func()
        assert result == "actual_value"

    def test_ignore_unimplemented_returns_default_on_unimplemented(self):
        """Test that UNIMPLEMENTED error returns default value."""

        @ignore_unimplemented(default_return_value="default")
        def unimplemented_func():
            raise MockUnimplementedError

        result = unimplemented_func()
        assert result == "default"

    def test_ignore_unimplemented_raises_other_grpc_errors(self):
        """Test that other gRPC errors are raised."""

        @ignore_unimplemented(default_return_value="default")
        def unavailable_func():
            raise MockUnavailableError

        with pytest.raises(grpc.RpcError):
            unavailable_func()

    def test_ignore_unimplemented_raises_non_grpc_errors(self):
        """Test that non-gRPC errors are raised."""

        @ignore_unimplemented(default_return_value="default")
        def error_func():
            raise ValueError("Some error")

        with pytest.raises(ValueError):
            error_func()

    def test_ignore_unimplemented_preserves_function_metadata(self):
        """Test that ignore_unimplemented preserves function metadata."""

        @ignore_unimplemented(default_return_value=None)
        def my_function():
            """My function docstring."""

        assert my_function.__name__ == "my_function"

    def test_ignore_unimplemented_with_none_default(self):
        """Test ignore_unimplemented with None as default value."""

        @ignore_unimplemented(default_return_value=None)
        def unimplemented_func():
            raise MockUnimplementedError

        result = unimplemented_func()
        assert result is None

    def test_ignore_unimplemented_with_dict_default(self):
        """Test ignore_unimplemented with dict as default value."""

        @ignore_unimplemented(default_return_value={"status": "unimplemented"})
        def unimplemented_func():
            raise MockUnimplementedError

        result = unimplemented_func()
        assert result == {"status": "unimplemented"}


class TestUpgradeReminderDecorator:
    """Tests for the upgrade_reminder decorator."""

    def test_upgrade_reminder_returns_value_on_success(self):
        """Test that successful function returns its value."""

        @upgrade_reminder
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_upgrade_reminder_raises_milvus_exception_on_unimplemented(self):
        """Test that UNIMPLEMENTED error raises MilvusException with upgrade message."""

        @upgrade_reminder
        def unimplemented_func():
            raise MockUnimplementedError

        with pytest.raises(MilvusException) as exc_info:
            unimplemented_func()

        assert "sdk is incompatible with server" in exc_info.value.message

    def test_upgrade_reminder_raises_other_grpc_errors(self):
        """Test that other gRPC errors are raised as-is."""

        @upgrade_reminder
        def unavailable_func():
            raise MockUnavailableError

        with pytest.raises(grpc.RpcError):
            unavailable_func()

    def test_upgrade_reminder_raises_non_grpc_errors(self):
        """Test that non-gRPC errors are raised as-is."""

        @upgrade_reminder
        def error_func():
            raise RuntimeError("Runtime error")

        with pytest.raises(RuntimeError):
            error_func()

    def test_upgrade_reminder_preserves_function_metadata(self):
        """Test that upgrade_reminder preserves function metadata."""

        @upgrade_reminder
        def my_upgraded_function():
            """Upgraded function docstring."""

        assert my_upgraded_function.__name__ == "my_upgraded_function"


class TestRetryOnSchemaMismatchDecorator:
    """Tests for the retry_on_schema_mismatch decorator."""

    def test_retry_on_schema_mismatch_success(self):
        """Test that successful function returns without retry."""
        mock_self = MagicMock()

        @retry_on_schema_mismatch()
        def successful_func(self_arg, collection_name, **kwargs):
            return "success"

        result = successful_func(mock_self, "test_collection")
        assert result == "success"
        mock_self._invalidate_schema.assert_not_called()

    def test_retry_on_schema_mismatch_retries_on_data_not_match(self):
        """Test that DataNotMatchException triggers retry."""
        mock_self = MagicMock()
        mock_context = MagicMock()
        mock_context.get_db_name.return_value = "test_db"
        call_count = 0

        @retry_on_schema_mismatch()
        def data_mismatch_func(self_arg, collection_name, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise DataNotMatchException(message="Data mismatch")
            return "success_after_retry"

        result = data_mismatch_func(mock_self, "test_collection", context=mock_context)
        assert result == "success_after_retry"
        assert call_count == 2
        mock_self._invalidate_schema.assert_called_once_with("test_collection", db_name="test_db")

    def test_retry_on_schema_mismatch_retries_on_schema_mismatch_exception(self):
        """Test that SchemaMismatchRetryableException triggers retry."""
        mock_self = MagicMock()
        mock_context = MagicMock()
        mock_context.get_db_name.return_value = "db1"
        call_count = 0

        @retry_on_schema_mismatch()
        def schema_mismatch_func(self_arg, collection_name, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SchemaMismatchRetryableException(message="Schema mismatch")
            return "retry_success"

        result = schema_mismatch_func(mock_self, "my_collection", context=mock_context)
        assert result == "retry_success"
        assert call_count == 2
        mock_self._invalidate_schema.assert_called_once_with("my_collection", db_name="db1")

    def test_retry_on_schema_mismatch_raises_after_max_retries(self):
        """Test that exception is raised after max retries (2 attempts)."""
        mock_self = MagicMock()
        mock_context = MagicMock()
        mock_context.get_db_name.return_value = "test_db"
        call_count = 0

        @retry_on_schema_mismatch()
        def always_failing_func(self_arg, collection_name, **kwargs):
            nonlocal call_count
            call_count += 1
            raise DataNotMatchException(message="Always fails")

        with pytest.raises(DataNotMatchException):
            always_failing_func(mock_self, "test_collection", context=mock_context)

        assert call_count == 2  # Initial + 1 retry

    def test_retry_on_schema_mismatch_raises_param_error_without_context(self):
        """Test that ParamError is raised when context is not provided."""
        mock_self = MagicMock()

        @retry_on_schema_mismatch()
        def func_without_context(self_arg, collection_name, **kwargs):
            raise DataNotMatchException(message="Data mismatch")

        with pytest.raises(ParamError) as exc_info:
            func_without_context(mock_self, "test_collection")

        assert "context is required" in exc_info.value.message

    def test_retry_on_schema_mismatch_preserves_function_metadata(self):
        """Test that retry_on_schema_mismatch preserves function metadata."""

        @retry_on_schema_mismatch()
        def my_schema_func(self_arg, collection_name):
            """Schema function docstring."""

        assert my_schema_func.__name__ == "my_schema_func"

    @pytest.mark.asyncio
    async def test_async_retry_on_schema_mismatch_success(self):
        """Test async retry_on_schema_mismatch with successful function."""
        mock_self = MagicMock()

        @retry_on_schema_mismatch()
        async def async_successful_func(self_arg, collection_name, **kwargs):
            return "async_success"

        result = await async_successful_func(mock_self, "test_collection")
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_retry_on_schema_mismatch_retries(self):
        """Test async retry_on_schema_mismatch triggers retry on mismatch."""
        mock_self = MagicMock()
        mock_context = MagicMock()
        mock_context.get_db_name.return_value = "async_db"
        call_count = 0

        @retry_on_schema_mismatch()
        async def async_mismatch_func(self_arg, collection_name, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise DataNotMatchException(message="Async data mismatch")
            return "async_retry_success"

        result = await async_mismatch_func(mock_self, "async_collection", context=mock_context)
        assert result == "async_retry_success"
        assert call_count == 2
        mock_self._invalidate_schema.assert_called_once_with("async_collection", db_name="async_db")


class TestErrorHandlerEdgeCases:
    """Additional edge case tests for error_handler decorator."""

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_with_empty_func_name(self, mock_logger):
        """Test error_handler uses function name when func_name is empty."""

        @error_handler(func_name="")
        def my_function_name():
            raise MilvusException(ErrorCode.UNEXPECTED_ERROR, "test error")

        with pytest.raises(MilvusException):
            my_function_name()

        assert mock_logger.error.called
        log_message = mock_logger.error.call_args[0][0]
        assert "my_function_name" in log_message

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_grpc_future_timeout_error(self, mock_logger):
        """Test error_handler handles grpc.FutureTimeoutError."""

        @error_handler(func_name="timeout_func")
        def func_that_times_out():
            raise MockFutureTimeoutError

        with pytest.raises(grpc.FutureTimeoutError):
            func_that_times_out()

        assert mock_logger.error.called
        log_message = mock_logger.error.call_args[0][0]
        assert "gRPC timeout" in log_message.lower() or "grpc Timeout" in log_message

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_successful_function_no_logging(self, mock_logger):
        """Test that successful function does not trigger error logging."""

        @error_handler(func_name="successful_func")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"
        mock_logger.error.assert_not_called()


class MockUnimplementedError(grpc.RpcError):
    """Mock gRPC UNIMPLEMENTED error."""

    def code(self):
        return grpc.StatusCode.UNIMPLEMENTED

    def details(self):
        return "Method not implemented"


class MockFutureTimeoutError(grpc.FutureTimeoutError):
    """Mock gRPC FutureTimeoutError."""

    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "Future timed out"
