import asyncio
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


def assert_preserves_metadata(func, expected_name):
    assert func.__name__ == expected_name


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


class _MockMilvusError(MilvusException):
    def __init__(self, code, message, compatible_code):
        super(MilvusException, self).__init__(message)
        self._code = code
        self._message = message
        self._compatible_code = compatible_code

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message

    @property
    def compatible_code(self):
        return self._compatible_code


class MockForceDenyError(_MockMilvusError):
    def __init__(self, code=ErrorCode.FORCE_DENY, message="force deny"):
        super().__init__(code, message, common_pb2.ForceDeny)


class MockRateLimitError(_MockMilvusError):
    def __init__(self, code=ErrorCode.RATE_LIMIT, message="rate limit"):
        super().__init__(code, message, common_pb2.RateLimit)


class MockUnRetriableError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, message="unretriable error"):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "details of unretriable error"


class MockUnimplementedError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNIMPLEMENTED

    def details(self):
        return "Method not implemented"


class MockFutureTimeoutError(grpc.FutureTimeoutError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "Future timed out"


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
        count = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code):
            nonlocal count
            count += 1
            self.mock_failure(code)

        with pytest.raises(MilvusException, match="unavailable"):
            test_api(self, grpc.StatusCode.UNAVAILABLE)

        assert count == times + 1

    def test_retry_decorators_timeout(self):
        count = 0

        @retry_on_rpc_failure()
        def test_api(self, code, timeout=None):
            nonlocal count
            count += 1
            time.sleep(1)
            self.mock_failure(code)

        with pytest.raises(MilvusException):
            test_api(self, grpc.StatusCode.UNAVAILABLE, timeout=1)

        assert count == 1

    @pytest.mark.skip("Do not open this unless you have loads of time, get some coffee and wait")
    def test_retry_decorators_default_behaviour(self):
        count = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            nonlocal count
            count += 1
            self.mock_failure(code)

        with pytest.raises(MilvusException):
            test_api(self, grpc.StatusCode.UNAVAILABLE)

        assert count == 7 + 1

    def test_retry_decorators_force_deny(self):
        count = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            nonlocal count
            count += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException, match="force deny"):
            test_api(self, ErrorCode.FORCE_DENY)

        assert count == 1

    def test_retry_decorators_set_retry_times(self):
        count = 0

        @retry_on_rpc_failure()
        def test_api(self, code, retry_on_rate_limit, **kwargs):
            nonlocal count
            count += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException):
            test_api(self, ErrorCode.RATE_LIMIT, retry_on_rate_limit=True, retry_times=3)

        assert count == 3 + 1

    @pytest.mark.parametrize(
        "do_retry, expected_count_fn",
        [
            (False, lambda times: 1),
            (True, lambda times: times + 1),
        ],
        ids=["without_retry", "with_retry"],
    )
    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_rate_limit(self, times, do_retry, expected_count_fn):
        count = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code, retry_on_rate_limit):
            nonlocal count
            count += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException, match="rate limit"):
            test_api(self, ErrorCode.RATE_LIMIT, retry_on_rate_limit=do_retry)

        assert count == expected_count_fn(times)

    @pytest.mark.parametrize("code", IGNORE_RETRY_CODES)
    def test_donot_retry_codes(self, code):
        count = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            nonlocal count
            count += 1
            self.mock_failure(code)

        with pytest.raises(grpc.RpcError):
            test_api(self, code)

        assert count == 1


class MockSelfForTracing:
    def set_onetime_loglevel(self, level):
        pass


class TestErrorHandlerTraceback:
    @pytest.mark.parametrize(
        "func_name, raise_exc, expected_strings, expected_exc",
        [
            (
                "test_func",
                MilvusException(ErrorCode.UNEXPECTED_ERROR, "test error"),
                ["Traceback:", "inner_func", "test_func"],
                MilvusException,
            ),
            (
                "test_grpc_func",
                MockUnavailableError(),
                ["Traceback:", "inner_func", "test_grpc_func"],
                grpc.RpcError,
            ),
            (
                "test_generic_func",
                ValueError("test generic error"),
                ["Traceback:", "inner_func", "ValueError"],
                MilvusException,
            ),
        ],
        ids=["milvus_exception", "grpc_error", "generic_exception"],
    )
    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_includes_traceback(
        self, mock_logger, func_name, raise_exc, expected_strings, expected_exc
    ):
        @error_handler(func_name=func_name)
        def func_that_raises():
            def inner_func():
                raise raise_exc

            inner_func()

        with pytest.raises(expected_exc):
            func_that_raises()

        assert mock_logger.error.called
        log_message = mock_logger.error.call_args[0][0]
        for s in expected_strings:
            assert s in log_message

    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_traceback_shows_call_stack(self, mock_logger):
        @error_handler(func_name="outer_func")
        def outer_function():
            def middle_function():
                def inner_function():
                    raise MilvusException(ErrorCode.UNEXPECTED_ERROR, "deep error")

                inner_function()

            middle_function()

        with pytest.raises(MilvusException):
            outer_function()

        assert mock_logger.error.called
        log_message = mock_logger.error.call_args[0][0]
        assert "Traceback:" in log_message
        assert "outer_function" in log_message
        assert "middle_function" in log_message
        assert "inner_function" in log_message

    @pytest.mark.asyncio
    @patch("pymilvus.decorators.LOGGER")
    async def test_async_error_handler_includes_traceback(self, mock_logger):
        @error_handler(func_name="test_async_func")
        async def async_func_that_raises():
            def inner_func():
                raise MilvusException(ErrorCode.UNEXPECTED_ERROR, "async test error")

            inner_func()

        with pytest.raises(MilvusException):
            await async_func_that_raises()

        assert mock_logger.error.called
        log_message = mock_logger.error.call_args[0][0]
        assert "Traceback:" in log_message
        assert "inner_func" in log_message
        assert "test_async_func" in log_message


class TestRetryDecoratorEdgeCases:
    def test_decorator_preserves_function_metadata(self):
        @retry_on_rpc_failure()
        def my_documented_function(self_arg):
            """This is a docstring."""

        assert_preserves_metadata(my_documented_function, "my_documented_function")
        assert "docstring" in my_documented_function.__doc__

    def test_decorator_with_custom_initial_back_off(self):
        mock_self = MockSelfForTracing()
        call_times = []

        @retry_on_rpc_failure(retry_times=2, initial_back_off=0.001, max_back_off=0.01)
        def failing_func(self_arg):
            call_times.append(time.time())
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            failing_func(mock_self)

        assert len(call_times) == 3

    def test_decorator_with_custom_max_back_off(self):
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=3, initial_back_off=0.001, max_back_off=0.002)
        def failing_func(self_arg):
            nonlocal call_count
            call_count += 1
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            failing_func(mock_self)

        assert call_count == 4

    def test_decorator_with_zero_retry_times(self):
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=0)
        def failing_func(self_arg):
            nonlocal call_count
            call_count += 1
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            failing_func(mock_self)

        assert call_count == 1

    def test_successful_function_returns_value(self):
        mock_self = MockSelfForTracing()

        @retry_on_rpc_failure()
        def successful_func(self_arg):
            return "success"

        result = successful_func(mock_self)
        assert result == "success"

    def test_function_succeeds_after_initial_failures(self):
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
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=5)
        def func_deadline_exceeded(self_arg):
            nonlocal call_count
            call_count += 1
            raise MockDeadlineExceededError

        with pytest.raises(grpc.RpcError):
            func_deadline_exceeded(mock_self)

        assert call_count == 1

    def test_back_off_multiplier_effect(self):
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
        first_interval = call_times[1] - call_times[0]
        second_interval = call_times[2] - call_times[1]
        assert second_interval > first_interval * 2


class TestDeprecatedDecorator:
    @patch("pymilvus.decorators.LOGGER")
    def test_deprecated_logs_warning(self, mock_logger):
        @deprecated
        def old_function():
            return "result"

        result = old_function()
        assert result == "result"
        assert mock_logger.warning.called

    def test_deprecated_preserves_function_metadata(self):
        @deprecated
        def my_old_function():
            """Old function docstring."""

        assert_preserves_metadata(my_old_function, "my_old_function")

    @patch("pymilvus.decorators.LOGGER")
    def test_deprecated_passes_arguments(self, mock_logger):
        @deprecated
        def old_function_with_args(a, b, c=None):
            return a + b + (c or 0)

        result = old_function_with_args(1, 2, c=3)
        assert result == 6


class TestTracingRequestDecorator:
    def test_tracing_request_calls_function(self):
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg):
            return "result"

        result = test_func(mock_self)
        assert result == "result"

    def test_tracing_request_with_log_level(self):
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg, **kwargs):
            return "result"

        result = test_func(mock_self, log_level="DEBUG")
        assert result == "result"
        mock_self.set_onetime_loglevel.assert_called_once_with("DEBUG")

    def test_tracing_request_with_log_level_hyphen(self):
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg, **kwargs):
            return "result"

        result = test_func(mock_self, **{"log-level": "INFO"})
        assert result == "result"
        mock_self.set_onetime_loglevel.assert_called_once_with("INFO")

    def test_tracing_request_without_log_level(self):
        mock_self = MagicMock()

        @tracing_request()
        def test_func(self_arg, **kwargs):
            return "result"

        result = test_func(mock_self)
        assert result == "result"
        mock_self.set_onetime_loglevel.assert_not_called()

    def test_tracing_request_preserves_function_metadata(self):
        @tracing_request()
        def my_traced_function(self_arg):
            """Traced function docstring."""

        assert_preserves_metadata(my_traced_function, "my_traced_function")

    @pytest.mark.asyncio
    async def test_async_tracing_request_with_log_level(self):
        mock_self = MagicMock()

        @tracing_request()
        async def async_test_func(self_arg, **kwargs):
            return "async_result"

        result = await async_test_func(mock_self, log_level="WARNING")
        assert result == "async_result"
        mock_self.set_onetime_loglevel.assert_called_once_with("WARNING")


class TestIgnoreUnimplementedDecorator:
    def test_ignore_unimplemented_returns_value_on_success(self):
        @ignore_unimplemented(default_return_value="default")
        def successful_func():
            return "actual_value"

        result = successful_func()
        assert result == "actual_value"

    def test_ignore_unimplemented_returns_default_on_unimplemented(self):
        @ignore_unimplemented(default_return_value="default")
        def unimplemented_func():
            raise MockUnimplementedError

        result = unimplemented_func()
        assert result == "default"

    def test_ignore_unimplemented_raises_other_grpc_errors(self):
        @ignore_unimplemented(default_return_value="default")
        def unavailable_func():
            raise MockUnavailableError

        with pytest.raises(grpc.RpcError):
            unavailable_func()

    def test_ignore_unimplemented_raises_non_grpc_errors(self):
        @ignore_unimplemented(default_return_value="default")
        def error_func():
            raise ValueError("Some error")

        with pytest.raises(ValueError):
            error_func()

    def test_ignore_unimplemented_preserves_function_metadata(self):
        @ignore_unimplemented(default_return_value=None)
        def my_function():
            """My function docstring."""

        assert_preserves_metadata(my_function, "my_function")

    def test_ignore_unimplemented_with_none_default(self):
        @ignore_unimplemented(default_return_value=None)
        def unimplemented_func():
            raise MockUnimplementedError

        result = unimplemented_func()
        assert result is None

    def test_ignore_unimplemented_with_dict_default(self):
        @ignore_unimplemented(default_return_value={"status": "unimplemented"})
        def unimplemented_func():
            raise MockUnimplementedError

        result = unimplemented_func()
        assert result == {"status": "unimplemented"}


class TestUpgradeReminderDecorator:
    def test_upgrade_reminder_returns_value_on_success(self):
        @upgrade_reminder
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_upgrade_reminder_raises_milvus_exception_on_unimplemented(self):
        @upgrade_reminder
        def unimplemented_func():
            raise MockUnimplementedError

        with pytest.raises(MilvusException) as exc_info:
            unimplemented_func()

        assert "sdk is incompatible with server" in exc_info.value.message

    def test_upgrade_reminder_raises_other_grpc_errors(self):
        @upgrade_reminder
        def unavailable_func():
            raise MockUnavailableError

        with pytest.raises(grpc.RpcError):
            unavailable_func()

    def test_upgrade_reminder_raises_non_grpc_errors(self):
        @upgrade_reminder
        def error_func():
            raise RuntimeError("Runtime error")

        with pytest.raises(RuntimeError):
            error_func()

    def test_upgrade_reminder_preserves_function_metadata(self):
        @upgrade_reminder
        def my_upgraded_function():
            """Upgraded function docstring."""

        assert_preserves_metadata(my_upgraded_function, "my_upgraded_function")


class TestRetryOnSchemaMismatchDecorator:
    def test_retry_on_schema_mismatch_success(self):
        mock_self = MagicMock()

        @retry_on_schema_mismatch()
        def successful_func(self_arg, collection_name, **kwargs):
            return "success"

        result = successful_func(mock_self, "test_collection")
        assert result == "success"
        mock_self._invalidate_schema.assert_not_called()

    def test_retry_on_schema_mismatch_retries_on_data_not_match(self):
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

        assert call_count == 2

    def test_retry_on_schema_mismatch_raises_param_error_without_context(self):
        mock_self = MagicMock()

        @retry_on_schema_mismatch()
        def func_without_context(self_arg, collection_name, **kwargs):
            raise DataNotMatchException(message="Data mismatch")

        with pytest.raises(ParamError) as exc_info:
            func_without_context(mock_self, "test_collection")

        assert "context is required" in exc_info.value.message

    def test_retry_on_schema_mismatch_preserves_function_metadata(self):
        @retry_on_schema_mismatch()
        def my_schema_func(self_arg, collection_name):
            """Schema function docstring."""

        assert_preserves_metadata(my_schema_func, "my_schema_func")

    @pytest.mark.asyncio
    async def test_async_retry_on_schema_mismatch_success(self):
        mock_self = MagicMock()

        @retry_on_schema_mismatch()
        async def async_successful_func(self_arg, collection_name, **kwargs):
            return "async_success"

        result = await async_successful_func(mock_self, "test_collection")
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_retry_on_schema_mismatch_retries(self):
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
    @patch("pymilvus.decorators.LOGGER")
    def test_error_handler_with_empty_func_name(self, mock_logger):
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
        @error_handler(func_name="successful_func")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"
        mock_logger.error.assert_not_called()


class TestAsyncRetryTimeout:
    @pytest.mark.asyncio
    async def test_async_timeout_enforced_on_blocking_call(self):
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure()
        async def blocking_func(self_arg, timeout=None):
            nonlocal call_count
            call_count += 1
            # Simulate a call that blocks much longer than timeout
            await asyncio.sleep(60)

        start = time.time()
        with pytest.raises(MilvusException, match="Retry timeout"):
            await blocking_func(mock_self, timeout=1)
        elapsed = time.time() - start

        assert elapsed < 5, f"Expected to finish within 5s, took {elapsed:.1f}s"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_timeout_not_affected_when_no_timeout(self):
        mock_self = MockSelfForTracing()
        call_count = 0

        @retry_on_rpc_failure(retry_times=2)
        async def failing_func(self_arg, timeout=None):
            nonlocal call_count
            call_count += 1
            raise MockUnavailableError

        with pytest.raises(MilvusException):
            await failing_func(mock_self)

        assert call_count == 3  # initial + 2 retries
