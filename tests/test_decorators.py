import time

import grpc
import pytest
from pymilvus.decorators import IGNORE_RETRY_CODES, retry_on_rpc_failure
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
