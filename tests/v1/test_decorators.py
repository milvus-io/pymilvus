import grpc
import pytest

from pymilvus.decorators import retry_on_rpc_failure, error_handler
from pymilvus.exceptions import MilvusUnavailableException, MilvusException
from pymilvus.grpc_gen import common_pb2


class TestDecorators:
    def mock_failure(self, code: grpc.StatusCode):
        if code == MockUnavailableError().code():
            raise MockUnavailableError()
        if code == MockDeadlineExceededError().code():
            raise MockDeadlineExceededError()

    def mock_milvus_exception(self, code: common_pb2.ErrorCode):
        if code == common_pb2.ForceDeny:
            raise MockForceDenyError()
        if code == common_pb2.RateLimit:
            raise MockRateLimitError()
        raise MilvusException(common_pb2.UNEXPECTED_ERROR, str("unexpected error"))

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_unavailable(self, times):
        self.count_test_retry_decorators_Unavailable = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code):
            self.count_test_retry_decorators_Unavailable += 1
            self.mock_failure(code)

        with pytest.raises(MilvusUnavailableException) as e:
            test_api(self, grpc.StatusCode.UNAVAILABLE)
            assert "unavailable" in e.reason()
            print(e)

        # the first execute + retry times
        assert self.count_test_retry_decorators_Unavailable == times + 1

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_deadline_exceeded(self, times):
        self.count_test_retry_decorators_deadline_exceeded = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code):
            self.count_test_retry_decorators_deadline_exceeded += 1
            self.mock_failure(code)

        with pytest.raises(MilvusException) as e:
            test_api(self, grpc.StatusCode.DEADLINE_EXCEEDED)
            assert "deadline exceeded" in e.reason()
            print(e)

        # the first execute + retry times
        assert self.count_test_retry_decorators_deadline_exceeded == times + 1

    def test_retry_decorators_timeout(self):
        self.count_test_retry_decorators_timeout = 0

        @retry_on_rpc_failure()
        def test_api(self, code, timeout=None):
            self.count_test_retry_decorators_timeout += 1
            self.mock_failure(code)

        with pytest.raises(MilvusException) as e:
            test_api(self, grpc.StatusCode.DEADLINE_EXCEEDED, timeout=1)
            print(e)

        assert self.count_test_retry_decorators_timeout == 6

    @pytest.mark.skip("Do not open this unless you have loads of time, get some coffee and wait")
    def test_retry_decorators_default_behaviour(self):
        self.test_retry_decorators_default_retry_times = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            self.test_retry_decorators_default_retry_times += 1
            self.mock_failure(code)

        with pytest.raises(MilvusException) as e:
            test_api(self, grpc.StatusCode.DEADLINE_EXCEEDED)
            print(e)

        assert self.test_retry_decorators_default_retry_times == 7 + 1

    def test_retry_decorators_force_deny(self):
        self.execute_times = 0

        @retry_on_rpc_failure()
        def test_api(self, code):
            self.execute_times += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException) as e:
            test_api(self, common_pb2.ForceDeny)
        print(e)
        assert "force deny" in e.value.message

        # the first execute + 0 retry times
        assert self.execute_times == 1

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_rate_limit_without_retry(self, times):
        self.count_test_retry_decorators_force_deny = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code, retry_on_rate_limit):
            self.count_test_retry_decorators_force_deny += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException) as e:
            test_api(self, common_pb2.RateLimit, retry_on_rate_limit=False)
        print(e)
        assert "rate limit" in e.value.message

        # the first execute + 0 retry times
        assert self.count_test_retry_decorators_force_deny == 1

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_rate_limit_with_retry(self, times):
        self.count_test_retry_decorators_force_deny = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code, retry_on_rate_limit):
            self.count_test_retry_decorators_force_deny += 1
            self.mock_milvus_exception(code)

        with pytest.raises(MilvusException) as e:
            test_api(self, common_pb2.RateLimit, retry_on_rate_limit=True)
        print(e)
        assert "rate limit" in e.value.message

        # the first execute + retry times
        assert self.count_test_retry_decorators_force_deny == times + 1


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
    def __init__(self, code=common_pb2.ForceDeny, message="force deny"):
        super(MilvusException, self).__init__(message)
        self._code = code
        self._message = message

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message

class MockRateLimitError(MilvusException):
    def __init__(self, code=common_pb2.RateLimit, message="rate limit"):
        super(MilvusException, self).__init__(message)
        self._code = code
        self._message = message

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message
