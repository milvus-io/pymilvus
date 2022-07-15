import grpc
import pytest

from pymilvus.decorators import retry_on_rpc_failure, error_handler
from pymilvus.exceptions import MilvusUnavaliableException, MilvusException


class TestDecorators:
    def mock_failure(self, code: grpc.StatusCode):
        if code == MockUnavailableError().code():
            raise MockUnavailableError()
        if code == MockDeadlineExceededError().code():
            raise MockDeadlineExceededError()

    @pytest.mark.parametrize("times", [0, 1, 2, 3])
    def test_retry_decorators_unavailable(self, times):
        self.count_test_retry_decorators_unavaliable = 0

        @retry_on_rpc_failure(retry_times=times)
        def test_api(self, code):
            self.count_test_retry_decorators_unavaliable += 1
            self.mock_failure(code)

        with pytest.raises(MilvusUnavaliableException) as e:
            test_api(self, grpc.StatusCode.UNAVAILABLE)
            assert "unavailable" in e.reason()
            print(e)

        # the first execute + retry times
        assert self.count_test_retry_decorators_unavaliable == times + 1

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
