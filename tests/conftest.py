import sys
import pytest
import grpc_testing

# https://github.com/grpc/grpc/blob/5918f98ecbf5ace77f30fa97f7fc3e8bdac08e04/src/python/grpcio_tests/tests/testing/_client_test.py
from grpc.framework.foundation import logging_pool

from pymilvus.grpc_gen import milvus_pb2

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

descriptor = milvus_pb2.DESCRIPTOR.services_by_name['MilvusService']


@pytest.fixture(scope="function")
def channel(request):
    channel = grpc_testing.channel([descriptor], grpc_testing.strict_real_time())

    return channel


@pytest.fixture(scope="function")
def client_thread(request):
    client_execution_thread_pool = logging_pool.pool(2)

    def teardown():
        client_execution_thread_pool.shutdown(wait=True)

    return client_execution_thread_pool
