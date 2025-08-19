import logging
import sys
from pathlib import Path

import grpc_testing
import pytest

# https://github.com/grpc/grpc/blob/5918f98ecbf5ace77f30fa97f7fc3e8bdac08e04/src/python/grpcio_tests/tests/testing/_client_test.py
from grpc.framework.foundation import logging_pool

logging.getLogger("faker").setLevel(logging.WARNING)

from os.path import abspath, dirname

from pymilvus.grpc_gen import milvus_pb2

sys.path.append(Path(__file__).absolute().parent.parent)


descriptor = milvus_pb2.DESCRIPTOR.services_by_name['MilvusService']


@pytest.fixture(scope="function")
def channel(request):
    return grpc_testing.channel([descriptor], grpc_testing.strict_real_time())


@pytest.fixture(scope="function")
def client_thread(request):
    client_execution_thread_pool = logging_pool.pool(2)

    def teardown():
        client_execution_thread_pool.shutdown(wait=True)

    return client_execution_thread_pool
