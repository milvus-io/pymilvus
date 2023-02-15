import sys
import pytest
import grpc_testing

# https://github.com/grpc/grpc/blob/5918f98ecbf5ace77f30fa97f7fc3e8bdac08e04/src/python/grpcio_tests/tests/testing/_client_test.py
from grpc.framework.foundation import logging_pool

from pymilvus.grpc_gen import milvus_pb2

# test import
from pymilvus import *
import pymilvus.v2
import pymilvus.aio

from pymilvus.v2 import MilvusClient

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

descriptor = milvus_pb2.DESCRIPTOR.services_by_name['MilvusService']


@pytest.fixture(scope="function")
def channel(request):
    channel = grpc_testing.channel([descriptor], grpc_testing.strict_real_time())

    return channel


@pytest.fixture(scope="module")
def client_thread(request):
    client_execution_thread_pool = logging_pool.pool(2)

    def teardown():
        client_execution_thread_pool.shutdown(wait=True)
    return client_execution_thread_pool

@pytest.fixture(scope="module")
def client(request):
    channel = grpc_testing.channel([descriptor], grpc_testing.strict_real_time())

    client = MilvusClient("fake", "fake", _channel=channel)
    return client

@pytest.fixture(scope="function")
def rpc_future_GetVersion(client_thread):
    channel = grpc_testing.channel([descriptor], grpc_testing.strict_real_time())
    client = MilvusClient("fake", "fake", _channel=channel)

    get_server_version_future = client_thread.submit(client.get_server_version)
    (invocation_metadata, request, rpc) = (
        channel.take_unary_unary(descriptor.methods_by_name['GetVersion']))

    rpc.send_initial_metadata(())
    return rpc, get_server_version_future

