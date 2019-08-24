import pytest
import sys

sys.path.append(".")
from milvus import Milvus

from milvus.client.Exceptions import *


def test_host_port_connect(args):
    milvus = Milvus()

    _args = {'host': args['ip'], 'port': args['port']}

    status = milvus.connect(**_args)
    assert status.OK()

    milvus.disconnect()


@pytest.mark.skip("host support only")
def test_default_connect():
    milvus = Milvus()

    status = milvus.connect()
    assert status.OK()

    milvus.disconnect()


def test_uri_connect(args):
    uri = 'tcp://{}:{}'.format(args['ip'], args['port'])
    milvus = Milvus()
    status = milvus.connect(uri=uri)
    assert status.OK()

    milvus.disconnect()


@pytest.mark.skip("host support only")
def test_uri_not_host_connect():
    uri = 'tcp://:19530'
    milvus = Milvus()

    # try:
    status = milvus.connect(uri=uri)
    # except NotConnectError as e:
    #     return
    assert status.OK(), ""


@pytest.mark.skip("may failed in server")
def test_uri_not_port_connect():
    uri = 'tcp://localhost:'

    milvus = Milvus()

    # try:
    status = milvus.connect(uri=uri)
    # except NotConnectError as e:
    #     return
    assert status.OK(), ""


@pytest.mark.skip("host support only")
def test_port_connect():
    port = "19530"

    milvus = Milvus()
    try:
        status = milvus.connect(port=port)
    except NotConnectError as e:
        assert False
    assert status.OK(), "Error excepted, but passed"
