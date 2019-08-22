import sys

sys.path.append(".")
from milvus import Milvus

from milvus.client.Exceptions import *


def test_host_port_connect():
    milvus = Milvus()

    args = {'host': 'localhost', 'port': 19530}

    status = milvus.connect(**args)
    assert status.OK()

    milvus.disconnect()


def test__connect():
    milvus = Milvus()

    status = milvus.connect()
    assert status.OK()

    milvus.disconnect()


def test_uri_connect():
    uri = 'tcp://127.0.0.1:19530'
    milvus = Milvus()
    status = milvus.connect(uri=uri)
    assert status.OK()

    milvus.disconnect()


def test_uri_not_host_connect():
    uri = 'tcp://:19530'
    # import pdb;pdb.set_trace()

    milvus = Milvus()

    # try:
    status = milvus.connect(uri=uri)
    # except NotConnectError as e:
    #     return
    assert status.OK(), ""


def test_uri_not_port_connect():
    uri = 'tcp://localhost:'

    milvus = Milvus()

    # try:
    status = milvus.connect(uri=uri)
    # except NotConnectError as e:
    #     return
    assert status.OK(), ""


def test_port_connect():
    port = "19530"

    milvus = Milvus()
    try:
        status = milvus.connect(port=port)
    except NotConnectError as e:
        assert False
    assert status.OK(), "Error excepted, but passed"
