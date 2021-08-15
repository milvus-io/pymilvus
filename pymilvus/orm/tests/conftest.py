import sys
import pytest

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

sys.modules['pymilvus'] = __import__('mock_milvus')
import pymilvus_orm.connections as connections


@pytest.fixture(scope='session', autouse=True)
def create_collection():
    connections.connect()
    yield
    connections.remove_connection(alias='default')
