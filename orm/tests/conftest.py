import sys
sys.modules['milvus'] = __import__('mock_milvus')

import pytest
import pymilvus_orm.connections as connections


@pytest.fixture(scope='session', autouse=True)
def create_collection():
    connections.create_connection()
    yield
    connections.remove_connection(alias='default')
