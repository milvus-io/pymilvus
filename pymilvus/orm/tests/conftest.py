import sys
import pytest

from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from mock_milvus import MockMilvus
#  sys.modules['pymilvus'] = __import__('mock_milvus')
from pymilvus import connections


#  @pytest.fixture(scope='session', autouse=True)
#  def create_collection():
#      connections.connect()
#      yield
#      connections.remove_connection(alias='default')
