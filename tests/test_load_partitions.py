from tests.utils import *
from tests.constants import *

uniq_id = "load_partitions"

class TestLoadPartitions:
    """
    ******************************************************************
      The following cases are used to test `load_partitions` function
    ******************************************************************
    """
    def test_load_partitions(self, connect, collection):
        '''
        target: test load collection and wait for loading collection
        method: insert then flush, when flushed, try load collection
        expected: no errors
        '''
        partition_name = "lvn9pq34u8rasjk"
        connect.create_partition(collection, partition_name + "1")
        ids = connect.insert(collection, default_entities, partition_name=partition_name + "1")

        connect.create_partition(collection, partition_name + "2")
        ids = connect.insert(collection, default_entity, partition_name=partition_name + "2")

        connect.flush([collection])
        connect.load_partitions(collection, [partition_name + "2"])
