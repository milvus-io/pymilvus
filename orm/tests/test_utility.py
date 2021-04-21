import unittest
from utils import *
from pymilvus_orm.utility import *


class TestCase(unittest.TestCase):
    def test_loading_progress(self):
        loading_progress(gen_collection_name(), gen_partition_name())

    def test_wait_for_loading_complete(self):
        wait_for_loading_complete(gen_collection_name(), gen_partition_name())

    def test_index_building_progress(self):
        index_building_progress(gen_collection_name(), gen_index_name())

    def test_wait_for_index_building_complete(self):
        wait_for_index_building_complete(gen_collection_name(), gen_index_name())

    def test_has_collection(self):
        has_collection(gen_collection_name())

    def test_has_partition(self):
        has_partition(gen_collection_name(), gen_partition_name())


if __name__ == '__main__':
    unittest.main()
