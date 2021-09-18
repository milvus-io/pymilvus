import pytest
from utils import *
from pymilvus import utility


@pytest.mark.xfail
class TestCollectionSchema:
    def test_loading_progress(self):
        utility.loading_progress(gen_collection_name(), [gen_partition_name()])

    def test_wait_for_loading_complete(self):
        utility.wait_for_loading_complete(gen_collection_name(), [gen_partition_name()])

    def test_index_building_progress(self):
        utility.index_building_progress(gen_collection_name(), gen_index_name())

    def test_wait_for_index_building_complete(self):
        utility.wait_for_index_building_complete(gen_collection_name(), gen_index_name())

    def test_has_collection(self):
        assert utility.has_collection(gen_collection_name()) is False

    def test_has_partition(self):
        with pytest.raises(BaseException):
            utility.has_partition(gen_collection_name(), gen_partition_name())

    def test_drop_collection(self):
        with pytest.raises(BaseException):
            utility.drop_collection(gen_collection_name())
