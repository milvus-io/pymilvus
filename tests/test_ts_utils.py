import threading

from pymilvus.client import ts_utils


from pymilvus.client.cache import GlobalCache

class TestTsUtils:
    def setup_method(self):
        GlobalCache._reset_for_testing()

    def test_update_and_get(self):
        # Verify update_collection_ts and get_collection_ts
        collection_name = "coll1"
        endpoint = "localhost:19530"
        db_name = "default"

        # Initial state should be 0
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 0

        # Update and verify
        ts_utils.update_collection_ts(collection_name, 100, endpoint, db_name)
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 100

        # Update with smaller timestamp should be ignored
        ts_utils.update_collection_ts(collection_name, 50, endpoint, db_name)
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 100

        # Update with larger timestamp should succeed
        ts_utils.update_collection_ts(collection_name, 200, endpoint, db_name)
        assert ts_utils.get_collection_ts(collection_name, endpoint, db_name) == 200

        # Test with empty endpoint/db (default behavior)
        ts_utils.update_collection_ts(collection_name, 300)
        assert ts_utils.get_collection_ts(collection_name) == 300


    def test_get_current_bounded_ts(self):
        ts = ts_utils.get_bounded_ts()
        assert ts == ts_utils.BOUNDED_TS

    def test_get_eventually_ts(self):
        ts = ts_utils.get_eventually_ts()
        assert ts == ts_utils.EVENTUALLY_TS
