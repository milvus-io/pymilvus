import unittest

from pymilvus.client import ts_utils
from pymilvus.client.cache import CollectionTsCache, GlobalCache


class TestCollectionTsCache(unittest.TestCase):
    def setUp(self):
        GlobalCache._reset_for_testing()

    def test_cache_isolation(self):
        cache = CollectionTsCache()

        # Test basic set/get
        cache.set("ep1", "db1", "coll1", 100)
        self.assertEqual(cache.get("ep1", "db1", "coll1"), 100)

        # Test isolation between collections
        cache.set("ep1", "db1", "coll2", 200)
        self.assertEqual(cache.get("ep1", "db1", "coll1"), 100)
        self.assertEqual(cache.get("ep1", "db1", "coll2"), 200)

        # Test isolation between dbs
        cache.set("ep1", "db2", "coll1", 300)
        self.assertEqual(cache.get("ep1", "db1", "coll1"), 100)
        self.assertEqual(cache.get("ep1", "db2", "coll1"), 300)

        # Test isolation between endpoints
        cache.set("ep2", "db1", "coll1", 400)
        self.assertEqual(cache.get("ep1", "db1", "coll1"), 100)
        self.assertEqual(cache.get("ep2", "db1", "coll1"), 400)

    def test_cache_monotonicity(self):
        cache = CollectionTsCache()

        cache.set("ep1", "db1", "coll1", 100)

        # Should update
        cache.set("ep1", "db1", "coll1", 110)
        self.assertEqual(cache.get("ep1", "db1", "coll1"), 110)

        # Should NOT update (smaller ts)
        cache.set("ep1", "db1", "coll1", 105)
        self.assertEqual(cache.get("ep1", "db1", "coll1"), 110)

    def test_ts_utils_integration(self):
        # Verify ts_utils uses GlobalCache.collection_ts

        ts_utils.update_collection_ts("coll1", 1000, "ep1", "db1")
        self.assertEqual(GlobalCache.collection_ts.get("ep1", "db1", "coll1"), 1000)

        # Test default args (empty strings)
        ts_utils.update_collection_ts("coll1", 500)
        self.assertEqual(GlobalCache.collection_ts.get("", "", "coll1"), 500)

        # Test retrieval
        self.assertEqual(ts_utils.get_collection_ts("coll1", "ep1", "db1"), 1000)
        self.assertEqual(ts_utils.get_collection_ts("coll1"), 500)

    def test_construct_guarantee_ts(self):
        # Mock get_eventually_ts to return a fixed value if needed,
        # but ts_utils.get_eventually_ts returns constant 1.

        ts_utils.update_collection_ts("coll1", 2000, "ep1", "db1")

        kwargs = {}
        # Default behavior: use session/customized consistency -> get cached ts
        ts_utils.construct_guarantee_ts("coll1", kwargs, "ep1", "db1")
        self.assertEqual(kwargs.get("guarantee_timestamp"), 2000)

        # Different endpoint
        kwargs = {}
        ts_utils.construct_guarantee_ts("coll1", kwargs, "ep2", "db1")
        # Should not find 2000, should default to EVENTUALLY_TS (1)
        # Note: construct_guarantee_ts logic: get_collection_ts(...) or get_eventually_ts()
        # If cache miss, get_collection_ts returns 0 (from my implementation of CollectionTsCache.get using .get(key, 0) and returning SUPER.get(key)??
        # Wait, CacheRegion.get returns None if missing?
        # My implementation: return super().get(key) or 0.
        # So it returns 0.
        # 0 or 1 -> 1.
        self.assertEqual(kwargs.get("guarantee_timestamp"), 1)


if __name__ == "__main__":
    unittest.main()
