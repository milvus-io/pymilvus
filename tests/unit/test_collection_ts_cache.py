import unittest

from pymilvus.client.cache import CollectionTsCache


class TestCollectionTsCache(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
