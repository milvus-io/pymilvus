import threading

from pymilvus.client.cache import CacheRegion, GlobalCache, SchemaCache


class TestSchemaCache:
    """Unit tests for SchemaCache and GlobalCache."""

    def setup_method(self):
        """Reset singleton and clear cache before each test."""
        GlobalCache._reset_for_testing()
        self.cache = GlobalCache.schema

    def teardown_method(self):
        """Clean up after each test."""
        GlobalCache._reset_for_testing()

    def test_global_cache_returns_same_schema_instance(self):
        """Verify that GlobalCache.schema is always the same instance."""
        cache1 = GlobalCache.schema
        cache2 = GlobalCache.schema
        assert cache1 is cache2

    def test_get_returns_none_for_missing_key(self):
        """get() should return None for non-existent keys."""
        assert self.cache.get("localhost:19530", "default", "non_existent") is None

    def test_set_and_get(self):
        """Basic set/get functionality with tuple keys."""
        schema = {"fields": [{"name": "id", "type": "INT64"}]}
        self.cache.set("localhost:19530", "mydb", "collection1", schema)
        result = self.cache.get("localhost:19530", "mydb", "collection1")
        assert result == schema

    def test_empty_db_defaults_to_default(self):
        """Empty db_name should be treated as 'default'."""
        schema = {"fields": []}
        self.cache.set("localhost:19530", "", "collection1", schema)

        # Both empty string and explicit "default" should retrieve the same value
        result1 = self.cache.get("localhost:19530", "", "collection1")
        result2 = self.cache.get("localhost:19530", "default", "collection1")
        assert result1 == schema
        assert result2 == schema

    def test_lru_eviction(self):
        """Verify LRU eviction when capacity is exceeded."""

        # Create a small capacity cache for testing
        small_cache = SchemaCache(capacity=3)

        small_cache.set("host", "db", "c1", {"name": "c1"})
        small_cache.set("host", "db", "c2", {"name": "c2"})
        small_cache.set("host", "db", "c3", {"name": "c3"})

        assert len(small_cache) == 3

        # Access c1 to make it recently used (order: c2, c3, c1)
        small_cache.get("host", "db", "c1")

        # Add c4, should evict c2 (least recently used)
        small_cache.set("host", "db", "c4", {"name": "c4"})

        assert len(small_cache) == 3
        assert small_cache.get("host", "db", "c2") is None  # Evicted
        assert small_cache.get("host", "db", "c1") == {"name": "c1"}  # Still present
        assert small_cache.get("host", "db", "c3") == {"name": "c3"}  # Still present
        assert small_cache.get("host", "db", "c4") == {"name": "c4"}  # Newly added

    def test_set_updates_existing_key(self):
        """set() on existing key should update value and move to end."""
        small_cache = SchemaCache(capacity=3)

        small_cache.set("host", "db", "c1", {"v": 1})
        small_cache.set("host", "db", "c2", {"v": 2})
        small_cache.set("host", "db", "c3", {"v": 3})

        # Update c1 (moves it to end)
        small_cache.set("host", "db", "c1", {"v": 1, "updated": True})

        assert small_cache.get("host", "db", "c1") == {"v": 1, "updated": True}

        # Add c4, should evict c2 (not c1 since c1 was recently updated)
        small_cache.set("host", "db", "c4", {"v": 4})

        assert small_cache.get("host", "db", "c2") is None  # Evicted
        assert small_cache.get("host", "db", "c1") == {"v": 1, "updated": True}  # Still present

    def test_invalidate_removes_key(self):
        """invalidate() should remove specific key."""
        self.cache.set("host", "db", "c1", {"v": 1})
        self.cache.set("host", "db", "c2", {"v": 2})

        self.cache.invalidate("host", "db", "c1")

        assert self.cache.get("host", "db", "c1") is None
        assert self.cache.get("host", "db", "c2") == {"v": 2}

    def test_invalidate_nonexistent_key_is_safe(self):
        """invalidate() on non-existent key should not raise."""
        self.cache.invalidate("host", "db", "non_existent")  # Should not raise

    def test_invalidate_db(self):
        """invalidate_db() should remove all entries for a database."""
        self.cache.set("host", "db1", "c1", {"db": "db1", "c": "c1"})
        self.cache.set("host", "db1", "c2", {"db": "db1", "c": "c2"})
        self.cache.set("host", "db2", "c1", {"db": "db2", "c": "c1"})
        self.cache.set("host2", "db1", "c1", {"host": "host2", "db": "db1"})

        assert len(self.cache) == 4

        self.cache.invalidate_db("host", "db1")

        assert len(self.cache) == 2
        assert self.cache.get("host", "db1", "c1") is None
        assert self.cache.get("host", "db1", "c2") is None
        assert self.cache.get("host", "db2", "c1") == {"db": "db2", "c": "c1"}
        assert self.cache.get("host2", "db1", "c1") == {"host": "host2", "db": "db1"}

    def test_clear_removes_all_entries(self):
        """clear() should remove all entries."""
        self.cache.set("host", "db", "c1", {"v": 1})
        self.cache.set("host", "db", "c2", {"v": 2})
        self.cache.set("host", "db", "c3", {"v": 3})

        self.cache.clear()

        assert len(self.cache) == 0
        assert self.cache.get("host", "db", "c1") is None

    def test_thread_safety(self):
        """Verify thread-safe concurrent access."""
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    collection = f"thread_{thread_id}_coll_{i}"
                    schema = {"thread": thread_id, "iter": i}
                    self.cache.set("host", "db", collection, schema)
                    result = self.cache.get("host", "db", collection)
                    # Value might be evicted due to capacity, but should not error
                    if result is not None:
                        assert result["thread"] == thread_id
                        assert result["iter"] == i
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(self.cache) > 0  # Cache should have some entries

    def test_make_key_static_method(self):
        """Verify tuple keys are created correctly."""
        key = SchemaCache._make_key("localhost:19530", "mydb", "collection1")
        assert key == ("localhost:19530", "mydb", "collection1")

        # Empty db should default to "default"
        key = SchemaCache._make_key("localhost:19530", "", "collection1")
        assert key == ("localhost:19530", "default", "collection1")

    def test_len(self):
        """__len__ should return number of cached entries."""
        assert len(self.cache) == 0

        self.cache.set("host", "db", "c1", {"v": 1})
        assert len(self.cache) == 1

        self.cache.set("host", "db", "c2", {"v": 2})
        assert len(self.cache) == 2

        self.cache.invalidate("host", "db", "c1")
        assert len(self.cache) == 1


class TestCacheRegion:
    """Unit tests for the base CacheRegion class."""

    def test_basic_operations(self):
        """Test basic get/set/invalidate operations."""
        cache = CacheRegion(capacity=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        """Test clear removes all entries."""
        cache = CacheRegion(capacity=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
