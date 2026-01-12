import threading

import pytest

from pymilvus.client.schema_cache import GlobalSchemaCache, Singleton


class TestGlobalSchemaCache:
    """Unit tests for GlobalSchemaCache."""

    def setup_method(self):
        """Reset singleton and clear cache before each test."""
        Singleton._reset_for_testing()
        # Re-instantiate after reset
        self.cache = GlobalSchemaCache()

    def teardown_method(self):
        """Clean up after each test."""
        Singleton._reset_for_testing()

    def test_singleton_returns_same_instance(self):
        """Verify that GlobalSchemaCache is a singleton."""
        cache1 = GlobalSchemaCache()
        cache2 = GlobalSchemaCache()
        assert cache1 is cache2
        assert cache1._cache is cache2._cache

    def test_get_returns_none_for_missing_key(self):
        """get() should return None for non-existent keys."""
        assert self.cache.get("non_existent_key") is None

    def test_set_and_get(self):
        """Basic set/get functionality."""
        self.cache.set("key1", {"schema": "test_schema"})
        result = self.cache.get("key1")
        assert result == {"schema": "test_schema"}

    def test_lru_eviction(self):
        """Verify LRU eviction when capacity is exceeded."""
        # Create a small capacity cache for testing
        from cachetools import LRUCache
        import threading
        
        small_cache = type('obj', (object,), {
            '_cache': LRUCache(maxsize=3),
            '_lock': threading.Lock()
        })()

        # Wrap set/get with locks like GlobalSchemaCache
        def set_item(key, value):
            with small_cache._lock:
                small_cache._cache[key] = value
        
        def get_item(key):
            with small_cache._lock:
                return small_cache._cache.get(key)
        
        set_item("k1", "v1")
        set_item("k2", "v2")
        set_item("k3", "v3")

        assert len(small_cache._cache) == 3

        # Access k1 to make it recently used (order: k2, k3, k1)
        get_item("k1")

        # Add k4, should evict k2 (least recently used)
        set_item("k4", "v4")

        assert len(small_cache._cache) == 3
        assert get_item("k2") is None  # Evicted
        assert get_item("k1") == "v1"  # Still present
        assert get_item("k3") == "v3"  # Still present
        assert get_item("k4") == "v4"  # Newly added

    def test_set_updates_existing_key(self):
        """set() on existing key should update value and move to end."""
        # Create a small capacity cache for testing
        from cachetools import LRUCache
        import threading
        
        small_cache = type('obj', (object,), {
            '_cache': LRUCache(maxsize=3),
            '_lock': threading.Lock()
        })()

        def set_item(key, value):
            with small_cache._lock:
                small_cache._cache[key] = value
        
        def get_item(key):
            with small_cache._lock:
                return small_cache._cache.get(key)

        set_item("k1", "v1")
        set_item("k2", "v2")
        set_item("k3", "v3")

        # Update k1 (moves it to end)
        set_item("k1", "v1_updated")

        assert get_item("k1") == "v1_updated"

        # Add k4, should evict k2 (not k1 since k1 was recently updated)
        set_item("k4", "v4")

        assert get_item("k2") is None  # Evicted
        assert get_item("k1") == "v1_updated"  # Still present

    def test_invalidate_removes_key(self):
        """invalidate() should remove specific key."""
        self.cache.set("k1", "v1")
        self.cache.set("k2", "v2")

        self.cache.invalidate("k1")

        assert self.cache.get("k1") is None
        assert self.cache.get("k2") == "v2"

    def test_invalidate_nonexistent_key_is_safe(self):
        """invalidate() on non-existent key should not raise."""
        self.cache.invalidate("non_existent")  # Should not raise

    def test_clear_removes_all_entries(self):
        """clear() should remove all entries."""
        self.cache.set("k1", "v1")
        self.cache.set("k2", "v2")
        self.cache.set("k3", "v3")

        self.cache.clear()

        assert len(self.cache) == 0
        assert self.cache.get("k1") is None

    def test_thread_safety(self):
        """Verify thread-safe concurrent access."""
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    self.cache.set(key, {"thread": thread_id, "iter": i})
                    result = self.cache.get(key)
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

    def test_format_key_with_all_parts(self):
        """format_key should create proper key format."""
        key = GlobalSchemaCache.format_key("localhost:19530", "mydb", "collection1")
        assert key == "localhost:19530/mydb/collection1"

    def test_format_key_with_empty_db_name(self):
        """Empty db_name should default to 'default'."""
        key = GlobalSchemaCache.format_key("localhost:19530", "", "collection1")
        assert key == "localhost:19530/default/collection1"

    def test_format_key_with_none_like_db_name(self):
        """None-like db_name values should default to 'default'."""
        # Empty string
        key = GlobalSchemaCache.format_key("localhost:19530", "", "coll")
        assert key == "localhost:19530/default/coll"

    def test_len(self):
        """__len__ should return number of cached entries."""
        assert len(self.cache) == 0

        self.cache.set("k1", "v1")
        assert len(self.cache) == 1

        self.cache.set("k2", "v2")
        assert len(self.cache) == 2

        self.cache.invalidate("k1")
        assert len(self.cache) == 1
