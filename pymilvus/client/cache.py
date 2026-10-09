import logging
import threading
from typing import Any, ClassVar, MutableMapping, Optional, Tuple

from cachetools import LRUCache

logger = logging.getLogger(__name__)


class CacheRegion:
    """
    Thread-safe cache base class, LRU-bounded by default.

    Subclasses should define specific key types and value types.
    """

    DEFAULT_CAPACITY = 4096

    def __init__(self, capacity: Optional[int] = DEFAULT_CAPACITY):
        """Create a cache region.

        Args:
            capacity: maximum number of entries. ``None`` makes the region
                unbounded, so entries are only dropped by ``invalidate`` or
                ``clear``. Use it for regions where an evicted entry changes
                behaviour rather than just costing a round trip.
        """
        self._cache: MutableMapping[Any, Any] = (
            {} if capacity is None else LRUCache(maxsize=capacity)
        )
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache. Returns None if not found."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: Any, value: Any) -> None:
        """Set value in cache. Bounded regions evict the LRU entry when over capacity."""
        with self._lock:
            self._cache[key] = value

    def invalidate(self, key: Any) -> None:
        """Remove a specific key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: Tuple[Any, ...]) -> None:
        """Remove every entry whose tuple key starts with ``prefix``.

        Used to drop a whole database or endpoint at once, so an unbounded
        region still shrinks when the objects it describes are gone.
        """
        size = len(prefix)
        with self._lock:
            stale = [k for k in self._cache if isinstance(k, tuple) and k[:size] == prefix]
            for key in stale:
                self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            return len(self._cache)


class SchemaCache(CacheRegion):
    """
    Schema-specific cache with tuple-based keys.

    Key: (endpoint, db_name, collection_name)
    Value: schema dict
    """

    def get(self, endpoint: str, db_name: str, collection_name: str) -> Optional[dict]:
        """Get schema from cache."""
        key = self._make_key(endpoint, db_name, collection_name)
        return super().get(key)

    def set(self, endpoint: str, db_name: str, collection_name: str, schema: dict) -> None:
        """Set schema in cache."""
        key = self._make_key(endpoint, db_name, collection_name)
        super().set(key, schema)

    def invalidate(self, endpoint: str, db_name: str, collection_name: str) -> None:
        """Invalidate schema for a specific collection."""
        key = self._make_key(endpoint, db_name, collection_name)
        super().invalidate(key)

    def invalidate_db(self, endpoint: str, db_name: str) -> None:
        """Invalidate all schemas for a database."""
        self.invalidate_prefix((endpoint, db_name or "default"))

    def invalidate_endpoint(self, endpoint: str) -> None:
        """Invalidate all schemas cached for an endpoint."""
        self.invalidate_prefix((endpoint,))

    @staticmethod
    def _make_key(endpoint: str, db_name: str, collection_name: str) -> Tuple[str, str, str]:
        """Create tuple key from components."""
        db = db_name if db_name else "default"
        return (endpoint, db, collection_name)


class CollectionTsCache(CacheRegion):
    """
    Collection timestamp cache with tuple-based keys.

    Key: (endpoint, db_name, collection_name)
    Value: timestamp (int)

    Unbounded on purpose. A missing entry makes ``get`` return 0, and
    ``construct_guarantee_ts`` then falls back to ``EVENTUALLY_TS``, so an
    evicted collection silently downgrades a Session-consistency read to the
    weakest guarantee with nothing logged. An entry is one small int, and
    stale ones are dropped by ``invalidate``.
    """

    def __init__(self, capacity: Optional[int] = None):
        super().__init__(capacity)

    def get(self, endpoint: str, db_name: str, collection_name: str) -> int:
        """Get timestamp from cache."""
        key = self._make_key(endpoint, db_name, collection_name)
        return super().get(key) or 0

    def set(self, endpoint: str, db_name: str, collection_name: str, ts: int) -> None:
        """Set timestamp in cache."""
        key = self._make_key(endpoint, db_name, collection_name)
        with self._lock:
            # Only update if new timestamp is greater
            old_ts = self._cache.get(key, 0)
            if ts > old_ts:
                self._cache[key] = ts

    def invalidate(self, endpoint: str, db_name: str, collection_name: str) -> None:
        """Invalidate timestamp for a specific collection."""
        key = self._make_key(endpoint, db_name, collection_name)
        super().invalidate(key)

    def invalidate_db(self, endpoint: str, db_name: str) -> None:
        """Invalidate all timestamps for a database."""
        self.invalidate_prefix((endpoint, db_name or "default"))

    def invalidate_endpoint(self, endpoint: str) -> None:
        """Invalidate all timestamps cached for an endpoint."""
        self.invalidate_prefix((endpoint,))

    @staticmethod
    def _make_key(endpoint: str, db_name: str, collection_name: str) -> Tuple[str, str, str]:
        """Create tuple key from components."""
        db = db_name if db_name else "default"
        return (endpoint, db, collection_name)


class GlobalCache:
    """
    Global access point for all cache instances.

    Usage:
        GlobalCache.schema.get(endpoint, db_name, collection_name)
        GlobalCache.schema.set(endpoint, db_name, collection_name, schema)
        GlobalCache.collection_ts.get(endpoint, db_name, collection_name)
        GlobalCache.collection_ts.set(endpoint, db_name, collection_name, ts)
    """

    schema: ClassVar[SchemaCache] = SchemaCache()
    collection_ts: ClassVar[CollectionTsCache] = CollectionTsCache()

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Reset cache for testing. Creates new instances."""
        cls.schema = SchemaCache()
        cls.collection_ts = CollectionTsCache()
