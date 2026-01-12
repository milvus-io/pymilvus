import logging
import threading
from typing import Any, ClassVar, Optional, Tuple

from cachetools import LRUCache

logger = logging.getLogger(__name__)


class Singleton(type):
    """Thread-safe singleton metaclass."""

    _instances: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def _reset_for_testing(cls):
        """Reset all singleton instances. Only for testing."""
        with cls._lock:
            cls._instances.clear()


class CacheRegion:
    """
    Thread-safe LRU cache base class.

    Subclasses should define specific key types and value types.
    """

    DEFAULT_CAPACITY = 4096

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self._cache: LRUCache = LRUCache(maxsize=capacity)
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache. Returns None if not found."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: Any, value: Any) -> None:
        """Set value in cache. Evicts LRU entry if over capacity."""
        with self._lock:
            self._cache[key] = value

    def invalidate(self, key: Any) -> None:
        """Remove a specific key from cache."""
        with self._lock:
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
        prefix = (endpoint, db_name or "default")
        with self._lock:
            keys_to_remove = [k for k in self._cache if k[:2] == prefix]
            for key in keys_to_remove:
                self._cache.pop(key, None)

    @staticmethod
    def _make_key(endpoint: str, db_name: str, collection_name: str) -> Tuple[str, str, str]:
        """Create tuple key from components."""
        db = db_name if db_name else "default"
        return (endpoint, db, collection_name)


class GlobalCache(metaclass=Singleton):
    """
    Global access point for all cache instances.

    Usage:
        GlobalCache.schema.get(endpoint, db_name, collection_name)
        GlobalCache.schema.set(endpoint, db_name, collection_name, schema)
    """

    schema: ClassVar[SchemaCache] = SchemaCache()

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Reset cache for testing. Creates new instances."""
        cls.schema = SchemaCache()
