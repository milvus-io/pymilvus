import logging
import threading
from typing import Any, ClassVar, Optional

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


class GlobalSchemaCache(metaclass=Singleton):
    """
    A thread-safe, singleton LRU cache for collection schemas.

    Cache keys are formatted as "{endpoint}/{db_name}/{collection_name}" to ensure
    isolation between different Milvus clusters and databases.
    """

    DEFAULT_CAPACITY = 4096

    def __init__(self):
        self._cache: LRUCache = LRUCache(maxsize=self.DEFAULT_CAPACITY)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache. Returns None if not found."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache. Evicts LRU entry if over capacity."""
        with self._lock:
            self._cache[key] = value

    def invalidate(self, key: str) -> None:
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

    @staticmethod
    def format_key(endpoint: str, db_name: str, collection_name: str) -> str:
        """
        Generate a unique cache key.

        Args:
            endpoint: Server address (e.g., "localhost:19530")
            db_name: Database name (empty string defaults to "default")
            collection_name: Collection name

        Returns:
            Cache key in format "{endpoint}/{db_name}/{collection_name}"
        """
        db = db_name if db_name else "default"
        return f"{endpoint}/{db}/{collection_name}"
