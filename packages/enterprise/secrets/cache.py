"""Caching layer for secret management.

This module provides caching utilities for secret values,
integrating with the common.cache module.

Example:
    >>> from packages.enterprise.secrets import SecretCache
    >>>
    >>> cache = SecretCache(ttl_seconds=300.0, max_size=1000)
    >>> cache.set("db/password", secret_value)
    >>> cached = cache.get("db/password")
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import SecretValue


@dataclass
class CacheEntry:
    """Entry in the secret cache.

    Attributes:
        value: The cached secret value.
        cached_at: Timestamp when the entry was cached.
        expires_at: Timestamp when the entry expires.
        access_count: Number of times the entry was accessed.
        last_accessed: Timestamp of last access.
    """

    value: SecretValue
    cached_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """Statistics for the secret cache.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of evictions.
        size: Current cache size.
        max_size: Maximum cache size.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate.

        Returns:
            Hit rate as a ratio (0.0 to 1.0).
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class SecretCache:
    """LRU cache for secret values with TTL support.

    Thread-safe cache implementation with automatic expiration
    and LRU eviction.

    Attributes:
        ttl_seconds: Time-to-live for cache entries.
        max_size: Maximum number of entries.

    Example:
        >>> cache = SecretCache(ttl_seconds=300.0, max_size=1000)
        >>> cache.set("key", secret_value)
        >>> value = cache.get("key")
        >>> cache.invalidate("key")
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_size: int = 1000,
    ) -> None:
        """Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for entries.
            max_size: Maximum cache size.
        """
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)

    @property
    def ttl_seconds(self) -> float:
        """Get the TTL in seconds."""
        return self._ttl_seconds

    @property
    def max_size(self) -> int:
        """Get the maximum size."""
        return self._max_size

    def get(self, key: str) -> SecretValue | None:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found or expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._stats.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: SecretValue,
        ttl_seconds: float | None = None,
    ) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl_seconds: Optional override for TTL.
        """
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()

            ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
            now = time.time()
            self._cache[key] = CacheEntry(
                value=value,
                cached_at=now,
                expires_at=now + ttl,
            )
            self._stats.size = len(self._cache)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if the entry was removed.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all entries matching a prefix.

        Args:
            prefix: Key prefix.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
            self._stats.size = len(self._cache)
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]
        self._stats.evictions += 1

    def _evict_expired(self) -> int:
        """Evict all expired entries.

        Returns:
            Number of entries evicted.
        """
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._cache.items() if now > v.expires_at]
            for key in expired:
                del self._cache[key]
            self._stats.size = len(self._cache)
            return len(expired)

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Cache statistics.
        """
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self._stats.max_size,
            )

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.evictions = 0

    def contains(self, key: str) -> bool:
        """Check if a key is in the cache (and not expired).

        Args:
            key: Cache key.

        Returns:
            True if the key is cached and not expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return False
            return True

    def __len__(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return self.contains(key)


class TieredSecretCache:
    """Two-tier cache for secrets with hot and cold tiers.

    Hot tier: Frequently accessed secrets with short TTL.
    Cold tier: Less frequently accessed with longer TTL.

    Example:
        >>> cache = TieredSecretCache(
        ...     hot_ttl_seconds=60.0,
        ...     cold_ttl_seconds=300.0,
        ... )
    """

    def __init__(
        self,
        hot_ttl_seconds: float = 60.0,
        cold_ttl_seconds: float = 300.0,
        hot_max_size: int = 100,
        cold_max_size: int = 1000,
        promotion_threshold: int = 3,
    ) -> None:
        """Initialize the tiered cache.

        Args:
            hot_ttl_seconds: TTL for hot tier.
            cold_ttl_seconds: TTL for cold tier.
            hot_max_size: Max size of hot tier.
            cold_max_size: Max size of cold tier.
            promotion_threshold: Access count to promote to hot.
        """
        self._hot = SecretCache(ttl_seconds=hot_ttl_seconds, max_size=hot_max_size)
        self._cold = SecretCache(ttl_seconds=cold_ttl_seconds, max_size=cold_max_size)
        self._promotion_threshold = promotion_threshold
        self._access_counts: dict[str, int] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> SecretValue | None:
        """Get from cache, checking hot tier first.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found.
        """
        # Check hot tier first
        value = self._hot.get(key)
        if value is not None:
            return value

        # Check cold tier
        value = self._cold.get(key)
        if value is not None:
            # Track access for promotion
            with self._lock:
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                if self._access_counts[key] >= self._promotion_threshold:
                    # Promote to hot tier
                    self._hot.set(key, value)
                    self._access_counts[key] = 0
            return value

        return None

    def set(
        self,
        key: str,
        value: SecretValue,
        hot: bool = False,
    ) -> None:
        """Set in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            hot: Whether to put directly in hot tier.
        """
        if hot:
            self._hot.set(key, value)
        else:
            self._cold.set(key, value)

    def invalidate(self, key: str) -> bool:
        """Invalidate from both tiers.

        Args:
            key: Cache key.

        Returns:
            True if removed from either tier.
        """
        hot_removed = self._hot.invalidate(key)
        cold_removed = self._cold.invalidate(key)
        with self._lock:
            self._access_counts.pop(key, None)
        return hot_removed or cold_removed

    def clear(self) -> None:
        """Clear both tiers."""
        self._hot.clear()
        self._cold.clear()
        with self._lock:
            self._access_counts.clear()

    def get_stats(self) -> dict[str, CacheStats]:
        """Get statistics for both tiers.

        Returns:
            Dict with 'hot' and 'cold' stats.
        """
        return {
            "hot": self._hot.get_stats(),
            "cold": self._cold.get_stats(),
        }
