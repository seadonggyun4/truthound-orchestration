"""Caching utilities for Truthound Integrations.

This module provides flexible, production-ready caching implementations
designed for reducing redundant computations and improving performance.

Features:
    - Multiple backends: In-memory, LRU, TTL-based caching
    - Configurable eviction policies
    - Key-based cache partitioning for multi-tenant scenarios
    - Async and sync function support
    - Hook system for monitoring and alerting
    - Cache statistics and metrics

Design Principles:
    1. Protocol-based: Easy to extend with custom cache backends
    2. Immutable Config: Thread-safe configuration using frozen dataclass
    3. Observable: Hook system for monitoring cache events
    4. Composable: Works well with retry, circuit breaker, and other patterns

Cache Eviction Policies:
    - LRU: Least Recently Used - evicts oldest accessed items first
    - LFU: Least Frequently Used - evicts least accessed items first
    - TTL: Time To Live - evicts items after a specified duration
    - FIFO: First In First Out - evicts oldest inserted items first

Example:
    >>> from common.cache import cached, CacheConfig
    >>> @cached(ttl_seconds=300.0)
    ... def fetch_user(user_id: str) -> dict:
    ...     return db.query(f"SELECT * FROM users WHERE id = {user_id}")

    >>> # With custom configuration
    >>> config = CacheConfig(
    ...     max_size=1000,
    ...     ttl_seconds=3600.0,
    ...     eviction_policy=EvictionPolicy.LRU,
    ... )
    >>> @cached(config=config)
    ... async def async_fetch(user_id: str) -> dict:
    ...     return await db.async_query(user_id)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import pickle
import threading
import time
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heappop, heappush
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# =============================================================================
# Exceptions
# =============================================================================


class CacheError(TruthoundIntegrationError):
    """Base exception for cache errors.

    Attributes:
        cache_name: Name of the cache that caused the error.
        key: The cache key involved in the error.
    """

    def __init__(
        self,
        message: str,
        *,
        cache_name: str | None = None,
        key: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize cache error.

        Args:
            message: Human-readable error description.
            cache_name: Name of the cache that caused the error.
            key: The cache key involved in the error.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if cache_name is not None:
            details["cache_name"] = cache_name
        if key is not None:
            details["key"] = key
        super().__init__(message, details=details, cause=cause)
        self.cache_name = cache_name
        self.key = key


class CacheFullError(CacheError):
    """Exception raised when cache is full and cannot accept new entries.

    Attributes:
        max_size: Maximum size of the cache.
        current_size: Current size of the cache.
    """

    def __init__(
        self,
        message: str = "Cache is full",
        *,
        cache_name: str | None = None,
        max_size: int | None = None,
        current_size: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize cache full error.

        Args:
            message: Human-readable error description.
            cache_name: Name of the cache.
            max_size: Maximum size of the cache.
            current_size: Current size of the cache.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if max_size is not None:
            details["max_size"] = max_size
        if current_size is not None:
            details["current_size"] = current_size
        super().__init__(message, cache_name=cache_name, details=details, cause=cause)
        self.max_size = max_size
        self.current_size = current_size


class CacheKeyError(CacheError):
    """Exception raised when a cache key is not found."""

    def __init__(
        self,
        key: str,
        *,
        cache_name: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize cache key error.

        Args:
            key: The cache key that was not found.
            cache_name: Name of the cache.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        message = f"Cache key '{key}' not found"
        super().__init__(message, cache_name=cache_name, key=key, details=details, cause=cause)


class CacheSerializationError(CacheError):
    """Exception raised when cache serialization/deserialization fails."""

    pass


# =============================================================================
# Enums
# =============================================================================


class EvictionPolicy(Enum):
    """Cache eviction policies.

    Attributes:
        LRU: Least Recently Used - evicts oldest accessed items first.
        LFU: Least Frequently Used - evicts least accessed items first.
        TTL: Time To Live - evicts items after a specified duration.
        FIFO: First In First Out - evicts oldest inserted items first.
        NONE: No eviction - cache grows unbounded (use with caution).
    """

    LRU = auto()
    LFU = auto()
    TTL = auto()
    FIFO = auto()
    NONE = auto()


class CacheAction(Enum):
    """Action to take on cache miss.

    Attributes:
        COMPUTE: Compute the value and cache it.
        RAISE: Raise a CacheKeyError.
        RETURN_NONE: Return None without caching.
    """

    COMPUTE = auto()
    RAISE = auto()
    RETURN_NONE = auto()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class CacheBackend(Protocol[K, V]):
    """Protocol for cache backend implementations.

    Implement this protocol to create custom cache storage backends.
    """

    @abstractmethod
    def get(self, key: K) -> V | None:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found.
        """
        ...

    @abstractmethod
    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Optional TTL override.
        """
        ...

    @abstractmethod
    def delete(self, key: K) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key.

        Returns:
            True if the key was deleted, False if not found.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache."""
        ...

    @abstractmethod
    def contains(self, key: K) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The cache key.

        Returns:
            True if the key exists.
        """
        ...

    @abstractmethod
    def size(self) -> int:
        """Get the number of items in the cache.

        Returns:
            Number of cached items.
        """
        ...


@runtime_checkable
class CacheHook(Protocol):
    """Protocol for cache event hooks.

    Implement this to receive notifications about cache events.
    """

    @abstractmethod
    def on_hit(
        self,
        key: str,
        value: Any,
        context: dict[str, Any],
    ) -> None:
        """Called when a cache hit occurs.

        Args:
            key: The cache key.
            value: The cached value.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_miss(
        self,
        key: str,
        context: dict[str, Any],
    ) -> None:
        """Called when a cache miss occurs.

        Args:
            key: The cache key.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None,
        context: dict[str, Any],
    ) -> None:
        """Called when a value is set in the cache.

        Args:
            key: The cache key.
            value: The value being cached.
            ttl_seconds: TTL for the cached value.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_evict(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
    ) -> None:
        """Called when a value is evicted from the cache.

        Args:
            key: The cache key.
            reason: Reason for eviction (e.g., "ttl_expired", "lru", "manual").
            context: Additional context information.
        """
        ...


@runtime_checkable
class KeyGenerator(Protocol):
    """Protocol for generating cache keys from function calls.

    Implement this to create custom key generation strategies.
    """

    @abstractmethod
    def generate_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Generate a cache key from function call.

        Args:
            func: The function being cached.
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.

        Returns:
            The generated cache key.
        """
        ...


# =============================================================================
# Key Generators
# =============================================================================


class DefaultKeyGenerator:
    """Generate cache key based on function name and arguments."""

    def __init__(
        self,
        prefix: str = "",
        include_module: bool = True,
        hash_args: bool = False,
    ) -> None:
        """Initialize default key generator.

        Args:
            prefix: Optional prefix for all keys.
            include_module: Include module name in key.
            hash_args: Hash arguments instead of string representation.
        """
        self.prefix = prefix
        self.include_module = include_module
        self.hash_args = hash_args

    def generate_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Generate cache key from function and arguments.

        Args:
            func: The function being cached.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Generated cache key.
        """
        parts = []

        if self.prefix:
            parts.append(self.prefix)

        if self.include_module:
            parts.append(f"{func.__module__}.{func.__name__}")
        else:
            parts.append(func.__name__)

        if self.hash_args:
            arg_hash = self._hash_args(args, kwargs)
            parts.append(arg_hash)
        else:
            if args:
                parts.append(str(args))
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                parts.append(str(sorted_kwargs))

        return ":".join(parts)

    def _hash_args(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Hash arguments for cache key.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Hashed string representation.
        """
        try:
            data = pickle.dumps((args, tuple(sorted(kwargs.items()))))
            return hashlib.sha256(data).hexdigest()[:16]
        except (TypeError, pickle.PicklingError):
            return hashlib.sha256(
                f"{args}:{sorted(kwargs.items())}".encode()
            ).hexdigest()[:16]


class ArgumentKeyGenerator:
    """Generate cache key from specific function arguments."""

    def __init__(
        self,
        arg_names: tuple[str, ...] | None = None,
        arg_indices: tuple[int, ...] | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize argument key generator.

        Args:
            arg_names: Names of keyword arguments to use in key.
            arg_indices: Indices of positional arguments to use in key.
            prefix: Optional prefix for all keys.
        """
        self.arg_names = arg_names or ()
        self.arg_indices = arg_indices or ()
        self.prefix = prefix

    def generate_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Generate cache key from specified arguments.

        Args:
            func: The function being cached.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Generated cache key.
        """
        key_parts = []

        if self.prefix:
            key_parts.append(self.prefix)

        key_parts.append(func.__name__)

        key_parts.extend(
            str(args[idx]) for idx in self.arg_indices if idx < len(args)
        )

        key_parts.extend(
            f"{name}={kwargs[name]}" for name in self.arg_names if name in kwargs
        )

        return ":".join(key_parts)


class CallableKeyGenerator:
    """Generate cache key using a callable."""

    def __init__(
        self,
        generator: Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], str],
    ) -> None:
        """Initialize callable key generator.

        Args:
            generator: Function to generate the key.
        """
        self._generator = generator

    def generate_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Generate cache key using the callable.

        Args:
            func: The function being cached.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Generated cache key.
        """
        return self._generator(func, args, kwargs)


# =============================================================================
# Hooks
# =============================================================================


class LoggingCacheHook:
    """Hook that logs cache events.

    Uses the Truthound logging system for structured logging.
    """

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.cache).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.cache")

    def on_hit(
        self,
        key: str,
        value: Any,
        context: dict[str, Any],
    ) -> None:
        """Log cache hit.

        Args:
            key: The cache key.
            value: The cached value.
            context: Additional context.
        """
        self._logger.debug(
            "Cache hit",
            key=key,
            **context,
        )

    def on_miss(
        self,
        key: str,
        context: dict[str, Any],
    ) -> None:
        """Log cache miss.

        Args:
            key: The cache key.
            context: Additional context.
        """
        self._logger.debug(
            "Cache miss",
            key=key,
            **context,
        )

    def on_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None,
        context: dict[str, Any],
    ) -> None:
        """Log cache set.

        Args:
            key: The cache key.
            value: The value being cached.
            ttl_seconds: TTL for the cached value.
            context: Additional context.
        """
        self._logger.debug(
            "Cache set",
            key=key,
            ttl_seconds=ttl_seconds,
            **context,
        )

    def on_evict(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
    ) -> None:
        """Log cache eviction.

        Args:
            key: The cache key.
            reason: Reason for eviction.
            context: Additional context.
        """
        self._logger.debug(
            "Cache eviction",
            key=key,
            reason=reason,
            **context,
        )


class MetricsCacheHook:
    """Hook that collects cache metrics.

    Useful for monitoring cache performance and hit rates.
    """

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._hits: int = 0
        self._misses: int = 0
        self._sets: int = 0
        self._evictions: int = 0
        self._key_stats: dict[str, dict[str, int]] = {}
        self._eviction_reasons: dict[str, int] = {}
        self._lock = threading.Lock()

    def on_hit(
        self,
        key: str,
        value: Any,
        context: dict[str, Any],
    ) -> None:
        """Record cache hit.

        Args:
            key: The cache key.
            value: The cached value.
            context: Additional context.
        """
        with self._lock:
            self._hits += 1
            if key not in self._key_stats:
                self._key_stats[key] = {"hits": 0, "misses": 0, "sets": 0}
            self._key_stats[key]["hits"] += 1

    def on_miss(
        self,
        key: str,
        context: dict[str, Any],
    ) -> None:
        """Record cache miss.

        Args:
            key: The cache key.
            context: Additional context.
        """
        with self._lock:
            self._misses += 1
            if key not in self._key_stats:
                self._key_stats[key] = {"hits": 0, "misses": 0, "sets": 0}
            self._key_stats[key]["misses"] += 1

    def on_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None,
        context: dict[str, Any],
    ) -> None:
        """Record cache set.

        Args:
            key: The cache key.
            value: The value being cached.
            ttl_seconds: TTL for the cached value.
            context: Additional context.
        """
        with self._lock:
            self._sets += 1
            if key not in self._key_stats:
                self._key_stats[key] = {"hits": 0, "misses": 0, "sets": 0}
            self._key_stats[key]["sets"] += 1

    def on_evict(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
    ) -> None:
        """Record cache eviction.

        Args:
            key: The cache key.
            reason: Reason for eviction.
            context: Additional context.
        """
        with self._lock:
            self._evictions += 1
            self._eviction_reasons[reason] = self._eviction_reasons.get(reason, 0) + 1

    @property
    def hits(self) -> int:
        """Get total hit count."""
        return self._hits

    @property
    def misses(self) -> int:
        """Get total miss count."""
        return self._misses

    @property
    def sets(self) -> int:
        """Get total set count."""
        return self._sets

    @property
    def evictions(self) -> int:
        """Get total eviction count."""
        return self._evictions

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def miss_rate(self) -> float:
        """Get cache miss rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._misses / total

    def get_key_stats(self, key: str) -> dict[str, int]:
        """Get stats for a specific key.

        Args:
            key: The cache key.

        Returns:
            Dictionary with hits, misses, sets counts.
        """
        with self._lock:
            return dict(self._key_stats.get(key, {"hits": 0, "misses": 0, "sets": 0}))

    def get_eviction_reasons(self) -> dict[str, int]:
        """Get eviction reasons breakdown.

        Returns:
            Dictionary mapping reasons to counts.
        """
        with self._lock:
            return dict(self._eviction_reasons)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._key_stats.clear()
            self._eviction_reasons.clear()


class CompositeCacheHook:
    """Combine multiple cache hooks."""

    def __init__(self, hooks: Sequence[CacheHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: CacheHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: CacheHook) -> None:
        """Remove a hook.

        Args:
            hook: Hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_hit(
        self,
        key: str,
        value: Any,
        context: dict[str, Any],
    ) -> None:
        """Call on_hit on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_hit(key, value, context)

    def on_miss(
        self,
        key: str,
        context: dict[str, Any],
    ) -> None:
        """Call on_miss on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_miss(key, context)

    def on_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_set on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_set(key, value, ttl_seconds, context)

    def on_evict(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
    ) -> None:
        """Call on_evict on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_evict(key, reason, context)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Configuration for cache behavior.

    Immutable configuration object for caching operations.
    Use builder methods to create modified copies.

    Attributes:
        max_size: Maximum number of items in the cache.
        ttl_seconds: Default time-to-live in seconds for cached items.
        eviction_policy: Policy for evicting items when cache is full.
        on_miss: Action to take on cache miss.
        name: Optional name for the cache (for logging/metrics).
        namespace: Optional namespace for key isolation.

    Example:
        >>> config = CacheConfig(
        ...     max_size=1000,
        ...     ttl_seconds=300.0,
        ...     eviction_policy=EvictionPolicy.LRU,
        ... )
        >>> smaller_config = config.with_max_size(100)
    """

    max_size: int = 1000
    ttl_seconds: float | None = None
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    on_miss: CacheAction = CacheAction.COMPUTE
    name: str | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_size < 1:
            raise ValueError("max_size must be at least 1")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive if specified")

    def with_max_size(self, max_size: int) -> CacheConfig:
        """Create config with new max_size.

        Args:
            max_size: New maximum size.

        Returns:
            New CacheConfig with updated value.
        """
        return CacheConfig(
            max_size=max_size,
            ttl_seconds=self.ttl_seconds,
            eviction_policy=self.eviction_policy,
            on_miss=self.on_miss,
            name=self.name,
            namespace=self.namespace,
        )

    def with_ttl(self, ttl_seconds: float | None) -> CacheConfig:
        """Create config with new TTL.

        Args:
            ttl_seconds: New TTL in seconds.

        Returns:
            New CacheConfig with updated value.
        """
        return CacheConfig(
            max_size=self.max_size,
            ttl_seconds=ttl_seconds,
            eviction_policy=self.eviction_policy,
            on_miss=self.on_miss,
            name=self.name,
            namespace=self.namespace,
        )

    def with_eviction_policy(self, policy: EvictionPolicy) -> CacheConfig:
        """Create config with new eviction policy.

        Args:
            policy: New eviction policy.

        Returns:
            New CacheConfig with updated value.
        """
        return CacheConfig(
            max_size=self.max_size,
            ttl_seconds=self.ttl_seconds,
            eviction_policy=policy,
            on_miss=self.on_miss,
            name=self.name,
            namespace=self.namespace,
        )

    def with_on_miss(self, on_miss: CacheAction) -> CacheConfig:
        """Create config with new on_miss action.

        Args:
            on_miss: New action on cache miss.

        Returns:
            New CacheConfig with updated value.
        """
        return CacheConfig(
            max_size=self.max_size,
            ttl_seconds=self.ttl_seconds,
            eviction_policy=self.eviction_policy,
            on_miss=on_miss,
            name=self.name,
            namespace=self.namespace,
        )

    def with_name(self, name: str) -> CacheConfig:
        """Create config with a name.

        Args:
            name: Name for the cache.

        Returns:
            New CacheConfig with updated value.
        """
        return CacheConfig(
            max_size=self.max_size,
            ttl_seconds=self.ttl_seconds,
            eviction_policy=self.eviction_policy,
            on_miss=self.on_miss,
            name=name,
            namespace=self.namespace,
        )

    def with_namespace(self, namespace: str) -> CacheConfig:
        """Create config with a namespace.

        Args:
            namespace: Namespace for key isolation.

        Returns:
            New CacheConfig with updated value.
        """
        return CacheConfig(
            max_size=self.max_size,
            ttl_seconds=self.ttl_seconds,
            eviction_policy=self.eviction_policy,
            on_miss=self.on_miss,
            name=self.name,
            namespace=namespace,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "eviction_policy": self.eviction_policy.name,
            "on_miss": self.on_miss.name,
            "name": self.name,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create CacheConfig from dictionary.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New CacheConfig instance.
        """
        eviction_policy = EvictionPolicy[data.get("eviction_policy", "LRU")]
        on_miss = CacheAction[data.get("on_miss", "COMPUTE")]
        return cls(
            max_size=data.get("max_size", 1000),
            ttl_seconds=data.get("ttl_seconds"),
            eviction_policy=eviction_policy,
            on_miss=on_miss,
            name=data.get("name"),
            namespace=data.get("namespace"),
        )


# Default configurations for common use cases
DEFAULT_CACHE_CONFIG = CacheConfig()

# Small cache for frequently accessed data
SMALL_CACHE_CONFIG = CacheConfig(
    max_size=100,
    ttl_seconds=60.0,
    eviction_policy=EvictionPolicy.LRU,
)

# Large cache for persistent data
LARGE_CACHE_CONFIG = CacheConfig(
    max_size=10000,
    ttl_seconds=3600.0,
    eviction_policy=EvictionPolicy.LRU,
)

# Short-lived cache for transient data
SHORT_TTL_CACHE_CONFIG = CacheConfig(
    max_size=500,
    ttl_seconds=30.0,
    eviction_policy=EvictionPolicy.TTL,
)

# Long-lived cache for stable data
LONG_TTL_CACHE_CONFIG = CacheConfig(
    max_size=1000,
    ttl_seconds=86400.0,  # 24 hours
    eviction_policy=EvictionPolicy.LRU,
)

# No eviction cache (use with caution)
NO_EVICTION_CACHE_CONFIG = CacheConfig(
    max_size=10000,
    eviction_policy=EvictionPolicy.NONE,
)


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata.

    Attributes:
        value: The cached value.
        created_at: Timestamp when the entry was created.
        accessed_at: Timestamp of last access.
        access_count: Number of times the entry was accessed.
        expires_at: Optional expiration timestamp.
    """

    value: V
    created_at: float = field(default_factory=time.monotonic)
    accessed_at: float = field(default_factory=time.monotonic)
    access_count: int = 0
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if the entry has expired.

        Returns:
            True if expired.
        """
        if self.expires_at is None:
            return False
        return time.monotonic() > self.expires_at

    def touch(self) -> None:
        """Update access metadata."""
        self.accessed_at = time.monotonic()
        self.access_count += 1


# =============================================================================
# Cache Statistics
# =============================================================================


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Cache statistics snapshot.

    Attributes:
        size: Current number of items in cache.
        max_size: Maximum cache size.
        hits: Total cache hits.
        misses: Total cache misses.
        evictions: Total evictions.
        hit_rate: Cache hit rate (0.0 to 1.0).
    """

    size: int
    max_size: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float

    @property
    def miss_rate(self) -> float:
        """Get cache miss rate."""
        return 1.0 - self.hit_rate

    @property
    def utilization(self) -> float:
        """Get cache utilization (size / max_size)."""
        if self.max_size == 0:
            return 0.0
        return self.size / self.max_size


# =============================================================================
# Cache Backend Implementations
# =============================================================================


class InMemoryCache(Generic[K, V]):
    """Simple in-memory cache implementation.

    Thread-safe dictionary-based cache with no eviction policy.

    Example:
        >>> cache = InMemoryCache[str, dict]()
        >>> cache.set("user:1", {"name": "Alice"})
        >>> cache.get("user:1")
        {'name': 'Alice'}
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of items.
        """
        self._cache: dict[K, CacheEntry[V]] = {}
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> V | None:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return None

            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Optional TTL in seconds.
        """
        with self._lock:
            expires_at = None
            if ttl_seconds is not None:
                expires_at = time.monotonic() + ttl_seconds

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )

    def delete(self, key: K) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from the cache."""
        with self._lock:
            self._cache.clear()

    def contains(self, key: K) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The cache key.

        Returns:
            True if exists and not expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    def size(self) -> int:
        """Get the number of items in the cache.

        Returns:
            Number of cached items.
        """
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Current cache statistics.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return CacheStats(
                size=len(self._cache),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                hit_rate=hit_rate,
            )


class LRUCache(Generic[K, V]):
    """LRU (Least Recently Used) cache implementation.

    Evicts least recently accessed items when the cache is full.

    Example:
        >>> cache = LRUCache[str, dict](max_size=100)
        >>> cache.set("user:1", {"name": "Alice"})
        >>> cache.get("user:1")
        {'name': 'Alice'}
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float | None = None,
    ) -> None:
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of items.
            ttl_seconds: Default TTL for entries.
        """
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> V | None:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Optional TTL override.
        """
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = None
            if ttl is not None:
                expires_at = time.monotonic() + ttl

            # Remove existing key if present
            if key in self._cache:
                del self._cache[key]

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )

    def delete(self, key: K) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from the cache."""
        with self._lock:
            self._cache.clear()

    def contains(self, key: K) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The cache key.

        Returns:
            True if exists and not expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    def size(self) -> int:
        """Get the number of items in the cache.

        Returns:
            Number of cached items.
        """
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Current cache statistics.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return CacheStats(
                size=len(self._cache),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                hit_rate=hit_rate,
            )


class LFUCache(Generic[K, V]):
    """LFU (Least Frequently Used) cache implementation.

    Evicts least frequently accessed items when the cache is full.

    Example:
        >>> cache = LFUCache[str, dict](max_size=100)
        >>> cache.set("user:1", {"name": "Alice"})
        >>> cache.get("user:1")
        {'name': 'Alice'}
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float | None = None,
    ) -> None:
        """Initialize LFU cache.

        Args:
            max_size: Maximum number of items.
            ttl_seconds: Default TTL for entries.
        """
        self._cache: dict[K, CacheEntry[V]] = {}
        self._freq_map: dict[K, int] = {}
        self._max_size = max_size
        self._default_ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _get_min_freq_key(self) -> K | None:
        """Get the key with minimum frequency.

        Returns:
            The key with lowest access count, or None if empty.
        """
        if not self._freq_map:
            return None

        min_freq = min(self._freq_map.values())
        for key, freq in self._freq_map.items():
            if freq == min_freq:
                return key
        return None

    def get(self, key: K) -> V | None:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                del self._freq_map[key]
                self._misses += 1
                self._evictions += 1
                return None

            entry.touch()
            self._freq_map[key] = self._freq_map.get(key, 0) + 1
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Optional TTL override.
        """
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = None
            if ttl is not None:
                expires_at = time.monotonic() + ttl

            # Update existing key
            if key in self._cache:
                self._cache[key] = CacheEntry(
                    value=value,
                    expires_at=expires_at,
                    access_count=self._cache[key].access_count,
                )
                return

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                evict_key = self._get_min_freq_key()
                if evict_key is not None:
                    del self._cache[evict_key]
                    del self._freq_map[evict_key]
                    self._evictions += 1
                else:
                    break

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )
            self._freq_map[key] = 1

    def delete(self, key: K) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._freq_map[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from the cache."""
        with self._lock:
            self._cache.clear()
            self._freq_map.clear()

    def contains(self, key: K) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The cache key.

        Returns:
            True if exists and not expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                del self._freq_map[key]
                return False
            return True

    def size(self) -> int:
        """Get the number of items in the cache.

        Returns:
            Number of cached items.
        """
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Current cache statistics.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return CacheStats(
                size=len(self._cache),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                hit_rate=hit_rate,
            )


class TTLCache(Generic[K, V]):
    """TTL-based cache implementation.

    Items are automatically evicted after their TTL expires.

    Example:
        >>> cache = TTLCache[str, dict](ttl_seconds=300.0)
        >>> cache.set("user:1", {"name": "Alice"})
        >>> cache.get("user:1")
        {'name': 'Alice'}
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
    ) -> None:
        """Initialize TTL cache.

        Args:
            max_size: Maximum number of items.
            ttl_seconds: Default TTL for entries.
        """
        self._cache: dict[K, CacheEntry[V]] = {}
        self._expiry_heap: list[tuple[float, K]] = []
        self._max_size = max_size
        self._default_ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.monotonic()
        while self._expiry_heap:
            expires_at, key = self._expiry_heap[0]
            if expires_at > now:
                break

            heappop(self._expiry_heap)
            entry = self._cache.get(key)
            if entry is not None and entry.expires_at == expires_at:
                del self._cache[key]
                self._evictions += 1

    def get(self, key: K) -> V | None:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found/expired.
        """
        with self._lock:
            self._cleanup_expired()

            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return None

            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Optional TTL override.
        """
        with self._lock:
            self._cleanup_expired()

            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = time.monotonic() + ttl

            # Remove existing key
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                if self._expiry_heap:
                    _, evict_key = heappop(self._expiry_heap)
                    if evict_key in self._cache:
                        del self._cache[evict_key]
                        self._evictions += 1
                else:
                    break

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )
            heappush(self._expiry_heap, (expires_at, key))

    def delete(self, key: K) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all values from the cache."""
        with self._lock:
            self._cache.clear()
            self._expiry_heap.clear()

    def contains(self, key: K) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The cache key.

        Returns:
            True if exists and not expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    def size(self) -> int:
        """Get the number of items in the cache.

        Returns:
            Number of cached items.
        """
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Current cache statistics.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return CacheStats(
                size=len(self._cache),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                hit_rate=hit_rate,
            )


# =============================================================================
# Cache Factory
# =============================================================================


def create_cache(
    config: CacheConfig,
    hooks: Sequence[CacheHook] | None = None,  # noqa: ARG001
) -> LRUCache[str, Any] | LFUCache[str, Any] | TTLCache[str, Any] | InMemoryCache[str, Any]:
    """Create a cache based on configuration.

    Args:
        config: Cache configuration.
        hooks: Cache event hooks (reserved for future use).

    Returns:
        Cache instance.
    """
    if config.eviction_policy == EvictionPolicy.LRU:
        return LRUCache(
            max_size=config.max_size,
            ttl_seconds=config.ttl_seconds,
        )
    elif config.eviction_policy == EvictionPolicy.LFU:
        return LFUCache(
            max_size=config.max_size,
            ttl_seconds=config.ttl_seconds,
        )
    elif config.eviction_policy == EvictionPolicy.TTL:
        return TTLCache(
            max_size=config.max_size,
            ttl_seconds=config.ttl_seconds or 300.0,
        )
    else:
        return InMemoryCache(max_size=config.max_size)


# =============================================================================
# Cache Executor
# =============================================================================


class CacheExecutor:
    """Executes functions with caching.

    This class encapsulates the caching logic and can be
    used directly or through the cached decorator.

    Example:
        >>> executor = CacheExecutor(CacheConfig(max_size=100))
        >>> result = executor.execute(my_function, "arg1", key="value")
    """

    def __init__(
        self,
        config: CacheConfig,
        cache: CacheBackend[str, Any] | None = None,
        key_generator: KeyGenerator | None = None,
        hooks: Sequence[CacheHook] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            config: Cache configuration.
            cache: Custom cache backend (created from config if None).
            key_generator: Custom key generator.
            hooks: Cache event hooks.
        """
        self.config = config
        self._cache = cache or create_cache(config, hooks)
        self._key_generator = key_generator or DefaultKeyGenerator()
        self._hook: CacheHook | None = None
        if hooks:
            self._hook = CompositeCacheHook(list(hooks))

    def _create_context(self, func: Callable[..., Any], key: str) -> dict[str, Any]:
        """Create context dictionary for hooks.

        Args:
            func: Function being cached.
            key: Cache key.

        Returns:
            Context dictionary.
        """
        return {
            "function": func.__name__,
            "module": func.__module__,
            "cache_name": self.config.name or func.__name__,
            "key": key,
            "max_size": self.config.max_size,
        }

    def _get_namespaced_key(self, key: str) -> str:
        """Add namespace to key if configured.

        Args:
            key: Original key.

        Returns:
            Namespaced key.
        """
        if self.config.namespace:
            return f"{self.config.namespace}:{key}"
        return key

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with caching.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value (cached or computed).
        """
        key = self._key_generator.generate_key(func, args, kwargs)
        key = self._get_namespaced_key(key)
        context = self._create_context(func, key)

        # Check cache
        cached_value = self._cache.get(key)
        if cached_value is not None:
            if self._hook:
                self._hook.on_hit(key, cached_value, context)
            return cached_value

        # Cache miss
        if self._hook:
            self._hook.on_miss(key, context)

        if self.config.on_miss == CacheAction.RAISE:
            raise CacheKeyError(key, cache_name=self.config.name)

        if self.config.on_miss == CacheAction.RETURN_NONE:
            return None  # type: ignore[return-value]

        # Compute value
        result = func(*args, **kwargs)

        # Cache result
        self._cache.set(key, result, self.config.ttl_seconds)
        if self._hook:
            self._hook.on_set(key, result, self.config.ttl_seconds, context)

        return result

    async def execute_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute async function with caching.

        Args:
            func: Async function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value (cached or computed).
        """
        key = self._key_generator.generate_key(func, args, kwargs)
        key = self._get_namespaced_key(key)
        context = self._create_context(func, key)

        # Check cache
        cached_value = self._cache.get(key)
        if cached_value is not None:
            if self._hook:
                self._hook.on_hit(key, cached_value, context)
            return cached_value

        # Cache miss
        if self._hook:
            self._hook.on_miss(key, context)

        if self.config.on_miss == CacheAction.RAISE:
            raise CacheKeyError(key, cache_name=self.config.name)

        if self.config.on_miss == CacheAction.RETURN_NONE:
            return None  # type: ignore[return-value]

        # Compute value
        result = await func(*args, **kwargs)

        # Cache result
        self._cache.set(key, result, self.config.ttl_seconds)
        if self._hook:
            self._hook.on_set(key, result, self.config.ttl_seconds, context)

        return result

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if invalidated, False if not found.
        """
        key = self._get_namespaced_key(key)
        return self._cache.delete(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()


# =============================================================================
# Cache Registry
# =============================================================================


class CacheRegistry:
    """Registry for managing multiple caches.

    Provides a central location to create, retrieve, and manage
    caches by name.

    Example:
        >>> registry = CacheRegistry()
        >>> cache = registry.get_or_create("users", config=my_config)
        >>> cache.set("user:1", user_data)
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._caches: dict[str, CacheBackend[str, Any]] = {}
        self._configs: dict[str, CacheConfig] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CacheBackend[str, Any] | None:
        """Get cache by name.

        Args:
            name: Cache name.

        Returns:
            Cache if found, None otherwise.
        """
        with self._lock:
            return self._caches.get(name)

    def get_or_create(
        self,
        name: str,
        config: CacheConfig | None = None,
        hooks: Sequence[CacheHook] | None = None,
    ) -> CacheBackend[str, Any]:
        """Get existing or create new cache.

        Args:
            name: Cache name.
            config: Configuration (uses default if None).
            hooks: Cache event hooks.

        Returns:
            Cache instance.
        """
        with self._lock:
            if name in self._caches:
                return self._caches[name]

            cache_config = (config or DEFAULT_CACHE_CONFIG).with_name(name)
            cache = create_cache(cache_config, hooks)
            self._caches[name] = cache
            self._configs[name] = cache_config
            return cache

    def remove(self, name: str) -> bool:
        """Remove cache by name.

        Args:
            name: Cache name.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                del self._configs[name]
                return True
            return False

    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()

    def get_all_stats(self) -> dict[str, CacheStats]:
        """Get stats for all caches.

        Returns:
            Dictionary mapping names to stats.
        """
        with self._lock:
            result = {}
            for name, cache in self._caches.items():
                if hasattr(cache, "get_stats"):
                    result[name] = cache.get_stats()
            return result

    @property
    def names(self) -> list[str]:
        """Get all cache names."""
        with self._lock:
            return list(self._caches.keys())


# Global registry instance
_default_registry = CacheRegistry()


def get_cache(
    name: str,
    config: CacheConfig | None = None,
    hooks: Sequence[CacheHook] | None = None,
) -> CacheBackend[str, Any]:
    """Get or create a cache from the global registry.

    Args:
        name: Cache name.
        config: Configuration (uses default if None).
        hooks: Cache event hooks.

    Returns:
        Cache instance.

    Example:
        >>> cache = get_cache("users")
        >>> cache.set("user:1", user_data)
    """
    return _default_registry.get_or_create(name, config=config, hooks=hooks)


def get_cache_registry() -> CacheRegistry:
    """Get the global cache registry.

    Returns:
        The global CacheRegistry instance.
    """
    return _default_registry


# =============================================================================
# Cache Decorator
# =============================================================================


def cached(
    *,
    config: CacheConfig | None = None,
    name: str | None = None,
    max_size: int | None = None,
    ttl_seconds: float | None = None,
    eviction_policy: EvictionPolicy | None = None,
    hooks: Sequence[CacheHook] | None = None,
    key_generator: KeyGenerator | None = None,
    use_registry: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add caching to functions.

    Can be used with a CacheConfig object or individual parameters.
    Supports both sync and async functions.

    Args:
        config: Complete cache configuration (takes precedence).
        name: Name for the cache (for registry and logging).
        max_size: Maximum number of cached items.
        ttl_seconds: Time-to-live for cached items.
        eviction_policy: Policy for evicting items.
        hooks: Cache event hooks.
        key_generator: Custom key generator.
        use_registry: If True, use global registry (enables sharing).

    Returns:
        Decorator function.

    Example:
        >>> @cached(ttl_seconds=300.0)
        ... def fetch_user(user_id: str) -> dict:
        ...     return db.query(user_id)

        >>> @cached(config=CacheConfig(max_size=1000))
        ... async def async_fetch(user_id: str) -> dict:
        ...     return await db.async_query(user_id)
    """
    # Build config from parameters if not provided
    if config is None:
        config = CacheConfig(
            max_size=max_size if max_size is not None else 1000,
            ttl_seconds=ttl_seconds,
            eviction_policy=eviction_policy if eviction_policy is not None else EvictionPolicy.LRU,
            name=name,
        )
    elif name is not None:
        config = config.with_name(name)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache_name = config.name or func.__name__

        if use_registry:
            cache = get_cache(cache_name, config=config, hooks=hooks)
        else:
            cache = create_cache(config.with_name(cache_name), hooks)

        executor = CacheExecutor(
            config=config,
            cache=cache,
            key_generator=key_generator,
            hooks=hooks,
        )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await executor.execute_async(func, *args, **kwargs)

            # Add cache control methods
            async_wrapper.invalidate = executor.invalidate  # type: ignore[attr-defined]
            async_wrapper.clear = executor.clear  # type: ignore[attr-defined]
            async_wrapper.cache = cache  # type: ignore[attr-defined]

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return executor.execute(func, *args, **kwargs)

            # Add cache control methods
            sync_wrapper.invalidate = executor.invalidate  # type: ignore[attr-defined]
            sync_wrapper.clear = executor.clear  # type: ignore[attr-defined]
            sync_wrapper.cache = cache  # type: ignore[attr-defined]

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def cache_get(
    name: str,
    key: str,
    default: T | None = None,
) -> T | None:
    """Get a value from a named cache.

    Args:
        name: Cache name.
        key: Cache key.
        default: Default value if not found.

    Returns:
        Cached value or default.

    Example:
        >>> user = cache_get("users", "user:1")
    """
    cache = get_cache(name)
    result = cache.get(key)
    return result if result is not None else default


def cache_set(
    name: str,
    key: str,
    value: Any,
    ttl_seconds: float | None = None,
) -> None:
    """Set a value in a named cache.

    Args:
        name: Cache name.
        key: Cache key.
        value: Value to cache.
        ttl_seconds: Optional TTL.

    Example:
        >>> cache_set("users", "user:1", user_data, ttl_seconds=300.0)
    """
    cache = get_cache(name)
    cache.set(key, value, ttl_seconds)


def cache_delete(
    name: str,
    key: str,
) -> bool:
    """Delete a value from a named cache.

    Args:
        name: Cache name.
        key: Cache key.

    Returns:
        True if deleted.

    Example:
        >>> cache_delete("users", "user:1")
    """
    cache = get_cache(name)
    return cache.delete(key)


def cache_clear(name: str) -> None:
    """Clear all values from a named cache.

    Args:
        name: Cache name.

    Example:
        >>> cache_clear("users")
    """
    cache = get_cache(name)
    cache.clear()


def cache_stats(name: str) -> CacheStats | None:
    """Get statistics for a named cache.

    Args:
        name: Cache name.

    Returns:
        Cache statistics or None if cache not found.

    Example:
        >>> stats = cache_stats("users")
        >>> print(f"Hit rate: {stats.hit_rate:.2%}")
    """
    cache = get_cache(name)
    if hasattr(cache, "get_stats"):
        return cache.get_stats()
    return None
