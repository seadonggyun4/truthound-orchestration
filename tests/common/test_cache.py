"""Tests for the cache module.

This module contains comprehensive tests for all caching functionality:
- Cache backends (InMemory, LRU, LFU, TTL)
- Configuration validation
- Hook system
- Decorator and executor patterns
- Registry management
- Async support
- Key generators
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import Any

import pytest


# Check if pytest-asyncio is available
HAS_PYTEST_ASYNCIO = importlib.util.find_spec("pytest_asyncio") is not None

asyncio_test = pytest.mark.skipif(
    not HAS_PYTEST_ASYNCIO,
    reason="pytest-asyncio not installed",
)

from common.cache import (
    DEFAULT_CACHE_CONFIG,
    LARGE_CACHE_CONFIG,
    LONG_TTL_CACHE_CONFIG,
    NO_EVICTION_CACHE_CONFIG,
    SHORT_TTL_CACHE_CONFIG,
    SMALL_CACHE_CONFIG,
    ArgumentKeyGenerator,
    CacheAction,
    # Configuration
    CacheConfig,
    # Data Types
    CacheEntry,
    # Exceptions
    CacheError,
    # Executor
    CacheExecutor,
    CacheFullError,
    CacheKeyError,
    # Registry
    CacheRegistry,
    CacheStats,
    CallableKeyGenerator,
    CompositeCacheHook,
    # Key Generators
    DefaultKeyGenerator,
    EvictionPolicy,
    # Backends
    InMemoryCache,
    LFUCache,
    # Hooks
    LoggingCacheHook,
    LRUCache,
    MetricsCacheHook,
    TTLCache,
    cache_clear,
    cache_delete,
    cache_get,
    cache_set,
    cache_stats,
    # Decorators and Functions
    cached,
    # Factory
    create_cache,
    get_cache,
    get_cache_registry,
)


# =============================================================================
# Exception Tests
# =============================================================================


class TestCacheError:
    """Tests for CacheError exception."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = CacheError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_error_with_details(self) -> None:
        """Test error with details."""
        error = CacheError(
            "Cache error",
            cache_name="test_cache",
            key="test_key",
        )
        assert error.cache_name == "test_cache"
        assert error.key == "test_key"
        assert error.details["cache_name"] == "test_cache"
        assert error.details["key"] == "test_key"


class TestCacheFullError:
    """Tests for CacheFullError exception."""

    def test_full_error(self) -> None:
        """Test cache full error creation."""
        error = CacheFullError(
            cache_name="test_cache",
            max_size=100,
            current_size=100,
        )
        assert error.cache_name == "test_cache"
        assert error.max_size == 100
        assert error.current_size == 100


class TestCacheKeyError:
    """Tests for CacheKeyError exception."""

    def test_key_error(self) -> None:
        """Test cache key error creation."""
        error = CacheKeyError(
            "missing_key",
            cache_name="test_cache",
        )
        assert error.key == "missing_key"
        assert error.cache_name == "test_cache"
        assert "missing_key" in str(error)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.max_size == 1000
        assert config.ttl_seconds is None
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.on_miss == CacheAction.COMPUTE

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CacheConfig(
            max_size=500,
            ttl_seconds=300.0,
            eviction_policy=EvictionPolicy.LFU,
            name="custom_cache",
        )
        assert config.max_size == 500
        assert config.ttl_seconds == 300.0
        assert config.eviction_policy == EvictionPolicy.LFU
        assert config.name == "custom_cache"

    def test_validation_max_size(self) -> None:
        """Test max_size validation."""
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            CacheConfig(max_size=0)

    def test_validation_ttl_seconds(self) -> None:
        """Test ttl_seconds validation."""
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            CacheConfig(ttl_seconds=0)

        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            CacheConfig(ttl_seconds=-1.0)

    def test_builder_methods(self) -> None:
        """Test builder methods return new instances."""
        config = CacheConfig()

        new_config = config.with_max_size(500)
        assert new_config.max_size == 500
        assert config.max_size == 1000  # Original unchanged

        new_config = config.with_ttl(300.0)
        assert new_config.ttl_seconds == 300.0

        new_config = config.with_eviction_policy(EvictionPolicy.LFU)
        assert new_config.eviction_policy == EvictionPolicy.LFU

        new_config = config.with_on_miss(CacheAction.RAISE)
        assert new_config.on_miss == CacheAction.RAISE

        new_config = config.with_name("test")
        assert new_config.name == "test"

        new_config = config.with_namespace("ns")
        assert new_config.namespace == "ns"

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        config = CacheConfig(
            max_size=500,
            ttl_seconds=300.0,
            eviction_policy=EvictionPolicy.LFU,
            name="test",
        )
        data = config.to_dict()
        restored = CacheConfig.from_dict(data)

        assert restored.max_size == config.max_size
        assert restored.ttl_seconds == config.ttl_seconds
        assert restored.eviction_policy == config.eviction_policy
        assert restored.name == config.name

    def test_preset_configs(self) -> None:
        """Test preset configurations."""
        assert DEFAULT_CACHE_CONFIG.max_size == 1000
        assert SMALL_CACHE_CONFIG.max_size == 100
        assert LARGE_CACHE_CONFIG.max_size == 10000
        assert SHORT_TTL_CACHE_CONFIG.ttl_seconds == 30.0
        assert LONG_TTL_CACHE_CONFIG.ttl_seconds == 86400.0
        assert NO_EVICTION_CACHE_CONFIG.eviction_policy == EvictionPolicy.NONE


# =============================================================================
# Cache Entry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_basic_entry(self) -> None:
        """Test basic entry creation."""
        entry = CacheEntry(value="test")
        assert entry.value == "test"
        assert entry.access_count == 0

    def test_is_expired_no_ttl(self) -> None:
        """Test is_expired with no TTL."""
        entry = CacheEntry(value="test")
        assert not entry.is_expired()

    def test_is_expired_with_ttl(self) -> None:
        """Test is_expired with TTL."""
        entry = CacheEntry(value="test", expires_at=time.monotonic() + 1.0)
        assert not entry.is_expired()

        entry = CacheEntry(value="test", expires_at=time.monotonic() - 1.0)
        assert entry.is_expired()

    def test_touch(self) -> None:
        """Test touch updates metadata."""
        entry = CacheEntry(value="test")
        original_accessed = entry.accessed_at
        time.sleep(0.01)
        entry.touch()
        assert entry.access_count == 1
        assert entry.accessed_at > original_accessed


# =============================================================================
# Cache Stats Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_stats(self) -> None:
        """Test stats properties."""
        stats = CacheStats(
            size=50,
            max_size=100,
            hits=80,
            misses=20,
            evictions=5,
            hit_rate=0.8,
        )
        assert abs(stats.miss_rate - 0.2) < 1e-9
        assert stats.utilization == 0.5


# =============================================================================
# Key Generator Tests
# =============================================================================


class TestDefaultKeyGenerator:
    """Tests for DefaultKeyGenerator."""

    def test_basic_key_generation(self) -> None:
        """Test basic key generation."""
        generator = DefaultKeyGenerator()

        def test_func(a: int, b: str) -> str:
            return f"{a}-{b}"

        key = generator.generate_key(test_func, (1,), {"b": "test"})
        assert "test_func" in key
        assert "(1,)" in key or "1" in key

    def test_with_prefix(self) -> None:
        """Test key generation with prefix."""
        generator = DefaultKeyGenerator(prefix="cache")

        def test_func() -> None:
            pass

        key = generator.generate_key(test_func, (), {})
        assert key.startswith("cache:")

    def test_hash_args(self) -> None:
        """Test key generation with hashed args."""
        generator = DefaultKeyGenerator(hash_args=True)

        def test_func(data: dict[str, Any]) -> None:
            pass

        key = generator.generate_key(test_func, ({"a": 1},), {})
        # Should contain a hash instead of the dict representation
        assert "test_func" in key


class TestArgumentKeyGenerator:
    """Tests for ArgumentKeyGenerator."""

    def test_arg_name_extraction(self) -> None:
        """Test extraction by argument name."""
        generator = ArgumentKeyGenerator(arg_names=("user_id",))

        def test_func(user_id: str) -> None:
            pass

        key = generator.generate_key(test_func, (), {"user_id": "123"})
        assert "user_id=123" in key

    def test_arg_index_extraction(self) -> None:
        """Test extraction by argument index."""
        generator = ArgumentKeyGenerator(arg_indices=(0,))

        def test_func(user_id: str) -> None:
            pass

        key = generator.generate_key(test_func, ("123",), {})
        assert "123" in key


class TestCallableKeyGenerator:
    """Tests for CallableKeyGenerator."""

    def test_callable_extraction(self) -> None:
        """Test custom callable extraction."""
        def custom_extractor(
            func: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> str:
            return f"custom:{kwargs.get('id', 'default')}"

        generator = CallableKeyGenerator(custom_extractor)

        def test_func() -> None:
            pass

        key = generator.generate_key(test_func, (), {"id": "abc"})
        assert key == "custom:abc"


# =============================================================================
# Hook Tests
# =============================================================================


class TestLoggingCacheHook:
    """Tests for LoggingCacheHook."""

    def test_hook_methods_exist(self) -> None:
        """Test that all hook methods exist."""
        hook = LoggingCacheHook()
        assert hasattr(hook, "on_hit")
        assert hasattr(hook, "on_miss")
        assert hasattr(hook, "on_set")
        assert hasattr(hook, "on_evict")

    def test_hook_methods_dont_raise(self) -> None:
        """Test that hook methods don't raise exceptions."""
        hook = LoggingCacheHook()
        hook.on_hit("key", "value", {})
        hook.on_miss("key", {})
        hook.on_set("key", "value", 60.0, {})
        hook.on_evict("key", "lru", {})


class TestMetricsCacheHook:
    """Tests for MetricsCacheHook."""

    def test_metrics_collection(self) -> None:
        """Test metrics are collected correctly."""
        hook = MetricsCacheHook()

        hook.on_hit("key1", "value", {})
        hook.on_hit("key1", "value", {})
        hook.on_miss("key2", {})
        hook.on_set("key3", "value", None, {})
        hook.on_evict("key4", "lru", {})

        assert hook.hits == 2
        assert hook.misses == 1
        assert hook.sets == 1
        assert hook.evictions == 1
        assert hook.hit_rate == 2 / 3

    def test_key_stats(self) -> None:
        """Test per-key statistics."""
        hook = MetricsCacheHook()

        hook.on_hit("key1", "value", {})
        hook.on_hit("key1", "value", {})
        hook.on_miss("key1", {})

        stats = hook.get_key_stats("key1")
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_eviction_reasons(self) -> None:
        """Test eviction reason tracking."""
        hook = MetricsCacheHook()

        hook.on_evict("key1", "lru", {})
        hook.on_evict("key2", "lru", {})
        hook.on_evict("key3", "ttl_expired", {})

        reasons = hook.get_eviction_reasons()
        assert reasons["lru"] == 2
        assert reasons["ttl_expired"] == 1

    def test_reset(self) -> None:
        """Test metrics reset."""
        hook = MetricsCacheHook()
        hook.on_hit("key", "value", {})
        hook.reset()
        assert hook.hits == 0


class TestCompositeCacheHook:
    """Tests for CompositeCacheHook."""

    def test_composite_calls_all_hooks(self) -> None:
        """Test composite hook calls all child hooks."""
        hook1 = MetricsCacheHook()
        hook2 = MetricsCacheHook()
        composite = CompositeCacheHook([hook1, hook2])

        composite.on_hit("key", "value", {})
        assert hook1.hits == 1
        assert hook2.hits == 1

    def test_add_remove_hook(self) -> None:
        """Test adding and removing hooks."""
        hook1 = MetricsCacheHook()
        composite = CompositeCacheHook([])

        composite.add_hook(hook1)
        composite.on_hit("key", "value", {})
        assert hook1.hits == 1

        composite.remove_hook(hook1)
        composite.on_hit("key", "value", {})
        assert hook1.hits == 1  # Not incremented


# =============================================================================
# Cache Backend Tests
# =============================================================================


class TestInMemoryCache:
    """Tests for InMemoryCache."""

    def test_basic_operations(self) -> None:
        """Test basic get/set/delete."""
        cache: InMemoryCache[str, str] = InMemoryCache()

        cache.set("key", "value")
        assert cache.get("key") == "value"
        assert cache.contains("key")
        assert cache.size() == 1

        assert cache.delete("key")
        assert cache.get("key") is None
        assert not cache.contains("key")
        assert cache.size() == 0

    def test_get_nonexistent(self) -> None:
        """Test get returns None for nonexistent key."""
        cache: InMemoryCache[str, str] = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_clear(self) -> None:
        """Test clear removes all entries."""
        cache: InMemoryCache[str, str] = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size() == 0

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        cache: InMemoryCache[str, str] = InMemoryCache()
        cache.set("key", "value", ttl_seconds=0.1)

        assert cache.get("key") == "value"
        time.sleep(0.15)
        assert cache.get("key") is None

    def test_stats(self) -> None:
        """Test statistics tracking."""
        cache: InMemoryCache[str, str] = InMemoryCache()
        cache.set("key", "value")
        cache.get("key")  # hit
        cache.get("missing")  # miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1


class TestLRUCache:
    """Tests for LRUCache."""

    def test_basic_operations(self) -> None:
        """Test basic get/set/delete."""
        cache: LRUCache[str, str] = LRUCache()

        cache.set("key", "value")
        assert cache.get("key") == "value"
        assert cache.contains("key")

    def test_lru_eviction(self) -> None:
        """Test LRU eviction policy."""
        cache: LRUCache[str, int] = LRUCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' to make it most recently used
        cache.get("a")

        # Add new item, should evict 'b' (least recently used)
        cache.set("d", 4)

        assert cache.get("a") == 1  # Still present
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_ttl_with_lru(self) -> None:
        """Test TTL works with LRU."""
        cache: LRUCache[str, str] = LRUCache(ttl_seconds=0.1)
        cache.set("key", "value")

        assert cache.get("key") == "value"
        time.sleep(0.15)
        assert cache.get("key") is None


class TestLFUCache:
    """Tests for LFUCache."""

    def test_basic_operations(self) -> None:
        """Test basic get/set/delete."""
        cache: LFUCache[str, str] = LFUCache()

        cache.set("key", "value")
        assert cache.get("key") == "value"
        assert cache.contains("key")

    def test_lfu_eviction(self) -> None:
        """Test LFU eviction policy."""
        cache: LFUCache[str, int] = LFUCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' multiple times
        cache.get("a")
        cache.get("a")
        cache.get("a")

        # Access 'b' once
        cache.get("b")

        # 'c' is never accessed, should be evicted first
        cache.set("d", 4)

        assert cache.get("a") == 1  # Most frequently used
        assert cache.get("b") == 2  # Second most frequently used
        assert cache.get("c") is None  # Evicted (least frequently used)
        assert cache.get("d") == 4


class TestTTLCache:
    """Tests for TTLCache."""

    def test_basic_operations(self) -> None:
        """Test basic get/set/delete."""
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=300.0)

        cache.set("key", "value")
        assert cache.get("key") == "value"
        assert cache.contains("key")

    def test_ttl_expiration(self) -> None:
        """Test automatic TTL expiration."""
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=0.1)
        cache.set("key", "value")

        assert cache.get("key") == "value"
        time.sleep(0.15)
        assert cache.get("key") is None

    def test_custom_ttl_per_item(self) -> None:
        """Test custom TTL per item."""
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=300.0)

        cache.set("short", "value", ttl_seconds=0.1)
        cache.set("long", "value", ttl_seconds=300.0)

        time.sleep(0.15)
        assert cache.get("short") is None
        assert cache.get("long") == "value"


# =============================================================================
# Factory Tests
# =============================================================================


class TestCreateCache:
    """Tests for create_cache factory function."""

    def test_create_lru_cache(self) -> None:
        """Test creating LRU cache."""
        config = CacheConfig(eviction_policy=EvictionPolicy.LRU)
        cache = create_cache(config)
        assert isinstance(cache, LRUCache)

    def test_create_lfu_cache(self) -> None:
        """Test creating LFU cache."""
        config = CacheConfig(eviction_policy=EvictionPolicy.LFU)
        cache = create_cache(config)
        assert isinstance(cache, LFUCache)

    def test_create_ttl_cache(self) -> None:
        """Test creating TTL cache."""
        config = CacheConfig(eviction_policy=EvictionPolicy.TTL)
        cache = create_cache(config)
        assert isinstance(cache, TTLCache)

    def test_create_memory_cache(self) -> None:
        """Test creating in-memory cache."""
        config = CacheConfig(eviction_policy=EvictionPolicy.NONE)
        cache = create_cache(config)
        assert isinstance(cache, InMemoryCache)


# =============================================================================
# Executor Tests
# =============================================================================


class TestCacheExecutor:
    """Tests for CacheExecutor."""

    def test_execute_caches_result(self) -> None:
        """Test that executor caches results."""
        config = CacheConfig()
        executor = CacheExecutor(config)

        call_count = 0

        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = executor.execute(compute, 5)
        result2 = executor.execute(compute, 5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once

    def test_execute_different_args(self) -> None:
        """Test that different args get different cache entries."""
        config = CacheConfig()
        executor = CacheExecutor(config)

        call_count = 0

        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = executor.execute(compute, 5)
        result2 = executor.execute(compute, 10)

        assert result1 == 10
        assert result2 == 20
        assert call_count == 2

    def test_invalidate(self) -> None:
        """Test cache invalidation."""
        config = CacheConfig()
        executor = CacheExecutor(config)

        call_count = 0

        def compute() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        executor.execute(compute)
        assert call_count == 1

        # Invalidate and recompute
        key_gen = DefaultKeyGenerator()
        key = key_gen.generate_key(compute, (), {})
        executor.invalidate(key)

        executor.execute(compute)
        assert call_count == 2

    def test_clear(self) -> None:
        """Test clearing all cache entries."""
        config = CacheConfig()
        executor = CacheExecutor(config)

        call_count = 0

        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        executor.execute(compute, 1)
        executor.execute(compute, 2)
        assert call_count == 2

        executor.clear()

        executor.execute(compute, 1)
        executor.execute(compute, 2)
        assert call_count == 4

    def test_on_miss_raise(self) -> None:
        """Test RAISE action on cache miss."""
        config = CacheConfig(on_miss=CacheAction.RAISE)
        executor = CacheExecutor(config)

        def compute() -> str:
            return "result"

        with pytest.raises(CacheKeyError):
            executor.execute(compute)

    def test_namespace(self) -> None:
        """Test namespace isolation."""
        config1 = CacheConfig(namespace="ns1")
        config2 = CacheConfig(namespace="ns2")

        executor1 = CacheExecutor(config1)
        executor2 = CacheExecutor(config2)

        call_count = 0

        def compute() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result1 = executor1.execute(compute)
        result2 = executor2.execute(compute)

        # Different namespaces should not share cache
        assert result1 != result2


# =============================================================================
# Registry Tests
# =============================================================================


class TestCacheRegistry:
    """Tests for CacheRegistry."""

    def test_get_or_create(self) -> None:
        """Test get_or_create returns same cache."""
        registry = CacheRegistry()

        cache1 = registry.get_or_create("test")
        cache2 = registry.get_or_create("test")

        assert cache1 is cache2

    def test_get_nonexistent(self) -> None:
        """Test get returns None for nonexistent cache."""
        registry = CacheRegistry()
        assert registry.get("nonexistent") is None

    def test_remove(self) -> None:
        """Test removing a cache."""
        registry = CacheRegistry()
        registry.get_or_create("test")

        assert registry.remove("test")
        assert registry.get("test") is None
        assert not registry.remove("test")  # Already removed

    def test_clear_all(self) -> None:
        """Test clearing all caches."""
        registry = CacheRegistry()
        cache1 = registry.get_or_create("test1")
        cache2 = registry.get_or_create("test2")

        cache1.set("key", "value")
        cache2.set("key", "value")

        registry.clear_all()

        assert cache1.size() == 0
        assert cache2.size() == 0

    def test_names(self) -> None:
        """Test listing cache names."""
        registry = CacheRegistry()
        registry.get_or_create("a")
        registry.get_or_create("b")

        names = registry.names
        assert "a" in names
        assert "b" in names


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_cache(self) -> None:
        """Test get_cache function."""
        cache = get_cache("global_test_1")
        assert cache is not None

    def test_get_cache_registry(self) -> None:
        """Test get_cache_registry function."""
        registry = get_cache_registry()
        assert isinstance(registry, CacheRegistry)


# =============================================================================
# Decorator Tests
# =============================================================================


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_basic_caching(self) -> None:
        """Test basic function caching."""
        call_count = 0

        @cached(ttl_seconds=60.0, name="test_basic_caching")
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = compute(5)
        result2 = compute(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    def test_different_args_cached_separately(self) -> None:
        """Test different args are cached separately."""
        call_count = 0

        @cached(name="test_different_args")
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(1)
        compute(2)
        compute(1)  # Should be cached

        assert call_count == 2

    def test_cache_clear_method(self) -> None:
        """Test clear method on decorated function."""
        call_count = 0

        @cached(name="test_clear_method", use_registry=False)
        def compute() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        compute()
        assert call_count == 1

        compute.clear()  # type: ignore[attr-defined]
        compute()
        assert call_count == 2

    def test_with_config(self) -> None:
        """Test decorator with config object."""
        config = CacheConfig(max_size=10, ttl_seconds=60.0)

        @cached(config=config, name="test_with_config")
        def compute(x: int) -> int:
            return x * 2

        assert compute(5) == 10


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestCacheUtilityFunctions:
    """Tests for cache utility functions."""

    def test_cache_get_set(self) -> None:
        """Test cache_get and cache_set."""
        cache_set("utility_test", "key", "value")
        result = cache_get("utility_test", "key")
        assert result == "value"

    def test_cache_get_default(self) -> None:
        """Test cache_get with default."""
        result = cache_get("utility_test_2", "missing", default="default")
        assert result == "default"

    def test_cache_delete(self) -> None:
        """Test cache_delete."""
        cache_set("utility_test_3", "key", "value")
        assert cache_delete("utility_test_3", "key")
        assert cache_get("utility_test_3", "key") is None

    def test_cache_clear(self) -> None:
        """Test cache_clear."""
        cache_set("utility_test_4", "key1", "value1")
        cache_set("utility_test_4", "key2", "value2")
        cache_clear("utility_test_4")
        assert cache_get("utility_test_4", "key1") is None
        assert cache_get("utility_test_4", "key2") is None

    def test_cache_stats(self) -> None:
        """Test cache_stats."""
        cache_name = "utility_test_5"
        cache = get_cache(cache_name)
        cache.set("key", "value")
        cache.get("key")  # hit
        cache.get("missing")  # miss

        stats = cache_stats(cache_name)
        assert stats is not None
        assert stats.hits == 1
        assert stats.misses == 1


# =============================================================================
# Async Tests
# =============================================================================


@asyncio_test
class TestAsyncCaching:
    """Tests for async caching support."""

    @pytest.mark.asyncio
    async def test_async_cached_decorator(self) -> None:
        """Test @cached with async function."""
        call_count = 0

        @cached(name="test_async_cached")
        async def async_compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        result1 = await async_compute(5)
        result2 = await async_compute(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_executor(self) -> None:
        """Test CacheExecutor with async functions."""
        config = CacheConfig()
        executor = CacheExecutor(config)

        call_count = 0

        async def async_compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        result1 = await executor.execute_async(async_compute, 5)
        result2 = await executor.execute_async(async_compute, 5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestCacheIntegration:
    """Integration tests for cache module."""

    def test_cache_with_hooks(self) -> None:
        """Test caching with hooks."""
        metrics_hook = MetricsCacheHook()
        config = CacheConfig(ttl_seconds=60.0)
        executor = CacheExecutor(config, hooks=[metrics_hook])

        def compute(x: int) -> int:
            return x * 2

        executor.execute(compute, 5)  # miss + set
        executor.execute(compute, 5)  # hit

        assert metrics_hook.misses == 1
        assert metrics_hook.sets == 1
        assert metrics_hook.hits == 1

    def test_lru_cache_eviction_with_stats(self) -> None:
        """Test LRU eviction with statistics."""
        cache: LRUCache[str, int] = LRUCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict 'a'

        stats = cache.get_stats()
        assert stats.evictions >= 1
        assert cache.get("a") is None

    def test_ttl_cleanup(self) -> None:
        """Test TTL cache cleanup."""
        cache: TTLCache[str, str] = TTLCache(max_size=10, ttl_seconds=0.1)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        time.sleep(0.15)

        # Access should trigger cleanup
        cache.get("key1")

        stats = cache.get_stats()
        assert stats.evictions >= 1
