"""Tests for the rate limiter module.

This module contains comprehensive tests for all rate limiting functionality:
- Rate limit algorithms (Token Bucket, Sliding Window, Fixed Window, Leaky Bucket)
- Configuration validation
- Hook system
- Decorator and executor patterns
- Registry management
- Async support
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

# Check if pytest-asyncio is available
HAS_PYTEST_ASYNCIO = importlib.util.find_spec("pytest_asyncio") is not None

asyncio_test = pytest.mark.skipif(
    not HAS_PYTEST_ASYNCIO,
    reason="pytest-asyncio not installed",
)

from common.rate_limiter import (
    # Exceptions
    RateLimitError,
    RateLimitExceededError,
    # Enums
    RateLimitAlgorithm,
    RateLimitAction,
    # Configuration
    RateLimitConfig,
    DEFAULT_RATE_LIMIT_CONFIG,
    STRICT_RATE_LIMIT_CONFIG,
    LENIENT_RATE_LIMIT_CONFIG,
    BURST_RATE_LIMIT_CONFIG,
    API_RATE_LIMIT_CONFIG,
    # Key Extractors
    DefaultKeyExtractor,
    ArgumentKeyExtractor,
    CallableKeyExtractor,
    # Hooks
    LoggingRateLimitHook,
    MetricsRateLimitHook,
    CompositeRateLimitHook,
    # Implementations
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    FixedWindowRateLimiter,
    LeakyBucketRateLimiter,
    # Factory
    create_rate_limiter,
    # Executor
    RateLimitExecutor,
    # Registry
    RateLimiterRegistry,
    get_rate_limiter,
    get_rate_limiter_registry,
    # Decorators and Functions
    rate_limit,
    rate_limit_call,
    rate_limit_call_async,
)


# =============================================================================
# Exception Tests
# =============================================================================


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = RateLimitError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_error_with_details(self) -> None:
        """Test error with details."""
        error = RateLimitError(
            "Rate limit exceeded",
            limit=100,
            window_seconds=60.0,
            retry_after_seconds=5.0,
        )
        assert error.limit == 100
        assert error.window_seconds == 60.0
        assert error.retry_after_seconds == 5.0
        assert error.details["limit"] == 100


class TestRateLimitExceededError:
    """Tests for RateLimitExceededError exception."""

    def test_exceeded_error(self) -> None:
        """Test exceeded error creation."""
        error = RateLimitExceededError(
            key="user_123",
            current_count=100,
            limit=100,
            window_seconds=60.0,
            retry_after_seconds=5.0,
        )
        assert error.key == "user_123"
        assert error.current_count == 100
        assert error.limit == 100
        assert error.retry_after_seconds == 5.0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.max_requests == 100
        assert config.window_seconds == 60.0
        assert config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET
        assert config.on_limit == RateLimitAction.REJECT

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RateLimitConfig(
            max_requests=50,
            window_seconds=30.0,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            burst_size=10,
        )
        assert config.max_requests == 50
        assert config.window_seconds == 30.0
        assert config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW
        assert config.burst_size == 10

    def test_validation_max_requests(self) -> None:
        """Test max_requests validation."""
        with pytest.raises(ValueError, match="max_requests must be at least 1"):
            RateLimitConfig(max_requests=0)

    def test_validation_window_seconds(self) -> None:
        """Test window_seconds validation."""
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimitConfig(window_seconds=0)

    def test_validation_burst_size(self) -> None:
        """Test burst_size validation."""
        with pytest.raises(ValueError, match="burst_size must be at least 1"):
            RateLimitConfig(burst_size=0)

    def test_effective_burst_size(self) -> None:
        """Test effective_burst_size property."""
        config = RateLimitConfig(max_requests=100, burst_size=None)
        assert config.effective_burst_size == 100

        config = RateLimitConfig(max_requests=100, burst_size=50)
        assert config.effective_burst_size == 50

    def test_tokens_per_second(self) -> None:
        """Test tokens_per_second property."""
        config = RateLimitConfig(max_requests=60, window_seconds=60.0)
        assert config.tokens_per_second == 1.0

        config = RateLimitConfig(max_requests=120, window_seconds=60.0)
        assert config.tokens_per_second == 2.0

    def test_builder_methods(self) -> None:
        """Test builder methods return new instances."""
        config = RateLimitConfig()

        new_config = config.with_max_requests(50)
        assert new_config.max_requests == 50
        assert config.max_requests == 100  # Original unchanged

        new_config = config.with_window(30.0)
        assert new_config.window_seconds == 30.0

        new_config = config.with_algorithm(RateLimitAlgorithm.SLIDING_WINDOW)
        assert new_config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW

        new_config = config.with_burst_size(10)
        assert new_config.burst_size == 10

        new_config = config.with_on_limit(RateLimitAction.WAIT)
        assert new_config.on_limit == RateLimitAction.WAIT

        new_config = config.with_name("test")
        assert new_config.name == "test"

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        config = RateLimitConfig(
            max_requests=50,
            window_seconds=30.0,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            name="test",
        )

        data = config.to_dict()
        assert data["max_requests"] == 50
        assert data["algorithm"] == "SLIDING_WINDOW"

        restored = RateLimitConfig.from_dict(data)
        assert restored.max_requests == 50
        assert restored.algorithm == RateLimitAlgorithm.SLIDING_WINDOW

    def test_preset_configs(self) -> None:
        """Test preset configurations exist."""
        assert DEFAULT_RATE_LIMIT_CONFIG.max_requests == 100
        assert STRICT_RATE_LIMIT_CONFIG.max_requests == 10
        assert LENIENT_RATE_LIMIT_CONFIG.max_requests == 1000
        assert BURST_RATE_LIMIT_CONFIG.burst_size == 50
        assert API_RATE_LIMIT_CONFIG.window_seconds == 1.0


# =============================================================================
# Token Bucket Rate Limiter Tests
# =============================================================================


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_basic_acquire(self) -> None:
        """Test basic token acquisition."""
        config = RateLimitConfig(max_requests=10, window_seconds=1.0)
        limiter = TokenBucketRateLimiter(config)

        # Should be able to acquire tokens
        assert limiter.acquire("test") is True
        assert limiter.get_remaining("test") == 9

    def test_burst_limit(self) -> None:
        """Test burst limit enforcement."""
        config = RateLimitConfig(max_requests=10, window_seconds=10.0, burst_size=5)
        limiter = TokenBucketRateLimiter(config)

        # Acquire all burst tokens
        for _ in range(5):
            assert limiter.acquire("test") is True

        # Next acquire should fail
        assert limiter.acquire("test") is False

    def test_token_refill(self) -> None:
        """Test token refill over time."""
        config = RateLimitConfig(max_requests=10, window_seconds=1.0)
        limiter = TokenBucketRateLimiter(config)

        # Drain all tokens
        for _ in range(10):
            limiter.acquire("test")

        assert limiter.get_remaining("test") == 0

        # Wait for refill
        time.sleep(0.2)

        # Should have some tokens now
        assert limiter.get_remaining("test") >= 1

    def test_multiple_keys(self) -> None:
        """Test separate rate limits per key."""
        config = RateLimitConfig(max_requests=5, window_seconds=10.0)
        limiter = TokenBucketRateLimiter(config)

        # Drain key1
        for _ in range(5):
            limiter.acquire("key1")

        assert limiter.acquire("key1") is False
        assert limiter.acquire("key2") is True  # key2 unaffected

    def test_get_wait_time(self) -> None:
        """Test wait time calculation."""
        config = RateLimitConfig(max_requests=10, window_seconds=1.0)
        limiter = TokenBucketRateLimiter(config)

        # With tokens available
        assert limiter.get_wait_time("test") == 0.0

        # Drain all tokens
        for _ in range(10):
            limiter.acquire("test")

        # Wait time should be positive
        wait_time = limiter.get_wait_time("test")
        assert wait_time > 0

    def test_reset(self) -> None:
        """Test reset functionality."""
        config = RateLimitConfig(max_requests=10, window_seconds=10.0)
        limiter = TokenBucketRateLimiter(config)

        # Use some tokens
        for _ in range(5):
            limiter.acquire("test")

        assert limiter.get_remaining("test") == 5

        # Reset
        limiter.reset("test")
        assert limiter.get_remaining("test") == 10

    def test_reset_all(self) -> None:
        """Test reset all keys."""
        config = RateLimitConfig(max_requests=10, window_seconds=10.0)
        limiter = TokenBucketRateLimiter(config)

        limiter.acquire("key1")
        limiter.acquire("key2")

        limiter.reset()

        assert limiter.get_remaining("key1") == 10
        assert limiter.get_remaining("key2") == 10


# =============================================================================
# Sliding Window Rate Limiter Tests
# =============================================================================


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    def test_basic_acquire(self) -> None:
        """Test basic request acquisition."""
        config = RateLimitConfig(
            max_requests=5,
            window_seconds=1.0,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
        )
        limiter = SlidingWindowRateLimiter(config)

        for _ in range(5):
            assert limiter.acquire("test") is True

        assert limiter.acquire("test") is False

    def test_sliding_window_expiry(self) -> None:
        """Test requests expire after window."""
        config = RateLimitConfig(
            max_requests=5,
            window_seconds=0.2,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
        )
        limiter = SlidingWindowRateLimiter(config)

        # Use all requests
        for _ in range(5):
            limiter.acquire("test")

        assert limiter.acquire("test") is False

        # Wait for window to expire
        time.sleep(0.25)

        # Should be able to acquire again
        assert limiter.acquire("test") is True

    def test_get_remaining(self) -> None:
        """Test remaining count."""
        config = RateLimitConfig(
            max_requests=10,
            window_seconds=10.0,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
        )
        limiter = SlidingWindowRateLimiter(config)

        assert limiter.get_remaining("test") == 10

        for _ in range(3):
            limiter.acquire("test")

        assert limiter.get_remaining("test") == 7


# =============================================================================
# Fixed Window Rate Limiter Tests
# =============================================================================


class TestFixedWindowRateLimiter:
    """Tests for FixedWindowRateLimiter."""

    def test_basic_acquire(self) -> None:
        """Test basic request acquisition."""
        config = RateLimitConfig(
            max_requests=5,
            window_seconds=10.0,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
        )
        limiter = FixedWindowRateLimiter(config)

        for _ in range(5):
            assert limiter.acquire("test") is True

        assert limiter.acquire("test") is False

    def test_window_reset(self) -> None:
        """Test window resets after time."""
        config = RateLimitConfig(
            max_requests=5,
            window_seconds=0.2,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
        )
        limiter = FixedWindowRateLimiter(config)

        # Use all requests
        for _ in range(5):
            limiter.acquire("test")

        # Wait for window to reset
        time.sleep(0.25)

        # Should have full quota again
        assert limiter.get_remaining("test") == 5


# =============================================================================
# Leaky Bucket Rate Limiter Tests
# =============================================================================


class TestLeakyBucketRateLimiter:
    """Tests for LeakyBucketRateLimiter."""

    def test_basic_acquire(self) -> None:
        """Test basic request acquisition."""
        config = RateLimitConfig(
            max_requests=10,
            window_seconds=10.0,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
        )
        limiter = LeakyBucketRateLimiter(config)

        # Fill the bucket
        for _ in range(10):
            assert limiter.acquire("test") is True

        assert limiter.acquire("test") is False

    def test_leak_over_time(self) -> None:
        """Test bucket leaks over time."""
        config = RateLimitConfig(
            max_requests=10,
            window_seconds=1.0,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
        )
        limiter = LeakyBucketRateLimiter(config)

        # Fill the bucket
        for _ in range(10):
            limiter.acquire("test")

        assert limiter.acquire("test") is False

        # Wait for some leakage
        time.sleep(0.2)

        # Should have capacity now
        assert limiter.get_remaining("test") >= 1


# =============================================================================
# Rate Limiter Factory Tests
# =============================================================================


class TestRateLimiterFactory:
    """Tests for rate limiter factory."""

    def test_create_token_bucket(self) -> None:
        """Test creating token bucket limiter."""
        config = RateLimitConfig(algorithm=RateLimitAlgorithm.TOKEN_BUCKET)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, TokenBucketRateLimiter)

    def test_create_sliding_window(self) -> None:
        """Test creating sliding window limiter."""
        config = RateLimitConfig(algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, SlidingWindowRateLimiter)

    def test_create_fixed_window(self) -> None:
        """Test creating fixed window limiter."""
        config = RateLimitConfig(algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, FixedWindowRateLimiter)

    def test_create_leaky_bucket(self) -> None:
        """Test creating leaky bucket limiter."""
        config = RateLimitConfig(algorithm=RateLimitAlgorithm.LEAKY_BUCKET)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, LeakyBucketRateLimiter)


# =============================================================================
# Key Extractor Tests
# =============================================================================


class TestKeyExtractors:
    """Tests for key extractors."""

    def test_default_key_extractor(self) -> None:
        """Test default key extractor."""
        extractor = DefaultKeyExtractor()

        def test_func():
            pass

        key = extractor.extract_key(test_func, (), {})
        assert "test_func" in key

    def test_argument_key_extractor_by_name(self) -> None:
        """Test argument key extractor with keyword argument."""
        extractor = ArgumentKeyExtractor(arg_name="user_id", prefix="user")

        def test_func(user_id: str):
            pass

        key = extractor.extract_key(test_func, (), {"user_id": "123"})
        assert key == "user:123"

    def test_argument_key_extractor_by_index(self) -> None:
        """Test argument key extractor with positional argument."""
        extractor = ArgumentKeyExtractor(arg_index=0, prefix="api")

        def test_func(endpoint: str):
            pass

        key = extractor.extract_key(test_func, ("/users",), {})
        assert key == "api:/users"

    def test_callable_key_extractor(self) -> None:
        """Test callable key extractor."""
        def custom_extractor(func, args, kwargs):
            return f"custom:{kwargs.get('key', 'default')}"

        extractor = CallableKeyExtractor(custom_extractor)

        def test_func(key: str):
            pass

        key = extractor.extract_key(test_func, (), {"key": "test"})
        assert key == "custom:test"


# =============================================================================
# Hook Tests
# =============================================================================


class TestRateLimitHooks:
    """Tests for rate limit hooks."""

    def test_metrics_hook(self) -> None:
        """Test metrics hook collects stats."""
        hook = MetricsRateLimitHook()

        hook.on_acquire("key1", 1, 9, {})
        hook.on_acquire("key1", 1, 8, {})
        hook.on_reject("key1", 1, 5.0, {})
        hook.on_wait("key1", 1, 2.0, {})

        assert hook.acquired_count == 2
        assert hook.rejected_count == 1
        assert hook.waited_count == 1
        assert hook.total_wait_time == 2.0

        stats = hook.get_key_stats("key1")
        assert stats["acquired"] == 2
        assert stats["rejected"] == 1
        assert stats["waited"] == 1

    def test_metrics_hook_reset(self) -> None:
        """Test metrics hook reset."""
        hook = MetricsRateLimitHook()

        hook.on_acquire("key1", 1, 9, {})
        hook.reset()

        assert hook.acquired_count == 0
        assert hook.rejected_count == 0

    def test_composite_hook(self) -> None:
        """Test composite hook calls all hooks."""
        mock_hook1 = MagicMock()
        mock_hook2 = MagicMock()

        composite = CompositeRateLimitHook([mock_hook1, mock_hook2])
        composite.on_acquire("key", 1, 9, {})

        mock_hook1.on_acquire.assert_called_once()
        mock_hook2.on_acquire.assert_called_once()

    def test_composite_hook_add_remove(self) -> None:
        """Test adding and removing hooks."""
        hook = MetricsRateLimitHook()
        composite = CompositeRateLimitHook([])

        composite.add_hook(hook)
        composite.on_acquire("key", 1, 9, {})
        assert hook.acquired_count == 1

        composite.remove_hook(hook)
        composite.on_acquire("key", 1, 8, {})
        assert hook.acquired_count == 1  # Not incremented


# =============================================================================
# Executor Tests
# =============================================================================


class TestRateLimitExecutor:
    """Tests for RateLimitExecutor."""

    def test_execute_within_limit(self) -> None:
        """Test execution within rate limit."""
        config = RateLimitConfig(max_requests=10, window_seconds=10.0)
        executor = RateLimitExecutor(config)

        def test_func():
            return "success"

        result = executor.execute(test_func)
        assert result == "success"

    def test_execute_exceeds_limit_reject(self) -> None:
        """Test execution exceeds limit with REJECT action."""
        config = RateLimitConfig(
            max_requests=2,
            window_seconds=10.0,
            on_limit=RateLimitAction.REJECT,
        )
        executor = RateLimitExecutor(config)

        def test_func():
            return "success"

        # First two should succeed
        executor.execute(test_func)
        executor.execute(test_func)

        # Third should raise
        with pytest.raises(RateLimitExceededError):
            executor.execute(test_func)

    def test_execute_exceeds_limit_wait(self) -> None:
        """Test execution exceeds limit with WAIT action."""
        config = RateLimitConfig(
            max_requests=2,
            window_seconds=0.2,
            on_limit=RateLimitAction.WAIT,
            max_wait_seconds=1.0,
        )
        executor = RateLimitExecutor(config)

        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        # Make rapid calls
        start = time.time()
        executor.execute(test_func)
        executor.execute(test_func)
        executor.execute(test_func)  # Should wait
        elapsed = time.time() - start

        assert call_count == 3
        assert elapsed >= 0.1  # Should have waited


# =============================================================================
# Registry Tests
# =============================================================================


class TestRateLimiterRegistry:
    """Tests for RateLimiterRegistry."""

    def test_get_or_create(self) -> None:
        """Test get_or_create returns same instance."""
        registry = RateLimiterRegistry()

        limiter1 = registry.get_or_create("test")
        limiter2 = registry.get_or_create("test")

        assert limiter1 is limiter2

    def test_get_nonexistent(self) -> None:
        """Test get returns None for nonexistent."""
        registry = RateLimiterRegistry()
        assert registry.get("nonexistent") is None

    def test_remove(self) -> None:
        """Test removing a limiter."""
        registry = RateLimiterRegistry()
        registry.get_or_create("test")

        assert registry.remove("test") is True
        assert registry.get("test") is None
        assert registry.remove("test") is False  # Already removed

    def test_reset_all(self) -> None:
        """Test resetting all limiters."""
        registry = RateLimiterRegistry()
        config = RateLimitConfig(max_requests=5, window_seconds=10.0)

        limiter = registry.get_or_create("test", config=config)
        for _ in range(3):
            limiter.acquire()

        registry.reset_all()
        assert limiter.get_remaining() == 5

    def test_get_all_stats(self) -> None:
        """Test getting stats for all limiters."""
        registry = RateLimiterRegistry()
        config = RateLimitConfig(max_requests=10, window_seconds=60.0)

        registry.get_or_create("api1", config=config)
        registry.get_or_create("api2", config=config)

        stats = registry.get_all_stats()
        assert "api1" in stats
        assert "api2" in stats
        assert stats["api1"]["max_requests"] == 10

    def test_names(self) -> None:
        """Test getting all limiter names."""
        registry = RateLimiterRegistry()
        registry.get_or_create("api1")
        registry.get_or_create("api2")

        names = registry.names
        assert "api1" in names
        assert "api2" in names


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_rate_limiter(self) -> None:
        """Test getting limiter from global registry."""
        limiter = get_rate_limiter("global_test")
        assert limiter is not None

        # Same instance returned
        limiter2 = get_rate_limiter("global_test")
        assert limiter is limiter2

    def test_get_rate_limiter_registry(self) -> None:
        """Test getting global registry."""
        registry = get_rate_limiter_registry()
        assert isinstance(registry, RateLimiterRegistry)


# =============================================================================
# Decorator Tests
# =============================================================================


class TestRateLimitDecorator:
    """Tests for rate_limit decorator."""

    def test_decorator_basic(self) -> None:
        """Test basic decorator usage."""
        @rate_limit(max_requests=5, window_seconds=10.0, use_registry=False)
        def test_func():
            return "success"

        for _ in range(5):
            assert test_func() == "success"

        with pytest.raises(RateLimitExceededError):
            test_func()

    def test_decorator_with_config(self) -> None:
        """Test decorator with config object."""
        config = RateLimitConfig(max_requests=3, window_seconds=10.0)

        @rate_limit(config=config, use_registry=False)
        def test_func():
            return "success"

        for _ in range(3):
            assert test_func() == "success"

        with pytest.raises(RateLimitExceededError):
            test_func()

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test decorator preserves function name and docstring."""
        @rate_limit(max_requests=10, window_seconds=10.0, use_registry=False)
        def documented_function():
            """This is a docstring."""
            return "success"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a docstring."


class TestAsyncRateLimit:
    """Tests for async rate limiting."""

    @asyncio_test
    @pytest.mark.asyncio
    async def test_async_decorator(self) -> None:
        """Test async decorator."""
        @rate_limit(max_requests=3, window_seconds=10.0, use_registry=False)
        async def async_func():
            return "async_success"

        for _ in range(3):
            result = await async_func()
            assert result == "async_success"

        with pytest.raises(RateLimitExceededError):
            await async_func()

    @asyncio_test
    @pytest.mark.asyncio
    async def test_rate_limit_call_async(self) -> None:
        """Test rate_limit_call_async function."""
        async def async_func():
            return "success"

        result = await rate_limit_call_async(
            async_func,
            name="async_test",
            config=RateLimitConfig(max_requests=10, window_seconds=10.0),
        )
        assert result == "success"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRateLimitIntegration:
    """Integration tests for rate limiting."""

    def test_rate_limit_call_function(self) -> None:
        """Test rate_limit_call helper function."""
        def test_func():
            return "success"

        result = rate_limit_call(
            test_func,
            name="call_test",
            config=RateLimitConfig(max_requests=10, window_seconds=10.0),
        )
        assert result == "success"

    def test_with_hooks(self) -> None:
        """Test rate limiting with hooks."""
        metrics_hook = MetricsRateLimitHook()
        config = RateLimitConfig(max_requests=3, window_seconds=10.0)

        @rate_limit(config=config, hooks=[metrics_hook], use_registry=False)
        def test_func():
            return "success"

        for _ in range(3):
            test_func()

        assert metrics_hook.acquired_count == 3

        with pytest.raises(RateLimitExceededError):
            test_func()

        assert metrics_hook.rejected_count == 1

    def test_with_custom_key_extractor(self) -> None:
        """Test rate limiting with custom key extractor."""
        config = RateLimitConfig(max_requests=2, window_seconds=10.0)
        extractor = ArgumentKeyExtractor(arg_name="user_id")

        @rate_limit(config=config, key_extractor=extractor, use_registry=False)
        def test_func(user_id: str):
            return f"success:{user_id}"

        # Each user has their own limit
        assert test_func(user_id="user1") == "success:user1"
        assert test_func(user_id="user1") == "success:user1"
        assert test_func(user_id="user2") == "success:user2"  # Different key

        with pytest.raises(RateLimitExceededError):
            test_func(user_id="user1")  # user1 exhausted

        assert test_func(user_id="user2") == "success:user2"  # user2 still has quota


class TestConcurrency:
    """Tests for concurrent rate limiting."""

    def test_thread_safety(self) -> None:
        """Test thread safety of rate limiter."""
        import concurrent.futures

        config = RateLimitConfig(max_requests=100, window_seconds=10.0)
        limiter = TokenBucketRateLimiter(config)

        def acquire_token():
            return limiter.acquire("concurrent")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(acquire_token) for _ in range(100)]
            results = [f.result() for f in futures]

        # All 100 should succeed
        assert sum(results) == 100
        assert limiter.get_remaining("concurrent") == 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_acquire_multiple_tokens(self) -> None:
        """Test acquiring multiple tokens at once."""
        config = RateLimitConfig(max_requests=10, window_seconds=10.0)
        limiter = TokenBucketRateLimiter(config)

        assert limiter.acquire("test", tokens=5) is True
        assert limiter.get_remaining("test") == 5

        assert limiter.acquire("test", tokens=5) is True
        assert limiter.get_remaining("test") == 0

        assert limiter.acquire("test", tokens=1) is False

    def test_very_short_window(self) -> None:
        """Test with very short time window."""
        config = RateLimitConfig(max_requests=100, window_seconds=0.01)
        limiter = TokenBucketRateLimiter(config)

        # Should refill quickly
        limiter.acquire("test", tokens=50)
        time.sleep(0.02)
        assert limiter.get_remaining("test") >= 50

    def test_zero_remaining(self) -> None:
        """Test handling of zero remaining tokens."""
        config = RateLimitConfig(max_requests=1, window_seconds=10.0)
        limiter = TokenBucketRateLimiter(config)

        assert limiter.acquire("test") is True
        assert limiter.get_remaining("test") == 0
        assert limiter.acquire("test") is False
