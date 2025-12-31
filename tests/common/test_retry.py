"""Tests for common.retry module."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from common.retry import (
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
    DEFAULT_RETRY_CONFIG,
    NO_DELAY_RETRY_CONFIG,
    CallableExceptionFilter,
    CompositeExceptionFilter,
    CompositeRetryHook,
    ExponentialDelayCalculator,
    FibonacciDelayCalculator,
    FixedDelayCalculator,
    LinearDelayCalculator,
    LoggingRetryHook,
    MetricsRetryHook,
    NonRetryableError,
    RetryConfig,
    RetryError,
    RetryExecutor,
    RetryExhaustedError,
    RetryStrategy,
    TypeBasedExceptionFilter,
    retry,
    retry_call,
    retry_call_async,
)


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all expected strategies exist."""
        assert RetryStrategy.FIXED
        assert RetryStrategy.EXPONENTIAL
        assert RetryStrategy.LINEAR
        assert RetryStrategy.FIBONACCI


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.jitter is True
        assert config.exceptions == (Exception,)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay_seconds=2.0,
            max_delay_seconds=120.0,
            strategy=RetryStrategy.LINEAR,
            jitter=False,
            exceptions=(ValueError, ConnectionError),
        )
        assert config.max_attempts == 5
        assert config.base_delay_seconds == 2.0
        assert config.max_delay_seconds == 120.0
        assert config.strategy == RetryStrategy.LINEAR
        assert config.jitter is False
        assert config.exceptions == (ValueError, ConnectionError)

    def test_validation_max_attempts(self):
        """Test validation of max_attempts."""
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            RetryConfig(max_attempts=0)

    def test_validation_base_delay(self):
        """Test validation of base_delay_seconds."""
        with pytest.raises(ValueError, match="base_delay_seconds must be non-negative"):
            RetryConfig(base_delay_seconds=-1.0)

    def test_validation_max_delay(self):
        """Test validation of max_delay_seconds."""
        with pytest.raises(ValueError, match="max_delay_seconds must be >= base_delay_seconds"):
            RetryConfig(base_delay_seconds=10.0, max_delay_seconds=5.0)

    def test_validation_jitter_factor(self):
        """Test validation of jitter_factor."""
        with pytest.raises(ValueError, match=r"jitter_factor must be between 0\.0 and 1\.0"):
            RetryConfig(jitter_factor=1.5)

    def test_with_max_attempts(self):
        """Test with_max_attempts builder method."""
        config = RetryConfig(max_attempts=3)
        new_config = config.with_max_attempts(5)
        assert new_config.max_attempts == 5
        assert config.max_attempts == 3  # Original unchanged

    def test_with_delays(self):
        """Test with_delays builder method."""
        config = RetryConfig()
        new_config = config.with_delays(base_delay_seconds=2.0, max_delay_seconds=120.0)
        assert new_config.base_delay_seconds == 2.0
        assert new_config.max_delay_seconds == 120.0

    def test_with_strategy(self):
        """Test with_strategy builder method."""
        config = RetryConfig()
        new_config = config.with_strategy(RetryStrategy.FIXED)
        assert new_config.strategy == RetryStrategy.FIXED

    def test_with_exceptions(self):
        """Test with_exceptions builder method."""
        config = RetryConfig()
        new_config = config.with_exceptions(
            exceptions=(ValueError,),
            non_retryable=(KeyError,),
        )
        assert new_config.exceptions == (ValueError,)
        assert new_config.non_retryable_exceptions == (KeyError,)

    def test_with_jitter(self):
        """Test with_jitter builder method."""
        config = RetryConfig()
        new_config = config.with_jitter(enabled=False, factor=0.5)
        assert new_config.jitter is False
        assert new_config.jitter_factor == 0.5

    def test_to_dict(self):
        """Test to_dict method."""
        config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.LINEAR,
            exceptions=(ValueError,),
        )
        result = config.to_dict()
        assert result["max_attempts"] == 5
        assert result["strategy"] == "LINEAR"
        assert "ValueError" in result["exceptions"]

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "max_attempts": 5,
            "base_delay_seconds": 2.0,
            "strategy": "LINEAR",
        }
        config = RetryConfig.from_dict(data)
        assert config.max_attempts == 5
        assert config.base_delay_seconds == 2.0
        assert config.strategy == RetryStrategy.LINEAR

    def test_immutability(self):
        """Test that config is immutable."""
        config = RetryConfig()
        with pytest.raises(AttributeError):
            config.max_attempts = 10  # type: ignore


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_config(self):
        """Test DEFAULT_RETRY_CONFIG."""
        assert DEFAULT_RETRY_CONFIG.max_attempts == 3
        assert DEFAULT_RETRY_CONFIG.strategy == RetryStrategy.EXPONENTIAL

    def test_aggressive_config(self):
        """Test AGGRESSIVE_RETRY_CONFIG."""
        assert AGGRESSIVE_RETRY_CONFIG.max_attempts == 10
        assert AGGRESSIVE_RETRY_CONFIG.base_delay_seconds == 0.5

    def test_conservative_config(self):
        """Test CONSERVATIVE_RETRY_CONFIG."""
        assert CONSERVATIVE_RETRY_CONFIG.max_attempts == 3
        assert CONSERVATIVE_RETRY_CONFIG.base_delay_seconds == 5.0

    def test_no_delay_config(self):
        """Test NO_DELAY_RETRY_CONFIG."""
        assert NO_DELAY_RETRY_CONFIG.base_delay_seconds == 0.0
        assert NO_DELAY_RETRY_CONFIG.jitter is False


class TestDelayCalculators:
    """Tests for delay calculators."""

    def test_fixed_delay(self):
        """Test FixedDelayCalculator."""
        calc = FixedDelayCalculator()
        assert calc.calculate_delay(1, 1.0, 60.0) == 1.0
        assert calc.calculate_delay(5, 1.0, 60.0) == 1.0
        assert calc.calculate_delay(10, 1.0, 60.0) == 1.0

    def test_fixed_delay_respects_max(self):
        """Test FixedDelayCalculator respects max_delay."""
        calc = FixedDelayCalculator()
        assert calc.calculate_delay(1, 10.0, 5.0) == 5.0

    def test_exponential_delay(self):
        """Test ExponentialDelayCalculator."""
        calc = ExponentialDelayCalculator(base=2.0)
        assert calc.calculate_delay(1, 1.0, 60.0) == 1.0  # 1 * 2^0 = 1
        assert calc.calculate_delay(2, 1.0, 60.0) == 2.0  # 1 * 2^1 = 2
        assert calc.calculate_delay(3, 1.0, 60.0) == 4.0  # 1 * 2^2 = 4
        assert calc.calculate_delay(4, 1.0, 60.0) == 8.0  # 1 * 2^3 = 8

    def test_exponential_delay_capped(self):
        """Test ExponentialDelayCalculator caps at max_delay."""
        calc = ExponentialDelayCalculator(base=2.0)
        assert calc.calculate_delay(10, 1.0, 30.0) == 30.0

    def test_linear_delay(self):
        """Test LinearDelayCalculator."""
        calc = LinearDelayCalculator(increment=2.0)
        assert calc.calculate_delay(1, 1.0, 60.0) == 1.0  # 1 + 0*2 = 1
        assert calc.calculate_delay(2, 1.0, 60.0) == 3.0  # 1 + 1*2 = 3
        assert calc.calculate_delay(3, 1.0, 60.0) == 5.0  # 1 + 2*2 = 5

    def test_fibonacci_delay(self):
        """Test FibonacciDelayCalculator."""
        calc = FibonacciDelayCalculator()
        # Fibonacci: 1, 1, 2, 3, 5, 8...
        assert calc.calculate_delay(1, 1.0, 60.0) == 1.0
        assert calc.calculate_delay(2, 1.0, 60.0) == 1.0
        assert calc.calculate_delay(3, 1.0, 60.0) == 2.0
        assert calc.calculate_delay(4, 1.0, 60.0) == 3.0
        assert calc.calculate_delay(5, 1.0, 60.0) == 5.0


class TestExceptionFilters:
    """Tests for exception filters."""

    def test_type_based_filter_retryable(self):
        """Test TypeBasedExceptionFilter with retryable exceptions."""
        filter = TypeBasedExceptionFilter(retryable=(ValueError, TypeError))
        assert filter.should_retry(ValueError("test"), 1) is True
        assert filter.should_retry(TypeError("test"), 1) is True
        assert filter.should_retry(KeyError("test"), 1) is False

    def test_type_based_filter_non_retryable(self):
        """Test TypeBasedExceptionFilter with non-retryable exceptions."""
        filter = TypeBasedExceptionFilter(
            retryable=(Exception,),
            non_retryable=(KeyError,),
        )
        assert filter.should_retry(ValueError("test"), 1) is True
        assert filter.should_retry(KeyError("test"), 1) is False

    def test_callable_filter(self):
        """Test CallableExceptionFilter."""

        def predicate(exc: Exception, attempt: int) -> bool:
            return attempt < 3

        filter = CallableExceptionFilter(predicate)
        assert filter.should_retry(ValueError(), 1) is True
        assert filter.should_retry(ValueError(), 2) is True
        assert filter.should_retry(ValueError(), 3) is False

    def test_composite_filter_any(self):
        """Test CompositeExceptionFilter with any mode."""
        filter1 = TypeBasedExceptionFilter(retryable=(ValueError,))
        filter2 = TypeBasedExceptionFilter(retryable=(TypeError,))
        composite = CompositeExceptionFilter([filter1, filter2], require_all=False)

        assert composite.should_retry(ValueError(), 1) is True
        assert composite.should_retry(TypeError(), 1) is True
        assert composite.should_retry(KeyError(), 1) is False

    def test_composite_filter_all(self):
        """Test CompositeExceptionFilter with all mode."""

        def attempt_filter(exc: Exception, attempt: int) -> bool:
            return attempt < 3

        filter1 = TypeBasedExceptionFilter(retryable=(Exception,))
        filter2 = CallableExceptionFilter(attempt_filter)
        composite = CompositeExceptionFilter([filter1, filter2], require_all=True)

        assert composite.should_retry(ValueError(), 1) is True
        assert composite.should_retry(ValueError(), 3) is False


class TestRetryHooks:
    """Tests for retry hooks."""

    def test_metrics_hook(self):
        """Test MetricsRetryHook tracks metrics."""
        hook = MetricsRetryHook()

        hook.on_retry(1, ValueError("test"), 1.0, {"function": "test_func"})
        hook.on_retry(2, ValueError("test"), 2.0, {"function": "test_func"})
        hook.on_success(3, "result", {"function": "test_func"})

        assert hook.total_retries == 2
        assert hook.successful_retries == 1
        assert hook.retry_counts_by_function["test_func"] == 2

    def test_metrics_hook_failure(self):
        """Test MetricsRetryHook tracks failures."""
        hook = MetricsRetryHook()

        hook.on_failure(3, (ValueError("test"),), {"function": "test_func"})

        assert hook.failed_operations == 1

    def test_metrics_hook_reset(self):
        """Test MetricsRetryHook reset."""
        hook = MetricsRetryHook()
        hook.on_retry(1, ValueError("test"), 1.0, {"function": "test_func"})
        hook.reset()

        assert hook.total_retries == 0
        assert hook.successful_retries == 0
        assert hook.failed_operations == 0
        assert hook.retry_counts_by_function == {}

    def test_composite_hook(self):
        """Test CompositeRetryHook calls all hooks."""
        hook1 = MagicMock()
        hook2 = MagicMock()
        composite = CompositeRetryHook([hook1, hook2])

        composite.on_retry(1, ValueError("test"), 1.0, {})
        hook1.on_retry.assert_called_once()
        hook2.on_retry.assert_called_once()

        composite.on_success(1, "result", {})
        hook1.on_success.assert_called_once()
        hook2.on_success.assert_called_once()

        composite.on_failure(1, (), {})
        hook1.on_failure.assert_called_once()
        hook2.on_failure.assert_called_once()

    def test_composite_hook_add_remove(self):
        """Test CompositeRetryHook add/remove."""
        hook1 = MagicMock()
        hook2 = MagicMock()
        composite = CompositeRetryHook([hook1])

        composite.add_hook(hook2)
        composite.on_retry(1, ValueError("test"), 1.0, {})
        assert hook2.on_retry.called

        composite.remove_hook(hook2)
        hook2.reset_mock()
        composite.on_retry(2, ValueError("test"), 1.0, {})
        assert not hook2.on_retry.called

    def test_composite_hook_exception_resilience(self):
        """Test CompositeRetryHook handles hook exceptions."""
        failing_hook = MagicMock()
        failing_hook.on_retry.side_effect = RuntimeError("Hook failed")
        working_hook = MagicMock()
        composite = CompositeRetryHook([failing_hook, working_hook])

        # Should not raise, and second hook should still be called
        composite.on_retry(1, ValueError("test"), 1.0, {})
        assert working_hook.on_retry.called


class TestRetryExceptions:
    """Tests for retry exceptions."""

    def test_retry_error(self):
        """Test RetryError."""
        exc = RetryError("Test error", attempts=3)
        assert exc.attempts == 3
        assert "attempts" in exc.details

    def test_retry_exhausted_error(self):
        """Test RetryExhaustedError."""
        original = ValueError("Original error")
        exc = RetryExhaustedError(
            "Retry exhausted",
            attempts=3,
            last_exception=original,
            exceptions=(original,),
        )
        assert exc.attempts == 3
        assert exc.last_exception is original
        assert exc.exceptions == (original,)
        assert exc.cause is original

    def test_non_retryable_error(self):
        """Test NonRetryableError."""
        original = ValueError("Non-retryable")
        exc = NonRetryableError(
            "Non-retryable exception",
            attempts=1,
            last_exception=original,
        )
        assert exc.last_exception is original


class TestRetryExecutor:
    """Tests for RetryExecutor."""

    def test_success_on_first_attempt(self):
        """Test success on first attempt."""
        config = NO_DELAY_RETRY_CONFIG
        executor = RetryExecutor(config)

        def success_func() -> str:
            return "success"

        result = executor.execute(success_func)
        assert result == "success"

    def test_success_after_retry(self):
        """Test success after retrying."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        executor = RetryExecutor(config)

        attempts = 0

        def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Retry")
            return "success"

        result = executor.execute(failing_then_success)
        assert result == "success"
        assert attempts == 3

    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        executor = RetryExecutor(config)

        def always_fails() -> None:
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            executor.execute(always_fails)

        assert exc_info.value.attempts == 3
        assert len(exc_info.value.exceptions) == 3

    def test_non_retryable_exception(self):
        """Test non-retryable exception stops retry."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.0,
            jitter=False,
            exceptions=(ValueError,),
            non_retryable_exceptions=(KeyError,),
        )
        executor = RetryExecutor(config)

        def raises_keyerror() -> None:
            raise KeyError("Non-retryable")

        with pytest.raises(NonRetryableError) as exc_info:
            executor.execute(raises_keyerror)

        assert exc_info.value.attempts == 1  # Only one attempt

    def test_hooks_called(self):
        """Test hooks are called during retry."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        hook = MetricsRetryHook()
        executor = RetryExecutor(config, hooks=[hook])

        attempts = 0

        def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry")
            return "success"

        executor.execute(failing_then_success)

        assert hook.total_retries == 1
        assert hook.successful_retries == 1

    def test_delay_applied(self):
        """Test delay is applied between retries."""
        config = RetryConfig(
            max_attempts=2,
            base_delay_seconds=0.1,
            max_delay_seconds=0.1,
            jitter=False,
            strategy=RetryStrategy.FIXED,
        )
        executor = RetryExecutor(config)

        attempts = 0

        def failing_once() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry")
            return "success"

        start = time.monotonic()
        executor.execute(failing_once)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.1

    def test_retry_on_result(self):
        """Test retry on specific result."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=0.0,
            jitter=False,
            retry_on_result=lambda r: r is None,
        )
        executor = RetryExecutor(config)

        attempts = 0

        def returns_none_then_value() -> str | None:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return None
            return "value"

        result = executor.execute(returns_none_then_value)
        assert result == "value"
        assert attempts == 3


class TestRetryExecutorAsync:
    """Tests for RetryExecutor async functionality."""

    def test_async_success_on_first_attempt(self):
        """Test async success on first attempt."""
        config = NO_DELAY_RETRY_CONFIG
        executor = RetryExecutor(config)

        async def success_func() -> str:
            return "success"

        result = asyncio.get_event_loop().run_until_complete(
            executor.execute_async(success_func)
        )
        assert result == "success"

    def test_async_success_after_retry(self):
        """Test async success after retrying."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        executor = RetryExecutor(config)

        attempts = 0

        async def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Retry")
            return "success"

        result = asyncio.get_event_loop().run_until_complete(
            executor.execute_async(failing_then_success)
        )
        assert result == "success"
        assert attempts == 3

    def test_async_retry_exhausted(self):
        """Test async retry exhaustion."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        executor = RetryExecutor(config)

        async def always_fails() -> None:
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                executor.execute_async(always_fails)
            )

        assert exc_info.value.attempts == 3

    def test_async_delay_applied(self):
        """Test async delay uses asyncio.sleep."""
        config = RetryConfig(
            max_attempts=2,
            base_delay_seconds=0.05,
            max_delay_seconds=0.05,
            jitter=False,
            strategy=RetryStrategy.FIXED,
        )
        executor = RetryExecutor(config)

        attempts = 0

        async def failing_once() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry")
            return "success"

        start = time.monotonic()
        asyncio.get_event_loop().run_until_complete(
            executor.execute_async(failing_once)
        )
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05


class TestRetryDecorator:
    """Tests for retry decorator."""

    def test_decorator_basic(self):
        """Test basic decorator usage."""
        attempts = 0

        @retry(max_attempts=3, base_delay_seconds=0.0, jitter=False)
        def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Retry")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert attempts == 3

    def test_decorator_with_config(self):
        """Test decorator with RetryConfig."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        attempts = 0

        @retry(config=config)
        def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry")
            return "success"

        result = failing_then_success()
        assert result == "success"

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @retry(max_attempts=3, base_delay_seconds=0.0)
        def documented_func() -> str:
            """This is a docstring."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    def test_decorator_with_arguments(self):
        """Test decorator with function arguments."""

        @retry(max_attempts=2, base_delay_seconds=0.0, jitter=False)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_decorator_with_exceptions(self):
        """Test decorator with specific exceptions."""
        attempts = 0

        @retry(
            max_attempts=3,
            base_delay_seconds=0.0,
            jitter=False,
            exceptions=(ValueError,),
        )
        def raises_valueerror() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Retry")
            return "success"

        result = raises_valueerror()
        assert result == "success"

    def test_decorator_non_retryable(self):
        """Test decorator with non-retryable exception."""

        @retry(
            max_attempts=3,
            base_delay_seconds=0.0,
            jitter=False,
            exceptions=(ValueError,),
            non_retryable=(KeyError,),
        )
        def raises_keyerror() -> None:
            raise KeyError("Non-retryable")

        with pytest.raises(NonRetryableError):
            raises_keyerror()


class TestRetryDecoratorAsync:
    """Tests for retry decorator with async functions."""

    def test_decorator_async_basic(self):
        """Test async decorator usage."""
        attempts = 0

        @retry(max_attempts=3, base_delay_seconds=0.0, jitter=False)
        async def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Retry")
            return "success"

        result = asyncio.get_event_loop().run_until_complete(failing_then_success())
        assert result == "success"
        assert attempts == 3

    def test_decorator_async_with_config(self):
        """Test async decorator with RetryConfig."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        attempts = 0

        @retry(config=config)
        async def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry")
            return "success"

        result = asyncio.get_event_loop().run_until_complete(failing_then_success())
        assert result == "success"


class TestRetryCallFunctions:
    """Tests for retry_call and retry_call_async."""

    def test_retry_call_basic(self):
        """Test retry_call function."""
        attempts = 0

        def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry")
            return "success"

        result = retry_call(
            failing_then_success,
            config=NO_DELAY_RETRY_CONFIG.with_max_attempts(3),
        )
        assert result == "success"

    def test_retry_call_with_args(self):
        """Test retry_call with function arguments."""

        def add(a: int, b: int) -> int:
            return a + b

        result = retry_call(add, 1, 2, config=NO_DELAY_RETRY_CONFIG)
        assert result == 3

    def test_retry_call_async_basic(self):
        """Test retry_call_async function."""
        attempts = 0

        async def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Retry")
            return "success"

        result = asyncio.get_event_loop().run_until_complete(
            retry_call_async(
                failing_then_success,
                config=NO_DELAY_RETRY_CONFIG.with_max_attempts(3),
            )
        )
        assert result == "success"


class TestLoggingRetryHook:
    """Tests for LoggingRetryHook."""

    def test_logging_hook_on_retry(self):
        """Test LoggingRetryHook logs retries."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingRetryHook()
            hook.on_retry(1, ValueError("test"), 1.0, {"function": "test"})

            mock_logger.warning.assert_called_once()

    def test_logging_hook_on_success(self):
        """Test LoggingRetryHook logs success after retry."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingRetryHook()
            hook.on_success(2, "result", {"function": "test"})  # attempt > 1

            mock_logger.info.assert_called_once()

    def test_logging_hook_on_success_first_attempt(self):
        """Test LoggingRetryHook doesn't log success on first attempt."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingRetryHook()
            hook.on_success(1, "result", {"function": "test"})  # first attempt

            mock_logger.info.assert_not_called()

    def test_logging_hook_on_failure(self):
        """Test LoggingRetryHook logs failures."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingRetryHook()
            hook.on_failure(3, (ValueError("test"),), {"function": "test"})

            mock_logger.error.assert_called_once()


class TestJitter:
    """Tests for jitter behavior."""

    def test_jitter_adds_randomness(self):
        """Test jitter adds randomness to delays."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
            jitter=True,
            jitter_factor=0.5,
            strategy=RetryStrategy.FIXED,
        )
        executor = RetryExecutor(config)

        # Calculate multiple delays and check they're not all the same
        delays = set()
        for _ in range(10):
            delay = executor._calculate_delay(1)
            delays.add(round(delay, 4))  # Round to avoid floating point issues

        # With jitter, we should see variation
        assert len(delays) > 1

    def test_no_jitter(self):
        """Test no jitter produces consistent delays."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
            jitter=False,
            strategy=RetryStrategy.FIXED,
        )
        executor = RetryExecutor(config)

        delays = set()
        for _ in range(10):
            delay = executor._calculate_delay(1)
            delays.add(delay)

        # Without jitter, all delays should be the same
        assert len(delays) == 1
        assert 1.0 in delays


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_attempt(self):
        """Test with single attempt (no retries)."""
        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(1)

        @retry(config=config)
        def always_fails() -> None:
            raise ValueError("Fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            always_fails()

        assert exc_info.value.attempts == 1

    def test_zero_delay(self):
        """Test with zero delay."""
        config = NO_DELAY_RETRY_CONFIG
        executor = RetryExecutor(config)

        delay = executor._calculate_delay(5)
        assert delay == 0.0

    def test_custom_delay_calculator(self):
        """Test with custom delay calculator."""

        class CustomCalculator:
            def calculate_delay(self, attempt: int, base: float, max_d: float) -> float:
                return 0.001 * attempt

        config = RetryConfig(max_attempts=3, base_delay_seconds=0.0)
        executor = RetryExecutor(config, delay_calculator=CustomCalculator())

        attempts = 0

        def failing_then_success() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Retry")
            return "success"

        result = executor.execute(failing_then_success)
        assert result == "success"

    def test_custom_exception_filter(self):
        """Test with custom exception filter."""

        class CustomFilter:
            def should_retry(self, exc: Exception, attempt: int) -> bool:
                return "retry" in str(exc).lower()

        config = NO_DELAY_RETRY_CONFIG.with_max_attempts(3)
        executor = RetryExecutor(config, exception_filter=CustomFilter())

        attempts = 0

        def sometimes_retryable() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("Please retry this")
            if attempts == 2:
                raise ValueError("Do not continue")
            return "success"

        # First raises retryable, second raises non-retryable
        with pytest.raises(NonRetryableError):
            executor.execute(sometimes_retryable)
