"""Retry utilities for Truthound Integrations.

This module provides a flexible, extensible retry system designed for
resilient operation in distributed environments. It supports:

- Configurable retry strategies (exponential backoff, fixed delay, etc.)
- Customizable exception filtering
- Pre/post retry hooks for logging and monitoring
- Async and sync function support
- Jitter for avoiding thundering herd problems

Design Principles:
    1. Protocol-based: Easy to extend with custom retry strategies
    2. Composable: Combine retry with other decorators
    3. Observable: Hook system for monitoring retry behavior
    4. Safe by default: Sensible defaults that prevent infinite loops

Example:
    >>> from common.retry import retry, RetryConfig
    >>> @retry(max_attempts=3, exceptions=(ConnectionError,))
    ... def fetch_data():
    ...     return api.get("/data")

    >>> # With custom configuration
    >>> config = RetryConfig(
    ...     max_attempts=5,
    ...     base_delay_seconds=1.0,
    ...     max_delay_seconds=60.0,
    ...     exponential_base=2.0,
    ... )
    >>> @retry(config=config)
    ... async def async_fetch():
    ...     return await api.async_get("/data")
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class RetryError(TruthoundIntegrationError):
    """Exception raised when all retry attempts are exhausted.

    Attributes:
        attempts: Number of attempts made.
        last_exception: The last exception that caused retry failure.
        exceptions: All exceptions encountered during retry attempts.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        last_exception: Exception | None = None,
        exceptions: tuple[Exception, ...] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize retry error.

        Args:
            message: Human-readable error description.
            attempts: Number of attempts made.
            last_exception: The last exception that caused retry failure.
            exceptions: All exceptions encountered during retry attempts.
            details: Optional dictionary with additional error context.
        """
        details = details or {}
        details["attempts"] = attempts
        if last_exception:
            details["last_exception_type"] = type(last_exception).__name__
        super().__init__(message, details=details, cause=last_exception)
        self.attempts = attempts
        self.last_exception = last_exception
        self.exceptions = exceptions


class RetryExhaustedError(RetryError):
    """Exception raised when maximum retry attempts are exhausted."""

    pass


class NonRetryableError(RetryError):
    """Exception raised when an error is not retryable.

    Use this to wrap exceptions that should not trigger retries.
    """

    pass


# =============================================================================
# Enums
# =============================================================================


class RetryStrategy(Enum):
    """Built-in retry delay strategies.

    Attributes:
        FIXED: Use a fixed delay between retries.
        EXPONENTIAL: Exponentially increase delay between retries.
        LINEAR: Linearly increase delay between retries.
        FIBONACCI: Use Fibonacci sequence for delays.
    """

    FIXED = auto()
    EXPONENTIAL = auto()
    LINEAR = auto()
    FIBONACCI = auto()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class DelayCalculator(Protocol):
    """Protocol for calculating retry delays.

    Implement this protocol to create custom delay strategies.
    """

    @abstractmethod
    def calculate_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        """Calculate delay for the given attempt.

        Args:
            attempt: Current attempt number (1-indexed).
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.

        Returns:
            Delay in seconds.
        """
        ...


@runtime_checkable
class RetryHook(Protocol):
    """Protocol for retry event hooks.

    Implement this to receive notifications about retry events.
    """

    @abstractmethod
    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
        context: dict[str, Any],
    ) -> None:
        """Called before a retry attempt.

        Args:
            attempt: Current attempt number.
            exception: Exception that triggered retry.
            delay: Delay before next attempt.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_success(
        self,
        attempt: int,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Called on successful execution.

        Args:
            attempt: Attempt number that succeeded.
            result: Return value from the function.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_failure(
        self,
        attempts: int,
        exceptions: tuple[Exception, ...],
        context: dict[str, Any],
    ) -> None:
        """Called when all retry attempts are exhausted.

        Args:
            attempts: Total number of attempts made.
            exceptions: All exceptions encountered.
            context: Additional context information.
        """
        ...


@runtime_checkable
class ExceptionFilter(Protocol):
    """Protocol for determining if an exception should trigger retry.

    Implement this for complex retry conditions.
    """

    @abstractmethod
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the exception should trigger a retry.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if retry should be attempted.
        """
        ...


# =============================================================================
# Delay Calculators
# =============================================================================


class FixedDelayCalculator:
    """Fixed delay between retries."""

    def calculate_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        """Return fixed base delay.

        Args:
            attempt: Current attempt number (unused).
            base_delay: Base delay to return.
            max_delay: Maximum delay (unused for fixed).

        Returns:
            Base delay in seconds.
        """
        return min(base_delay, max_delay)


class ExponentialDelayCalculator:
    """Exponential backoff delay calculator."""

    def __init__(self, base: float = 2.0) -> None:
        """Initialize calculator.

        Args:
            base: Exponential base (default: 2.0 for doubling).
        """
        self.base = base

    def calculate_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        """Calculate exponential delay.

        Args:
            attempt: Current attempt number.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.

        Returns:
            Delay in seconds (capped at max_delay).
        """
        delay = base_delay * (self.base ** (attempt - 1))
        return min(delay, max_delay)


class LinearDelayCalculator:
    """Linear increase delay calculator."""

    def __init__(self, increment: float = 1.0) -> None:
        """Initialize calculator.

        Args:
            increment: Delay increment per attempt in seconds.
        """
        self.increment = increment

    def calculate_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        """Calculate linear delay.

        Args:
            attempt: Current attempt number.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.

        Returns:
            Delay in seconds (capped at max_delay).
        """
        delay = base_delay + (self.increment * (attempt - 1))
        return min(delay, max_delay)


class FibonacciDelayCalculator:
    """Fibonacci sequence delay calculator."""

    def calculate_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        """Calculate Fibonacci-based delay.

        Args:
            attempt: Current attempt number.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.

        Returns:
            Delay in seconds (capped at max_delay).
        """
        # Calculate Fibonacci number for attempt
        a, b = 0, 1
        for _ in range(attempt):
            a, b = b, a + b
        delay = base_delay * a
        return min(delay, max_delay)


# =============================================================================
# Exception Filters
# =============================================================================


class TypeBasedExceptionFilter:
    """Filter exceptions by type."""

    def __init__(
        self,
        retryable: tuple[type[Exception], ...] = (Exception,),
        non_retryable: tuple[type[Exception], ...] = (),
    ) -> None:
        """Initialize filter.

        Args:
            retryable: Exception types that should trigger retry.
            non_retryable: Exception types that should not trigger retry.
        """
        self.retryable = retryable
        self.non_retryable = non_retryable

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if exception type is retryable.

        Args:
            exception: The exception to check.
            attempt: Current attempt number (unused).

        Returns:
            True if exception should trigger retry.
        """
        # Non-retryable takes precedence
        if isinstance(exception, self.non_retryable):
            return False
        return isinstance(exception, self.retryable)


class CallableExceptionFilter:
    """Filter exceptions using a callable."""

    def __init__(
        self,
        predicate: Callable[[Exception, int], bool],
    ) -> None:
        """Initialize filter.

        Args:
            predicate: Function that returns True if retry should occur.
        """
        self.predicate = predicate

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check using predicate function.

        Args:
            exception: The exception to check.
            attempt: Current attempt number.

        Returns:
            Result of predicate function.
        """
        return self.predicate(exception, attempt)


class CompositeExceptionFilter:
    """Combine multiple exception filters."""

    def __init__(
        self,
        filters: Sequence[ExceptionFilter],
        require_all: bool = False,
    ) -> None:
        """Initialize composite filter.

        Args:
            filters: List of filters to combine.
            require_all: If True, all filters must agree. If False, any filter.
        """
        self.filters = filters
        self.require_all = require_all

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check all filters.

        Args:
            exception: The exception to check.
            attempt: Current attempt number.

        Returns:
            Combined result from all filters.
        """
        if self.require_all:
            return all(f.should_retry(exception, attempt) for f in self.filters)
        return any(f.should_retry(exception, attempt) for f in self.filters)


# =============================================================================
# Retry Hooks
# =============================================================================


class LoggingRetryHook:
    """Hook that logs retry events.

    Uses the Truthound logging system for structured logging.
    """

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.retry).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.retry")

    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
        context: dict[str, Any],
    ) -> None:
        """Log retry attempt.

        Args:
            attempt: Current attempt number.
            exception: Exception that triggered retry.
            delay: Delay before next attempt.
            context: Additional context.
        """
        self._logger.warning(
            "Retry attempt",
            attempt=attempt,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            delay_seconds=delay,
            **context,
        )

    def on_success(
        self,
        attempt: int,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Log successful execution.

        Args:
            attempt: Attempt number that succeeded.
            result: Return value (not logged for security).
            context: Additional context.
        """
        if attempt > 1:
            self._logger.info(
                "Retry succeeded",
                attempt=attempt,
                **context,
            )

    def on_failure(
        self,
        attempts: int,
        exceptions: tuple[Exception, ...],
        context: dict[str, Any],
    ) -> None:
        """Log retry exhaustion.

        Args:
            attempts: Total attempts made.
            exceptions: All exceptions encountered.
            context: Additional context.
        """
        last_exc = exceptions[-1] if exceptions else None
        self._logger.error(
            "Retry exhausted",
            attempts=attempts,
            last_exception_type=type(last_exc).__name__ if last_exc else None,
            last_exception_message=str(last_exc) if last_exc else None,
            **context,
        )


class MetricsRetryHook:
    """Hook that collects retry metrics.

    Useful for monitoring and alerting on retry patterns.
    """

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._total_retries: int = 0
        self._successful_retries: int = 0
        self._failed_operations: int = 0
        self._retry_counts: dict[str, int] = {}

    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
        context: dict[str, Any],
    ) -> None:
        """Record retry attempt.

        Args:
            attempt: Current attempt number.
            exception: Exception that triggered retry.
            delay: Delay before next attempt.
            context: Additional context.
        """
        self._total_retries += 1
        func_name = context.get("function", "unknown")
        self._retry_counts[func_name] = self._retry_counts.get(func_name, 0) + 1

    def on_success(
        self,
        attempt: int,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Record successful retry.

        Args:
            attempt: Attempt number that succeeded.
            result: Return value.
            context: Additional context.
        """
        if attempt > 1:
            self._successful_retries += 1

    def on_failure(
        self,
        attempts: int,
        exceptions: tuple[Exception, ...],
        context: dict[str, Any],
    ) -> None:
        """Record failed operation.

        Args:
            attempts: Total attempts made.
            exceptions: All exceptions encountered.
            context: Additional context.
        """
        self._failed_operations += 1

    @property
    def total_retries(self) -> int:
        """Get total retry count."""
        return self._total_retries

    @property
    def successful_retries(self) -> int:
        """Get successful retry count."""
        return self._successful_retries

    @property
    def failed_operations(self) -> int:
        """Get failed operation count."""
        return self._failed_operations

    @property
    def retry_counts_by_function(self) -> dict[str, int]:
        """Get retry counts by function name."""
        return dict(self._retry_counts)

    def reset(self) -> None:
        """Reset all metrics."""
        self._total_retries = 0
        self._successful_retries = 0
        self._failed_operations = 0
        self._retry_counts.clear()


class CompositeRetryHook:
    """Combine multiple retry hooks."""

    def __init__(self, hooks: Sequence[RetryHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: RetryHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: RetryHook) -> None:
        """Remove a hook.

        Args:
            hook: Hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_retry on all hooks.

        Args:
            attempt: Current attempt number.
            exception: Exception that triggered retry.
            delay: Delay before next attempt.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_retry(attempt, exception, delay, context)

    def on_success(
        self,
        attempt: int,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Call on_success on all hooks.

        Args:
            attempt: Attempt number that succeeded.
            result: Return value.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_success(attempt, result, context)

    def on_failure(
        self,
        attempts: int,
        exceptions: tuple[Exception, ...],
        context: dict[str, Any],
    ) -> None:
        """Call on_failure on all hooks.

        Args:
            attempts: Total attempts made.
            exceptions: All exceptions encountered.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_failure(attempts, exceptions, context)


# =============================================================================
# Configuration
# =============================================================================


def _get_delay_calculator(
    strategy: RetryStrategy,
    exponential_base: float = 2.0,
    linear_increment: float = 1.0,
) -> DelayCalculator:
    """Get delay calculator for strategy.

    Args:
        strategy: Retry strategy.
        exponential_base: Base for exponential strategy.
        linear_increment: Increment for linear strategy.

    Returns:
        DelayCalculator instance.
    """
    if strategy == RetryStrategy.FIXED:
        return FixedDelayCalculator()
    elif strategy == RetryStrategy.EXPONENTIAL:
        return ExponentialDelayCalculator(base=exponential_base)
    elif strategy == RetryStrategy.LINEAR:
        return LinearDelayCalculator(increment=linear_increment)
    elif strategy == RetryStrategy.FIBONACCI:
        return FibonacciDelayCalculator()
    else:
        return ExponentialDelayCalculator()


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for retry behavior.

    Immutable configuration object for retry operations.
    Use builder methods to create modified copies.

    Attributes:
        max_attempts: Maximum number of attempts (including initial).
        base_delay_seconds: Base delay between retries in seconds.
        max_delay_seconds: Maximum delay between retries in seconds.
        strategy: Delay calculation strategy.
        exponential_base: Base for exponential backoff (default: 2.0).
        linear_increment: Increment for linear backoff in seconds.
        jitter: Whether to add random jitter to delays.
        jitter_factor: Maximum jitter as fraction of delay (0.0 to 1.0).
        exceptions: Exception types to retry on.
        non_retryable_exceptions: Exception types to never retry.
        retry_on_result: Function to check if result should trigger retry.

    Example:
        >>> config = RetryConfig(
        ...     max_attempts=5,
        ...     base_delay_seconds=1.0,
        ...     strategy=RetryStrategy.EXPONENTIAL,
        ... )
        >>> faster_config = config.with_max_attempts(3)
    """

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    exponential_base: float = 2.0
    linear_increment: float = 1.0
    jitter: bool = True
    jitter_factor: float = 0.1
    exceptions: tuple[type[Exception], ...] = (Exception,)
    non_retryable_exceptions: tuple[type[Exception], ...] = ()
    retry_on_result: Callable[[Any], bool] | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not 0.0 <= self.jitter_factor <= 1.0:
            raise ValueError("jitter_factor must be between 0.0 and 1.0")

    def with_max_attempts(self, max_attempts: int) -> RetryConfig:
        """Create config with new max_attempts.

        Args:
            max_attempts: New maximum attempts.

        Returns:
            New RetryConfig with updated value.
        """
        return RetryConfig(
            max_attempts=max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            strategy=self.strategy,
            exponential_base=self.exponential_base,
            linear_increment=self.linear_increment,
            jitter=self.jitter,
            jitter_factor=self.jitter_factor,
            exceptions=self.exceptions,
            non_retryable_exceptions=self.non_retryable_exceptions,
            retry_on_result=self.retry_on_result,
        )

    def with_delays(
        self,
        base_delay_seconds: float | None = None,
        max_delay_seconds: float | None = None,
    ) -> RetryConfig:
        """Create config with new delay settings.

        Args:
            base_delay_seconds: New base delay.
            max_delay_seconds: New max delay.

        Returns:
            New RetryConfig with updated values.
        """
        return RetryConfig(
            max_attempts=self.max_attempts,
            base_delay_seconds=base_delay_seconds
            if base_delay_seconds is not None
            else self.base_delay_seconds,
            max_delay_seconds=max_delay_seconds
            if max_delay_seconds is not None
            else self.max_delay_seconds,
            strategy=self.strategy,
            exponential_base=self.exponential_base,
            linear_increment=self.linear_increment,
            jitter=self.jitter,
            jitter_factor=self.jitter_factor,
            exceptions=self.exceptions,
            non_retryable_exceptions=self.non_retryable_exceptions,
            retry_on_result=self.retry_on_result,
        )

    def with_strategy(self, strategy: RetryStrategy) -> RetryConfig:
        """Create config with new strategy.

        Args:
            strategy: New retry strategy.

        Returns:
            New RetryConfig with updated value.
        """
        return RetryConfig(
            max_attempts=self.max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            strategy=strategy,
            exponential_base=self.exponential_base,
            linear_increment=self.linear_increment,
            jitter=self.jitter,
            jitter_factor=self.jitter_factor,
            exceptions=self.exceptions,
            non_retryable_exceptions=self.non_retryable_exceptions,
            retry_on_result=self.retry_on_result,
        )

    def with_exceptions(
        self,
        exceptions: tuple[type[Exception], ...] | None = None,
        non_retryable: tuple[type[Exception], ...] | None = None,
    ) -> RetryConfig:
        """Create config with new exception settings.

        Args:
            exceptions: Exception types to retry on.
            non_retryable: Exception types to never retry.

        Returns:
            New RetryConfig with updated values.
        """
        return RetryConfig(
            max_attempts=self.max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            strategy=self.strategy,
            exponential_base=self.exponential_base,
            linear_increment=self.linear_increment,
            jitter=self.jitter,
            jitter_factor=self.jitter_factor,
            exceptions=exceptions if exceptions is not None else self.exceptions,
            non_retryable_exceptions=non_retryable
            if non_retryable is not None
            else self.non_retryable_exceptions,
            retry_on_result=self.retry_on_result,
        )

    def with_jitter(
        self,
        enabled: bool = True,
        factor: float | None = None,
    ) -> RetryConfig:
        """Create config with new jitter settings.

        Args:
            enabled: Whether to enable jitter.
            factor: Jitter factor (0.0 to 1.0).

        Returns:
            New RetryConfig with updated values.
        """
        return RetryConfig(
            max_attempts=self.max_attempts,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            strategy=self.strategy,
            exponential_base=self.exponential_base,
            linear_increment=self.linear_increment,
            jitter=enabled,
            jitter_factor=factor if factor is not None else self.jitter_factor,
            exceptions=self.exceptions,
            non_retryable_exceptions=self.non_retryable_exceptions,
            retry_on_result=self.retry_on_result,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_attempts": self.max_attempts,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "strategy": self.strategy.name,
            "exponential_base": self.exponential_base,
            "linear_increment": self.linear_increment,
            "jitter": self.jitter,
            "jitter_factor": self.jitter_factor,
            "exceptions": [e.__name__ for e in self.exceptions],
            "non_retryable_exceptions": [e.__name__ for e in self.non_retryable_exceptions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create RetryConfig from dictionary.

        Note: Exception types cannot be restored from dict.
        Use default exception types.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New RetryConfig instance.
        """
        strategy = RetryStrategy[data.get("strategy", "EXPONENTIAL")]
        return cls(
            max_attempts=data.get("max_attempts", 3),
            base_delay_seconds=data.get("base_delay_seconds", 1.0),
            max_delay_seconds=data.get("max_delay_seconds", 60.0),
            strategy=strategy,
            exponential_base=data.get("exponential_base", 2.0),
            linear_increment=data.get("linear_increment", 1.0),
            jitter=data.get("jitter", True),
            jitter_factor=data.get("jitter_factor", 0.1),
        )


# Default configurations for common use cases
DEFAULT_RETRY_CONFIG = RetryConfig()

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=10,
    base_delay_seconds=0.5,
    max_delay_seconds=30.0,
    strategy=RetryStrategy.EXPONENTIAL,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay_seconds=5.0,
    max_delay_seconds=120.0,
    strategy=RetryStrategy.EXPONENTIAL,
)

NO_DELAY_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay_seconds=0.0,
    max_delay_seconds=0.0,
    jitter=False,
)


# =============================================================================
# Retry Executor
# =============================================================================


class RetryExecutor:
    """Executes functions with retry logic.

    This class encapsulates the retry execution logic and can be
    used directly or through the retry decorator.

    Example:
        >>> executor = RetryExecutor(RetryConfig(max_attempts=3))
        >>> result = executor.execute(my_function, "arg1", key="value")
    """

    def __init__(
        self,
        config: RetryConfig,
        delay_calculator: DelayCalculator | None = None,
        exception_filter: ExceptionFilter | None = None,
        hooks: Sequence[RetryHook] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            config: Retry configuration.
            delay_calculator: Custom delay calculator (uses config.strategy if None).
            exception_filter: Custom exception filter (uses config.exceptions if None).
            hooks: Retry event hooks.
        """
        self.config = config
        self._delay_calculator = delay_calculator or _get_delay_calculator(
            config.strategy,
            config.exponential_base,
            config.linear_increment,
        )
        self._exception_filter = exception_filter or TypeBasedExceptionFilter(
            retryable=config.exceptions,
            non_retryable=config.non_retryable_exceptions,
        )
        self._hook: RetryHook | None = None
        if hooks:
            self._hook = CompositeRetryHook(list(hooks))

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt with jitter.

        Args:
            attempt: Current attempt number.

        Returns:
            Delay in seconds with optional jitter.
        """
        base_delay = self._delay_calculator.calculate_delay(
            attempt,
            self.config.base_delay_seconds,
            self.config.max_delay_seconds,
        )

        if self.config.jitter and base_delay > 0:
            jitter = base_delay * self.config.jitter_factor * random.random()
            base_delay += jitter

        return min(base_delay, self.config.max_delay_seconds)

    def _should_retry_result(self, result: Any) -> bool:
        """Check if result should trigger retry.

        Args:
            result: Function return value.

        Returns:
            True if result should trigger retry.
        """
        if self.config.retry_on_result is None:
            return False
        return self.config.retry_on_result(result)

    def _create_context(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Create context dictionary for hooks.

        Args:
            func: Function being retried.

        Returns:
            Context dictionary.
        """
        return {
            "function": func.__name__,
            "module": func.__module__,
            "max_attempts": self.config.max_attempts,
        }

    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value.

        Raises:
            RetryExhaustedError: When all retry attempts fail.
            NonRetryableError: When exception is not retryable.
        """
        exceptions: list[Exception] = []
        context = self._create_context(func)

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)

                # Check if result should trigger retry
                if self._should_retry_result(result) and attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    if self._hook:
                        # Create a synthetic exception for logging
                        result_exc = ValueError(f"Retry triggered by result: {result}")
                        self._hook.on_retry(attempt, result_exc, delay, context)
                    time.sleep(delay)
                    continue

                # Success
                if self._hook:
                    self._hook.on_success(attempt, result, context)
                return result

            except Exception as exc:
                exceptions.append(exc)

                # Check if exception is retryable
                if not self._exception_filter.should_retry(exc, attempt):
                    if self._hook:
                        self._hook.on_failure(attempt, tuple(exceptions), context)
                    raise NonRetryableError(
                        f"Non-retryable exception: {exc}",
                        attempts=attempt,
                        last_exception=exc,
                        exceptions=tuple(exceptions),
                    ) from exc

                # Last attempt - raise error
                if attempt >= self.config.max_attempts:
                    if self._hook:
                        self._hook.on_failure(attempt, tuple(exceptions), context)
                    raise RetryExhaustedError(
                        f"Retry exhausted after {attempt} attempts",
                        attempts=attempt,
                        last_exception=exc,
                        exceptions=tuple(exceptions),
                    ) from exc

                # Calculate delay and notify hooks
                delay = self._calculate_delay(attempt)
                if self._hook:
                    self._hook.on_retry(attempt, exc, delay, context)

                # Wait before retry
                time.sleep(delay)

        # Should not reach here
        raise RetryExhaustedError(
            "Retry exhausted",
            attempts=self.config.max_attempts,
            exceptions=tuple(exceptions),
        )

    async def execute_async(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with retry logic.

        Args:
            func: Async function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value.

        Raises:
            RetryExhaustedError: When all retry attempts fail.
            NonRetryableError: When exception is not retryable.
        """
        exceptions: list[Exception] = []
        context = self._create_context(func)

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)

                # Check if result should trigger retry
                if self._should_retry_result(result) and attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    if self._hook:
                        result_exc = ValueError(f"Retry triggered by result: {result}")
                        self._hook.on_retry(attempt, result_exc, delay, context)
                    await asyncio.sleep(delay)
                    continue

                # Success
                if self._hook:
                    self._hook.on_success(attempt, result, context)
                return result

            except Exception as exc:
                exceptions.append(exc)

                # Check if exception is retryable
                if not self._exception_filter.should_retry(exc, attempt):
                    if self._hook:
                        self._hook.on_failure(attempt, tuple(exceptions), context)
                    raise NonRetryableError(
                        f"Non-retryable exception: {exc}",
                        attempts=attempt,
                        last_exception=exc,
                        exceptions=tuple(exceptions),
                    ) from exc

                # Last attempt - raise error
                if attempt >= self.config.max_attempts:
                    if self._hook:
                        self._hook.on_failure(attempt, tuple(exceptions), context)
                    raise RetryExhaustedError(
                        f"Retry exhausted after {attempt} attempts",
                        attempts=attempt,
                        last_exception=exc,
                        exceptions=tuple(exceptions),
                    ) from exc

                # Calculate delay and notify hooks
                delay = self._calculate_delay(attempt)
                if self._hook:
                    self._hook.on_retry(attempt, exc, delay, context)

                # Wait before retry
                await asyncio.sleep(delay)

        # Should not reach here
        raise RetryExhaustedError(
            "Retry exhausted",
            attempts=self.config.max_attempts,
            exceptions=tuple(exceptions),
        )


# =============================================================================
# Retry Decorator
# =============================================================================


def retry(
    *,
    config: RetryConfig | None = None,
    max_attempts: int | None = None,
    base_delay_seconds: float | None = None,
    max_delay_seconds: float | None = None,
    strategy: RetryStrategy | None = None,
    exceptions: tuple[type[Exception], ...] | None = None,
    non_retryable: tuple[type[Exception], ...] | None = None,
    jitter: bool | None = None,
    hooks: Sequence[RetryHook] | None = None,
    delay_calculator: DelayCalculator | None = None,
    exception_filter: ExceptionFilter | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to add retry behavior to functions.

    Can be used with a RetryConfig object or individual parameters.
    Supports both sync and async functions.

    Args:
        config: Complete retry configuration (takes precedence).
        max_attempts: Maximum number of attempts.
        base_delay_seconds: Base delay between retries.
        max_delay_seconds: Maximum delay between retries.
        strategy: Delay calculation strategy.
        exceptions: Exception types to retry on.
        non_retryable: Exception types to never retry.
        jitter: Whether to add jitter to delays.
        hooks: Retry event hooks.
        delay_calculator: Custom delay calculator.
        exception_filter: Custom exception filter.

    Returns:
        Decorator function.

    Example:
        >>> @retry(max_attempts=3, exceptions=(ConnectionError,))
        ... def fetch_data():
        ...     return api.get("/data")

        >>> @retry(config=RetryConfig(max_attempts=5))
        ... async def async_fetch():
        ...     return await api.async_get("/data")
    """
    # Build config from parameters if not provided
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts if max_attempts is not None else 3,
            base_delay_seconds=base_delay_seconds if base_delay_seconds is not None else 1.0,
            max_delay_seconds=max_delay_seconds if max_delay_seconds is not None else 60.0,
            strategy=strategy if strategy is not None else RetryStrategy.EXPONENTIAL,
            exceptions=exceptions if exceptions is not None else (Exception,),
            non_retryable_exceptions=non_retryable if non_retryable is not None else (),
            jitter=jitter if jitter is not None else True,
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        executor = RetryExecutor(
            config=config,  # type: ignore
            delay_calculator=delay_calculator,
            exception_filter=exception_filter,
            hooks=hooks,
        )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await executor.execute_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return executor.execute(func, *args, **kwargs)

            return sync_wrapper

    return decorator


def retry_call(
    func: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    hooks: Sequence[RetryHook] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with retry logic without using decorator.

    Useful when you can't modify the function definition.

    Args:
        func: Function to execute.
        *args: Positional arguments for function.
        config: Retry configuration.
        hooks: Retry event hooks.
        **kwargs: Keyword arguments for function.

    Returns:
        Function return value.

    Example:
        >>> result = retry_call(
        ...     external_api.fetch,
        ...     endpoint="/data",
        ...     config=RetryConfig(max_attempts=3),
        ... )
    """
    executor = RetryExecutor(
        config=config or DEFAULT_RETRY_CONFIG,
        hooks=hooks,
    )
    return executor.execute(func, *args, **kwargs)


async def retry_call_async(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    config: RetryConfig | None = None,
    hooks: Sequence[RetryHook] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an async function with retry logic without using decorator.

    Args:
        func: Async function to execute.
        *args: Positional arguments for function.
        config: Retry configuration.
        hooks: Retry event hooks.
        **kwargs: Keyword arguments for function.

    Returns:
        Function return value.

    Example:
        >>> result = await retry_call_async(
        ...     async_api.fetch,
        ...     endpoint="/data",
        ...     config=RetryConfig(max_attempts=3),
        ... )
    """
    executor = RetryExecutor(
        config=config or DEFAULT_RETRY_CONFIG,
        hooks=hooks,
    )
    return await executor.execute_async(func, *args, **kwargs)
