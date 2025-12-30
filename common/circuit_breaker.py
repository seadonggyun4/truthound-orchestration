"""Circuit Breaker pattern implementation for Truthound Integrations.

This module provides a production-ready Circuit Breaker pattern implementation
designed for resilient operation in distributed environments. It prevents
cascade failures by temporarily blocking requests to failing services.

Circuit States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests are blocked
    - HALF_OPEN: Testing if service has recovered

Design Principles:
    1. Protocol-based: Easy to extend with custom failure detection
    2. Immutable Config: Thread-safe configuration using frozen dataclass
    3. Observable: Hook system for monitoring state transitions
    4. Composable: Works well with retry and other patterns

Example:
    >>> from common.circuit_breaker import circuit_breaker, CircuitBreakerConfig
    >>> @circuit_breaker(failure_threshold=5, recovery_timeout_seconds=30.0)
    ... def call_external_service():
    ...     return api.get("/data")

    >>> # With custom configuration
    >>> config = CircuitBreakerConfig(
    ...     failure_threshold=10,
    ...     success_threshold=3,
    ...     recovery_timeout_seconds=60.0,
    ... )
    >>> @circuit_breaker(config=config)
    ... async def async_call():
    ...     return await api.async_get("/data")
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
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


class CircuitBreakerError(TruthoundIntegrationError):
    """Base exception for circuit breaker errors.

    Attributes:
        state: Current circuit breaker state.
        failure_count: Number of failures recorded.
    """

    def __init__(
        self,
        message: str,
        *,
        state: CircuitState | None = None,
        failure_count: int = 0,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize circuit breaker error.

        Args:
            message: Human-readable error description.
            state: Current circuit breaker state.
            failure_count: Number of failures recorded.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        details["failure_count"] = failure_count
        if state:
            details["state"] = state.name
        super().__init__(message, details=details, cause=cause)
        self.state = state
        self.failure_count = failure_count


class CircuitOpenError(CircuitBreakerError):
    """Exception raised when circuit is open and blocking requests.

    Attributes:
        remaining_seconds: Seconds until circuit may try half-open.
    """

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        *,
        remaining_seconds: float = 0.0,
        state: CircuitState | None = None,
        failure_count: int = 0,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize circuit open error.

        Args:
            message: Human-readable error description.
            remaining_seconds: Seconds until circuit may try half-open.
            state: Current circuit breaker state.
            failure_count: Number of failures recorded.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        details["remaining_seconds"] = remaining_seconds
        super().__init__(
            message,
            state=state or CircuitState.OPEN,
            failure_count=failure_count,
            details=details,
            cause=cause,
        )
        self.remaining_seconds = remaining_seconds


# =============================================================================
# Enums
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states.

    Attributes:
        CLOSED: Normal operation, requests pass through.
        OPEN: Failure threshold exceeded, requests are blocked.
        HALF_OPEN: Testing if service has recovered.
    """

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class FailureDetector(Protocol):
    """Protocol for detecting failures.

    Implement this protocol to create custom failure detection logic.
    """

    @abstractmethod
    def is_failure(self, exception: Exception) -> bool:
        """Determine if exception should count as a failure.

        Args:
            exception: The exception that occurred.

        Returns:
            True if exception should count as a failure.
        """
        ...

    @abstractmethod
    def is_success(self, result: Any) -> bool:
        """Determine if result indicates success.

        Args:
            result: The return value from the function.

        Returns:
            True if result indicates success.
        """
        ...


@runtime_checkable
class CircuitBreakerHook(Protocol):
    """Protocol for circuit breaker event hooks.

    Implement this to receive notifications about circuit events.
    """

    @abstractmethod
    def on_state_change(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Called when circuit state changes.

        Args:
            old_state: Previous circuit state.
            new_state: New circuit state.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_success(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Called on successful execution.

        Args:
            state: Current circuit state.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_failure(
        self,
        exception: Exception,
        state: CircuitState,
        failure_count: int,
        context: dict[str, Any],
    ) -> None:
        """Called on failure.

        Args:
            exception: Exception that occurred.
            state: Current circuit state.
            failure_count: Current failure count.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_rejected(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Called when request is rejected due to open circuit.

        Args:
            state: Current circuit state (OPEN).
            context: Additional context information.
        """
        ...


# =============================================================================
# Failure Detectors
# =============================================================================


class TypeBasedFailureDetector:
    """Detect failures based on exception types.

    By default, all exceptions are considered failures.
    """

    def __init__(
        self,
        failure_exceptions: tuple[type[Exception], ...] = (Exception,),
        ignored_exceptions: tuple[type[Exception], ...] = (),
    ) -> None:
        """Initialize detector.

        Args:
            failure_exceptions: Exception types that count as failures.
            ignored_exceptions: Exception types that don't count as failures.
        """
        self.failure_exceptions = failure_exceptions
        self.ignored_exceptions = ignored_exceptions

    def is_failure(self, exception: Exception) -> bool:
        """Check if exception is a failure.

        Args:
            exception: The exception to check.

        Returns:
            True if exception counts as a failure.
        """
        if isinstance(exception, self.ignored_exceptions):
            return False
        return isinstance(exception, self.failure_exceptions)

    def is_success(self, result: Any) -> bool:
        """Check if result indicates success.

        Args:
            result: The return value.

        Returns:
            Always True for type-based detector.
        """
        return True


class CallableFailureDetector:
    """Detect failures using callable predicates."""

    def __init__(
        self,
        failure_predicate: Callable[[Exception], bool] | None = None,
        success_predicate: Callable[[Any], bool] | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            failure_predicate: Function to check if exception is failure.
            success_predicate: Function to check if result is success.
        """
        self._failure_predicate = failure_predicate or (lambda _: True)
        self._success_predicate = success_predicate or (lambda _: True)

    def is_failure(self, exception: Exception) -> bool:
        """Check using failure predicate.

        Args:
            exception: The exception to check.

        Returns:
            Result of failure predicate.
        """
        return self._failure_predicate(exception)

    def is_success(self, result: Any) -> bool:
        """Check using success predicate.

        Args:
            result: The return value.

        Returns:
            Result of success predicate.
        """
        return self._success_predicate(result)


class CompositeFailureDetector:
    """Combine multiple failure detectors."""

    def __init__(
        self,
        detectors: Sequence[FailureDetector],
        require_all_for_failure: bool = False,
    ) -> None:
        """Initialize composite detector.

        Args:
            detectors: List of detectors to combine.
            require_all_for_failure: If True, all must agree for failure.
        """
        self._detectors = list(detectors)
        self._require_all = require_all_for_failure

    def is_failure(self, exception: Exception) -> bool:
        """Check all detectors.

        Args:
            exception: The exception to check.

        Returns:
            Combined result from all detectors.
        """
        if self._require_all:
            return all(d.is_failure(exception) for d in self._detectors)
        return any(d.is_failure(exception) for d in self._detectors)

    def is_success(self, result: Any) -> bool:
        """Check all detectors for success.

        Args:
            result: The return value.

        Returns:
            True only if all detectors agree on success.
        """
        return all(d.is_success(result) for d in self._detectors)


# =============================================================================
# Hooks
# =============================================================================


class LoggingCircuitBreakerHook:
    """Hook that logs circuit breaker events.

    Uses the Truthound logging system for structured logging.
    """

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.circuit_breaker).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.circuit_breaker")

    def on_state_change(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Log state transition.

        Args:
            old_state: Previous circuit state.
            new_state: New circuit state.
            context: Additional context.
        """
        if new_state == CircuitState.OPEN:
            self._logger.warning(
                "Circuit breaker opened",
                old_state=old_state.name,
                new_state=new_state.name,
                **context,
            )
        elif new_state == CircuitState.CLOSED:
            self._logger.info(
                "Circuit breaker closed",
                old_state=old_state.name,
                new_state=new_state.name,
                **context,
            )
        else:
            self._logger.info(
                "Circuit breaker half-open",
                old_state=old_state.name,
                new_state=new_state.name,
                **context,
            )

    def on_success(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Log success (only in half-open state).

        Args:
            state: Current circuit state.
            context: Additional context.
        """
        if state == CircuitState.HALF_OPEN:
            self._logger.info(
                "Circuit breaker success in half-open state",
                state=state.name,
                **context,
            )

    def on_failure(
        self,
        exception: Exception,
        state: CircuitState,
        failure_count: int,
        context: dict[str, Any],
    ) -> None:
        """Log failure.

        Args:
            exception: Exception that occurred.
            state: Current circuit state.
            failure_count: Current failure count.
            context: Additional context.
        """
        self._logger.warning(
            "Circuit breaker recorded failure",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            state=state.name,
            failure_count=failure_count,
            **context,
        )

    def on_rejected(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Log rejected request.

        Args:
            state: Current circuit state.
            context: Additional context.
        """
        self._logger.warning(
            "Circuit breaker rejected request",
            state=state.name,
            **context,
        )


class MetricsCircuitBreakerHook:
    """Hook that collects circuit breaker metrics.

    Useful for monitoring and alerting on circuit breaker behavior.
    """

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._state_changes: list[tuple[CircuitState, CircuitState, datetime]] = []
        self._success_count: int = 0
        self._failure_count: int = 0
        self._rejected_count: int = 0
        self._lock = threading.Lock()

    def on_state_change(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Record state transition.

        Args:
            old_state: Previous circuit state.
            new_state: New circuit state.
            context: Additional context.
        """
        with self._lock:
            self._state_changes.append(
                (old_state, new_state, datetime.now(UTC))
            )

    def on_success(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Record success.

        Args:
            state: Current circuit state.
            context: Additional context.
        """
        with self._lock:
            self._success_count += 1

    def on_failure(
        self,
        exception: Exception,
        state: CircuitState,
        failure_count: int,
        context: dict[str, Any],
    ) -> None:
        """Record failure.

        Args:
            exception: Exception that occurred.
            state: Current circuit state.
            failure_count: Current failure count.
            context: Additional context.
        """
        with self._lock:
            self._failure_count += 1

    def on_rejected(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Record rejected request.

        Args:
            state: Current circuit state.
            context: Additional context.
        """
        with self._lock:
            self._rejected_count += 1

    @property
    def success_count(self) -> int:
        """Get total success count."""
        return self._success_count

    @property
    def failure_count(self) -> int:
        """Get total failure count."""
        return self._failure_count

    @property
    def rejected_count(self) -> int:
        """Get total rejected request count."""
        return self._rejected_count

    @property
    def state_changes(self) -> list[tuple[CircuitState, CircuitState, datetime]]:
        """Get state change history."""
        with self._lock:
            return list(self._state_changes)

    @property
    def times_opened(self) -> int:
        """Get number of times circuit opened."""
        return sum(
            1 for _, new, _ in self._state_changes if new == CircuitState.OPEN
        )

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._state_changes.clear()
            self._success_count = 0
            self._failure_count = 0
            self._rejected_count = 0


class CompositeCircuitBreakerHook:
    """Combine multiple circuit breaker hooks."""

    def __init__(self, hooks: Sequence[CircuitBreakerHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: CircuitBreakerHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: CircuitBreakerHook) -> None:
        """Remove a hook.

        Args:
            hook: Hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_state_change(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Call on_state_change on all hooks.

        Args:
            old_state: Previous circuit state.
            new_state: New circuit state.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_state_change(old_state, new_state, context)

    def on_success(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Call on_success on all hooks.

        Args:
            state: Current circuit state.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_success(state, context)

    def on_failure(
        self,
        exception: Exception,
        state: CircuitState,
        failure_count: int,
        context: dict[str, Any],
    ) -> None:
        """Call on_failure on all hooks.

        Args:
            exception: Exception that occurred.
            state: Current circuit state.
            failure_count: Current failure count.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_failure(exception, state, failure_count, context)

    def on_rejected(
        self,
        state: CircuitState,
        context: dict[str, Any],
    ) -> None:
        """Call on_rejected on all hooks.

        Args:
            state: Current circuit state.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_rejected(state, context)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Immutable configuration object for circuit breaker operations.
    Use builder methods to create modified copies.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        success_threshold: Successes in half-open before closing circuit.
        recovery_timeout_seconds: Seconds before trying half-open after opening.
        half_open_max_calls: Max concurrent calls allowed in half-open state.
        exceptions: Exception types that count as failures.
        ignored_exceptions: Exception types that don't count as failures.
        exclude_from_failure_count: If True, ignored exceptions don't count.
        name: Optional name for the circuit breaker (for logging/metrics).

    Example:
        >>> config = CircuitBreakerConfig(
        ...     failure_threshold=5,
        ...     success_threshold=2,
        ...     recovery_timeout_seconds=30.0,
        ... )
        >>> stricter_config = config.with_failure_threshold(3)
    """

    failure_threshold: int = 5
    success_threshold: int = 1
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1
    exceptions: tuple[type[Exception], ...] = (Exception,)
    ignored_exceptions: tuple[type[Exception], ...] = ()
    exclude_from_failure_count: bool = True
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.recovery_timeout_seconds < 0:
            raise ValueError("recovery_timeout_seconds must be non-negative")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")

    def with_failure_threshold(self, failure_threshold: int) -> CircuitBreakerConfig:
        """Create config with new failure threshold.

        Args:
            failure_threshold: New failure threshold.

        Returns:
            New CircuitBreakerConfig with updated value.
        """
        return CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=self.success_threshold,
            recovery_timeout_seconds=self.recovery_timeout_seconds,
            half_open_max_calls=self.half_open_max_calls,
            exceptions=self.exceptions,
            ignored_exceptions=self.ignored_exceptions,
            exclude_from_failure_count=self.exclude_from_failure_count,
            name=self.name,
        )

    def with_success_threshold(self, success_threshold: int) -> CircuitBreakerConfig:
        """Create config with new success threshold.

        Args:
            success_threshold: New success threshold.

        Returns:
            New CircuitBreakerConfig with updated value.
        """
        return CircuitBreakerConfig(
            failure_threshold=self.failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout_seconds=self.recovery_timeout_seconds,
            half_open_max_calls=self.half_open_max_calls,
            exceptions=self.exceptions,
            ignored_exceptions=self.ignored_exceptions,
            exclude_from_failure_count=self.exclude_from_failure_count,
            name=self.name,
        )

    def with_recovery_timeout(
        self,
        recovery_timeout_seconds: float,
    ) -> CircuitBreakerConfig:
        """Create config with new recovery timeout.

        Args:
            recovery_timeout_seconds: New recovery timeout in seconds.

        Returns:
            New CircuitBreakerConfig with updated value.
        """
        return CircuitBreakerConfig(
            failure_threshold=self.failure_threshold,
            success_threshold=self.success_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
            half_open_max_calls=self.half_open_max_calls,
            exceptions=self.exceptions,
            ignored_exceptions=self.ignored_exceptions,
            exclude_from_failure_count=self.exclude_from_failure_count,
            name=self.name,
        )

    def with_exceptions(
        self,
        exceptions: tuple[type[Exception], ...] | None = None,
        ignored: tuple[type[Exception], ...] | None = None,
    ) -> CircuitBreakerConfig:
        """Create config with new exception settings.

        Args:
            exceptions: Exception types that count as failures.
            ignored: Exception types that don't count as failures.

        Returns:
            New CircuitBreakerConfig with updated values.
        """
        return CircuitBreakerConfig(
            failure_threshold=self.failure_threshold,
            success_threshold=self.success_threshold,
            recovery_timeout_seconds=self.recovery_timeout_seconds,
            half_open_max_calls=self.half_open_max_calls,
            exceptions=exceptions if exceptions is not None else self.exceptions,
            ignored_exceptions=ignored
            if ignored is not None
            else self.ignored_exceptions,
            exclude_from_failure_count=self.exclude_from_failure_count,
            name=self.name,
        )

    def with_name(self, name: str) -> CircuitBreakerConfig:
        """Create config with a name.

        Args:
            name: Name for the circuit breaker.

        Returns:
            New CircuitBreakerConfig with updated value.
        """
        return CircuitBreakerConfig(
            failure_threshold=self.failure_threshold,
            success_threshold=self.success_threshold,
            recovery_timeout_seconds=self.recovery_timeout_seconds,
            half_open_max_calls=self.half_open_max_calls,
            exceptions=self.exceptions,
            ignored_exceptions=self.ignored_exceptions,
            exclude_from_failure_count=self.exclude_from_failure_count,
            name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "half_open_max_calls": self.half_open_max_calls,
            "exceptions": [e.__name__ for e in self.exceptions],
            "ignored_exceptions": [e.__name__ for e in self.ignored_exceptions],
            "exclude_from_failure_count": self.exclude_from_failure_count,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create CircuitBreakerConfig from dictionary.

        Note: Exception types cannot be restored from dict.
        Use default exception types.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New CircuitBreakerConfig instance.
        """
        return cls(
            failure_threshold=data.get("failure_threshold", 5),
            success_threshold=data.get("success_threshold", 1),
            recovery_timeout_seconds=data.get("recovery_timeout_seconds", 30.0),
            half_open_max_calls=data.get("half_open_max_calls", 1),
            exclude_from_failure_count=data.get("exclude_from_failure_count", True),
            name=data.get("name"),
        )


# Default configurations for common use cases
DEFAULT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig()

SENSITIVE_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    success_threshold=2,
    recovery_timeout_seconds=60.0,
)

RESILIENT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=10,
    success_threshold=1,
    recovery_timeout_seconds=15.0,
)

AGGRESSIVE_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=2,
    success_threshold=3,
    recovery_timeout_seconds=120.0,
)


# =============================================================================
# Circuit Breaker State
# =============================================================================


@dataclass
class CircuitBreakerState:
    """Mutable state for circuit breaker.

    This class tracks the current state of a circuit breaker, including
    failure counts, timestamps, and success counts during recovery.

    Attributes:
        state: Current circuit state.
        failure_count: Consecutive failure count.
        success_count: Consecutive success count in half-open state.
        last_failure_time: Timestamp of last failure.
        opened_at: Timestamp when circuit opened.
        half_open_calls: Number of calls in current half-open period.
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    opened_at: float | None = None
    half_open_calls: int = 0


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """Circuit breaker implementation.

    This class provides the core circuit breaker logic with thread-safe
    state management and configurable behavior.

    Example:
        >>> cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
        >>> result = cb.call(my_function, arg1, arg2)

        >>> # Check state
        >>> if cb.is_open:
        ...     print("Circuit is open!")
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        failure_detector: FailureDetector | None = None,
        hooks: Sequence[CircuitBreakerHook] | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration.
            failure_detector: Custom failure detector (uses config if None).
            hooks: Circuit breaker event hooks.
        """
        self.config = config
        self._failure_detector = failure_detector or TypeBasedFailureDetector(
            failure_exceptions=config.exceptions,
            ignored_exceptions=config.ignored_exceptions,
        )
        self._hook: CircuitBreakerHook | None = None
        if hooks:
            self._hook = CompositeCircuitBreakerHook(list(hooks))
        self._state = CircuitBreakerState()
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_recovery()
            return self._state.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._state.failure_count

    @property
    def success_count(self) -> int:
        """Get current success count (in half-open state)."""
        with self._lock:
            return self._state.success_count

    def _create_context(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Create context dictionary for hooks.

        Args:
            func: Function being protected.

        Returns:
            Context dictionary.
        """
        return {
            "function": func.__name__,
            "module": func.__module__,
            "circuit_name": self.config.name or func.__name__,
            "failure_threshold": self.config.failure_threshold,
        }

    def _check_recovery(self) -> None:
        """Check if circuit should transition to half-open.

        Must be called with lock held.
        """
        if self._state.state != CircuitState.OPEN:
            return

        if self._state.opened_at is None:
            return

        elapsed = time.monotonic() - self._state.opened_at
        if elapsed >= self.config.recovery_timeout_seconds:
            old_state = self._state.state
            self._state.state = CircuitState.HALF_OPEN
            self._state.success_count = 0
            self._state.half_open_calls = 0
            if self._hook:
                self._hook.on_state_change(
                    old_state,
                    CircuitState.HALF_OPEN,
                    {"elapsed_seconds": elapsed},
                )

    def _record_success(self, context: dict[str, Any]) -> None:
        """Record a successful call.

        Args:
            context: Context for hooks.
        """
        with self._lock:
            if self._hook:
                self._hook.on_success(self._state.state, context)

            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                if self._state.success_count >= self.config.success_threshold:
                    # Recovery successful - close circuit
                    old_state = self._state.state
                    self._state.state = CircuitState.CLOSED
                    self._state.failure_count = 0
                    self._state.success_count = 0
                    self._state.opened_at = None
                    if self._hook:
                        self._hook.on_state_change(
                            old_state,
                            CircuitState.CLOSED,
                            context,
                        )
            elif self._state.state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._state.failure_count = 0

    def _record_failure(self, exception: Exception, context: dict[str, Any]) -> None:
        """Record a failed call.

        Args:
            exception: Exception that occurred.
            context: Context for hooks.
        """
        with self._lock:
            # Check if this exception should be counted
            if not self._failure_detector.is_failure(exception):
                return

            self._state.failure_count += 1
            self._state.last_failure_time = time.monotonic()

            if self._hook:
                self._hook.on_failure(
                    exception,
                    self._state.state,
                    self._state.failure_count,
                    context,
                )

            if self._state.state == CircuitState.HALF_OPEN:
                # Failure during recovery - reopen circuit
                old_state = self._state.state
                self._state.state = CircuitState.OPEN
                self._state.opened_at = time.monotonic()
                self._state.success_count = 0
                if self._hook:
                    self._hook.on_state_change(
                        old_state,
                        CircuitState.OPEN,
                        context,
                    )

            elif self._state.state == CircuitState.CLOSED:
                if self._state.failure_count >= self.config.failure_threshold:
                    # Threshold exceeded - open circuit
                    old_state = self._state.state
                    self._state.state = CircuitState.OPEN
                    self._state.opened_at = time.monotonic()
                    if self._hook:
                        self._hook.on_state_change(
                            old_state,
                            CircuitState.OPEN,
                            context,
                        )

    def _can_execute(self) -> bool:
        """Check if execution is allowed.

        Returns:
            True if execution is allowed.
        """
        with self._lock:
            self._check_recovery()

            if self._state.state == CircuitState.CLOSED:
                return True

            if self._state.state == CircuitState.HALF_OPEN:
                if self._state.half_open_calls < self.config.half_open_max_calls:
                    self._state.half_open_calls += 1
                    return True
                return False

            return False

    def remaining_recovery_time(self) -> float:
        """Get remaining time until recovery attempt.

        Returns:
            Seconds until circuit may transition to half-open, or 0 if not open.
        """
        with self._lock:
            if self._state.state != CircuitState.OPEN:
                return 0.0

            if self._state.opened_at is None:
                return 0.0

            elapsed = time.monotonic() - self._state.opened_at
            remaining = self.config.recovery_timeout_seconds - elapsed
            return max(0.0, remaining)

    def reset(self) -> None:
        """Reset circuit breaker to closed state.

        Use with caution - typically only for testing or manual recovery.
        """
        with self._lock:
            old_state = self._state.state
            self._state = CircuitBreakerState()
            if old_state != CircuitState.CLOSED and self._hook:
                self._hook.on_state_change(
                    old_state,
                    CircuitState.CLOSED,
                    {"reason": "manual_reset"},
                )

    def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value.

        Raises:
            CircuitOpenError: When circuit is open and blocking requests.
        """
        context = self._create_context(func)

        if not self._can_execute():
            if self._hook:
                self._hook.on_rejected(self._state.state, context)
            raise CircuitOpenError(
                f"Circuit breaker '{self.config.name or func.__name__}' is open",
                remaining_seconds=self.remaining_recovery_time(),
                state=self._state.state,
                failure_count=self._state.failure_count,
            )

        try:
            result = func(*args, **kwargs)

            # Check if result indicates success
            if self._failure_detector.is_success(result):
                self._record_success(context)
            else:
                # Treat unsuccessful result as failure
                self._record_failure(
                    ValueError(f"Unsuccessful result: {result}"),
                    context,
                )

            return result

        except Exception as exc:
            self._record_failure(exc, context)
            raise

    async def call_async(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value.

        Raises:
            CircuitOpenError: When circuit is open and blocking requests.
        """
        context = self._create_context(func)

        if not self._can_execute():
            if self._hook:
                self._hook.on_rejected(self._state.state, context)
            raise CircuitOpenError(
                f"Circuit breaker '{self.config.name or func.__name__}' is open",
                remaining_seconds=self.remaining_recovery_time(),
                state=self._state.state,
                failure_count=self._state.failure_count,
            )

        try:
            result = await func(*args, **kwargs)

            if self._failure_detector.is_success(result):
                self._record_success(context)
            else:
                self._record_failure(
                    ValueError(f"Unsuccessful result: {result}"),
                    context,
                )

            return result

        except Exception as exc:
            self._record_failure(exc, context)
            raise


# =============================================================================
# Circuit Breaker Registry
# =============================================================================


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.

    Provides a central location to create, retrieve, and manage
    circuit breakers by name.

    Example:
        >>> registry = CircuitBreakerRegistry()
        >>> cb = registry.get_or_create("api_service", config=my_config)
        >>> result = cb.call(api_call)
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name.

        Args:
            name: Circuit breaker name.

        Returns:
            CircuitBreaker if found, None otherwise.
        """
        with self._lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        failure_detector: FailureDetector | None = None,
        hooks: Sequence[CircuitBreakerHook] | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker.

        Args:
            name: Circuit breaker name.
            config: Configuration (uses default if None).
            failure_detector: Custom failure detector.
            hooks: Circuit breaker event hooks.

        Returns:
            CircuitBreaker instance.
        """
        with self._lock:
            if name in self._breakers:
                return self._breakers[name]

            cb_config = (config or DEFAULT_CIRCUIT_BREAKER_CONFIG).with_name(name)
            cb = CircuitBreaker(
                config=cb_config,
                failure_detector=failure_detector,
                hooks=hooks,
            )
            self._breakers[name] = cb
            return cb

    def remove(self, name: str) -> bool:
        """Remove circuit breaker by name.

        Args:
            name: Circuit breaker name.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()

    def get_all_states(self) -> dict[str, CircuitState]:
        """Get states of all circuit breakers.

        Returns:
            Dictionary mapping names to states.
        """
        with self._lock:
            return {name: cb.state for name, cb in self._breakers.items()}

    @property
    def names(self) -> list[str]:
        """Get all circuit breaker names."""
        with self._lock:
            return list(self._breakers.keys())


# Global registry instance
_default_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    failure_detector: FailureDetector | None = None,
    hooks: Sequence[CircuitBreakerHook] | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry.

    Args:
        name: Circuit breaker name.
        config: Configuration (uses default if None).
        failure_detector: Custom failure detector.
        hooks: Circuit breaker event hooks.

    Returns:
        CircuitBreaker instance.

    Example:
        >>> cb = get_circuit_breaker("external_api")
        >>> result = cb.call(api_request)
    """
    return _default_registry.get_or_create(
        name,
        config=config,
        failure_detector=failure_detector,
        hooks=hooks,
    )


def get_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.

    Returns:
        The global CircuitBreakerRegistry instance.
    """
    return _default_registry


# =============================================================================
# Circuit Breaker Decorator
# =============================================================================


def circuit_breaker(
    *,
    config: CircuitBreakerConfig | None = None,
    name: str | None = None,
    failure_threshold: int | None = None,
    success_threshold: int | None = None,
    recovery_timeout_seconds: float | None = None,
    exceptions: tuple[type[Exception], ...] | None = None,
    ignored_exceptions: tuple[type[Exception], ...] | None = None,
    hooks: Sequence[CircuitBreakerHook] | None = None,
    failure_detector: FailureDetector | None = None,
    use_registry: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to add circuit breaker protection to functions.

    Can be used with a CircuitBreakerConfig object or individual parameters.
    Supports both sync and async functions.

    Args:
        config: Complete circuit breaker configuration (takes precedence).
        name: Name for the circuit breaker (for registry and logging).
        failure_threshold: Number of failures before opening circuit.
        success_threshold: Successes in half-open before closing.
        recovery_timeout_seconds: Seconds before trying half-open.
        exceptions: Exception types that count as failures.
        ignored_exceptions: Exception types that don't count as failures.
        hooks: Circuit breaker event hooks.
        failure_detector: Custom failure detector.
        use_registry: If True, use global registry (enables sharing).

    Returns:
        Decorator function.

    Example:
        >>> @circuit_breaker(failure_threshold=5)
        ... def call_api():
        ...     return api.get("/data")

        >>> @circuit_breaker(config=CircuitBreakerConfig(failure_threshold=3))
        ... async def async_call():
        ...     return await api.async_get("/data")
    """
    # Build config from parameters if not provided
    if config is None:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold if failure_threshold is not None else 5,
            success_threshold=success_threshold if success_threshold is not None else 1,
            recovery_timeout_seconds=(
                recovery_timeout_seconds
                if recovery_timeout_seconds is not None
                else 30.0
            ),
            exceptions=exceptions if exceptions is not None else (Exception,),
            ignored_exceptions=(
                ignored_exceptions if ignored_exceptions is not None else ()
            ),
            name=name,
        )
    elif name is not None:
        config = config.with_name(name)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cb_name = config.name or func.__name__

        if use_registry:
            cb = get_circuit_breaker(
                cb_name,
                config=config,
                failure_detector=failure_detector,
                hooks=hooks,
            )
        else:
            cb = CircuitBreaker(
                config=config.with_name(cb_name),
                failure_detector=failure_detector,
                hooks=hooks,
            )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await cb.call_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return cb.call(func, *args, **kwargs)

            return sync_wrapper

    return decorator


def circuit_breaker_call(
    func: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    config: CircuitBreakerConfig | None = None,
    hooks: Sequence[CircuitBreakerHook] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with circuit breaker without using decorator.

    Useful when you can't modify the function definition.

    Args:
        func: Function to execute.
        *args: Positional arguments for function.
        name: Circuit breaker name (uses function name if None).
        config: Circuit breaker configuration.
        hooks: Circuit breaker event hooks.
        **kwargs: Keyword arguments for function.

    Returns:
        Function return value.

    Example:
        >>> result = circuit_breaker_call(
        ...     external_api.fetch,
        ...     endpoint="/data",
        ...     name="external_api",
        ...     config=CircuitBreakerConfig(failure_threshold=3),
        ... )
    """
    cb_name = name or func.__name__
    cb = get_circuit_breaker(cb_name, config=config, hooks=hooks)
    return cb.call(func, *args, **kwargs)


async def circuit_breaker_call_async(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    name: str | None = None,
    config: CircuitBreakerConfig | None = None,
    hooks: Sequence[CircuitBreakerHook] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an async function with circuit breaker without using decorator.

    Args:
        func: Async function to execute.
        *args: Positional arguments for function.
        name: Circuit breaker name (uses function name if None).
        config: Circuit breaker configuration.
        hooks: Circuit breaker event hooks.
        **kwargs: Keyword arguments for function.

    Returns:
        Function return value.

    Example:
        >>> result = await circuit_breaker_call_async(
        ...     async_api.fetch,
        ...     endpoint="/data",
        ...     name="async_api",
        ... )
    """
    cb_name = name or func.__name__
    cb = get_circuit_breaker(cb_name, config=config, hooks=hooks)
    return await cb.call_async(func, *args, **kwargs)
