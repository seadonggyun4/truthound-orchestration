"""Engine Chain and Fallback patterns for Data Quality Engines.

This module provides flexible engine chaining and fallback capabilities,
allowing multiple engines to be composed together with various execution
strategies.

Key Components:
    - EngineChain: Execute engines in sequence with fallback on failure
    - ParallelEngineChain: Execute multiple engines concurrently
    - ConditionalEngineChain: Route to engines based on conditions
    - EngineSelector: Dynamic engine selection based on data/rules

Design Principles:
    1. Protocol-based: All chains implement DataQualityEngine protocol
    2. Composable: Chains can contain other chains for complex patterns
    3. Observable: Hook system for chain execution events
    4. Configurable: Fine-grained control over fallback behavior
    5. Fail-safe: Comprehensive error handling and recovery

Use Cases:
    - Primary/backup engine failover
    - Load balancing across multiple engines
    - Engine selection based on data characteristics
    - Graceful degradation in distributed systems
    - A/B testing of engine implementations

Example:
    >>> from common.engines.chain import EngineChain, FallbackConfig
    >>> chain = EngineChain(
    ...     engines=[primary_engine, backup_engine],
    ...     config=FallbackConfig(retry_count=2),
    ... )
    >>> result = chain.check(data, rules)  # Falls back on failure

    >>> # With conditional routing
    >>> chain = ConditionalEngineChain()
    >>> chain.add_route(lambda data, rules: len(rules) > 10, heavy_engine)
    >>> chain.add_route(lambda data, rules: True, light_engine)  # default
    >>> result = chain.check(data, rules)
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

from common.base import AnomalyResult, CheckResult, DriftResult, LearnResult, ProfileResult
from common.engines.base import (
    AnomalyDetectionEngine,
    DataQualityEngine,
    DriftDetectionEngine,
    EngineCapabilities,
    EngineInfoMixin,
    supports_anomaly,
    supports_drift,
)
from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class EngineChainError(TruthoundIntegrationError):
    """Base exception for engine chain errors.

    Attributes:
        chain_name: Name of the chain.
        engine_name: Name of the engine that failed.
        attempted_engines: List of engines that were tried.
    """

    def __init__(
        self,
        message: str,
        *,
        chain_name: str | None = None,
        engine_name: str | None = None,
        attempted_engines: tuple[str, ...] = (),
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize chain error.

        Args:
            message: Human-readable error description.
            chain_name: Name of the chain.
            engine_name: Name of the engine that failed.
            attempted_engines: List of engines that were tried.
            details: Additional error context.
            cause: Original exception.
        """
        details = details or {}
        if chain_name:
            details["chain_name"] = chain_name
        if engine_name:
            details["engine_name"] = engine_name
        if attempted_engines:
            details["attempted_engines"] = list(attempted_engines)
        super().__init__(message, details=details, cause=cause)
        self.chain_name = chain_name
        self.engine_name = engine_name
        self.attempted_engines = attempted_engines


class AllEnginesFailedError(EngineChainError):
    """Exception raised when all engines in a chain fail.

    This indicates that no engine in the chain could successfully
    complete the requested operation.

    Attributes:
        exceptions: Map of engine names to their exceptions.
    """

    def __init__(
        self,
        message: str,
        *,
        chain_name: str | None = None,
        attempted_engines: tuple[str, ...] = (),
        exceptions: dict[str, Exception] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize all engines failed error."""
        details = details or {}
        if exceptions:
            details["exception_types"] = {
                name: type(exc).__name__ for name, exc in exceptions.items()
            }
        last_exception = None
        if exceptions:
            last_exception = list(exceptions.values())[-1]
        super().__init__(
            message,
            chain_name=chain_name,
            attempted_engines=attempted_engines,
            details=details,
            cause=last_exception,
        )
        self.exceptions = exceptions or {}


class NoEngineSelectedError(EngineChainError):
    """Exception raised when no engine could be selected.

    This occurs in conditional chains when no condition matches,
    or when an engine selector returns None.
    """

    pass


class EngineChainConfigError(EngineChainError):
    """Exception raised for invalid chain configuration."""

    pass


# =============================================================================
# Enums
# =============================================================================


class FallbackStrategy(Enum):
    """Strategy for handling engine failures in a chain.

    Attributes:
        SEQUENTIAL: Try engines in order until one succeeds.
        FIRST_AVAILABLE: Use first engine that is healthy.
        ROUND_ROBIN: Rotate through engines for load distribution.
        RANDOM: Randomly select from available engines.
        PRIORITY: Select based on engine priority scores.
        WEIGHTED: Select based on weighted random distribution.
    """

    SEQUENTIAL = auto()
    FIRST_AVAILABLE = auto()
    ROUND_ROBIN = auto()
    RANDOM = auto()
    PRIORITY = auto()
    WEIGHTED = auto()


class ChainExecutionMode(Enum):
    """Execution mode for engine chains.

    Attributes:
        FAIL_FAST: Stop on first failure, no fallback.
        FALLBACK: Try next engine on failure.
        ALL: Execute all engines, aggregate results.
        FIRST_SUCCESS: Execute until first success.
        QUORUM: Execute until quorum of successes.
    """

    FAIL_FAST = auto()
    FALLBACK = auto()
    ALL = auto()
    FIRST_SUCCESS = auto()
    QUORUM = auto()


class FailureReason(Enum):
    """Reason for engine failure in a chain.

    Attributes:
        EXCEPTION: Engine raised an exception.
        TIMEOUT: Engine operation timed out.
        UNHEALTHY: Engine health check failed.
        RESULT_CHECK: Result did not meet criteria.
        SKIPPED: Engine was intentionally skipped.
    """

    EXCEPTION = auto()
    TIMEOUT = auto()
    UNHEALTHY = auto()
    RESULT_CHECK = auto()
    SKIPPED = auto()


# =============================================================================
# Chain Execution Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class ChainExecutionAttempt:
    """Record of a single engine execution attempt.

    Attributes:
        engine_name: Name of the engine attempted.
        success: Whether the attempt succeeded.
        failure_reason: Reason for failure if unsuccessful.
        exception: Exception if one occurred.
        duration_ms: Execution duration in milliseconds.
        result: Result if successful.
    """

    engine_name: str
    success: bool
    failure_reason: FailureReason | None = None
    exception: Exception | None = None
    duration_ms: float = 0.0
    result: CheckResult | ProfileResult | LearnResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "engine_name": self.engine_name,
            "success": self.success,
            "failure_reason": self.failure_reason.name if self.failure_reason else None,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
            "duration_ms": self.duration_ms,
            "has_result": self.result is not None,
        }


@dataclass(frozen=True, slots=True)
class ChainExecutionResult:
    """Complete result of a chain execution.

    Provides detailed information about all attempts made during
    chain execution, useful for debugging and monitoring.

    Attributes:
        chain_name: Name of the chain.
        success: Whether any engine succeeded.
        final_engine: Name of engine that produced the final result.
        attempts: All execution attempts made.
        total_duration_ms: Total chain execution time.
        result: Final result if successful.
    """

    chain_name: str
    success: bool
    final_engine: str | None = None
    attempts: tuple[ChainExecutionAttempt, ...] = ()
    total_duration_ms: float = 0.0
    result: CheckResult | ProfileResult | LearnResult | None = None

    @property
    def attempt_count(self) -> int:
        """Get total number of attempts."""
        return len(self.attempts)

    @property
    def failed_engines(self) -> tuple[str, ...]:
        """Get names of engines that failed."""
        return tuple(a.engine_name for a in self.attempts if not a.success)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_name": self.chain_name,
            "success": self.success,
            "final_engine": self.final_engine,
            "attempt_count": self.attempt_count,
            "attempts": [a.to_dict() for a in self.attempts],
            "failed_engines": list(self.failed_engines),
            "total_duration_ms": self.total_duration_ms,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class FallbackConfig:
    """Configuration for engine fallback behavior.

    Immutable configuration object for controlling how chains
    handle failures and select fallback engines.

    Attributes:
        strategy: Fallback selection strategy.
        mode: Chain execution mode.
        retry_count: Number of retries per engine before fallback.
        retry_delay_seconds: Delay between retries.
        timeout_seconds: Per-engine operation timeout.
        check_health: Whether to check engine health before use.
        skip_unhealthy: Whether to skip unhealthy engines.
        result_validator: Optional function to validate results.
        quorum_count: Required successes for QUORUM mode.
        weights: Engine weights for WEIGHTED strategy.
        priorities: Engine priorities for PRIORITY strategy.

    Example:
        >>> config = FallbackConfig(
        ...     strategy=FallbackStrategy.SEQUENTIAL,
        ...     retry_count=2,
        ...     timeout_seconds=30.0,
        ... )
    """

    strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL
    mode: ChainExecutionMode = ChainExecutionMode.FALLBACK
    retry_count: int = 1
    retry_delay_seconds: float = 0.0
    timeout_seconds: float | None = None
    check_health: bool = False
    skip_unhealthy: bool = True
    result_validator: Callable[[Any], bool] | None = None
    quorum_count: int = 1
    weights: dict[str, float] = field(default_factory=dict)
    priorities: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.quorum_count < 1:
            raise ValueError("quorum_count must be at least 1")

    def with_strategy(self, strategy: FallbackStrategy) -> Self:
        """Create config with new strategy."""
        return self._copy_with(strategy=strategy)

    def with_mode(self, mode: ChainExecutionMode) -> Self:
        """Create config with new mode."""
        return self._copy_with(mode=mode)

    def with_retry(
        self,
        count: int | None = None,
        delay_seconds: float | None = None,
    ) -> Self:
        """Create config with retry settings."""
        updates: dict[str, Any] = {}
        if count is not None:
            updates["retry_count"] = count
        if delay_seconds is not None:
            updates["retry_delay_seconds"] = delay_seconds
        return self._copy_with(**updates)

    def with_timeout(self, timeout_seconds: float) -> Self:
        """Create config with timeout."""
        return self._copy_with(timeout_seconds=timeout_seconds)

    def with_health_check(self, enabled: bool = True, skip_unhealthy: bool = True) -> Self:
        """Create config with health check settings."""
        return self._copy_with(check_health=enabled, skip_unhealthy=skip_unhealthy)

    def with_result_validator(
        self,
        validator: Callable[[Any], bool],
    ) -> Self:
        """Create config with result validator."""
        return self._copy_with(result_validator=validator)

    def with_weights(self, **weights: float) -> Self:
        """Create config with engine weights."""
        return self._copy_with(weights={**self.weights, **weights})

    def with_priorities(self, **priorities: int) -> Self:
        """Create config with engine priorities."""
        return self._copy_with(priorities={**self.priorities, **priorities})

    def _copy_with(self, **updates: Any) -> Self:
        """Create a copy with updated fields."""
        from dataclasses import fields as dataclass_fields

        current_values = {}
        for f in dataclass_fields(self):
            current_values[f.name] = getattr(self, f.name)
        current_values.update(updates)
        return self.__class__(**current_values)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy.name,
            "mode": self.mode.name,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "check_health": self.check_health,
            "skip_unhealthy": self.skip_unhealthy,
            "has_result_validator": self.result_validator is not None,
            "quorum_count": self.quorum_count,
            "weights": dict(self.weights),
            "priorities": dict(self.priorities),
            "metadata": dict(self.metadata),
        }


# Default configurations for common use cases
DEFAULT_FALLBACK_CONFIG = FallbackConfig()

RETRY_FALLBACK_CONFIG = FallbackConfig(
    strategy=FallbackStrategy.SEQUENTIAL,
    retry_count=3,
    retry_delay_seconds=1.0,
)

HEALTH_AWARE_FALLBACK_CONFIG = FallbackConfig(
    strategy=FallbackStrategy.FIRST_AVAILABLE,
    check_health=True,
    skip_unhealthy=True,
)

LOAD_BALANCED_CONFIG = FallbackConfig(
    strategy=FallbackStrategy.ROUND_ROBIN,
    check_health=True,
)

WEIGHTED_CONFIG = FallbackConfig(
    strategy=FallbackStrategy.WEIGHTED,
)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class ChainHook(Protocol):
    """Protocol for chain execution event hooks.

    Implement this to receive notifications about chain execution events.
    """

    @abstractmethod
    def on_chain_start(
        self,
        chain_name: str,
        operation: str,
        context: dict[str, Any],
    ) -> None:
        """Called when chain execution starts.

        Args:
            chain_name: Name of the chain.
            operation: Operation being executed (check, profile, learn).
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_engine_attempt(
        self,
        chain_name: str,
        engine_name: str,
        attempt: int,
        context: dict[str, Any],
    ) -> None:
        """Called before each engine attempt.

        Args:
            chain_name: Name of the chain.
            engine_name: Name of the engine being tried.
            attempt: Attempt number (1-indexed).
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_engine_success(
        self,
        chain_name: str,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when an engine succeeds.

        Args:
            chain_name: Name of the chain.
            engine_name: Name of the successful engine.
            duration_ms: Execution duration.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_engine_failure(
        self,
        chain_name: str,
        engine_name: str,
        reason: FailureReason,
        exception: Exception | None,
        context: dict[str, Any],
    ) -> None:
        """Called when an engine fails.

        Args:
            chain_name: Name of the chain.
            engine_name: Name of the failed engine.
            reason: Reason for failure.
            exception: Exception if applicable.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_fallback(
        self,
        chain_name: str,
        from_engine: str,
        to_engine: str,
        context: dict[str, Any],
    ) -> None:
        """Called when falling back to another engine.

        Args:
            chain_name: Name of the chain.
            from_engine: Name of failed engine.
            to_engine: Name of fallback engine.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_chain_complete(
        self,
        chain_name: str,
        result: ChainExecutionResult,
        context: dict[str, Any],
    ) -> None:
        """Called when chain execution completes.

        Args:
            chain_name: Name of the chain.
            result: Chain execution result.
            context: Additional context.
        """
        ...


@runtime_checkable
class EngineSelector(Protocol):
    """Protocol for dynamic engine selection.

    Implement this for custom engine selection logic based on
    data characteristics, rules, or other criteria.
    """

    @abstractmethod
    def select_engine(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        available_engines: Sequence[DataQualityEngine],
        context: dict[str, Any],
    ) -> DataQualityEngine | None:
        """Select an engine for the given data and rules.

        Args:
            data: Data to be processed.
            rules: Validation rules.
            available_engines: List of available engines.
            context: Additional context.

        Returns:
            Selected engine, or None if no suitable engine.
        """
        ...


@runtime_checkable
class EngineCondition(Protocol):
    """Protocol for conditional engine routing.

    Implement this for custom conditions that determine whether
    an engine should be used for given data/rules.
    """

    @abstractmethod
    def matches(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        context: dict[str, Any],
    ) -> bool:
        """Check if condition matches for the given data and rules.

        Args:
            data: Data to be processed.
            rules: Validation rules.
            context: Additional context.

        Returns:
            True if condition matches, False otherwise.
        """
        ...


# =============================================================================
# Hook Implementations
# =============================================================================


class LoggingChainHook:
    """Hook that logs chain execution events."""

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.engines.chain).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.engines.chain")

    def on_chain_start(
        self,
        chain_name: str,
        operation: str,
        context: dict[str, Any],
    ) -> None:
        """Log chain start."""
        self._logger.info(
            "Chain execution starting",
            chain_name=chain_name,
            operation=operation,
            **context,
        )

    def on_engine_attempt(
        self,
        chain_name: str,
        engine_name: str,
        attempt: int,
        context: dict[str, Any],
    ) -> None:
        """Log engine attempt."""
        self._logger.debug(
            "Attempting engine",
            chain_name=chain_name,
            engine_name=engine_name,
            attempt=attempt,
            **context,
        )

    def on_engine_success(
        self,
        chain_name: str,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log engine success."""
        self._logger.info(
            "Engine succeeded",
            chain_name=chain_name,
            engine_name=engine_name,
            duration_ms=duration_ms,
            **context,
        )

    def on_engine_failure(
        self,
        chain_name: str,
        engine_name: str,
        reason: FailureReason,
        exception: Exception | None,
        context: dict[str, Any],
    ) -> None:
        """Log engine failure."""
        self._logger.warning(
            "Engine failed",
            chain_name=chain_name,
            engine_name=engine_name,
            reason=reason.name,
            exception_type=type(exception).__name__ if exception else None,
            exception_message=str(exception) if exception else None,
            **context,
        )

    def on_fallback(
        self,
        chain_name: str,
        from_engine: str,
        to_engine: str,
        context: dict[str, Any],
    ) -> None:
        """Log fallback."""
        self._logger.info(
            "Falling back to next engine",
            chain_name=chain_name,
            from_engine=from_engine,
            to_engine=to_engine,
            **context,
        )

    def on_chain_complete(
        self,
        chain_name: str,
        result: ChainExecutionResult,
        context: dict[str, Any],
    ) -> None:
        """Log chain completion."""
        log_method = self._logger.info if result.success else self._logger.error
        log_method(
            "Chain execution complete",
            chain_name=chain_name,
            success=result.success,
            attempt_count=result.attempt_count,
            final_engine=result.final_engine,
            total_duration_ms=result.total_duration_ms,
            **context,
        )


class MetricsChainHook:
    """Hook that collects chain execution metrics."""

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._chain_executions: dict[str, int] = {}
        self._chain_successes: dict[str, int] = {}
        self._chain_failures: dict[str, int] = {}
        self._engine_attempts: dict[str, dict[str, int]] = {}
        self._engine_successes: dict[str, dict[str, int]] = {}
        self._engine_failures: dict[str, dict[str, int]] = {}
        self._fallback_counts: dict[str, int] = {}
        self._total_duration_ms: dict[str, float] = {}
        self._lock = threading.Lock()

    def on_chain_start(
        self,
        chain_name: str,
        operation: str,
        context: dict[str, Any],
    ) -> None:
        """Record chain start."""
        with self._lock:
            self._chain_executions[chain_name] = (
                self._chain_executions.get(chain_name, 0) + 1
            )

    def on_engine_attempt(
        self,
        chain_name: str,
        engine_name: str,
        attempt: int,
        context: dict[str, Any],
    ) -> None:
        """Record engine attempt."""
        with self._lock:
            if chain_name not in self._engine_attempts:
                self._engine_attempts[chain_name] = {}
            attempts = self._engine_attempts[chain_name]
            attempts[engine_name] = attempts.get(engine_name, 0) + 1

    def on_engine_success(
        self,
        chain_name: str,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record engine success."""
        with self._lock:
            if chain_name not in self._engine_successes:
                self._engine_successes[chain_name] = {}
            successes = self._engine_successes[chain_name]
            successes[engine_name] = successes.get(engine_name, 0) + 1

    def on_engine_failure(
        self,
        chain_name: str,
        engine_name: str,
        reason: FailureReason,
        exception: Exception | None,
        context: dict[str, Any],
    ) -> None:
        """Record engine failure."""
        with self._lock:
            if chain_name not in self._engine_failures:
                self._engine_failures[chain_name] = {}
            failures = self._engine_failures[chain_name]
            failures[engine_name] = failures.get(engine_name, 0) + 1

    def on_fallback(
        self,
        chain_name: str,
        from_engine: str,
        to_engine: str,
        context: dict[str, Any],
    ) -> None:
        """Record fallback."""
        with self._lock:
            self._fallback_counts[chain_name] = (
                self._fallback_counts.get(chain_name, 0) + 1
            )

    def on_chain_complete(
        self,
        chain_name: str,
        result: ChainExecutionResult,
        context: dict[str, Any],
    ) -> None:
        """Record chain completion."""
        with self._lock:
            if result.success:
                self._chain_successes[chain_name] = (
                    self._chain_successes.get(chain_name, 0) + 1
                )
            else:
                self._chain_failures[chain_name] = (
                    self._chain_failures.get(chain_name, 0) + 1
                )
            self._total_duration_ms[chain_name] = (
                self._total_duration_ms.get(chain_name, 0.0) + result.total_duration_ms
            )

    def get_chain_success_rate(self, chain_name: str) -> float:
        """Get success rate for a chain."""
        with self._lock:
            total = self._chain_executions.get(chain_name, 0)
            if total == 0:
                return 0.0
            successes = self._chain_successes.get(chain_name, 0)
            return successes / total

    def get_fallback_rate(self, chain_name: str) -> float:
        """Get fallback rate for a chain."""
        with self._lock:
            total = self._chain_executions.get(chain_name, 0)
            if total == 0:
                return 0.0
            fallbacks = self._fallback_counts.get(chain_name, 0)
            return fallbacks / total

    def get_engine_stats(self, chain_name: str) -> dict[str, dict[str, int]]:
        """Get engine statistics for a chain."""
        with self._lock:
            return {
                "attempts": dict(self._engine_attempts.get(chain_name, {})),
                "successes": dict(self._engine_successes.get(chain_name, {})),
                "failures": dict(self._engine_failures.get(chain_name, {})),
            }

    def get_average_duration_ms(self, chain_name: str) -> float:
        """Get average execution duration for a chain."""
        with self._lock:
            total = self._chain_executions.get(chain_name, 0)
            if total == 0:
                return 0.0
            return self._total_duration_ms.get(chain_name, 0.0) / total

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._chain_executions.clear()
            self._chain_successes.clear()
            self._chain_failures.clear()
            self._engine_attempts.clear()
            self._engine_successes.clear()
            self._engine_failures.clear()
            self._fallback_counts.clear()
            self._total_duration_ms.clear()


class CompositeChainHook:
    """Combines multiple chain hooks."""

    def __init__(self, hooks: Sequence[ChainHook] | None = None) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks: list[ChainHook] = list(hooks or [])

    def add_hook(self, hook: ChainHook) -> None:
        """Add a hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: ChainHook) -> None:
        """Remove a hook."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def _call_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call method on all hooks, suppressing exceptions."""
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                getattr(hook, method)(*args, **kwargs)

    def on_chain_start(
        self,
        chain_name: str,
        operation: str,
        context: dict[str, Any],
    ) -> None:
        """Call on_chain_start on all hooks."""
        self._call_hooks("on_chain_start", chain_name, operation, context)

    def on_engine_attempt(
        self,
        chain_name: str,
        engine_name: str,
        attempt: int,
        context: dict[str, Any],
    ) -> None:
        """Call on_engine_attempt on all hooks."""
        self._call_hooks("on_engine_attempt", chain_name, engine_name, attempt, context)

    def on_engine_success(
        self,
        chain_name: str,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_engine_success on all hooks."""
        self._call_hooks("on_engine_success", chain_name, engine_name, duration_ms, context)

    def on_engine_failure(
        self,
        chain_name: str,
        engine_name: str,
        reason: FailureReason,
        exception: Exception | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_engine_failure on all hooks."""
        self._call_hooks(
            "on_engine_failure", chain_name, engine_name, reason, exception, context
        )

    def on_fallback(
        self,
        chain_name: str,
        from_engine: str,
        to_engine: str,
        context: dict[str, Any],
    ) -> None:
        """Call on_fallback on all hooks."""
        self._call_hooks("on_fallback", chain_name, from_engine, to_engine, context)

    def on_chain_complete(
        self,
        chain_name: str,
        result: ChainExecutionResult,
        context: dict[str, Any],
    ) -> None:
        """Call on_chain_complete on all hooks."""
        self._call_hooks("on_chain_complete", chain_name, result, context)


# =============================================================================
# Engine Selection Strategies
# =============================================================================


class SequentialEngineIterator:
    """Iterator that returns engines in sequence."""

    def __init__(self, engines: Sequence[DataQualityEngine]) -> None:
        """Initialize iterator."""
        self._engines = list(engines)
        self._index = 0

    def next(self) -> DataQualityEngine | None:
        """Get next engine."""
        if self._index >= len(self._engines):
            return None
        engine = self._engines[self._index]
        self._index += 1
        return engine

    def reset(self) -> None:
        """Reset iterator to start."""
        self._index = 0


class RoundRobinEngineIterator:
    """Iterator that rotates through engines."""

    def __init__(self, engines: Sequence[DataQualityEngine]) -> None:
        """Initialize iterator."""
        self._engines = list(engines)
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> DataQualityEngine | None:
        """Get next engine in rotation."""
        if not self._engines:
            return None
        with self._lock:
            engine = self._engines[self._index % len(self._engines)]
            self._index += 1
            return engine

    def reset(self) -> None:
        """Reset is not needed for round robin."""
        pass


class WeightedRandomSelector:
    """Selector that chooses engines based on weights."""

    def __init__(
        self,
        engines: Sequence[DataQualityEngine],
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize selector.

        Args:
            engines: Available engines.
            weights: Weight for each engine by name. Default weight is 1.0.
        """
        import random

        self._engines = list(engines)
        self._weights = weights or {}
        self._random = random.Random()

    def select(self) -> DataQualityEngine | None:
        """Select engine based on weights."""
        if not self._engines:
            return None

        engine_weights = [
            self._weights.get(e.engine_name, 1.0) for e in self._engines
        ]
        total = sum(engine_weights)
        if total <= 0:
            return self._engines[0]

        r = self._random.uniform(0, total)
        cumulative = 0.0
        for engine, weight in zip(self._engines, engine_weights):
            cumulative += weight
            if r <= cumulative:
                return engine
        return self._engines[-1]


class PrioritySelector:
    """Selector that chooses engines based on priority."""

    def __init__(
        self,
        engines: Sequence[DataQualityEngine],
        priorities: dict[str, int] | None = None,
    ) -> None:
        """Initialize selector.

        Args:
            engines: Available engines.
            priorities: Priority for each engine by name. Higher = more preferred.
        """
        self._priorities = priorities or {}
        self._sorted_engines = sorted(
            engines,
            key=lambda e: self._priorities.get(e.engine_name, 0),
            reverse=True,
        )
        self._index = 0

    def next(self) -> DataQualityEngine | None:
        """Get next engine by priority."""
        if self._index >= len(self._sorted_engines):
            return None
        engine = self._sorted_engines[self._index]
        self._index += 1
        return engine

    def reset(self) -> None:
        """Reset to highest priority engine."""
        self._index = 0


# =============================================================================
# Engine Chain Implementation
# =============================================================================


class EngineChain(EngineInfoMixin):
    """Chain of engines with fallback support.

    Executes engines in sequence, falling back to the next engine
    when one fails. Implements DataQualityEngine protocol, so it
    can be used anywhere a single engine is expected.

    Features:
        - Configurable fallback strategies
        - Retry support per engine
        - Health check integration
        - Result validation
        - Comprehensive execution tracking
        - Hook-based observability

    Example:
        >>> # Simple fallback chain
        >>> chain = EngineChain([primary, backup])
        >>> result = chain.check(data, rules)  # Falls back on failure

        >>> # With configuration
        >>> chain = EngineChain(
        ...     engines=[primary, secondary, tertiary],
        ...     config=FallbackConfig(
        ...         retry_count=2,
        ...         check_health=True,
        ...     ),
        ...     name="production_chain",
        ... )

        >>> # Using context manager
        >>> with chain:
        ...     result = chain.check(data, rules)

        >>> # Access execution details
        >>> result = chain.check(data, rules)
        >>> exec_result = chain.last_execution_result
        >>> print(f"Tried {exec_result.attempt_count} engines")
    """

    def __init__(
        self,
        engines: Sequence[DataQualityEngine],
        config: FallbackConfig | None = None,
        hooks: Sequence[ChainHook] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize engine chain.

        Args:
            engines: Sequence of engines to chain (in priority order).
            config: Fallback configuration.
            hooks: Chain execution hooks.
            name: Optional chain name (default: "engine_chain").

        Raises:
            EngineChainConfigError: If engines sequence is empty.
        """
        if not engines:
            raise EngineChainConfigError(
                "EngineChain requires at least one engine",
                chain_name=name,
            )

        self._engines = list(engines)
        self._config = config or DEFAULT_FALLBACK_CONFIG
        self._hook: ChainHook | None = None
        if hooks:
            self._hook = CompositeChainHook(list(hooks))
        self._name = name or "engine_chain"
        self._last_execution_result: ChainExecutionResult | None = None
        self._round_robin_index = 0
        self._lock = threading.RLock()

    @property
    def engine_name(self) -> str:
        """Return chain name."""
        return self._name

    @property
    def engine_version(self) -> str:
        """Return version based on contained engines."""
        return "chain:1.0.0"

    @property
    def engines(self) -> tuple[DataQualityEngine, ...]:
        """Get the engines in this chain."""
        return tuple(self._engines)

    @property
    def config(self) -> FallbackConfig:
        """Get the chain configuration."""
        return self._config

    @property
    def last_execution_result(self) -> ChainExecutionResult | None:
        """Get the result of the last chain execution."""
        return self._last_execution_result

    def _get_capabilities(self) -> EngineCapabilities:
        """Return combined capabilities of all engines."""
        all_data_types: set[str] = set()
        all_rule_types: set[str] = set()

        for engine in self._engines:
            if hasattr(engine, "get_capabilities"):
                caps = engine.get_capabilities()
                all_data_types.update(caps.supported_data_types)
                all_rule_types.update(caps.supported_rule_types)

        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
            supported_data_types=tuple(sorted(all_data_types)),
            supported_rule_types=tuple(sorted(all_rule_types)),
            extra={"chain_length": len(self._engines)},
        )

    def _get_description(self) -> str:
        """Return chain description."""
        engine_names = [e.engine_name for e in self._engines]
        return f"Engine chain with {len(self._engines)} engines: {', '.join(engine_names)}"

    def _create_context(self) -> dict[str, Any]:
        """Create context dictionary for hooks."""
        return {
            "chain_name": self._name,
            "engine_count": len(self._engines),
            "config": self._config.to_dict(),
        }

    def _get_next_engine_iterator(self) -> SequentialEngineIterator | RoundRobinEngineIterator | WeightedRandomSelector | PrioritySelector:
        """Get engine iterator based on strategy."""
        strategy = self._config.strategy

        if strategy == FallbackStrategy.SEQUENTIAL:
            return SequentialEngineIterator(self._engines)
        elif strategy == FallbackStrategy.ROUND_ROBIN:
            return RoundRobinEngineIterator(self._engines)
        elif strategy == FallbackStrategy.WEIGHTED:
            return WeightedRandomSelector(self._engines, self._config.weights)
        elif strategy == FallbackStrategy.PRIORITY:
            return PrioritySelector(self._engines, self._config.priorities)
        elif strategy == FallbackStrategy.RANDOM:
            import random
            shuffled = list(self._engines)
            random.shuffle(shuffled)
            return SequentialEngineIterator(shuffled)
        elif strategy == FallbackStrategy.FIRST_AVAILABLE:
            return SequentialEngineIterator(self._engines)
        else:
            return SequentialEngineIterator(self._engines)

    def _check_engine_health(self, engine: DataQualityEngine) -> bool:
        """Check if engine is healthy.

        Args:
            engine: Engine to check.

        Returns:
            True if engine is healthy or health check disabled.
        """
        if not self._config.check_health:
            return True

        from common.engines.lifecycle import ManagedEngine
        from common.health import HealthStatus

        if isinstance(engine, ManagedEngine):
            try:
                result = engine.health_check()
                return result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            except Exception:
                return False

        return True

    def _validate_result(self, result: Any) -> bool:
        """Validate result using configured validator.

        Args:
            result: Result to validate.

        Returns:
            True if result is valid or no validator configured.
        """
        if self._config.result_validator is None:
            return True

        try:
            return self._config.result_validator(result)
        except Exception:
            return False

    def _execute_with_retry(
        self,
        engine: DataQualityEngine,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any | None, Exception | None, float]:
        """Execute operation with retry logic.

        Args:
            engine: Engine to use.
            operation: Operation to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (result, exception, duration_ms).
        """
        last_exception: Exception | None = None
        total_duration = 0.0

        for attempt in range(1, self._config.retry_count + 1):
            if self._hook:
                self._hook.on_engine_attempt(
                    self._name,
                    engine.engine_name,
                    attempt,
                    self._create_context(),
                )

            start_time = time.perf_counter()
            try:
                result = operation(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                total_duration += duration_ms

                if not self._validate_result(result):
                    last_exception = ValueError("Result validation failed")
                    if self._hook:
                        self._hook.on_engine_failure(
                            self._name,
                            engine.engine_name,
                            FailureReason.RESULT_CHECK,
                            last_exception,
                            self._create_context(),
                        )
                    if attempt < self._config.retry_count and self._config.retry_delay_seconds > 0:
                        time.sleep(self._config.retry_delay_seconds)
                    continue

                return result, None, total_duration

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                total_duration += duration_ms
                last_exception = e

                if self._hook:
                    self._hook.on_engine_failure(
                        self._name,
                        engine.engine_name,
                        FailureReason.EXCEPTION,
                        e,
                        self._create_context(),
                    )

                if attempt < self._config.retry_count and self._config.retry_delay_seconds > 0:
                    time.sleep(self._config.retry_delay_seconds)

        return None, last_exception, total_duration

    def _execute_chain(
        self,
        operation_name: str,
        operation_getter: Callable[[DataQualityEngine], Callable[..., Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute operation across engine chain.

        Args:
            operation_name: Name of operation (check, profile, learn).
            operation_getter: Function to get operation from engine.
            *args: Positional arguments for operation.
            **kwargs: Keyword arguments for operation.

        Returns:
            Result from successful engine.

        Raises:
            AllEnginesFailedError: If all engines fail.
        """
        context = self._create_context()
        context["operation"] = operation_name

        if self._hook:
            self._hook.on_chain_start(self._name, operation_name, context)

        start_time = time.perf_counter()
        attempts: list[ChainExecutionAttempt] = []
        exceptions: dict[str, Exception] = {}
        previous_engine: str | None = None

        iterator = self._get_next_engine_iterator()
        tried_engines: set[str] = set()

        while True:
            if isinstance(iterator, WeightedRandomSelector):
                engine = iterator.select()
            else:
                engine = iterator.next()

            if engine is None:
                break

            if engine.engine_name in tried_engines:
                break
            tried_engines.add(engine.engine_name)

            if self._config.check_health and self._config.skip_unhealthy:
                if not self._check_engine_health(engine):
                    attempts.append(ChainExecutionAttempt(
                        engine_name=engine.engine_name,
                        success=False,
                        failure_reason=FailureReason.UNHEALTHY,
                    ))
                    if self._hook:
                        self._hook.on_engine_failure(
                            self._name,
                            engine.engine_name,
                            FailureReason.UNHEALTHY,
                            None,
                            context,
                        )
                    continue

            if previous_engine is not None and self._hook:
                self._hook.on_fallback(
                    self._name,
                    previous_engine,
                    engine.engine_name,
                    context,
                )

            operation = operation_getter(engine)
            result, exception, duration_ms = self._execute_with_retry(
                engine, operation, *args, **kwargs
            )

            if exception is None:
                attempts.append(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=True,
                    duration_ms=duration_ms,
                    result=result,
                ))

                if self._hook:
                    self._hook.on_engine_success(
                        self._name,
                        engine.engine_name,
                        duration_ms,
                        context,
                    )

                total_duration_ms = (time.perf_counter() - start_time) * 1000
                exec_result = ChainExecutionResult(
                    chain_name=self._name,
                    success=True,
                    final_engine=engine.engine_name,
                    attempts=tuple(attempts),
                    total_duration_ms=total_duration_ms,
                    result=result,
                )

                if self._hook:
                    self._hook.on_chain_complete(self._name, exec_result, context)

                with self._lock:
                    self._last_execution_result = exec_result

                return result
            else:
                attempts.append(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=False,
                    failure_reason=FailureReason.EXCEPTION,
                    exception=exception,
                    duration_ms=duration_ms,
                ))
                exceptions[engine.engine_name] = exception
                previous_engine = engine.engine_name

                if self._config.mode == ChainExecutionMode.FAIL_FAST:
                    break

        total_duration_ms = (time.perf_counter() - start_time) * 1000
        exec_result = ChainExecutionResult(
            chain_name=self._name,
            success=False,
            attempts=tuple(attempts),
            total_duration_ms=total_duration_ms,
        )

        if self._hook:
            self._hook.on_chain_complete(self._name, exec_result, context)

        with self._lock:
            self._last_execution_result = exec_result

        attempted_engines = tuple(a.engine_name for a in attempts)
        raise AllEnginesFailedError(
            f"All {len(attempted_engines)} engines in chain '{self._name}' failed",
            chain_name=self._name,
            attempted_engines=attempted_engines,
            exceptions=exceptions,
        )

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation check across engine chain.

        Args:
            data: Data to validate.
            rules: Validation rules.
            **kwargs: Engine-specific parameters.

        Returns:
            CheckResult from first successful engine.

        Raises:
            AllEnginesFailedError: If all engines fail.
        """
        return self._execute_chain(
            "check",
            lambda e: lambda d, r, **kw: e.check(d, r, **kw),
            data,
            rules,
            **kwargs,
        )

    def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Execute profiling across engine chain.

        Args:
            data: Data to profile.
            **kwargs: Engine-specific parameters.

        Returns:
            ProfileResult from first successful engine.

        Raises:
            AllEnginesFailedError: If all engines fail.
        """
        return self._execute_chain(
            "profile",
            lambda e: lambda d, **kw: e.profile(d, **kw),
            data,
            **kwargs,
        )

    def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Execute rule learning across engine chain.

        Args:
            data: Data to learn from.
            **kwargs: Engine-specific parameters.

        Returns:
            LearnResult from first successful engine.

        Raises:
            AllEnginesFailedError: If all engines fail.
        """
        return self._execute_chain(
            "learn",
            lambda e: lambda d, **kw: e.learn(d, **kw),
            data,
            **kwargs,
        )

    def detect_drift(
        self,
        baseline: Any,
        current: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute drift detection across engine chain.

        Only engines implementing DriftDetectionEngine are considered.
        Non-drift engines are automatically skipped.

        Args:
            baseline: Baseline data for comparison.
            current: Current data to check for drift.
            **kwargs: Engine-specific parameters.

        Returns:
            DriftResult from first successful drift-capable engine.

        Raises:
            NoEngineSelectedError: If no engines support drift detection.
            AllEnginesFailedError: If all drift-capable engines fail.
        """
        drift_engines = [e for e in self._engines if supports_drift(e)]
        if not drift_engines:
            raise NoEngineSelectedError(
                f"No engines in chain '{self._name}' support drift detection",
                chain_name=self._name,
            )

        # Temporarily swap engines to only drift-capable ones
        original_engines = self._engines
        self._engines = drift_engines
        try:
            return self._execute_chain(
                "detect_drift",
                lambda e: lambda b, c, **kw: e.detect_drift(b, c, **kw),
                baseline,
                current,
                **kwargs,
            )
        finally:
            self._engines = original_engines

    def detect_anomalies(
        self,
        data: Any,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Execute anomaly detection across engine chain.

        Only engines implementing AnomalyDetectionEngine are considered.
        Non-anomaly engines are automatically skipped.

        Args:
            data: Data to check for anomalies.
            **kwargs: Engine-specific parameters.

        Returns:
            AnomalyResult from first successful anomaly-capable engine.

        Raises:
            NoEngineSelectedError: If no engines support anomaly detection.
            AllEnginesFailedError: If all anomaly-capable engines fail.
        """
        anomaly_engines = [e for e in self._engines if supports_anomaly(e)]
        if not anomaly_engines:
            raise NoEngineSelectedError(
                f"No engines in chain '{self._name}' support anomaly detection",
                chain_name=self._name,
            )

        original_engines = self._engines
        self._engines = anomaly_engines
        try:
            return self._execute_chain(
                "detect_anomalies",
                lambda e: lambda d, **kw: e.detect_anomalies(d, **kw),
                data,
                **kwargs,
            )
        finally:
            self._engines = original_engines

    def add_engine(
        self,
        engine: DataQualityEngine,
        position: int | None = None,
    ) -> None:
        """Add an engine to the chain.

        Args:
            engine: Engine to add.
            position: Position in chain (default: append at end).
        """
        with self._lock:
            if position is None:
                self._engines.append(engine)
            else:
                self._engines.insert(position, engine)

    def remove_engine(self, engine_name: str) -> bool:
        """Remove an engine from the chain.

        Args:
            engine_name: Name of engine to remove.

        Returns:
            True if engine was removed, False if not found.
        """
        with self._lock:
            for i, engine in enumerate(self._engines):
                if engine.engine_name == engine_name:
                    del self._engines[i]
                    return True
            return False

    def __enter__(self) -> Self:
        """Enter context manager - start all engines that support it."""
        from common.engines.lifecycle import ManagedEngine

        for engine in self._engines:
            if isinstance(engine, ManagedEngine):
                engine.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager - stop all engines that support it."""
        from common.engines.lifecycle import ManagedEngine

        for engine in self._engines:
            if isinstance(engine, ManagedEngine):
                try:
                    engine.stop()
                except Exception:
                    pass


# =============================================================================
# Conditional Engine Chain
# =============================================================================


@dataclass
class ConditionalRoute:
    """A route in a conditional engine chain.

    Attributes:
        condition: Condition that must match.
        engine: Engine to use when condition matches.
        priority: Priority for route ordering (higher = earlier).
        name: Optional name for the route.
    """

    condition: EngineCondition | Callable[[Any, Sequence[Mapping[str, Any]]], bool]
    engine: DataQualityEngine
    priority: int = 0
    name: str | None = None


class ConditionalEngineChain(EngineInfoMixin):
    """Engine chain that routes based on conditions.

    Routes requests to different engines based on data/rule characteristics.
    Useful for selecting specialized engines for different scenarios.

    Example:
        >>> chain = ConditionalEngineChain(name="smart_router")
        >>> chain.add_route(
        ...     lambda data, rules: len(data) > 1_000_000,
        ...     heavy_engine,
        ...     name="large_data",
        ... )
        >>> chain.add_route(
        ...     lambda data, rules: any(r.get("type") == "regex" for r in rules),
        ...     regex_engine,
        ...     name="regex_rules",
        ... )
        >>> chain.set_default_engine(general_engine)
        >>> result = chain.check(data, rules)  # Routes to appropriate engine
    """

    def __init__(
        self,
        routes: Sequence[ConditionalRoute] | None = None,
        default_engine: DataQualityEngine | None = None,
        hooks: Sequence[ChainHook] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize conditional chain.

        Args:
            routes: Initial routes.
            default_engine: Fallback engine when no condition matches.
            hooks: Chain execution hooks.
            name: Chain name.
        """
        self._routes: list[ConditionalRoute] = list(routes or [])
        self._default_engine = default_engine
        self._hook: ChainHook | None = None
        if hooks:
            self._hook = CompositeChainHook(list(hooks))
        self._name = name or "conditional_chain"
        self._last_execution_result: ChainExecutionResult | None = None
        self._lock = threading.RLock()

    @property
    def engine_name(self) -> str:
        """Return chain name."""
        return self._name

    @property
    def engine_version(self) -> str:
        """Return version."""
        return "conditional_chain:1.0.0"

    @property
    def routes(self) -> tuple[ConditionalRoute, ...]:
        """Get all routes."""
        return tuple(self._routes)

    @property
    def default_engine(self) -> DataQualityEngine | None:
        """Get default engine."""
        return self._default_engine

    @property
    def last_execution_result(self) -> ChainExecutionResult | None:
        """Get last execution result."""
        return self._last_execution_result

    def _get_capabilities(self) -> EngineCapabilities:
        """Return combined capabilities."""
        all_engines = [r.engine for r in self._routes]
        if self._default_engine:
            all_engines.append(self._default_engine)

        all_data_types: set[str] = set()
        all_rule_types: set[str] = set()

        for engine in all_engines:
            if hasattr(engine, "get_capabilities"):
                caps = engine.get_capabilities()
                all_data_types.update(caps.supported_data_types)
                all_rule_types.update(caps.supported_rule_types)

        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
            supported_data_types=tuple(sorted(all_data_types)),
            supported_rule_types=tuple(sorted(all_rule_types)),
            extra={"route_count": len(self._routes)},
        )

    def add_route(
        self,
        condition: EngineCondition | Callable[[Any, Sequence[Mapping[str, Any]]], bool],
        engine: DataQualityEngine,
        priority: int = 0,
        name: str | None = None,
    ) -> None:
        """Add a conditional route.

        Args:
            condition: Condition for this route.
            engine: Engine to use when condition matches.
            priority: Route priority (higher = evaluated earlier).
            name: Optional route name.
        """
        route = ConditionalRoute(
            condition=condition,
            engine=engine,
            priority=priority,
            name=name,
        )
        with self._lock:
            self._routes.append(route)
            self._routes.sort(key=lambda r: r.priority, reverse=True)

    def remove_route(self, name: str) -> bool:
        """Remove a route by name.

        Args:
            name: Name of route to remove.

        Returns:
            True if route was removed.
        """
        with self._lock:
            for i, route in enumerate(self._routes):
                if route.name == name:
                    del self._routes[i]
                    return True
            return False

    def set_default_engine(self, engine: DataQualityEngine) -> None:
        """Set the default fallback engine.

        Args:
            engine: Engine to use when no condition matches.
        """
        with self._lock:
            self._default_engine = engine

    def _select_engine(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
    ) -> DataQualityEngine | None:
        """Select engine based on conditions.

        Args:
            data: Data to process.
            rules: Validation rules.

        Returns:
            Selected engine or None.
        """
        context: dict[str, Any] = {}

        for route in self._routes:
            condition = route.condition
            try:
                if isinstance(condition, EngineCondition):
                    matches = condition.matches(data, rules, context)
                else:
                    matches = condition(data, rules)

                if matches:
                    return route.engine
            except Exception:
                continue

        return self._default_engine

    def _execute(
        self,
        operation_name: str,
        data: Any,
        operation: Callable[[DataQualityEngine, Any], Any],
    ) -> Any:
        """Execute operation with selected engine.

        Args:
            operation_name: Name of operation.
            data: Data (includes rules for check).
            operation: Operation to execute.

        Returns:
            Operation result.

        Raises:
            NoEngineSelectedError: If no engine could be selected.
        """
        start_time = time.perf_counter()
        context = {
            "chain_name": self._name,
            "operation": operation_name,
            "route_count": len(self._routes),
        }

        if self._hook:
            self._hook.on_chain_start(self._name, operation_name, context)

        if isinstance(data, tuple) and len(data) == 2:
            actual_data, rules = data
        else:
            actual_data = data
            rules = []

        engine = self._select_engine(actual_data, rules)

        if engine is None:
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            exec_result = ChainExecutionResult(
                chain_name=self._name,
                success=False,
                total_duration_ms=total_duration_ms,
            )
            with self._lock:
                self._last_execution_result = exec_result

            if self._hook:
                self._hook.on_chain_complete(self._name, exec_result, context)

            raise NoEngineSelectedError(
                f"No engine selected for chain '{self._name}'",
                chain_name=self._name,
            )

        if self._hook:
            self._hook.on_engine_attempt(self._name, engine.engine_name, 1, context)

        try:
            result = operation(engine, data)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                self._hook.on_engine_success(
                    self._name, engine.engine_name, duration_ms, context
                )

            exec_result = ChainExecutionResult(
                chain_name=self._name,
                success=True,
                final_engine=engine.engine_name,
                attempts=(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=True,
                    duration_ms=duration_ms,
                    result=result,
                ),),
                total_duration_ms=duration_ms,
                result=result,
            )

            if self._hook:
                self._hook.on_chain_complete(self._name, exec_result, context)

            with self._lock:
                self._last_execution_result = exec_result

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                self._hook.on_engine_failure(
                    self._name,
                    engine.engine_name,
                    FailureReason.EXCEPTION,
                    e,
                    context,
                )

            exec_result = ChainExecutionResult(
                chain_name=self._name,
                success=False,
                attempts=(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=False,
                    failure_reason=FailureReason.EXCEPTION,
                    exception=e,
                    duration_ms=duration_ms,
                ),),
                total_duration_ms=duration_ms,
            )

            if self._hook:
                self._hook.on_chain_complete(self._name, exec_result, context)

            with self._lock:
                self._last_execution_result = exec_result

            raise

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute check with conditionally selected engine."""
        return self._execute(
            "check",
            (data, rules),
            lambda e, d: e.check(d[0], d[1], **kwargs),
        )

    def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Execute profile with conditionally selected engine."""
        return self._execute(
            "profile",
            (data, []),
            lambda e, d: e.profile(d[0], **kwargs),
        )

    def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Execute learn with conditionally selected engine."""
        return self._execute(
            "learn",
            (data, []),
            lambda e, d: e.learn(d[0], **kwargs),
        )

    def detect_drift(
        self,
        baseline: Any,
        current: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute drift detection with conditionally selected engine.

        Raises:
            NoEngineSelectedError: If selected engine doesn't support drift.
        """
        def _op(e: Any, d: tuple[Any, ...]) -> DriftResult:
            if not supports_drift(e):
                raise NoEngineSelectedError(
                    f"Selected engine '{e.engine_name}' does not support drift detection",
                    chain_name=self._name,
                )
            return e.detect_drift(d[0], d[1], **kwargs)

        return self._execute("detect_drift", (baseline, current), _op)

    def detect_anomalies(
        self,
        data: Any,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Execute anomaly detection with conditionally selected engine.

        Raises:
            NoEngineSelectedError: If selected engine doesn't support anomaly.
        """
        def _op(e: Any, d: tuple[Any, ...]) -> AnomalyResult:
            if not supports_anomaly(e):
                raise NoEngineSelectedError(
                    f"Selected engine '{e.engine_name}' does not support anomaly detection",
                    chain_name=self._name,
                )
            return e.detect_anomalies(d[0], **kwargs)

        return self._execute("detect_anomalies", (data, []), _op)

    def __enter__(self) -> Self:
        """Enter context manager."""
        from common.engines.lifecycle import ManagedEngine

        all_engines = [r.engine for r in self._routes]
        if self._default_engine:
            all_engines.append(self._default_engine)

        for engine in all_engines:
            if isinstance(engine, ManagedEngine):
                engine.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        from common.engines.lifecycle import ManagedEngine

        all_engines = [r.engine for r in self._routes]
        if self._default_engine:
            all_engines.append(self._default_engine)

        for engine in all_engines:
            if isinstance(engine, ManagedEngine):
                try:
                    engine.stop()
                except Exception:
                    pass


# =============================================================================
# Selector-Based Engine Chain
# =============================================================================


class SelectorEngineChain(EngineInfoMixin):
    """Engine chain using custom selector logic.

    Delegates engine selection to an EngineSelector implementation,
    allowing for complex selection logic based on data characteristics.

    Example:
        >>> class DataSizeSelector:
        ...     def select_engine(self, data, rules, engines, context):
        ...         if len(data) > 1_000_000:
        ...             return next((e for e in engines if "heavy" in e.engine_name), None)
        ...         return engines[0] if engines else None
        ...
        >>> chain = SelectorEngineChain(
        ...     engines=[light_engine, heavy_engine],
        ...     selector=DataSizeSelector(),
        ... )
    """

    def __init__(
        self,
        engines: Sequence[DataQualityEngine],
        selector: EngineSelector,
        fallback_chain: EngineChain | None = None,
        hooks: Sequence[ChainHook] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize selector chain.

        Args:
            engines: Available engines.
            selector: Engine selector implementation.
            fallback_chain: Chain to use if selector returns None.
            hooks: Chain execution hooks.
            name: Chain name.
        """
        self._engines = list(engines)
        self._selector = selector
        self._fallback_chain = fallback_chain
        self._hook: ChainHook | None = None
        if hooks:
            self._hook = CompositeChainHook(list(hooks))
        self._name = name or "selector_chain"
        self._last_execution_result: ChainExecutionResult | None = None
        self._lock = threading.RLock()

    @property
    def engine_name(self) -> str:
        """Return chain name."""
        return self._name

    @property
    def engine_version(self) -> str:
        """Return version."""
        return "selector_chain:1.0.0"

    @property
    def engines(self) -> tuple[DataQualityEngine, ...]:
        """Get available engines."""
        return tuple(self._engines)

    @property
    def last_execution_result(self) -> ChainExecutionResult | None:
        """Get last execution result."""
        return self._last_execution_result

    def _get_capabilities(self) -> EngineCapabilities:
        """Return combined capabilities."""
        all_data_types: set[str] = set()
        all_rule_types: set[str] = set()

        for engine in self._engines:
            if hasattr(engine, "get_capabilities"):
                caps = engine.get_capabilities()
                all_data_types.update(caps.supported_data_types)
                all_rule_types.update(caps.supported_rule_types)

        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
            supported_data_types=tuple(sorted(all_data_types)),
            supported_rule_types=tuple(sorted(all_rule_types)),
        )

    def _execute(
        self,
        operation_name: str,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        operation: Callable[[DataQualityEngine], Any],
    ) -> Any:
        """Execute with selected engine.

        Args:
            operation_name: Operation name.
            data: Data to process.
            rules: Validation rules.
            operation: Operation to execute.

        Returns:
            Operation result.
        """
        start_time = time.perf_counter()
        context = {
            "chain_name": self._name,
            "operation": operation_name,
            "engine_count": len(self._engines),
        }

        if self._hook:
            self._hook.on_chain_start(self._name, operation_name, context)

        engine = self._selector.select_engine(data, rules, self._engines, context)

        if engine is None:
            if self._fallback_chain is not None:
                return operation(self._fallback_chain)  # type: ignore

            total_duration_ms = (time.perf_counter() - start_time) * 1000
            exec_result = ChainExecutionResult(
                chain_name=self._name,
                success=False,
                total_duration_ms=total_duration_ms,
            )

            if self._hook:
                self._hook.on_chain_complete(self._name, exec_result, context)

            with self._lock:
                self._last_execution_result = exec_result

            raise NoEngineSelectedError(
                f"Selector returned no engine for chain '{self._name}'",
                chain_name=self._name,
            )

        if self._hook:
            self._hook.on_engine_attempt(self._name, engine.engine_name, 1, context)

        try:
            result = operation(engine)
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                self._hook.on_engine_success(
                    self._name, engine.engine_name, duration_ms, context
                )

            exec_result = ChainExecutionResult(
                chain_name=self._name,
                success=True,
                final_engine=engine.engine_name,
                attempts=(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=True,
                    duration_ms=duration_ms,
                    result=result,
                ),),
                total_duration_ms=duration_ms,
                result=result,
            )

            if self._hook:
                self._hook.on_chain_complete(self._name, exec_result, context)

            with self._lock:
                self._last_execution_result = exec_result

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                self._hook.on_engine_failure(
                    self._name,
                    engine.engine_name,
                    FailureReason.EXCEPTION,
                    e,
                    context,
                )

            if self._fallback_chain is not None:
                return operation(self._fallback_chain)  # type: ignore

            exec_result = ChainExecutionResult(
                chain_name=self._name,
                success=False,
                attempts=(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=False,
                    failure_reason=FailureReason.EXCEPTION,
                    exception=e,
                    duration_ms=duration_ms,
                ),),
                total_duration_ms=duration_ms,
            )

            if self._hook:
                self._hook.on_chain_complete(self._name, exec_result, context)

            with self._lock:
                self._last_execution_result = exec_result

            raise

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute check with selected engine."""
        return self._execute(
            "check",
            data,
            rules,
            lambda e: e.check(data, rules, **kwargs),
        )

    def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Execute profile with selected engine."""
        return self._execute(
            "profile",
            data,
            [],
            lambda e: e.profile(data, **kwargs),
        )

    def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Execute learn with selected engine."""
        return self._execute(
            "learn",
            data,
            [],
            lambda e: e.learn(data, **kwargs),
        )

    def detect_drift(
        self,
        baseline: Any,
        current: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute drift detection with selected engine."""
        def _op(e: Any) -> DriftResult:
            if not supports_drift(e):
                raise NoEngineSelectedError(
                    f"Selected engine '{e.engine_name}' does not support drift detection",
                    chain_name=self._name,
                )
            return e.detect_drift(baseline, current, **kwargs)

        return self._execute("detect_drift", baseline, [], _op)

    def detect_anomalies(
        self,
        data: Any,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Execute anomaly detection with selected engine."""
        def _op(e: Any) -> AnomalyResult:
            if not supports_anomaly(e):
                raise NoEngineSelectedError(
                    f"Selected engine '{e.engine_name}' does not support anomaly detection",
                    chain_name=self._name,
                )
            return e.detect_anomalies(data, **kwargs)

        return self._execute("detect_anomalies", data, [], _op)


# =============================================================================
# Async Engine Chain
# =============================================================================


class AsyncEngineChain:
    """Async version of EngineChain for async engines.

    Provides the same functionality as EngineChain but with async methods
    for use with AsyncDataQualityEngine implementations.

    Example:
        >>> chain = AsyncEngineChain([async_primary, async_backup])
        >>> result = await chain.check(data, rules)
    """

    def __init__(
        self,
        engines: Sequence[Any],  # AsyncDataQualityEngine
        config: FallbackConfig | None = None,
        hooks: Sequence[ChainHook] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize async chain.

        Args:
            engines: Sequence of async engines.
            config: Fallback configuration.
            hooks: Chain execution hooks.
            name: Chain name.
        """
        if not engines:
            raise EngineChainConfigError(
                "AsyncEngineChain requires at least one engine",
                chain_name=name,
            )

        self._engines = list(engines)
        self._config = config or DEFAULT_FALLBACK_CONFIG
        self._hook: ChainHook | None = None
        if hooks:
            self._hook = CompositeChainHook(list(hooks))
        self._name = name or "async_engine_chain"
        self._last_execution_result: ChainExecutionResult | None = None
        self._lock = threading.RLock()

    @property
    def engine_name(self) -> str:
        """Return chain name."""
        return self._name

    @property
    def engine_version(self) -> str:
        """Return version."""
        return "async_chain:1.0.0"

    @property
    def engines(self) -> tuple[Any, ...]:
        """Get engines."""
        return tuple(self._engines)

    @property
    def last_execution_result(self) -> ChainExecutionResult | None:
        """Get last execution result."""
        return self._last_execution_result

    async def _execute_with_retry(
        self,
        engine: Any,
        operation: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any | None, Exception | None, float]:
        """Execute async operation with retry."""
        last_exception: Exception | None = None
        total_duration = 0.0

        for attempt in range(1, self._config.retry_count + 1):
            if self._hook:
                self._hook.on_engine_attempt(
                    self._name,
                    engine.engine_name,
                    attempt,
                    {"operation": "async"},
                )

            start_time = time.perf_counter()
            try:
                result = await operation(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                total_duration += duration_ms
                return result, None, total_duration

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                total_duration += duration_ms
                last_exception = e

                if self._hook:
                    self._hook.on_engine_failure(
                        self._name,
                        engine.engine_name,
                        FailureReason.EXCEPTION,
                        e,
                        {"operation": "async"},
                    )

                if (
                    attempt < self._config.retry_count
                    and self._config.retry_delay_seconds > 0
                ):
                    await asyncio.sleep(self._config.retry_delay_seconds)

        return None, last_exception, total_duration

    async def _execute_chain(
        self,
        operation_name: str,
        operation_getter: Callable[[Any], Callable[..., Awaitable[Any]]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute async operation across chain."""
        context = {"chain_name": self._name, "operation": operation_name}

        if self._hook:
            self._hook.on_chain_start(self._name, operation_name, context)

        start_time = time.perf_counter()
        attempts: list[ChainExecutionAttempt] = []
        exceptions: dict[str, Exception] = {}
        previous_engine: str | None = None

        for engine in self._engines:
            if previous_engine is not None and self._hook:
                self._hook.on_fallback(
                    self._name,
                    previous_engine,
                    engine.engine_name,
                    context,
                )

            operation = operation_getter(engine)
            result, exception, duration_ms = await self._execute_with_retry(
                engine, operation, *args, **kwargs
            )

            if exception is None:
                attempts.append(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=True,
                    duration_ms=duration_ms,
                    result=result,
                ))

                if self._hook:
                    self._hook.on_engine_success(
                        self._name, engine.engine_name, duration_ms, context
                    )

                total_duration_ms = (time.perf_counter() - start_time) * 1000
                exec_result = ChainExecutionResult(
                    chain_name=self._name,
                    success=True,
                    final_engine=engine.engine_name,
                    attempts=tuple(attempts),
                    total_duration_ms=total_duration_ms,
                    result=result,
                )

                if self._hook:
                    self._hook.on_chain_complete(self._name, exec_result, context)

                with self._lock:
                    self._last_execution_result = exec_result

                return result
            else:
                attempts.append(ChainExecutionAttempt(
                    engine_name=engine.engine_name,
                    success=False,
                    failure_reason=FailureReason.EXCEPTION,
                    exception=exception,
                    duration_ms=duration_ms,
                ))
                exceptions[engine.engine_name] = exception
                previous_engine = engine.engine_name

                if self._config.mode == ChainExecutionMode.FAIL_FAST:
                    break

        total_duration_ms = (time.perf_counter() - start_time) * 1000
        exec_result = ChainExecutionResult(
            chain_name=self._name,
            success=False,
            attempts=tuple(attempts),
            total_duration_ms=total_duration_ms,
        )

        if self._hook:
            self._hook.on_chain_complete(self._name, exec_result, context)

        with self._lock:
            self._last_execution_result = exec_result

        attempted_engines = tuple(a.engine_name for a in attempts)
        raise AllEnginesFailedError(
            f"All engines in async chain '{self._name}' failed",
            chain_name=self._name,
            attempted_engines=attempted_engines,
            exceptions=exceptions,
        )

    async def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute async check across chain."""
        return await self._execute_chain(
            "check",
            lambda e: lambda d, r, **kw: e.check(d, r, **kw),
            data,
            rules,
            **kwargs,
        )

    async def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Execute async profile across chain."""
        return await self._execute_chain(
            "profile",
            lambda e: lambda d, **kw: e.profile(d, **kw),
            data,
            **kwargs,
        )

    async def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Execute async learn across chain."""
        return await self._execute_chain(
            "learn",
            lambda e: lambda d, **kw: e.learn(d, **kw),
            data,
            **kwargs,
        )

    async def detect_drift(
        self,
        baseline: Any,
        current: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute async drift detection across chain.

        Only engines implementing DriftDetectionEngine are considered.

        Raises:
            NoEngineSelectedError: If no engines support drift detection.
            AllEnginesFailedError: If all drift-capable engines fail.
        """
        drift_engines = [e for e in self._engines if supports_drift(e)]
        if not drift_engines:
            raise NoEngineSelectedError(
                f"No engines in async chain '{self._name}' support drift detection",
                chain_name=self._name,
            )

        original_engines = self._engines
        self._engines = drift_engines
        try:
            return await self._execute_chain(
                "detect_drift",
                lambda e: lambda b, c, **kw: e.detect_drift(b, c, **kw),
                baseline,
                current,
                **kwargs,
            )
        finally:
            self._engines = original_engines

    async def detect_anomalies(
        self,
        data: Any,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Execute async anomaly detection across chain.

        Only engines implementing AnomalyDetectionEngine are considered.

        Raises:
            NoEngineSelectedError: If no engines support anomaly detection.
            AllEnginesFailedError: If all anomaly-capable engines fail.
        """
        anomaly_engines = [e for e in self._engines if supports_anomaly(e)]
        if not anomaly_engines:
            raise NoEngineSelectedError(
                f"No engines in async chain '{self._name}' support anomaly detection",
                chain_name=self._name,
            )

        original_engines = self._engines
        self._engines = anomaly_engines
        try:
            return await self._execute_chain(
                "detect_anomalies",
                lambda e: lambda d, **kw: e.detect_anomalies(d, **kw),
                data,
                **kwargs,
            )
        finally:
            self._engines = original_engines

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        for engine in self._engines:
            if hasattr(engine, "start"):
                await engine.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        for engine in self._engines:
            if hasattr(engine, "stop"):
                try:
                    await engine.stop()
                except Exception:
                    pass


# =============================================================================
# Factory Functions
# =============================================================================


def create_fallback_chain(
    primary: DataQualityEngine,
    *fallbacks: DataQualityEngine,
    retry_count: int = 1,
    check_health: bool = False,
    name: str | None = None,
    hooks: Sequence[ChainHook] | None = None,
) -> EngineChain:
    """Create a simple fallback chain.

    Convenience function for creating a chain with primary and backup engines.

    Args:
        primary: Primary engine.
        *fallbacks: Fallback engines in priority order.
        retry_count: Retries per engine before fallback.
        check_health: Whether to check engine health.
        name: Chain name.
        hooks: Chain hooks.

    Returns:
        Configured EngineChain.

    Example:
        >>> chain = create_fallback_chain(primary, backup1, backup2)
        >>> result = chain.check(data, rules)
    """
    engines = [primary, *fallbacks]
    config = FallbackConfig(
        strategy=FallbackStrategy.SEQUENTIAL,
        retry_count=retry_count,
        check_health=check_health,
    )
    return EngineChain(engines, config=config, hooks=hooks, name=name)


def create_load_balanced_chain(
    *engines: DataQualityEngine,
    strategy: FallbackStrategy = FallbackStrategy.ROUND_ROBIN,
    weights: dict[str, float] | None = None,
    check_health: bool = True,
    name: str | None = None,
    hooks: Sequence[ChainHook] | None = None,
) -> EngineChain:
    """Create a load-balanced engine chain.

    Distributes requests across multiple engines for load balancing.

    Args:
        *engines: Engines to balance across.
        strategy: Load balancing strategy.
        weights: Engine weights for WEIGHTED strategy.
        check_health: Whether to check engine health.
        name: Chain name.
        hooks: Chain hooks.

    Returns:
        Configured EngineChain.

    Example:
        >>> chain = create_load_balanced_chain(
        ...     engine1, engine2, engine3,
        ...     strategy=FallbackStrategy.WEIGHTED,
        ...     weights={"engine1": 2.0, "engine2": 1.0, "engine3": 1.0},
        ... )
    """
    config = FallbackConfig(
        strategy=strategy,
        check_health=check_health,
        weights=weights or {},
    )
    return EngineChain(list(engines), config=config, hooks=hooks, name=name)


def create_async_fallback_chain(
    primary: Any,  # AsyncDataQualityEngine
    *fallbacks: Any,
    retry_count: int = 1,
    name: str | None = None,
    hooks: Sequence[ChainHook] | None = None,
) -> AsyncEngineChain:
    """Create an async fallback chain.

    Args:
        primary: Primary async engine.
        *fallbacks: Fallback async engines.
        retry_count: Retries per engine.
        name: Chain name.
        hooks: Chain hooks.

    Returns:
        Configured AsyncEngineChain.
    """
    engines = [primary, *fallbacks]
    config = FallbackConfig(
        strategy=FallbackStrategy.SEQUENTIAL,
        retry_count=retry_count,
    )
    return AsyncEngineChain(engines, config=config, hooks=hooks, name=name)
