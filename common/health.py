"""Health Check utilities for Truthound Integrations.

This module provides a flexible, extensible health check system designed for
monitoring component health in distributed environments. It supports:

- Multiple health check statuses (healthy, degraded, unhealthy, unknown)
- Configurable timeouts and caching
- Composite health checks for aggregating multiple checks
- Pre/post check hooks for logging and monitoring
- Async and sync function support
- Platform-specific adapters (Airflow, Dagster, Prefect)

Design Principles:
    1. Protocol-based: Easy to extend with custom health checkers
    2. Immutable Config: Thread-safe configuration using frozen dataclass
    3. Observable: Hook system for monitoring health check behavior
    4. Composable: Aggregate multiple health checks into one

Example:
    >>> from common.health import HealthCheck, HealthCheckConfig
    >>> @health_check(name="database", timeout_seconds=5.0)
    ... def check_database():
    ...     return db.ping()

    >>> # With composite checks
    >>> composite = CompositeHealthChecker(
    ...     checkers=[db_checker, cache_checker, api_checker],
    ...     name="system",
    ... )
    >>> result = composite.check()
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
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


class HealthCheckError(TruthoundIntegrationError):
    """Base exception for health check errors.

    Attributes:
        check_name: Name of the health check.
        status: Health check status.
    """

    def __init__(
        self,
        message: str,
        *,
        check_name: str | None = None,
        status: HealthStatus | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize health check error.

        Args:
            message: Human-readable error description.
            check_name: Name of the health check.
            status: Health check status.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if check_name:
            details["check_name"] = check_name
        if status:
            details["status"] = status.name
        super().__init__(message, details=details, cause=cause)
        self.check_name = check_name
        self.status = status


class HealthCheckTimeoutError(HealthCheckError):
    """Exception raised when health check times out.

    Attributes:
        timeout_seconds: Timeout value in seconds.
    """

    def __init__(
        self,
        message: str = "Health check timed out",
        *,
        check_name: str | None = None,
        timeout_seconds: float = 0.0,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Human-readable error description.
            check_name: Name of the health check.
            timeout_seconds: Timeout value in seconds.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(
            message,
            check_name=check_name,
            status=HealthStatus.UNHEALTHY,
            details=details,
            cause=cause,
        )
        self.timeout_seconds = timeout_seconds


# =============================================================================
# Enums
# =============================================================================


class HealthStatus(Enum):
    """Health check status values.

    Attributes:
        HEALTHY: Component is fully operational.
        DEGRADED: Component is operational but with reduced functionality.
        UNHEALTHY: Component is not operational.
        UNKNOWN: Health status could not be determined.
    """

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()

    @property
    def is_healthy(self) -> bool:
        """Check if status represents a healthy state."""
        return self == HealthStatus.HEALTHY

    @property
    def is_operational(self) -> bool:
        """Check if status represents an operational state."""
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    @property
    def weight(self) -> int:
        """Return numeric weight for comparison (higher = healthier)."""
        weights = {
            HealthStatus.HEALTHY: 100,
            HealthStatus.DEGRADED: 50,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 25,
        }
        return weights[self]

    def __lt__(self, other: HealthStatus) -> bool:
        """Compare status by weight."""
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.weight < other.weight

    def __le__(self, other: HealthStatus) -> bool:
        """Compare status by weight."""
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.weight <= other.weight

    def __gt__(self, other: HealthStatus) -> bool:
        """Compare status by weight."""
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.weight > other.weight

    def __ge__(self, other: HealthStatus) -> bool:
        """Compare status by weight."""
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.weight >= other.weight


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class HealthChecker(Protocol):
    """Protocol for health checkers.

    Implement this protocol to create custom health checks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this health checker."""
        ...

    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Execute health check.

        Returns:
            HealthCheckResult with status and details.
        """
        ...


@runtime_checkable
class AsyncHealthChecker(Protocol):
    """Protocol for async health checkers.

    Implement this protocol for async health checks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this health checker."""
        ...

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Execute async health check.

        Returns:
            HealthCheckResult with status and details.
        """
        ...


@runtime_checkable
class HealthCheckHook(Protocol):
    """Protocol for health check event hooks.

    Implement this to receive notifications about health check events.
    """

    @abstractmethod
    def on_check_start(
        self,
        name: str,
        context: dict[str, Any],
    ) -> None:
        """Called before health check starts.

        Args:
            name: Name of the health check.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_check_complete(
        self,
        name: str,
        result: HealthCheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called after health check completes.

        Args:
            name: Name of the health check.
            result: Health check result.
            duration_ms: Check duration in milliseconds.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_check_error(
        self,
        name: str,
        exception: Exception,
        context: dict[str, Any],
    ) -> None:
        """Called when health check encounters an error.

        Args:
            name: Name of the health check.
            exception: Exception that occurred.
            context: Additional context information.
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class HealthCheckConfig:
    """Configuration for health check behavior.

    Immutable configuration object for health check operations.
    Use builder methods to create modified copies.

    Attributes:
        timeout_seconds: Maximum time for health check execution.
        cache_ttl_seconds: Time to cache health check results (0 = no caching).
        include_details: Whether to include detailed information in results.
        fail_on_timeout: Whether to return UNHEALTHY on timeout.
        tags: Tags for categorization.
        metadata: Additional metadata.

    Example:
        >>> config = HealthCheckConfig(
        ...     timeout_seconds=5.0,
        ...     cache_ttl_seconds=30.0,
        ... )
        >>> strict_config = config.with_timeout(2.0)
    """

    timeout_seconds: float = 5.0
    cache_ttl_seconds: float = 0.0
    include_details: bool = True
    fail_on_timeout: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")

    def with_timeout(self, timeout_seconds: float) -> HealthCheckConfig:
        """Create config with new timeout.

        Args:
            timeout_seconds: New timeout value.

        Returns:
            New HealthCheckConfig with updated value.
        """
        return HealthCheckConfig(
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=self.cache_ttl_seconds,
            include_details=self.include_details,
            fail_on_timeout=self.fail_on_timeout,
            tags=self.tags,
            metadata=self.metadata,
        )

    def with_cache_ttl(self, cache_ttl_seconds: float) -> HealthCheckConfig:
        """Create config with new cache TTL.

        Args:
            cache_ttl_seconds: New cache TTL value.

        Returns:
            New HealthCheckConfig with updated value.
        """
        return HealthCheckConfig(
            timeout_seconds=self.timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
            include_details=self.include_details,
            fail_on_timeout=self.fail_on_timeout,
            tags=self.tags,
            metadata=self.metadata,
        )

    def with_tags(self, *tags: str) -> HealthCheckConfig:
        """Create config with additional tags.

        Args:
            *tags: Additional tags to add.

        Returns:
            New HealthCheckConfig with merged tags.
        """
        return HealthCheckConfig(
            timeout_seconds=self.timeout_seconds,
            cache_ttl_seconds=self.cache_ttl_seconds,
            include_details=self.include_details,
            fail_on_timeout=self.fail_on_timeout,
            tags=self.tags | frozenset(tags),
            metadata=self.metadata,
        )

    def with_metadata(self, **kwargs: Any) -> HealthCheckConfig:
        """Create config with additional metadata.

        Args:
            **kwargs: Additional metadata.

        Returns:
            New HealthCheckConfig with merged metadata.
        """
        return HealthCheckConfig(
            timeout_seconds=self.timeout_seconds,
            cache_ttl_seconds=self.cache_ttl_seconds,
            include_details=self.include_details,
            fail_on_timeout=self.fail_on_timeout,
            tags=self.tags,
            metadata={**self.metadata, **kwargs},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "include_details": self.include_details,
            "fail_on_timeout": self.fail_on_timeout,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create HealthCheckConfig from dictionary.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New HealthCheckConfig instance.
        """
        return cls(
            timeout_seconds=data.get("timeout_seconds", 5.0),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 0.0),
            include_details=data.get("include_details", True),
            fail_on_timeout=data.get("fail_on_timeout", True),
            tags=frozenset(data.get("tags", [])),
            metadata=data.get("metadata", {}),
        )


# Default configurations for common use cases
DEFAULT_HEALTH_CHECK_CONFIG = HealthCheckConfig()

FAST_HEALTH_CHECK_CONFIG = HealthCheckConfig(
    timeout_seconds=1.0,
    cache_ttl_seconds=10.0,
)

THOROUGH_HEALTH_CHECK_CONFIG = HealthCheckConfig(
    timeout_seconds=30.0,
    cache_ttl_seconds=0.0,
    include_details=True,
)

CACHED_HEALTH_CHECK_CONFIG = HealthCheckConfig(
    timeout_seconds=5.0,
    cache_ttl_seconds=60.0,
)


# =============================================================================
# Result Types
# =============================================================================


def _utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of a health check execution.

    Immutable result object containing health status and metadata.

    Attributes:
        name: Name of the health check.
        status: Health status.
        message: Human-readable status message.
        duration_ms: Check duration in milliseconds.
        timestamp: ISO format timestamp of check.
        details: Additional check details.
        dependencies: Results from dependent health checks.
        metadata: Additional result metadata.

    Example:
        >>> result = HealthCheckResult(
        ...     name="database",
        ...     status=HealthStatus.HEALTHY,
        ...     message="Database is operational",
        ...     duration_ms=45.2,
        ... )
        >>> result.is_healthy
        True
    """

    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=_utc_now_iso)
    details: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[HealthCheckResult, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if the result represents a healthy state."""
        return self.status.is_healthy

    @property
    def is_operational(self) -> bool:
        """Check if the result represents an operational state."""
        return self.status.is_operational

    def with_status(self, status: HealthStatus) -> HealthCheckResult:
        """Create a new result with updated status.

        Args:
            status: New health status.

        Returns:
            New HealthCheckResult with updated status.
        """
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=self.message,
            duration_ms=self.duration_ms,
            timestamp=self.timestamp,
            details=self.details,
            dependencies=self.dependencies,
            metadata=self.metadata,
        )

    def with_message(self, message: str) -> HealthCheckResult:
        """Create a new result with updated message.

        Args:
            message: New message.

        Returns:
            New HealthCheckResult with updated message.
        """
        return HealthCheckResult(
            name=self.name,
            status=self.status,
            message=message,
            duration_ms=self.duration_ms,
            timestamp=self.timestamp,
            details=self.details,
            dependencies=self.dependencies,
            metadata=self.metadata,
        )

    def with_details(self, **kwargs: Any) -> HealthCheckResult:
        """Create a new result with additional details.

        Args:
            **kwargs: Additional details.

        Returns:
            New HealthCheckResult with merged details.
        """
        return HealthCheckResult(
            name=self.name,
            status=self.status,
            message=self.message,
            duration_ms=self.duration_ms,
            timestamp=self.timestamp,
            details={**self.details, **kwargs},
            dependencies=self.dependencies,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.name,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "details": self.details,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "metadata": self.metadata,
            "is_healthy": self.is_healthy,
            "is_operational": self.is_operational,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create HealthCheckResult from dictionary.

        Args:
            data: Dictionary with result data.

        Returns:
            New HealthCheckResult instance.
        """
        dependencies = tuple(
            cls.from_dict(d) for d in data.get("dependencies", [])
        )
        return cls(
            name=data["name"],
            status=HealthStatus[data["status"]],
            message=data.get("message", ""),
            duration_ms=data.get("duration_ms", 0.0),
            timestamp=data.get("timestamp", _utc_now_iso()),
            details=data.get("details", {}),
            dependencies=dependencies,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def healthy(
        cls,
        name: str,
        message: str = "Healthy",
        **details: Any,
    ) -> HealthCheckResult:
        """Create a healthy result.

        Args:
            name: Name of the health check.
            message: Status message.
            **details: Additional details.

        Returns:
            HealthCheckResult with HEALTHY status.
        """
        return cls(
            name=name,
            status=HealthStatus.HEALTHY,
            message=message,
            details=details,
        )

    @classmethod
    def degraded(
        cls,
        name: str,
        message: str = "Degraded",
        **details: Any,
    ) -> HealthCheckResult:
        """Create a degraded result.

        Args:
            name: Name of the health check.
            message: Status message.
            **details: Additional details.

        Returns:
            HealthCheckResult with DEGRADED status.
        """
        return cls(
            name=name,
            status=HealthStatus.DEGRADED,
            message=message,
            details=details,
        )

    @classmethod
    def unhealthy(
        cls,
        name: str,
        message: str = "Unhealthy",
        **details: Any,
    ) -> HealthCheckResult:
        """Create an unhealthy result.

        Args:
            name: Name of the health check.
            message: Status message.
            **details: Additional details.

        Returns:
            HealthCheckResult with UNHEALTHY status.
        """
        return cls(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=message,
            details=details,
        )

    @classmethod
    def unknown(
        cls,
        name: str,
        message: str = "Unknown",
        **details: Any,
    ) -> HealthCheckResult:
        """Create an unknown result.

        Args:
            name: Name of the health check.
            message: Status message.
            **details: Additional details.

        Returns:
            HealthCheckResult with UNKNOWN status.
        """
        return cls(
            name=name,
            status=HealthStatus.UNKNOWN,
            message=message,
            details=details,
        )


# =============================================================================
# Hooks
# =============================================================================


class LoggingHealthCheckHook:
    """Hook that logs health check events.

    Uses the Truthound logging system for structured logging.
    """

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.health).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.health")

    def on_check_start(
        self,
        name: str,
        context: dict[str, Any],
    ) -> None:
        """Log check start.

        Args:
            name: Name of the health check.
            context: Additional context.
        """
        self._logger.debug(
            "Health check starting",
            check_name=name,
            **context,
        )

    def on_check_complete(
        self,
        name: str,
        result: HealthCheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log check completion.

        Args:
            name: Name of the health check.
            result: Health check result.
            duration_ms: Check duration in milliseconds.
            context: Additional context.
        """
        log_method = self._logger.info
        if result.status == HealthStatus.UNHEALTHY:
            log_method = self._logger.error
        elif result.status in {HealthStatus.DEGRADED, HealthStatus.UNKNOWN}:
            log_method = self._logger.warning

        log_method(
            "Health check completed",
            check_name=name,
            status=result.status.name,
            duration_ms=duration_ms,
            result_message=result.message,
            **context,
        )

    def on_check_error(
        self,
        name: str,
        exception: Exception,
        context: dict[str, Any],
    ) -> None:
        """Log check error.

        Args:
            name: Name of the health check.
            exception: Exception that occurred.
            context: Additional context.
        """
        self._logger.error(
            "Health check failed with exception",
            check_name=name,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            **context,
        )


class MetricsHealthCheckHook:
    """Hook that collects health check metrics.

    Useful for monitoring and alerting on health check behavior.
    """

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._check_counts: dict[str, int] = {}
        self._status_counts: dict[str, dict[HealthStatus, int]] = {}
        self._total_duration_ms: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}
        self._last_results: dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()

    def on_check_start(
        self,
        name: str,
        context: dict[str, Any],
    ) -> None:
        """Record check start.

        Args:
            name: Name of the health check.
            context: Additional context.
        """
        with self._lock:
            if name not in self._check_counts:
                self._check_counts[name] = 0
                self._status_counts[name] = {}
                self._total_duration_ms[name] = 0.0
                self._error_counts[name] = 0

    def on_check_complete(
        self,
        name: str,
        result: HealthCheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record check completion.

        Args:
            name: Name of the health check.
            result: Health check result.
            duration_ms: Check duration in milliseconds.
            context: Additional context.
        """
        with self._lock:
            self._check_counts[name] = self._check_counts.get(name, 0) + 1
            self._total_duration_ms[name] = (
                self._total_duration_ms.get(name, 0.0) + duration_ms
            )

            if name not in self._status_counts:
                self._status_counts[name] = {}
            status_dict = self._status_counts[name]
            status_dict[result.status] = status_dict.get(result.status, 0) + 1

            self._last_results[name] = result

    def on_check_error(
        self,
        name: str,
        exception: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record check error.

        Args:
            name: Name of the health check.
            exception: Exception that occurred.
            context: Additional context.
        """
        with self._lock:
            self._error_counts[name] = self._error_counts.get(name, 0) + 1

    def get_check_count(self, name: str) -> int:
        """Get total check count for a name."""
        with self._lock:
            return self._check_counts.get(name, 0)

    def get_status_counts(self, name: str) -> dict[HealthStatus, int]:
        """Get status counts for a name."""
        with self._lock:
            return dict(self._status_counts.get(name, {}))

    def get_average_duration_ms(self, name: str) -> float:
        """Get average check duration for a name."""
        with self._lock:
            count = self._check_counts.get(name, 0)
            if count == 0:
                return 0.0
            return self._total_duration_ms.get(name, 0.0) / count

    def get_error_count(self, name: str) -> int:
        """Get error count for a name."""
        with self._lock:
            return self._error_counts.get(name, 0)

    def get_last_result(self, name: str) -> HealthCheckResult | None:
        """Get last result for a name."""
        with self._lock:
            return self._last_results.get(name)

    def get_all_last_results(self) -> dict[str, HealthCheckResult]:
        """Get all last results."""
        with self._lock:
            return dict(self._last_results)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._check_counts.clear()
            self._status_counts.clear()
            self._total_duration_ms.clear()
            self._error_counts.clear()
            self._last_results.clear()


class CompositeHealthCheckHook:
    """Combine multiple health check hooks."""

    def __init__(self, hooks: Sequence[HealthCheckHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: HealthCheckHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: HealthCheckHook) -> None:
        """Remove a hook.

        Args:
            hook: Hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_check_start(
        self,
        name: str,
        context: dict[str, Any],
    ) -> None:
        """Call on_check_start on all hooks.

        Args:
            name: Name of the health check.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_check_start(name, context)

    def on_check_complete(
        self,
        name: str,
        result: HealthCheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_check_complete on all hooks.

        Args:
            name: Name of the health check.
            result: Health check result.
            duration_ms: Check duration in milliseconds.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_check_complete(name, result, duration_ms, context)

    def on_check_error(
        self,
        name: str,
        exception: Exception,
        context: dict[str, Any],
    ) -> None:
        """Call on_check_error on all hooks.

        Args:
            name: Name of the health check.
            exception: Exception that occurred.
            context: Additional context.
        """
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_check_error(name, exception, context)


# =============================================================================
# Health Check Executor
# =============================================================================


class HealthCheckExecutor:
    """Executes health checks with timeout, caching, and hooks.

    This class provides the core health check execution logic with
    configurable behavior and observability.

    Example:
        >>> executor = HealthCheckExecutor(
        ...     config=HealthCheckConfig(timeout_seconds=5.0),
        ...     hooks=[LoggingHealthCheckHook()],
        ... )
        >>> result = executor.execute("database", check_database)
    """

    def __init__(
        self,
        config: HealthCheckConfig | None = None,
        hooks: Sequence[HealthCheckHook] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            config: Health check configuration.
            hooks: Health check event hooks.
        """
        self.config = config or DEFAULT_HEALTH_CHECK_CONFIG
        self._hook: HealthCheckHook | None = None
        if hooks:
            self._hook = CompositeHealthCheckHook(list(hooks))
        self._cache: dict[str, tuple[HealthCheckResult, float]] = {}
        self._lock = threading.RLock()

    def _create_context(self, name: str) -> dict[str, Any]:
        """Create context dictionary for hooks.

        Args:
            name: Name of the health check.

        Returns:
            Context dictionary.
        """
        return {
            "check_name": name,
            "timeout_seconds": self.config.timeout_seconds,
            "cache_ttl_seconds": self.config.cache_ttl_seconds,
        }

    def _get_cached(self, name: str) -> HealthCheckResult | None:
        """Get cached result if valid.

        Args:
            name: Name of the health check.

        Returns:
            Cached result if valid, None otherwise.
        """
        if self.config.cache_ttl_seconds <= 0:
            return None

        with self._lock:
            cached = self._cache.get(name)
            if cached is None:
                return None

            result, cached_at = cached
            age = time.monotonic() - cached_at
            if age > self.config.cache_ttl_seconds:
                del self._cache[name]
                return None

            return result

    def _set_cached(self, name: str, result: HealthCheckResult) -> None:
        """Cache a result.

        Args:
            name: Name of the health check.
            result: Result to cache.
        """
        if self.config.cache_ttl_seconds <= 0:
            return

        with self._lock:
            self._cache[name] = (result, time.monotonic())

    def clear_cache(self, name: str | None = None) -> None:
        """Clear cached results.

        Args:
            name: Specific check name to clear, or None for all.
        """
        with self._lock:
            if name is None:
                self._cache.clear()
            elif name in self._cache:
                del self._cache[name]

    def execute(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult | bool | None],
    ) -> HealthCheckResult:
        """Execute a health check.

        Args:
            name: Name of the health check.
            check_func: Function to execute.

        Returns:
            HealthCheckResult.
        """
        context = self._create_context(name)

        # Check cache first
        cached = self._get_cached(name)
        if cached is not None:
            return cached

        if self._hook:
            self._hook.on_check_start(name, context)

        start_time = time.perf_counter()

        try:
            # Execute with timeout using threading
            result_container: list[Any] = []
            exception_container: list[Exception] = []

            def run_check() -> None:
                try:
                    result_container.append(check_func())
                except Exception as e:
                    exception_container.append(e)

            thread = threading.Thread(target=run_check)
            thread.start()
            thread.join(timeout=self.config.timeout_seconds)

            if thread.is_alive():
                # Timeout occurred
                duration_ms = (time.perf_counter() - start_time) * 1000
                result = HealthCheckResult(
                    name=name,
                    status=(
                        HealthStatus.UNHEALTHY
                        if self.config.fail_on_timeout
                        else HealthStatus.UNKNOWN
                    ),
                    message=f"Health check timed out after {self.config.timeout_seconds}s",
                    duration_ms=duration_ms,
                )
                if self._hook:
                    self._hook.on_check_complete(name, result, duration_ms, context)
                return result

            if exception_container:
                raise exception_container[0]

            raw_result = result_container[0] if result_container else None
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Normalize result
            result = self._normalize_result(name, raw_result, duration_ms)

            if self._hook:
                self._hook.on_check_complete(name, result, duration_ms, context)

            self._set_cached(name, result)
            return result

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                self._hook.on_check_error(name, exc, context)

            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {exc}",
                duration_ms=duration_ms,
                details={"exception": type(exc).__name__, "error": str(exc)},
            )

            if self._hook:
                self._hook.on_check_complete(name, result, duration_ms, context)

            return result

    async def execute_async(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult | bool | None]],
    ) -> HealthCheckResult:
        """Execute an async health check.

        Args:
            name: Name of the health check.
            check_func: Async function to execute.

        Returns:
            HealthCheckResult.
        """
        context = self._create_context(name)

        # Check cache first
        cached = self._get_cached(name)
        if cached is not None:
            return cached

        if self._hook:
            self._hook.on_check_start(name, context)

        start_time = time.perf_counter()

        try:
            raw_result = await asyncio.wait_for(
                check_func(),
                timeout=self.config.timeout_seconds,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            result = self._normalize_result(name, raw_result, duration_ms)

            if self._hook:
                self._hook.on_check_complete(name, result, duration_ms, context)

            self._set_cached(name, result)
            return result

        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                name=name,
                status=(
                    HealthStatus.UNHEALTHY
                    if self.config.fail_on_timeout
                    else HealthStatus.UNKNOWN
                ),
                message=f"Health check timed out after {self.config.timeout_seconds}s",
                duration_ms=duration_ms,
            )
            if self._hook:
                self._hook.on_check_complete(name, result, duration_ms, context)
            return result

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                self._hook.on_check_error(name, exc, context)

            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {exc}",
                duration_ms=duration_ms,
                details={"exception": type(exc).__name__, "error": str(exc)},
            )

            if self._hook:
                self._hook.on_check_complete(name, result, duration_ms, context)

            return result

    def _normalize_result(
        self,
        name: str,
        result: HealthCheckResult | bool | None,
        duration_ms: float,
    ) -> HealthCheckResult:
        """Normalize check result to HealthCheckResult.

        Args:
            name: Name of the health check.
            result: Raw result from check function.
            duration_ms: Check duration in milliseconds.

        Returns:
            Normalized HealthCheckResult.
        """
        if isinstance(result, HealthCheckResult):
            # Update duration if not set
            if result.duration_ms == 0.0:
                return HealthCheckResult(
                    name=result.name,
                    status=result.status,
                    message=result.message,
                    duration_ms=duration_ms,
                    timestamp=result.timestamp,
                    details=result.details,
                    dependencies=result.dependencies,
                    metadata=result.metadata,
                )
            return result

        if isinstance(result, bool):
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                message="Healthy" if result else "Unhealthy",
                duration_ms=duration_ms,
            )

        if result is None:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message="Healthy",
                duration_ms=duration_ms,
            )

        # Treat any other truthy value as healthy
        return HealthCheckResult(
            name=name,
            status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
            message="Healthy" if result else "Unhealthy",
            duration_ms=duration_ms,
        )


# =============================================================================
# Composite Health Checker
# =============================================================================


class CompositeHealthChecker:
    """Aggregates multiple health checkers into a single check.

    Combines results from multiple health checks using configurable
    aggregation strategy.

    Example:
        >>> composite = CompositeHealthChecker(
        ...     checkers=[db_checker, cache_checker],
        ...     name="system",
        ...     strategy=AggregationStrategy.WORST,
        ... )
        >>> result = composite.check()
    """

    def __init__(
        self,
        checkers: Sequence[HealthChecker],
        name: str = "composite",
        strategy: AggregationStrategy | None = None,
        parallel: bool = True,
    ) -> None:
        """Initialize composite checker.

        Args:
            checkers: List of health checkers to aggregate.
            name: Name for the composite check.
            strategy: Aggregation strategy (default: WORST).
            parallel: Whether to run checks in parallel.
        """
        self._checkers = list(checkers)
        self._name = name
        self._strategy = strategy or AggregationStrategy.WORST
        self._parallel = parallel

    @property
    def name(self) -> str:
        """Return the name of this health checker."""
        return self._name

    def add_checker(self, checker: HealthChecker) -> None:
        """Add a health checker.

        Args:
            checker: Health checker to add.
        """
        self._checkers.append(checker)

    def remove_checker(self, name: str) -> bool:
        """Remove a health checker by name.

        Args:
            name: Name of the checker to remove.

        Returns:
            True if removed, False if not found.
        """
        for i, checker in enumerate(self._checkers):
            if checker.name == name:
                self._checkers.pop(i)
                return True
        return False

    def check(self) -> HealthCheckResult:
        """Execute all health checks and aggregate results.

        Returns:
            Aggregated HealthCheckResult.
        """
        start_time = time.perf_counter()
        results: list[HealthCheckResult] = []

        if self._parallel:
            threads: list[tuple[threading.Thread, list[HealthCheckResult]]] = []
            for checker in self._checkers:
                result_container: list[HealthCheckResult] = []

                def run_check(c: HealthChecker, container: list[HealthCheckResult]) -> None:
                    try:
                        container.append(c.check())
                    except Exception as e:
                        container.append(
                            HealthCheckResult.unhealthy(
                                c.name,
                                message=f"Check failed: {e}",
                                exception=type(e).__name__,
                            )
                        )

                thread = threading.Thread(
                    target=run_check,
                    args=(checker, result_container),
                )
                thread.start()
                threads.append((thread, result_container))

            for thread, container in threads:
                thread.join()
                if container:
                    results.append(container[0])
        else:
            for checker in self._checkers:
                try:
                    results.append(checker.check())
                except Exception as e:
                    results.append(
                        HealthCheckResult.unhealthy(
                            checker.name,
                            message=f"Check failed: {e}",
                            exception=type(e).__name__,
                        )
                    )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return self._aggregate_results(results, duration_ms)

    async def check_async(self) -> HealthCheckResult:
        """Execute all async health checks and aggregate results.

        Returns:
            Aggregated HealthCheckResult.
        """
        start_time = time.perf_counter()
        results: list[HealthCheckResult] = []

        async def run_check(checker: HealthChecker) -> HealthCheckResult:
            try:
                if isinstance(checker, AsyncHealthChecker):
                    return await checker.check()
                # Run sync checker in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, checker.check)
            except Exception as e:
                return HealthCheckResult.unhealthy(
                    checker.name,
                    message=f"Check failed: {e}",
                    exception=type(e).__name__,
                )

        if self._parallel:
            results = await asyncio.gather(
                *[run_check(c) for c in self._checkers]
            )
        else:
            for checker in self._checkers:
                results.append(await run_check(checker))

        duration_ms = (time.perf_counter() - start_time) * 1000
        return self._aggregate_results(list(results), duration_ms)

    def _aggregate_results(
        self,
        results: list[HealthCheckResult],
        duration_ms: float,
    ) -> HealthCheckResult:
        """Aggregate multiple results into one.

        Args:
            results: List of results to aggregate.
            duration_ms: Total duration in milliseconds.

        Returns:
            Aggregated HealthCheckResult.
        """
        if not results:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNKNOWN,
                message="No health checks configured",
                duration_ms=duration_ms,
            )

        status = self._strategy.aggregate([r.status for r in results])
        healthy_count = sum(1 for r in results if r.is_healthy)
        total_count = len(results)

        return HealthCheckResult(
            name=self._name,
            status=status,
            message=f"{healthy_count}/{total_count} checks healthy",
            duration_ms=duration_ms,
            details={
                "healthy_count": healthy_count,
                "total_count": total_count,
                "check_statuses": {r.name: r.status.name for r in results},
            },
            dependencies=tuple(results),
        )


class AggregationStrategy(Enum):
    """Strategy for aggregating multiple health check results.

    Attributes:
        WORST: Use the worst status (most unhealthy).
        BEST: Use the best status (most healthy).
        MAJORITY: Use the most common status.
        ALL_HEALTHY: HEALTHY only if all are healthy.
        ANY_HEALTHY: HEALTHY if any are healthy.
    """

    WORST = auto()
    BEST = auto()
    MAJORITY = auto()
    ALL_HEALTHY = auto()
    ANY_HEALTHY = auto()

    def aggregate(self, statuses: Sequence[HealthStatus]) -> HealthStatus:  # noqa: PLR0911
        """Aggregate multiple statuses into one.

        Args:
            statuses: Statuses to aggregate.

        Returns:
            Aggregated status.
        """
        if not statuses:
            return HealthStatus.UNKNOWN

        status_list = list(statuses)

        match self:
            case AggregationStrategy.WORST:
                return min(status_list)
            case AggregationStrategy.BEST:
                return max(status_list)
            case AggregationStrategy.MAJORITY:
                from collections import Counter
                counter = Counter(status_list)
                return counter.most_common(1)[0][0]
            case AggregationStrategy.ALL_HEALTHY:
                if all(s == HealthStatus.HEALTHY for s in status_list):
                    return HealthStatus.HEALTHY
                if any(s == HealthStatus.UNHEALTHY for s in status_list):
                    return HealthStatus.UNHEALTHY
                return HealthStatus.DEGRADED
            case AggregationStrategy.ANY_HEALTHY:
                if any(s == HealthStatus.HEALTHY for s in status_list):
                    return HealthStatus.HEALTHY
                return min(status_list)
            case _:
                return HealthStatus.UNKNOWN


# =============================================================================
# Simple Health Checker Implementation
# =============================================================================


class SimpleHealthChecker:
    """Simple health checker wrapping a callable.

    Example:
        >>> checker = SimpleHealthChecker(
        ...     name="database",
        ...     check_func=lambda: db.ping(),
        ... )
        >>> result = checker.check()
    """

    def __init__(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult | bool | None],
        config: HealthCheckConfig | None = None,
        hooks: Sequence[HealthCheckHook] | None = None,
    ) -> None:
        """Initialize simple health checker.

        Args:
            name: Name of the health check.
            check_func: Function to execute for health check.
            config: Health check configuration.
            hooks: Health check event hooks.
        """
        self._name = name
        self._check_func = check_func
        self._executor = HealthCheckExecutor(config=config, hooks=hooks)

    @property
    def name(self) -> str:
        """Return the name of this health checker."""
        return self._name

    def check(self) -> HealthCheckResult:
        """Execute the health check.

        Returns:
            HealthCheckResult.
        """
        return self._executor.execute(self._name, self._check_func)


class AsyncSimpleHealthChecker:
    """Simple async health checker wrapping a coroutine function.

    Example:
        >>> checker = AsyncSimpleHealthChecker(
        ...     name="api",
        ...     check_func=lambda: api_client.health(),
        ... )
        >>> result = await checker.check()
    """

    def __init__(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult | bool | None]],
        config: HealthCheckConfig | None = None,
        hooks: Sequence[HealthCheckHook] | None = None,
    ) -> None:
        """Initialize async health checker.

        Args:
            name: Name of the health check.
            check_func: Async function to execute for health check.
            config: Health check configuration.
            hooks: Health check event hooks.
        """
        self._name = name
        self._check_func = check_func
        self._executor = HealthCheckExecutor(config=config, hooks=hooks)

    @property
    def name(self) -> str:
        """Return the name of this health checker."""
        return self._name

    async def check(self) -> HealthCheckResult:
        """Execute the async health check.

        Returns:
            HealthCheckResult.
        """
        return await self._executor.execute_async(self._name, self._check_func)


# =============================================================================
# Health Check Registry
# =============================================================================


class HealthCheckRegistry:
    """Registry for managing health checkers.

    Provides a central location to register, retrieve, and execute
    health checks by name.

    Example:
        >>> registry = HealthCheckRegistry()
        >>> registry.register("database", db_checker)
        >>> result = registry.check("database")
        >>> all_results = registry.check_all()
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._checkers: dict[str, HealthChecker] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        checker: HealthChecker | Callable[[], HealthCheckResult | bool | None],
        config: HealthCheckConfig | None = None,
        hooks: Sequence[HealthCheckHook] | None = None,
    ) -> None:
        """Register a health checker.

        Args:
            name: Name for the health checker.
            checker: Health checker or callable to register.
            config: Health check configuration (for callable).
            hooks: Health check event hooks (for callable).
        """
        with self._lock:
            if isinstance(checker, HealthChecker):
                self._checkers[name] = checker
            else:
                self._checkers[name] = SimpleHealthChecker(
                    name=name,
                    check_func=checker,
                    config=config,
                    hooks=hooks,
                )

    def unregister(self, name: str) -> bool:
        """Unregister a health checker.

        Args:
            name: Name of the checker to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._checkers:
                del self._checkers[name]
                return True
            return False

    def get(self, name: str) -> HealthChecker | None:
        """Get a health checker by name.

        Args:
            name: Name of the checker.

        Returns:
            HealthChecker if found, None otherwise.
        """
        with self._lock:
            return self._checkers.get(name)

    def check(self, name: str) -> HealthCheckResult:
        """Execute a health check by name.

        Args:
            name: Name of the checker.

        Returns:
            HealthCheckResult.

        Raises:
            HealthCheckError: If checker not found.
        """
        checker = self.get(name)
        if checker is None:
            return HealthCheckResult.unknown(
                name,
                message=f"Health checker '{name}' not found",
            )
        return checker.check()

    def check_all(
        self,
        parallel: bool = True,
        strategy: AggregationStrategy = AggregationStrategy.WORST,
    ) -> HealthCheckResult:
        """Execute all registered health checks.

        Args:
            parallel: Whether to run checks in parallel.
            strategy: Aggregation strategy.

        Returns:
            Aggregated HealthCheckResult.
        """
        with self._lock:
            checkers = list(self._checkers.values())

        if not checkers:
            return HealthCheckResult.unknown(
                "all",
                message="No health checks registered",
            )

        composite = CompositeHealthChecker(
            checkers=checkers,
            name="all",
            strategy=strategy,
            parallel=parallel,
        )
        return composite.check()

    @property
    def names(self) -> list[str]:
        """Get all registered checker names."""
        with self._lock:
            return list(self._checkers.keys())

    def clear(self) -> None:
        """Clear all registered checkers."""
        with self._lock:
            self._checkers.clear()


# Global registry instance
_default_registry = HealthCheckRegistry()


def get_health_registry() -> HealthCheckRegistry:
    """Get the global health check registry.

    Returns:
        The global HealthCheckRegistry instance.
    """
    return _default_registry


def register_health_check(
    name: str,
    checker: HealthChecker | Callable[[], HealthCheckResult | bool | None],
    config: HealthCheckConfig | None = None,
    hooks: Sequence[HealthCheckHook] | None = None,
) -> None:
    """Register a health check in the global registry.

    Args:
        name: Name for the health checker.
        checker: Health checker or callable to register.
        config: Health check configuration (for callable).
        hooks: Health check event hooks (for callable).
    """
    _default_registry.register(name, checker, config, hooks)


def check_health(name: str) -> HealthCheckResult:
    """Execute a health check from the global registry.

    Args:
        name: Name of the checker.

    Returns:
        HealthCheckResult.
    """
    return _default_registry.check(name)


def check_all_health(
    parallel: bool = True,
    strategy: AggregationStrategy = AggregationStrategy.WORST,
) -> HealthCheckResult:
    """Execute all health checks from the global registry.

    Args:
        parallel: Whether to run checks in parallel.
        strategy: Aggregation strategy.

    Returns:
        Aggregated HealthCheckResult.
    """
    return _default_registry.check_all(parallel=parallel, strategy=strategy)


# =============================================================================
# Health Check Decorator
# =============================================================================


def health_check(
    *,
    name: str | None = None,
    config: HealthCheckConfig | None = None,
    timeout_seconds: float | None = None,
    cache_ttl_seconds: float | None = None,
    hooks: Sequence[HealthCheckHook] | None = None,
    register: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., HealthCheckResult]]:
    """Decorator to create a health check from a function.

    The decorated function can return:
    - HealthCheckResult: Used directly
    - bool: Converted to HEALTHY/UNHEALTHY
    - None: Treated as HEALTHY
    - Any exception: Converted to UNHEALTHY

    Args:
        name: Name for the health check (defaults to function name).
        config: Complete health check configuration.
        timeout_seconds: Timeout in seconds (overrides config).
        cache_ttl_seconds: Cache TTL in seconds (overrides config).
        hooks: Health check event hooks.
        register: Whether to register in the global registry.

    Returns:
        Decorator function.

    Example:
        >>> @health_check(name="database", timeout_seconds=5.0)
        ... def check_database():
        ...     return db.ping()

        >>> result = check_database()  # Returns HealthCheckResult
    """
    # Build config from parameters
    if config is None:
        config = HealthCheckConfig(
            timeout_seconds=timeout_seconds if timeout_seconds is not None else 5.0,
            cache_ttl_seconds=cache_ttl_seconds if cache_ttl_seconds is not None else 0.0,
        )
    else:
        if timeout_seconds is not None:
            config = config.with_timeout(timeout_seconds)
        if cache_ttl_seconds is not None:
            config = config.with_cache_ttl(cache_ttl_seconds)

    def decorator(func: Callable[..., Any]) -> Callable[..., HealthCheckResult]:
        check_name = name or func.__name__
        executor = HealthCheckExecutor(config=config, hooks=hooks)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> HealthCheckResult:
                async def check_func() -> Any:
                    return await func(*args, **kwargs)

                return await executor.execute_async(check_name, check_func)

            if register:
                # Create an async checker and register
                async_checker = AsyncSimpleHealthChecker(
                    name=check_name,
                    check_func=lambda: func(),
                    config=config,
                    hooks=hooks,
                )
                _default_registry._checkers[check_name] = async_checker  # type: ignore

            return async_wrapper  # type: ignore

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> HealthCheckResult:
            def check_func() -> Any:
                return func(*args, **kwargs)

            return executor.execute(check_name, check_func)

        if register:
            checker = SimpleHealthChecker(
                name=check_name,
                check_func=lambda: func(),
                config=config,
                hooks=hooks,
            )
            _default_registry.register(check_name, checker)

        return sync_wrapper

    return decorator


# =============================================================================
# Utility Functions
# =============================================================================


def create_health_checker(
    name: str,
    check_func: Callable[[], HealthCheckResult | bool | None],
    config: HealthCheckConfig | None = None,
    hooks: Sequence[HealthCheckHook] | None = None,
) -> HealthChecker:
    """Create a health checker from a callable.

    Args:
        name: Name of the health check.
        check_func: Function to execute for health check.
        config: Health check configuration.
        hooks: Health check event hooks.

    Returns:
        HealthChecker instance.

    Example:
        >>> checker = create_health_checker(
        ...     name="database",
        ...     check_func=lambda: db.ping(),
        ...     config=HealthCheckConfig(timeout_seconds=5.0),
        ... )
    """
    return SimpleHealthChecker(
        name=name,
        check_func=check_func,
        config=config,
        hooks=hooks,
    )


def create_async_health_checker(
    name: str,
    check_func: Callable[[], Awaitable[HealthCheckResult | bool | None]],
    config: HealthCheckConfig | None = None,
    hooks: Sequence[HealthCheckHook] | None = None,
) -> AsyncHealthChecker:
    """Create an async health checker from a coroutine function.

    Args:
        name: Name of the health check.
        check_func: Async function to execute for health check.
        config: Health check configuration.
        hooks: Health check event hooks.

    Returns:
        AsyncHealthChecker instance.

    Example:
        >>> checker = create_async_health_checker(
        ...     name="api",
        ...     check_func=lambda: api_client.health(),
        ... )
    """
    return AsyncSimpleHealthChecker(
        name=name,
        check_func=check_func,
        config=config,
        hooks=hooks,
    )


def create_composite_checker(
    name: str,
    checkers: Sequence[HealthChecker],
    strategy: AggregationStrategy = AggregationStrategy.WORST,
    parallel: bool = True,
) -> CompositeHealthChecker:
    """Create a composite health checker from multiple checkers.

    Args:
        name: Name for the composite check.
        checkers: List of health checkers.
        strategy: Aggregation strategy.
        parallel: Whether to run checks in parallel.

    Returns:
        CompositeHealthChecker instance.

    Example:
        >>> composite = create_composite_checker(
        ...     name="system",
        ...     checkers=[db_checker, cache_checker],
        ...     strategy=AggregationStrategy.WORST,
        ... )
    """
    return CompositeHealthChecker(
        checkers=checkers,
        name=name,
        strategy=strategy,
        parallel=parallel,
    )
