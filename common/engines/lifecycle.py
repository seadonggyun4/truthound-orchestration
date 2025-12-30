"""Engine Lifecycle Management for Data Quality Engines.

This module provides lifecycle management capabilities for data quality engines,
including initialization, shutdown, health checks, and resource cleanup.

Key Components:
    - ManagedEngine: Protocol for engines with lifecycle management
    - EngineConfig: Base configuration for engine initialization
    - EngineState: Engine runtime state tracking
    - EngineLifecycleManager: Manages engine lifecycle with hooks
    - EngineHealthChecker: Integrates with common.health module

Design Principles:
    1. Protocol-based: Optional lifecycle support via ManagedEngine
    2. Backwards-compatible: Existing engines continue to work
    3. Observable: Hook system for lifecycle events
    4. Resource-safe: Automatic cleanup with context managers

Example:
    >>> from common.engines.lifecycle import ManagedEngine, EngineConfig
    >>> engine = TruthoundEngine(config=TruthoundEngineConfig())
    >>> with engine:
    ...     result = engine.check(data)
    >>> # Resources automatically cleaned up

    >>> # Or with lifecycle manager
    >>> manager = EngineLifecycleManager(engine)
    >>> manager.start()
    >>> try:
    ...     result = engine.check(data)
    ... finally:
    ...     manager.stop()
"""

from __future__ import annotations

import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
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
from common.health import (
    HealthCheckResult,
    HealthStatus,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from common.engines.base import DataQualityEngine


# =============================================================================
# Exceptions
# =============================================================================


class EngineLifecycleError(TruthoundIntegrationError):
    """Base exception for engine lifecycle errors.

    Attributes:
        engine_name: Name of the engine.
        state: Current engine state.
    """

    def __init__(
        self,
        message: str,
        *,
        engine_name: str | None = None,
        state: EngineState | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize lifecycle error.

        Args:
            message: Human-readable error description.
            engine_name: Name of the engine.
            state: Current engine state.
            details: Additional error context.
            cause: Original exception.
        """
        details = details or {}
        if engine_name:
            details["engine_name"] = engine_name
        if state:
            details["state"] = state.name
        super().__init__(message, details=details, cause=cause)
        self.engine_name = engine_name
        self.state = state


class EngineNotStartedError(EngineLifecycleError):
    """Exception raised when engine operation is called before start."""

    def __init__(
        self,
        engine_name: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize not started error."""
        super().__init__(
            f"Engine '{engine_name}' is not started. Call start() first.",
            engine_name=engine_name,
            state=EngineState.CREATED,
            details=details,
            cause=cause,
        )


class EngineAlreadyStartedError(EngineLifecycleError):
    """Exception raised when start is called on already running engine."""

    def __init__(
        self,
        engine_name: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize already started error."""
        super().__init__(
            f"Engine '{engine_name}' is already started.",
            engine_name=engine_name,
            state=EngineState.RUNNING,
            details=details,
            cause=cause,
        )


class EngineStoppedError(EngineLifecycleError):
    """Exception raised when operation is called on stopped engine."""

    def __init__(
        self,
        engine_name: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize stopped error."""
        super().__init__(
            f"Engine '{engine_name}' has been stopped and cannot be reused.",
            engine_name=engine_name,
            state=EngineState.STOPPED,
            details=details,
            cause=cause,
        )


class EngineInitializationError(EngineLifecycleError):
    """Exception raised when engine initialization fails."""

    pass


class EngineShutdownError(EngineLifecycleError):
    """Exception raised when engine shutdown fails."""

    pass


# =============================================================================
# Enums
# =============================================================================


class EngineState(Enum):
    """Engine lifecycle state.

    State transitions:
        CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED
                            -> FAILED (from any state)
    """

    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()

    @property
    def is_active(self) -> bool:
        """Check if engine is in an active state."""
        return self in (EngineState.STARTING, EngineState.RUNNING)

    @property
    def can_start(self) -> bool:
        """Check if engine can be started."""
        return self == EngineState.CREATED

    @property
    def can_stop(self) -> bool:
        """Check if engine can be stopped."""
        return self in (EngineState.RUNNING, EngineState.STARTING, EngineState.FAILED)

    @property
    def is_terminal(self) -> bool:
        """Check if engine is in terminal state."""
        return self == EngineState.STOPPED


# =============================================================================
# Configuration Types
# =============================================================================


ConfigT = TypeVar("ConfigT", bound="EngineConfig")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Base configuration for data quality engines.

    Immutable configuration object that can be extended by specific engines.
    Use builder methods to create modified copies.

    Attributes:
        auto_start: Whether to automatically start engine on creation.
        auto_stop: Whether to automatically stop on context exit.
        health_check_enabled: Whether to enable health checks.
        health_check_interval_seconds: Interval between health checks.
        startup_timeout_seconds: Maximum time for engine startup.
        shutdown_timeout_seconds: Maximum time for engine shutdown.
        max_retries_on_failure: Max retries for failed operations.
        tags: Tags for categorization.
        metadata: Additional configuration metadata.

    Example:
        >>> config = EngineConfig(
        ...     auto_start=True,
        ...     health_check_enabled=True,
        ...     startup_timeout_seconds=30.0,
        ... )
        >>> engine = TruthoundEngine(config=config)
    """

    auto_start: bool = False
    auto_stop: bool = True
    health_check_enabled: bool = True
    health_check_interval_seconds: float = 30.0
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    max_retries_on_failure: int = 3
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self._validate_base_config()

    def _validate_base_config(self) -> None:
        """Validate base configuration values.

        This method is called from __post_init__ and can be called by
        subclasses to validate inherited fields.
        """
        if self.startup_timeout_seconds < 0:
            raise ValueError("startup_timeout_seconds must be non-negative")
        if self.shutdown_timeout_seconds < 0:
            raise ValueError("shutdown_timeout_seconds must be non-negative")
        if self.health_check_interval_seconds < 0:
            raise ValueError("health_check_interval_seconds must be non-negative")
        if self.max_retries_on_failure < 0:
            raise ValueError("max_retries_on_failure must be non-negative")

    def with_auto_start(self, auto_start: bool) -> Self:
        """Create config with auto_start setting.

        Args:
            auto_start: Whether to automatically start engine on creation.

        Returns:
            New configuration with auto_start setting.
        """
        return self._copy_with(auto_start=auto_start)

    def with_auto_stop(self, auto_stop: bool) -> Self:
        """Create config with auto_stop setting.

        Args:
            auto_stop: Whether to automatically stop on context exit.

        Returns:
            New configuration with auto_stop setting.
        """
        return self._copy_with(auto_stop=auto_stop)

    def with_health_check(
        self,
        enabled: bool = True,
        interval_seconds: float | None = None,
    ) -> Self:
        """Create config with health check settings.

        Args:
            enabled: Whether health checks are enabled.
            interval_seconds: Interval between health checks.

        Returns:
            New configuration with health check settings.
        """
        updates: dict[str, Any] = {"health_check_enabled": enabled}
        if interval_seconds is not None:
            updates["health_check_interval_seconds"] = interval_seconds
        return self._copy_with(**updates)

    def with_timeouts(
        self,
        startup_seconds: float | None = None,
        shutdown_seconds: float | None = None,
    ) -> Self:
        """Create config with timeout settings.

        Args:
            startup_seconds: Maximum startup time.
            shutdown_seconds: Maximum shutdown time.

        Returns:
            New configuration with timeout settings.
        """
        updates: dict[str, Any] = {}
        if startup_seconds is not None:
            updates["startup_timeout_seconds"] = startup_seconds
        if shutdown_seconds is not None:
            updates["shutdown_timeout_seconds"] = shutdown_seconds
        return self._copy_with(**updates)

    def with_retries(self, max_retries: int) -> Self:
        """Create config with retry setting.

        Args:
            max_retries: Maximum number of retries on failure.

        Returns:
            New configuration with retry setting.
        """
        return self._copy_with(max_retries_on_failure=max_retries)

    def with_tags(self, *tags: str) -> Self:
        """Create config with additional tags.

        Args:
            *tags: Tags to add.

        Returns:
            New configuration with tags.
        """
        return self._copy_with(tags=self.tags | frozenset(tags))

    def with_metadata(self, **kwargs: Any) -> Self:
        """Create config with additional metadata.

        Args:
            **kwargs: Metadata key-value pairs to add.

        Returns:
            New configuration with metadata.
        """
        return self._copy_with(metadata={**self.metadata, **kwargs})

    def _copy_with(self: Self, **updates: Any) -> Self:
        """Create a copy with updated fields.

        This is a helper method for builder pattern implementation.
        Subclasses can use this to implement their builder methods.

        Args:
            **updates: Fields to update.

        Returns:
            New configuration with updated fields.
        """
        from dataclasses import fields as dataclass_fields

        current_values = {}
        for f in dataclass_fields(self):
            current_values[f.name] = getattr(self, f.name)
        current_values.update(updates)
        return self.__class__(**current_values)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import fields as dataclass_fields

        result: dict[str, Any] = {}
        for f in dataclass_fields(self):
            value = getattr(self, f.name)
            if isinstance(value, frozenset):
                value = list(value)
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary."""
        from dataclasses import fields as dataclass_fields

        # Convert list back to frozenset for tags
        if "tags" in data and isinstance(data["tags"], list):
            data = dict(data)
            data["tags"] = frozenset(data["tags"])

        # Filter to only known fields
        known_fields = {f.name for f in dataclass_fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)


# Default configurations
DEFAULT_ENGINE_CONFIG = EngineConfig()

PRODUCTION_ENGINE_CONFIG = EngineConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    health_check_interval_seconds=60.0,
    startup_timeout_seconds=60.0,
    shutdown_timeout_seconds=30.0,
    max_retries_on_failure=5,
)

DEVELOPMENT_ENGINE_CONFIG = EngineConfig(
    auto_start=False,
    auto_stop=True,
    health_check_enabled=False,
    startup_timeout_seconds=10.0,
    shutdown_timeout_seconds=5.0,
    max_retries_on_failure=1,
)

TESTING_ENGINE_CONFIG = EngineConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=False,
    startup_timeout_seconds=5.0,
    shutdown_timeout_seconds=2.0,
    max_retries_on_failure=0,
)


# =============================================================================
# Engine State Snapshot
# =============================================================================


@dataclass(frozen=True, slots=True)
class EngineStateSnapshot:
    """Immutable snapshot of engine state at a point in time.

    Attributes:
        state: Current engine state.
        started_at: Timestamp when engine started.
        stopped_at: Timestamp when engine stopped.
        last_health_check_at: Timestamp of last health check.
        last_health_status: Result of last health check.
        error_count: Number of errors since start.
        operation_count: Number of operations executed.
        uptime_seconds: Time since engine started.
        metadata: Additional state metadata.
    """

    state: EngineState
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_health_check_at: datetime | None = None
    last_health_status: HealthStatus | None = None
    error_count: int = 0
    operation_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def uptime_seconds(self) -> float | None:
        """Calculate uptime in seconds."""
        if self.started_at is None:
            return None
        end_time = self.stopped_at or datetime.now(UTC)
        return (end_time - self.started_at).total_seconds()

    @property
    def is_healthy(self) -> bool:
        """Check if engine is healthy."""
        return (
            self.state == EngineState.RUNNING
            and self.last_health_status in (HealthStatus.HEALTHY, None)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state.name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "last_health_check_at": (
                self.last_health_check_at.isoformat()
                if self.last_health_check_at
                else None
            ),
            "last_health_status": (
                self.last_health_status.name if self.last_health_status else None
            ),
            "error_count": self.error_count,
            "operation_count": self.operation_count,
            "uptime_seconds": self.uptime_seconds,
            "is_healthy": self.is_healthy,
            "metadata": self.metadata,
        }


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class ManagedEngine(Protocol):
    """Protocol for engines with lifecycle management.

    This extends the base DataQualityEngine protocol with lifecycle methods.
    Engines implementing this protocol support:
    - Explicit start/stop lifecycle
    - Health checking
    - Resource management via context managers
    - State introspection

    State Diagram:
        ┌─────────┐    start()    ┌──────────┐
        │ CREATED │──────────────>│ STARTING │
        └─────────┘               └────┬─────┘
                                       │
                                       v
        ┌─────────┐    stop()     ┌──────────┐
        │ STOPPED │<──────────────│ RUNNING  │
        └─────────┘               └────┬─────┘
                                       │
                                       v
                                  ┌──────────┐
                                  │ STOPPING │
                                  └────┬─────┘
                                       │
                                       v
                                  ┌──────────┐
                                  │ STOPPED  │
                                  └──────────┘

    Example:
        >>> class MyEngine:
        ...     def start(self) -> None:
        ...         # Initialize resources
        ...         self._connection = create_connection()
        ...
        ...     def stop(self) -> None:
        ...         # Cleanup resources
        ...         self._connection.close()
        ...
        ...     def health_check(self) -> HealthCheckResult:
        ...         return HealthCheckResult.healthy(self.engine_name)
    """

    @property
    def engine_name(self) -> str:
        """Return engine name."""
        ...

    @property
    def engine_version(self) -> str:
        """Return engine version."""
        ...

    def start(self) -> None:
        """Start the engine and initialize resources.

        This method should:
        - Initialize any connections (database, API, etc.)
        - Allocate resources (memory pools, thread pools)
        - Perform any required setup

        Raises:
            EngineAlreadyStartedError: If engine is already running.
            EngineInitializationError: If initialization fails.
        """
        ...

    def stop(self) -> None:
        """Stop the engine and cleanup resources.

        This method should:
        - Close all connections
        - Release all resources
        - Perform cleanup operations

        Raises:
            EngineShutdownError: If shutdown fails.
        """
        ...

    def health_check(self) -> HealthCheckResult:
        """Perform health check on the engine.

        Returns:
            HealthCheckResult with engine health status.
        """
        ...

    def get_state(self) -> EngineState:
        """Return current engine state.

        Returns:
            Current EngineState.
        """
        ...


@runtime_checkable
class AsyncManagedEngine(Protocol):
    """Protocol for async engines with lifecycle management."""

    @property
    def engine_name(self) -> str:
        """Return engine name."""
        ...

    @property
    def engine_version(self) -> str:
        """Return engine version."""
        ...

    async def start(self) -> None:
        """Start the engine asynchronously."""
        ...

    async def stop(self) -> None:
        """Stop the engine asynchronously."""
        ...

    async def health_check(self) -> HealthCheckResult:
        """Perform async health check."""
        ...

    def get_state(self) -> EngineState:
        """Return current engine state."""
        ...


@runtime_checkable
class LifecycleHook(Protocol):
    """Protocol for engine lifecycle event hooks."""

    @abstractmethod
    def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Called before engine starts."""
        ...

    @abstractmethod
    def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called after engine starts successfully."""
        ...

    @abstractmethod
    def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Called before engine stops."""
        ...

    @abstractmethod
    def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called after engine stops."""
        ...

    @abstractmethod
    def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Called when lifecycle error occurs."""
        ...

    @abstractmethod
    def on_health_check(
        self,
        engine_name: str,
        result: HealthCheckResult,
        context: dict[str, Any],
    ) -> None:
        """Called after health check."""
        ...


# =============================================================================
# Lifecycle Hooks Implementation
# =============================================================================


class LoggingLifecycleHook:
    """Hook that logs lifecycle events."""

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.engines.lifecycle).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.engines.lifecycle")

    def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Log engine starting."""
        self._logger.info(
            "Engine starting",
            engine_name=engine_name,
            **context,
        )

    def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log engine started."""
        self._logger.info(
            "Engine started",
            engine_name=engine_name,
            duration_ms=duration_ms,
            **context,
        )

    def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Log engine stopping."""
        self._logger.info(
            "Engine stopping",
            engine_name=engine_name,
            **context,
        )

    def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log engine stopped."""
        self._logger.info(
            "Engine stopped",
            engine_name=engine_name,
            duration_ms=duration_ms,
            **context,
        )

    def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Log engine error."""
        self._logger.error(
            "Engine lifecycle error",
            engine_name=engine_name,
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
        )

    def on_health_check(
        self,
        engine_name: str,
        result: HealthCheckResult,
        context: dict[str, Any],
    ) -> None:
        """Log health check result."""
        log_method = self._logger.info
        if result.status == HealthStatus.UNHEALTHY:
            log_method = self._logger.error
        elif result.status in {HealthStatus.DEGRADED, HealthStatus.UNKNOWN}:
            log_method = self._logger.warning

        log_method(
            "Engine health check",
            engine_name=engine_name,
            health_status=result.status.name,
            message=result.message,
            **context,
        )


class MetricsLifecycleHook:
    """Hook that collects lifecycle metrics."""

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._start_counts: dict[str, int] = {}
        self._stop_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._health_check_counts: dict[str, dict[HealthStatus, int]] = {}
        self._total_startup_time_ms: dict[str, float] = {}
        self._total_shutdown_time_ms: dict[str, float] = {}
        self._lock = threading.Lock()

    def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record start attempt."""
        pass  # Recorded in on_started

    def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record successful start."""
        with self._lock:
            self._start_counts[engine_name] = (
                self._start_counts.get(engine_name, 0) + 1
            )
            self._total_startup_time_ms[engine_name] = (
                self._total_startup_time_ms.get(engine_name, 0.0) + duration_ms
            )

    def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record stop attempt."""
        pass  # Recorded in on_stopped

    def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record successful stop."""
        with self._lock:
            self._stop_counts[engine_name] = self._stop_counts.get(engine_name, 0) + 1
            self._total_shutdown_time_ms[engine_name] = (
                self._total_shutdown_time_ms.get(engine_name, 0.0) + duration_ms
            )

    def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record error."""
        with self._lock:
            self._error_counts[engine_name] = (
                self._error_counts.get(engine_name, 0) + 1
            )

    def on_health_check(
        self,
        engine_name: str,
        result: HealthCheckResult,
        context: dict[str, Any],
    ) -> None:
        """Record health check result."""
        with self._lock:
            if engine_name not in self._health_check_counts:
                self._health_check_counts[engine_name] = {}
            status_counts = self._health_check_counts[engine_name]
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

    def get_start_count(self, engine_name: str) -> int:
        """Get start count for engine."""
        with self._lock:
            return self._start_counts.get(engine_name, 0)

    def get_stop_count(self, engine_name: str) -> int:
        """Get stop count for engine."""
        with self._lock:
            return self._stop_counts.get(engine_name, 0)

    def get_error_count(self, engine_name: str) -> int:
        """Get error count for engine."""
        with self._lock:
            return self._error_counts.get(engine_name, 0)

    def get_average_startup_time_ms(self, engine_name: str) -> float:
        """Get average startup time for engine."""
        with self._lock:
            count = self._start_counts.get(engine_name, 0)
            if count == 0:
                return 0.0
            return self._total_startup_time_ms.get(engine_name, 0.0) / count

    def get_health_status_counts(self, engine_name: str) -> dict[HealthStatus, int]:
        """Get health check status counts for engine."""
        with self._lock:
            return dict(self._health_check_counts.get(engine_name, {}))

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._start_counts.clear()
            self._stop_counts.clear()
            self._error_counts.clear()
            self._health_check_counts.clear()
            self._total_startup_time_ms.clear()
            self._total_shutdown_time_ms.clear()


class CompositeLifecycleHook:
    """Combines multiple lifecycle hooks."""

    def __init__(self, hooks: Sequence[LifecycleHook] | None = None) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks: list[LifecycleHook] = list(hooks or [])

    def add_hook(self, hook: LifecycleHook) -> None:
        """Add a hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: LifecycleHook) -> None:
        """Remove a hook."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def _call_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call method on all hooks, suppressing exceptions."""
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                getattr(hook, method)(*args, **kwargs)

    def on_starting(self, engine_name: str, context: dict[str, Any]) -> None:
        """Call on_starting on all hooks."""
        self._call_hooks("on_starting", engine_name, context)

    def on_started(
        self, engine_name: str, duration_ms: float, context: dict[str, Any]
    ) -> None:
        """Call on_started on all hooks."""
        self._call_hooks("on_started", engine_name, duration_ms, context)

    def on_stopping(self, engine_name: str, context: dict[str, Any]) -> None:
        """Call on_stopping on all hooks."""
        self._call_hooks("on_stopping", engine_name, context)

    def on_stopped(
        self, engine_name: str, duration_ms: float, context: dict[str, Any]
    ) -> None:
        """Call on_stopped on all hooks."""
        self._call_hooks("on_stopped", engine_name, duration_ms, context)

    def on_error(
        self, engine_name: str, error: Exception, context: dict[str, Any]
    ) -> None:
        """Call on_error on all hooks."""
        self._call_hooks("on_error", engine_name, error, context)

    def on_health_check(
        self, engine_name: str, result: HealthCheckResult, context: dict[str, Any]
    ) -> None:
        """Call on_health_check on all hooks."""
        self._call_hooks("on_health_check", engine_name, result, context)


# =============================================================================
# Engine State Tracker
# =============================================================================


class EngineStateTracker:
    """Thread-safe engine state tracking.

    Tracks engine state transitions, timing, and operation counts.
    """

    def __init__(self, engine_name: str) -> None:
        """Initialize state tracker.

        Args:
            engine_name: Name of the engine being tracked.
        """
        self._engine_name = engine_name
        self._state = EngineState.CREATED
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._last_health_check_at: datetime | None = None
        self._last_health_status: HealthStatus | None = None
        self._error_count = 0
        self._operation_count = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> EngineState:
        """Get current state."""
        with self._lock:
            return self._state

    def transition_to(self, new_state: EngineState) -> EngineState:
        """Transition to new state.

        Args:
            new_state: Target state.

        Returns:
            Previous state.

        Raises:
            EngineLifecycleError: If transition is invalid.
        """
        with self._lock:
            old_state = self._state
            self._state = new_state

            if new_state == EngineState.RUNNING:
                self._started_at = datetime.now(UTC)
            elif new_state == EngineState.STOPPED:
                self._stopped_at = datetime.now(UTC)

            return old_state

    def record_health_check(self, status: HealthStatus) -> None:
        """Record health check result.

        Args:
            status: Health check status.
        """
        with self._lock:
            self._last_health_check_at = datetime.now(UTC)
            self._last_health_status = status

    def record_error(self) -> int:
        """Record an error.

        Returns:
            New error count.
        """
        with self._lock:
            self._error_count += 1
            return self._error_count

    def record_operation(self) -> int:
        """Record an operation.

        Returns:
            New operation count.
        """
        with self._lock:
            self._operation_count += 1
            return self._operation_count

    def get_snapshot(self) -> EngineStateSnapshot:
        """Get immutable snapshot of current state.

        Returns:
            EngineStateSnapshot.
        """
        with self._lock:
            return EngineStateSnapshot(
                state=self._state,
                started_at=self._started_at,
                stopped_at=self._stopped_at,
                last_health_check_at=self._last_health_check_at,
                last_health_status=self._last_health_status,
                error_count=self._error_count,
                operation_count=self._operation_count,
            )


# =============================================================================
# Engine Lifecycle Manager
# =============================================================================


class EngineLifecycleManager(Generic[ConfigT]):
    """Manages engine lifecycle with hooks and state tracking.

    Provides a wrapper around engines that adds:
    - Lifecycle management (start/stop)
    - State tracking
    - Health check integration
    - Hook notifications
    - Context manager support

    Example:
        >>> engine = TruthoundEngine()
        >>> manager = EngineLifecycleManager(engine)
        >>> with manager:
        ...     result = engine.check(data)
    """

    def __init__(
        self,
        engine: DataQualityEngine | ManagedEngine,
        config: ConfigT | None = None,
        hooks: Sequence[LifecycleHook] | None = None,
    ) -> None:
        """Initialize lifecycle manager.

        Args:
            engine: Engine to manage.
            config: Lifecycle configuration.
            hooks: Lifecycle event hooks.
        """
        self._engine = engine
        self._config: ConfigT = config or DEFAULT_ENGINE_CONFIG  # type: ignore
        self._hook: LifecycleHook | None = None
        if hooks:
            self._hook = CompositeLifecycleHook(list(hooks))
        self._state_tracker = EngineStateTracker(engine.engine_name)
        self._lock = threading.RLock()

    @property
    def engine(self) -> DataQualityEngine | ManagedEngine:
        """Get the managed engine."""
        return self._engine

    @property
    def engine_name(self) -> str:
        """Get engine name."""
        return self._engine.engine_name

    @property
    def state(self) -> EngineState:
        """Get current engine state."""
        return self._state_tracker.state

    @property
    def config(self) -> ConfigT:
        """Get configuration."""
        return self._config

    def get_state_snapshot(self) -> EngineStateSnapshot:
        """Get current state snapshot."""
        return self._state_tracker.get_snapshot()

    def _create_context(self) -> dict[str, Any]:
        """Create context dictionary for hooks."""
        return {
            "engine_name": self.engine_name,
            "engine_version": self._engine.engine_version,
            "config": self._config.to_dict(),
        }

    def start(self) -> None:
        """Start the engine.

        Raises:
            EngineAlreadyStartedError: If already started.
            EngineStoppedError: If engine was stopped.
            EngineInitializationError: If start fails.
        """
        with self._lock:
            current_state = self._state_tracker.state

            if current_state.is_terminal:
                raise EngineStoppedError(self.engine_name)
            if current_state.is_active:
                raise EngineAlreadyStartedError(self.engine_name)

            context = self._create_context()

            if self._hook:
                self._hook.on_starting(self.engine_name, context)

            self._state_tracker.transition_to(EngineState.STARTING)
            start_time = time.perf_counter()

            try:
                # Call start if engine supports it
                if isinstance(self._engine, ManagedEngine):
                    self._engine.start()

                self._state_tracker.transition_to(EngineState.RUNNING)
                duration_ms = (time.perf_counter() - start_time) * 1000

                if self._hook:
                    self._hook.on_started(self.engine_name, duration_ms, context)

            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                self._state_tracker.record_error()

                if self._hook:
                    self._hook.on_error(self.engine_name, e, context)

                raise EngineInitializationError(
                    f"Failed to start engine '{self.engine_name}': {e}",
                    engine_name=self.engine_name,
                    cause=e,
                ) from e

    def stop(self) -> None:
        """Stop the engine.

        Raises:
            EngineShutdownError: If stop fails.
        """
        with self._lock:
            current_state = self._state_tracker.state

            if current_state == EngineState.STOPPED:
                return  # Already stopped

            if not current_state.can_stop:
                return  # Nothing to stop

            context = self._create_context()

            if self._hook:
                self._hook.on_stopping(self.engine_name, context)

            self._state_tracker.transition_to(EngineState.STOPPING)
            start_time = time.perf_counter()

            try:
                # Call stop if engine supports it
                if isinstance(self._engine, ManagedEngine):
                    self._engine.stop()

                self._state_tracker.transition_to(EngineState.STOPPED)
                duration_ms = (time.perf_counter() - start_time) * 1000

                if self._hook:
                    self._hook.on_stopped(self.engine_name, duration_ms, context)

            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                self._state_tracker.record_error()

                if self._hook:
                    self._hook.on_error(self.engine_name, e, context)

                raise EngineShutdownError(
                    f"Failed to stop engine '{self.engine_name}': {e}",
                    engine_name=self.engine_name,
                    cause=e,
                ) from e

    def health_check(self) -> HealthCheckResult:
        """Perform health check on the engine.

        Returns:
            HealthCheckResult with engine health status.
        """
        context = self._create_context()

        # Check state first
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
                details={"state": state.name},
            )
            self._state_tracker.record_health_check(result.status)
            if self._hook:
                self._hook.on_health_check(self.engine_name, result, context)
            return result

        # Delegate to engine if it supports health check
        if isinstance(self._engine, ManagedEngine):
            result = self._engine.health_check()
        else:
            # Basic health check for non-managed engines
            result = HealthCheckResult.healthy(
                self.engine_name,
                message="Engine is running",
            )

        self._state_tracker.record_health_check(result.status)

        if self._hook:
            self._hook.on_health_check(self.engine_name, result, context)

        return result

    def __enter__(self) -> EngineLifecycleManager[ConfigT]:
        """Enter context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if self._config.auto_stop:
            self.stop()


# =============================================================================
# Engine Health Checker
# =============================================================================


class EngineHealthChecker:
    """Health checker that wraps an engine for use with health.py.

    Integrates engine health checks with the common.health module.

    Example:
        >>> from common.health import register_health_check
        >>> checker = EngineHealthChecker(engine)
        >>> register_health_check("truthound", checker)
    """

    def __init__(
        self,
        engine: DataQualityEngine | ManagedEngine | EngineLifecycleManager[Any],
        name: str | None = None,
    ) -> None:
        """Initialize engine health checker.

        Args:
            engine: Engine or lifecycle manager to check.
            name: Override name for health check (default: engine name).
        """
        if isinstance(engine, EngineLifecycleManager):
            self._manager = engine
            self._engine = engine.engine
        else:
            self._manager = None
            self._engine = engine

        self._name = name or self._engine.engine_name

    @property
    def name(self) -> str:
        """Return health check name."""
        return self._name

    def check(self) -> HealthCheckResult:
        """Perform health check.

        Returns:
            HealthCheckResult.
        """
        start_time = time.perf_counter()

        try:
            # Use lifecycle manager if available
            if self._manager is not None:
                result = self._manager.health_check()
            elif isinstance(self._engine, ManagedEngine):
                result = self._engine.health_check()
            else:
                # Basic health check - just verify engine is accessible
                result = HealthCheckResult.healthy(
                    self._name,
                    message=f"{self._engine.engine_name} v{self._engine.engine_version}",
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update duration
            return HealthCheckResult(
                name=result.name,
                status=result.status,
                message=result.message,
                duration_ms=duration_ms,
                details=result.details,
                dependencies=result.dependencies,
                metadata=result.metadata,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                duration_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__},
            )


# =============================================================================
# Mixin for Managed Engine Implementation
# =============================================================================


class ManagedEngineMixin:
    """Mixin providing default ManagedEngine implementation.

    Add this mixin to your engine class to get lifecycle management.
    Override _do_start, _do_stop, and _do_health_check for custom behavior.

    Example:
        >>> class MyEngine(ManagedEngineMixin, EngineInfoMixin):
        ...     def _do_start(self) -> None:
        ...         self._connection = create_connection()
        ...
        ...     def _do_stop(self) -> None:
        ...         self._connection.close()
        ...
        ...     def _do_health_check(self) -> HealthCheckResult:
        ...         if self._connection.is_alive():
        ...             return HealthCheckResult.healthy(self.engine_name)
        ...         return HealthCheckResult.unhealthy(self.engine_name)
    """

    engine_name: str
    engine_version: str

    def __init__(self, config: EngineConfig | None = None) -> None:
        """Initialize mixin.

        Args:
            config: Engine configuration.
        """
        self._lifecycle_config = config or DEFAULT_ENGINE_CONFIG
        self._state_tracker = EngineStateTracker(
            getattr(self, "engine_name", "unknown")
        )
        self._lifecycle_lock = threading.RLock()

    def _do_start(self) -> None:
        """Override to implement custom start logic."""
        pass

    def _do_stop(self) -> None:
        """Override to implement custom stop logic."""
        pass

    def _do_health_check(self) -> HealthCheckResult:
        """Override to implement custom health check logic."""
        return HealthCheckResult.healthy(
            getattr(self, "engine_name", "unknown"),
            message="Engine is running",
        )

    def start(self) -> None:
        """Start the engine."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state.is_terminal:
                raise EngineStoppedError(self.engine_name)
            if state.is_active:
                raise EngineAlreadyStartedError(self.engine_name)

            self._state_tracker.transition_to(EngineState.STARTING)
            try:
                self._do_start()
                self._state_tracker.transition_to(EngineState.RUNNING)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineInitializationError(
                    f"Failed to start engine: {e}",
                    engine_name=self.engine_name,
                    cause=e,
                ) from e

    def stop(self) -> None:
        """Stop the engine."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state == EngineState.STOPPED:
                return
            if not state.can_stop:
                return

            self._state_tracker.transition_to(EngineState.STOPPING)
            try:
                self._do_stop()
                self._state_tracker.transition_to(EngineState.STOPPED)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineShutdownError(
                    f"Failed to stop engine: {e}",
                    engine_name=self.engine_name,
                    cause=e,
                ) from e

    def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            return HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
            )

        result = self._do_health_check()
        self._state_tracker.record_health_check(result.status)
        return result

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self._state_tracker.state

    def get_state_snapshot(self) -> EngineStateSnapshot:
        """Get state snapshot."""
        return self._state_tracker.get_snapshot()

    def __enter__(self) -> Self:
        """Enter context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if self._lifecycle_config.auto_stop:
            self.stop()


# =============================================================================
# Utility Functions
# =============================================================================


def create_managed_engine(
    engine: DataQualityEngine,
    config: EngineConfig | None = None,
    hooks: Sequence[LifecycleHook] | None = None,
) -> EngineLifecycleManager[EngineConfig]:
    """Create a lifecycle manager for an engine.

    Args:
        engine: Engine to manage.
        config: Lifecycle configuration.
        hooks: Lifecycle event hooks.

    Returns:
        EngineLifecycleManager wrapping the engine.

    Example:
        >>> engine = TruthoundEngine()
        >>> managed = create_managed_engine(engine)
        >>> with managed:
        ...     result = engine.check(data)
    """
    return EngineLifecycleManager(engine, config=config, hooks=hooks)


# =============================================================================
# Async Lifecycle Hook Protocol
# =============================================================================


@runtime_checkable
class AsyncLifecycleHook(Protocol):
    """Protocol for async engine lifecycle event hooks.

    Async version of LifecycleHook for use with async engines.
    All methods are coroutines that can perform async operations.

    Example:
        >>> class MyAsyncHook:
        ...     async def on_starting(self, engine_name, context):
        ...         await notify_service(f"Engine {engine_name} starting")
    """

    @abstractmethod
    async def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Called before engine starts."""
        ...

    @abstractmethod
    async def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called after engine starts successfully."""
        ...

    @abstractmethod
    async def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Called before engine stops."""
        ...

    @abstractmethod
    async def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called after engine stops."""
        ...

    @abstractmethod
    async def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Called when lifecycle error occurs."""
        ...

    @abstractmethod
    async def on_health_check(
        self,
        engine_name: str,
        result: HealthCheckResult,
        context: dict[str, Any],
    ) -> None:
        """Called after health check."""
        ...


# =============================================================================
# Async Lifecycle Hooks Implementation
# =============================================================================


class AsyncLoggingLifecycleHook:
    """Async hook that logs lifecycle events.

    Provides the same functionality as LoggingLifecycleHook but with
    async method signatures for use with async engines.

    Example:
        >>> hook = AsyncLoggingLifecycleHook()
        >>> manager = AsyncEngineLifecycleManager(engine, hooks=[hook])
    """

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize async logging hook.

        Args:
            logger_name: Logger name (default: common.engines.lifecycle).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.engines.lifecycle")

    async def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Log engine starting."""
        self._logger.info(
            "Engine starting",
            engine_name=engine_name,
            **context,
        )

    async def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log engine started."""
        self._logger.info(
            "Engine started",
            engine_name=engine_name,
            duration_ms=duration_ms,
            **context,
        )

    async def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Log engine stopping."""
        self._logger.info(
            "Engine stopping",
            engine_name=engine_name,
            **context,
        )

    async def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log engine stopped."""
        self._logger.info(
            "Engine stopped",
            engine_name=engine_name,
            duration_ms=duration_ms,
            **context,
        )

    async def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Log engine error."""
        self._logger.error(
            "Engine lifecycle error",
            engine_name=engine_name,
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
        )

    async def on_health_check(
        self,
        engine_name: str,
        result: HealthCheckResult,
        context: dict[str, Any],
    ) -> None:
        """Log health check result."""
        log_method = self._logger.info
        if result.status == HealthStatus.UNHEALTHY:
            log_method = self._logger.error
        elif result.status in {HealthStatus.DEGRADED, HealthStatus.UNKNOWN}:
            log_method = self._logger.warning

        log_method(
            "Engine health check",
            engine_name=engine_name,
            health_status=result.status.name,
            health_message=result.message,
            **context,
        )


class AsyncMetricsLifecycleHook:
    """Async hook that collects lifecycle metrics.

    Provides the same functionality as MetricsLifecycleHook but with
    async method signatures for use with async engines.

    Example:
        >>> hook = AsyncMetricsLifecycleHook()
        >>> await engine_manager.start()
        >>> print(hook.get_start_count("truthound"))
    """

    def __init__(self) -> None:
        """Initialize async metrics collection."""
        self._start_counts: dict[str, int] = {}
        self._stop_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._health_check_counts: dict[str, dict[HealthStatus, int]] = {}
        self._total_startup_time_ms: dict[str, float] = {}
        self._total_shutdown_time_ms: dict[str, float] = {}
        self._lock = threading.Lock()

    async def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record start attempt."""
        pass  # Recorded in on_started

    async def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record successful start."""
        with self._lock:
            self._start_counts[engine_name] = (
                self._start_counts.get(engine_name, 0) + 1
            )
            self._total_startup_time_ms[engine_name] = (
                self._total_startup_time_ms.get(engine_name, 0.0) + duration_ms
            )

    async def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record stop attempt."""
        pass  # Recorded in on_stopped

    async def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record successful stop."""
        with self._lock:
            self._stop_counts[engine_name] = self._stop_counts.get(engine_name, 0) + 1
            self._total_shutdown_time_ms[engine_name] = (
                self._total_shutdown_time_ms.get(engine_name, 0.0) + duration_ms
            )

    async def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record error."""
        with self._lock:
            self._error_counts[engine_name] = (
                self._error_counts.get(engine_name, 0) + 1
            )

    async def on_health_check(
        self,
        engine_name: str,
        result: HealthCheckResult,
        context: dict[str, Any],
    ) -> None:
        """Record health check result."""
        with self._lock:
            if engine_name not in self._health_check_counts:
                self._health_check_counts[engine_name] = {}
            status_counts = self._health_check_counts[engine_name]
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

    def get_start_count(self, engine_name: str) -> int:
        """Get start count for engine."""
        with self._lock:
            return self._start_counts.get(engine_name, 0)

    def get_stop_count(self, engine_name: str) -> int:
        """Get stop count for engine."""
        with self._lock:
            return self._stop_counts.get(engine_name, 0)

    def get_error_count(self, engine_name: str) -> int:
        """Get error count for engine."""
        with self._lock:
            return self._error_counts.get(engine_name, 0)

    def get_average_startup_time_ms(self, engine_name: str) -> float:
        """Get average startup time for engine."""
        with self._lock:
            count = self._start_counts.get(engine_name, 0)
            if count == 0:
                return 0.0
            return self._total_startup_time_ms.get(engine_name, 0.0) / count

    def get_health_status_counts(self, engine_name: str) -> dict[HealthStatus, int]:
        """Get health check status counts for engine."""
        with self._lock:
            return dict(self._health_check_counts.get(engine_name, {}))

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._start_counts.clear()
            self._stop_counts.clear()
            self._error_counts.clear()
            self._health_check_counts.clear()
            self._total_startup_time_ms.clear()
            self._total_shutdown_time_ms.clear()


class AsyncCompositeLifecycleHook:
    """Combines multiple async lifecycle hooks.

    Calls all hooks in order, suppressing exceptions from individual hooks.

    Example:
        >>> hooks = AsyncCompositeLifecycleHook([
        ...     AsyncLoggingLifecycleHook(),
        ...     AsyncMetricsLifecycleHook(),
        ... ])
        >>> manager = AsyncEngineLifecycleManager(engine, hooks=[hooks])
    """

    def __init__(self, hooks: Sequence[AsyncLifecycleHook] | None = None) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of async hooks to call.
        """
        self._hooks: list[AsyncLifecycleHook] = list(hooks or [])

    def add_hook(self, hook: AsyncLifecycleHook) -> None:
        """Add a hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: AsyncLifecycleHook) -> None:
        """Remove a hook."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    async def _call_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call method on all hooks, suppressing exceptions."""
        import contextlib

        for hook in self._hooks:
            with contextlib.suppress(Exception):
                await getattr(hook, method)(*args, **kwargs)

    async def on_starting(self, engine_name: str, context: dict[str, Any]) -> None:
        """Call on_starting on all hooks."""
        await self._call_hooks("on_starting", engine_name, context)

    async def on_started(
        self, engine_name: str, duration_ms: float, context: dict[str, Any]
    ) -> None:
        """Call on_started on all hooks."""
        await self._call_hooks("on_started", engine_name, duration_ms, context)

    async def on_stopping(self, engine_name: str, context: dict[str, Any]) -> None:
        """Call on_stopping on all hooks."""
        await self._call_hooks("on_stopping", engine_name, context)

    async def on_stopped(
        self, engine_name: str, duration_ms: float, context: dict[str, Any]
    ) -> None:
        """Call on_stopped on all hooks."""
        await self._call_hooks("on_stopped", engine_name, duration_ms, context)

    async def on_error(
        self, engine_name: str, error: Exception, context: dict[str, Any]
    ) -> None:
        """Call on_error on all hooks."""
        await self._call_hooks("on_error", engine_name, error, context)

    async def on_health_check(
        self, engine_name: str, result: HealthCheckResult, context: dict[str, Any]
    ) -> None:
        """Call on_health_check on all hooks."""
        await self._call_hooks("on_health_check", engine_name, result, context)


class SyncToAsyncLifecycleHookAdapter:
    """Adapter to use sync LifecycleHook with async engines.

    Wraps a synchronous LifecycleHook to work with AsyncEngineLifecycleManager.
    Sync hooks are called in a thread pool executor to avoid blocking.

    Example:
        >>> sync_hook = LoggingLifecycleHook()
        >>> async_hook = SyncToAsyncLifecycleHookAdapter(sync_hook)
        >>> manager = AsyncEngineLifecycleManager(engine, hooks=[async_hook])
    """

    def __init__(
        self,
        sync_hook: LifecycleHook,
        *,
        use_thread_pool: bool = True,
    ) -> None:
        """Initialize adapter.

        Args:
            sync_hook: Synchronous hook to wrap.
            use_thread_pool: Whether to run sync hook in thread pool.
        """
        self._sync_hook = sync_hook
        self._use_thread_pool = use_thread_pool

    async def _call_sync(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Call sync method, optionally in thread pool."""
        import asyncio

        func = getattr(self._sync_hook, method)
        if self._use_thread_pool:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        else:
            func(*args, **kwargs)

    async def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Call sync on_starting."""
        await self._call_sync("on_starting", engine_name, context)

    async def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync on_started."""
        await self._call_sync("on_started", engine_name, duration_ms, context)

    async def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Call sync on_stopping."""
        await self._call_sync("on_stopping", engine_name, context)

    async def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync on_stopped."""
        await self._call_sync("on_stopped", engine_name, duration_ms, context)

    async def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Call sync on_error."""
        await self._call_sync("on_error", engine_name, error, context)

    async def on_health_check(
        self,
        engine_name: str,
        result: HealthCheckResult,
        context: dict[str, Any],
    ) -> None:
        """Call sync on_health_check."""
        await self._call_sync("on_health_check", engine_name, result, context)


# =============================================================================
# Async Engine Lifecycle Manager
# =============================================================================


class AsyncEngineLifecycleManager(Generic[ConfigT]):
    """Async engine lifecycle manager with hooks and state tracking.

    Provides async lifecycle management for AsyncManagedEngine or
    AsyncDataQualityEngine implementations.

    Features:
        - Async start/stop lifecycle
        - State tracking (shared with sync version)
        - Health check integration
        - Async hook notifications
        - Async context manager support

    Example:
        >>> engine = AsyncTruthoundEngine()
        >>> manager = AsyncEngineLifecycleManager(engine)
        >>> async with manager:
        ...     result = await engine.check(data)

        >>> # Or explicit lifecycle
        >>> await manager.start()
        >>> result = await engine.check(data)
        >>> await manager.stop()
    """

    def __init__(
        self,
        engine: Any,  # AsyncDataQualityEngine | AsyncManagedEngine
        config: ConfigT | None = None,
        hooks: Sequence[AsyncLifecycleHook] | None = None,
    ) -> None:
        """Initialize async lifecycle manager.

        Args:
            engine: Async engine to manage.
            config: Lifecycle configuration.
            hooks: Async lifecycle event hooks.
        """
        self._engine = engine
        self._config: ConfigT = config or DEFAULT_ENGINE_CONFIG  # type: ignore
        self._hook: AsyncLifecycleHook | None = None
        if hooks:
            self._hook = AsyncCompositeLifecycleHook(list(hooks))
        self._state_tracker = EngineStateTracker(engine.engine_name)
        self._lock = threading.RLock()

    @property
    def engine(self) -> Any:
        """Get the managed async engine."""
        return self._engine

    @property
    def engine_name(self) -> str:
        """Get engine name."""
        return self._engine.engine_name

    @property
    def state(self) -> EngineState:
        """Get current engine state."""
        return self._state_tracker.state

    @property
    def config(self) -> ConfigT:
        """Get configuration."""
        return self._config

    def get_state_snapshot(self) -> EngineStateSnapshot:
        """Get current state snapshot."""
        return self._state_tracker.get_snapshot()

    def _create_context(self) -> dict[str, Any]:
        """Create context dictionary for hooks."""
        return {
            "engine_name": self.engine_name,
            "engine_version": self._engine.engine_version,
            "config": self._config.to_dict(),
            "async": True,
        }

    async def start(self) -> None:
        """Start the engine asynchronously.

        Raises:
            EngineAlreadyStartedError: If already started.
            EngineStoppedError: If engine was stopped.
            EngineInitializationError: If start fails.
        """
        with self._lock:
            current_state = self._state_tracker.state

            if current_state.is_terminal:
                raise EngineStoppedError(self.engine_name)
            if current_state.is_active:
                raise EngineAlreadyStartedError(self.engine_name)

        context = self._create_context()

        if self._hook:
            await self._hook.on_starting(self.engine_name, context)

        with self._lock:
            self._state_tracker.transition_to(EngineState.STARTING)

        start_time = time.perf_counter()

        try:
            # Call start if engine supports it
            if isinstance(self._engine, AsyncManagedEngine):
                await self._engine.start()

            with self._lock:
                self._state_tracker.transition_to(EngineState.RUNNING)

            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                await self._hook.on_started(self.engine_name, duration_ms, context)

        except Exception as e:
            with self._lock:
                self._state_tracker.transition_to(EngineState.FAILED)
                self._state_tracker.record_error()

            if self._hook:
                await self._hook.on_error(self.engine_name, e, context)

            raise EngineInitializationError(
                f"Failed to start async engine '{self.engine_name}': {e}",
                engine_name=self.engine_name,
                cause=e,
            ) from e

    async def stop(self) -> None:
        """Stop the engine asynchronously.

        Raises:
            EngineShutdownError: If stop fails.
        """
        with self._lock:
            current_state = self._state_tracker.state

            if current_state == EngineState.STOPPED:
                return  # Already stopped

            if not current_state.can_stop:
                return  # Nothing to stop

        context = self._create_context()

        if self._hook:
            await self._hook.on_stopping(self.engine_name, context)

        with self._lock:
            self._state_tracker.transition_to(EngineState.STOPPING)

        start_time = time.perf_counter()

        try:
            # Call stop if engine supports it
            if isinstance(self._engine, AsyncManagedEngine):
                await self._engine.stop()

            with self._lock:
                self._state_tracker.transition_to(EngineState.STOPPED)

            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                await self._hook.on_stopped(self.engine_name, duration_ms, context)

        except Exception as e:
            with self._lock:
                self._state_tracker.transition_to(EngineState.FAILED)
                self._state_tracker.record_error()

            if self._hook:
                await self._hook.on_error(self.engine_name, e, context)

            raise EngineShutdownError(
                f"Failed to stop async engine '{self.engine_name}': {e}",
                engine_name=self.engine_name,
                cause=e,
            ) from e

    async def health_check(self) -> HealthCheckResult:
        """Perform async health check on the engine.

        Returns:
            HealthCheckResult with engine health status.
        """
        context = self._create_context()

        # Check state first
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
                details={"state": state.name},
            )
            self._state_tracker.record_health_check(result.status)
            if self._hook:
                await self._hook.on_health_check(self.engine_name, result, context)
            return result

        # Delegate to engine if it supports health check
        if isinstance(self._engine, AsyncManagedEngine):
            result = await self._engine.health_check()
        else:
            # Basic health check for non-managed engines
            result = HealthCheckResult.healthy(
                self.engine_name,
                message="Async engine is running",
            )

        self._state_tracker.record_health_check(result.status)

        if self._hook:
            await self._hook.on_health_check(self.engine_name, result, context)

        return result

    async def __aenter__(self) -> AsyncEngineLifecycleManager[ConfigT]:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        if self._config.auto_stop:
            await self.stop()


# =============================================================================
# Async Engine Health Checker
# =============================================================================


class AsyncEngineHealthChecker:
    """Async health checker for async engines.

    Integrates async engine health checks with health monitoring systems.

    Example:
        >>> checker = AsyncEngineHealthChecker(async_engine)
        >>> result = await checker.check()
    """

    def __init__(
        self,
        engine: Any,  # AsyncDataQualityEngine | AsyncManagedEngine | AsyncEngineLifecycleManager
        name: str | None = None,
    ) -> None:
        """Initialize async engine health checker.

        Args:
            engine: Async engine or lifecycle manager to check.
            name: Override name for health check (default: engine name).
        """
        if isinstance(engine, AsyncEngineLifecycleManager):
            self._manager = engine
            self._engine = engine.engine
        else:
            self._manager = None
            self._engine = engine

        self._name = name or self._engine.engine_name

    @property
    def name(self) -> str:
        """Return health check name."""
        return self._name

    async def check(self) -> HealthCheckResult:
        """Perform async health check.

        Returns:
            HealthCheckResult.
        """
        start_time = time.perf_counter()

        try:
            # Use lifecycle manager if available
            if self._manager is not None:
                result = await self._manager.health_check()
            elif isinstance(self._engine, AsyncManagedEngine):
                result = await self._engine.health_check()
            else:
                # Basic health check - just verify engine is accessible
                result = HealthCheckResult.healthy(
                    self._name,
                    message=f"{self._engine.engine_name} v{self._engine.engine_version}",
                )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update duration
            return HealthCheckResult(
                name=result.name,
                status=result.status,
                message=result.message,
                duration_ms=duration_ms,
                details=result.details,
                dependencies=result.dependencies,
                metadata=result.metadata,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Async health check failed: {e}",
                duration_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__},
            )


# =============================================================================
# Async Managed Engine Mixin
# =============================================================================


class AsyncManagedEngineMixin:
    """Mixin providing default AsyncManagedEngine implementation.

    Add this mixin to your async engine class to get lifecycle management.
    Override _do_start, _do_stop, and _do_health_check for custom behavior.

    All lifecycle methods are async and can perform async operations.

    Example:
        >>> class MyAsyncEngine(AsyncManagedEngineMixin, EngineInfoMixin):
        ...     async def _do_start(self) -> None:
        ...         self._connection = await create_async_connection()
        ...
        ...     async def _do_stop(self) -> None:
        ...         await self._connection.close()
        ...
        ...     async def _do_health_check(self) -> HealthCheckResult:
        ...         if await self._connection.ping():
        ...             return HealthCheckResult.healthy(self.engine_name)
        ...         return HealthCheckResult.unhealthy(self.engine_name)
        ...
        ...     async def check(self, data, rules, **kwargs):
        ...         # Async validation
        ...         ...

    Usage:
        >>> async with MyAsyncEngine() as engine:
        ...     result = await engine.check(data, rules)
    """

    engine_name: str
    engine_version: str

    def __init__(self, config: EngineConfig | None = None) -> None:
        """Initialize async mixin.

        Args:
            config: Engine configuration.
        """
        self._lifecycle_config = config or DEFAULT_ENGINE_CONFIG
        self._state_tracker = EngineStateTracker(
            getattr(self, "engine_name", "unknown")
        )
        self._lifecycle_lock = threading.RLock()

    async def _do_start(self) -> None:
        """Override to implement custom async start logic.

        This method is called during start() and should perform any
        async initialization (database connections, API setup, etc.).
        """
        pass

    async def _do_stop(self) -> None:
        """Override to implement custom async stop logic.

        This method is called during stop() and should perform any
        async cleanup (close connections, flush buffers, etc.).
        """
        pass

    async def _do_health_check(self) -> HealthCheckResult:
        """Override to implement custom async health check logic.

        Returns:
            HealthCheckResult indicating engine health.
        """
        return HealthCheckResult.healthy(
            getattr(self, "engine_name", "unknown"),
            message="Async engine is running",
        )

    async def start(self) -> None:
        """Start the async engine."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state.is_terminal:
                raise EngineStoppedError(self.engine_name)
            if state.is_active:
                raise EngineAlreadyStartedError(self.engine_name)

            self._state_tracker.transition_to(EngineState.STARTING)

        try:
            await self._do_start()
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.RUNNING)
        except Exception as e:
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.FAILED)
            raise EngineInitializationError(
                f"Failed to start async engine: {e}",
                engine_name=self.engine_name,
                cause=e,
            ) from e

    async def stop(self) -> None:
        """Stop the async engine."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state == EngineState.STOPPED:
                return
            if not state.can_stop:
                return

            self._state_tracker.transition_to(EngineState.STOPPING)

        try:
            await self._do_stop()
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.STOPPED)
        except Exception as e:
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.FAILED)
            raise EngineShutdownError(
                f"Failed to stop async engine: {e}",
                engine_name=self.engine_name,
                cause=e,
            ) from e

    async def health_check(self) -> HealthCheckResult:
        """Perform async health check."""
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            return HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
            )

        result = await self._do_health_check()
        self._state_tracker.record_health_check(result.status)
        return result

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self._state_tracker.state

    def get_state_snapshot(self) -> EngineStateSnapshot:
        """Get state snapshot."""
        return self._state_tracker.get_snapshot()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        if self._lifecycle_config.auto_stop:
            await self.stop()


# =============================================================================
# Sync Engine Wrapper for Async Context
# =============================================================================


class SyncEngineAsyncAdapter:
    """Adapter to use sync DataQualityEngine in async context.

    Wraps a synchronous engine to provide an async interface by running
    sync operations in a thread pool executor. This allows using existing
    sync engines with async frameworks like Prefect or FastAPI.

    Features:
        - Async check/profile/learn methods
        - Thread pool execution for sync operations
        - Configurable executor
        - Full lifecycle support if underlying engine supports it

    Example:
        >>> from common.engines import TruthoundEngine
        >>> sync_engine = TruthoundEngine()
        >>> async_engine = SyncEngineAsyncAdapter(sync_engine)
        >>> async with async_engine:
        ...     result = await async_engine.check(data, rules)
    """

    def __init__(
        self,
        engine: Any,  # DataQualityEngine | ManagedEngine
        *,
        executor: Any | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            engine: Synchronous engine to wrap.
            executor: Optional ThreadPoolExecutor. If None, uses default.
        """
        self._engine = engine
        self._executor = executor
        self._lifecycle_config = getattr(engine, "_lifecycle_config", DEFAULT_ENGINE_CONFIG)
        self._state_tracker = EngineStateTracker(engine.engine_name)
        self._lifecycle_lock = threading.RLock()

    @property
    def engine_name(self) -> str:
        """Return the engine name."""
        return self._engine.engine_name

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        return self._engine.engine_version

    @property
    def wrapped_engine(self) -> Any:
        """Get the wrapped synchronous engine."""
        return self._engine

    async def _run_in_executor(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a sync function in thread pool executor."""
        import asyncio
        from functools import partial

        loop = asyncio.get_event_loop()
        if kwargs:
            func = partial(func, **kwargs)
        return await loop.run_in_executor(self._executor, func, *args)

    async def check(
        self,
        data: Any,
        rules: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute validation check asynchronously.

        Args:
            data: Data to validate.
            rules: Validation rules.
            **kwargs: Engine-specific parameters.

        Returns:
            CheckResult from underlying engine.
        """
        return await self._run_in_executor(
            self._engine.check, data, rules, **kwargs
        )

    async def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Profile data asynchronously.

        Args:
            data: Data to profile.
            **kwargs: Engine-specific parameters.

        Returns:
            ProfileResult from underlying engine.
        """
        return await self._run_in_executor(self._engine.profile, data, **kwargs)

    async def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Learn rules asynchronously.

        Args:
            data: Data to learn from.
            **kwargs: Engine-specific parameters.

        Returns:
            LearnResult from underlying engine.
        """
        return await self._run_in_executor(self._engine.learn, data, **kwargs)

    async def start(self) -> None:
        """Start the engine asynchronously."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state.is_terminal:
                raise EngineStoppedError(self.engine_name)
            if state.is_active:
                raise EngineAlreadyStartedError(self.engine_name)

            self._state_tracker.transition_to(EngineState.STARTING)

        try:
            if isinstance(self._engine, ManagedEngine):
                await self._run_in_executor(self._engine.start)
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.RUNNING)
        except Exception as e:
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.FAILED)
            raise EngineInitializationError(
                f"Failed to start engine: {e}",
                engine_name=self.engine_name,
                cause=e,
            ) from e

    async def stop(self) -> None:
        """Stop the engine asynchronously."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state == EngineState.STOPPED:
                return
            if not state.can_stop:
                return

            self._state_tracker.transition_to(EngineState.STOPPING)

        try:
            if isinstance(self._engine, ManagedEngine):
                await self._run_in_executor(self._engine.stop)
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.STOPPED)
        except Exception as e:
            with self._lifecycle_lock:
                self._state_tracker.transition_to(EngineState.FAILED)
            raise EngineShutdownError(
                f"Failed to stop engine: {e}",
                engine_name=self.engine_name,
                cause=e,
            ) from e

    async def health_check(self) -> HealthCheckResult:
        """Perform async health check."""
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            return HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
            )

        if isinstance(self._engine, ManagedEngine):
            result = await self._run_in_executor(self._engine.health_check)
        else:
            result = HealthCheckResult.healthy(
                self.engine_name,
                message="Engine is running",
            )

        self._state_tracker.record_health_check(result.status)
        return result

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self._state_tracker.state

    def get_state_snapshot(self) -> EngineStateSnapshot:
        """Get state snapshot."""
        return self._state_tracker.get_snapshot()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        if self._lifecycle_config.auto_stop:
            await self.stop()


def create_engine_health_checker(
    engine: DataQualityEngine | ManagedEngine | EngineLifecycleManager[Any],
    name: str | None = None,
) -> EngineHealthChecker:
    """Create a health checker for an engine.

    Args:
        engine: Engine to check.
        name: Override name for health check.

    Returns:
        EngineHealthChecker instance.

    Example:
        >>> checker = create_engine_health_checker(engine)
        >>> result = checker.check()
    """
    return EngineHealthChecker(engine, name=name)


def register_engine_health_check(
    engine: DataQualityEngine | ManagedEngine | EngineLifecycleManager[Any],
    name: str | None = None,
) -> None:
    """Register engine health check in global registry.

    Args:
        engine: Engine to register.
        name: Override name for registration.

    Example:
        >>> register_engine_health_check(engine)
        >>> from common.health import check_health
        >>> result = check_health("truthound")
    """
    from common.health import register_health_check

    checker = create_engine_health_checker(engine, name)
    register_health_check(checker.name, checker)
