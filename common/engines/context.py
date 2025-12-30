"""Engine Context Manager for Data Quality Engines.

This module provides advanced context management capabilities for data quality engines,
including transaction-like sessions, nested contexts, multi-engine coordination,
resource tracking, and savepoint patterns.

Key Components:
    - EngineContext: Enhanced context manager with resource tracking
    - EngineSession: Transaction-like session management with commit/rollback
    - MultiEngineContext: Coordinate multiple engines together
    - ContextStack: Nested context management
    - ResourceTracker: Track and cleanup resources
    - Savepoint: Partial rollback support

Design Principles:
    1. Protocol-based: Flexible implementation via protocols
    2. Resource-safe: Automatic cleanup even on exceptions
    3. Composable: Contexts can be nested and combined
    4. Observable: Hook system for context events
    5. Thread-safe: Safe for concurrent use

Example:
    >>> # Simple context usage
    >>> with EngineContext(engine) as ctx:
    ...     result = ctx.execute(lambda e: e.check(data))

    >>> # Session with commit/rollback
    >>> with EngineSession(engine) as session:
    ...     session.execute(lambda e: e.check(data1))
    ...     session.create_savepoint("after_data1")
    ...     session.execute(lambda e: e.check(data2))
    ...     session.rollback_to_savepoint("after_data1")  # Rollback data2 effects

    >>> # Multiple engines
    >>> with MultiEngineContext(engine1, engine2) as ctx:
    ...     results = ctx.execute_all(lambda e: e.check(data))
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
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
from common.health import HealthCheckResult, HealthStatus


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence



# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
E = TypeVar("E")  # Engine type
R = TypeVar("R")  # Result type


# =============================================================================
# Exceptions
# =============================================================================


class ContextError(TruthoundIntegrationError):
    """Base exception for context management errors.

    Attributes:
        context_id: Unique identifier of the context.
    """

    def __init__(
        self,
        message: str,
        *,
        context_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize context error.

        Args:
            message: Human-readable error description.
            context_id: Context identifier.
            details: Additional error context.
            cause: Original exception.
        """
        details = details or {}
        if context_id:
            details["context_id"] = context_id
        super().__init__(message, details=details, cause=cause)
        self.context_id = context_id


class ContextNotActiveError(ContextError):
    """Exception raised when operation is called outside active context."""

    def __init__(
        self,
        context_id: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize error."""
        super().__init__(
            "Context is not active. Use 'with' statement or call enter().",
            context_id=context_id,
            details=details,
            cause=cause,
        )


class ContextAlreadyActiveError(ContextError):
    """Exception raised when context is already active."""

    def __init__(
        self,
        context_id: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize error."""
        super().__init__(
            "Context is already active.",
            context_id=context_id,
            details=details,
            cause=cause,
        )


class SavepointError(ContextError):
    """Exception raised for savepoint-related errors."""

    def __init__(
        self,
        message: str,
        savepoint_name: str | None = None,
        *,
        context_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize savepoint error."""
        details = details or {}
        if savepoint_name:
            details["savepoint_name"] = savepoint_name
        super().__init__(message, context_id=context_id, details=details, cause=cause)
        self.savepoint_name = savepoint_name


class ResourceCleanupError(ContextError):
    """Exception raised when resource cleanup fails."""

    def __init__(
        self,
        message: str,
        failed_resources: list[str] | None = None,
        *,
        context_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize cleanup error."""
        details = details or {}
        if failed_resources:
            details["failed_resources"] = failed_resources
        super().__init__(message, context_id=context_id, details=details, cause=cause)
        self.failed_resources = failed_resources or []


# =============================================================================
# Enums
# =============================================================================


class ContextState(Enum):
    """State of an engine context.

    State transitions:
        CREATED -> ENTERING -> ACTIVE -> EXITING -> EXITED
                                     -> FAILED (from any active state)
    """

    CREATED = auto()
    ENTERING = auto()
    ACTIVE = auto()
    EXITING = auto()
    EXITED = auto()
    FAILED = auto()

    @property
    def is_active(self) -> bool:
        """Check if context is in an active state."""
        return self == ContextState.ACTIVE

    @property
    def can_enter(self) -> bool:
        """Check if context can be entered."""
        return self == ContextState.CREATED

    @property
    def can_exit(self) -> bool:
        """Check if context can be exited."""
        return self in (ContextState.ACTIVE, ContextState.ENTERING, ContextState.FAILED)

    @property
    def is_terminal(self) -> bool:
        """Check if context is in terminal state."""
        return self in (ContextState.EXITED, ContextState.FAILED)


class SessionState(Enum):
    """State of an engine session.

    Similar to database transaction states.
    """

    PENDING = auto()  # Session created but not started
    ACTIVE = auto()  # Session is active
    COMMITTING = auto()  # Commit in progress
    COMMITTED = auto()  # Successfully committed
    ROLLING_BACK = auto()  # Rollback in progress
    ROLLED_BACK = auto()  # Successfully rolled back
    FAILED = auto()  # Session failed


class CleanupStrategy(Enum):
    """Strategy for resource cleanup on context exit."""

    ALWAYS = auto()  # Always cleanup, regardless of success/failure
    ON_SUCCESS = auto()  # Only cleanup on successful exit
    ON_FAILURE = auto()  # Only cleanup on failure
    NEVER = auto()  # Never auto-cleanup (manual cleanup required)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Configuration for engine contexts.

    Immutable configuration for context behavior.

    Attributes:
        auto_start_engine: Automatically start engine on context enter.
        auto_stop_engine: Automatically stop engine on context exit.
        cleanup_strategy: When to cleanup resources.
        timeout_seconds: Maximum time for context operations.
        track_resources: Whether to track resources.
        enable_savepoints: Whether to enable savepoint support.
        max_nested_depth: Maximum nesting depth (0 = unlimited).
        propagate_exceptions: Whether to propagate exceptions from cleanup.
        tags: Tags for categorization.
        metadata: Additional configuration metadata.
    """

    auto_start_engine: bool = True
    auto_stop_engine: bool = True
    cleanup_strategy: CleanupStrategy = CleanupStrategy.ALWAYS
    timeout_seconds: float | None = None
    track_resources: bool = True
    enable_savepoints: bool = True
    max_nested_depth: int = 0
    propagate_exceptions: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_auto_start(self, auto_start: bool) -> Self:
        """Create config with auto_start setting."""
        return self._copy_with(auto_start_engine=auto_start)

    def with_auto_stop(self, auto_stop: bool) -> Self:
        """Create config with auto_stop setting."""
        return self._copy_with(auto_stop_engine=auto_stop)

    def with_cleanup_strategy(self, strategy: CleanupStrategy) -> Self:
        """Create config with cleanup strategy."""
        return self._copy_with(cleanup_strategy=strategy)

    def with_timeout(self, timeout_seconds: float | None) -> Self:
        """Create config with timeout setting."""
        return self._copy_with(timeout_seconds=timeout_seconds)

    def with_savepoints(self, enabled: bool) -> Self:
        """Create config with savepoint setting."""
        return self._copy_with(enable_savepoints=enabled)

    def with_resource_tracking(self, enabled: bool) -> Self:
        """Create config with resource tracking setting."""
        return self._copy_with(track_resources=enabled)

    def _copy_with(self, **updates: Any) -> Self:
        """Create a copy with updated fields."""
        from dataclasses import fields as dataclass_fields

        current_values = {}
        for f in dataclass_fields(self):
            current_values[f.name] = getattr(self, f.name)
        current_values.update(updates)
        return self.__class__(**current_values)


# Preset configurations
DEFAULT_CONTEXT_CONFIG = ContextConfig()

LIGHTWEIGHT_CONTEXT_CONFIG = ContextConfig(
    auto_start_engine=False,
    auto_stop_engine=False,
    track_resources=False,
    enable_savepoints=False,
)

STRICT_CONTEXT_CONFIG = ContextConfig(
    cleanup_strategy=CleanupStrategy.ALWAYS,
    propagate_exceptions=True,
    track_resources=True,
    enable_savepoints=True,
)

TESTING_CONTEXT_CONFIG = ContextConfig(
    auto_start_engine=True,
    auto_stop_engine=True,
    timeout_seconds=5.0,
    track_resources=True,
)


# =============================================================================
# Resource Tracking
# =============================================================================


@dataclass
class TrackedResource:
    """Information about a tracked resource.

    Attributes:
        resource_id: Unique identifier for the resource.
        resource_type: Type of resource (e.g., "connection", "file").
        name: Human-readable name.
        created_at: When the resource was created.
        cleanup_func: Optional cleanup function.
        metadata: Additional resource metadata.
    """

    resource_id: str
    resource_type: str
    name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    cleanup_func: Callable[[], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> bool:
        """Execute cleanup function if available.

        Returns:
            True if cleanup was successful, False if no cleanup needed.

        Raises:
            Exception: If cleanup function raises an exception.
        """
        if self.cleanup_func:
            self.cleanup_func()
            return True
        return False


class ResourceTracker:
    """Tracks resources created during context execution.

    Thread-safe resource tracking with automatic cleanup support.

    Example:
        >>> tracker = ResourceTracker()
        >>> tracker.register("conn1", "connection", cleanup_func=conn.close)
        >>> tracker.cleanup_all()  # Calls conn.close()
    """

    def __init__(self) -> None:
        """Initialize resource tracker."""
        self._resources: dict[str, TrackedResource] = {}
        self._lock = threading.RLock()
        self._cleanup_order: list[str] = []

    def register(
        self,
        resource_id: str,
        resource_type: str,
        *,
        name: str | None = None,
        cleanup_func: Callable[[], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrackedResource:
        """Register a resource for tracking.

        Args:
            resource_id: Unique identifier for the resource.
            resource_type: Type of resource.
            name: Human-readable name (defaults to resource_id).
            cleanup_func: Function to call for cleanup.
            metadata: Additional metadata.

        Returns:
            TrackedResource instance.
        """
        with self._lock:
            resource = TrackedResource(
                resource_id=resource_id,
                resource_type=resource_type,
                name=name or resource_id,
                cleanup_func=cleanup_func,
                metadata=metadata or {},
            )
            self._resources[resource_id] = resource
            self._cleanup_order.append(resource_id)
            return resource

    def unregister(self, resource_id: str) -> TrackedResource | None:
        """Unregister a resource.

        Args:
            resource_id: Resource identifier.

        Returns:
            The unregistered resource, or None if not found.
        """
        with self._lock:
            resource = self._resources.pop(resource_id, None)
            if resource_id in self._cleanup_order:
                self._cleanup_order.remove(resource_id)
            return resource

    def get(self, resource_id: str) -> TrackedResource | None:
        """Get a tracked resource by ID.

        Args:
            resource_id: Resource identifier.

        Returns:
            TrackedResource if found, None otherwise.
        """
        with self._lock:
            return self._resources.get(resource_id)

    def list_resources(self, resource_type: str | None = None) -> list[TrackedResource]:
        """List all tracked resources.

        Args:
            resource_type: Optional filter by resource type.

        Returns:
            List of tracked resources.
        """
        with self._lock:
            resources = list(self._resources.values())
            if resource_type:
                resources = [r for r in resources if r.resource_type == resource_type]
            return resources

    def cleanup_all(self, reverse_order: bool = True) -> list[str]:
        """Cleanup all tracked resources.

        Args:
            reverse_order: Cleanup in reverse registration order (LIFO).

        Returns:
            List of resource IDs that failed cleanup.
        """
        failed: list[str] = []

        with self._lock:
            order = list(reversed(self._cleanup_order)) if reverse_order else list(self._cleanup_order)

        for resource_id in order:
            try:
                resource = self.get(resource_id)
                if resource:
                    resource.cleanup()
                    self.unregister(resource_id)
            except Exception:
                failed.append(resource_id)

        return failed

    def __len__(self) -> int:
        """Return number of tracked resources."""
        with self._lock:
            return len(self._resources)

    def __contains__(self, resource_id: str) -> bool:
        """Check if resource is tracked."""
        with self._lock:
            return resource_id in self._resources

    def __bool__(self) -> bool:
        """Return True always (tracker exists, may be empty)."""
        return True


# =============================================================================
# Savepoint Support
# =============================================================================


@dataclass(frozen=True, slots=True)
class Savepoint:
    """Represents a savepoint in a session.

    Savepoints allow partial rollback to a specific point.

    Attributes:
        name: Unique name for the savepoint.
        created_at: When the savepoint was created.
        sequence: Sequence number for ordering.
        state_snapshot: Captured state at savepoint creation.
        metadata: Additional savepoint metadata.
    """

    name: str
    created_at: datetime
    sequence: int
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SavepointManager:
    """Manages savepoints within a session.

    Thread-safe savepoint management with rollback support.

    Example:
        >>> manager = SavepointManager()
        >>> manager.create("sp1", {"data": [1, 2, 3]})
        >>> manager.create("sp2", {"data": [1, 2, 3, 4]})
        >>> snapshot = manager.rollback_to("sp1")  # Returns sp1's snapshot
    """

    def __init__(self) -> None:
        """Initialize savepoint manager."""
        self._savepoints: dict[str, Savepoint] = {}
        self._sequence = 0
        self._lock = threading.RLock()

    def create(
        self,
        name: str,
        state_snapshot: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Savepoint:
        """Create a new savepoint.

        Args:
            name: Unique name for the savepoint.
            state_snapshot: State to capture.
            metadata: Additional metadata.

        Returns:
            Created Savepoint.

        Raises:
            SavepointError: If savepoint name already exists.
        """
        with self._lock:
            if name in self._savepoints:
                raise SavepointError(f"Savepoint '{name}' already exists", name)

            self._sequence += 1
            savepoint = Savepoint(
                name=name,
                created_at=datetime.now(UTC),
                sequence=self._sequence,
                state_snapshot=state_snapshot or {},
                metadata=metadata or {},
            )
            self._savepoints[name] = savepoint
            return savepoint

    def get(self, name: str) -> Savepoint | None:
        """Get a savepoint by name.

        Args:
            name: Savepoint name.

        Returns:
            Savepoint if found, None otherwise.
        """
        with self._lock:
            return self._savepoints.get(name)

    def exists(self, name: str) -> bool:
        """Check if savepoint exists.

        Args:
            name: Savepoint name.

        Returns:
            True if exists, False otherwise.
        """
        with self._lock:
            return name in self._savepoints

    def release(self, name: str) -> Savepoint | None:
        """Release (delete) a savepoint.

        Args:
            name: Savepoint name.

        Returns:
            Released savepoint, or None if not found.
        """
        with self._lock:
            return self._savepoints.pop(name, None)

    def rollback_to(self, name: str) -> dict[str, Any]:
        """Rollback to a savepoint.

        Releases all savepoints created after the target savepoint.

        Args:
            name: Savepoint name to rollback to.

        Returns:
            State snapshot from the savepoint.

        Raises:
            SavepointError: If savepoint not found.
        """
        with self._lock:
            savepoint = self._savepoints.get(name)
            if not savepoint:
                raise SavepointError(f"Savepoint '{name}' not found", name)

            # Remove all savepoints created after this one
            to_remove = [
                sp_name
                for sp_name, sp in self._savepoints.items()
                if sp.sequence > savepoint.sequence
            ]
            for sp_name in to_remove:
                del self._savepoints[sp_name]

            return dict(savepoint.state_snapshot)

    def list_savepoints(self) -> list[Savepoint]:
        """List all savepoints in creation order.

        Returns:
            List of savepoints ordered by creation sequence.
        """
        with self._lock:
            return sorted(self._savepoints.values(), key=lambda sp: sp.sequence)

    def clear(self) -> int:
        """Clear all savepoints.

        Returns:
            Number of savepoints cleared.
        """
        with self._lock:
            count = len(self._savepoints)
            self._savepoints.clear()
            return count


# =============================================================================
# Context Protocols
# =============================================================================


@runtime_checkable
class ContextHook(Protocol):
    """Protocol for context event hooks."""

    @abstractmethod
    def on_enter(
        self,
        context_id: str,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Called when context is entered."""
        ...

    @abstractmethod
    def on_exit(
        self,
        context_id: str,
        engine_name: str,
        success: bool,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when context is exited."""
        ...

    @abstractmethod
    def on_execute(
        self,
        context_id: str,
        engine_name: str,
        operation: str,
        duration_ms: float,
        success: bool,
        context: dict[str, Any],
    ) -> None:
        """Called after operation execution."""
        ...

    @abstractmethod
    def on_error(
        self,
        context_id: str,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Called when error occurs."""
        ...


# =============================================================================
# Context Hook Implementations
# =============================================================================


class LoggingContextHook:
    """Hook that logs context events."""

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.engines.context).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.engines.context")

    def on_enter(
        self,
        context_id: str,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Log context enter."""
        self._logger.info(
            "Context entered",
            context_id=context_id,
            engine_name=engine_name,
            **{k: v for k, v in context.items() if k not in ("context_id", "engine_name")},
        )

    def on_exit(
        self,
        context_id: str,
        engine_name: str,
        success: bool,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log context exit."""
        log_method = self._logger.info if success else self._logger.warning
        log_method(
            "Context exited",
            context_id=context_id,
            engine_name=engine_name,
            success=success,
            duration_ms=duration_ms,
        )

    def on_execute(
        self,
        context_id: str,
        engine_name: str,
        operation: str,
        duration_ms: float,
        success: bool,
        context: dict[str, Any],
    ) -> None:
        """Log operation execution."""
        log_method = self._logger.info if success else self._logger.warning
        log_method(
            "Operation executed",
            context_id=context_id,
            engine_name=engine_name,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
        )

    def on_error(
        self,
        context_id: str,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Log error."""
        self._logger.error(
            "Context error",
            context_id=context_id,
            engine_name=engine_name,
            error_type=type(error).__name__,
            error_message=str(error),
        )


class MetricsContextHook:
    """Hook that collects context metrics."""

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._enter_counts: dict[str, int] = {}
        self._exit_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}
        self._total_duration_ms: dict[str, float] = {}
        self._operation_counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def on_enter(
        self,
        context_id: str,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record context enter."""
        with self._lock:
            self._enter_counts[engine_name] = self._enter_counts.get(engine_name, 0) + 1

    def on_exit(
        self,
        context_id: str,
        engine_name: str,
        success: bool,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record context exit."""
        with self._lock:
            self._exit_counts[engine_name] = self._exit_counts.get(engine_name, 0) + 1
            self._total_duration_ms[engine_name] = (
                self._total_duration_ms.get(engine_name, 0.0) + duration_ms
            )
            if success:
                self._success_counts[engine_name] = self._success_counts.get(engine_name, 0) + 1
            else:
                self._failure_counts[engine_name] = self._failure_counts.get(engine_name, 0) + 1

    def on_execute(
        self,
        context_id: str,
        engine_name: str,
        operation: str,
        duration_ms: float,
        success: bool,
        context: dict[str, Any],
    ) -> None:
        """Record operation execution."""
        with self._lock:
            if engine_name not in self._operation_counts:
                self._operation_counts[engine_name] = {}
            ops = self._operation_counts[engine_name]
            ops[operation] = ops.get(operation, 0) + 1

    def on_error(
        self,
        context_id: str,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record error (counted in on_exit with success=False)."""
        pass

    def get_enter_count(self, engine_name: str) -> int:
        """Get enter count for engine."""
        with self._lock:
            return self._enter_counts.get(engine_name, 0)

    def get_success_rate(self, engine_name: str) -> float:
        """Get success rate for engine."""
        with self._lock:
            total = self._exit_counts.get(engine_name, 0)
            if total == 0:
                return 0.0
            return self._success_counts.get(engine_name, 0) / total

    def get_average_duration_ms(self, engine_name: str) -> float:
        """Get average context duration for engine."""
        with self._lock:
            count = self._exit_counts.get(engine_name, 0)
            if count == 0:
                return 0.0
            return self._total_duration_ms.get(engine_name, 0.0) / count

    def get_operation_counts(self, engine_name: str) -> dict[str, int]:
        """Get operation counts for engine."""
        with self._lock:
            return dict(self._operation_counts.get(engine_name, {}))

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._enter_counts.clear()
            self._exit_counts.clear()
            self._success_counts.clear()
            self._failure_counts.clear()
            self._total_duration_ms.clear()
            self._operation_counts.clear()


class CompositeContextHook:
    """Combines multiple context hooks."""

    def __init__(self, hooks: Sequence[ContextHook] | None = None) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks: list[ContextHook] = list(hooks or [])

    def add_hook(self, hook: ContextHook) -> None:
        """Add a hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: ContextHook) -> None:
        """Remove a hook."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def _call_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call method on all hooks, suppressing exceptions."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                getattr(hook, method)(*args, **kwargs)

    def on_enter(
        self,
        context_id: str,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Call on_enter on all hooks."""
        self._call_hooks("on_enter", context_id, engine_name, context)

    def on_exit(
        self,
        context_id: str,
        engine_name: str,
        success: bool,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_exit on all hooks."""
        self._call_hooks("on_exit", context_id, engine_name, success, duration_ms, context)

    def on_execute(
        self,
        context_id: str,
        engine_name: str,
        operation: str,
        duration_ms: float,
        success: bool,
        context: dict[str, Any],
    ) -> None:
        """Call on_execute on all hooks."""
        self._call_hooks(
            "on_execute", context_id, engine_name, operation, duration_ms, success, context
        )

    def on_error(
        self,
        context_id: str,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Call on_error on all hooks."""
        self._call_hooks("on_error", context_id, engine_name, error, context)


# =============================================================================
# Engine Context
# =============================================================================


class EngineContext(Generic[E]):
    """Enhanced context manager for engine operations.

    Provides resource tracking, cleanup guarantees, and operation execution
    within a managed context.

    Features:
        - Automatic engine start/stop
        - Resource tracking and cleanup
        - Operation timing and logging
        - Hook notifications
        - Error handling

    Example:
        >>> with EngineContext(engine) as ctx:
        ...     result = ctx.execute(lambda e: e.check(data))
        ...     ctx.register_resource("conn", "connection", cleanup_func=conn.close)

        >>> # With custom config
        >>> config = ContextConfig(auto_stop_engine=False)
        >>> with EngineContext(engine, config=config) as ctx:
        ...     pass  # Engine not stopped on exit
    """

    def __init__(
        self,
        engine: E,
        *,
        config: ContextConfig | None = None,
        hooks: Sequence[ContextHook] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize engine context.

        Args:
            engine: Engine to manage.
            config: Context configuration.
            hooks: Context event hooks.
            name: Optional name for the context.
        """
        self._engine = engine
        self._config = config or DEFAULT_CONTEXT_CONFIG
        self._hook: ContextHook | None = None
        if hooks:
            self._hook = CompositeContextHook(list(hooks))

        self._context_id = str(uuid.uuid4())[:8]
        self._name = name or f"ctx-{self._context_id}"
        self._state = ContextState.CREATED
        self._enter_time: float | None = None
        self._exit_time: float | None = None
        self._exception: Exception | None = None
        self._resource_tracker = ResourceTracker() if self._config.track_resources else None
        self._lock = threading.RLock()

    @property
    def context_id(self) -> str:
        """Get unique context identifier."""
        return self._context_id

    @property
    def name(self) -> str:
        """Get context name."""
        return self._name

    @property
    def engine(self) -> E:
        """Get the managed engine."""
        self._ensure_active()
        return self._engine

    @property
    def state(self) -> ContextState:
        """Get current context state."""
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return self.state == ContextState.ACTIVE

    @property
    def duration_ms(self) -> float | None:
        """Get context duration in milliseconds."""
        if self._enter_time is None:
            return None
        end_time = self._exit_time or time.perf_counter()
        return (end_time - self._enter_time) * 1000

    def _ensure_active(self) -> None:
        """Ensure context is active.

        Raises:
            ContextNotActiveError: If context is not active.
        """
        if not self.is_active:
            raise ContextNotActiveError(self._context_id)

    def _create_hook_context(self) -> dict[str, Any]:
        """Create context dictionary for hooks."""
        return {
            "context_id": self._context_id,
            "name": self._name,
            "config": self._config,
            "state": self._state.name,
        }

    def _start_engine(self) -> None:
        """Start the engine if configured and engine supports it."""
        from common.engines.lifecycle import ManagedEngine

        if self._config.auto_start_engine and isinstance(self._engine, ManagedEngine):
            self._engine.start()

    def _stop_engine(self) -> None:
        """Stop the engine if configured and engine supports it."""
        from common.engines.lifecycle import ManagedEngine

        if self._config.auto_stop_engine and isinstance(self._engine, ManagedEngine):
            self._engine.stop()

    def enter(self) -> Self:
        """Enter the context explicitly.

        Returns:
            Self for chaining.

        Raises:
            ContextAlreadyActiveError: If context is already active.
        """
        with self._lock:
            if not self._state.can_enter:
                raise ContextAlreadyActiveError(self._context_id)

            self._state = ContextState.ENTERING
            self._enter_time = time.perf_counter()

        try:
            self._start_engine()

            with self._lock:
                self._state = ContextState.ACTIVE

            if self._hook:
                self._hook.on_enter(
                    self._context_id,
                    self._engine.engine_name,
                    self._create_hook_context(),
                )

        except Exception as e:
            with self._lock:
                self._state = ContextState.FAILED
                self._exception = e

            if self._hook:
                self._hook.on_error(
                    self._context_id,
                    self._engine.engine_name,
                    e,
                    self._create_hook_context(),
                )

            raise

        return self

    def exit(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: Any = None,
    ) -> bool:
        """Exit the context explicitly.

        Args:
            exc_type: Exception type if exiting due to exception.
            exc_val: Exception value if exiting due to exception.
            exc_tb: Exception traceback if exiting due to exception.

        Returns:
            True if exception was suppressed, False otherwise.
        """
        success = exc_type is None

        with self._lock:
            if not self._state.can_exit:
                return False

            self._state = ContextState.EXITING

        cleanup_errors: list[str] = []

        try:
            # Cleanup resources based on strategy
            if self._resource_tracker:
                should_cleanup = (
                    self._config.cleanup_strategy == CleanupStrategy.ALWAYS
                    or (self._config.cleanup_strategy == CleanupStrategy.ON_SUCCESS and success)
                    or (self._config.cleanup_strategy == CleanupStrategy.ON_FAILURE and not success)
                )
                if should_cleanup:
                    cleanup_errors = self._resource_tracker.cleanup_all()

            # Stop engine
            self._stop_engine()

            with self._lock:
                self._state = ContextState.EXITED
                self._exit_time = time.perf_counter()

            if self._hook:
                self._hook.on_exit(
                    self._context_id,
                    self._engine.engine_name,
                    success,
                    self.duration_ms or 0.0,
                    self._create_hook_context(),
                )

        except Exception as e:
            with self._lock:
                self._state = ContextState.FAILED
                self._exception = e

            if self._hook:
                self._hook.on_error(
                    self._context_id,
                    self._engine.engine_name,
                    e,
                    self._create_hook_context(),
                )

            if self._config.propagate_exceptions:
                raise

        if cleanup_errors and self._config.propagate_exceptions:
            raise ResourceCleanupError(
                f"Failed to cleanup {len(cleanup_errors)} resources",
                cleanup_errors,
                context_id=self._context_id,
            )

        return False

    def execute(
        self,
        operation: Callable[[E], R],
        *,
        name: str | None = None,
    ) -> R:
        """Execute an operation on the engine.

        Args:
            operation: Callable that receives the engine.
            name: Optional name for the operation.

        Returns:
            Result from the operation.

        Raises:
            ContextNotActiveError: If context is not active.
        """
        self._ensure_active()

        op_name = name or operation.__name__ if hasattr(operation, "__name__") else "operation"
        start_time = time.perf_counter()
        success = False

        try:
            result = operation(self._engine)
            success = True
            return result

        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._hook:
                self._hook.on_execute(
                    self._context_id,
                    self._engine.engine_name,
                    op_name,
                    duration_ms,
                    success,
                    self._create_hook_context(),
                )

    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        *,
        name: str | None = None,
        cleanup_func: Callable[[], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrackedResource | None:
        """Register a resource for tracking.

        Args:
            resource_id: Unique identifier for the resource.
            resource_type: Type of resource.
            name: Human-readable name.
            cleanup_func: Function to call for cleanup.
            metadata: Additional metadata.

        Returns:
            TrackedResource if tracking is enabled, None otherwise.
        """
        if not self._resource_tracker:
            return None

        return self._resource_tracker.register(
            resource_id,
            resource_type,
            name=name,
            cleanup_func=cleanup_func,
            metadata=metadata,
        )

    def unregister_resource(self, resource_id: str) -> TrackedResource | None:
        """Unregister a resource.

        Args:
            resource_id: Resource identifier.

        Returns:
            The unregistered resource, or None.
        """
        if not self._resource_tracker:
            return None

        return self._resource_tracker.unregister(resource_id)

    def list_resources(self, resource_type: str | None = None) -> list[TrackedResource]:
        """List tracked resources.

        Args:
            resource_type: Optional filter by type.

        Returns:
            List of tracked resources.
        """
        if not self._resource_tracker:
            return []

        return self._resource_tracker.list_resources(resource_type)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self.enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager."""
        return self.exit(exc_type, exc_val, exc_tb)


# =============================================================================
# Engine Session
# =============================================================================


class EngineSession(Generic[E]):
    """Transaction-like session management for engines.

    Provides commit/rollback semantics with savepoint support,
    similar to database transactions.

    Features:
        - Session start/commit/rollback
        - Savepoint creation and rollback
        - Operation history tracking
        - Automatic rollback on exception

    Example:
        >>> with EngineSession(engine) as session:
        ...     result1 = session.execute(lambda e: e.check(data1))
        ...     session.create_savepoint("sp1")
        ...     result2 = session.execute(lambda e: e.check(data2))
        ...     if result2.failed_count > 0:
        ...         session.rollback_to_savepoint("sp1")
        ...     session.commit()

        >>> # Automatic rollback on exception
        >>> with EngineSession(engine) as session:
        ...     session.execute(lambda e: e.check(data))
        ...     raise ValueError("error")  # Session rolled back
    """

    @dataclass
    class OperationRecord:
        """Record of an executed operation."""

        operation_id: str
        name: str
        timestamp: datetime
        duration_ms: float
        success: bool
        result: Any
        savepoint: str | None = None

    def __init__(
        self,
        engine: E,
        *,
        config: ContextConfig | None = None,
        hooks: Sequence[ContextHook] | None = None,
        name: str | None = None,
        auto_commit: bool = False,
    ) -> None:
        """Initialize engine session.

        Args:
            engine: Engine to manage.
            config: Context configuration.
            hooks: Context event hooks.
            name: Optional name for the session.
            auto_commit: Automatically commit on successful exit.
        """
        self._context = EngineContext(engine, config=config, hooks=hooks, name=name)
        self._auto_commit = auto_commit
        self._session_id = str(uuid.uuid4())[:8]
        self._state = SessionState.PENDING
        self._savepoint_manager = SavepointManager()
        self._operations: list[EngineSession.OperationRecord] = []
        self._rollback_handlers: list[Callable[[], None]] = []
        self._lock = threading.RLock()

    @property
    def session_id(self) -> str:
        """Get unique session identifier."""
        return self._session_id

    @property
    def context(self) -> EngineContext[E]:
        """Get underlying context."""
        return self._context

    @property
    def engine(self) -> E:
        """Get the managed engine."""
        return self._context.engine

    @property
    def state(self) -> SessionState:
        """Get current session state."""
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    @property
    def operations(self) -> list[OperationRecord]:
        """Get list of executed operations."""
        with self._lock:
            return list(self._operations)

    def begin(self) -> Self:
        """Begin the session.

        Returns:
            Self for chaining.
        """
        with self._lock:
            if self._state != SessionState.PENDING:
                raise ContextError(
                    f"Cannot begin session in state {self._state.name}",
                    context_id=self._session_id,
                )
            self._state = SessionState.ACTIVE

        self._context.enter()
        return self

    def commit(self) -> None:
        """Commit the session.

        Marks the session as committed. All operations are finalized.
        """
        with self._lock:
            if self._state != SessionState.ACTIVE:
                raise ContextError(
                    f"Cannot commit session in state {self._state.name}",
                    context_id=self._session_id,
                )
            self._state = SessionState.COMMITTING

        try:
            # Clear savepoints on commit
            self._savepoint_manager.clear()

            with self._lock:
                self._state = SessionState.COMMITTED

        except Exception:
            with self._lock:
                self._state = SessionState.FAILED
            raise

    def rollback(self) -> None:
        """Rollback the session.

        Executes all registered rollback handlers in reverse order.
        """
        with self._lock:
            if self._state not in (SessionState.ACTIVE, SessionState.FAILED):
                raise ContextError(
                    f"Cannot rollback session in state {self._state.name}",
                    context_id=self._session_id,
                )
            self._state = SessionState.ROLLING_BACK

        try:
            # Execute rollback handlers in reverse order
            for handler in reversed(self._rollback_handlers):
                with contextlib.suppress(Exception):
                    handler()

            # Clear savepoints
            self._savepoint_manager.clear()

            with self._lock:
                self._state = SessionState.ROLLED_BACK

        except Exception:
            with self._lock:
                self._state = SessionState.FAILED
            raise

    def create_savepoint(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Savepoint:
        """Create a savepoint.

        Args:
            name: Unique name for the savepoint.
            metadata: Additional metadata.

        Returns:
            Created Savepoint.
        """
        if not self.is_active:
            raise ContextError(
                "Cannot create savepoint in inactive session",
                context_id=self._session_id,
            )

        # Capture current state
        with self._lock:
            state_snapshot = {
                "operation_count": len(self._operations),
                "rollback_handler_count": len(self._rollback_handlers),
            }

        return self._savepoint_manager.create(
            name,
            state_snapshot,
            metadata=metadata,
        )

    def rollback_to_savepoint(self, name: str) -> dict[str, Any]:
        """Rollback to a savepoint.

        Args:
            name: Savepoint name.

        Returns:
            State snapshot from the savepoint.
        """
        if not self.is_active:
            raise ContextError(
                "Cannot rollback to savepoint in inactive session",
                context_id=self._session_id,
            )

        snapshot = self._savepoint_manager.rollback_to(name)

        # Rollback operations after savepoint
        with self._lock:
            op_count = snapshot.get("operation_count", 0)
            handler_count = snapshot.get("rollback_handler_count", 0)

            # Execute rollback handlers for operations after savepoint
            handlers_to_run = self._rollback_handlers[handler_count:]
            for handler in reversed(handlers_to_run):
                with contextlib.suppress(Exception):
                    handler()

            # Truncate operation and handler lists
            self._operations = self._operations[:op_count]
            self._rollback_handlers = self._rollback_handlers[:handler_count]

        return snapshot

    def release_savepoint(self, name: str) -> Savepoint | None:
        """Release (delete) a savepoint.

        Args:
            name: Savepoint name.

        Returns:
            Released savepoint, or None if not found.
        """
        return self._savepoint_manager.release(name)

    def list_savepoints(self) -> list[Savepoint]:
        """List all savepoints.

        Returns:
            List of savepoints.
        """
        return self._savepoint_manager.list_savepoints()

    def execute(
        self,
        operation: Callable[[E], R],
        *,
        name: str | None = None,
        rollback_handler: Callable[[], None] | None = None,
    ) -> R:
        """Execute an operation within the session.

        Args:
            operation: Callable that receives the engine.
            name: Optional name for the operation.
            rollback_handler: Optional handler for rollback.

        Returns:
            Result from the operation.
        """
        if not self.is_active:
            raise ContextError(
                "Cannot execute in inactive session",
                context_id=self._session_id,
            )

        op_name = name or (operation.__name__ if hasattr(operation, "__name__") else "operation")
        op_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        success = False
        result: Any = None

        # Get current savepoint name
        savepoints = self._savepoint_manager.list_savepoints()
        current_savepoint = savepoints[-1].name if savepoints else None

        try:
            result = self._context.execute(operation, name=op_name)
            success = True

            if rollback_handler:
                self._rollback_handlers.append(rollback_handler)

            return result

        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            with self._lock:
                self._operations.append(
                    EngineSession.OperationRecord(
                        operation_id=op_id,
                        name=op_name,
                        timestamp=datetime.now(UTC),
                        duration_ms=duration_ms,
                        success=success,
                        result=result,
                        savepoint=current_savepoint,
                    )
                )

    def register_rollback_handler(self, handler: Callable[[], None]) -> None:
        """Register a rollback handler.

        Args:
            handler: Function to call on rollback.
        """
        with self._lock:
            self._rollback_handlers.append(handler)

    def __enter__(self) -> Self:
        """Enter session context manager."""
        return self.begin()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit session context manager.

        Auto-commits on success if auto_commit is True.
        Auto-rollbacks on exception.
        """
        try:
            if exc_type is not None:
                # Rollback on exception
                if self._state == SessionState.ACTIVE:
                    self.rollback()
            elif self._auto_commit and self._state == SessionState.ACTIVE:
                # Auto-commit on success
                self.commit()

        finally:
            self._context.exit(exc_type, exc_val, exc_tb)

        return False


# =============================================================================
# Multi-Engine Context
# =============================================================================


class MultiEngineContext:
    """Context manager for coordinating multiple engines.

    Provides coordinated start/stop, health checks, and operation
    execution across multiple engines.

    Features:
        - Coordinated engine lifecycle
        - Parallel operation execution
        - Aggregated health checks
        - Failure handling strategies

    Example:
        >>> with MultiEngineContext(engine1, engine2, engine3) as ctx:
        ...     # Execute on all engines
        ...     results = ctx.execute_all(lambda e: e.check(data))
        ...
        ...     # Execute on specific engine
        ...     result = ctx.execute_on("truthound", lambda e: e.profile(data))
        ...
        ...     # Get aggregated health
        ...     health = ctx.health_check()
    """

    def __init__(
        self,
        *engines: Any,
        config: ContextConfig | None = None,
        hooks: Sequence[ContextHook] | None = None,
        fail_fast: bool = False,
        parallel: bool = False,
    ) -> None:
        """Initialize multi-engine context.

        Args:
            *engines: Engines to manage.
            config: Context configuration.
            hooks: Context event hooks.
            fail_fast: Stop on first engine failure.
            parallel: Execute operations in parallel.
        """
        self._engines: dict[str, Any] = {}
        for engine in engines:
            name = engine.engine_name
            if name in self._engines:
                raise ValueError(f"Duplicate engine name: {name}")
            self._engines[name] = engine

        self._config = config or DEFAULT_CONTEXT_CONFIG
        self._hook: ContextHook | None = None
        if hooks:
            self._hook = CompositeContextHook(list(hooks))

        self._fail_fast = fail_fast
        self._parallel = parallel
        self._context_id = str(uuid.uuid4())[:8]
        self._contexts: dict[str, EngineContext[Any]] = {}
        self._state = ContextState.CREATED
        self._lock = threading.RLock()

    @property
    def context_id(self) -> str:
        """Get unique context identifier."""
        return self._context_id

    @property
    def engine_names(self) -> list[str]:
        """Get list of managed engine names."""
        return list(self._engines.keys())

    @property
    def state(self) -> ContextState:
        """Get current context state."""
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return self.state == ContextState.ACTIVE

    def get_engine(self, name: str) -> Any:
        """Get engine by name.

        Args:
            name: Engine name.

        Returns:
            The engine.

        Raises:
            KeyError: If engine not found.
        """
        if name not in self._engines:
            raise KeyError(f"Engine '{name}' not found")
        return self._engines[name]

    def enter(self) -> Self:
        """Enter the multi-engine context.

        Returns:
            Self for chaining.
        """
        with self._lock:
            if not self._state.can_enter:
                raise ContextAlreadyActiveError(self._context_id)
            self._state = ContextState.ENTERING

        errors: list[tuple[str, Exception]] = []

        for name, engine in self._engines.items():
            try:
                ctx = EngineContext(engine, config=self._config)
                ctx.enter()
                self._contexts[name] = ctx
            except Exception as e:
                errors.append((name, e))
                if self._fail_fast:
                    # Cleanup already started contexts
                    for ctx in self._contexts.values():
                        with contextlib.suppress(Exception):
                            ctx.exit()
                    with self._lock:
                        self._state = ContextState.FAILED
                    raise ContextError(
                        f"Failed to enter context for engine '{name}': {e}",
                        context_id=self._context_id,
                        cause=e,
                    ) from e

        if errors and not self._fail_fast:
            # Continue with available engines
            pass

        with self._lock:
            self._state = ContextState.ACTIVE

        return self

    def exit(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: Any = None,
    ) -> bool:
        """Exit the multi-engine context.

        Args:
            exc_type: Exception type if exiting due to exception.
            exc_val: Exception value.
            exc_tb: Exception traceback.

        Returns:
            True if exception was suppressed.
        """
        with self._lock:
            if not self._state.can_exit:
                return False
            self._state = ContextState.EXITING

        errors: list[tuple[str, Exception]] = []

        for name, ctx in self._contexts.items():
            try:
                ctx.exit(exc_type, exc_val, exc_tb)
            except Exception as e:
                errors.append((name, e))

        with self._lock:
            self._state = ContextState.EXITED if not errors else ContextState.FAILED

        self._contexts.clear()

        if errors and self._config.propagate_exceptions:
            raise ContextError(
                f"Failed to exit contexts: {[e[0] for e in errors]}",
                context_id=self._context_id,
            )

        return False

    def execute_on(
        self,
        engine_name: str,
        operation: Callable[[Any], R],
        *,
        name: str | None = None,
    ) -> R:
        """Execute operation on a specific engine.

        Args:
            engine_name: Target engine name.
            operation: Callable that receives the engine.
            name: Optional operation name.

        Returns:
            Result from the operation.
        """
        if not self.is_active:
            raise ContextNotActiveError(self._context_id)

        if engine_name not in self._contexts:
            raise KeyError(f"Engine '{engine_name}' not found or not active")

        return self._contexts[engine_name].execute(operation, name=name)

    def execute_all(
        self,
        operation: Callable[[Any], R],
        *,
        name: str | None = None,
    ) -> dict[str, R | Exception]:
        """Execute operation on all engines.

        Args:
            operation: Callable that receives the engine.
            name: Optional operation name.

        Returns:
            Dictionary mapping engine names to results or exceptions.
        """
        if not self.is_active:
            raise ContextNotActiveError(self._context_id)

        results: dict[str, R | Exception] = {}

        if self._parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(ctx.execute, operation, name=name): engine_name
                    for engine_name, ctx in self._contexts.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    engine_name = futures[future]
                    try:
                        results[engine_name] = future.result()
                    except Exception as e:
                        results[engine_name] = e
                        if self._fail_fast:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
        else:
            for engine_name, ctx in self._contexts.items():
                try:
                    results[engine_name] = ctx.execute(operation, name=name)
                except Exception as e:
                    results[engine_name] = e
                    if self._fail_fast:
                        break

        return results

    def health_check(self) -> dict[str, HealthCheckResult]:
        """Perform health check on all engines.

        Returns:
            Dictionary mapping engine names to health check results.
        """
        from common.engines.lifecycle import ManagedEngine

        results: dict[str, HealthCheckResult] = {}

        for name, engine in self._engines.items():
            if isinstance(engine, ManagedEngine):
                try:
                    results[name] = engine.health_check()
                except Exception as e:
                    results[name] = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {e}",
                    )
            else:
                results[name] = HealthCheckResult.healthy(
                    name,
                    message=f"{engine.engine_name} v{engine.engine_version}",
                )

        return results

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self.enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager."""
        return self.exit(exc_type, exc_val, exc_tb)


# =============================================================================
# Context Stack (Nested Contexts)
# =============================================================================


class ContextStack:
    """Manages a stack of nested contexts.

    Allows entering and exiting multiple contexts in order,
    with automatic cleanup in reverse order.

    Example:
        >>> stack = ContextStack()
        >>> stack.push(EngineContext(engine1))
        >>> stack.push(EngineContext(engine2))
        >>> with stack:
        ...     # Both contexts are active
        ...     ctx = stack.current  # Returns engine2 context
        >>> # Both contexts automatically exited
    """

    def __init__(self, max_depth: int = 0) -> None:
        """Initialize context stack.

        Args:
            max_depth: Maximum stack depth (0 = unlimited).
        """
        self._stack: list[EngineContext[Any]] = []
        self._max_depth = max_depth
        self._lock = threading.RLock()

    @property
    def depth(self) -> int:
        """Get current stack depth."""
        with self._lock:
            return len(self._stack)

    @property
    def current(self) -> EngineContext[Any] | None:
        """Get current (top) context."""
        with self._lock:
            return self._stack[-1] if self._stack else None

    @property
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        with self._lock:
            return len(self._stack) == 0

    def push(self, context: EngineContext[Any]) -> None:
        """Push a context onto the stack.

        Args:
            context: Context to push.

        Raises:
            ContextError: If max depth exceeded.
        """
        with self._lock:
            if self._max_depth > 0 and len(self._stack) >= self._max_depth:
                raise ContextError(
                    f"Maximum context depth ({self._max_depth}) exceeded",
                )
            self._stack.append(context)

    def pop(self) -> EngineContext[Any] | None:
        """Pop the top context from the stack.

        Returns:
            The popped context, or None if empty.
        """
        with self._lock:
            return self._stack.pop() if self._stack else None

    def enter_all(self) -> Self:
        """Enter all contexts in the stack.

        Returns:
            Self for chaining.
        """
        for ctx in self._stack:
            ctx.enter()
        return self

    def exit_all(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: Any = None,
    ) -> None:
        """Exit all contexts in reverse order.

        Args:
            exc_type: Exception type if exiting due to exception.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        for ctx in reversed(self._stack):
            with contextlib.suppress(Exception):
                ctx.exit(exc_type, exc_val, exc_tb)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self.enter_all()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager."""
        self.exit_all(exc_type, exc_val, exc_tb)
        return False


# =============================================================================
# Async Context Implementations
# =============================================================================


class AsyncEngineContext(Generic[E]):
    """Async version of EngineContext.

    Provides async context management for engines that support async operations.

    Example:
        >>> async with AsyncEngineContext(engine) as ctx:
        ...     result = await ctx.execute(lambda e: e.check(data))
    """

    def __init__(
        self,
        engine: E,
        *,
        config: ContextConfig | None = None,
        hooks: Sequence[ContextHook] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize async engine context.

        Args:
            engine: Engine to manage.
            config: Context configuration.
            hooks: Context event hooks.
            name: Optional name for the context.
        """
        self._engine = engine
        self._config = config or DEFAULT_CONTEXT_CONFIG
        self._hook: ContextHook | None = None
        if hooks:
            self._hook = CompositeContextHook(list(hooks))

        self._context_id = str(uuid.uuid4())[:8]
        self._name = name or f"async-ctx-{self._context_id}"
        self._state = ContextState.CREATED
        self._enter_time: float | None = None
        self._exit_time: float | None = None
        self._resource_tracker = ResourceTracker() if self._config.track_resources else None
        self._lock = threading.RLock()

    @property
    def context_id(self) -> str:
        """Get unique context identifier."""
        return self._context_id

    @property
    def engine(self) -> E:
        """Get the managed engine."""
        self._ensure_active()
        return self._engine

    @property
    def state(self) -> ContextState:
        """Get current context state."""
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return self.state == ContextState.ACTIVE

    def _ensure_active(self) -> None:
        """Ensure context is active."""
        if not self.is_active:
            raise ContextNotActiveError(self._context_id)

    async def _start_engine(self) -> None:
        """Start the engine if configured."""
        from common.engines.lifecycle import AsyncManagedEngine, ManagedEngine

        if self._config.auto_start_engine:
            if isinstance(self._engine, AsyncManagedEngine):
                await self._engine.start()
            elif isinstance(self._engine, ManagedEngine):
                self._engine.start()

    async def _stop_engine(self) -> None:
        """Stop the engine if configured."""
        from common.engines.lifecycle import AsyncManagedEngine, ManagedEngine

        if self._config.auto_stop_engine:
            if isinstance(self._engine, AsyncManagedEngine):
                await self._engine.stop()
            elif isinstance(self._engine, ManagedEngine):
                self._engine.stop()

    async def enter(self) -> Self:
        """Enter the context asynchronously.

        Returns:
            Self for chaining.
        """
        with self._lock:
            if not self._state.can_enter:
                raise ContextAlreadyActiveError(self._context_id)
            self._state = ContextState.ENTERING
            self._enter_time = time.perf_counter()

        try:
            await self._start_engine()
            with self._lock:
                self._state = ContextState.ACTIVE
        except Exception:
            with self._lock:
                self._state = ContextState.FAILED
            raise

        return self

    async def exit(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: Any = None,
    ) -> bool:
        """Exit the context asynchronously.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.

        Returns:
            True if exception was suppressed.
        """
        success = exc_type is None

        with self._lock:
            if not self._state.can_exit:
                return False
            self._state = ContextState.EXITING

        try:
            if self._resource_tracker:
                should_cleanup = (
                    self._config.cleanup_strategy == CleanupStrategy.ALWAYS
                    or (self._config.cleanup_strategy == CleanupStrategy.ON_SUCCESS and success)
                    or (self._config.cleanup_strategy == CleanupStrategy.ON_FAILURE and not success)
                )
                if should_cleanup:
                    self._resource_tracker.cleanup_all()

            await self._stop_engine()

            with self._lock:
                self._state = ContextState.EXITED
                self._exit_time = time.perf_counter()

        except Exception:
            with self._lock:
                self._state = ContextState.FAILED
            if self._config.propagate_exceptions:
                raise

        return False

    async def execute(
        self,
        operation: Callable[[E], R],
        *,
        name: str | None = None,
    ) -> R:
        """Execute an operation on the engine.

        For async operations, the operation should return an awaitable.

        Args:
            operation: Callable that receives the engine.
            name: Optional operation name.

        Returns:
            Result from the operation.
        """
        self._ensure_active()

        import inspect

        result = operation(self._engine)
        if inspect.isawaitable(result):
            return await result
        return result

    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        *,
        name: str | None = None,
        cleanup_func: Callable[[], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrackedResource | None:
        """Register a resource for tracking."""
        if not self._resource_tracker:
            return None
        return self._resource_tracker.register(
            resource_id, resource_type, name=name, cleanup_func=cleanup_func, metadata=metadata
        )

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return await self.enter()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit async context manager."""
        return await self.exit(exc_type, exc_val, exc_tb)


# =============================================================================
# Utility Functions
# =============================================================================


def create_engine_context(
    engine: Any,
    *,
    config: ContextConfig | None = None,
    hooks: Sequence[ContextHook] | None = None,
    name: str | None = None,
) -> EngineContext[Any]:
    """Create an engine context.

    Args:
        engine: Engine to manage.
        config: Context configuration.
        hooks: Context event hooks.
        name: Optional name.

    Returns:
        EngineContext instance.
    """
    return EngineContext(engine, config=config, hooks=hooks, name=name)


def create_engine_session(
    engine: Any,
    *,
    config: ContextConfig | None = None,
    hooks: Sequence[ContextHook] | None = None,
    name: str | None = None,
    auto_commit: bool = False,
) -> EngineSession[Any]:
    """Create an engine session.

    Args:
        engine: Engine to manage.
        config: Context configuration.
        hooks: Context event hooks.
        name: Optional name.
        auto_commit: Auto-commit on success.

    Returns:
        EngineSession instance.
    """
    return EngineSession(
        engine, config=config, hooks=hooks, name=name, auto_commit=auto_commit
    )


def create_multi_engine_context(
    *engines: Any,
    config: ContextConfig | None = None,
    hooks: Sequence[ContextHook] | None = None,
    fail_fast: bool = False,
    parallel: bool = False,
) -> MultiEngineContext:
    """Create a multi-engine context.

    Args:
        *engines: Engines to manage.
        config: Context configuration.
        hooks: Context event hooks.
        fail_fast: Stop on first failure.
        parallel: Execute in parallel.

    Returns:
        MultiEngineContext instance.
    """
    return MultiEngineContext(
        *engines, config=config, hooks=hooks, fail_fast=fail_fast, parallel=parallel
    )


@contextlib.contextmanager
def engine_context(
    engine: Any,
    *,
    config: ContextConfig | None = None,
) -> Generator[EngineContext[Any], None, None]:
    """Convenience context manager function.

    Args:
        engine: Engine to manage.
        config: Context configuration.

    Yields:
        EngineContext instance.

    Example:
        >>> with engine_context(engine) as ctx:
        ...     result = ctx.execute(lambda e: e.check(data))
    """
    ctx = EngineContext(engine, config=config)
    with ctx:
        yield ctx


@contextlib.asynccontextmanager
async def async_engine_context(
    engine: Any,
    *,
    config: ContextConfig | None = None,
) -> Generator[AsyncEngineContext[Any], None, None]:
    """Convenience async context manager function.

    Args:
        engine: Engine to manage.
        config: Context configuration.

    Yields:
        AsyncEngineContext instance.

    Example:
        >>> async with async_engine_context(engine) as ctx:
        ...     result = await ctx.execute(lambda e: e.check(data))
    """
    ctx = AsyncEngineContext(engine, config=config)
    async with ctx:
        yield ctx
