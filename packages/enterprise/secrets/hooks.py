"""Hook system for secret management operations.

This module provides hooks that are called during secret operations,
enabling audit logging, metrics collection, and custom behaviors.

IMPORTANT: Hooks must NEVER log actual secret values. Only paths, operations,
and metadata should be logged.

Example:
    >>> from packages.enterprise.secrets import (
    ...     AuditLoggingHook,
    ...     MetricsSecretHook,
    ...     CompositeSecretHook,
    ... )
    >>>
    >>> hooks = CompositeSecretHook([
    ...     AuditLoggingHook(),
    ...     MetricsSecretHook(),
    ... ])
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base import SecretValue


logger = logging.getLogger(__name__)


class SecretOperation(Enum):
    """Types of secret operations.

    Attributes:
        GET: Retrieve a secret.
        SET: Store a secret.
        DELETE: Delete a secret.
        LIST: List secrets.
        EXISTS: Check if a secret exists.
        ROTATE: Rotate a secret.
    """

    GET = auto()
    SET = auto()
    DELETE = auto()
    LIST = auto()
    EXISTS = auto()
    ROTATE = auto()


@dataclass(frozen=True, slots=True)
class SecretOperationContext:
    """Context for a secret operation.

    Provides information about the operation being performed.
    Never contains actual secret values.

    Attributes:
        operation: The type of operation.
        path: The secret path (may be empty for list operations).
        timestamp: When the operation started.
        provider_name: Name of the provider.
        tenant_id: Optional tenant identifier.
        user_id: Optional user identifier.
        request_id: Optional request correlation ID.
        metadata: Additional context metadata.
    """

    operation: SecretOperation
    path: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provider_name: str = ""
    tenant_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SecretHook(Protocol):
    """Protocol for secret operation hooks.

    Hooks are called before and after secret operations, enabling
    audit logging, metrics, and custom behaviors.

    IMPORTANT: Implementations must NEVER access or log actual secret values.
    """

    def on_before_get(self, context: SecretOperationContext) -> None:
        """Called before retrieving a secret.

        Args:
            context: Operation context.
        """
        ...

    def on_after_get(
        self,
        context: SecretOperationContext,
        result: SecretValue | None,
        duration_ms: float,
    ) -> None:
        """Called after retrieving a secret.

        Args:
            context: Operation context.
            result: The retrieved secret (DO NOT log the value).
            duration_ms: Operation duration in milliseconds.
        """
        ...

    def on_before_set(self, context: SecretOperationContext) -> None:
        """Called before storing a secret.

        Args:
            context: Operation context.
        """
        ...

    def on_after_set(
        self,
        context: SecretOperationContext,
        result: SecretValue,
        duration_ms: float,
    ) -> None:
        """Called after storing a secret.

        Args:
            context: Operation context.
            result: The stored secret (DO NOT log the value).
            duration_ms: Operation duration in milliseconds.
        """
        ...

    def on_before_delete(self, context: SecretOperationContext) -> None:
        """Called before deleting a secret.

        Args:
            context: Operation context.
        """
        ...

    def on_after_delete(
        self,
        context: SecretOperationContext,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Called after deleting a secret.

        Args:
            context: Operation context.
            success: Whether the deletion was successful.
            duration_ms: Operation duration in milliseconds.
        """
        ...

    def on_error(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Called when an operation fails.

        Args:
            context: Operation context.
            error: The exception that occurred.
            duration_ms: Operation duration in milliseconds.
        """
        ...


class BaseSecretHook(ABC):
    """Base class for secret hooks with no-op defaults.

    Provides default no-op implementations for all hook methods.
    Subclasses can override only the methods they need.
    """

    def on_before_get(self, context: SecretOperationContext) -> None:
        """Called before retrieving a secret."""
        pass

    def on_after_get(
        self,
        context: SecretOperationContext,
        result: SecretValue | None,
        duration_ms: float,
    ) -> None:
        """Called after retrieving a secret."""
        pass

    def on_before_set(self, context: SecretOperationContext) -> None:
        """Called before storing a secret."""
        pass

    def on_after_set(
        self,
        context: SecretOperationContext,
        result: SecretValue,
        duration_ms: float,
    ) -> None:
        """Called after storing a secret."""
        pass

    def on_before_delete(self, context: SecretOperationContext) -> None:
        """Called before deleting a secret."""
        pass

    def on_after_delete(
        self,
        context: SecretOperationContext,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Called after deleting a secret."""
        pass

    def on_error(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Called when an operation fails."""
        pass


class AuditLoggingHook(BaseSecretHook):
    """Hook for audit logging of secret operations.

    Logs all secret operations for compliance and security auditing.
    NEVER logs actual secret values.

    Attributes:
        logger: The logger to use.
        log_level: Log level for audit entries.

    Example:
        >>> hook = AuditLoggingHook(log_level=logging.INFO)
        >>> # Use with a provider
    """

    def __init__(
        self,
        logger_name: str = "secrets.audit",
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize the audit logging hook.

        Args:
            logger_name: Name for the audit logger.
            log_level: Log level for audit entries.
        """
        self._logger = logging.getLogger(logger_name)
        self._log_level = log_level

    def _log(self, message: str, context: SecretOperationContext, **kwargs: Any) -> None:
        """Log an audit entry.

        Args:
            message: The log message.
            context: Operation context.
            **kwargs: Additional log fields.
        """
        extra = {
            "operation": context.operation.name,
            "path": context.path,
            "provider": context.provider_name,
            "timestamp": context.timestamp.isoformat(),
        }
        if context.tenant_id:
            extra["tenant_id"] = context.tenant_id
        if context.user_id:
            extra["user_id"] = context.user_id
        if context.request_id:
            extra["request_id"] = context.request_id
        extra.update(kwargs)
        self._logger.log(self._log_level, message, extra=extra)

    def on_before_get(self, context: SecretOperationContext) -> None:
        """Log before retrieving a secret."""
        self._log("SECRET_GET_START", context)

    def on_after_get(
        self,
        context: SecretOperationContext,
        result: SecretValue | None,
        duration_ms: float,
    ) -> None:
        """Log after retrieving a secret."""
        self._log(
            "SECRET_GET_SUCCESS" if result else "SECRET_GET_NOT_FOUND",
            context,
            duration_ms=duration_ms,
            found=result is not None,
            version=result.version if result else None,
        )

    def on_before_set(self, context: SecretOperationContext) -> None:
        """Log before storing a secret."""
        self._log("SECRET_SET_START", context)

    def on_after_set(
        self,
        context: SecretOperationContext,
        result: SecretValue,
        duration_ms: float,
    ) -> None:
        """Log after storing a secret."""
        self._log(
            "SECRET_SET_SUCCESS",
            context,
            duration_ms=duration_ms,
            version=result.version,
        )

    def on_before_delete(self, context: SecretOperationContext) -> None:
        """Log before deleting a secret."""
        self._log("SECRET_DELETE_START", context)

    def on_after_delete(
        self,
        context: SecretOperationContext,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Log after deleting a secret."""
        self._log(
            "SECRET_DELETE_SUCCESS" if success else "SECRET_DELETE_NOT_FOUND",
            context,
            duration_ms=duration_ms,
            success=success,
        )

    def on_error(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Log operation errors."""
        self._logger.error(
            "SECRET_OPERATION_ERROR",
            extra={
                "operation": context.operation.name,
                "path": context.path,
                "provider": context.provider_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "duration_ms": duration_ms,
            },
        )


@dataclass
class SecretMetrics:
    """Metrics collected by MetricsSecretHook.

    Thread-safe metrics container.

    Attributes:
        get_count: Number of get operations.
        get_hit_count: Number of successful gets.
        get_miss_count: Number of get misses.
        set_count: Number of set operations.
        delete_count: Number of delete operations.
        error_count: Number of errors.
        total_duration_ms: Total duration of all operations.
    """

    get_count: int = 0
    get_hit_count: int = 0
    get_miss_count: int = 0
    set_count: int = 0
    delete_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate.

        Returns:
            Hit rate as a ratio (0.0 to 1.0).
        """
        if self.get_count == 0:
            return 0.0
        return self.get_hit_count / self.get_count

    @property
    def average_duration_ms(self) -> float:
        """Calculate the average operation duration.

        Returns:
            Average duration in milliseconds.
        """
        total = self.get_count + self.set_count + self.delete_count
        if total == 0:
            return 0.0
        return self.total_duration_ms / total

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.get_count = 0
        self.get_hit_count = 0
        self.get_miss_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.error_count = 0
        self.total_duration_ms = 0.0


class MetricsSecretHook(BaseSecretHook):
    """Hook for collecting metrics on secret operations.

    Tracks counts, hit rates, and latencies for secret operations.

    Example:
        >>> hook = MetricsSecretHook()
        >>> # Use with a provider...
        >>> print(f"Hit rate: {hook.metrics.hit_rate:.2%}")
        >>> print(f"Avg latency: {hook.metrics.average_duration_ms:.2f}ms")
    """

    def __init__(self) -> None:
        """Initialize the metrics hook."""
        self._metrics = SecretMetrics()
        self._lock = threading.Lock()

    @property
    def metrics(self) -> SecretMetrics:
        """Get current metrics (thread-safe copy)."""
        with self._lock:
            return SecretMetrics(
                get_count=self._metrics.get_count,
                get_hit_count=self._metrics.get_hit_count,
                get_miss_count=self._metrics.get_miss_count,
                set_count=self._metrics.set_count,
                delete_count=self._metrics.delete_count,
                error_count=self._metrics.error_count,
                total_duration_ms=self._metrics.total_duration_ms,
            )

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.reset()

    def on_after_get(
        self,
        context: SecretOperationContext,
        result: SecretValue | None,
        duration_ms: float,
    ) -> None:
        """Record get operation metrics."""
        with self._lock:
            self._metrics.get_count += 1
            self._metrics.total_duration_ms += duration_ms
            if result is not None:
                self._metrics.get_hit_count += 1
            else:
                self._metrics.get_miss_count += 1

    def on_after_set(
        self,
        context: SecretOperationContext,
        result: SecretValue,
        duration_ms: float,
    ) -> None:
        """Record set operation metrics."""
        with self._lock:
            self._metrics.set_count += 1
            self._metrics.total_duration_ms += duration_ms

    def on_after_delete(
        self,
        context: SecretOperationContext,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record delete operation metrics."""
        with self._lock:
            self._metrics.delete_count += 1
            self._metrics.total_duration_ms += duration_ms

    def on_error(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Record error metrics."""
        with self._lock:
            self._metrics.error_count += 1
            self._metrics.total_duration_ms += duration_ms


class CompositeSecretHook(BaseSecretHook):
    """Hook that combines multiple hooks.

    Calls all registered hooks in order, with error isolation.
    If one hook fails, the others still execute.

    Example:
        >>> composite = CompositeSecretHook([
        ...     AuditLoggingHook(),
        ...     MetricsSecretHook(),
        ... ])
    """

    def __init__(self, hooks: Sequence[SecretHook]) -> None:
        """Initialize with a sequence of hooks.

        Args:
            hooks: Hooks to combine.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: SecretHook) -> None:
        """Add a hook to the composite.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: SecretHook) -> bool:
        """Remove a hook from the composite.

        Args:
            hook: Hook to remove.

        Returns:
            True if the hook was removed.
        """
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False

    def _call_hooks(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Call a method on all hooks with error isolation.

        Args:
            method_name: Name of the method to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        for hook in self._hooks:
            try:
                method = getattr(hook, method_name, None)
                if method:
                    method(*args, **kwargs)
            except Exception as e:
                # Log but don't propagate hook errors
                logger.exception(f"Hook {type(hook).__name__}.{method_name} failed: {e}")

    def on_before_get(self, context: SecretOperationContext) -> None:
        """Call on_before_get on all hooks."""
        self._call_hooks("on_before_get", context)

    def on_after_get(
        self,
        context: SecretOperationContext,
        result: SecretValue | None,
        duration_ms: float,
    ) -> None:
        """Call on_after_get on all hooks."""
        self._call_hooks("on_after_get", context, result, duration_ms)

    def on_before_set(self, context: SecretOperationContext) -> None:
        """Call on_before_set on all hooks."""
        self._call_hooks("on_before_set", context)

    def on_after_set(
        self,
        context: SecretOperationContext,
        result: SecretValue,
        duration_ms: float,
    ) -> None:
        """Call on_after_set on all hooks."""
        self._call_hooks("on_after_set", context, result, duration_ms)

    def on_before_delete(self, context: SecretOperationContext) -> None:
        """Call on_before_delete on all hooks."""
        self._call_hooks("on_before_delete", context)

    def on_after_delete(
        self,
        context: SecretOperationContext,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Call on_after_delete on all hooks."""
        self._call_hooks("on_after_delete", context, success, duration_ms)

    def on_error(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Call on_error on all hooks."""
        self._call_hooks("on_error", context, error, duration_ms)


class TenantAwareSecretHook(BaseSecretHook):
    """Hook that injects tenant context from the current thread.

    Automatically enriches operation context with the current tenant ID
    from the multi-tenant context system.

    Example:
        >>> hook = TenantAwareSecretHook(wrapped_hook=AuditLoggingHook())
    """

    def __init__(self, wrapped_hook: SecretHook) -> None:
        """Initialize with a wrapped hook.

        Args:
            wrapped_hook: The hook to wrap.
        """
        self._wrapped = wrapped_hook

    def _enrich_context(self, context: SecretOperationContext) -> SecretOperationContext:
        """Enrich context with tenant information.

        Args:
            context: Original context.

        Returns:
            Enriched context with tenant ID if available.
        """
        if context.tenant_id:
            return context

        try:
            from packages.enterprise.multi_tenant.context import get_current_tenant_id

            tenant_id = get_current_tenant_id()
            if tenant_id:
                return SecretOperationContext(
                    operation=context.operation,
                    path=context.path,
                    timestamp=context.timestamp,
                    provider_name=context.provider_name,
                    tenant_id=tenant_id,
                    user_id=context.user_id,
                    request_id=context.request_id,
                    metadata=context.metadata,
                )
        except ImportError:
            pass
        return context

    def on_before_get(self, context: SecretOperationContext) -> None:
        """Call wrapped hook with enriched context."""
        self._wrapped.on_before_get(self._enrich_context(context))

    def on_after_get(
        self,
        context: SecretOperationContext,
        result: SecretValue | None,
        duration_ms: float,
    ) -> None:
        """Call wrapped hook with enriched context."""
        self._wrapped.on_after_get(self._enrich_context(context), result, duration_ms)

    def on_before_set(self, context: SecretOperationContext) -> None:
        """Call wrapped hook with enriched context."""
        self._wrapped.on_before_set(self._enrich_context(context))

    def on_after_set(
        self,
        context: SecretOperationContext,
        result: SecretValue,
        duration_ms: float,
    ) -> None:
        """Call wrapped hook with enriched context."""
        self._wrapped.on_after_set(self._enrich_context(context), result, duration_ms)

    def on_before_delete(self, context: SecretOperationContext) -> None:
        """Call wrapped hook with enriched context."""
        self._wrapped.on_before_delete(self._enrich_context(context))

    def on_after_delete(
        self,
        context: SecretOperationContext,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Call wrapped hook with enriched context."""
        self._wrapped.on_after_delete(self._enrich_context(context), success, duration_ms)

    def on_error(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Call wrapped hook with enriched context."""
        self._wrapped.on_error(self._enrich_context(context), error, duration_ms)
