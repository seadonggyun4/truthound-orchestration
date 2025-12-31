"""Tenant lifecycle hooks.

This module provides hooks for observing and reacting to tenant
lifecycle events. Hooks enable loose coupling between the tenant
system and other parts of the application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import TenantConfig
    from .types import TenantStatus


logger = logging.getLogger(__name__)


# =============================================================================
# Hook Protocols
# =============================================================================


@runtime_checkable
class TenantHook(Protocol):
    """Protocol for tenant lifecycle hooks.

    Implementations can observe and react to tenant events.
    All methods are optional - implement only what you need.
    """

    def on_tenant_created(self, tenant: TenantConfig) -> None:
        """Called when a new tenant is created.

        Args:
            tenant: The newly created tenant configuration.
        """
        ...

    def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """Called when a tenant is updated.

        Args:
            tenant: The updated tenant configuration.
            previous: The previous configuration (if available).
        """
        ...

    def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Called when a tenant is deleted.

        Args:
            tenant: The deleted tenant configuration.
            hard_delete: True if hard deleted, False if soft deleted.
        """
        ...

    def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """Called when a tenant's status changes.

        Args:
            tenant: The tenant with new status.
            previous_status: The previous status.
        """
        ...

    def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """Called when a tenant is suspended.

        Args:
            tenant: The suspended tenant.
            reason: The reason for suspension.
        """
        ...


@runtime_checkable
class AsyncTenantHook(Protocol):
    """Async protocol for tenant lifecycle hooks."""

    async def on_tenant_created(self, tenant: TenantConfig) -> None:
        """Called when a new tenant is created."""
        ...

    async def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """Called when a tenant is updated."""
        ...

    async def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Called when a tenant is deleted."""
        ...

    async def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """Called when a tenant's status changes."""
        ...

    async def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """Called when a tenant is suspended."""
        ...


# =============================================================================
# Base Hook Implementation
# =============================================================================


class BaseTenantHook:
    """Base implementation of TenantHook with no-op methods.

    Subclass and override only the methods you need.
    """

    def on_tenant_created(self, tenant: TenantConfig) -> None:
        """No-op implementation."""
        pass

    def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """No-op implementation."""
        pass

    def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """No-op implementation."""
        pass

    def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """No-op implementation."""
        pass

    def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """No-op implementation."""
        pass


# =============================================================================
# Logging Hook
# =============================================================================


class LoggingTenantHook(BaseTenantHook):
    """Hook that logs tenant lifecycle events.

    Logs all events at the configured log level.
    """

    def __init__(
        self,
        *,
        log_level: int = logging.INFO,
        include_details: bool = True,
    ) -> None:
        self.log_level = log_level
        self.include_details = include_details

    def on_tenant_created(self, tenant: TenantConfig) -> None:
        """Log tenant creation."""
        extra = {"tenant_id": tenant.tenant_id} if self.include_details else {}
        logger.log(
            self.log_level,
            f"Tenant created: {tenant.tenant_id} (tier={tenant.tier.value})",
            extra=extra,
        )

    def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """Log tenant update."""
        extra = {"tenant_id": tenant.tenant_id} if self.include_details else {}
        logger.log(
            self.log_level,
            f"Tenant updated: {tenant.tenant_id}",
            extra=extra,
        )

    def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Log tenant deletion."""
        delete_type = "hard deleted" if hard_delete else "soft deleted"
        extra = {"tenant_id": tenant.tenant_id} if self.include_details else {}
        logger.log(
            self.log_level,
            f"Tenant {delete_type}: {tenant.tenant_id}",
            extra=extra,
        )

    def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """Log status change."""
        prev = previous_status.value if previous_status else "unknown"
        extra = {"tenant_id": tenant.tenant_id} if self.include_details else {}
        logger.log(
            self.log_level,
            f"Tenant status changed: {tenant.tenant_id} ({prev} -> {tenant.status.value})",
            extra=extra,
        )

    def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """Log tenant suspension."""
        msg = f"Tenant suspended: {tenant.tenant_id}"
        if reason:
            msg += f" (reason: {reason})"
        extra = {"tenant_id": tenant.tenant_id} if self.include_details else {}
        logger.log(logging.WARNING, msg, extra=extra)


# =============================================================================
# Metrics Hook
# =============================================================================


@dataclass
class TenantMetrics:
    """Container for tenant metrics."""

    tenants_created: int = 0
    tenants_deleted: int = 0
    tenants_updated: int = 0
    tenants_suspended: int = 0
    status_changes: int = 0
    last_event_time: datetime | None = None


class MetricsTenantHook(BaseTenantHook):
    """Hook that collects tenant metrics.

    Tracks counts of tenant operations for monitoring.
    """

    def __init__(self) -> None:
        self._metrics = TenantMetrics()

    @property
    def metrics(self) -> TenantMetrics:
        """Get the collected metrics."""
        return self._metrics

    def on_tenant_created(self, tenant: TenantConfig) -> None:
        """Increment created counter."""
        self._metrics.tenants_created += 1
        self._metrics.last_event_time = datetime.now()

    def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """Increment updated counter."""
        self._metrics.tenants_updated += 1
        self._metrics.last_event_time = datetime.now()

    def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Increment deleted counter."""
        self._metrics.tenants_deleted += 1
        self._metrics.last_event_time = datetime.now()

    def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """Increment status changes counter."""
        self._metrics.status_changes += 1
        self._metrics.last_event_time = datetime.now()

    def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """Increment suspended counter."""
        self._metrics.tenants_suspended += 1
        self._metrics.last_event_time = datetime.now()

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = TenantMetrics()


# =============================================================================
# Audit Hook
# =============================================================================


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Represents an audit event for a tenant operation."""

    event_type: str
    tenant_id: str
    timestamp: datetime
    details: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "details": dict(self.details),
        }


class AuditTenantHook(BaseTenantHook):
    """Hook that creates audit trails for tenant operations.

    Maintains a list of audit events that can be persisted
    or forwarded to an audit logging system.
    """

    def __init__(self, *, max_events: int = 10000) -> None:
        self.max_events = max_events
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> list[AuditEvent]:
        """Get all audit events."""
        return list(self._events)

    def _add_event(
        self,
        event_type: str,
        tenant_id: str,
        **details: Any,
    ) -> None:
        """Add an audit event."""
        event = AuditEvent(
            event_type=event_type,
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            details=tuple(details.items()),
        )
        self._events.append(event)

        # Trim if over limit
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def on_tenant_created(self, tenant: TenantConfig) -> None:
        """Record tenant creation."""
        self._add_event(
            "tenant_created",
            tenant.tenant_id,
            name=tenant.name,
            tier=tenant.tier.value,
            status=tenant.status.value,
        )

    def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """Record tenant update."""
        changes = {}
        if previous:
            if previous.name != tenant.name:
                changes["name"] = (previous.name, tenant.name)
            if previous.tier != tenant.tier:
                changes["tier"] = (previous.tier.value, tenant.tier.value)
        self._add_event("tenant_updated", tenant.tenant_id, changes=changes)

    def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Record tenant deletion."""
        self._add_event(
            "tenant_deleted",
            tenant.tenant_id,
            hard_delete=hard_delete,
        )

    def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """Record status change."""
        self._add_event(
            "status_changed",
            tenant.tenant_id,
            previous=previous_status.value if previous_status else None,
            new=tenant.status.value,
        )

    def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """Record tenant suspension."""
        self._add_event(
            "tenant_suspended",
            tenant.tenant_id,
            reason=reason,
        )

    def clear(self) -> None:
        """Clear all audit events."""
        self._events.clear()


# =============================================================================
# Composite Hook
# =============================================================================


class CompositeTenantHook(BaseTenantHook):
    """Hook that delegates to multiple hooks.

    Runs all hooks for each event, ignoring individual failures.
    """

    def __init__(self, hooks: list[TenantHook] | None = None) -> None:
        self.hooks = hooks or []

    def add_hook(self, hook: TenantHook) -> None:
        """Add a hook."""
        self.hooks.append(hook)

    def remove_hook(self, hook: TenantHook) -> None:
        """Remove a hook."""
        self.hooks.remove(hook)

    def _call_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call a method on all hooks."""
        for hook in self.hooks:
            handler = getattr(hook, method, None)
            if handler:
                try:
                    handler(*args, **kwargs)
                except Exception:
                    # Log but don't propagate
                    logger.exception(f"Hook {hook} failed on {method}")

    def on_tenant_created(self, tenant: TenantConfig) -> None:
        """Delegate to all hooks."""
        self._call_hooks("on_tenant_created", tenant)

    def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """Delegate to all hooks."""
        self._call_hooks("on_tenant_updated", tenant, previous=previous)

    def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Delegate to all hooks."""
        self._call_hooks("on_tenant_deleted", tenant, hard_delete=hard_delete)

    def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """Delegate to all hooks."""
        self._call_hooks("on_status_changed", tenant, previous_status=previous_status)

    def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """Delegate to all hooks."""
        self._call_hooks("on_tenant_suspended", tenant, reason=reason)


# =============================================================================
# Callback Hook
# =============================================================================


class CallbackTenantHook(BaseTenantHook):
    """Hook that calls custom callbacks.

    Allows registering callbacks without subclassing.
    """

    def __init__(self) -> None:
        self._callbacks: dict[str, list[Any]] = {}

    def on(self, event: str, callback: Any) -> None:
        """Register a callback for an event.

        Args:
            event: Event name (e.g., "tenant_created").
            callback: Callable to invoke.
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def off(self, event: str, callback: Any) -> None:
        """Unregister a callback.

        Args:
            event: Event name.
            callback: Callable to remove.
        """
        if event in self._callbacks:
            self._callbacks[event].remove(callback)

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.exception(f"Callback failed for event {event}")

    def on_tenant_created(self, tenant: TenantConfig) -> None:
        """Emit tenant_created event."""
        self._emit("tenant_created", tenant)

    def on_tenant_updated(
        self,
        tenant: TenantConfig,
        *,
        previous: TenantConfig | None = None,
    ) -> None:
        """Emit tenant_updated event."""
        self._emit("tenant_updated", tenant, previous=previous)

    def on_tenant_deleted(
        self,
        tenant: TenantConfig,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Emit tenant_deleted event."""
        self._emit("tenant_deleted", tenant, hard_delete=hard_delete)

    def on_status_changed(
        self,
        tenant: TenantConfig,
        *,
        previous_status: TenantStatus | None = None,
    ) -> None:
        """Emit status_changed event."""
        self._emit("status_changed", tenant, previous_status=previous_status)

    def on_tenant_suspended(
        self,
        tenant: TenantConfig,
        *,
        reason: str | None = None,
    ) -> None:
        """Emit tenant_suspended event."""
        self._emit("tenant_suspended", tenant, reason=reason)
