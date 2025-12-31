"""Isolation validation implementations.

This module provides validators and violation handlers for
enforcing and monitoring tenant isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..exceptions import CrossTenantAccessError, TenantIsolationError
from ..types import Permission, ResourceType

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class IsolationViolation:
    """Represents an isolation violation.

    Captures details about a violation for logging, alerting,
    and audit purposes.
    """

    violation_type: str
    source_tenant_id: str
    target_tenant_id: str | None
    resource_type: ResourceType
    resource_id: str | None
    permission: Permission | None
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "violation_type": self.violation_type,
            "source_tenant_id": self.source_tenant_id,
            "target_tenant_id": self.target_tenant_id,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "permission": self.permission.value if self.permission else None,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": dict(self.context),
        }


@dataclass(frozen=True, slots=True)
class IsolationResult:
    """Result of an isolation check.

    Provides detailed information about whether isolation
    was maintained and any violations that occurred.
    """

    is_allowed: bool
    violation: IsolationViolation | None = None
    message: str | None = None

    @classmethod
    def allowed(cls, message: str | None = None) -> IsolationResult:
        """Create an allowed result."""
        return cls(is_allowed=True, message=message)

    @classmethod
    def denied(
        cls,
        violation: IsolationViolation,
        message: str | None = None,
    ) -> IsolationResult:
        """Create a denied result with violation."""
        return cls(is_allowed=False, violation=violation, message=message)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_allowed


# =============================================================================
# Violation Handlers
# =============================================================================


@runtime_checkable
class IsolationViolationHandler(Protocol):
    """Protocol for handling isolation violations.

    Implementations can log, alert, or take other actions
    when violations occur.
    """

    def handle(self, violation: IsolationViolation) -> None:
        """Handle an isolation violation.

        Args:
            violation: The violation details.
        """
        ...


class LoggingViolationHandler:
    """Handler that logs isolation violations.

    Logs violations at WARNING level by default.
    """

    def __init__(
        self,
        *,
        log_level: int = logging.WARNING,
        include_context: bool = True,
    ) -> None:
        self.log_level = log_level
        self.include_context = include_context

    def handle(self, violation: IsolationViolation) -> None:
        """Log the violation."""
        msg = (
            f"Isolation violation: {violation.violation_type} - "
            f"Tenant '{violation.source_tenant_id}' attempted to access "
            f"{violation.resource_type.value}"
        )
        if violation.target_tenant_id:
            msg += f" owned by tenant '{violation.target_tenant_id}'"
        if violation.resource_id:
            msg += f" (resource: {violation.resource_id})"

        extra: dict[str, Any] = {
            "violation_type": violation.violation_type,
            "source_tenant_id": violation.source_tenant_id,
            "target_tenant_id": violation.target_tenant_id,
            "resource_type": violation.resource_type.value,
        }
        if self.include_context:
            extra["context"] = dict(violation.context)

        logger.log(self.log_level, msg, extra=extra)


class RaisingViolationHandler:
    """Handler that raises exceptions for isolation violations.

    Converts violations to appropriate exceptions.
    """

    def __init__(
        self,
        *,
        exception_factory: Callable[[IsolationViolation], Exception] | None = None,
    ) -> None:
        self.exception_factory = exception_factory or self._default_factory

    def _default_factory(self, violation: IsolationViolation) -> Exception:
        """Create a default exception from a violation."""
        if violation.target_tenant_id and violation.source_tenant_id != violation.target_tenant_id:
            return CrossTenantAccessError(
                source_tenant_id=violation.source_tenant_id,
                target_tenant_id=violation.target_tenant_id,
                resource_type=violation.resource_type.value,
                resource_id=violation.resource_id,
                message=violation.message,
            )
        return TenantIsolationError(
            violation.message,
            source_tenant_id=violation.source_tenant_id,
            target_tenant_id=violation.target_tenant_id,
            resource_type=violation.resource_type.value,
            resource_id=violation.resource_id,
        )

    def handle(self, violation: IsolationViolation) -> None:
        """Raise an exception for the violation."""
        raise self.exception_factory(violation)


class CompositeViolationHandler:
    """Handler that delegates to multiple handlers.

    Runs all handlers for each violation.
    """

    def __init__(self, handlers: list[IsolationViolationHandler]) -> None:
        self.handlers = handlers

    def add_handler(self, handler: IsolationViolationHandler) -> None:
        """Add a handler."""
        self.handlers.append(handler)

    def handle(self, violation: IsolationViolation) -> None:
        """Handle the violation with all handlers."""
        for handler in self.handlers:
            try:
                handler.handle(violation)
            except Exception:
                # Don't let one handler failure stop others
                # (unless it's the raising handler, which is expected to raise)
                if isinstance(handler, RaisingViolationHandler):
                    raise


class MetricsViolationHandler:
    """Handler that collects violation metrics.

    Useful for monitoring and alerting.
    """

    def __init__(self) -> None:
        self._violations: list[IsolationViolation] = []
        self._counts: dict[str, int] = {}
        self._by_tenant: dict[str, int] = {}

    @property
    def total_violations(self) -> int:
        """Total number of violations."""
        return len(self._violations)

    @property
    def violations_by_type(self) -> dict[str, int]:
        """Violations grouped by type."""
        return dict(self._counts)

    @property
    def violations_by_tenant(self) -> dict[str, int]:
        """Violations grouped by source tenant."""
        return dict(self._by_tenant)

    def handle(self, violation: IsolationViolation) -> None:
        """Record the violation in metrics."""
        self._violations.append(violation)
        self._counts[violation.violation_type] = (
            self._counts.get(violation.violation_type, 0) + 1
        )
        self._by_tenant[violation.source_tenant_id] = (
            self._by_tenant.get(violation.source_tenant_id, 0) + 1
        )

    def get_recent_violations(self, limit: int = 100) -> list[IsolationViolation]:
        """Get recent violations."""
        return self._violations[-limit:]

    def clear(self) -> None:
        """Clear all recorded violations."""
        self._violations.clear()
        self._counts.clear()
        self._by_tenant.clear()


# =============================================================================
# Resource Ownership Validators
# =============================================================================


class DefaultResourceOwnershipValidator:
    """Default implementation of resource ownership validation.

    Uses prefix-based ownership detection and an optional registry.
    """

    def __init__(
        self,
        *,
        prefix_separator: str = ":",
        registry: dict[tuple[str, ResourceType], str] | None = None,
    ) -> None:
        self.prefix_separator = prefix_separator
        self._registry = registry or {}

    def register_owner(
        self,
        resource_id: str,
        resource_type: ResourceType,
        tenant_id: str,
    ) -> None:
        """Register a resource owner."""
        self._registry[(resource_id, resource_type)] = tenant_id

    def unregister(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Unregister a resource."""
        self._registry.pop((resource_id, resource_type), None)

    def get_owner(
        self,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Get the owner of a resource."""
        # Check registry first
        owner = self._registry.get((resource_id, resource_type))
        if owner:
            return owner

        # Try to extract from prefix
        if self.prefix_separator in resource_id:
            parts = resource_id.split(self.prefix_separator)
            if len(parts) >= 2 and parts[0] in ("tenant", "phy", "shared"):
                return parts[1]

        return None

    def is_owner(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check if a tenant owns a resource."""
        owner = self.get_owner(resource_id, resource_type, context=context)
        return owner == tenant_id if owner else False


# =============================================================================
# Access Validators
# =============================================================================


class IsolationAccessValidator:
    """Validates access and generates violations for denials.

    Combines isolation checking with violation handling.
    """

    def __init__(
        self,
        *,
        enforcer: Any,  # IsolationEnforcer
        violation_handler: IsolationViolationHandler | None = None,
    ) -> None:
        self.enforcer = enforcer
        self.violation_handler = violation_handler or LoggingViolationHandler()

    def validate_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        resource_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> IsolationResult:
        """Validate access and handle violations."""
        is_allowed = self.enforcer.check_access(
            source_tenant_id,
            target_tenant_id,
            resource_type,
            permission,
            context=context,
        )

        if is_allowed:
            return IsolationResult.allowed()

        # Create violation
        violation = IsolationViolation(
            violation_type="cross_tenant_access",
            source_tenant_id=source_tenant_id,
            target_tenant_id=target_tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            permission=permission,
            message=(
                f"Tenant '{source_tenant_id}' denied access to "
                f"{resource_type.value} owned by tenant '{target_tenant_id}'"
            ),
            context=tuple(context.items()) if context else (),
        )

        # Handle violation
        self.violation_handler.handle(violation)

        return IsolationResult.denied(violation)

    def validate_ownership(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> IsolationResult:
        """Validate resource ownership and handle violations."""
        is_owner = self.enforcer.validate_resource_ownership(
            tenant_id, resource_id, resource_type, context=context
        )

        if is_owner:
            return IsolationResult.allowed()

        # Create violation
        violation = IsolationViolation(
            violation_type="ownership_violation",
            source_tenant_id=tenant_id,
            target_tenant_id=None,
            resource_type=resource_type,
            resource_id=resource_id,
            permission=None,
            message=(
                f"Tenant '{tenant_id}' claimed ownership of "
                f"{resource_type.value} '{resource_id}' but validation failed"
            ),
            context=tuple(context.items()) if context else (),
        )

        # Handle violation
        self.violation_handler.handle(violation)

        return IsolationResult.denied(violation)
