"""Base protocols for tenant isolation.

This module defines the core protocols that isolation strategies
and validators must implement. Using protocols enables duck typing
and easy extension without tight coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..types import IsolationLevel, Permission, ResourceType


@runtime_checkable
class IsolationEnforcer(Protocol):
    """Protocol for enforcing tenant isolation.

    Implementations ensure that tenants cannot access
    each other's resources based on the configured isolation level.
    """

    @property
    def isolation_level(self) -> IsolationLevel:
        """Return the isolation level this enforcer implements."""
        ...

    def check_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check if access is allowed between tenants.

        Args:
            source_tenant_id: The tenant requesting access.
            target_tenant_id: The tenant owning the resource.
            resource_type: The type of resource being accessed.
            permission: The permission being requested.
            context: Additional context for the check.

        Returns:
            True if access is allowed, False otherwise.
        """
        ...

    def validate_resource_ownership(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Validate that a resource belongs to the tenant.

        Args:
            tenant_id: The tenant claiming ownership.
            resource_id: The resource identifier.
            resource_type: The type of resource.
            context: Additional context for validation.

        Returns:
            True if the resource belongs to the tenant, False otherwise.
        """
        ...

    def get_tenant_resource_prefix(self, tenant_id: str) -> str:
        """Get the resource prefix for a tenant.

        Used for partitioning resources by tenant.

        Args:
            tenant_id: The tenant identifier.

        Returns:
            A prefix string for tenant resources.
        """
        ...


@runtime_checkable
class IsolationValidator(Protocol):
    """Protocol for validating isolation constraints.

    Validators check that isolation rules are being followed
    and can be composed for complex validation scenarios.
    """

    def validate(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> IsolationValidationResult:
        """Validate isolation for a resource access.

        Args:
            tenant_id: The tenant attempting access.
            resource_id: The resource being accessed.
            resource_type: The type of resource.
            context: Additional context for validation.

        Returns:
            Validation result with details.
        """
        ...


@runtime_checkable
class ResourceOwnershipValidator(Protocol):
    """Protocol for validating resource ownership.

    Implementations determine whether a resource belongs to a tenant.
    """

    def get_owner(
        self,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Get the owner tenant ID for a resource.

        Args:
            resource_id: The resource identifier.
            resource_type: The type of resource.
            context: Additional context.

        Returns:
            The owner tenant ID, or None if not found.
        """
        ...

    def is_owner(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check if a tenant owns a resource.

        Args:
            tenant_id: The tenant to check.
            resource_id: The resource identifier.
            resource_type: The type of resource.
            context: Additional context.

        Returns:
            True if the tenant owns the resource, False otherwise.
        """
        ...


@runtime_checkable
class CrossTenantAccessPolicy(Protocol):
    """Protocol for policies governing cross-tenant access.

    Some resources may be shared between tenants or accessible
    system-wide. This protocol defines policies for such access.
    """

    def allows_cross_tenant_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check if cross-tenant access is allowed.

        Args:
            source_tenant_id: The tenant requesting access.
            target_tenant_id: The tenant owning the resource.
            resource_type: The type of resource.
            permission: The permission requested.
            context: Additional context.

        Returns:
            True if cross-tenant access is allowed, False otherwise.
        """
        ...

    def get_shared_tenants(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> Sequence[str]:
        """Get tenants with whom a resource is shared.

        Args:
            resource_id: The resource identifier.
            resource_type: The type of resource.

        Returns:
            List of tenant IDs the resource is shared with.
        """
        ...


# =============================================================================
# Result Types
# =============================================================================


class IsolationValidationResult:
    """Result of an isolation validation.

    Encapsulates the result of validating tenant isolation,
    including whether it passed and any details.
    """

    __slots__ = ("is_valid", "message", "details", "violations")

    def __init__(
        self,
        is_valid: bool,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        violations: list[str] | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}
        self.violations = violations or []

    @classmethod
    def success(
        cls,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> IsolationValidationResult:
        """Create a successful validation result."""
        return cls(True, message=message, details=details)

    @classmethod
    def failure(
        cls,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        violations: list[str] | None = None,
    ) -> IsolationValidationResult:
        """Create a failed validation result."""
        return cls(False, message=message, details=details, violations=violations)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid

    def __repr__(self) -> str:
        return f"IsolationValidationResult(is_valid={self.is_valid}, message={self.message!r})"
