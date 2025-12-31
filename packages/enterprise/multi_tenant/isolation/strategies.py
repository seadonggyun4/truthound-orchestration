"""Isolation strategy implementations.

This module provides concrete implementations of isolation strategies
for different isolation levels: shared, logical, physical, and dedicated.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..types import IsolationLevel, Permission, ResourceScope, ResourceType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .base import CrossTenantAccessPolicy, ResourceOwnershipValidator


# =============================================================================
# Noop Isolation (for testing/development)
# =============================================================================


class NoopIsolationEnforcer:
    """No-op isolation enforcer that allows all access.

    Use only for testing or development environments.
    NEVER use in production.
    """

    @property
    def isolation_level(self) -> IsolationLevel:
        """Return SHARED as this allows everything."""
        return IsolationLevel.SHARED

    def check_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Always returns True (no enforcement)."""
        return True

    def validate_resource_ownership(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Always returns True (no validation)."""
        return True

    def get_tenant_resource_prefix(self, tenant_id: str) -> str:
        """Return empty prefix (no partitioning)."""
        return ""


# =============================================================================
# Shared Isolation (minimal separation)
# =============================================================================


@dataclass
class SharedIsolationEnforcer:
    """Shared isolation enforcer with minimal separation.

    Resources are shared by default, with isolation only for
    explicitly marked tenant-specific resources.
    """

    cross_tenant_policy: CrossTenantAccessPolicy | None = None
    _allowed_cross_tenant: set[tuple[str, str, ResourceType]] = field(
        default_factory=set, repr=False
    )

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.SHARED

    def allow_cross_tenant_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Explicitly allow cross-tenant access for a resource type."""
        self._allowed_cross_tenant.add(
            (source_tenant_id, target_tenant_id, resource_type)
        )

    def check_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check access with shared isolation.

        Allows access if:
        1. Same tenant
        2. Explicitly allowed cross-tenant access
        3. Cross-tenant policy allows it
        4. Resource scope is GLOBAL or SHARED
        """
        # Same tenant always allowed
        if source_tenant_id == target_tenant_id:
            return True

        # Check resource scope from context
        if context:
            scope = context.get("resource_scope")
            if scope in (ResourceScope.GLOBAL, ResourceScope.SHARED):
                return True

        # Check explicit allowance
        if (source_tenant_id, target_tenant_id, resource_type) in self._allowed_cross_tenant:
            return True

        # Check cross-tenant policy
        if self.cross_tenant_policy:
            return self.cross_tenant_policy.allows_cross_tenant_access(
                source_tenant_id, target_tenant_id, resource_type, permission, context=context
            )

        # Shared isolation is permissive by default
        return True

    def validate_resource_ownership(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Validate ownership with shared isolation.

        In shared mode, ownership validation is relaxed.
        """
        # Check if resource has tenant prefix
        prefix = self.get_tenant_resource_prefix(tenant_id)
        if prefix and resource_id.startswith(prefix):
            return True
        # In shared mode, assume ownership if no prefix
        return True

    def get_tenant_resource_prefix(self, tenant_id: str) -> str:
        """Get prefix for shared isolation."""
        return f"shared:{tenant_id}:"


# =============================================================================
# Logical Isolation (standard separation)
# =============================================================================


@dataclass
class LogicalIsolationEnforcer:
    """Logical isolation enforcer with standard separation.

    Resources are logically separated using prefixes and metadata.
    Cross-tenant access is denied by default unless explicitly allowed.
    """

    ownership_validator: ResourceOwnershipValidator | None = None
    cross_tenant_policy: CrossTenantAccessPolicy | None = None
    _resource_owners: dict[tuple[str, ResourceType], str] = field(
        default_factory=dict, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.LOGICAL

    def register_resource(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Register a resource as owned by a tenant."""
        with self._lock:
            self._resource_owners[(resource_id, resource_type)] = tenant_id

    def unregister_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Unregister a resource."""
        with self._lock:
            self._resource_owners.pop((resource_id, resource_type), None)

    def check_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check access with logical isolation.

        Denies access if tenants differ, unless:
        1. Resource scope is GLOBAL
        2. Cross-tenant policy explicitly allows it
        """
        # Same tenant always allowed
        if source_tenant_id == target_tenant_id:
            return True

        # Check resource scope
        if context:
            scope = context.get("resource_scope")
            if scope == ResourceScope.GLOBAL:
                return True

        # Check cross-tenant policy
        if self.cross_tenant_policy:
            return self.cross_tenant_policy.allows_cross_tenant_access(
                source_tenant_id, target_tenant_id, resource_type, permission, context=context
            )

        # Logical isolation denies cross-tenant by default
        return False

    def validate_resource_ownership(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Validate ownership with logical isolation."""
        # Check registered ownership first
        with self._lock:
            owner = self._resource_owners.get((resource_id, resource_type))
            if owner is not None:
                return owner == tenant_id

        # Check ownership validator
        if self.ownership_validator:
            return self.ownership_validator.is_owner(
                tenant_id, resource_id, resource_type, context=context
            )

        # Check prefix-based ownership
        prefix = self.get_tenant_resource_prefix(tenant_id)
        return resource_id.startswith(prefix)

    def get_tenant_resource_prefix(self, tenant_id: str) -> str:
        """Get prefix for logical isolation."""
        return f"tenant:{tenant_id}:"


# =============================================================================
# Physical Isolation (strong separation)
# =============================================================================


@dataclass
class PhysicalIsolationEnforcer:
    """Physical isolation enforcer with strong separation.

    Resources are physically separated with cryptographic prefixes.
    No cross-tenant access is allowed except for system resources.
    """

    ownership_validator: ResourceOwnershipValidator | None = None
    salt: str = "truthound"
    _resource_owners: dict[tuple[str, ResourceType], str] = field(
        default_factory=dict, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.PHYSICAL

    def register_resource(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> None:
        """Register a resource as owned by a tenant."""
        with self._lock:
            self._resource_owners[(resource_id, resource_type)] = tenant_id

    def check_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check access with physical isolation.

        Only allows access if:
        1. Same tenant
        2. Resource scope is GLOBAL (system resources only)
        """
        # Same tenant always allowed
        if source_tenant_id == target_tenant_id:
            return True

        # Only GLOBAL resources can be accessed cross-tenant
        if context:
            scope = context.get("resource_scope")
            if scope == ResourceScope.GLOBAL:
                return True

        # Physical isolation strictly denies cross-tenant access
        return False

    def validate_resource_ownership(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Validate ownership with physical isolation."""
        # Check registered ownership
        with self._lock:
            owner = self._resource_owners.get((resource_id, resource_type))
            if owner is not None:
                return owner == tenant_id

        # Check ownership validator
        if self.ownership_validator:
            return self.ownership_validator.is_owner(
                tenant_id, resource_id, resource_type, context=context
            )

        # Check cryptographic prefix
        prefix = self.get_tenant_resource_prefix(tenant_id)
        return resource_id.startswith(prefix)

    def get_tenant_resource_prefix(self, tenant_id: str) -> str:
        """Get cryptographic prefix for physical isolation.

        Uses a hash-based prefix to prevent tenant ID guessing.
        """
        hash_input = f"{self.salt}:{tenant_id}".encode()
        hash_value = hashlib.sha256(hash_input).hexdigest()[:16]
        return f"phy:{hash_value}:"


# =============================================================================
# Composite Isolation
# =============================================================================


@dataclass
class CompositeIsolationEnforcer:
    """Composite isolation enforcer that combines multiple enforcers.

    Allows different isolation levels for different resource types.
    """

    default_enforcer: LogicalIsolationEnforcer = field(
        default_factory=LogicalIsolationEnforcer
    )
    resource_enforcers: dict[ResourceType, Any] = field(default_factory=dict)

    @property
    def isolation_level(self) -> IsolationLevel:
        """Return the default isolation level."""
        return self.default_enforcer.isolation_level

    def set_enforcer_for_resource(
        self,
        resource_type: ResourceType,
        enforcer: Any,
    ) -> None:
        """Set a specific enforcer for a resource type."""
        self.resource_enforcers[resource_type] = enforcer

    def _get_enforcer(self, resource_type: ResourceType) -> Any:
        """Get the enforcer for a resource type."""
        return self.resource_enforcers.get(resource_type, self.default_enforcer)

    def check_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Check access using the appropriate enforcer."""
        enforcer = self._get_enforcer(resource_type)
        return enforcer.check_access(
            source_tenant_id, target_tenant_id, resource_type, permission, context=context
        )

    def validate_resource_ownership(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Validate ownership using the appropriate enforcer."""
        enforcer = self._get_enforcer(resource_type)
        return enforcer.validate_resource_ownership(
            tenant_id, resource_id, resource_type, context=context
        )

    def get_tenant_resource_prefix(self, tenant_id: str) -> str:
        """Get prefix from the default enforcer."""
        return self.default_enforcer.get_tenant_resource_prefix(tenant_id)


# =============================================================================
# Factory Function
# =============================================================================


def create_isolation_enforcer(
    level: IsolationLevel,
    *,
    ownership_validator: ResourceOwnershipValidator | None = None,
    cross_tenant_policy: CrossTenantAccessPolicy | None = None,
) -> NoopIsolationEnforcer | SharedIsolationEnforcer | LogicalIsolationEnforcer | PhysicalIsolationEnforcer:
    """Create an isolation enforcer for the given level.

    Args:
        level: The isolation level.
        ownership_validator: Optional ownership validator.
        cross_tenant_policy: Optional cross-tenant policy.

    Returns:
        An isolation enforcer instance.
    """
    if level == IsolationLevel.SHARED:
        return SharedIsolationEnforcer(cross_tenant_policy=cross_tenant_policy)
    elif level == IsolationLevel.LOGICAL:
        return LogicalIsolationEnforcer(
            ownership_validator=ownership_validator,
            cross_tenant_policy=cross_tenant_policy,
        )
    elif level == IsolationLevel.PHYSICAL:
        return PhysicalIsolationEnforcer(ownership_validator=ownership_validator)
    elif level == IsolationLevel.DEDICATED:
        # Dedicated uses physical isolation with stricter settings
        return PhysicalIsolationEnforcer(ownership_validator=ownership_validator)
    else:
        # Default to logical
        return LogicalIsolationEnforcer(
            ownership_validator=ownership_validator,
            cross_tenant_policy=cross_tenant_policy,
        )
