"""Multi-tenant type definitions and enumerations.

This module defines the core types, enums, and protocols used throughout
the multi-tenant module. All enums use string values for JSON serialization
compatibility and human readability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Tenant Status and Lifecycle
# =============================================================================


class TenantStatus(str, Enum):
    """Status of a tenant in the system.

    Defines the lifecycle states a tenant can be in:
    - PENDING: Tenant created but not yet activated
    - ACTIVE: Tenant is fully operational
    - SUSPENDED: Tenant temporarily disabled (can be reactivated)
    - DISABLED: Tenant permanently disabled
    - ARCHIVED: Tenant data retained but not accessible
    - DELETED: Tenant marked for deletion (soft delete)
    """

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISABLED = "disabled"
    ARCHIVED = "archived"
    DELETED = "deleted"

    @property
    def is_operational(self) -> bool:
        """Check if the tenant can perform operations."""
        return self == TenantStatus.ACTIVE

    @property
    def is_accessible(self) -> bool:
        """Check if tenant data is accessible (read-only possible)."""
        return self in (TenantStatus.ACTIVE, TenantStatus.SUSPENDED)


class TenantTier(str, Enum):
    """Subscription tier for a tenant.

    Defines the service level and quota limits:
    - FREE: Basic functionality with limited quotas
    - STARTER: Entry-level paid tier
    - PROFESSIONAL: Standard paid tier
    - ENTERPRISE: Full-featured enterprise tier
    - CUSTOM: Custom tier with negotiated limits
    """

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


# =============================================================================
# Isolation Levels
# =============================================================================


class IsolationLevel(str, Enum):
    """Level of isolation between tenants.

    Defines how strictly tenants are separated:
    - SHARED: Resources shared between tenants (cost-effective)
    - LOGICAL: Logical separation with shared infrastructure
    - PHYSICAL: Physical separation (e.g., separate databases)
    - DEDICATED: Fully dedicated resources per tenant
    """

    SHARED = "shared"
    LOGICAL = "logical"
    PHYSICAL = "physical"
    DEDICATED = "dedicated"

    @property
    def security_level(self) -> int:
        """Return a numeric security level (higher = more secure)."""
        levels = {
            IsolationLevel.SHARED: 1,
            IsolationLevel.LOGICAL: 2,
            IsolationLevel.PHYSICAL: 3,
            IsolationLevel.DEDICATED: 4,
        }
        return levels[self]


class ResourceScope(str, Enum):
    """Scope of a resource in relation to tenants.

    Defines resource ownership and visibility:
    - GLOBAL: Resource shared across all tenants (system-level)
    - TENANT: Resource belongs to a specific tenant
    - USER: Resource belongs to a specific user within a tenant
    - SHARED: Resource explicitly shared between tenants
    """

    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    SHARED = "shared"


# =============================================================================
# Quota and Resource Types
# =============================================================================


class QuotaType(str, Enum):
    """Types of quotas that can be applied to tenants.

    Defines various quota categories:
    - API_CALLS: Number of API calls allowed
    - STORAGE_BYTES: Storage space in bytes
    - ENGINES: Number of data quality engines
    - RULES: Number of validation rules
    - EXECUTIONS: Number of validation executions
    - USERS: Number of users per tenant
    - CONNECTIONS: Number of data source connections
    - CONCURRENT_JOBS: Number of concurrent jobs
    """

    API_CALLS = "api_calls"
    STORAGE_BYTES = "storage_bytes"
    ENGINES = "engines"
    RULES = "rules"
    EXECUTIONS = "executions"
    USERS = "users"
    CONNECTIONS = "connections"
    CONCURRENT_JOBS = "concurrent_jobs"


class QuotaPeriod(str, Enum):
    """Time period for quota measurement.

    Defines quota reset periods:
    - HOURLY: Quota resets every hour
    - DAILY: Quota resets every day
    - WEEKLY: Quota resets every week
    - MONTHLY: Quota resets every month
    - UNLIMITED: No time-based limit (absolute quota)
    """

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    UNLIMITED = "unlimited"


# =============================================================================
# Resolution Strategies
# =============================================================================


class TenantResolutionStrategy(str, Enum):
    """Strategy for resolving the current tenant from context.

    Defines how to identify the tenant for a request:
    - HEADER: Extract from HTTP header (e.g., X-Tenant-ID)
    - SUBDOMAIN: Extract from request subdomain
    - PATH: Extract from URL path (e.g., /tenants/{id}/...)
    - JWT_CLAIM: Extract from JWT token claim
    - API_KEY: Extract from API key lookup
    - QUERY_PARAM: Extract from query parameter
    - CONTEXT: Use explicitly set context
    - COMPOSITE: Try multiple strategies in order
    """

    HEADER = "header"
    SUBDOMAIN = "subdomain"
    PATH = "path"
    JWT_CLAIM = "jwt_claim"
    API_KEY = "api_key"
    QUERY_PARAM = "query_param"
    CONTEXT = "context"
    COMPOSITE = "composite"


# =============================================================================
# Permission and Authorization
# =============================================================================


class Permission(str, Enum):
    """Permissions that can be granted to tenants.

    Defines granular permissions for operations:
    - READ: Read access to resources
    - WRITE: Write/modify access to resources
    - DELETE: Delete access to resources
    - ADMIN: Administrative access
    - EXECUTE: Execute operations (e.g., run validations)
    - SHARE: Share resources with other tenants/users
    - EXPORT: Export data
    - CONFIGURE: Modify configuration
    """

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    SHARE = "share"
    EXPORT = "export"
    CONFIGURE = "configure"


class ResourceType(str, Enum):
    """Types of resources that can be managed per tenant.

    Defines resource categories:
    - ENGINE: Data quality engine instances
    - RULE: Validation rules
    - SCHEMA: Data schemas
    - CONNECTION: Data source connections
    - DATASET: Data sets
    - REPORT: Validation reports
    - SCHEDULE: Scheduled jobs
    - ALERT: Alert configurations
    - USER: User accounts
    - CONFIG: Configuration settings
    """

    ENGINE = "engine"
    RULE = "rule"
    SCHEMA = "schema"
    CONNECTION = "connection"
    DATASET = "dataset"
    REPORT = "report"
    SCHEDULE = "schedule"
    ALERT = "alert"
    USER = "user"
    CONFIG = "config"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class TenantId:
    """Value object representing a tenant identifier.

    Encapsulates tenant ID with validation and formatting.
    Using a value object ensures consistent handling of tenant IDs.

    Attributes:
        value: The actual tenant identifier string.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("Tenant ID cannot be empty")
        if not self.value.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Tenant ID must be alphanumeric (with - and _): {self.value}"
            )

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    @classmethod
    def from_string(cls, value: str) -> TenantId:
        """Create a TenantId from a string, normalizing format."""
        return cls(value.strip().lower())


@dataclass(frozen=True, slots=True)
class QuotaLimit:
    """Represents a quota limit for a tenant.

    Attributes:
        quota_type: The type of quota.
        limit: The maximum allowed value.
        period: The reset period for the quota.
        warning_threshold: Percentage at which to warn (0.0-1.0).
    """

    quota_type: QuotaType
    limit: int
    period: QuotaPeriod = QuotaPeriod.MONTHLY
    warning_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError(f"Quota limit must be non-negative: {self.limit}")
        if not 0.0 <= self.warning_threshold <= 1.0:
            raise ValueError(
                f"Warning threshold must be between 0.0 and 1.0: {self.warning_threshold}"
            )


@dataclass(frozen=True, slots=True)
class QuotaUsage:
    """Represents current quota usage for a tenant.

    Attributes:
        quota_type: The type of quota.
        current: Current usage value.
        limit: The maximum allowed value.
        period: The reset period.
        period_start: When the current period started.
        period_end: When the current period ends.
    """

    quota_type: QuotaType
    current: int
    limit: int
    period: QuotaPeriod
    period_start: datetime
    period_end: datetime

    @property
    def remaining(self) -> int:
        """Calculate remaining quota."""
        return max(0, self.limit - self.current)

    @property
    def usage_percentage(self) -> float:
        """Calculate usage as a percentage (0.0-1.0)."""
        if self.limit == 0:
            return 1.0 if self.current > 0 else 0.0
        return min(1.0, self.current / self.limit)

    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.current >= self.limit


@dataclass(frozen=True, slots=True)
class TenantMetadata:
    """Metadata associated with a tenant.

    Immutable metadata for audit and management purposes.

    Attributes:
        created_at: When the tenant was created.
        updated_at: When the tenant was last updated.
        created_by: Who created the tenant.
        updated_by: Who last updated the tenant.
        tags: Tags for categorization.
        labels: Key-value labels for organization.
    """

    created_at: datetime
    updated_at: datetime
    created_by: str | None = None
    updated_by: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    labels: tuple[tuple[str, str], ...] = ()

    def with_update(
        self,
        updated_by: str | None = None,
        tags: frozenset[str] | None = None,
        labels: tuple[tuple[str, str], ...] | None = None,
    ) -> TenantMetadata:
        """Create a new metadata with updated values."""
        return TenantMetadata(
            created_at=self.created_at,
            updated_at=datetime.now(),
            created_by=self.created_by,
            updated_by=updated_by or self.updated_by,
            tags=tags if tags is not None else self.tags,
            labels=labels if labels is not None else self.labels,
        )

    def get_label(self, key: str) -> str | None:
        """Get a label value by key."""
        for k, v in self.labels:
            if k == key:
                return v
        return None


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class TenantAware(Protocol):
    """Protocol for objects that are tenant-aware.

    Any class that needs to be associated with a tenant should
    implement this protocol.
    """

    @property
    def tenant_id(self) -> str:
        """Return the tenant ID this object belongs to."""
        ...


@runtime_checkable
class TenantResolver(Protocol):
    """Protocol for resolving tenant context from various sources.

    Implementations determine how to extract the tenant ID
    from different contexts (headers, tokens, etc.).
    """

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve the tenant ID from the given context.

        Args:
            context: A mapping containing request context (headers, params, etc.)

        Returns:
            The resolved tenant ID, or None if not found.
        """
        ...

    @property
    def strategy(self) -> TenantResolutionStrategy:
        """Return the resolution strategy used by this resolver."""
        ...


@runtime_checkable
class TenantValidator(Protocol):
    """Protocol for validating tenant operations.

    Implementations can validate tenant configurations,
    permissions, and other constraints.
    """

    def validate(self, tenant_id: str, context: Mapping[str, Any]) -> bool:
        """Validate a tenant operation.

        Args:
            tenant_id: The tenant to validate.
            context: Additional context for validation.

        Returns:
            True if validation passes, False otherwise.
        """
        ...


@runtime_checkable
class QuotaEnforcer(Protocol):
    """Protocol for enforcing tenant quotas.

    Implementations track and enforce quota limits.
    """

    def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        requested_amount: int = 1,
    ) -> bool:
        """Check if a quota allows the requested amount.

        Args:
            tenant_id: The tenant to check.
            quota_type: The type of quota.
            requested_amount: The amount being requested.

        Returns:
            True if quota allows the operation, False otherwise.
        """
        ...

    def record_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> None:
        """Record quota usage.

        Args:
            tenant_id: The tenant to record for.
            quota_type: The type of quota.
            amount: The amount used.
        """
        ...

    def get_usage(self, tenant_id: str, quota_type: QuotaType) -> QuotaUsage:
        """Get current quota usage.

        Args:
            tenant_id: The tenant to check.
            quota_type: The type of quota.

        Returns:
            Current quota usage information.
        """
        ...


@runtime_checkable
class IsolationEnforcer(Protocol):
    """Protocol for enforcing tenant isolation.

    Implementations ensure that tenants cannot access
    each other's resources.
    """

    def check_access(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        resource_type: ResourceType,
        permission: Permission,
    ) -> bool:
        """Check if access is allowed between tenants.

        Args:
            source_tenant_id: The tenant requesting access.
            target_tenant_id: The tenant owning the resource.
            resource_type: The type of resource.
            permission: The permission being requested.

        Returns:
            True if access is allowed, False otherwise.
        """
        ...

    def validate_isolation(
        self,
        tenant_id: str,
        resource_id: str,
        resource_type: ResourceType,
    ) -> bool:
        """Validate that a resource belongs to the tenant.

        Args:
            tenant_id: The tenant claiming ownership.
            resource_id: The resource identifier.
            resource_type: The type of resource.

        Returns:
            True if the resource belongs to the tenant, False otherwise.
        """
        ...


# =============================================================================
# Type Aliases
# =============================================================================

# Configuration type alias
TenantConfigDict = dict[str, Any]

# Quota limits mapping
QuotaLimitsMap = Mapping[QuotaType, QuotaLimit]

# Permission set
PermissionSet = frozenset[Permission]

# Resource permissions mapping
ResourcePermissions = Mapping[ResourceType, PermissionSet]

# Labels type
Labels = Sequence[tuple[str, str]]
