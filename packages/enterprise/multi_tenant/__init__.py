"""Multi-tenant support for Truthound Orchestration.

This module provides comprehensive multi-tenant functionality including:

- **Tenant Management**: Create, update, delete, and manage tenant lifecycles
- **Context Management**: Thread-safe and async-safe tenant context propagation
- **Isolation Strategies**: Configurable isolation levels (shared, logical, physical)
- **Storage Backends**: Pluggable storage for tenant configurations
- **Middleware**: HTTP request tenant resolution
- **Hooks**: Lifecycle event hooks for monitoring and integration

Quick Start
-----------

Basic tenant management::

    from packages.enterprise.multi_tenant import (
        TenantRegistry,
        TenantConfig,
        TenantContextManager,
        get_current_tenant_id,
    )

    # Create a registry
    registry = TenantRegistry()

    # Create a tenant
    tenant = registry.create_tenant(
        "acme-corp",
        "ACME Corporation",
        activate=True,
    )

    # Use tenant context
    with TenantContextManager("acme-corp", config=tenant):
        print(get_current_tenant_id())  # "acme-corp"
        # All operations here run in tenant context

Using the global registry::

    from packages.enterprise.multi_tenant import (
        get_registry,
        create_tenant,
        get_tenant,
    )

    # Configure global registry
    registry = get_registry()

    # Create and get tenants
    tenant = create_tenant("tenant-1", "Tenant One", activate=True)
    retrieved = get_tenant("tenant-1")

Isolation enforcement::

    from packages.enterprise.multi_tenant import (
        create_isolation_enforcer,
        IsolationLevel,
        ResourceType,
        Permission,
    )

    # Create enforcer for logical isolation
    enforcer = create_isolation_enforcer(IsolationLevel.LOGICAL)

    # Check access between tenants
    allowed = enforcer.check_access(
        source_tenant_id="tenant-a",
        target_tenant_id="tenant-b",
        resource_type=ResourceType.ENGINE,
        permission=Permission.READ,
    )

Middleware for web frameworks::

    from packages.enterprise.multi_tenant import (
        TenantMiddleware,
        HeaderTenantResolver,
        create_default_middleware,
    )

    # Create middleware with default settings
    middleware = create_default_middleware(
        registry=registry,
        require_tenant=True,
    )

    # Resolve tenant from request context
    ctx_manager = middleware.get_context_manager({
        "headers": {"X-Tenant-ID": "my-tenant"},
    })

Architecture
------------

The module follows these design principles:

1. **Protocol-based**: Uses protocols for extensibility without tight coupling
2. **Immutable configs**: All configurations are frozen dataclasses
3. **Thread-safe**: Context management uses contextvars for thread/async safety
4. **Pluggable storage**: Storage backends implement a common protocol
5. **Observable hooks**: Lifecycle events can be monitored via hooks

Directory Structure::

    multi_tenant/
    ├── __init__.py      # Public API exports (this file)
    ├── config.py        # Configuration classes
    ├── context.py       # Context management
    ├── exceptions.py    # Exception hierarchy
    ├── hooks.py         # Lifecycle hooks
    ├── middleware.py    # HTTP middleware
    ├── registry.py      # Tenant registry
    ├── types.py         # Enums and protocols
    ├── isolation/       # Isolation strategies
    │   ├── base.py      # Protocols
    │   ├── strategies.py # Implementations
    │   └── validators.py # Validation
    └── storage/         # Storage backends
        ├── base.py      # Protocols
        ├── memory.py    # In-memory
        └── file.py      # File-based
"""

from __future__ import annotations

# =============================================================================
# Exceptions
# =============================================================================

from .exceptions import (
    # Base
    MultiTenantError,
    # Lifecycle
    TenantNotFoundError,
    TenantAlreadyExistsError,
    TenantDisabledError,
    TenantSuspendedError,
    # Configuration
    TenantConfigurationError,
    TenantConfigValidationError,
    # Isolation
    TenantIsolationError,
    CrossTenantAccessError,
    # Authorization
    TenantAuthorizationError,
    TenantPermissionDeniedError,
    # Quota
    TenantQuotaError,
    TenantQuotaExceededError,
    TenantResourceLimitError,
    # Context
    TenantContextError,
    NoTenantContextError,
    TenantContextAlreadySetError,
    # Storage
    TenantStorageError,
    TenantDataNotFoundError,
    # Middleware
    TenantMiddlewareError,
    TenantResolutionError,
)

# =============================================================================
# Types and Enums
# =============================================================================

from .types import (
    # Status and Lifecycle
    TenantStatus,
    TenantTier,
    # Isolation
    IsolationLevel,
    ResourceScope,
    # Quota
    QuotaType,
    QuotaPeriod,
    # Resolution
    TenantResolutionStrategy,
    # Permission
    Permission,
    ResourceType,
    # Data classes
    TenantId,
    QuotaLimit,
    QuotaUsage,
    TenantMetadata,
    # Protocols
    TenantAware,
    TenantResolver,
    TenantValidator,
    QuotaEnforcer,
    IsolationEnforcer,
)

# =============================================================================
# Configuration
# =============================================================================

from .config import (
    TenantConfig,
    MultiTenantConfig,
    TIER_QUOTA_DEFAULTS,
    # Presets
    DEFAULT_CONFIG,
    TESTING_CONFIG,
    PRODUCTION_CONFIG,
    STRICT_CONFIG,
    SINGLE_TENANT_CONFIG,
)

# =============================================================================
# Context Management
# =============================================================================

from .context import (
    # Context class
    TenantContext,
    # Context managers
    TenantContextManager,
    AdminTenantContext,
    tenant_context,
    # Access functions
    get_current_tenant,
    get_current_tenant_required,
    get_current_tenant_id,
    get_current_tenant_id_required,
    is_tenant_context_set,
    # Manual context control
    set_tenant_context,
    reset_tenant_context,
    clear_tenant_context,
    get_tenant_stack,
    copy_context_to_thread,
    # Decorators
    require_tenant_context,
    require_tenant_context_async,
    with_tenant,
    with_tenant_async,
    # Propagation
    TenantContextPropagator,
    context_propagator,
)

# =============================================================================
# Registry
# =============================================================================

from .registry import (
    TenantRegistry,
    # Singleton management
    get_registry,
    configure_registry,
    reset_registry,
    # Convenience functions
    create_tenant,
    get_tenant,
    list_tenants,
    tenant_exists,
)

# =============================================================================
# Isolation
# =============================================================================

from .isolation import (
    # Protocols
    IsolationEnforcer,
    IsolationValidator,
    ResourceOwnershipValidator,
    # Strategies
    NoopIsolationEnforcer,
    SharedIsolationEnforcer,
    LogicalIsolationEnforcer,
    PhysicalIsolationEnforcer,
    CompositeIsolationEnforcer,
    create_isolation_enforcer,
    # Validators
    IsolationResult,
    IsolationViolation,
    IsolationViolationHandler,
    LoggingViolationHandler,
    RaisingViolationHandler,
    DefaultResourceOwnershipValidator,
)

# =============================================================================
# Storage
# =============================================================================

from .storage import (
    # Protocols
    TenantStorage,
    AsyncTenantStorage,
    TenantDataStorage,
    # Implementations
    InMemoryTenantStorage,
    InMemoryTenantDataStorage,
    FileTenantStorage,
)

# =============================================================================
# Middleware
# =============================================================================

from .middleware import (
    # Resolvers
    HeaderTenantResolver,
    SubdomainTenantResolver,
    PathTenantResolver,
    QueryParamTenantResolver,
    JWTClaimTenantResolver,
    ContextTenantResolver,
    CompositeTenantResolver,
    # Middleware
    TenantMiddleware,
    # Decorators
    with_tenant_middleware,
    with_tenant_middleware_async,
    # Factory
    create_default_middleware,
)

# =============================================================================
# Hooks
# =============================================================================

from .hooks import (
    # Protocols
    TenantHook,
    AsyncTenantHook,
    # Base
    BaseTenantHook,
    # Implementations
    LoggingTenantHook,
    MetricsTenantHook,
    AuditTenantHook,
    CompositeTenantHook,
    CallbackTenantHook,
    # Data
    TenantMetrics,
    AuditEvent,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Exceptions
    "MultiTenantError",
    "TenantNotFoundError",
    "TenantAlreadyExistsError",
    "TenantDisabledError",
    "TenantSuspendedError",
    "TenantConfigurationError",
    "TenantConfigValidationError",
    "TenantIsolationError",
    "CrossTenantAccessError",
    "TenantAuthorizationError",
    "TenantPermissionDeniedError",
    "TenantQuotaError",
    "TenantQuotaExceededError",
    "TenantResourceLimitError",
    "TenantContextError",
    "NoTenantContextError",
    "TenantContextAlreadySetError",
    "TenantStorageError",
    "TenantDataNotFoundError",
    "TenantMiddlewareError",
    "TenantResolutionError",
    # Types and Enums
    "TenantStatus",
    "TenantTier",
    "IsolationLevel",
    "ResourceScope",
    "QuotaType",
    "QuotaPeriod",
    "TenantResolutionStrategy",
    "Permission",
    "ResourceType",
    "TenantId",
    "QuotaLimit",
    "QuotaUsage",
    "TenantMetadata",
    "TenantAware",
    "TenantResolver",
    "TenantValidator",
    "QuotaEnforcer",
    "IsolationEnforcer",
    # Configuration
    "TenantConfig",
    "MultiTenantConfig",
    "TIER_QUOTA_DEFAULTS",
    "DEFAULT_CONFIG",
    "TESTING_CONFIG",
    "PRODUCTION_CONFIG",
    "STRICT_CONFIG",
    "SINGLE_TENANT_CONFIG",
    # Context
    "TenantContext",
    "TenantContextManager",
    "AdminTenantContext",
    "tenant_context",
    "get_current_tenant",
    "get_current_tenant_required",
    "get_current_tenant_id",
    "get_current_tenant_id_required",
    "is_tenant_context_set",
    "set_tenant_context",
    "reset_tenant_context",
    "clear_tenant_context",
    "get_tenant_stack",
    "copy_context_to_thread",
    "require_tenant_context",
    "require_tenant_context_async",
    "with_tenant",
    "with_tenant_async",
    "TenantContextPropagator",
    "context_propagator",
    # Registry
    "TenantRegistry",
    "get_registry",
    "configure_registry",
    "reset_registry",
    "create_tenant",
    "get_tenant",
    "list_tenants",
    "tenant_exists",
    # Isolation
    "IsolationValidator",
    "ResourceOwnershipValidator",
    "NoopIsolationEnforcer",
    "SharedIsolationEnforcer",
    "LogicalIsolationEnforcer",
    "PhysicalIsolationEnforcer",
    "CompositeIsolationEnforcer",
    "create_isolation_enforcer",
    "IsolationResult",
    "IsolationViolation",
    "IsolationViolationHandler",
    "LoggingViolationHandler",
    "RaisingViolationHandler",
    "DefaultResourceOwnershipValidator",
    # Storage
    "TenantStorage",
    "AsyncTenantStorage",
    "TenantDataStorage",
    "InMemoryTenantStorage",
    "InMemoryTenantDataStorage",
    "FileTenantStorage",
    # Middleware
    "HeaderTenantResolver",
    "SubdomainTenantResolver",
    "PathTenantResolver",
    "QueryParamTenantResolver",
    "JWTClaimTenantResolver",
    "ContextTenantResolver",
    "CompositeTenantResolver",
    "TenantMiddleware",
    "with_tenant_middleware",
    "with_tenant_middleware_async",
    "create_default_middleware",
    # Hooks
    "TenantHook",
    "AsyncTenantHook",
    "BaseTenantHook",
    "LoggingTenantHook",
    "MetricsTenantHook",
    "AuditTenantHook",
    "CompositeTenantHook",
    "CallbackTenantHook",
    "TenantMetrics",
    "AuditEvent",
]
