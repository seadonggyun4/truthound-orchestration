"""Tenant registry for managing tenant lifecycle.

This module provides a central registry for managing tenants,
including creation, retrieval, update, and deletion operations.
Uses the singleton pattern with thread-safe access.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterator

from .config import MultiTenantConfig, TenantConfig, TIER_QUOTA_DEFAULTS
from .context import TenantContext, TenantContextManager
from .exceptions import (
    TenantAlreadyExistsError,
    TenantDisabledError,
    TenantNotFoundError,
    TenantSuspendedError,
)
from .types import IsolationLevel, TenantMetadata, TenantStatus, TenantTier

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .hooks import TenantHook
    from .storage.base import TenantStorage


# =============================================================================
# Tenant Registry
# =============================================================================


@dataclass
class TenantRegistry:
    """Central registry for managing tenants.

    Provides CRUD operations for tenants with:
    - Thread-safe access
    - Event hooks for lifecycle events
    - Caching support
    - Integration with storage backends

    The registry uses a singleton pattern by default but can be
    instantiated directly for testing or multi-registry scenarios.
    """

    config: MultiTenantConfig = field(default_factory=MultiTenantConfig)
    storage: TenantStorage | None = None
    hooks: list[TenantHook] = field(default_factory=list)

    # Internal state
    _tenants: dict[str, TenantConfig] = field(default_factory=dict, repr=False)
    _cache: dict[str, tuple[TenantConfig, datetime]] = field(
        default_factory=dict, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        # Initialize storage if configured
        if self.storage is None and self.config.storage_backend != "memory":
            # Defer storage creation - user should set storage explicitly
            pass

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        *,
        tier: TenantTier | None = None,
        isolation_level: IsolationLevel | None = None,
        settings: Mapping[str, Any] | None = None,
        features: frozenset[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        activate: bool = False,
    ) -> TenantConfig:
        """Create a new tenant.

        Args:
            tenant_id: Unique identifier for the tenant.
            name: Human-readable name.
            tier: Subscription tier (defaults to config default).
            isolation_level: Isolation level (defaults to config default).
            settings: Initial settings.
            features: Enabled features.
            metadata: Initial metadata.
            activate: Whether to immediately activate the tenant.

        Returns:
            The created tenant configuration.

        Raises:
            TenantAlreadyExistsError: If a tenant with the ID already exists.
        """
        with self._lock:
            if tenant_id in self._tenants:
                raise TenantAlreadyExistsError(tenant_id)

            # Apply defaults
            tier = tier or self.config.default_tier
            isolation_level = isolation_level or self.config.default_isolation_level
            quotas = TIER_QUOTA_DEFAULTS.get(tier, ())

            # Create metadata
            now = datetime.now()
            tenant_metadata = TenantMetadata(
                created_at=now,
                updated_at=now,
                tags=frozenset(metadata.get("tags", [])) if metadata else frozenset(),
                labels=tuple(metadata.get("labels", {}).items()) if metadata else (),
            )

            # Create config
            tenant = TenantConfig(
                tenant_id=tenant_id,
                name=name,
                status=TenantStatus.ACTIVE if activate else TenantStatus.PENDING,
                tier=tier,
                isolation_level=isolation_level,
                quotas=quotas,
                metadata=tenant_metadata,
                settings=tuple(settings.items()) if settings else (),
                features=features or frozenset(),
            )

            # Store
            self._tenants[tenant_id] = tenant
            self._invalidate_cache(tenant_id)

            # Persist if storage available
            if self.storage:
                self.storage.save(tenant)

            # Notify hooks
            self._notify_hooks("on_tenant_created", tenant)

            return tenant

    def get_tenant(
        self,
        tenant_id: str,
        *,
        include_disabled: bool = False,
    ) -> TenantConfig:
        """Get a tenant by ID.

        Args:
            tenant_id: The tenant identifier.
            include_disabled: Whether to include disabled tenants.

        Returns:
            The tenant configuration.

        Raises:
            TenantNotFoundError: If the tenant does not exist.
            TenantDisabledError: If the tenant is disabled and include_disabled is False.
            TenantSuspendedError: If the tenant is suspended.
        """
        # Check cache first
        if self.config.cache_tenant_configs:
            cached = self._get_from_cache(tenant_id)
            if cached:
                return self._validate_tenant_access(cached, include_disabled)

        with self._lock:
            tenant = self._tenants.get(tenant_id)

        # Try storage if not in memory
        if tenant is None and self.storage:
            tenant = self.storage.load(tenant_id)
            if tenant:
                with self._lock:
                    self._tenants[tenant_id] = tenant

        if tenant is None:
            raise TenantNotFoundError(tenant_id)

        # Cache the result
        if self.config.cache_tenant_configs:
            self._add_to_cache(tenant_id, tenant)

        return self._validate_tenant_access(tenant, include_disabled)

    def _validate_tenant_access(
        self,
        tenant: TenantConfig,
        include_disabled: bool,
    ) -> TenantConfig:
        """Validate that a tenant can be accessed."""
        if tenant.status == TenantStatus.DISABLED and not include_disabled:
            raise TenantDisabledError(tenant.tenant_id)
        if tenant.status == TenantStatus.SUSPENDED:
            raise TenantSuspendedError(tenant.tenant_id)
        if tenant.status == TenantStatus.DELETED:
            raise TenantNotFoundError(tenant.tenant_id)
        return tenant

    def get_tenant_or_none(
        self,
        tenant_id: str,
        *,
        include_disabled: bool = False,
    ) -> TenantConfig | None:
        """Get a tenant by ID, returning None if not found.

        Args:
            tenant_id: The tenant identifier.
            include_disabled: Whether to include disabled tenants.

        Returns:
            The tenant configuration or None.
        """
        try:
            return self.get_tenant(tenant_id, include_disabled=include_disabled)
        except (TenantNotFoundError, TenantDisabledError, TenantSuspendedError):
            return None

    def update_tenant(
        self,
        tenant_id: str,
        *,
        name: str | None = None,
        tier: TenantTier | None = None,
        isolation_level: IsolationLevel | None = None,
        settings: Mapping[str, Any] | None = None,
        features: frozenset[str] | None = None,
    ) -> TenantConfig:
        """Update a tenant's configuration.

        Args:
            tenant_id: The tenant identifier.
            name: New name (optional).
            tier: New tier (optional).
            isolation_level: New isolation level (optional).
            settings: Settings to update/add (optional).
            features: New features (optional).

        Returns:
            The updated tenant configuration.

        Raises:
            TenantNotFoundError: If the tenant does not exist.
        """
        with self._lock:
            current = self._tenants.get(tenant_id)
            if current is None:
                raise TenantNotFoundError(tenant_id)

            # Build updated config
            updated = current
            if name:
                updated = TenantConfig(
                    tenant_id=updated.tenant_id,
                    name=name,
                    status=updated.status,
                    tier=updated.tier,
                    isolation_level=updated.isolation_level,
                    quotas=updated.quotas,
                    metadata=updated.metadata,
                    settings=updated.settings,
                    features=updated.features,
                    allowed_engines=updated.allowed_engines,
                    default_engine=updated.default_engine,
                )
            if tier:
                updated = updated.with_tier(tier)
            if isolation_level:
                updated = updated.with_isolation_level(isolation_level)
            if settings:
                for key, value in settings.items():
                    updated = updated.with_setting(key, value)
            if features is not None:
                updated = updated.with_features(features)

            # Update metadata
            if updated.metadata:
                updated = TenantConfig(
                    tenant_id=updated.tenant_id,
                    name=updated.name,
                    status=updated.status,
                    tier=updated.tier,
                    isolation_level=updated.isolation_level,
                    quotas=updated.quotas,
                    metadata=updated.metadata.with_update(),
                    settings=updated.settings,
                    features=updated.features,
                    allowed_engines=updated.allowed_engines,
                    default_engine=updated.default_engine,
                )

            self._tenants[tenant_id] = updated
            self._invalidate_cache(tenant_id)

            # Persist
            if self.storage:
                self.storage.save(updated)

            # Notify hooks
            self._notify_hooks("on_tenant_updated", updated, previous=current)

            return updated

    def delete_tenant(
        self,
        tenant_id: str,
        *,
        hard_delete: bool = False,
    ) -> None:
        """Delete a tenant.

        Args:
            tenant_id: The tenant identifier.
            hard_delete: If True, completely remove. If False, soft delete.

        Raises:
            TenantNotFoundError: If the tenant does not exist.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                raise TenantNotFoundError(tenant_id)

            if hard_delete:
                del self._tenants[tenant_id]
                if self.storage:
                    self.storage.delete(tenant_id)
            else:
                # Soft delete - mark as deleted
                updated = tenant.with_status(TenantStatus.DELETED)
                self._tenants[tenant_id] = updated
                if self.storage:
                    self.storage.save(updated)

            self._invalidate_cache(tenant_id)

            # Notify hooks
            self._notify_hooks("on_tenant_deleted", tenant, hard_delete=hard_delete)

    # -------------------------------------------------------------------------
    # Status Operations
    # -------------------------------------------------------------------------

    def activate_tenant(self, tenant_id: str) -> TenantConfig:
        """Activate a tenant."""
        return self._set_status(tenant_id, TenantStatus.ACTIVE)

    def suspend_tenant(
        self,
        tenant_id: str,
        *,
        reason: str | None = None,
    ) -> TenantConfig:
        """Suspend a tenant."""
        tenant = self._set_status(tenant_id, TenantStatus.SUSPENDED)
        self._notify_hooks("on_tenant_suspended", tenant, reason=reason)
        return tenant

    def disable_tenant(self, tenant_id: str) -> TenantConfig:
        """Disable a tenant."""
        return self._set_status(tenant_id, TenantStatus.DISABLED)

    def _set_status(
        self,
        tenant_id: str,
        status: TenantStatus,
    ) -> TenantConfig:
        """Set tenant status."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                raise TenantNotFoundError(tenant_id)

            updated = tenant.with_status(status)
            self._tenants[tenant_id] = updated
            self._invalidate_cache(tenant_id)

            if self.storage:
                self.storage.save(updated)

            self._notify_hooks("on_status_changed", updated, previous_status=tenant.status)

            return updated

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def list_tenants(
        self,
        *,
        status: TenantStatus | Sequence[TenantStatus] | None = None,
        tier: TenantTier | Sequence[TenantTier] | None = None,
        include_deleted: bool = False,
    ) -> list[TenantConfig]:
        """List tenants with optional filtering.

        Args:
            status: Filter by status(es).
            tier: Filter by tier(s).
            include_deleted: Whether to include deleted tenants.

        Returns:
            List of matching tenant configurations.
        """
        with self._lock:
            tenants = list(self._tenants.values())

        # Apply filters
        if status is not None:
            statuses = [status] if isinstance(status, TenantStatus) else list(status)
            tenants = [t for t in tenants if t.status in statuses]

        if tier is not None:
            tiers = [tier] if isinstance(tier, TenantTier) else list(tier)
            tenants = [t for t in tenants if t.tier in tiers]

        if not include_deleted:
            tenants = [t for t in tenants if t.status != TenantStatus.DELETED]

        return tenants

    def count_tenants(
        self,
        *,
        status: TenantStatus | None = None,
        include_deleted: bool = False,
    ) -> int:
        """Count tenants.

        Args:
            status: Filter by status.
            include_deleted: Whether to include deleted tenants.

        Returns:
            Number of matching tenants.
        """
        return len(self.list_tenants(status=status, include_deleted=include_deleted))

    def exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        with self._lock:
            return tenant_id in self._tenants

    def __iter__(self) -> Iterator[TenantConfig]:
        """Iterate over all non-deleted tenants."""
        return iter(self.list_tenants())

    def __len__(self) -> int:
        """Return the number of non-deleted tenants."""
        return self.count_tenants()

    def __contains__(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        return self.exists(tenant_id)

    # -------------------------------------------------------------------------
    # Context Integration
    # -------------------------------------------------------------------------

    def get_context(
        self,
        tenant_id: str,
        *,
        user_id: str | None = None,
        correlation_id: str | None = None,
    ) -> TenantContextManager:
        """Get a context manager for a tenant.

        Args:
            tenant_id: The tenant identifier.
            user_id: Optional user ID.
            correlation_id: Optional correlation ID.

        Returns:
            A context manager for the tenant.
        """
        tenant = self.get_tenant(tenant_id)
        return TenantContextManager(
            tenant_id=tenant_id,
            config=tenant,
            user_id=user_id,
            correlation_id=correlation_id,
        )

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def _get_from_cache(self, tenant_id: str) -> TenantConfig | None:
        """Get a tenant from cache if not expired."""
        cached = self._cache.get(tenant_id)
        if cached is None:
            return None

        tenant, cached_at = cached
        age = (datetime.now() - cached_at).total_seconds()
        if age > self.config.cache_ttl_seconds:
            del self._cache[tenant_id]
            return None

        return tenant

    def _add_to_cache(self, tenant_id: str, tenant: TenantConfig) -> None:
        """Add a tenant to the cache."""
        self._cache[tenant_id] = (tenant, datetime.now())

    def _invalidate_cache(self, tenant_id: str) -> None:
        """Invalidate a tenant's cache entry."""
        self._cache.pop(tenant_id, None)

    def clear_cache(self) -> None:
        """Clear all cached tenant configurations."""
        self._cache.clear()

    # -------------------------------------------------------------------------
    # Hook Management
    # -------------------------------------------------------------------------

    def add_hook(self, hook: TenantHook) -> None:
        """Add a lifecycle hook."""
        self.hooks.append(hook)

    def remove_hook(self, hook: TenantHook) -> None:
        """Remove a lifecycle hook."""
        self.hooks.remove(hook)

    def _notify_hooks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Notify all hooks of an event."""
        for hook in self.hooks:
            handler = getattr(hook, event, None)
            if handler:
                try:
                    handler(*args, **kwargs)
                except Exception:
                    # Hooks should not break the registry
                    pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the registry (for testing).

        Clears all tenants and cache.
        """
        with self._lock:
            self._tenants.clear()
            self._cache.clear()


# =============================================================================
# Singleton Management
# =============================================================================

_registry: TenantRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> TenantRegistry:
    """Get the global tenant registry singleton.

    Returns:
        The global tenant registry.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = TenantRegistry()
    return _registry


def configure_registry(
    config: MultiTenantConfig | None = None,
    storage: TenantStorage | None = None,
    hooks: list[TenantHook] | None = None,
) -> TenantRegistry:
    """Configure the global tenant registry.

    Args:
        config: Multi-tenant configuration.
        storage: Storage backend.
        hooks: Lifecycle hooks.

    Returns:
        The configured registry.
    """
    global _registry
    with _registry_lock:
        _registry = TenantRegistry(
            config=config or MultiTenantConfig(),
            storage=storage,
            hooks=hooks or [],
        )
    return _registry


def reset_registry() -> None:
    """Reset the global tenant registry.

    Useful for testing.
    """
    global _registry
    with _registry_lock:
        if _registry:
            _registry.reset()
        _registry = None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_tenant(
    tenant_id: str,
    name: str,
    **kwargs: Any,
) -> TenantConfig:
    """Create a tenant using the global registry."""
    return get_registry().create_tenant(tenant_id, name, **kwargs)


def get_tenant(tenant_id: str, **kwargs: Any) -> TenantConfig:
    """Get a tenant using the global registry."""
    return get_registry().get_tenant(tenant_id, **kwargs)


def list_tenants(**kwargs: Any) -> list[TenantConfig]:
    """List tenants using the global registry."""
    return get_registry().list_tenants(**kwargs)


def tenant_exists(tenant_id: str) -> bool:
    """Check if a tenant exists using the global registry."""
    return get_registry().exists(tenant_id)
