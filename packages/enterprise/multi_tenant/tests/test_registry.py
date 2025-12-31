"""Tests for tenant registry."""

from __future__ import annotations

import pytest

from ..config import MultiTenantConfig, TenantConfig
from ..context import TenantContextManager
from ..exceptions import (
    TenantAlreadyExistsError,
    TenantDisabledError,
    TenantNotFoundError,
    TenantSuspendedError,
)
from ..hooks import LoggingTenantHook, MetricsTenantHook
from ..registry import (
    TenantRegistry,
    configure_registry,
    create_tenant,
    get_registry,
    get_tenant,
    list_tenants,
    reset_registry,
    tenant_exists,
)
from ..storage import InMemoryTenantStorage
from ..types import IsolationLevel, TenantStatus, TenantTier


class TestTenantRegistry:
    """Tests for TenantRegistry."""

    def test_create_tenant(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test creating a tenant."""
        tenant = tenant_registry.create_tenant(
            "test-tenant",
            "Test Tenant",
        )
        assert tenant.tenant_id == "test-tenant"
        assert tenant.name == "Test Tenant"
        assert tenant.status == TenantStatus.PENDING

    def test_create_tenant_with_activation(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test creating an activated tenant."""
        tenant = tenant_registry.create_tenant(
            "test-tenant",
            "Test Tenant",
            activate=True,
        )
        assert tenant.status == TenantStatus.ACTIVE

    def test_create_tenant_with_tier(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test creating tenant with specific tier."""
        tenant = tenant_registry.create_tenant(
            "test-tenant",
            "Test Tenant",
            tier=TenantTier.ENTERPRISE,
        )
        assert tenant.tier == TenantTier.ENTERPRISE
        # Should have default quotas for enterprise tier
        assert len(tenant.quotas) > 0

    def test_create_tenant_with_settings(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test creating tenant with settings."""
        tenant = tenant_registry.create_tenant(
            "test-tenant",
            "Test Tenant",
            settings={"key": "value"},
        )
        assert tenant.get_setting("key") == "value"

    def test_create_duplicate_tenant_raises(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test that creating duplicate tenant raises."""
        tenant_registry.create_tenant("test-tenant", "Test 1")
        with pytest.raises(TenantAlreadyExistsError):
            tenant_registry.create_tenant("test-tenant", "Test 2")

    def test_get_tenant(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test getting a tenant."""
        tenant_registry.create_tenant("test-tenant", "Test", activate=True)
        tenant = tenant_registry.get_tenant("test-tenant")
        assert tenant.tenant_id == "test-tenant"

    def test_get_tenant_not_found(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test getting non-existent tenant."""
        with pytest.raises(TenantNotFoundError):
            tenant_registry.get_tenant("non-existent")

    def test_get_disabled_tenant_raises(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test getting disabled tenant raises."""
        tenant_registry.create_tenant("test-tenant", "Test")
        tenant_registry.disable_tenant("test-tenant")
        with pytest.raises(TenantDisabledError):
            tenant_registry.get_tenant("test-tenant")

    def test_get_disabled_tenant_with_include_disabled(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test getting disabled tenant with include_disabled."""
        tenant_registry.create_tenant("test-tenant", "Test")
        tenant_registry.disable_tenant("test-tenant")
        tenant = tenant_registry.get_tenant("test-tenant", include_disabled=True)
        assert tenant.status == TenantStatus.DISABLED

    def test_get_suspended_tenant_raises(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test getting suspended tenant raises."""
        tenant_registry.create_tenant("test-tenant", "Test", activate=True)
        tenant_registry.suspend_tenant("test-tenant")
        with pytest.raises(TenantSuspendedError):
            tenant_registry.get_tenant("test-tenant")

    def test_get_tenant_or_none(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test get_tenant_or_none."""
        assert tenant_registry.get_tenant_or_none("non-existent") is None
        tenant_registry.create_tenant("test-tenant", "Test", activate=True)
        assert tenant_registry.get_tenant_or_none("test-tenant") is not None

    def test_update_tenant(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test updating a tenant."""
        tenant_registry.create_tenant("test-tenant", "Original")
        updated = tenant_registry.update_tenant(
            "test-tenant",
            name="Updated",
            tier=TenantTier.PROFESSIONAL,
        )
        assert updated.name == "Updated"
        assert updated.tier == TenantTier.PROFESSIONAL

    def test_update_tenant_settings(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test updating tenant settings."""
        tenant_registry.create_tenant("test-tenant", "Test")
        updated = tenant_registry.update_tenant(
            "test-tenant",
            settings={"new_key": "new_value"},
        )
        assert updated.get_setting("new_key") == "new_value"

    def test_update_non_existent_tenant_raises(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test updating non-existent tenant raises."""
        with pytest.raises(TenantNotFoundError):
            tenant_registry.update_tenant("non-existent", name="New")

    def test_delete_tenant_soft(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test soft deleting a tenant."""
        tenant_registry.create_tenant("test-tenant", "Test")
        tenant_registry.delete_tenant("test-tenant")
        # Soft deleted - still exists but status is DELETED
        with pytest.raises(TenantNotFoundError):
            tenant_registry.get_tenant("test-tenant")

    def test_delete_tenant_hard(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test hard deleting a tenant."""
        tenant_registry.create_tenant("test-tenant", "Test")
        tenant_registry.delete_tenant("test-tenant", hard_delete=True)
        assert not tenant_registry.exists("test-tenant")

    def test_activate_tenant(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test activating a tenant."""
        tenant_registry.create_tenant("test-tenant", "Test")
        tenant = tenant_registry.activate_tenant("test-tenant")
        assert tenant.status == TenantStatus.ACTIVE

    def test_suspend_tenant(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test suspending a tenant."""
        tenant_registry.create_tenant("test-tenant", "Test", activate=True)
        tenant = tenant_registry.suspend_tenant("test-tenant", reason="Testing")
        assert tenant.status == TenantStatus.SUSPENDED

    def test_disable_tenant(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test disabling a tenant."""
        tenant_registry.create_tenant("test-tenant", "Test", activate=True)
        tenant = tenant_registry.disable_tenant("test-tenant")
        assert tenant.status == TenantStatus.DISABLED

    def test_list_tenants(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test listing tenants."""
        tenant_registry.create_tenant("tenant-1", "Tenant 1", activate=True)
        tenant_registry.create_tenant("tenant-2", "Tenant 2", activate=True)
        tenant_registry.create_tenant("tenant-3", "Tenant 3")

        all_tenants = tenant_registry.list_tenants()
        assert len(all_tenants) == 3

        active_tenants = tenant_registry.list_tenants(status=TenantStatus.ACTIVE)
        assert len(active_tenants) == 2

    def test_list_tenants_by_tier(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test listing tenants by tier."""
        tenant_registry.create_tenant(
            "tenant-1", "Tenant 1", tier=TenantTier.FREE
        )
        tenant_registry.create_tenant(
            "tenant-2", "Tenant 2", tier=TenantTier.ENTERPRISE
        )

        free_tenants = tenant_registry.list_tenants(tier=TenantTier.FREE)
        assert len(free_tenants) == 1

    def test_count_tenants(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test counting tenants."""
        tenant_registry.create_tenant("tenant-1", "T1", activate=True)
        tenant_registry.create_tenant("tenant-2", "T2", activate=True)
        tenant_registry.create_tenant("tenant-3", "T3")

        assert tenant_registry.count_tenants() == 3
        assert tenant_registry.count_tenants(status=TenantStatus.ACTIVE) == 2

    def test_exists(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test checking tenant existence."""
        assert not tenant_registry.exists("test-tenant")
        tenant_registry.create_tenant("test-tenant", "Test")
        assert tenant_registry.exists("test-tenant")

    def test_contains_protocol(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test __contains__ protocol."""
        tenant_registry.create_tenant("test-tenant", "Test")
        assert "test-tenant" in tenant_registry
        assert "non-existent" not in tenant_registry

    def test_len_protocol(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test __len__ protocol."""
        assert len(tenant_registry) == 0
        tenant_registry.create_tenant("tenant-1", "T1")
        tenant_registry.create_tenant("tenant-2", "T2")
        assert len(tenant_registry) == 2

    def test_iter_protocol(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test __iter__ protocol."""
        tenant_registry.create_tenant("tenant-1", "T1")
        tenant_registry.create_tenant("tenant-2", "T2")

        tenant_ids = [t.tenant_id for t in tenant_registry]
        assert "tenant-1" in tenant_ids
        assert "tenant-2" in tenant_ids

    def test_get_context(
        self, tenant_registry: TenantRegistry
    ) -> None:
        """Test getting context manager for tenant."""
        tenant_registry.create_tenant("test-tenant", "Test", activate=True)
        ctx_manager = tenant_registry.get_context("test-tenant")

        with ctx_manager as ctx:
            assert ctx.tenant_id == "test-tenant"
            assert ctx.config is not None

    def test_cache_invalidation(
        self, multi_tenant_config: MultiTenantConfig,
        tenant_storage: InMemoryTenantStorage,
    ) -> None:
        """Test that cache is invalidated on updates."""
        # Enable caching
        config = multi_tenant_config.with_cache(enabled=True, ttl_seconds=300)
        registry = TenantRegistry(config=config, storage=tenant_storage)

        registry.create_tenant("test-tenant", "Test", activate=True)
        original = registry.get_tenant("test-tenant")
        assert original.name == "Test"

        registry.update_tenant("test-tenant", name="Updated")
        updated = registry.get_tenant("test-tenant")
        assert updated.name == "Updated"

    def test_hooks_are_called(
        self, tenant_registry: TenantRegistry,
        metrics_hook: MetricsTenantHook,
    ) -> None:
        """Test that hooks are called on operations."""
        tenant_registry.add_hook(metrics_hook)

        tenant_registry.create_tenant("test-tenant", "Test")
        assert metrics_hook.metrics.tenants_created == 1

        tenant_registry.update_tenant("test-tenant", name="Updated")
        assert metrics_hook.metrics.tenants_updated == 1

        tenant_registry.delete_tenant("test-tenant")
        assert metrics_hook.metrics.tenants_deleted == 1


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry(self) -> None:
        """Test getting global registry."""
        reset_registry()
        registry = get_registry()
        assert registry is not None
        # Same instance returned
        assert get_registry() is registry

    def test_configure_registry(self) -> None:
        """Test configuring global registry."""
        reset_registry()
        config = MultiTenantConfig(max_tenants=100)
        storage = InMemoryTenantStorage()

        registry = configure_registry(config=config, storage=storage)
        assert registry.config.max_tenants == 100
        assert registry.storage is storage

    def test_reset_registry(self) -> None:
        """Test resetting global registry."""
        registry = get_registry()
        registry.create_tenant("test", "Test")
        assert len(registry) == 1

        reset_registry()
        new_registry = get_registry()
        assert len(new_registry) == 0

    def test_convenience_functions(self) -> None:
        """Test convenience functions use global registry."""
        reset_registry()

        tenant = create_tenant("test-tenant", "Test", activate=True)
        assert tenant.tenant_id == "test-tenant"

        retrieved = get_tenant("test-tenant")
        assert retrieved.tenant_id == "test-tenant"

        tenants = list_tenants()
        assert len(tenants) == 1

        assert tenant_exists("test-tenant")
        assert not tenant_exists("non-existent")
