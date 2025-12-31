"""Tests for tenant configuration."""

from __future__ import annotations

import pytest

from ..config import (
    MultiTenantConfig,
    TenantConfig,
    TIER_QUOTA_DEFAULTS,
    DEFAULT_CONFIG,
    TESTING_CONFIG,
    PRODUCTION_CONFIG,
)
from ..types import (
    IsolationLevel,
    QuotaLimit,
    QuotaPeriod,
    QuotaType,
    TenantStatus,
    TenantTier,
)


class TestTenantConfig:
    """Tests for TenantConfig."""

    def test_create_basic_config(self) -> None:
        """Test creating a basic tenant configuration."""
        config = TenantConfig(
            tenant_id="test-tenant",
            name="Test Tenant",
        )
        assert config.tenant_id == "test-tenant"
        assert config.name == "Test Tenant"
        assert config.status == TenantStatus.PENDING
        assert config.tier == TenantTier.FREE
        assert config.isolation_level == IsolationLevel.LOGICAL

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        config = TenantConfig(
            tenant_id="test-tenant",
            name="Test Tenant",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            config.name = "New Name"  # type: ignore

    def test_with_status(self) -> None:
        """Test updating status."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        updated = config.with_status(TenantStatus.ACTIVE)
        assert updated.status == TenantStatus.ACTIVE
        assert config.status == TenantStatus.PENDING  # Original unchanged

    def test_with_tier(self) -> None:
        """Test updating tier with default quotas."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        updated = config.with_tier(TenantTier.ENTERPRISE)
        assert updated.tier == TenantTier.ENTERPRISE
        assert len(updated.quotas) > 0  # Default quotas applied

    def test_with_tier_no_default_quotas(self) -> None:
        """Test updating tier without applying default quotas."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        updated = config.with_tier(TenantTier.ENTERPRISE, apply_default_quotas=False)
        assert updated.tier == TenantTier.ENTERPRISE
        assert len(updated.quotas) == 0

    def test_with_quota(self) -> None:
        """Test adding a quota."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        quota = QuotaLimit(QuotaType.API_CALLS, 1000, QuotaPeriod.DAILY)
        updated = config.with_quota(quota)
        assert config.get_quota(QuotaType.API_CALLS) is None
        assert updated.get_quota(QuotaType.API_CALLS) == quota

    def test_with_features(self) -> None:
        """Test updating features."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        updated = config.with_features(frozenset(["feature_a", "feature_b"]))
        assert updated.has_feature("feature_a")
        assert updated.has_feature("feature_b")
        assert not config.has_feature("feature_a")

    def test_with_feature(self) -> None:
        """Test adding a single feature."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        updated = config.with_feature("new_feature")
        assert updated.has_feature("new_feature")

    def test_without_feature(self) -> None:
        """Test removing a feature."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
            features=frozenset(["feature_a", "feature_b"]),
        )
        updated = config.without_feature("feature_a")
        assert not updated.has_feature("feature_a")
        assert updated.has_feature("feature_b")

    def test_with_setting(self) -> None:
        """Test adding/updating settings."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        updated = config.with_setting("key", "value")
        assert updated.get_setting("key") == "value"
        assert config.get_setting("key") is None

    def test_with_engines(self) -> None:
        """Test updating allowed engines."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
        )
        updated = config.with_engines(
            frozenset(["truthound", "great_expectations"]),
            default_engine="great_expectations",
        )
        assert updated.is_engine_allowed("truthound")
        assert updated.is_engine_allowed("great_expectations")
        assert updated.default_engine == "great_expectations"

    def test_is_active(self) -> None:
        """Test is_active property."""
        active = TenantConfig(
            tenant_id="test",
            name="Test",
            status=TenantStatus.ACTIVE,
        )
        pending = TenantConfig(
            tenant_id="test",
            name="Test",
            status=TenantStatus.PENDING,
        )
        assert active.is_active
        assert not pending.is_active

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        config = TenantConfig(
            tenant_id="test",
            name="Test",
            status=TenantStatus.ACTIVE,
            tier=TenantTier.PROFESSIONAL,
            features=frozenset(["feature_a"]),
        ).with_setting("key", "value").with_quota(
            QuotaLimit(QuotaType.API_CALLS, 1000)
        )

        data = config.to_dict()
        restored = TenantConfig.from_dict(data)

        assert restored.tenant_id == config.tenant_id
        assert restored.name == config.name
        assert restored.status == config.status
        assert restored.tier == config.tier
        assert restored.has_feature("feature_a")

    def test_empty_tenant_id_raises(self) -> None:
        """Test that empty tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="Tenant ID cannot be empty"):
            TenantConfig(tenant_id="", name="Test")

    def test_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Tenant name cannot be empty"):
            TenantConfig(tenant_id="test", name="")


class TestMultiTenantConfig:
    """Tests for MultiTenantConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MultiTenantConfig()
        assert config.enabled is True
        assert config.default_isolation_level == IsolationLevel.LOGICAL
        assert config.default_tier == TenantTier.FREE
        assert config.enable_quota_enforcement is True

    def test_testing_config_preset(self) -> None:
        """Test testing configuration preset."""
        assert TESTING_CONFIG.require_authentication is False
        assert TESTING_CONFIG.enable_quota_enforcement is False
        assert TESTING_CONFIG.cache_tenant_configs is False

    def test_production_config_preset(self) -> None:
        """Test production configuration preset."""
        assert PRODUCTION_CONFIG.require_authentication is True
        assert PRODUCTION_CONFIG.enable_quota_enforcement is True
        assert PRODUCTION_CONFIG.storage_backend == "database"

    def test_with_enabled(self) -> None:
        """Test updating enabled state."""
        config = MultiTenantConfig()
        disabled = config.with_enabled(False)
        assert disabled.enabled is False
        assert config.enabled is True

    def test_with_defaults(self) -> None:
        """Test updating defaults."""
        config = MultiTenantConfig()
        updated = config.with_defaults(
            isolation_level=IsolationLevel.PHYSICAL,
            tier=TenantTier.ENTERPRISE,
        )
        assert updated.default_isolation_level == IsolationLevel.PHYSICAL
        assert updated.default_tier == TenantTier.ENTERPRISE

    def test_with_enforcement(self) -> None:
        """Test updating enforcement settings."""
        config = MultiTenantConfig()
        updated = config.with_enforcement(
            quota_enforcement=False,
            isolation_validation=False,
        )
        assert updated.enable_quota_enforcement is False
        assert updated.enable_isolation_validation is False

    def test_with_cache(self) -> None:
        """Test updating cache settings."""
        config = MultiTenantConfig()
        updated = config.with_cache(enabled=False, ttl_seconds=60)
        assert updated.cache_tenant_configs is False
        assert updated.cache_ttl_seconds == 60

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        config = MultiTenantConfig(
            default_isolation_level=IsolationLevel.PHYSICAL,
            max_tenants=100,
        )
        data = config.to_dict()
        restored = MultiTenantConfig.from_dict(data)

        assert restored.default_isolation_level == config.default_isolation_level
        assert restored.max_tenants == config.max_tenants


class TestTierQuotaDefaults:
    """Tests for tier quota defaults."""

    def test_free_tier_quotas(self) -> None:
        """Test free tier has quotas."""
        quotas = TIER_QUOTA_DEFAULTS[TenantTier.FREE]
        assert len(quotas) > 0
        # Check API calls quota
        api_quota = next(
            (q for q in quotas if q.quota_type == QuotaType.API_CALLS), None
        )
        assert api_quota is not None
        assert api_quota.limit == 1000

    def test_enterprise_tier_has_higher_quotas(self) -> None:
        """Test enterprise tier has higher quotas than free."""
        free_quotas = {q.quota_type: q.limit for q in TIER_QUOTA_DEFAULTS[TenantTier.FREE]}
        ent_quotas = {q.quota_type: q.limit for q in TIER_QUOTA_DEFAULTS[TenantTier.ENTERPRISE]}

        for quota_type in free_quotas:
            assert ent_quotas[quota_type] >= free_quotas[quota_type]

    def test_custom_tier_has_no_defaults(self) -> None:
        """Test custom tier has no default quotas."""
        quotas = TIER_QUOTA_DEFAULTS[TenantTier.CUSTOM]
        assert len(quotas) == 0
