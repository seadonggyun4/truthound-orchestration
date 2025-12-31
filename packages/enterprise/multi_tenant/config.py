"""Tenant configuration classes.

This module provides immutable configuration classes for tenant management.
All configurations use frozen dataclasses with builder pattern support
for thread-safe, immutable configuration updates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from .types import (
    IsolationLevel,
    Labels,
    Permission,
    QuotaLimit,
    QuotaLimitsMap,
    QuotaPeriod,
    QuotaType,
    ResourcePermissions,
    ResourceType,
    TenantMetadata,
    TenantStatus,
    TenantTier,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


# =============================================================================
# Quota Presets
# =============================================================================


def _default_free_quotas() -> tuple[QuotaLimit, ...]:
    """Default quotas for free tier."""
    return (
        QuotaLimit(QuotaType.API_CALLS, 1000, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.STORAGE_BYTES, 100 * 1024 * 1024),  # 100 MB
        QuotaLimit(QuotaType.ENGINES, 1, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.RULES, 50, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.EXECUTIONS, 100, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.USERS, 3, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONNECTIONS, 2, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONCURRENT_JOBS, 1, QuotaPeriod.UNLIMITED),
    )


def _default_starter_quotas() -> tuple[QuotaLimit, ...]:
    """Default quotas for starter tier."""
    return (
        QuotaLimit(QuotaType.API_CALLS, 10000, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.STORAGE_BYTES, 1024 * 1024 * 1024),  # 1 GB
        QuotaLimit(QuotaType.ENGINES, 3, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.RULES, 200, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.EXECUTIONS, 1000, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.USERS, 10, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONNECTIONS, 5, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONCURRENT_JOBS, 3, QuotaPeriod.UNLIMITED),
    )


def _default_professional_quotas() -> tuple[QuotaLimit, ...]:
    """Default quotas for professional tier."""
    return (
        QuotaLimit(QuotaType.API_CALLS, 100000, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.STORAGE_BYTES, 10 * 1024 * 1024 * 1024),  # 10 GB
        QuotaLimit(QuotaType.ENGINES, 10, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.RULES, 1000, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.EXECUTIONS, 10000, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.USERS, 50, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONNECTIONS, 20, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONCURRENT_JOBS, 10, QuotaPeriod.UNLIMITED),
    )


def _default_enterprise_quotas() -> tuple[QuotaLimit, ...]:
    """Default quotas for enterprise tier."""
    return (
        QuotaLimit(QuotaType.API_CALLS, 1000000, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.STORAGE_BYTES, 100 * 1024 * 1024 * 1024),  # 100 GB
        QuotaLimit(QuotaType.ENGINES, 100, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.RULES, 10000, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.EXECUTIONS, 100000, QuotaPeriod.DAILY),
        QuotaLimit(QuotaType.USERS, 500, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONNECTIONS, 100, QuotaPeriod.UNLIMITED),
        QuotaLimit(QuotaType.CONCURRENT_JOBS, 50, QuotaPeriod.UNLIMITED),
    )


TIER_QUOTA_DEFAULTS: dict[TenantTier, tuple[QuotaLimit, ...]] = {
    TenantTier.FREE: _default_free_quotas(),
    TenantTier.STARTER: _default_starter_quotas(),
    TenantTier.PROFESSIONAL: _default_professional_quotas(),
    TenantTier.ENTERPRISE: _default_enterprise_quotas(),
    TenantTier.CUSTOM: (),  # No defaults for custom tier
}


# =============================================================================
# Tenant Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class TenantConfig:
    """Configuration for a single tenant.

    Immutable configuration class with builder pattern support.
    All changes return a new instance.

    Attributes:
        tenant_id: Unique identifier for the tenant.
        name: Human-readable name.
        status: Current status of the tenant.
        tier: Subscription tier.
        isolation_level: Level of isolation from other tenants.
        quotas: Quota limits for this tenant.
        metadata: Audit and management metadata.
        settings: Additional tenant-specific settings.
        features: Enabled features for this tenant.
        allowed_engines: List of allowed data quality engines.
        default_engine: Default engine to use.
    """

    tenant_id: str
    name: str
    status: TenantStatus = TenantStatus.PENDING
    tier: TenantTier = TenantTier.FREE
    isolation_level: IsolationLevel = IsolationLevel.LOGICAL
    quotas: tuple[QuotaLimit, ...] = ()
    metadata: TenantMetadata | None = None
    settings: tuple[tuple[str, Any], ...] = ()
    features: frozenset[str] = field(default_factory=frozenset)
    allowed_engines: frozenset[str] = field(
        default_factory=lambda: frozenset(("truthound",))
    )
    default_engine: str = "truthound"

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("Tenant ID cannot be empty")
        if not self.name:
            raise ValueError("Tenant name cannot be empty")

    # -------------------------------------------------------------------------
    # Builder Methods
    # -------------------------------------------------------------------------

    def with_status(self, status: TenantStatus) -> Self:
        """Return a new config with updated status."""
        return TenantConfig(
            tenant_id=self.tenant_id,
            name=self.name,
            status=status,
            tier=self.tier,
            isolation_level=self.isolation_level,
            quotas=self.quotas,
            metadata=self.metadata,
            settings=self.settings,
            features=self.features,
            allowed_engines=self.allowed_engines,
            default_engine=self.default_engine,
        )

    def with_tier(
        self,
        tier: TenantTier,
        *,
        apply_default_quotas: bool = True,
    ) -> Self:
        """Return a new config with updated tier.

        Args:
            tier: The new tier.
            apply_default_quotas: If True, apply default quotas for the tier.
        """
        quotas = self.quotas
        if apply_default_quotas and tier in TIER_QUOTA_DEFAULTS:
            quotas = TIER_QUOTA_DEFAULTS[tier]
        return TenantConfig(
            tenant_id=self.tenant_id,
            name=self.name,
            status=self.status,
            tier=tier,
            isolation_level=self.isolation_level,
            quotas=quotas,
            metadata=self.metadata,
            settings=self.settings,
            features=self.features,
            allowed_engines=self.allowed_engines,
            default_engine=self.default_engine,
        )

    def with_isolation_level(self, level: IsolationLevel) -> Self:
        """Return a new config with updated isolation level."""
        return TenantConfig(
            tenant_id=self.tenant_id,
            name=self.name,
            status=self.status,
            tier=self.tier,
            isolation_level=level,
            quotas=self.quotas,
            metadata=self.metadata,
            settings=self.settings,
            features=self.features,
            allowed_engines=self.allowed_engines,
            default_engine=self.default_engine,
        )

    def with_quotas(self, quotas: tuple[QuotaLimit, ...]) -> Self:
        """Return a new config with updated quotas."""
        return TenantConfig(
            tenant_id=self.tenant_id,
            name=self.name,
            status=self.status,
            tier=self.tier,
            isolation_level=self.isolation_level,
            quotas=quotas,
            metadata=self.metadata,
            settings=self.settings,
            features=self.features,
            allowed_engines=self.allowed_engines,
            default_engine=self.default_engine,
        )

    def with_quota(self, quota: QuotaLimit) -> Self:
        """Return a new config with a quota added/updated."""
        # Replace existing quota of same type or add new one
        new_quotas = tuple(q for q in self.quotas if q.quota_type != quota.quota_type)
        new_quotas = (*new_quotas, quota)
        return self.with_quotas(new_quotas)

    def with_features(self, features: frozenset[str]) -> Self:
        """Return a new config with updated features."""
        return TenantConfig(
            tenant_id=self.tenant_id,
            name=self.name,
            status=self.status,
            tier=self.tier,
            isolation_level=self.isolation_level,
            quotas=self.quotas,
            metadata=self.metadata,
            settings=self.settings,
            features=features,
            allowed_engines=self.allowed_engines,
            default_engine=self.default_engine,
        )

    def with_feature(self, feature: str) -> Self:
        """Return a new config with a feature enabled."""
        return self.with_features(self.features | {feature})

    def without_feature(self, feature: str) -> Self:
        """Return a new config with a feature disabled."""
        return self.with_features(self.features - {feature})

    def with_setting(self, key: str, value: Any) -> Self:
        """Return a new config with a setting added/updated."""
        new_settings = tuple((k, v) for k, v in self.settings if k != key)
        new_settings = (*new_settings, (key, value))
        return TenantConfig(
            tenant_id=self.tenant_id,
            name=self.name,
            status=self.status,
            tier=self.tier,
            isolation_level=self.isolation_level,
            quotas=self.quotas,
            metadata=self.metadata,
            settings=new_settings,
            features=self.features,
            allowed_engines=self.allowed_engines,
            default_engine=self.default_engine,
        )

    def with_engines(
        self,
        allowed_engines: frozenset[str],
        *,
        default_engine: str | None = None,
    ) -> Self:
        """Return a new config with updated engine settings."""
        default = default_engine or self.default_engine
        if default not in allowed_engines:
            default = next(iter(allowed_engines)) if allowed_engines else "truthound"
        return TenantConfig(
            tenant_id=self.tenant_id,
            name=self.name,
            status=self.status,
            tier=self.tier,
            isolation_level=self.isolation_level,
            quotas=self.quotas,
            metadata=self.metadata,
            settings=self.settings,
            features=self.features,
            allowed_engines=allowed_engines,
            default_engine=default,
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key."""
        for k, v in self.settings:
            if k == key:
                return v
        return default

    def get_quota(self, quota_type: QuotaType) -> QuotaLimit | None:
        """Get a quota limit by type."""
        for quota in self.quotas:
            if quota.quota_type == quota_type:
                return quota
        return None

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return feature in self.features

    def is_engine_allowed(self, engine: str) -> bool:
        """Check if an engine is allowed for this tenant."""
        return engine in self.allowed_engines

    @property
    def is_active(self) -> bool:
        """Check if the tenant is active."""
        return self.status == TenantStatus.ACTIVE

    @property
    def is_operational(self) -> bool:
        """Check if the tenant can perform operations."""
        return self.status.is_operational

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status.value,
            "tier": self.tier.value,
            "isolation_level": self.isolation_level.value,
            "quotas": [
                {
                    "quota_type": q.quota_type.value,
                    "limit": q.limit,
                    "period": q.period.value,
                    "warning_threshold": q.warning_threshold,
                }
                for q in self.quotas
            ],
            "settings": dict(self.settings),
            "features": list(self.features),
            "allowed_engines": list(self.allowed_engines),
            "default_engine": self.default_engine,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TenantConfig:
        """Create from dictionary."""
        quotas = tuple(
            QuotaLimit(
                quota_type=QuotaType(q["quota_type"]),
                limit=q["limit"],
                period=QuotaPeriod(q.get("period", "monthly")),
                warning_threshold=q.get("warning_threshold", 0.8),
            )
            for q in data.get("quotas", [])
        )
        settings_dict = data.get("settings", {})
        settings = tuple(settings_dict.items()) if isinstance(settings_dict, dict) else ()
        return cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            status=TenantStatus(data.get("status", "pending")),
            tier=TenantTier(data.get("tier", "free")),
            isolation_level=IsolationLevel(data.get("isolation_level", "logical")),
            quotas=quotas,
            settings=settings,
            features=frozenset(data.get("features", [])),
            allowed_engines=frozenset(data.get("allowed_engines", ["truthound"])),
            default_engine=data.get("default_engine", "truthound"),
        )


# =============================================================================
# Multi-Tenant System Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class MultiTenantConfig:
    """Global configuration for the multi-tenant system.

    Configures system-wide behavior for tenant management.

    Attributes:
        enabled: Whether multi-tenancy is enabled.
        default_isolation_level: Default isolation level for new tenants.
        default_tier: Default tier for new tenants.
        allow_tenant_creation: Whether new tenants can be created.
        require_authentication: Whether authentication is required.
        enable_quota_enforcement: Whether to enforce quotas.
        enable_isolation_validation: Whether to validate isolation.
        default_tenant_id: Fallback tenant ID when none is set.
        tenant_header_name: HTTP header for tenant ID.
        tenant_query_param: Query parameter for tenant ID.
        storage_backend: Storage backend type.
        cache_tenant_configs: Whether to cache tenant configurations.
        cache_ttl_seconds: TTL for cached configurations.
        max_tenants: Maximum number of tenants allowed.
    """

    enabled: bool = True
    default_isolation_level: IsolationLevel = IsolationLevel.LOGICAL
    default_tier: TenantTier = TenantTier.FREE
    allow_tenant_creation: bool = True
    require_authentication: bool = True
    enable_quota_enforcement: bool = True
    enable_isolation_validation: bool = True
    default_tenant_id: str | None = None
    tenant_header_name: str = "X-Tenant-ID"
    tenant_query_param: str = "tenant_id"
    storage_backend: str = "memory"
    cache_tenant_configs: bool = True
    cache_ttl_seconds: int = 300
    max_tenants: int | None = None

    # -------------------------------------------------------------------------
    # Builder Methods
    # -------------------------------------------------------------------------

    def with_enabled(self, enabled: bool) -> Self:
        """Return a new config with updated enabled state."""
        return MultiTenantConfig(
            enabled=enabled,
            default_isolation_level=self.default_isolation_level,
            default_tier=self.default_tier,
            allow_tenant_creation=self.allow_tenant_creation,
            require_authentication=self.require_authentication,
            enable_quota_enforcement=self.enable_quota_enforcement,
            enable_isolation_validation=self.enable_isolation_validation,
            default_tenant_id=self.default_tenant_id,
            tenant_header_name=self.tenant_header_name,
            tenant_query_param=self.tenant_query_param,
            storage_backend=self.storage_backend,
            cache_tenant_configs=self.cache_tenant_configs,
            cache_ttl_seconds=self.cache_ttl_seconds,
            max_tenants=self.max_tenants,
        )

    def with_defaults(
        self,
        *,
        isolation_level: IsolationLevel | None = None,
        tier: TenantTier | None = None,
    ) -> Self:
        """Return a new config with updated defaults."""
        return MultiTenantConfig(
            enabled=self.enabled,
            default_isolation_level=isolation_level or self.default_isolation_level,
            default_tier=tier or self.default_tier,
            allow_tenant_creation=self.allow_tenant_creation,
            require_authentication=self.require_authentication,
            enable_quota_enforcement=self.enable_quota_enforcement,
            enable_isolation_validation=self.enable_isolation_validation,
            default_tenant_id=self.default_tenant_id,
            tenant_header_name=self.tenant_header_name,
            tenant_query_param=self.tenant_query_param,
            storage_backend=self.storage_backend,
            cache_tenant_configs=self.cache_tenant_configs,
            cache_ttl_seconds=self.cache_ttl_seconds,
            max_tenants=self.max_tenants,
        )

    def with_enforcement(
        self,
        *,
        quota_enforcement: bool | None = None,
        isolation_validation: bool | None = None,
    ) -> Self:
        """Return a new config with updated enforcement settings."""
        return MultiTenantConfig(
            enabled=self.enabled,
            default_isolation_level=self.default_isolation_level,
            default_tier=self.default_tier,
            allow_tenant_creation=self.allow_tenant_creation,
            require_authentication=self.require_authentication,
            enable_quota_enforcement=(
                quota_enforcement
                if quota_enforcement is not None
                else self.enable_quota_enforcement
            ),
            enable_isolation_validation=(
                isolation_validation
                if isolation_validation is not None
                else self.enable_isolation_validation
            ),
            default_tenant_id=self.default_tenant_id,
            tenant_header_name=self.tenant_header_name,
            tenant_query_param=self.tenant_query_param,
            storage_backend=self.storage_backend,
            cache_tenant_configs=self.cache_tenant_configs,
            cache_ttl_seconds=self.cache_ttl_seconds,
            max_tenants=self.max_tenants,
        )

    def with_cache(
        self,
        *,
        enabled: bool | None = None,
        ttl_seconds: int | None = None,
    ) -> Self:
        """Return a new config with updated cache settings."""
        return MultiTenantConfig(
            enabled=self.enabled,
            default_isolation_level=self.default_isolation_level,
            default_tier=self.default_tier,
            allow_tenant_creation=self.allow_tenant_creation,
            require_authentication=self.require_authentication,
            enable_quota_enforcement=self.enable_quota_enforcement,
            enable_isolation_validation=self.enable_isolation_validation,
            default_tenant_id=self.default_tenant_id,
            tenant_header_name=self.tenant_header_name,
            tenant_query_param=self.tenant_query_param,
            storage_backend=self.storage_backend,
            cache_tenant_configs=(
                enabled if enabled is not None else self.cache_tenant_configs
            ),
            cache_ttl_seconds=(
                ttl_seconds if ttl_seconds is not None else self.cache_ttl_seconds
            ),
            max_tenants=self.max_tenants,
        )

    def with_storage(self, backend: str) -> Self:
        """Return a new config with updated storage backend."""
        return MultiTenantConfig(
            enabled=self.enabled,
            default_isolation_level=self.default_isolation_level,
            default_tier=self.default_tier,
            allow_tenant_creation=self.allow_tenant_creation,
            require_authentication=self.require_authentication,
            enable_quota_enforcement=self.enable_quota_enforcement,
            enable_isolation_validation=self.enable_isolation_validation,
            default_tenant_id=self.default_tenant_id,
            tenant_header_name=self.tenant_header_name,
            tenant_query_param=self.tenant_query_param,
            storage_backend=backend,
            cache_tenant_configs=self.cache_tenant_configs,
            cache_ttl_seconds=self.cache_ttl_seconds,
            max_tenants=self.max_tenants,
        )

    # -------------------------------------------------------------------------
    # Loading Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_env(cls, prefix: str = "TRUTHOUND_TENANT") -> MultiTenantConfig:
        """Load configuration from environment variables.

        Environment variables:
        - {prefix}_ENABLED: "true" or "false"
        - {prefix}_DEFAULT_ISOLATION_LEVEL: isolation level string
        - {prefix}_DEFAULT_TIER: tier string
        - {prefix}_STORAGE_BACKEND: storage backend name
        - {prefix}_CACHE_TTL_SECONDS: cache TTL in seconds
        - etc.
        """

        def _get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(f"{prefix}_{key}", "").lower()
            if val in ("true", "1", "yes"):
                return True
            if val in ("false", "0", "no"):
                return False
            return default

        def _get_int(key: str, default: int | None) -> int | None:
            val = os.environ.get(f"{prefix}_{key}")
            if val:
                try:
                    return int(val)
                except ValueError:
                    pass
            return default

        def _get_str(key: str, default: str) -> str:
            return os.environ.get(f"{prefix}_{key}", default)

        isolation_str = _get_str("DEFAULT_ISOLATION_LEVEL", "logical")
        tier_str = _get_str("DEFAULT_TIER", "free")

        return cls(
            enabled=_get_bool("ENABLED", True),
            default_isolation_level=IsolationLevel(isolation_str),
            default_tier=TenantTier(tier_str),
            allow_tenant_creation=_get_bool("ALLOW_CREATION", True),
            require_authentication=_get_bool("REQUIRE_AUTH", True),
            enable_quota_enforcement=_get_bool("ENFORCE_QUOTA", True),
            enable_isolation_validation=_get_bool("VALIDATE_ISOLATION", True),
            default_tenant_id=os.environ.get(f"{prefix}_DEFAULT_ID"),
            tenant_header_name=_get_str("HEADER_NAME", "X-Tenant-ID"),
            tenant_query_param=_get_str("QUERY_PARAM", "tenant_id"),
            storage_backend=_get_str("STORAGE_BACKEND", "memory"),
            cache_tenant_configs=_get_bool("CACHE_CONFIGS", True),
            cache_ttl_seconds=_get_int("CACHE_TTL_SECONDS", 300) or 300,
            max_tenants=_get_int("MAX_TENANTS", None),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> MultiTenantConfig:
        """Load configuration from a JSON or YAML file."""
        path = Path(path)
        with path.open() as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    data = yaml.safe_load(f)
                except ImportError as e:
                    raise ImportError(
                        "PyYAML is required to load YAML configuration files. "
                        "Install it with: pip install pyyaml"
                    ) from e
            else:
                data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultiTenantConfig:
        """Create from dictionary."""
        isolation_level = data.get("default_isolation_level", "logical")
        tier = data.get("default_tier", "free")
        return cls(
            enabled=data.get("enabled", True),
            default_isolation_level=IsolationLevel(isolation_level),
            default_tier=TenantTier(tier),
            allow_tenant_creation=data.get("allow_tenant_creation", True),
            require_authentication=data.get("require_authentication", True),
            enable_quota_enforcement=data.get("enable_quota_enforcement", True),
            enable_isolation_validation=data.get("enable_isolation_validation", True),
            default_tenant_id=data.get("default_tenant_id"),
            tenant_header_name=data.get("tenant_header_name", "X-Tenant-ID"),
            tenant_query_param=data.get("tenant_query_param", "tenant_id"),
            storage_backend=data.get("storage_backend", "memory"),
            cache_tenant_configs=data.get("cache_tenant_configs", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 300),
            max_tenants=data.get("max_tenants"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "default_isolation_level": self.default_isolation_level.value,
            "default_tier": self.default_tier.value,
            "allow_tenant_creation": self.allow_tenant_creation,
            "require_authentication": self.require_authentication,
            "enable_quota_enforcement": self.enable_quota_enforcement,
            "enable_isolation_validation": self.enable_isolation_validation,
            "default_tenant_id": self.default_tenant_id,
            "tenant_header_name": self.tenant_header_name,
            "tenant_query_param": self.tenant_query_param,
            "storage_backend": self.storage_backend,
            "cache_tenant_configs": self.cache_tenant_configs,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_tenants": self.max_tenants,
        }


# =============================================================================
# Preset Configurations
# =============================================================================


# Default configuration for development
DEFAULT_CONFIG = MultiTenantConfig()

# Configuration for testing (no enforcement, no authentication)
TESTING_CONFIG = MultiTenantConfig(
    enabled=True,
    require_authentication=False,
    enable_quota_enforcement=False,
    enable_isolation_validation=False,
    storage_backend="memory",
    cache_tenant_configs=False,
)

# Configuration for production
PRODUCTION_CONFIG = MultiTenantConfig(
    enabled=True,
    require_authentication=True,
    enable_quota_enforcement=True,
    enable_isolation_validation=True,
    default_isolation_level=IsolationLevel.LOGICAL,
    storage_backend="database",
    cache_tenant_configs=True,
    cache_ttl_seconds=300,
)

# Configuration for strict security
STRICT_CONFIG = MultiTenantConfig(
    enabled=True,
    require_authentication=True,
    enable_quota_enforcement=True,
    enable_isolation_validation=True,
    default_isolation_level=IsolationLevel.PHYSICAL,
    allow_tenant_creation=False,  # Only admins can create tenants
    cache_tenant_configs=True,
    cache_ttl_seconds=60,  # Shorter cache for security
)

# Configuration for single-tenant mode (multi-tenancy disabled)
SINGLE_TENANT_CONFIG = MultiTenantConfig(
    enabled=False,
    require_authentication=False,
    enable_quota_enforcement=False,
    enable_isolation_validation=False,
    default_tenant_id="default",
)
