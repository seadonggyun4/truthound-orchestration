"""Pytest fixtures for multi-tenant tests."""

from __future__ import annotations

import pytest

from ..config import MultiTenantConfig, TenantConfig, TESTING_CONFIG
from ..context import clear_tenant_context
from ..hooks import LoggingTenantHook, MetricsTenantHook
from ..registry import TenantRegistry, reset_registry
from ..storage import InMemoryTenantStorage, InMemoryTenantDataStorage
from ..types import IsolationLevel, TenantStatus, TenantTier


@pytest.fixture
def multi_tenant_config() -> MultiTenantConfig:
    """Create a test configuration."""
    return TESTING_CONFIG


@pytest.fixture
def tenant_storage() -> InMemoryTenantStorage:
    """Create an in-memory storage backend."""
    return InMemoryTenantStorage()


@pytest.fixture
def tenant_data_storage() -> InMemoryTenantDataStorage:
    """Create an in-memory data storage backend."""
    return InMemoryTenantDataStorage()


@pytest.fixture
def tenant_registry(
    multi_tenant_config: MultiTenantConfig,
    tenant_storage: InMemoryTenantStorage,
) -> TenantRegistry:
    """Create a tenant registry for testing."""
    return TenantRegistry(
        config=multi_tenant_config,
        storage=tenant_storage,
    )


@pytest.fixture
def sample_tenant_config() -> TenantConfig:
    """Create a sample tenant configuration."""
    return TenantConfig(
        tenant_id="test-tenant",
        name="Test Tenant",
        status=TenantStatus.ACTIVE,
        tier=TenantTier.PROFESSIONAL,
        isolation_level=IsolationLevel.LOGICAL,
    )


@pytest.fixture
def inactive_tenant_config() -> TenantConfig:
    """Create an inactive tenant configuration."""
    return TenantConfig(
        tenant_id="inactive-tenant",
        name="Inactive Tenant",
        status=TenantStatus.PENDING,
    )


@pytest.fixture
def suspended_tenant_config() -> TenantConfig:
    """Create a suspended tenant configuration."""
    return TenantConfig(
        tenant_id="suspended-tenant",
        name="Suspended Tenant",
        status=TenantStatus.SUSPENDED,
    )


@pytest.fixture
def logging_hook() -> LoggingTenantHook:
    """Create a logging hook."""
    return LoggingTenantHook()


@pytest.fixture
def metrics_hook() -> MetricsTenantHook:
    """Create a metrics hook."""
    return MetricsTenantHook()


@pytest.fixture(autouse=True)
def cleanup_context():
    """Clean up tenant context after each test."""
    yield
    clear_tenant_context()


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Reset global registry after each test."""
    yield
    reset_registry()
