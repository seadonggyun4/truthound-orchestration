"""Tests for tenant isolation."""

from __future__ import annotations

import pytest

from ..isolation import (
    CompositeIsolationEnforcer,
    DefaultResourceOwnershipValidator,
    IsolationResult,
    IsolationViolation,
    LoggingViolationHandler,
    LogicalIsolationEnforcer,
    MetricsViolationHandler,
    NoopIsolationEnforcer,
    PhysicalIsolationEnforcer,
    RaisingViolationHandler,
    SharedIsolationEnforcer,
    create_isolation_enforcer,
)
from ..isolation.validators import IsolationAccessValidator
from ..exceptions import CrossTenantAccessError
from ..types import IsolationLevel, Permission, ResourceScope, ResourceType


class TestNoopIsolationEnforcer:
    """Tests for NoopIsolationEnforcer."""

    def test_allows_all_access(self) -> None:
        """Test that noop enforcer allows all access."""
        enforcer = NoopIsolationEnforcer()
        assert enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_validates_all_ownership(self) -> None:
        """Test that noop enforcer validates all ownership."""
        enforcer = NoopIsolationEnforcer()
        assert enforcer.validate_resource_ownership(
            "any-tenant",
            "any-resource",
            ResourceType.ENGINE,
        )

    def test_empty_prefix(self) -> None:
        """Test that noop enforcer returns empty prefix."""
        enforcer = NoopIsolationEnforcer()
        assert enforcer.get_tenant_resource_prefix("tenant") == ""


class TestSharedIsolationEnforcer:
    """Tests for SharedIsolationEnforcer."""

    def test_same_tenant_access(self) -> None:
        """Test same tenant access is allowed."""
        enforcer = SharedIsolationEnforcer()
        assert enforcer.check_access(
            "tenant-a",
            "tenant-a",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_cross_tenant_access_allowed_by_default(self) -> None:
        """Test cross-tenant access is allowed by default in shared mode."""
        enforcer = SharedIsolationEnforcer()
        assert enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_explicit_cross_tenant_allowance(self) -> None:
        """Test explicit cross-tenant allowance."""
        enforcer = SharedIsolationEnforcer()
        enforcer.allow_cross_tenant_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
        )
        assert enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_prefix_format(self) -> None:
        """Test resource prefix format."""
        enforcer = SharedIsolationEnforcer()
        prefix = enforcer.get_tenant_resource_prefix("tenant-123")
        assert prefix == "shared:tenant-123:"


class TestLogicalIsolationEnforcer:
    """Tests for LogicalIsolationEnforcer."""

    def test_same_tenant_access(self) -> None:
        """Test same tenant access is allowed."""
        enforcer = LogicalIsolationEnforcer()
        assert enforcer.check_access(
            "tenant-a",
            "tenant-a",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_cross_tenant_access_denied(self) -> None:
        """Test cross-tenant access is denied."""
        enforcer = LogicalIsolationEnforcer()
        assert not enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_global_resource_access(self) -> None:
        """Test global resources can be accessed cross-tenant."""
        enforcer = LogicalIsolationEnforcer()
        assert enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
            context={"resource_scope": ResourceScope.GLOBAL},
        )

    def test_ownership_by_prefix(self) -> None:
        """Test ownership validation by prefix."""
        enforcer = LogicalIsolationEnforcer()
        prefix = enforcer.get_tenant_resource_prefix("tenant-a")
        resource_id = f"{prefix}resource-123"

        assert enforcer.validate_resource_ownership(
            "tenant-a",
            resource_id,
            ResourceType.ENGINE,
        )
        assert not enforcer.validate_resource_ownership(
            "tenant-b",
            resource_id,
            ResourceType.ENGINE,
        )

    def test_registered_ownership(self) -> None:
        """Test registered resource ownership."""
        enforcer = LogicalIsolationEnforcer()
        enforcer.register_resource("tenant-a", "resource-123", ResourceType.ENGINE)

        assert enforcer.validate_resource_ownership(
            "tenant-a",
            "resource-123",
            ResourceType.ENGINE,
        )
        assert not enforcer.validate_resource_ownership(
            "tenant-b",
            "resource-123",
            ResourceType.ENGINE,
        )

    def test_unregister_resource(self) -> None:
        """Test unregistering a resource."""
        enforcer = LogicalIsolationEnforcer()
        enforcer.register_resource("tenant-a", "resource-123", ResourceType.ENGINE)
        enforcer.unregister_resource("resource-123", ResourceType.ENGINE)

        # Falls back to prefix-based check (fails without prefix)
        assert not enforcer.validate_resource_ownership(
            "tenant-a",
            "resource-123",
            ResourceType.ENGINE,
        )

    def test_prefix_format(self) -> None:
        """Test resource prefix format."""
        enforcer = LogicalIsolationEnforcer()
        prefix = enforcer.get_tenant_resource_prefix("tenant-123")
        assert prefix == "tenant:tenant-123:"


class TestPhysicalIsolationEnforcer:
    """Tests for PhysicalIsolationEnforcer."""

    def test_same_tenant_access(self) -> None:
        """Test same tenant access is allowed."""
        enforcer = PhysicalIsolationEnforcer()
        assert enforcer.check_access(
            "tenant-a",
            "tenant-a",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_cross_tenant_access_strictly_denied(self) -> None:
        """Test cross-tenant access is strictly denied."""
        enforcer = PhysicalIsolationEnforcer()
        assert not enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_global_resource_access(self) -> None:
        """Test only global resources can be accessed cross-tenant."""
        enforcer = PhysicalIsolationEnforcer()
        assert enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
            context={"resource_scope": ResourceScope.GLOBAL},
        )

    def test_cryptographic_prefix(self) -> None:
        """Test that prefix is cryptographically derived."""
        enforcer = PhysicalIsolationEnforcer()
        prefix = enforcer.get_tenant_resource_prefix("tenant-123")
        assert prefix.startswith("phy:")
        # Prefix should be consistent
        assert prefix == enforcer.get_tenant_resource_prefix("tenant-123")
        # Different tenants get different prefixes
        assert prefix != enforcer.get_tenant_resource_prefix("tenant-456")


class TestCompositeIsolationEnforcer:
    """Tests for CompositeIsolationEnforcer."""

    def test_uses_default_enforcer(self) -> None:
        """Test using default enforcer."""
        enforcer = CompositeIsolationEnforcer()
        # Default is logical
        assert not enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )

    def test_resource_specific_enforcer(self) -> None:
        """Test resource-specific enforcer."""
        enforcer = CompositeIsolationEnforcer()
        enforcer.set_enforcer_for_resource(
            ResourceType.CONFIG,
            SharedIsolationEnforcer(),
        )

        # Default (logical) - denied
        assert not enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )

        # CONFIG uses shared - allowed
        assert enforcer.check_access(
            "tenant-a",
            "tenant-b",
            ResourceType.CONFIG,
            Permission.READ,
        )


class TestCreateIsolationEnforcer:
    """Tests for create_isolation_enforcer factory."""

    def test_create_shared(self) -> None:
        """Test creating shared enforcer."""
        enforcer = create_isolation_enforcer(IsolationLevel.SHARED)
        assert isinstance(enforcer, SharedIsolationEnforcer)

    def test_create_logical(self) -> None:
        """Test creating logical enforcer."""
        enforcer = create_isolation_enforcer(IsolationLevel.LOGICAL)
        assert isinstance(enforcer, LogicalIsolationEnforcer)

    def test_create_physical(self) -> None:
        """Test creating physical enforcer."""
        enforcer = create_isolation_enforcer(IsolationLevel.PHYSICAL)
        assert isinstance(enforcer, PhysicalIsolationEnforcer)

    def test_create_dedicated(self) -> None:
        """Test creating dedicated enforcer (uses physical)."""
        enforcer = create_isolation_enforcer(IsolationLevel.DEDICATED)
        assert isinstance(enforcer, PhysicalIsolationEnforcer)


class TestIsolationViolation:
    """Tests for IsolationViolation."""

    def test_create_violation(self) -> None:
        """Test creating a violation."""
        violation = IsolationViolation(
            violation_type="cross_tenant_access",
            source_tenant_id="tenant-a",
            target_tenant_id="tenant-b",
            resource_type=ResourceType.ENGINE,
            resource_id="engine-123",
            permission=Permission.READ,
            message="Access denied",
        )
        assert violation.source_tenant_id == "tenant-a"
        assert violation.target_tenant_id == "tenant-b"

    def test_to_dict(self) -> None:
        """Test converting violation to dict."""
        violation = IsolationViolation(
            violation_type="test",
            source_tenant_id="tenant-a",
            target_tenant_id=None,
            resource_type=ResourceType.ENGINE,
            resource_id=None,
            permission=None,
            message="Test",
        )
        data = violation.to_dict()
        assert data["violation_type"] == "test"
        assert data["source_tenant_id"] == "tenant-a"


class TestIsolationResult:
    """Tests for IsolationResult."""

    def test_allowed_result(self) -> None:
        """Test allowed result."""
        result = IsolationResult.allowed("Access granted")
        assert result.is_allowed
        assert result.violation is None
        assert bool(result) is True

    def test_denied_result(self) -> None:
        """Test denied result."""
        violation = IsolationViolation(
            violation_type="test",
            source_tenant_id="a",
            target_tenant_id="b",
            resource_type=ResourceType.ENGINE,
            resource_id=None,
            permission=None,
            message="Denied",
        )
        result = IsolationResult.denied(violation)
        assert not result.is_allowed
        assert result.violation == violation
        assert bool(result) is False


class TestViolationHandlers:
    """Tests for violation handlers."""

    def test_logging_handler(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging violation handler."""
        import logging
        handler = LoggingViolationHandler(log_level=logging.WARNING)
        violation = IsolationViolation(
            violation_type="test",
            source_tenant_id="tenant-a",
            target_tenant_id="tenant-b",
            resource_type=ResourceType.ENGINE,
            resource_id="resource-123",
            permission=Permission.READ,
            message="Test violation",
        )

        with caplog.at_level(logging.WARNING):
            handler.handle(violation)

        assert "Isolation violation" in caplog.text
        assert "tenant-a" in caplog.text

    def test_raising_handler(self) -> None:
        """Test raising violation handler."""
        handler = RaisingViolationHandler()
        violation = IsolationViolation(
            violation_type="cross_tenant_access",
            source_tenant_id="tenant-a",
            target_tenant_id="tenant-b",
            resource_type=ResourceType.ENGINE,
            resource_id=None,
            permission=None,
            message="Cross tenant access",
        )

        with pytest.raises(CrossTenantAccessError):
            handler.handle(violation)

    def test_metrics_handler(self) -> None:
        """Test metrics violation handler."""
        handler = MetricsViolationHandler()
        violation = IsolationViolation(
            violation_type="test",
            source_tenant_id="tenant-a",
            target_tenant_id=None,
            resource_type=ResourceType.ENGINE,
            resource_id=None,
            permission=None,
            message="Test",
        )

        handler.handle(violation)
        handler.handle(violation)

        assert handler.total_violations == 2
        assert handler.violations_by_type["test"] == 2
        assert handler.violations_by_tenant["tenant-a"] == 2


class TestResourceOwnershipValidator:
    """Tests for DefaultResourceOwnershipValidator."""

    def test_register_and_get_owner(self) -> None:
        """Test registering and getting owner."""
        validator = DefaultResourceOwnershipValidator()
        validator.register_owner("resource-1", ResourceType.ENGINE, "tenant-a")

        assert validator.get_owner("resource-1", ResourceType.ENGINE) == "tenant-a"
        assert validator.is_owner("tenant-a", "resource-1", ResourceType.ENGINE)
        assert not validator.is_owner("tenant-b", "resource-1", ResourceType.ENGINE)

    def test_prefix_based_ownership(self) -> None:
        """Test prefix-based ownership detection."""
        validator = DefaultResourceOwnershipValidator()

        # Detects owner from prefix
        assert validator.get_owner(
            "tenant:acme:resource-1",
            ResourceType.ENGINE,
        ) == "acme"

    def test_unregister(self) -> None:
        """Test unregistering ownership."""
        validator = DefaultResourceOwnershipValidator()
        validator.register_owner("resource-1", ResourceType.ENGINE, "tenant-a")
        validator.unregister("resource-1", ResourceType.ENGINE)

        assert validator.get_owner("resource-1", ResourceType.ENGINE) is None


class TestIsolationAccessValidator:
    """Tests for IsolationAccessValidator."""

    def test_validate_access_allowed(self) -> None:
        """Test access validation when allowed."""
        enforcer = LogicalIsolationEnforcer()
        validator = IsolationAccessValidator(enforcer=enforcer)

        result = validator.validate_access(
            "tenant-a",
            "tenant-a",
            ResourceType.ENGINE,
            Permission.READ,
        )
        assert result.is_allowed

    def test_validate_access_denied(self) -> None:
        """Test access validation when denied."""
        enforcer = LogicalIsolationEnforcer()
        metrics_handler = MetricsViolationHandler()
        validator = IsolationAccessValidator(
            enforcer=enforcer,
            violation_handler=metrics_handler,
        )

        result = validator.validate_access(
            "tenant-a",
            "tenant-b",
            ResourceType.ENGINE,
            Permission.READ,
        )
        assert not result.is_allowed
        assert result.violation is not None
        assert metrics_handler.total_violations == 1
