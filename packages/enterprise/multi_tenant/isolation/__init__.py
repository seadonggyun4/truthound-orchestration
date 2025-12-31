"""Tenant isolation module.

This module provides strategies and validators for enforcing
tenant isolation in multi-tenant environments.
"""

from .base import (
    IsolationEnforcer,
    IsolationValidator,
    ResourceOwnershipValidator,
)
from .strategies import (
    CompositeIsolationEnforcer,
    LogicalIsolationEnforcer,
    NoopIsolationEnforcer,
    PhysicalIsolationEnforcer,
    SharedIsolationEnforcer,
    create_isolation_enforcer,
)
from .validators import (
    DefaultResourceOwnershipValidator,
    IsolationResult,
    IsolationViolation,
    IsolationViolationHandler,
    LoggingViolationHandler,
    MetricsViolationHandler,
    RaisingViolationHandler,
)

__all__ = [
    # Base protocols
    "IsolationEnforcer",
    "IsolationValidator",
    "ResourceOwnershipValidator",
    # Strategies
    "NoopIsolationEnforcer",
    "SharedIsolationEnforcer",
    "LogicalIsolationEnforcer",
    "PhysicalIsolationEnforcer",
    "CompositeIsolationEnforcer",
    "create_isolation_enforcer",
    # Validators
    "IsolationResult",
    "IsolationViolation",
    "IsolationViolationHandler",
    "LoggingViolationHandler",
    "MetricsViolationHandler",
    "RaisingViolationHandler",
    "DefaultResourceOwnershipValidator",
]
