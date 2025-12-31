"""Multi-tenant exception hierarchy.

This module defines a comprehensive exception hierarchy for multi-tenant operations.
All exceptions inherit from MultiTenantError, which itself inherits from
TruthoundIntegrationError for consistency with the rest of the codebase.
"""

from __future__ import annotations

from typing import Any


class MultiTenantError(Exception):
    """Base exception for all multi-tenant operations.

    This is the root exception class for the multi-tenant module.
    All other multi-tenant exceptions inherit from this class.

    Attributes:
        message: Human-readable error message.
        details: Additional context about the error.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# =============================================================================
# Tenant Lifecycle Exceptions
# =============================================================================


class TenantNotFoundError(MultiTenantError):
    """Raised when a requested tenant does not exist.

    Attributes:
        tenant_id: The ID of the tenant that was not found.
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        msg = message or f"Tenant '{tenant_id}' not found"
        super().__init__(msg, details=details)


class TenantAlreadyExistsError(MultiTenantError):
    """Raised when attempting to create a tenant that already exists.

    Attributes:
        tenant_id: The ID of the tenant that already exists.
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        msg = message or f"Tenant '{tenant_id}' already exists"
        super().__init__(msg, details=details)


class TenantDisabledError(MultiTenantError):
    """Raised when attempting to use a disabled tenant.

    Attributes:
        tenant_id: The ID of the disabled tenant.
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        msg = message or f"Tenant '{tenant_id}' is disabled"
        super().__init__(msg, details=details)


class TenantSuspendedError(MultiTenantError):
    """Raised when attempting to use a suspended tenant.

    Attributes:
        tenant_id: The ID of the suspended tenant.
        reason: The reason for suspension (if available).
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        reason: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.reason = reason
        msg = message or f"Tenant '{tenant_id}' is suspended"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, details=details)


# =============================================================================
# Configuration Exceptions
# =============================================================================


class TenantConfigurationError(MultiTenantError):
    """Raised when there is an error in tenant configuration.

    Attributes:
        tenant_id: The ID of the tenant with configuration issues.
        config_key: The specific configuration key that caused the error (if applicable).
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: str | None = None,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.config_key = config_key
        details = details or {}
        if tenant_id:
            details["tenant_id"] = tenant_id
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details)


class TenantConfigValidationError(TenantConfigurationError):
    """Raised when tenant configuration fails validation.

    Attributes:
        validation_errors: List of validation error messages.
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: str | None = None,
        validation_errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.validation_errors = validation_errors or []
        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, tenant_id=tenant_id, details=details)


# =============================================================================
# Isolation Exceptions
# =============================================================================


class TenantIsolationError(MultiTenantError):
    """Raised when tenant isolation is violated.

    This is a critical security exception that indicates a breach
    in tenant isolation boundaries.

    Attributes:
        source_tenant_id: The tenant that initiated the action.
        target_tenant_id: The tenant whose resources were accessed.
        resource_type: The type of resource that was accessed.
        resource_id: The specific resource identifier.
    """

    def __init__(
        self,
        message: str,
        *,
        source_tenant_id: str | None = None,
        target_tenant_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.source_tenant_id = source_tenant_id
        self.target_tenant_id = target_tenant_id
        self.resource_type = resource_type
        self.resource_id = resource_id
        details = details or {}
        if source_tenant_id:
            details["source_tenant_id"] = source_tenant_id
        if target_tenant_id:
            details["target_tenant_id"] = target_tenant_id
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        super().__init__(message, details=details)


class CrossTenantAccessError(TenantIsolationError):
    """Raised when a tenant attempts to access another tenant's resources.

    This is a specific type of isolation violation where one tenant
    explicitly tries to access resources belonging to another tenant.
    """

    def __init__(
        self,
        source_tenant_id: str,
        target_tenant_id: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = message or (
            f"Tenant '{source_tenant_id}' attempted to access "
            f"resources of tenant '{target_tenant_id}'"
        )
        super().__init__(
            msg,
            source_tenant_id=source_tenant_id,
            target_tenant_id=target_tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
        )


# =============================================================================
# Authorization Exceptions
# =============================================================================


class TenantAuthorizationError(MultiTenantError):
    """Raised when a tenant authorization check fails.

    Attributes:
        tenant_id: The tenant that failed authorization.
        action: The action that was denied.
        resource: The resource that was being accessed.
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.action = action
        self.resource = resource
        details = details or {}
        if tenant_id:
            details["tenant_id"] = tenant_id
        if action:
            details["action"] = action
        if resource:
            details["resource"] = resource
        super().__init__(message, details=details)


class TenantPermissionDeniedError(TenantAuthorizationError):
    """Raised when a tenant lacks permission for an action.

    Attributes:
        required_permission: The permission that was required.
    """

    def __init__(
        self,
        tenant_id: str,
        action: str,
        *,
        resource: str | None = None,
        required_permission: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.required_permission = required_permission
        msg = message or f"Tenant '{tenant_id}' lacks permission for action '{action}'"
        if resource:
            msg += f" on resource '{resource}'"
        details = details or {}
        if required_permission:
            details["required_permission"] = required_permission
        super().__init__(
            msg,
            tenant_id=tenant_id,
            action=action,
            resource=resource,
            details=details,
        )


# =============================================================================
# Quota and Resource Exceptions
# =============================================================================


class TenantQuotaError(MultiTenantError):
    """Base exception for tenant quota-related errors.

    Attributes:
        tenant_id: The tenant with the quota issue.
        quota_type: The type of quota (e.g., "api_calls", "storage").
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: str | None = None,
        quota_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.quota_type = quota_type
        details = details or {}
        if tenant_id:
            details["tenant_id"] = tenant_id
        if quota_type:
            details["quota_type"] = quota_type
        super().__init__(message, details=details)


class TenantQuotaExceededError(TenantQuotaError):
    """Raised when a tenant exceeds their quota limits.

    Attributes:
        current_usage: Current usage amount.
        quota_limit: The maximum allowed limit.
    """

    def __init__(
        self,
        tenant_id: str,
        quota_type: str,
        *,
        current_usage: int | float | None = None,
        quota_limit: int | float | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.current_usage = current_usage
        self.quota_limit = quota_limit
        msg = message or f"Tenant '{tenant_id}' exceeded {quota_type} quota"
        if current_usage is not None and quota_limit is not None:
            msg += f" (usage: {current_usage}, limit: {quota_limit})"
        details = details or {}
        if current_usage is not None:
            details["current_usage"] = current_usage
        if quota_limit is not None:
            details["quota_limit"] = quota_limit
        super().__init__(
            msg,
            tenant_id=tenant_id,
            quota_type=quota_type,
            details=details,
        )


class TenantResourceLimitError(TenantQuotaError):
    """Raised when a tenant hits resource limits.

    Attributes:
        resource_type: The type of resource (e.g., "engines", "connections").
        current_count: Current resource count.
        max_count: Maximum allowed count.
    """

    def __init__(
        self,
        tenant_id: str,
        resource_type: str,
        *,
        current_count: int | None = None,
        max_count: int | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.resource_type = resource_type
        self.current_count = current_count
        self.max_count = max_count
        msg = message or f"Tenant '{tenant_id}' reached {resource_type} limit"
        if current_count is not None and max_count is not None:
            msg += f" (current: {current_count}, max: {max_count})"
        details = details or {}
        details["resource_type"] = resource_type
        if current_count is not None:
            details["current_count"] = current_count
        if max_count is not None:
            details["max_count"] = max_count
        super().__init__(
            msg,
            tenant_id=tenant_id,
            quota_type=resource_type,
            details=details,
        )


# =============================================================================
# Context Exceptions
# =============================================================================


class TenantContextError(MultiTenantError):
    """Raised when there is an error with tenant context management.

    Attributes:
        tenant_id: The tenant ID associated with the context error.
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        details = details or {}
        if tenant_id:
            details["tenant_id"] = tenant_id
        super().__init__(message, details=details)


class NoTenantContextError(TenantContextError):
    """Raised when an operation requires a tenant context but none is set."""

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = message or "No tenant context is currently set"
        super().__init__(msg, details=details)


class TenantContextAlreadySetError(TenantContextError):
    """Raised when attempting to set a tenant context when one is already active."""

    def __init__(
        self,
        current_tenant_id: str,
        new_tenant_id: str,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.current_tenant_id = current_tenant_id
        self.new_tenant_id = new_tenant_id
        msg = message or (
            f"Tenant context already set to '{current_tenant_id}', "
            f"cannot switch to '{new_tenant_id}'"
        )
        details = details or {}
        details["current_tenant_id"] = current_tenant_id
        details["new_tenant_id"] = new_tenant_id
        super().__init__(msg, tenant_id=current_tenant_id, details=details)


# =============================================================================
# Storage Exceptions
# =============================================================================


class TenantStorageError(MultiTenantError):
    """Raised when there is an error with tenant storage operations.

    Attributes:
        tenant_id: The tenant ID associated with the storage error.
        operation: The storage operation that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        tenant_id: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.operation = operation
        details = details or {}
        if tenant_id:
            details["tenant_id"] = tenant_id
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details)


class TenantDataNotFoundError(TenantStorageError):
    """Raised when requested tenant data is not found in storage.

    Attributes:
        key: The key that was not found.
    """

    def __init__(
        self,
        tenant_id: str,
        key: str,
        *,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.key = key
        msg = message or f"Data not found for tenant '{tenant_id}': key '{key}'"
        details = details or {}
        details["key"] = key
        super().__init__(msg, tenant_id=tenant_id, operation="get", details=details)


# =============================================================================
# Middleware Exceptions
# =============================================================================


class TenantMiddlewareError(MultiTenantError):
    """Raised when there is an error in tenant middleware processing.

    Attributes:
        middleware_name: The name of the middleware that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        middleware_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.middleware_name = middleware_name
        details = details or {}
        if middleware_name:
            details["middleware_name"] = middleware_name
        super().__init__(message, details=details)


class TenantResolutionError(TenantMiddlewareError):
    """Raised when the tenant cannot be resolved from a request.

    Attributes:
        resolution_strategy: The resolution strategy that was used.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        resolution_strategy: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.resolution_strategy = resolution_strategy
        msg = message or "Could not resolve tenant from request"
        details = details or {}
        if resolution_strategy:
            details["resolution_strategy"] = resolution_strategy
        super().__init__(msg, details=details)
