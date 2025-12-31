"""Multi-tenant integration for secret management.

This module provides tenant-aware secret providers that automatically
scope secrets to the current tenant context.

Example:
    >>> from packages.enterprise.secrets import TenantAwareSecretProvider
    >>> from packages.enterprise.multi_tenant import TenantContext
    >>>
    >>> provider = TenantAwareSecretProvider(base_provider)
    >>> with TenantContext("tenant-123"):
    ...     secret = provider.get("db/password")  # Gets tenants/tenant-123/db/password
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .base import SecretMetadata, SecretProvider, SecretType, SecretValue
from .exceptions import SecretAccessDeniedError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class TenantAwareSecretProvider:
    """Secret provider that scopes secrets to the current tenant.

    Automatically prefixes paths with the tenant ID from the current
    context, providing tenant isolation.

    Attributes:
        wrapped: The underlying provider.
        prefix_template: Template for path prefixing.

    Example:
        >>> provider = TenantAwareSecretProvider(
        ...     base_provider,
        ...     prefix_template="tenants/{tenant_id}",
        ... )
        >>> with TenantContext("my-tenant"):
        ...     provider.get("db/password")  # Gets tenants/my-tenant/db/password
    """

    def __init__(
        self,
        wrapped: SecretProvider,
        prefix_template: str = "tenants/{tenant_id}",
        separator: str = "/",
        require_tenant: bool = True,
        fallback_to_global: bool = False,
        global_prefix: str = "global",
    ) -> None:
        """Initialize the tenant-aware provider.

        Args:
            wrapped: The underlying provider.
            prefix_template: Template with {tenant_id} placeholder.
            separator: Path separator.
            require_tenant: Require tenant context for operations.
            fallback_to_global: Fall back to global secrets if tenant secret not found.
            global_prefix: Prefix for global secrets.
        """
        self._wrapped = wrapped
        self._prefix_template = prefix_template
        self._separator = separator
        self._require_tenant = require_tenant
        self._fallback_to_global = fallback_to_global
        self._global_prefix = global_prefix

    def _get_tenant_id(self) -> str | None:
        """Get the current tenant ID from context.

        Returns:
            Tenant ID or None if not in tenant context.
        """
        try:
            from packages.enterprise.multi_tenant.context import get_current_tenant_id

            return get_current_tenant_id()
        except ImportError:
            return None

    def _tenant_path(self, path: str, tenant_id: str) -> str:
        """Create a tenant-scoped path.

        Args:
            path: Original path.
            tenant_id: Tenant ID.

        Returns:
            Tenant-scoped path.
        """
        prefix = self._prefix_template.format(tenant_id=tenant_id)
        return f"{prefix}{self._separator}{path}"

    def _global_path(self, path: str) -> str:
        """Create a global path.

        Args:
            path: Original path.

        Returns:
            Global path.
        """
        return f"{self._global_prefix}{self._separator}{path}"

    def _check_tenant(self) -> str:
        """Check and return tenant ID.

        Returns:
            Tenant ID.

        Raises:
            SecretAccessDeniedError: If tenant required but not set.
        """
        tenant_id = self._get_tenant_id()
        if tenant_id is None and self._require_tenant:
            raise SecretAccessDeniedError(
                "Tenant context required for secret access",
                operation="get",
            )
        return tenant_id or ""

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a tenant-scoped secret.

        Args:
            path: Secret path.
            version: Optional version.

        Returns:
            Secret value or None.
        """
        tenant_id = self._check_tenant()

        if tenant_id:
            tenant_path = self._tenant_path(path, tenant_id)
            result = self._wrapped.get(tenant_path, version=version)

            if result is not None:
                return result

            # Fall back to global if enabled
            if self._fallback_to_global:
                global_path = self._global_path(path)
                return self._wrapped.get(global_path, version=version)

            return None

        # No tenant context, try global only if allowed
        if not self._require_tenant:
            global_path = self._global_path(path)
            return self._wrapped.get(global_path, version=version)

        return None

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set a tenant-scoped secret.

        Args:
            path: Secret path.
            value: Secret value.
            secret_type: Secret type.
            expires_at: Expiration time.
            metadata: Additional metadata.

        Returns:
            Stored secret.
        """
        tenant_id = self._check_tenant()

        # Add tenant_id to metadata
        meta = dict(metadata) if metadata else {}
        if tenant_id:
            meta["tenant_id"] = tenant_id
            full_path = self._tenant_path(path, tenant_id)
        else:
            full_path = self._global_path(path)

        return self._wrapped.set(
            full_path,
            value,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=meta,
        )

    def delete(self, path: str) -> bool:
        """Delete a tenant-scoped secret.

        Args:
            path: Secret path.

        Returns:
            True if deleted.
        """
        tenant_id = self._check_tenant()

        if tenant_id:
            full_path = self._tenant_path(path, tenant_id)
        else:
            full_path = self._global_path(path)

        return self._wrapped.delete(full_path)

    def exists(self, path: str) -> bool:
        """Check if a tenant-scoped secret exists.

        Args:
            path: Secret path.

        Returns:
            True if exists.
        """
        tenant_id = self._check_tenant()

        if tenant_id:
            full_path = self._tenant_path(path, tenant_id)
            if self._wrapped.exists(full_path):
                return True
            if self._fallback_to_global:
                return self._wrapped.exists(self._global_path(path))
            return False

        if not self._require_tenant:
            return self._wrapped.exists(self._global_path(path))

        return False

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List tenant-scoped secrets.

        Args:
            prefix: Path prefix.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of secret metadata.
        """
        tenant_id = self._check_tenant()

        if tenant_id:
            full_prefix = self._tenant_path(prefix, tenant_id)
        else:
            full_prefix = self._global_path(prefix)

        results = self._wrapped.list(full_prefix, limit=limit, offset=offset)

        # Strip tenant prefix from paths
        prefix_to_strip = (
            self._tenant_path("", tenant_id) if tenant_id else self._global_path("")
        )

        return [
            SecretMetadata(
                path=m.path[len(prefix_to_strip) :] if m.path.startswith(prefix_to_strip) else m.path,
                version=m.version,
                created_at=m.created_at,
                updated_at=m.updated_at,
                expires_at=m.expires_at,
                secret_type=m.secret_type,
                tags=m.tags,
                description=m.description,
            )
            for m in results
        ]

    def get_global(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a global (non-tenant) secret.

        Args:
            path: Secret path.
            version: Optional version.

        Returns:
            Secret value or None.
        """
        global_path = self._global_path(path)
        return self._wrapped.get(global_path, version=version)

    def set_global(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set a global (non-tenant) secret.

        Args:
            path: Secret path.
            value: Secret value.
            secret_type: Secret type.
            expires_at: Expiration time.
            metadata: Additional metadata.

        Returns:
            Stored secret.
        """
        global_path = self._global_path(path)
        meta = dict(metadata) if metadata else {}
        meta["global"] = True

        return self._wrapped.set(
            global_path,
            value,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=meta,
        )


class TenantSecretIsolator:
    """Ensures strict tenant isolation for secrets.

    Validates that secrets are only accessed within proper tenant
    context and provides audit trail.

    Example:
        >>> isolator = TenantSecretIsolator(provider)
        >>> with TenantContext("tenant-1"):
        ...     isolator.get("secret")  # OK
        ...     isolator.get("../other-tenant/secret")  # Raises error
    """

    def __init__(
        self,
        wrapped: SecretProvider,
        allowed_paths: frozenset[str] | None = None,
        denied_paths: frozenset[str] | None = None,
    ) -> None:
        """Initialize the isolator.

        Args:
            wrapped: Provider to wrap.
            allowed_paths: Whitelist of allowed path patterns.
            denied_paths: Blacklist of denied path patterns.
        """
        self._wrapped = TenantAwareSecretProvider(
            wrapped,
            require_tenant=True,
            fallback_to_global=False,
        )
        self._allowed_paths = allowed_paths
        self._denied_paths = denied_paths or frozenset()

    def _validate_path(self, path: str) -> None:
        """Validate that path is allowed.

        Args:
            path: Path to validate.

        Raises:
            SecretAccessDeniedError: If path is not allowed.
        """
        # Check for path traversal
        if ".." in path or path.startswith("/"):
            raise SecretAccessDeniedError(
                "Path traversal not allowed",
                path=path,
            )

        # Check denied paths
        for denied in self._denied_paths:
            if path.startswith(denied):
                raise SecretAccessDeniedError(
                    f"Access to path denied: {path}",
                    path=path,
                )

        # Check allowed paths
        if self._allowed_paths is not None:
            allowed = any(path.startswith(p) for p in self._allowed_paths)
            if not allowed:
                raise SecretAccessDeniedError(
                    f"Path not in allowed list: {path}",
                    path=path,
                )

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get with path validation."""
        self._validate_path(path)
        return self._wrapped.get(path, version=version)

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set with path validation."""
        self._validate_path(path)
        return self._wrapped.set(
            path,
            value,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=metadata,
        )

    def delete(self, path: str) -> bool:
        """Delete with path validation."""
        self._validate_path(path)
        return self._wrapped.delete(path)

    def exists(self, path: str) -> bool:
        """Check existence with path validation."""
        self._validate_path(path)
        return self._wrapped.exists(path)

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List with path validation."""
        if prefix:
            self._validate_path(prefix)
        return self._wrapped.list(prefix, limit=limit, offset=offset)


def create_tenant_provider(
    provider: SecretProvider,
    require_tenant: bool = True,
    fallback_to_global: bool = False,
    isolate: bool = False,
    allowed_paths: frozenset[str] | None = None,
) -> SecretProvider:
    """Factory function to create a tenant-aware provider.

    Args:
        provider: Base provider to wrap.
        require_tenant: Require tenant context.
        fallback_to_global: Fall back to global secrets.
        isolate: Use strict isolation.
        allowed_paths: Allowed path patterns (for isolation).

    Returns:
        Tenant-aware provider.

    Example:
        >>> provider = create_tenant_provider(
        ...     base_provider,
        ...     require_tenant=True,
        ...     isolate=True,
        ... )
    """
    if isolate:
        return TenantSecretIsolator(
            provider,
            allowed_paths=allowed_paths,
        )

    return TenantAwareSecretProvider(
        provider,
        require_tenant=require_tenant,
        fallback_to_global=fallback_to_global,
    )
