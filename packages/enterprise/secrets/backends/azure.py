"""Azure Key Vault secret provider.

This module provides a secret provider that integrates with
Azure Key Vault.

Requires: pip install azure-keyvault-secrets azure-identity

Example:
    >>> from packages.enterprise.secrets.backends import AzureKeyVaultProvider
    >>> from packages.enterprise.secrets import AzureKeyVaultConfig
    >>>
    >>> config = AzureKeyVaultConfig(
    ...     vault_url="https://myvault.vault.azure.net/",
    ... )
    >>> provider = AzureKeyVaultProvider(config)
    >>> secret = provider.get("database-password")
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..base import (
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
    SecretMetadata,
    SecretType,
    SecretValue,
)
from ..exceptions import (
    SecretAccessDeniedError,
    SecretAuthenticationError,
    SecretBackendError,
    SecretConnectionError,
    SecretNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..config import AzureKeyVaultConfig


class AzureKeyVaultProvider(HealthCheckable):
    """Azure Key Vault secret provider.

    Supports versioned secrets with automatic version management.

    Example:
        >>> provider = AzureKeyVaultProvider(config)
        >>> provider.set("db-password", "secret123")
        >>> secret = provider.get("db-password")
        >>> secret = provider.get("db-password", version="abc123")  # Specific version
    """

    def __init__(self, config: AzureKeyVaultConfig) -> None:
        """Initialize the provider.

        Args:
            config: Azure Key Vault configuration.

        Raises:
            ImportError: If azure-keyvault-secrets or azure-identity is not installed.
        """
        try:
            from azure.core.exceptions import (
                ClientAuthenticationError,
                HttpResponseError,
                ResourceNotFoundError,
                ServiceRequestError,
            )
            from azure.identity import (
                ClientSecretCredential,
                DefaultAzureCredential,
                ManagedIdentityCredential,
            )
            from azure.keyvault.secrets import SecretClient
        except ImportError as e:
            raise ImportError(
                "azure-keyvault-secrets and azure-identity packages required for "
                "AzureKeyVaultProvider. Install with: pip install azure-keyvault-secrets azure-identity"
            ) from e

        self._config = config
        self._SecretClient = SecretClient
        self._DefaultAzureCredential = DefaultAzureCredential
        self._ClientSecretCredential = ClientSecretCredential
        self._ManagedIdentityCredential = ManagedIdentityCredential
        self._ResourceNotFoundError = ResourceNotFoundError
        self._HttpResponseError = HttpResponseError
        self._ClientAuthenticationError = ClientAuthenticationError
        self._ServiceRequestError = ServiceRequestError
        self._client = None

    def _get_client(self):
        """Get or create the Key Vault client.

        Returns:
            Key Vault SecretClient.
        """
        if self._client is not None:
            return self._client

        # Determine credential type
        if self._config.client_id and self._config.client_secret and self._config.tenant_id:
            # Service principal authentication
            credential = self._ClientSecretCredential(
                tenant_id=self._config.tenant_id,
                client_id=self._config.client_id,
                client_secret=self._config.client_secret,
            )
        elif self._config.use_managed_identity:
            # Managed identity authentication
            credential = self._ManagedIdentityCredential(
                client_id=self._config.managed_identity_client_id,
            )
        else:
            # Default Azure credential chain
            credential = self._DefaultAzureCredential()

        self._client = self._SecretClient(
            vault_url=self._config.vault_url,
            credential=credential,
        )

        return self._client

    def _secret_name(self, path: str) -> str:
        """Convert path to secret name.

        Azure Key Vault secret names can only contain alphanumeric and dashes.

        Args:
            path: Secret path.

        Returns:
            Valid secret name.
        """
        # Replace / and _ with -
        name = path.replace("/", "-").replace("_", "-")
        if self._config.prefix:
            name = f"{self._config.prefix}-{name}"
        return name

    def _path_from_name(self, name: str) -> str:
        """Convert secret name back to path.

        Args:
            name: Secret name.

        Returns:
            Original path.
        """
        if self._config.prefix and name.startswith(f"{self._config.prefix}-"):
            name = name[len(self._config.prefix) + 1:]
        # Note: We can't distinguish between - that was / and - that was -
        # So we keep - as is
        return name

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a secret from Azure Key Vault.

        Args:
            path: Secret path.
            version: Optional version ID.

        Returns:
            Secret value or None.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            secret = client.get_secret(secret_name, version=version)

            return SecretValue(
                value=secret.value,
                version=secret.properties.version or "unknown",
                created_at=secret.properties.created_on or datetime.now(timezone.utc),
                expires_at=secret.properties.expires_on,
                secret_type=SecretType.STRING,
                metadata={
                    "id": secret.id,
                    "name": secret.name,
                    "content_type": secret.properties.content_type,
                    "enabled": secret.properties.enabled,
                    "tags": dict(secret.properties.tags) if secret.properties.tags else {},
                    "updated_on": secret.properties.updated_on,
                },
            )

        except self._ResourceNotFoundError:
            return None
        except self._ClientAuthenticationError as e:
            raise SecretAuthenticationError(
                f"Authentication failed: {e}",
                backend="azure_key_vault",
                auth_method="azure_ad",
            )
        except self._HttpResponseError as e:
            if e.status_code == 403:
                raise SecretAccessDeniedError(
                    f"Access denied to secret: {path}",
                    path=path,
                )
            raise SecretBackendError(
                f"Azure Key Vault error: {e}",
                backend="azure_key_vault",
                path=path,
            )
        except self._ServiceRequestError as e:
            raise SecretConnectionError(
                f"Failed to connect to Azure Key Vault: {e}",
                backend="azure_key_vault",
                endpoint=self._config.vault_url,
            )

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Store a secret in Azure Key Vault.

        Args:
            path: Secret path.
            value: Secret value.
            secret_type: Type of secret.
            expires_at: Optional expiration time.
            metadata: Additional metadata (tags, content_type).

        Returns:
            The stored secret.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            kwargs: dict[str, Any] = {}

            if expires_at:
                kwargs["expires_on"] = expires_at

            if metadata:
                if "content_type" in metadata:
                    kwargs["content_type"] = metadata["content_type"]
                if "tags" in metadata:
                    kwargs["tags"] = metadata["tags"]
                if "enabled" in metadata:
                    kwargs["enabled"] = metadata["enabled"]

            secret = client.set_secret(secret_name, value, **kwargs)

            return SecretValue(
                value=secret.value,
                version=secret.properties.version or "unknown",
                created_at=secret.properties.created_on or datetime.now(timezone.utc),
                expires_at=secret.properties.expires_on,
                secret_type=secret_type,
                metadata={
                    "id": secret.id,
                    "name": secret.name,
                },
            )

        except self._ClientAuthenticationError as e:
            raise SecretAuthenticationError(
                f"Authentication failed: {e}",
                backend="azure_key_vault",
                auth_method="azure_ad",
            )
        except self._HttpResponseError as e:
            if e.status_code == 403:
                raise SecretAccessDeniedError(
                    f"Access denied to set secret: {path}",
                    path=path,
                )
            raise SecretBackendError(
                f"Failed to store secret: {e}",
                backend="azure_key_vault",
                path=path,
            )

    def delete(self, path: str, *, purge: bool = False) -> bool:
        """Delete a secret from Azure Key Vault.

        Args:
            path: Secret path.
            purge: If True, permanently delete (requires soft-delete disabled or purge permission).

        Returns:
            True if deleted.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            # Start deletion
            poller = client.begin_delete_secret(secret_name)
            poller.wait()  # Wait for deletion to complete

            if purge:
                # Purge the deleted secret
                try:
                    client.purge_deleted_secret(secret_name)
                except self._HttpResponseError:
                    # Purge may not be allowed
                    pass

            return True

        except self._ResourceNotFoundError:
            return False
        except self._HttpResponseError as e:
            raise SecretBackendError(
                f"Failed to delete secret: {e}",
                backend="azure_key_vault",
                path=path,
            )

    def exists(self, path: str) -> bool:
        """Check if a secret exists.

        Args:
            path: Secret path.

        Returns:
            True if exists.
        """
        return self.get(path) is not None

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List secrets matching a prefix.

        Args:
            prefix: Path prefix.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of secret metadata.
        """
        try:
            client = self._get_client()
            full_prefix = self._secret_name(prefix) if prefix else self._config.prefix or ""

            results: list[SecretMetadata] = []

            for secret_props in client.list_properties_of_secrets():
                # Filter by prefix
                if full_prefix and not secret_props.name.startswith(full_prefix):
                    continue

                # Skip disabled secrets
                if not secret_props.enabled:
                    continue

                path = self._path_from_name(secret_props.name)

                results.append(
                    SecretMetadata(
                        path=path,
                        version=secret_props.version or "latest",
                        created_at=secret_props.created_on,
                        expires_at=secret_props.expires_on,
                        secret_type=SecretType.STRING,
                        metadata={
                            "id": secret_props.id,
                            "name": secret_props.name,
                            "content_type": secret_props.content_type,
                            "tags": dict(secret_props.tags) if secret_props.tags else {},
                            "updated_on": secret_props.updated_on,
                        },
                    )
                )

            # Sort by path
            results.sort(key=lambda m: m.path)

            # Apply offset and limit
            if offset:
                results = results[offset:]
            if limit:
                results = results[:limit]

            return results

        except self._HttpResponseError as e:
            raise SecretBackendError(
                f"Failed to list secrets: {e}",
                backend="azure_key_vault",
            )

    def get_versions(self, path: str) -> Sequence[SecretMetadata]:
        """List all versions of a secret.

        Args:
            path: Secret path.

        Returns:
            List of version metadata.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            versions: list[SecretMetadata] = []

            for secret_props in client.list_properties_of_secret_versions(secret_name):
                versions.append(
                    SecretMetadata(
                        path=path,
                        version=secret_props.version or "unknown",
                        created_at=secret_props.created_on,
                        expires_at=secret_props.expires_on,
                        secret_type=SecretType.STRING,
                        metadata={
                            "id": secret_props.id,
                            "enabled": secret_props.enabled,
                            "updated_on": secret_props.updated_on,
                        },
                    )
                )

            return versions

        except self._ResourceNotFoundError:
            return []
        except self._HttpResponseError as e:
            raise SecretBackendError(
                f"Failed to list versions: {e}",
                backend="azure_key_vault",
                path=path,
            )

    def recover_deleted(self, path: str) -> bool:
        """Recover a soft-deleted secret.

        Args:
            path: Secret path.

        Returns:
            True if recovered.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            poller = client.begin_recover_deleted_secret(secret_name)
            poller.wait()
            return True

        except self._ResourceNotFoundError:
            return False
        except self._HttpResponseError as e:
            raise SecretBackendError(
                f"Failed to recover secret: {e}",
                backend="azure_key_vault",
                path=path,
            )

    def backup(self, path: str) -> bytes | None:
        """Backup a secret.

        Args:
            path: Secret path.

        Returns:
            Backup blob or None.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            backup = client.backup_secret(secret_name)
            return backup

        except self._ResourceNotFoundError:
            return None
        except self._HttpResponseError as e:
            raise SecretBackendError(
                f"Failed to backup secret: {e}",
                backend="azure_key_vault",
                path=path,
            )

    def restore(self, backup: bytes) -> SecretMetadata | None:
        """Restore a secret from backup.

        Args:
            backup: Backup blob from backup() method.

        Returns:
            Restored secret metadata or None.
        """
        try:
            client = self._get_client()

            secret_props = client.restore_secret_backup(backup)

            return SecretMetadata(
                path=self._path_from_name(secret_props.name),
                version=secret_props.version or "unknown",
                created_at=secret_props.created_on,
                expires_at=secret_props.expires_on,
                secret_type=SecretType.STRING,
                metadata={"id": secret_props.id},
            )

        except self._HttpResponseError as e:
            raise SecretBackendError(
                f"Failed to restore secret: {e}",
                backend="azure_key_vault",
            )

    def health_check(self) -> HealthCheckResult:
        """Check Azure Key Vault health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            client = self._get_client()

            # List secrets with minimal iteration as a health check
            iterator = client.list_properties_of_secrets()
            # Just try to get the first one
            try:
                next(iter(iterator))
            except StopIteration:
                pass  # Empty vault is fine

            duration = (time.perf_counter() - start) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Azure Key Vault is healthy",
                details={
                    "vault_url": self._config.vault_url,
                },
                latency_ms=duration,
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
