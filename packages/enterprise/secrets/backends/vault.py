"""HashiCorp Vault secret provider.

This module provides a secret provider that integrates with
HashiCorp Vault for secure secret storage.

Requires: pip install hvac

Example:
    >>> from packages.enterprise.secrets.backends import VaultSecretProvider
    >>> from packages.enterprise.secrets import VaultConfig
    >>>
    >>> config = VaultConfig(
    ...     address="https://vault.example.com",
    ...     token_path="/run/secrets/vault-token",
    ... )
    >>> provider = VaultSecretProvider(config)
    >>> secret = provider.get("database/password")
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..base import (
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
    SecretMetadata,
    SecretProvider,
    SecretType,
    SecretValue,
)
from ..exceptions import (
    SecretAuthenticationError,
    SecretBackendError,
    SecretConnectionError,
    SecretNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..config import VaultConfig


class VaultSecretProvider(HealthCheckable):
    """HashiCorp Vault secret provider.

    Supports KV v1 and v2 secrets engines.

    Example:
        >>> provider = VaultSecretProvider(config)
        >>> provider.set("secret/data/db", "password")
        >>> secret = provider.get("secret/data/db")
    """

    def __init__(self, config: VaultConfig) -> None:
        """Initialize the provider.

        Args:
            config: Vault configuration.

        Raises:
            ImportError: If hvac is not installed.
        """
        try:
            import hvac
        except ImportError as e:
            raise ImportError(
                "hvac package required for VaultSecretProvider. "
                "Install with: pip install hvac"
            ) from e

        self._config = config
        self._client: hvac.Client | None = None
        self._hvac = hvac

    def _get_client(self):
        """Get or create the Vault client.

        Returns:
            Vault client.
        """
        if self._client is not None:
            return self._client

        # Load token
        token = self._config.token
        if self._config.token_path:
            token = Path(self._config.token_path).read_text().strip()

        # Create client
        self._client = self._hvac.Client(
            url=self._config.address,
            token=token,
            namespace=self._config.namespace,
            verify=self._config.verify_ssl,
            cert=(
                (self._config.client_cert_path, self._config.client_key_path)
                if self._config.client_cert_path
                else None
            ),
            timeout=self._config.timeout_seconds,
        )

        # Verify authentication
        if not self._client.is_authenticated():
            raise SecretAuthenticationError(
                "Failed to authenticate with Vault",
                backend="vault",
                auth_method="token",
            )

        return self._client

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a secret from Vault.

        Args:
            path: Secret path.
            version: Optional version (KV v2 only).

        Returns:
            Secret value or None.
        """
        try:
            client = self._get_client()

            if self._config.kv_version == 2:
                response = client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self._config.mount_point,
                    version=int(version) if version else None,
                )
                if response is None:
                    return None

                data = response.get("data", {})
                secret_data = data.get("data", {})
                metadata = data.get("metadata", {})

                # Get the secret value (assume "value" key or first key)
                value = secret_data.get("value") or next(iter(secret_data.values()), None)
                if value is None:
                    return None

                return SecretValue(
                    value=value,
                    version=str(metadata.get("version", "1")),
                    created_at=datetime.fromisoformat(
                        metadata["created_time"].replace("Z", "+00:00")
                    ) if "created_time" in metadata else datetime.now(timezone.utc),
                    secret_type=SecretType.STRING,
                    metadata=secret_data,
                )
            else:
                # KV v1
                response = client.secrets.kv.v1.read_secret(
                    path=path,
                    mount_point=self._config.mount_point,
                )
                if response is None:
                    return None

                data = response.get("data", {})
                value = data.get("value") or next(iter(data.values()), None)

                return SecretValue(
                    value=value,
                    version="1",
                    secret_type=SecretType.STRING,
                    metadata=data,
                )

        except self._hvac.exceptions.InvalidPath:
            return None
        except self._hvac.exceptions.Forbidden as e:
            raise SecretAuthenticationError(
                f"Access denied: {e}",
                backend="vault",
            )
        except Exception as e:
            if "Connection" in str(type(e).__name__):
                raise SecretConnectionError(
                    f"Failed to connect to Vault: {e}",
                    backend="vault",
                    endpoint=self._config.address,
                )
            raise SecretBackendError(
                f"Vault error: {e}",
                backend="vault",
                path=path,
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
        """Store a secret in Vault.

        Args:
            path: Secret path.
            value: Secret value.
            secret_type: Type of secret.
            expires_at: Ignored (Vault handles TTL separately).
            metadata: Additional data to store.

        Returns:
            The stored secret.
        """
        try:
            client = self._get_client()

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            secret_data = {"value": value}
            if metadata:
                secret_data.update(metadata)

            if self._config.kv_version == 2:
                response = client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=secret_data,
                    mount_point=self._config.mount_point,
                )
                version = str(response.get("data", {}).get("version", "1"))
            else:
                client.secrets.kv.v1.create_or_update_secret(
                    path=path,
                    secret=secret_data,
                    mount_point=self._config.mount_point,
                )
                version = "1"

            return SecretValue(
                value=value,
                version=version,
                created_at=datetime.now(timezone.utc),
                secret_type=secret_type,
                metadata=secret_data,
            )

        except Exception as e:
            raise SecretBackendError(
                f"Failed to store secret: {e}",
                backend="vault",
                path=path,
            )

    def delete(self, path: str) -> bool:
        """Delete a secret from Vault.

        Args:
            path: Secret path.

        Returns:
            True if deleted.
        """
        try:
            client = self._get_client()

            if self._config.kv_version == 2:
                client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=path,
                    mount_point=self._config.mount_point,
                )
            else:
                client.secrets.kv.v1.delete_secret(
                    path=path,
                    mount_point=self._config.mount_point,
                )

            return True

        except self._hvac.exceptions.InvalidPath:
            return False
        except Exception as e:
            raise SecretBackendError(
                f"Failed to delete secret: {e}",
                backend="vault",
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

            if self._config.kv_version == 2:
                response = client.secrets.kv.v2.list_secrets(
                    path=prefix,
                    mount_point=self._config.mount_point,
                )
            else:
                response = client.secrets.kv.v1.list_secrets(
                    path=prefix,
                    mount_point=self._config.mount_point,
                )

            keys = response.get("data", {}).get("keys", [])
            results = [
                SecretMetadata(
                    path=f"{prefix}/{key}".lstrip("/"),
                    version="1",
                    secret_type=SecretType.STRING,
                )
                for key in keys
            ]

            if offset:
                results = results[offset:]
            if limit:
                results = results[:limit]

            return results

        except self._hvac.exceptions.InvalidPath:
            return []
        except Exception as e:
            raise SecretBackendError(
                f"Failed to list secrets: {e}",
                backend="vault",
            )

    def health_check(self) -> HealthCheckResult:
        """Check Vault health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            client = self._get_client()
            health = client.sys.read_health_status(method="GET")

            duration = (time.perf_counter() - start) * 1000

            if health.get("sealed", True):
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Vault is sealed",
                    latency_ms=duration,
                )

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Vault is healthy",
                details={
                    "version": health.get("version"),
                    "cluster_name": health.get("cluster_name"),
                },
                latency_ms=duration,
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
