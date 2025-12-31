"""GCP Secret Manager secret provider.

This module provides a secret provider that integrates with
Google Cloud Secret Manager.

Requires: pip install google-cloud-secret-manager

Example:
    >>> from packages.enterprise.secrets.backends import GCPSecretManagerProvider
    >>> from packages.enterprise.secrets import GCPSecretManagerConfig
    >>>
    >>> config = GCPSecretManagerConfig(
    ...     project_id="my-project",
    ... )
    >>> provider = GCPSecretManagerProvider(config)
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

    from ..config import GCPSecretManagerConfig


class GCPSecretManagerProvider(HealthCheckable):
    """GCP Secret Manager secret provider.

    Supports versioned secrets with automatic version management.

    Example:
        >>> provider = GCPSecretManagerProvider(config)
        >>> provider.set("db-password", "secret123")
        >>> secret = provider.get("db-password")
        >>> secret = provider.get("db-password", version="1")  # Specific version
    """

    def __init__(self, config: GCPSecretManagerConfig) -> None:
        """Initialize the provider.

        Args:
            config: GCP Secret Manager configuration.

        Raises:
            ImportError: If google-cloud-secret-manager is not installed.
        """
        try:
            from google.api_core import exceptions as gcp_exceptions
            from google.cloud import secretmanager
        except ImportError as e:
            raise ImportError(
                "google-cloud-secret-manager package required for GCPSecretManagerProvider. "
                "Install with: pip install google-cloud-secret-manager"
            ) from e

        self._config = config
        self._secretmanager = secretmanager
        self._gcp_exceptions = gcp_exceptions
        self._client = None

    def _get_client(self):
        """Get or create the Secret Manager client.

        Returns:
            Secret Manager client.
        """
        if self._client is not None:
            return self._client

        # Create client with optional credentials
        if self._config.credentials_path:
            self._client = self._secretmanager.SecretManagerServiceClient.from_service_account_json(
                self._config.credentials_path
            )
        else:
            # Use default credentials (ADC)
            self._client = self._secretmanager.SecretManagerServiceClient()

        return self._client

    def _secret_path(self, path: str) -> str:
        """Build the secret resource path.

        Args:
            path: Secret name.

        Returns:
            Full resource path.
        """
        # GCP secret names can't have /, so replace with -
        secret_id = path.replace("/", "-")
        if self._config.prefix:
            secret_id = f"{self._config.prefix}-{secret_id}"

        return f"projects/{self._config.project_id}/secrets/{secret_id}"

    def _version_path(self, path: str, version: str = "latest") -> str:
        """Build the secret version resource path.

        Args:
            path: Secret name.
            version: Version number or 'latest'.

        Returns:
            Full version resource path.
        """
        return f"{self._secret_path(path)}/versions/{version}"

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a secret from GCP Secret Manager.

        Args:
            path: Secret name.
            version: Optional version number (default: latest).

        Returns:
            Secret value or None.
        """
        try:
            client = self._get_client()
            version_name = self._version_path(path, version or "latest")

            response = client.access_secret_version(
                request={"name": version_name}
            )

            # Decode the payload
            payload = response.payload.data.decode("utf-8")

            # Extract version number from name
            # Format: projects/.../secrets/.../versions/N
            version_num = response.name.split("/")[-1]

            return SecretValue(
                value=payload,
                version=version_num,
                created_at=response.create_time if hasattr(response, "create_time") else datetime.now(timezone.utc),
                secret_type=SecretType.STRING,
                metadata={
                    "name": response.name,
                    "state": response.state.name if hasattr(response.state, "name") else str(response.state),
                },
            )

        except self._gcp_exceptions.NotFound:
            return None
        except self._gcp_exceptions.PermissionDenied as e:
            raise SecretAccessDeniedError(
                f"Access denied to secret: {path}",
                path=path,
            )
        except self._gcp_exceptions.Unauthenticated as e:
            raise SecretAuthenticationError(
                f"Authentication failed: {e}",
                backend="gcp_secret_manager",
                auth_method="service_account",
            )
        except self._gcp_exceptions.GoogleAPIError as e:
            raise SecretBackendError(
                f"GCP Secret Manager error: {e}",
                backend="gcp_secret_manager",
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
        """Store a secret in GCP Secret Manager.

        Args:
            path: Secret name.
            value: Secret value.
            secret_type: Type of secret.
            expires_at: Ignored (GCP handles TTL via policies).
            metadata: Labels to apply to the secret.

        Returns:
            The stored secret.
        """
        try:
            client = self._get_client()
            secret_path = self._secret_path(path)
            parent = f"projects/{self._config.project_id}"

            # Extract secret ID from path
            secret_id = secret_path.split("/")[-1]

            # Encode value
            if isinstance(value, str):
                payload = value.encode("utf-8")
            else:
                payload = value

            # Check if secret exists
            try:
                client.get_secret(request={"name": secret_path})
                exists = True
            except self._gcp_exceptions.NotFound:
                exists = False

            if not exists:
                # Create the secret
                create_request: dict[str, Any] = {
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {
                        "replication": {
                            "automatic": {},
                        },
                    },
                }

                # Add labels if provided
                if metadata and "labels" in metadata:
                    create_request["secret"]["labels"] = metadata["labels"]

                client.create_secret(request=create_request)

            # Add the secret version
            version_response = client.add_secret_version(
                request={
                    "parent": secret_path,
                    "payload": {"data": payload},
                }
            )

            version_num = version_response.name.split("/")[-1]

            return SecretValue(
                value=value if isinstance(value, str) else value.decode("utf-8"),
                version=version_num,
                created_at=datetime.now(timezone.utc),
                secret_type=secret_type,
                metadata={"name": version_response.name},
            )

        except self._gcp_exceptions.PermissionDenied as e:
            raise SecretAccessDeniedError(
                f"Access denied to set secret: {path}",
                path=path,
            )
        except self._gcp_exceptions.GoogleAPIError as e:
            raise SecretBackendError(
                f"Failed to store secret: {e}",
                backend="gcp_secret_manager",
                path=path,
            )

    def delete(self, path: str) -> bool:
        """Delete a secret from GCP Secret Manager.

        Args:
            path: Secret name.

        Returns:
            True if deleted.
        """
        try:
            client = self._get_client()
            secret_path = self._secret_path(path)

            client.delete_secret(request={"name": secret_path})
            return True

        except self._gcp_exceptions.NotFound:
            return False
        except self._gcp_exceptions.GoogleAPIError as e:
            raise SecretBackendError(
                f"Failed to delete secret: {e}",
                backend="gcp_secret_manager",
                path=path,
            )

    def exists(self, path: str) -> bool:
        """Check if a secret exists.

        Args:
            path: Secret name.

        Returns:
            True if exists.
        """
        try:
            client = self._get_client()
            secret_path = self._secret_path(path)
            client.get_secret(request={"name": secret_path})
            return True
        except self._gcp_exceptions.NotFound:
            return False

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List secrets matching a prefix.

        Args:
            prefix: Name prefix (will be matched against secret names).
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of secret metadata.
        """
        try:
            client = self._get_client()
            parent = f"projects/{self._config.project_id}"

            results: list[SecretMetadata] = []
            full_prefix = f"{self._config.prefix}-{prefix}" if self._config.prefix else prefix

            # List all secrets and filter
            for secret in client.list_secrets(request={"parent": parent}):
                # Extract secret name
                secret_id = secret.name.split("/")[-1]

                # Filter by prefix
                if full_prefix and not secret_id.startswith(full_prefix):
                    continue

                # Strip prefix for path
                if self._config.prefix and secret_id.startswith(f"{self._config.prefix}-"):
                    path = secret_id[len(self._config.prefix) + 1:]
                else:
                    path = secret_id

                # Convert - back to / for path
                path = path.replace("-", "/")

                results.append(
                    SecretMetadata(
                        path=path,
                        version="latest",
                        created_at=secret.create_time if hasattr(secret, "create_time") else None,
                        secret_type=SecretType.STRING,
                        metadata={
                            "name": secret.name,
                            "labels": dict(secret.labels) if secret.labels else {},
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

        except self._gcp_exceptions.GoogleAPIError as e:
            raise SecretBackendError(
                f"Failed to list secrets: {e}",
                backend="gcp_secret_manager",
            )

    def get_versions(self, path: str) -> Sequence[SecretMetadata]:
        """List all versions of a secret.

        Args:
            path: Secret name.

        Returns:
            List of version metadata.
        """
        try:
            client = self._get_client()
            secret_path = self._secret_path(path)

            versions: list[SecretMetadata] = []
            for version in client.list_secret_versions(request={"parent": secret_path}):
                version_num = version.name.split("/")[-1]

                versions.append(
                    SecretMetadata(
                        path=path,
                        version=version_num,
                        created_at=version.create_time if hasattr(version, "create_time") else None,
                        secret_type=SecretType.STRING,
                        metadata={
                            "name": version.name,
                            "state": version.state.name if hasattr(version.state, "name") else str(version.state),
                        },
                    )
                )

            return versions

        except self._gcp_exceptions.NotFound:
            return []
        except self._gcp_exceptions.GoogleAPIError as e:
            raise SecretBackendError(
                f"Failed to list versions: {e}",
                backend="gcp_secret_manager",
                path=path,
            )

    def disable_version(self, path: str, version: str) -> bool:
        """Disable a secret version.

        Args:
            path: Secret name.
            version: Version number to disable.

        Returns:
            True if disabled.
        """
        try:
            client = self._get_client()
            version_name = self._version_path(path, version)

            client.disable_secret_version(request={"name": version_name})
            return True

        except self._gcp_exceptions.NotFound:
            return False
        except self._gcp_exceptions.GoogleAPIError as e:
            raise SecretBackendError(
                f"Failed to disable version: {e}",
                backend="gcp_secret_manager",
                path=path,
            )

    def destroy_version(self, path: str, version: str) -> bool:
        """Destroy a secret version (irreversible).

        Args:
            path: Secret name.
            version: Version number to destroy.

        Returns:
            True if destroyed.
        """
        try:
            client = self._get_client()
            version_name = self._version_path(path, version)

            client.destroy_secret_version(request={"name": version_name})
            return True

        except self._gcp_exceptions.NotFound:
            return False
        except self._gcp_exceptions.GoogleAPIError as e:
            raise SecretBackendError(
                f"Failed to destroy version: {e}",
                backend="gcp_secret_manager",
                path=path,
            )

    def health_check(self) -> HealthCheckResult:
        """Check GCP Secret Manager health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            client = self._get_client()
            parent = f"projects/{self._config.project_id}"

            # List secrets with minimal results as a health check
            list(client.list_secrets(request={"parent": parent, "page_size": 1}))
            duration = (time.perf_counter() - start) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="GCP Secret Manager is healthy",
                details={
                    "project_id": self._config.project_id,
                },
                latency_ms=duration,
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
