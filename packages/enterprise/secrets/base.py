"""Base protocols and types for secret management.

This module defines the core protocols and data types for the secret management system.
All secret providers must implement the SecretProvider or AsyncSecretProvider protocol.

Example:
    >>> from packages.enterprise.secrets import SecretProvider, SecretValue
    >>>
    >>> class MyProvider(SecretProvider):
    ...     def get(self, path: str) -> SecretValue | None:
    ...         return SecretValue(value="secret", version="1")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class SecretType(Enum):
    """Types of secrets that can be stored.

    Attributes:
        STRING: Plain text string secret (default).
        BINARY: Binary data (base64 encoded when serialized).
        JSON: JSON-structured data.
        CERTIFICATE: X.509 certificate.
        KEY_PAIR: Public/private key pair.
        API_KEY: API key or token.
        PASSWORD: Password credential.
        CONNECTION_STRING: Database or service connection string.
    """

    STRING = auto()
    BINARY = auto()
    JSON = auto()
    CERTIFICATE = auto()
    KEY_PAIR = auto()
    API_KEY = auto()
    PASSWORD = auto()
    CONNECTION_STRING = auto()


@dataclass(frozen=True, slots=True)
class SecretValue:
    """Immutable container for a secret value.

    Attributes:
        value: The actual secret value.
        version: Version identifier for the secret.
        created_at: When the secret was created.
        expires_at: When the secret expires (None for non-expiring).
        secret_type: Type classification of the secret.
        metadata: Additional metadata about the secret.

    Example:
        >>> secret = SecretValue(
        ...     value="my-secret",
        ...     version="v1",
        ...     secret_type=SecretType.PASSWORD,
        ... )
        >>> print(secret.is_expired)
        False
    """

    value: str | bytes
    version: str = "1"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    secret_type: SecretType = SecretType.STRING
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the secret has expired.

        Returns:
            True if expires_at is set and is in the past.
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_binary(self) -> bool:
        """Check if the secret value is binary.

        Returns:
            True if value is bytes.
        """
        return isinstance(self.value, bytes)

    def with_metadata(self, **kwargs: Any) -> SecretValue:
        """Create a new SecretValue with additional metadata.

        Args:
            **kwargs: Metadata key-value pairs to add.

        Returns:
            New SecretValue with merged metadata.
        """
        merged = {**self.metadata, **kwargs}
        return SecretValue(
            value=self.value,
            version=self.version,
            created_at=self.created_at,
            expires_at=self.expires_at,
            secret_type=self.secret_type,
            metadata=merged,
        )

    def __repr__(self) -> str:
        """Return safe representation without exposing the value."""
        return (
            f"SecretValue(version={self.version!r}, "
            f"type={self.secret_type.name}, "
            f"expired={self.is_expired})"
        )


@dataclass(frozen=True, slots=True)
class SecretMetadata:
    """Metadata about a secret without the actual value.

    Used for listing and inspecting secrets without retrieving values.

    Attributes:
        path: The secret path/key.
        version: Current version identifier.
        created_at: When the secret was created.
        updated_at: When the secret was last updated.
        expires_at: When the secret expires.
        secret_type: Type classification.
        tags: Tags for categorization.
        description: Human-readable description.
    """

    path: str
    version: str = "1"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    secret_type: SecretType = SecretType.STRING
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str = ""


@dataclass(frozen=True, slots=True)
class SecretVersion:
    """Information about a specific secret version.

    Attributes:
        version: Version identifier.
        created_at: When this version was created.
        is_current: Whether this is the current version.
        is_deprecated: Whether this version is deprecated.
    """

    version: str
    created_at: datetime
    is_current: bool = False
    is_deprecated: bool = False


@runtime_checkable
class SecretProvider(Protocol):
    """Protocol for synchronous secret providers.

    Implementations store and retrieve secrets from various backends
    such as Vault, AWS Secrets Manager, environment variables, etc.

    Example:
        >>> class MyProvider:
        ...     def get(self, path: str, *, version: str | None = None) -> SecretValue | None:
        ...         return SecretValue(value="secret", version="1")
        ...
        ...     def set(
        ...         self,
        ...         path: str,
        ...         value: str | bytes,
        ...         *,
        ...         secret_type: SecretType = SecretType.STRING,
        ...         expires_at: datetime | None = None,
        ...         metadata: Mapping[str, Any] | None = None,
        ...     ) -> SecretValue:
        ...         return SecretValue(value=value, version="1")
    """

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Retrieve a secret by path.

        Args:
            path: The secret path/key.
            version: Optional specific version to retrieve.

        Returns:
            The secret value, or None if not found.
        """
        ...

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Store a secret.

        Args:
            path: The secret path/key.
            value: The secret value to store.
            secret_type: Type classification of the secret.
            expires_at: Optional expiration time.
            metadata: Optional metadata to store with the secret.

        Returns:
            The stored secret value with version info.
        """
        ...

    def delete(self, path: str) -> bool:
        """Delete a secret.

        Args:
            path: The secret path/key.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if a secret exists.

        Args:
            path: The secret path/key.

        Returns:
            True if the secret exists.
        """
        ...

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List secrets matching a prefix.

        Args:
            prefix: Path prefix to filter by.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of secret metadata.
        """
        ...


@runtime_checkable
class AsyncSecretProvider(Protocol):
    """Protocol for asynchronous secret providers.

    Async version of SecretProvider for use with async frameworks.
    """

    async def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Retrieve a secret by path asynchronously."""
        ...

    async def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Store a secret asynchronously."""
        ...

    async def delete(self, path: str) -> bool:
        """Delete a secret asynchronously."""
        ...

    async def exists(self, path: str) -> bool:
        """Check if a secret exists asynchronously."""
        ...

    async def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List secrets matching a prefix asynchronously."""
        ...


@runtime_checkable
class VersionedSecretProvider(Protocol):
    """Protocol for providers supporting version management.

    Extends SecretProvider with version-specific operations.
    """

    def get_version(self, path: str, version: str) -> SecretValue | None:
        """Get a specific version of a secret.

        Args:
            path: The secret path/key.
            version: The version to retrieve.

        Returns:
            The secret value for that version, or None.
        """
        ...

    def list_versions(self, path: str) -> Sequence[SecretVersion]:
        """List all versions of a secret.

        Args:
            path: The secret path/key.

        Returns:
            List of version information.
        """
        ...

    def deprecate_version(self, path: str, version: str) -> bool:
        """Mark a version as deprecated.

        Args:
            path: The secret path/key.
            version: The version to deprecate.

        Returns:
            True if successful.
        """
        ...

    def delete_version(self, path: str, version: str) -> bool:
        """Delete a specific version.

        Args:
            path: The secret path/key.
            version: The version to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for providers supporting health checks.

    Providers implementing this can report their health status.
    """

    def health_check(self) -> HealthCheckResult:
        """Check the health of the provider.

        Returns:
            Health check result with status and details.
        """
        ...


@runtime_checkable
class AsyncHealthCheckable(Protocol):
    """Async protocol for providers supporting health checks."""

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the provider asynchronously."""
        ...


class HealthStatus(Enum):
    """Health status values.

    Attributes:
        HEALTHY: Provider is fully operational.
        DEGRADED: Provider is operational but with issues.
        UNHEALTHY: Provider is not operational.
        UNKNOWN: Health status cannot be determined.
    """

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of a health check operation.

    Attributes:
        status: The health status.
        message: Human-readable status message.
        details: Additional status details.
        latency_ms: Response latency in milliseconds.
        checked_at: When the check was performed.
    """

    status: HealthStatus
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_healthy(self) -> bool:
        """Check if status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @classmethod
    def healthy(cls, message: str = "OK", **details: Any) -> HealthCheckResult:
        """Create a healthy result.

        Args:
            message: Status message.
            **details: Additional details.

        Returns:
            Healthy check result.
        """
        return cls(status=HealthStatus.HEALTHY, message=message, details=details)

    @classmethod
    def unhealthy(cls, message: str, **details: Any) -> HealthCheckResult:
        """Create an unhealthy result.

        Args:
            message: Error message.
            **details: Additional details.

        Returns:
            Unhealthy check result.
        """
        return cls(status=HealthStatus.UNHEALTHY, message=message, details=details)

    @classmethod
    def degraded(cls, message: str, **details: Any) -> HealthCheckResult:
        """Create a degraded result.

        Args:
            message: Status message.
            **details: Additional details.

        Returns:
            Degraded check result.
        """
        return cls(status=HealthStatus.DEGRADED, message=message, details=details)
