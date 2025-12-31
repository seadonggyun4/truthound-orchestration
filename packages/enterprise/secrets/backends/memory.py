"""In-memory secret provider for testing.

This module provides a thread-safe in-memory implementation of
SecretProvider, useful for testing and development.

Example:
    >>> from packages.enterprise.secrets.backends import InMemorySecretProvider
    >>>
    >>> provider = InMemorySecretProvider()
    >>> provider.set("db/password", "secret123")
    >>> secret = provider.get("db/password")
    >>> print(secret.value)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..base import (
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
    SecretMetadata,
    SecretProvider,
    SecretType,
    SecretValue,
    SecretVersion,
    VersionedSecretProvider,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass
class StoredSecret:
    """Internal representation of a stored secret.

    Maintains version history for the secret.
    """

    path: str
    versions: dict[str, SecretValue] = field(default_factory=dict)
    current_version: str = "1"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)


class InMemorySecretProvider(VersionedSecretProvider, HealthCheckable):
    """In-memory secret provider for testing.

    Thread-safe implementation that maintains version history.

    Attributes:
        max_versions: Maximum versions to keep per secret.

    Example:
        >>> provider = InMemorySecretProvider(max_versions=5)
        >>> provider.set("secret", "value1")
        >>> provider.set("secret", "value2")
        >>> versions = provider.list_versions("secret")
        >>> print(len(versions))  # 2
    """

    def __init__(self, max_versions: int = 10) -> None:
        """Initialize the provider.

        Args:
            max_versions: Maximum versions to keep per secret.
        """
        self._secrets: dict[str, StoredSecret] = {}
        self._lock = threading.RLock()
        self._max_versions = max_versions

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a secret by path.

        Args:
            path: Secret path.
            version: Optional specific version.

        Returns:
            Secret value or None if not found.
        """
        with self._lock:
            stored = self._secrets.get(path)
            if stored is None:
                return None

            if version:
                return stored.versions.get(version)

            return stored.versions.get(stored.current_version)

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
            path: Secret path.
            value: Secret value.
            secret_type: Type of secret.
            expires_at: Expiration time.
            metadata: Additional metadata.

        Returns:
            The stored secret with version info.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            if path in self._secrets:
                stored = self._secrets[path]
                # Increment version
                new_version = str(int(stored.current_version) + 1)
                stored.current_version = new_version
                stored.updated_at = now
            else:
                stored = StoredSecret(path=path, created_at=now, updated_at=now)
                new_version = "1"
                stored.current_version = new_version
                self._secrets[path] = stored

            # Create secret value
            secret = SecretValue(
                value=value,
                version=new_version,
                created_at=now,
                expires_at=expires_at,
                secret_type=secret_type,
                metadata=metadata or {},
            )

            stored.versions[new_version] = secret

            # Cleanup old versions
            self._cleanup_versions(stored)

            return secret

    def delete(self, path: str) -> bool:
        """Delete a secret.

        Args:
            path: Secret path.

        Returns:
            True if deleted.
        """
        with self._lock:
            if path in self._secrets:
                del self._secrets[path]
                return True
            return False

    def exists(self, path: str) -> bool:
        """Check if a secret exists.

        Args:
            path: Secret path.

        Returns:
            True if exists.
        """
        with self._lock:
            return path in self._secrets

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
        with self._lock:
            results = []
            for path, stored in self._secrets.items():
                if path.startswith(prefix):
                    current = stored.versions.get(stored.current_version)
                    results.append(
                        SecretMetadata(
                            path=path,
                            version=stored.current_version,
                            created_at=stored.created_at,
                            updated_at=stored.updated_at,
                            expires_at=current.expires_at if current else None,
                            secret_type=current.secret_type if current else SecretType.STRING,
                            tags=stored.tags,
                            description=stored.description,
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

    def get_version(self, path: str, version: str) -> SecretValue | None:
        """Get a specific version.

        Args:
            path: Secret path.
            version: Version to get.

        Returns:
            Secret value or None.
        """
        with self._lock:
            stored = self._secrets.get(path)
            if stored is None:
                return None
            return stored.versions.get(version)

    def list_versions(self, path: str) -> Sequence[SecretVersion]:
        """List all versions of a secret.

        Args:
            path: Secret path.

        Returns:
            List of version info.
        """
        with self._lock:
            stored = self._secrets.get(path)
            if stored is None:
                return []

            return [
                SecretVersion(
                    version=v,
                    created_at=secret.created_at,
                    is_current=(v == stored.current_version),
                    is_deprecated=False,
                )
                for v, secret in sorted(
                    stored.versions.items(),
                    key=lambda x: int(x[0]),
                    reverse=True,
                )
            ]

    def deprecate_version(self, path: str, version: str) -> bool:
        """Mark a version as deprecated.

        Note: This implementation doesn't track deprecation.

        Args:
            path: Secret path.
            version: Version to deprecate.

        Returns:
            True if successful.
        """
        with self._lock:
            stored = self._secrets.get(path)
            if stored is None:
                return False
            return version in stored.versions

    def delete_version(self, path: str, version: str) -> bool:
        """Delete a specific version.

        Args:
            path: Secret path.
            version: Version to delete.

        Returns:
            True if deleted.
        """
        with self._lock:
            stored = self._secrets.get(path)
            if stored is None:
                return False

            if version not in stored.versions:
                return False

            # Don't delete current version if it's the only one
            if version == stored.current_version and len(stored.versions) == 1:
                return False

            del stored.versions[version]

            # Update current version if deleted
            if version == stored.current_version:
                stored.current_version = max(stored.versions.keys(), key=int)

            return True

    def health_check(self) -> HealthCheckResult:
        """Check provider health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            # Simple check: can we access the store?
            with self._lock:
                count = len(self._secrets)

            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="In-memory provider is healthy",
                details={"secret_count": count},
                latency_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=duration,
            )

    def _cleanup_versions(self, stored: StoredSecret) -> None:
        """Remove excess versions.

        Args:
            stored: Stored secret to clean up.
        """
        if len(stored.versions) <= self._max_versions:
            return

        # Sort versions and keep newest
        versions = sorted(stored.versions.keys(), key=int)
        to_delete = versions[: -self._max_versions]

        for v in to_delete:
            if v != stored.current_version:
                del stored.versions[v]

    def clear(self) -> None:
        """Clear all secrets (for testing)."""
        with self._lock:
            self._secrets.clear()

    def __len__(self) -> int:
        """Get number of secrets stored."""
        with self._lock:
            return len(self._secrets)
