"""Environment variable secret provider.

This module provides a secret provider that reads secrets from
environment variables.

Example:
    >>> import os
    >>> os.environ["SECRET_DB_PASSWORD"] = "secret123"
    >>>
    >>> from packages.enterprise.secrets.backends import EnvironmentSecretProvider
    >>> provider = EnvironmentSecretProvider(prefix="SECRET_")
    >>> secret = provider.get("DB_PASSWORD")
    >>> print(secret.value)  # "secret123"
"""

from __future__ import annotations

import os
import time
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
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class EnvironmentSecretProvider(HealthCheckable):
    """Secret provider that reads from environment variables.

    Read-only by default. Can optionally set environment variables.

    Attributes:
        prefix: Environment variable prefix.
        strip_prefix: Whether to strip prefix from names.

    Example:
        >>> # With SECRET_DB_PASSWORD=mysecret
        >>> provider = EnvironmentSecretProvider(prefix="SECRET_")
        >>> secret = provider.get("DB_PASSWORD")
    """

    def __init__(
        self,
        prefix: str = "SECRET_",
        strip_prefix: bool = True,
        case_sensitive: bool = True,
        allow_set: bool = False,
    ) -> None:
        """Initialize the provider.

        Args:
            prefix: Prefix to filter environment variables.
            strip_prefix: Strip prefix from secret names.
            case_sensitive: Whether names are case-sensitive.
            allow_set: Allow setting environment variables.
        """
        self._prefix = prefix
        self._strip_prefix = strip_prefix
        self._case_sensitive = case_sensitive
        self._allow_set = allow_set

    def _env_name(self, path: str) -> str:
        """Convert path to environment variable name.

        Args:
            path: Secret path.

        Returns:
            Environment variable name.
        """
        # Replace / with _ and uppercase
        name = path.replace("/", "_").replace("-", "_")
        if not self._case_sensitive:
            name = name.upper()

        if self._strip_prefix:
            return f"{self._prefix}{name}"
        return name

    def _secret_name(self, env_name: str) -> str:
        """Convert environment variable name to path.

        Args:
            env_name: Environment variable name.

        Returns:
            Secret path.
        """
        if self._strip_prefix and env_name.startswith(self._prefix):
            return env_name[len(self._prefix) :]
        return env_name

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a secret from environment.

        Args:
            path: Secret path (will be converted to env var name).
            version: Ignored (env vars don't have versions).

        Returns:
            Secret value or None if not found.
        """
        env_name = self._env_name(path)

        # Handle case insensitivity
        if not self._case_sensitive:
            # Search case-insensitively
            for key, value in os.environ.items():
                if key.upper() == env_name.upper():
                    return SecretValue(
                        value=value,
                        version="env",
                        secret_type=SecretType.STRING,
                        metadata={"env_var": key},
                    )
            return None

        value = os.environ.get(env_name)
        if value is None:
            return None

        return SecretValue(
            value=value,
            version="env",
            secret_type=SecretType.STRING,
            metadata={"env_var": env_name},
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
        """Set an environment variable.

        Args:
            path: Secret path.
            value: Secret value.
            secret_type: Ignored (env vars are strings).
            expires_at: Ignored.
            metadata: Ignored.

        Returns:
            The stored secret.

        Raises:
            PermissionError: If allow_set is False.
        """
        if not self._allow_set:
            raise PermissionError("Setting environment variables is not allowed")

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        env_name = self._env_name(path)
        os.environ[env_name] = value

        return SecretValue(
            value=value,
            version="env",
            created_at=datetime.now(timezone.utc),
            secret_type=SecretType.STRING,
            metadata={"env_var": env_name},
        )

    def delete(self, path: str) -> bool:
        """Delete an environment variable.

        Args:
            path: Secret path.

        Returns:
            True if deleted.

        Raises:
            PermissionError: If allow_set is False.
        """
        if not self._allow_set:
            raise PermissionError("Deleting environment variables is not allowed")

        env_name = self._env_name(path)

        if env_name in os.environ:
            del os.environ[env_name]
            return True
        return False

    def exists(self, path: str) -> bool:
        """Check if an environment variable exists.

        Args:
            path: Secret path.

        Returns:
            True if exists.
        """
        env_name = self._env_name(path)

        if not self._case_sensitive:
            return any(k.upper() == env_name.upper() for k in os.environ)

        return env_name in os.environ

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List environment variables matching prefix.

        Args:
            prefix: Path prefix.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of secret metadata.
        """
        results = []
        search_prefix = self._env_name(prefix) if prefix else self._prefix

        for key in os.environ:
            check_key = key if self._case_sensitive else key.upper()
            check_prefix = search_prefix if self._case_sensitive else search_prefix.upper()

            if check_key.startswith(check_prefix):
                secret_name = self._secret_name(key)
                results.append(
                    SecretMetadata(
                        path=secret_name,
                        version="env",
                        secret_type=SecretType.STRING,
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

    def health_check(self) -> HealthCheckResult:
        """Check provider health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            # Count matching env vars
            count = sum(1 for k in os.environ if k.startswith(self._prefix))
            duration = (time.perf_counter() - start) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Environment provider is healthy",
                details={"secret_count": count, "prefix": self._prefix},
                latency_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=duration,
            )
