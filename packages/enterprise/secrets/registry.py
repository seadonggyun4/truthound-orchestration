"""Thread-safe singleton registry for secret providers.

This module provides a centralized registry for managing secret providers.
The registry supports plugin discovery via entry points.

Example:
    >>> from packages.enterprise.secrets import (
    ...     get_secret_registry,
    ...     get_secret,
    ...     set_secret,
    ... )
    >>>
    >>> # Register a provider
    >>> registry = get_secret_registry()
    >>> registry.register("vault", VaultSecretProvider(config))
    >>>
    >>> # Use convenience functions
    >>> secret = get_secret("database/password")
    >>> set_secret("api/key", "new-value")
"""

from __future__ import annotations

import threading
from datetime import datetime
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, ClassVar

from .base import (
    AsyncSecretProvider,
    SecretMetadata,
    SecretProvider,
    SecretType,
    SecretValue,
)
from .exceptions import ProviderNotFoundError, SecretConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .config import SecretConfig


ENTRY_POINT_GROUP = "truthound.secret_providers"
"""Entry point group for discovering secret providers."""


class SecretProviderRegistry:
    """Thread-safe singleton registry for secret providers.

    Manages registration, retrieval, and lifecycle of secret providers.
    Supports plugin discovery via Python entry points.

    Attributes:
        providers: Read-only view of registered providers.
        default_provider_name: Name of the default provider.

    Example:
        >>> registry = SecretProviderRegistry()
        >>> registry.register("vault", VaultSecretProvider(config))
        >>> provider = registry.get("vault")
        >>> registry.set_default("vault")
    """

    _instance: ClassVar[SecretProviderRegistry | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> SecretProviderRegistry:
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._providers: dict[str, SecretProvider] = {}
                    instance._async_providers: dict[str, AsyncSecretProvider] = {}
                    instance._default_name: str | None = None
                    instance._initialized = True
                    cls._instance = instance
        return cls._instance

    @property
    def providers(self) -> Mapping[str, SecretProvider]:
        """Get read-only view of registered providers."""
        return dict(self._providers)

    @property
    def async_providers(self) -> Mapping[str, AsyncSecretProvider]:
        """Get read-only view of registered async providers."""
        return dict(self._async_providers)

    @property
    def default_provider_name(self) -> str | None:
        """Get the name of the default provider."""
        return self._default_name

    def register(
        self,
        name: str,
        provider: SecretProvider,
        *,
        set_default: bool = False,
    ) -> None:
        """Register a secret provider.

        Args:
            name: Unique name for the provider.
            provider: The provider instance.
            set_default: Whether to set this as the default provider.

        Raises:
            SecretConfigurationError: If name is empty.

        Example:
            >>> registry.register("vault", VaultSecretProvider(config))
            >>> registry.register("aws", AWSProvider(config), set_default=True)
        """
        if not name:
            raise SecretConfigurationError("Provider name cannot be empty")

        with self._lock:
            self._providers[name] = provider
            if set_default or self._default_name is None:
                self._default_name = name

    def register_async(
        self,
        name: str,
        provider: AsyncSecretProvider,
        *,
        set_default: bool = False,
    ) -> None:
        """Register an async secret provider.

        Args:
            name: Unique name for the provider.
            provider: The async provider instance.
            set_default: Whether to set this as the default provider.

        Raises:
            SecretConfigurationError: If name is empty.
        """
        if not name:
            raise SecretConfigurationError("Provider name cannot be empty")

        with self._lock:
            self._async_providers[name] = provider
            if set_default or self._default_name is None:
                self._default_name = name

    def unregister(self, name: str) -> bool:
        """Unregister a provider.

        Args:
            name: Name of the provider to unregister.

        Returns:
            True if the provider was unregistered, False if not found.
        """
        with self._lock:
            sync_removed = self._providers.pop(name, None) is not None
            async_removed = self._async_providers.pop(name, None) is not None
            if self._default_name == name:
                # Set new default from remaining providers
                if self._providers:
                    self._default_name = next(iter(self._providers))
                elif self._async_providers:
                    self._default_name = next(iter(self._async_providers))
                else:
                    self._default_name = None
            return sync_removed or async_removed

    def get(self, name: str | None = None) -> SecretProvider:
        """Get a registered provider.

        Args:
            name: Provider name. Uses default if None.

        Returns:
            The registered provider.

        Raises:
            ProviderNotFoundError: If the provider is not found.

        Example:
            >>> provider = registry.get("vault")
            >>> default = registry.get()  # Uses default
        """
        provider_name = name or self._default_name
        if provider_name is None:
            raise ProviderNotFoundError("No providers registered")

        provider = self._providers.get(provider_name)
        if provider is None:
            raise ProviderNotFoundError(provider_name)
        return provider

    def get_async(self, name: str | None = None) -> AsyncSecretProvider:
        """Get a registered async provider.

        Args:
            name: Provider name. Uses default if None.

        Returns:
            The registered async provider.

        Raises:
            ProviderNotFoundError: If the provider is not found.
        """
        provider_name = name or self._default_name
        if provider_name is None:
            raise ProviderNotFoundError("No providers registered")

        provider = self._async_providers.get(provider_name)
        if provider is None:
            raise ProviderNotFoundError(provider_name)
        return provider

    def set_default(self, name: str) -> None:
        """Set the default provider.

        Args:
            name: Name of the provider to set as default.

        Raises:
            ProviderNotFoundError: If the provider is not found.
        """
        if name not in self._providers and name not in self._async_providers:
            raise ProviderNotFoundError(name)
        self._default_name = name

    def exists(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name to check.

        Returns:
            True if the provider exists.
        """
        return name in self._providers or name in self._async_providers

    def list_providers(self) -> Sequence[str]:
        """List all registered provider names.

        Returns:
            List of provider names.
        """
        names = set(self._providers.keys())
        names.update(self._async_providers.keys())
        return sorted(names)

    def clear(self) -> None:
        """Remove all registered providers."""
        with self._lock:
            self._providers.clear()
            self._async_providers.clear()
            self._default_name = None

    def discover_plugins(self) -> int:
        """Discover and register providers from entry points.

        Looks for providers registered under the 'truthound.secret_providers'
        entry point group.

        Returns:
            Number of providers discovered.

        Example:
            # In pyproject.toml:
            # [project.entry-points."truthound.secret_providers"]
            # my_provider = "my_package:MySecretProvider"

            >>> count = registry.discover_plugins()
            >>> print(f"Discovered {count} providers")
        """
        discovered = 0
        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
            for ep in eps:
                try:
                    provider_class = ep.load()
                    # Only instantiate if it's a callable
                    if callable(provider_class):
                        provider = provider_class()
                        if isinstance(provider, SecretProvider):
                            self.register(ep.name, provider)
                            discovered += 1
                        elif isinstance(provider, AsyncSecretProvider):
                            self.register_async(ep.name, provider)
                            discovered += 1
                except Exception:
                    # Skip providers that fail to load
                    pass
        except TypeError:
            # Python 3.9 compatibility
            pass
        return discovered

    def create_from_config(
        self,
        name: str,
        config: SecretConfig,
    ) -> SecretProvider:
        """Create and register a provider from configuration.

        Args:
            name: Name to register the provider under.
            config: Configuration for the provider.

        Returns:
            The created provider.

        Raises:
            SecretConfigurationError: If the backend type is not supported.
        """
        from .config import BackendType

        # Import backends lazily
        if config.backend_type == BackendType.MEMORY:
            from .backends.memory import InMemorySecretProvider

            provider = InMemorySecretProvider()
        elif config.backend_type == BackendType.ENV:
            from .backends.env import EnvironmentSecretProvider

            provider = EnvironmentSecretProvider()
        elif config.backend_type == BackendType.FILE:
            from .backends.file import FileSecretProvider

            provider = FileSecretProvider()
        else:
            raise SecretConfigurationError(
                f"Unsupported backend type: {config.backend_type.name}",
                config_key="backend_type",
            )

        self.register(name, provider)
        return provider


# =============================================================================
# Global Singleton Access
# =============================================================================


_registry: SecretProviderRegistry | None = None
_registry_lock = threading.Lock()


def get_secret_registry() -> SecretProviderRegistry:
    """Get the global secret provider registry.

    Returns:
        The singleton registry instance.

    Example:
        >>> registry = get_secret_registry()
        >>> registry.register("vault", provider)
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = SecretProviderRegistry()
    return _registry


def reset_secret_registry() -> None:
    """Reset the global registry (for testing).

    Clears all registered providers and resets the singleton.
    """
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.clear()
        _registry = None
        # Also reset the class singleton
        SecretProviderRegistry._instance = None


# =============================================================================
# Convenience Functions
# =============================================================================


def get_secret(
    path: str,
    *,
    version: str | None = None,
    provider: str | None = None,
) -> SecretValue | None:
    """Get a secret from the default or specified provider.

    Args:
        path: The secret path.
        version: Optional specific version.
        provider: Optional provider name.

    Returns:
        The secret value, or None if not found.

    Example:
        >>> secret = get_secret("database/password")
        >>> if secret:
        ...     print(f"Password version: {secret.version}")
    """
    registry = get_secret_registry()
    return registry.get(provider).get(path, version=version)


def set_secret(
    path: str,
    value: str | bytes,
    *,
    secret_type: SecretType = SecretType.STRING,
    expires_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
    provider: str | None = None,
) -> SecretValue:
    """Set a secret in the default or specified provider.

    Args:
        path: The secret path.
        value: The secret value.
        secret_type: Type of secret.
        expires_at: Optional expiration time.
        metadata: Optional metadata.
        provider: Optional provider name.

    Returns:
        The stored secret value.

    Example:
        >>> result = set_secret("api/key", "new-key-value")
        >>> print(f"Stored version: {result.version}")
    """
    registry = get_secret_registry()
    return registry.get(provider).set(
        path,
        value,
        secret_type=secret_type,
        expires_at=expires_at,
        metadata=metadata,
    )


def delete_secret(
    path: str,
    *,
    provider: str | None = None,
) -> bool:
    """Delete a secret from the default or specified provider.

    Args:
        path: The secret path.
        provider: Optional provider name.

    Returns:
        True if deleted, False if not found.

    Example:
        >>> if delete_secret("old/secret"):
        ...     print("Secret deleted")
    """
    registry = get_secret_registry()
    return registry.get(provider).delete(path)


def secret_exists(
    path: str,
    *,
    provider: str | None = None,
) -> bool:
    """Check if a secret exists.

    Args:
        path: The secret path.
        provider: Optional provider name.

    Returns:
        True if the secret exists.

    Example:
        >>> if secret_exists("database/password"):
        ...     print("Secret is configured")
    """
    registry = get_secret_registry()
    return registry.get(provider).exists(path)


def list_secrets(
    prefix: str = "",
    *,
    limit: int | None = None,
    offset: int = 0,
    provider: str | None = None,
) -> Sequence[SecretMetadata]:
    """List secrets matching a prefix.

    Args:
        prefix: Path prefix to filter by.
        limit: Maximum number of results.
        offset: Number of results to skip.
        provider: Optional provider name.

    Returns:
        List of secret metadata.

    Example:
        >>> secrets = list_secrets("database/")
        >>> for s in secrets:
        ...     print(s.path)
    """
    registry = get_secret_registry()
    return registry.get(provider).list(prefix, limit=limit, offset=offset)
