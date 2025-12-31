"""Composable middleware wrappers for secret providers.

This module provides wrapper classes that add functionality to secret providers
using the decorator pattern. Multiple wrappers can be composed together.

Example:
    >>> from packages.enterprise.secrets import (
    ...     create_wrapped_provider,
    ...     CachingProviderWrapper,
    ...     NamespacedProviderWrapper,
    ... )
    >>>
    >>> # Compose wrappers
    >>> wrapped = create_wrapped_provider(
    ...     provider,
    ...     config=PRODUCTION_SECRET_CONFIG,
    ...     hooks=[AuditLoggingHook()],
    ... )
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .base import SecretMetadata, SecretProvider, SecretType, SecretValue
from .hooks import SecretHook, SecretOperation, SecretOperationContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .config import SecretConfig
    from .encryption import SecretEncryptor


class ProviderWrapper:
    """Base class for provider wrappers (decorator pattern).

    Wraps a SecretProvider and delegates all calls to it.
    Subclasses can override methods to add behavior.

    Attributes:
        wrapped: The wrapped provider.

    Example:
        >>> class LoggingWrapper(ProviderWrapper):
        ...     def get(self, path, *, version=None):
        ...         print(f"Getting {path}")
        ...         return self.wrapped.get(path, version=version)
    """

    def __init__(self, wrapped: SecretProvider) -> None:
        """Initialize the wrapper.

        Args:
            wrapped: The provider to wrap.
        """
        self._wrapped = wrapped

    @property
    def wrapped(self) -> SecretProvider:
        """Get the wrapped provider."""
        return self._wrapped

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Delegate to wrapped provider."""
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
        """Delegate to wrapped provider."""
        return self._wrapped.set(
            path,
            value,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=metadata,
        )

    def delete(self, path: str) -> bool:
        """Delegate to wrapped provider."""
        return self._wrapped.delete(path)

    def exists(self, path: str) -> bool:
        """Delegate to wrapped provider."""
        return self._wrapped.exists(path)

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """Delegate to wrapped provider."""
        return self._wrapped.list(prefix, limit=limit, offset=offset)


class HookedProviderWrapper(ProviderWrapper):
    """Wrapper that invokes hooks around operations.

    Calls hooks before and after each operation, enabling
    audit logging, metrics, and custom behaviors.

    Example:
        >>> hooks = [AuditLoggingHook(), MetricsSecretHook()]
        >>> wrapped = HookedProviderWrapper(provider, hooks=hooks)
    """

    def __init__(
        self,
        wrapped: SecretProvider,
        hooks: Sequence[SecretHook],
        provider_name: str = "",
    ) -> None:
        """Initialize the hooked wrapper.

        Args:
            wrapped: The provider to wrap.
            hooks: Hooks to invoke.
            provider_name: Name of the provider for context.
        """
        super().__init__(wrapped)
        self._hooks = list(hooks)
        self._provider_name = provider_name

    def _create_context(
        self,
        operation: SecretOperation,
        path: str = "",
    ) -> SecretOperationContext:
        """Create an operation context.

        Args:
            operation: The operation type.
            path: The secret path.

        Returns:
            Operation context.
        """
        return SecretOperationContext(
            operation=operation,
            path=path,
            provider_name=self._provider_name,
        )

    def _call_before_hooks(
        self,
        method_name: str,
        context: SecretOperationContext,
    ) -> None:
        """Call before hooks."""
        for hook in self._hooks:
            try:
                getattr(hook, method_name)(context)
            except Exception:
                pass  # Hooks should not affect operation

    def _call_after_hooks(
        self,
        method_name: str,
        context: SecretOperationContext,
        *args: Any,
    ) -> None:
        """Call after hooks."""
        for hook in self._hooks:
            try:
                getattr(hook, method_name)(context, *args)
            except Exception:
                pass

    def _call_error_hooks(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Call error hooks."""
        for hook in self._hooks:
            try:
                hook.on_error(context, error, duration_ms)
            except Exception:
                pass

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get with hook invocation."""
        context = self._create_context(SecretOperation.GET, path)
        self._call_before_hooks("on_before_get", context)
        start = time.perf_counter()
        try:
            result = self._wrapped.get(path, version=version)
            duration_ms = (time.perf_counter() - start) * 1000
            self._call_after_hooks("on_after_get", context, result, duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._call_error_hooks(context, e, duration_ms)
            raise

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set with hook invocation."""
        context = self._create_context(SecretOperation.SET, path)
        self._call_before_hooks("on_before_set", context)
        start = time.perf_counter()
        try:
            result = self._wrapped.set(
                path,
                value,
                secret_type=secret_type,
                expires_at=expires_at,
                metadata=metadata,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            self._call_after_hooks("on_after_set", context, result, duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._call_error_hooks(context, e, duration_ms)
            raise

    def delete(self, path: str) -> bool:
        """Delete with hook invocation."""
        context = self._create_context(SecretOperation.DELETE, path)
        self._call_before_hooks("on_before_delete", context)
        start = time.perf_counter()
        try:
            result = self._wrapped.delete(path)
            duration_ms = (time.perf_counter() - start) * 1000
            self._call_after_hooks("on_after_delete", context, result, duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._call_error_hooks(context, e, duration_ms)
            raise


class NamespacedProviderWrapper(ProviderWrapper):
    """Wrapper that adds a namespace prefix to all paths.

    Useful for tenant isolation or environment separation.

    Example:
        >>> # All paths will be prefixed with "prod/"
        >>> wrapped = NamespacedProviderWrapper(provider, namespace="prod")
        >>> wrapped.get("db/password")  # Actually gets "prod/db/password"
    """

    def __init__(
        self,
        wrapped: SecretProvider,
        namespace: str,
        separator: str = "/",
    ) -> None:
        """Initialize the namespaced wrapper.

        Args:
            wrapped: The provider to wrap.
            namespace: Namespace prefix.
            separator: Path separator.
        """
        super().__init__(wrapped)
        self._namespace = namespace.rstrip(separator)
        self._separator = separator

    def _prefixed_path(self, path: str) -> str:
        """Add namespace prefix to path.

        Args:
            path: Original path.

        Returns:
            Prefixed path.
        """
        if not self._namespace:
            return path
        return f"{self._namespace}{self._separator}{path}"

    def _strip_prefix(self, path: str) -> str:
        """Remove namespace prefix from path.

        Args:
            path: Prefixed path.

        Returns:
            Original path.
        """
        prefix = f"{self._namespace}{self._separator}"
        if path.startswith(prefix):
            return path[len(prefix) :]
        return path

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get with namespace prefix."""
        return self._wrapped.get(self._prefixed_path(path), version=version)

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set with namespace prefix."""
        return self._wrapped.set(
            self._prefixed_path(path),
            value,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=metadata,
        )

    def delete(self, path: str) -> bool:
        """Delete with namespace prefix."""
        return self._wrapped.delete(self._prefixed_path(path))

    def exists(self, path: str) -> bool:
        """Check existence with namespace prefix."""
        return self._wrapped.exists(self._prefixed_path(path))

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List with namespace prefix and stripped results."""
        prefixed = self._prefixed_path(prefix)
        results = self._wrapped.list(prefixed, limit=limit, offset=offset)
        # Strip namespace from returned paths
        return [
            SecretMetadata(
                path=self._strip_prefix(m.path),
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


class EncryptingProviderWrapper(ProviderWrapper):
    """Wrapper that encrypts/decrypts secrets client-side.

    Provides an additional layer of encryption beyond what the
    backend provides.

    Example:
        >>> encryptor = FernetEncryptor(key="...")
        >>> wrapped = EncryptingProviderWrapper(provider, encryptor)
        >>> wrapped.set("secret", "value")  # Stored encrypted
        >>> wrapped.get("secret")  # Returns decrypted value
    """

    def __init__(
        self,
        wrapped: SecretProvider,
        encryptor: SecretEncryptor,
    ) -> None:
        """Initialize the encrypting wrapper.

        Args:
            wrapped: The provider to wrap.
            encryptor: The encryptor to use.
        """
        super().__init__(wrapped)
        self._encryptor = encryptor

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get and decrypt secret."""
        result = self._wrapped.get(path, version=version)
        if result is None:
            return None

        # Decrypt the value
        if isinstance(result.value, bytes):
            decrypted = self._encryptor.decrypt(result.value)
        else:
            # Value might be stored as base64 string
            import base64

            encrypted_bytes = base64.b64decode(result.value)
            decrypted = self._encryptor.decrypt(encrypted_bytes)

        return SecretValue(
            value=decrypted,
            version=result.version,
            created_at=result.created_at,
            expires_at=result.expires_at,
            secret_type=result.secret_type,
            metadata=result.metadata,
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
        """Encrypt and set secret."""
        # Encrypt the value
        if isinstance(value, str):
            value_bytes = value.encode("utf-8")
        else:
            value_bytes = value

        encrypted = self._encryptor.encrypt(value_bytes)

        # Store as base64 string for compatibility
        import base64

        encrypted_str = base64.b64encode(encrypted).decode("ascii")

        # Mark as encrypted in metadata
        meta = dict(metadata) if metadata else {}
        meta["encrypted"] = True
        meta["encryption_algorithm"] = type(self._encryptor).__name__

        return self._wrapped.set(
            path,
            encrypted_str,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=meta,
        )


class CachingProviderWrapper(ProviderWrapper):
    """Wrapper that caches secret values.

    Reduces backend calls by caching retrieved secrets.
    Cache is invalidated on set/delete operations.

    Example:
        >>> wrapped = CachingProviderWrapper(provider, ttl_seconds=300.0)
        >>> wrapped.get("secret")  # Fetches from backend
        >>> wrapped.get("secret")  # Returns cached value
    """

    def __init__(
        self,
        wrapped: SecretProvider,
        ttl_seconds: float = 300.0,
        max_size: int = 1000,
    ) -> None:
        """Initialize the caching wrapper.

        Args:
            wrapped: The provider to wrap.
            ttl_seconds: Cache TTL in seconds.
            max_size: Maximum cache size.
        """
        super().__init__(wrapped)
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._cache: dict[str, tuple[SecretValue, float]] = {}

    def _cache_key(self, path: str, version: str | None = None) -> str:
        """Generate cache key.

        Args:
            path: Secret path.
            version: Optional version.

        Returns:
            Cache key.
        """
        if version:
            return f"{path}@{version}"
        return path

    def _is_expired(self, cached_at: float) -> bool:
        """Check if cache entry is expired.

        Args:
            cached_at: Timestamp when cached.

        Returns:
            True if expired.
        """
        return time.time() - cached_at > self._ttl_seconds

    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, cached_at) in self._cache.items()
            if current_time - cached_at > self._ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get with caching."""
        key = self._cache_key(path, version)

        # Check cache
        if key in self._cache:
            value, cached_at = self._cache[key]
            if not self._is_expired(cached_at):
                return value
            # Expired, remove from cache
            del self._cache[key]

        # Fetch from backend
        result = self._wrapped.get(path, version=version)

        # Cache result
        if result is not None:
            # Evict if at max size
            if len(self._cache) >= self._max_size:
                self._evict_expired()
                # If still at max, remove oldest
                if len(self._cache) >= self._max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
            self._cache[key] = (result, time.time())

        return result

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set and invalidate cache."""
        result = self._wrapped.set(
            path,
            value,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=metadata,
        )

        # Invalidate cache for this path
        self._invalidate(path)

        # Cache the new value
        key = self._cache_key(path)
        self._cache[key] = (result, time.time())

        return result

    def delete(self, path: str) -> bool:
        """Delete and invalidate cache."""
        result = self._wrapped.delete(path)
        self._invalidate(path)
        return result

    def _invalidate(self, path: str) -> None:
        """Invalidate cache entries for a path.

        Args:
            path: Path to invalidate.
        """
        # Remove exact match and any versioned entries
        keys_to_remove = [k for k in self._cache if k == path or k.startswith(f"{path}@")]
        for key in keys_to_remove:
            del self._cache[key]

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


class ValidatingProviderWrapper(ProviderWrapper):
    """Wrapper that validates secrets on retrieval.

    Checks for expired secrets and optionally validates
    secret format/content.

    Example:
        >>> wrapped = ValidatingProviderWrapper(
        ...     provider,
        ...     reject_expired=True,
        ... )
    """

    def __init__(
        self,
        wrapped: SecretProvider,
        reject_expired: bool = True,
    ) -> None:
        """Initialize the validating wrapper.

        Args:
            wrapped: The provider to wrap.
            reject_expired: Whether to reject expired secrets.
        """
        super().__init__(wrapped)
        self._reject_expired = reject_expired

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get with validation."""
        from .exceptions import SecretExpiredError

        result = self._wrapped.get(path, version=version)

        if result is not None and self._reject_expired and result.is_expired:
            raise SecretExpiredError(path, expired_at=result.expires_at)

        return result


def create_wrapped_provider(
    provider: SecretProvider,
    *,
    config: SecretConfig | None = None,
    hooks: Sequence[SecretHook] | None = None,
    encryptor: SecretEncryptor | None = None,
    namespace: str | None = None,
    provider_name: str = "",
) -> SecretProvider:
    """Create a wrapped provider with middleware applied.

    Factory function that composes multiple wrappers based on
    configuration.

    Args:
        provider: The base provider.
        config: Optional configuration for wrappers.
        hooks: Optional hooks to apply.
        encryptor: Optional encryptor for client-side encryption.
        namespace: Optional namespace prefix.
        provider_name: Name of the provider for context.

    Returns:
        Wrapped provider with middleware applied.

    Example:
        >>> wrapped = create_wrapped_provider(
        ...     VaultSecretProvider(vault_config),
        ...     config=PRODUCTION_SECRET_CONFIG,
        ...     hooks=[AuditLoggingHook()],
        ...     namespace="myapp",
        ... )
    """
    result: SecretProvider = provider

    # Apply namespace first (innermost)
    if namespace:
        result = NamespacedProviderWrapper(result, namespace)
    elif config and config.namespace:
        result = NamespacedProviderWrapper(result, config.namespace)

    # Apply encryption
    if encryptor:
        result = EncryptingProviderWrapper(result, encryptor)

    # Apply caching
    if config and config.cache_enabled:
        result = CachingProviderWrapper(
            result,
            ttl_seconds=config.cache_ttl_seconds,
        )

    # Apply validation
    if config and config.validate_on_get:
        result = ValidatingProviderWrapper(result, reject_expired=True)

    # Apply hooks last (outermost)
    if hooks:
        result = HookedProviderWrapper(result, hooks, provider_name)

    return result
