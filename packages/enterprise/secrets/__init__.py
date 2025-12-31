"""Secret management system for enterprise deployments.

This package provides a comprehensive secret management system with support for
multiple backends, caching, encryption, rotation, and multi-tenant isolation.

Supported backends:
    - HashiCorp Vault
    - AWS Secrets Manager / Parameter Store
    - GCP Secret Manager
    - Azure Key Vault
    - Environment variables
    - Encrypted files
    - In-memory (for testing)

Example:
    >>> from packages.enterprise.secrets import (
    ...     get_secret_registry,
    ...     get_secret,
    ...     set_secret,
    ... )
    >>>
    >>> # Register a provider
    >>> from packages.enterprise.secrets.backends import InMemorySecretProvider
    >>> registry = get_secret_registry()
    >>> registry.register("memory", InMemorySecretProvider())
    >>>
    >>> # Use convenience functions
    >>> set_secret("database/password", "secret123")
    >>> secret = get_secret("database/password")
    >>> print(secret.value)
"""

from __future__ import annotations

# Core types and protocols
from .base import (
    AsyncSecretProvider,
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

# Configuration
from .config import (
    AWSSecretsManagerConfig,
    AzureKeyVaultConfig,
    BackendType,
    EnvSecretConfig,
    FileSecretConfig,
    GCPSecretManagerConfig,
    SecretConfig,
    VaultConfig,
    # Presets
    DEFAULT_SECRET_CONFIG,
    DEVELOPMENT_SECRET_CONFIG,
    HIGH_SECURITY_SECRET_CONFIG,
    PRODUCTION_SECRET_CONFIG,
    TESTING_SECRET_CONFIG,
)

# Exceptions
from .exceptions import (
    ProviderNotFoundError,
    RotationGeneratorError,
    RotationScheduleError,
    SecretAccessDeniedError,
    SecretAuthenticationError,
    SecretBackendError,
    SecretCacheError,
    SecretConfigurationError,
    SecretConnectionError,
    SecretDecryptError,
    SecretEncryptError,
    SecretEncryptionError,
    SecretError,
    SecretExpiredError,
    SecretNotFoundError,
    SecretRotationError,
    SecretValidationError,
)

# Registry and convenience functions
from .registry import (
    SecretProviderRegistry,
    delete_secret,
    get_secret,
    get_secret_registry,
    list_secrets,
    reset_secret_registry,
    secret_exists,
    set_secret,
)

# Hooks
from .hooks import (
    AuditLoggingHook,
    CompositeSecretHook,
    MetricsSecretHook,
    SecretHook,
    SecretOperationContext,
    TenantAwareSecretHook,
)

# Middleware
from .middleware import (
    CachingProviderWrapper,
    EncryptingProviderWrapper,
    HookedProviderWrapper,
    NamespacedProviderWrapper,
    ProviderWrapper,
    ValidatingProviderWrapper,
    create_wrapped_provider,
)

# Caching
from .cache import (
    CacheStats,
    SecretCache,
    TieredSecretCache,
)

# Encryption
from .encryption import (
    AESGCMEncryptor,
    ChaCha20Poly1305Encryptor,
    FernetEncryptor,
    PasswordDerivedEncryptor,
    SecretEncryptor,
    generate_aes_key,
    generate_chacha_key,
    generate_fernet_key,
)

# Rotation
from .rotation import (
    APIKeyGenerator,
    PasswordGenerator,
    RotationConfig,
    RotationResult,
    RotationSchedule,
    SecretGenerator,
    SecretRotationManager,
    TokenGenerator,
    UUIDGenerator,
)

# Multi-tenant
from .tenant import (
    TenantAwareSecretProvider,
    TenantSecretIsolator,
    create_tenant_provider,
)

__all__ = [
    # Core types
    "SecretType",
    "SecretValue",
    "SecretMetadata",
    "SecretVersion",
    "HealthStatus",
    "HealthCheckResult",
    # Protocols
    "SecretProvider",
    "AsyncSecretProvider",
    "VersionedSecretProvider",
    "HealthCheckable",
    # Configuration
    "SecretConfig",
    "VaultConfig",
    "AWSSecretsManagerConfig",
    "GCPSecretManagerConfig",
    "AzureKeyVaultConfig",
    "FileSecretConfig",
    "EnvSecretConfig",
    "BackendType",
    # Config presets
    "DEFAULT_SECRET_CONFIG",
    "PRODUCTION_SECRET_CONFIG",
    "DEVELOPMENT_SECRET_CONFIG",
    "TESTING_SECRET_CONFIG",
    "HIGH_SECURITY_SECRET_CONFIG",
    # Exceptions
    "SecretError",
    "SecretNotFoundError",
    "SecretAccessDeniedError",
    "SecretExpiredError",
    "SecretBackendError",
    "SecretConnectionError",
    "SecretAuthenticationError",
    "SecretEncryptionError",
    "SecretEncryptError",
    "SecretDecryptError",
    "SecretRotationError",
    "RotationScheduleError",
    "RotationGeneratorError",
    "SecretValidationError",
    "SecretConfigurationError",
    "SecretCacheError",
    "ProviderNotFoundError",
    # Registry
    "SecretProviderRegistry",
    "get_secret_registry",
    "reset_secret_registry",
    "get_secret",
    "set_secret",
    "delete_secret",
    "secret_exists",
    "list_secrets",
    # Hooks
    "SecretHook",
    "SecretOperationContext",
    "AuditLoggingHook",
    "MetricsSecretHook",
    "CompositeSecretHook",
    "TenantAwareSecretHook",
    # Middleware
    "ProviderWrapper",
    "HookedProviderWrapper",
    "NamespacedProviderWrapper",
    "EncryptingProviderWrapper",
    "CachingProviderWrapper",
    "ValidatingProviderWrapper",
    "create_wrapped_provider",
    # Caching
    "SecretCache",
    "TieredSecretCache",
    "CacheStats",
    # Encryption
    "SecretEncryptor",
    "FernetEncryptor",
    "AESGCMEncryptor",
    "ChaCha20Poly1305Encryptor",
    "PasswordDerivedEncryptor",
    "generate_fernet_key",
    "generate_aes_key",
    "generate_chacha_key",
    # Rotation
    "SecretRotationManager",
    "RotationSchedule",
    "RotationConfig",
    "RotationResult",
    "SecretGenerator",
    "PasswordGenerator",
    "UUIDGenerator",
    "APIKeyGenerator",
    "TokenGenerator",
    # Multi-tenant
    "TenantAwareSecretProvider",
    "TenantSecretIsolator",
    "create_tenant_provider",
]
