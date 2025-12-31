"""Configuration system for secret management.

This module provides immutable configuration classes with builder patterns
for the secret management system.

Example:
    >>> from packages.enterprise.secrets import SecretConfig, BackendType
    >>>
    >>> config = SecretConfig(
    ...     backend_type=BackendType.VAULT,
    ...     cache_enabled=True,
    ...     cache_ttl_seconds=300.0,
    ... )
    >>>
    >>> # Builder pattern
    >>> config = (
    ...     SecretConfig()
    ...     .with_cache(enabled=True, ttl_seconds=300.0)
    ...     .with_retry(enabled=True, max_attempts=3)
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class BackendType(Enum):
    """Types of secret storage backends.

    Attributes:
        MEMORY: In-memory storage (for testing).
        ENV: Environment variables.
        FILE: Encrypted file storage.
        VAULT: HashiCorp Vault.
        AWS_SECRETS_MANAGER: AWS Secrets Manager.
        AWS_PARAMETER_STORE: AWS Systems Manager Parameter Store.
        AWS_KMS: AWS Key Management Service.
        GCP_SECRET_MANAGER: Google Cloud Secret Manager.
        AZURE_KEY_VAULT: Azure Key Vault.
    """

    MEMORY = auto()
    ENV = auto()
    FILE = auto()
    VAULT = auto()
    AWS_SECRETS_MANAGER = auto()
    AWS_PARAMETER_STORE = auto()
    AWS_KMS = auto()
    GCP_SECRET_MANAGER = auto()
    AZURE_KEY_VAULT = auto()


class EncryptionAlgorithm(Enum):
    """Encryption algorithms for client-side encryption.

    Attributes:
        FERNET: Fernet (AES-128-CBC with HMAC).
        AES_256_GCM: AES-256 in GCM mode.
        CHACHA20_POLY1305: ChaCha20-Poly1305.
    """

    FERNET = auto()
    AES_256_GCM = auto()
    CHACHA20_POLY1305 = auto()


@dataclass(frozen=True, slots=True)
class SecretConfig:
    """Configuration for secret management.

    All fields are immutable. Use builder methods to create modified copies.

    Attributes:
        backend_type: Type of secret storage backend.
        namespace: Namespace prefix for all secrets.
        cache_enabled: Whether to cache secrets.
        cache_ttl_seconds: Cache time-to-live in seconds.
        retry_enabled: Whether to retry failed operations.
        retry_max_attempts: Maximum retry attempts.
        retry_delay_seconds: Delay between retries.
        encryption_enabled: Whether to encrypt secrets client-side.
        encryption_algorithm: Algorithm for client-side encryption.
        audit_enabled: Whether to enable audit logging.
        timeout_seconds: Operation timeout in seconds.
        validate_on_get: Whether to validate secrets on retrieval.
        mask_in_logs: Whether to mask secret values in logs.

    Example:
        >>> config = SecretConfig(
        ...     cache_enabled=True,
        ...     cache_ttl_seconds=300.0,
        ... )
        >>> config = config.with_retry(enabled=True, max_attempts=5)
    """

    backend_type: BackendType = BackendType.MEMORY
    namespace: str = ""
    cache_enabled: bool = False
    cache_ttl_seconds: float = 300.0
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    encryption_enabled: bool = False
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET
    audit_enabled: bool = True
    timeout_seconds: float = 30.0
    validate_on_get: bool = True
    mask_in_logs: bool = True

    def with_backend(self, backend_type: BackendType) -> SecretConfig:
        """Create a copy with a different backend type.

        Args:
            backend_type: The backend type to use.

        Returns:
            New config with updated backend type.
        """
        return SecretConfig(
            backend_type=backend_type,
            namespace=self.namespace,
            cache_enabled=self.cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
            retry_enabled=self.retry_enabled,
            retry_max_attempts=self.retry_max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            encryption_enabled=self.encryption_enabled,
            encryption_algorithm=self.encryption_algorithm,
            audit_enabled=self.audit_enabled,
            timeout_seconds=self.timeout_seconds,
            validate_on_get=self.validate_on_get,
            mask_in_logs=self.mask_in_logs,
        )

    def with_namespace(self, namespace: str) -> SecretConfig:
        """Create a copy with a different namespace.

        Args:
            namespace: The namespace prefix.

        Returns:
            New config with updated namespace.
        """
        return SecretConfig(
            backend_type=self.backend_type,
            namespace=namespace,
            cache_enabled=self.cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
            retry_enabled=self.retry_enabled,
            retry_max_attempts=self.retry_max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            encryption_enabled=self.encryption_enabled,
            encryption_algorithm=self.encryption_algorithm,
            audit_enabled=self.audit_enabled,
            timeout_seconds=self.timeout_seconds,
            validate_on_get=self.validate_on_get,
            mask_in_logs=self.mask_in_logs,
        )

    def with_cache(
        self,
        enabled: bool = True,
        ttl_seconds: float | None = None,
    ) -> SecretConfig:
        """Create a copy with cache settings.

        Args:
            enabled: Whether to enable caching.
            ttl_seconds: Cache TTL in seconds.

        Returns:
            New config with updated cache settings.
        """
        return SecretConfig(
            backend_type=self.backend_type,
            namespace=self.namespace,
            cache_enabled=enabled,
            cache_ttl_seconds=ttl_seconds if ttl_seconds is not None else self.cache_ttl_seconds,
            retry_enabled=self.retry_enabled,
            retry_max_attempts=self.retry_max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            encryption_enabled=self.encryption_enabled,
            encryption_algorithm=self.encryption_algorithm,
            audit_enabled=self.audit_enabled,
            timeout_seconds=self.timeout_seconds,
            validate_on_get=self.validate_on_get,
            mask_in_logs=self.mask_in_logs,
        )

    def with_retry(
        self,
        enabled: bool = True,
        max_attempts: int | None = None,
        delay_seconds: float | None = None,
    ) -> SecretConfig:
        """Create a copy with retry settings.

        Args:
            enabled: Whether to enable retries.
            max_attempts: Maximum retry attempts.
            delay_seconds: Delay between retries.

        Returns:
            New config with updated retry settings.
        """
        return SecretConfig(
            backend_type=self.backend_type,
            namespace=self.namespace,
            cache_enabled=self.cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
            retry_enabled=enabled,
            retry_max_attempts=max_attempts if max_attempts is not None else self.retry_max_attempts,
            retry_delay_seconds=delay_seconds if delay_seconds is not None else self.retry_delay_seconds,
            encryption_enabled=self.encryption_enabled,
            encryption_algorithm=self.encryption_algorithm,
            audit_enabled=self.audit_enabled,
            timeout_seconds=self.timeout_seconds,
            validate_on_get=self.validate_on_get,
            mask_in_logs=self.mask_in_logs,
        )

    def with_encryption(
        self,
        enabled: bool = True,
        algorithm: EncryptionAlgorithm | None = None,
    ) -> SecretConfig:
        """Create a copy with encryption settings.

        Args:
            enabled: Whether to enable client-side encryption.
            algorithm: Encryption algorithm to use.

        Returns:
            New config with updated encryption settings.
        """
        return SecretConfig(
            backend_type=self.backend_type,
            namespace=self.namespace,
            cache_enabled=self.cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
            retry_enabled=self.retry_enabled,
            retry_max_attempts=self.retry_max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            encryption_enabled=enabled,
            encryption_algorithm=algorithm if algorithm is not None else self.encryption_algorithm,
            audit_enabled=self.audit_enabled,
            timeout_seconds=self.timeout_seconds,
            validate_on_get=self.validate_on_get,
            mask_in_logs=self.mask_in_logs,
        )

    def with_audit(self, enabled: bool = True) -> SecretConfig:
        """Create a copy with audit settings.

        Args:
            enabled: Whether to enable audit logging.

        Returns:
            New config with updated audit settings.
        """
        return SecretConfig(
            backend_type=self.backend_type,
            namespace=self.namespace,
            cache_enabled=self.cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
            retry_enabled=self.retry_enabled,
            retry_max_attempts=self.retry_max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            encryption_enabled=self.encryption_enabled,
            encryption_algorithm=self.encryption_algorithm,
            audit_enabled=enabled,
            timeout_seconds=self.timeout_seconds,
            validate_on_get=self.validate_on_get,
            mask_in_logs=self.mask_in_logs,
        )

    def with_timeout(self, timeout_seconds: float) -> SecretConfig:
        """Create a copy with timeout settings.

        Args:
            timeout_seconds: Operation timeout in seconds.

        Returns:
            New config with updated timeout.
        """
        return SecretConfig(
            backend_type=self.backend_type,
            namespace=self.namespace,
            cache_enabled=self.cache_enabled,
            cache_ttl_seconds=self.cache_ttl_seconds,
            retry_enabled=self.retry_enabled,
            retry_max_attempts=self.retry_max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            encryption_enabled=self.encryption_enabled,
            encryption_algorithm=self.encryption_algorithm,
            audit_enabled=self.audit_enabled,
            timeout_seconds=timeout_seconds,
            validate_on_get=self.validate_on_get,
            mask_in_logs=self.mask_in_logs,
        )


# =============================================================================
# Backend-Specific Configurations
# =============================================================================


@dataclass(frozen=True, slots=True)
class VaultConfig:
    """Configuration for HashiCorp Vault backend.

    Attributes:
        address: Vault server address.
        token: Authentication token (prefer token_path for security).
        token_path: Path to file containing token.
        mount_point: KV secrets engine mount point.
        kv_version: KV engine version (1 or 2).
        namespace: Vault namespace (Enterprise only).
        verify_ssl: Whether to verify SSL certificates.
        ca_cert_path: Path to CA certificate.
        client_cert_path: Path to client certificate.
        client_key_path: Path to client key.
        timeout_seconds: Request timeout.

    Example:
        >>> config = VaultConfig(
        ...     address="https://vault.example.com",
        ...     token_path="/run/secrets/vault-token",
        ...     mount_point="secret",
        ...     kv_version=2,
        ... )
    """

    address: str = "http://localhost:8200"
    token: str | None = None
    token_path: str | None = None
    mount_point: str = "secret"
    kv_version: int = 2
    namespace: str | None = None
    verify_ssl: bool = True
    ca_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None
    timeout_seconds: float = 30.0

    def with_address(self, address: str) -> VaultConfig:
        """Create a copy with a different address."""
        return VaultConfig(
            address=address,
            token=self.token,
            token_path=self.token_path,
            mount_point=self.mount_point,
            kv_version=self.kv_version,
            namespace=self.namespace,
            verify_ssl=self.verify_ssl,
            ca_cert_path=self.ca_cert_path,
            client_cert_path=self.client_cert_path,
            client_key_path=self.client_key_path,
            timeout_seconds=self.timeout_seconds,
        )

    def with_auth(
        self,
        token: str | None = None,
        token_path: str | None = None,
    ) -> VaultConfig:
        """Create a copy with authentication settings."""
        return VaultConfig(
            address=self.address,
            token=token,
            token_path=token_path,
            mount_point=self.mount_point,
            kv_version=self.kv_version,
            namespace=self.namespace,
            verify_ssl=self.verify_ssl,
            ca_cert_path=self.ca_cert_path,
            client_cert_path=self.client_cert_path,
            client_key_path=self.client_key_path,
            timeout_seconds=self.timeout_seconds,
        )

    def with_kv(self, mount_point: str, version: int = 2) -> VaultConfig:
        """Create a copy with KV engine settings."""
        return VaultConfig(
            address=self.address,
            token=self.token,
            token_path=self.token_path,
            mount_point=mount_point,
            kv_version=version,
            namespace=self.namespace,
            verify_ssl=self.verify_ssl,
            ca_cert_path=self.ca_cert_path,
            client_cert_path=self.client_cert_path,
            client_key_path=self.client_key_path,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class AWSSecretsManagerConfig:
    """Configuration for AWS Secrets Manager backend.

    Attributes:
        region_name: AWS region.
        access_key_id: AWS access key (prefer IAM roles).
        secret_access_key: AWS secret key (prefer IAM roles).
        session_token: AWS session token for temporary credentials.
        endpoint_url: Custom endpoint URL (for LocalStack, etc.).
        kms_key_id: KMS key ID for encryption.
        prefix: Prefix for secret names.

    Example:
        >>> config = AWSSecretsManagerConfig(
        ...     region_name="us-east-1",
        ...     prefix="myapp/",
        ... )
    """

    region_name: str = "us-east-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    endpoint_url: str | None = None
    kms_key_id: str | None = None
    prefix: str = ""

    def with_region(self, region_name: str) -> AWSSecretsManagerConfig:
        """Create a copy with a different region."""
        return AWSSecretsManagerConfig(
            region_name=region_name,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            session_token=self.session_token,
            endpoint_url=self.endpoint_url,
            kms_key_id=self.kms_key_id,
            prefix=self.prefix,
        )

    def with_credentials(
        self,
        access_key_id: str,
        secret_access_key: str,
        session_token: str | None = None,
    ) -> AWSSecretsManagerConfig:
        """Create a copy with explicit credentials."""
        return AWSSecretsManagerConfig(
            region_name=self.region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            endpoint_url=self.endpoint_url,
            kms_key_id=self.kms_key_id,
            prefix=self.prefix,
        )

    def with_kms(self, kms_key_id: str) -> AWSSecretsManagerConfig:
        """Create a copy with a KMS key."""
        return AWSSecretsManagerConfig(
            region_name=self.region_name,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            session_token=self.session_token,
            endpoint_url=self.endpoint_url,
            kms_key_id=kms_key_id,
            prefix=self.prefix,
        )


@dataclass(frozen=True, slots=True)
class AWSParameterStoreConfig:
    """Configuration for AWS Systems Manager Parameter Store.

    Attributes:
        region_name: AWS region.
        access_key_id: AWS access key.
        secret_access_key: AWS secret key.
        session_token: AWS session token.
        endpoint_url: Custom endpoint URL.
        kms_key_id: KMS key ID for SecureString parameters.
        prefix: Prefix for parameter names.
        with_decryption: Whether to decrypt SecureString parameters.

    Example:
        >>> config = AWSParameterStoreConfig(
        ...     region_name="us-east-1",
        ...     prefix="/myapp/",
        ...     with_decryption=True,
        ... )
    """

    region_name: str = "us-east-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    endpoint_url: str | None = None
    kms_key_id: str | None = None
    prefix: str = ""
    with_decryption: bool = True


@dataclass(frozen=True, slots=True)
class GCPSecretManagerConfig:
    """Configuration for Google Cloud Secret Manager.

    Attributes:
        project_id: GCP project ID.
        credentials_path: Path to service account JSON file.
        credentials: Credentials dict (prefer credentials_path).

    Example:
        >>> config = GCPSecretManagerConfig(
        ...     project_id="my-project",
        ...     credentials_path="/path/to/service-account.json",
        ... )
    """

    project_id: str = ""
    credentials_path: str | None = None
    credentials: Mapping[str, Any] | None = None

    def with_project(self, project_id: str) -> GCPSecretManagerConfig:
        """Create a copy with a different project ID."""
        return GCPSecretManagerConfig(
            project_id=project_id,
            credentials_path=self.credentials_path,
            credentials=self.credentials,
        )

    def with_credentials(
        self,
        credentials_path: str | None = None,
        credentials: Mapping[str, Any] | None = None,
    ) -> GCPSecretManagerConfig:
        """Create a copy with different credentials."""
        return GCPSecretManagerConfig(
            project_id=self.project_id,
            credentials_path=credentials_path,
            credentials=credentials,
        )


@dataclass(frozen=True, slots=True)
class AzureKeyVaultConfig:
    """Configuration for Azure Key Vault.

    Attributes:
        vault_url: Key Vault URL (e.g., https://myvault.vault.azure.net/).
        tenant_id: Azure AD tenant ID.
        client_id: Azure AD client ID.
        client_secret: Azure AD client secret.
        certificate_path: Path to client certificate.
        use_managed_identity: Whether to use managed identity.

    Example:
        >>> config = AzureKeyVaultConfig(
        ...     vault_url="https://myvault.vault.azure.net/",
        ...     use_managed_identity=True,
        ... )
    """

    vault_url: str = ""
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    certificate_path: str | None = None
    use_managed_identity: bool = False

    def with_vault(self, vault_url: str) -> AzureKeyVaultConfig:
        """Create a copy with a different vault URL."""
        return AzureKeyVaultConfig(
            vault_url=vault_url,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
            certificate_path=self.certificate_path,
            use_managed_identity=self.use_managed_identity,
        )

    def with_service_principal(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
    ) -> AzureKeyVaultConfig:
        """Create a copy with service principal credentials."""
        return AzureKeyVaultConfig(
            vault_url=self.vault_url,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            certificate_path=None,
            use_managed_identity=False,
        )

    def with_managed_identity(self) -> AzureKeyVaultConfig:
        """Create a copy using managed identity."""
        return AzureKeyVaultConfig(
            vault_url=self.vault_url,
            tenant_id=None,
            client_id=None,
            client_secret=None,
            certificate_path=None,
            use_managed_identity=True,
        )


@dataclass(frozen=True, slots=True)
class FileSecretConfig:
    """Configuration for file-based secret storage.

    Attributes:
        directory: Directory to store secrets.
        encryption_key: Encryption key for secrets.
        encryption_key_path: Path to encryption key file.
        file_permissions: Unix file permissions (octal).
        create_directory: Whether to create directory if missing.

    Example:
        >>> config = FileSecretConfig(
        ...     directory="/etc/secrets",
        ...     encryption_key_path="/etc/secret-key",
        ... )
    """

    directory: str = ".secrets"
    encryption_key: str | None = None
    encryption_key_path: str | None = None
    file_permissions: int = 0o600
    create_directory: bool = True

    def with_directory(self, directory: str) -> FileSecretConfig:
        """Create a copy with a different directory."""
        return FileSecretConfig(
            directory=directory,
            encryption_key=self.encryption_key,
            encryption_key_path=self.encryption_key_path,
            file_permissions=self.file_permissions,
            create_directory=self.create_directory,
        )

    def with_encryption_key(
        self,
        key: str | None = None,
        key_path: str | None = None,
    ) -> FileSecretConfig:
        """Create a copy with encryption key settings."""
        return FileSecretConfig(
            directory=self.directory,
            encryption_key=key,
            encryption_key_path=key_path,
            file_permissions=self.file_permissions,
            create_directory=self.create_directory,
        )


@dataclass(frozen=True, slots=True)
class EnvSecretConfig:
    """Configuration for environment variable secrets.

    Attributes:
        prefix: Prefix to filter environment variables.
        strip_prefix: Whether to strip prefix from secret names.
        case_sensitive: Whether secret names are case-sensitive.

    Example:
        >>> config = EnvSecretConfig(
        ...     prefix="SECRET_",
        ...     strip_prefix=True,
        ... )
    """

    prefix: str = "SECRET_"
    strip_prefix: bool = True
    case_sensitive: bool = True


# =============================================================================
# Preset Configurations
# =============================================================================


DEFAULT_SECRET_CONFIG = SecretConfig()
"""Default configuration with minimal settings."""

PRODUCTION_SECRET_CONFIG = SecretConfig(
    cache_enabled=True,
    cache_ttl_seconds=300.0,
    retry_enabled=True,
    retry_max_attempts=5,
    retry_delay_seconds=2.0,
    audit_enabled=True,
    timeout_seconds=30.0,
    validate_on_get=True,
    mask_in_logs=True,
)
"""Production configuration with caching, retries, and audit."""

DEVELOPMENT_SECRET_CONFIG = SecretConfig(
    cache_enabled=False,
    retry_enabled=False,
    audit_enabled=False,
    timeout_seconds=10.0,
    validate_on_get=False,
    mask_in_logs=False,
)
"""Development configuration with minimal overhead."""

TESTING_SECRET_CONFIG = SecretConfig(
    backend_type=BackendType.MEMORY,
    cache_enabled=False,
    retry_enabled=False,
    audit_enabled=False,
    timeout_seconds=5.0,
    validate_on_get=False,
    mask_in_logs=False,
)
"""Testing configuration using in-memory backend."""

HIGH_SECURITY_SECRET_CONFIG = SecretConfig(
    cache_enabled=False,
    encryption_enabled=True,
    encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
    audit_enabled=True,
    validate_on_get=True,
    mask_in_logs=True,
)
"""High security configuration with encryption and no caching."""
