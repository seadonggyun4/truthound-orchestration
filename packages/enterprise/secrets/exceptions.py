"""Exception hierarchy for secret management.

This module provides a structured exception hierarchy for the secret management
system. All exceptions inherit from SecretError.

Exception Hierarchy:
    SecretError (base)
    ├── SecretNotFoundError
    ├── SecretAccessDeniedError
    ├── SecretExpiredError
    ├── SecretValidationError
    ├── SecretConfigurationError
    ├── SecretBackendError
    │   ├── SecretConnectionError
    │   └── SecretAuthenticationError
    ├── SecretEncryptionError
    │   ├── SecretEncryptError
    │   └── SecretDecryptError
    └── SecretRotationError
        ├── RotationScheduleError
        └── RotationGeneratorError

Example:
    >>> try:
    ...     secret = provider.get("nonexistent/path")
    ... except SecretNotFoundError as e:
    ...     print(f"Secret not found: {e.path}")
    ... except SecretError as e:
    ...     print(f"Secret error: {e}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.exceptions import TruthoundIntegrationError

if TYPE_CHECKING:
    pass


class SecretError(TruthoundIntegrationError):
    """Base exception for all secret management errors.

    All exceptions in the secret management system inherit from this class,
    allowing callers to catch any secret-related error at a single point.

    Attributes:
        message: Human-readable error description.
        path: Optional secret path involved in the error.
        details: Optional dictionary with additional error context.
        cause: Optional original exception that caused this error.

    Example:
        >>> try:
        ...     raise SecretError("Something went wrong", path="db/password")
        ... except SecretError as e:
        ...     print(f"Error for {e.path}: {e.message}")
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            path: Optional secret path involved in the error.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if path:
            details["path"] = path
        super().__init__(message, details=details, cause=cause)
        self.path = path


class SecretNotFoundError(SecretError):
    """Exception raised when a secret is not found.

    Attributes:
        path: The path that was not found.
        version: Optional version that was requested.

    Example:
        >>> raise SecretNotFoundError(path="db/password")
        SecretNotFoundError: Secret not found: db/password
    """

    def __init__(
        self,
        path: str,
        *,
        version: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            path: The secret path that was not found.
            version: Optional version that was requested.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if version:
            details["version"] = version
            message = f"Secret not found: {path} (version: {version})"
        else:
            message = f"Secret not found: {path}"
        super().__init__(message, path=path, details=details, cause=cause)
        self.version = version


class SecretAccessDeniedError(SecretError):
    """Exception raised when access to a secret is denied.

    Attributes:
        path: The path that access was denied for.
        operation: The operation that was attempted.

    Example:
        >>> raise SecretAccessDeniedError(path="admin/key", operation="read")
    """

    def __init__(
        self,
        message: str = "Access denied",
        *,
        path: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            path: The secret path.
            operation: The operation that was attempted.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if operation:
            details["operation"] = operation
        super().__init__(message, path=path, details=details, cause=cause)
        self.operation = operation


class SecretExpiredError(SecretError):
    """Exception raised when a secret has expired.

    Attributes:
        path: The path of the expired secret.
        expired_at: When the secret expired.

    Example:
        >>> from datetime import datetime
        >>> raise SecretExpiredError(path="api/token", expired_at=datetime.now())
    """

    def __init__(
        self,
        path: str,
        *,
        expired_at: Any | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            path: The secret path.
            expired_at: When the secret expired.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if expired_at:
            details["expired_at"] = str(expired_at)
        message = f"Secret expired: {path}"
        super().__init__(message, path=path, details=details, cause=cause)
        self.expired_at = expired_at


class SecretValidationError(SecretError):
    """Exception raised when secret validation fails.

    Attributes:
        path: The secret path.
        validation_errors: List of validation error messages.

    Example:
        >>> raise SecretValidationError(
        ...     path="db/password",
        ...     validation_errors=["Value too short", "Missing uppercase"],
        ... )
    """

    def __init__(
        self,
        message: str = "Secret validation failed",
        *,
        path: str | None = None,
        validation_errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            path: The secret path.
            validation_errors: List of validation error messages.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, path=path, details=details, cause=cause)
        self.validation_errors = validation_errors or []


class SecretConfigurationError(SecretError):
    """Exception raised for configuration errors.

    Attributes:
        config_key: The configuration key that caused the error.

    Example:
        >>> raise SecretConfigurationError(
        ...     "Invalid backend configuration",
        ...     config_key="vault_address",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            config_key: The configuration key that caused the error.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, cause=cause)
        self.config_key = config_key


class SecretBackendError(SecretError):
    """Exception raised for backend-related errors.

    Base class for errors from the secret storage backend.

    Attributes:
        backend: Name of the backend that caused the error.

    Example:
        >>> raise SecretBackendError(
        ...     "Backend unavailable",
        ...     backend="vault",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            backend: Name of the backend.
            path: Optional secret path.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if backend:
            details["backend"] = backend
        super().__init__(message, path=path, details=details, cause=cause)
        self.backend = backend


class SecretConnectionError(SecretBackendError):
    """Exception raised when connection to backend fails.

    Attributes:
        endpoint: The endpoint that could not be reached.

    Example:
        >>> raise SecretConnectionError(
        ...     "Failed to connect to Vault",
        ...     backend="vault",
        ...     endpoint="https://vault.example.com",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        endpoint: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            backend: Name of the backend.
            endpoint: The endpoint URL.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, backend=backend, details=details, cause=cause)
        self.endpoint = endpoint


class SecretAuthenticationError(SecretBackendError):
    """Exception raised when authentication to backend fails.

    Attributes:
        auth_method: The authentication method that failed.

    Example:
        >>> raise SecretAuthenticationError(
        ...     "Token expired",
        ...     backend="vault",
        ...     auth_method="token",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        auth_method: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            backend: Name of the backend.
            auth_method: The authentication method that failed.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if auth_method:
            details["auth_method"] = auth_method
        super().__init__(message, backend=backend, details=details, cause=cause)
        self.auth_method = auth_method


class SecretEncryptionError(SecretError):
    """Exception raised for encryption/decryption errors.

    Base class for encryption-related errors.

    Attributes:
        algorithm: The encryption algorithm involved.

    Example:
        >>> raise SecretEncryptionError(
        ...     "Encryption failed",
        ...     algorithm="AES-256-GCM",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        algorithm: str | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            algorithm: The encryption algorithm.
            path: Optional secret path.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if algorithm:
            details["algorithm"] = algorithm
        super().__init__(message, path=path, details=details, cause=cause)
        self.algorithm = algorithm


class SecretEncryptError(SecretEncryptionError):
    """Exception raised when encryption fails."""

    pass


class SecretDecryptError(SecretEncryptionError):
    """Exception raised when decryption fails."""

    pass


class SecretRotationError(SecretError):
    """Exception raised for secret rotation errors.

    Base class for rotation-related errors.

    Example:
        >>> raise SecretRotationError(
        ...     "Rotation failed",
        ...     path="api/key",
        ... )
    """

    pass


class RotationScheduleError(SecretRotationError):
    """Exception raised when scheduling rotation fails.

    Attributes:
        schedule_id: The schedule ID that failed.

    Example:
        >>> raise RotationScheduleError(
        ...     "Invalid rotation schedule",
        ...     schedule_id="weekly-rotation",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        schedule_id: str | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            schedule_id: The schedule ID.
            path: Optional secret path.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if schedule_id:
            details["schedule_id"] = schedule_id
        super().__init__(message, path=path, details=details, cause=cause)
        self.schedule_id = schedule_id


class RotationGeneratorError(SecretRotationError):
    """Exception raised when secret generation fails during rotation.

    Attributes:
        generator: The generator that failed.

    Example:
        >>> raise RotationGeneratorError(
        ...     "Failed to generate password",
        ...     generator="password",
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        generator: str | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            generator: The generator name.
            path: Optional secret path.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        if generator:
            details["generator"] = generator
        super().__init__(message, path=path, details=details, cause=cause)
        self.generator = generator


class SecretCacheError(SecretError):
    """Exception raised for cache-related errors.

    Example:
        >>> raise SecretCacheError("Cache full", path="api/key")
    """

    pass


class ProviderNotFoundError(SecretError):
    """Exception raised when a provider is not found in the registry.

    Attributes:
        provider_name: The name of the provider that was not found.

    Example:
        >>> raise ProviderNotFoundError("vault")
    """

    def __init__(
        self,
        provider_name: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            provider_name: The name of the provider.
            details: Optional dictionary with additional error context.
            cause: Optional original exception.
        """
        details = details or {}
        details["provider_name"] = provider_name
        message = f"Provider not found: {provider_name}"
        super().__init__(message, details=details, cause=cause)
        self.provider_name = provider_name
