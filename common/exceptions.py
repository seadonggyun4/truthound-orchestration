"""Exception hierarchy for Truthound Integrations.

This module provides a structured exception hierarchy for consistent error handling
across all platform integrations. All exceptions inherit from TruthoundIntegrationError
to allow catching any integration-related error at a single point.

Exception Hierarchy:
    TruthoundIntegrationError (base)
    ├── ConfigurationError
    │   ├── InvalidConfigValueError
    │   └── MissingConfigError
    ├── ValidationExecutionError
    │   ├── RuleExecutionError
    │   └── DataAccessError
    ├── SerializationError
    │   ├── SerializeError
    │   └── DeserializeError
    ├── PlatformConnectionError
    │   └── AuthenticationError
    ├── IntegrationTimeoutError
    └── QualityGateError
        └── ThresholdExceededError

Example:
    >>> try:
    ...     result = adapter.check(data, config)
    ... except QualityGateError as e:
    ...     logger.warning(f"Quality gate failed: {e}")
    ... except TruthoundIntegrationError as e:
    ...     logger.error(f"Integration error: {e}")
"""

from __future__ import annotations

from typing import Any


class TruthoundIntegrationError(Exception):
    """Base exception for all Truthound integration errors.

    All exceptions in the Truthound integration system inherit from this class,
    allowing callers to catch any integration-related error at a single point.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
        cause: Optional original exception that caused this error.

    Example:
        >>> try:
        ...     raise TruthoundIntegrationError("Something went wrong", details={"key": "value"})
        ... except TruthoundIntegrationError as e:
        ...     print(f"Error: {e.message}, Details: {e.details}")
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation with details if present."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"cause={self.cause!r})"
        )

    def with_context(self, **kwargs: Any) -> TruthoundIntegrationError:
        """Create a new exception with additional context details.

        This method returns a new instance with merged details, preserving
        immutability of the original exception.

        Args:
            **kwargs: Additional context to add to details.

        Returns:
            New exception instance with merged details.

        Example:
            >>> e = TruthoundIntegrationError("Error", details={"key": "value"})
            >>> e2 = e.with_context(platform="airflow")
            >>> e2.details
            {'key': 'value', 'platform': 'airflow'}
        """
        merged_details = {**self.details, **kwargs}
        return self.__class__(self.message, details=merged_details, cause=self.cause)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(TruthoundIntegrationError):
    """Exception for configuration-related errors.

    Raised when there are issues with configuration values, missing required
    configuration, or invalid configuration formats.

    Attributes:
        config_key: Optional key that caused the configuration error.
    """

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Human-readable error description.
            config_key: Optional key that caused the configuration error.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, cause=cause)
        self.config_key = config_key


class InvalidConfigValueError(ConfigurationError):
    """Exception for invalid configuration values.

    Raised when a configuration value fails validation, is of wrong type,
    or is outside acceptable bounds.

    Attributes:
        config_key: The configuration key with invalid value.
        value: The invalid value that was provided.
        expected: Description of what was expected.
    """

    def __init__(
        self,
        message: str,
        *,
        config_key: str,
        value: Any = None,
        expected: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize invalid config value error.

        Args:
            message: Human-readable error description.
            config_key: The configuration key with invalid value.
            value: The invalid value that was provided.
            expected: Description of what was expected.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        details["value"] = value
        if expected:
            details["expected"] = expected
        super().__init__(message, config_key=config_key, details=details, cause=cause)
        self.value = value
        self.expected = expected


class MissingConfigError(ConfigurationError):
    """Exception for missing required configuration.

    Raised when a required configuration key is not provided.

    Attributes:
        config_key: The missing required configuration key.
    """

    def __init__(
        self,
        config_key: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize missing config error.

        Args:
            config_key: The missing required configuration key.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        message = f"Required configuration key '{config_key}' is missing"
        super().__init__(message, config_key=config_key, details=details, cause=cause)


# =============================================================================
# Validation Execution Errors
# =============================================================================


class ValidationExecutionError(TruthoundIntegrationError):
    """Exception for errors during validation execution.

    Raised when validation fails due to execution issues rather than
    data quality problems.

    Attributes:
        rule_name: Optional name of the rule that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        rule_name: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize validation execution error.

        Args:
            message: Human-readable error description.
            rule_name: Optional name of the rule that failed.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if rule_name:
            details["rule_name"] = rule_name
        super().__init__(message, details=details, cause=cause)
        self.rule_name = rule_name


class RuleExecutionError(ValidationExecutionError):
    """Exception for errors during rule execution.

    Raised when a specific validation rule fails to execute properly
    (not when data fails validation).

    Attributes:
        rule_name: Name of the rule that failed.
        column: Optional column name involved in the error.
    """

    def __init__(
        self,
        message: str,
        *,
        rule_name: str,
        column: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize rule execution error.

        Args:
            message: Human-readable error description.
            rule_name: Name of the rule that failed.
            column: Optional column name involved in the error.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if column:
            details["column"] = column
        super().__init__(message, rule_name=rule_name, details=details, cause=cause)
        self.column = column


class DataAccessError(ValidationExecutionError):
    """Exception for errors accessing data during validation.

    Raised when the validation system cannot read, parse, or access
    the data source.

    Attributes:
        source: Description of the data source that could not be accessed.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize data access error.

        Args:
            message: Human-readable error description.
            source: Description of the data source that could not be accessed.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if source:
            details["source"] = source
        super().__init__(message, details=details, cause=cause)
        self.source = source


# =============================================================================
# Serialization Errors
# =============================================================================


class SerializationError(TruthoundIntegrationError):
    """Exception for serialization-related errors.

    Base exception for all serialization and deserialization errors.

    Attributes:
        target_type: Optional type being serialized to/from.
    """

    def __init__(
        self,
        message: str,
        *,
        target_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize serialization error.

        Args:
            message: Human-readable error description.
            target_type: Optional type being serialized to/from.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if target_type:
            details["target_type"] = target_type
        super().__init__(message, details=details, cause=cause)
        self.target_type = target_type


class SerializeError(SerializationError):
    """Exception for errors during serialization.

    Raised when converting an object to a serialized format fails.
    """

    pass


class DeserializeError(SerializationError):
    """Exception for errors during deserialization.

    Raised when converting a serialized format back to an object fails.
    """

    pass


# =============================================================================
# Platform Connection Errors
# =============================================================================


class PlatformConnectionError(TruthoundIntegrationError):
    """Exception for platform connection errors.

    Raised when there are issues connecting to the workflow platform
    or external services.

    Attributes:
        platform: Name of the platform that failed to connect.
        endpoint: Optional endpoint URL that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        platform: str | None = None,
        endpoint: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize platform connection error.

        Args:
            message: Human-readable error description.
            platform: Name of the platform that failed to connect.
            endpoint: Optional endpoint URL that failed.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if platform:
            details["platform"] = platform
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, details=details, cause=cause)
        self.platform = platform
        self.endpoint = endpoint


class AuthenticationError(PlatformConnectionError):
    """Exception for authentication failures.

    Raised when authentication to a platform or service fails.
    """

    pass


# =============================================================================
# Timeout Errors
# =============================================================================


class IntegrationTimeoutError(TruthoundIntegrationError):
    """Exception for timeout errors.

    Raised when an operation exceeds its configured timeout.

    Attributes:
        timeout_seconds: The timeout value that was exceeded.
        operation: Description of the operation that timed out.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Human-readable error description.
            timeout_seconds: The timeout value that was exceeded.
            operation: Description of the operation that timed out.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, cause=cause)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


# =============================================================================
# Quality Gate Errors
# =============================================================================


class QualityGateError(TruthoundIntegrationError):
    """Exception for quality gate failures.

    Raised when data quality validation results fail to meet defined
    quality gates or thresholds. This is distinct from execution errors;
    the validation ran successfully, but the data quality did not meet
    requirements.

    Attributes:
        failed_count: Number of failed validations.
        pass_rate: Actual pass rate achieved.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_count: int | None = None,
        pass_rate: float | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize quality gate error.

        Args:
            message: Human-readable error description.
            failed_count: Number of failed validations.
            pass_rate: Actual pass rate achieved.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if failed_count is not None:
            details["failed_count"] = failed_count
        if pass_rate is not None:
            details["pass_rate"] = pass_rate
        super().__init__(message, details=details, cause=cause)
        self.failed_count = failed_count
        self.pass_rate = pass_rate


class ThresholdExceededError(QualityGateError):
    """Exception for threshold violations.

    Raised when a specific quality threshold is exceeded.

    Attributes:
        threshold: The threshold value that was exceeded.
        actual: The actual value that exceeded the threshold.
        threshold_type: Type of threshold (e.g., 'pass_rate', 'failure_count').
    """

    def __init__(
        self,
        message: str,
        *,
        threshold: float | None = None,
        actual: float | None = None,
        threshold_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize threshold exceeded error.

        Args:
            message: Human-readable error description.
            threshold: The threshold value that was exceeded.
            actual: The actual value that exceeded the threshold.
            threshold_type: Type of threshold (e.g., 'pass_rate', 'failure_count').
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if threshold is not None:
            details["threshold"] = threshold
        if actual is not None:
            details["actual"] = actual
        if threshold_type:
            details["threshold_type"] = threshold_type
        super().__init__(message, details=details, cause=cause)
        self.threshold = threshold
        self.actual = actual
        self.threshold_type = threshold_type


# =============================================================================
# Utility Functions
# =============================================================================


def wrap_exception(
    exception: Exception,
    wrapper_class: type[TruthoundIntegrationError] = TruthoundIntegrationError,
    message: str | None = None,
    **kwargs: Any,
) -> TruthoundIntegrationError:
    """Wrap an exception in a TruthoundIntegrationError.

    Utility function to convert arbitrary exceptions into the Truthound
    exception hierarchy while preserving the original exception as cause.

    Args:
        exception: The original exception to wrap.
        wrapper_class: The exception class to wrap with.
        message: Optional custom message. Defaults to original exception message.
        **kwargs: Additional arguments to pass to the wrapper class.

    Returns:
        A new exception instance wrapping the original.

    Example:
        >>> try:
        ...     raise ValueError("bad value")
        ... except ValueError as e:
        ...     raise wrap_exception(e, InvalidConfigValueError, config_key="key")
    """
    msg = message if message is not None else str(exception)
    return wrapper_class(msg, cause=exception, **kwargs)
