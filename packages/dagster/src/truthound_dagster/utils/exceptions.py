"""Custom Exceptions for Dagster Integration.

This module provides custom exception types for data quality
operations in Dagster pipelines.

Example:
    >>> from truthound_dagster.utils import DataQualityError
    >>>
    >>> try:
    ...     check_quality(data)
    ... except DataQualityError as e:
    ...     print(f"Quality check failed: {e.message}")
    ...     print(f"Failures: {e.result.failed_count}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from common.base import CheckResult

    from truthound_dagster.sla.config import SLAViolation


class DataQualityError(Exception):
    """Base exception for data quality errors.

    This exception is raised when data quality operations fail.
    It includes the check result for detailed error information.

    Attributes:
        message: Error message.
        result: Optional check result with failure details.
        metadata: Additional context.

    Example:
        >>> try:
        ...     check_quality(data)
        ... except DataQualityError as e:
        ...     print(f"Failed: {e.message}")
        ...     if e.result:
        ...         print(f"Failures: {e.result.failed_count}")
    """

    def __init__(
        self,
        message: str,
        result: CheckResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize data quality error.

        Args:
            message: Error message.
            result: Optional check result.
            metadata: Additional context.
        """
        super().__init__(message)
        self.message = message
        self.result = result
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """Return string representation."""
        if self.result:
            return (
                f"{self.message} "
                f"(failed={self.result.failed_count}, "
                f"rate={self.result.failure_rate:.2%})"
            )
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data: dict[str, Any] = {
            "message": self.message,
            "metadata": self.metadata,
        }
        if self.result:
            data["result"] = {
                "status": self.result.status.value,
                "passed_count": self.result.passed_count,
                "failed_count": self.result.failed_count,
                "failure_rate": self.result.failure_rate,
            }
        return data


class ConfigurationError(DataQualityError):
    """Exception for configuration errors.

    Raised when resource or op configuration is invalid.

    Attributes:
        field: Name of the invalid field.
        value: The invalid value.
        reason: Why the value is invalid.

    Example:
        >>> raise ConfigurationError(
        ...     message="Invalid threshold",
        ...     field="warning_threshold",
        ...     value=1.5,
        ...     reason="Must be between 0 and 1",
        ... )
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        reason: str | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message.
            field: Name of invalid field.
            value: The invalid value.
            reason: Why value is invalid.
        """
        super().__init__(message)
        self.field = field
        self.value = value
        self.reason = reason

    def __str__(self) -> str:
        """Return string representation."""
        parts = [self.message]
        if self.field:
            parts.append(f"field={self.field}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        return " ".join(parts)


class EngineError(DataQualityError):
    """Exception for engine errors.

    Raised when the data quality engine encounters an error.

    Attributes:
        engine_name: Name of the engine.
        operation: The operation that failed.
        original_error: The underlying exception.

    Example:
        >>> raise EngineError(
        ...     message="Engine check failed",
        ...     engine_name="truthound",
        ...     operation="check",
        ...     original_error=original_exception,
        ... )
    """

    def __init__(
        self,
        message: str,
        engine_name: str | None = None,
        operation: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize engine error.

        Args:
            message: Error message.
            engine_name: Name of the engine.
            operation: The failing operation.
            original_error: The underlying exception.
        """
        super().__init__(message)
        self.engine_name = engine_name
        self.operation = operation
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation."""
        parts = [self.message]
        if self.engine_name:
            parts.append(f"engine={self.engine_name}")
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.original_error:
            parts.append(f"cause={type(self.original_error).__name__}")
        return " ".join(parts)


class SLAViolationError(DataQualityError):
    """Exception for SLA violations.

    Raised when SLA thresholds are exceeded and alert_on_violation is True.

    Attributes:
        violations: List of SLA violations.

    Example:
        >>> raise SLAViolationError(
        ...     message="SLA violated",
        ...     violations=detected_violations,
        ... )
    """

    def __init__(
        self,
        message: str,
        violations: list[SLAViolation] | None = None,
    ) -> None:
        """Initialize SLA violation error.

        Args:
            message: Error message.
            violations: List of violations.
        """
        super().__init__(message)
        self.violations = violations or []

    def __str__(self) -> str:
        """Return string representation."""
        if self.violations:
            violation_msgs = [v.message for v in self.violations[:3]]
            suffix = f" (+{len(self.violations) - 3} more)" if len(self.violations) > 3 else ""
            return f"{self.message}: {', '.join(violation_msgs)}{suffix}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "violations": [v.to_dict() for v in self.violations],
        }
