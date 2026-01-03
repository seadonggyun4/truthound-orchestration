"""Exceptions for Mage Data Quality blocks.

This module defines the exception hierarchy for data quality
operations in Mage AI pipelines.

Exception Hierarchy:
    DataQualityBlockError
    ├── BlockConfigurationError
    ├── BlockExecutionError
    ├── DataLoadError
    └── SLAViolationError
"""

from __future__ import annotations

from typing import Any


class DataQualityBlockError(Exception):
    """Base exception for all data quality block errors.

    All exceptions raised by data quality blocks inherit from this
    base class, allowing for unified exception handling.

    Example:
        >>> try:
        ...     result = transformer.execute(data)
        ... except DataQualityBlockError as e:
        ...     logger.error(f"Data quality error: {e}")
    """

    def __init__(
        self,
        message: str,
        *,
        block_uuid: str | None = None,
        pipeline_uuid: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize exception.

        Args:
            message: Error message.
            block_uuid: UUID of the block that raised the error.
            pipeline_uuid: UUID of the pipeline.
            details: Additional error details.
        """
        super().__init__(message)
        self.message = message
        self.block_uuid = block_uuid
        self.pipeline_uuid = pipeline_uuid
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation."""
        parts = [self.message]
        if self.block_uuid:
            parts.append(f"block_uuid={self.block_uuid}")
        if self.pipeline_uuid:
            parts.append(f"pipeline_uuid={self.pipeline_uuid}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with exception details.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "block_uuid": self.block_uuid,
            "pipeline_uuid": self.pipeline_uuid,
            "details": self.details,
        }


class BlockConfigurationError(DataQualityBlockError):
    """Exception raised for block configuration errors.

    This exception indicates that the block configuration is invalid
    or contains incompatible settings.

    Example:
        >>> if config.timeout_seconds < 0:
        ...     raise BlockConfigurationError(
        ...         "Timeout must be positive",
        ...         field="timeout_seconds",
        ...         value=config.timeout_seconds,
        ...     )
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        block_uuid: str | None = None,
        pipeline_uuid: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message.
            field: Name of the invalid configuration field.
            value: The invalid value.
            block_uuid: UUID of the block.
            pipeline_uuid: UUID of the pipeline.
            details: Additional error details.
        """
        super().__init__(
            message,
            block_uuid=block_uuid,
            pipeline_uuid=pipeline_uuid,
            details=details,
        )
        self.field = field
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["field"] = self.field
        result["value"] = self.value
        return result


class BlockExecutionError(DataQualityBlockError):
    """Exception raised during block execution.

    This exception indicates that an error occurred while executing
    the data quality check, profile, or learn operation.

    Example:
        >>> try:
        ...     result = engine.check(data, rules)
        ... except Exception as e:
        ...     raise BlockExecutionError(
        ...         "Check operation failed",
        ...         original_error=e,
        ...     )
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: BaseException | None = None,
        operation: str | None = None,
        block_uuid: str | None = None,
        pipeline_uuid: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize execution error.

        Args:
            message: Error message.
            original_error: The underlying exception.
            operation: The operation that failed (check, profile, learn).
            block_uuid: UUID of the block.
            pipeline_uuid: UUID of the pipeline.
            details: Additional error details.
        """
        super().__init__(
            message,
            block_uuid=block_uuid,
            pipeline_uuid=pipeline_uuid,
            details=details,
        )
        self.original_error = original_error
        self.operation = operation

    def __str__(self) -> str:
        """Return string representation."""
        base = super().__str__()
        if self.original_error:
            return f"{base} | caused_by={type(self.original_error).__name__}: {self.original_error}"
        return base

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["operation"] = self.operation
        if self.original_error:
            result["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error),
            }
        return result


class DataLoadError(DataQualityBlockError):
    """Exception raised when data loading fails.

    This exception indicates that the input data could not be loaded
    or parsed into the expected format.

    Example:
        >>> try:
        ...     df = load_from_source(config)
        ... except Exception as e:
        ...     raise DataLoadError(
        ...         "Failed to load data from source",
        ...         source=config.source_name,
        ...         original_error=e,
        ...     )
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        original_error: BaseException | None = None,
        block_uuid: str | None = None,
        pipeline_uuid: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize data load error.

        Args:
            message: Error message.
            source: Data source identifier.
            original_error: The underlying exception.
            block_uuid: UUID of the block.
            pipeline_uuid: UUID of the pipeline.
            details: Additional error details.
        """
        super().__init__(
            message,
            block_uuid=block_uuid,
            pipeline_uuid=pipeline_uuid,
            details=details,
        )
        self.source = source
        self.original_error = original_error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["source"] = self.source
        if self.original_error:
            result["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error),
            }
        return result


class SLAViolationError(DataQualityBlockError):
    """Exception raised when SLA violations are critical.

    This exception is raised when SLA violations meet the criteria
    for raising an error (e.g., fail_on_critical=True and critical
    violations detected).

    Example:
        >>> if violations and config.fail_on_critical:
        ...     raise SLAViolationError(
        ...         "Critical SLA violations detected",
        ...         violations=violations,
        ...     )
    """

    def __init__(
        self,
        message: str,
        *,
        violations: list[dict[str, Any]] | None = None,
        block_uuid: str | None = None,
        pipeline_uuid: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize SLA violation error.

        Args:
            message: Error message.
            violations: List of SLA violations.
            block_uuid: UUID of the block.
            pipeline_uuid: UUID of the pipeline.
            details: Additional error details.
        """
        super().__init__(
            message,
            block_uuid=block_uuid,
            pipeline_uuid=pipeline_uuid,
            details=details,
        )
        self.violations = violations or []

    @property
    def violation_count(self) -> int:
        """Number of violations."""
        return len(self.violations)

    @property
    def critical_count(self) -> int:
        """Number of critical violations."""
        return sum(
            1
            for v in self.violations
            if v.get("alert_level") == "critical"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["violations"] = self.violations
        result["violation_count"] = self.violation_count
        result["critical_count"] = self.critical_count
        return result


class SensorTimeoutError(DataQualityBlockError):
    """Exception raised when a sensor times out.

    This exception is raised when a sensor's poke loop exceeds the
    configured timeout or max poke attempts.

    Example:
        >>> if elapsed > config.timeout_seconds:
        ...     raise SensorTimeoutError(
        ...         "Sensor timed out",
        ...         timeout_seconds=config.timeout_seconds,
        ...         poke_count=poke_count,
        ...     )
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float | None = None,
        poke_count: int = 0,
        block_uuid: str | None = None,
        pipeline_uuid: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize sensor timeout error.

        Args:
            message: Error message.
            timeout_seconds: Configured timeout in seconds.
            poke_count: Number of pokes performed before timeout.
            block_uuid: UUID of the block.
            pipeline_uuid: UUID of the pipeline.
            details: Additional error details.
        """
        super().__init__(
            message,
            block_uuid=block_uuid,
            pipeline_uuid=pipeline_uuid,
            details=details,
        )
        self.timeout_seconds = timeout_seconds
        self.poke_count = poke_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result["timeout_seconds"] = self.timeout_seconds
        result["poke_count"] = self.poke_count
        return result
