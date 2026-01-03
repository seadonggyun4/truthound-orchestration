"""Exception hierarchy for Kestra data quality integration.

This module defines a comprehensive exception hierarchy for handling
errors in data quality operations within Kestra workflows.

Exception Hierarchy:
    DataQualityError (base)
    ├── ConfigurationError (configuration validation)
    ├── EngineError (engine execution)
    ├── ScriptError (Kestra script execution)
    ├── FlowError (Kestra flow generation)
    ├── OutputError (Kestra output handling)
    └── SLAViolationError (SLA violations)

Example:
    >>> from truthound_kestra.utils.exceptions import DataQualityError
    >>>
    >>> try:
    ...     result = check_quality(data)
    ... except DataQualityError as e:
    ...     print(f"Quality check failed: {e}")
    ...     print(f"Details: {e.to_dict()}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from truthound_kestra.sla.config import SLAViolation

__all__ = [
    "DataQualityError",
    "ConfigurationError",
    "EngineError",
    "ScriptError",
    "FlowError",
    "OutputError",
    "SLAViolationError",
    "SerializationError",
]


@dataclass
class DataQualityError(Exception):
    """Base exception for all data quality errors in Kestra integration.

    This is the root exception class for the truthound-kestra package.
    All other exceptions inherit from this class.

    Attributes:
        message: Human-readable error description.
        result: Optional check result data associated with the error.
        metadata: Additional context information for debugging.

    Example:
        >>> raise DataQualityError(
        ...     message="Quality check failed",
        ...     result={"status": "FAILED", "failed_count": 5},
        ...     metadata={"task_id": "check_users"}
        ... )
    """

    message: str
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the exception with the message."""
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation with result summary if available."""
        if self.result:
            failed_count = self.result.get("failed_count", 0)
            status = self.result.get("status", "UNKNOWN")
            return f"{self.message} (status={status}, failed={failed_count})"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"result={self.result!r}, "
            f"metadata={self.metadata!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization.

        Returns:
            Dictionary containing exception details.
        """
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "result": self.result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataQualityError:
        """Create exception from dictionary.

        Args:
            data: Dictionary containing exception details.

        Returns:
            DataQualityError instance.
        """
        return cls(
            message=data.get("message", "Unknown error"),
            result=data.get("result"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConfigurationError(DataQualityError):
    """Exception raised for configuration validation errors.

    Attributes:
        field: Name of the configuration field that caused the error.
        value: The invalid value that was provided.
        reason: Explanation of why the value is invalid.

    Example:
        >>> raise ConfigurationError(
        ...     message="Invalid timeout value",
        ...     field="timeout_seconds",
        ...     value=-1,
        ...     reason="Timeout must be a positive number"
        ... )
    """

    field: str | None = None
    value: Any = None
    reason: str | None = None

    def __str__(self) -> str:
        """Return detailed configuration error message."""
        parts = [self.message]
        if self.field:
            parts.append(f"field={self.field!r}")
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        if self.reason:
            parts.append(f"reason={self.reason!r}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "field": self.field,
            "value": self.value,
            "reason": self.reason,
        })
        return base


@dataclass
class EngineError(DataQualityError):
    """Exception raised when a data quality engine operation fails.

    Attributes:
        engine_name: Name of the engine that failed.
        operation: The operation that was being performed (check, profile, learn).
        original_error: The underlying exception that caused the failure.

    Example:
        >>> try:
        ...     result = engine.check(data, rules)
        ... except Exception as e:
        ...     raise EngineError(
        ...         message="Engine check failed",
        ...         engine_name="truthound",
        ...         operation="check",
        ...         original_error=e
        ...     )
    """

    engine_name: str | None = None
    operation: str | None = None
    original_error: Exception | None = None

    def __str__(self) -> str:
        """Return detailed engine error message."""
        parts = [self.message]
        if self.engine_name:
            parts.append(f"engine={self.engine_name!r}")
        if self.operation:
            parts.append(f"operation={self.operation!r}")
        if self.original_error:
            parts.append(f"caused_by={type(self.original_error).__name__}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "engine_name": self.engine_name,
            "operation": self.operation,
            "original_error": (
                str(self.original_error) if self.original_error else None
            ),
            "original_error_type": (
                type(self.original_error).__name__ if self.original_error else None
            ),
        })
        return base


@dataclass
class ScriptError(DataQualityError):
    """Exception raised when a Kestra script execution fails.

    This exception is specific to errors that occur during the execution
    of Python scripts within Kestra tasks.

    Attributes:
        script_name: Name or identifier of the script that failed.
        task_id: Kestra task ID where the script was executed.
        execution_id: Kestra execution ID for the flow run.

    Example:
        >>> raise ScriptError(
        ...     message="Failed to load input data",
        ...     script_name="check_quality_script",
        ...     task_id="validate_users",
        ...     execution_id="abc123"
        ... )
    """

    script_name: str | None = None
    task_id: str | None = None
    execution_id: str | None = None

    def __str__(self) -> str:
        """Return detailed script error message."""
        parts = [self.message]
        if self.script_name:
            parts.append(f"script={self.script_name!r}")
        if self.task_id:
            parts.append(f"task_id={self.task_id!r}")
        if self.execution_id:
            parts.append(f"execution_id={self.execution_id!r}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "script_name": self.script_name,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
        })
        return base


@dataclass
class FlowError(DataQualityError):
    """Exception raised when Kestra flow generation or execution fails.

    Attributes:
        flow_id: Kestra flow identifier.
        namespace: Kestra namespace where the flow resides.
        trigger: The trigger type if the error is trigger-related.

    Example:
        >>> raise FlowError(
        ...     message="Failed to generate flow YAML",
        ...     flow_id="data_quality_pipeline",
        ...     namespace="production"
        ... )
    """

    flow_id: str | None = None
    namespace: str | None = None
    trigger: str | None = None

    def __str__(self) -> str:
        """Return detailed flow error message."""
        parts = [self.message]
        if self.flow_id:
            parts.append(f"flow_id={self.flow_id!r}")
        if self.namespace:
            parts.append(f"namespace={self.namespace!r}")
        if self.trigger:
            parts.append(f"trigger={self.trigger!r}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "flow_id": self.flow_id,
            "namespace": self.namespace,
            "trigger": self.trigger,
        })
        return base


@dataclass
class OutputError(DataQualityError):
    """Exception raised when Kestra output handling fails.

    This exception is raised when there are issues with producing
    or consuming Kestra task outputs.

    Attributes:
        output_name: Name of the output that failed.
        output_type: Expected type of the output (e.g., 'json', 'file').

    Example:
        >>> raise OutputError(
        ...     message="Failed to serialize output to JSON",
        ...     output_name="check_result",
        ...     output_type="json"
        ... )
    """

    output_name: str | None = None
    output_type: str | None = None

    def __str__(self) -> str:
        """Return detailed output error message."""
        parts = [self.message]
        if self.output_name:
            parts.append(f"output={self.output_name!r}")
        if self.output_type:
            parts.append(f"type={self.output_type!r}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "output_name": self.output_name,
            "output_type": self.output_type,
        })
        return base


@dataclass
class SLAViolationError(DataQualityError):
    """Exception raised when SLA thresholds are violated.

    This exception aggregates one or more SLA violations that occurred
    during a data quality check.

    Attributes:
        violations: List of SLAViolation objects describing each violation.

    Example:
        >>> from truthound_kestra.sla.config import SLAViolation, SLAViolationType
        >>> raise SLAViolationError(
        ...     message="SLA thresholds exceeded",
        ...     violations=[
        ...         SLAViolation(
        ...             violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
        ...             message="Failure rate 15% exceeds maximum 5%",
        ...             threshold=0.05,
        ...             actual=0.15
        ...         )
        ...     ]
        ... )
    """

    violations: list[SLAViolation] = field(default_factory=list)

    def __str__(self) -> str:
        """Return summary of SLA violations."""
        if not self.violations:
            return self.message
        return f"{self.message} ({len(self.violations)} violation(s))"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base["violations"] = [
            v.to_dict() if hasattr(v, "to_dict") else str(v)
            for v in self.violations
        ]
        base["violation_count"] = len(self.violations)
        return base


@dataclass
class SerializationError(DataQualityError):
    """Exception raised when serialization or deserialization fails.

    Attributes:
        format: The serialization format that failed (e.g., 'json', 'yaml').
        direction: Whether it was 'serialize' or 'deserialize'.

    Example:
        >>> raise SerializationError(
        ...     message="Failed to serialize result to JSON",
        ...     format="json",
        ...     direction="serialize"
        ... )
    """

    format: str | None = None
    direction: str | None = None

    def __str__(self) -> str:
        """Return detailed serialization error message."""
        parts = [self.message]
        if self.format:
            parts.append(f"format={self.format!r}")
        if self.direction:
            parts.append(f"direction={self.direction!r}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "format": self.format,
            "direction": self.direction,
        })
        return base
