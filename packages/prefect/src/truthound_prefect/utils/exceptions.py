"""Custom exceptions for truthound-prefect.

Exception hierarchy:
    DataQualityError (base)
    ├── ConfigurationError (invalid config values)
    ├── EngineError (engine execution failures)
    ├── BlockError (Prefect Block errors)
    └── SLAViolationError (SLA threshold violations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from truthound_prefect.sla.config import SLAViolation


@dataclass
class DataQualityError(Exception):
    """Base exception for all data quality errors.

    Attributes:
        message: Human-readable error message.
        result: Optional check result that caused the error.
        metadata: Additional context about the error.
    """

    message: str
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "result": self.result,
            "metadata": self.metadata,
        }


@dataclass
class ConfigurationError(DataQualityError):
    """Exception raised for configuration errors.

    Attributes:
        field: The configuration field that has an error.
        value: The invalid value.
        reason: Explanation of why the value is invalid.
    """

    field: str | None = None
    value: Any = None
    reason: str | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.field:
            parts.append(f"Field: {self.field}")
        if self.value is not None:
            parts.append(f"Value: {self.value!r}")
        if self.reason:
            parts.append(f"Reason: {self.reason}")
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
    """Exception raised for engine execution errors.

    Attributes:
        engine_name: Name of the engine that failed.
        operation: The operation that failed (check, profile, learn).
        original_error: The original exception that was caught.
    """

    engine_name: str | None = None
    operation: str | None = None
    original_error: Exception | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.engine_name:
            parts.append(f"Engine: {self.engine_name}")
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.original_error:
            parts.append(f"Cause: {self.original_error!s}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "engine_name": self.engine_name,
            "operation": self.operation,
            "original_error": str(self.original_error) if self.original_error else None,
        })
        return base


@dataclass
class BlockError(DataQualityError):
    """Exception raised for Prefect Block errors.

    Attributes:
        block_name: Name of the block that failed.
        block_type: Type of the block (e.g., DataQualityBlock).
        operation: The operation that failed.
    """

    block_name: str | None = None
    block_type: str | None = None
    operation: str | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.block_name:
            parts.append(f"Block: {self.block_name}")
        if self.block_type:
            parts.append(f"Type: {self.block_type}")
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "block_name": self.block_name,
            "block_type": self.block_type,
            "operation": self.operation,
        })
        return base


@dataclass
class SLAViolationError(DataQualityError):
    """Exception raised when SLA thresholds are violated.

    Attributes:
        violations: List of SLA violations that occurred.
    """

    violations: list[SLAViolation] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.violations:
            return self.message
        violation_msgs = [f"- {v.message}" for v in self.violations]
        return f"{self.message}\nViolations:\n" + "\n".join(violation_msgs)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        base = super().to_dict()
        base["violations"] = [
            {
                "type": v.violation_type.value,
                "message": v.message,
                "threshold": v.threshold,
                "actual": v.actual,
                "alert_level": v.alert_level.value,
            }
            for v in self.violations
        ]
        return base


__all__ = [
    "DataQualityError",
    "ConfigurationError",
    "EngineError",
    "BlockError",
    "SLAViolationError",
]
