"""Type definitions and output containers for truthound-prefect.

This module provides generic output containers for wrapping data quality
results with their associated data, following the immutable dataclass pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")


class QualityCheckMode(str, Enum):
    """Mode for quality checking in flows and tasks."""

    BEFORE = "before"  # Check before processing
    AFTER = "after"  # Check after processing
    BOTH = "both"  # Check before and after
    NONE = "none"  # Skip quality checks


class OperationType(str, Enum):
    """Type of data quality operation."""

    CHECK = "check"
    PROFILE = "profile"
    LEARN = "learn"


class OperationStatus(str, Enum):
    """Status of a data quality operation."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class DataQualityOutput(Generic[DataT, ResultT]):
    """Generic container for data with quality result.

    This is the base output type that wraps any data with its associated
    quality check result and metadata.

    Type Parameters:
        DataT: Type of the data being validated.
        ResultT: Type of the quality result.

    Attributes:
        data: The data that was validated.
        result: The quality check result (None if not checked).
        metadata: Additional metadata about the operation.
        is_success: Whether the quality check passed.
        timestamp: When the operation was performed.
    """

    data: DataT
    result: ResultT | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

    def with_metadata(self, **kwargs: Any) -> DataQualityOutput[DataT, ResultT]:
        """Return a new output with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return DataQualityOutput(
            data=self.data,
            result=self.result,
            metadata=new_metadata,
            is_success=self.is_success,
            timestamp=self.timestamp,
        )

    def with_result(self, result: ResultT, is_success: bool = True) -> DataQualityOutput[DataT, ResultT]:
        """Return a new output with a different result."""
        return DataQualityOutput(
            data=self.data,
            result=result,
            metadata=self.metadata,
            is_success=is_success,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_type": type(self.data).__name__,
            "result": self.result if isinstance(self.result, dict) else None,
            "metadata": self.metadata,
            "is_success": self.is_success,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class QualityCheckOutput(Generic[DataT]):
    """Output container for quality check results.

    Specialized container for check operations.

    Attributes:
        data: The data that was checked.
        result: The check result as a dictionary.
        is_success: Whether all checks passed.
        passed_count: Number of passed rules.
        failed_count: Number of failed rules.
        failure_rate: Rate of failures (0.0 to 1.0).
        metadata: Additional metadata.
        timestamp: When the check was performed.
    """

    data: DataT
    result: dict[str, Any] = field(default_factory=dict)
    is_success: bool = True
    passed_count: int = 0
    failed_count: int = 0
    failure_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_result(
        cls,
        data: DataT,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> QualityCheckOutput[DataT]:
        """Create output from a serialized check result."""
        return cls(
            data=data,
            result=result,
            is_success=result.get("is_success", True),
            passed_count=result.get("passed_count", 0),
            failed_count=result.get("failed_count", 0),
            failure_rate=result.get("failure_rate", 0.0),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_type": type(self.data).__name__,
            "result": self.result,
            "is_success": self.is_success,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "failure_rate": self.failure_rate,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_prefect_artifact(self) -> dict[str, Any]:
        """Convert to Prefect artifact format."""
        return {
            "type": "table",
            "data": [
                {"metric": "Status", "value": "PASSED" if self.is_success else "FAILED"},
                {"metric": "Passed Rules", "value": self.passed_count},
                {"metric": "Failed Rules", "value": self.failed_count},
                {"metric": "Failure Rate", "value": f"{self.failure_rate:.2%}"},
            ],
        }


@dataclass(frozen=True, slots=True)
class ProfileOutput(Generic[DataT]):
    """Output container for profile results.

    Specialized container for profile operations.

    Attributes:
        data: The data that was profiled.
        result: The profile result as a dictionary.
        row_count: Number of rows in the data.
        column_count: Number of columns in the data.
        metadata: Additional metadata.
        timestamp: When the profile was performed.
    """

    data: DataT
    result: dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    column_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_result(
        cls,
        data: DataT,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ProfileOutput[DataT]:
        """Create output from a serialized profile result."""
        return cls(
            data=data,
            result=result,
            row_count=result.get("row_count", 0),
            column_count=result.get("column_count", 0),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_type": type(self.data).__name__,
            "result": self.result,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class LearnOutput(Generic[DataT]):
    """Output container for learn results.

    Specialized container for schema learning operations.

    Attributes:
        data: The data that was analyzed.
        result: The learn result as a dictionary.
        rules_count: Number of rules learned.
        metadata: Additional metadata.
        timestamp: When the learning was performed.
    """

    data: DataT
    result: dict[str, Any] = field(default_factory=dict)
    rules_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_result(
        cls,
        data: DataT,
        result: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> LearnOutput[DataT]:
        """Create output from a serialized learn result."""
        rules = result.get("rules", [])
        return cls(
            data=data,
            result=result,
            rules_count=len(rules),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_type": type(self.data).__name__,
            "result": self.result,
            "rules_count": self.rules_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# Type aliases for convenience
AnyDataQualityOutput = DataQualityOutput[Any, Any]
AnyQualityCheckOutput = QualityCheckOutput[Any]
AnyProfileOutput = ProfileOutput[Any]
AnyLearnOutput = LearnOutput[Any]


__all__ = [
    # Enums
    "QualityCheckMode",
    "OperationType",
    "OperationStatus",
    # Output types
    "DataQualityOutput",
    "QualityCheckOutput",
    "ProfileOutput",
    "LearnOutput",
    # Type aliases
    "AnyDataQualityOutput",
    "AnyQualityCheckOutput",
    "AnyProfileOutput",
    "AnyLearnOutput",
]
