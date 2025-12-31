"""Type Definitions for Dagster Integration.

This module provides type definitions and output containers
for data quality operations in Dagster.

Example:
    >>> from truthound_dagster.utils.types import DataQualityOutput
    >>>
    >>> output = DataQualityOutput(
    ...     data=df,
    ...     result=check_result,
    ...     metadata={"source": "s3"},
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from common.base import CheckResult, LearnResult, ProfileResult

# Type variables for generic output types
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class DataQualityOutput(Generic[DataT, ResultT]):
    """Generic output container for data quality operations.

    This class wraps the original data along with quality check
    results and metadata. It can be used as the return type
    for quality-checked assets.

    Parameters
    ----------
    data : DataT
        The original or processed data.

    result : ResultT | None
        The quality check/profile/learn result.

    metadata : dict[str, Any]
        Additional metadata about the operation.

    is_success : bool
        Whether the operation succeeded.

    Example:
        >>> output = DataQualityOutput(
        ...     data=df,
        ...     result=check_result,
        ...     metadata={"row_count": 1000},
        ... )
        >>> if output.is_success:
        ...     process(output.data)
    """

    data: DataT
    result: ResultT | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_success: bool = True

    def with_metadata(self, **kwargs: Any) -> DataQualityOutput[DataT, ResultT]:
        """Create new output with additional metadata.

        Args:
            **kwargs: Metadata to add.

        Returns:
            DataQualityOutput: New output with merged metadata.
        """
        new_metadata = {**self.metadata, **kwargs}
        return DataQualityOutput(
            data=self.data,
            result=self.result,
            metadata=new_metadata,
            is_success=self.is_success,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            dict[str, Any]: Dictionary representation.
        """
        result_dict: dict[str, Any] | None = None
        if self.result is not None and hasattr(self.result, "__dict__"):
            result_dict = {
                k: v for k, v in self.result.__dict__.items() if not k.startswith("_")
            }

        return {
            "data": self.data,
            "result": result_dict,
            "metadata": self.metadata,
            "is_success": self.is_success,
        }


@dataclass(frozen=True, slots=True)
class QualityCheckOutput(Generic[DataT]):
    """Output container specifically for quality check operations.

    Parameters
    ----------
    data : DataT
        The checked data.

    result : CheckResult
        The check result.

    passed : bool
        Whether all checks passed.

    failure_count : int
        Number of failed checks.

    warning_count : int
        Number of warnings.

    metadata : dict[str, Any]
        Additional metadata.

    Example:
        >>> output = QualityCheckOutput(
        ...     data=df,
        ...     result=check_result,
        ...     passed=True,
        ...     failure_count=0,
        ... )
    """

    data: DataT
    result: CheckResult
    passed: bool = True
    failure_count: int = 0
    warning_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        data: DataT,
        result: CheckResult,
        metadata: dict[str, Any] | None = None,
    ) -> QualityCheckOutput[DataT]:
        """Create output from CheckResult.

        Args:
            data: The checked data.
            result: The check result.
            metadata: Additional metadata.

        Returns:
            QualityCheckOutput: New output instance.
        """
        return cls(
            data=data,
            result=result,
            passed=result.is_success,
            failure_count=result.failed_count,
            warning_count=result.warning_count,
            metadata=metadata or {},
        )

    @property
    def is_success(self) -> bool:
        """Whether the check was successful."""
        return self.passed

    @property
    def failure_rate(self) -> float:
        """Get failure rate from result."""
        return self.result.failure_rate

    def to_dagster_metadata(self) -> dict[str, Any]:
        """Convert to Dagster metadata format.

        Returns:
            dict[str, Any]: Dagster-compatible metadata.
        """
        return {
            "status": self.result.status.value,
            "passed": self.passed,
            "passed_count": self.result.passed_count,
            "failed_count": self.failure_count,
            "warning_count": self.warning_count,
            "failure_rate": self.failure_rate,
            "execution_time_ms": self.result.execution_time_ms,
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class ProfileOutput(Generic[DataT]):
    """Output container for profiling operations.

    Parameters
    ----------
    data : DataT
        The profiled data.

    result : ProfileResult
        The profile result.

    row_count : int
        Number of rows in the data.

    column_count : int
        Number of columns in the data.

    metadata : dict[str, Any]
        Additional metadata.

    Example:
        >>> output = ProfileOutput(
        ...     data=df,
        ...     result=profile_result,
        ...     row_count=1000,
        ...     column_count=10,
        ... )
    """

    data: DataT
    result: ProfileResult
    row_count: int = 0
    column_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        data: DataT,
        result: ProfileResult,
        metadata: dict[str, Any] | None = None,
    ) -> ProfileOutput[DataT]:
        """Create output from ProfileResult.

        Args:
            data: The profiled data.
            result: The profile result.
            metadata: Additional metadata.

        Returns:
            ProfileOutput: New output instance.
        """
        return cls(
            data=data,
            result=result,
            row_count=result.row_count,
            column_count=result.column_count,
            metadata=metadata or {},
        )

    def get_column_profile(self, column_name: str) -> dict[str, Any] | None:
        """Get profile for a specific column.

        Args:
            column_name: Name of the column.

        Returns:
            dict[str, Any] | None: Column profile or None.
        """
        for col in self.result.columns:
            if col.column_name == column_name:
                return {
                    "column_name": col.column_name,
                    "dtype": str(col.dtype),
                    "null_count": col.null_count,
                    "null_percentage": col.null_percentage,
                    "unique_count": col.unique_count,
                    "unique_percentage": col.unique_percentage,
                }
        return None

    def to_dagster_metadata(self) -> dict[str, Any]:
        """Convert to Dagster metadata format.

        Returns:
            dict[str, Any]: Dagster-compatible metadata.
        """
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "execution_time_ms": self.result.execution_time_ms,
            "columns": [col.column_name for col in self.result.columns],
            **self.metadata,
        }


@dataclass(frozen=True, slots=True)
class LearnOutput(Generic[DataT]):
    """Output container for schema learning operations.

    Parameters
    ----------
    data : DataT
        The analyzed data.

    result : LearnResult
        The learn result.

    rule_count : int
        Number of learned rules.

    metadata : dict[str, Any]
        Additional metadata.

    Example:
        >>> output = LearnOutput(
        ...     data=df,
        ...     result=learn_result,
        ...     rule_count=15,
        ... )
    """

    data: DataT
    result: LearnResult
    rule_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        data: DataT,
        result: LearnResult,
        metadata: dict[str, Any] | None = None,
    ) -> LearnOutput[DataT]:
        """Create output from LearnResult.

        Args:
            data: The analyzed data.
            result: The learn result.
            metadata: Additional metadata.

        Returns:
            LearnOutput: New output instance.
        """
        return cls(
            data=data,
            result=result,
            rule_count=len(result.rules),
            metadata=metadata or {},
        )

    def get_rules_for_column(self, column_name: str) -> list[dict[str, Any]]:
        """Get learned rules for a specific column.

        Args:
            column_name: Name of the column.

        Returns:
            list[dict[str, Any]]: Rules for the column.
        """
        return [
            {
                "column": rule.column,
                "rule_type": rule.rule_type,
                "confidence": rule.confidence,
                "parameters": rule.parameters,
            }
            for rule in self.result.rules
            if rule.column == column_name
        ]

    def get_high_confidence_rules(
        self,
        min_confidence: float = 0.9,
    ) -> list[dict[str, Any]]:
        """Get rules with high confidence.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            list[dict[str, Any]]: High confidence rules.
        """
        return [
            {
                "column": rule.column,
                "rule_type": rule.rule_type,
                "confidence": rule.confidence,
                "parameters": rule.parameters,
            }
            for rule in self.result.rules
            if rule.confidence >= min_confidence
        ]

    def to_dagster_metadata(self) -> dict[str, Any]:
        """Convert to Dagster metadata format.

        Returns:
            dict[str, Any]: Dagster-compatible metadata.
        """
        return {
            "rule_count": self.rule_count,
            "execution_time_ms": self.result.execution_time_ms,
            "columns_with_rules": list({rule.column for rule in self.result.rules}),
            **self.metadata,
        }


# Type aliases for common use cases
AnyDataQualityOutput = DataQualityOutput[Any, Any]
AnyQualityCheckOutput = QualityCheckOutput[Any]
AnyProfileOutput = ProfileOutput[Any]
AnyLearnOutput = LearnOutput[Any]
