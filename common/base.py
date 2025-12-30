"""Base types for Truthound Integrations.

This module defines the core protocols, enums, configuration objects, and result
types used across all platform integrations. It follows a protocol-based design
for maximum flexibility and uses immutable dataclasses for thread-safety.

Key Components:
    - Protocols: WorkflowIntegration, AsyncWorkflowIntegration
    - Enums: CheckStatus, FailureAction, Severity
    - Config: CheckConfig, ProfileConfig, LearnConfig
    - Results: CheckResult, ProfileResult, LearnResult, ValidationFailure

Design Principles:
    1. Protocol-based: Use structural typing for flexible implementations
    2. Immutable: All config and result types are frozen dataclasses
    3. Type-safe: Full type annotations for static analysis
    4. Composable: Builder-style methods for creating modified instances

Example:
    >>> from common.base import CheckConfig, FailureAction
    >>> config = CheckConfig(
    ...     rules=({"type": "not_null", "column": "id"},),
    ...     failure_action=FailureAction.RAISE,
    ... )
    >>> new_config = config.with_timeout(60)
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl


# =============================================================================
# Enums
# =============================================================================


class CheckStatus(Enum):
    """Status of a validation check execution.

    Attributes:
        PASSED: All validations passed successfully.
        FAILED: One or more validations failed.
        WARNING: Validations passed but with warnings.
        SKIPPED: Validation was skipped (e.g., due to sampling or conditions).
        ERROR: An error occurred during validation execution.
    """

    PASSED = auto()
    FAILED = auto()
    WARNING = auto()
    SKIPPED = auto()
    ERROR = auto()

    def is_success(self) -> bool:
        """Check if the status represents a successful outcome."""
        return self in (CheckStatus.PASSED, CheckStatus.WARNING, CheckStatus.SKIPPED)

    def is_terminal_failure(self) -> bool:
        """Check if the status represents a terminal failure."""
        return self in (CheckStatus.FAILED, CheckStatus.ERROR)


class FailureAction(Enum):
    """Action to take when validation fails.

    Attributes:
        RAISE: Raise an exception immediately.
        WARN: Log a warning but continue execution.
        LOG: Log the failure at info level and continue.
        CONTINUE: Silently continue without logging.
    """

    RAISE = auto()
    WARN = auto()
    LOG = auto()
    CONTINUE = auto()


class Severity(Enum):
    """Severity level for validation failures.

    Attributes:
        CRITICAL: Critical failure that must be addressed immediately.
        ERROR: Error that should be fixed.
        WARNING: Warning that should be reviewed.
        INFO: Informational note.
    """

    CRITICAL = auto()
    ERROR = auto()
    WARNING = auto()
    INFO = auto()

    @property
    def weight(self) -> int:
        """Return numeric weight for comparison (higher = more severe)."""
        weights = {
            Severity.CRITICAL: 100,
            Severity.ERROR: 75,
            Severity.WARNING: 50,
            Severity.INFO: 25,
        }
        return weights[self]

    def __lt__(self, other: Severity) -> bool:
        """Compare severity by weight (less severe < more severe)."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.weight < other.weight

    def __le__(self, other: Severity) -> bool:
        """Compare severity by weight."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.weight <= other.weight

    def __gt__(self, other: Severity) -> bool:
        """Compare severity by weight."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.weight > other.weight

    def __ge__(self, other: Severity) -> bool:
        """Compare severity by weight."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.weight >= other.weight


class ProfileStatus(Enum):
    """Status of a profiling operation.

    Attributes:
        COMPLETED: Profiling completed successfully.
        PARTIAL: Profiling completed with some columns skipped.
        FAILED: Profiling failed to complete.
    """

    COMPLETED = auto()
    PARTIAL = auto()
    FAILED = auto()


class LearnStatus(Enum):
    """Status of a schema learning operation.

    Attributes:
        COMPLETED: Learning completed successfully.
        PARTIAL: Learning completed with some rules not generated.
        FAILED: Learning failed to complete.
    """

    COMPLETED = auto()
    PARTIAL = auto()
    FAILED = auto()


# =============================================================================
# Validation Failure
# =============================================================================


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    """Represents a single validation failure.

    Immutable record of a validation failure with all relevant context
    for debugging and reporting.

    Attributes:
        rule_name: Name of the validation rule that failed.
        column: Column name where the failure occurred (if applicable).
        message: Human-readable failure message.
        severity: Severity level of the failure.
        failed_count: Number of records that failed this validation.
        total_count: Total number of records checked.
        sample_values: Sample of failing values for debugging.
        metadata: Additional rule-specific metadata.

    Example:
        >>> failure = ValidationFailure(
        ...     rule_name="not_null",
        ...     column="email",
        ...     message="Found 5 null values",
        ...     severity=Severity.ERROR,
        ...     failed_count=5,
        ...     total_count=100,
        ... )
    """

    rule_name: str
    column: str | None = None
    message: str = ""
    severity: Severity = Severity.ERROR
    failed_count: int = 0
    total_count: int = 0
    sample_values: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        """Calculate the failure rate as a percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.failed_count / self.total_count) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_name": self.rule_name,
            "column": self.column,
            "message": self.message,
            "severity": self.severity.name,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "sample_values": list(self.sample_values),
            "failure_rate": self.failure_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a ValidationFailure from a dictionary.

        Args:
            data: Dictionary containing failure data.

        Returns:
            New ValidationFailure instance.
        """
        return cls(
            rule_name=data["rule_name"],
            column=data.get("column"),
            message=data.get("message", ""),
            severity=Severity[data.get("severity", "ERROR")],
            failed_count=data.get("failed_count", 0),
            total_count=data.get("total_count", 0),
            sample_values=tuple(data.get("sample_values", [])),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class CheckConfig:
    """Configuration for validation checks.

    Immutable configuration object for running data quality checks.
    Use builder methods to create modified copies.

    Attributes:
        rules: Tuple of validation rule dictionaries.
        fail_on_error: Whether to fail on execution errors.
        failure_action: Action to take on validation failure.
        sample_size: Optional sample size for large datasets.
        parallel: Whether to run rules in parallel.
        timeout_seconds: Execution timeout in seconds.
        tags: Frozenset of tags for categorization.
        extra: Platform-specific extra configuration.

    Example:
        >>> config = CheckConfig(
        ...     rules=({"type": "not_null", "column": "id"},),
        ...     failure_action=FailureAction.WARN,
        ...     timeout_seconds=60,
        ... )
        >>> strict_config = config.with_failure_action(FailureAction.RAISE)
    """

    rules: tuple[dict[str, Any], ...] = ()
    fail_on_error: bool = True
    failure_action: FailureAction = FailureAction.RAISE
    sample_size: int | None = None
    parallel: bool = False
    timeout_seconds: int | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    extra: dict[str, Any] = field(default_factory=dict)

    def with_rules(self, *rules: dict[str, Any]) -> CheckConfig:
        """Create a new config with additional rules.

        Args:
            *rules: Additional validation rules to add.

        Returns:
            New CheckConfig with merged rules.
        """
        return CheckConfig(
            rules=self.rules + rules,
            fail_on_error=self.fail_on_error,
            failure_action=self.failure_action,
            sample_size=self.sample_size,
            parallel=self.parallel,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            extra=self.extra,
        )

    def with_timeout(self, timeout_seconds: int) -> CheckConfig:
        """Create a new config with specified timeout.

        Args:
            timeout_seconds: New timeout value in seconds.

        Returns:
            New CheckConfig with updated timeout.
        """
        return CheckConfig(
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            failure_action=self.failure_action,
            sample_size=self.sample_size,
            parallel=self.parallel,
            timeout_seconds=timeout_seconds,
            tags=self.tags,
            extra=self.extra,
        )

    def with_failure_action(self, action: FailureAction) -> CheckConfig:
        """Create a new config with specified failure action.

        Args:
            action: New failure action.

        Returns:
            New CheckConfig with updated failure action.
        """
        return CheckConfig(
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            failure_action=action,
            sample_size=self.sample_size,
            parallel=self.parallel,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            extra=self.extra,
        )

    def with_tags(self, *tags: str) -> CheckConfig:
        """Create a new config with additional tags.

        Args:
            *tags: Additional tags to add.

        Returns:
            New CheckConfig with merged tags.
        """
        return CheckConfig(
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            failure_action=self.failure_action,
            sample_size=self.sample_size,
            parallel=self.parallel,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags | frozenset(tags),
            extra=self.extra,
        )

    def with_extra(self, **kwargs: Any) -> CheckConfig:
        """Create a new config with additional extra parameters.

        Args:
            **kwargs: Additional platform-specific parameters.

        Returns:
            New CheckConfig with merged extra parameters.
        """
        return CheckConfig(
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            failure_action=self.failure_action,
            sample_size=self.sample_size,
            parallel=self.parallel,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            extra={**self.extra, **kwargs},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rules": list(self.rules),
            "fail_on_error": self.fail_on_error,
            "failure_action": self.failure_action.name,
            "sample_size": self.sample_size,
            "parallel": self.parallel,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a CheckConfig from a dictionary.

        Args:
            data: Dictionary containing configuration data.

        Returns:
            New CheckConfig instance.
        """
        return cls(
            rules=tuple(data.get("rules", [])),
            fail_on_error=data.get("fail_on_error", True),
            failure_action=FailureAction[data.get("failure_action", "RAISE")],
            sample_size=data.get("sample_size"),
            parallel=data.get("parallel", False),
            timeout_seconds=data.get("timeout_seconds"),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
        )

    def to_truthound_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for Truthound API calls.

        Returns:
            Dictionary suitable for passing to Truthound functions.
        """
        kwargs: dict[str, Any] = {
            "rules": list(self.rules),
            "fail_on_error": self.fail_on_error,
        }
        if self.sample_size is not None:
            kwargs["sample_size"] = self.sample_size
        if self.parallel:
            kwargs["parallel"] = self.parallel
        if self.timeout_seconds is not None:
            kwargs["timeout"] = self.timeout_seconds
        return kwargs


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Configuration for data profiling.

    Immutable configuration for running data profiling operations.

    Attributes:
        columns: Specific columns to profile (None = all).
        include_histograms: Whether to compute histograms.
        include_correlations: Whether to compute correlations.
        sample_size: Optional sample size for large datasets.
        timeout_seconds: Execution timeout in seconds.
        extra: Platform-specific extra configuration.
    """

    columns: tuple[str, ...] | None = None
    include_histograms: bool = True
    include_correlations: bool = False
    sample_size: int | None = None
    timeout_seconds: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def with_columns(self, *columns: str) -> ProfileConfig:
        """Create a new config with specified columns.

        Args:
            *columns: Columns to profile.

        Returns:
            New ProfileConfig with specified columns.
        """
        return ProfileConfig(
            columns=columns,
            include_histograms=self.include_histograms,
            include_correlations=self.include_correlations,
            sample_size=self.sample_size,
            timeout_seconds=self.timeout_seconds,
            extra=self.extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "columns": list(self.columns) if self.columns else None,
            "include_histograms": self.include_histograms,
            "include_correlations": self.include_correlations,
            "sample_size": self.sample_size,
            "timeout_seconds": self.timeout_seconds,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a ProfileConfig from a dictionary."""
        columns = data.get("columns")
        return cls(
            columns=tuple(columns) if columns else None,
            include_histograms=data.get("include_histograms", True),
            include_correlations=data.get("include_correlations", False),
            sample_size=data.get("sample_size"),
            timeout_seconds=data.get("timeout_seconds"),
            extra=data.get("extra", {}),
        )


@dataclass(frozen=True, slots=True)
class LearnConfig:
    """Configuration for schema learning.

    Immutable configuration for learning validation rules from data.

    Attributes:
        columns: Specific columns to learn (None = all).
        include_types: Whether to learn type constraints.
        include_ranges: Whether to learn value ranges.
        include_patterns: Whether to learn string patterns.
        sample_size: Optional sample size for large datasets.
        confidence_threshold: Minimum confidence for learned rules.
        timeout_seconds: Execution timeout in seconds.
        extra: Platform-specific extra configuration.
    """

    columns: tuple[str, ...] | None = None
    include_types: bool = True
    include_ranges: bool = True
    include_patterns: bool = False
    sample_size: int | None = None
    confidence_threshold: float = 0.95
    timeout_seconds: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "columns": list(self.columns) if self.columns else None,
            "include_types": self.include_types,
            "include_ranges": self.include_ranges,
            "include_patterns": self.include_patterns,
            "sample_size": self.sample_size,
            "confidence_threshold": self.confidence_threshold,
            "timeout_seconds": self.timeout_seconds,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a LearnConfig from a dictionary."""
        columns = data.get("columns")
        return cls(
            columns=tuple(columns) if columns else None,
            include_types=data.get("include_types", True),
            include_ranges=data.get("include_ranges", True),
            include_patterns=data.get("include_patterns", False),
            sample_size=data.get("sample_size"),
            confidence_threshold=data.get("confidence_threshold", 0.95),
            timeout_seconds=data.get("timeout_seconds"),
            extra=data.get("extra", {}),
        )


# =============================================================================
# Result Types
# =============================================================================


def _utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a validation check execution.

    Immutable result object containing all validation outcomes and metadata.
    Designed for serialization to XCom and other platform-specific formats.

    Attributes:
        status: Overall status of the check.
        passed_count: Number of validations that passed.
        failed_count: Number of validations that failed.
        warning_count: Number of validations with warnings.
        skipped_count: Number of validations skipped.
        failures: Tuple of validation failures.
        execution_time_ms: Execution time in milliseconds.
        timestamp: ISO format timestamp of execution.
        metadata: Additional result metadata.

    Example:
        >>> result = CheckResult(
        ...     status=CheckStatus.PASSED,
        ...     passed_count=10,
        ...     failed_count=0,
        ... )
        >>> result.is_success
        True
        >>> result.pass_rate
        100.0
    """

    status: CheckStatus
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    skipped_count: int = 0
    failures: tuple[ValidationFailure, ...] = ()
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if the result represents a successful outcome."""
        return self.status.is_success()

    @property
    def total_count(self) -> int:
        """Return total number of validations run."""
        return self.passed_count + self.failed_count + self.warning_count + self.skipped_count

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as a percentage."""
        if self.total_count == 0:
            return 100.0
        return (self.passed_count / self.total_count) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as a percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.failed_count / self.total_count) * 100

    def iter_failures(self, min_severity: Severity | None = None) -> Iterator[ValidationFailure]:
        """Iterate over failures with optional severity filtering.

        Args:
            min_severity: Minimum severity to include. None includes all.

        Yields:
            ValidationFailure objects matching the criteria.
        """
        for failure in self.failures:
            if min_severity is None or failure.severity >= min_severity:
                yield failure

    def get_critical_failures(self) -> tuple[ValidationFailure, ...]:
        """Get all critical severity failures.

        Returns:
            Tuple of failures with CRITICAL severity.
        """
        return tuple(self.iter_failures(min_severity=Severity.CRITICAL))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (XCom compatible).

        Returns:
            Dictionary representation suitable for XCom storage.
        """
        return {
            "status": self.status.name,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "skipped_count": self.skipped_count,
            "failures": [f.to_dict() for f in self.failures],
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "is_success": self.is_success,
            "pass_rate": self.pass_rate,
            "failure_rate": self.failure_rate,
            "total_count": self.total_count,
        }

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a CheckResult from a dictionary.

        Args:
            data: Dictionary containing result data.

        Returns:
            New CheckResult instance.
        """
        failures = tuple(
            ValidationFailure.from_dict(f) for f in data.get("failures", [])
        )
        return cls(
            status=CheckStatus[data["status"]],
            passed_count=data.get("passed_count", 0),
            failed_count=data.get("failed_count", 0),
            warning_count=data.get("warning_count", 0),
            skipped_count=data.get("skipped_count", 0),
            failures=failures,
            execution_time_ms=data.get("execution_time_ms", 0.0),
            timestamp=data.get("timestamp", _utc_now_iso()),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_truthound(cls, truthound_result: Any) -> Self:
        """Create a CheckResult from a Truthound validation result.

        Args:
            truthound_result: Result object from Truthound validation.

        Returns:
            New CheckResult instance.
        """
        # Map Truthound result to common format
        # This will be implemented based on actual Truthound result structure
        failures: list[ValidationFailure] = []
        passed = 0
        failed = 0
        warnings = 0

        # Handle Truthound result structure
        if hasattr(truthound_result, "results"):
            for result in truthound_result.results:
                if hasattr(result, "is_valid") and result.is_valid:
                    passed += 1
                elif hasattr(result, "severity") and result.severity == "warning":
                    warnings += 1
                else:
                    failed += 1
                    failures.append(
                        ValidationFailure(
                            rule_name=getattr(result, "rule_name", "unknown"),
                            column=getattr(result, "column", None),
                            message=getattr(result, "message", ""),
                            severity=Severity.ERROR,
                            failed_count=getattr(result, "failed_count", 0),
                            total_count=getattr(result, "total_count", 0),
                        )
                    )

        status = CheckStatus.PASSED
        if failed > 0:
            status = CheckStatus.FAILED
        elif warnings > 0:
            status = CheckStatus.WARNING

        return cls(
            status=status,
            passed_count=passed,
            failed_count=failed,
            warning_count=warnings,
            failures=tuple(failures),
            execution_time_ms=getattr(truthound_result, "execution_time_ms", 0.0),
        )


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Profile of a single column.

    Attributes:
        column_name: Name of the column.
        dtype: Data type of the column.
        null_count: Number of null values.
        null_percentage: Percentage of null values.
        unique_count: Number of unique values.
        unique_percentage: Percentage of unique values.
        min_value: Minimum value (for comparable types).
        max_value: Maximum value (for comparable types).
        mean: Mean value (for numeric types).
        std: Standard deviation (for numeric types).
        histogram: Histogram data if computed.
        metadata: Additional column-specific metadata.
    """

    column_name: str
    dtype: str
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    min_value: Any = None
    max_value: Any = None
    mean: float | None = None
    std: float | None = None
    histogram: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "column_name": self.column_name,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "null_percentage": self.null_percentage,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "std": self.std,
            "histogram": self.histogram,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a ColumnProfile from a dictionary."""
        return cls(
            column_name=data["column_name"],
            dtype=data["dtype"],
            null_count=data.get("null_count", 0),
            null_percentage=data.get("null_percentage", 0.0),
            unique_count=data.get("unique_count", 0),
            unique_percentage=data.get("unique_percentage", 0.0),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            mean=data.get("mean"),
            std=data.get("std"),
            histogram=data.get("histogram"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Result of a profiling operation.

    Attributes:
        status: Overall status of the profiling.
        row_count: Total number of rows profiled.
        column_count: Number of columns profiled.
        columns: Tuple of column profiles.
        correlations: Correlation matrix if computed.
        execution_time_ms: Execution time in milliseconds.
        timestamp: ISO format timestamp of execution.
        metadata: Additional result metadata.
    """

    status: ProfileStatus
    row_count: int = 0
    column_count: int = 0
    columns: tuple[ColumnProfile, ...] = ()
    correlations: dict[str, dict[str, float]] | None = None
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if profiling completed successfully."""
        return self.status == ProfileStatus.COMPLETED

    def get_column(self, name: str) -> ColumnProfile | None:
        """Get profile for a specific column.

        Args:
            name: Column name to look up.

        Returns:
            ColumnProfile if found, None otherwise.
        """
        for col in self.columns:
            if col.column_name == name:
                return col
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [c.to_dict() for c in self.columns],
            "correlations": self.correlations,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a ProfileResult from a dictionary."""
        columns = tuple(
            ColumnProfile.from_dict(c) for c in data.get("columns", [])
        )
        return cls(
            status=ProfileStatus[data["status"]],
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            columns=columns,
            correlations=data.get("correlations"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            timestamp=data.get("timestamp", _utc_now_iso()),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class LearnedRule:
    """A rule learned from data.

    Attributes:
        rule_type: Type of the rule (e.g., "not_null", "range").
        column: Column the rule applies to.
        parameters: Rule parameters learned from data.
        confidence: Confidence score for the learned rule.
        sample_size: Number of samples used for learning.
        metadata: Additional rule-specific metadata.
    """

    rule_type: str
    column: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    sample_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rule_dict(self) -> dict[str, Any]:
        """Convert to a rule dictionary for CheckConfig.

        Returns:
            Dictionary suitable for use in CheckConfig.rules.
        """
        return {
            "type": self.rule_type,
            "column": self.column,
            **self.parameters,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_type": self.rule_type,
            "column": self.column,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a LearnedRule from a dictionary."""
        return cls(
            rule_type=data["rule_type"],
            column=data["column"],
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 1.0),
            sample_size=data.get("sample_size", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class LearnResult:
    """Result of a schema learning operation.

    Attributes:
        status: Overall status of the learning.
        rules: Tuple of learned rules.
        columns_analyzed: Number of columns analyzed.
        execution_time_ms: Execution time in milliseconds.
        timestamp: ISO format timestamp of execution.
        metadata: Additional result metadata.
    """

    status: LearnStatus
    rules: tuple[LearnedRule, ...] = ()
    columns_analyzed: int = 0
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if learning completed successfully."""
        return self.status == LearnStatus.COMPLETED

    def to_check_config(self, **kwargs: Any) -> CheckConfig:
        """Convert learned rules to a CheckConfig.

        Args:
            **kwargs: Additional CheckConfig parameters.

        Returns:
            CheckConfig with learned rules.
        """
        rules = tuple(rule.to_rule_dict() for rule in self.rules)
        return CheckConfig(rules=rules, **kwargs)

    def filter_by_confidence(self, min_confidence: float) -> tuple[LearnedRule, ...]:
        """Filter rules by minimum confidence.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            Tuple of rules meeting the threshold.
        """
        return tuple(rule for rule in self.rules if rule.confidence >= min_confidence)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.name,
            "rules": [r.to_dict() for r in self.rules],
            "columns_analyzed": self.columns_analyzed,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a LearnResult from a dictionary."""
        rules = tuple(LearnedRule.from_dict(r) for r in data.get("rules", []))
        return cls(
            status=LearnStatus[data["status"]],
            rules=rules,
            columns_analyzed=data.get("columns_analyzed", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            timestamp=data.get("timestamp", _utc_now_iso()),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class WorkflowIntegration(Protocol):
    """Protocol for synchronous workflow platform integrations.

    All platform adapters (Airflow, Dagster, Prefect) must implement this
    protocol to ensure consistent integration patterns.

    Example:
        >>> class AirflowAdapter:
        ...     @property
        ...     def platform_name(self) -> str:
        ...         return "airflow"
        ...
        ...     @property
        ...     def platform_version(self) -> str:
        ...         return "2.7.0"
        ...
        ...     def check(self, data: pl.DataFrame, config: CheckConfig) -> CheckResult:
        ...         # Implementation
        ...         ...
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform (e.g., 'airflow', 'dagster')."""
        ...

    @property
    @abstractmethod
    def platform_version(self) -> str:
        """Return the version of the platform integration."""
        ...

    @abstractmethod
    def check(self, data: pl.DataFrame, config: CheckConfig) -> CheckResult:
        """Execute validation checks on the data.

        Args:
            data: Polars DataFrame to validate.
            config: Configuration for the validation.

        Returns:
            CheckResult with validation outcomes.

        Raises:
            ValidationExecutionError: If validation execution fails.
            QualityGateError: If quality gate conditions are not met.
        """
        ...

    @abstractmethod
    def profile(self, data: pl.DataFrame, config: ProfileConfig) -> ProfileResult:
        """Execute profiling on the data.

        Args:
            data: Polars DataFrame to profile.
            config: Configuration for the profiling.

        Returns:
            ProfileResult with profiling outcomes.

        Raises:
            ValidationExecutionError: If profiling execution fails.
        """
        ...

    @abstractmethod
    def learn(self, data: pl.DataFrame, config: LearnConfig) -> LearnResult:
        """Learn validation rules from the data.

        Args:
            data: Polars DataFrame to learn from.
            config: Configuration for the learning.

        Returns:
            LearnResult with learned rules.

        Raises:
            ValidationExecutionError: If learning execution fails.
        """
        ...


@runtime_checkable
class AsyncWorkflowIntegration(Protocol):
    """Protocol for asynchronous workflow platform integrations.

    Async version of WorkflowIntegration for platforms that support
    asynchronous operations.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of the platform."""
        ...

    @property
    @abstractmethod
    def platform_version(self) -> str:
        """Return the version of the platform integration."""
        ...

    @abstractmethod
    async def check(self, data: pl.DataFrame, config: CheckConfig) -> CheckResult:
        """Execute validation checks asynchronously.

        Args:
            data: Polars DataFrame to validate.
            config: Configuration for the validation.

        Returns:
            CheckResult with validation outcomes.
        """
        ...

    @abstractmethod
    async def profile(self, data: pl.DataFrame, config: ProfileConfig) -> ProfileResult:
        """Execute profiling asynchronously.

        Args:
            data: Polars DataFrame to profile.
            config: Configuration for the profiling.

        Returns:
            ProfileResult with profiling outcomes.
        """
        ...

    @abstractmethod
    async def learn(self, data: pl.DataFrame, config: LearnConfig) -> LearnResult:
        """Learn validation rules asynchronously.

        Args:
            data: Polars DataFrame to learn from.
            config: Configuration for the learning.

        Returns:
            LearnResult with learned rules.
        """
        ...


# =============================================================================
# Result Builders
# =============================================================================


class CheckResultBuilder:
    """Builder for creating CheckResult instances.

    Provides a fluent API for constructing CheckResult objects, useful when
    building results incrementally during validation execution.

    Example:
        >>> builder = CheckResultBuilder()
        >>> result = (
        ...     builder
        ...     .with_passed(10)
        ...     .with_failed(2)
        ...     .add_failure(failure)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the builder with defaults."""
        self._passed_count: int = 0
        self._failed_count: int = 0
        self._warning_count: int = 0
        self._skipped_count: int = 0
        self._failures: list[ValidationFailure] = []
        self._execution_time_ms: float = 0.0
        self._metadata: dict[str, Any] = {}

    def with_passed(self, count: int) -> CheckResultBuilder:
        """Set the passed count."""
        self._passed_count = count
        return self

    def with_failed(self, count: int) -> CheckResultBuilder:
        """Set the failed count."""
        self._failed_count = count
        return self

    def with_warnings(self, count: int) -> CheckResultBuilder:
        """Set the warning count."""
        self._warning_count = count
        return self

    def with_skipped(self, count: int) -> CheckResultBuilder:
        """Set the skipped count."""
        self._skipped_count = count
        return self

    def add_failure(self, failure: ValidationFailure) -> CheckResultBuilder:
        """Add a validation failure."""
        self._failures.append(failure)
        return self

    def with_execution_time(self, ms: float) -> CheckResultBuilder:
        """Set the execution time in milliseconds."""
        self._execution_time_ms = ms
        return self

    def with_metadata(self, **kwargs: Any) -> CheckResultBuilder:
        """Add metadata entries."""
        self._metadata.update(kwargs)
        return self

    def build(self) -> CheckResult:
        """Build the CheckResult.

        Returns:
            Immutable CheckResult instance.
        """
        # Determine status based on counts
        if self._failed_count > 0:
            status = CheckStatus.FAILED
        elif self._warning_count > 0:
            status = CheckStatus.WARNING
        elif self._passed_count > 0:
            status = CheckStatus.PASSED
        else:
            status = CheckStatus.SKIPPED

        return CheckResult(
            status=status,
            passed_count=self._passed_count,
            failed_count=self._failed_count,
            warning_count=self._warning_count,
            skipped_count=self._skipped_count,
            failures=tuple(self._failures),
            execution_time_ms=self._execution_time_ms,
            metadata=self._metadata,
        )
