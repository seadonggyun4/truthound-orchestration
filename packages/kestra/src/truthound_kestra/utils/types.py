"""Type definitions and enums for Kestra data quality integration.

This module provides type definitions, enums, and data classes used
throughout the truthound-kestra package.

Example:
    >>> from truthound_kestra.utils.types import (
    ...     CheckStatus,
    ...     ScriptOutput,
    ...     ExecutionContext,
    ... )
    >>>
    >>> output = ScriptOutput(
    ...     status=CheckStatus.PASSED,
    ...     passed_count=100,
    ...     failed_count=0
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

__all__ = [
    # Enums
    "CheckStatus",
    "Severity",
    "OperationType",
    "OutputFormat",
    "DataSourceType",
    # Data classes
    "ScriptOutput",
    "ExecutionContext",
    "ValidationFailure",
    "ColumnProfile",
    "LearnedRule",
    # Type aliases
    "RuleDict",
    "MetadataDict",
]

# Type aliases
RuleDict = dict[str, Any]
MetadataDict = dict[str, Any]


class CheckStatus(str, Enum):
    """Status of a data quality check.

    Values:
        PASSED: All validations passed successfully.
        FAILED: One or more validations failed.
        WARNING: Validations passed but with warnings.
        SKIPPED: Check was skipped (e.g., no data).
        ERROR: An error occurred during check execution.
    """

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"

    def is_success(self) -> bool:
        """Check if status indicates success."""
        return self in (CheckStatus.PASSED, CheckStatus.WARNING)

    def is_failure(self) -> bool:
        """Check if status indicates failure."""
        return self in (CheckStatus.FAILED, CheckStatus.ERROR)


class Severity(str, Enum):
    """Severity level for validation failures and alerts.

    Values:
        CRITICAL: Highest severity, requires immediate attention.
        HIGH: High priority issue.
        MEDIUM: Moderate priority issue.
        LOW: Low priority issue.
        INFO: Informational, no action required.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def __lt__(self, other: Severity) -> bool:
        """Compare severity levels (CRITICAL > HIGH > MEDIUM > LOW > INFO)."""
        order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        return order.index(self) < order.index(other)


class OperationType(str, Enum):
    """Type of data quality operation.

    Values:
        CHECK: Validate data against rules.
        PROFILE: Generate data profile statistics.
        LEARN: Learn schema/rules from data.
    """

    CHECK = "check"
    PROFILE = "profile"
    LEARN = "learn"


class OutputFormat(str, Enum):
    """Output format for Kestra task outputs.

    Values:
        JSON: JSON format (default).
        YAML: YAML format.
        CSV: CSV format for tabular data.
        MARKDOWN: Markdown format for reports.
    """

    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    MARKDOWN = "markdown"


class DataSourceType(str, Enum):
    """Type of data source for input.

    Values:
        FILE: Local file path.
        URI: Remote URI (S3, GCS, HTTP, etc.).
        INLINE: Inline data in the task.
        OUTPUT: Output from a previous task.
        SECRET: Secret reference from Kestra secrets.
    """

    FILE = "file"
    URI = "uri"
    INLINE = "inline"
    OUTPUT = "output"
    SECRET = "secret"


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    """Represents a single validation failure.

    Attributes:
        rule_type: Type of the rule that failed.
        column: Column name where the failure occurred.
        message: Human-readable failure message.
        severity: Severity level of the failure.
        failed_count: Number of rows that failed this rule.
        failed_indices: Sample of row indices that failed (optional).
        metadata: Additional context information.

    Example:
        >>> failure = ValidationFailure(
        ...     rule_type="not_null",
        ...     column="email",
        ...     message="Found 5 null values in column 'email'",
        ...     severity=Severity.HIGH,
        ...     failed_count=5
        ... )
    """

    rule_type: str
    column: str
    message: str
    severity: Severity = Severity.MEDIUM
    failed_count: int = 0
    failed_indices: tuple[int, ...] = ()
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rule_type": self.rule_type,
            "column": self.column,
            "message": self.message,
            "severity": self.severity.value,
            "failed_count": self.failed_count,
            "failed_indices": list(self.failed_indices),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationFailure:
        """Create from dictionary."""
        return cls(
            rule_type=data["rule_type"],
            column=data["column"],
            message=data["message"],
            severity=Severity(data.get("severity", "medium")),
            failed_count=data.get("failed_count", 0),
            failed_indices=tuple(data.get("failed_indices", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Profile statistics for a single column.

    Attributes:
        column_name: Name of the column.
        dtype: Data type of the column.
        null_count: Number of null values.
        null_percentage: Percentage of null values.
        unique_count: Number of unique values.
        unique_percentage: Percentage of unique values.
        min_value: Minimum value (for numeric/date columns).
        max_value: Maximum value (for numeric/date columns).
        mean: Mean value (for numeric columns).
        std: Standard deviation (for numeric columns).
        metadata: Additional statistics.

    Example:
        >>> profile = ColumnProfile(
        ...     column_name="age",
        ...     dtype="int64",
        ...     null_count=0,
        ...     null_percentage=0.0,
        ...     unique_count=50,
        ...     unique_percentage=50.0,
        ...     min_value=18,
        ...     max_value=85,
        ...     mean=42.5
        ... )
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
    metadata: MetadataDict = field(default_factory=dict)

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
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnProfile:
        """Create from dictionary."""
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
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class LearnedRule:
    """A rule learned from data analysis.

    Attributes:
        rule_type: Type of the learned rule.
        column: Column the rule applies to.
        parameters: Rule parameters (e.g., min/max values).
        confidence: Confidence score for the rule (0.0 to 1.0).
        sample_size: Number of samples used to learn the rule.
        metadata: Additional learning context.

    Example:
        >>> rule = LearnedRule(
        ...     rule_type="in_range",
        ...     column="age",
        ...     parameters={"min": 0, "max": 150},
        ...     confidence=0.95
        ... )
    """

    rule_type: str
    column: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    sample_size: int = 0
    metadata: MetadataDict = field(default_factory=dict)

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

    def to_rule_dict(self) -> RuleDict:
        """Convert to a rule dictionary for engine consumption."""
        rule: RuleDict = {
            "type": self.rule_type,
            "column": self.column,
            **self.parameters,
        }
        return rule

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedRule:
        """Create from dictionary."""
        return cls(
            rule_type=data["rule_type"],
            column=data["column"],
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 1.0),
            sample_size=data.get("sample_size", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ScriptOutput:
    """Standard output structure for Kestra scripts.

    This class provides a consistent output format for all data quality
    scripts executed within Kestra tasks.

    Attributes:
        status: Overall status of the operation.
        operation: Type of operation performed.
        passed_count: Number of passed validations.
        failed_count: Number of failed validations.
        warning_count: Number of warnings.
        total_rows: Total number of rows processed.
        execution_time_ms: Execution time in milliseconds.
        failures: List of validation failures (for check operations).
        columns: List of column profiles (for profile operations).
        rules: List of learned rules (for learn operations).
        metadata: Additional output context.
        timestamp: When the operation completed.

    Example:
        >>> output = ScriptOutput(
        ...     status=CheckStatus.PASSED,
        ...     operation=OperationType.CHECK,
        ...     passed_count=10,
        ...     failed_count=0,
        ...     total_rows=1000
        ... )
        >>> print(output.to_dict())
    """

    status: CheckStatus
    operation: OperationType = OperationType.CHECK
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    total_rows: int = 0
    execution_time_ms: float = 0.0
    failures: tuple[ValidationFailure, ...] = ()
    columns: tuple[ColumnProfile, ...] = ()
    rules: tuple[LearnedRule, ...] = ()
    metadata: MetadataDict = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_success(self) -> bool:
        """Check if the operation was successful."""
        return self.status.is_success()

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate (0.0 to 1.0)."""
        total = self.passed_count + self.failed_count
        if total == 0:
            return 1.0
        return self.passed_count / total

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 to 1.0)."""
        return 1.0 - self.pass_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Kestra output."""
        return {
            "status": self.status.value,
            "operation": self.operation.value,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "total_rows": self.total_rows,
            "execution_time_ms": self.execution_time_ms,
            "failures": [f.to_dict() for f in self.failures],
            "columns": [c.to_dict() for c in self.columns],
            "rules": [r.to_dict() for r in self.rules],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "is_success": self.is_success,
            "pass_rate": self.pass_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScriptOutput:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            status=CheckStatus(data.get("status", "error")),
            operation=OperationType(data.get("operation", "check")),
            passed_count=data.get("passed_count", 0),
            failed_count=data.get("failed_count", 0),
            warning_count=data.get("warning_count", 0),
            total_rows=data.get("total_rows", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            failures=tuple(
                ValidationFailure.from_dict(f)
                for f in data.get("failures", [])
            ),
            columns=tuple(
                ColumnProfile.from_dict(c)
                for c in data.get("columns", [])
            ),
            rules=tuple(
                LearnedRule.from_dict(r)
                for r in data.get("rules", [])
            ),
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
        )


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Kestra execution context information.

    This class captures the Kestra-specific execution context for
    logging, monitoring, and correlation purposes.

    Attributes:
        execution_id: Unique identifier for the flow execution.
        flow_id: Identifier of the flow being executed.
        namespace: Kestra namespace.
        task_id: Identifier of the current task.
        attempt: Current attempt number (for retries).
        trigger_type: Type of trigger that started the execution.
        trigger_date: When the execution was triggered.
        variables: Flow variables available in the execution.
        labels: Labels attached to the execution.

    Example:
        >>> context = ExecutionContext(
        ...     execution_id="abc123",
        ...     flow_id="data_quality_pipeline",
        ...     namespace="production",
        ...     task_id="validate_users"
        ... )
    """

    execution_id: str
    flow_id: str
    namespace: str = "default"
    task_id: str | None = None
    attempt: int = 1
    trigger_type: str | None = None
    trigger_date: datetime | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    labels: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_id": self.execution_id,
            "flow_id": self.flow_id,
            "namespace": self.namespace,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "trigger_type": self.trigger_type,
            "trigger_date": (
                self.trigger_date.isoformat() if self.trigger_date else None
            ),
            "variables": self.variables,
            "labels": list(self.labels),
        }

    def to_log_context(self) -> dict[str, Any]:
        """Convert to context for structured logging."""
        return {
            "kestra.execution_id": self.execution_id,
            "kestra.flow_id": self.flow_id,
            "kestra.namespace": self.namespace,
            "kestra.task_id": self.task_id,
            "kestra.attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        """Create from dictionary."""
        trigger_date = data.get("trigger_date")
        if isinstance(trigger_date, str):
            trigger_date = datetime.fromisoformat(trigger_date)

        return cls(
            execution_id=data["execution_id"],
            flow_id=data["flow_id"],
            namespace=data.get("namespace", "default"),
            task_id=data.get("task_id"),
            attempt=data.get("attempt", 1),
            trigger_type=data.get("trigger_type"),
            trigger_date=trigger_date,
            variables=data.get("variables", {}),
            labels=frozenset(data.get("labels", [])),
        )

    @classmethod
    def from_kestra_env(cls) -> ExecutionContext:
        """Create from Kestra environment variables.

        This method reads the standard Kestra environment variables
        that are automatically set during task execution.

        Returns:
            ExecutionContext populated from environment.
        """
        import os

        return cls(
            execution_id=os.environ.get("KESTRA_EXECUTION_ID", "unknown"),
            flow_id=os.environ.get("KESTRA_FLOW_ID", "unknown"),
            namespace=os.environ.get("KESTRA_NAMESPACE", "default"),
            task_id=os.environ.get("KESTRA_TASK_ID"),
            attempt=int(os.environ.get("KESTRA_ATTEMPT", "1")),
            trigger_type=os.environ.get("KESTRA_TRIGGER_TYPE"),
        )

    def with_task_id(self, task_id: str) -> ExecutionContext:
        """Return new context with updated task_id."""
        return ExecutionContext(
            execution_id=self.execution_id,
            flow_id=self.flow_id,
            namespace=self.namespace,
            task_id=task_id,
            attempt=self.attempt,
            trigger_type=self.trigger_type,
            trigger_date=self.trigger_date,
            variables=self.variables,
            labels=self.labels,
        )

    def with_attempt(self, attempt: int) -> ExecutionContext:
        """Return new context with updated attempt number."""
        return ExecutionContext(
            execution_id=self.execution_id,
            flow_id=self.flow_id,
            namespace=self.namespace,
            task_id=self.task_id,
            attempt=attempt,
            trigger_type=self.trigger_type,
            trigger_date=self.trigger_date,
            variables=self.variables,
            labels=self.labels,
        )
