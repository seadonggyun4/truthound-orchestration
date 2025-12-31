"""Semantic conventions for data quality telemetry.

This module defines standard attribute names and values for data quality
operations, following OpenTelemetry semantic conventions patterns.

These attributes provide consistent naming across all telemetry data,
enabling better correlation, filtering, and analysis in observability
platforms.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    # Attribute namespace classes
    "DataQualityAttributes",
    "EngineAttributes",
    "CheckAttributes",
    "ProfileAttributes",
    "LearnAttributes",
    "RuleAttributes",
    "DatasetAttributes",
    # Helper functions
    "create_check_attributes",
    "create_profile_attributes",
    "create_learn_attributes",
    "create_engine_attributes",
]


class DataQualityAttributes:
    """Standard attribute names for data quality telemetry.

    These follow the OpenTelemetry semantic conventions naming pattern:
    `<namespace>.<attribute>` in snake_case.

    Example:
        span.set_attribute(DataQualityAttributes.ENGINE_NAME, "truthound")
        span.set_attribute(DataQualityAttributes.CHECK_STATUS, "passed")
    """

    # Namespace prefix
    NAMESPACE = "dq"

    # Engine attributes
    ENGINE_NAME = "dq.engine.name"
    """Name of the data quality engine (e.g., 'truthound', 'great_expectations')."""

    ENGINE_VERSION = "dq.engine.version"
    """Version of the data quality engine."""

    ENGINE_TYPE = "dq.engine.type"
    """Type of engine (e.g., 'native', 'adapter')."""

    # Operation attributes
    OPERATION_TYPE = "dq.operation.type"
    """Type of operation (check, profile, learn)."""

    OPERATION_STATUS = "dq.operation.status"
    """Status of the operation (started, completed, failed)."""

    OPERATION_DURATION_MS = "dq.operation.duration_ms"
    """Duration of the operation in milliseconds."""

    # Check attributes
    CHECK_STATUS = "dq.check.status"
    """Overall check status (passed, failed, warning, error)."""

    CHECK_PASSED_COUNT = "dq.check.passed_count"
    """Number of passed validations."""

    CHECK_FAILED_COUNT = "dq.check.failed_count"
    """Number of failed validations."""

    CHECK_WARNING_COUNT = "dq.check.warning_count"
    """Number of warnings."""

    CHECK_ERROR_COUNT = "dq.check.error_count"
    """Number of errors during validation."""

    CHECK_RULES_COUNT = "dq.check.rules_count"
    """Total number of rules applied."""

    CHECK_SUCCESS_RATE = "dq.check.success_rate"
    """Ratio of passed rules to total rules (0.0 to 1.0)."""

    # Profile attributes
    PROFILE_ROW_COUNT = "dq.profile.row_count"
    """Number of rows in the profiled dataset."""

    PROFILE_COLUMN_COUNT = "dq.profile.column_count"
    """Number of columns in the profiled dataset."""

    PROFILE_NULL_PERCENTAGE = "dq.profile.null_percentage"
    """Overall null value percentage."""

    PROFILE_UNIQUE_RATIO = "dq.profile.unique_ratio"
    """Ratio of unique values."""

    # Learn attributes
    LEARN_RULES_GENERATED = "dq.learn.rules_generated"
    """Number of rules generated."""

    LEARN_CONFIDENCE_AVG = "dq.learn.confidence_avg"
    """Average confidence of generated rules."""

    LEARN_COVERAGE = "dq.learn.coverage"
    """Percentage of columns covered by generated rules."""

    # Rule attributes
    RULE_TYPE = "dq.rule.type"
    """Type of validation rule (not_null, unique, in_range, etc.)."""

    RULE_COLUMN = "dq.rule.column"
    """Column the rule applies to."""

    RULE_SEVERITY = "dq.rule.severity"
    """Severity level (critical, high, medium, low, info)."""

    RULE_PASSED = "dq.rule.passed"
    """Whether the rule passed (true/false)."""

    # Dataset attributes
    DATASET_NAME = "dq.dataset.name"
    """Name or identifier of the dataset."""

    DATASET_SOURCE = "dq.dataset.source"
    """Source of the dataset (file path, table name, etc.)."""

    DATASET_FORMAT = "dq.dataset.format"
    """Format of the dataset (parquet, csv, json, etc.)."""

    DATASET_SIZE_BYTES = "dq.dataset.size_bytes"
    """Size of the dataset in bytes."""

    DATASET_ROW_COUNT = "dq.dataset.row_count"
    """Number of rows in the dataset."""

    DATASET_COLUMN_COUNT = "dq.dataset.column_count"
    """Number of columns in the dataset."""

    # Error attributes
    ERROR_TYPE = "dq.error.type"
    """Type of error that occurred."""

    ERROR_MESSAGE = "dq.error.message"
    """Error message."""

    ERROR_STACK_TRACE = "dq.error.stack_trace"
    """Stack trace of the error (if available)."""

    # Platform integration attributes
    PLATFORM_NAME = "dq.platform.name"
    """Orchestration platform (airflow, dagster, prefect, dbt)."""

    PLATFORM_VERSION = "dq.platform.version"
    """Version of the orchestration platform."""

    PLATFORM_TASK_ID = "dq.platform.task_id"
    """Task identifier within the platform."""

    PLATFORM_RUN_ID = "dq.platform.run_id"
    """Run/execution identifier within the platform."""

    PLATFORM_DAG_ID = "dq.platform.dag_id"
    """DAG/workflow identifier (Airflow-specific)."""

    PLATFORM_JOB_NAME = "dq.platform.job_name"
    """Job name (Dagster-specific)."""

    PLATFORM_FLOW_NAME = "dq.platform.flow_name"
    """Flow name (Prefect-specific)."""


class OperationType(Enum):
    """Types of data quality operations."""

    CHECK = "check"
    PROFILE = "profile"
    LEARN = "learn"


class OperationStatus(Enum):
    """Status of data quality operations."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckStatus(Enum):
    """Status of validation checks."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class Severity(Enum):
    """Severity levels for validation failures."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True)
class EngineAttributes:
    """Structured engine attributes for telemetry."""

    name: str
    version: str | None = None
    engine_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to attribute dictionary."""
        attrs: dict[str, Any] = {
            DataQualityAttributes.ENGINE_NAME: self.name,
        }
        if self.version:
            attrs[DataQualityAttributes.ENGINE_VERSION] = self.version
        if self.engine_type:
            attrs[DataQualityAttributes.ENGINE_TYPE] = self.engine_type
        return attrs


@dataclass(frozen=True)
class CheckAttributes:
    """Structured check operation attributes for telemetry."""

    status: CheckStatus
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    rules_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.rules_count == 0:
            return 1.0
        return self.passed_count / self.rules_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to attribute dictionary."""
        return {
            DataQualityAttributes.CHECK_STATUS: self.status.value,
            DataQualityAttributes.CHECK_PASSED_COUNT: self.passed_count,
            DataQualityAttributes.CHECK_FAILED_COUNT: self.failed_count,
            DataQualityAttributes.CHECK_WARNING_COUNT: self.warning_count,
            DataQualityAttributes.CHECK_ERROR_COUNT: self.error_count,
            DataQualityAttributes.CHECK_RULES_COUNT: self.rules_count,
            DataQualityAttributes.CHECK_SUCCESS_RATE: self.success_rate,
        }


@dataclass(frozen=True)
class ProfileAttributes:
    """Structured profile operation attributes for telemetry."""

    row_count: int = 0
    column_count: int = 0
    null_percentage: float = 0.0
    unique_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to attribute dictionary."""
        return {
            DataQualityAttributes.PROFILE_ROW_COUNT: self.row_count,
            DataQualityAttributes.PROFILE_COLUMN_COUNT: self.column_count,
            DataQualityAttributes.PROFILE_NULL_PERCENTAGE: self.null_percentage,
            DataQualityAttributes.PROFILE_UNIQUE_RATIO: self.unique_ratio,
        }


@dataclass(frozen=True)
class LearnAttributes:
    """Structured learn operation attributes for telemetry."""

    rules_generated: int = 0
    confidence_avg: float = 0.0
    coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to attribute dictionary."""
        return {
            DataQualityAttributes.LEARN_RULES_GENERATED: self.rules_generated,
            DataQualityAttributes.LEARN_CONFIDENCE_AVG: self.confidence_avg,
            DataQualityAttributes.LEARN_COVERAGE: self.coverage,
        }


@dataclass(frozen=True)
class RuleAttributes:
    """Structured rule attributes for telemetry."""

    rule_type: str
    column: str | None = None
    severity: Severity = Severity.MEDIUM
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to attribute dictionary."""
        attrs: dict[str, Any] = {
            DataQualityAttributes.RULE_TYPE: self.rule_type,
            DataQualityAttributes.RULE_SEVERITY: self.severity.value,
            DataQualityAttributes.RULE_PASSED: self.passed,
        }
        if self.column:
            attrs[DataQualityAttributes.RULE_COLUMN] = self.column
        return attrs


@dataclass(frozen=True)
class DatasetAttributes:
    """Structured dataset attributes for telemetry."""

    name: str | None = None
    source: str | None = None
    format: str | None = None
    size_bytes: int | None = None
    row_count: int | None = None
    column_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to attribute dictionary."""
        attrs: dict[str, Any] = {}
        if self.name:
            attrs[DataQualityAttributes.DATASET_NAME] = self.name
        if self.source:
            attrs[DataQualityAttributes.DATASET_SOURCE] = self.source
        if self.format:
            attrs[DataQualityAttributes.DATASET_FORMAT] = self.format
        if self.size_bytes is not None:
            attrs[DataQualityAttributes.DATASET_SIZE_BYTES] = self.size_bytes
        if self.row_count is not None:
            attrs[DataQualityAttributes.DATASET_ROW_COUNT] = self.row_count
        if self.column_count is not None:
            attrs[DataQualityAttributes.DATASET_COLUMN_COUNT] = self.column_count
        return attrs


def create_engine_attributes(
    name: str,
    version: str | None = None,
    engine_type: str | None = None,
) -> dict[str, Any]:
    """Create engine attributes dictionary.

    Args:
        name: Engine name.
        version: Engine version.
        engine_type: Engine type (native, adapter).

    Returns:
        Dictionary of engine attributes.
    """
    return EngineAttributes(
        name=name,
        version=version,
        engine_type=engine_type,
    ).to_dict()


def create_check_attributes(
    status: CheckStatus | str,
    passed_count: int = 0,
    failed_count: int = 0,
    warning_count: int = 0,
    error_count: int = 0,
    rules_count: int = 0,
) -> dict[str, Any]:
    """Create check operation attributes dictionary.

    Args:
        status: Check status (passed, failed, warning, error).
        passed_count: Number of passed validations.
        failed_count: Number of failed validations.
        warning_count: Number of warnings.
        error_count: Number of errors.
        rules_count: Total number of rules.

    Returns:
        Dictionary of check attributes.
    """
    if isinstance(status, str):
        status = CheckStatus(status)
    return CheckAttributes(
        status=status,
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=warning_count,
        error_count=error_count,
        rules_count=rules_count,
    ).to_dict()


def create_profile_attributes(
    row_count: int = 0,
    column_count: int = 0,
    null_percentage: float = 0.0,
    unique_ratio: float = 0.0,
) -> dict[str, Any]:
    """Create profile operation attributes dictionary.

    Args:
        row_count: Number of rows.
        column_count: Number of columns.
        null_percentage: Overall null percentage.
        unique_ratio: Ratio of unique values.

    Returns:
        Dictionary of profile attributes.
    """
    return ProfileAttributes(
        row_count=row_count,
        column_count=column_count,
        null_percentage=null_percentage,
        unique_ratio=unique_ratio,
    ).to_dict()


def create_learn_attributes(
    rules_generated: int = 0,
    confidence_avg: float = 0.0,
    coverage: float = 0.0,
) -> dict[str, Any]:
    """Create learn operation attributes dictionary.

    Args:
        rules_generated: Number of rules generated.
        confidence_avg: Average confidence score.
        coverage: Column coverage percentage.

    Returns:
        Dictionary of learn attributes.
    """
    return LearnAttributes(
        rules_generated=rules_generated,
        confidence_avg=confidence_avg,
        coverage=coverage,
    ).to_dict()
