"""SLA Configuration Types for Data Quality Operations.

This module provides immutable configuration types for SLA management
in data quality operations.

Example:
    >>> config = SLAConfig(
    ...     max_failure_rate=0.05,
    ...     min_pass_rate=0.95,
    ...     max_execution_time_seconds=300.0,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AlertLevel(Enum):
    """Alert severity level for SLA violations.

    Attributes:
        INFO: Informational, no action required.
        WARNING: Warning, should be reviewed.
        ERROR: Error, requires attention.
        CRITICAL: Critical, immediate action required.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SLAViolationType(Enum):
    """Types of SLA violations.

    Attributes:
        FAILURE_RATE_EXCEEDED: Failure rate exceeds threshold.
        PASS_RATE_BELOW_MINIMUM: Pass rate below minimum.
        EXECUTION_TIME_EXCEEDED: Execution time exceeded limit.
        ROW_COUNT_BELOW_MINIMUM: Row count below minimum.
        ROW_COUNT_ABOVE_MAXIMUM: Row count above maximum.
        CONSECUTIVE_FAILURES: Too many consecutive failures.
        CUSTOM: Custom violation type.
    """

    FAILURE_RATE_EXCEEDED = "failure_rate_exceeded"
    PASS_RATE_BELOW_MINIMUM = "pass_rate_below_minimum"
    EXECUTION_TIME_EXCEEDED = "execution_time_exceeded"
    ROW_COUNT_BELOW_MINIMUM = "row_count_below_minimum"
    ROW_COUNT_ABOVE_MAXIMUM = "row_count_above_maximum"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class SLAConfig:
    """Immutable SLA configuration for data quality operations.

    This configuration defines thresholds and limits for SLA monitoring.

    Attributes:
        max_failure_rate: Maximum acceptable failure rate (0.0-1.0).
        min_pass_rate: Minimum acceptable pass rate (0.0-1.0).
        max_execution_time_seconds: Maximum execution time in seconds.
        min_row_count: Minimum expected row count.
        max_row_count: Maximum expected row count.
        max_consecutive_failures: Maximum consecutive failures before alert.
        alert_on_warning: Whether to alert on warnings.
        alert_level: Default alert level for violations.
        enabled: Whether SLA monitoring is enabled.
        tags: Metadata tags for the SLA.

    Example:
        >>> config = SLAConfig(
        ...     max_failure_rate=0.05,
        ...     min_pass_rate=0.95,
        ...     max_execution_time_seconds=300.0,
        ... )
    """

    max_failure_rate: float | None = None
    min_pass_rate: float | None = None
    max_execution_time_seconds: float | None = None
    min_row_count: int | None = None
    max_row_count: int | None = None
    max_consecutive_failures: int | None = 3
    alert_on_warning: bool = False
    alert_level: AlertLevel = AlertLevel.ERROR
    enabled: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_failure_rate is not None:
            if not 0 <= self.max_failure_rate <= 1:
                msg = "max_failure_rate must be between 0 and 1"
                raise ValueError(msg)

        if self.min_pass_rate is not None:
            if not 0 <= self.min_pass_rate <= 1:
                msg = "min_pass_rate must be between 0 and 1"
                raise ValueError(msg)

        if self.max_execution_time_seconds is not None:
            if self.max_execution_time_seconds <= 0:
                msg = "max_execution_time_seconds must be positive"
                raise ValueError(msg)

        if self.min_row_count is not None and self.min_row_count < 0:
            msg = "min_row_count must be non-negative"
            raise ValueError(msg)

        if self.max_row_count is not None and self.max_row_count < 0:
            msg = "max_row_count must be non-negative"
            raise ValueError(msg)

        if (
            self.min_row_count is not None
            and self.max_row_count is not None
            and self.min_row_count > self.max_row_count
        ):
            msg = "min_row_count cannot exceed max_row_count"
            raise ValueError(msg)

    def with_failure_rate(self, max_rate: float) -> SLAConfig:
        """Return new config with updated max failure rate."""
        return SLAConfig(
            max_failure_rate=max_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_pass_rate(self, min_rate: float) -> SLAConfig:
        """Return new config with updated min pass rate."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=min_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_execution_time(self, max_seconds: float) -> SLAConfig:
        """Return new config with updated max execution time."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=max_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_row_count_range(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> SLAConfig:
        """Return new config with updated row count range."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=min_count if min_count is not None else self.min_row_count,
            max_row_count=max_count if max_count is not None else self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_alert_level(self, level: AlertLevel) -> SLAConfig:
        """Return new config with updated alert level."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_enabled(self, enabled: bool) -> SLAConfig:
        """Return new config with updated enabled status."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=enabled,
            tags=self.tags,
        )


@dataclass(frozen=True, slots=True)
class SLAMetrics:
    """Metrics captured for SLA evaluation.

    Attributes:
        passed_count: Number of passed checks.
        failed_count: Number of failed checks.
        warning_count: Number of warnings.
        execution_time_ms: Execution time in milliseconds.
        row_count: Number of rows processed.
        timestamp: When metrics were captured.
        task_id: Airflow task ID.
        dag_id: Airflow DAG ID.
        run_id: Airflow run ID.

    Example:
        >>> metrics = SLAMetrics(
        ...     passed_count=10,
        ...     failed_count=1,
        ...     execution_time_ms=1500.0,
        ...     row_count=10000,
        ... )
    """

    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    execution_time_ms: float = 0.0
    row_count: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: str | None = None
    dag_id: str | None = None
    run_id: str | None = None

    @property
    def total_count(self) -> int:
        """Total number of checks."""
        return self.passed_count + self.failed_count

    @property
    def pass_rate(self) -> float:
        """Pass rate (0.0-1.0)."""
        if self.total_count == 0:
            return 1.0
        return self.passed_count / self.total_count

    @property
    def failure_rate(self) -> float:
        """Failure rate (0.0-1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count

    @property
    def execution_time_seconds(self) -> float:
        """Execution time in seconds."""
        return self.execution_time_ms / 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
            "failure_rate": self.failure_rate,
            "execution_time_ms": self.execution_time_ms,
            "execution_time_seconds": self.execution_time_seconds,
            "row_count": self.row_count,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "dag_id": self.dag_id,
            "run_id": self.run_id,
        }

    @classmethod
    def from_check_result(
        cls,
        result: dict[str, Any],
        task_id: str | None = None,
        dag_id: str | None = None,
        run_id: str | None = None,
    ) -> SLAMetrics:
        """Create metrics from check result dictionary.

        Args:
            result: Check result dictionary from XCom.
            task_id: Airflow task ID.
            dag_id: Airflow DAG ID.
            run_id: Airflow run ID.

        Returns:
            SLAMetrics: Metrics from result.
        """
        return cls(
            passed_count=result.get("passed_count", 0),
            failed_count=result.get("failed_count", 0),
            warning_count=result.get("warning_count", 0),
            execution_time_ms=result.get("execution_time_ms", 0.0),
            row_count=result.get("_metadata", {}).get("row_count"),
            task_id=task_id,
            dag_id=dag_id,
            run_id=run_id,
        )


@dataclass(frozen=True, slots=True)
class SLAViolation:
    """Represents an SLA violation.

    Attributes:
        violation_type: Type of violation.
        message: Human-readable violation message.
        threshold: SLA threshold value.
        actual: Actual measured value.
        alert_level: Severity level.
        timestamp: When violation was detected.
        task_id: Airflow task ID.
        dag_id: Airflow DAG ID.
        metadata: Additional context.

    Example:
        >>> violation = SLAViolation(
        ...     violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
        ...     message="Failure rate 10% exceeds threshold 5%",
        ...     threshold=0.05,
        ...     actual=0.10,
        ...     alert_level=AlertLevel.ERROR,
        ... )
    """

    violation_type: SLAViolationType
    message: str
    threshold: float | int | None = None
    actual: float | int | None = None
    alert_level: AlertLevel = AlertLevel.ERROR
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: str | None = None
    dag_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_type": self.violation_type.value,
            "message": self.message,
            "threshold": self.threshold,
            "actual": self.actual,
            "alert_level": self.alert_level.value,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "dag_id": self.dag_id,
            "metadata": self.metadata,
        }


# Preset configurations
DEFAULT_SLA_CONFIG = SLAConfig()

STRICT_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.01,
    min_pass_rate=0.99,
    max_execution_time_seconds=60.0,
    max_consecutive_failures=1,
    alert_level=AlertLevel.CRITICAL,
)

LENIENT_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.10,
    min_pass_rate=0.90,
    max_execution_time_seconds=600.0,
    max_consecutive_failures=5,
    alert_on_warning=False,
    alert_level=AlertLevel.WARNING,
)

PRODUCTION_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.05,
    min_pass_rate=0.95,
    max_execution_time_seconds=300.0,
    max_consecutive_failures=3,
    alert_on_warning=True,
    alert_level=AlertLevel.ERROR,
)
