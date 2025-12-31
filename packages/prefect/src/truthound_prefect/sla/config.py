"""SLA configuration and data types.

This module provides immutable configuration classes and data containers
for SLA monitoring, following the frozen dataclass pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AlertLevel(str, Enum):
    """Severity level for SLA alerts."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SLAViolationType(str, Enum):
    """Type of SLA violation."""

    FAILURE_RATE_EXCEEDED = "failure_rate_exceeded"
    PASS_RATE_BELOW_MINIMUM = "pass_rate_below_minimum"
    EXECUTION_TIME_EXCEEDED = "execution_time_exceeded"
    ROW_COUNT_BELOW_MINIMUM = "row_count_below_minimum"
    ROW_COUNT_ABOVE_MAXIMUM = "row_count_above_maximum"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class SLAConfig:
    """Configuration for SLA monitoring.

    Immutable configuration with validation and builder methods.

    Attributes:
        max_failure_rate: Maximum allowed failure rate (0.0 to 1.0).
        min_pass_rate: Minimum required pass rate (0.0 to 1.0).
        max_execution_time_seconds: Maximum allowed execution time.
        min_row_count: Minimum required row count.
        max_row_count: Maximum allowed row count.
        max_consecutive_failures: Maximum allowed consecutive failures.
        alert_on_warning: Send alerts for warnings too.
        alert_level: Default alert level for violations.
        enabled: Whether SLA monitoring is enabled.
        tags: Tags for categorization.
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
            if not 0.0 <= self.max_failure_rate <= 1.0:
                raise ValueError("max_failure_rate must be between 0.0 and 1.0")

        if self.min_pass_rate is not None:
            if not 0.0 <= self.min_pass_rate <= 1.0:
                raise ValueError("min_pass_rate must be between 0.0 and 1.0")

        if self.max_execution_time_seconds is not None:
            if self.max_execution_time_seconds <= 0:
                raise ValueError("max_execution_time_seconds must be positive")

        if self.min_row_count is not None and self.max_row_count is not None:
            if self.min_row_count > self.max_row_count:
                raise ValueError("min_row_count cannot exceed max_row_count")

        if self.max_consecutive_failures is not None:
            if self.max_consecutive_failures < 1:
                raise ValueError("max_consecutive_failures must be at least 1")

    def with_failure_rate(self, max_rate: float | None) -> SLAConfig:
        """Return a new config with max_failure_rate changed."""
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

    def with_pass_rate(self, min_rate: float | None) -> SLAConfig:
        """Return a new config with min_pass_rate changed."""
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

    def with_execution_time(self, max_seconds: float | None) -> SLAConfig:
        """Return a new config with max_execution_time_seconds changed."""
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

    def with_row_count_bounds(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> SLAConfig:
        """Return a new config with row count bounds changed."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=min_count,
            max_row_count=max_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_consecutive_failures(self, max_failures: int | None) -> SLAConfig:
        """Return a new config with max_consecutive_failures changed."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=max_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_alert_settings(
        self,
        alert_level: AlertLevel = AlertLevel.ERROR,
        alert_on_warning: bool = False,
    ) -> SLAConfig:
        """Return a new config with alert settings changed."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=alert_on_warning,
            alert_level=alert_level,
            enabled=self.enabled,
            tags=self.tags,
        )

    def with_tags(self, *tags: str) -> SLAConfig:
        """Return a new config with additional tags."""
        new_tags = self.tags | frozenset(tags)
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=self.alert_level,
            enabled=self.enabled,
            tags=new_tags,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_failure_rate": self.max_failure_rate,
            "min_pass_rate": self.min_pass_rate,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "min_row_count": self.min_row_count,
            "max_row_count": self.max_row_count,
            "max_consecutive_failures": self.max_consecutive_failures,
            "alert_on_warning": self.alert_on_warning,
            "alert_level": self.alert_level.value,
            "enabled": self.enabled,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class SLAMetrics:
    """Metrics captured for SLA evaluation.

    Immutable container for operation metrics.

    Attributes:
        passed_count: Number of passed checks.
        failed_count: Number of failed checks.
        warning_count: Number of warnings.
        execution_time_ms: Execution time in milliseconds.
        row_count: Number of rows in the data.
        timestamp: When the metrics were captured.
        flow_name: Name of the flow.
        task_name: Name of the task.
        run_id: Prefect run ID.
    """

    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    execution_time_ms: float = 0.0
    row_count: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    flow_name: str | None = None
    task_name: str | None = None
    run_id: str | None = None

    @property
    def total_count(self) -> int:
        """Total number of checks."""
        return self.passed_count + self.failed_count

    @property
    def pass_rate(self) -> float:
        """Pass rate (0.0 to 1.0)."""
        if self.total_count == 0:
            return 1.0
        return self.passed_count / self.total_count

    @property
    def failure_rate(self) -> float:
        """Failure rate (0.0 to 1.0)."""
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count

    @property
    def execution_time_seconds(self) -> float:
        """Execution time in seconds."""
        return self.execution_time_ms / 1000.0

    @property
    def is_success(self) -> bool:
        """Whether all checks passed."""
        return self.failed_count == 0

    @classmethod
    def from_check_result(
        cls,
        result: dict[str, Any],
        flow_name: str | None = None,
        task_name: str | None = None,
        run_id: str | None = None,
    ) -> SLAMetrics:
        """Create metrics from a serialized check result."""
        return cls(
            passed_count=result.get("passed_count", 0),
            failed_count=result.get("failed_count", 0),
            warning_count=0,  # Would need to be derived
            execution_time_ms=result.get("execution_time_ms", 0.0),
            row_count=result.get("metadata", {}).get("row_count"),
            flow_name=flow_name,
            task_name=task_name,
            run_id=run_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
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
            "is_success": self.is_success,
            "timestamp": self.timestamp.isoformat(),
            "flow_name": self.flow_name,
            "task_name": self.task_name,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class SLAViolation:
    """Represents an SLA violation.

    Immutable container for violation details.

    Attributes:
        violation_type: Type of violation.
        message: Human-readable message.
        threshold: The configured threshold.
        actual: The actual value.
        alert_level: Severity of the violation.
        timestamp: When the violation was detected.
        flow_name: Name of the flow.
        task_name: Name of the task.
        run_id: Prefect run ID.
        metadata: Additional context.
    """

    violation_type: SLAViolationType
    message: str
    threshold: float | int | None = None
    actual: float | int | None = None
    alert_level: AlertLevel = AlertLevel.ERROR
    timestamp: datetime = field(default_factory=datetime.now)
    flow_name: str | None = None
    task_name: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_type": self.violation_type.value,
            "message": self.message,
            "threshold": self.threshold,
            "actual": self.actual,
            "alert_level": self.alert_level.value,
            "timestamp": self.timestamp.isoformat(),
            "flow_name": self.flow_name,
            "task_name": self.task_name,
            "run_id": self.run_id,
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
    tags=frozenset({"strict"}),
)

LENIENT_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.10,
    min_pass_rate=0.90,
    max_execution_time_seconds=600.0,
    max_consecutive_failures=5,
    alert_level=AlertLevel.WARNING,
    tags=frozenset({"lenient"}),
)

PRODUCTION_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.05,
    min_pass_rate=0.95,
    max_execution_time_seconds=300.0,
    max_consecutive_failures=3,
    alert_on_warning=True,
    alert_level=AlertLevel.ERROR,
    tags=frozenset({"production"}),
)


__all__ = [
    # Enums
    "AlertLevel",
    "SLAViolationType",
    # Data types
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    # Presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
]
