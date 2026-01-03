"""SLA configuration for Kestra data quality integration.

This module provides configuration classes for SLA monitoring
in Kestra workflows.

Example:
    >>> from truthound_kestra.sla.config import (
    ...     SLAConfig,
    ...     SLAMetrics,
    ...     AlertLevel,
    ... )
    >>>
    >>> config = SLAConfig(
    ...     max_failure_rate=0.05,
    ...     min_pass_rate=0.95,
    ...     max_execution_time_seconds=300.0
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

__all__ = [
    # Enums
    "AlertLevel",
    "SLAViolationType",
    # Config classes
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    # Presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
]


class AlertLevel(str, Enum):
    """Alert level for SLA violations.

    Values:
        INFO: Informational, no immediate action needed.
        WARNING: Warning, should be investigated.
        ERROR: Error, requires attention.
        CRITICAL: Critical, requires immediate action.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def _order_index(self) -> int:
        """Get the order index for comparison."""
        order = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        return order.index(self)

    def __lt__(self, other: AlertLevel) -> bool:
        """Compare alert levels."""
        return self._order_index() < other._order_index()

    def __le__(self, other: AlertLevel) -> bool:
        """Compare alert levels."""
        return self._order_index() <= other._order_index()

    def __gt__(self, other: AlertLevel) -> bool:
        """Compare alert levels."""
        return self._order_index() > other._order_index()

    def __ge__(self, other: AlertLevel) -> bool:
        """Compare alert levels."""
        return self._order_index() >= other._order_index()


class SLAViolationType(str, Enum):
    """Type of SLA violation.

    Values:
        FAILURE_RATE_EXCEEDED: Failure rate above threshold.
        PASS_RATE_BELOW_MINIMUM: Pass rate below minimum.
        EXECUTION_TIME_EXCEEDED: Execution time above limit.
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
    """Configuration for SLA monitoring.

    Attributes:
        max_failure_rate: Maximum allowed failure rate (0.0 to 1.0).
        min_pass_rate: Minimum required pass rate (0.0 to 1.0).
        max_execution_time_seconds: Maximum execution time in seconds.
        min_row_count: Minimum expected row count.
        max_row_count: Maximum expected row count.
        max_consecutive_failures: Maximum consecutive failures allowed.
        alert_on_warning: Whether to alert on warnings.
        alert_level: Default alert level for violations.
        enabled: Whether SLA monitoring is enabled.
        tags: Tags for categorizing SLA.
        flow_id: Associated Kestra flow ID.
        task_id: Associated Kestra task ID.
        namespace: Kestra namespace.

    Example:
        >>> config = SLAConfig(
        ...     max_failure_rate=0.05,
        ...     min_pass_rate=0.95,
        ...     alert_level=AlertLevel.ERROR
        ... )
    """

    max_failure_rate: float | None = None
    min_pass_rate: float | None = None
    max_execution_time_seconds: float | None = None
    min_row_count: int | None = None
    max_row_count: int | None = None
    max_consecutive_failures: int = 3
    alert_on_warning: bool = False
    alert_level: AlertLevel = AlertLevel.ERROR
    enabled: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)
    flow_id: str | None = None
    task_id: str | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_failure_rate is not None:
            if not 0.0 <= self.max_failure_rate <= 1.0:
                raise ValueError("max_failure_rate must be between 0.0 and 1.0")
        if self.min_pass_rate is not None:
            if not 0.0 <= self.min_pass_rate <= 1.0:
                raise ValueError("min_pass_rate must be between 0.0 and 1.0")

    def with_failure_rate(self, max_rate: float) -> SLAConfig:
        """Return new config with updated max_failure_rate."""
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
            flow_id=self.flow_id,
            task_id=self.task_id,
            namespace=self.namespace,
        )

    def with_pass_rate(self, min_rate: float) -> SLAConfig:
        """Return new config with updated min_pass_rate."""
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
            flow_id=self.flow_id,
            task_id=self.task_id,
            namespace=self.namespace,
        )

    def with_execution_time(self, max_seconds: float) -> SLAConfig:
        """Return new config with updated max_execution_time_seconds."""
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
            flow_id=self.flow_id,
            task_id=self.task_id,
            namespace=self.namespace,
        )

    def with_row_count(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> SLAConfig:
        """Return new config with updated row count limits."""
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
            flow_id=self.flow_id,
            task_id=self.task_id,
            namespace=self.namespace,
        )

    def with_kestra_context(
        self,
        flow_id: str,
        task_id: str | None = None,
        namespace: str | None = None,
    ) -> SLAConfig:
        """Return new config with Kestra context."""
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
            tags=self.tags,
            flow_id=flow_id,
            task_id=task_id,
            namespace=namespace,
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
            "flow_id": self.flow_id,
            "task_id": self.task_id,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SLAConfig:
        """Create from dictionary."""
        return cls(
            max_failure_rate=data.get("max_failure_rate"),
            min_pass_rate=data.get("min_pass_rate"),
            max_execution_time_seconds=data.get("max_execution_time_seconds"),
            min_row_count=data.get("min_row_count"),
            max_row_count=data.get("max_row_count"),
            max_consecutive_failures=data.get("max_consecutive_failures", 3),
            alert_on_warning=data.get("alert_on_warning", False),
            alert_level=AlertLevel(data.get("alert_level", "error")),
            enabled=data.get("enabled", True),
            tags=frozenset(data.get("tags", [])),
            flow_id=data.get("flow_id"),
            task_id=data.get("task_id"),
            namespace=data.get("namespace"),
        )


@dataclass(frozen=True, slots=True)
class SLAMetrics:
    """Metrics for SLA evaluation.

    Attributes:
        passed_count: Number of passed validations.
        failed_count: Number of failed validations.
        warning_count: Number of warnings.
        execution_time_ms: Execution time in milliseconds.
        row_count: Number of rows processed.
        timestamp: When metrics were collected.
        flow_id: Associated Kestra flow ID.
        task_id: Associated Kestra task ID.
        execution_id: Kestra execution ID.

    Example:
        >>> metrics = SLAMetrics(
        ...     passed_count=100,
        ...     failed_count=5,
        ...     execution_time_ms=1500.0,
        ...     row_count=10000
        ... )
    """

    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    execution_time_ms: float = 0.0
    row_count: int | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    flow_id: str | None = None
    task_id: str | None = None
    execution_id: str | None = None

    @property
    def total_count(self) -> int:
        """Get total validation count."""
        return self.passed_count + self.failed_count

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate (0.0 to 1.0)."""
        if self.total_count == 0:
            return 1.0
        return self.passed_count / self.total_count

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 to 1.0)."""
        return 1.0 - self.pass_rate

    @property
    def execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        return self.execution_time_ms / 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "execution_time_ms": self.execution_time_ms,
            "row_count": self.row_count,
            "timestamp": self.timestamp.isoformat(),
            "flow_id": self.flow_id,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "pass_rate": self.pass_rate,
            "failure_rate": self.failure_rate,
        }

    @classmethod
    def from_check_result(
        cls,
        result: dict[str, Any],
        flow_id: str | None = None,
        task_id: str | None = None,
        execution_id: str | None = None,
    ) -> SLAMetrics:
        """Create metrics from a check result.

        Args:
            result: Check result dictionary.
            flow_id: Optional Kestra flow ID.
            task_id: Optional Kestra task ID.
            execution_id: Optional Kestra execution ID.

        Returns:
            SLAMetrics instance.
        """
        return cls(
            passed_count=result.get("passed_count", 0),
            failed_count=result.get("failed_count", 0),
            warning_count=result.get("warning_count", 0),
            execution_time_ms=result.get("execution_time_ms", 0.0),
            row_count=result.get("total_rows"),
            flow_id=flow_id,
            task_id=task_id,
            execution_id=execution_id,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SLAMetrics:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            passed_count=data.get("passed_count", 0),
            failed_count=data.get("failed_count", 0),
            warning_count=data.get("warning_count", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            row_count=data.get("row_count"),
            timestamp=timestamp,
            flow_id=data.get("flow_id"),
            task_id=data.get("task_id"),
            execution_id=data.get("execution_id"),
        )


@dataclass(frozen=True, slots=True)
class SLAViolation:
    """Represents an SLA violation.

    Attributes:
        violation_type: Type of the violation.
        message: Human-readable violation message.
        threshold: The threshold that was violated.
        actual: The actual value that caused the violation.
        alert_level: Severity level of the violation.
        timestamp: When the violation occurred.
        metadata: Additional context information.

    Example:
        >>> violation = SLAViolation(
        ...     violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
        ...     message="Failure rate 10% exceeds maximum 5%",
        ...     threshold=0.05,
        ...     actual=0.10
        ... )
    """

    violation_type: SLAViolationType
    message: str
    threshold: float | int | None = None
    actual: float | int | None = None
    alert_level: AlertLevel = AlertLevel.ERROR
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
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
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SLAViolation:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            violation_type=SLAViolationType(data["violation_type"]),
            message=data["message"],
            threshold=data.get("threshold"),
            actual=data.get("actual"),
            alert_level=AlertLevel(data.get("alert_level", "error")),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


# Preset configurations
DEFAULT_SLA_CONFIG = SLAConfig()
"""Default SLA configuration with no thresholds set."""

STRICT_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.01,
    min_pass_rate=0.99,
    max_execution_time_seconds=60.0,
    max_consecutive_failures=1,
    alert_level=AlertLevel.CRITICAL,
)
"""Strict SLA configuration for critical data."""

LENIENT_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.10,
    min_pass_rate=0.90,
    max_execution_time_seconds=600.0,
    max_consecutive_failures=5,
    alert_level=AlertLevel.WARNING,
)
"""Lenient SLA configuration for less critical data."""

PRODUCTION_SLA_CONFIG = SLAConfig(
    max_failure_rate=0.05,
    min_pass_rate=0.95,
    max_execution_time_seconds=300.0,
    max_consecutive_failures=3,
    alert_on_warning=True,
    alert_level=AlertLevel.ERROR,
)
"""Production SLA configuration with balanced thresholds."""
