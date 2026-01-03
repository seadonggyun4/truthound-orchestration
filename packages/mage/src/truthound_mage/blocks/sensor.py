"""Sensor blocks for data quality gates in Mage.

This module provides sensor blocks that wait for data quality conditions
to be met before allowing downstream blocks to proceed.

Components:
    - BaseSensorBlock: Abstract base for all sensors
    - DataQualitySensor: General data quality sensor
    - QualityGateSensor: Quality gate with pass/fail thresholds

Example:
    >>> from truthound_mage import QualityGateSensor, SensorBlockConfig
    >>>
    >>> config = SensorBlockConfig(
    ...     min_pass_rate=0.95,
    ...     max_failure_rate=0.05,
    ... )
    >>> sensor = QualityGateSensor(config=config)
    >>> passed = sensor.check(check_result)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from truthound_mage.blocks.base import (
    BlockConfig,
    BlockExecutionContext,
)

if TYPE_CHECKING:
    from common.base import CheckResult


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class SensorBlockConfig(BlockConfig):
    """Configuration for sensor blocks.

    Attributes:
        min_pass_rate: Minimum pass rate required (0.0-1.0).
        max_failure_rate: Maximum failure rate allowed (0.0-1.0).
        min_row_count: Minimum row count required.
        max_row_count: Maximum row count allowed.
        required_columns: Columns that must be present.
        poke_interval_seconds: Seconds between sensor checks.
        timeout_seconds: Total timeout for sensor.
        mode: Sensor mode ('poke' or 'reschedule').
        soft_fail: Whether to soft fail instead of hard fail.
        exponential_backoff: Whether to use exponential backoff.

    Example:
        >>> config = SensorBlockConfig(
        ...     min_pass_rate=0.95,
        ...     poke_interval_seconds=30,
        ... )
    """

    min_pass_rate: float | None = None
    max_failure_rate: float | None = None
    min_row_count: int | None = None
    max_row_count: int | None = None
    required_columns: frozenset[str] = field(default_factory=frozenset)
    poke_interval_seconds: float = 60.0
    mode: str = "poke"  # 'poke' or 'reschedule'
    soft_fail: bool = False
    exponential_backoff: bool = False
    max_poke_attempts: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        BlockConfig.__post_init__(self)

        if self.min_pass_rate is not None:
            if not 0 <= self.min_pass_rate <= 1:
                msg = "min_pass_rate must be between 0 and 1"
                raise ValueError(msg)

        if self.max_failure_rate is not None:
            if not 0 <= self.max_failure_rate <= 1:
                msg = "max_failure_rate must be between 0 and 1"
                raise ValueError(msg)

        if self.min_row_count is not None and self.min_row_count < 0:
            msg = "min_row_count must be non-negative"
            raise ValueError(msg)

        if self.max_row_count is not None and self.max_row_count < 0:
            msg = "max_row_count must be non-negative"
            raise ValueError(msg)

        if self.poke_interval_seconds <= 0:
            msg = "poke_interval_seconds must be positive"
            raise ValueError(msg)

        if self.mode not in ("poke", "reschedule"):
            msg = "mode must be 'poke' or 'reschedule'"
            raise ValueError(msg)

    def with_pass_rate(self, min_rate: float) -> SensorBlockConfig:
        """Return new config with updated min_pass_rate."""
        return SensorBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            min_pass_rate=min_rate,
            max_failure_rate=self.max_failure_rate,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            required_columns=self.required_columns,
            poke_interval_seconds=self.poke_interval_seconds,
            mode=self.mode,
            soft_fail=self.soft_fail,
            exponential_backoff=self.exponential_backoff,
            max_poke_attempts=self.max_poke_attempts,
        )

    def with_failure_rate(self, max_rate: float) -> SensorBlockConfig:
        """Return new config with updated max_failure_rate."""
        return SensorBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            min_pass_rate=self.min_pass_rate,
            max_failure_rate=max_rate,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            required_columns=self.required_columns,
            poke_interval_seconds=self.poke_interval_seconds,
            mode=self.mode,
            soft_fail=self.soft_fail,
            exponential_backoff=self.exponential_backoff,
            max_poke_attempts=self.max_poke_attempts,
        )

    def with_row_count_range(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> SensorBlockConfig:
        """Return new config with updated row count range."""
        return SensorBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            min_pass_rate=self.min_pass_rate,
            max_failure_rate=self.max_failure_rate,
            min_row_count=min_count,
            max_row_count=max_count,
            required_columns=self.required_columns,
            poke_interval_seconds=self.poke_interval_seconds,
            mode=self.mode,
            soft_fail=self.soft_fail,
            exponential_backoff=self.exponential_backoff,
            max_poke_attempts=self.max_poke_attempts,
        )

    def with_poke_settings(
        self,
        interval_seconds: float,
        max_attempts: int | None = None,
        exponential_backoff: bool = False,
    ) -> SensorBlockConfig:
        """Return new config with updated poke settings."""
        return SensorBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            min_pass_rate=self.min_pass_rate,
            max_failure_rate=self.max_failure_rate,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            required_columns=self.required_columns,
            poke_interval_seconds=interval_seconds,
            mode=self.mode,
            soft_fail=self.soft_fail,
            exponential_backoff=exponential_backoff,
            max_poke_attempts=max_attempts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        base = BlockConfig.to_dict(self)
        base.update({
            "min_pass_rate": self.min_pass_rate,
            "max_failure_rate": self.max_failure_rate,
            "min_row_count": self.min_row_count,
            "max_row_count": self.max_row_count,
            "required_columns": list(self.required_columns),
            "poke_interval_seconds": self.poke_interval_seconds,
            "mode": self.mode,
            "soft_fail": self.soft_fail,
            "exponential_backoff": self.exponential_backoff,
            "max_poke_attempts": self.max_poke_attempts,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensorBlockConfig:
        """Create configuration from dictionary."""
        required_columns = data.get("required_columns", [])
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
            min_pass_rate=data.get("min_pass_rate"),
            max_failure_rate=data.get("max_failure_rate"),
            min_row_count=data.get("min_row_count"),
            max_row_count=data.get("max_row_count"),
            required_columns=frozenset(required_columns),
            poke_interval_seconds=data.get("poke_interval_seconds", 60.0),
            mode=data.get("mode", "poke"),
            soft_fail=data.get("soft_fail", False),
            exponential_backoff=data.get("exponential_backoff", False),
            max_poke_attempts=data.get("max_poke_attempts"),
        )


# =============================================================================
# Sensor Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class SensorResult:
    """Result of a sensor check.

    Attributes:
        passed: Whether the sensor condition was met.
        message: Human-readable message about the result.
        metrics: Metrics that were evaluated.
        violations: List of violated conditions.
        poke_count: Number of pokes performed.
        total_time_seconds: Total time spent waiting.

    Example:
        >>> result = SensorResult(
        ...     passed=True,
        ...     message="All conditions met",
        ...     metrics={"pass_rate": 0.98},
        ... )
    """

    passed: bool
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    violations: tuple[str, ...] = field(default_factory=tuple)
    poke_count: int = 1
    total_time_seconds: float = 0.0

    @property
    def is_passed(self) -> bool:
        """Alias for passed property."""
        return self.passed

    @property
    def has_violations(self) -> bool:
        """Check if there are violations."""
        return len(self.violations) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "passed": self.passed,
            "message": self.message,
            "metrics": self.metrics,
            "violations": list(self.violations),
            "poke_count": self.poke_count,
            "total_time_seconds": self.total_time_seconds,
        }


# =============================================================================
# Base Sensor Block
# =============================================================================


class BaseSensorBlock(ABC):
    """Abstract base class for data quality sensor blocks.

    Sensor blocks wait for data quality conditions to be met before
    allowing downstream blocks to proceed.

    Subclasses must implement:
        - poke: Check if condition is met

    Attributes:
        config: Sensor block configuration.

    Example:
        >>> class CustomSensor(BaseSensorBlock):
        ...     def poke(self, check_result, context):
        ...         return check_result.pass_rate >= 0.95
    """

    def __init__(
        self,
        config: SensorBlockConfig | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize sensor block.

        Args:
            config: Sensor configuration. Uses default if None.
            hooks: Lifecycle hooks to invoke during execution.
        """
        self.config = config or SensorBlockConfig()
        self._hooks = list(hooks) if hooks else []

    @abstractmethod
    def poke(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> bool:
        """Check if the sensor condition is met.

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            True if condition is met, False otherwise.
        """
        ...

    def execute(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> SensorResult:
        """Execute the sensor with configured poke behavior.

        This method implements the sensor execution pattern:
        1. Poke to check condition
        2. If not met and within timeout, wait and retry
        3. Return result when condition met or timeout reached

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            SensorResult indicating whether condition was met.

        Raises:
            SensorTimeoutError: If timeout reached and soft_fail is False.
        """
        from truthound_mage.utils.exceptions import SensorTimeoutError

        context = context or BlockExecutionContext()
        start_time = time.time()
        poke_count = 0
        current_interval = self.config.poke_interval_seconds

        while True:
            poke_count += 1

            # Check condition
            if self.poke(check_result, context):
                elapsed = time.time() - start_time
                metrics = self._extract_metrics(check_result)
                return SensorResult(
                    passed=True,
                    message="Sensor condition met",
                    metrics=metrics,
                    poke_count=poke_count,
                    total_time_seconds=elapsed,
                )

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= self.config.timeout_seconds:
                metrics = self._extract_metrics(check_result)
                violations = self._get_violations(check_result)

                if self.config.soft_fail:
                    return SensorResult(
                        passed=False,
                        message="Sensor timed out (soft fail)",
                        metrics=metrics,
                        violations=tuple(violations),
                        poke_count=poke_count,
                        total_time_seconds=elapsed,
                    )
                else:
                    raise SensorTimeoutError(
                        f"Sensor timed out after {elapsed:.1f}s and {poke_count} pokes",
                        timeout_seconds=self.config.timeout_seconds,
                        poke_count=poke_count,
                    )

            # Check max poke attempts
            if (
                self.config.max_poke_attempts is not None
                and poke_count >= self.config.max_poke_attempts
            ):
                metrics = self._extract_metrics(check_result)
                violations = self._get_violations(check_result)

                if self.config.soft_fail:
                    return SensorResult(
                        passed=False,
                        message=f"Max poke attempts ({self.config.max_poke_attempts}) reached",
                        metrics=metrics,
                        violations=tuple(violations),
                        poke_count=poke_count,
                        total_time_seconds=elapsed,
                    )
                else:
                    raise SensorTimeoutError(
                        f"Max poke attempts ({self.config.max_poke_attempts}) reached",
                        timeout_seconds=elapsed,
                        poke_count=poke_count,
                    )

            # Wait before next poke
            time.sleep(current_interval)

            # Apply exponential backoff if configured
            if self.config.exponential_backoff:
                current_interval = min(current_interval * 2, 300)  # Cap at 5 minutes

    def check(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> SensorResult:
        """Single check without waiting (no poke loop).

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            SensorResult indicating whether condition is met.
        """
        start_time = time.time()
        passed = self.poke(check_result, context)
        elapsed = time.time() - start_time

        metrics = self._extract_metrics(check_result)
        violations = [] if passed else self._get_violations(check_result)

        return SensorResult(
            passed=passed,
            message="Condition met" if passed else "Condition not met",
            metrics=metrics,
            violations=tuple(violations),
            poke_count=1,
            total_time_seconds=elapsed,
        )

    def _extract_metrics(
        self,
        check_result: CheckResult | dict[str, Any],
    ) -> dict[str, Any]:
        """Extract metrics from check result.

        Args:
            check_result: Check result to extract metrics from.

        Returns:
            Dictionary of metrics.
        """
        if isinstance(check_result, dict):
            return {
                "pass_rate": check_result.get("pass_rate", 0.0),
                "failure_rate": check_result.get("failure_rate", 0.0),
                "passed_count": check_result.get("passed_count", 0),
                "failed_count": check_result.get("failed_count", 0),
                "total_count": check_result.get("total_count", 0),
            }
        else:
            return {
                "pass_rate": check_result.pass_rate,
                "failure_rate": check_result.failure_rate,
                "passed_count": check_result.passed_count,
                "failed_count": check_result.failed_count,
                "total_count": check_result.passed_count + check_result.failed_count,
            }

    def _get_violations(
        self,
        check_result: CheckResult | dict[str, Any],
    ) -> list[str]:
        """Get list of violated conditions.

        Args:
            check_result: Check result to analyze.

        Returns:
            List of violation messages.
        """
        violations = []
        metrics = self._extract_metrics(check_result)

        if self.config.min_pass_rate is not None:
            if metrics["pass_rate"] < self.config.min_pass_rate:
                violations.append(
                    f"Pass rate {metrics['pass_rate']:.2%} below minimum "
                    f"{self.config.min_pass_rate:.2%}"
                )

        if self.config.max_failure_rate is not None:
            if metrics["failure_rate"] > self.config.max_failure_rate:
                violations.append(
                    f"Failure rate {metrics['failure_rate']:.2%} above maximum "
                    f"{self.config.max_failure_rate:.2%}"
                )

        total_count = metrics.get("total_count", 0)
        if self.config.min_row_count is not None:
            if total_count < self.config.min_row_count:
                violations.append(
                    f"Row count {total_count} below minimum {self.config.min_row_count}"
                )

        if self.config.max_row_count is not None:
            if total_count > self.config.max_row_count:
                violations.append(
                    f"Row count {total_count} above maximum {self.config.max_row_count}"
                )

        return violations


# =============================================================================
# Data Quality Sensor
# =============================================================================


class DataQualitySensor(BaseSensorBlock):
    """General data quality sensor.

    Waits for data quality conditions to be met based on check results.
    Supports custom condition functions for flexible criteria.

    Attributes:
        config: Sensor configuration.
        condition_fn: Optional custom condition function.

    Example:
        >>> sensor = DataQualitySensor(
        ...     config=SensorBlockConfig(min_pass_rate=0.95),
        ... )
        >>> result = sensor.execute(check_result)
        >>> if result.passed:
        ...     print("Quality conditions met!")
    """

    def __init__(
        self,
        config: SensorBlockConfig | None = None,
        condition_fn: Callable[[CheckResult | dict], bool] | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize data quality sensor.

        Args:
            config: Sensor configuration.
            condition_fn: Custom condition function.
            hooks: Lifecycle hooks.
        """
        super().__init__(config=config, hooks=hooks)
        self._condition_fn = condition_fn

    def poke(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> bool:
        """Check if quality conditions are met.

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            True if all conditions are met.
        """
        # Use custom condition if provided
        if self._condition_fn is not None:
            return self._condition_fn(check_result)

        metrics = self._extract_metrics(check_result)

        # Check pass rate
        if self.config.min_pass_rate is not None:
            if metrics["pass_rate"] < self.config.min_pass_rate:
                return False

        # Check failure rate
        if self.config.max_failure_rate is not None:
            if metrics["failure_rate"] > self.config.max_failure_rate:
                return False

        # Check row count
        total_count = metrics.get("total_count", 0)
        if self.config.min_row_count is not None:
            if total_count < self.config.min_row_count:
                return False

        if self.config.max_row_count is not None:
            if total_count > self.config.max_row_count:
                return False

        return True


# =============================================================================
# Quality Gate Sensor
# =============================================================================


class QualityGateSensor(BaseSensorBlock):
    """Quality gate sensor with pass/fail thresholds.

    A specialized sensor for implementing quality gates in data pipelines.
    Provides a simple pass/fail interface based on configured thresholds.

    Attributes:
        config: Sensor configuration.

    Example:
        >>> sensor = QualityGateSensor(
        ...     config=SensorBlockConfig(
        ...         min_pass_rate=0.99,
        ...         max_failure_rate=0.01,
        ...     ),
        ... )
        >>> if sensor.check(result).passed:
        ...     proceed_to_production()
    """

    def poke(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> bool:
        """Check if quality gate conditions are met.

        A quality gate passes if ALL of the following are true:
        - Pass rate >= min_pass_rate (if configured)
        - Failure rate <= max_failure_rate (if configured)
        - Row count within bounds (if configured)
        - Check status is PASSED (not FAILED or ERROR)

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            True if all gate conditions are met.
        """
        # Check status first
        if isinstance(check_result, dict):
            status = check_result.get("status", "").upper()
            if status in ("FAILED", "ERROR"):
                return False
        else:
            from common.base import CheckStatus

            if check_result.status in (CheckStatus.FAILED, CheckStatus.ERROR):
                return False

        metrics = self._extract_metrics(check_result)

        # Check pass rate threshold
        if self.config.min_pass_rate is not None:
            if metrics["pass_rate"] < self.config.min_pass_rate:
                return False

        # Check failure rate threshold
        if self.config.max_failure_rate is not None:
            if metrics["failure_rate"] > self.config.max_failure_rate:
                return False

        # Check row count bounds
        total_count = metrics.get("total_count", 0)
        if self.config.min_row_count is not None:
            if total_count < self.config.min_row_count:
                return False

        if self.config.max_row_count is not None:
            if total_count > self.config.max_row_count:
                return False

        return True


# =============================================================================
# Factory Functions
# =============================================================================


def create_quality_sensor(
    min_pass_rate: float | None = None,
    max_failure_rate: float | None = None,
    min_row_count: int | None = None,
    max_row_count: int | None = None,
    poke_interval_seconds: float = 60.0,
    timeout_seconds: int = 300,
    soft_fail: bool = False,
    condition_fn: Callable[[CheckResult | dict], bool] | None = None,
    **kwargs: Any,
) -> DataQualitySensor:
    """Create a data quality sensor with the given configuration.

    Args:
        min_pass_rate: Minimum pass rate required.
        max_failure_rate: Maximum failure rate allowed.
        min_row_count: Minimum row count required.
        max_row_count: Maximum row count allowed.
        poke_interval_seconds: Seconds between checks.
        timeout_seconds: Total timeout.
        soft_fail: Whether to soft fail on timeout.
        condition_fn: Custom condition function.
        **kwargs: Additional configuration options.

    Returns:
        Configured DataQualitySensor instance.

    Example:
        >>> sensor = create_quality_sensor(
        ...     min_pass_rate=0.95,
        ...     timeout_seconds=120,
        ... )
    """
    config = SensorBlockConfig(
        min_pass_rate=min_pass_rate,
        max_failure_rate=max_failure_rate,
        min_row_count=min_row_count,
        max_row_count=max_row_count,
        poke_interval_seconds=poke_interval_seconds,
        timeout_seconds=timeout_seconds,
        soft_fail=soft_fail,
        **kwargs,
    )
    return DataQualitySensor(config=config, condition_fn=condition_fn)
