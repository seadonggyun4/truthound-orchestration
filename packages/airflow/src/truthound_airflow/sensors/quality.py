"""Data Quality Sensor for Apache Airflow.

This module provides the DataQualitySensor for waiting until data quality
conditions are met before proceeding with downstream tasks.

Example:
    >>> from truthound_airflow import DataQualitySensor
    >>>
    >>> wait_for_quality = DataQualitySensor(
    ...     task_id="wait_for_quality",
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     data_path="s3://bucket/incoming/data.parquet",
    ...     min_pass_rate=0.99,
    ...     poke_interval=300,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from airflow.sensors.base import BaseSensorOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.engines.base import DataQualityEngine


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Configuration for data quality sensor.

    Attributes:
        min_pass_rate: Minimum pass rate to consider condition met (0.0-1.0).
        min_row_count: Minimum number of rows required.
        max_failure_count: Maximum allowed failures.
        check_data_exists: Whether to check for data existence first.
        continue_on_error: Whether to continue if check fails (vs raise).

    Example:
        >>> config = SensorConfig(
        ...     min_pass_rate=0.99,
        ...     min_row_count=100,
        ... )
    """

    min_pass_rate: float = 1.0
    min_row_count: int | None = None
    max_failure_count: int | None = None
    check_data_exists: bool = True
    continue_on_error: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.min_pass_rate <= 1:
            msg = "min_pass_rate must be between 0 and 1"
            raise ValueError(msg)
        if self.min_row_count is not None and self.min_row_count < 0:
            msg = "min_row_count must be non-negative"
            raise ValueError(msg)


# =============================================================================
# Sensor
# =============================================================================


class DataQualitySensor(BaseSensorOperator):
    """Wait until data quality conditions are met.

    This sensor periodically checks data quality and returns True (pokes
    successfully) when the quality conditions are satisfied. Use this to
    gate downstream processing until data meets quality requirements.

    The sensor is engine-agnostic and works with any DataQualityEngine.

    Parameters
    ----------
    rules : list[dict[str, Any]]
        Validation rules that must pass.

    data_path : str | None
        Path to data file. Mutually exclusive with `sql`.

    sql : str | None
        SQL query to fetch data. Mutually exclusive with `data_path`.

    connection_id : str
        Airflow Connection ID. Default: "truthound_default"

    min_pass_rate : float
        Minimum pass rate (0.0-1.0) to consider quality met.
        Default: 1.0 (all rules must pass)

    min_row_count : int | None
        Minimum number of rows required.
        If data has fewer rows, sensor continues waiting.

    engine : DataQualityEngine | None
        Custom engine instance. Default: Truthound.

    poke_interval : float
        Seconds between pokes. Default: 60

    timeout : float
        Total timeout in seconds. Default: 3600

    mode : str
        Sensor mode: "poke" or "reschedule".
        "reschedule" is more resource-efficient for long waits.
        Default: "poke"

    soft_fail : bool
        If True, sensor is skipped on failure instead of failing.
        Default: False

    Examples
    --------
    Basic quality gate:

    >>> wait = DataQualitySensor(
    ...     task_id="wait_for_quality",
    ...     rules=[
    ...         {"column": "id", "type": "not_null"},
    ...         {"column": "amount", "type": "positive"},
    ...     ],
    ...     data_path="s3://bucket/incoming/{{ ds }}/data.parquet",
    ...     min_pass_rate=0.99,
    ... )

    With reschedule mode (resource-efficient):

    >>> wait = DataQualitySensor(
    ...     task_id="wait_for_quality",
    ...     rules=[...],
    ...     data_path="...",
    ...     poke_interval=300,  # 5 minutes
    ...     timeout=7200,       # 2 hours
    ...     mode="reschedule",
    ... )

    With minimum row count:

    >>> wait = DataQualitySensor(
    ...     task_id="wait_for_enough_data",
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     data_path="...",
    ...     min_row_count=1000,
    ...     min_pass_rate=0.95,
    ... )

    Notes
    -----
    - Sensor returns True when ALL conditions are met:
      - Data exists (if check_data_exists=True)
      - Row count >= min_row_count (if specified)
      - Pass rate >= min_pass_rate
    - Use "reschedule" mode for long waits to free up worker slots
    - The sensor does NOT modify data or push results to XCom
    """

    template_fields: Sequence[str] = (
        "rules",
        "data_path",
        "sql",
        "connection_id",
    )
    ui_color: str = "#E67E22"

    def __init__(
        self,
        *,
        rules: list[dict[str, Any]],
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        min_pass_rate: float = 1.0,
        min_row_count: int | None = None,
        max_failure_count: int | None = None,
        check_data_exists: bool = True,
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        timeout_seconds: int = 300,
        **kwargs: Any,
    ) -> None:
        """Initialize data quality sensor."""
        # Validate data source
        if data_path and sql:
            msg = "Cannot specify both data_path and sql"
            raise ValueError(msg)
        if not data_path and not sql:
            msg = "Must specify either data_path or sql"
            raise ValueError(msg)

        # Validate pass rate
        if not 0 <= min_pass_rate <= 1:
            msg = "min_pass_rate must be between 0 and 1"
            raise ValueError(msg)

        super().__init__(**kwargs)

        self.rules = rules
        self.data_path = data_path
        self.sql = sql
        self.connection_id = connection_id
        self.min_pass_rate = min_pass_rate
        self.min_row_count = min_row_count
        self.max_failure_count = max_failure_count
        self.check_data_exists = check_data_exists
        self.timeout_seconds = timeout_seconds

        # Engine configuration
        self._engine = engine
        self._engine_name = engine_name

    @property
    def engine(self) -> DataQualityEngine:
        """Get the data quality engine instance."""
        if self._engine is None:
            from common.engines import get_engine

            self._engine = get_engine(self._engine_name)
        return self._engine

    def poke(self, context: Context) -> bool:
        """Check if quality conditions are met.

        Args:
            context: Airflow execution context.

        Returns:
            bool: True if conditions are met, False to continue waiting.
        """
        from truthound_airflow.hooks.base import DataQualityHook

        self.log.info("Checking data quality conditions...")

        hook = DataQualityHook(connection_id=self.connection_id)

        # Try to load data
        try:
            if self.data_path:
                data = hook.load_data(self.data_path)
            else:
                data = hook.query(self.sql)
        except FileNotFoundError:
            if self.check_data_exists:
                self.log.info("Data not found yet, will retry...")
                return False
            raise
        except Exception as e:
            self.log.warning(f"Error loading data: {e}")
            return False

        # Check minimum row count
        row_count = len(data) if hasattr(data, "__len__") else 0
        if self.min_row_count is not None:
            if row_count < self.min_row_count:
                self.log.info(
                    f"Row count {row_count} < minimum {self.min_row_count}, "
                    "will retry..."
                )
                return False

        # Execute quality check
        self.log.info(f"Checking {len(self.rules)} rules on {row_count} rows")

        result = self.engine.check(
            data,
            self.rules,
            timeout=self.timeout_seconds,
        )

        # Calculate pass rate
        total_rules = result.passed_count + result.failed_count
        if total_rules == 0:
            self.log.warning("No rules were evaluated")
            return False

        pass_rate = result.passed_count / total_rules

        self.log.info(
            f"Current pass rate: {pass_rate:.2%} "
            f"(required: {self.min_pass_rate:.2%})"
        )

        # Check max failure count
        if self.max_failure_count is not None:
            if result.failed_count > self.max_failure_count:
                self.log.info(
                    f"Failure count {result.failed_count} > max {self.max_failure_count}, "
                    "will retry..."
                )
                return False

        # Check pass rate
        if pass_rate >= self.min_pass_rate:
            self.log.info(
                f"Quality conditions MET: "
                f"pass_rate={pass_rate:.2%} >= {self.min_pass_rate:.2%}"
            )
            return True

        self.log.info(
            f"Quality conditions NOT met: "
            f"pass_rate={pass_rate:.2%} < {self.min_pass_rate:.2%}, "
            "will retry..."
        )
        return False


# Alias for backwards compatibility
TruthoundSensor = DataQualitySensor
