"""Data Quality Check Operator for Apache Airflow.

This module provides the DataQualityCheckOperator for executing data quality
validations in Airflow DAGs. The operator is engine-agnostic and supports
Truthound, Great Expectations, Pandera, and custom engines.

Example:
    >>> from truthound_airflow import DataQualityCheckOperator
    >>>
    >>> check_quality = DataQualityCheckOperator(
    ...     task_id="check_users_quality",
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "unique"},
    ...     ],
    ...     data_path="s3://bucket/users/{{ ds }}/data.parquet",
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from airflow.exceptions import AirflowException

from truthound_airflow.operators.base import BaseDataQualityOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.base import CheckResult
    from common.engines.base import DataQualityEngine


class DataQualityCheckOperator(BaseDataQualityOperator):
    """Execute data quality validation checks.

    This operator loads data from a file or SQL query and runs validation
    rules using the configured DataQualityEngine. Results are pushed to
    XCom and can be used by downstream tasks.

    The operator is engine-agnostic: by default it uses Truthound, but any
    DataQualityEngine implementation can be plugged in.

    Parameters
    ----------
    rules : list[dict[str, Any]]
        Validation rules to apply. Format depends on the engine.
        For Truthound/common format:
            [{"column": "id", "type": "not_null"}, ...]
        Jinja templates are supported.

    data_path : str | None
        Path to data file (S3, GCS, local, etc.).
        Mutually exclusive with `sql`.
        Example: "s3://bucket/data/{{ ds }}/events.parquet"

    sql : str | None
        SQL query to fetch data.
        Mutually exclusive with `data_path`.
        Example: "SELECT * FROM events WHERE date = '{{ ds }}'"

    connection_id : str
        Airflow Connection ID for data source.
        Default: "truthound_default"

    engine : DataQualityEngine | None
        Custom engine instance to use.
        If None, uses engine_name or default (Truthound).

    engine_name : str | None
        Name of engine to get from registry.
        Options: "truthound", "great_expectations", "pandera"

    fail_on_error : bool
        Whether to raise AirflowException on validation failure.
        Default: True

    warning_threshold : float | None
        Failure rate threshold for warning instead of failure.
        Value between 0.0 and 1.0.
        Example: 0.05 means 5% failure rate triggers warning only.

    sample_size : int | None
        Number of rows to sample. None means all rows.

    timeout_seconds : int
        Validation timeout in seconds. Default: 300

    xcom_push_key : str
        Key for pushing result to XCom.
        Default: "data_quality_result"

    Attributes
    ----------
    template_fields : Sequence[str]
        Fields supporting Jinja templating.

    ui_color : str
        Operator color in Airflow UI.

    Examples
    --------
    Basic usage with Truthound (default engine):

    >>> check = DataQualityCheckOperator(
    ...     task_id="check_quality",
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "unique"},
    ...         {"column": "age", "type": "in_range", "min": 0, "max": 150},
    ...     ],
    ...     data_path="s3://bucket/users.parquet",
    ... )

    With SQL query:

    >>> check = DataQualityCheckOperator(
    ...     task_id="check_transactions",
    ...     rules=[{"column": "amount", "type": "positive"}],
    ...     sql="SELECT * FROM transactions WHERE date = '{{ ds }}'",
    ...     connection_id="postgres_default",
    ... )

    With warning threshold (don't fail on minor issues):

    >>> check = DataQualityCheckOperator(
    ...     task_id="check_with_threshold",
    ...     rules=[...],
    ...     data_path="...",
    ...     warning_threshold=0.01,  # 1% failure rate is acceptable
    ... )

    With Great Expectations engine:

    >>> from common.engines import GreatExpectationsAdapter
    >>>
    >>> check = DataQualityCheckOperator(
    ...     task_id="check_with_ge",
    ...     rules=[...],
    ...     data_path="...",
    ...     engine=GreatExpectationsAdapter(),
    ... )

    Raises
    ------
    ValueError
        If both data_path and sql are specified.
        If neither data_path nor sql is specified.
        If warning_threshold is not between 0 and 1.

    AirflowException
        If validation fails and fail_on_error is True.

    Notes
    -----
    - Results are always pushed to XCom regardless of success/failure
    - Use warning_threshold for soft failures (log warning but continue)
    - Large datasets should use sample_size to limit memory usage
    """

    template_fields: Sequence[str] = (
        "rules",
        "data_path",
        "sql",
        "connection_id",
    )
    ui_color: str = "#4A90D9"

    def __init__(
        self,
        *,
        rules: list[dict[str, Any]],
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        fail_on_error: bool = True,
        warning_threshold: float | None = None,
        sample_size: int | None = None,
        timeout_seconds: int = 300,
        xcom_push_key: str = "data_quality_result",
        **kwargs: Any,
    ) -> None:
        """Initialize data quality check operator."""
        # Validate warning threshold
        if warning_threshold is not None:
            if not 0 <= warning_threshold <= 1:
                msg = "warning_threshold must be between 0 and 1"
                raise ValueError(msg)

        super().__init__(
            data_path=data_path,
            sql=sql,
            connection_id=connection_id,
            engine=engine,
            engine_name=engine_name,
            fail_on_error=fail_on_error,
            timeout_seconds=timeout_seconds,
            xcom_push_key=xcom_push_key,
            **kwargs,
        )

        self.rules = rules
        self.warning_threshold = warning_threshold
        self.sample_size = sample_size

    def _execute_operation(
        self,
        data: Any,
        context: Context,
    ) -> CheckResult:
        """Execute validation check.

        Args:
            data: Loaded data (typically Polars DataFrame).
            context: Airflow execution context.

        Returns:
            CheckResult: Validation result.
        """
        self.log.info(f"Executing quality check with {len(self.rules)} rules")

        # Sample data if configured
        if self.sample_size and hasattr(data, "__len__") and len(data) > self.sample_size:
            self.log.info(f"Sampling {self.sample_size} rows from {len(data)}")
            if hasattr(data, "sample"):
                data = data.sample(n=self.sample_size)

        # Execute check with engine
        result = self.engine.check(
            data,
            self.rules,
            timeout=self.timeout_seconds,
        )

        return result

    def _serialize_result(self, result: CheckResult) -> dict[str, Any]:
        """Serialize CheckResult for XCom.

        Args:
            result: The check result.

        Returns:
            dict[str, Any]: XCom-compatible dictionary.
        """
        return {
            "status": result.status.value,
            "is_success": result.is_success,
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "warning_count": result.warning_count,
            "skipped_count": getattr(result, "skipped_count", 0),
            "failure_rate": result.failure_rate,
            "failures": [
                {
                    "rule_name": f.rule_name,
                    "column": f.column,
                    "message": f.message,
                    "severity": f.severity.value,
                    "failed_count": f.failed_count,
                    "total_count": f.total_count,
                    "failure_rate": f.failure_rate,
                }
                for f in result.failures
            ],
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
        }

    def _handle_result(
        self,
        result: CheckResult,
        result_dict: dict[str, Any],
        context: Context,
    ) -> None:
        """Handle check result, potentially raising on failure.

        Args:
            result: The check result.
            result_dict: Serialized result dictionary.
            context: Airflow execution context.

        Raises:
            AirflowException: If check failed and fail_on_error is True.
        """
        if result.is_success:
            self.log.info(
                f"Quality check PASSED: "
                f"{result.passed_count} rules passed, "
                f"duration={result.execution_time_ms:.2f}ms"
            )
            return

        failure_rate = result.failure_rate

        # Check warning threshold
        if self.warning_threshold is not None and failure_rate <= self.warning_threshold:
            self.log.warning(
                f"Quality check has warnings: "
                f"{result.failed_count} failures "
                f"({failure_rate:.2%} <= threshold {self.warning_threshold:.2%})"
            )
            return

        # Log failure details
        self.log.error(
            f"Quality check FAILED: "
            f"{result.failed_count} rules failed ({failure_rate:.2%})"
        )

        for failure in result.failures:
            self.log.error(
                f"  - {failure.rule_name} [{failure.column}]: "
                f"{failure.message} "
                f"({failure.failed_count}/{failure.total_count} failed)"
            )

        # Raise if configured
        if self.fail_on_error:
            failure_summary = ", ".join(
                f"{f.rule_name}[{f.column}]" for f in result.failures[:3]
            )
            if len(result.failures) > 3:
                failure_summary += f", ... ({len(result.failures) - 3} more)"

            raise AirflowException(
                f"Quality check failed: {result.failed_count} rules failed "
                f"({failure_rate:.2%}). Failures: {failure_summary}"
            )

    def _log_metrics(self, result_dict: dict[str, Any]) -> None:
        """Log check metrics.

        Args:
            result_dict: Serialized result dictionary.
        """
        self.log.info(
            f"Quality Check Results: "
            f"status={result_dict['status']}, "
            f"passed={result_dict['passed_count']}, "
            f"failed={result_dict['failed_count']}, "
            f"duration={result_dict['execution_time_ms']:.2f}ms"
        )


# Alias for backwards compatibility
TruthoundCheckOperator = DataQualityCheckOperator
