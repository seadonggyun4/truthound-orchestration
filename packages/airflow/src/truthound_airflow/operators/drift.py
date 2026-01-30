"""Data Quality Drift Detection Operator for Apache Airflow.

This module provides the DataQualityDriftOperator for executing drift
detection between baseline and current datasets in Airflow DAGs.

Example:
    >>> from truthound_airflow.operators.drift import DataQualityDriftOperator
    >>>
    >>> detect_drift = DataQualityDriftOperator(
    ...     task_id="detect_drift",
    ...     baseline_data_path="s3://bucket/baseline/{{ ds }}/data.parquet",
    ...     current_data_path="s3://bucket/current/{{ ds }}/data.parquet",
    ...     method="ks",
    ...     threshold=0.05,
    ... )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from airflow.exceptions import AirflowException
from airflow.models import BaseOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.base import DriftResult
    from common.engines.base import DataQualityEngine


class DataQualityDriftOperator(BaseOperator):
    """Execute drift detection between baseline and current datasets.

    This operator compares a baseline dataset against a current dataset
    to detect statistical drift using the configured engine. Results are
    pushed to XCom for downstream consumption.

    The engine must implement ``DriftDetectionEngine`` protocol. If it
    does not, the operator raises ``AirflowException`` with a clear message.

    Parameters
    ----------
    baseline_data_path : str | None
        Path to baseline data file. Mutually exclusive with baseline_sql.
    current_data_path : str | None
        Path to current data file. Mutually exclusive with current_sql.
    baseline_sql : str | None
        SQL query to fetch baseline data.
    current_sql : str | None
        SQL query to fetch current data.
    method : str
        Statistical method for drift detection. Default: "auto".
    columns : list[str] | None
        Columns to check. None means all columns.
    threshold : float | None
        Drift detection threshold. None uses method default.
    fail_on_drift : bool
        Whether to raise AirflowException when drift is detected.
    connection_id : str
        Airflow Connection ID for data source.
    engine : DataQualityEngine | None
        Custom engine instance. If None, uses registry.
    engine_name : str | None
        Engine name to get from registry.
    timeout_seconds : int
        Operation timeout in seconds.
    xcom_push_key : str
        Key for pushing result to XCom.

    Raises
    ------
    AirflowException
        If drift detected and fail_on_drift is True.
        If the engine does not support drift detection.

    Example
    -------
    >>> detect_drift = DataQualityDriftOperator(
    ...     task_id="detect_drift",
    ...     baseline_data_path="s3://bucket/baseline.parquet",
    ...     current_data_path="s3://bucket/current.parquet",
    ...     method="psi",
    ...     threshold=0.1,
    ...     fail_on_drift=True,
    ... )
    """

    template_fields: Sequence[str] = (
        "baseline_data_path",
        "current_data_path",
        "baseline_sql",
        "current_sql",
        "connection_id",
    )
    template_ext: Sequence[str] = (".sql", ".json", ".yaml")
    ui_color: str = "#E67E22"
    ui_fgcolor: str = "#FFFFFF"

    def __init__(
        self,
        *,
        baseline_data_path: str | None = None,
        current_data_path: str | None = None,
        baseline_sql: str | None = None,
        current_sql: str | None = None,
        method: str = "auto",
        columns: list[str] | None = None,
        threshold: float | None = None,
        fail_on_drift: bool = True,
        connection_id: str = "truthound_default",
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        timeout_seconds: int = 300,
        xcom_push_key: str = "drift_result",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate data sources
        if not baseline_data_path and not baseline_sql:
            msg = "Must specify either baseline_data_path or baseline_sql"
            raise ValueError(msg)
        if not current_data_path and not current_sql:
            msg = "Must specify either current_data_path or current_sql"
            raise ValueError(msg)

        self.baseline_data_path = baseline_data_path
        self.current_data_path = current_data_path
        self.baseline_sql = baseline_sql
        self.current_sql = current_sql
        self.method = method
        self.columns = columns
        self.threshold = threshold
        self.fail_on_drift = fail_on_drift
        self.connection_id = connection_id
        self.timeout_seconds = timeout_seconds
        self.xcom_push_key = xcom_push_key

        self._engine = engine
        self._engine_name = engine_name

    @property
    def engine(self) -> DataQualityEngine:
        """Get the data quality engine instance (lazy initialization)."""
        if self._engine is None:
            from common.engines import get_engine

            self._engine = get_engine(self._engine_name)
        return self._engine

    def execute(self, context: Context) -> dict[str, Any]:
        """Execute drift detection.

        Args:
            context: Airflow execution context.

        Returns:
            Serialized DriftResult dictionary.

        Raises:
            AirflowException: If engine doesn't support drift or drift detected
                              and fail_on_drift is True.
        """
        from common.engines.base import supports_drift

        # Verify engine supports drift detection
        if not supports_drift(self.engine):
            raise AirflowException(
                f"Engine '{self.engine.engine_name}' does not support drift detection. "
                f"Use an engine that implements DriftDetectionEngine protocol."
            )

        self.log.info(
            f"Starting drift detection with engine: {self.engine.engine_name}, "
            f"method={self.method}"
        )

        # Load data
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook(connection_id=self.connection_id)

        if self.baseline_data_path:
            baseline = hook.load_data(self.baseline_data_path)
        else:
            baseline = hook.query(self.baseline_sql)

        if self.current_data_path:
            current = hook.load_data(self.current_data_path)
        else:
            current = hook.query(self.current_sql)

        self.log.info(
            f"Loaded baseline ({len(baseline) if hasattr(baseline, '__len__') else '?'} rows) "
            f"and current ({len(current) if hasattr(current, '__len__') else '?'} rows)"
        )

        # Build kwargs
        detect_kwargs: dict[str, Any] = {"method": self.method}
        if self.columns is not None:
            detect_kwargs["columns"] = self.columns
        if self.threshold is not None:
            detect_kwargs["threshold"] = self.threshold

        # Execute drift detection
        result: DriftResult = self.engine.detect_drift(
            baseline, current, **detect_kwargs
        )

        # Serialize
        result_dict = result.to_dict()
        result_dict["_metadata"] = {
            "engine": self.engine.engine_name,
            "engine_version": self.engine.engine_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": self.task_id,
            "dag_id": context.get("dag", {}).dag_id if context.get("dag") else None,
        }

        # Push to XCom
        context["ti"].xcom_push(key=self.xcom_push_key, value=result_dict)
        self.log.info(f"Pushed drift result to XCom with key: {self.xcom_push_key}")

        # Log summary
        self.log.info(
            f"Drift Detection Results: status={result.status.name}, "
            f"drifted={result.drifted_count}/{result.total_columns} columns, "
            f"drift_rate={result.drift_rate:.2f}%, "
            f"duration={result.execution_time_ms:.2f}ms"
        )

        # Handle drift
        if result.is_drifted:
            drifted_names = [c.column for c in result.drifted_columns if c.is_drifted]
            self.log.warning(
                f"Drift detected in columns: {', '.join(drifted_names[:5])}"
                + (f" ... ({len(drifted_names) - 5} more)" if len(drifted_names) > 5 else "")
            )

            if self.fail_on_drift:
                raise AirflowException(
                    f"Drift detected: {result.drifted_count}/{result.total_columns} columns "
                    f"({result.drift_rate:.2f}%). "
                    f"Drifted: {', '.join(drifted_names[:3])}"
                    + (f" ... ({len(drifted_names) - 3} more)" if len(drifted_names) > 3 else "")
                )
        else:
            self.log.info("No drift detected.")

        return result_dict
