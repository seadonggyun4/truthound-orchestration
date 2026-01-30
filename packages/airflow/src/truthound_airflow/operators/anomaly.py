"""Data Quality Anomaly Detection Operator for Apache Airflow.

This module provides the DataQualityAnomalyOperator for detecting anomalies
in datasets within Airflow DAGs.

Example:
    >>> from truthound_airflow.operators.anomaly import DataQualityAnomalyOperator
    >>>
    >>> detect_anomaly = DataQualityAnomalyOperator(
    ...     task_id="detect_anomalies",
    ...     data_path="s3://bucket/data/{{ ds }}/events.parquet",
    ...     detector="isolation_forest",
    ...     contamination=0.05,
    ... )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from airflow.exceptions import AirflowException

from truthound_airflow.operators.base import BaseDataQualityOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.base import AnomalyResult
    from common.engines.base import DataQualityEngine


class DataQualityAnomalyOperator(BaseDataQualityOperator):
    """Execute anomaly detection on a dataset.

    This operator loads data and runs anomaly detection using the configured
    engine. Results are pushed to XCom for downstream consumption.

    The engine must implement ``AnomalyDetectionEngine`` protocol.

    Parameters
    ----------
    data_path : str | None
        Path to data file. Mutually exclusive with sql.
    sql : str | None
        SQL query to fetch data. Mutually exclusive with data_path.
    detector : str
        Anomaly detector to use. Default: "isolation_forest".
        Options: "isolation_forest", "z_score", "lof", "ensemble".
    columns : list[str] | None
        Columns to analyze. None means all numeric columns.
    contamination : float
        Expected proportion of anomalies (0 < x < 0.5). Default: 0.05.
    fail_on_anomaly : bool
        Whether to raise AirflowException when anomalies are detected.
    connection_id : str
        Airflow Connection ID.
    engine : DataQualityEngine | None
        Custom engine instance.
    engine_name : str | None
        Engine name for registry lookup.
    timeout_seconds : int
        Operation timeout in seconds.
    xcom_push_key : str
        Key for XCom push.

    Raises
    ------
    AirflowException
        If anomalies detected and fail_on_anomaly is True.
        If the engine does not support anomaly detection.

    Example
    -------
    >>> detect = DataQualityAnomalyOperator(
    ...     task_id="detect_anomalies",
    ...     data_path="s3://bucket/transactions.parquet",
    ...     detector="isolation_forest",
    ...     contamination=0.01,
    ...     fail_on_anomaly=True,
    ... )
    """

    template_fields: Sequence[str] = (
        "data_path",
        "sql",
        "connection_id",
    )
    ui_color: str = "#E74C3C"
    ui_fgcolor: str = "#FFFFFF"

    def __init__(
        self,
        *,
        data_path: str | None = None,
        sql: str | None = None,
        detector: str = "isolation_forest",
        columns: list[str] | None = None,
        contamination: float = 0.05,
        fail_on_anomaly: bool = True,
        connection_id: str = "truthound_default",
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        fail_on_error: bool = True,
        timeout_seconds: int = 300,
        xcom_push_key: str = "anomaly_result",
        **kwargs: Any,
    ) -> None:
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

        self.detector = detector
        self.columns = columns
        self.contamination = contamination
        self.fail_on_anomaly = fail_on_anomaly

    def _execute_operation(
        self,
        data: Any,
        context: Context,
    ) -> AnomalyResult:
        """Execute anomaly detection.

        Args:
            data: Loaded data.
            context: Airflow execution context.

        Returns:
            AnomalyResult from the engine.

        Raises:
            AirflowException: If engine doesn't support anomaly detection.
        """
        from common.engines.base import supports_anomaly

        if not supports_anomaly(self.engine):
            raise AirflowException(
                f"Engine '{self.engine.engine_name}' does not support anomaly detection. "
                f"Use an engine that implements AnomalyDetectionEngine protocol."
            )

        self.log.info(
            f"Executing anomaly detection with detector={self.detector}, "
            f"contamination={self.contamination}"
        )

        detect_kwargs: dict[str, Any] = {
            "detector": self.detector,
            "contamination": self.contamination,
        }
        if self.columns is not None:
            detect_kwargs["columns"] = self.columns

        return self.engine.detect_anomalies(data, **detect_kwargs)

    def _serialize_result(self, result: AnomalyResult) -> dict[str, Any]:
        """Serialize AnomalyResult for XCom.

        Args:
            result: The anomaly detection result.

        Returns:
            XCom-compatible dictionary.
        """
        return result.to_dict()

    def _handle_result(
        self,
        result: AnomalyResult,
        result_dict: dict[str, Any],
        context: Context,
    ) -> None:
        """Handle anomaly result.

        Args:
            result: The anomaly result.
            result_dict: Serialized result.
            context: Airflow execution context.

        Raises:
            AirflowException: If anomalies found and fail_on_anomaly is True.
        """
        if result.has_anomalies:
            anomaly_cols = [a.column for a in result.anomalies if a.is_anomaly]
            self.log.warning(
                f"Anomalies detected: {result.anomalous_row_count}/{result.total_row_count} rows "
                f"({result.anomaly_rate:.2f}%). Columns: {', '.join(anomaly_cols[:5])}"
            )

            if self.fail_on_anomaly:
                raise AirflowException(
                    f"Anomalies detected: {result.anomalous_row_count} anomalous rows "
                    f"({result.anomaly_rate:.2f}%) using {result.detector} detector."
                )
        else:
            self.log.info("No anomalies detected.")

    def _log_metrics(self, result_dict: dict[str, Any]) -> None:
        """Log anomaly detection metrics."""
        self.log.info(
            f"Anomaly Detection Results: "
            f"status={result_dict.get('status', 'unknown')}, "
            f"anomalous_rows={result_dict.get('anomalous_row_count', 0)}, "
            f"total_rows={result_dict.get('total_row_count', 0)}, "
            f"detector={result_dict.get('detector', 'unknown')}, "
            f"duration={result_dict.get('execution_time_ms', 0):.2f}ms"
        )
