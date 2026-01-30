"""Data Quality Anomaly Detection Tasks for Prefect.

This module provides anomaly detection tasks for identifying
anomalous patterns in datasets within Prefect flows.

Example:
    >>> from truthound_prefect.tasks.anomaly import data_quality_anomaly_task
    >>> from truthound_prefect.blocks.base import DataQualityBlock
    >>>
    >>> @flow
    ... async def my_flow():
    ...     block = DataQualityBlock(engine_name="truthound")
    ...     result = await data_quality_anomaly_task(
    ...         block=block,
    ...         data_path="s3://bucket/data.parquet",
    ...         detector="isolation_forest",
    ...     )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact

if TYPE_CHECKING:
    from common.base import AnomalyResult
    from truthound_prefect.blocks.base import DataQualityBlock


def _serialize_anomaly_result(result: AnomalyResult) -> dict[str, Any]:
    """Serialize AnomalyResult to a dictionary."""
    result_dict = result.to_dict()
    result_dict["_metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result_dict


@task(
    name="data_quality_anomaly",
    description="Detect anomalous patterns in a dataset.",
    tags=["data-quality", "anomaly"],
    retries=0,
)
async def data_quality_anomaly_task(
    block: DataQualityBlock,
    *,
    data_path: str | None = None,
    data_sql: str | None = None,
    detector: str = "isolation_forest",
    columns: list[str] | None = None,
    contamination: float = 0.05,
    fail_on_anomaly: bool = True,
) -> dict[str, Any]:
    """Detect anomalous patterns in a dataset.

    Args:
        block: DataQualityBlock with engine configuration.
        data_path: Path to data file.
        data_sql: SQL query for data.
        detector: Anomaly detection algorithm.
        columns: Columns to check. None means all.
        contamination: Expected proportion of anomalies.
        fail_on_anomaly: Whether to raise on anomaly detection.

    Returns:
        Serialized AnomalyResult dictionary.

    Raises:
        ValueError: If engine doesn't support anomaly detection.
        RuntimeError: If anomalies detected and fail_on_anomaly is True.
    """
    from common.engines.base import supports_anomaly

    logger = get_run_logger()
    engine = block.get_engine()

    # Verify engine supports anomaly detection
    if not supports_anomaly(engine):
        raise ValueError(
            f"Engine '{engine.engine_name}' does not support anomaly detection. "
            f"Use an engine that implements AnomalyDetectionEngine protocol."
        )

    logger.info(
        "Starting anomaly detection with engine: %s, detector=%s",
        engine.engine_name,
        detector,
    )

    # Validate data source
    if not data_path and not data_sql:
        raise ValueError("Must specify either data_path or data_sql")

    # Load data
    data = block.load_data(data_path=data_path, sql=data_sql)

    # Build kwargs
    detect_kwargs: dict[str, Any] = {
        "detector": detector,
        "contamination": contamination,
    }
    if columns is not None:
        detect_kwargs["columns"] = columns

    # Execute anomaly detection
    result: AnomalyResult = engine.detect_anomalies(data, **detect_kwargs)

    # Serialize
    result_dict = _serialize_anomaly_result(result)
    result_dict["_metadata"]["engine"] = engine.engine_name
    result_dict["_metadata"]["engine_version"] = engine.engine_version

    # Log summary
    logger.info(
        "Anomaly Detection Results: status=%s, anomalies=%d/%d columns, "
        "anomaly_rate=%.2f%%, duration=%.2fms",
        result.status.name,
        result.anomaly_count,
        result.total_columns,
        result.anomaly_rate,
        result.execution_time_ms,
    )

    # Create artifact
    artifact_data = [
        {
            "Column": a.column,
            "Detector": a.detector if hasattr(a, "detector") else detector,
            "Score": f"{a.score:.4f}",
            "Threshold": f"{a.threshold:.4f}",
            "Anomaly": "Yes" if a.is_anomaly else "No",
        }
        for a in (result.anomalies if hasattr(result, "anomalies") else [])
    ]
    if artifact_data:
        await create_table_artifact(
            key="anomaly-result",
            table=artifact_data,
            description=f"Anomaly detection: {result.status.name}",
        )

    # Handle anomalies
    if result.has_anomalies:
        anomaly_names = [a.column for a in result.anomalies if a.is_anomaly]
        logger.warning(
            "Anomalies detected in columns: %s%s",
            ", ".join(anomaly_names[:5]),
            f" ... ({len(anomaly_names) - 5} more)" if len(anomaly_names) > 5 else "",
        )

        if fail_on_anomaly:
            raise RuntimeError(
                f"Anomalies detected: {result.anomaly_count}/{result.total_columns} columns "
                f"({result.anomaly_rate:.2f}%). "
                f"Anomalous: {', '.join(anomaly_names[:3])}"
                + (f" ... ({len(anomaly_names) - 3} more)" if len(anomaly_names) > 3 else "")
            )
    else:
        logger.info("No anomalies detected.")

    return result_dict


def create_anomaly_task(
    name: str = "data_quality_anomaly",
    description: str | None = None,
    tags: list[str] | None = None,
    retries: int = 0,
    retry_delay_seconds: float = 10.0,
) -> Any:
    """Factory function to create a configured anomaly detection task.

    Args:
        name: Task name.
        description: Task description.
        tags: Task tags.
        retries: Number of retries.
        retry_delay_seconds: Delay between retries.

    Returns:
        Configured Prefect task.
    """
    task_tags = ["data-quality", "anomaly"]
    if tags:
        task_tags.extend(tags)

    @task(
        name=name,
        description=description or "Detect anomalies in data.",
        tags=task_tags,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    async def _anomaly_task(
        block: DataQualityBlock,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await data_quality_anomaly_task(block, **kwargs)

    return _anomaly_task


# Pre-configured tasks
strict_anomaly_task = create_anomaly_task(
    name="strict_anomaly_check",
    description="Strict anomaly detection (low contamination).",
    tags=["strict"],
)

lenient_anomaly_task = create_anomaly_task(
    name="lenient_anomaly_check",
    description="Lenient anomaly detection (high contamination).",
    tags=["lenient"],
)


__all__ = [
    # Main task
    "data_quality_anomaly_task",
    # Factory
    "create_anomaly_task",
    # Pre-configured tasks
    "strict_anomaly_task",
    "lenient_anomaly_task",
]
