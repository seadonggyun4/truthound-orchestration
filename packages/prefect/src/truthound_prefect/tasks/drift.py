"""Data Quality Drift Detection Tasks for Prefect.

This module provides drift detection tasks for comparing baseline
and current datasets within Prefect flows.

Example:
    >>> from truthound_prefect.tasks.drift import data_quality_drift_task
    >>> from truthound_prefect.blocks.base import DataQualityBlock
    >>>
    >>> @flow
    ... async def my_flow():
    ...     block = DataQualityBlock(engine_name="truthound")
    ...     result = await data_quality_drift_task(
    ...         block=block,
    ...         baseline_data_path="s3://bucket/baseline.parquet",
    ...         current_data_path="s3://bucket/current.parquet",
    ...         method="ks",
    ...     )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact

if TYPE_CHECKING:
    from common.base import DriftResult
    from truthound_prefect.blocks.base import DataQualityBlock


def _serialize_drift_result(result: DriftResult) -> dict[str, Any]:
    """Serialize DriftResult to a dictionary."""
    result_dict = result.to_dict()
    result_dict["_metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result_dict


@task(
    name="data_quality_drift",
    description="Detect drift between baseline and current datasets.",
    tags=["data-quality", "drift"],
    retries=0,
)
async def data_quality_drift_task(
    block: DataQualityBlock,
    *,
    baseline_data_path: str | None = None,
    current_data_path: str | None = None,
    baseline_sql: str | None = None,
    current_sql: str | None = None,
    method: str = "auto",
    columns: list[str] | None = None,
    threshold: float | None = None,
    fail_on_drift: bool = True,
) -> dict[str, Any]:
    """Detect drift between baseline and current datasets.

    Args:
        block: DataQualityBlock with engine configuration.
        baseline_data_path: Path to baseline data file.
        current_data_path: Path to current data file.
        baseline_sql: SQL query for baseline data.
        current_sql: SQL query for current data.
        method: Statistical method for drift detection.
        columns: Columns to check. None means all.
        threshold: Drift detection threshold.
        fail_on_drift: Whether to raise on drift detection.

    Returns:
        Serialized DriftResult dictionary.

    Raises:
        ValueError: If engine doesn't support drift detection.
        RuntimeError: If drift detected and fail_on_drift is True.
    """
    from common.engines.base import supports_drift

    logger = get_run_logger()
    engine = block.get_engine()

    # Verify engine supports drift detection
    if not supports_drift(engine):
        raise ValueError(
            f"Engine '{engine.engine_name}' does not support drift detection. "
            f"Use an engine that implements DriftDetectionEngine protocol."
        )

    logger.info(
        "Starting drift detection with engine: %s, method=%s",
        engine.engine_name,
        method,
    )

    # Validate data sources
    if not baseline_data_path and not baseline_sql:
        raise ValueError("Must specify either baseline_data_path or baseline_sql")
    if not current_data_path and not current_sql:
        raise ValueError("Must specify either current_data_path or current_sql")

    # Load data
    baseline = block.load_data(
        data_path=baseline_data_path, sql=baseline_sql
    )
    current = block.load_data(
        data_path=current_data_path, sql=current_sql
    )

    # Build kwargs
    detect_kwargs: dict[str, Any] = {"method": method}
    if columns is not None:
        detect_kwargs["columns"] = columns
    if threshold is not None:
        detect_kwargs["threshold"] = threshold

    # Execute drift detection
    result: DriftResult = engine.detect_drift(
        baseline, current, **detect_kwargs
    )

    # Serialize
    result_dict = _serialize_drift_result(result)
    result_dict["_metadata"]["engine"] = engine.engine_name
    result_dict["_metadata"]["engine_version"] = engine.engine_version

    # Log summary
    logger.info(
        "Drift Detection Results: status=%s, drifted=%d/%d columns, "
        "drift_rate=%.2f%%, duration=%.2fms",
        result.status.name,
        result.drifted_count,
        result.total_columns,
        result.drift_rate,
        result.execution_time_ms,
    )

    # Create artifact
    artifact_data = [
        {
            "Column": c.column,
            "Method": c.method,
            "Statistic": f"{c.statistic:.4f}" if c.statistic else "N/A",
            "P-Value": f"{c.p_value:.4f}" if c.p_value else "N/A",
            "Drifted": "Yes" if c.is_drifted else "No",
            "Severity": c.severity if hasattr(c, "severity") else "N/A",
        }
        for c in (result.drifted_columns if hasattr(result, "drifted_columns") else [])
    ]
    if artifact_data:
        await create_table_artifact(
            key="drift-result",
            table=artifact_data,
            description=f"Drift detection: {result.status.name}",
        )

    # Handle drift
    if result.is_drifted:
        drifted_names = [c.column for c in result.drifted_columns if c.is_drifted]
        logger.warning(
            "Drift detected in columns: %s%s",
            ", ".join(drifted_names[:5]),
            f" ... ({len(drifted_names) - 5} more)" if len(drifted_names) > 5 else "",
        )

        if fail_on_drift:
            raise RuntimeError(
                f"Drift detected: {result.drifted_count}/{result.total_columns} columns "
                f"({result.drift_rate:.2f}%). "
                f"Drifted: {', '.join(drifted_names[:3])}"
                + (f" ... ({len(drifted_names) - 3} more)" if len(drifted_names) > 3 else "")
            )
    else:
        logger.info("No drift detected.")

    return result_dict


def create_drift_task(
    name: str = "data_quality_drift",
    description: str | None = None,
    tags: list[str] | None = None,
    retries: int = 0,
    retry_delay_seconds: float = 10.0,
) -> Any:
    """Factory function to create a configured drift detection task.

    Args:
        name: Task name.
        description: Task description.
        tags: Task tags.
        retries: Number of retries.
        retry_delay_seconds: Delay between retries.

    Returns:
        Configured Prefect task.
    """
    task_tags = ["data-quality", "drift"]
    if tags:
        task_tags.extend(tags)

    @task(
        name=name,
        description=description or "Detect drift between datasets.",
        tags=task_tags,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    async def _drift_task(
        block: DataQualityBlock,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await data_quality_drift_task(block, **kwargs)

    return _drift_task


# Pre-configured tasks
strict_drift_task = create_drift_task(
    name="strict_drift_check",
    description="Strict drift detection (low threshold).",
    tags=["strict"],
)

lenient_drift_task = create_drift_task(
    name="lenient_drift_check",
    description="Lenient drift detection (high threshold).",
    tags=["lenient"],
)


__all__ = [
    # Main task
    "data_quality_drift_task",
    # Factory
    "create_drift_task",
    # Pre-configured tasks
    "strict_drift_task",
    "lenient_drift_task",
]
