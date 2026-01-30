"""Data Quality Drift Detection Op for Dagster.

This module provides the drift detection op for executing drift
detection between baseline and current datasets in Dagster jobs.

Example:
    >>> from truthound_dagster.ops.drift import data_quality_drift_op
    >>>
    >>> @job(resource_defs={"data_quality": data_quality_resource})
    ... def my_job():
    ...     data_quality_drift_op()
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from dagster import Config, In, Nothing, OpExecutionContext, Out, op

from truthound_dagster.ops.base import DriftOpConfig

if TYPE_CHECKING:
    from common.base import DriftResult


# Use string annotation for DriftResult since it's TYPE_CHECKING only
class DriftOpDagsterConfig(Config):
    """Dagster configuration schema for drift detection ops.

    This class defines the configuration fields that can be set
    in Dagster's configuration system (YAML, Python, etc.).

    Note: Columns are passed as a JSON string due to Dagster's config
    limitations with nested list structures. Use json.dumps() to
    convert columns list to string format.
    """

    baseline_data_path: Optional[str] = None
    current_data_path: Optional[str] = None
    baseline_sql: Optional[str] = None
    current_sql: Optional[str] = None
    method: str = "auto"
    columns_json: Optional[str] = None
    threshold: float = -1.0  # -1 means not set
    fail_on_drift: bool = True
    timeout_seconds: float = 300.0

    def get_columns(self) -> Optional[List[str]]:
        """Parse columns from JSON string."""
        if self.columns_json is None:
            return None
        import json

        return json.loads(self.columns_json)

    def get_threshold(self) -> Optional[float]:
        """Get threshold, None if not set."""
        return None if self.threshold < 0 else self.threshold


def _serialize_drift_result(result: "DriftResult") -> Dict[str, Any]:
    """Serialize DriftResult to a dictionary for Dagster output.

    Args:
        result: Drift detection result to serialize.

    Returns:
        Dict[str, Any]: Serialized result.
    """
    result_dict = result.to_dict()
    result_dict["_metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result_dict


@op(
    name="data_quality_drift",
    description="Execute drift detection between baseline and current datasets.",
    ins={
        "baseline": In(description="Baseline data (DataFrame, path, etc.)"),
        "current": In(description="Current data (DataFrame, path, etc.)"),
    },
    out=Out(description="Drift detection result dictionary"),
    tags={"kind": "data_quality", "operation": "drift"},
    required_resource_keys={"data_quality"},
)
def data_quality_drift_op(
    context: OpExecutionContext,
    baseline: Any,
    current: Any,
    config: DriftOpDagsterConfig,
) -> Dict[str, Any]:
    """Execute drift detection between baseline and current datasets.

    This op detects statistical drift between a baseline dataset and
    a current dataset using the configured DataQualityResource. It logs
    the results and returns a serialized result dictionary.

    Parameters
    ----------
    context : OpExecutionContext
        Dagster execution context.

    baseline : Any
        Baseline data. Can be a DataFrame, file path, or other
        data source supported by the engine.

    current : Any
        Current data to compare against baseline.

    config : DriftOpDagsterConfig
        Op configuration with drift detection settings.

    Returns
    -------
    Dict[str, Any]
        Serialized drift result containing:
        - status: Drift status (no_drift, drift_detected, warning, error)
        - drifted_columns: List of drifted column details
        - drift_rate: Percentage of columns with drift
        - execution_time_ms: Execution time

    Raises
    ------
    ValueError
        If engine does not support drift detection.
    Exception
        If drift is detected and fail_on_drift is True.

    Examples
    --------
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def drift_job():
    ...     baseline = load_baseline_op()
    ...     current = load_current_op()
    ...     result = data_quality_drift_op(baseline, current)
    ...     return result
    """
    from common.engines.base import supports_drift

    dq_resource = context.resources.data_quality
    engine = dq_resource.engine if hasattr(dq_resource, "engine") else dq_resource

    # Verify engine supports drift detection
    if not supports_drift(engine):
        raise ValueError(
            f"Engine '{engine.engine_name}' does not support drift detection. "
            f"Use an engine that implements DriftDetectionEngine protocol."
        )

    context.log.info(
        f"Starting drift detection with engine: {engine.engine_name}, "
        f"method={config.method}"
    )

    # Build kwargs
    detect_kwargs: Dict[str, Any] = {"method": config.method}
    columns = config.get_columns()
    if columns is not None:
        detect_kwargs["columns"] = columns
    threshold = config.get_threshold()
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
    context.log.info(
        f"Drift Detection Results: status={result.status.name}, "
        f"drifted={result.drifted_count}/{result.total_columns} columns, "
        f"drift_rate={result.drift_rate:.2f}%, "
        f"duration={result.execution_time_ms:.2f}ms"
    )

    # Add output metadata
    context.add_output_metadata(
        {
            "status": result.status.name,
            "drifted_columns": result.drifted_count,
            "total_columns": result.total_columns,
            "drift_rate": result.drift_rate,
            "method": config.method,
            "engine": engine.engine_name,
        }
    )

    # Handle drift
    if result.is_drifted:
        drifted_names = [c.column for c in result.drifted_columns if c.is_drifted]
        context.log.warning(
            f"Drift detected in columns: {', '.join(drifted_names[:5])}"
            + (f" ... ({len(drifted_names) - 5} more)" if len(drifted_names) > 5 else "")
        )

        if config.fail_on_drift:
            raise Exception(
                f"Drift detected: {result.drifted_count}/{result.total_columns} columns "
                f"({result.drift_rate:.2f}%). "
                f"Drifted: {', '.join(drifted_names[:3])}"
                + (f" ... ({len(drifted_names) - 3} more)" if len(drifted_names) > 3 else "")
            )
    else:
        context.log.info("No drift detected.")

    return result_dict


def create_drift_op(
    name: str,
    *,
    method: str = "auto",
    columns: Optional[Sequence[str]] = None,
    threshold: float | None = None,
    fail_on_drift: bool = True,
    timeout_seconds: float = 300.0,
    description: str | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Create a customized drift detection op.

    This factory function creates a new op with preset configuration.
    Use this when you need the same drift detection configuration
    in multiple places.

    Parameters
    ----------
    name : str
        Name for the op.

    method : str
        Drift detection method.

    columns : Optional[Sequence[str]]
        Columns to check for drift.

    threshold : float | None
        Drift threshold.

    fail_on_drift : bool
        Whether to raise on drift detection.

    timeout_seconds : float
        Operation timeout.

    description : str | None
        Op description.

    tags : Optional[Dict[str, str]]
        Additional op tags.

    Returns
    -------
    Callable
        Configured Dagster op.

    Examples
    --------
    >>> drift_check = create_drift_op(
    ...     name="users_drift",
    ...     method="ks",
    ...     threshold=0.05,
    ...     fail_on_drift=True,
    ... )
    >>>
    >>> @job
    ... def pipeline():
    ...     baseline = load_baseline()
    ...     current = load_current()
    ...     drift_check(baseline, current)
    """
    # Build configuration
    config = DriftOpConfig(
        method=method,
        columns=tuple(columns) if columns else None,
        threshold=threshold,
        fail_on_drift=fail_on_drift,
        timeout_seconds=timeout_seconds,
    )

    # Build tags
    op_tags = {"kind": "data_quality", "operation": "drift"}
    if tags:
        op_tags.update(tags)

    # Build description
    op_description = description or f"Data quality drift detection: {name}"

    @op(
        name=name,
        description=op_description,
        ins={
            "baseline": In(description="Baseline data"),
            "current": In(description="Current data"),
        },
        out=Out(description="Drift detection result"),
        tags=op_tags,
        required_resource_keys={"data_quality"},
    )
    def drift_op_impl(
        context: OpExecutionContext,
        baseline: Any,
        current: Any,
    ) -> Dict[str, Any]:
        """Execute configured drift detection."""
        from common.engines.base import supports_drift

        dq_resource = context.resources.data_quality
        engine = dq_resource.engine if hasattr(dq_resource, "engine") else dq_resource

        if not supports_drift(engine):
            raise ValueError(
                f"Engine '{engine.engine_name}' does not support drift detection."
            )

        context.log.info(
            f"Starting {name} with method={config.method}"
        )

        # Build kwargs
        detect_kwargs: Dict[str, Any] = {"method": config.method}
        if config.columns is not None:
            detect_kwargs["columns"] = list(config.columns)
        if config.threshold is not None:
            detect_kwargs["threshold"] = config.threshold

        # Execute
        result: DriftResult = engine.detect_drift(
            baseline, current, **detect_kwargs
        )

        # Log results
        if result.is_drifted:
            drifted_names = [c.column for c in result.drifted_columns if c.is_drifted]
            context.log.warning(
                f"{name} detected drift in {len(drifted_names)} columns"
            )

            if config.fail_on_drift:
                raise Exception(
                    f"Drift detected: {result.drifted_count}/{result.total_columns} columns"
                )
        else:
            context.log.info(f"{name}: No drift detected.")

        result_dict = _serialize_drift_result(result)
        result_dict["_metadata"]["engine"] = engine.engine_name
        result_dict["_metadata"]["engine_version"] = engine.engine_version

        context.add_output_metadata(
            {
                "status": result.status.name,
                "drifted_columns": result.drifted_count,
                "total_columns": result.total_columns,
                "drift_rate": result.drift_rate,
            }
        )

        return result_dict

    return drift_op_impl
