"""Data Quality Anomaly Detection Op for Dagster.

This module provides the anomaly detection op for identifying
anomalous patterns in datasets within Dagster jobs.

Example:
    >>> from truthound_dagster.ops.anomaly import data_quality_anomaly_op
    >>>
    >>> @job(resource_defs={"data_quality": data_quality_resource})
    ... def my_job():
    ...     data_quality_anomaly_op()
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

from dagster import Config, In, OpExecutionContext, Out, op

from truthound_dagster.ops.base import AnomalyOpConfig

if TYPE_CHECKING:
    from common.base import AnomalyResult


class AnomalyOpDagsterConfig(Config):
    """Dagster configuration schema for anomaly detection ops.

    This class defines the configuration fields that can be set
    in Dagster's configuration system (YAML, Python, etc.).

    Note: Columns are passed as a JSON string due to Dagster's config
    limitations with nested list structures. Use json.dumps() to
    convert columns list to string format.
    """

    data_path: Optional[str] = None
    data_sql: Optional[str] = None
    detector: str = "isolation_forest"
    columns_json: Optional[str] = None
    contamination: float = 0.05
    fail_on_anomaly: bool = True
    timeout_seconds: float = 300.0

    def get_columns(self) -> Optional[List[str]]:
        """Parse columns from JSON string."""
        if self.columns_json is None:
            return None
        import json

        return json.loads(self.columns_json)


def _serialize_anomaly_result(result: "AnomalyResult") -> Dict[str, Any]:
    """Serialize AnomalyResult to a dictionary for Dagster output.

    Args:
        result: Anomaly detection result to serialize.

    Returns:
        Dict[str, Any]: Serialized result.
    """
    result_dict = result.to_dict()
    result_dict["_metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result_dict


@op(
    name="data_quality_anomaly",
    description="Detect anomalous patterns in a dataset.",
    ins={"data": In(description="Data to analyze (DataFrame, path, etc.)")},
    out=Out(description="Anomaly detection result dictionary"),
    tags={"kind": "data_quality", "operation": "anomaly"},
    required_resource_keys={"data_quality"},
)
def data_quality_anomaly_op(
    context: OpExecutionContext,
    data: Any,
    config: AnomalyOpDagsterConfig,
) -> Dict[str, Any]:
    """Detect anomalous patterns in a dataset.

    This op identifies anomalies in data using the configured
    DataQualityResource. It logs the results and returns a serialized
    result dictionary.

    Parameters
    ----------
    context : OpExecutionContext
        Dagster execution context.

    data : Any
        Data to analyze. Can be a DataFrame, file path, or other
        data source supported by the engine.

    config : AnomalyOpDagsterConfig
        Op configuration with anomaly detection settings.

    Returns
    -------
    Dict[str, Any]
        Serialized anomaly result containing:
        - status: Anomaly status (normal, anomaly_detected, warning, error)
        - anomalies: List of anomaly details per column
        - anomaly_rate: Percentage of columns with anomalies
        - execution_time_ms: Execution time

    Raises
    ------
    ValueError
        If engine does not support anomaly detection.
    Exception
        If anomalies are detected and fail_on_anomaly is True.

    Examples
    --------
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def anomaly_job():
    ...     data = load_data_op()
    ...     result = data_quality_anomaly_op(data)
    ...     return result
    """
    from common.engines.base import supports_anomaly

    dq_resource = context.resources.data_quality
    engine = dq_resource.engine if hasattr(dq_resource, "engine") else dq_resource

    # Verify engine supports anomaly detection
    if not supports_anomaly(engine):
        raise ValueError(
            f"Engine '{engine.engine_name}' does not support anomaly detection. "
            f"Use an engine that implements AnomalyDetectionEngine protocol."
        )

    context.log.info(
        f"Starting anomaly detection with engine: {engine.engine_name}, "
        f"detector={config.detector}"
    )

    # Build kwargs
    detect_kwargs: Dict[str, Any] = {
        "detector": config.detector,
        "contamination": config.contamination,
    }
    columns = config.get_columns()
    if columns is not None:
        detect_kwargs["columns"] = columns

    # Execute anomaly detection
    result: AnomalyResult = engine.detect_anomalies(data, **detect_kwargs)

    # Serialize
    result_dict = _serialize_anomaly_result(result)
    result_dict["_metadata"]["engine"] = engine.engine_name
    result_dict["_metadata"]["engine_version"] = engine.engine_version

    # Log summary
    context.log.info(
        f"Anomaly Detection Results: status={result.status.name}, "
        f"anomalies={result.anomaly_count}/{result.total_columns} columns, "
        f"anomaly_rate={result.anomaly_rate:.2f}%, "
        f"duration={result.execution_time_ms:.2f}ms"
    )

    # Add output metadata
    context.add_output_metadata(
        {
            "status": result.status.name,
            "anomaly_count": result.anomaly_count,
            "total_columns": result.total_columns,
            "anomaly_rate": result.anomaly_rate,
            "detector": config.detector,
            "engine": engine.engine_name,
        }
    )

    # Handle anomalies
    if result.has_anomalies:
        anomaly_names = [a.column for a in result.anomalies if a.is_anomaly]
        context.log.warning(
            f"Anomalies detected in columns: {', '.join(anomaly_names[:5])}"
            + (f" ... ({len(anomaly_names) - 5} more)" if len(anomaly_names) > 5 else "")
        )

        if config.fail_on_anomaly:
            raise Exception(
                f"Anomalies detected: {result.anomaly_count}/{result.total_columns} columns "
                f"({result.anomaly_rate:.2f}%). "
                f"Anomalous: {', '.join(anomaly_names[:3])}"
                + (f" ... ({len(anomaly_names) - 3} more)" if len(anomaly_names) > 3 else "")
            )
    else:
        context.log.info("No anomalies detected.")

    return result_dict


def create_anomaly_op(
    name: str,
    *,
    detector: str = "isolation_forest",
    columns: Optional[Sequence[str]] = None,
    contamination: float = 0.05,
    fail_on_anomaly: bool = True,
    timeout_seconds: float = 300.0,
    description: str | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Create a customized anomaly detection op.

    This factory function creates a new op with preset configuration.
    Use this when you need the same anomaly detection configuration
    in multiple places.

    Parameters
    ----------
    name : str
        Name for the op.

    detector : str
        Anomaly detection algorithm.

    columns : Optional[Sequence[str]]
        Columns to check for anomalies.

    contamination : float
        Expected proportion of anomalies.

    fail_on_anomaly : bool
        Whether to raise on anomaly detection.

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
    >>> anomaly_check = create_anomaly_op(
    ...     name="users_anomaly",
    ...     detector="isolation_forest",
    ...     contamination=0.05,
    ...     fail_on_anomaly=True,
    ... )
    >>>
    >>> @job
    ... def pipeline():
    ...     data = load_data()
    ...     anomaly_check(data)
    """
    # Build configuration
    config = AnomalyOpConfig(
        detector=detector,
        columns=tuple(columns) if columns else None,
        contamination=contamination,
        fail_on_anomaly=fail_on_anomaly,
        timeout_seconds=timeout_seconds,
    )

    # Build tags
    op_tags = {"kind": "data_quality", "operation": "anomaly"}
    if tags:
        op_tags.update(tags)

    # Build description
    op_description = description or f"Data quality anomaly detection: {name}"

    @op(
        name=name,
        description=op_description,
        ins={"data": In(description="Data to analyze")},
        out=Out(description="Anomaly detection result"),
        tags=op_tags,
        required_resource_keys={"data_quality"},
    )
    def anomaly_op_impl(
        context: OpExecutionContext,
        data: Any,
    ) -> Dict[str, Any]:
        """Execute configured anomaly detection."""
        from common.engines.base import supports_anomaly

        dq_resource = context.resources.data_quality
        engine = dq_resource.engine if hasattr(dq_resource, "engine") else dq_resource

        if not supports_anomaly(engine):
            raise ValueError(
                f"Engine '{engine.engine_name}' does not support anomaly detection."
            )

        context.log.info(
            f"Starting {name} with detector={config.detector}"
        )

        # Build kwargs
        detect_kwargs: Dict[str, Any] = {
            "detector": config.detector,
            "contamination": config.contamination,
        }
        if config.columns is not None:
            detect_kwargs["columns"] = list(config.columns)

        # Execute
        result: AnomalyResult = engine.detect_anomalies(data, **detect_kwargs)

        # Log results
        if result.has_anomalies:
            anomaly_names = [a.column for a in result.anomalies if a.is_anomaly]
            context.log.warning(
                f"{name} detected anomalies in {len(anomaly_names)} columns"
            )

            if config.fail_on_anomaly:
                raise Exception(
                    f"Anomalies detected: {result.anomaly_count}/{result.total_columns} columns"
                )
        else:
            context.log.info(f"{name}: No anomalies detected.")

        result_dict = _serialize_anomaly_result(result)
        result_dict["_metadata"]["engine"] = engine.engine_name
        result_dict["_metadata"]["engine_version"] = engine.engine_version

        context.add_output_metadata(
            {
                "status": result.status.name,
                "anomaly_count": result.anomaly_count,
                "total_columns": result.total_columns,
                "anomaly_rate": result.anomaly_rate,
            }
        )

        return result_dict

    return anomaly_op_impl
