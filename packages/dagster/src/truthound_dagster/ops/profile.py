"""Data Quality Profile Ops for Dagster.

This module provides Dagster ops for executing data profiling operations.
Profiling analyzes data to understand its structure, statistics, and patterns.

Example:
    >>> from dagster import job
    >>> from truthound_dagster.ops import data_quality_profile_op
    >>> from truthound_dagster.resources import DataQualityResource
    >>>
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def profile_job():
    ...     data_quality_profile_op()
"""



from typing import TYPE_CHECKING, Any, Callable, Optional, Dict, List

from dagster import Config, In, OpExecutionContext, Out, op

from truthound_dagster.ops.base import ProfileOpConfig

if TYPE_CHECKING:
    from common.base import ProfileResult

    from truthound_dagster.resources import DataQualityResource


class ProfileOpDagsterConfig(Config):
    """Dagster configuration schema for profile ops."""

    include_histograms: bool = True
    include_samples: bool = True
    sample_size: int = 10
    timeout_seconds: float = 300.0


def _serialize_profile_result(result: "ProfileResult") -> Dict[str, Any]:
    """Serialize ProfileResult for Dagster metadata.

    Args:
        result: Profile result to serialize.

    Returns:
        Dict[str, Any]: Serialized result.
    """
    columns = []
    for col in result.columns:
        col_data = {
            "column_name": col.column_name,
            "dtype": str(col.dtype),
            "null_count": col.null_count,
            "null_percentage": col.null_percentage,
            "unique_count": col.unique_count,
            "unique_percentage": col.unique_percentage,
        }

        # Add numeric stats if available
        if hasattr(col, "min_value") and col.min_value is not None:
            col_data.update(
                {
                    "min_value": col.min_value,
                    "max_value": col.max_value,
                    "mean": getattr(col, "mean", None),
                    "std": getattr(col, "std", None),
                }
            )

        columns.append(col_data)

    return {
        "row_count": result.row_count,
        "column_count": result.column_count,
        "columns": columns,
        "execution_time_ms": result.execution_time_ms,
        "timestamp": result.timestamp.isoformat(),
    }


@op(
    name="data_quality_profile",
    description="Execute data profiling to analyze data structure and statistics.",
    ins={"data": In(description="Data to profile (DataFrame, path, etc.)")},
    out=Out(description="Profile result with statistics"),
    tags={"kind": "data_quality", "operation": "profile"},
    required_resource_keys={"data_quality"},
)
def data_quality_profile_op(
    context: OpExecutionContext,
    data: Any,
    config: ProfileOpDagsterConfig,
) -> Dict[str, Any]:
    """Execute data profiling.

    This op profiles data to understand its structure, statistics,
    and patterns. Results include column types, null counts,
    unique counts, and optional statistical summaries.

    Parameters
    ----------
    context : OpExecutionContext
        Dagster execution context.

    data : Any
        Data to profile.

    config : ProfileOpDagsterConfig
        Op configuration.

    Returns
    -------
    Dict[str, Any]
        Serialized profile result containing:
        - row_count: Number of rows
        - column_count: Number of columns
        - columns: List of column profiles
        - execution_time_ms: Execution time

    Examples
    --------
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def profile_job():
    ...     data = load_data_op()
    ...     profile = data_quality_profile_op(data)
    ...     return profile
    """
    dq_resource: DataQualityResource = context.resources.data_quality

    context.log.info("Starting data profiling")

    # Execute profiling
    result = dq_resource.profile(
        data=data,
        timeout=config.timeout_seconds,
        include_histograms=config.include_histograms,
        include_samples=config.include_samples,
        sample_size=config.sample_size,
    )

    context.log.info(
        f"Profiling complete: {result.row_count} rows, "
        f"{result.column_count} columns in {result.execution_time_ms:.2f}ms"
    )

    # Log column summaries
    for col in result.columns[:5]:
        context.log.info(
            f"  {col.column_name}: {col.dtype}, "
            f"null={col.null_percentage:.1f}%, "
            f"unique={col.unique_percentage:.1f}%"
        )
    if len(result.columns) > 5:
        context.log.info(f"  ... and {len(result.columns) - 5} more columns")

    # Serialize and return
    result_dict = _serialize_profile_result(result)

    # Add metadata
    context.add_output_metadata(
        {
            "row_count": result.row_count,
            "column_count": result.column_count,
            "execution_time_ms": result.execution_time_ms,
        }
    )

    return result_dict


def create_profile_op(
    name: str,
    *,
    include_histograms: bool = True,
    include_samples: bool = True,
    sample_size: int = 10,
    timeout_seconds: float = 300.0,
    description: str | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Create a customized data profiling op.

    Parameters
    ----------
    name : str
        Name for the op.

    include_histograms : bool
        Include histogram data.

    include_samples : bool
        Include sample values.

    sample_size : int
        Number of sample values.

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
    >>> users_profile = create_profile_op(
    ...     name="users_profile",
    ...     include_histograms=True,
    ... )
    """
    config = ProfileOpConfig(
        include_histograms=include_histograms,
        include_samples=include_samples,
        sample_size=sample_size,
        timeout_seconds=timeout_seconds,
    )

    op_tags = {"kind": "data_quality", "operation": "profile"}
    if tags:
        op_tags.update(tags)

    op_description = description or f"Data profiling: {name}"

    @op(
        name=name,
        description=op_description,
        ins={"data": In(description="Data to profile")},
        out=Out(description="Profile result"),
        tags=op_tags,
        required_resource_keys={"data_quality"},
    )
    def profile_op_impl(context: OpExecutionContext, data: Any) -> Dict[str, Any]:
        """Execute configured data profiling."""
        dq_resource: DataQualityResource = context.resources.data_quality

        context.log.info(f"Starting {name}")

        result = dq_resource.profile(
            data=data,
            timeout=config.timeout_seconds,
            include_histograms=config.include_histograms,
            include_samples=config.include_samples,
            sample_size=config.sample_size,
        )

        context.log.info(
            f"{name} complete: {result.row_count} rows, "
            f"{result.column_count} columns"
        )

        result_dict = _serialize_profile_result(result)

        context.add_output_metadata(
            {
                "row_count": result.row_count,
                "column_count": result.column_count,
            }
        )

        return result_dict

    return profile_op_impl
