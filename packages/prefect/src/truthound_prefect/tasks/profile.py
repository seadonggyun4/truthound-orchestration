"""Profile tasks for data profiling.

This module provides Prefect tasks for executing data profiling operations.
Tasks can be used directly or created dynamically using factory functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact

from truthound_prefect.tasks.base import DEFAULT_PROFILE_CONFIG, ProfileTaskConfig
from truthound_prefect.utils.helpers import format_count, format_duration, summarize_profile_result

if TYPE_CHECKING:
    from truthound_prefect.blocks.engine import DataQualityBlock


@task(
    name="data_quality_profile",
    description="Profile the provided data to understand its characteristics",
    tags=["data-quality", "profile"],
    retries=0,
    retry_delay_seconds=10,
)
async def data_quality_profile_task(
    data: Any,
    block: DataQualityBlock,
    include_histograms: bool = False,
    sample_size: int | None = None,
    store_artifact: bool = True,
    artifact_key: str = "profile_result",
    **kwargs: Any,
) -> dict[str, Any]:
    """Profile data to understand its characteristics.

    This is the standard profile task that can be used in any Prefect flow.

    Args:
        data: The data to profile.
        block: DataQualityBlock to use for profiling.
        include_histograms: Include histogram data in profile.
        sample_size: Maximum rows to sample for profiling.
        store_artifact: Store result as Prefect artifact.
        artifact_key: Key for the artifact.
        **kwargs: Additional engine-specific arguments.

    Returns:
        Serialized profile result as a dictionary.

    Example:
        >>> @flow
        ... async def my_flow():
        ...     block = DataQualityBlock(engine_name="truthound")
        ...     data = load_data()
        ...     profile = await data_quality_profile_task(
        ...         data=data,
        ...         block=block,
        ...         include_histograms=True,
        ...     )
        ...     return profile
    """
    logger = get_run_logger()
    logger.info("Starting data profiling")

    # Add options to kwargs
    if include_histograms:
        kwargs["include_histograms"] = True
    if sample_size is not None:
        kwargs["sample_size"] = sample_size

    # Execute profile
    result = block.profile(data, **kwargs)

    # Log summary
    logger.info(summarize_profile_result(result))
    if "execution_time_ms" in result:
        logger.info(f"Duration: {format_duration(result['execution_time_ms'])}")

    # Store artifact if requested
    if store_artifact:
        await _create_profile_artifact(result, artifact_key)

    return result


async def _create_profile_artifact(result: dict[str, Any], key: str) -> None:
    """Create a Prefect table artifact for the profile result."""
    rows = [
        {"Metric": "Row Count", "Value": format_count(result.get("row_count", 0))},
        {"Metric": "Column Count", "Value": str(result.get("column_count", 0))},
    ]

    if "execution_time_ms" in result:
        rows.append({
            "Metric": "Duration",
            "Value": format_duration(result["execution_time_ms"]),
        })

    # Add column summaries (limit to first 10)
    columns = result.get("columns", [])[:10]
    for col in columns:
        null_pct = col.get("null_percentage", 0)
        rows.append({
            "Metric": f"Column: {col.get('column_name', 'N/A')}",
            "Value": f"{col.get('dtype', 'unknown')} | {null_pct:.1%} null",
        })

    if len(result.get("columns", [])) > 10:
        rows.append({
            "Metric": "...",
            "Value": f"and {len(result.get('columns', [])) - 10} more columns",
        })

    await create_table_artifact(
        key=key,
        table=rows,
        description="Data Profile Result",
    )


def create_profile_task(
    name: str,
    config: ProfileTaskConfig | None = None,
    block: DataQualityBlock | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Create a configured profile task.

    Factory function to create a profile task with pre-configured settings.
    The returned task can be used in Prefect flows.

    Args:
        name: Name of the task.
        config: Task configuration.
        block: Optional pre-configured block.
        description: Optional task description.

    Returns:
        A configured Prefect task.

    Example:
        >>> # Create a minimal profile task
        >>> quick_profile = create_profile_task(
        ...     name="quick_profile",
        ...     config=MINIMAL_PROFILE_CONFIG,
        ... )
        >>>
        >>> @flow
        ... async def my_flow():
        ...     block = DataQualityBlock(engine_name="truthound")
        ...     result = await quick_profile(data=df, block=block)
    """
    cfg = config or DEFAULT_PROFILE_CONFIG

    @task(
        name=name,
        description=description or f"Data profiling task: {name}",
        tags=list(cfg.tags) + ["data-quality", "profile"],
        retries=cfg.retries,
        retry_delay_seconds=cfg.retry_delay_seconds,
        timeout_seconds=cfg.timeout_seconds if cfg.timeout_seconds else None,
    )
    async def profile_task_impl(
        data: Any,
        block: DataQualityBlock | None = block,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the configured profile task."""
        if block is None:
            from truthound_prefect.blocks import DataQualityBlock as DQBlock
            block = DQBlock(engine_name="truthound")

        return await data_quality_profile_task(
            data=data,
            block=block,
            include_histograms=cfg.include_histograms,
            sample_size=cfg.sample_size,
            store_artifact=cfg.store_result,
            artifact_key=cfg.result_key,
            **kwargs,
        )

    return profile_task_impl


# Pre-configured tasks
minimal_profile_task = create_profile_task(
    name="minimal_data_profile",
    config=ProfileTaskConfig(
        include_histograms=False,
        sample_size=10000,
        tags=frozenset({"minimal"}),
    ),
    description="Minimal data profiling with sampling",
)

full_profile_task = create_profile_task(
    name="full_data_profile",
    config=ProfileTaskConfig(
        include_histograms=True,
        sample_size=None,
        tags=frozenset({"full"}),
    ),
    description="Full data profiling with histograms",
)


__all__ = [
    # Main task
    "data_quality_profile_task",
    # Factory
    "create_profile_task",
    # Pre-configured tasks
    "minimal_profile_task",
    "full_profile_task",
]
