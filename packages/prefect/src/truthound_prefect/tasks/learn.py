"""Learn tasks for schema learning.

This module provides Prefect tasks for executing schema learning operations.
Tasks can be used directly or created dynamically using factory functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact

from truthound_prefect.tasks.base import DEFAULT_LEARN_CONFIG, LearnTaskConfig
from truthound_prefect.utils.helpers import format_duration, summarize_learn_result

if TYPE_CHECKING:
    from truthound_prefect.blocks.engine import DataQualityBlock


@task(
    name="data_quality_learn",
    description="Learn data quality rules from the provided data",
    tags=["data-quality", "learn"],
    retries=0,
    retry_delay_seconds=10,
)
async def data_quality_learn_task(
    data: Any,
    block: DataQualityBlock,
    infer_constraints: bool = True,
    min_confidence: float = 0.9,
    categorical_threshold: int = 20,
    store_artifact: bool = True,
    artifact_key: str = "learn_result",
    **kwargs: Any,
) -> dict[str, Any]:
    """Learn data quality rules from data.

    This is the standard learn task that can be used in any Prefect flow.

    Args:
        data: The data to learn from.
        block: DataQualityBlock to use for learning.
        infer_constraints: Infer constraints from data.
        min_confidence: Minimum confidence for learned rules.
        categorical_threshold: Threshold for categorical column detection.
        store_artifact: Store result as Prefect artifact.
        artifact_key: Key for the artifact.
        **kwargs: Additional engine-specific arguments.

    Returns:
        Serialized learn result as a dictionary.

    Example:
        >>> @flow
        ... async def my_flow():
        ...     block = DataQualityBlock(engine_name="truthound")
        ...     baseline_data = load_baseline()
        ...     rules = await data_quality_learn_task(
        ...         data=baseline_data,
        ...         block=block,
        ...         min_confidence=0.95,
        ...     )
        ...     return rules
    """
    logger = get_run_logger()
    logger.info("Starting schema learning")

    # Add options to kwargs
    if infer_constraints:
        kwargs["infer_constraints"] = True
    if min_confidence != 0.9:
        kwargs["min_confidence"] = min_confidence
    if categorical_threshold != 20:
        kwargs["categorical_threshold"] = categorical_threshold

    # Execute learn
    result = block.learn(data, **kwargs)

    # Log summary
    logger.info(summarize_learn_result(result))
    if "execution_time_ms" in result:
        logger.info(f"Duration: {format_duration(result['execution_time_ms'])}")

    # Store artifact if requested
    if store_artifact:
        await _create_learn_artifact(result, artifact_key)

    return result


async def _create_learn_artifact(result: dict[str, Any], key: str) -> None:
    """Create a Prefect table artifact for the learn result."""
    rules = result.get("rules", [])

    rows = [
        {"Metric": "Rules Learned", "Value": str(len(rules))},
    ]

    if "execution_time_ms" in result:
        rows.append({
            "Metric": "Duration",
            "Value": format_duration(result["execution_time_ms"]),
        })

    # Add rule summaries (limit to first 10)
    for rule in rules[:10]:
        confidence = rule.get("confidence", 0)
        rows.append({
            "Metric": f"Rule: {rule.get('rule_type', 'unknown')}",
            "Value": f"{rule.get('column', 'N/A')} | {confidence:.0%} confidence",
        })

    if len(rules) > 10:
        rows.append({
            "Metric": "...",
            "Value": f"and {len(rules) - 10} more rules",
        })

    await create_table_artifact(
        key=key,
        table=rows,
        description="Schema Learning Result",
    )


def create_learn_task(
    name: str,
    config: LearnTaskConfig | None = None,
    block: DataQualityBlock | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Create a configured learn task.

    Factory function to create a learn task with pre-configured settings.
    The returned task can be used in Prefect flows.

    Args:
        name: Name of the task.
        config: Task configuration.
        block: Optional pre-configured block.
        description: Optional task description.

    Returns:
        A configured Prefect task.

    Example:
        >>> # Create a strict learn task
        >>> strict_learn = create_learn_task(
        ...     name="strict_schema_learning",
        ...     config=STRICT_LEARN_CONFIG,
        ... )
        >>>
        >>> @flow
        ... async def my_flow():
        ...     block = DataQualityBlock(engine_name="truthound")
        ...     rules = await strict_learn(data=df, block=block)
    """
    cfg = config or DEFAULT_LEARN_CONFIG

    @task(
        name=name,
        description=description or f"Schema learning task: {name}",
        tags=list(cfg.tags) + ["data-quality", "learn"],
        retries=cfg.retries,
        retry_delay_seconds=cfg.retry_delay_seconds,
        timeout_seconds=cfg.timeout_seconds if cfg.timeout_seconds else None,
    )
    async def learn_task_impl(
        data: Any,
        block: DataQualityBlock | None = block,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the configured learn task."""
        if block is None:
            from truthound_prefect.blocks import DataQualityBlock as DQBlock
            block = DQBlock(engine_name="truthound")

        return await data_quality_learn_task(
            data=data,
            block=block,
            infer_constraints=cfg.infer_constraints,
            min_confidence=cfg.min_confidence,
            categorical_threshold=cfg.categorical_threshold,
            store_artifact=cfg.store_result,
            artifact_key=cfg.result_key,
            **kwargs,
        )

    return learn_task_impl


# Pre-configured tasks
standard_learn_task = create_learn_task(
    name="standard_schema_learning",
    config=DEFAULT_LEARN_CONFIG,
    description="Standard schema learning with default settings",
)

strict_learn_task = create_learn_task(
    name="strict_schema_learning",
    config=LearnTaskConfig(
        min_confidence=0.95,
        tags=frozenset({"strict"}),
    ),
    description="Strict schema learning with 95% confidence threshold",
)


__all__ = [
    # Main task
    "data_quality_learn_task",
    # Factory
    "create_learn_task",
    # Pre-configured tasks
    "standard_learn_task",
    "strict_learn_task",
]
