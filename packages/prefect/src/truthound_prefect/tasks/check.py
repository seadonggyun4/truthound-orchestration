"""Check tasks for data quality validation.

This module provides Prefect tasks for executing data quality checks.
Tasks can be used directly or created dynamically using factory functions.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Sequence

from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact

from truthound_prefect.tasks.base import CheckTaskConfig, DEFAULT_CHECK_CONFIG
from truthound_prefect.utils.exceptions import DataQualityError
from truthound_prefect.utils.helpers import format_duration, summarize_check_result
from truthound_prefect.utils.serialization import serialize_result

if TYPE_CHECKING:
    from truthound_prefect.blocks.engine import DataQualityBlock


@task(
    name="data_quality_check",
    description="Execute a data quality check on the provided data",
    tags=["data-quality", "check"],
    retries=0,
    retry_delay_seconds=10,
)
async def data_quality_check_task(
    data: Any,
    block: DataQualityBlock,
    rules: Sequence[dict[str, Any]] | None = None,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    auto_schema: bool = False,
    store_artifact: bool = True,
    artifact_key: str = "check_result",
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute a data quality check.

    This is the standard check task that can be used in any Prefect flow.

    Args:
        data: The data to check.
        block: DataQualityBlock to use for the check.
        rules: Optional list of rules to check.
        fail_on_error: Raise exception on check failures.
        warning_threshold: Failure rate threshold for warnings (0.0 to 1.0).
        auto_schema: Use auto-schema mode (Truthound only).
        store_artifact: Store result as Prefect artifact.
        artifact_key: Key for the artifact.
        **kwargs: Additional engine-specific arguments.

    Returns:
        Serialized check result as a dictionary.

    Raises:
        DataQualityError: If the check fails and fail_on_error is True.

    Example:
        >>> @flow
        ... async def my_flow():
        ...     block = DataQualityBlock(engine_name="truthound")
        ...     data = load_data()
        ...     result = await data_quality_check_task(
        ...         data=data,
        ...         block=block,
        ...         auto_schema=True,
        ...     )
        ...     return result
    """
    logger = get_run_logger()
    logger.info("Starting data quality check")

    # Apply auto_schema if specified
    if auto_schema:
        kwargs["auto_schema"] = True

    # Execute check
    result = block.check(data, rules=list(rules) if rules else None, **kwargs)

    # Log summary
    logger.info(summarize_check_result(result))
    if "execution_time_ms" in result:
        logger.info(f"Duration: {format_duration(result['execution_time_ms'])}")

    # Store artifact if requested
    if store_artifact:
        await _create_check_artifact(result, artifact_key)

    # Handle failure/warning
    if not result.get("is_success", True):
        failure_rate = result.get("failure_rate", 0.0)
        is_warning = (
            warning_threshold is not None
            and failure_rate <= warning_threshold
        )

        if is_warning:
            logger.warning(
                f"Data quality check has warnings: {result['failed_count']} failures "
                f"(within threshold of {warning_threshold:.1%})"
            )
        elif fail_on_error:
            raise DataQualityError(
                message=f"Data quality check failed with {result['failed_count']} failures",
                result=result,
            )
        else:
            logger.error(f"Data quality check failed with {result['failed_count']} failures")

    return result


async def _create_check_artifact(result: dict[str, Any], key: str) -> None:
    """Create a Prefect table artifact for the check result."""
    rows = [
        {"Metric": "Status", "Value": result.get("status", "unknown")},
        {"Metric": "Passed Rules", "Value": str(result.get("passed_count", 0))},
        {"Metric": "Failed Rules", "Value": str(result.get("failed_count", 0))},
        {"Metric": "Failure Rate", "Value": f"{result.get('failure_rate', 0):.2%}"},
    ]

    if "execution_time_ms" in result:
        rows.append({
            "Metric": "Duration",
            "Value": format_duration(result["execution_time_ms"]),
        })

    await create_table_artifact(
        key=key,
        table=rows,
        description="Data Quality Check Result",
    )


def create_check_task(
    name: str,
    config: CheckTaskConfig | None = None,
    block: DataQualityBlock | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Create a configured check task.

    Factory function to create a check task with pre-configured settings.
    The returned task can be used in Prefect flows.

    Args:
        name: Name of the task.
        config: Task configuration.
        block: Optional pre-configured block.
        description: Optional task description.

    Returns:
        A configured Prefect task.

    Example:
        >>> # Create a strict check task
        >>> strict_check = create_check_task(
        ...     name="strict_validation",
        ...     config=STRICT_CHECK_CONFIG,
        ... )
        >>>
        >>> @flow
        ... async def my_flow():
        ...     block = DataQualityBlock(engine_name="truthound")
        ...     result = await strict_check(data=df, block=block)
    """
    cfg = config or DEFAULT_CHECK_CONFIG

    @task(
        name=name,
        description=description or f"Data quality check task: {name}",
        tags=list(cfg.tags) + ["data-quality", "check"],
        retries=cfg.retries,
        retry_delay_seconds=cfg.retry_delay_seconds,
        timeout_seconds=cfg.timeout_seconds if cfg.timeout_seconds else None,
    )
    async def check_task_impl(
        data: Any,
        block: DataQualityBlock | None = block,
        rules: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the configured check task."""
        if block is None:
            from truthound_prefect.blocks import DataQualityBlock as DQBlock
            block = DQBlock(engine_name="truthound", auto_schema=cfg.auto_schema)

        # Use config rules if not provided
        effective_rules = rules or list(cfg.rules)

        return await data_quality_check_task(
            data=data,
            block=block,
            rules=effective_rules,
            fail_on_error=cfg.fail_on_error,
            warning_threshold=cfg.warning_threshold,
            auto_schema=cfg.auto_schema,
            store_artifact=cfg.store_result,
            artifact_key=cfg.result_key,
            **kwargs,
        )

    return check_task_impl


# Pre-configured tasks
strict_check_task = create_check_task(
    name="strict_data_quality_check",
    config=CheckTaskConfig(
        fail_on_error=True,
        warning_threshold=None,
        tags=frozenset({"strict"}),
    ),
    description="Strict data quality check that fails on any error",
)

lenient_check_task = create_check_task(
    name="lenient_data_quality_check",
    config=CheckTaskConfig(
        fail_on_error=False,
        warning_threshold=0.10,
        tags=frozenset({"lenient"}),
    ),
    description="Lenient data quality check with 10% warning threshold",
)

auto_schema_check_task = create_check_task(
    name="auto_schema_check",
    config=CheckTaskConfig(
        auto_schema=True,
        tags=frozenset({"auto-schema"}),
    ),
    description="Data quality check using Truthound auto-schema",
)


__all__ = [
    # Main task
    "data_quality_check_task",
    # Factory
    "create_check_task",
    # Pre-configured tasks
    "strict_check_task",
    "lenient_check_task",
    "auto_schema_check_task",
]
