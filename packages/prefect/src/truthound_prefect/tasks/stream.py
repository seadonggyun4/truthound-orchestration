"""Streaming tasks for Prefect data quality execution."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence

from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact

if False:  # pragma: no cover
    from truthound_prefect.blocks.engine import DataQualityBlock


@task(
    name="data_quality_stream",
    description="Execute bounded-memory streaming quality checks",
    tags=["data-quality", "stream"],
    retries=0,
    retry_delay_seconds=10,
)
async def data_quality_stream_task(
    stream: Any,
    block: "DataQualityBlock",
    rules: Sequence[dict[str, Any]] | None = None,
    *,
    batch_size: int = 1000,
    checkpoint: dict[str, Any] | None = None,
    max_batches: int | None = None,
    store_artifact: bool = True,
    artifact_key: str = "stream_result",
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute a streaming quality run and optionally emit Prefect artifacts."""

    logger = get_run_logger()
    logger.info("Starting streaming data quality run")

    if inspect.isasyncgen(stream) or hasattr(stream, "__aiter__"):
        result = await block.astream(
            stream,
            rules=rules,
            batch_size=batch_size,
            checkpoint=checkpoint,
            max_batches=max_batches,
            **kwargs,
        )
    else:
        result = block.stream(
            stream,
            rules=rules,
            batch_size=batch_size,
            checkpoint=checkpoint,
            max_batches=max_batches,
            **kwargs,
        )

    summary = result["summary"]
    logger.info(
        "Streaming quality summary: %s batches, %s failed",
        summary.get("total_batches", 0),
        summary.get("failed_batches", 0),
    )

    if store_artifact:
        await create_table_artifact(
            key=artifact_key,
            table=[
                {"Metric": "Total Batches", "Value": str(summary.get("total_batches", 0))},
                {"Metric": "Total Records", "Value": str(summary.get("total_records", 0))},
                {"Metric": "Failed Batches", "Value": str(summary.get("failed_batches", 0))},
                {"Metric": "Final Status", "Value": str(summary.get("final_status", "unknown"))},
            ],
            description="Streaming data quality summary",
        )

    return result


def create_stream_task(
    name: str,
    *,
    block: "DataQualityBlock",
    batch_size: int = 1000,
    max_batches: int | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Create a configured streaming task."""

    @task(
        name=name,
        description=description or f"Streaming data quality task: {name}",
        tags=["data-quality", "stream"],
    )
    async def stream_task_impl(
        stream: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await data_quality_stream_task(
            stream=stream,
            block=block,
            rules=rules,
            batch_size=batch_size,
            max_batches=max_batches,
            **kwargs,
        )

    return stream_task_impl


__all__ = ["data_quality_stream_task", "create_stream_task"]
