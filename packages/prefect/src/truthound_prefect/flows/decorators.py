"""Flow decorators for data quality integration.

This module provides decorators that wrap Prefect flows with data quality
checks, following the decorator pattern used in the dagster package.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar

from prefect import flow as prefect_flow
from prefect import get_run_logger

from truthound_prefect.flows.config import QualityFlowConfig
from truthound_prefect.utils.exceptions import DataQualityError
from truthound_prefect.utils.helpers import summarize_check_result
from truthound_prefect.utils.types import QualityCheckMode

if TYPE_CHECKING:
    from truthound_prefect.blocks.engine import DataQualityBlock

F = TypeVar("F", bound=Callable[..., Any])


def quality_checked_flow(
    rules: Sequence[dict[str, Any]] | None = None,
    check_mode: QualityCheckMode | str = QualityCheckMode.AFTER,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    auto_schema: bool = False,
    engine_name: str = "truthound",
    store_result: bool = True,
    result_key: str = "quality_result",
    block: DataQualityBlock | None = None,
    config: QualityFlowConfig | None = None,
    **flow_kwargs: Any,
) -> Callable[[F], F]:
    """Decorator to add quality checks to a Prefect flow.

    This decorator wraps a flow function to automatically perform data
    quality checks on the returned data. The wrapped function should
    return data that can be validated.

    Args:
        rules: List of rules to check.
        check_mode: When to check (before, after, both, none).
        fail_on_error: Raise exception on check failures.
        warning_threshold: Failure rate threshold for warnings.
        auto_schema: Use auto-schema mode (Truthound only).
        engine_name: Name of the engine to use.
        store_result: Store result as Prefect artifact.
        result_key: Key for the artifact.
        block: Optional pre-configured block.
        config: Optional QualityFlowConfig (overrides other params).
        **flow_kwargs: Additional arguments passed to @flow decorator.

    Returns:
        Decorated flow function.

    Example:
        >>> @quality_checked_flow(
        ...     rules=[{"type": "not_null", "column": "id"}],
        ...     fail_on_error=True,
        ... )
        ... async def process_users():
        ...     data = load_users()
        ...     return transform(data)
        >>>
        >>> # The flow will check the returned data
        >>> result = await process_users()
    """
    # Build config from parameters or use provided config
    if config is not None:
        cfg = config
    else:
        if isinstance(check_mode, str):
            check_mode = QualityCheckMode(check_mode)
        cfg = QualityFlowConfig(
            check_mode=check_mode,
            rules=tuple(rules) if rules else (),
            fail_on_error=fail_on_error,
            warning_threshold=warning_threshold,
            auto_schema=auto_schema,
            store_results=store_result,
            engine_name=engine_name,
        )

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_run_logger()

            # Get or create block
            dq_block = block
            if dq_block is None:
                from truthound_prefect.blocks import DataQualityBlock
                dq_block = DataQualityBlock(
                    engine_name=cfg.engine_name,
                    auto_schema=cfg.auto_schema,
                )

            # Skip if disabled
            if cfg.check_mode == QualityCheckMode.NONE:
                return await fn(*args, **kwargs)

            # Execute the flow function
            result_data = await fn(*args, **kwargs)

            # Perform quality check on returned data
            if cfg.check_mode in (QualityCheckMode.AFTER, QualityCheckMode.BOTH):
                logger.info("Running data quality check on output...")

                check_kwargs: dict[str, Any] = {}
                if cfg.auto_schema:
                    check_kwargs["auto_schema"] = True

                check_result = dq_block.check(
                    data=result_data,
                    rules=list(cfg.rules) if cfg.rules else None,
                    **check_kwargs,
                )

                # Log summary
                logger.info(summarize_check_result(check_result))

                # Store artifact if requested
                if cfg.store_results:
                    from prefect.artifacts import create_table_artifact
                    await create_table_artifact(
                        key=result_key,
                        table=[
                            {"Metric": "Status", "Value": check_result.get("status", "unknown")},
                            {"Metric": "Passed", "Value": str(check_result.get("passed_count", 0))},
                            {"Metric": "Failed", "Value": str(check_result.get("failed_count", 0))},
                        ],
                        description="Flow Quality Check Result",
                    )

                # Handle failure/warning
                if not check_result.get("is_success", True):
                    failure_rate = check_result.get("failure_rate", 0.0)
                    is_warning = (
                        cfg.warning_threshold is not None
                        and failure_rate <= cfg.warning_threshold
                    )

                    if is_warning:
                        logger.warning(
                            f"Quality check has warnings: {check_result['failed_count']} failures"
                        )
                    elif cfg.fail_on_error:
                        raise DataQualityError(
                            message=f"Quality check failed with {check_result['failed_count']} failures",
                            result=check_result,
                        )

            return result_data

        # Apply Prefect flow decorator
        # Note: Prefect 3.x removed 'tags' from flow decorator
        flow_name = flow_kwargs.pop("name", fn.__name__)
        flow_kwargs.pop("tags", None)  # Remove tags if passed

        decorated = prefect_flow(
            name=flow_name,
            **flow_kwargs,
        )(wrapper)

        return decorated  # type: ignore

    return decorator


def profiled_flow(
    include_histograms: bool = False,
    sample_size: int | None = None,
    store_result: bool = True,
    result_key: str = "profile_result",
    block: DataQualityBlock | None = None,
    **flow_kwargs: Any,
) -> Callable[[F], F]:
    """Decorator to add data profiling to a Prefect flow.

    This decorator wraps a flow function to automatically profile
    the returned data.

    Args:
        include_histograms: Include histogram data in profile.
        sample_size: Maximum rows to sample for profiling.
        store_result: Store result as Prefect artifact.
        result_key: Key for the artifact.
        block: Optional pre-configured block.
        **flow_kwargs: Additional arguments passed to @flow decorator.

    Returns:
        Decorated flow function.

    Example:
        >>> @profiled_flow(include_histograms=True)
        ... async def load_data():
        ...     return read_parquet("data.parquet")
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_run_logger()

            # Get or create block
            dq_block = block
            if dq_block is None:
                from truthound_prefect.blocks import DataQualityBlock
                dq_block = DataQualityBlock(engine_name="truthound")

            # Execute the flow function
            result_data = await fn(*args, **kwargs)

            # Profile the returned data
            logger.info("Profiling output data...")

            profile_kwargs: dict[str, Any] = {}
            if include_histograms:
                profile_kwargs["include_histograms"] = True
            if sample_size is not None:
                profile_kwargs["sample_size"] = sample_size

            profile_result = dq_block.profile(result_data, **profile_kwargs)

            # Log summary
            row_count = profile_result.get("row_count", 0)
            col_count = profile_result.get("column_count", 0)
            logger.info(f"Profile: {row_count:,} rows × {col_count} columns")

            # Store artifact if requested
            if store_result:
                from prefect.artifacts import create_table_artifact
                await create_table_artifact(
                    key=result_key,
                    table=[
                        {"Metric": "Row Count", "Value": f"{row_count:,}"},
                        {"Metric": "Column Count", "Value": str(col_count)},
                    ],
                    description="Flow Profile Result",
                )

            return result_data

        # Apply Prefect flow decorator
        # Note: Prefect 3.x removed 'tags' from flow decorator
        flow_name = flow_kwargs.pop("name", fn.__name__)
        flow_kwargs.pop("tags", None)  # Remove tags if passed

        decorated = prefect_flow(
            name=flow_name,
            **flow_kwargs,
        )(wrapper)

        return decorated  # type: ignore

    return decorator


def validated_flow(
    check_before: bool = False,
    check_after: bool = True,
    rules: Sequence[dict[str, Any]] | None = None,
    auto_schema: bool = False,
    fail_on_error: bool = True,
    **flow_kwargs: Any,
) -> Callable[[F], F]:
    """Decorator for flows with input/output validation.

    Simplified decorator that validates flow inputs and/or outputs.

    Args:
        check_before: Validate first positional argument.
        check_after: Validate returned data.
        rules: List of rules to check.
        auto_schema: Use auto-schema mode.
        fail_on_error: Raise exception on failures.
        **flow_kwargs: Additional arguments passed to @flow decorator.

    Returns:
        Decorated flow function.

    Example:
        >>> @validated_flow(check_before=True, check_after=True)
        ... async def transform_data(input_df):
        ...     return input_df.with_columns(...)
    """
    check_mode = QualityCheckMode.NONE
    if check_before and check_after:
        check_mode = QualityCheckMode.BOTH
    elif check_before:
        check_mode = QualityCheckMode.BEFORE
    elif check_after:
        check_mode = QualityCheckMode.AFTER

    return quality_checked_flow(
        rules=rules,
        check_mode=check_mode,
        fail_on_error=fail_on_error,
        auto_schema=auto_schema,
        **flow_kwargs,
    )


__all__ = [
    "quality_checked_flow",
    "profiled_flow",
    "validated_flow",
]
