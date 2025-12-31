"""Flow factory functions for creating data quality flows.

This module provides factory functions to create configured Prefect flows
for common data quality patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from prefect import flow as prefect_flow
from prefect import get_run_logger

from truthound_prefect.flows.config import PipelineFlowConfig, QualityFlowConfig
from truthound_prefect.utils.exceptions import DataQualityError
from truthound_prefect.utils.helpers import summarize_check_result
from truthound_prefect.utils.types import QualityCheckMode

if TYPE_CHECKING:
    from truthound_prefect.blocks.engine import DataQualityBlock


def create_quality_flow(
    name: str,
    loader: Callable[..., Any],
    rules: Sequence[dict[str, Any]] | None = None,
    config: QualityFlowConfig | None = None,
    block: DataQualityBlock | None = None,
    description: str | None = None,
    **flow_kwargs: Any,
) -> Callable[..., Any]:
    """Create a flow that loads data and performs quality checks.

    Factory function to create a complete data quality flow.

    Args:
        name: Name of the flow.
        loader: Callable that loads and returns the data.
        rules: List of rules to check.
        config: Optional flow configuration.
        block: Optional pre-configured block.
        description: Optional flow description.
        **flow_kwargs: Additional arguments passed to @flow decorator.

    Returns:
        A configured Prefect flow.

    Example:
        >>> def load_users():
        ...     return pl.read_parquet("users.parquet")
        >>>
        >>> users_quality_flow = create_quality_flow(
        ...     name="users_quality_check",
        ...     loader=load_users,
        ...     rules=[{"type": "not_null", "column": "id"}],
        ... )
        >>>
        >>> # Run the flow
        >>> result = await users_quality_flow()
    """
    cfg = config or QualityFlowConfig(
        rules=tuple(rules) if rules else (),
    )

    # Note: Prefect 3.x removed 'tags' from flow decorator
    @prefect_flow(
        name=name,
        description=description or f"Data quality flow: {name}",
        **flow_kwargs,
    )
    async def quality_flow_impl(**kwargs: Any) -> dict[str, Any]:
        """Execute the quality flow."""
        logger = get_run_logger()

        # Get or create block
        dq_block = block
        if dq_block is None:
            from truthound_prefect.blocks import DataQualityBlock
            dq_block = DataQualityBlock(
                engine_name=cfg.engine_name,
                auto_schema=cfg.auto_schema,
            )

        # Load data
        logger.info(f"Loading data using {loader.__name__}...")
        if callable(loader):
            data = loader(**kwargs) if kwargs else loader()
        else:
            data = loader

        # Skip check if disabled
        if cfg.check_mode == QualityCheckMode.NONE:
            return {"data": data, "check_result": None}

        # Perform quality check
        logger.info("Running quality check...")

        check_kwargs: dict[str, Any] = {}
        if cfg.auto_schema:
            check_kwargs["auto_schema"] = True

        check_result = dq_block.check(
            data=data,
            rules=list(cfg.rules) if cfg.rules else None,
            **check_kwargs,
        )

        # Log summary
        logger.info(summarize_check_result(check_result))

        # Store artifact if requested
        if cfg.store_results:
            from prefect.artifacts import create_table_artifact
            await create_table_artifact(
                key=f"{name}_result",
                table=[
                    {"Metric": "Status", "Value": check_result.get("status", "unknown")},
                    {"Metric": "Passed", "Value": str(check_result.get("passed_count", 0))},
                    {"Metric": "Failed", "Value": str(check_result.get("failed_count", 0))},
                ],
                description=f"Quality Check Result: {name}",
            )

        # Handle failure
        if not check_result.get("is_success", True):
            failure_rate = check_result.get("failure_rate", 0.0)
            is_warning = (
                cfg.warning_threshold is not None
                and failure_rate <= cfg.warning_threshold
            )

            if is_warning:
                logger.warning(f"Quality check has warnings: {check_result['failed_count']} failures")
            elif cfg.fail_on_error:
                raise DataQualityError(
                    message=f"Quality check failed with {check_result['failed_count']} failures",
                    result=check_result,
                )

        return {"data": data, "check_result": check_result}

    return quality_flow_impl


def create_validation_flow(
    name: str,
    source: str | Callable[..., Any],
    rules: Sequence[dict[str, Any]],
    fail_on_error: bool = True,
    store_result: bool = True,
    description: str | None = None,
    **flow_kwargs: Any,
) -> Callable[..., Any]:
    """Create a standalone validation flow.

    Creates a flow specifically for validating data without returning it.

    Args:
        name: Name of the flow.
        source: Data source (path or loader function).
        rules: List of rules to check.
        fail_on_error: Raise exception on failures.
        store_result: Store result as Prefect artifact.
        description: Optional flow description.
        **flow_kwargs: Additional arguments passed to @flow decorator.

    Returns:
        A configured Prefect flow.

    Example:
        >>> validate_orders = create_validation_flow(
        ...     name="validate_orders",
        ...     source="s3://bucket/orders.parquet",
        ...     rules=[
        ...         {"type": "not_null", "column": "order_id"},
        ...         {"type": "in_range", "column": "amount", "min": 0},
        ...     ],
        ... )
    """
    config = QualityFlowConfig(
        rules=tuple(rules),
        fail_on_error=fail_on_error,
        store_results=store_result,
    )

    def loader() -> Any:
        if callable(source):
            return source()
        else:
            # Assume it's a path
            try:
                import polars as pl
                if source.endswith(".parquet"):
                    return pl.read_parquet(source)
                elif source.endswith(".csv"):
                    return pl.read_csv(source)
                else:
                    return pl.read_parquet(source)
            except ImportError:
                import pandas as pd
                if source.endswith(".parquet"):
                    return pd.read_parquet(source)
                elif source.endswith(".csv"):
                    return pd.read_csv(source)
                else:
                    return pd.read_parquet(source)

    return create_quality_flow(
        name=name,
        loader=loader,
        config=config,
        description=description or f"Validation flow: {name}",
        **flow_kwargs,
    )


def create_pipeline_flow(
    name: str,
    stages: Sequence[dict[str, Any]],
    config: PipelineFlowConfig | None = None,
    block: DataQualityBlock | None = None,
    description: str | None = None,
    **flow_kwargs: Any,
) -> Callable[..., Any]:
    """Create a multi-stage pipeline flow with quality checks.

    Creates a flow that executes multiple stages with optional quality
    checks at each stage.

    Args:
        name: Name of the flow.
        stages: List of stage definitions, each with:
            - name: Stage name
            - loader: Callable that returns data
            - rules: Optional rules to check (default: auto_schema)
            - check: Whether to check (default: True)
        config: Optional pipeline configuration.
        block: Optional pre-configured block.
        description: Optional flow description.
        **flow_kwargs: Additional arguments passed to @flow decorator.

    Returns:
        A configured Prefect flow.

    Example:
        >>> pipeline = create_pipeline_flow(
        ...     name="etl_pipeline",
        ...     stages=[
        ...         {"name": "extract", "loader": extract_data, "check": True},
        ...         {"name": "transform", "loader": transform_data, "check": True},
        ...         {"name": "load", "loader": load_data, "check": False},
        ...     ],
        ... )
    """
    cfg = config or PipelineFlowConfig()

    # Note: Prefect 3.x removed 'tags' from flow decorator
    @prefect_flow(
        name=name,
        description=description or f"Pipeline flow: {name}",
        **flow_kwargs,
    )
    async def pipeline_flow_impl(**kwargs: Any) -> dict[str, Any]:
        """Execute the pipeline flow."""
        logger = get_run_logger()

        # Get or create block
        dq_block = block
        if dq_block is None:
            from truthound_prefect.blocks import DataQualityBlock
            dq_block = DataQualityBlock(
                engine_name=cfg.engine_name,
                auto_schema=cfg.auto_schema,
            )

        results: dict[str, Any] = {"stages": {}}
        current_data = None

        for stage in stages:
            stage_name = stage["name"]
            loader = stage["loader"]
            should_check = stage.get("check", True)
            stage_rules = stage.get("rules", [])

            logger.info(f"Executing stage: {stage_name}")

            # Execute stage
            if callable(loader):
                # Pass current_data if loader accepts it
                import inspect
                sig = inspect.signature(loader)
                if len(sig.parameters) > 0 and current_data is not None:
                    current_data = loader(current_data)
                else:
                    current_data = loader()
            else:
                current_data = loader

            stage_result: dict[str, Any] = {"data_loaded": True}

            # Profile if configured
            if cfg.profile_data:
                profile = dq_block.profile(current_data)
                stage_result["profile"] = profile
                logger.info(
                    f"Stage {stage_name}: "
                    f"{profile.get('row_count', 0):,} rows × "
                    f"{profile.get('column_count', 0)} columns"
                )

            # Check if configured
            if should_check and cfg.check_mode != QualityCheckMode.NONE:
                check_kwargs: dict[str, Any] = {}
                if cfg.auto_schema and not stage_rules:
                    check_kwargs["auto_schema"] = True

                check_result = dq_block.check(
                    data=current_data,
                    rules=list(stage_rules) if stage_rules else None,
                    **check_kwargs,
                )

                stage_result["check"] = check_result
                logger.info(f"Stage {stage_name}: {summarize_check_result(check_result)}")

                # Handle failure
                if not check_result.get("is_success", True) and cfg.fail_on_error:
                    raise DataQualityError(
                        message=f"Stage {stage_name} failed quality check",
                        result=check_result,
                    )

            results["stages"][stage_name] = stage_result

        results["final_data"] = current_data
        return results

    return pipeline_flow_impl


def create_multi_table_quality_flows(
    tables: dict[str, dict[str, Any]],
    group_name: str | None = None,
    config: QualityFlowConfig | None = None,
) -> dict[str, Callable[..., Any]]:
    """Create quality flows for multiple tables.

    Factory function to create flows for multiple tables at once.

    Args:
        tables: Dictionary mapping table names to their configuration:
            - loader: Callable that returns the data
            - rules: Optional list of rules
            - fail_on_error: Optional override
        group_name: Optional group name prefix.
        config: Optional base configuration.

    Returns:
        Dictionary mapping table names to flows.

    Example:
        >>> tables = {
        ...     "users": {
        ...         "loader": lambda: pl.read_parquet("users.parquet"),
        ...         "rules": [{"type": "not_null", "column": "id"}],
        ...     },
        ...     "orders": {
        ...         "loader": lambda: pl.read_parquet("orders.parquet"),
        ...         "rules": [{"type": "not_null", "column": "order_id"}],
        ...     },
        ... }
        >>> flows = create_multi_table_quality_flows(tables, group_name="etl")
    """
    flows: dict[str, Callable[..., Any]] = {}

    for table_name, table_config in tables.items():
        flow_name = f"{group_name}_{table_name}" if group_name else table_name

        # Build table-specific config
        table_rules = table_config.get("rules", [])
        base_cfg = config or QualityFlowConfig()

        table_flow_config = QualityFlowConfig(
            enabled=base_cfg.enabled,
            name=flow_name,
            timeout_seconds=base_cfg.timeout_seconds,
            retries=base_cfg.retries,
            retry_delay_seconds=base_cfg.retry_delay_seconds,
            tags=base_cfg.tags,
            check_mode=base_cfg.check_mode,
            rules=tuple(table_rules),
            fail_on_error=table_config.get("fail_on_error", base_cfg.fail_on_error),
            warning_threshold=table_config.get("warning_threshold", base_cfg.warning_threshold),
            auto_schema=table_config.get("auto_schema", base_cfg.auto_schema),
            store_results=base_cfg.store_results,
            engine_name=base_cfg.engine_name,
        )

        flows[table_name] = create_quality_flow(
            name=flow_name,
            loader=table_config["loader"],
            config=table_flow_config,
            description=f"Quality flow for {table_name}",
        )

    return flows


__all__ = [
    "create_quality_flow",
    "create_validation_flow",
    "create_pipeline_flow",
    "create_multi_table_quality_flows",
]
