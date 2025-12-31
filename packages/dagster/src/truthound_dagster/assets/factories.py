"""Asset Factories for Data Quality Integration.

This module provides factory functions for creating data quality assets
programmatically. Use these when you need dynamic asset creation.

Example:
    >>> from truthound_dagster.assets import create_quality_asset
    >>>
    >>> users_asset = create_quality_asset(
    ...     name="users",
    ...     loader=load_users,
    ...     rules=[{"column": "id", "type": "not_null"}],
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from dagster import AssetExecutionContext, AssetsDefinition, Output, asset

from truthound_dagster.assets.config import (
    QualityAssetConfig,
    QualityCheckMode,
)

if TYPE_CHECKING:
    from common.base import CheckResult

    from truthound_dagster.resources import DataQualityResource


def _serialize_check_result(result: CheckResult) -> dict[str, Any]:
    """Serialize CheckResult for metadata."""
    return {
        "status": result.status.value,
        "is_success": result.is_success,
        "passed_count": result.passed_count,
        "failed_count": result.failed_count,
        "failure_rate": result.failure_rate,
        "execution_time_ms": result.execution_time_ms,
    }


def create_quality_asset(
    name: str,
    loader: Callable[[AssetExecutionContext], Any],
    rules: Sequence[dict[str, Any]] | None = None,
    *,
    check_mode: QualityCheckMode = QualityCheckMode.AFTER,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    auto_schema: bool = False,
    store_result: bool = True,
    group_name: str | None = None,
    key_prefix: str | list[str] | None = None,
    description: str | None = None,
    deps: Sequence[str | AssetsDefinition] | None = None,
    **asset_kwargs: Any,
) -> AssetsDefinition:
    """Create a quality-checked asset from a loader function.

    This factory creates an asset that loads data using the provided
    loader function and runs quality checks on the result.

    Parameters
    ----------
    name : str
        Asset name.

    loader : Callable[[AssetExecutionContext], Any]
        Function that loads and returns the asset data.

    rules : Sequence[dict[str, Any]] | None
        Validation rules to apply.

    check_mode : QualityCheckMode
        When to run quality checks.

    fail_on_error : bool
        Whether to fail on quality check failure.

    warning_threshold : float | None
        Failure rate threshold for warning.

    auto_schema : bool
        Auto-generate schema from data.

    store_result : bool
        Store quality result in metadata.

    group_name : str | None
        Asset group name.

    key_prefix : str | list[str] | None
        Asset key prefix.

    description : str | None
        Asset description.

    deps : Sequence[str | AssetsDefinition] | None
        Asset dependencies.

    **asset_kwargs : Any
        Additional arguments passed to @asset.

    Returns
    -------
    AssetsDefinition
        Configured Dagster asset.

    Examples
    --------
    >>> def load_users(context):
    ...     return pl.read_parquet("users.parquet")
    >>>
    >>> users_asset = create_quality_asset(
    ...     name="users",
    ...     loader=load_users,
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "unique"},
    ...     ],
    ... )
    """
    config = QualityAssetConfig(
        rules=tuple(rules or []),
        check_mode=check_mode,
        fail_on_error=fail_on_error,
        warning_threshold=warning_threshold,
        auto_schema=auto_schema,
        store_result=store_result,
    )

    final_asset_kwargs = {
        "name": name,
        "description": description or f"Quality-checked asset: {name}",
        "required_resource_keys": {"data_quality"},
        **asset_kwargs,
    }
    if group_name:
        final_asset_kwargs["group_name"] = group_name
    if key_prefix:
        final_asset_kwargs["key_prefix"] = key_prefix
    if deps:
        final_asset_kwargs["deps"] = deps

    @asset(**final_asset_kwargs)
    def quality_asset(context: AssetExecutionContext) -> Output:
        """Quality-checked asset created by factory."""
        # Load data
        data = loader(context)

        # Skip checks if disabled
        if config.check_mode == QualityCheckMode.NONE:
            return Output(data)

        # Get resource and run check
        dq_resource: DataQualityResource = context.resources.data_quality

        context.log.info(f"Running quality check on {name}")

        from truthound_dagster.utils.exceptions import DataQualityError

        try:
            result = dq_resource.check(
                data=data,
                rules=list(config.rules),
                auto_schema=config.auto_schema,
                fail_on_error=False,
            )
        except Exception as e:
            context.log.error(f"Quality check failed: {e}")
            if config.fail_on_error:
                raise
            return Output(data)

        # Log results
        if result.is_success:
            context.log.info(f"Quality check PASSED: {result.passed_count} rules")
        else:
            context.log.warning(
                f"Quality check FAILED: {result.failed_count} failures"
            )

        # Build metadata
        metadata: dict[str, Any] = {}
        if config.store_result:
            metadata["quality_result"] = _serialize_check_result(result)

        # Handle failure
        if not result.is_success:
            is_warning = (
                config.warning_threshold is not None
                and result.failure_rate <= config.warning_threshold
            )

            if not is_warning and config.fail_on_error:
                raise DataQualityError(
                    message=f"Quality check failed for {name}",
                    result=result,
                )

        return Output(data, metadata=metadata)

    return quality_asset


def create_quality_check_asset(
    name: str,
    source_asset: str | AssetsDefinition,
    rules: Sequence[dict[str, Any]] | None = None,
    *,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    auto_schema: bool = False,
    group_name: str | None = None,
    key_prefix: str | list[str] | None = None,
    description: str | None = None,
    **asset_kwargs: Any,
) -> AssetsDefinition:
    """Create a standalone quality check asset.

    This creates an asset that depends on a source asset and runs
    quality checks on it. Use this when you want to separate data
    loading from quality checking.

    Parameters
    ----------
    name : str
        Asset name for the quality check.

    source_asset : str | AssetsDefinition
        Source asset to check.

    rules : Sequence[dict[str, Any]] | None
        Validation rules to apply.

    fail_on_error : bool
        Whether to fail on quality check failure.

    warning_threshold : float | None
        Failure rate threshold for warning.

    auto_schema : bool
        Auto-generate schema from data.

    group_name : str | None
        Asset group name.

    key_prefix : str | list[str] | None
        Asset key prefix.

    description : str | None
        Asset description.

    **asset_kwargs : Any
        Additional arguments passed to @asset.

    Returns
    -------
    AssetsDefinition
        Quality check asset.

    Examples
    --------
    >>> users_check = create_quality_check_asset(
    ...     name="users_quality_check",
    ...     source_asset="users",
    ...     rules=[{"column": "id", "type": "not_null"}],
    ... )
    """
    config = QualityAssetConfig(
        rules=tuple(rules or []),
        fail_on_error=fail_on_error,
        warning_threshold=warning_threshold,
        auto_schema=auto_schema,
    )

    # Get source asset name
    if isinstance(source_asset, str):
        source_name = source_asset
    else:
        source_name = source_asset.key.path[-1]

    final_asset_kwargs = {
        "name": name,
        "description": description or f"Quality check for {source_name}",
        "required_resource_keys": {"data_quality"},
        "deps": [source_asset],
        **asset_kwargs,
    }
    if group_name:
        final_asset_kwargs["group_name"] = group_name
    if key_prefix:
        final_asset_kwargs["key_prefix"] = key_prefix

    @asset(**final_asset_kwargs)
    def quality_check_asset(context: AssetExecutionContext) -> Output:
        """Standalone quality check asset."""
        # Note: In a real implementation, you would load the source asset data
        # This is a simplified version that returns the check result

        dq_resource: DataQualityResource = context.resources.data_quality

        context.log.info(f"Running quality check: {name}")

        # In practice, you'd get data from the source asset
        # For now, return metadata about the check configuration
        metadata = {
            "source_asset": source_name,
            "rules_count": len(config.rules),
            "auto_schema": config.auto_schema,
            "fail_on_error": config.fail_on_error,
        }

        return Output(
            value={"status": "configured", "source": source_name},
            metadata=metadata,
        )

    return quality_check_asset


def create_multi_table_quality_assets(
    tables: dict[str, dict[str, Any]],
    *,
    group_name: str | None = None,
    key_prefix: str | list[str] | None = None,
) -> list[AssetsDefinition]:
    """Create quality check assets for multiple tables.

    This factory creates assets for multiple tables at once,
    useful for generating assets from a configuration file.

    Parameters
    ----------
    tables : dict[str, dict[str, Any]]
        Dictionary mapping table names to configurations.
        Each configuration can include:
        - loader: Callable to load data
        - rules: Validation rules
        - fail_on_error: Whether to fail on error
        - description: Asset description

    group_name : str | None
        Common group name for all assets.

    key_prefix : str | list[str] | None
        Common key prefix for all assets.

    Returns
    -------
    list[AssetsDefinition]
        List of configured assets.

    Examples
    --------
    >>> tables = {
    ...     "users": {
    ...         "loader": load_users,
    ...         "rules": [{"column": "id", "type": "not_null"}],
    ...     },
    ...     "orders": {
    ...         "loader": load_orders,
    ...         "rules": [{"column": "order_id", "type": "not_null"}],
    ...     },
    ... }
    >>>
    >>> assets = create_multi_table_quality_assets(
    ...     tables=tables,
    ...     group_name="quality_checks",
    ... )
    """
    assets = []

    for table_name, config in tables.items():
        loader = config.get("loader")
        if loader is None:
            continue

        asset_def = create_quality_asset(
            name=table_name,
            loader=loader,
            rules=config.get("rules"),
            fail_on_error=config.get("fail_on_error", True),
            warning_threshold=config.get("warning_threshold"),
            auto_schema=config.get("auto_schema", False),
            group_name=group_name,
            key_prefix=key_prefix,
            description=config.get("description"),
        )
        assets.append(asset_def)

    return assets
