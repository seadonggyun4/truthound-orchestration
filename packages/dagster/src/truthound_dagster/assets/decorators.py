"""Asset Decorators for Data Quality Integration.

This module provides decorators for creating quality-aware Dagster assets.
The decorators wrap asset functions to automatically run quality checks.

Example:
    >>> from truthound_dagster.assets import quality_checked_asset
    >>>
    >>> @quality_checked_asset(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ... )
    ... def users(context):
    ...     return load_users()
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Sequence

from dagster import AssetExecutionContext, Output, asset

from truthound_dagster.assets.config import (
    ProfileAssetConfig,
    QualityAssetConfig,
    QualityCheckMode,
)

if TYPE_CHECKING:
    from common.base import CheckResult, ProfileResult

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


def _serialize_profile_result(result: ProfileResult) -> dict[str, Any]:
    """Serialize ProfileResult for metadata."""
    return {
        "row_count": result.row_count,
        "column_count": result.column_count,
        "execution_time_ms": result.execution_time_ms,
    }


def quality_checked_asset(
    rules: Sequence[dict[str, Any]] | None = None,
    *,
    check_mode: QualityCheckMode = QualityCheckMode.AFTER,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    auto_schema: bool = False,
    store_result: bool = True,
    result_metadata_key: str = "quality_result",
    timeout_seconds: float = 300.0,
    name: str | None = None,
    group_name: str | None = None,
    key_prefix: str | list[str] | None = None,
    description: str | None = None,
    **asset_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Output]]:
    """Decorator for creating quality-checked assets.

    This decorator wraps an asset function to automatically run
    data quality checks on the asset's output. The checks are
    configured via rules or auto-schema generation.

    Parameters
    ----------
    rules : Sequence[dict[str, Any]] | None
        Validation rules to apply to the asset output.

    check_mode : QualityCheckMode
        When to run quality checks:
        - BEFORE: Not applicable (asset hasn't produced data yet)
        - AFTER: Check after asset function returns (default)
        - NONE: Skip quality checks

    fail_on_error : bool
        Whether to fail the asset on quality check failure.

    warning_threshold : float | None
        Failure rate threshold for warning instead of failure.

    auto_schema : bool
        Auto-generate schema from data (Truthound).

    store_result : bool
        Store quality result in asset metadata.

    result_metadata_key : str
        Key for storing result in metadata.

    timeout_seconds : float
        Quality check timeout.

    name : str | None
        Asset name. Defaults to function name.

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
    Callable
        Decorated asset function.

    Examples
    --------
    Basic usage:

    >>> @quality_checked_asset(
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "unique"},
    ...     ],
    ... )
    ... def users(context):
    ...     return load_users()

    With auto-schema:

    >>> @quality_checked_asset(auto_schema=True)
    ... def transactions(context):
    ...     return load_transactions()

    With warning threshold (don't fail on minor issues):

    >>> @quality_checked_asset(
    ...     rules=[...],
    ...     warning_threshold=0.05,  # 5% failure rate acceptable
    ... )
    ... def events(context):
    ...     return load_events()
    """
    config = QualityAssetConfig(
        rules=tuple(rules or []),
        check_mode=check_mode,
        fail_on_error=fail_on_error,
        warning_threshold=warning_threshold,
        auto_schema=auto_schema,
        store_result=store_result,
        result_metadata_key=result_metadata_key,
        timeout_seconds=timeout_seconds,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Output]:
        # Build asset kwargs
        final_asset_kwargs = {
            "name": name or fn.__name__,
            "description": description or fn.__doc__ or f"Quality-checked asset: {fn.__name__}",
            "required_resource_keys": {"data_quality"},
            **asset_kwargs,
        }
        if group_name:
            final_asset_kwargs["group_name"] = group_name
        if key_prefix:
            final_asset_kwargs["key_prefix"] = key_prefix

        @asset(**final_asset_kwargs)
        @wraps(fn)
        def wrapper(context: AssetExecutionContext, **kwargs: Any) -> Output:
            """Wrapped asset function with quality checks."""
            # Execute original asset function
            data = fn(context, **kwargs)

            # Skip checks if disabled
            if config.check_mode == QualityCheckMode.NONE:
                return Output(data)

            # Get data quality resource
            dq_resource: DataQualityResource = context.resources.data_quality

            # Run quality check
            context.log.info(f"Running quality check on {fn.__name__}")

            from truthound_dagster.utils.exceptions import DataQualityError

            try:
                result = dq_resource.check(
                    data=data,
                    rules=list(config.rules),
                    auto_schema=config.auto_schema,
                    fail_on_error=False,  # Handle failure ourselves
                    timeout=config.timeout_seconds,
                )
            except Exception as e:
                context.log.error(f"Quality check failed with error: {e}")
                if config.fail_on_error:
                    raise
                # Return data without quality metadata
                return Output(data)

            # Log results
            if result.is_success:
                context.log.info(
                    f"Quality check PASSED: {result.passed_count} rules "
                    f"in {result.execution_time_ms:.2f}ms"
                )
            else:
                context.log.warning(
                    f"Quality check FAILED: {result.failed_count} failures "
                    f"({result.failure_rate:.2%})"
                )

            # Build metadata
            metadata: dict[str, Any] = {}
            if config.store_result:
                metadata[config.result_metadata_key] = _serialize_check_result(result)

            # Handle failure
            if not result.is_success:
                # Check warning threshold
                is_warning = (
                    config.warning_threshold is not None
                    and result.failure_rate <= config.warning_threshold
                )

                if not is_warning and config.fail_on_error:
                    raise DataQualityError(
                        message=f"Quality check failed for {fn.__name__}",
                        result=result,
                    )

            return Output(data, metadata=metadata)

        return wrapper

    return decorator


def profiled_asset(
    *,
    include_histograms: bool = True,
    include_samples: bool = True,
    sample_size: int = 10,
    store_result: bool = True,
    result_metadata_key: str = "profile_result",
    timeout_seconds: float = 300.0,
    name: str | None = None,
    group_name: str | None = None,
    key_prefix: str | list[str] | None = None,
    description: str | None = None,
    **asset_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Output]]:
    """Decorator for creating profiled assets.

    This decorator wraps an asset function to automatically run
    data profiling on the asset's output. The profile is stored
    in the asset metadata.

    Parameters
    ----------
    include_histograms : bool
        Include histogram data in profile.

    include_samples : bool
        Include sample values.

    sample_size : int
        Number of sample values.

    store_result : bool
        Store profile in metadata.

    result_metadata_key : str
        Key for storing profile.

    timeout_seconds : float
        Profiling timeout.

    name : str | None
        Asset name.

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
    Callable
        Decorated asset function.

    Examples
    --------
    >>> @profiled_asset()
    ... def users(context):
    ...     return load_users()
    """
    config = ProfileAssetConfig(
        include_histograms=include_histograms,
        include_samples=include_samples,
        sample_size=sample_size,
        store_result=store_result,
        result_metadata_key=result_metadata_key,
        timeout_seconds=timeout_seconds,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Output]:
        final_asset_kwargs = {
            "name": name or fn.__name__,
            "description": description or fn.__doc__ or f"Profiled asset: {fn.__name__}",
            "required_resource_keys": {"data_quality"},
            **asset_kwargs,
        }
        if group_name:
            final_asset_kwargs["group_name"] = group_name
        if key_prefix:
            final_asset_kwargs["key_prefix"] = key_prefix

        @asset(**final_asset_kwargs)
        @wraps(fn)
        def wrapper(context: AssetExecutionContext, **kwargs: Any) -> Output:
            """Wrapped asset function with profiling."""
            # Execute original asset function
            data = fn(context, **kwargs)

            # Get data quality resource
            dq_resource: DataQualityResource = context.resources.data_quality

            # Run profiling
            context.log.info(f"Running profile on {fn.__name__}")

            try:
                result = dq_resource.profile(
                    data=data,
                    timeout=config.timeout_seconds,
                    include_histograms=config.include_histograms,
                    include_samples=config.include_samples,
                    sample_size=config.sample_size,
                )
            except Exception as e:
                context.log.warning(f"Profiling failed: {e}")
                return Output(data)

            context.log.info(
                f"Profiling complete: {result.row_count} rows, "
                f"{result.column_count} columns"
            )

            # Build metadata
            metadata: dict[str, Any] = {}
            if config.store_result:
                metadata[config.result_metadata_key] = _serialize_profile_result(result)

            return Output(data, metadata=metadata)

        return wrapper

    return decorator
