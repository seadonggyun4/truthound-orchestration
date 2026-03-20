"""First-class Dagster asset checks for Truthound data quality."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from dagster import AssetCheckResult, asset_check


def create_asset_check(
    *,
    asset: Any,
    name: str,
    rules: Sequence[dict[str, Any]] | None = None,
    auto_schema: bool = False,
    fail_on_error: bool = True,
    timeout_seconds: float = 300.0,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Any]:
    """Create a first-class Dagster asset check backed by Truthound."""

    def decorator(fn: Callable[..., Any]) -> Any:
        @asset_check(
            asset=asset,
            name=name,
            description=description or f"Truthound asset check for {name}",
            required_resource_keys={"data_quality"},
            blocking=fail_on_error,
        )
        def _asset_check(context, **kwargs: Any) -> AssetCheckResult:
            data = fn(context, **kwargs)
            result = context.resources.data_quality.check(
                data=data,
                rules=list(rules) if rules is not None else None,
                auto_schema=auto_schema,
                fail_on_error=fail_on_error,
                timeout=timeout_seconds,
                dagster_context=context,
                check_name=name,
            )
            return AssetCheckResult(
                passed=result.is_success,
                asset_key=context.asset_key,
                check_name=name,
                metadata={
                    "status": result.status.name,
                    "passed_count": result.passed_count,
                    "failed_count": result.failed_count,
                    "failure_rate": result.failure_rate,
                    "execution_time_ms": result.execution_time_ms,
                },
                description=f"Truthound quality check completed with status {result.status.name}",
            )

        return _asset_check

    return decorator


def quality_asset_check(
    *,
    asset: Any,
    rules: Sequence[dict[str, Any]] | None = None,
    auto_schema: bool = False,
    fail_on_error: bool = True,
    timeout_seconds: float = 300.0,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Any]:
    """Decorator helper mirroring Dagster's `asset_check` ergonomics."""

    def decorator(fn: Callable[..., Any]) -> Any:
        return create_asset_check(
            asset=asset,
            name=name or f"{fn.__name__}_quality",
            rules=rules,
            auto_schema=auto_schema,
            fail_on_error=fail_on_error,
            timeout_seconds=timeout_seconds,
            description=description,
        )(fn)

    return decorator


__all__ = ["create_asset_check", "quality_asset_check"]
