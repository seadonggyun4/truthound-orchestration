"""Helper utilities for truthound-prefect.

This module provides formatting, logging, and other utility functions
used throughout the package.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def format_duration(duration_ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        Human-readable duration string.

    Examples:
        >>> format_duration(500)
        '500.00ms'
        >>> format_duration(5000)
        '5.00s'
        >>> format_duration(120000)
        '2.00min'
    """
    if duration_ms < 1000:
        return f"{duration_ms:.2f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms / 1000:.2f}s"
    elif duration_ms < 3600000:
        return f"{duration_ms / 60000:.2f}min"
    else:
        return f"{duration_ms / 3600000:.2f}h"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a value as a percentage string.

    Args:
        value: Value between 0 and 1.
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string.

    Examples:
        >>> format_percentage(0.9523)
        '95.23%'
        >>> format_percentage(0.5, decimals=0)
        '50%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_count(count: int) -> str:
    """Format a count with thousands separators.

    Args:
        count: The count to format.

    Returns:
        Formatted count string.

    Examples:
        >>> format_count(1234567)
        '1,234,567'
    """
    return f"{count:,}"


def summarize_check_result(result: dict[str, Any]) -> str:
    """Create a one-line summary of a check result.

    Args:
        result: Serialized check result dictionary.

    Returns:
        Summary string.

    Examples:
        >>> summarize_check_result({"is_success": True, "passed_count": 5, "failed_count": 0})
        '✅ PASSED (5 passed, 0 failed)'
    """
    status = "✅ PASSED" if result.get("is_success", True) else "❌ FAILED"
    passed = result.get("passed_count", 0)
    failed = result.get("failed_count", 0)
    return f"{status} ({passed} passed, {failed} failed)"


def summarize_profile_result(result: dict[str, Any]) -> str:
    """Create a one-line summary of a profile result.

    Args:
        result: Serialized profile result dictionary.

    Returns:
        Summary string.
    """
    rows = result.get("row_count", 0)
    cols = result.get("column_count", 0)
    return f"📊 Profile: {format_count(rows)} rows × {cols} columns"


def summarize_learn_result(result: dict[str, Any]) -> str:
    """Create a one-line summary of a learn result.

    Args:
        result: Serialized learn result dictionary.

    Returns:
        Summary string.
    """
    rules_count = len(result.get("rules", []))
    return f"📝 Learned {rules_count} rules"


def create_quality_metadata(
    result: dict[str, Any],
    flow_name: str | None = None,
    task_name: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create metadata dictionary for Prefect artifacts.

    Args:
        result: Serialized result dictionary.
        flow_name: Name of the flow.
        task_name: Name of the task.
        run_id: Prefect run ID.

    Returns:
        Metadata dictionary for Prefect.
    """
    metadata: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
    }

    if flow_name:
        metadata["flow_name"] = flow_name
    if task_name:
        metadata["task_name"] = task_name
    if run_id:
        metadata["run_id"] = run_id

    # Add result-specific metadata
    if "is_success" in result:
        metadata["quality_status"] = "passed" if result["is_success"] else "failed"
    if "passed_count" in result:
        metadata["passed_count"] = result["passed_count"]
        metadata["failed_count"] = result["failed_count"]
    if "row_count" in result:
        metadata["row_count"] = result["row_count"]
    if "rules" in result:
        metadata["rules_count"] = len(result["rules"])
    if "execution_time_ms" in result:
        metadata["execution_time_ms"] = result["execution_time_ms"]

    return metadata


def get_data_info(data: Any) -> dict[str, Any]:
    """Extract basic information about data.

    Args:
        data: The data to inspect (DataFrame, dict, etc.).

    Returns:
        Dictionary with data information.
    """
    info: dict[str, Any] = {
        "type": type(data).__name__,
    }

    # Polars DataFrame
    if hasattr(data, "shape") and hasattr(data, "columns"):
        info["row_count"] = data.shape[0]
        info["column_count"] = data.shape[1]
        info["columns"] = list(data.columns)
    # Pandas DataFrame (duck typed)
    elif hasattr(data, "shape") and hasattr(data, "columns") and hasattr(data, "dtypes"):
        info["row_count"] = data.shape[0]
        info["column_count"] = data.shape[1]
        info["columns"] = list(data.columns)
    # Dict/list
    elif isinstance(data, dict):
        info["key_count"] = len(data)
    elif isinstance(data, (list, tuple)):
        info["length"] = len(data)

    return info


def calculate_timeout(
    base_timeout: float,
    data_size: int | None = None,
    scale_factor: float = 0.001,
    min_timeout: float = 30.0,
    max_timeout: float = 3600.0,
) -> float:
    """Calculate dynamic timeout based on data size.

    Args:
        base_timeout: Base timeout in seconds.
        data_size: Size of data (e.g., row count).
        scale_factor: Seconds to add per data unit.
        min_timeout: Minimum timeout.
        max_timeout: Maximum timeout.

    Returns:
        Calculated timeout in seconds.
    """
    timeout = base_timeout
    if data_size is not None:
        timeout += data_size * scale_factor
    return max(min_timeout, min(timeout, max_timeout))


def parse_rules_from_string(rules_json: str) -> list[dict[str, Any]]:
    """Parse rules from a JSON string.

    Args:
        rules_json: JSON string containing rules.

    Returns:
        List of rule dictionaries.

    Raises:
        ValueError: If the JSON is invalid.
    """
    import json

    if not rules_json or rules_json.strip() == "":
        return []

    try:
        rules = json.loads(rules_json)
        if not isinstance(rules, list):
            rules = [rules]
        return rules
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid rules JSON: {e}") from e


def merge_results(
    results: list[dict[str, Any]],
    strategy: str = "combine",
) -> dict[str, Any]:
    """Merge multiple check results into one.

    Args:
        results: List of serialized check results.
        strategy: Merge strategy ('combine', 'worst', 'best').

    Returns:
        Merged result dictionary.
    """
    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    if strategy == "worst":
        # Return the result with the most failures
        return max(results, key=lambda r: r.get("failed_count", 0))
    elif strategy == "best":
        # Return the result with the most passes
        return max(results, key=lambda r: r.get("passed_count", 0))
    else:  # combine
        # Combine all results
        total_passed = sum(r.get("passed_count", 0) for r in results)
        total_failed = sum(r.get("failed_count", 0) for r in results)
        all_failures = []
        for r in results:
            all_failures.extend(r.get("failures", []))

        total = total_passed + total_failed
        is_success = total_failed == 0

        return {
            "status": "passed" if is_success else "failed",
            "is_success": is_success,
            "passed_count": total_passed,
            "failed_count": total_failed,
            "failure_rate": total_failed / total if total > 0 else 0.0,
            "failures": all_failures,
            "merged_from": len(results),
            "timestamp": datetime.now().isoformat(),
        }


def create_run_context(
    flow_name: str | None = None,
    task_name: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a context dictionary for the current run.

    Args:
        flow_name: Name of the current flow.
        task_name: Name of the current task.
        **kwargs: Additional context values.

    Returns:
        Context dictionary.
    """
    context: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
    }

    if flow_name:
        context["flow_name"] = flow_name
    if task_name:
        context["task_name"] = task_name

    context.update(kwargs)
    return context


__all__ = [
    "format_duration",
    "format_percentage",
    "format_count",
    "summarize_check_result",
    "summarize_profile_result",
    "summarize_learn_result",
    "create_quality_metadata",
    "get_data_info",
    "calculate_timeout",
    "parse_rules_from_string",
    "merge_results",
    "create_run_context",
]
