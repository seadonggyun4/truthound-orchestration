"""Helper Functions for Dagster Integration.

This module provides utility functions for common operations
in data quality workflows with Dagster.

Example:
    >>> from truthound_dagster.utils.helpers import format_duration, format_percentage
    >>>
    >>> duration = format_duration(1234.5)  # "1.23s"
    >>> percentage = format_percentage(0.95)  # "95.00%"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from common.base import CheckResult, ProfileResult


def format_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable string.

    Args:
        ms: Duration in milliseconds.

    Returns:
        str: Formatted duration string.

    Example:
        >>> format_duration(1234.5)
        '1.23s'
        >>> format_duration(65000)
        '1m 5.00s'
        >>> format_duration(500)
        '500.00ms'
    """
    if ms < 1000:
        return f"{ms:.2f}ms"
    elif ms < 60000:
        return f"{ms / 1000:.2f}s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.2f}s"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a ratio as percentage.

    Args:
        value: Value between 0 and 1.
        decimals: Number of decimal places.

    Returns:
        str: Formatted percentage string.

    Example:
        >>> format_percentage(0.9523)
        '95.23%'
        >>> format_percentage(0.5, decimals=0)
        '50%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_count(count: int, total: int | None = None) -> str:
    """Format count with optional total.

    Args:
        count: The count value.
        total: Optional total for ratio display.

    Returns:
        str: Formatted count string.

    Example:
        >>> format_count(5)
        '5'
        >>> format_count(5, 10)
        '5/10 (50.00%)'
    """
    if total is None:
        return str(count)
    if total == 0:
        return f"{count}/{total}"
    percentage = format_percentage(count / total)
    return f"{count}/{total} ({percentage})"


def get_current_timestamp() -> datetime:
    """Get current UTC timestamp.

    Returns:
        datetime: Current UTC datetime.
    """
    return datetime.now(timezone.utc)


def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO format string.

    Args:
        dt: Datetime to format.

    Returns:
        str: ISO format string.
    """
    return dt.isoformat()


def parse_timestamp(s: str) -> datetime:
    """Parse ISO format string to datetime.

    Args:
        s: ISO format string.

    Returns:
        datetime: Parsed datetime.
    """
    return datetime.fromisoformat(s)


def summarize_check_result(result: CheckResult) -> dict[str, Any]:
    """Create a summary of check result.

    Args:
        result: Check result to summarize.

    Returns:
        dict[str, Any]: Summary dictionary.

    Example:
        >>> summary = summarize_check_result(result)
        >>> print(summary["status"])
        'PASSED'
    """
    return {
        "status": result.status.value,
        "is_success": result.is_success,
        "passed_count": result.passed_count,
        "failed_count": result.failed_count,
        "warning_count": result.warning_count,
        "failure_rate": result.failure_rate,
        "failure_rate_formatted": format_percentage(result.failure_rate),
        "execution_time_ms": result.execution_time_ms,
        "execution_time_formatted": format_duration(result.execution_time_ms),
        "failure_summary": [
            {
                "rule": f.rule_name,
                "column": f.column,
                "message": f.message,
            }
            for f in result.failures[:5]  # Top 5 failures
        ],
    }


def summarize_profile_result(result: ProfileResult) -> dict[str, Any]:
    """Create a summary of profile result.

    Args:
        result: Profile result to summarize.

    Returns:
        dict[str, Any]: Summary dictionary.
    """
    columns_with_nulls = [col for col in result.columns if col.null_count > 0]

    return {
        "row_count": result.row_count,
        "column_count": result.column_count,
        "execution_time_ms": result.execution_time_ms,
        "execution_time_formatted": format_duration(result.execution_time_ms),
        "columns_with_nulls": len(columns_with_nulls),
        "null_summary": [
            {
                "column": col.column_name,
                "null_count": col.null_count,
                "null_percentage": format_percentage(col.null_percentage / 100),
            }
            for col in columns_with_nulls[:5]  # Top 5
        ],
    }


def merge_metadata(*dicts: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge multiple metadata dictionaries.

    Later dictionaries override earlier ones for duplicate keys.

    Args:
        *dicts: Dictionaries to merge.

    Returns:
        dict[str, Any]: Merged dictionary.

    Example:
        >>> merge_metadata({"a": 1}, {"b": 2}, {"a": 3})
        {'a': 3, 'b': 2}
    """
    result: dict[str, Any] = {}
    for d in dicts:
        if d is not None:
            result.update(d)
    return result


def filter_metadata(
    metadata: Mapping[str, Any],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Filter metadata dictionary.

    Args:
        metadata: Source metadata.
        include: Keys to include (if specified, only these keys).
        exclude: Keys to exclude.

    Returns:
        dict[str, Any]: Filtered metadata.

    Example:
        >>> filter_metadata({"a": 1, "b": 2}, include=["a"])
        {'a': 1}
        >>> filter_metadata({"a": 1, "b": 2}, exclude=["a"])
        {'b': 2}
    """
    result = dict(metadata)

    if include is not None:
        result = {k: v for k, v in result.items() if k in include}

    if exclude is not None:
        result = {k: v for k, v in result.items() if k not in exclude}

    return result


def safe_get(
    mapping: Mapping[str, Any],
    key: str,
    default: Any = None,
    expected_type: type | None = None,
) -> Any:
    """Safely get value from mapping with type checking.

    Args:
        mapping: Source mapping.
        key: Key to get.
        default: Default value if key not found.
        expected_type: Expected type (returns default if type mismatch).

    Returns:
        Value or default.

    Example:
        >>> safe_get({"a": 1}, "a", expected_type=int)
        1
        >>> safe_get({"a": "1"}, "a", default=0, expected_type=int)
        0
    """
    value = mapping.get(key, default)

    if expected_type is not None and value is not default:
        if not isinstance(value, expected_type):
            return default

    return value


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length.

    Args:
        s: String to truncate.
        max_length: Maximum length.
        suffix: Suffix to append if truncated.

    Returns:
        str: Truncated string.

    Example:
        >>> truncate_string("Hello World", max_length=8)
        'Hello...'
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def build_asset_key(*parts: str) -> tuple[str, ...]:
    """Build asset key from parts.

    Args:
        *parts: Key parts.

    Returns:
        tuple[str, ...]: Asset key tuple.

    Example:
        >>> build_asset_key("schema", "table", "quality_check")
        ('schema', 'table', 'quality_check')
    """
    return tuple(p for p in parts if p)


def extract_rules_from_config(
    config: Mapping[str, Any],
    key: str = "rules",
) -> list[dict[str, Any]]:
    """Extract rules from configuration.

    Args:
        config: Configuration dictionary.
        key: Key for rules in config.

    Returns:
        list[dict[str, Any]]: List of rules.
    """
    rules = config.get(key, [])
    if isinstance(rules, (list, tuple)):
        return [dict(r) for r in rules if isinstance(r, Mapping)]
    return []


def validate_rule_format(rule: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Validate basic rule format.

    Args:
        rule: Rule to validate.

    Returns:
        tuple[bool, str | None]: (is_valid, error_message)

    Example:
        >>> validate_rule_format({"type": "not_null", "column": "id"})
        (True, None)
        >>> validate_rule_format({"column": "id"})
        (False, "Rule missing required field: type")
    """
    if not isinstance(rule, Mapping):
        return False, "Rule must be a mapping"

    if "type" not in rule:
        return False, "Rule missing required field: type"

    rule_type = rule.get("type")
    if not isinstance(rule_type, str):
        return False, "Rule type must be a string"

    # Column-based rules require column field
    column_rules = {
        "not_null",
        "unique",
        "in_set",
        "in_range",
        "regex",
        "dtype",
        "min_length",
        "max_length",
        "greater_than",
        "less_than",
    }

    if rule_type in column_rules and "column" not in rule:
        return False, f"Rule type '{rule_type}' requires 'column' field"

    return True, None


def validate_rules(rules: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Validate a list of rules.

    Args:
        rules: Rules to validate.

    Returns:
        tuple[bool, list[str]]: (all_valid, list of error messages)
    """
    errors: list[str] = []

    for i, rule in enumerate(rules):
        is_valid, error = validate_rule_format(rule)
        if not is_valid:
            errors.append(f"Rule {i}: {error}")

    return len(errors) == 0, errors


def create_quality_metadata(
    result: CheckResult,
    additional: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create standard quality metadata for Dagster.

    Args:
        result: Check result.
        additional: Additional metadata.

    Returns:
        dict[str, Any]: Quality metadata.
    """
    metadata = {
        "quality_status": result.status.value,
        "quality_passed": result.is_success,
        "quality_passed_count": result.passed_count,
        "quality_failed_count": result.failed_count,
        "quality_failure_rate": result.failure_rate,
        "quality_execution_time_ms": result.execution_time_ms,
        "quality_timestamp": result.timestamp.isoformat(),
    }

    if additional:
        metadata.update(additional)

    return metadata
