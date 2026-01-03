"""Helper functions for Mage Data Quality blocks.

This module provides utility functions for formatting results,
creating metadata, and common data operations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from common.base import CheckResult, CheckStatus

from truthound_mage.utils.types import BlockMetadata, DataQualityOutput


def format_check_result(
    result: CheckResult,
    *,
    include_failures: bool = True,
    max_failures: int = 10,
    include_metadata: bool = True,
) -> str:
    """Format CheckResult as a human-readable string.

    Args:
        result: CheckResult to format.
        include_failures: Include failure details.
        max_failures: Maximum failures to include.
        include_metadata: Include metadata.

    Returns:
        Formatted string representation.

    Example:
        >>> result = engine.check(data, rules)
        >>> print(format_check_result(result))
        Data Quality Check: PASSED
        Passed: 10/10 (100.0%)
    """
    lines: list[str] = []

    # Status line
    status_name = result.status.name if hasattr(result.status, "name") else str(result.status)
    lines.append(f"Data Quality Check: {status_name}")
    lines.append("-" * 40)

    # Counts
    total = result.passed_count + result.failed_count
    if total > 0:
        pass_rate = (result.passed_count / total) * 100
        lines.append(f"Passed: {result.passed_count}/{total} ({pass_rate:.1f}%)")
        lines.append(f"Failed: {result.failed_count}/{total} ({100 - pass_rate:.1f}%)")
    else:
        lines.append("No rules evaluated")

    # Warning and skipped counts
    if hasattr(result, "warning_count") and result.warning_count:
        lines.append(f"Warnings: {result.warning_count}")
    if hasattr(result, "skipped_count") and result.skipped_count:
        lines.append(f"Skipped: {result.skipped_count}")

    # Execution time
    if hasattr(result, "execution_time_ms") and result.execution_time_ms:
        lines.append(f"Execution time: {result.execution_time_ms:.2f}ms")

    # Failures
    if include_failures and hasattr(result, "failures") and result.failures:
        lines.append("")
        lines.append("Failures:")
        for i, failure in enumerate(result.failures[:max_failures]):
            failure_str = _format_single_failure(failure)
            lines.append(f"  {i + 1}. {failure_str}")

        remaining = len(result.failures) - max_failures
        if remaining > 0:
            lines.append(f"  ... and {remaining} more failures")

    # Metadata
    if include_metadata and hasattr(result, "metadata") and result.metadata:
        lines.append("")
        lines.append("Metadata:")
        for key, value in result.metadata.items():
            if not key.startswith("_"):
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def format_violations(
    violations: Sequence[Any],
    *,
    max_violations: int = 10,
    include_details: bool = True,
) -> str:
    """Format SLA violations as a human-readable string.

    Args:
        violations: List of SLA violations.
        max_violations: Maximum violations to include.
        include_details: Include violation details.

    Returns:
        Formatted string representation.

    Example:
        >>> violations = monitor.check(metrics)
        >>> print(format_violations(violations))
        SLA Violations: 2
        1. [CRITICAL] Pass rate below threshold: 85.0% < 95.0%
    """
    if not violations:
        return "No SLA violations detected"

    lines: list[str] = []
    lines.append(f"SLA Violations: {len(violations)}")
    lines.append("-" * 40)

    for i, violation in enumerate(violations[:max_violations]):
        violation_str = _format_single_violation(violation, include_details)
        lines.append(f"{i + 1}. {violation_str}")

    remaining = len(violations) - max_violations
    if remaining > 0:
        lines.append(f"... and {remaining} more violations")

    return "\n".join(lines)


def create_block_metadata(
    block_uuid: str,
    *,
    pipeline_uuid: str | None = None,
    block_type: str = "transformer",
    operation: str | None = None,
    engine_name: str | None = None,
    engine_version: str | None = None,
    tags: Sequence[str] | None = None,
    **extra: Any,
) -> BlockMetadata:
    """Create BlockMetadata with current timestamp.

    Args:
        block_uuid: Unique identifier for the block.
        pipeline_uuid: UUID of the containing pipeline.
        block_type: Type of block.
        operation: Operation being performed.
        engine_name: Name of the data quality engine.
        engine_version: Version of the engine.
        tags: Custom tags.
        **extra: Additional metadata fields.

    Returns:
        BlockMetadata with started_at set to current time.

    Example:
        >>> metadata = create_block_metadata(
        ...     "check_1",
        ...     pipeline_uuid="pipeline_1",
        ...     operation="check",
        ...     engine_name="truthound",
        ... )
    """
    return BlockMetadata(
        block_uuid=block_uuid,
        pipeline_uuid=pipeline_uuid,
        block_type=block_type,  # type: ignore[arg-type]
        operation=operation,  # type: ignore[arg-type]
        started_at=datetime.now(timezone.utc),
        engine_name=engine_name,
        engine_version=engine_version,
        tags=frozenset(tags) if tags else frozenset(),
        extra=tuple(extra.items()),
    )


def get_data_size(data: Any) -> tuple[int, int]:
    """Get the size of data as (rows, columns).

    Args:
        data: Data to measure (DataFrame, dict, or list).

    Returns:
        Tuple of (row_count, column_count).

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> get_data_size(df)
        (2, 2)
    """
    # Polars DataFrame
    if hasattr(data, "shape"):
        shape = data.shape
        if isinstance(shape, tuple) and len(shape) >= 2:
            return (shape[0], shape[1])
        return (shape[0], 1) if shape else (0, 0)

    # Dictionary (column-oriented)
    if isinstance(data, dict):
        if not data:
            return (0, 0)
        first_col = next(iter(data.values()))
        row_count = len(first_col) if hasattr(first_col, "__len__") else 0
        return (row_count, len(data))

    # List of dicts (row-oriented)
    if isinstance(data, list):
        if not data:
            return (0, 0)
        if isinstance(data[0], dict):
            return (len(data), len(data[0]))
        return (len(data), 1)

    return (0, 0)


def validate_data_input(
    data: Any,
    *,
    allow_empty: bool = False,
    min_rows: int | None = None,
    required_columns: Sequence[str] | None = None,
) -> tuple[bool, str | None]:
    """Validate data input for data quality operations.

    Args:
        data: Data to validate.
        allow_empty: Allow empty data.
        min_rows: Minimum required rows.
        required_columns: Required column names.

    Returns:
        Tuple of (is_valid, error_message).

    Example:
        >>> valid, error = validate_data_input(df, min_rows=1)
        >>> if not valid:
        ...     raise ValueError(error)
    """
    if data is None:
        return (False, "Data is None")

    rows, cols = get_data_size(data)

    if not allow_empty and rows == 0:
        return (False, "Data is empty (0 rows)")

    if min_rows is not None and rows < min_rows:
        return (False, f"Data has {rows} rows, minimum required is {min_rows}")

    if required_columns:
        # Get column names
        if hasattr(data, "columns"):
            columns = set(data.columns)
        elif isinstance(data, dict):
            columns = set(data.keys())
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            columns = set(data[0].keys())
        else:
            columns = set()

        missing = set(required_columns) - columns
        if missing:
            return (False, f"Missing required columns: {sorted(missing)}")

    return (True, None)


def create_output(
    result: Any,
    metadata: BlockMetadata,
    *,
    data: Any = None,
    **summary: Any,
) -> DataQualityOutput:
    """Create DataQualityOutput from result.

    Args:
        result: The result from data quality operation.
        metadata: Block execution metadata.
        data: Optional output data.
        **summary: Summary key-value pairs.

    Returns:
        DataQualityOutput instance.

    Example:
        >>> output = create_output(
        ...     result,
        ...     metadata,
        ...     passed=10,
        ...     failed=2,
        ... )
    """
    # Determine success based on result
    success = True
    if hasattr(result, "status"):
        status = result.status
        if hasattr(status, "name"):
            success = status.name in ("PASSED", "WARNING")
        else:
            success = str(status).upper() in ("PASSED", "WARNING")
    elif hasattr(result, "failed_count"):
        success = result.failed_count == 0

    # Complete metadata
    completed_metadata = metadata.with_completion()

    return DataQualityOutput(
        success=success,
        result=result,
        metadata=completed_metadata,
        data=data,
        summary=tuple(summary.items()),
    )


def merge_rules(
    *rule_sets: Sequence[dict[str, Any]],
    deduplicate: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Merge multiple rule sets into one.

    Args:
        *rule_sets: Rule sets to merge.
        deduplicate: Remove duplicate rules.

    Returns:
        Merged tuple of rules.

    Example:
        >>> rules1 = [{"type": "not_null", "column": "id"}]
        >>> rules2 = [{"type": "unique", "column": "id"}]
        >>> merged = merge_rules(rules1, rules2)
    """
    all_rules: list[dict[str, Any]] = []

    for rules in rule_sets:
        if rules:
            all_rules.extend(rules)

    if deduplicate:
        # Use JSON string for comparison (order-independent)
        import json

        seen: set[str] = set()
        unique_rules: list[dict[str, Any]] = []

        for rule in all_rules:
            # Sort keys for consistent comparison
            key = json.dumps(rule, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)

        return tuple(unique_rules)

    return tuple(all_rules)


def _format_single_failure(failure: Any) -> str:
    """Format a single validation failure."""
    if isinstance(failure, dict):
        column = failure.get("column", "unknown")
        rule_type = failure.get("rule_type", "unknown")
        message = failure.get("message", "")
        severity = failure.get("severity", "")
        if severity:
            return f"[{severity}] {column}: {rule_type} - {message}"
        return f"{column}: {rule_type} - {message}"

    if hasattr(failure, "column") and hasattr(failure, "message"):
        column = failure.column
        message = failure.message
        severity = getattr(failure, "severity", None)
        if severity:
            severity_name = severity.name if hasattr(severity, "name") else str(severity)
            return f"[{severity_name}] {column}: {message}"
        return f"{column}: {message}"

    return str(failure)


def _format_single_violation(violation: Any, include_details: bool) -> str:
    """Format a single SLA violation."""
    if isinstance(violation, dict):
        alert_level = violation.get("alert_level", "").upper()
        message = violation.get("message", "")
        if include_details:
            threshold = violation.get("threshold")
            actual = violation.get("actual")
            if threshold is not None and actual is not None:
                return f"[{alert_level}] {message} (threshold={threshold}, actual={actual})"
        return f"[{alert_level}] {message}"

    if hasattr(violation, "alert_level") and hasattr(violation, "message"):
        alert_level = violation.alert_level
        if hasattr(alert_level, "name"):
            alert_level = alert_level.name.upper()
        message = violation.message
        if include_details:
            threshold = getattr(violation, "threshold", None)
            actual = getattr(violation, "actual", None)
            if threshold is not None and actual is not None:
                return f"[{alert_level}] {message} (threshold={threshold}, actual={actual})"
        return f"[{alert_level}] {message}"

    return str(violation)
