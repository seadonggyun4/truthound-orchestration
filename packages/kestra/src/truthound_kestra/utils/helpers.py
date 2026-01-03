"""Helper utilities for Kestra data quality integration.

This module provides utility functions for common operations such as
data loading, result formatting, logging, and Kestra-specific helpers.

Example:
    >>> from truthound_kestra.utils.helpers import (
    ...     load_data,
    ...     format_duration,
    ...     create_kestra_output,
    ... )
    >>>
    >>> df = load_data("s3://bucket/data.parquet")
    >>> print(format_duration(1234.5))  # "1.23s"
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, TypeVar

from truthound_kestra.utils.exceptions import ConfigurationError, ScriptError
from truthound_kestra.utils.types import (
    CheckStatus,
    DataSourceType,
    ExecutionContext,
    OperationType,
    ScriptOutput,
)

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    # Data loading
    "load_data",
    "detect_data_source_type",
    "parse_uri",
    # Formatting
    "format_duration",
    "format_percentage",
    "format_count",
    "format_status_badge",
    # Kestra helpers
    "create_kestra_output",
    "get_kestra_variable",
    "get_execution_context",
    "kestra_outputs",
    # Timing
    "Timer",
    "timed",
    # Logging
    "get_logger",
    "log_operation",
    # Validation
    "validate_rules",
    "merge_rules",
]

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Data Loading
# =============================================================================


def detect_data_source_type(source: str) -> DataSourceType:
    """Detect the type of data source from a string.

    Args:
        source: Source string (path, URI, or reference).

    Returns:
        Detected DataSourceType.

    Example:
        >>> detect_data_source_type("s3://bucket/data.csv")
        DataSourceType.URI
        >>> detect_data_source_type("/path/to/file.csv")
        DataSourceType.FILE
    """
    source = source.strip()

    # Check for URI schemes
    uri_prefixes = (
        "s3://",
        "gs://",
        "gcs://",
        "http://",
        "https://",
        "az://",
        "azure://",
        "abfs://",
    )
    if any(source.startswith(prefix) for prefix in uri_prefixes):
        return DataSourceType.URI

    # Check for Kestra output reference
    if source.startswith("{{") and source.endswith("}}"):
        return DataSourceType.OUTPUT

    # Check for secret reference
    if source.startswith("secret://") or source.startswith("{{secret."):
        return DataSourceType.SECRET

    # Check for file path
    if os.path.exists(source) or source.startswith("/") or source.startswith("./"):
        return DataSourceType.FILE

    # Default to inline data
    return DataSourceType.INLINE


def parse_uri(uri: str) -> dict[str, str]:
    """Parse a URI into its components.

    Args:
        uri: URI string to parse.

    Returns:
        Dictionary with scheme, bucket, key, and path.

    Example:
        >>> parse_uri("s3://my-bucket/path/to/file.csv")
        {'scheme': 's3', 'bucket': 'my-bucket', 'key': 'path/to/file.csv', 'path': '/path/to/file.csv'}
    """
    from urllib.parse import urlparse

    parsed = urlparse(uri)

    return {
        "scheme": parsed.scheme,
        "bucket": parsed.netloc,
        "key": parsed.path.lstrip("/"),
        "path": parsed.path,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def load_data(
    source: str,
    file_type: str | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    """Load data from various sources into a Polars DataFrame.

    Supports local files, S3, GCS, HTTP(S), and Kestra output references.

    Args:
        source: Data source (path, URI, or inline JSON).
        file_type: Optional file type override (csv, parquet, json).
        **kwargs: Additional arguments passed to Polars read functions.

    Returns:
        Polars DataFrame.

    Raises:
        ScriptError: If data loading fails.

    Example:
        >>> df = load_data("data.csv")
        >>> df = load_data("s3://bucket/data.parquet")
        >>> df = load_data('{"col1": [1,2,3]}')
    """
    try:
        import polars as pl
    except ImportError as e:
        raise ScriptError(
            message="Polars is required for data loading",
            script_name="load_data",
        ) from e

    try:
        source_type = detect_data_source_type(source)

        if source_type == DataSourceType.INLINE:
            # Try parsing as JSON
            try:
                data = json.loads(source)
                return pl.DataFrame(data)
            except json.JSONDecodeError:
                raise ScriptError(
                    message=f"Cannot parse inline data: {source[:100]}...",
                    script_name="load_data",
                )

        # Determine file type from extension or override
        if file_type is None:
            parsed = parse_uri(source) if source_type == DataSourceType.URI else {"path": source}
            path = parsed.get("key") or parsed.get("path", "")
            suffix = Path(path).suffix.lower()
            file_type = suffix.lstrip(".")

        # Load based on file type
        if file_type in ("csv", "tsv"):
            return pl.read_csv(source, **kwargs)
        elif file_type in ("parquet", "pq"):
            return pl.read_parquet(source, **kwargs)
        elif file_type == "json":
            return pl.read_json(source, **kwargs)
        elif file_type == "ndjson":
            return pl.read_ndjson(source, **kwargs)
        elif file_type == "ipc":
            return pl.read_ipc(source, **kwargs)
        else:
            # Try parquet as default
            return pl.read_parquet(source, **kwargs)

    except Exception as e:
        raise ScriptError(
            message=f"Failed to load data from {source}: {e}",
            script_name="load_data",
        ) from e


# =============================================================================
# Formatting
# =============================================================================


def format_duration(milliseconds: float) -> str:
    """Format duration in human-readable form.

    Args:
        milliseconds: Duration in milliseconds.

    Returns:
        Formatted duration string.

    Example:
        >>> format_duration(100)
        '100.00ms'
        >>> format_duration(1500)
        '1.50s'
        >>> format_duration(90000)
        '1m 30s'
    """
    if milliseconds < 1000:
        return f"{milliseconds:.2f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds / 1000:.2f}s"
    else:
        minutes = int(milliseconds // 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.0f}s"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a ratio as a percentage string.

    Args:
        value: Value between 0.0 and 1.0.
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string.

    Example:
        >>> format_percentage(0.956)
        '95.6%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_count(count: int) -> str:
    """Format a count with thousand separators.

    Args:
        count: Integer count.

    Returns:
        Formatted count string.

    Example:
        >>> format_count(1234567)
        '1,234,567'
    """
    return f"{count:,}"


def format_status_badge(status: CheckStatus | str) -> str:
    """Format status as a badge-like string.

    Args:
        status: Check status or status string.

    Returns:
        Badge string with emoji.

    Example:
        >>> format_status_badge(CheckStatus.PASSED)
        '✅ PASSED'
    """
    if isinstance(status, str):
        try:
            status = CheckStatus(status.lower())
        except ValueError:
            return f"❓ {status.upper()}"

    badges = {
        CheckStatus.PASSED: "✅ PASSED",
        CheckStatus.FAILED: "❌ FAILED",
        CheckStatus.WARNING: "⚠️ WARNING",
        CheckStatus.SKIPPED: "⏭️ SKIPPED",
        CheckStatus.ERROR: "💥 ERROR",
    }
    return badges.get(status, f"❓ {status.value.upper()}")


# =============================================================================
# Kestra Helpers
# =============================================================================


def get_kestra_variable(name: str, default: Any = None) -> Any:
    """Get a Kestra flow variable from environment.

    Args:
        name: Variable name.
        default: Default value if not found.

    Returns:
        Variable value or default.

    Example:
        >>> threshold = get_kestra_variable("MAX_FAILURES", 10)
    """
    # Kestra sets flow variables as environment variables
    env_name = f"KESTRA_{name.upper()}"
    value = os.environ.get(env_name)

    if value is None:
        return default

    # Try to parse as JSON for complex types
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def get_execution_context() -> ExecutionContext:
    """Get the current Kestra execution context.

    Returns:
        ExecutionContext populated from environment.

    Example:
        >>> context = get_execution_context()
        >>> print(f"Running in flow: {context.flow_id}")
    """
    return ExecutionContext.from_kestra_env()


def create_kestra_output(
    result: ScriptOutput | dict[str, Any],
    include_summary: bool = True,
) -> dict[str, Any]:
    """Create a Kestra-compatible output dictionary.

    This function creates an output dictionary suitable for use with
    Kestra's output mechanism.

    Args:
        result: Script output or result dictionary.
        include_summary: Whether to include a human-readable summary.

    Returns:
        Kestra-compatible output dictionary.

    Example:
        >>> output = create_kestra_output(script_output)
        >>> Kestra.outputs(output)
    """
    if isinstance(result, ScriptOutput):
        data = result.to_dict()
    else:
        data = dict(result)

    output = {
        "result": data,
        "status": data.get("status", "unknown"),
        "is_success": data.get("is_success", False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if include_summary:
        passed = data.get("passed_count", 0)
        failed = data.get("failed_count", 0)
        total = passed + failed
        pass_rate = passed / total if total > 0 else 1.0

        output["summary"] = {
            "passed_count": passed,
            "failed_count": failed,
            "total_count": total,
            "pass_rate": pass_rate,
            "pass_rate_formatted": format_percentage(pass_rate),
            "execution_time": format_duration(data.get("execution_time_ms", 0)),
        }

    return output


def kestra_outputs(outputs: dict[str, Any]) -> None:
    """Send outputs to Kestra.

    This function attempts to use the Kestra Python SDK to send outputs.
    If the SDK is not available, it falls back to printing JSON.

    Args:
        outputs: Dictionary of outputs to send.

    Example:
        >>> kestra_outputs({"check_result": result_dict})
    """
    try:
        from kestra import Kestra

        Kestra.outputs(outputs)
    except ImportError:
        # Fallback: print as JSON (Kestra can parse this)
        print(f"::outputs:: {json.dumps(outputs, default=str)}")


# =============================================================================
# Timing
# =============================================================================


@dataclass
class Timer:
    """Context manager for timing operations.

    Attributes:
        name: Name of the operation being timed.
        start_time: Start timestamp.
        end_time: End timestamp.
        elapsed_ms: Elapsed time in milliseconds.

    Example:
        >>> with Timer("data_load") as t:
        ...     data = load_data("file.csv")
        >>> print(f"Loaded in {t.elapsed_ms:.2f}ms")
    """

    name: str = "operation"
    start_time: float = field(default=0.0, init=False)
    end_time: float = field(default=0.0, init=False)
    elapsed_ms: float = field(default=0.0, init=False)

    def __enter__(self) -> Timer:
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer and calculate elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

    @property
    def elapsed_formatted(self) -> str:
        """Get formatted elapsed time."""
        return format_duration(self.elapsed_ms)


def timed(name: str | None = None) -> Callable[[F], F]:
    """Decorator to time function execution.

    Args:
        name: Optional name for the operation (defaults to function name).

    Returns:
        Decorated function that logs timing information.

    Example:
        >>> @timed("data_validation")
        ... def validate(data):
        ...     return engine.check(data)
    """

    def decorator(func: F) -> F:
        operation_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(operation_name) as timer:
                result = func(*args, **kwargs)

            logger = get_logger(func.__module__)
            logger.debug(
                f"{operation_name} completed in {timer.elapsed_formatted}",
                extra={"elapsed_ms": timer.elapsed_ms},
            )

            return result

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Logging
# =============================================================================


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger configured for Kestra scripts.

    Args:
        name: Logger name (defaults to 'truthound_kestra').

    Returns:
        Configured logger.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting validation")
    """
    logger = logging.getLogger(name or "truthound_kestra")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


@contextmanager
def log_operation(
    operation: str,
    logger: logging.Logger | None = None,
    **context: Any,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for logging operation start/end with timing.

    Args:
        operation: Name of the operation.
        logger: Optional logger (uses default if not provided).
        **context: Additional context to log.

    Yields:
        Dictionary for collecting operation results.

    Example:
        >>> with log_operation("check", table="users") as op:
        ...     result = engine.check(data)
        ...     op["result"] = result
    """
    log = logger or get_logger()
    result_dict: dict[str, Any] = {}

    log.info(f"Starting {operation}", extra=context)
    start = time.perf_counter()

    try:
        yield result_dict
        elapsed = (time.perf_counter() - start) * 1000
        log.info(
            f"Completed {operation} in {format_duration(elapsed)}",
            extra={**context, "elapsed_ms": elapsed, "success": True},
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        log.error(
            f"Failed {operation} after {format_duration(elapsed)}: {e}",
            extra={**context, "elapsed_ms": elapsed, "success": False, "error": str(e)},
        )
        raise


# =============================================================================
# Validation
# =============================================================================


def validate_rules(rules: list[dict[str, Any]]) -> list[str]:
    """Validate rule definitions and return any errors.

    Args:
        rules: List of rule dictionaries.

    Returns:
        List of validation error messages (empty if valid).

    Example:
        >>> errors = validate_rules([{"type": "not_null"}])  # Missing 'column'
        >>> if errors:
        ...     raise ConfigurationError(f"Invalid rules: {errors}")
    """
    errors = []

    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"Rule {i}: must be a dictionary, got {type(rule).__name__}")
            continue

        if "type" not in rule:
            errors.append(f"Rule {i}: missing required field 'type'")
            continue

        rule_type = rule["type"]

        # Rules that require 'column'
        column_required = {
            "not_null",
            "unique",
            "in_set",
            "in_range",
            "regex",
            "dtype",
            "min_length",
            "max_length",
        }

        if rule_type in column_required and "column" not in rule:
            errors.append(f"Rule {i} ({rule_type}): missing required field 'column'")

        # Validate specific rule types
        if rule_type == "in_set" and "values" not in rule:
            errors.append(f"Rule {i} (in_set): missing required field 'values'")

        if rule_type == "regex" and "pattern" not in rule:
            errors.append(f"Rule {i} (regex): missing required field 'pattern'")

        if rule_type == "dtype" and "dtype" not in rule:
            errors.append(f"Rule {i} (dtype): missing required field 'dtype'")

    return errors


def merge_rules(
    *rule_lists: list[dict[str, Any]] | None,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """Merge multiple rule lists into one.

    Args:
        *rule_lists: Variable number of rule lists to merge.
        deduplicate: Whether to remove duplicate rules.

    Returns:
        Merged list of rules.

    Example:
        >>> base_rules = [{"type": "not_null", "column": "id"}]
        >>> extra_rules = [{"type": "unique", "column": "id"}]
        >>> all_rules = merge_rules(base_rules, extra_rules)
    """
    merged = []
    seen = set()

    for rule_list in rule_lists:
        if rule_list is None:
            continue

        for rule in rule_list:
            if deduplicate:
                # Create a hashable key for deduplication
                key = (rule.get("type"), rule.get("column"), json.dumps(rule, sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)

            merged.append(rule)

    return merged
