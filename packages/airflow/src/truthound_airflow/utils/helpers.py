"""Common Helper Utilities for Data Quality Operations.

This module provides common utility functions used across
the Airflow data quality package.

Example:
    >>> from truthound_airflow.utils import (
    ...     format_duration,
    ...     format_percentage,
    ...     merge_dicts,
    ... )
    >>>
    >>> print(format_duration(1500.5))  # "1.50s"
    >>> print(format_percentage(0.95))  # "95.00%"
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# =============================================================================
# Dictionary Utilities
# =============================================================================


def safe_get(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Safely get nested dictionary value.

    Args:
        data: Dictionary to search.
        *keys: Nested keys to traverse.
        default: Default value if not found.

    Returns:
        Any: Found value or default.

    Example:
        >>> data = {"a": {"b": {"c": 1}}}
        >>> safe_get(data, "a", "b", "c")
        1
        >>> safe_get(data, "a", "x", "y", default=0)
        0
    """
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
            if result is None:
                return default
        else:
            return default
    return result


def merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
    deep: bool = True,
) -> dict[str, Any]:
    """Merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Dictionary to merge on top.
        deep: Whether to recursively merge nested dicts.

    Returns:
        dict[str, Any]: Merged dictionary.

    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> override = {"b": {"d": 3}}
        >>> merge_dicts(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = dict(base)

    for key, value in override.items():
        if (
            deep
            and key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value

    return result


def flatten_dict(
    data: dict[str, Any],
    separator: str = ".",
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested dictionary.

    Args:
        data: Dictionary to flatten.
        separator: Key separator.
        prefix: Key prefix.

    Returns:
        dict[str, Any]: Flattened dictionary.

    Example:
        >>> data = {"a": {"b": 1, "c": 2}}
        >>> flatten_dict(data)
        {'a.b': 1, 'a.c': 2}
    """
    result = {}

    for key, value in data.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_dict(value, separator, full_key))
        else:
            result[full_key] = value

    return result


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_duration(
    milliseconds: float,
    precision: int = 2,
) -> str:
    """Format duration in human-readable form.

    Args:
        milliseconds: Duration in milliseconds.
        precision: Decimal precision.

    Returns:
        str: Formatted duration string.

    Example:
        >>> format_duration(1500.5)
        '1.50s'
        >>> format_duration(125.3)
        '125.30ms'
        >>> format_duration(65000)
        '1m 5.00s'
    """
    if milliseconds < 1000:
        return f"{milliseconds:.{precision}f}ms"

    seconds = milliseconds / 1000

    if seconds < 60:
        return f"{seconds:.{precision}f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.{precision}f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    return f"{hours}h {remaining_minutes}m {remaining_seconds:.{precision}f}s"


def format_percentage(
    value: float,
    precision: int = 2,
) -> str:
    """Format value as percentage.

    Args:
        value: Value between 0 and 1.
        precision: Decimal precision.

    Returns:
        str: Formatted percentage string.

    Example:
        >>> format_percentage(0.95)
        '95.00%'
        >>> format_percentage(0.001)
        '0.10%'
    """
    return f"{value * 100:.{precision}f}%"


def format_count(count: int) -> str:
    """Format large count with suffixes.

    Args:
        count: Number to format.

    Returns:
        str: Formatted count string.

    Example:
        >>> format_count(1234567)
        '1.23M'
        >>> format_count(1500)
        '1.50K'
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.2f}K"
    elif count < 1_000_000_000:
        return f"{count / 1_000_000:.2f}M"
    else:
        return f"{count / 1_000_000_000:.2f}B"


def truncate_string(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """Truncate string to maximum length.

    Args:
        text: String to truncate.
        max_length: Maximum length.
        suffix: Suffix to add when truncated.

    Returns:
        str: Truncated string.

    Example:
        >>> truncate_string("Hello World", 8)
        'Hello...'
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


# =============================================================================
# Collection Utilities
# =============================================================================


def chunk_list(
    items: list[T],
    chunk_size: int,
) -> list[list[T]]:
    """Split list into chunks.

    Args:
        items: List to split.
        chunk_size: Size of each chunk.

    Returns:
        list[list[T]]: List of chunks.

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def unique_by(
    items: list[T],
    key: Callable[[T], Any],
) -> list[T]:
    """Get unique items by key function.

    Args:
        items: List of items.
        key: Function to extract key.

    Returns:
        list[T]: Unique items (first occurrence).

    Example:
        >>> items = [{"id": 1, "name": "a"}, {"id": 1, "name": "b"}]
        >>> unique_by(items, lambda x: x["id"])
        [{'id': 1, 'name': 'a'}]
    """
    seen = set()
    result = []

    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            result.append(item)

    return result


def group_by(
    items: list[T],
    key: Callable[[T], Any],
) -> dict[Any, list[T]]:
    """Group items by key function.

    Args:
        items: List of items.
        key: Function to extract key.

    Returns:
        dict[Any, list[T]]: Grouped items.

    Example:
        >>> items = [{"type": "a", "value": 1}, {"type": "b", "value": 2}]
        >>> group_by(items, lambda x: x["type"])
        {'a': [{'type': 'a', 'value': 1}], 'b': [{'type': 'b', 'value': 2}]}
    """
    result: dict[Any, list[T]] = {}

    for item in items:
        k = key(item)
        if k not in result:
            result[k] = []
        result[k].append(item)

    return result


# =============================================================================
# Retry Utility
# =============================================================================


def retry_operation(
    operation: Callable[[], T],
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry an operation with exponential backoff.

    Args:
        operation: Function to execute.
        max_attempts: Maximum number of attempts.
        delay_seconds: Initial delay between attempts.
        backoff_factor: Multiplier for delay.
        exceptions: Exceptions to catch and retry.

    Returns:
        T: Result of operation.

    Raises:
        Exception: Last exception if all attempts fail.

    Example:
        >>> result = retry_operation(
        ...     lambda: unreliable_api_call(),
        ...     max_attempts=3,
        ...     delay_seconds=1.0,
        ... )
    """
    last_exception: Exception | None = None
    delay = delay_seconds

    for attempt in range(max_attempts):
        try:
            return operation()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= backoff_factor

    if last_exception:
        raise last_exception

    # Should not reach here, but satisfy type checker
    raise RuntimeError("Retry failed with no exception captured")


def retry_decorator(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions.

    Args:
        max_attempts: Maximum number of attempts.
        delay_seconds: Initial delay between attempts.
        backoff_factor: Multiplier for delay.
        exceptions: Exceptions to catch and retry.

    Returns:
        Callable: Decorated function.

    Example:
        >>> @retry_decorator(max_attempts=3)
        ... def unreliable_function():
        ...     return api.call()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_operation(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                delay_seconds=delay_seconds,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
            )

        return wrapper

    return decorator


# =============================================================================
# Timing Utilities
# =============================================================================


class Timer:
    """Simple timer for measuring execution time.

    Example:
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do work ...
        >>> timer.stop()
        >>> print(timer.elapsed_ms)
        125.5
    """

    def __init__(self) -> None:
        """Initialize timer."""
        self._start: float | None = None
        self._end: float | None = None

    def start(self) -> None:
        """Start timer."""
        self._start = time.perf_counter()
        self._end = None

    def stop(self) -> float:
        """Stop timer and return elapsed time in milliseconds."""
        if self._start is None:
            raise RuntimeError("Timer not started")
        self._end = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self._start is None:
            return 0.0

        end = self._end or time.perf_counter()
        return (end - self._start) * 1000

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ms / 1000

    def __enter__(self) -> "Timer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()


def timed(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator that returns result and execution time.

    Args:
        func: Function to time.

    Returns:
        Callable: Function returning (result, elapsed_ms).

    Example:
        >>> @timed
        ... def process_data():
        ...     return do_work()
        >>>
        >>> result, elapsed_ms = process_data()
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        with Timer() as timer:
            result = func(*args, **kwargs)
        return result, timer.elapsed_ms

    return wrapper
