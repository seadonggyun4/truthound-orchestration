"""Tests for helper utilities."""

from __future__ import annotations

from typing import Any

import pytest


class TestSafeGet:
    """Tests for safe_get function."""

    def test_single_key(self) -> None:
        """Test getting a single key."""
        from truthound_airflow.utils.helpers import safe_get

        data = {"a": 1, "b": 2}

        assert safe_get(data, "a") == 1
        assert safe_get(data, "b") == 2

    def test_nested_keys(self) -> None:
        """Test getting nested keys."""
        from truthound_airflow.utils.helpers import safe_get

        data = {"a": {"b": {"c": 1}}}

        assert safe_get(data, "a", "b", "c") == 1

    def test_missing_key_returns_default(self) -> None:
        """Test that missing key returns default."""
        from truthound_airflow.utils.helpers import safe_get

        data = {"a": 1}

        assert safe_get(data, "b") is None
        assert safe_get(data, "b", default=0) == 0

    def test_nested_missing_key_returns_default(self) -> None:
        """Test that missing nested key returns default."""
        from truthound_airflow.utils.helpers import safe_get

        data = {"a": {"b": 1}}

        assert safe_get(data, "a", "x", "y", default=0) == 0


class TestMergeDicts:
    """Tests for merge_dicts function."""

    def test_simple_merge(self) -> None:
        """Test simple merge."""
        from truthound_airflow.utils.helpers import merge_dicts

        base = {"a": 1}
        override = {"b": 2}

        result = merge_dicts(base, override)

        assert result == {"a": 1, "b": 2}

    def test_override(self) -> None:
        """Test override behavior."""
        from truthound_airflow.utils.helpers import merge_dicts

        base = {"a": 1}
        override = {"a": 2}

        result = merge_dicts(base, override)

        assert result == {"a": 2}

    def test_deep_merge(self) -> None:
        """Test deep merge of nested dicts."""
        from truthound_airflow.utils.helpers import merge_dicts

        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}}

        result = merge_dicts(base, override, deep=True)

        assert result == {"a": 1, "b": {"c": 2, "d": 3}}

    def test_shallow_merge(self) -> None:
        """Test shallow merge replaces nested dicts."""
        from truthound_airflow.utils.helpers import merge_dicts

        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}}

        result = merge_dicts(base, override, deep=False)

        assert result == {"a": 1, "b": {"d": 3}}

    def test_does_not_modify_original(self) -> None:
        """Test that original dicts are not modified."""
        from truthound_airflow.utils.helpers import merge_dicts

        base = {"a": 1}
        override = {"b": 2}

        merge_dicts(base, override)

        assert base == {"a": 1}
        assert override == {"b": 2}


class TestFlattenDict:
    """Tests for flatten_dict function."""

    def test_simple_dict(self) -> None:
        """Test flattening simple dict."""
        from truthound_airflow.utils.helpers import flatten_dict

        data = {"a": 1, "b": 2}

        result = flatten_dict(data)

        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self) -> None:
        """Test flattening nested dict."""
        from truthound_airflow.utils.helpers import flatten_dict

        data = {"a": {"b": 1, "c": 2}}

        result = flatten_dict(data)

        assert result == {"a.b": 1, "a.c": 2}

    def test_custom_separator(self) -> None:
        """Test custom separator."""
        from truthound_airflow.utils.helpers import flatten_dict

        data = {"a": {"b": 1}}

        result = flatten_dict(data, separator="_")

        assert result == {"a_b": 1}


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_milliseconds(self) -> None:
        """Test formatting milliseconds."""
        from truthound_airflow.utils.helpers import format_duration

        assert format_duration(125.5) == "125.50ms"

    def test_seconds(self) -> None:
        """Test formatting seconds."""
        from truthound_airflow.utils.helpers import format_duration

        assert format_duration(1500.0) == "1.50s"

    def test_minutes(self) -> None:
        """Test formatting minutes."""
        from truthound_airflow.utils.helpers import format_duration

        assert format_duration(65000.0) == "1m 5.00s"

    def test_hours(self) -> None:
        """Test formatting hours."""
        from truthound_airflow.utils.helpers import format_duration

        assert format_duration(3700000.0) == "1h 1m 40.00s"

    def test_custom_precision(self) -> None:
        """Test custom precision."""
        from truthound_airflow.utils.helpers import format_duration

        assert format_duration(1500.123, precision=1) == "1.5s"


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_basic(self) -> None:
        """Test basic percentage formatting."""
        from truthound_airflow.utils.helpers import format_percentage

        assert format_percentage(0.95) == "95.00%"

    def test_small_value(self) -> None:
        """Test small percentage value."""
        from truthound_airflow.utils.helpers import format_percentage

        assert format_percentage(0.001) == "0.10%"

    def test_custom_precision(self) -> None:
        """Test custom precision."""
        from truthound_airflow.utils.helpers import format_percentage

        assert format_percentage(0.956789, precision=1) == "95.7%"


class TestFormatCount:
    """Tests for format_count function."""

    def test_small_number(self) -> None:
        """Test small number formatting."""
        from truthound_airflow.utils.helpers import format_count

        assert format_count(999) == "999"

    def test_thousands(self) -> None:
        """Test thousands formatting."""
        from truthound_airflow.utils.helpers import format_count

        assert format_count(1500) == "1.50K"

    def test_millions(self) -> None:
        """Test millions formatting."""
        from truthound_airflow.utils.helpers import format_count

        assert format_count(1234567) == "1.23M"

    def test_billions(self) -> None:
        """Test billions formatting."""
        from truthound_airflow.utils.helpers import format_count

        assert format_count(1234567890) == "1.23B"


class TestTruncateString:
    """Tests for truncate_string function."""

    def test_short_string_unchanged(self) -> None:
        """Test short string is unchanged."""
        from truthound_airflow.utils.helpers import truncate_string

        result = truncate_string("Hello", max_length=10)

        assert result == "Hello"

    def test_long_string_truncated(self) -> None:
        """Test long string is truncated."""
        from truthound_airflow.utils.helpers import truncate_string

        result = truncate_string("Hello World", max_length=8)

        assert result == "Hello..."

    def test_custom_suffix(self) -> None:
        """Test custom suffix."""
        from truthound_airflow.utils.helpers import truncate_string

        result = truncate_string("Hello World", max_length=8, suffix="~")

        assert result == "Hello W~"


class TestChunkList:
    """Tests for chunk_list function."""

    def test_even_chunks(self) -> None:
        """Test even chunks."""
        from truthound_airflow.utils.helpers import chunk_list

        result = chunk_list([1, 2, 3, 4], 2)

        assert result == [[1, 2], [3, 4]]

    def test_uneven_chunks(self) -> None:
        """Test uneven chunks."""
        from truthound_airflow.utils.helpers import chunk_list

        result = chunk_list([1, 2, 3, 4, 5], 2)

        assert result == [[1, 2], [3, 4], [5]]

    def test_empty_list(self) -> None:
        """Test empty list."""
        from truthound_airflow.utils.helpers import chunk_list

        result = chunk_list([], 2)

        assert result == []


class TestUniqueBy:
    """Tests for unique_by function."""

    def test_unique_items(self) -> None:
        """Test unique items by key."""
        from truthound_airflow.utils.helpers import unique_by

        items = [{"id": 1, "name": "a"}, {"id": 1, "name": "b"}, {"id": 2, "name": "c"}]

        result = unique_by(items, lambda x: x["id"])

        assert len(result) == 2
        assert result[0]["name"] == "a"

    def test_all_unique(self) -> None:
        """Test all unique items."""
        from truthound_airflow.utils.helpers import unique_by

        items = [{"id": 1}, {"id": 2}, {"id": 3}]

        result = unique_by(items, lambda x: x["id"])

        assert len(result) == 3


class TestGroupBy:
    """Tests for group_by function."""

    def test_group_items(self) -> None:
        """Test grouping items by key."""
        from truthound_airflow.utils.helpers import group_by

        items = [
            {"type": "a", "value": 1},
            {"type": "b", "value": 2},
            {"type": "a", "value": 3},
        ]

        result = group_by(items, lambda x: x["type"])

        assert len(result["a"]) == 2
        assert len(result["b"]) == 1


class TestTimer:
    """Tests for Timer class."""

    def test_start_stop(self) -> None:
        """Test basic start/stop."""
        import time

        from truthound_airflow.utils.helpers import Timer

        timer = Timer()
        timer.start()
        time.sleep(0.01)  # 10ms
        elapsed = timer.stop()

        assert elapsed > 0

    def test_elapsed_ms_property(self) -> None:
        """Test elapsed_ms property."""
        import time

        from truthound_airflow.utils.helpers import Timer

        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.stop()

        assert timer.elapsed_ms > 0

    def test_elapsed_seconds_property(self) -> None:
        """Test elapsed_seconds property."""
        from truthound_airflow.utils.helpers import Timer

        timer = Timer()
        timer.start()
        timer.stop()

        assert timer.elapsed_seconds >= 0

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        import time

        from truthound_airflow.utils.helpers import Timer

        with Timer() as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms > 0

    def test_not_started_raises(self) -> None:
        """Test stop before start raises."""
        from truthound_airflow.utils.helpers import Timer

        timer = Timer()

        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.stop()


class TestTimedDecorator:
    """Tests for timed decorator."""

    def test_timed_returns_tuple(self) -> None:
        """Test timed decorator returns result and time."""
        from truthound_airflow.utils.helpers import timed

        @timed
        def test_func():
            return "result"

        result, elapsed = test_func()

        assert result == "result"
        assert elapsed > 0
