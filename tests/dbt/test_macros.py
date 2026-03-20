"""Tests for first-party dbt macro surface."""

from pathlib import Path


def test_window_macro_file_exports_incremental_surface() -> None:
    macro_file = Path("packages/dbt/macros/truthound_window.sql")
    content = macro_file.read_text()

    assert "macro truthound_window_predicate" in content
    assert "macro truthound_windowed_check" in content
    assert "macro truthound_incremental_check" in content
    assert "partition_by" in content
