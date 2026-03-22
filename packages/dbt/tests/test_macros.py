"""Tests for first-party dbt macro surface."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_window_macro_file_exports_incremental_surface() -> None:
    macro_file = ROOT / "packages" / "dbt" / "macros" / "truthound_window.sql"
    content = macro_file.read_text()

    assert "macro truthound_window_predicate" in content
    assert "macro truthound_windowed_check" in content
    assert "macro truthound_incremental_check" in content
    assert "partition_by" in content


def test_summary_macros_use_numeric_percentage_rounding() -> None:
    utils_file = ROOT / "packages" / "dbt" / "macros" / "truthound_utils.sql"
    check_file = ROOT / "packages" / "dbt" / "macros" / "truthound_check.sql"

    utils_content = utils_file.read_text()
    check_content = check_file.read_text()

    assert "macro round_percentage" in utils_content
    assert "cast(" in utils_content
    assert " as numeric" in utils_content
    assert "truthound.round_percentage('r.failure_count', 't.cnt'" in check_content
    assert "truthound.round_percentage('p.null_count', 't.cnt'" in check_content
    assert "truthound.round_percentage('p.distinct_count', 't.cnt'" in check_content
    assert "round(cast(r.failure_count as {{ dbt.type_float() }}) / t.cnt * 100, 2)" not in check_content
