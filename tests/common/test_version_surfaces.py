"""Regression tests for single-source version surface synchronization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "ci" / "sync_version_surfaces.py"


spec = importlib.util.spec_from_file_location("sync_version_surfaces", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
sync_version_surfaces = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sync_version_surfaces
spec.loader.exec_module(sync_version_surfaces)


def _copy_managed_tree(tmp_path: Path) -> Path:
    temp_root = tmp_path / "repo"
    temp_root.mkdir()

    managed_paths = set(sync_version_surfaces.build_managed_surfaces(ROOT))
    managed_paths.add(Path("ci/version-surfaces.toml"))

    for relative_path in managed_paths:
        source = ROOT / relative_path
        destination = temp_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    return temp_root


def _bump_patch(version: str) -> str:
    major, minor, patch = version.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def test_version_manifest_derives_expected_release_values() -> None:
    config = sync_version_surfaces.load_version_surface_config()

    assert config.package_version == "3.0.1"
    assert config.release_line == "3.x"
    assert config.release_tag == "v3.0.1"
    assert config.package_version_tuple == (3, 0, 1)
    assert config.orchestration_self_requirement == ">=3.0.1,<4.0.0"
    assert config.package_series_requirement == ">=3.0.1,<4.0.0"
    assert config.truthound_primary_version == "3.0.1"
    assert config.truthound_install_range == ">=3.0,<4.0"
    assert config.truthound_runtime_requirement == ">=3.0.1,<4.0.0"


def test_repository_version_surfaces_are_clean() -> None:
    config = sync_version_surfaces.load_version_surface_config()

    assert sync_version_surfaces.validate_expected_tag(config, config.release_tag) is None
    assert sync_version_surfaces.collect_version_surface_drift(ROOT) == {}


def test_detects_stale_adapter_tuple_surface(tmp_path: Path) -> None:
    temp_root = _copy_managed_tree(tmp_path)
    version_file = temp_root / "packages" / "airflow" / "src" / "truthound_airflow" / "version.py"
    config = sync_version_surfaces.load_version_surface_config()
    stale_tuple = str((0, 1, 0)) if config.package_version_tuple != (0, 1, 0) else str((0, 1, 1))
    version_file.write_text(
        version_file.read_text(encoding="utf-8").replace(str(config.package_version_tuple), stale_tuple),
        encoding="utf-8",
    )

    drifts = sync_version_surfaces.collect_version_surface_drift(temp_root)

    assert Path("packages/airflow/src/truthound_airflow/version.py") in drifts


def test_detects_stale_package_pyproject_version(tmp_path: Path) -> None:
    temp_root = _copy_managed_tree(tmp_path)
    pyproject_file = temp_root / "packages" / "prefect" / "pyproject.toml"
    config = sync_version_surfaces.load_version_surface_config()
    stale_version = _bump_patch(config.package_version)
    pyproject_file.write_text(
        pyproject_file.read_text(encoding="utf-8").replace(
            f'version = "{config.package_version}"',
            f'version = "{stale_version}"',
            1,
        ),
        encoding="utf-8",
    )

    drifts = sync_version_surfaces.collect_version_surface_drift(temp_root)

    assert Path("packages/prefect/pyproject.toml") in drifts


def test_detects_stale_dbt_project_version(tmp_path: Path) -> None:
    temp_root = _copy_managed_tree(tmp_path)
    dbt_project = temp_root / "packages" / "dbt" / "dbt_project.yml"
    config = sync_version_surfaces.load_version_surface_config()
    stale_version = _bump_patch(config.package_version)
    dbt_project.write_text(
        dbt_project.read_text(encoding="utf-8").replace(
            f"version: '{config.package_version}'",
            f"version: '{stale_version}'",
            1,
        ),
        encoding="utf-8",
    )

    drifts = sync_version_surfaces.collect_version_surface_drift(temp_root)

    assert Path("packages/dbt/dbt_project.yml") in drifts


def test_detects_stale_support_matrix_truthound_section(tmp_path: Path) -> None:
    temp_root = _copy_managed_tree(tmp_path)
    support_matrix = temp_root / "ci" / "support-matrix.toml"
    config = sync_version_surfaces.load_version_surface_config()
    stale_version = _bump_patch(config.truthound_primary_version)
    support_matrix.write_text(
        support_matrix.read_text(encoding="utf-8").replace(
            f'primary = "{config.truthound_primary_version}"',
            f'primary = "{stale_version}"',
            1,
        ),
        encoding="utf-8",
    )

    drifts = sync_version_surfaces.collect_version_surface_drift(temp_root)

    assert Path("ci/support-matrix.toml") in drifts


def test_detects_stale_root_extra_floor(tmp_path: Path) -> None:
    temp_root = _copy_managed_tree(tmp_path)
    pyproject_file = temp_root / "pyproject.toml"
    pyproject_file.write_text(
        pyproject_file.read_text(encoding="utf-8").replace("kestra>=1.3.0", "kestra>=0.15.0", 1),
        encoding="utf-8",
    )

    drifts = sync_version_surfaces.collect_version_surface_drift(temp_root)

    assert Path("pyproject.toml") in drifts


def test_tag_validation_rejects_mismatched_release_tag() -> None:
    config = sync_version_surfaces.load_version_surface_config()
    stale_tag = f"v{_bump_patch(config.package_version)}"

    error = sync_version_surfaces.validate_expected_tag(config, stale_tag)

    assert error == f"tag {stale_tag!r} does not match manifest release tag {config.release_tag!r}"
