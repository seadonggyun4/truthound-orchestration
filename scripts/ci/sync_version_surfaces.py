"""Synchronize repository version-bearing surfaces from a single manifest."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


ROOT = Path(__file__).resolve().parents[2]
VERSION_MANIFEST_PATH = ROOT / "ci" / "version-surfaces.toml"
SUPPORT_MATRIX_PATH = ROOT / "ci" / "support-matrix.toml"

ROOT_PYPROJECT = Path("pyproject.toml")
SUPPORT_MATRIX = Path("ci/support-matrix.toml")
COMMON_INIT = Path("common/__init__.py")
TRUTHOUND_RUNTIME = Path("common/engines/truthound_runtime.py")
ROOT_README = Path("README.md")

AIRFLOW_VERSION = Path("packages/airflow/src/truthound_airflow/version.py")
DAGSTER_VERSION = Path("packages/dagster/src/truthound_dagster/version.py")
PREFECT_VERSION = Path("packages/prefect/src/truthound_prefect/version.py")
KESTRA_VERSION = Path("packages/kestra/src/truthound_kestra/version.py")
MAGE_VERSION = Path("packages/mage/src/truthound_mage/version.py")
DBT_INIT = Path("packages/dbt/src/truthound_dbt/__init__.py")

AIRFLOW_PYPROJECT = Path("packages/airflow/pyproject.toml")
DAGSTER_PYPROJECT = Path("packages/dagster/pyproject.toml")
PREFECT_PYPROJECT = Path("packages/prefect/pyproject.toml")
KESTRA_PYPROJECT = Path("packages/kestra/pyproject.toml")
MAGE_PYPROJECT = Path("packages/mage/pyproject.toml")
DBT_PROJECT = Path("packages/dbt/dbt_project.yml")

DAGSTER_README = Path("packages/dagster/README.md")
PREFECT_README = Path("packages/prefect/README.md")
DBT_README = Path("packages/dbt/README.md")

DOC_GETTING_STARTED = Path("docs/getting-started.md")
DOC_COMPATIBILITY = Path("docs/compatibility.md")
DOC_MIGRATION = Path("docs/migration/3.0.md")
DOC_AIRFLOW_INDEX = Path("docs/airflow/index.md")
DOC_AIRFLOW_INSTALL = Path("docs/airflow/install-compatibility.md")
DOC_DAGSTER_INDEX = Path("docs/dagster/index.md")
DOC_DAGSTER_INSTALL = Path("docs/dagster/install-compatibility.md")
DOC_PREFECT_INDEX = Path("docs/prefect/index.md")
DOC_PREFECT_INSTALL = Path("docs/prefect/install-compatibility.md")
DOC_DBT_INDEX = Path("docs/dbt/index.md")
DOC_DBT_PACKAGE_SETUP = Path("docs/dbt/package-setup.md")
DOC_DBT_PACKAGE_UPGRADES = Path("docs/dbt/package-upgrades.md")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _parse_version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"Expected semantic version 'X.Y.Z', got: {value}")
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


def _replace_once(text: str, pattern: str, replacement: str, *, context: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise ValueError(f"Expected one match for {context}, found {count}")
    return updated


def _replace_all(text: str, pattern: str, replacement: str, *, context: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count < 1:
        raise ValueError(f"Expected at least one match for {context}, found 0")
    return updated


def _replace_optional_all(text: str, pattern: str, replacement: str) -> str:
    updated, _ = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    return updated


class PlatformSpec(TypedDict):
    host_dependency: tuple[str, str] | None
    optional_host_extra: tuple[str, str] | None


@dataclass(frozen=True)
class VersionSurfaceConfig:
    package_version: str
    release_line: str
    release_tag: str
    truthound_primary_version: str
    truthound_install_range: str
    truthound_runtime_requirement: str

    @property
    def package_version_tuple(self) -> tuple[int, int, int]:
        return _parse_version(self.package_version)

    @property
    def truthound_primary_tuple(self) -> tuple[int, int, int]:
        return _parse_version(self.truthound_primary_version)

    @property
    def orchestration_self_requirement(self) -> str:
        major, _, _ = self.package_version_tuple
        return f">={self.package_version},<{major + 1}.0.0"

    @property
    def package_series_requirement(self) -> str:
        major, _, _ = self.package_version_tuple
        return f">={self.package_version},<{major + 1}.0.0"

    @property
    def truthound_major(self) -> int:
        major, _, _ = self.truthound_primary_tuple
        return major

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionSurfaceConfig:
        release = data["release"]
        truthound = data["truthound"]
        config = cls(
            package_version=str(release["package_version"]),
            release_line=str(release["line"]),
            release_tag=str(release["tag"]),
            truthound_primary_version=str(truthound["primary_version"]),
            truthound_install_range=str(truthound["install_range"]),
            truthound_runtime_requirement=str(truthound["runtime_requirement"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        package_major, _, _ = self.package_version_tuple
        truthound_major, _, _ = self.truthound_primary_tuple
        expected_line = f"{package_major}.x"
        expected_tag = f"v{self.package_version}"

        if self.release_line != expected_line:
            raise ValueError(
                f"Manifest release line {self.release_line!r} does not match package version {self.package_version!r}"
            )
        if self.release_tag != expected_tag:
            raise ValueError(
                f"Manifest release tag {self.release_tag!r} does not match package version {self.package_version!r}"
            )

        expected_install_range = f">={truthound_major}.0,<{truthound_major + 1}.0"
        if self.truthound_install_range != expected_install_range:
            raise ValueError(
                "Manifest truthound.install_range does not match truthound.primary_version major: "
                f"{self.truthound_install_range!r} != {expected_install_range!r}"
            )

        expected_runtime = f">={self.truthound_primary_version},<{truthound_major + 1}.0.0"
        if self.truthound_runtime_requirement != expected_runtime:
            raise ValueError(
                "Manifest truthound.runtime_requirement does not match truthound.primary_version: "
                f"{self.truthound_runtime_requirement!r} != {expected_runtime!r}"
            )


def load_version_surface_config(path: Path = VERSION_MANIFEST_PATH) -> VersionSurfaceConfig:
    return VersionSurfaceConfig.from_dict(_load_toml(path))


def load_support_matrix(path: Path = SUPPORT_MATRIX_PATH) -> dict[str, Any]:
    return _load_toml(path)


def render_airflow_version_module(config: VersionSurfaceConfig) -> str:
    return (
        '"""Version Information for truthound-airflow.\n\n'
        "This module provides version information for the package.\n"
        "The version follows Semantic Versioning (SemVer).\n"
        '"""\n\n'
        f'__version__ = "{config.package_version}"\n'
        f"__version_tuple__ = {config.package_version_tuple}\n"
    )


def render_dagster_version_module(config: VersionSurfaceConfig) -> str:
    return (
        '"""Version information for truthound-dagster."""\n\n'
        f'__version__ = "{config.package_version}"\n'
        f"__version_tuple__ = {config.package_version_tuple}\n"
        "__version_info__ = __version_tuple__  # Alias for compatibility\n"
    )


def render_prefect_version_module(config: VersionSurfaceConfig) -> str:
    return (
        '"""Version information for truthound-prefect."""\n\n'
        f'__version__ = "{config.package_version}"\n'
        f"__version_info__ = {config.package_version_tuple}\n"
    )


def render_kestra_version_module(config: VersionSurfaceConfig) -> str:
    return (
        '"""Version information for truthound-kestra."""\n\n'
        f'__version__ = "{config.package_version}"\n'
        f"__version_tuple__ = {config.package_version_tuple}\n"
        "__version_info__ = __version_tuple__\n"
    )


def render_mage_version_module(config: VersionSurfaceConfig) -> str:
    return (
        '"""Truthound Mage package version information."""\n\n'
        f'__version__ = "{config.package_version}"\n'
    )


def sync_root_pyproject(text: str, config: VersionSurfaceConfig, support: dict[str, Any]) -> str:
    platforms = support["platforms"]
    dbt_compile = support["dbt"]["compile_adapters"]

    text = _replace_once(
        text,
        r'(^\[project\]\n.*?^version = ")([^"]+)(")',
        rf'\g<1>{config.package_version}\g<3>',
        context="root pyproject project.version",
    )
    text = _replace_once(
        text,
        r'"truthound>=[^"]+"',
        f'"truthound{config.truthound_install_range}"',
        context="root pyproject truthound dependency",
    )
    replacements = {
        "apache-airflow": f'>={platforms["airflow"]["min"]}',
        "dagster": f'>={platforms["dagster"]["min"]}',
        "prefect": f'>={platforms["prefect"]["min"]}',
        "kestra": f'>={platforms["kestra"]["min"]}',
        "dbt-core": support["platforms"]["dbt"]["core_range"].split("dbt-core", 1)[1],
        "dbt-postgres": dbt_compile["postgres"].split("dbt-postgres", 1)[1],
    }
    for package, requirement in replacements.items():
        pattern = rf'"{re.escape(package)}[<>=][^"]*"'
        text = _replace_all(
            text,
            pattern,
            f'"{package}{requirement}"',
            context=f"root pyproject {package} surfaces",
        )
    return text


def sync_package_pyproject(
    text: str,
    *,
    config: VersionSurfaceConfig,
    support: dict[str, Any],
    host_dependency: tuple[str, str] | None,
    optional_host_extra: tuple[str, str] | None = None,
) -> str:
    text = _replace_once(
        text,
        r'(^\[project\]\n.*?^version = ")([^"]+)(")',
        rf'\g<1>{config.package_version}\g<3>',
        context="package pyproject project.version",
    )
    text = _replace_all(
        text,
        r'"truthound-orchestration>=[^"]+"',
        f'"truthound-orchestration{config.orchestration_self_requirement}"',
        context="package pyproject truthound-orchestration dependency",
    )
    text = _replace_all(
        text,
        r'"truthound>=[^"]+"',
        f'"truthound{config.truthound_install_range}"',
        context="package pyproject truthound dependency",
    )
    if host_dependency is not None:
        package_name, requirement = host_dependency
        text = _replace_all(
            text,
            rf'"{re.escape(package_name)}[<>=][^"]*"',
            f'"{package_name}{requirement}"',
            context=f"package pyproject host dependency {package_name}",
        )
    if optional_host_extra is not None:
        package_name, requirement = optional_host_extra
        text = _replace_all(
            text,
            rf'"{re.escape(package_name)}[<>=][^"]*"',
            f'"{package_name}{requirement}"',
            context=f"package pyproject optional host dependency {package_name}",
        )
    return text


def sync_support_matrix(text: str, config: VersionSurfaceConfig) -> str:
    replacement = (
        "[truthound]\n"
        f'range = "{config.truthound_install_range}"\n'
        f'primary = "{config.truthound_primary_version}"\n'
    )
    return _replace_once(
        text,
        r"^\[truthound\]\nrange = \".*?\"\nprimary = \".*?\"\n",
        replacement,
        context="support matrix truthound section",
    )


def sync_common_init(text: str, config: VersionSurfaceConfig) -> str:
    return _replace_once(
        text,
        r'^__version__ = ".*?"$',
        f'__version__ = "{config.package_version}"',
        context="common.__version__",
    )


def sync_truthound_runtime(text: str, config: VersionSurfaceConfig) -> str:
    text = _replace_once(
        text,
        r'^TRUTHOUND_VERSION_REQUIREMENT = ".*?"$',
        f'TRUTHOUND_VERSION_REQUIREMENT = "{config.truthound_runtime_requirement}"',
        context="truthound runtime requirement",
    )
    return _replace_all(
        text,
        r"truthound>=[^,'\"]+,<[^'\"]+",
        f"truthound{config.truthound_install_range}",
        context="truthound runtime install hints",
    )


def sync_dbt_init(text: str, config: VersionSurfaceConfig) -> str:
    return _replace_once(
        text,
        r'^__version__ = ".*?"$',
        f'__version__ = "{config.package_version}"',
        context="dbt __version__",
    )


def sync_dbt_project(text: str, config: VersionSurfaceConfig) -> str:
    return _replace_once(
        text,
        r"^version:\s*['\"].*?['\"]$",
        f"version: '{config.package_version}'",
        context="dbt_project version",
    )


def sync_generic_truthound_docs(text: str, config: VersionSurfaceConfig) -> str:
    text = _replace_optional_all(
        text,
        r"truthound>=\d+\.\d+,<\d+\.\d+",
        f"truthound{config.truthound_install_range}",
    )
    text = _replace_optional_all(
        text,
        r"`>=\d+\.\d+,<\d+\.\d+`",
        f"`{config.truthound_install_range}`",
    )
    return _replace_optional_all(
        text,
        r'"[>]*=\d+\.\d+\.\d+,<\d+\.\d+\.\d+"',
        f'"{config.package_series_requirement}"',
    )


def sync_dbt_readme(text: str, config: VersionSurfaceConfig) -> str:
    text = _replace_all(
        text,
        r'"[>]*=\d+\.\d+\.\d+,<\d+\.\d+\.\d+"',
        f'"{config.package_series_requirement}"',
        context="dbt readme package series",
    )
    return _replace_all(
        text,
        r'"v\d+\.\d+\.\d+"',
        f'"{config.release_tag}"',
        context="dbt readme git revision",
    )


def build_managed_surfaces(root: Path = ROOT) -> dict[Path, str]:
    config = load_version_surface_config(root / VERSION_MANIFEST_PATH.relative_to(ROOT))
    support = load_support_matrix(root / SUPPORT_MATRIX_PATH.relative_to(ROOT))

    platform_specs: dict[Path, PlatformSpec] = {
        AIRFLOW_PYPROJECT: {
            "host_dependency": ("apache-airflow", f'>={support["platforms"]["airflow"]["min"]}'),
            "optional_host_extra": None,
        },
        DAGSTER_PYPROJECT: {
            "host_dependency": ("dagster", f'>={support["platforms"]["dagster"]["min"]}'),
            "optional_host_extra": None,
        },
        PREFECT_PYPROJECT: {
            "host_dependency": ("prefect", f'>={support["platforms"]["prefect"]["min"]}'),
            "optional_host_extra": None,
        },
        KESTRA_PYPROJECT: {
            "host_dependency": None,
            "optional_host_extra": ("kestra", f'>={support["platforms"]["kestra"]["min"]}'),
        },
        MAGE_PYPROJECT: {
            "host_dependency": None,
            "optional_host_extra": None,
        },
    }

    surfaces: dict[Path, str] = {}

    root_pyproject_path = root / ROOT_PYPROJECT
    surfaces[ROOT_PYPROJECT] = sync_root_pyproject(
        root_pyproject_path.read_text(encoding="utf-8"),
        config,
        support,
    )

    support_matrix_path = root / SUPPORT_MATRIX
    surfaces[SUPPORT_MATRIX] = sync_support_matrix(
        support_matrix_path.read_text(encoding="utf-8"),
        config,
    )

    common_init_path = root / COMMON_INIT
    surfaces[COMMON_INIT] = sync_common_init(
        common_init_path.read_text(encoding="utf-8"),
        config,
    )

    runtime_path = root / TRUTHOUND_RUNTIME
    surfaces[TRUTHOUND_RUNTIME] = sync_truthound_runtime(
        runtime_path.read_text(encoding="utf-8"),
        config,
    )

    surfaces[AIRFLOW_VERSION] = render_airflow_version_module(config)
    surfaces[DAGSTER_VERSION] = render_dagster_version_module(config)
    surfaces[PREFECT_VERSION] = render_prefect_version_module(config)
    surfaces[KESTRA_VERSION] = render_kestra_version_module(config)
    surfaces[MAGE_VERSION] = render_mage_version_module(config)

    dbt_init_path = root / DBT_INIT
    surfaces[DBT_INIT] = sync_dbt_init(
        dbt_init_path.read_text(encoding="utf-8"),
        config,
    )

    for relative_path, kwargs in platform_specs.items():
        current_text = (root / relative_path).read_text(encoding="utf-8")
        surfaces[relative_path] = sync_package_pyproject(
            current_text,
            config=config,
            support=support,
            host_dependency=kwargs["host_dependency"],
            optional_host_extra=kwargs["optional_host_extra"],
        )

    dbt_project_path = root / DBT_PROJECT
    surfaces[DBT_PROJECT] = sync_dbt_project(
        dbt_project_path.read_text(encoding="utf-8"),
        config,
    )

    generic_doc_paths = (
        ROOT_README,
        DAGSTER_README,
        PREFECT_README,
        DOC_GETTING_STARTED,
        DOC_COMPATIBILITY,
        DOC_MIGRATION,
        DOC_AIRFLOW_INDEX,
        DOC_AIRFLOW_INSTALL,
        DOC_DAGSTER_INDEX,
        DOC_DAGSTER_INSTALL,
        DOC_PREFECT_INDEX,
        DOC_PREFECT_INSTALL,
        DOC_DBT_INDEX,
        DOC_DBT_PACKAGE_SETUP,
        DOC_DBT_PACKAGE_UPGRADES,
    )
    for relative_path in generic_doc_paths:
        current_text = (root / relative_path).read_text(encoding="utf-8")
        surfaces[relative_path] = sync_generic_truthound_docs(current_text, config)

    dbt_readme_path = root / DBT_README
    surfaces[DBT_README] = sync_dbt_readme(
        dbt_readme_path.read_text(encoding="utf-8"),
        config,
    )

    return surfaces


def collect_version_surface_drift(root: Path = ROOT) -> dict[Path, str]:
    drifts: dict[Path, str] = {}
    for relative_path, expected_text in build_managed_surfaces(root).items():
        current_text = (root / relative_path).read_text(encoding="utf-8")
        if current_text != expected_text:
            drifts[relative_path] = "content differs from generated version surface"
    return drifts


def validate_expected_tag(config: VersionSurfaceConfig, expected_tag: str | None) -> str | None:
    if not expected_tag:
        return None
    if expected_tag != config.release_tag:
        return f"tag {expected_tag!r} does not match manifest release tag {config.release_tag!r}"
    return None


def write_version_surfaces(root: Path = ROOT) -> list[Path]:
    written: list[Path] = []
    for relative_path, expected_text in build_managed_surfaces(root).items():
        path = root / relative_path
        current_text = path.read_text(encoding="utf-8")
        if current_text != expected_text:
            path.write_text(expected_text, encoding="utf-8")
            written.append(relative_path)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "write"))
    parser.add_argument("--expected-tag", help="Expected git tag, e.g. v3.0.0")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Repository root to operate on",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    config = load_version_surface_config(root / VERSION_MANIFEST_PATH.relative_to(ROOT))

    print(f"Loaded version manifest: {VERSION_MANIFEST_PATH.relative_to(ROOT)}")
    print(f"  package_version={config.package_version}")
    print(f"  release_line={config.release_line}")
    print(f"  release_tag={config.release_tag}")
    print(f"  truthound_primary_version={config.truthound_primary_version}")
    print(f"  truthound_install_range={config.truthound_install_range}")
    print(f"  truthound_runtime_requirement={config.truthound_runtime_requirement}")

    tag_error = validate_expected_tag(config, args.expected_tag)
    if tag_error:
        print(f"Tag validation failed: {tag_error}", file=sys.stderr)
        return 1
    if args.expected_tag:
        print(f"Validated expected tag: {args.expected_tag}")

    managed_paths = sorted(build_managed_surfaces(root))
    print("Managed version surfaces:")
    for relative_path in managed_paths:
        print(f"  - {relative_path}")

    if args.command == "write":
        written = write_version_surfaces(root)
        if written:
            print("Updated files:")
            for relative_path in written:
                print(f"  - {relative_path}")
        else:
            print("No version surface updates were necessary.")
        return 0

    drifts = collect_version_surface_drift(root)
    if drifts:
        print("Version surface drift detected:", file=sys.stderr)
        for relative_path, reason in drifts.items():
            print(f"  - {relative_path}: {reason}", file=sys.stderr)
        return 1

    print("All managed version surfaces are in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
