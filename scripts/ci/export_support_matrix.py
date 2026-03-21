"""Export CI support matrix values for workflows and docs."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_MATRIX_PATH = ROOT / "ci" / "support-matrix.toml"
DEFAULT_DOC_PATH = ROOT / "docs" / "compatibility.md"
BEGIN_MARKER = "<!-- BEGIN GENERATED SUPPORT MATRIX -->"
END_MARKER = "<!-- END GENERATED SUPPORT MATRIX -->"

WHEEL_IMPORTS = {
    "base": ["common"],
    "airflow": ["common", "truthound_airflow"],
    "prefect": ["common", "truthound_prefect"],
    "dagster": ["common", "truthound_dagster"],
    "dbt": ["common", "truthound_dbt", "dbt"],
    "opentelemetry": ["common", "opentelemetry"],
}


def load_support_matrix() -> dict[str, Any]:
    with SUPPORT_MATRIX_PATH.open("rb") as handle:
        return tomllib.load(handle)


def expand_python_versions(data: dict[str, Any], names: list[str]) -> list[str]:
    return [data["python"][name] for name in names]


def expand_platform_versions(
    data: dict[str, Any],
    platform: str,
    names: list[str],
) -> list[str]:
    return [data["platforms"][platform][name] for name in names]


def build_workflow_payload(
    data: dict[str, Any],
    workflow: str,
    lane: str,
) -> dict[str, Any]:
    lane_config = data["lanes"][lane]
    truthound_range = data["truthound"]["range"]
    primary_python = data["python"]["primary"]

    if workflow == "foundation":
        extras = data["wheel_smoke"]["extras"]
        return {
            "python_version": primary_python,
            "truthound_range": truthound_range,
            "wheel_smoke_matrix": {
                "include": [
                    {
                        "label": label,
                        "extra": "" if label == "base" else label,
                        "imports": WHEEL_IMPORTS[label],
                    }
                    for label in extras
                ]
            },
        }

    if workflow == "shared-runtime":
        return {
            "truthound_range": truthound_range,
            "python_matrix": {
                "python_version": expand_python_versions(
                    data,
                    lane_config["shared_runtime_python"],
                )
            },
        }

    if workflow in {"airflow", "prefect", "dagster"}:
        version_key = f"{workflow}_versions"
        return {
            "python_version": primary_python,
            "truthound_range": truthound_range,
            "version_matrix": {
                "version": expand_platform_versions(data, workflow, lane_config[version_key])
            },
        }

    if workflow == "mage-kestra":
        return {
            "python_version": data["python"][lane_config["mage_kestra_python"][0]],
            "truthound_range": truthound_range,
        }

    if workflow == "dbt":
        compile_targets = lane_config["dbt_compile_targets"]
        compile_adapters = data["dbt"]["compile_adapters"]
        execution_target = data["platforms"]["dbt"]["execution_target"]
        return {
            "python_version": primary_python,
            "truthound_range": truthound_range,
            "compile_matrix": {
                "include": [
                    {"target": target, "adapter": compile_adapters[target]}
                    for target in compile_targets
                ]
            },
            "run_execution": bool(lane_config["dbt_execution"]),
            "execution_target": execution_target,
            "execution_adapter": compile_adapters[execution_target],
            "coverage_threshold": str(data["platforms"]["dbt"]["coverage_threshold"]),
        }

    raise ValueError(f"Unsupported workflow: {workflow}")


def build_security_surface(
    data: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    surface = data["security"]["surfaces"][name]
    return {
        "label": name,
        "extra": surface["extra"],
        "host_requirements": list(surface.get("host_requirements", [])),
        "constraints": list(surface.get("constraints", [])),
        "constraint_urls": list(surface.get("constraint_urls", [])),
    }


def build_security_audit_inputs(data: dict[str, Any]) -> dict[str, Any]:
    security = data["security"]
    return {
        "python_version": data["python"][security["python"]],
        "blocking_matrix": {
            "include": [
                build_security_surface(data, name)
                for name in security["blocking_surfaces"]
            ]
        },
        "canary_matrix": {
            "include": [
                build_security_surface(data, name)
                for name in security["canary_surfaces"]
            ]
        },
    }


def render_generated_support_block(data: dict[str, Any]) -> str:
    def join_versions(values: list[str]) -> str:
        return ", ".join(f"`{value}`" for value in values)

    def install_surface(extra: str) -> str:
        if not extra:
            return "`truthound-orchestration`"
        return f"`truthound-orchestration[{extra}]`"

    rows = [
        (
            "PR",
            join_versions(expand_python_versions(data, data["lanes"]["pr"]["foundation_python"])),
            join_versions(expand_platform_versions(data, "airflow", data["lanes"]["pr"]["airflow_versions"])),
            join_versions(expand_platform_versions(data, "prefect", data["lanes"]["pr"]["prefect_versions"])),
            join_versions(expand_platform_versions(data, "dagster", data["lanes"]["pr"]["dagster_versions"])),
            "Primary host smoke",
            "`postgres`",
            "No",
        ),
        (
            "Main",
            join_versions(expand_python_versions(data, data["lanes"]["main"]["foundation_python"])),
            join_versions(expand_platform_versions(data, "airflow", data["lanes"]["main"]["airflow_versions"])),
            join_versions(expand_platform_versions(data, "prefect", data["lanes"]["main"]["prefect_versions"])),
            join_versions(expand_platform_versions(data, "dagster", data["lanes"]["main"]["dagster_versions"])),
            "Primary host smoke",
            ", ".join(
                f"`{target}`" for target in data["lanes"]["main"]["dbt_compile_targets"]
            ),
            "Yes (`postgres`)",
        ),
        (
            "Release",
            join_versions(expand_python_versions(data, data["lanes"]["release"]["foundation_python"])),
            join_versions(
                expand_platform_versions(data, "airflow", data["lanes"]["release"]["airflow_versions"])
            ),
            join_versions(
                expand_platform_versions(data, "prefect", data["lanes"]["release"]["prefect_versions"])
            ),
            join_versions(
                expand_platform_versions(data, "dagster", data["lanes"]["release"]["dagster_versions"])
            ),
            "Primary host smoke",
            ", ".join(
                f"`{target}`" for target in data["lanes"]["release"]["dbt_compile_targets"]
            ),
            "Yes (`postgres`)",
        ),
        (
            "Nightly",
            join_versions(
                expand_python_versions(data, data["lanes"]["nightly"]["shared_runtime_python"])
            ),
            join_versions(
                expand_platform_versions(data, "airflow", data["lanes"]["nightly"]["airflow_versions"])
            ),
            join_versions(
                expand_platform_versions(data, "prefect", data["lanes"]["nightly"]["prefect_versions"])
            ),
            join_versions(
                expand_platform_versions(data, "dagster", data["lanes"]["nightly"]["dagster_versions"])
            ),
            "Primary host smoke + advanced-tier canary",
            ", ".join(
                f"`{target}`" for target in data["lanes"]["nightly"]["dbt_compile_targets"]
            ),
            "Yes (`postgres`)",
        ),
    ]

    platform_rows = [
        (
            "Airflow",
            f"`{data['platforms']['airflow']['min']}`",
            f"`{data['platforms']['airflow']['primary']}`",
        ),
        (
            "Prefect",
            f"`{data['platforms']['prefect']['min']}`",
            f"`{data['platforms']['prefect']['primary']}`",
        ),
        (
            "Dagster",
            f"`{data['platforms']['dagster']['min']}`",
            f"`{data['platforms']['dagster']['primary']}`",
        ),
        (
            "Kestra",
            f"`{data['platforms']['kestra']['min']}`",
            f"`{data['platforms']['kestra']['primary']}`",
        ),
        (
            "dbt",
            (
                f"`{data['platforms']['dbt']['core_range']}` + "
                f"`{data['dbt']['compile_adapters'][data['platforms']['dbt']['execution_target']]}`"
            ),
            (
                f"`dbt-core {data['platforms']['dbt']['execution_core_version']}` + "
                f"`dbt-{data['platforms']['dbt']['execution_target']} "
                f"{data['platforms']['dbt']['execution_adapter_version']}`"
            ),
        ),
    ]

    lines = [
        "## Generated CI Support Matrix",
        "",
        f"Truthound release line: `{data['truthound']['range']}`",
        "",
        "| Lane | Python | Airflow | Prefect | Dagster | Mage / Kestra | dbt Compile | dbt Execute |",
        "|------|--------|---------|---------|---------|----------------|-------------|-------------|",
    ]
    lines.extend(
        f"| {lane} | {python_versions} | {airflow} | {prefect} | {dagster} | {mage_kestra} | {dbt_compile} | {dbt_execute} |"
        for lane, python_versions, airflow, prefect, dagster, mage_kestra, dbt_compile, dbt_execute in rows
    )
    lines.extend(
        [
            "",
            "## Supported Host Version Anchors",
            "",
            "| Platform | Minimum Supported | Primary Supported |",
            "|----------|-------------------|-------------------|",
        ]
    )
    lines.extend(
        f"| {platform} | {minimum} | {primary} |"
        for platform, minimum, primary in platform_rows
    )
    lines.extend(
        [
            "",
            "## Security Audit Surfaces",
            "",
            "| Surface | Install Surface | Release Blocking |",
            "|---------|-----------------|------------------|",
        ]
    )
    lines.extend(
        f"| `{name}` | {install_surface(data['security']['surfaces'][name]['extra'])} | {'Yes' if name in data['security']['blocking_surfaces'] else 'Nightly canary only'} |"
        for name in [*data["security"]["blocking_surfaces"], *data["security"]["canary_surfaces"]]
    )
    lines.extend(
        [
            "",
            "First-party release guarantees apply to per-surface installs. "
            "`truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.",
            "Airflow security audits install with the official Airflow constraints file for the pinned version and Python version.",
        ]
    )
    return "\n".join(lines).strip()


def extract_generated_block(text: str) -> str | None:
    if BEGIN_MARKER not in text or END_MARKER not in text:
        return None
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER)
    return text[start:end].strip()


def update_docs(path: Path, *, write: bool) -> int:
    data = load_support_matrix()
    expected = render_generated_support_block(data)
    text = path.read_text(encoding="utf-8")
    current = extract_generated_block(text)
    if current is None:
        print(f"Missing generated support matrix markers in {path}")
        return 1
    if current == expected:
        print(f"support matrix docs are up to date: {path}")
        return 0
    if not write:
        print(f"support matrix docs are stale: {path}")
        return 1
    updated = text.replace(
        f"{BEGIN_MARKER}\n{current}\n{END_MARKER}",
        f"{BEGIN_MARKER}\n{expected}\n{END_MARKER}",
    )
    path.write_text(updated, encoding="utf-8")
    print(f"updated generated support matrix in {path}")
    return 0


def write_github_output(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in payload.items():
            serialized = json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
            handle.write(f"{key}<<__TRUTHOUND_EOF__\n{serialized}\n__TRUTHOUND_EOF__\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    workflow = subparsers.add_parser("workflow", help="Export workflow metadata.")
    workflow.add_argument("--workflow", required=True)
    workflow.add_argument("--lane", choices=["pr", "main", "release", "nightly"], required=True)
    workflow.add_argument("--github-output", type=Path, default=None)

    docs = subparsers.add_parser("sync-docs", help="Check or update generated docs support matrix.")
    docs.add_argument("--path", type=Path, default=DEFAULT_DOC_PATH)
    docs.add_argument("--write", action="store_true")

    render = subparsers.add_parser("render-docs", help="Render the generated docs support matrix.")
    render.add_argument("--path", type=Path, default=None)

    security = subparsers.add_parser(
        "render-security",
        help="Render deterministic security audit surfaces.",
    )
    security.add_argument("--github-output", type=Path, default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "workflow":
        payload = build_workflow_payload(load_support_matrix(), args.workflow, args.lane)
        if args.github_output is not None:
            write_github_output(args.github_output, payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "sync-docs":
        return update_docs(args.path, write=args.write)
    if args.command == "render-docs":
        output = render_generated_support_block(load_support_matrix())
        if args.path is not None:
            args.path.write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0
    if args.command == "render-security":
        payload = build_security_audit_inputs(load_support_matrix())
        if args.github_output is not None:
            write_github_output(args.github_output, payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
