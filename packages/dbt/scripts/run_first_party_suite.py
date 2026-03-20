"""Run the first-party Truthound dbt verification suite.

This runner keeps the dbt verification contract in-repo and is designed to be
used both locally and from CI. It supports two primary lanes:

- compile parity across adapter-dispatch families
- execution parity on the canonical PostgreSQL baseline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any

from common.orchestration import (
    OperationEventType,
    OperationKind,
    create_observability_emitter,
    emit_runtime_event,
)
from common.runtime import ObservabilityBackend, ObservabilityConfig, normalize_runtime_context


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[1] / "integration_tests"
DEFAULT_PROFILES_DIR = DEFAULT_PROJECT_DIR

DEFAULT_ENV = {
    "BIGQUERY_PROJECT": "truthound-local",
    "DATABRICKS_HOST": "https://example.databricks.local",
    "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/local",
    "DATABRICKS_TOKEN": "local-token",
    "REDSHIFT_DATABASE": "dev",
    "REDSHIFT_HOST": "localhost",
    "REDSHIFT_PASSWORD": "postgres",
    "REDSHIFT_USER": "postgres",
    "SNOWFLAKE_ACCOUNT": "local-account",
    "SNOWFLAKE_DATABASE": "ANALYTICS",
    "SNOWFLAKE_PASSWORD": "local-password",
    "SNOWFLAKE_USER": "local-user",
}


def resolve_dbt_executable() -> str:
    """Resolve the dbt CLI path from the active environment."""

    override = os.environ.get("DBT_EXECUTABLE")
    if override:
        return override

    sibling = Path(sys.executable).with_name("dbt")
    if sibling.exists():
        return str(sibling)
    return "dbt"


@dataclass(frozen=True, slots=True)
class SuiteCommand:
    """A single dbt command in the first-party verification suite."""

    name: str
    operation: OperationKind
    argv: tuple[str, ...]
    metadata: dict[str, Any]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("compile", "execute", "all"),
        default="all",
        help="Verification lane to run.",
    )
    parser.add_argument(
        "--target",
        default="postgres",
        help="dbt target name from integration_tests/profiles.yml.",
    )
    parser.add_argument(
        "--project-dir",
        default=str(DEFAULT_PROJECT_DIR),
        help="Path to the dbt integration project.",
    )
    parser.add_argument(
        "--profiles-dir",
        default=str(DEFAULT_PROFILES_DIR),
        help="Path to the dbt profiles directory.",
    )
    parser.add_argument(
        "--skip-macro-smoke",
        action="store_true",
        help="Skip Truthound macro smoke checks.",
    )
    parser.add_argument(
        "--openlineage-endpoint",
        default=None,
        help="Optional OpenLineage HTTP endpoint for runtime events.",
    )
    parser.add_argument(
        "--openlineage-namespace",
        default="truthound",
        help="OpenLineage namespace to use when emitting runtime events.",
    )
    parser.add_argument(
        "--openlineage-job",
        default="dbt-first-party-suite",
        help="OpenLineage job name prefix.",
    )
    return parser.parse_args(argv)


def build_commands(
    mode: str,
    target: str,
    *,
    macro_smoke: bool,
    dbt_executable: str | None = None,
) -> list[SuiteCommand]:
    dbt = dbt_executable or resolve_dbt_executable()
    commands: list[SuiteCommand] = [
        SuiteCommand(
            name="deps",
            operation=OperationKind.OBSERVABILITY,
            argv=(dbt, "deps", "--target", target),
            metadata={"phase": "deps"},
        )
    ]
    if mode in {"compile", "all"}:
        commands.append(
            SuiteCommand(
                name="compile",
                operation=OperationKind.CHECK,
                argv=(dbt, "parse", "--target", target),
                metadata={
                    "phase": "compile",
                    "target": target,
                    "mode": "static-parse",
                },
            )
        )
    if mode in {"execute", "all"}:
        commands.extend(
            [
                SuiteCommand(
                    name="seed",
                    operation=OperationKind.LEARN,
                    argv=(dbt, "seed", "--target", target, "--full-refresh"),
                    metadata={"phase": "seed", "target": target},
                ),
                SuiteCommand(
                    name="run",
                    operation=OperationKind.LEARN,
                    argv=(dbt, "run", "--target", target),
                    metadata={"phase": "run", "target": target},
                ),
                SuiteCommand(
                    name="test",
                    operation=OperationKind.CHECK,
                    argv=(dbt, "test", "--target", target),
                    metadata={"phase": "test", "target": target},
                ),
            ]
        )
        if macro_smoke:
            rule_args = json.dumps(
                {
                    "model_name": "test_model_valid",
                    "rules": [{"column": "id", "check": "not_null"}],
                    "options": {"limit": 50},
                }
            )
            summary_args = json.dumps(
                {
                    "model_name": "test_model_valid",
                    "rules": [{"column": "name", "check": "not_null"}],
                    "options": {"limit": 50},
                }
            )
            commands.extend(
                [
                    SuiteCommand(
                        name="run_truthound_check",
                        operation=OperationKind.CHECK,
                        argv=(
                            dbt,
                            "run-operation",
                            "run_truthound_check",
                            "--target",
                            target,
                            "--args",
                            rule_args,
                        ),
                        metadata={
                            "phase": "run-operation",
                            "macro": "run_truthound_check",
                            "model_name": "test_model_valid",
                        },
                    ),
                    SuiteCommand(
                        name="run_truthound_summary",
                        operation=OperationKind.CHECK,
                        argv=(
                            dbt,
                            "run-operation",
                            "run_truthound_summary",
                            "--target",
                            target,
                            "--args",
                            summary_args,
                        ),
                        metadata={
                            "phase": "run-operation",
                            "macro": "run_truthound_summary",
                            "model_name": "test_model_valid",
                        },
                    ),
                ]
            )
    return commands


def build_env(project_dir: Path, profiles_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["DBT_PROFILES_DIR"] = str(profiles_dir)
    env["DBT_PROJECT_DIR"] = str(project_dir)
    for key, value in DEFAULT_ENV.items():
        env.setdefault(key, value)
    return env


def build_runtime_context(
    *,
    project_dir: Path,
    target: str,
    phase: str,
    invocation_id: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    host_execution = {
        "invocation_id": invocation_id,
        "node_unique_id": metadata.get("node_unique_id") if metadata else None,
        "model_name": metadata.get("model_name") if metadata else None,
        "test_name": metadata.get("test_name") if metadata else None,
    }
    return normalize_runtime_context(
        platform="dbt",
        project_root=str(project_dir),
        host_metadata={"suite": "first_party", "phase": phase, "target": target},
        host_execution=host_execution,
    )


def build_observability_config(args: argparse.Namespace) -> ObservabilityConfig | None:
    if args.openlineage_endpoint is None:
        return None
    return ObservabilityConfig(
        backend=ObservabilityBackend.OPENLINEAGE,
        endpoint=args.openlineage_endpoint,
        namespace=args.openlineage_namespace,
        job_name=args.openlineage_job,
    )


def run_command(
    command: SuiteCommand,
    *,
    project_dir: Path,
    env: dict[str, str],
    target: str,
    emitter: Any,
    invocation_id: str,
) -> None:
    runtime_context = build_runtime_context(
        project_dir=project_dir,
        target=target,
        phase=command.name,
        invocation_id=invocation_id,
        metadata=command.metadata,
    )
    emit_runtime_event(
        emitter,
        event_type=OperationEventType.STARTED,
        operation=command.operation,
        engine_name="truthound-dbt",
        runtime_context=runtime_context,
        metadata={"argv": list(command.argv), **command.metadata},
    )
    try:
        completed = subprocess.run(
            command.argv,
            cwd=project_dir,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        emit_runtime_event(
            emitter,
            event_type=OperationEventType.COMPLETED,
            operation=command.operation,
            engine_name="truthound-dbt",
            runtime_context=runtime_context,
            metadata={
                "argv": list(command.argv),
                "returncode": completed.returncode,
                **command.metadata,
            },
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        emit_runtime_event(
            emitter,
            event_type=OperationEventType.FAILED,
            operation=command.operation,
            engine_name="truthound-dbt",
            runtime_context=runtime_context,
            metadata={
                "argv": list(command.argv),
                "returncode": exc.returncode,
                **command.metadata,
            },
            error=exc,
        )
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    project_dir = Path(args.project_dir).resolve()
    profiles_dir = Path(args.profiles_dir).resolve()
    env = build_env(project_dir, profiles_dir)
    commands = build_commands(
        args.mode,
        args.target,
        macro_smoke=not args.skip_macro_smoke,
        dbt_executable=resolve_dbt_executable(),
    )
    observability = build_observability_config(args)
    emitter = create_observability_emitter(observability)
    invocation_id = str(uuid.uuid4())

    print(
        json.dumps(
            {
                "mode": args.mode,
                "target": args.target,
                "project_dir": str(project_dir),
                "profiles_dir": str(profiles_dir),
                "commands": [command.name for command in commands],
            }
        )
    )
    for command in commands:
        run_command(
            command,
            project_dir=project_dir,
            env=env,
            target=args.target,
            emitter=emitter,
            invocation_id=invocation_id,
        )
    emitter.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
