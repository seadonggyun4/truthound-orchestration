"""Bind independent release checks and immutable distribution hashes to one commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "ci-foundation.yml": (
        "CI Foundation · release",
        {"matrix": 1, "quality": 1, "wheel-smoke (": 6, "summary": 1},
    ),
    "shared-runtime.yml": (
        "Shared Runtime · release · full",
        {"matrix": 1, "shared-runtime (": 1, "summary": 1},
    ),
    "airflow.yml": ("Airflow · release · full", {"matrix": 1, "airflow (": 2, "summary": 1}),
    "prefect.yml": ("Prefect · release · full", {"matrix": 1, "prefect (": 2, "summary": 1}),
    "dagster.yml": ("Dagster · release · full", {"matrix": 1, "dagster (": 2, "summary": 1}),
    "mage.yml": ("Mage · release · full", {"matrix": 1, "mage": 1, "summary": 1}),
    "kestra.yml": ("Kestra · release · full", {"matrix": 1, "kestra (": 2, "summary": 1}),
    "dbt.yml": (
        "dbt · release · full",
        {
            "matrix": 1,
            "dbt-python": 1,
            "compile-matrix (": 5,
            "postgres-execution (": 2,
            "summary": 1,
        },
    ),
    "security.yml": (
        "Security · release",
        {"matrix": 1, "actionlint": 1, "pip-audit (": 7, "codeql (": 2, "summary-blocking": 1},
    ),
}
OPTIONAL_SKIPS = {"dependency-review", "pip-audit-advisory", "pip-audit-canary"}


def api(path: str) -> dict[str, Any]:
    result = subprocess.run(["gh", "api", path], check=True, capture_output=True, text=True)
    value: dict[str, Any] = json.loads(result.stdout)
    return value


def validate_run(run: dict[str, Any], jobs: list[dict[str, Any]], workflow: str, sha: str) -> None:
    """Fail closed on a wrong commit, smoke-only run, missing job or skipped gate."""
    title, required_jobs = REQUIRED[workflow]
    if (
        run.get("path") != f".github/workflows/{workflow}"
        or run.get("head_sha") != sha
        or run.get("event") != "workflow_dispatch"
        or run.get("display_title") != title
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        raise ValueError(f"{workflow}: not a successful full release run for {sha}")
    for prefix, count in required_jobs.items():
        matches = [
            job
            for job in jobs
            if (job["name"].startswith(prefix) if prefix.endswith("(") else job["name"] == prefix)
        ]
        if len(matches) != count or any(
            job.get("status") != "completed" or job.get("conclusion") != "success"
            for job in matches
        ):
            raise ValueError(f"{workflow}: missing or unsuccessful required jobs: {prefix}")
    for job in jobs:
        if (
            job.get("conclusion") == "skipped"
            and workflow == "security.yml"
            and job["name"] in OPTIONAL_SKIPS
        ):
            continue
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise ValueError(f"{workflow}: unsuccessful job {job['name']}")


def verify_runs(run_ids: dict[str, int], sha: str, repository: str) -> dict[str, Any]:
    if set(run_ids) != set(REQUIRED):
        raise ValueError("Provide exactly the nine independent release workflow run IDs")
    verified = {}
    for workflow, run_id in run_ids.items():
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
            raise ValueError("Run IDs must be positive integers")
        path = f"repos/{repository}/actions/runs/{run_id}"
        run = api(path)
        attempt = run["run_attempt"]
        jobs: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = api(f"{path}/attempts/{attempt}/jobs?per_page=100&page={page}")["jobs"]
            jobs.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        validate_run(run, jobs, workflow, sha)
        current = api(path)
        if current["run_attempt"] != attempt or current["conclusion"] != "success":
            raise ValueError(f"{workflow}: run changed during verification")
        verified[workflow] = {"run_id": run_id, "attempt": attempt, "url": run["html_url"]}
    return {"sha": sha, "version": version(), "runs": verified}


def version() -> str:
    return str(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"])


def manifest(dist: Path, sha: str) -> dict[str, Any]:
    files = sorted([*dist.glob("*.whl"), *dist.glob("*.tar.gz")])
    expected = {
        f"truthound_orchestration-{version()}-py3-none-any.whl",
        f"truthound_orchestration-{version()}.tar.gz",
    }
    if {file.name for file in files} != expected:
        raise ValueError("Expected exactly this version's root wheel and sdist")
    return {
        "sha": sha,
        "version": version(),
        "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files},
    }


def verify_artifacts(dist: Path, sha: str) -> None:
    stored = json.loads((dist / "release-manifest.json").read_text())
    if stored != manifest(dist, sha):
        raise ValueError("Distribution version, commit or SHA-256 differs from verified artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["manifest", "verify-artifacts", "verify-runs", "verify-gate"]
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--gate-run", type=int)
    parser.add_argument("--output", type=Path, default=Path("release-evidence.json"))
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.command == "manifest":
        (args.dist_dir / "release-manifest.json").write_text(
            json.dumps(manifest(args.dist_dir, sha), indent=2)
        )
    elif args.command == "verify-artifacts":
        verify_artifacts(args.dist_dir, sha)
    elif args.command == "verify-gate":
        run = api(f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{args.gate_run}")
        if (
            run.get("path") != ".github/workflows/release-gate.yml"
            or run.get("head_sha") != sha
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
        ):
            raise ValueError("Release Gate must have succeeded for this exact commit")
    else:
        if args.runs is None:
            parser.error("--runs is required")
        inputs = json.loads(args.runs.read_text())
        # Publish consumes the verified evidence, then rechecks the referenced runs.
        if "runs" in inputs:
            if inputs["sha"] != sha or inputs["version"] != version():
                raise ValueError("Release evidence is for a different commit or version")
            inputs = {key: value["run_id"] for key, value in inputs["runs"].items()}
        result = verify_runs(inputs, sha, os.environ["GITHUB_REPOSITORY"])
        args.output.write_text(json.dumps(result, indent=2))
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with Path(output).open("a") as handle:
                handle.write(f"foundation_run_id={inputs['ci-foundation.yml']}\n")


if __name__ == "__main__":
    main()
