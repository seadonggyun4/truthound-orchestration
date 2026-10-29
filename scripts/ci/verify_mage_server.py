"""Verify Mage's real scheduler with a dependency-isolated Core interpreter.

Mage 0.9.80 and Core 3.3 require incompatible Typer versions. Both interpreters
must have normally resolved installations; do not use --no-deps overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-python", type=Path, required=True)
    parser.add_argument("--core-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    project = work / "project"
    fixtures = Path(__file__).with_name("host_servers")
    host_python = args.host_python.absolute()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "PATH": str(host_python.parent) + os.pathsep + os.environ["PATH"],
        "MAGE_DATA_DIR": str(work / "variables"),
        "MAGE_DATABASE_CONNECTION_URL": "sqlite:///" + str(work / "mage.db"),
        "REQUIRE_USER_AUTHENTICATION": "false",
        "DISABLE_AUTO_BROWSER_OPEN": "true",
        "SCHEDULER_TRIGGER_INTERVAL": "1",
        "TRUTHOUND_CORE_PYTHON": str(args.core_python.absolute()),
        "TRUTHOUND_CORE_CHECK": str(fixtures / "mage_core_check.py"),
    }
    init = """
import sys
from mage_ai.data_preparation.repo_manager import init_repo
from mage_ai.settings.repo import set_repo_path
from mage_ai.data_preparation.models.pipeline import Pipeline
from mage_ai.data_preparation.models.block import Block
repo=sys.argv[1]
init_repo(repo);set_repo_path(repo)
pipeline=Pipeline.create('core33_server',repo_path=repo)
Block.create('quality','data_loader',repo_path=repo,pipeline=pipeline)
"""
    with (work / "init.log").open("w") as log:
        subprocess.run(
            [str(host_python), "-c", init, str(project)],
            env=env,
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )
    metadata = project / "metadata.yaml"
    metadata.write_text(
        metadata.read_text().replace(
            "variables_dir: ~/.mage_data", "variables_dir: " + json.dumps(str(work / "variables"))
        )
    )
    (project / "data_loaders/quality.py").write_text((fixtures / "mage_block.py").read_text())

    def request(path, payload=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api" + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    with (work / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(host_python.with_name("mage")),
                "start",
                str(project),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError("Mage server exited; inspect server.log")
                try:
                    if "pipeline_schedules" in request(
                        "/pipelines/core33_server/pipeline_schedules"
                    ):
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Mage API startup")
            response = request(
                "/pipelines/core33_server/pipeline_schedules",
                {
                    "pipeline_schedule": {
                        "name": "core33-verification",
                        "schedule_type": "api",
                        "schedule_interval": "@once",
                        "status": "active",
                    }
                },
            )
            schedule = response.get("pipeline_schedule")
            if not schedule:
                raise RuntimeError("Mage rejected schedule: " + str(response.get("error")))
            results = []
            for bad in (False, True):
                response = request(
                    f"/pipeline_schedules/{schedule['id']}/pipeline_runs",
                    {"pipeline_run": {"variables": {"bad": bad}}},
                )
                run = response["pipeline_run"]
                for _ in range(600):
                    run = request(f"/pipeline_runs/{run['id']}")["pipeline_run"]
                    if run["status"] in ("completed", "failed", "cancelled"):
                        break
                    time.sleep(1)
                assert run["status"] == ("failed" if bad else "completed"), run["status"]
                if bad:
                    errors = [(b.get("metrics") or {}).get("error") for b in run["block_runs"]]
                    assert "Isolated Core validation failed" in json.dumps(errors), errors
                results.append({"bad": bad, "run_id": run["id"], "status": run["status"]})
            (work / "results.json").write_text(json.dumps(results, indent=2))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
