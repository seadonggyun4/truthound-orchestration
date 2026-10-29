"""Isolated real-host execution regression; invoked by verify_host_server.py."""

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any


root = pathlib.Path(sys.argv[1])
name = "runtime"
py = sys.executable
port = sys.argv[2]
work = root / name
work.mkdir(exist_ok=True)
env = dict(
    os.environ,
    PATH=str(pathlib.Path(py).parent) + os.pathsep + os.environ["PATH"],
    DAGSTER_HOME=str(work),
)
(work / "dagster.yaml").write_text("telemetry:\n  enabled: false\n")
f = (work / "server.log").open("w")
p = subprocess.Popen(
    [py, "-m", "dagster", "dev", "-f", str(root / "dagster_job.py"), "-h", "127.0.0.1", "-p", port],
    env=env,
    cwd=root,
    stdout=f,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)


def query(q: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/graphql",
        data=json.dumps({"query": q, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        result: dict[str, Any] = json.load(r)
        return result


try:
    for _ in range(600):
        if p.poll() is not None:
            raise RuntimeError("server exited")
        try:
            data = query(
                "{repositoriesOrError{... on RepositoryConnection{nodes{name location{name}}}}}"
            )
            repos = data["data"]["repositoriesOrError"]["nodes"]
            if repos:
                break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError("repository load")
    repo = repos[0]
    results = []
    for bad in [False, True]:
        params = {
            "selector": {
                "repositoryLocationName": repo["location"]["name"],
                "repositoryName": repo["name"],
                "pipelineName": "quality_job",
            },
            "runConfigData": {"ops": {"validate": {"config": {"bad": bad}}}},
            "mode": "default",
        }
        launched = query(
            "mutation($p: ExecutionParams!){launchRun(executionParams:$p){__typename ... on LaunchRunSuccess{run{runId}} ... on PythonError{message stack}}}",
            {"p": params},
        )
        (work / ("launch-" + str(bad) + ".json")).write_text(json.dumps(launched, indent=2))
        run_id = launched["data"]["launchRun"]["run"]["runId"]
        for _ in range(600):
            result = query(
                "query($id:ID!){runOrError(runId:$id){... on Run{status}}}", {"id": run_id}
            )
            status = result["data"]["runOrError"].get("status")
            if status in ["SUCCESS", "FAILURE", "CANCELED"]:
                break
            time.sleep(1)
        assert status == ("FAILURE" if bad else "SUCCESS"), result
        results.append({"bad": bad, "run_id": run_id, "status": status})
    (work / "results.json").write_text(json.dumps(results, indent=2))
    print(name, "PASSED")
finally:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
