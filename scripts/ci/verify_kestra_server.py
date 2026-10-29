"""Verify real Kestra execution and DNS deny-list enforcement using synthetic data.

The output directory must not exist. Credentials/configuration remain private;
only results.json is suitable for CI upload. Requires the released server jar
and Java 25. No existing service, database or network configuration is changed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--server-jar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    server_sha256 = hashlib.sha256(args.server_jar.read_bytes()).hexdigest()
    assert importlib.metadata.version("truthound") == "3.3.0"
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "home", "storage"):
        (work / name).mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    hits: list[str] = []

    class Canary(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"synthetic-deny-list-canary")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    canary = HTTPServer(("127.0.0.1", 0), Canary)
    thread = threading.Thread(target=canary.serve_forever, daemon=True)
    thread.start()
    password = secrets.token_urlsafe(32)
    authorization = (
        "Basic " + base64.b64encode(("verification@example.invalid:" + password).encode()).decode()
    )
    config = work / "application.yml"
    config.write_text(f"""micronaut:
  server:
    host: 127.0.0.1
    port: {port}
kestra:
  server:
    basic-auth:
      username: verification@example.invalid
      password: {json.dumps(password)}
  anonymous-usage-report:
    enabled: false
  storage:
    type: local
    local:
      base-path: {json.dumps(str(work / "storage"))}
  tasks:
    tmp-dir:
      path: {json.dumps(str(work / "tmp"))}
    http:
      denied-list:
        - 127.0.0.0/8
        - "::1/128"
""")
    config.chmod(0o600)
    base = f"http://127.0.0.1:{port}/api/v1"

    def request(
        path: str, payload: bytes | None = None, content_type: str = "application/json"
    ) -> Any:
        req = urllib.request.Request(
            base + path,
            data=payload,
            headers={"Authorization": authorization, "Content-Type": content_type},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    process: subprocess.Popen[bytes] | None = None
    try:
        with (work / "server.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    str(args.java.absolute()),
                    "-Xmx768m",
                    f"-Djava.io.tmpdir={work / 'tmp'}",
                    f"-Duser.home={work / 'home'}",
                    "-jar",
                    str(args.server_jar.resolve()),
                    "server",
                    "local",
                    f"--port={port}",
                    "--worker-thread=2",
                    "--no-tutorials",
                    "--config",
                    str(config),
                ],
                cwd=work,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError("Kestra server exited; inspect private server.log")
                try:
                    request("/configs")
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("Kestra API startup")
            try:
                urllib.request.urlopen(base + "/main/flows/search", timeout=10)
            except urllib.error.HTTPError as exc:
                assert exc.code in (401, 403), exc.code
            else:
                raise AssertionError("Unauthenticated flow search was accepted")
            results = []
            fixture = Path(__file__).with_name("host_servers") / "kestra_check.py"
            for kind in ("good", "bad", "denied_ip", "denied_dns"):
                flow = f"id: verification_{kind}\nnamespace: verification.security\ntasks:\n"
                if kind in ("good", "bad"):
                    command = shlex.join([sys.executable, "-I", str(fixture.resolve()), kind])
                    flow += (
                        "  - id: quality\n    type: io.kestra.plugin.scripts.shell.Commands\n"
                        "    taskRunner:\n      type: io.kestra.plugin.core.runner.Process\n"
                        f"    commands:\n      - {json.dumps(command)}\n"
                    )
                else:
                    host = "127.0.0.1" if kind == "denied_ip" else "localhost"
                    flow += (
                        "  - id: denied\n    type: io.kestra.plugin.core.http.Download\n"
                        f"    uri: http://{host}:{canary.server_port}/{kind}\n"
                    )
                request("/main/flows", flow.encode(), "application/x-yaml")
                result = request(
                    f"/main/executions/verification.security/verification_{kind}",
                    b"",
                    "multipart/form-data; boundary=verification",
                )
                run_id = result["id"]
                for _ in range(600):
                    result = request("/main/executions/" + run_id)
                    state = result["state"]["current"]
                    if state in ("SUCCESS", "FAILED", "KILLED", "CANCELLED", "WARNING"):
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError(f"Kestra execution {kind}")
                expected = "SUCCESS" if kind == "good" else "FAILED"
                results.append({"kind": kind, "state": state, "expected": expected})
                (work / "results.json").write_text(
                    json.dumps({"runs": results, "canary_hits": hits}, indent=2)
                )
                assert state == expected, results[-1]
            assert not hits, hits
            (work / "results.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "runs": results,
                        "canary_hits": hits,
                        "server_sha256": server_sha256,
                        "sdk": importlib.metadata.version("kestra"),
                        "core": importlib.metadata.version("truthound"),
                        "python": sys.version,
                    },
                    indent=2,
                )
            )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        canary.shutdown()
        canary.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
