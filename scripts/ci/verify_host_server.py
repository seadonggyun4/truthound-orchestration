"""Run real host services and assert persisted success and quality-failure states.

Use a disposable, previously nonexistent output directory. The child owns only
its newly started processes; logs and database evidence remain after shutdown.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import socket
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=["airflow", "dagster", "prefect"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = Path(__file__).with_name("host_servers")
    for fixture in ["prefect_flow.py", "dagster_job.py"]:
        (output / fixture).write_text((fixtures / fixture).read_text())
    if args.port is None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    else:
        port = args.port
    package = "apache-airflow" if args.platform == "airflow" else args.platform
    metadata = {
        "platform": args.platform,
        "host_version": importlib.metadata.version(package),
        "core_version": importlib.metadata.version("truthound"),
        "python": sys.version,
        "status": "started",
    }
    assert metadata["core_version"] == "3.3.0", metadata
    (output / "verification.json").write_text(json.dumps(metadata, indent=2))
    result = subprocess.run(
        [sys.executable, "-I", str(fixtures / (args.platform + ".py")), str(output), str(port)],
        check=False,
    )
    metadata["status"] = "passed" if result.returncode == 0 else "failed"
    (output / "verification.json").write_text(json.dumps(metadata, indent=2))
    result.check_returncode()


if __name__ == "__main__":
    main()
