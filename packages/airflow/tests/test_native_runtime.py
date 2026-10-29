"""Exercise real Airflow imports outside the mock-module test fixtures."""

import os
import subprocess
import sys


def test_native_task_runner_models_initialize(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.metadata
if int(importlib.metadata.version('apache-airflow').split('.')[0]) >= 3:
    import airflow.sdk.execution_time.task_runner
else:
    import airflow.task.task_runner.standard_task_runner
""",
        ],
        env={**os.environ, "AIRFLOW_HOME": str(tmp_path), "AIRFLOW__CORE__LOAD_EXAMPLES": "false"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
