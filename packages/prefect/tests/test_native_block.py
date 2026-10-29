"""Run the real Prefect model outside the package's mock-module fixtures."""

import subprocess
import sys


def test_native_block_runs_core_and_preserves_quality_failure(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import polars as pl
from truthound_prefect.blocks.engine import DataQualityBlock
block = DataQualityBlock(fail_on_error=False)
good = pl.DataFrame({'id': [1, 2, 3]})
bad = pl.DataFrame({'id': [1, None, 3]})
assert block.check(good)['is_success']
assert not block.check(bad)['is_success']
assert block.profile(good)['row_count'] == 3
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_server_application_initializes(tmp_path):
    """Catch server-only dependency and FastAPI incompatibilities hidden by Block tests."""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from prefect.server.api.server import create_app; "
            "app = create_app(); assert app is not None",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_server_fixture_entrypoint_loads(tmp_path):
    """Prefect 2 must retrieve the decorated flow through its persisted name."""
    from pathlib import Path

    fixture = Path(__file__).resolve().parents[3] / "scripts/ci/host_servers/prefect_flow.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; from prefect.flows import load_flow_from_entrypoint; "
            "flow = load_flow_from_entrypoint(sys.argv[1]); "
            "assert flow.name == 'truthound-core33-server'",
            str(fixture) + ":quality_flow",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
