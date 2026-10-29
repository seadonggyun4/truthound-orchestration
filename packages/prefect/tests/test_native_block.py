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
