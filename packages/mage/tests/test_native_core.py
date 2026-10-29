"""Exercise real Core outside mocks and the Mage host interpreter."""

import subprocess
import sys


def test_native_core_default_parallel_check_and_quality_failure(tmp_path):
    """A minimal companion install must supply Core's NumPy runtime dependency."""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import polars as pl
from truthound_mage import CheckBlockConfig, CheckTransformer
transformer = CheckTransformer(config=CheckBlockConfig(fail_on_error=False))
good = transformer.execute(pl.DataFrame({"id": [1, 2, 3]}))
bad = transformer.execute(pl.DataFrame({"id": [1, None, 3]}))
assert good.result.is_success
assert not bad.result.is_success
assert bad.result.failed_count > 0
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
