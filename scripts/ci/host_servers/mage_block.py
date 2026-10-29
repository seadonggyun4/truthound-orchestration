"""Mage loader that crosses the supported isolated-Core process boundary."""

import json
import os
import subprocess


if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def load_data(*args, **kwargs):
    completed = subprocess.run(
        [os.environ["TRUTHOUND_CORE_PYTHON"], "-I", os.environ["TRUTHOUND_CORE_CHECK"]],
        input=json.dumps(bool(kwargs.get("bad", False))),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode:
        raise RuntimeError("Isolated Core validation failed: " + completed.stderr[-2000:])
    return json.loads(completed.stdout)
