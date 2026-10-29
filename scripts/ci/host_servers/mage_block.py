"""Mage loader that crosses the supported isolated-Core process boundary."""

import json
import os
import subprocess
from importlib import import_module
from typing import Any


if "data_loader" not in globals():
    data_loader = import_module("mage_ai.data_preparation.decorators").data_loader


def load_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
    completed = subprocess.run(
        [os.environ["TRUTHOUND_CORE_PYTHON"], "-I", os.environ["TRUTHOUND_CORE_CHECK"]],
        input=json.dumps(bool(kwargs.get("bad", False))),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode:
        raise RuntimeError("Isolated Core validation failed: " + completed.stderr[-2000:])
    result: dict[str, Any] = json.loads(completed.stdout)
    return result


load_data = data_loader(load_data)
