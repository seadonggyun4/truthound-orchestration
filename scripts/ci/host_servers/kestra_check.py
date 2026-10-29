"""Run a synthetic Core check in a real Kestra process task."""

import sys
from importlib import import_module

import polars as pl


adapter = import_module("truthound_kestra.scripts.check")
config = import_module("truthound_kestra.scripts.base").CheckScriptConfig
sdk = import_module("kestra").Kestra
bad = sys.argv[1] == "bad"
result = adapter.CheckScriptExecutor(config=config(fail_on_error=False)).execute(
    pl.DataFrame({"id": [1, None, 3] if bad else [1, 2, 3]})
)
sdk.outputs({"is_success": result.is_success, "failed_count": len(result.failures)})
if not result.is_success:
    raise RuntimeError("Expected quality failure propagated to Kestra execution")
