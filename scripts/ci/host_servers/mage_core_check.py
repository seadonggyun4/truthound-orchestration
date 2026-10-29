"""Execute the first-party transformer in the separately installed Core runtime."""

import importlib.metadata
import json
import sys
from importlib import import_module

import polars as pl


mage_adapter = import_module("truthound_mage")
CheckBlockConfig = mage_adapter.CheckBlockConfig
CheckTransformer = mage_adapter.CheckTransformer


assert importlib.metadata.version("truthound") == "3.3.0"
bad = json.loads(sys.stdin.read())
result = CheckTransformer(config=CheckBlockConfig(fail_on_error=True)).execute(
    pl.DataFrame({"id": [1, None, 3] if bad else [1, 2, 3]})
)
print(json.dumps(result.result.to_dict()))
