"""Real Core failures must survive the Kestra output boundary."""

import polars as pl

from truthound_kestra.scripts.base import CheckScriptConfig
from truthound_kestra.scripts.check import CheckScriptExecutor


def test_core_failure_serializes_to_kestra_output():
    runner = CheckScriptExecutor(config=CheckScriptConfig(fail_on_error=False))
    result = runner.execute(pl.DataFrame({"id": [1, None, 3]}))
    assert not result.is_success
    assert result.failed_count > 0
    payload = result.to_dict()
    assert payload["output"]["failures"]
    assert all(item["rule_type"] for item in payload["output"]["failures"])
