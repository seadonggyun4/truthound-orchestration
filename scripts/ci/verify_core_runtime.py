"""Verify installed Core 3.3 and optional hosts in an isolated interpreter.

The child owns host imports so this launcher works without every host installed
and cannot accidentally import the repository checkout through sys.path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


SMOKE_PROGRAM = r'''
"""Exercise installed Core 3.3 and host adapters without source-tree imports."""

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import truthound
from truthound.core.results import CheckResult as CoreCheckResult, ValidationRunResult

import common
from common.base import CheckResult, CheckStatus
from common.engines import TruthoundEngine
from common.engines.truthound_mapping import map_validation_run_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "platform", choices=["base", "airflow", "dagster", "prefect", "mage", "kestra"]
    )
    parser.add_argument(
        "--advanced", action="store_true", help="Exercise optional NumPy/SciPy ML paths"
    )
    args = parser.parse_args()
    assert truthound.__version__ == "3.3.0"
    assert "site-packages" in truthound.__file__

    assert "site-packages" in common.__file__, common.__file__
    good = pl.DataFrame({"id": [1, 2, 3], "amount": [10, 20, 30]})
    bad = pl.DataFrame({"id": [1, None, 3], "amount": [10, 20, 30]})
    engine = TruthoundEngine()
    assert engine.check(good).status == CheckStatus.PASSED
    assert engine.check(bad).status == CheckStatus.FAILED
    assert engine.profile(good).row_count == 3
    assert engine.learn(good).columns_analyzed == 2
    run = ValidationRunResult(
        suite_name="summary",
        source="synthetic",
        row_count=3,
        column_count=2,
        checks=(CoreCheckResult(name="required", success=False, issue_count=1),),
    )
    mapped = map_validation_run_result(
        ValidationRunResult.from_dict(run.to_dict()), engine_name="truthound", execution_time_ms=0
    )
    assert not CheckResult.from_dict(mapped.to_dict()).is_success
    if args.advanced:
        baseline = pl.DataFrame({"x": list(range(100))})
        shifted = pl.DataFrame({"x": list(range(200, 300))})
        assert engine.detect_drift(baseline, shifted, method="ks").is_drifted
        assert not engine.detect_drift(baseline, baseline, method="ks").is_drifted
        outlier = pl.DataFrame({"x": [float(i % 10) for i in range(200)] + [10000.0]})
        for detector in ("isolation_forest", "z_score", "ensemble"):
            result = engine.detect_anomalies(outlier, detector=detector, columns=["x"])
            assert result.has_anomalies
            assert result.total_row_count == 201
            assert any(score.metadata["row_index"] == 200 for score in result.anomalies)
    if args.platform == "dagster":
        from dagster import OpExecutionContext, job, op

        from truthound_dagster import DataQualityResource

        @op(required_resource_keys={"quality"})
        def verify(context: OpExecutionContext) -> None:
            assert context.resources.quality.check(good).is_success
            result = context.resources.quality.check(bad, fail_on_error=False)
            assert not result.is_success

        @job(resource_defs={"quality": DataQualityResource()})
        def verify_job() -> None:
            verify()

        assert verify_job.execute_in_process().success
    elif args.platform == "prefect":
        from truthound_prefect.blocks.engine import DataQualityBlock

        prefect_block = DataQualityBlock(fail_on_error=False)
        assert prefect_block.check(good)["is_success"]
        assert not prefect_block.check(bad)["is_success"]
        assert prefect_block.profile(good)["row_count"] == 3
    elif args.platform == "mage":
        from truthound_mage import CheckBlockConfig, CheckTransformer

        mage_block = CheckTransformer(config=CheckBlockConfig(fail_on_error=False))
        assert mage_block.execute(good).result.is_success
        assert not mage_block.execute(bad).result.is_success
    elif args.platform == "kestra":
        from truthound_kestra.scripts.base import CheckScriptConfig
        from truthound_kestra.scripts.check import CheckScriptExecutor

        runner = CheckScriptExecutor(config=CheckScriptConfig(fail_on_error=False))
        assert runner.execute(good).is_success
        assert not runner.execute(bad).is_success
    elif args.platform == "airflow":
        from truthound_airflow import DataQualityCheckOperator

        class XComCapture:
            def xcom_push(self, *, key: str, value: dict[str, Any]) -> None:
                self.value = value

        with tempfile.TemporaryDirectory() as tmp:
            for frame, expected in [(good, True), (bad, False)]:
                path = Path(tmp) / "input.csv"
                frame.write_csv(path)
                sink = XComCapture()
                operator = DataQualityCheckOperator(
                    task_id="quality", data_path=str(path), fail_on_error=False
                )
                result = operator.execute({"ti": sink})
                assert result["is_success"] is expected
                assert sink.value["is_success"] is expected
    print(
        json.dumps(
            {
                "platform": args.platform,
                "core": importlib.metadata.version("truthound"),
                "core_path": truthound.__file__,
                "orchestration_path": common.__file__,
                "status": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=["base", "airflow", "dagster", "prefect", "mage", "kestra"])
    parser.add_argument("--advanced", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, "-I", "-c", SMOKE_PROGRAM, args.platform]
    if args.advanced:
        command.append("--advanced")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
