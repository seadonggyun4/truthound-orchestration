from importlib import import_module
from typing import Any

import polars as pl
from prefect import flow, task


DataQualityBlock = import_module("truthound_prefect.blocks.engine").DataQualityBlock


def validate(bad: bool) -> dict[str, Any]:
    result: dict[str, Any] = DataQualityBlock(fail_on_error=True).check(
        pl.DataFrame({"id": [1, None, 3] if bad else [1, 2, 3]})
    )
    assert result["is_success"]
    return result


validate_task = task(validate)


def quality_flow(bad: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = validate_task(bad)
    return result


served_flow = flow(name="truthound-core33-server")(quality_flow)
# Prefect 2 persists the original function name as its import entrypoint.
quality_flow = served_flow


if __name__ == "__main__":
    served_flow.serve(name="verification", pause_on_shutdown=False)
