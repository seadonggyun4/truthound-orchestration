from importlib import import_module

import polars as pl
from dagster import Definitions, OpExecutionContext, job, op


DataQualityResource = import_module("truthound_dagster").DataQualityResource


def validate(context: OpExecutionContext) -> None:
    bad = context.op_config["bad"]
    result = context.resources.quality.check(
        pl.DataFrame({"id": [1, None, 3] if bad else [1, 2, 3]}), fail_on_error=True
    )
    assert result.is_success
    context.add_output_metadata({"core_version": "3.3.0", "is_success": result.is_success})


validate_op = op(required_resource_keys={"quality"}, config_schema={"bad": bool})(validate)


def quality_job() -> None:
    validate_op()


quality_job_definition = job(resource_defs={"quality": DataQualityResource()})(quality_job)


defs = Definitions(jobs=[quality_job_definition])
