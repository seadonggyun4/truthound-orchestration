import polars as pl
from dagster import Definitions, OpExecutionContext, job, op

from truthound_dagster import DataQualityResource


@op(required_resource_keys={"quality"}, config_schema={"bad": bool})
def validate(context: OpExecutionContext):
    bad = context.op_config["bad"]
    result = context.resources.quality.check(
        pl.DataFrame({"id": [1, None, 3] if bad else [1, 2, 3]}), fail_on_error=True
    )
    assert result.is_success
    context.add_output_metadata({"core_version": "3.3.0", "is_success": result.is_success})


@job(resource_defs={"quality": DataQualityResource()})
def quality_job():
    validate()


defs = Definitions(jobs=[quality_job])
