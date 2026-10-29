import polars as pl
from prefect import flow, task

from truthound_prefect.blocks.engine import DataQualityBlock


@task
def validate(bad: bool):
    result = DataQualityBlock(fail_on_error=True).check(
        pl.DataFrame({"id": [1, None, 3] if bad else [1, 2, 3]})
    )
    assert result["is_success"]
    return result


@flow(name="truthound-core33-server")
def quality_flow(bad: bool = False):
    return validate(bad)


if __name__ == "__main__":
    quality_flow.serve(name="verification", pause_on_shutdown=False)
