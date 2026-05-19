---
title: Mage Recipes
---

# Mage Recipes

This page collects a few high-signal patterns that map well to the current package
surface.

## Validate A Dataset Before A Load

```python
from truthound_mage import CheckBlockConfig, CheckTransformer


def transform(df, *args, **kwargs):
    result = CheckTransformer(
        config=CheckBlockConfig(
            rules=[
                {"column": "id", "check": "not_null"},
                {"column": "email", "check": "email_format"},
            ]
        )
    ).execute(df)
    return result.result_dict
```

Use this pattern when a pipeline should fail before data is written downstream.

## Generate A Profile For Operator Review

Use `ProfileTransformer` for baselining, onboarding, and change review.

This is especially useful for pipelines that ingest third-party feeds whose quality
shape changes over time.

## Gate Downstream Work With A Sensor

Use `QualityGateSensor` or `DataQualitySensor` when downstream blocks should run only
after a check result meets pass-rate or failure-rate thresholds.

This keeps quality gate semantics explicit instead of scattering them across block code.

## Route Hard Failures And Soft Warnings Differently

Combine:

- strict `CheckTransformer` configs for blocking datasets
- condition or sensor blocks for soft rollout paths
- SLA hooks for operator visibility

## Keep Shared Config In One Place

Create one helper module that returns prebuilt `CheckBlockConfig`,
`ProfileBlockConfig`, and `SensorBlockConfig` objects. This prevents configuration
drift across pipelines.

## Depot Pipeline Happy Path

Use the Depot block helpers when a Mage pipeline needs shared scheduled sync or branch validation behavior without hiding execution state inside ad hoc block code.

```python
from truthound_mage.blocks.depot import scheduled_sync
from truthound_mage.blocks.base import BlockExecutionContext


def transform(*args, **kwargs):
    return scheduled_sync(
        depot_id="customer-platform",
        asset_id="users",
        context=BlockExecutionContext(block_uuid="sync-users", pipeline_uuid="users-pipeline"),
    )
```

The block returns the shared compact Depot flow payload. Mage still owns block wiring and pipeline-local execution, while Depot keeps ownership of approval, release safety, rollback safety, and business state. For the shared status and failure semantics, see [Depot Pipelines](../depot-pipelines.md).

## Related Pages

- [Project Layout](project-layout.md)
- [`io_config.yaml`](io-config.md)
- [Troubleshooting](troubleshooting.md)
- [Depot Pipelines](../depot-pipelines.md)
