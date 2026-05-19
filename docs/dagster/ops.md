---
title: Dagster Ops
---

# Dagster Ops

Prebuilt ops are the right choice when you want Dagster jobs to compose explicit validation stages instead of putting all behavior inside assets.

## Pre-built Ops

### `data_quality_check_op`

Use it for standard validation steps inside jobs and graphs.

### `data_quality_profile_op`

Use it when the job needs descriptive profile output.

### `data_quality_learn_op`

Use it when the job should infer candidate rules from baseline data.

## Op Factories

Use `create_check_op`, `create_profile_op`, and `create_learn_op` when you want preconfigured ops without rewriting boilerplate.

## Op Configuration

`CheckOpConfig`, `ProfileOpConfig`, and `LearnOpConfig` expose operation-level configuration surfaces for explicit job composition.

## Preset Configurations

`STRICT_CHECK_CONFIG` and `LENIENT_CHECK_CONFIG` are useful when teams want named policy modes inside job code.

## Op Chaining

Ops are a strong fit when you want separate stages for:

- load
- validate
- profile
- decide
- notify or persist

## Inputs And Outputs

The op outputs follow the same shared result contract used across the repository, with Dagster-native wrapping where appropriate.

## Depot Pipeline Happy Path

Use the Depot ops when a job needs branch-aware orchestration or scheduled Depot execution while keeping Dagster-native graph structure and metadata emission.

```python
from dagster import job
from truthound_dagster.ops import scheduled_validation_op
from truthound_dagster.resources import DepotResource


@job(resource_defs={"depot": DepotResource(base_url="https://depot.example", api_token="token")})
def validate_users_branch():
    scheduled_validation_op(depot_id="customer-platform", asset_id="users", branch_id="main", wait=True)
```

The op emits a compact shared Depot payload plus Dagster-native metadata. Dagster still owns graph wiring and run metadata, while Depot keeps ownership of approval, release safety, rollback safety, and business state. For the shared flow and failure semantics, see [Depot Pipelines](../depot-pipelines.md).

## Related Reading

- [Resources](resources.md)
- [Assets and Asset Checks](assets.md)
- [Recipes](recipes.md)
- [Depot Pipelines](../depot-pipelines.md)
