---
title: Prefect Tasks
---

# Prefect Tasks

Tasks are the easiest Prefect execution surface to adopt. Use them when a flow already exists and you want to insert validation without committing to persisted block configuration yet.

## Pre-built Tasks

### `data_quality_check_task`

Use it for standard validation inside a flow.

### `data_quality_profile_task`

Use it when the flow should profile a dataset.

### `data_quality_learn_task`

Use it to learn candidate rules from baseline data.

## Specialized Tasks

Other useful helpers include:

- strict and lenient check presets
- auto-schema check helpers
- profile presets
- streaming tasks

## Task Factories

Use `create_check_task`, `create_profile_task`, and related factory helpers when you want reusable task construction without rewriting shared logic.

## Task Configuration

`CheckTaskConfig`, `ProfileTaskConfig`, and `LearnTaskConfig` expose task-level policy choices while keeping task wiring consistent.

## Task Chaining

Prefect tasks compose well when you separate load, validate, profile, notify, and persist steps.

## Using With Blocks

Pass a `DataQualityBlock` when the flow should consume saved configuration instead of ephemeral defaults.

## Retry Configuration

Keep business logic retries and platform retries understandable. Do not hide configuration mistakes behind excessive retry settings.

## Depot Pipeline Happy Path

Use the Depot tasks when a Prefect flow should submit or wait on branch-aware execution without rebuilding the shared Depot contract in flow code.

```python
from prefect import flow
from truthound_prefect.blocks.depot import DepotBlock
from truthound_prefect.tasks.depot import scheduled_validation_task


@flow
def validate_users_branch() -> dict:
    depot = DepotBlock(base_url="https://depot.example", api_token="token")
    return scheduled_validation_task(
        depot_id="customer-platform",
        asset_id="users",
        branch_id="main",
        wait=True,
        depot_block=depot,
    )
```

The task returns the shared compact Depot payload. Prefect still owns task retries and flow composition, while Depot keeps ownership of approval, release safety, rollback safety, and business state. For the shared contract behind the returned payload, see [Depot Pipelines](../depot-pipelines.md).

## Related Reading

- [Blocks](blocks.md)
- [Flows](flows.md)
- [Deployment Patterns](deployment-patterns.md)
- [Depot Pipelines](../depot-pipelines.md)
