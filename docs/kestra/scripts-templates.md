---
title: Kestra Scripts and Flow Templates
---

# Kestra Scripts and Flow Templates

Truthound supports Kestra in two complementary ways:

- script entry points for Python script tasks
- flow generators for teams that want reusable YAML templates

## Script Entry Points

The main script entry points are:

- `check_quality_script`
- `stream_quality_script`
- `profile_data_script`
- `learn_schema_script`

The package also exposes drift and anomaly helpers through the script module.

## Why Scripts Work Well In Kestra

- the flow stays readable in YAML
- execution logic can evolve without duplicating long inline scripts
- output handling and serialization stay consistent across flows

## Flow Generators

The flow module exposes:

- `generate_flow_yaml`
- `generate_check_flow`
- `generate_profile_flow`
- `generate_learn_flow`
- `generate_quality_pipeline`

These are useful when platform teams want a standardized starter flow for multiple
domains or environments.

## Depot Pipeline Happy Path

Use the Depot script and template surface when Kestra should keep YAML as the orchestration entrypoint but still emit the shared Depot payload contract.

```python
from truthound_kestra.flows.depot import generate_depot_scheduled_validate_flow

flow_yaml = generate_depot_scheduled_validate_flow(
    flow_id="scheduled-validate-users",
    namespace="company.data",
    depot_id="customer-platform",
    asset_id="users",
    schedule="0 6 * * *",
)
```

The generated flow calls the shared Depot script entrypoint and preserves the compact shared Depot payload in Kestra output handling. Kestra still owns YAML flow composition, while Depot keeps ownership of approval, release safety, rollback safety, and business state. For the shared execution contract, see [Depot Pipelines](../depot-pipelines.md).

## Production Guidance

- use script entry points for task-level execution
- use flow generators when you want a repeatable scaffold, not when every flow is
  highly bespoke
- keep environment-specific values in Kestra variables or secrets, not inside the
  generated template itself

## Related Pages

- [Recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
- [Depot Pipelines](../depot-pipelines.md)
