---
title: Dagster Integration
---

# Dagster Integration

Provides Resources, Ops, and Assets for data quality validation within Dagster.

## Installation

```bash
pip install truthound-orchestration[dagster]
```

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| Resources | Engine resource management | [resources.md](resources.md) |
| Ops | Data quality Ops | [ops.md](ops.md) |
| Assets | Quality validation Assets | [assets.md](assets.md) |
| SLA | SLA monitoring | [sla.md](sla.md) |

## Quick Start

```python
from dagster import job, op
from packages.dagster.resources import DataQualityResource
from packages.dagster.ops import data_quality_check_op

@job(resource_defs={"data_quality": DataQualityResource()})
def quality_job():
    data_quality_check_op()
```

## Resources

### DataQualityResource

Provides data quality engines as Dagster resources:

```python
from dagster import Definitions
from packages.dagster.resources import DataQualityResource

defs = Definitions(
    resources={
        "data_quality": DataQualityResource(
            engine_name="truthound",
        ),
    },
)
```

## Ops

### data_quality_check_op

Data validation Op:

```python
from dagster import job, op
from packages.dagster.ops import data_quality_check_op

@job
def my_job():
    data_quality_check_op()
```

### data_quality_profile_op

Profiling Op:

```python
from packages.dagster.ops import data_quality_profile_op

@job
def profile_job():
    data_quality_profile_op()
```

### data_quality_learn_op

Schema learning Op:

```python
from packages.dagster.ops import data_quality_learn_op

@job
def learn_job():
    data_quality_learn_op()
```

## Assets

### @quality_checked_asset

Asset decorator with quality validation:

```python
from packages.dagster.assets import quality_checked_asset

@quality_checked_asset(
    auto_schema=True,
    fail_on_error=True,
)
def my_asset():
    return load_data()
```

### @profiled_asset

Asset with profiling:

```python
from packages.dagster.assets import profiled_asset

@profiled_asset
def profiled_data():
    return load_data()
```

## SLA Monitoring

```python
from packages.dagster.sla import SLAConfig, SLAMonitor, DataQualitySLAHook

config = SLAConfig(
    min_pass_rate=0.95,
    max_duration_seconds=3600,
)

@op(required_resource_keys={"sla_monitor"})
def monitored_op(context):
    result = check_data()
    context.resources.sla_monitor.record(result)
```

## Navigation

- [Resources](resources.md) - Detailed resource usage
- [Ops](ops.md) - Op configuration
- [Assets](assets.md) - Asset utilization
- [SLA](sla.md) - SLA monitoring
