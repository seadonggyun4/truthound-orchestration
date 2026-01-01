---
title: Dagster Resources
---

# Dagster Resources

Provides data quality engines as Dagster resources.

## DataQualityResource

```python
from dagster import Definitions, job, op
from packages.dagster.resources import DataQualityResource

@op(required_resource_keys={"data_quality"})
def check_op(context):
    engine = context.resources.data_quality.get_engine()
    data = load_data()
    result = engine.check(data, auto_schema=True)
    return result

@job(resource_defs={"data_quality": DataQualityResource()})
def quality_job():
    check_op()
```

### Configuration

```python
from packages.dagster.resources import DataQualityResource, EngineResourceConfig

config = EngineResourceConfig(
    engine_name="truthound",
    auto_start=True,
    parallel=True,
    max_workers=4,
)

resource = DataQualityResource(config=config)
```

## EngineResource

Engine-specific resource:

```python
from packages.dagster.resources import EngineResource

resource = EngineResource(engine_name="truthound")

@op(required_resource_keys={"engine"})
def my_op(context):
    engine = context.resources.engine
    result = engine.check(data, auto_schema=True)
```

## Preset Configurations

```python
from packages.dagster.resources import (
    DEFAULT_ENGINE_CONFIG,      # Default settings
    PARALLEL_ENGINE_CONFIG,     # Parallel processing
    PRODUCTION_ENGINE_CONFIG,   # Production optimized
)

resource = DataQualityResource(config=PRODUCTION_ENGINE_CONFIG)
```

## Resource Methods

| Method | Description |
|--------|-------------|
| `get_engine()` | Return engine instance |
| `check(data, ...)` | Data validation |
| `profile(data)` | Data profiling |
| `learn(data)` | Schema learning |

## Lifecycle

Resources are automatically started and stopped during job execution:

```python
@op(required_resource_keys={"data_quality"})
def my_op(context):
    # Resource is already started
    engine = context.resources.data_quality.get_engine()
    # Automatically cleaned up when job ends
```

## Usage in Definitions

```python
from dagster import Definitions
from packages.dagster.resources import DataQualityResource
from packages.dagster.ops import data_quality_check_op

defs = Definitions(
    resources={
        "data_quality": DataQualityResource(
            engine_name="truthound",
        ),
    },
    jobs=[my_quality_job],
)
```
