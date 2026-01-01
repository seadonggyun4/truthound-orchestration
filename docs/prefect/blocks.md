---
title: Prefect Blocks
---

# Prefect Blocks

Blocks for storing and reusing data quality engines and configurations.

## DataQualityBlock

Data quality engine Block:

```python
from packages.prefect.blocks import DataQualityBlock

# Create and save Block
block = DataQualityBlock(
    engine_name="truthound",
    auto_schema=True,
)
block.save("my-quality-block")

# Load and use Block
block = DataQualityBlock.load("my-quality-block")
result = block.check(data)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `engine_name` | str | Name of the engine to use |
| `auto_schema` | bool | Automatic schema generation |
| `fail_on_error` | bool | Raise exception on failure |

### Methods

| Method | Description |
|--------|-------------|
| `check(data, ...)` | Data validation |
| `profile(data)` | Data profiling |
| `learn(data)` | Schema learning |
| `get_engine()` | Return engine instance |

## EngineBlock

Engine-specific Block:

```python
from packages.prefect.blocks import EngineBlock, EngineBlockConfig

config = EngineBlockConfig(
    engine_name="truthound",
    parallel=True,
    max_workers=4,
)

block = EngineBlock(config=config)
block.save("my-engine-block")
```

## SLABlock

SLA configuration Block:

```python
from packages.prefect.blocks import SLABlock

block = SLABlock(
    min_pass_rate=0.95,
    max_duration_seconds=3600,
)
block.save("my-sla-block")

# Usage
block = SLABlock.load("my-sla-block")
is_compliant = block.check_compliance(metrics)
```

## Block Configuration

### BlockConfig

```python
from packages.prefect.blocks import BlockConfig

config = BlockConfig(
    engine_name="truthound",
    auto_schema=True,
    fail_on_error=True,
    timeout_seconds=3600,
)

block = DataQualityBlock(config=config)
```

### EngineBlockConfig

```python
from packages.prefect.blocks import EngineBlockConfig

config = EngineBlockConfig(
    engine_name="truthound",
    parallel=True,
    max_workers=4,
    cache_schemas=True,
)
```

## Preset Configurations

```python
from packages.prefect.blocks import (
    DEFAULT_BLOCK_CONFIG,
    STRICT_BLOCK_CONFIG,
    LENIENT_BLOCK_CONFIG,
    PARALLEL_BLOCK_CONFIG,
    PRODUCTION_BLOCK_CONFIG,
    TESTING_BLOCK_CONFIG,
)

block = DataQualityBlock(config=PRODUCTION_BLOCK_CONFIG)
```

## Using Blocks in Flows

```python
from prefect import flow
from packages.prefect.blocks import DataQualityBlock

@flow
def quality_flow():
    block = DataQualityBlock.load("my-quality-block")
    data = load_data()
    result = block.check(data)
    return result
```

## Updating Blocks

```python
block = DataQualityBlock.load("my-quality-block")
block.auto_schema = False
block.save("my-quality-block", overwrite=True)
```

## Deleting Blocks

```python
DataQualityBlock.delete("my-quality-block")
```

## Managing in Prefect UI

Blocks can also be created and managed from the Prefect UI:

1. Prefect UI → Blocks
2. Click "+" button
3. Select "DataQualityBlock"
4. Enter configuration and save
