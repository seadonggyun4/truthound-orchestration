---
title: Prefect Integration
---

# Prefect Integration

Provides Blocks, Tasks, and Flows for data quality validation within Prefect.

## Installation

```bash
pip install truthound-orchestration[prefect]
```

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| Blocks | Engine and configuration storage | [blocks.md](blocks.md) |
| Tasks | Data quality tasks | [tasks.md](tasks.md) |
| Flows | Quality flow templates | [flows.md](flows.md) |

## Quick Start

```python
from prefect import flow
from packages.prefect.tasks import data_quality_check_task

@flow
def quality_flow():
    data = load_data()
    result = data_quality_check_task(data, auto_schema=True)
    return result
```

## Blocks

### DataQualityBlock

Data quality engine Block:

```python
from packages.prefect.blocks import DataQualityBlock

block = DataQualityBlock(engine_name="truthound")
block.save("my-quality-block")

# Load later
block = DataQualityBlock.load("my-quality-block")
result = block.check(data, auto_schema=True)
```

## Tasks

### data_quality_check_task

Data validation task:

```python
from packages.prefect.tasks import data_quality_check_task

@flow
def my_flow():
    result = data_quality_check_task(data, auto_schema=True)
```

### data_quality_profile_task

Profiling task:

```python
from packages.prefect.tasks import data_quality_profile_task

@flow
def profile_flow():
    profile = data_quality_profile_task(data)
```

### data_quality_learn_task

Schema learning task:

```python
from packages.prefect.tasks import data_quality_learn_task

@flow
def learn_flow():
    learn_result = data_quality_learn_task(data)
```

## Specialized Tasks

```python
from packages.prefect.tasks import (
    auto_schema_check_task,   # Auto schema validation
    strict_check_task,        # Strict validation
    lenient_check_task,       # Lenient validation
)

@flow
def specialized_flow():
    result = strict_check_task(data)
```

## Flows

Flow templates:

```python
from packages.prefect.flows import FlowConfig, QualityFlowConfig

config = QualityFlowConfig(
    auto_schema=True,
    fail_on_error=True,
)
```

## Navigation

- [Blocks](blocks.md) - Detailed Block usage
- [Tasks](tasks.md) - Task configuration
- [Flows](flows.md) - Flow utilization
