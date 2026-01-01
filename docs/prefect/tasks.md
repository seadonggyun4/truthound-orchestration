---
title: Prefect Tasks
---

# Prefect Tasks

Tasks for data quality validation, profiling, and schema learning.

## Pre-built Tasks

### data_quality_check_task

Data validation task:

```python
from prefect import flow
from packages.prefect.tasks import data_quality_check_task

@flow
def my_flow():
    data = load_data()
    result = data_quality_check_task(data, auto_schema=True)
    return result
```

### data_quality_profile_task

Profiling task:

```python
from packages.prefect.tasks import data_quality_profile_task

@flow
def profile_flow():
    data = load_data()
    profile = data_quality_profile_task(data)
    return profile
```

### data_quality_learn_task

Schema learning task:

```python
from packages.prefect.tasks import data_quality_learn_task

@flow
def learn_flow():
    data = load_data()
    learn_result = data_quality_learn_task(data)
    return learn_result
```

## Specialized Tasks

```python
from packages.prefect.tasks import (
    auto_schema_check_task,   # Auto schema validation
    strict_check_task,        # Strict validation (fail_on_error=True)
    lenient_check_task,       # Lenient validation (fail_on_error=False)
)

@flow
def specialized_flow():
    data = load_data()

    # Strict validation - raises exception on failure
    strict_result = strict_check_task(data)

    # Lenient validation - continues on failure
    lenient_result = lenient_check_task(data)
```

## Task Factories

### create_check_task

Create custom validation tasks:

```python
from packages.prefect.tasks import create_check_task, CheckTaskConfig

config = CheckTaskConfig(
    auto_schema=True,
    fail_on_error=True,
    engine_name="truthound",
)

my_check_task = create_check_task(
    name="my_check",
    config=config,
)

@flow
def my_flow():
    result = my_check_task(data)
```

### create_profile_task

Create custom profiling tasks:

```python
from packages.prefect.tasks import create_profile_task, ProfileTaskConfig

config = ProfileTaskConfig(
    include_statistics=True,
)

my_profile_task = create_profile_task(
    name="my_profile",
    config=config,
)
```

### create_learn_task

Create custom learning tasks:

```python
from packages.prefect.tasks import create_learn_task, LearnTaskConfig

config = LearnTaskConfig(
    min_confidence=0.8,
)

my_learn_task = create_learn_task(
    name="my_learn",
    config=config,
)
```

## Task Configuration

### CheckTaskConfig

```python
from packages.prefect.tasks import CheckTaskConfig

config = CheckTaskConfig(
    auto_schema=True,
    rules=[
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
    ],
    fail_on_error=True,
    engine_name="truthound",
    timeout_seconds=3600,
)
```

### ProfileTaskConfig

```python
from packages.prefect.tasks import ProfileTaskConfig

config = ProfileTaskConfig(
    include_statistics=True,
)
```

### LearnTaskConfig

```python
from packages.prefect.tasks import LearnTaskConfig

config = LearnTaskConfig(
    min_confidence=0.8,
)
```

## Task Chaining

```python
from prefect import flow, task
from packages.prefect.tasks import (
    data_quality_check_task,
    data_quality_profile_task,
)

@task
def load_data():
    return pl.read_parquet("data.parquet")

@task
def process_results(check_result, profile_result):
    if check_result.status.name == "FAILED":
        raise Exception("Quality check failed")
    return {"check": check_result, "profile": profile_result}

@flow
def quality_pipeline():
    data = load_data()
    check = data_quality_check_task(data, auto_schema=True)
    profile = data_quality_profile_task(data)
    result = process_results(check, profile)
    return result
```

## Using with Blocks

```python
from prefect import flow
from packages.prefect.blocks import DataQualityBlock
from packages.prefect.tasks import data_quality_check_task

@flow
def flow_with_block():
    block = DataQualityBlock.load("my-quality-block")
    data = load_data()

    # Use Block configuration
    result = data_quality_check_task(
        data,
        auto_schema=block.auto_schema,
        engine_name=block.engine_name,
    )
    return result
```

## Retry Configuration

```python
from prefect import flow
from packages.prefect.tasks import data_quality_check_task

@flow
def flow_with_retry():
    # Task uses Prefect's retry mechanism
    result = data_quality_check_task.with_options(
        retries=3,
        retry_delay_seconds=60,
    )(data, auto_schema=True)
    return result
```
