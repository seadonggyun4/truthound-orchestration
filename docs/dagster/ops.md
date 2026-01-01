---
title: Dagster Ops
---

# Dagster Ops

Ops for data quality validation, profiling, and schema learning.

## Pre-built Ops

### data_quality_check_op

Data validation Op:

```python
from dagster import job
from packages.dagster.ops import data_quality_check_op

@job(resource_defs={"data_quality": DataQualityResource()})
def check_job():
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

## Op Factories

### create_check_op

Create custom validation Op:

```python
from packages.dagster.ops import create_check_op, CheckOpConfig

config = CheckOpConfig(
    auto_schema=True,
    fail_on_error=True,
)

my_check_op = create_check_op(
    name="my_check",
    config=config,
)

@job
def my_job():
    my_check_op()
```

### create_profile_op

Create custom profiling Op:

```python
from packages.dagster.ops import create_profile_op, ProfileOpConfig

config = ProfileOpConfig(
    include_statistics=True,
)

my_profile_op = create_profile_op(
    name="my_profile",
    config=config,
)
```

### create_learn_op

Create custom learning Op:

```python
from packages.dagster.ops import create_learn_op, LearnOpConfig

config = LearnOpConfig(
    min_confidence=0.8,
)

my_learn_op = create_learn_op(
    name="my_learn",
    config=config,
)
```

## Op Configuration

### CheckOpConfig

```python
from packages.dagster.ops import CheckOpConfig

config = CheckOpConfig(
    auto_schema=True,
    rules=[
        {"type": "not_null", "column": "id"},
    ],
    fail_on_error=True,
    timeout_seconds=3600,
)
```

### ProfileOpConfig

```python
from packages.dagster.ops import ProfileOpConfig

config = ProfileOpConfig(
    include_statistics=True,
)
```

### LearnOpConfig

```python
from packages.dagster.ops import LearnOpConfig

config = LearnOpConfig(
    min_confidence=0.8,
)
```

## Preset Configurations

```python
from packages.dagster.ops import (
    STRICT_CHECK_CONFIG,    # Strict validation
    LENIENT_CHECK_CONFIG,   # Lenient validation
)

strict_op = create_check_op("strict", config=STRICT_CHECK_CONFIG)
lenient_op = create_check_op("lenient", config=LENIENT_CHECK_CONFIG)
```

## Op Chaining

```python
from dagster import job, op
from packages.dagster.ops import data_quality_check_op, data_quality_profile_op

@op
def load_data():
    return pl.read_parquet("data.parquet")

@op
def process_result(check_result, profile_result):
    if check_result.status.name == "FAILED":
        raise Exception("Quality check failed")
    return {"check": check_result, "profile": profile_result}

@job
def quality_pipeline():
    data = load_data()
    check = data_quality_check_op(data)
    profile = data_quality_profile_op(data)
    process_result(check, profile)
```

## Inputs and Outputs

Ops use `In` and `Out` to exchange data:

```python
from dagster import In, Out, op
from packages.dagster.ops import data_quality_check_op

# check_op receives DataFrame as input and outputs CheckResult
@job
def my_job():
    data = load_data()
    result = data_quality_check_op(data)
    # result is of type CheckResult
```
