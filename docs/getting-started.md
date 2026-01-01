---
title: Getting Started
---

# Getting Started

## Installation

### Single Package with Optional Dependencies

The framework employs a single-package architecture with optional dependencies to optimize installation size and minimize unnecessary dependencies.

```bash
# Core installation (common module + engine adapters)
pip install truthound-orchestration

# Platform-specific installation
pip install truthound-orchestration[airflow]
pip install truthound-orchestration[dagster]
pip install truthound-orchestration[prefect]

# Multiple platforms
pip install truthound-orchestration[airflow,dagster]

# Complete installation (development)
pip install truthound-orchestration[all]
```

### Engine Installation

Data quality engines require separate installation:

```bash
pip install truthound              # Default recommendation
pip install great-expectations     # For Great Expectations usage
pip install pandera                # For Pandera usage
```

## Requirements

- Python 3.11 or higher
- Polars (for data processing)

## Quick Start

### 1. Basic Engine Usage

```python
from common.engines import TruthoundEngine
import polars as pl

# Create engine instance
engine = TruthoundEngine()

# Context manager usage (recommended)
with engine:
    # Load data
    df = pl.read_csv("data.csv")

    # Learn schema from data
    schema = engine.get_schema(df)

    # Validate data against schema
    result = engine.check(df, schema=schema)

    print(f"Status: {result.status.name}")
    print(f"Passed: {result.passed_count}")
    print(f"Failed: {result.failed_count}")
```

### 2. Apache Airflow Integration

```python
from packages.airflow.operators import DataQualityCheckOperator

check_task = DataQualityCheckOperator(
    task_id="data_quality_check",
    data_source="s3://bucket/data.parquet",
    auto_schema=True,
)
```

### 3. Dagster Integration

```python
from packages.dagster.resources import DataQualityResource
from packages.dagster.ops import data_quality_check_op

@job
def quality_job():
    data_quality_check_op()
```

### 4. Prefect Integration

```python
from packages.prefect.blocks import DataQualityBlock
from packages.prefect.tasks import data_quality_check_task

@flow
def quality_flow():
    result = data_quality_check_task(data)
    return result
```

## Project Structure

```
truthound-orchestration/
├── common/                    # Shared module
│   ├── base.py               # Protocol, Config, Result types
│   ├── logging.py            # Structured logging
│   ├── retry.py              # Retry decorators
│   ├── circuit_breaker.py    # Circuit breaker pattern
│   ├── health.py             # Health check system
│   ├── metrics.py            # Metrics collection
│   ├── rate_limiter.py       # Rate limiting
│   ├── cache.py              # Caching utilities
│   └── engines/              # Engine implementations
├── packages/
│   ├── airflow/              # Airflow integration
│   ├── dagster/              # Dagster integration
│   ├── prefect/              # Prefect integration
│   ├── dbt/                  # dbt integration
│   └── enterprise/           # Enterprise features
└── tests/                    # Test suite
```

### 5. dbt Integration

```yaml
# models/schema.yml
version: 2

models:
  - name: stg_customers
    tests:
      - truthound_check:
          rules:
            - column: customer_id
              type: not_null
            - column: customer_id
              type: unique
            - column: email
              type: regex
              pattern: "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"
```

## Next Steps

- [Common Module](common/index.md) - Logging, retry, metrics, and more
- [Engines](engines/index.md) - Truthound, Great Expectations, Pandera
- [Airflow Integration](airflow/index.md) - Operators, Sensors, Hooks
- [Dagster Integration](dagster/index.md) - Resources, Ops, Assets
- [Prefect Integration](prefect/index.md) - Blocks, Tasks, Flows
- [dbt Integration](dbt/index.md) - Generic Tests, SQL Macros
- [Enterprise Features](enterprise/index.md) - Multi-tenancy, secrets, notifications
- [API Reference](api-reference/index.md) - Complete API documentation
