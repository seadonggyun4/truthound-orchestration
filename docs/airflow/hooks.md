---
title: Airflow Hooks
---

# Airflow Hooks

Hooks for connection management and data loading.

## DataQualityHook

Manages connections with data quality engines.

```python
from packages.airflow.hooks import DataQualityHook

hook = DataQualityHook(conn_id="my_data_connection")

# Data loading
data = hook.load_data("s3://bucket/data.parquet")

# Validation execution
result = hook.check(data, auto_schema=True)

# Profiling
profile = hook.profile(data)

# Schema learning
learn_result = hook.learn(data)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `conn_id` | str | Airflow Connection ID |
| `engine_name` | str | Engine name to use |

### Methods

| Method | Description |
|--------|-------------|
| `load_data(source)` | Load data |
| `check(data, ...)` | Data validation |
| `profile(data)` | Data profiling |
| `learn(data)` | Schema learning |
| `get_engine()` | Return engine instance |

## TruthoundHook

Truthound-specific Hook:

```python
from packages.airflow.hooks import TruthoundHook

hook = TruthoundHook(conn_id="my_connection")

# Use Truthound-specific features
data = hook.load_data("s3://bucket/data.parquet")
result = hook.check(data, auto_schema=True, parallel=True)
```

## ConnectionConfig

Connection configuration class:

```python
from packages.airflow.hooks import ConnectionConfig

config = ConnectionConfig(
    host="localhost",
    port=5432,
    login="user",
    password="pass",
    schema="database",
    extra={"ssl": True},
)
```

## DataLoader

Data loading utility:

```python
from packages.airflow.hooks import DataLoader

loader = DataLoader()

# Load data from various sources
df = loader.load("s3://bucket/data.parquet")
df = loader.load("gs://bucket/data.csv")
df = loader.load("/path/to/local/file.parquet")
```

## DataWriter

Data saving utility:

```python
from packages.airflow.hooks import DataWriter

writer = DataWriter()

# Save data to various destinations
writer.write(df, "s3://bucket/output.parquet")
writer.write(df, "/path/to/local/output.csv")
```

## Using Hooks in Operators

```python
from packages.airflow.operators import BaseDataQualityOperator
from packages.airflow.hooks import DataQualityHook

class MyOperator(BaseDataQualityOperator):
    def __init__(self, conn_id, **kwargs):
        super().__init__(**kwargs)
        self.conn_id = conn_id

    def execute(self, context):
        hook = DataQualityHook(conn_id=self.conn_id)
        data = hook.load_data(self.data_source)
        result = hook.check(data, auto_schema=True)
        return self.serialize_result(result)
```

## Connection Setup

Configure Connection via Airflow UI or CLI:

```bash
airflow connections add 'my_data_connection' \
    --conn-type 's3' \
    --conn-extra '{"aws_access_key_id": "...", "aws_secret_access_key": "..."}'
```

Or via environment variable:

```bash
export AIRFLOW_CONN_MY_DATA_CONNECTION='s3://?aws_access_key_id=...&aws_secret_access_key=...'
```
