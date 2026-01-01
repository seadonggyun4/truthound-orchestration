---
title: TruthoundEngine
---

# TruthoundEngine

Truthound serves as the default data quality engine, employing **schema-based validation** methodology.

## Basic Usage

```python
from common.engines import TruthoundEngine
import polars as pl

engine = TruthoundEngine()

# Context manager usage (recommended)
with engine:
    df = pl.read_csv("data.csv")

    # Schema learning
    schema = engine.get_schema(df)

    # Validation execution
    result = engine.check(df, schema=schema)

    print(f"Status: {result.status.name}")
    print(f"Passed: {result.passed_count}")
    print(f"Failed: {result.failed_count}")
```

## Schema-Based Validation

Truthound employs schema-based validation rather than conventional rules-based approaches:

```python
# Step 1: Learn schema from baseline data
schema = engine.get_schema(baseline_df)

# Step 2: Validate new data against learned schema
result = engine.check(new_df, schema=schema)

# Alternatively, use auto_schema (automatic schema generation from data)
result = engine.check(df, auto_schema=True)
```

## Truthound-Specific Parameters

```python
result = engine.check(
    data=df,
    schema=learned_schema,      # Schema object obtained from learn()
    auto_schema=True,           # Automatic schema generation from data
    parallel=True,              # Enable parallel validation
    max_workers=4,              # Maximum parallel worker count
    min_severity="medium",      # Minimum severity level to report
)
```

## Profiling

```python
profile = engine.profile(df)

for col in profile.columns:
    print(f"{col.column_name}: {col.dtype}")
    print(f"  Null: {col.null_percentage}%")
    print(f"  Unique: {col.unique_count}")
```

## Schema Learning

```python
learn_result = engine.learn(df)

for rule in learn_result.rules:
    print(f"{rule.column}: {rule.rule_type} (confidence={rule.confidence})")
```

## Engine Information

```python
info = engine.get_info()
print(f"Engine: {info.name} v{info.version}")

caps = engine.get_capabilities()
print(f"Streaming support: {caps.supports_streaming}")
```

## Severity Mapping

| Truthound Severity | Framework Severity |
|--------------------|--------------------|
| `critical` | `Severity.CRITICAL` |
| `high` | `Severity.ERROR` |
| `medium` | `Severity.WARNING` |
| `low` | `Severity.INFO` |

## Supported Data Types

| Data Type | Support |
|-----------|---------|
| Polars DataFrame | Native |
| Pandas DataFrame | Auto-conversion |
| CSV file path | Direct support |
| Parquet file path | Direct support |
| SQL connection | Direct support |

## Lifecycle Management

```python
from common.engines import TruthoundEngine

engine = TruthoundEngine()

# Explicit lifecycle management
engine.start()
try:
    result = engine.check(data)
    health = engine.health_check()
    print(f"Health: {health.status.name}")
finally:
    engine.stop()

# Or automatic management via context manager
with TruthoundEngine() as engine:
    result = engine.check(data, auto_schema=True)
```

## Configuration

```python
from common.engines import TruthoundEngineConfig

config = TruthoundEngineConfig(
    auto_start=True,
    parallel=True,
    max_workers=4,
    min_severity="medium",
    cache_schemas=True,
    infer_constraints=True,
    categorical_threshold=20,
)

engine = TruthoundEngine(config=config)
```

## Thread Safety

TruthoundEngine is thread-safe:

```python
from concurrent.futures import ThreadPoolExecutor

engine = TruthoundEngine()
engine.start()

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(engine.check, data, auto_schema=True)
        for data in data_batches
    ]
    results = [f.result() for f in futures]

engine.stop()
```
