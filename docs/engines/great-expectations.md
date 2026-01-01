---
title: GreatExpectationsAdapter
---

# GreatExpectationsAdapter

The Great Expectations adapter employs **rules-based** validation methodology.

## Basic Usage

```python
from common.engines import GreatExpectationsAdapter

engine = GreatExpectationsAdapter()

# Using common rule format (automatically converted to GE expectations)
result = engine.check(
    data=df,
    rules=[
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
        {"type": "in_range", "column": "age", "min": 0, "max": 150},
        {"type": "in_set", "column": "status", "values": ["active", "inactive"]},
    ],
)
```

## Rule Type Conversion

Common rule types are automatically converted to GE Expectations:

| Common Rule Type | GE Expectation |
|------------------|----------------|
| `not_null` | `expect_column_values_to_not_be_null` |
| `unique` | `expect_column_values_to_be_unique` |
| `in_set` | `expect_column_values_to_be_in_set` |
| `in_range` | `expect_column_values_to_be_between` |
| `regex` | `expect_column_values_to_match_regex` |
| `dtype` | `expect_column_values_to_be_of_type` |
| `min_length` / `max_length` | `expect_column_value_lengths_to_be_between` |
| `column_exists` | `expect_column_to_exist` |

## Direct GE Expectation Usage

```python
result = engine.check(
    data=df,
    rules=[
        {"type": "expect_column_to_exist", "column": "id"},
        {
            "type": "expect_column_mean_to_be_between",
            "column": "value",
            "min_value": 0,
            "max_value": 100,
        },
    ],
)
```

## GE-Specific Parameters

```python
result = engine.check(
    data=df,
    rules=rules,
    result_format="COMPLETE",  # BOOLEAN_ONLY, BASIC, COMPLETE
    fail_on_error=True,
)
```

## Data Conversion

GE requires Pandas DataFrames. Polars DataFrames are automatically converted:

```python
import polars as pl

polars_df = pl.read_csv("data.csv")
result = engine.check(polars_df, rules)  # Operates transparently
```

## Profiling

```python
profile = engine.profile(df)

for col in profile.columns:
    print(f"{col.column_name}: {col.dtype}")
    print(f"  Null: {col.null_percentage}%")
```

## Schema Learning

```python
learn_result = engine.learn(df)

for rule in learn_result.rules:
    print(f"{rule.column}: {rule.rule_type}")
```

## Lifecycle Management

```python
with GreatExpectationsAdapter() as engine:
    result = engine.check(df, rules)
    health = engine.health_check()
```

## Configuration

```python
from common.engines import GreatExpectationsConfig

config = GreatExpectationsConfig(
    result_format="COMPLETE",
    context_root_dir=None,
    include_profiling=True,
    catch_exceptions=True,
    enable_data_docs=False,
)

engine = GreatExpectationsAdapter(config=config)
```

## Supported Data Types

| Data Type | Support |
|-----------|---------|
| Pandas DataFrame | Native |
| Polars DataFrame | Auto-conversion |
