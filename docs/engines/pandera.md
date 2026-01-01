---
title: PanderaAdapter
---

# PanderaAdapter

The Pandera adapter supports **hybrid schema-based and rules-based** validation.

## Basic Usage

```python
from common.engines import PanderaAdapter

engine = PanderaAdapter()

result = engine.check(
    data=df,
    rules=[
        {"type": "not_null", "column": "id"},
        {"type": "dtype", "column": "value", "dtype": "float64"},
        {"type": "in_range", "column": "percentage", "min": 0, "max": 100},
        {"type": "regex", "column": "email", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
    ],
)
```

## Rule Type Conversion

Common rule types are automatically converted to Pandera Checks:

| Common Rule Type | Pandera Check |
|------------------|---------------|
| `not_null` | `nullable=False` |
| `unique` | `pa.Check.unique()` |
| `in_set` | `pa.Check.isin(values)` |
| `in_range` | `pa.Check.in_range(min, max)` |
| `regex` | `pa.Check.str_matches(pattern)` |
| `dtype` | `pa.Column(dtype=...)` |
| `greater_than` | `pa.Check.greater_than(value)` |
| `less_than` | `pa.Check.less_than(value)` |

## Pandera-Specific Parameters

```python
result = engine.check(
    data=df,
    rules=rules,
    lazy=True,          # Collect all errors (True) vs stop at first error (False)
    fail_on_error=True,
)
```

## Dtype Mapping

| Common Dtype | Pandera Dtype |
|--------------|---------------|
| `int`, `int32`, `int64` | `pa.Int`, `pa.Int32`, `pa.Int64` |
| `float`, `float32`, `float64` | `pa.Float`, `pa.Float32`, `pa.Float64` |
| `str`, `string` | `pa.String` |
| `bool`, `boolean` | `pa.Bool` |
| `datetime` | `pa.DateTime` |

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
    print(f"{rule.column}: {rule.rule_type}")
```

## Lifecycle Management

```python
with PanderaAdapter() as engine:
    result = engine.check(df, rules)
    health = engine.health_check()
```

## Configuration

```python
from common.engines import PanderaConfig

config = PanderaConfig(
    lazy=True,                  # Collect all errors
    strict=False,               # Strict mode
    coerce=False,               # Type coercion
    unique_column_names=False,  # Column name uniqueness check
    report_duplicates="all",    # Duplicate reporting mode
)

engine = PanderaAdapter(config=config)
```

## Supported Data Types

| Data Type | Support |
|-----------|---------|
| Pandas DataFrame | Native |
| Polars DataFrame | Auto-conversion |
