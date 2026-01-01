---
title: dbt Integration
---

# dbt Integration

The dbt integration provides SQL macros, generic tests, and a Python package for data quality validation within dbt projects. The implementation supports cross-adapter compatibility for major data warehouses.

## Installation

### dbt Hub

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=0.1.0"
```

```bash
dbt deps
```

### Git Installation

```yaml
# packages.yml
packages:
  - git: "https://github.com/seadonggyun4/truthound-orchestration.git"
    subdirectory: "packages/dbt"
    revision: "main"
```

## Package Structure

```
packages/dbt/
├── src/truthound_dbt/        # Python package
│   ├── adapters/             # Database-specific adapters
│   ├── converters/           # Rule converters
│   ├── generators/           # SQL, schema, and test generators
│   ├── parsers/              # Manifest and results parsers
│   └── hooks/                # dbt hook system
├── macros/                   # SQL macros
│   ├── truthound_check.sql   # Main validation macro
│   ├── truthound_rules.sql   # Rule-specific SQL generators
│   ├── truthound_utils.sql   # Cross-adapter utility functions
│   └── adapters/             # Database-specific optimizations
├── tests/generic/            # Generic test implementations
└── integration_tests/        # Integration test suite
```

## Components

| Component | Description |
|-----------|-------------|
| `truthound_check.sql` | Main data validation macro |
| `truthound_rules.sql` | Rule-specific SQL generators |
| `truthound_utils.sql` | Cross-adapter utility functions |
| `adapters/` | Database-specific macro implementations |
| Python Package | Adapters, converters, generators, parsers, and hooks |

## Supported Databases

| Database | SQL Adapter | Python Adapter |
|----------|-------------|----------------|
| PostgreSQL | `adapters/default.sql` | `adapters/postgres.py` |
| Snowflake | `adapters/snowflake.sql` | `adapters/snowflake.py` |
| BigQuery | `adapters/bigquery.sql` | `adapters/bigquery.py` |
| Redshift | `adapters/redshift.sql` | `adapters/redshift.py` |
| Databricks | `adapters/databricks.sql` | `adapters/databricks.py` |

## Macro Usage

### truthound_check

Data quality validation for tables/models:

```sql
-- models/staging/stg_orders.sql

{{ truthound_check(
    table='orders',
    rules=[
        {'type': 'not_null', 'column': 'order_id'},
        {'type': 'unique', 'column': 'order_id'},
        {'type': 'in_range', 'column': 'amount', 'min': 0},
    ]
) }}
```

### Rule Definition

```sql
-- macros/quality_rules.sql

{% macro order_rules() %}
    {{ return([
        {'type': 'not_null', 'column': 'order_id'},
        {'type': 'unique', 'column': 'order_id'},
        {'type': 'not_null', 'column': 'customer_id'},
        {'type': 'in_range', 'column': 'amount', 'min': 0},
        {'type': 'in_set', 'column': 'status', 'values': ['pending', 'completed', 'cancelled']},
    ]) }}
{% endmacro %}
```

## Tests

### Generic Test

```sql
-- tests/generic/test_truthound_check.sql

{% test truthound_check(model, rules) %}
    {{ truthound_check(model, rules) }}
{% endtest %}
```

Usage:

```yaml
# models/staging/schema.yml

models:
  - name: stg_orders
    tests:
      - truthound_check:
          rules:
            - {type: not_null, column: order_id}
            - {type: unique, column: order_id}
```

## Supported Rule Types

| Rule | Description | Example |
|------|-------------|---------|
| `not_null` | NULL value check | `{type: not_null, column: id}` |
| `unique` | Uniqueness check | `{type: unique, column: id}` |
| `in_range` | Range check | `{type: in_range, column: age, min: 0, max: 150}` |
| `in_set` | Value set check | `{type: in_set, column: status, values: [a, b]}` |
| `regex` | Regex check | `{type: regex, column: email, pattern: '^.*@.*$'}` |

## Python Package

The Python package provides programmatic access to dbt manifest parsing, SQL generation, and rule conversion.

### Manifest Parser

```python
from packages.dbt.src.truthound_dbt.parsers.manifest import parse_manifest

manifest = parse_manifest("target/manifest.json")

for model in manifest.models:
    print(f"Model: {model.name}")
    for test in model.tests:
        print(f"  Test: {test.name}")
```

### SQL Generator

```python
from packages.dbt.src.truthound_dbt.generators.sql import SQLGenerator

generator = SQLGenerator(adapter="snowflake")
sql = generator.generate_check_sql(
    table="users",
    rules=[{"type": "not_null", "column": "id"}]
)
```

### Rule Converter

```python
from packages.dbt.src.truthound_dbt.converters.rules import RuleConverter

converter = RuleConverter()
sql_condition = converter.to_sql({"type": "in_range", "column": "age", "min": 0, "max": 150})
```

## Requirements

| Dependency | Version |
|------------|---------|
| dbt-core | >= 1.6.0 |
| Python | >= 3.11 |

## Navigation

- [Macros](macros.md) - Detailed macro documentation
