# Truthound dbt Package

> Data quality checks for dbt using Truthound's declarative rule syntax.

## Overview

`truthound` is a dbt package that provides comprehensive data quality testing through Generic Tests and Jinja Macros. It integrates seamlessly with dbt's native testing framework while offering advanced validation capabilities.

## Features

- **20+ Built-in Check Types**: Completeness, uniqueness, validity, consistency, and more
- **Cross-Adapter Support**: PostgreSQL, Snowflake, BigQuery, Redshift, Databricks
- **Declarative YAML Configuration**: Define rules in schema.yml
- **Composable Rules**: Combine multiple checks in a single test
- **Performance Optimized**: Adapter-specific SQL optimizations
- **CI/CD Ready**: Manifest parser for coverage reporting

## Installation

### From dbt Hub (Recommended)

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=0.1.0"
```

### From Git

```yaml
# packages.yml
packages:
  - git: "https://github.com/seadonggyun4/truthound-integrations.git"
    subdirectory: "packages/dbt"
    revision: "v0.1.0"
```

### Local Development

```yaml
# packages.yml
packages:
  - local: "../path/to/truthound-orchestration/packages/dbt"
```

Then run:

```bash
dbt deps
```

## Quick Start

### Basic Usage

```yaml
# models/staging/schema.yml
version: 2

models:
  - name: stg_users
    tests:
      - truthound_check:
          rules:
            - column: user_id
              check: not_null
            - column: user_id
              check: unique
            - column: email
              check: email_format
            - column: age
              check: range
              min: 0
              max: 150
            - column: status
              check: in_set
              values: ['active', 'inactive', 'pending']
```

### Column-Level Tests

```yaml
columns:
  - name: email
    tests:
      - truthound_email_format
      - truthound_not_null

  - name: age
    tests:
      - truthound_range:
          min: 0
          max: 150
```

## Supported Check Types

### Completeness

| Check | Description | Parameters |
|-------|-------------|------------|
| `not_null` | Value must not be NULL | - |
| `not_empty` | Value must not be NULL or empty string | - |

### Uniqueness

| Check | Description | Parameters |
|-------|-------------|------------|
| `unique` | Values must be unique | - |
| `unique_combination` | Combination of columns must be unique | `columns: [col1, col2]` |

### Set Membership

| Check | Description | Parameters |
|-------|-------------|------------|
| `in_set` | Value must be in allowed set | `values: [...]` |
| `not_in_set` | Value must not be in forbidden set | `values: [...]` |

### Numeric Ranges

| Check | Description | Parameters |
|-------|-------------|------------|
| `range` | Value must be within range | `min`, `max` |
| `positive` | Value must be > 0 | - |
| `negative` | Value must be < 0 | - |
| `non_negative` | Value must be >= 0 | - |
| `greater_than` | Value must be > threshold | `value` |
| `less_than` | Value must be < threshold | `value` |

### String Patterns

| Check | Description | Parameters |
|-------|-------------|------------|
| `length` | String length must be within range | `min`, `max` |
| `regex` | Value must match regex pattern | `pattern` |
| `email_format` | Valid email format | - |
| `url_format` | Valid URL format | - |
| `uuid_format` | Valid UUID format | - |
| `phone_format` | Valid E.164 phone format | - |
| `ipv4_format` | Valid IPv4 address | - |

### Temporal

| Check | Description | Parameters |
|-------|-------------|------------|
| `not_future` | Date must not be in the future | - |
| `not_past` | Date must not be in the past | - |
| `date_format` | Must match date format | `format` (default: YYYY-MM-DD) |

### Referential Integrity

| Check | Description | Parameters |
|-------|-------------|------------|
| `referential_integrity` | Value must exist in another table | `to_model`, `to_column` |

### Custom

| Check | Description | Parameters |
|-------|-------------|------------|
| `expression` | Custom SQL expression (TRUE = failure) | `expression` |
| `row_count_range` | Table row count must be within range | `min`, `max` |

## Advanced Usage

### Multiple Rules with Configuration

```yaml
models:
  - name: fct_orders
    tests:
      - truthound_check:
          rules:
            - column: order_id
              check: not_null
            - column: amount
              check: positive
            - column: status
              check: in_set
              values: ['pending', 'shipped', 'delivered']
          sample_size: 10000  # Random sample for large tables
          fail_on_first: true  # Stop at first failure
          where: "created_at >= '2024-01-01'"  # Filter rows
          config:
            severity: warn
            tags: ['quality', 'orders']
```

### Referential Integrity

```yaml
models:
  - name: fct_orders
    tests:
      - truthound_check:
          rules:
            - column: customer_id
              check: referential_integrity
              to_model: dim_customers
              to_column: customer_id
```

### Custom Expression

```yaml
models:
  - name: events
    tests:
      - truthound_check:
          rules:
            - check: expression
              expression: "start_time > end_time"
              message: "Start time must be before end time"
```

### Source Testing

```yaml
sources:
  - name: raw_data
    tables:
      - name: raw_events
        tests:
          - truthound_check:
              rules:
                - column: event_id
                  check: not_null
                - column: event_type
                  check: in_set
                  values: ['click', 'view', 'purchase']
```

## Convenience Tests

For common single-check scenarios, use these shorthand tests:

```yaml
columns:
  - name: id
    tests:
      - truthound_not_null
      - truthound_unique

  - name: email
    tests:
      - truthound_email_format

  - name: amount
    tests:
      - truthound_positive
      - truthound_range:
          min: 0
          max: 10000

  - name: customer_id
    tests:
      - truthound_relationships:
          to: ref('dim_customers')
          field: customer_id
```

## Configuration

### Project-Level Variables

```yaml
# dbt_project.yml
vars:
  truthound:
    default_severity: 'warn'  # Default: 'error'
    debug_mode: true          # Enable debug logging
    sample_size: 10000        # Default sample size
    patterns:
      email: '^[a-z]+@company\.com$'  # Custom email pattern
```

### Test-Level Configuration

```yaml
- truthound_check:
    rules: [...]
    config:
      severity: warn
      tags: ['critical', 'pii']
      where: "is_active = true"
```

## Run Operations

### Ad-hoc Validation

```bash
# Check a specific model
dbt run-operation run_truthound_check --args '{
  "model_name": "stg_users",
  "rules": [{"column": "email", "check": "email_format"}]
}'

# Generate summary report
dbt run-operation run_truthound_summary --args '{
  "model_name": "stg_users",
  "rules": [
    {"column": "email", "check": "not_null"},
    {"column": "email", "check": "email_format"}
  ]
}'
```

## Manifest Parser (CI/CD)

The package includes a Python script for parsing manifest.json:

```bash
# Parse and list tests
python scripts/manifest_parser.py parse target/manifest.json

# Generate coverage report
python scripts/manifest_parser.py report target/manifest.json --format markdown

# Check coverage threshold
python scripts/manifest_parser.py coverage target/manifest.json --threshold 0.8
```

## Adapter Support

| Adapter | Support Level | Notes |
|---------|---------------|-------|
| PostgreSQL | Full | Default implementation |
| Snowflake | Full | QUALIFY, REGEXP_LIKE optimizations |
| BigQuery | Full | QUALIFY, REGEXP_CONTAINS optimizations |
| Redshift | Full | Subquery-based (no QUALIFY) |
| Databricks | Full | RLIKE, TRY_CAST optimizations |
| DuckDB | Full | Standard SQL implementation |

## Requirements

- dbt-core >= 1.6.0
- One of the supported adapters

## License

Apache 2.0

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
