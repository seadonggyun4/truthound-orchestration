---
title: dbt Macros
---

# dbt Macros

SQL macros for data quality validation.

## truthound_check

Data quality validation macro for tables/models:

```sql
{{ truthound_check(
    table='my_table',
    rules=[
        {'type': 'not_null', 'column': 'id'},
        {'type': 'unique', 'column': 'id'},
    ]
) }}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `table` | string | Table or model name |
| `rules` | list | List of validation rules |

### Rule Format

```sql
{
    'type': 'rule_type',
    'column': 'column_name',
    -- Additional parameters vary by rule type
}
```

## Rule Types

### not_null

NULL value check:

```sql
{{ truthound_check(
    table='orders',
    rules=[{'type': 'not_null', 'column': 'order_id'}]
) }}
```

### unique

Uniqueness check:

```sql
{{ truthound_check(
    table='orders',
    rules=[{'type': 'unique', 'column': 'order_id'}]
) }}
```

### in_range

Range check:

```sql
{{ truthound_check(
    table='orders',
    rules=[{
        'type': 'in_range',
        'column': 'amount',
        'min': 0,
        'max': 1000000
    }]
) }}
```

### in_set

Value set check:

```sql
{{ truthound_check(
    table='orders',
    rules=[{
        'type': 'in_set',
        'column': 'status',
        'values': ['pending', 'completed', 'cancelled']
    }]
) }}
```

### regex

Regex pattern check:

```sql
{{ truthound_check(
    table='customers',
    rules=[{
        'type': 'regex',
        'column': 'email',
        'pattern': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    }]
) }}
```

## Database-Specific Adapters

Handles SQL syntax differences across databases.

### BigQuery

```sql
-- adapters/bigquery.sql
{% macro bigquery_regex_check(column, pattern) %}
    REGEXP_CONTAINS({{ column }}, r'{{ pattern }}')
{% endmacro %}
```

### Snowflake

```sql
-- adapters/snowflake.sql
{% macro snowflake_regex_check(column, pattern) %}
    REGEXP_LIKE({{ column }}, '{{ pattern }}')
{% endmacro %}
```

### Redshift

```sql
-- adapters/redshift.sql
{% macro redshift_regex_check(column, pattern) %}
    {{ column }} ~ '{{ pattern }}'
{% endmacro %}
```

### Databricks

```sql
-- adapters/databricks.sql
{% macro databricks_regex_check(column, pattern) %}
    REGEXP({{ column }}, '{{ pattern }}')
{% endmacro %}
```

## Utility Macros

### truthound_utils.sql

```sql
-- Check if column exists
{% macro column_exists(table, column) %}
    ...
{% endmacro %}

-- Get table row count
{% macro row_count(table) %}
    ...
{% endmacro %}
```

## Rule Definition Macros

Reusable rule definitions:

```sql
-- macros/truthound_rules.sql

{% macro common_id_rules(column='id') %}
    {{ return([
        {'type': 'not_null', 'column': column},
        {'type': 'unique', 'column': column},
    ]) }}
{% endmacro %}

{% macro common_email_rules(column='email') %}
    {{ return([
        {'type': 'not_null', 'column': column},
        {'type': 'regex', 'column': column, 'pattern': '^.*@.*$'},
    ]) }}
{% endmacro %}
```

Usage:

```sql
{{ truthound_check(
    table='users',
    rules=common_id_rules('user_id') + common_email_rules()
) }}
```

## Test Models

Example test model:

```sql
-- tests/models/test_orders.sql

with validation as (
    {{ truthound_check(
        table=ref('stg_orders'),
        rules=[
            {'type': 'not_null', 'column': 'order_id'},
            {'type': 'unique', 'column': 'order_id'},
            {'type': 'in_range', 'column': 'amount', 'min': 0},
        ]
    ) }}
)

select * from validation where passed = false
```
