{#
    Truthound Check Macro
    =====================

    Main orchestration macro for running data quality checks.

    This macro combines multiple rules into a single unified check,
    returning all failing rows with metadata about which check failed.

    Architecture:
    - Takes a list of rule definitions
    - Generates SQL for each rule via truthound_rules.sql
    - Combines results with UNION ALL
    - Adds metadata columns for debugging

    Usage:
        {{ truthound.truthound_check(ref('users'), [
            {"column": "email", "check": "email_format"},
            {"column": "age", "check": "range", "min": 0, "max": 150}
        ]) }}
#}


{% macro truthound_check(model, rules, options={}) %}
{#
    Execute multiple data quality rules on a model.

    Parameters
    ----------
    model : Relation
        The dbt model or source to check
    rules : list[dict]
        List of rule definitions. Each rule must have:
        - check: str - The check type (e.g., 'not_null', 'unique')
        - column: str - The column to check (optional for some checks)
        - Additional parameters depending on check type

    options : dict
        Additional options:
        - sample_size: int - Limit rows to check (random sample)
        - fail_fast: bool - Stop after first failure
        - include_metadata: bool - Include metadata columns (default: true)
        - where_clause: str - Additional WHERE filter

    Returns
    -------
    SQL query string
        Query that returns all failing rows with metadata

    Example
    -------
    {{ truthound.truthound_check(
        ref('orders'),
        [
            {"column": "order_id", "check": "not_null"},
            {"column": "order_id", "check": "unique"},
            {"column": "amount", "check": "positive"},
            {"column": "status", "check": "in_set", "values": ["pending", "shipped"]}
        ],
        {"sample_size": 10000}
    ) }}
#}

{# Extract options with defaults #}
{% set sample_size = options.get('sample_size', truthound.get_config('sample_size')) %}
{% set fail_fast = options.get('fail_fast', truthound.get_config('fail_on_first', false)) %}
{% set include_metadata = options.get('include_metadata', true) %}
{% set where_clause = options.get('where_clause', options.get('where')) %}

{# Validate inputs #}
{% if rules is none or rules | length == 0 %}
    {{ truthound.raise_error("truthound_check requires at least one rule") }}
{% endif %}

{# Generate SQL for each rule #}
{% set rule_queries = [] %}
{% for rule in rules %}
    {% set rule_sql = truthound.generate_rule_sql(model, rule) %}
    {% do rule_queries.append(rule_sql) %}
{% endfor %}

{# Build the combined query #}
with
{# Base source with optional sampling and filtering #}
source_data as (
    select *
    from {{ model }}
    {% if where_clause %}
    where {{ where_clause }}
    {% endif %}
    {% if sample_size %}
    {{ truthound.limit_sample(sample_size) }}
    {% endif %}
),

{# Generate CTE for each rule check #}
{% for idx in range(rule_queries | length) %}
{% set rule = rules[idx] %}
rule_{{ idx }}_failures as (
    {{ rule_queries[idx] }}
),
{% endfor %}

{# Combine all failures #}
combined_failures as (
    {% for idx in range(rule_queries | length) %}
    {% set rule = rules[idx] %}
    select
        {% if include_metadata %}
        '{{ rule.get('column', '_model_') }}' as _truthound_column,
        '{{ rule.get('check', rule.get('type', 'unknown')) }}' as _truthound_check,
        '{{ rule.get('message', '') }}' as _truthound_message,
        {{ idx }} as _truthound_rule_index,
        {% endif %}
        t.*
    from rule_{{ idx }}_failures t
    {% if not loop.last %}
    union all
    {% endif %}
    {% endfor %}
)

select * from combined_failures
{% if fail_fast %}
limit 1
{% endif %}

{% endmacro %}


{# ============================================================================
   Convenience Macros
   ============================================================================ #}

{% macro check_not_null(model, column, options={}) %}
{#
    Convenience macro for single not_null check.

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    options : dict
        Additional options

    Returns
    -------
    SQL query
        Rows where column is NULL
#}
{{ return(truthound.truthound_check(model, [{"column": column, "check": "not_null"}], options)) }}
{% endmacro %}


{% macro check_unique(model, column, options={}) %}
{#
    Convenience macro for single unique check.

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    options : dict
        Additional options

    Returns
    -------
    SQL query
        Duplicate rows
#}
{{ return(truthound.truthound_check(model, [{"column": column, "check": "unique"}], options)) }}
{% endmacro %}


{% macro check_accepted_values(model, column, values, options={}) %}
{#
    Convenience macro for accepted values check.

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    values : list
        Allowed values
    options : dict
        Additional options

    Returns
    -------
    SQL query
        Rows with invalid values
#}
{{ return(truthound.truthound_check(model, [{"column": column, "check": "in_set", "values": values}], options)) }}
{% endmacro %}


{% macro check_referential_integrity(model, column, to_model, to_column, options={}) %}
{#
    Convenience macro for referential integrity check.

    Parameters
    ----------
    model : Relation
        Source model
    column : str
        Foreign key column
    to_model : str
        Target model name
    to_column : str
        Target column (primary key)
    options : dict
        Additional options

    Returns
    -------
    SQL query
        Rows with orphaned references
#}
{{ return(truthound.truthound_check(model, [{
    "column": column,
    "check": "referential_integrity",
    "to_model": to_model,
    "to_column": to_column
}], options)) }}
{% endmacro %}


{# ============================================================================
   Validation Summary Macros
   ============================================================================ #}

{% macro truthound_summary(model, rules, options={}) %}
{#
    Generate a summary of validation results.

    Instead of returning failing rows, this returns aggregated statistics.

    Parameters
    ----------
    model : Relation
        Model to check
    rules : list[dict]
        List of rule definitions
    options : dict
        Additional options

    Returns
    -------
    SQL query
        Summary with columns:
        - rule_index
        - column_name
        - check_type
        - failure_count
        - total_count
        - failure_rate

    Example
    -------
    {{ truthound.truthound_summary(ref('users'), [
        {"column": "email", "check": "not_null"},
        {"column": "email", "check": "email_format"}
    ]) }}
#}

{% set where_clause = options.get('where_clause', options.get('where')) %}

{# Generate SQL for each rule #}
{% set rule_queries = [] %}
{% for rule in rules %}
    {% set rule_sql = truthound.generate_rule_sql(model, rule) %}
    {% do rule_queries.append(rule_sql) %}
{% endfor %}

with
{# Get total count #}
total_count as (
    select count(*) as cnt
    from {{ model }}
    {% if where_clause %}
    where {{ where_clause }}
    {% endif %}
),

{# Generate CTEs for each rule #}
{% for idx in range(rule_queries | length) %}
{% set rule = rules[idx] %}
rule_{{ idx }}_count as (
    select count(*) as failure_count
    from ({{ rule_queries[idx] }}) failures
),
{% endfor %}

{# Combine into summary #}
summary as (
    {% for idx in range(rule_queries | length) %}
    {% set rule = rules[idx] %}
    select
        {{ idx }} as rule_index,
        '{{ rule.get('column', '_model_') }}' as column_name,
        '{{ rule.get('check', rule.get('type', 'unknown')) }}' as check_type,
        r.failure_count,
        t.cnt as total_count,
        case
            when t.cnt = 0 then 0.0
            else round(cast(r.failure_count as {{ dbt.type_float() }}) / t.cnt * 100, 2)
        end as failure_rate_pct
    from rule_{{ idx }}_count r
    cross join total_count t
    {% if not loop.last %}
    union all
    {% endif %}
    {% endfor %}
)

select * from summary
order by rule_index

{% endmacro %}


{% macro truthound_profile(model, columns=none, options={}) %}
{#
    Generate a data profile for specified columns.

    Returns statistics for each column including:
    - null_count, null_rate
    - distinct_count
    - min, max (for numeric)
    - min_length, max_length (for strings)

    Parameters
    ----------
    model : Relation
        Model to profile
    columns : list[str], optional
        Columns to profile (default: all)
    options : dict
        Additional options

    Returns
    -------
    SQL query
        Profile statistics
#}

{% set where_clause = options.get('where_clause', options.get('where')) %}

{# If columns not specified, we'll profile based on provided list #}
{% if columns is none or columns | length == 0 %}
    {{ truthound.raise_error("truthound_profile requires a list of columns") }}
{% endif %}

with
base_data as (
    select * from {{ model }}
    {% if where_clause %}
    where {{ where_clause }}
    {% endif %}
),

total_count as (
    select count(*) as cnt from base_data
),

{% for column in columns %}
profile_{{ loop.index0 }} as (
    select
        '{{ column }}' as column_name,
        count(*) - count({{ column }}) as null_count,
        count(distinct {{ column }}) as distinct_count,
        min({{ truthound.length(column) }}) as min_length,
        max({{ truthound.length(column) }}) as max_length
    from base_data
),
{% endfor %}

combined_profile as (
    {% for column in columns %}
    select
        p.column_name,
        p.null_count,
        round(cast(p.null_count as {{ dbt.type_float() }}) / nullif(t.cnt, 0) * 100, 2) as null_rate_pct,
        p.distinct_count,
        round(cast(p.distinct_count as {{ dbt.type_float() }}) / nullif(t.cnt, 0) * 100, 2) as distinct_rate_pct,
        p.min_length,
        p.max_length,
        t.cnt as total_count
    from profile_{{ loop.index0 }} p
    cross join total_count t
    {% if not loop.last %}
    union all
    {% endif %}
    {% endfor %}
)

select * from combined_profile
order by column_name

{% endmacro %}


{# ============================================================================
   Run Operation Helpers
   ============================================================================ #}

{% macro run_truthound_check(model_name, rules, options={}) %}
{#
    Run truthound check as a run-operation.

    Usage:
        dbt run-operation run_truthound_check --args '{
            "model_name": "stg_users",
            "rules": [{"column": "email", "check": "email_format"}]
        }'

    Parameters
    ----------
    model_name : str
        Name of the model to check
    rules : list[dict]
        List of rule definitions
    options : dict
        Additional options
#}

{% set check_query = truthound.truthound_check(ref(model_name), rules, options) %}

{{ log('Running Truthound check on ' ~ model_name ~ '...', info=True) }}

{% set results = run_query(check_query) %}

{% if results | length > 0 %}
    {{ log('FAILED: Found ' ~ (results | length) ~ ' failing rows', info=True) }}
    {% for row in results[:10] %}
        {{ log('  ' ~ row | string, info=True) }}
    {% endfor %}
    {% if results | length > 10 %}
        {{ log('  ... and ' ~ (results | length - 10) ~ ' more', info=True) }}
    {% endif %}
{% else %}
    {{ log('PASSED: All checks passed', info=True) }}
{% endif %}

{{ return(results | length == 0) }}

{% endmacro %}


{% macro run_truthound_summary(model_name, rules, options={}) %}
{#
    Run truthound summary as a run-operation.

    Usage:
        dbt run-operation run_truthound_summary --args '{
            "model_name": "stg_users",
            "rules": [{"column": "email", "check": "email_format"}]
        }'
#}

{% set summary_query = truthound.truthound_summary(ref(model_name), rules, options) %}

{{ log('Running Truthound summary on ' ~ model_name ~ '...', info=True) }}

{% set results = run_query(summary_query) %}

{{ log('', info=True) }}
{{ log('=== Truthound Validation Summary ===', info=True) }}
{{ log('Model: ' ~ model_name, info=True) }}
{{ log('', info=True) }}

{% for row in results %}
    {% set status = 'PASS' if row['failure_count'] == 0 else 'FAIL' %}
    {{ log(
        '[' ~ status ~ '] ' ~ row['column_name'] ~ ' / ' ~ row['check_type'] ~
        ' - ' ~ row['failure_count'] ~ '/' ~ row['total_count'] ~
        ' (' ~ row['failure_rate_pct'] ~ '%)',
        info=True
    ) }}
{% endfor %}

{{ log('', info=True) }}

{% endmacro %}
