{#
    Truthound Rule Macros
    =====================

    Individual rule implementations for data quality checks.

    Architecture:
    - Each rule is a standalone macro that returns SQL selecting failing rows
    - Rules are composable and can be combined in truthound_check
    - All rules follow the pattern: return rows that FAIL the check
    - Rules use adapter.dispatch() for cross-database compatibility where needed

    Rule Categories:
    - Completeness: not_null, required_columns
    - Uniqueness: unique, unique_combination
    - Validity: in_set, not_in_set, regex, format checks
    - Consistency: range, length, comparison
    - Temporal: not_future, not_past, date_format

    Usage:
        {% set failing_rows = truthound.rule_not_null(ref('users'), 'email') %}
#}


{# ============================================================================
   Rule Dispatcher
   ============================================================================ #}

{% macro generate_rule_sql(model, rule) %}
{#
    Generate SQL for a rule based on its type.

    This is the main dispatcher that routes to individual rule macros.

    Parameters
    ----------
    model : Relation
        The dbt model or source to check
    rule : dict
        Rule definition with at least 'check' key

    Returns
    -------
    SQL query string
        Query selecting rows that fail the check

    Example
    -------
    {% set sql = truthound.generate_rule_sql(
        ref('users'),
        {"column": "email", "check": "email_format"}
    ) %}
#}

{% set check_type = rule.get('check', rule.get('type')) %}
{% set column = rule.get('column') %}

{# Log debug info if enabled #}
{{ truthound.debug_log('Generating rule SQL: ' ~ check_type ~ ' on ' ~ column) }}

{# Dispatch to appropriate rule macro #}
{% if check_type == 'not_null' %}
    {{ return(truthound.rule_not_null(model, column, rule)) }}

{% elif check_type == 'unique' %}
    {{ return(truthound.rule_unique(model, column, rule)) }}

{% elif check_type == 'unique_combination' %}
    {{ return(truthound.rule_unique_combination(model, rule.get('columns', []), rule)) }}

{% elif check_type == 'in_set' or check_type == 'accepted_values' %}
    {{ return(truthound.rule_in_set(model, column, rule.get('values', []), rule)) }}

{% elif check_type == 'not_in_set' or check_type == 'rejected_values' %}
    {{ return(truthound.rule_not_in_set(model, column, rule.get('values', []), rule)) }}

{% elif check_type == 'range' or check_type == 'between' %}
    {{ return(truthound.rule_range(model, column, rule.get('min'), rule.get('max'), rule)) }}

{% elif check_type == 'length' %}
    {{ return(truthound.rule_length(model, column, rule.get('min'), rule.get('max'), rule)) }}

{% elif check_type == 'regex' or check_type == 'pattern' %}
    {{ return(truthound.rule_regex(model, column, rule.get('pattern'), rule)) }}

{% elif check_type == 'email_format' or check_type == 'email' %}
    {{ return(truthound.rule_email_format(model, column, rule)) }}

{% elif check_type == 'url_format' or check_type == 'url' %}
    {{ return(truthound.rule_url_format(model, column, rule)) }}

{% elif check_type == 'uuid_format' or check_type == 'uuid' %}
    {{ return(truthound.rule_uuid_format(model, column, rule)) }}

{% elif check_type == 'phone_format' or check_type == 'phone' %}
    {{ return(truthound.rule_phone_format(model, column, rule)) }}

{% elif check_type == 'ipv4_format' or check_type == 'ipv4' %}
    {{ return(truthound.rule_ipv4_format(model, column, rule)) }}

{% elif check_type == 'positive' %}
    {{ return(truthound.rule_positive(model, column, rule)) }}

{% elif check_type == 'negative' %}
    {{ return(truthound.rule_negative(model, column, rule)) }}

{% elif check_type == 'non_negative' %}
    {{ return(truthound.rule_non_negative(model, column, rule)) }}

{% elif check_type == 'non_positive' %}
    {{ return(truthound.rule_non_positive(model, column, rule)) }}

{% elif check_type == 'not_future' %}
    {{ return(truthound.rule_not_future(model, column, rule)) }}

{% elif check_type == 'not_past' %}
    {{ return(truthound.rule_not_past(model, column, rule)) }}

{% elif check_type == 'date_format' %}
    {{ return(truthound.rule_date_format(model, column, rule.get('format', 'YYYY-MM-DD'), rule)) }}

{% elif check_type == 'greater_than' or check_type == 'gt' %}
    {{ return(truthound.rule_greater_than(model, column, rule.get('value'), rule)) }}

{% elif check_type == 'greater_than_or_equal' or check_type == 'gte' %}
    {{ return(truthound.rule_greater_than_or_equal(model, column, rule.get('value'), rule)) }}

{% elif check_type == 'less_than' or check_type == 'lt' %}
    {{ return(truthound.rule_less_than(model, column, rule.get('value'), rule)) }}

{% elif check_type == 'less_than_or_equal' or check_type == 'lte' %}
    {{ return(truthound.rule_less_than_or_equal(model, column, rule.get('value'), rule)) }}

{% elif check_type == 'equal' or check_type == 'eq' %}
    {{ return(truthound.rule_equal(model, column, rule.get('value'), rule)) }}

{% elif check_type == 'not_equal' or check_type == 'neq' %}
    {{ return(truthound.rule_not_equal(model, column, rule.get('value'), rule)) }}

{% elif check_type == 'not_empty' %}
    {{ return(truthound.rule_not_empty(model, column, rule)) }}

{% elif check_type == 'referential_integrity' or check_type == 'foreign_key' %}
    {{ return(truthound.rule_referential_integrity(model, column, rule.get('to_model'), rule.get('to_column'), rule)) }}

{% elif check_type == 'expression' or check_type == 'custom' %}
    {{ return(truthound.rule_expression(model, rule.get('expression'), rule)) }}

{% elif check_type == 'row_count_range' %}
    {{ return(truthound.rule_row_count_range(model, rule.get('min'), rule.get('max'), rule)) }}

{% else %}
    {{ truthound.raise_error("Unknown check type: '" ~ check_type ~ "'. Available types: not_null, unique, unique_combination, in_set, not_in_set, range, length, regex, email_format, url_format, uuid_format, phone_format, ipv4_format, positive, negative, non_negative, non_positive, not_future, not_past, date_format, greater_than, greater_than_or_equal, less_than, less_than_or_equal, equal, not_equal, not_empty, referential_integrity, expression, row_count_range") }}

{% endif %}

{% endmacro %}


{# ============================================================================
   Completeness Rules
   ============================================================================ #}

{% macro rule_not_null(model, column, rule={}) %}
{#
    Check for NULL values.

    Fails when: column IS NULL

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    rule : dict
        Additional rule options

    Returns
    -------
    SQL query
        Rows where column is NULL
#}
select *
from {{ model }}
where {{ column }} is null
{% endmacro %}


{% macro rule_not_empty(model, column, rule={}) %}
{#
    Check for NULL or empty string values.

    Fails when: column IS NULL OR trim(column) = ''

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    rule : dict
        Additional rule options
#}
select *
from {{ model }}
where {{ truthound.is_null_or_empty(column) }}
{% endmacro %}


{# ============================================================================
   Uniqueness Rules
   ============================================================================ #}

{% macro rule_unique(model, column, rule={}) %}
{#
    Check for duplicate values.

    Fails when: column value appears more than once

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    rule : dict
        Additional rule options
#}
{{ return(adapter.dispatch('rule_unique_optimized', 'truthound')(model, column)) }}
{% endmacro %}


{% macro default__rule_unique_optimized(model, column) %}
{# Default implementation using subquery join #}
select t.*
from {{ model }} t
inner join (
    select {{ column }}
    from {{ model }}
    where {{ column }} is not null
    group by {{ column }}
    having count(*) > 1
) duplicates
on t.{{ column }} = duplicates.{{ column }}
{% endmacro %}


{% macro rule_unique_combination(model, columns, rule={}) %}
{#
    Check for duplicate combinations of multiple columns.

    Fails when: combination of column values appears more than once

    Parameters
    ----------
    model : Relation
        Model to check
    columns : list[str]
        List of column names
    rule : dict
        Additional rule options
#}
{% if columns | length == 0 %}
    {{ truthound.raise_error("unique_combination requires at least one column") }}
{% endif %}

{% set column_list = columns | join(', ') %}

select t.*
from {{ model }} t
inner join (
    select {{ column_list }}
    from {{ model }}
    group by {{ column_list }}
    having count(*) > 1
) duplicates
on {% for col in columns %}
    t.{{ col }} = duplicates.{{ col }}{% if not loop.last %} and {% endif %}
{% endfor %}
{% endmacro %}


{# ============================================================================
   Validity Rules - Set Membership
   ============================================================================ #}

{% macro rule_in_set(model, column, values, rule={}) %}
{#
    Check if values are in allowed set.

    Fails when: column value is NOT in the allowed set (and not NULL)

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    values : list
        Allowed values
    rule : dict
        Additional rule options
#}
{% if values | length == 0 %}
    {{ truthound.raise_error("in_set requires at least one value") }}
{% endif %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} not in ({{ truthound.quote_values(values) }})
{% endmacro %}


{% macro rule_not_in_set(model, column, values, rule={}) %}
{#
    Check if values are NOT in forbidden set.

    Fails when: column value IS in the forbidden set

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    values : list
        Forbidden values
    rule : dict
        Additional rule options
#}
{% if values | length == 0 %}
    {{ truthound.raise_error("not_in_set requires at least one value") }}
{% endif %}

select *
from {{ model }}
where {{ column }} in ({{ truthound.quote_values(values) }})
{% endmacro %}


{# ============================================================================
   Validity Rules - Numeric Ranges
   ============================================================================ #}

{% macro rule_range(model, column, min_val=none, max_val=none, rule={}) %}
{#
    Check if values are within a range.

    Fails when: value < min OR value > max

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    min_val : number, optional
        Minimum allowed value (inclusive)
    max_val : number, optional
        Maximum allowed value (inclusive)
    rule : dict
        Additional rule options

    Note
    ----
    At least one of min_val or max_val must be provided.
#}
{% if min_val is none and max_val is none %}
    {{ truthound.raise_error("range requires at least min or max value") }}
{% endif %}

select *
from {{ model }}
where {{ column }} is not null
  and (
    {% if min_val is not none %}
    {{ column }} < {{ min_val }}
    {% endif %}
    {% if min_val is not none and max_val is not none %}
    or
    {% endif %}
    {% if max_val is not none %}
    {{ column }} > {{ max_val }}
    {% endif %}
  )
{% endmacro %}


{% macro rule_positive(model, column, rule={}) %}
{#
    Check if values are positive (> 0).

    Fails when: value <= 0
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} <= 0
{% endmacro %}


{% macro rule_negative(model, column, rule={}) %}
{#
    Check if values are negative (< 0).

    Fails when: value >= 0
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} >= 0
{% endmacro %}


{% macro rule_non_negative(model, column, rule={}) %}
{#
    Check if values are non-negative (>= 0).

    Fails when: value < 0
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} < 0
{% endmacro %}


{% macro rule_non_positive(model, column, rule={}) %}
{#
    Check if values are non-positive (<= 0).

    Fails when: value > 0
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} > 0
{% endmacro %}


{% macro rule_greater_than(model, column, value, rule={}) %}
{#
    Check if values are greater than threshold.

    Fails when: value <= threshold
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} <= {{ value }}
{% endmacro %}


{% macro rule_greater_than_or_equal(model, column, value, rule={}) %}
{#
    Check if values are greater than or equal to threshold.

    Fails when: value < threshold
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} < {{ value }}
{% endmacro %}


{% macro rule_less_than(model, column, value, rule={}) %}
{#
    Check if values are less than threshold.

    Fails when: value >= threshold
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} >= {{ value }}
{% endmacro %}


{% macro rule_less_than_or_equal(model, column, value, rule={}) %}
{#
    Check if values are less than or equal to threshold.

    Fails when: value > threshold
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} > {{ value }}
{% endmacro %}


{% macro rule_equal(model, column, value, rule={}) %}
{#
    Check if values equal expected value.

    Fails when: value != expected
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} != {{ truthound.quote_value(value) }}
{% endmacro %}


{% macro rule_not_equal(model, column, value, rule={}) %}
{#
    Check if values do not equal forbidden value.

    Fails when: value == forbidden
#}
select *
from {{ model }}
where {{ column }} = {{ truthound.quote_value(value) }}
{% endmacro %}


{# ============================================================================
   Validity Rules - String Length
   ============================================================================ #}

{% macro rule_length(model, column, min_len=none, max_len=none, rule={}) %}
{#
    Check if string length is within range.

    Fails when: length < min OR length > max

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    min_len : int, optional
        Minimum length
    max_len : int, optional
        Maximum length
    rule : dict
        Additional rule options
#}
{% if min_len is none and max_len is none %}
    {{ truthound.raise_error("length requires at least min or max value") }}
{% endif %}

select *
from {{ model }}
where {{ column }} is not null
  and (
    {% if min_len is not none %}
    {{ truthound.length(column) }} < {{ min_len }}
    {% endif %}
    {% if min_len is not none and max_len is not none %}
    or
    {% endif %}
    {% if max_len is not none %}
    {{ truthound.length(column) }} > {{ max_len }}
    {% endif %}
  )
{% endmacro %}


{# ============================================================================
   Validity Rules - Pattern Matching
   ============================================================================ #}

{% macro rule_regex(model, column, pattern, rule={}) %}
{#
    Check if values match regex pattern.

    Fails when: value does NOT match pattern

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    pattern : str
        Regex pattern to match
    rule : dict
        Additional rule options
#}
{% if pattern is none or pattern == '' %}
    {{ truthound.raise_error("regex requires a pattern") }}
{% endif %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ truthound.regex_not_match(column, pattern) }}
{% endmacro %}


{% macro rule_email_format(model, column, rule={}) %}
{#
    Check if values are valid email format.

    Fails when: value is not a valid email format
#}
{% set email_pattern = truthound.get_pattern('email') %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ truthound.regex_not_match(column, email_pattern) }}
{% endmacro %}


{% macro rule_url_format(model, column, rule={}) %}
{#
    Check if values are valid URL format.

    Fails when: value is not a valid URL format
#}
{% set url_pattern = truthound.get_pattern('url') %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ truthound.regex_not_match(column, url_pattern) }}
{% endmacro %}


{% macro rule_uuid_format(model, column, rule={}) %}
{#
    Check if values are valid UUID format.

    Fails when: value is not a valid UUID format
#}
{% set uuid_pattern = truthound.get_pattern('uuid') %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ truthound.regex_not_match(column, uuid_pattern) }}
{% endmacro %}


{% macro rule_phone_format(model, column, rule={}) %}
{#
    Check if values are valid phone format (E.164).

    Fails when: value is not a valid phone format
#}
{% set phone_pattern = truthound.get_pattern('phone_e164') %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ truthound.regex_not_match(column, phone_pattern) }}
{% endmacro %}


{% macro rule_ipv4_format(model, column, rule={}) %}
{#
    Check if values are valid IPv4 format.

    Fails when: value is not a valid IPv4 address
#}
{% set ipv4_pattern = truthound.get_pattern('ipv4') %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ truthound.regex_not_match(column, ipv4_pattern) }}
{% endmacro %}


{# ============================================================================
   Temporal Rules
   ============================================================================ #}

{% macro rule_not_future(model, column, rule={}) %}
{#
    Check if dates are not in the future.

    Fails when: value > current timestamp
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} > {{ truthound.current_timestamp() }}
{% endmacro %}


{% macro rule_not_past(model, column, rule={}) %}
{#
    Check if dates are not in the past.

    Fails when: value < current timestamp
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} < {{ truthound.current_timestamp() }}
{% endmacro %}


{% macro rule_date_format(model, column, format, rule={}) %}
{#
    Check if values match date format.

    Fails when: value cannot be parsed as the specified date format

    Parameters
    ----------
    model : Relation
        Model to check
    column : str
        Column name
    format : str
        Expected date format (e.g., 'YYYY-MM-DD')
    rule : dict
        Additional rule options
#}
select *
from {{ model }}
where {{ column }} is not null
  and not {{ truthound.is_valid_date_format(column, format) }}
{% endmacro %}


{# ============================================================================
   Referential Integrity Rules
   ============================================================================ #}

{% macro rule_referential_integrity(model, column, to_model, to_column, rule={}) %}
{#
    Check referential integrity (foreign key).

    Fails when: value in column does not exist in to_model.to_column

    Parameters
    ----------
    model : Relation
        Source model
    column : str
        Source column (FK)
    to_model : str
        Target model name (for ref())
    to_column : str
        Target column (PK)
    rule : dict
        Additional rule options
#}
{% if to_model is none or to_column is none %}
    {{ truthound.raise_error("referential_integrity requires to_model and to_column") }}
{% endif %}

select t.*
from {{ model }} t
left join {{ ref(to_model) }} r
  on t.{{ column }} = r.{{ to_column }}
where t.{{ column }} is not null
  and r.{{ to_column }} is null
{% endmacro %}


{# ============================================================================
   Custom Expression Rules
   ============================================================================ #}

{% macro rule_expression(model, expression, rule={}) %}
{#
    Check custom SQL expression.

    Fails when: expression evaluates to TRUE

    Parameters
    ----------
    model : Relation
        Model to check
    expression : str
        SQL boolean expression (TRUE = failure)
    rule : dict
        Additional rule options

    Example
    -------
    - check: expression
      expression: "start_date > end_date"
#}
{% if expression is none or expression == '' %}
    {{ truthound.raise_error("expression rule requires an expression") }}
{% endif %}

select *
from {{ model }}
where {{ expression }}
{% endmacro %}


{# ============================================================================
   Aggregate Rules
   ============================================================================ #}

{% macro rule_row_count_range(model, min_count=none, max_count=none, rule={}) %}
{#
    Check if table row count is within range.

    Fails when: row count < min OR row count > max

    This returns a single dummy row if the check fails.

    Parameters
    ----------
    model : Relation
        Model to check
    min_count : int, optional
        Minimum row count
    max_count : int, optional
        Maximum row count
    rule : dict
        Additional rule options

    Note
    ----
    Unlike other rules, this returns 0 or 1 row indicating pass/fail.
#}
{% if min_count is none and max_count is none %}
    {{ truthound.raise_error("row_count_range requires at least min or max count") }}
{% endif %}

with row_count as (
    select count(*) as cnt from {{ model }}
)
select
    'row_count_check' as _truthound_check_type,
    cnt as _truthound_actual_count,
    {{ min_count if min_count is not none else 'null' }} as _truthound_min_count,
    {{ max_count if max_count is not none else 'null' }} as _truthound_max_count
from row_count
where
    {% if min_count is not none %}
    cnt < {{ min_count }}
    {% endif %}
    {% if min_count is not none and max_count is not none %}
    or
    {% endif %}
    {% if max_count is not none %}
    cnt > {{ max_count }}
    {% endif %}
{% endmacro %}
