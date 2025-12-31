{#
    Snowflake Adapter Implementations
    =================================

    Snowflake-specific optimizations for Truthound macros.

    Key optimizations:
    - QUALIFY clause for efficient window function filtering
    - REGEXP_LIKE for pattern matching
    - SAMPLE clause for efficient random sampling
    - TRY_* functions for safe type casting
#}


{# ============================================================================
   Unique Check - Snowflake Optimized
   ============================================================================ #}

{% macro snowflake__rule_unique_optimized(model, column) %}
{#
    Snowflake-optimized unique check using QUALIFY.

    QUALIFY allows filtering on window function results without subquery,
    which is more efficient in Snowflake.
#}
select *
from {{ model }}
where {{ column }} is not null
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{% macro snowflake__row_number_duplicates(model, column) %}
{#
    Find duplicates using Snowflake QUALIFY clause.
#}
select
    *,
    row_number() over (partition by {{ column }} order by {{ column }}) as _truthound_row_num
from {{ model }}
where {{ column }} is not null
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{# ============================================================================
   Regex Functions - Snowflake
   ============================================================================ #}

{% macro snowflake__regex_extract(column, pattern, group_num=0) %}
{#
    Extract substring matching regex pattern.

    Parameters
    ----------
    column : str
        Column name
    pattern : str
        Regex pattern with capture groups
    group_num : int
        Capture group number (0 = entire match)

    Returns
    -------
    SQL expression
        Extracted substring or NULL
#}
regexp_substr({{ column }}, '{{ pattern }}', 1, 1, 'e', {{ group_num }})
{% endmacro %}


{% macro snowflake__regex_replace(column, pattern, replacement) %}
{#
    Replace pattern matches in column.
#}
regexp_replace({{ column }}, '{{ pattern }}', '{{ replacement }}')
{% endmacro %}


{% macro snowflake__regex_count(column, pattern) %}
{#
    Count pattern matches in column.
#}
regexp_count({{ column }}, '{{ pattern }}')
{% endmacro %}


{# ============================================================================
   Date/Time Functions - Snowflake
   ============================================================================ #}

{% macro snowflake__date_diff(date1, date2, unit='day') %}
{#
    Calculate difference between two dates.

    Parameters
    ----------
    date1 : str
        First date column/expression
    date2 : str
        Second date column/expression
    unit : str
        Unit of difference (day, hour, minute, second, month, year)

    Returns
    -------
    SQL expression
        Numeric difference
#}
datediff({{ unit }}, {{ date2 }}, {{ date1 }})
{% endmacro %}


{% macro snowflake__date_trunc(column, unit='day') %}
{#
    Truncate date to specified precision.
#}
date_trunc('{{ unit }}', {{ column }})
{% endmacro %}


{% macro snowflake__try_parse_date(column, format) %}
{#
    Try to parse string as date, return NULL on failure.
#}
try_to_date({{ column }}, '{{ format }}')
{% endmacro %}


{% macro snowflake__try_parse_timestamp(column, format) %}
{#
    Try to parse string as timestamp, return NULL on failure.
#}
try_to_timestamp({{ column }}, '{{ format }}')
{% endmacro %}


{# ============================================================================
   Sampling - Snowflake
   ============================================================================ #}

{% macro snowflake__tablesample(n, method='ROWS') %}
{#
    Snowflake table sampling.

    Parameters
    ----------
    n : int
        Sample size (rows or percentage)
    method : str
        ROWS or PERCENT

    Returns
    -------
    SQL clause
        SAMPLE clause

    Example
    -------
    SELECT * FROM table {{ truthound.tablesample(1000, 'ROWS') }}
#}
{% if method | upper == 'ROWS' %}
sample ({{ n }} rows)
{% else %}
sample ({{ n }})
{% endif %}
{% endmacro %}


{% macro snowflake__bernoulli_sample(probability) %}
{#
    Bernoulli sampling (each row has probability of being included).

    Parameters
    ----------
    probability : float
        Probability (0.0 to 100.0)
#}
sample bernoulli ({{ probability }})
{% endmacro %}


{# ============================================================================
   Type Casting - Snowflake
   ============================================================================ #}

{% macro snowflake__safe_cast(column, data_type) %}
{#
    Safe cast using Snowflake TRY_CAST.
#}
try_cast({{ column }} as {{ data_type }})
{% endmacro %}


{% macro snowflake__try_to_number(column, precision=38, scale=0) %}
{#
    Try to convert to number with precision/scale.
#}
try_to_number({{ column }}, {{ precision }}, {{ scale }})
{% endmacro %}


{# ============================================================================
   String Functions - Snowflake
   ============================================================================ #}

{% macro snowflake__split_part(column, delimiter, part_num) %}
{#
    Split string and get specific part.
#}
split_part({{ column }}, '{{ delimiter }}', {{ part_num }})
{% endmacro %}


{% macro snowflake__array_contains(array_column, value) %}
{#
    Check if array contains value.
#}
array_contains({{ truthound.quote_value(value) }}::variant, {{ array_column }})
{% endmacro %}


{# ============================================================================
   JSON Functions - Snowflake
   ============================================================================ #}

{% macro snowflake__json_extract(column, path) %}
{#
    Extract value from JSON column.

    Parameters
    ----------
    column : str
        JSON column name
    path : str
        JSON path (e.g., 'user.name' or 'items[0].id')

    Returns
    -------
    SQL expression
        Extracted value as variant
#}
{{ column }}:{{ path }}
{% endmacro %}


{% macro snowflake__json_extract_string(column, path) %}
{#
    Extract string value from JSON.
#}
{{ column }}:{{ path }}::string
{% endmacro %}


{% macro snowflake__is_valid_json(column) %}
{#
    Check if column contains valid JSON.
#}
try_parse_json({{ column }}) is not null
{% endmacro %}


{# ============================================================================
   NULL Handling - Snowflake
   ============================================================================ #}

{% macro snowflake__nvl(column, default_value) %}
{#
    Snowflake NVL (null value logic).
#}
nvl({{ column }}, {{ default_value }})
{% endmacro %}


{% macro snowflake__nvl2(column, value_if_not_null, value_if_null) %}
{#
    Snowflake NVL2.
#}
nvl2({{ column }}, {{ value_if_not_null }}, {{ value_if_null }})
{% endmacro %}


{% macro snowflake__zeroifnull(column) %}
{#
    Return 0 if NULL.
#}
zeroifnull({{ column }})
{% endmacro %}


{# ============================================================================
   Conditional Logic - Snowflake
   ============================================================================ #}

{% macro snowflake__iff(condition, true_value, false_value) %}
{#
    Snowflake IFF (inline if).
#}
iff({{ condition }}, {{ true_value }}, {{ false_value }})
{% endmacro %}


{% macro snowflake__decode(column, mappings, default_value=none) %}
{#
    Snowflake DECODE (similar to CASE).

    Parameters
    ----------
    column : str
        Column to decode
    mappings : list[tuple]
        List of (search, result) pairs
    default_value : any
        Default if no match
#}
decode({{ column }}
{%- for search, result in mappings -%}
, {{ search }}, {{ result }}
{%- endfor -%}
{%- if default_value is not none -%}
, {{ default_value }}
{%- endif -%}
)
{% endmacro %}
