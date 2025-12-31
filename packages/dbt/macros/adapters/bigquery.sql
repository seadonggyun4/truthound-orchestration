{#
    BigQuery Adapter Implementations
    ================================

    Google BigQuery-specific optimizations for Truthound macros.

    Key optimizations:
    - QUALIFY clause for efficient window function filtering
    - REGEXP_CONTAINS/REGEXP_EXTRACT for pattern matching
    - SAFE_* functions for error-safe operations
    - Struct and Array handling
#}


{# ============================================================================
   Unique Check - BigQuery Optimized
   ============================================================================ #}

{% macro bigquery__rule_unique_optimized(model, column) %}
{#
    BigQuery-optimized unique check using QUALIFY.

    BigQuery supports QUALIFY clause for window function filtering.
#}
select *
from {{ model }}
where {{ column }} is not null
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{% macro bigquery__row_number_duplicates(model, column) %}
{#
    Find duplicates using BigQuery QUALIFY clause.
#}
select
    *,
    row_number() over (partition by {{ column }} order by {{ column }}) as _truthound_row_num
from {{ model }}
where {{ column }} is not null
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{# ============================================================================
   Regex Functions - BigQuery
   ============================================================================ #}

{% macro bigquery__regex_match(column, pattern) %}
{#
    BigQuery regex matching using REGEXP_CONTAINS.

    Note: BigQuery uses r'' prefix for raw strings in regex.
#}
regexp_contains({{ column }}, r'{{ pattern }}')
{% endmacro %}


{% macro bigquery__regex_extract(column, pattern, group_num=0) %}
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
#}
{% if group_num == 0 %}
regexp_extract({{ column }}, r'{{ pattern }}')
{% else %}
regexp_extract({{ column }}, r'{{ pattern }}', {{ group_num }})
{% endif %}
{% endmacro %}


{% macro bigquery__regex_replace(column, pattern, replacement) %}
{#
    Replace pattern matches in column.
#}
regexp_replace({{ column }}, r'{{ pattern }}', r'{{ replacement }}')
{% endmacro %}


{% macro bigquery__regex_extract_all(column, pattern) %}
{#
    Extract all matches as array.
#}
regexp_extract_all({{ column }}, r'{{ pattern }}')
{% endmacro %}


{# ============================================================================
   Date/Time Functions - BigQuery
   ============================================================================ #}

{% macro bigquery__date_diff(date1, date2, unit='DAY') %}
{#
    Calculate difference between two dates.

    Parameters
    ----------
    date1 : str
        First date column/expression
    date2 : str
        Second date column/expression
    unit : str
        Unit of difference (DAY, HOUR, MINUTE, SECOND, MONTH, YEAR)
#}
date_diff({{ date1 }}, {{ date2 }}, {{ unit }})
{% endmacro %}


{% macro bigquery__timestamp_diff(ts1, ts2, unit='SECOND') %}
{#
    Calculate difference between two timestamps.
#}
timestamp_diff({{ ts1 }}, {{ ts2 }}, {{ unit }})
{% endmacro %}


{% macro bigquery__date_trunc(column, unit='DAY') %}
{#
    Truncate date to specified precision.
#}
date_trunc({{ column }}, {{ unit }})
{% endmacro %}


{% macro bigquery__parse_date(column, format) %}
{#
    Parse string to date.

    Note: BigQuery uses strftime-style format codes.
#}
parse_date('{{ format }}', {{ column }})
{% endmacro %}


{% macro bigquery__safe_parse_date(column, format) %}
{#
    Safely parse string to date, return NULL on failure.
#}
safe.parse_date('{{ format }}', {{ column }})
{% endmacro %}


{% macro bigquery__format_date(column, format) %}
{#
    Format date to string.
#}
format_date('{{ format }}', {{ column }})
{% endmacro %}


{# ============================================================================
   Sampling - BigQuery
   ============================================================================ #}

{% macro bigquery__tablesample(percent) %}
{#
    BigQuery table sampling.

    Parameters
    ----------
    percent : float
        Percentage of table to sample (0.0 to 100.0)

    Note
    ----
    BigQuery TABLESAMPLE uses percentage, not row count.
    For row-based sampling, use ORDER BY RAND() LIMIT.
#}
tablesample system ({{ percent }} percent)
{% endmacro %}


{% macro bigquery__rand_limit(n) %}
{#
    Random sampling with exact row count.
#}
order by rand()
limit {{ n }}
{% endmacro %}


{# ============================================================================
   Type Casting - BigQuery
   ============================================================================ #}

{% macro bigquery__safe_cast(column, data_type) %}
{#
    Safe cast using BigQuery SAFE_CAST.
#}
safe_cast({{ column }} as {{ data_type }})
{% endmacro %}


{% macro bigquery__safe_divide(numerator, denominator) %}
{#
    Safe division (returns NULL if denominator is 0).
#}
safe_divide({{ numerator }}, {{ denominator }})
{% endmacro %}


{# ============================================================================
   String Functions - BigQuery
   ============================================================================ #}

{% macro bigquery__split(column, delimiter) %}
{#
    Split string into array.
#}
split({{ column }}, '{{ delimiter }}')
{% endmacro %}


{% macro bigquery__array_to_string(array_column, delimiter) %}
{#
    Join array elements into string.
#}
array_to_string({{ array_column }}, '{{ delimiter }}')
{% endmacro %}


{% macro bigquery__normalize_and_casefold(column) %}
{#
    Normalize unicode and casefold for comparison.
#}
normalize_and_casefold({{ column }})
{% endmacro %}


{% macro bigquery__byte_length(column) %}
{#
    Get byte length of string.
#}
byte_length({{ column }})
{% endmacro %}


{% macro bigquery__char_length(column) %}
{#
    Get character length of string.
#}
char_length({{ column }})
{% endmacro %}


{# ============================================================================
   Array Functions - BigQuery
   ============================================================================ #}

{% macro bigquery__array_contains(array_column, value) %}
{#
    Check if array contains value.
#}
{{ truthound.quote_value(value) }} in unnest({{ array_column }})
{% endmacro %}


{% macro bigquery__array_length(array_column) %}
{#
    Get array length.
#}
array_length({{ array_column }})
{% endmacro %}


{% macro bigquery__array_agg_distinct(column) %}
{#
    Aggregate distinct values into array.
#}
array_agg(distinct {{ column }} ignore nulls)
{% endmacro %}


{# ============================================================================
   JSON Functions - BigQuery
   ============================================================================ #}

{% macro bigquery__json_extract(column, path) %}
{#
    Extract value from JSON column.

    Parameters
    ----------
    column : str
        JSON column name
    path : str
        JSONPath expression (e.g., '$.user.name')
#}
json_extract({{ column }}, '{{ path }}')
{% endmacro %}


{% macro bigquery__json_extract_scalar(column, path) %}
{#
    Extract scalar value from JSON as string.
#}
json_extract_scalar({{ column }}, '{{ path }}')
{% endmacro %}


{% macro bigquery__json_value(column, path) %}
{#
    Extract scalar value from JSON (alternative syntax).
#}
json_value({{ column }}, '{{ path }}')
{% endmacro %}


{% macro bigquery__is_valid_json(column) %}
{#
    Check if column contains valid JSON.
#}
safe.parse_json({{ column }}) is not null
{% endmacro %}


{% macro bigquery__json_query(column, path) %}
{#
    Query JSON and return JSON result.
#}
json_query({{ column }}, '{{ path }}')
{% endmacro %}


{# ============================================================================
   Struct Functions - BigQuery
   ============================================================================ #}

{% macro bigquery__struct_field(struct_column, field_name) %}
{#
    Access struct field.
#}
{{ struct_column }}.{{ field_name }}
{% endmacro %}


{# ============================================================================
   NULL Handling - BigQuery
   ============================================================================ #}

{% macro bigquery__ifnull(column, default_value) %}
{#
    BigQuery IFNULL.
#}
ifnull({{ column }}, {{ default_value }})
{% endmacro %}


{% macro bigquery__nullif(column, value) %}
{#
    Return NULL if column equals value.
#}
nullif({{ column }}, {{ value }})
{% endmacro %}


{# ============================================================================
   Conditional Logic - BigQuery
   ============================================================================ #}

{% macro bigquery__if(condition, true_value, false_value) %}
{#
    BigQuery IF expression.
#}
if({{ condition }}, {{ true_value }}, {{ false_value }})
{% endmacro %}


{% macro bigquery__coalesce_array(columns) %}
{#
    Coalesce across multiple columns.
#}
coalesce({{ columns | join(', ') }})
{% endmacro %}


{# ============================================================================
   Geographic Functions - BigQuery
   ============================================================================ #}

{% macro bigquery__st_geogpoint(longitude, latitude) %}
{#
    Create geography point from coordinates.
#}
st_geogpoint({{ longitude }}, {{ latitude }})
{% endmacro %}


{% macro bigquery__st_distance(geo1, geo2) %}
{#
    Calculate distance between two geography points (in meters).
#}
st_distance({{ geo1 }}, {{ geo2 }})
{% endmacro %}
