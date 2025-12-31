{#
    Databricks Adapter Implementations
    ==================================

    Databricks/Spark SQL-specific optimizations for Truthound macros.

    Key features:
    - QUALIFY clause support (Databricks SQL)
    - RLIKE for regex
    - Delta Lake optimizations
    - Spark SQL functions
#}


{# ============================================================================
   Unique Check - Databricks Optimized
   ============================================================================ #}

{% macro databricks__rule_unique_optimized(model, column) %}
{#
    Databricks-optimized unique check using QUALIFY.

    Databricks SQL supports QUALIFY clause.
#}
select *
from {{ model }}
where {{ column }} is not null
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{% macro databricks__row_number_duplicates(model, column) %}
{#
    Find duplicates using Databricks QUALIFY clause.
#}
select
    *,
    row_number() over (partition by {{ column }} order by {{ column }}) as _truthound_row_num
from {{ model }}
where {{ column }} is not null
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{# ============================================================================
   Regex Functions - Databricks
   ============================================================================ #}

{% macro databricks__regex_match(column, pattern) %}
{#
    Databricks regex matching using RLIKE.
#}
{{ column }} rlike '{{ pattern }}'
{% endmacro %}


{% macro databricks__regex_extract(column, pattern, group_num=0) %}
{#
    Extract substring matching regex pattern.
#}
regexp_extract({{ column }}, '{{ pattern }}', {{ group_num }})
{% endmacro %}


{% macro databricks__regex_replace(column, pattern, replacement) %}
{#
    Replace pattern matches in column.
#}
regexp_replace({{ column }}, '{{ pattern }}', '{{ replacement }}')
{% endmacro %}


{% macro databricks__regex_extract_all(column, pattern) %}
{#
    Extract all matches as array.
#}
regexp_extract_all({{ column }}, '{{ pattern }}')
{% endmacro %}


{# ============================================================================
   Date/Time Functions - Databricks
   ============================================================================ #}

{% macro databricks__date_diff(date1, date2, unit='day') %}
{#
    Calculate difference between two dates.
#}
datediff({{ unit }}, {{ date2 }}, {{ date1 }})
{% endmacro %}


{% macro databricks__date_trunc(column, unit='day') %}
{#
    Truncate date to specified precision.
#}
date_trunc('{{ unit }}', {{ column }})
{% endmacro %}


{% macro databricks__date_add(column, days) %}
{#
    Add days to date.
#}
date_add({{ column }}, {{ days }})
{% endmacro %}


{% macro databricks__date_sub(column, days) %}
{#
    Subtract days from date.
#}
date_sub({{ column }}, {{ days }})
{% endmacro %}


{% macro databricks__months_between(date1, date2) %}
{#
    Calculate months between two dates.
#}
months_between({{ date1 }}, {{ date2 }})
{% endmacro %}


{% macro databricks__to_date(column, format=none) %}
{#
    Convert string to date.
#}
{% if format %}
to_date({{ column }}, '{{ format }}')
{% else %}
to_date({{ column }})
{% endif %}
{% endmacro %}


{% macro databricks__to_timestamp(column, format=none) %}
{#
    Convert string to timestamp.
#}
{% if format %}
to_timestamp({{ column }}, '{{ format }}')
{% else %}
to_timestamp({{ column }})
{% endif %}
{% endmacro %}


{% macro databricks__try_to_timestamp(column, format=none) %}
{#
    Safely convert string to timestamp, return NULL on failure.
#}
{% if format %}
try_to_timestamp({{ column }}, '{{ format }}')
{% else %}
try_to_timestamp({{ column }})
{% endif %}
{% endmacro %}


{# ============================================================================
   Sampling - Databricks
   ============================================================================ #}

{% macro databricks__limit_sample(n) %}
{#
    Random sampling with row limit.
#}
order by rand()
limit {{ n }}
{% endmacro %}


{% macro databricks__tablesample(fraction) %}
{#
    Databricks table sampling.

    Parameters
    ----------
    fraction : float
        Fraction of rows to sample (0.0 to 1.0)
#}
tablesample ({{ fraction * 100 }} percent)
{% endmacro %}


{% macro databricks__tablesample_rows(n) %}
{#
    Sample specific number of rows.
#}
tablesample ({{ n }} rows)
{% endmacro %}


{# ============================================================================
   Type Casting - Databricks
   ============================================================================ #}

{% macro databricks__safe_cast(column, data_type) %}
{#
    Safe cast using Databricks TRY_CAST.
#}
try_cast({{ column }} as {{ data_type }})
{% endmacro %}


{% macro databricks__try_divide(numerator, denominator) %}
{#
    Safe division (returns NULL if denominator is 0).
#}
try_divide({{ numerator }}, {{ denominator }})
{% endmacro %}


{% macro databricks__try_add(value1, value2) %}
{#
    Safe addition.
#}
try_add({{ value1 }}, {{ value2 }})
{% endmacro %}


{% macro databricks__try_multiply(value1, value2) %}
{#
    Safe multiplication.
#}
try_multiply({{ value1 }}, {{ value2 }})
{% endmacro %}


{# ============================================================================
   String Functions - Databricks
   ============================================================================ #}

{% macro databricks__split(column, delimiter) %}
{#
    Split string into array.
#}
split({{ column }}, '{{ delimiter }}')
{% endmacro %}


{% macro databricks__concat_ws(delimiter, columns) %}
{#
    Concatenate columns with separator.
#}
concat_ws('{{ delimiter }}', {{ columns | join(', ') }})
{% endmacro %}


{% macro databricks__substring(column, start, length=none) %}
{#
    Get substring.
#}
{% if length %}
substring({{ column }}, {{ start }}, {{ length }})
{% else %}
substring({{ column }}, {{ start }})
{% endif %}
{% endmacro %}


{% macro databricks__reverse(column) %}
{#
    Reverse string.
#}
reverse({{ column }})
{% endmacro %}


{% macro databricks__initcap(column) %}
{#
    Capitalize first letter of each word.
#}
initcap({{ column }})
{% endmacro %}


{# ============================================================================
   Array Functions - Databricks
   ============================================================================ #}

{% macro databricks__array_contains(array_column, value) %}
{#
    Check if array contains value.
#}
array_contains({{ array_column }}, {{ truthound.quote_value(value) }})
{% endmacro %}


{% macro databricks__array_size(array_column) %}
{#
    Get array size.
#}
size({{ array_column }})
{% endmacro %}


{% macro databricks__array_distinct(array_column) %}
{#
    Get distinct array elements.
#}
array_distinct({{ array_column }})
{% endmacro %}


{% macro databricks__array_sort(array_column) %}
{#
    Sort array elements.
#}
array_sort({{ array_column }})
{% endmacro %}


{% macro databricks__array_join(array_column, delimiter) %}
{#
    Join array elements into string.
#}
array_join({{ array_column }}, '{{ delimiter }}')
{% endmacro %}


{% macro databricks__explode(array_column) %}
{#
    Explode array into rows.
#}
explode({{ array_column }})
{% endmacro %}


{% macro databricks__collect_list(column) %}
{#
    Collect values into array (including duplicates).
#}
collect_list({{ column }})
{% endmacro %}


{% macro databricks__collect_set(column) %}
{#
    Collect distinct values into array.
#}
collect_set({{ column }})
{% endmacro %}


{# ============================================================================
   Map Functions - Databricks
   ============================================================================ #}

{% macro databricks__map_keys(map_column) %}
{#
    Get map keys as array.
#}
map_keys({{ map_column }})
{% endmacro %}


{% macro databricks__map_values(map_column) %}
{#
    Get map values as array.
#}
map_values({{ map_column }})
{% endmacro %}


{% macro databricks__map_contains_key(map_column, key) %}
{#
    Check if map contains key.
#}
map_contains_key({{ map_column }}, {{ truthound.quote_value(key) }})
{% endmacro %}


{% macro databricks__element_at(map_column, key) %}
{#
    Get value for key from map.
#}
element_at({{ map_column }}, {{ truthound.quote_value(key) }})
{% endmacro %}


{# ============================================================================
   JSON Functions - Databricks
   ============================================================================ #}

{% macro databricks__json_extract(column, path) %}
{#
    Extract value from JSON column.

    Parameters
    ----------
    column : str
        JSON column name
    path : str
        JSON path (e.g., '$.user.name')
#}
get_json_object({{ column }}, '{{ path }}')
{% endmacro %}


{% macro databricks__from_json(column, schema) %}
{#
    Parse JSON string to struct.
#}
from_json({{ column }}, '{{ schema }}')
{% endmacro %}


{% macro databricks__to_json(column) %}
{#
    Convert struct to JSON string.
#}
to_json({{ column }})
{% endmacro %}


{% macro databricks__is_valid_json(column) %}
{#
    Check if column contains valid JSON.
#}
try_parse_json({{ column }}) is not null
{% endmacro %}


{% macro databricks__json_tuple(column, keys) %}
{#
    Extract multiple values from JSON.
#}
json_tuple({{ column }}, {{ keys | map('tojson') | join(', ') }})
{% endmacro %}


{# ============================================================================
   NULL Handling - Databricks
   ============================================================================ #}

{% macro databricks__nvl(column, default_value) %}
{#
    Return default if NULL.
#}
nvl({{ column }}, {{ default_value }})
{% endmacro %}


{% macro databricks__nvl2(column, value_if_not_null, value_if_null) %}
{#
    NVL2 function.
#}
nvl2({{ column }}, {{ value_if_not_null }}, {{ value_if_null }})
{% endmacro %}


{% macro databricks__nanvl(column, default_value) %}
{#
    Return default if NaN.
#}
nanvl({{ column }}, {{ default_value }})
{% endmacro %}


{# ============================================================================
   Conditional Logic - Databricks
   ============================================================================ #}

{% macro databricks__if(condition, true_value, false_value) %}
{#
    Databricks IF expression.
#}
if({{ condition }}, {{ true_value }}, {{ false_value }})
{% endmacro %}


{% macro databricks__iff(condition, true_value, false_value) %}
{#
    Alias for IF.
#}
iff({{ condition }}, {{ true_value }}, {{ false_value }})
{% endmacro %}


{# ============================================================================
   Analytics Functions - Databricks
   ============================================================================ #}

{% macro databricks__approx_count_distinct(column) %}
{#
    Approximate count distinct.
#}
approx_count_distinct({{ column }})
{% endmacro %}


{% macro databricks__percentile_approx(column, percentile) %}
{#
    Approximate percentile.
#}
percentile_approx({{ column }}, {{ percentile }})
{% endmacro %}


{% macro databricks__histogram_numeric(column, num_bins) %}
{#
    Compute histogram.
#}
histogram_numeric({{ column }}, {{ num_bins }})
{% endmacro %}
