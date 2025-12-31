{#
    Redshift Adapter Implementations
    ================================

    Amazon Redshift-specific optimizations for Truthound macros.

    Key considerations:
    - Redshift is PostgreSQL-based but has limitations
    - No QUALIFY clause (use subqueries)
    - Limited regex support (POSIX regex)
    - Different JSON functions (JSON_EXTRACT_PATH_TEXT)
#}


{# ============================================================================
   Unique Check - Redshift
   ============================================================================ #}

{% macro redshift__rule_unique_optimized(model, column) %}
{#
    Redshift unique check using window function in subquery.

    Redshift doesn't support QUALIFY, so we use a subquery approach.
#}
select *
from (
    select
        *,
        count(*) over (partition by {{ column }}) as _truthound_dup_count
    from {{ model }}
    where {{ column }} is not null
) subq
where _truthound_dup_count > 1
{% endmacro %}


{% macro redshift__row_number_duplicates(model, column) %}
{#
    Find duplicates using window function in subquery.
#}
select *
from (
    select
        *,
        row_number() over (partition by {{ column }} order by {{ column }}) as _truthound_row_num,
        count(*) over (partition by {{ column }}) as _truthound_dup_count
    from {{ model }}
    where {{ column }} is not null
) subq
where _truthound_dup_count > 1
{% endmacro %}


{# ============================================================================
   Regex Functions - Redshift
   ============================================================================ #}

{% macro redshift__regex_match(column, pattern) %}
{#
    Redshift regex matching using POSIX operator.

    Note: Redshift uses PostgreSQL-style POSIX regex.
#}
{{ column }} ~ '{{ pattern }}'
{% endmacro %}


{% macro redshift__regex_replace(column, pattern, replacement) %}
{#
    Replace pattern matches in column.
#}
regexp_replace({{ column }}, '{{ pattern }}', '{{ replacement }}')
{% endmacro %}


{% macro redshift__regex_substr(column, pattern, start_pos=1, occurrence=1) %}
{#
    Extract substring matching regex pattern.

    Parameters
    ----------
    column : str
        Column name
    pattern : str
        Regex pattern
    start_pos : int
        Starting position (1-based)
    occurrence : int
        Which occurrence to match
#}
regexp_substr({{ column }}, '{{ pattern }}', {{ start_pos }}, {{ occurrence }})
{% endmacro %}


{% macro redshift__regex_count(column, pattern) %}
{#
    Count pattern matches in column.
#}
regexp_count({{ column }}, '{{ pattern }}')
{% endmacro %}


{% macro redshift__regex_instr(column, pattern) %}
{#
    Find position of first match.
#}
regexp_instr({{ column }}, '{{ pattern }}')
{% endmacro %}


{# ============================================================================
   Date/Time Functions - Redshift
   ============================================================================ #}

{% macro redshift__date_diff(date1, date2, unit='day') %}
{#
    Calculate difference between two dates.
#}
datediff({{ unit }}, {{ date2 }}, {{ date1 }})
{% endmacro %}


{% macro redshift__date_trunc(column, unit='day') %}
{#
    Truncate date to specified precision.
#}
date_trunc('{{ unit }}', {{ column }})
{% endmacro %}


{% macro redshift__date_add(column, interval_value, interval_unit='day') %}
{#
    Add interval to date.
#}
dateadd({{ interval_unit }}, {{ interval_value }}, {{ column }})
{% endmacro %}


{% macro redshift__to_date(column, format) %}
{#
    Convert string to date.

    Note: Redshift uses different format codes than standard.
#}
to_date({{ column }}, '{{ format }}')
{% endmacro %}


{% macro redshift__to_timestamp(column, format) %}
{#
    Convert string to timestamp.
#}
to_timestamp({{ column }}, '{{ format }}')
{% endmacro %}


{% macro redshift__current_timestamp() %}
{#
    Get current timestamp in Redshift.
#}
getdate()
{% endmacro %}


{% macro redshift__sysdate() %}
{#
    Get current date/time (Redshift specific).
#}
sysdate
{% endmacro %}


{# ============================================================================
   Sampling - Redshift
   ============================================================================ #}

{% macro redshift__limit_sample(n) %}
{#
    Random sampling with row limit.

    Note: Redshift doesn't have native TABLESAMPLE,
    so we use ORDER BY RANDOM() LIMIT.
#}
order by random()
limit {{ n }}
{% endmacro %}


{# ============================================================================
   Type Casting - Redshift
   ============================================================================ #}

{% macro redshift__safe_cast(column, data_type) %}
{#
    Safe cast for Redshift.

    Note: Redshift doesn't have TRY_CAST, so we use CASE with validation.
#}
{% if data_type | lower in ['integer', 'int', 'bigint', 'smallint'] %}
case
    when {{ column }} ~ '^-?[0-9]+$' then {{ column }}::{{ data_type }}
    else null
end
{% elif data_type | lower in ['float', 'double precision', 'real', 'numeric', 'decimal'] %}
case
    when {{ column }} ~ '^-?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$' then {{ column }}::{{ data_type }}
    else null
end
{% elif data_type | lower == 'boolean' %}
case
    when lower({{ column }}) in ('true', 't', 'yes', 'y', '1') then true
    when lower({{ column }}) in ('false', 'f', 'no', 'n', '0') then false
    else null
end
{% else %}
{{ column }}::{{ data_type }}
{% endif %}
{% endmacro %}


{# ============================================================================
   String Functions - Redshift
   ============================================================================ #}

{% macro redshift__length(column) %}
{#
    String length (Redshift uses LEN).
#}
len({{ column }})
{% endmacro %}


{% macro redshift__char_length(column) %}
{#
    Character length.
#}
char_length({{ column }})
{% endmacro %}


{% macro redshift__octet_length(column) %}
{#
    Byte length.
#}
octet_length({{ column }})
{% endmacro %}


{% macro redshift__split_part(column, delimiter, part_num) %}
{#
    Split string and get specific part.
#}
split_part({{ column }}, '{{ delimiter }}', {{ part_num }})
{% endmacro %}


{% macro redshift__concat(columns, separator='') %}
{#
    Concatenate columns.

    Note: Redshift || operator or CONCAT function.
#}
{% if separator == '' %}
{{ columns | join(' || ') }}
{% else %}
{{ columns | join(" || '" ~ separator ~ "' || ") }}
{% endif %}
{% endmacro %}


{% macro redshift__left(column, n) %}
{#
    Get leftmost n characters.
#}
left({{ column }}, {{ n }})
{% endmacro %}


{% macro redshift__right(column, n) %}
{#
    Get rightmost n characters.
#}
right({{ column }}, {{ n }})
{% endmacro %}


{# ============================================================================
   JSON Functions - Redshift
   ============================================================================ #}

{% macro redshift__json_extract(column, path) %}
{#
    Extract value from JSON column.

    Parameters
    ----------
    column : str
        JSON column name
    path : str
        Comma-separated path elements (e.g., 'user,name')

    Note
    ----
    Redshift uses JSON_EXTRACT_PATH_TEXT with comma-separated path.
#}
{% set path_parts = path.split('.') %}
json_extract_path_text({{ column }}, {{ path_parts | map('string') | map('tojson') | join(', ') }})
{% endmacro %}


{% macro redshift__json_extract_array_element(column, index) %}
{#
    Extract array element from JSON.
#}
json_extract_array_element_text({{ column }}, {{ index }})
{% endmacro %}


{% macro redshift__is_valid_json(column) %}
{#
    Check if column contains valid JSON.

    Note: Redshift doesn't have a direct function, so we try to extract.
#}
case
    when {{ column }} is null then false
    when json_extract_path_text({{ column }}, '') is not null then true
    else false
end
{% endmacro %}


{% macro redshift__json_typeof(column) %}
{#
    Get JSON value type.
#}
json_typeof({{ column }})
{% endmacro %}


{# ============================================================================
   NULL Handling - Redshift
   ============================================================================ #}

{% macro redshift__nvl(column, default_value) %}
{#
    Redshift NVL (same as COALESCE for 2 args).
#}
nvl({{ column }}, {{ default_value }})
{% endmacro %}


{% macro redshift__nvl2(column, value_if_not_null, value_if_null) %}
{#
    Redshift NVL2.
#}
nvl2({{ column }}, {{ value_if_not_null }}, {{ value_if_null }})
{% endmacro %}


{% macro redshift__decode(column, mappings, default_value=none) %}
{#
    Redshift DECODE function.

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


{# ============================================================================
   Window Functions - Redshift
   ============================================================================ #}

{% macro redshift__lead(column, offset=1, default=none) %}
{#
    LEAD window function.
#}
{% if default is none %}
lead({{ column }}, {{ offset }}) over ()
{% else %}
lead({{ column }}, {{ offset }}, {{ default }}) over ()
{% endif %}
{% endmacro %}


{% macro redshift__lag(column, offset=1, default=none) %}
{#
    LAG window function.
#}
{% if default is none %}
lag({{ column }}, {{ offset }}) over ()
{% else %}
lag({{ column }}, {{ offset }}, {{ default }}) over ()
{% endif %}
{% endmacro %}


{% macro redshift__first_value(column) %}
{#
    FIRST_VALUE window function.
#}
first_value({{ column }}) over ()
{% endmacro %}


{% macro redshift__last_value(column) %}
{#
    LAST_VALUE window function.
#}
last_value({{ column }}) over ()
{% endmacro %}


{# ============================================================================
   Compression/Encoding - Redshift Specific
   ============================================================================ #}

{% macro redshift__approximate_count_distinct(column) %}
{#
    Approximate count distinct (faster for large datasets).
#}
approximate count(distinct {{ column }})
{% endmacro %}


{% macro redshift__median(column) %}
{#
    Calculate median.
#}
median({{ column }})
{% endmacro %}


{% macro redshift__percentile_cont(column, percentile) %}
{#
    Calculate percentile.
#}
percentile_cont({{ percentile }}) within group (order by {{ column }})
{% endmacro %}
