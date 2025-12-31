{#
    Truthound Utility Macros
    ========================

    Cross-adapter utility macros for data quality checks.
    These macros abstract database-specific SQL syntax differences.

    Architecture:
    - Each utility macro uses adapter.dispatch() for cross-database compatibility
    - Default implementation works for PostgreSQL/standard SQL
    - Adapter-specific implementations in macros/adapters/

    Usage:
        {{ truthound.length('column_name') }}
        {{ truthound.regex_match('column_name', 'pattern') }}
        {{ truthound.current_timestamp() }}
#}


{# ============================================================================
   Configuration Utilities
   ============================================================================ #}

{% macro get_config(key, default=none) %}
{#
    Get Truthound configuration value with fallback.

    Parameters
    ----------
    key : str
        Configuration key to retrieve
    default : any
        Default value if key not found

    Returns
    -------
    any
        Configuration value or default

    Example
    -------
    {% set severity = truthound.get_config('default_severity', 'error') %}
#}
{% set truthound_config = var('truthound', {}) %}
{{ return(truthound_config.get(key, default)) }}
{% endmacro %}


{% macro get_pattern(pattern_name) %}
{#
    Get predefined regex pattern by name.

    Parameters
    ----------
    pattern_name : str
        Pattern name (email, uuid, url, phone_e164, ipv4)

    Returns
    -------
    str
        Regex pattern string

    Example
    -------
    {% set email_pattern = truthound.get_pattern('email') %}
#}
{% set patterns = truthound.get_config('patterns', {}) %}
{% set default_patterns = {
    'email': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$',
    'uuid': '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
    'url': '^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}',
    'phone_e164': '^\\+?[1-9]\\d{1,14}$',
    'ipv4': '^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
} %}
{{ return(patterns.get(pattern_name, default_patterns.get(pattern_name, ''))) }}
{% endmacro %}


{% macro debug_log(message) %}
{#
    Log debug message if debug mode is enabled.

    Parameters
    ----------
    message : str
        Message to log
#}
{% if truthound.get_config('debug_mode', false) %}
    {{ log('[TRUTHOUND DEBUG] ' ~ message, info=True) }}
{% endif %}
{% endmacro %}


{# ============================================================================
   String Functions
   ============================================================================ #}

{% macro length(column) %}
{#
    Get string length - cross-adapter compatible.

    Parameters
    ----------
    column : str
        Column name or expression

    Returns
    -------
    SQL expression
        Length of the string

    Example
    -------
    WHERE {{ truthound.length('name') }} > 10
#}
{{ return(adapter.dispatch('length', 'truthound')(column)) }}
{% endmacro %}


{% macro default__length(column) %}
length({{ column }})
{% endmacro %}


{% macro bigquery__length(column) %}
length({{ column }})
{% endmacro %}


{% macro snowflake__length(column) %}
length({{ column }})
{% endmacro %}


{% macro redshift__length(column) %}
len({{ column }})
{% endmacro %}


{% macro databricks__length(column) %}
length({{ column }})
{% endmacro %}


{# ============================================================================
   Regular Expression Functions
   ============================================================================ #}

{% macro regex_match(column, pattern) %}
{#
    Check if column matches regex pattern - cross-adapter compatible.

    Parameters
    ----------
    column : str
        Column name or expression
    pattern : str
        Regex pattern to match

    Returns
    -------
    SQL expression
        Boolean expression (true if matches)

    Example
    -------
    WHERE {{ truthound.regex_match('email', '^[a-z]+@') }}

    Note
    ----
    Pattern escaping may differ between adapters.
    Use truthound.escape_pattern() if needed.
#}
{{ return(adapter.dispatch('regex_match', 'truthound')(column, pattern)) }}
{% endmacro %}


{% macro default__regex_match(column, pattern) %}
{# PostgreSQL syntax #}
{{ column }} ~ '{{ pattern }}'
{% endmacro %}


{% macro bigquery__regex_match(column, pattern) %}
regexp_contains({{ column }}, r'{{ pattern }}')
{% endmacro %}


{% macro snowflake__regex_match(column, pattern) %}
regexp_like({{ column }}, '{{ pattern }}')
{% endmacro %}


{% macro redshift__regex_match(column, pattern) %}
{{ column }} ~ '{{ pattern }}'
{% endmacro %}


{% macro databricks__regex_match(column, pattern) %}
{{ column }} rlike '{{ pattern }}'
{% endmacro %}


{% macro duckdb__regex_match(column, pattern) %}
regexp_matches({{ column }}, '{{ pattern }}')
{% endmacro %}


{% macro regex_not_match(column, pattern) %}
{#
    Check if column does NOT match regex pattern.

    Parameters
    ----------
    column : str
        Column name or expression
    pattern : str
        Regex pattern

    Returns
    -------
    SQL expression
        Boolean expression (true if NOT matches)
#}
{{ return(adapter.dispatch('regex_not_match', 'truthound')(column, pattern)) }}
{% endmacro %}


{% macro default__regex_not_match(column, pattern) %}
not ({{ truthound.regex_match(column, pattern) }})
{% endmacro %}


{% macro bigquery__regex_not_match(column, pattern) %}
not regexp_contains({{ column }}, r'{{ pattern }}')
{% endmacro %}


{% macro snowflake__regex_not_match(column, pattern) %}
not regexp_like({{ column }}, '{{ pattern }}')
{% endmacro %}


{# ============================================================================
   Date/Time Functions
   ============================================================================ #}

{% macro current_timestamp() %}
{#
    Get current timestamp - cross-adapter compatible.

    Returns
    -------
    SQL expression
        Current timestamp

    Example
    -------
    WHERE created_at <= {{ truthound.current_timestamp() }}
#}
{{ return(adapter.dispatch('current_timestamp', 'truthound')()) }}
{% endmacro %}


{% macro default__current_timestamp() %}
current_timestamp
{% endmacro %}


{% macro bigquery__current_timestamp() %}
current_timestamp()
{% endmacro %}


{% macro snowflake__current_timestamp() %}
current_timestamp()
{% endmacro %}


{% macro redshift__current_timestamp() %}
getdate()
{% endmacro %}


{% macro databricks__current_timestamp() %}
current_timestamp()
{% endmacro %}


{% macro current_date() %}
{#
    Get current date - cross-adapter compatible.

    Returns
    -------
    SQL expression
        Current date (without time)
#}
{{ return(adapter.dispatch('current_date', 'truthound')()) }}
{% endmacro %}


{% macro default__current_date() %}
current_date
{% endmacro %}


{% macro bigquery__current_date() %}
current_date()
{% endmacro %}


{% macro snowflake__current_date() %}
current_date()
{% endmacro %}


{% macro is_valid_date_format(column, format) %}
{#
    Check if column value is a valid date in specified format.

    Parameters
    ----------
    column : str
        Column name
    format : str
        Expected date format (YYYY-MM-DD, etc.)

    Returns
    -------
    SQL expression
        Boolean expression (true if valid date)

    Note
    ----
    Format conversion varies by adapter:
    - YYYY -> %Y (BigQuery)
    - MM -> %m (BigQuery)
    - DD -> %d (BigQuery)
#}
{{ return(adapter.dispatch('is_valid_date_format', 'truthound')(column, format)) }}
{% endmacro %}


{% macro default__is_valid_date_format(column, format) %}
{# PostgreSQL: try to cast and check for error #}
(
    case
        when {{ column }} is null then true
        else {{ column }}::date is not null
    end
)
{% endmacro %}


{% macro snowflake__is_valid_date_format(column, format) %}
try_to_date({{ column }}, '{{ format }}') is not null
{% endmacro %}


{% macro bigquery__is_valid_date_format(column, format) %}
{# Convert standard format to BigQuery strftime format #}
{% set bq_format = format | replace("YYYY", "%Y") | replace("MM", "%m") | replace("DD", "%d") %}
safe.parse_date('{{ bq_format }}', {{ column }}) is not null
{% endmacro %}


{% macro redshift__is_valid_date_format(column, format) %}
{# Redshift doesn't have TRY_CAST, use exception handling workaround #}
({{ column }} is null or to_date({{ column }}, '{{ format }}') is not null)
{% endmacro %}


{% macro databricks__is_valid_date_format(column, format) %}
try_to_timestamp({{ column }}, '{{ format }}') is not null
{% endmacro %}


{# ============================================================================
   Sampling and Limiting
   ============================================================================ #}

{% macro limit_sample(n) %}
{#
    Generate sampling/limit clause - cross-adapter compatible.

    Parameters
    ----------
    n : int
        Number of rows to sample

    Returns
    -------
    SQL clause
        Sampling clause appropriate for the adapter

    Example
    -------
    SELECT * FROM table
    {{ truthound.limit_sample(1000) }}
#}
{{ return(adapter.dispatch('limit_sample', 'truthound')(n)) }}
{% endmacro %}


{% macro default__limit_sample(n) %}
order by random()
limit {{ n }}
{% endmacro %}


{% macro bigquery__limit_sample(n) %}
order by rand()
limit {{ n }}
{% endmacro %}


{% macro snowflake__limit_sample(n) %}
sample ({{ n }} rows)
{% endmacro %}


{% macro redshift__limit_sample(n) %}
order by random()
limit {{ n }}
{% endmacro %}


{% macro databricks__limit_sample(n) %}
order by rand()
limit {{ n }}
{% endmacro %}


{# ============================================================================
   Type Casting
   ============================================================================ #}

{% macro safe_cast(column, data_type) %}
{#
    Safely cast column to data type (returns NULL on failure).

    Parameters
    ----------
    column : str
        Column name or expression
    data_type : str
        Target data type (string, integer, float, date, timestamp)

    Returns
    -------
    SQL expression
        Casted value or NULL if cast fails

    Example
    -------
    SELECT {{ truthound.safe_cast('amount', 'float') }} as amount
#}
{{ return(adapter.dispatch('safe_cast', 'truthound')(column, data_type)) }}
{% endmacro %}


{% macro default__safe_cast(column, data_type) %}
{# PostgreSQL doesn't have TRY_CAST, use CASE with type check #}
{% if data_type == 'integer' %}
    case when {{ column }} ~ '^-?[0-9]+$' then {{ column }}::integer else null end
{% elif data_type == 'float' %}
    case when {{ column }} ~ '^-?[0-9]*\.?[0-9]+$' then {{ column }}::float else null end
{% else %}
    {{ column }}::{{ data_type }}
{% endif %}
{% endmacro %}


{% macro snowflake__safe_cast(column, data_type) %}
try_cast({{ column }} as {{ data_type }})
{% endmacro %}


{% macro bigquery__safe_cast(column, data_type) %}
safe_cast({{ column }} as {{ data_type }})
{% endmacro %}


{% macro databricks__safe_cast(column, data_type) %}
try_cast({{ column }} as {{ data_type }})
{% endmacro %}


{# ============================================================================
   Null Handling
   ============================================================================ #}

{% macro coalesce_empty(column, default_value="''") %}
{#
    Coalesce NULL and empty strings to default value.

    Parameters
    ----------
    column : str
        Column name
    default_value : str
        Value to use if NULL or empty

    Returns
    -------
    SQL expression
        Column value or default
#}
coalesce(nullif(trim({{ column }}), ''), {{ default_value }})
{% endmacro %}


{% macro is_null_or_empty(column) %}
{#
    Check if column is NULL or empty string.

    Parameters
    ----------
    column : str
        Column name

    Returns
    -------
    SQL expression
        Boolean (true if NULL or empty)
#}
({{ column }} is null or trim({{ column }}) = '')
{% endmacro %}


{# ============================================================================
   Value Quoting
   ============================================================================ #}

{% macro quote_value(value) %}
{#
    Quote a single value for SQL.

    Parameters
    ----------
    value : any
        Value to quote

    Returns
    -------
    str
        Quoted value
#}
{% if value is string %}
'{{ value | replace("'", "''") }}'
{% elif value is none %}
null
{% else %}
{{ value }}
{% endif %}
{% endmacro %}


{% macro quote_values(values) %}
{#
    Quote a list of values for SQL IN clause.

    Parameters
    ----------
    values : list
        List of values to quote

    Returns
    -------
    str
        Comma-separated quoted values

    Example
    -------
    WHERE status IN ({{ truthound.quote_values(['active', 'pending']) }})
    -- Output: WHERE status IN ('active', 'pending')
#}
{% set quoted = [] %}
{% for v in values %}
    {% do quoted.append(truthound.quote_value(v)) %}
{% endfor %}
{{ return(quoted | join(', ')) }}
{% endmacro %}


{# ============================================================================
   Error Handling
   ============================================================================ #}

{% macro raise_error(message) %}
{#
    Raise a compiler error with Truthound prefix.

    Parameters
    ----------
    message : str
        Error message
#}
{{ exceptions.raise_compiler_error('[TRUTHOUND] ' ~ message) }}
{% endmacro %}


{% macro warn(message) %}
{#
    Log a warning message with Truthound prefix.

    Parameters
    ----------
    message : str
        Warning message
#}
{{ log('[TRUTHOUND WARNING] ' ~ message, info=True) }}
{% endmacro %}
