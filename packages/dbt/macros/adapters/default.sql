{#
    Default Adapter Implementations
    ===============================

    Default (PostgreSQL-compatible) implementations for adapter-specific macros.
    These serve as fallbacks when no adapter-specific implementation exists.

    This file contains rule implementations that may need adapter-specific
    optimizations. The default implementations work for PostgreSQL and
    most SQL-compliant databases.
#}


{# ============================================================================
   Unique Check Optimization
   ============================================================================ #}

{% macro default__rule_unique_optimized(model, column) %}
{#
    Default unique check using standard SQL.

    Uses a subquery join to find duplicates.
    This is the fallback when QUALIFY is not available.
#}
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


{# ============================================================================
   Row Number for Duplicate Detection
   ============================================================================ #}

{% macro default__row_number_duplicates(model, column) %}
{#
    Find duplicates using ROW_NUMBER window function.

    Returns rows where the same value appears more than once,
    with a row_num indicating the occurrence order.
#}
select *
from (
    select
        *,
        row_number() over (partition by {{ column }} order by {{ column }}) as _truthound_row_num,
        count(*) over (partition by {{ column }}) as _truthound_duplicate_count
    from {{ model }}
    where {{ column }} is not null
) subq
where _truthound_duplicate_count > 1
{% endmacro %}


{# ============================================================================
   Conditional Aggregation
   ============================================================================ #}

{% macro default__count_failures(model, condition) %}
{#
    Count rows matching a failure condition.

    Parameters
    ----------
    model : Relation
        The model to check
    condition : str
        SQL WHERE condition

    Returns
    -------
    SQL query
        Query returning failure count
#}
select count(*) as failure_count
from {{ model }}
where {{ condition }}
{% endmacro %}


{# ============================================================================
   Boolean Expression Helpers
   ============================================================================ #}

{% macro default__bool_or(expressions) %}
{#
    Combine multiple boolean expressions with OR.

    Parameters
    ----------
    expressions : list[str]
        List of SQL boolean expressions

    Returns
    -------
    str
        Combined expression
#}
({{ expressions | join(' or ') }})
{% endmacro %}


{% macro default__bool_and(expressions) %}
{#
    Combine multiple boolean expressions with AND.

    Parameters
    ----------
    expressions : list[str]
        List of SQL boolean expressions

    Returns
    -------
    str
        Combined expression
#}
({{ expressions | join(' and ') }})
{% endmacro %}


{# ============================================================================
   Null-Safe Comparison
   ============================================================================ #}

{% macro default__null_safe_equals(column1, column2) %}
{#
    Null-safe equality comparison.

    Returns true if both values are equal OR both are NULL.

    Parameters
    ----------
    column1 : str
        First column/expression
    column2 : str
        Second column/expression

    Returns
    -------
    SQL expression
        Boolean result
#}
({{ column1 }} = {{ column2 }} or ({{ column1 }} is null and {{ column2 }} is null))
{% endmacro %}


{% macro default__null_safe_not_equals(column1, column2) %}
{#
    Null-safe inequality comparison.

    Returns true if values are different (NULL counts as a value).
#}
not {{ truthound.null_safe_equals(column1, column2) }}
{% endmacro %}


{# ============================================================================
   String Operations
   ============================================================================ #}

{% macro default__trim_whitespace(column) %}
{#
    Trim leading and trailing whitespace.
#}
trim({{ column }})
{% endmacro %}


{% macro default__lower(column) %}
{#
    Convert to lowercase.
#}
lower({{ column }})
{% endmacro %}


{% macro default__upper(column) %}
{#
    Convert to uppercase.
#}
upper({{ column }})
{% endmacro %}


{# ============================================================================
   Numeric Operations
   ============================================================================ #}

{% macro default__abs(column) %}
{#
    Absolute value.
#}
abs({{ column }})
{% endmacro %}


{% macro default__floor(column) %}
{#
    Floor (round down).
#}
floor({{ column }})
{% endmacro %}


{% macro default__ceil(column) %}
{#
    Ceiling (round up).
#}
ceil({{ column }})
{% endmacro %}
