{#
    Truthound Windowed / Incremental Check Macros
    =============================================

    First-party macros for partition-aware and incremental-quality execution.
#}

{% macro truthound_window_predicate(options={}) %}
{% set partition_by = options.get('partition_by') %}
{% set window_start = options.get('window_start') %}
{% set window_end = options.get('window_end') %}
{% set where_clause = options.get('where_clause', options.get('where')) %}

{% set predicates = [] %}
{% if where_clause %}
    {% do predicates.append(where_clause) %}
{% endif %}
{% if partition_by and window_start is not none %}
    {% do predicates.append(partition_by ~ " >= " ~ window_start) %}
{% endif %}
{% if partition_by and window_end is not none %}
    {% do predicates.append(partition_by ~ " < " ~ window_end) %}
{% endif %}

{% if predicates | length == 0 %}
    {{ return(none) }}
{% endif %}

{{ return(predicates | join(' and ')) }}
{% endmacro %}


{% macro truthound_windowed_check(model, rules, options={}) %}
{% set predicate = truthound.truthound_window_predicate(options) %}
{% set merged_options = options.copy() %}
{% do merged_options.update({'where_clause': predicate}) %}
{{ return(truthound.truthound_check(model, rules, merged_options)) }}
{% endmacro %}


{% macro truthound_incremental_check(model, rules, options={}) %}
{% set predicate = truthound.truthound_window_predicate(options) %}
{% if predicate is none and is_incremental() and options.get('partition_by') %}
    {% set predicate = options.get('partition_by') ~ " >= (select max(" ~ options.get('partition_by') ~ ") from " ~ this ~ ")" %}
{% endif %}
{% set merged_options = options.copy() %}
{% do merged_options.update({'where_clause': predicate}) %}
{{ return(truthound.truthound_check(model, rules, merged_options)) }}
{% endmacro %}
