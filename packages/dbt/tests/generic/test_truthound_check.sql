{% test truthound_check(model, rules, fail_on_first=false, sample_size=none, where=none) %}
{#
    Truthound Data Quality Generic Test
    ====================================

    A dbt generic test that validates data quality using Truthound rules.

    This test returns rows that fail any of the specified rules.
    If any rows are returned, the test fails.

    Parameters
    ----------
    model : Relation
        The dbt model or source to test

    rules : list[dict]
        List of rule definitions. Each rule requires:
        - check: str - Check type (not_null, unique, in_set, range, etc.)
        - column: str - Column to check (optional for some checks)
        - Additional parameters based on check type

    fail_on_first : bool (default: false)
        If true, stop and return after the first failure.
        Useful for faster feedback in large tables.

    sample_size : int (default: none)
        If specified, randomly sample this many rows for testing.
        Useful for very large tables where full scans are expensive.

    where : str (default: none)
        Additional WHERE clause to filter rows before testing.
        Example: "created_at > '2024-01-01'"

    Supported Check Types
    ---------------------
    Completeness:
        - not_null: Column must not be NULL
        - not_empty: Column must not be NULL or empty string

    Uniqueness:
        - unique: Column values must be unique
        - unique_combination: Combination of columns must be unique
          (use 'columns' instead of 'column')

    Set Membership:
        - in_set / accepted_values: Value must be in allowed set
          (requires 'values' parameter)
        - not_in_set / rejected_values: Value must not be in forbidden set
          (requires 'values' parameter)

    Numeric Ranges:
        - range / between: Value must be within range
          (requires 'min' and/or 'max' parameters)
        - positive: Value must be > 0
        - negative: Value must be < 0
        - non_negative: Value must be >= 0
        - greater_than / gt: Value must be > threshold
          (requires 'value' parameter)
        - less_than / lt: Value must be < threshold
          (requires 'value' parameter)

    String Patterns:
        - length: String length must be within range
          (requires 'min' and/or 'max' parameters)
        - regex / pattern: Value must match regex pattern
          (requires 'pattern' parameter)
        - email_format / email: Valid email format
        - url_format / url: Valid URL format
        - uuid_format / uuid: Valid UUID format
        - phone_format / phone: Valid E.164 phone format
        - ipv4_format / ipv4: Valid IPv4 address format

    Temporal:
        - not_future: Date/timestamp must not be in the future
        - not_past: Date/timestamp must not be in the past
        - date_format: Must match date format
          (optional 'format' parameter, default: 'YYYY-MM-DD')

    Referential:
        - referential_integrity / foreign_key: Value must exist in another table
          (requires 'to_model' and 'to_column' parameters)

    Custom:
        - expression / custom: Custom SQL expression
          (requires 'expression' parameter - TRUE = failure)

    Aggregate:
        - row_count_range: Table row count must be within range
          (requires 'min' and/or 'max' parameters)

    Returns
    -------
    SQL Query
        Query selecting rows that fail any check.
        Empty result = test passes.
        Any rows returned = test fails.

    Examples
    --------
    Basic usage in schema.yml:

    ```yaml
    models:
      - name: stg_users
        tests:
          - truthound_check:
              rules:
                - column: user_id
                  check: not_null
                - column: user_id
                  check: unique
                - column: email
                  check: email_format
    ```

    With configuration:

    ```yaml
    models:
      - name: stg_orders
        tests:
          - truthound_check:
              rules:
                - column: amount
                  check: positive
                - column: quantity
                  check: range
                  min: 1
                  max: 1000
                - column: status
                  check: in_set
                  values: ['pending', 'shipped', 'delivered']
              sample_size: 10000
              config:
                severity: warn
                tags: ['quality']
    ```

    Column-level test:

    ```yaml
    models:
      - name: dim_customers
        columns:
          - name: email
            tests:
              - truthound_check:
                  rules:
                    - check: email_format
    ```

    Referential integrity:

    ```yaml
    models:
      - name: fct_orders
        tests:
          - truthound_check:
              rules:
                - column: customer_id
                  check: referential_integrity
                  to_model: dim_customers
                  to_column: customer_id
    ```

    Custom expression:

    ```yaml
    models:
      - name: events
        tests:
          - truthound_check:
              rules:
                - check: expression
                  expression: "start_time > end_time"
                  message: "Start time must be before end time"
    ```
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality']
) }}

{# Build options dict #}
{% set options = {
    'fail_fast': fail_on_first,
    'sample_size': sample_size,
    'where_clause': where,
    'include_metadata': true
} %}

{# Generate and return the check query #}
{{ truthound.truthound_check(model, rules, options) }}

{% endtest %}


{% test truthound_not_null(model, column_name) %}
{#
    Convenience test for not_null check.

    Equivalent to:
    ```yaml
    - truthound_check:
        rules:
          - column: <column_name>
            check: not_null
    ```
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'completeness']
) }}

{{ truthound.rule_not_null(model, column_name) }}

{% endtest %}


{% test truthound_unique(model, column_name) %}
{#
    Convenience test for unique check.

    Equivalent to:
    ```yaml
    - truthound_check:
        rules:
          - column: <column_name>
            check: unique
    ```
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'uniqueness']
) }}

{{ truthound.rule_unique(model, column_name) }}

{% endtest %}


{% test truthound_accepted_values(model, column_name, values) %}
{#
    Convenience test for accepted values check.

    Equivalent to:
    ```yaml
    - truthound_check:
        rules:
          - column: <column_name>
            check: in_set
            values: <values>
    ```
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'validity']
) }}

{{ truthound.rule_in_set(model, column_name, values) }}

{% endtest %}


{% test truthound_relationships(model, column_name, to, field) %}
{#
    Convenience test for referential integrity.

    Compatible with dbt's built-in relationships test signature.

    Equivalent to:
    ```yaml
    - truthound_check:
        rules:
          - column: <column_name>
            check: referential_integrity
            to_model: <to>
            to_column: <field>
    ```
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'referential']
) }}

{{ truthound.rule_referential_integrity(model, column_name, to, field) }}

{% endtest %}


{% test truthound_email_format(model, column_name) %}
{#
    Convenience test for email format validation.
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'format']
) }}

{{ truthound.rule_email_format(model, column_name) }}

{% endtest %}


{% test truthound_uuid_format(model, column_name) %}
{#
    Convenience test for UUID format validation.
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'format']
) }}

{{ truthound.rule_uuid_format(model, column_name) }}

{% endtest %}


{% test truthound_positive(model, column_name) %}
{#
    Convenience test for positive values.
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'range']
) }}

{{ truthound.rule_positive(model, column_name) }}

{% endtest %}


{% test truthound_range(model, column_name, min=none, max=none) %}
{#
    Convenience test for numeric range.

    Parameters
    ----------
    column_name : str
        Column to check
    min : number, optional
        Minimum allowed value
    max : number, optional
        Maximum allowed value
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'range']
) }}

{{ truthound.rule_range(model, column_name, min, max) }}

{% endtest %}


{% test truthound_not_future(model, column_name) %}
{#
    Convenience test for dates not in future.
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'temporal']
) }}

{{ truthound.rule_not_future(model, column_name) }}

{% endtest %}


{% test truthound_regex(model, column_name, pattern) %}
{#
    Convenience test for regex pattern matching.

    Parameters
    ----------
    column_name : str
        Column to check
    pattern : str
        Regex pattern that values must match
#}

{{ config(
    severity = var('truthound', {}).get('default_severity', 'error'),
    tags = ['truthound', 'data-quality', 'pattern']
) }}

{{ truthound.rule_regex(model, column_name, pattern) }}

{% endtest %}
