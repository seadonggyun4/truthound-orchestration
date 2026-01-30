"""Rule to SQL Converters.

This module provides protocol-based converters for transforming data quality
rules into database-specific SQL expressions. The design follows the adapter
pattern, working in conjunction with SQLAdapters to generate optimized SQL.

Architecture:
    ```
    Rule (dict)
        │
        ▼
    RuleConverter
        │
        ├── Uses SQLAdapter for database-specific syntax
        │
        ▼
    RuleSQL (SQL expression)
    ```

Supported Rule Types:
    - Completeness: not_null, not_empty
    - Uniqueness: unique, unique_combination
    - Validity: in_set, not_in_set, range, regex, email_format, etc.
    - Temporal: not_future, not_past, date_format
    - Referential: referential_integrity, foreign_key
    - Custom: expression, row_count_range
    - Drift: drift_mean, drift_stddev, drift_null_rate, drift_distinct_count, drift_row_count
    - Anomaly: anomaly_zscore, anomaly_iqr, anomaly_range

Usage:
    >>> from truthound_dbt.converters import get_converter
    >>> from truthound_dbt.adapters import get_adapter
    >>>
    >>> adapter = get_adapter("snowflake")
    >>> converter = get_converter()
    >>>
    >>> rule = {"type": "email_format", "column": "email"}
    >>> result = converter.convert(rule, adapter)
    >>> print(result.sql)
    # Returns SQL for email validation

Custom Converter:
    >>> from truthound_dbt.converters import RuleConverter, register_converter
    >>>
    >>> class MyConverter:
    ...     def convert(self, rule, adapter):
    ...         # Custom conversion logic
    ...         return RuleSQL(...)
    >>>
    >>> register_converter("custom", MyConverter())
"""

from truthound_dbt.converters.base import (
    # Protocol
    RuleConverter,
    # Types
    ConversionResult,
    RuleSQL,
    ConversionContext,
    # Exceptions
    ConversionError,
    UnsupportedRuleError,
    InvalidRuleError,
    # Registry
    RuleConverterRegistry,
    get_converter,
    get_converter_registry,
    register_converter,
    list_converters,
    reset_converter_registry,
)
from truthound_dbt.converters.rules import (
    # Converter
    StandardRuleConverter,
    # Rule Handlers
    RuleHandler,
    NotNullHandler,
    UniqueHandler,
    InSetHandler,
    RangeHandler,
    RegexHandler,
    EmailFormatHandler,
    DateFormatHandler,
    ReferentialIntegrityHandler,
    ExpressionHandler,
    # Handler Registry
    RuleHandlerRegistry,
    get_handler,
    register_handler,
    list_handlers,
)
from truthound_dbt.converters.drift import (
    # Drift Handlers
    DriftMeanHandler,
    DriftStddevHandler,
    DriftNullRateHandler,
    DriftDistinctCountHandler,
    DriftRowCountHandler,
    # Anomaly Handlers
    AnomalyZScoreHandler,
    AnomalyIQRHandler,
    AnomalyRangeHandler,
    # Handler Collections
    DRIFT_HANDLERS,
    ANOMALY_HANDLERS,
    ALL_HANDLERS,
    # Registration
    register_drift_anomaly_handlers,
    # Factory Functions
    get_drift_handler,
    get_anomaly_handler,
)

# Register drift and anomaly handlers with the global handler registry
register_drift_anomaly_handlers()

__all__ = [
    # Protocol
    "RuleConverter",
    # Types
    "ConversionResult",
    "RuleSQL",
    "ConversionContext",
    # Exceptions
    "ConversionError",
    "UnsupportedRuleError",
    "InvalidRuleError",
    # Registry
    "RuleConverterRegistry",
    "get_converter",
    "get_converter_registry",
    "register_converter",
    "list_converters",
    "reset_converter_registry",
    # Converter
    "StandardRuleConverter",
    # Rule Handlers
    "RuleHandler",
    "NotNullHandler",
    "UniqueHandler",
    "InSetHandler",
    "RangeHandler",
    "RegexHandler",
    "EmailFormatHandler",
    "DateFormatHandler",
    "ReferentialIntegrityHandler",
    "ExpressionHandler",
    # Drift Handlers
    "DriftMeanHandler",
    "DriftStddevHandler",
    "DriftNullRateHandler",
    "DriftDistinctCountHandler",
    "DriftRowCountHandler",
    # Anomaly Handlers
    "AnomalyZScoreHandler",
    "AnomalyIQRHandler",
    "AnomalyRangeHandler",
    # Handler Collections
    "DRIFT_HANDLERS",
    "ANOMALY_HANDLERS",
    "ALL_HANDLERS",
    # Registration
    "register_drift_anomaly_handlers",
    # Factory Functions
    "get_drift_handler",
    "get_anomaly_handler",
    # Handler Registry
    "RuleHandlerRegistry",
    "get_handler",
    "register_handler",
    "list_handlers",
]
