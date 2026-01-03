"""Utility modules for Kestra data quality integration.

This package provides utility functions, types, and classes used
throughout the truthound-kestra integration.

Modules:
    exceptions: Exception hierarchy for error handling.
    types: Type definitions, enums, and data classes.
    serialization: Result serialization utilities.
    helpers: Helper functions for common operations.

Example:
    >>> from truthound_kestra.utils import (
    ...     DataQualityError,
    ...     CheckStatus,
    ...     serialize_result,
    ...     load_data,
    ... )
"""

from truthound_kestra.utils.exceptions import (
    ConfigurationError,
    DataQualityError,
    EngineError,
    FlowError,
    OutputError,
    ScriptError,
    SerializationError,
    SLAViolationError,
)
from truthound_kestra.utils.helpers import (
    Timer,
    create_kestra_output,
    detect_data_source_type,
    format_count,
    format_duration,
    format_percentage,
    format_status_badge,
    get_execution_context,
    get_kestra_variable,
    get_logger,
    kestra_outputs,
    load_data,
    log_operation,
    merge_rules,
    parse_uri,
    timed,
    validate_rules,
)
from truthound_kestra.utils.serialization import (
    COMPACT_SERIALIZER_CONFIG,
    DEFAULT_SERIALIZER_CONFIG,
    FULL_SERIALIZER_CONFIG,
    JsonSerializer,
    MarkdownSerializer,
    ResultSerializer,
    SerializerConfig,
    YamlSerializer,
    deserialize_result,
    get_serializer,
    serialize_result,
    serialize_to_format,
)
from truthound_kestra.utils.types import (
    CheckStatus,
    ColumnProfile,
    DataSourceType,
    ExecutionContext,
    LearnedRule,
    MetadataDict,
    OperationType,
    OutputFormat,
    RuleDict,
    ScriptOutput,
    Severity,
    ValidationFailure,
)

__all__ = [
    # Exceptions
    "DataQualityError",
    "ConfigurationError",
    "EngineError",
    "ScriptError",
    "FlowError",
    "OutputError",
    "SLAViolationError",
    "SerializationError",
    # Types and Enums
    "CheckStatus",
    "Severity",
    "OperationType",
    "OutputFormat",
    "DataSourceType",
    "RuleDict",
    "MetadataDict",
    # Data classes
    "ScriptOutput",
    "ExecutionContext",
    "ValidationFailure",
    "ColumnProfile",
    "LearnedRule",
    # Serialization
    "ResultSerializer",
    "JsonSerializer",
    "YamlSerializer",
    "MarkdownSerializer",
    "SerializerConfig",
    "DEFAULT_SERIALIZER_CONFIG",
    "COMPACT_SERIALIZER_CONFIG",
    "FULL_SERIALIZER_CONFIG",
    "serialize_result",
    "deserialize_result",
    "serialize_to_format",
    "get_serializer",
    # Helpers
    "Timer",
    "timed",
    "get_logger",
    "log_operation",
    "load_data",
    "detect_data_source_type",
    "parse_uri",
    "format_duration",
    "format_percentage",
    "format_count",
    "format_status_badge",
    "create_kestra_output",
    "get_kestra_variable",
    "get_execution_context",
    "kestra_outputs",
    "validate_rules",
    "merge_rules",
]
