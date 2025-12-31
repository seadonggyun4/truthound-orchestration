"""Utility Functions and Types for Dagster Integration.

This module provides utility functions, type definitions, and
helper classes for data quality operations in Dagster.

Available Utilities:
    - Serialization: Functions for serializing results
    - Types: Type definitions for Dagster integration
    - Exceptions: Custom exception types
    - Helpers: General helper functions

Example:
    >>> from truthound_dagster.utils import (
    ...     serialize_result,
    ...     DataQualityError,
    ... )
"""

from truthound_dagster.utils.serialization import (
    ResultSerializer,
    serialize_result,
    deserialize_result,
    to_dagster_metadata,
)
from truthound_dagster.utils.types import (
    DataQualityOutput,
    QualityCheckOutput,
    ProfileOutput,
    LearnOutput,
)
from truthound_dagster.utils.exceptions import (
    DataQualityError,
    ConfigurationError,
    EngineError,
    SLAViolationError,
)
from truthound_dagster.utils.helpers import (
    format_duration,
    format_percentage,
    format_count,
    format_timestamp,
    get_current_timestamp,
    parse_timestamp,
    truncate_string,
    merge_metadata,
    filter_metadata,
    safe_get,
    build_asset_key,
    summarize_check_result,
    summarize_profile_result,
    validate_rule_format,
    validate_rules,
    create_quality_metadata,
)

__all__ = [
    # Serialization
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "to_dagster_metadata",
    # Types
    "DataQualityOutput",
    "QualityCheckOutput",
    "ProfileOutput",
    "LearnOutput",
    # Exceptions
    "DataQualityError",
    "ConfigurationError",
    "EngineError",
    "SLAViolationError",
    # Helpers
    "format_duration",
    "format_percentage",
    "format_count",
    "format_timestamp",
    "get_current_timestamp",
    "parse_timestamp",
    "truncate_string",
    "merge_metadata",
    "filter_metadata",
    "safe_get",
    "build_asset_key",
    "summarize_check_result",
    "summarize_profile_result",
    "validate_rule_format",
    "validate_rules",
    "create_quality_metadata",
]
