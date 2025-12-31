"""Utility modules for truthound-prefect.

This package provides:
- Type definitions and output containers
- Custom exceptions
- Serialization utilities
- Helper functions
"""

from truthound_prefect.utils.exceptions import (
    BlockError,
    ConfigurationError,
    DataQualityError,
    EngineError,
    SLAViolationError,
)
from truthound_prefect.utils.helpers import (
    calculate_timeout,
    create_quality_metadata,
    create_run_context,
    format_count,
    format_duration,
    format_percentage,
    get_data_info,
    merge_results,
    parse_rules_from_string,
    summarize_check_result,
    summarize_learn_result,
    summarize_profile_result,
)
from truthound_prefect.utils.serialization import (
    ResultSerializer,
    deserialize_result,
    serialize_result,
    to_prefect_artifact,
)
from truthound_prefect.utils.types import (
    AnyDataQualityOutput,
    AnyLearnOutput,
    AnyProfileOutput,
    AnyQualityCheckOutput,
    DataQualityOutput,
    LearnOutput,
    OperationStatus,
    OperationType,
    ProfileOutput,
    QualityCheckMode,
    QualityCheckOutput,
)

__all__ = [
    # Exceptions
    "DataQualityError",
    "ConfigurationError",
    "EngineError",
    "BlockError",
    "SLAViolationError",
    # Types
    "QualityCheckMode",
    "OperationType",
    "OperationStatus",
    "DataQualityOutput",
    "QualityCheckOutput",
    "ProfileOutput",
    "LearnOutput",
    "AnyDataQualityOutput",
    "AnyQualityCheckOutput",
    "AnyProfileOutput",
    "AnyLearnOutput",
    # Serialization
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "to_prefect_artifact",
    # Helpers
    "format_duration",
    "format_percentage",
    "format_count",
    "summarize_check_result",
    "summarize_profile_result",
    "summarize_learn_result",
    "create_quality_metadata",
    "get_data_info",
    "calculate_timeout",
    "parse_rules_from_string",
    "merge_results",
    "create_run_context",
]
