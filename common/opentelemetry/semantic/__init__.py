"""Semantic conventions for data quality telemetry.

This module provides standardized attribute names and values for
data quality operations, following OpenTelemetry semantic conventions.
"""

from common.opentelemetry.semantic.attributes import (
    CheckAttributes,
    CheckStatus,
    DataQualityAttributes,
    DatasetAttributes,
    EngineAttributes,
    LearnAttributes,
    OperationStatus,
    OperationType,
    ProfileAttributes,
    RuleAttributes,
    Severity,
    create_check_attributes,
    create_engine_attributes,
    create_learn_attributes,
    create_profile_attributes,
)

__all__ = [
    # Main attribute namespace
    "DataQualityAttributes",
    # Structured attribute classes
    "EngineAttributes",
    "CheckAttributes",
    "ProfileAttributes",
    "LearnAttributes",
    "RuleAttributes",
    "DatasetAttributes",
    # Enums
    "OperationType",
    "OperationStatus",
    "CheckStatus",
    "Severity",
    # Factory functions
    "create_engine_attributes",
    "create_check_attributes",
    "create_profile_attributes",
    "create_learn_attributes",
]
