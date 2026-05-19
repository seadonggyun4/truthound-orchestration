"""Utility functions and types for Mage Data Quality blocks.

This module provides exceptions, types, serialization utilities,
and helper functions for data quality operations.
"""

from truthound_mage.utils.exceptions import (
    BlockConfigurationError,
    BlockExecutionError,
    DataLoadError,
    DataQualityBlockError,
    SLAViolationError,
)
from truthound_mage.utils.helpers import (
    create_block_metadata,
    format_check_result,
    format_violations,
    get_data_size,
    validate_data_input,
)
from truthound_mage.utils.serialization import (
    deserialize_check_result,
    deserialize_learn_result,
    deserialize_profile_result,
    deserialize_result,
    serialize_check_result,
    serialize_depot_result,
    serialize_learn_result,
    serialize_profile_result,
    serialize_result,
)
from truthound_mage.utils.types import (
    BlockMetadata,
    DataQualityOutput,
)

__all__ = [
    # Exceptions
    "DataQualityBlockError",
    "BlockConfigurationError",
    "BlockExecutionError",
    "DataLoadError",
    "SLAViolationError",
    # Types
    "DataQualityOutput",
    "BlockMetadata",
    # Serialization
    "serialize_result",
    "serialize_depot_result",
    "deserialize_result",
    "serialize_check_result",
    "serialize_profile_result",
    "serialize_learn_result",
    "deserialize_check_result",
    "deserialize_profile_result",
    "deserialize_learn_result",
    # Helpers
    "format_check_result",
    "format_violations",
    "create_block_metadata",
    "get_data_size",
    "validate_data_input",
]
