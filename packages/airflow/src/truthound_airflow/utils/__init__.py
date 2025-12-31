"""Utility Functions for Data Quality Operations.

This module provides utility functions for serialization,
connection management, and other common operations.

Components:
    - Serialization: XCom-compatible serialization utilities
    - Connection: Connection helper functions
    - Helpers: Common utility functions

Example:
    >>> from truthound_airflow.utils import (
    ...     serialize_result,
    ...     deserialize_result,
    ...     get_connection_config,
    ... )
    >>>
    >>> serialized = serialize_result(check_result)
    >>> result = deserialize_result(serialized)
"""

from truthound_airflow.utils.serialization import (
    ResultSerializer,
    serialize_result,
    deserialize_result,
    serialize_profile,
    deserialize_profile,
    serialize_learn_result,
    deserialize_learn_result,
    to_xcom_value,
    from_xcom_value,
)
from truthound_airflow.utils.connection import (
    ConnectionHelper,
    get_connection_config,
    parse_connection_uri,
    build_connection_uri,
    mask_sensitive_values,
)
from truthound_airflow.utils.helpers import (
    safe_get,
    merge_dicts,
    format_duration,
    format_percentage,
    truncate_string,
    chunk_list,
    retry_operation,
)

__all__ = [
    # Serialization
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "serialize_profile",
    "deserialize_profile",
    "serialize_learn_result",
    "deserialize_learn_result",
    "to_xcom_value",
    "from_xcom_value",
    # Connection
    "ConnectionHelper",
    "get_connection_config",
    "parse_connection_uri",
    "build_connection_uri",
    "mask_sensitive_values",
    # Helpers
    "safe_get",
    "merge_dicts",
    "format_duration",
    "format_percentage",
    "truncate_string",
    "chunk_list",
    "retry_operation",
]
