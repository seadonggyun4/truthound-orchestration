"""Data Quality Ops for Dagster.

This module provides Dagster ops for data quality operations.
Ops are the fundamental unit of computation in Dagster and can
be composed into jobs and graphs.

Available Ops:
    - data_quality_check_op: Execute validation checks
    - data_quality_profile_op: Run data profiling
    - data_quality_learn_op: Learn validation rules

Op Factories:
    - create_check_op: Create customized check op
    - create_profile_op: Create customized profile op
    - create_learn_op: Create customized learn op

Example:
    >>> from dagster import job
    >>> from truthound_dagster.ops import (
    ...     data_quality_check_op,
    ...     create_check_op,
    ... )
    >>>
    >>> # Use pre-built op
    >>> @job
    ... def quality_job():
    ...     data_quality_check_op()
    >>>
    >>> # Or create customized op
    >>> custom_check = create_check_op(
    ...     name="users_check",
    ...     rules=[{"column": "id", "type": "not_null"}],
    ... )
"""

from truthound_dagster.ops.base import (
    BaseOpConfig,
    CheckOpConfig,
    LearnOpConfig,
    ProfileOpConfig,
    DriftOpConfig,
    AnomalyOpConfig,
    # Presets
    DEFAULT_CHECK_CONFIG,
    STRICT_CHECK_CONFIG,
    LENIENT_CHECK_CONFIG,
    AUTO_SCHEMA_CHECK_CONFIG,
    DEFAULT_PROFILE_CONFIG,
    MINIMAL_PROFILE_CONFIG,
    DEFAULT_LEARN_CONFIG,
    HIGH_CONFIDENCE_LEARN_CONFIG,
    DEFAULT_DRIFT_CONFIG,
    STRICT_DRIFT_CONFIG,
    LENIENT_DRIFT_CONFIG,
    DEFAULT_ANOMALY_CONFIG,
    STRICT_ANOMALY_CONFIG,
    LENIENT_ANOMALY_CONFIG,
)
from truthound_dagster.ops.check import (
    create_check_op,
    data_quality_check_op,
)
from truthound_dagster.ops.profile import (
    create_profile_op,
    data_quality_profile_op,
)
from truthound_dagster.ops.learn import (
    create_learn_op,
    data_quality_learn_op,
)
from truthound_dagster.ops.drift import (
    create_drift_op,
    data_quality_drift_op,
)
from truthound_dagster.ops.anomaly import (
    create_anomaly_op,
    data_quality_anomaly_op,
)

__all__ = [
    # Configuration
    "BaseOpConfig",
    "CheckOpConfig",
    "ProfileOpConfig",
    "LearnOpConfig",
    "DriftOpConfig",
    "AnomalyOpConfig",
    # Check presets
    "DEFAULT_CHECK_CONFIG",
    "STRICT_CHECK_CONFIG",
    "LENIENT_CHECK_CONFIG",
    "AUTO_SCHEMA_CHECK_CONFIG",
    # Profile presets
    "DEFAULT_PROFILE_CONFIG",
    "MINIMAL_PROFILE_CONFIG",
    # Learn presets
    "DEFAULT_LEARN_CONFIG",
    "HIGH_CONFIDENCE_LEARN_CONFIG",
    # Drift presets
    "DEFAULT_DRIFT_CONFIG",
    "STRICT_DRIFT_CONFIG",
    "LENIENT_DRIFT_CONFIG",
    # Anomaly presets
    "DEFAULT_ANOMALY_CONFIG",
    "STRICT_ANOMALY_CONFIG",
    "LENIENT_ANOMALY_CONFIG",
    # Check ops
    "data_quality_check_op",
    "create_check_op",
    # Profile ops
    "data_quality_profile_op",
    "create_profile_op",
    # Learn ops
    "data_quality_learn_op",
    "create_learn_op",
    # Drift ops
    "data_quality_drift_op",
    "create_drift_op",
    # Anomaly ops
    "data_quality_anomaly_op",
    "create_anomaly_op",
]
