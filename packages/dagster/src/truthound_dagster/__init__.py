"""Truthound Dagster Integration Package.

This package provides seamless integration between Dagster and data quality
engines (Truthound, Great Expectations, Pandera). It follows Dagster's
Software-Defined Assets philosophy while providing extensible abstractions.

Main Components:
    - Resources: DataQualityResource for engine access
    - Ops: Pre-built ops for check, profile, learn operations
    - Assets: Decorators and factories for quality-aware assets
    - SLA: SLA monitoring and violation detection
    - Utils: Serialization, types, and helpers

Quick Start:
    >>> from truthound_dagster import (
    ...     DataQualityResource,
    ...     data_quality_check_op,
    ...     quality_checked_asset,
    ... )
    >>>
    >>> # Define resources
    >>> defs = Definitions(
    ...     resources={"data_quality": DataQualityResource()},
    ...     assets=[my_quality_asset],
    ... )

Example - Quality-Checked Asset:
    >>> @quality_checked_asset(
    ...     rules=[
    ...         {"type": "not_null", "column": "id"},
    ...         {"type": "unique", "column": "email"},
    ...     ],
    ...     fail_on_error=True,
    ... )
    >>> def users(context) -> pl.DataFrame:
    ...     return pl.read_parquet("users.parquet")

Example - Using SLA Monitor:
    >>> from truthound_dagster import SLAResource, SLAConfig
    >>>
    >>> sla_config = SLAConfig(
    ...     max_failure_rate=0.01,
    ...     max_execution_time_seconds=300.0,
    ... )
    >>> sla = SLAResource(default_config=sla_config.to_dict())
"""

from truthound_dagster.version import __version__, __version_info__

# Resources
from truthound_dagster.resources import (
    # Main resources
    DataQualityResource,
    EngineResource,
    # Configuration
    EngineResourceConfig,
    # Presets
    DEFAULT_ENGINE_CONFIG,
    PARALLEL_ENGINE_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
)

# Ops
from truthound_dagster.ops import (
    # Pre-built ops
    data_quality_check_op,
    data_quality_profile_op,
    data_quality_learn_op,
    # Factory functions
    create_check_op,
    create_profile_op,
    create_learn_op,
    # Configuration
    CheckOpConfig,
    ProfileOpConfig,
    LearnOpConfig,
    # Presets
    STRICT_CHECK_CONFIG,
    LENIENT_CHECK_CONFIG,
)

# Assets
from truthound_dagster.assets import (
    # Decorators
    quality_checked_asset,
    profiled_asset,
    # Factory functions
    create_quality_asset,
    create_quality_check_asset,
    # Configuration
    QualityAssetConfig,
    ProfileAssetConfig,
    QualityCheckMode,
)

# SLA
from truthound_dagster.sla import (
    # Configuration
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
    AlertLevel,
    # Monitor
    SLAMonitor,
    SLARegistry,
    get_sla_registry,
    reset_sla_registry,
    # Resource
    SLAResource,
    SLAResourceConfig,
    # Hooks
    SLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
    # Presets
    DEFAULT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
)

# Utils
from truthound_dagster.utils import (
    # Serialization
    ResultSerializer,
    serialize_result,
    deserialize_result,
    to_dagster_metadata,
    # Types
    DataQualityOutput,
    QualityCheckOutput,
    ProfileOutput,
    LearnOutput,
    # Exceptions
    DataQualityError,
    ConfigurationError,
    EngineError,
    SLAViolationError,
    # Helpers
    format_duration,
    format_percentage,
    summarize_check_result,
    create_quality_metadata,
)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Resources
    "DataQualityResource",
    "EngineResource",
    "EngineResourceConfig",
    "DEFAULT_ENGINE_CONFIG",
    "PARALLEL_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    # Ops
    "data_quality_check_op",
    "data_quality_profile_op",
    "data_quality_learn_op",
    "create_check_op",
    "create_profile_op",
    "create_learn_op",
    "CheckOpConfig",
    "ProfileOpConfig",
    "LearnOpConfig",
    "STRICT_CHECK_CONFIG",
    "LENIENT_CHECK_CONFIG",
    # Assets
    "quality_checked_asset",
    "profiled_asset",
    "create_quality_asset",
    "create_quality_check_asset",
    "QualityAssetConfig",
    "ProfileAssetConfig",
    "QualityCheckMode",
    # SLA
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    "SLAViolationType",
    "AlertLevel",
    "SLAMonitor",
    "SLARegistry",
    "get_sla_registry",
    "reset_sla_registry",
    "SLAResource",
    "SLAResourceConfig",
    "SLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CompositeSLAHook",
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    # Utils
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "to_dagster_metadata",
    "DataQualityOutput",
    "QualityCheckOutput",
    "ProfileOutput",
    "LearnOutput",
    "DataQualityError",
    "ConfigurationError",
    "EngineError",
    "SLAViolationError",
    "format_duration",
    "format_percentage",
    "summarize_check_result",
    "create_quality_metadata",
]
