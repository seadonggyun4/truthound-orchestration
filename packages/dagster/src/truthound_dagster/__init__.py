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

# Assets
from truthound_dagster.assets import (
    ProfileAssetConfig,
    # Configuration
    QualityAssetConfig,
    QualityCheckMode,
    create_asset_check,
    # Factory functions
    create_quality_asset,
    create_quality_check_asset,
    profiled_asset,
    quality_asset_check,
    # Decorators
    quality_checked_asset,
)

# Ops
from truthound_dagster.ops import (
    LENIENT_CHECK_CONFIG,
    # Presets
    STRICT_CHECK_CONFIG,
    # Configuration
    CheckOpConfig,
    LearnOpConfig,
    ProfileOpConfig,
    # Factory functions
    create_check_op,
    create_learn_op,
    create_profile_op,
    # Pre-built ops
    data_quality_check_op,
    data_quality_learn_op,
    data_quality_profile_op,
    merge_after_approval_op,
    pull_snapshot_op,
    release_tag_flow_op,
    rollback_flow_op,
    scheduled_sync_op,
    scheduled_validation_op,
    validate_branch_op,
)

# Resources
from truthound_dagster.resources import (
    # Presets
    DEFAULT_ENGINE_CONFIG,
    PARALLEL_ENGINE_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
    # Main resources
    DataQualityResource,
    DepotResource,
    DepotResourceConfig,
    EngineResource,
    # Configuration
    EngineResourceConfig,
)

# SLA
from truthound_dagster.sla import (
    # Presets
    DEFAULT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    AlertLevel,
    CompositeSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    # Configuration
    SLAConfig,
    # Hooks
    SLAHook,
    SLAMetrics,
    # Monitor
    SLAMonitor,
    SLARegistry,
    # Resource
    SLAResource,
    SLAResourceConfig,
    SLAViolation,
    SLAViolationType,
    get_sla_registry,
    reset_sla_registry,
)

# Utils
from truthound_dagster.utils import (
    ConfigurationError,
    # Exceptions
    DataQualityError,
    # Types
    DataQualityOutput,
    EngineError,
    LearnOutput,
    ProfileOutput,
    QualityCheckOutput,
    # Serialization
    ResultSerializer,
    SLAViolationError,
    create_quality_metadata,
    deserialize_result,
    # Helpers
    format_duration,
    format_percentage,
    serialize_depot_result,
    serialize_result,
    summarize_check_result,
    to_dagster_depot_metadata,
    to_dagster_metadata,
)
from truthound_dagster.version import __version__, __version_info__

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Resources
    "DataQualityResource",
    "DepotResource",
    "DepotResourceConfig",
    "EngineResource",
    "EngineResourceConfig",
    "DEFAULT_ENGINE_CONFIG",
    "PARALLEL_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    # Ops
    "data_quality_check_op",
    "data_quality_profile_op",
    "data_quality_learn_op",
    "pull_snapshot_op",
    "validate_branch_op",
    "merge_after_approval_op",
    "scheduled_sync_op",
    "scheduled_validation_op",
    "release_tag_flow_op",
    "rollback_flow_op",
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
    "quality_asset_check",
    "create_quality_asset",
    "create_quality_check_asset",
    "create_asset_check",
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
    "serialize_depot_result",
    "deserialize_result",
    "to_dagster_metadata",
    "to_dagster_depot_metadata",
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
