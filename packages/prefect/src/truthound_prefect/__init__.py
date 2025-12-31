"""Truthound Prefect Integration.

This package provides Prefect integration for data quality operations
with Truthound and other data quality engines.

Main Components:
    - Blocks: Prefect Blocks for engine and SLA configuration
    - Tasks: Prefect Tasks for check, profile, and learn operations
    - Flows: Flow decorators and factory functions
    - SLA: Service Level Agreement monitoring

Quick Start:
    >>> from prefect import flow
    >>> from truthound_prefect import DataQualityBlock, data_quality_check_task
    >>>
    >>> @flow
    ... async def validate_data():
    ...     block = DataQualityBlock(engine_name="truthound", auto_schema=True)
    ...     data = load_data()
    ...     result = await data_quality_check_task(data=data, block=block)
    ...     return result

Using Decorators:
    >>> from truthound_prefect import quality_checked_flow
    >>>
    >>> @quality_checked_flow(
    ...     rules=[{"type": "not_null", "column": "id"}],
    ...     fail_on_error=True,
    ... )
    ... async def process_users():
    ...     return load_and_transform_users()

Using SLA Monitoring:
    >>> from truthound_prefect import SLABlock, SLAConfig
    >>>
    >>> sla_block = SLABlock(max_failure_rate=0.05, max_execution_time_seconds=300)
    >>> violations = sla_block.check_result(check_result)
"""

from truthound_prefect.version import __version__, __version_info__

# Blocks
from truthound_prefect.blocks import (
    AUTO_SCHEMA_ENGINE_CONFIG,
    DEFAULT_BLOCK_CONFIG,
    DEFAULT_ENGINE_CONFIG,
    DEVELOPMENT_BLOCK_CONFIG,
    DEVELOPMENT_ENGINE_CONFIG,
    PARALLEL_ENGINE_CONFIG,
    PRODUCTION_BLOCK_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
    BaseBlock,
    BlockConfig,
    DataQualityBlock,
    EngineBlock,
    EngineBlockConfig,
)

# Tasks
from truthound_prefect.tasks import (
    AUTO_SCHEMA_CHECK_CONFIG,
    DEFAULT_CHECK_CONFIG,
    DEFAULT_LEARN_CONFIG,
    DEFAULT_PROFILE_CONFIG,
    FULL_PROFILE_CONFIG,
    LENIENT_CHECK_CONFIG,
    MINIMAL_PROFILE_CONFIG,
    STRICT_CHECK_CONFIG,
    STRICT_LEARN_CONFIG,
    BaseTaskConfig,
    CheckTaskConfig,
    LearnTaskConfig,
    ProfileTaskConfig,
    auto_schema_check_task,
    create_check_task,
    create_learn_task,
    create_profile_task,
    data_quality_check_task,
    data_quality_learn_task,
    data_quality_profile_task,
    full_profile_task,
    lenient_check_task,
    minimal_profile_task,
    standard_learn_task,
    strict_check_task,
    strict_learn_task,
)

# Flows
from truthound_prefect.flows import (
    AUTO_SCHEMA_FLOW_CONFIG,
    DEFAULT_FLOW_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_QUALITY_FLOW_CONFIG,
    FULL_PIPELINE_CONFIG,
    LENIENT_QUALITY_FLOW_CONFIG,
    STRICT_QUALITY_FLOW_CONFIG,
    FlowConfig,
    PipelineFlowConfig,
    QualityFlowConfig,
    create_multi_table_quality_flows,
    create_pipeline_flow,
    create_quality_flow,
    create_validation_flow,
    profiled_flow,
    quality_checked_flow,
    validated_flow,
)

# SLA
from truthound_prefect.sla import (
    DEFAULT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    AlertLevel,
    CallbackSLAHook,
    CompositeSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    SLABlock,
    SLAConfig,
    SLAHook,
    SLAHookStats,
    SLAMetrics,
    SLAMonitor,
    SLARegistry,
    SLAViolation,
    SLAViolationType,
    get_sla_registry,
    reset_sla_registry,
)

# Utils
from truthound_prefect.utils import (
    AnyDataQualityOutput,
    AnyLearnOutput,
    AnyProfileOutput,
    AnyQualityCheckOutput,
    BlockError,
    ConfigurationError,
    DataQualityError,
    DataQualityOutput,
    EngineError,
    LearnOutput,
    OperationStatus,
    OperationType,
    ProfileOutput,
    QualityCheckMode,
    QualityCheckOutput,
    ResultSerializer,
    SLAViolationError,
    calculate_timeout,
    create_quality_metadata,
    create_run_context,
    deserialize_result,
    format_count,
    format_duration,
    format_percentage,
    get_data_info,
    merge_results,
    parse_rules_from_string,
    serialize_result,
    summarize_check_result,
    summarize_learn_result,
    summarize_profile_result,
    to_prefect_artifact,
)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Blocks
    "BlockConfig",
    "BaseBlock",
    "EngineBlockConfig",
    "EngineBlock",
    "DataQualityBlock",
    "DEFAULT_BLOCK_CONFIG",
    "PRODUCTION_BLOCK_CONFIG",
    "DEVELOPMENT_BLOCK_CONFIG",
    "DEFAULT_ENGINE_CONFIG",
    "PARALLEL_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    "DEVELOPMENT_ENGINE_CONFIG",
    "AUTO_SCHEMA_ENGINE_CONFIG",
    # Task configs
    "BaseTaskConfig",
    "CheckTaskConfig",
    "ProfileTaskConfig",
    "LearnTaskConfig",
    "DEFAULT_CHECK_CONFIG",
    "STRICT_CHECK_CONFIG",
    "LENIENT_CHECK_CONFIG",
    "AUTO_SCHEMA_CHECK_CONFIG",
    "DEFAULT_PROFILE_CONFIG",
    "MINIMAL_PROFILE_CONFIG",
    "FULL_PROFILE_CONFIG",
    "DEFAULT_LEARN_CONFIG",
    "STRICT_LEARN_CONFIG",
    # Tasks
    "data_quality_check_task",
    "data_quality_profile_task",
    "data_quality_learn_task",
    "create_check_task",
    "create_profile_task",
    "create_learn_task",
    "strict_check_task",
    "lenient_check_task",
    "auto_schema_check_task",
    "minimal_profile_task",
    "full_profile_task",
    "standard_learn_task",
    "strict_learn_task",
    # Flow configs
    "FlowConfig",
    "QualityFlowConfig",
    "PipelineFlowConfig",
    "DEFAULT_FLOW_CONFIG",
    "DEFAULT_QUALITY_FLOW_CONFIG",
    "STRICT_QUALITY_FLOW_CONFIG",
    "LENIENT_QUALITY_FLOW_CONFIG",
    "AUTO_SCHEMA_FLOW_CONFIG",
    "DEFAULT_PIPELINE_CONFIG",
    "FULL_PIPELINE_CONFIG",
    # Flow decorators
    "quality_checked_flow",
    "profiled_flow",
    "validated_flow",
    # Flow factories
    "create_quality_flow",
    "create_validation_flow",
    "create_pipeline_flow",
    "create_multi_table_quality_flows",
    # SLA types
    "AlertLevel",
    "SLAViolationType",
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
    # SLA components
    "SLAMonitor",
    "SLARegistry",
    "SLABlock",
    "get_sla_registry",
    "reset_sla_registry",
    # SLA hooks
    "SLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CompositeSLAHook",
    "CallbackSLAHook",
    "SLAHookStats",
    # Utils - Types
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
    # Utils - Exceptions
    "DataQualityError",
    "ConfigurationError",
    "EngineError",
    "BlockError",
    "SLAViolationError",
    # Utils - Serialization
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "to_prefect_artifact",
    # Utils - Helpers
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
