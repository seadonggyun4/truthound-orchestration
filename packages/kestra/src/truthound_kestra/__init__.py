"""Truthound Kestra Integration - Data quality for Kestra workflows.

This package provides comprehensive data quality integration for Kestra
workflow orchestration, including script execution, flow generation,
output handling, and SLA monitoring.

Main Components:
    scripts: Entry points for Kestra Python Script tasks.
    flows: Flow configuration and YAML generation.
    outputs: Kestra output handling.
    sla: SLA monitoring and alerting.
    utils: Shared utilities and types.

Quick Start:
    >>> # In Kestra Python Script task
    >>> from truthound_kestra import check_quality_script
    >>>
    >>> check_quality_script(
    ...     input_uri="{{ outputs.extract.uri }}",
    ...     rules=[{"type": "not_null", "column": "id"}],
    ...     fail_on_error=True
    ... )

Flow Generation:
    >>> from truthound_kestra import FlowConfig, generate_flow_yaml
    >>>
    >>> config = FlowConfig(
    ...     id="data_quality_check",
    ...     namespace="company.data",
    ... )
    >>> yaml_content = generate_flow_yaml(config)

SLA Monitoring:
    >>> from truthound_kestra import SLAConfig, SLAMonitor, LoggingSLAHook
    >>>
    >>> config = SLAConfig(min_pass_rate=0.95)
    >>> monitor = SLAMonitor(config, hooks=[LoggingSLAHook()])
    >>> result = monitor.evaluate(metrics)
"""

# ============================================================================
# Flows - Flow configuration and YAML generation
# ============================================================================
from truthound_kestra.flows import (
    # Configuration
    FlowConfig,
    # Generator
    FlowGenerator,
    InputConfig,
    OutputConfig,
    RetryConfig,
    RetryPolicy,
    TaskConfig,
    # Enums
    TaskType,
    TriggerConfig,
    TriggerType,
    generate_check_flow,
    generate_depot_release_flow,
    generate_depot_scheduled_validate_flow,
    generate_depot_validate_flow,
    # Functions
    generate_flow_yaml,
    generate_learn_flow,
    generate_profile_flow,
    generate_quality_pipeline,
)

# ============================================================================
# Outputs - Kestra output handling
# ============================================================================
from truthound_kestra.outputs import (
    FileOutputHandler,
    # Handlers
    KestraOutputHandler,
    MultiOutputHandler,
    send_check_result,
    send_depot_result,
    send_learn_result,
    # Functions
    send_outputs,
    send_profile_result,
)
from truthound_kestra.outputs import (
    # Configuration
    OutputConfig as OutputHandlerConfig,
)

# ============================================================================
# Scripts - Kestra Python Script entry points
# ============================================================================
from truthound_kestra.scripts import (
    # Presets
    DEFAULT_SCRIPT_CONFIG,
    LENIENT_SCRIPT_CONFIG,
    PRODUCTION_SCRIPT_CONFIG,
    STRICT_SCRIPT_CONFIG,
    CheckScriptConfig,
    # Executors
    CheckScriptExecutor,
    # Results
    CheckScriptResult,
    # Protocols
    DataQualityEngineProtocol,
    LearnScriptConfig,
    LearnScriptExecutor,
    LearnScriptResult,
    ProfileScriptConfig,
    ProfileScriptExecutor,
    ProfileScriptResult,
    # Configuration
    ScriptConfig,
    ScriptExecutorProtocol,
    StreamScriptExecutor,
    # Main entry points
    check_quality_script,
    create_script_config,
    # Utilities
    get_engine,
    learn_schema_script,
    profile_data_script,
    pull_snapshot_script,
    release_tag_script,
    stream_quality_script,
    validate_branch_script,
)

# ============================================================================
# SLA - SLA monitoring and alerting
# ============================================================================
from truthound_kestra.sla import (
    # Presets
    DEFAULT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    # Enums
    AlertLevel,
    BaseSLAHook,
    CallbackSLAHook,
    CompositeSLAHook,
    KestraNotificationHook,
    LoggingSLAHook,
    MetricsSLAHook,
    # Configuration
    SLAConfig,
    SLAEvaluationResult,
    # Hooks
    SLAHookProtocol,
    SLAMetrics,
    # Monitor
    SLAMonitor,
    SLARegistry,
    SLAViolation,
    SLAViolationType,
    evaluate_sla,
    # Functions
    get_sla_registry,
    register_sla,
    reset_sla_registry,
)

# ============================================================================
# Utils - Shared utilities and types
# ============================================================================
from truthound_kestra.utils import (
    COMPACT_SERIALIZER_CONFIG,
    DEFAULT_SERIALIZER_CONFIG,
    FULL_SERIALIZER_CONFIG,
    # Types and Enums
    CheckStatus,
    ColumnProfile,
    ConfigurationError,
    # Exceptions
    DataQualityError,
    DataSourceType,
    EngineError,
    ExecutionContext,
    FlowError,
    JsonSerializer,
    LearnedRule,
    MarkdownSerializer,
    MetadataDict,
    OperationType,
    OutputError,
    OutputFormat,
    # Serialization
    ResultSerializer,
    RuleDict,
    ScriptError,
    # Data classes
    ScriptOutput,
    SerializationError,
    SerializerConfig,
    Severity,
    SLAViolationError,
    # Helpers
    Timer,
    ValidationFailure,
    YamlSerializer,
    create_kestra_output,
    deserialize_result,
    detect_data_source_type,
    format_count,
    format_duration,
    format_percentage,
    format_status_badge,
    get_execution_context,
    get_kestra_variable,
    get_logger,
    get_serializer,
    kestra_outputs,
    load_data,
    log_operation,
    merge_rules,
    parse_uri,
    serialize_depot_result,
    serialize_result,
    serialize_to_format,
    timed,
    validate_rules,
)
from truthound_kestra.version import (
    __version__,
    __version_info__,
    __version_tuple__,
)

__all__ = [
    # Version
    "__version__",
    "__version_tuple__",
    "__version_info__",
    # ========================================================================
    # Scripts
    # ========================================================================
    # Main entry points
    "check_quality_script",
    "pull_snapshot_script",
    "release_tag_script",
    "stream_quality_script",
    "validate_branch_script",
    "profile_data_script",
    "learn_schema_script",
    # Executors
    "CheckScriptExecutor",
    "StreamScriptExecutor",
    "ProfileScriptExecutor",
    "LearnScriptExecutor",
    # Results
    "CheckScriptResult",
    "ProfileScriptResult",
    "LearnScriptResult",
    # Configuration
    "ScriptConfig",
    "CheckScriptConfig",
    "ProfileScriptConfig",
    "LearnScriptConfig",
    # Script Presets
    "DEFAULT_SCRIPT_CONFIG",
    "STRICT_SCRIPT_CONFIG",
    "LENIENT_SCRIPT_CONFIG",
    "PRODUCTION_SCRIPT_CONFIG",
    # Protocols
    "DataQualityEngineProtocol",
    "ScriptExecutorProtocol",
    # Script Utilities
    "get_engine",
    "create_script_config",
    # ========================================================================
    # Flows
    # ========================================================================
    # Configuration
    "FlowConfig",
    "TaskConfig",
    "TriggerConfig",
    "InputConfig",
    "OutputConfig",
    "OutputHandlerConfig",
    "RetryConfig",
    # Enums
    "TaskType",
    "TriggerType",
    "RetryPolicy",
    # Generator
    "FlowGenerator",
    # Functions
    "generate_flow_yaml",
    "generate_check_flow",
    "generate_depot_validate_flow",
    "generate_depot_scheduled_validate_flow",
    "generate_depot_release_flow",
    "generate_profile_flow",
    "generate_learn_flow",
    "generate_quality_pipeline",
    # ========================================================================
    # Outputs
    # ========================================================================
    # Output Configuration (note: OutputConfig from outputs, different from flows)
    "KestraOutputHandler",
    "FileOutputHandler",
    "MultiOutputHandler",
    # Functions
    "send_outputs",
    "send_check_result",
    "send_depot_result",
    "send_profile_result",
    "send_learn_result",
    # ========================================================================
    # SLA
    # ========================================================================
    # Enums
    "AlertLevel",
    "SLAViolationType",
    # Configuration
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    # SLA Presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
    # Monitor
    "SLAMonitor",
    "SLAEvaluationResult",
    "SLARegistry",
    # Hooks
    "SLAHookProtocol",
    "BaseSLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CallbackSLAHook",
    "CompositeSLAHook",
    "KestraNotificationHook",
    # Functions
    "get_sla_registry",
    "reset_sla_registry",
    "register_sla",
    "evaluate_sla",
    # ========================================================================
    # Utils
    # ========================================================================
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
    "serialize_depot_result",
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
