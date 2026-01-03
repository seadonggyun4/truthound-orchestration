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

from truthound_kestra.version import (
    __version__,
    __version_info__,
    __version_tuple__,
)

# ============================================================================
# Scripts - Kestra Python Script entry points
# ============================================================================
from truthound_kestra.scripts import (
    # Main entry points
    check_quality_script,
    profile_data_script,
    learn_schema_script,
    # Executors
    CheckScriptExecutor,
    ProfileScriptExecutor,
    LearnScriptExecutor,
    # Results
    CheckScriptResult,
    ProfileScriptResult,
    LearnScriptResult,
    # Configuration
    ScriptConfig,
    CheckScriptConfig,
    ProfileScriptConfig,
    LearnScriptConfig,
    # Presets
    DEFAULT_SCRIPT_CONFIG,
    STRICT_SCRIPT_CONFIG,
    LENIENT_SCRIPT_CONFIG,
    PRODUCTION_SCRIPT_CONFIG,
    # Protocols
    DataQualityEngineProtocol,
    ScriptExecutorProtocol,
    # Utilities
    get_engine,
    create_script_config,
)

# ============================================================================
# Flows - Flow configuration and YAML generation
# ============================================================================
from truthound_kestra.flows import (
    # Configuration
    FlowConfig,
    TaskConfig,
    TriggerConfig,
    InputConfig,
    OutputConfig,
    RetryConfig,
    # Enums
    TaskType,
    TriggerType,
    RetryPolicy,
    # Generator
    FlowGenerator,
    # Functions
    generate_flow_yaml,
    generate_check_flow,
    generate_profile_flow,
    generate_learn_flow,
    generate_quality_pipeline,
)

# ============================================================================
# Outputs - Kestra output handling
# ============================================================================
from truthound_kestra.outputs import (
    # Configuration
    OutputConfig,
    # Handlers
    KestraOutputHandler,
    FileOutputHandler,
    MultiOutputHandler,
    # Functions
    send_outputs,
    send_check_result,
    send_profile_result,
    send_learn_result,
)

# ============================================================================
# SLA - SLA monitoring and alerting
# ============================================================================
from truthound_kestra.sla import (
    # Enums
    AlertLevel,
    SLAViolationType,
    # Configuration
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    # Presets
    DEFAULT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    # Monitor
    SLAMonitor,
    SLAEvaluationResult,
    SLARegistry,
    # Hooks
    SLAHookProtocol,
    BaseSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CallbackSLAHook,
    CompositeSLAHook,
    KestraNotificationHook,
    # Functions
    get_sla_registry,
    reset_sla_registry,
    register_sla,
    evaluate_sla,
)

# ============================================================================
# Utils - Shared utilities and types
# ============================================================================
from truthound_kestra.utils import (
    # Exceptions
    DataQualityError,
    ConfigurationError,
    EngineError,
    ScriptError,
    FlowError,
    OutputError,
    SLAViolationError,
    SerializationError,
    # Types and Enums
    CheckStatus,
    Severity,
    OperationType,
    OutputFormat,
    DataSourceType,
    RuleDict,
    MetadataDict,
    # Data classes
    ScriptOutput,
    ExecutionContext,
    ValidationFailure,
    ColumnProfile,
    LearnedRule,
    # Serialization
    ResultSerializer,
    JsonSerializer,
    YamlSerializer,
    MarkdownSerializer,
    SerializerConfig,
    DEFAULT_SERIALIZER_CONFIG,
    COMPACT_SERIALIZER_CONFIG,
    FULL_SERIALIZER_CONFIG,
    serialize_result,
    deserialize_result,
    serialize_to_format,
    get_serializer,
    # Helpers
    Timer,
    timed,
    get_logger,
    log_operation,
    load_data,
    detect_data_source_type,
    parse_uri,
    format_duration,
    format_percentage,
    format_count,
    format_status_badge,
    create_kestra_output,
    get_kestra_variable,
    get_execution_context,
    kestra_outputs,
    validate_rules,
    merge_rules,
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
    "profile_data_script",
    "learn_schema_script",
    # Executors
    "CheckScriptExecutor",
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
