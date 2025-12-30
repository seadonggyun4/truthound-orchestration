"""Data Quality Engine module for Truthound Orchestration.

This module provides a pluggable engine abstraction layer that allows
integration with various data quality frameworks. The core abstraction
is the DataQualityEngine Protocol, which can be implemented by any
data quality tool (Truthound, Great Expectations, Pandera, etc.).

All engines support lifecycle management with start/stop/health_check methods
and can be used as context managers for automatic resource cleanup.

Quick Start:
    >>> from common.engines import get_engine, TruthoundEngine
    >>> engine = get_engine("truthound")  # or use TruthoundEngine()
    >>> result = engine.check(data, rules=[{"type": "not_null", "column": "id"}])

Using Context Manager:
    >>> from common.engines import TruthoundEngine
    >>> with TruthoundEngine() as engine:
    ...     result = engine.check(data)

Lifecycle Management:
    >>> from common.engines import TruthoundEngine, EngineLifecycleManager
    >>> engine = TruthoundEngine()
    >>> engine.start()
    >>> result = engine.health_check()
    >>> engine.stop()

Using Registry:
    >>> from common.engines import EngineRegistry, register_engine, get_engine
    >>> register_engine("custom", CustomEngine())
    >>> engine = get_engine("custom")

Batch Operations:
    >>> from common.engines import TruthoundEngine, BatchExecutor, BatchConfig
    >>> engine = TruthoundEngine()
    >>> executor = BatchExecutor(engine)
    >>> results = executor.check_batch(
    ...     datasets=[df1, df2, df3],
    ...     config=BatchConfig(parallel=True, max_workers=4),
    ... )

Custom Engine Implementation:
    >>> from common.engines import DataQualityEngine, ManagedEngineMixin
    >>> class MyEngine(ManagedEngineMixin, EngineInfoMixin):
    ...     @property
    ...     def engine_name(self) -> str:
    ...         return "my_engine"
    ...     @property
    ...     def engine_version(self) -> str:
    ...         return "1.0.0"
    ...     def check(self, data, rules, **kwargs):
    ...         # Implementation
    ...         ...

Available Engines:
    - TruthoundEngine: Default engine using Truthound library
    - GreatExpectationsAdapter: Adapter for Great Expectations
    - PanderaAdapter: Adapter for Pandera schema validation
"""

from common.engines.base import (
    AsyncDataQualityEngine,
    DataQualityEngine,
    EngineCapabilities,
    EngineInfo,
    EngineInfoMixin,
)
from common.engines.batch import (
    # Exceptions
    AggregationError,
    # Strategies (Enums)
    AggregationStrategy,
    # Async Executor
    AsyncBatchExecutor,
    # Exceptions
    BatchConfig,
    BatchExecutionError,
    # Sync Executor
    BatchExecutor,
    # Hooks
    BatchHook,
    # Result Types
    BatchItemResult,
    BatchOperationError,
    BatchResult,
    ChunkingError,
    ChunkingStrategy,
    # Hooks
    CompositeBatchHook,
    # Chunkers
    DatasetListChunker,
    # Preset Configurations
    DEFAULT_BATCH_CONFIG,
    ExecutionStrategy,
    FAIL_FAST_BATCH_CONFIG,
    LARGE_DATA_BATCH_CONFIG,
    LoggingBatchHook,
    MetricsBatchHook,
    PARALLEL_BATCH_CONFIG,
    PolarsChunker,
    RowCountChunker,
    SEQUENTIAL_BATCH_CONFIG,
    # Convenience Functions
    create_async_batch_executor,
    create_batch_executor,
)
from common.engines.config import (
    # Base Configuration
    BaseEngineConfig,
    # Builder
    ConfigBuilder,
    # Enums
    ConfigEnvironment,
    ConfigLoader,
    ConfigRegistry,
    ConfigSource,
    ConfigValidator,
    # Environment-Aware
    EnvironmentConfig,
    # Exceptions
    ConfigLoadError,
    ConfigurationError,
    ConfigValidationError,
    # Validation
    FieldConstraint,
    MergeStrategy,
    ValidationError,
    ValidationResult,
    # Convenience Functions
    create_config_for_environment,
    load_config,
)
from common.engines.great_expectations import (
    DEFAULT_GE_CONFIG,
    DEVELOPMENT_GE_CONFIG,
    PRODUCTION_GE_CONFIG,
    GreatExpectationsAdapter,
    GreatExpectationsConfig,
)
from common.engines.lifecycle import (
    # Configuration
    DEFAULT_ENGINE_CONFIG,
    DEVELOPMENT_ENGINE_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
    TESTING_ENGINE_CONFIG,
    # Protocols
    AsyncLifecycleHook,
    AsyncManagedEngine,
    CompositeLifecycleHook,
    # Exceptions
    EngineAlreadyStartedError,
    EngineConfig,
    # Health
    EngineHealthChecker,
    EngineInitializationError,
    EngineLifecycleError,
    # Manager
    EngineLifecycleManager,
    EngineNotStartedError,
    EngineShutdownError,
    # State
    EngineState,
    EngineStateSnapshot,
    EngineStateTracker,
    EngineStoppedError,
    LifecycleHook,
    # Sync Hooks
    LoggingLifecycleHook,
    ManagedEngine,
    # Mixin
    ManagedEngineMixin,
    MetricsLifecycleHook,
    # Utilities
    create_engine_health_checker,
    create_managed_engine,
    register_engine_health_check,
    # Async Components
    AsyncCompositeLifecycleHook,
    AsyncEngineHealthChecker,
    AsyncEngineLifecycleManager,
    AsyncLoggingLifecycleHook,
    AsyncManagedEngineMixin,
    AsyncMetricsLifecycleHook,
    SyncEngineAsyncAdapter,
    SyncToAsyncLifecycleHookAdapter,
)
from common.engines.pandera import (
    DEFAULT_PANDERA_CONFIG,
    DEVELOPMENT_PANDERA_CONFIG,
    PRODUCTION_PANDERA_CONFIG,
    PanderaAdapter,
    PanderaConfig,
)
from common.engines.registry import (
    EngineRegistry,
    get_default_engine,
    get_engine,
    get_engine_registry,
    list_engines,
    register_engine,
    set_default_engine,
    # Plugin Discovery
    enable_plugin_discovery,
    disable_plugin_discovery,
    is_plugin_discovery_enabled,
    discover_and_register_plugins,
)
from common.engines.plugin import (
    # Constants
    ENTRY_POINT_GROUP,
    DEFAULT_PLUGIN_PRIORITY,
    # Enums
    PluginState,
    PluginType,
    LoadStrategy,
    # Exceptions
    PluginError,
    PluginDiscoveryError,
    PluginLoadError,
    PluginValidationError,
    PluginConflictError,
    PluginNotFoundError,
    # Data Types
    PluginMetadata,
    PluginSpec,
    PluginInstance,
    ValidationResult as PluginValidationResult,
    DiscoveryConfig,
    DEFAULT_DISCOVERY_CONFIG,
    # Protocols
    PluginSource,
    PluginHook,
    PluginValidator,
    PluginFactory,
    # Hook Implementations
    BasePluginHook,
    LoggingPluginHook,
    MetricsPluginHook,
    CompositePluginHook,
    # Plugin Sources
    EntryPointPluginSource,
    BuiltinPluginSource,
    DictPluginSource,
    # Validators
    DataQualityEngineValidator,
    CompositeValidator,
    DEFAULT_VALIDATOR,
    # Factory
    DefaultPluginFactory,
    DEFAULT_FACTORY,
    # Loader
    PluginLoader,
    # Registry
    PluginRegistry,
    get_plugin_registry,
    reset_plugin_registry,
    # Convenience Functions
    discover_plugins,
    load_plugins,
    get_plugin_engine,
    validate_plugin,
    auto_discover_engines,
    register_plugin,
)
from common.engines.metrics import (
    # Protocols
    AsyncEngineMetricsHook,
    EngineMetricsHook,
    # Configuration
    DEFAULT_ENGINE_METRICS_CONFIG,
    DISABLED_ENGINE_METRICS_CONFIG,
    EngineMetricsConfig,
    FULL_ENGINE_METRICS_CONFIG,
    MINIMAL_ENGINE_METRICS_CONFIG,
    # Enums
    EngineOperation,
    OperationStatus,
    # Base Hooks
    AsyncBaseEngineMetricsHook,
    BaseEngineMetricsHook,
    # Hook Implementations
    CompositeEngineHook,
    AsyncCompositeEngineHook,
    LoggingEngineHook,
    MetricsEngineHook,
    StatsCollectorHook,
    TracingEngineHook,
    # Adapters
    SyncToAsyncEngineHookAdapter,
    # Instrumented Engines
    AsyncInstrumentedEngine,
    InstrumentedEngine,
    # Factory Functions
    create_async_instrumented_engine,
    create_instrumented_engine,
    # Statistics
    EngineOperationStats,
    # Exceptions
    EngineMetricsError,
)
from common.engines.context import (
    # Configuration
    ContextConfig,
    DEFAULT_CONTEXT_CONFIG,
    LIGHTWEIGHT_CONTEXT_CONFIG,
    STRICT_CONTEXT_CONFIG,
    TESTING_CONTEXT_CONFIG,
    # Enums
    ContextState,
    SessionState,
    CleanupStrategy,
    # Resource Tracking
    TrackedResource,
    ResourceTracker,
    # Savepoint Support
    Savepoint,
    SavepointManager,
    # Protocols
    ContextHook,
    # Hooks
    LoggingContextHook,
    MetricsContextHook,
    CompositeContextHook,
    # Main Context Classes
    EngineContext,
    EngineSession,
    MultiEngineContext,
    ContextStack,
    AsyncEngineContext,
    # Exceptions
    ContextError,
    ContextNotActiveError,
    ContextAlreadyActiveError,
    SavepointError,
    ResourceCleanupError,
    # Factory Functions
    create_engine_context,
    create_engine_session,
    create_multi_engine_context,
    engine_context,
    async_engine_context,
)
from common.engines.truthound import (
    DEFAULT_TRUTHOUND_CONFIG,
    PARALLEL_TRUTHOUND_CONFIG,
    PRODUCTION_TRUTHOUND_CONFIG,
    TruthoundEngine,
    TruthoundEngineConfig,
)
from common.engines.version import (
    # Core Types
    SemanticVersion,
    VersionConstraint,
    VersionRange,
    VersionCompatibilityResult,
    EngineVersionRequirement,
    # Enums
    VersionOperator,
    CompatibilityLevel,
    CompatibilityStrategy,
    # Checker
    VersionCompatibilityChecker,
    VersionCompatibilityConfig,
    DEFAULT_VERSION_COMPATIBILITY_CONFIG,
    STRICT_VERSION_COMPATIBILITY_CONFIG,
    LENIENT_VERSION_COMPATIBILITY_CONFIG,
    # Registry
    VersionRegistry,
    get_version_registry,
    reset_version_registry,
    # Convenience Functions
    parse_version,
    parse_constraint,
    parse_range,
    check_version_compatibility,
    require_version,
    is_version_compatible,
    compare_versions,
    versions_compatible,
    # Decorator
    version_required,
    # Exceptions
    VersionError,
    VersionParseError,
    VersionConstraintError,
    VersionIncompatibleError,
)
from common.engines.aggregation import (
    # Enums
    ResultAggregationStrategy,
    ConflictResolution,
    StatusPriority,
    # Configuration
    AggregationConfig,
    DEFAULT_AGGREGATION_CONFIG,
    STRICT_AGGREGATION_CONFIG,
    LENIENT_AGGREGATION_CONFIG,
    CONSENSUS_AGGREGATION_CONFIG,
    WORST_CASE_AGGREGATION_CONFIG,
    # Result Types
    EngineResultEntry,
    AggregatedResult,
    ComparisonResult,
    # Protocols
    ResultAggregator,
    EngineResultAggregator,
    ResultComparator,
    AggregationHook,
    # Base Classes
    BaseResultAggregator,
    # Aggregator Implementations
    CheckResultMergeAggregator,
    CheckResultWeightedAggregator,
    ProfileResultAggregator as ProfileResultMergeAggregator,
    LearnResultAggregator as LearnResultMergeAggregator,
    # Multi-Engine Aggregator
    MultiEngineAggregator,
    # Hooks
    BaseAggregationHook,
    LoggingAggregationHook,
    MetricsAggregationHook,
    CompositeAggregationHook,
    # Registry
    AggregatorRegistry,
    get_aggregator_registry,
    get_aggregator,
    register_aggregator,
    list_aggregators,
    # Convenience Functions
    aggregate_check_results,
    aggregate_profile_results,
    aggregate_learn_results,
    compare_check_results,
    create_multi_engine_aggregator,
    # Exceptions
    AggregationError as ResultAggregationError,
    NoResultsError,
    ConflictError,
    AggregatorNotFoundError,
    InvalidWeightError,
)
from common.engines.chain import (
    # Exceptions
    EngineChainError,
    AllEnginesFailedError,
    NoEngineSelectedError,
    EngineChainConfigError,
    # Enums
    FallbackStrategy,
    ChainExecutionMode,
    FailureReason,
    # Result Types
    ChainExecutionAttempt,
    ChainExecutionResult,
    # Configuration
    FallbackConfig,
    DEFAULT_FALLBACK_CONFIG,
    RETRY_FALLBACK_CONFIG,
    HEALTH_AWARE_FALLBACK_CONFIG,
    LOAD_BALANCED_CONFIG,
    WEIGHTED_CONFIG,
    # Protocols
    ChainHook,
    EngineSelector,
    EngineCondition,
    # Hooks
    LoggingChainHook,
    MetricsChainHook,
    CompositeChainHook,
    # Engine Iterators/Selectors
    SequentialEngineIterator,
    RoundRobinEngineIterator,
    WeightedRandomSelector,
    PrioritySelector,
    # Main Chain Classes
    EngineChain,
    ConditionalEngineChain,
    ConditionalRoute,
    SelectorEngineChain,
    AsyncEngineChain,
    # Factory Functions
    create_fallback_chain,
    create_load_balanced_chain,
    create_async_fallback_chain,
)


__all__ = [
    # Base Types / Protocols
    "DataQualityEngine",
    "AsyncDataQualityEngine",
    "ManagedEngine",
    "AsyncManagedEngine",
    "LifecycleHook",
    "AsyncLifecycleHook",
    "EngineCapabilities",
    "EngineInfo",
    "EngineInfoMixin",
    # State
    "EngineState",
    "EngineStateSnapshot",
    "EngineStateTracker",
    # Base Configuration
    "EngineConfig",
    "BaseEngineConfig",
    "DEFAULT_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    "DEVELOPMENT_ENGINE_CONFIG",
    "TESTING_ENGINE_CONFIG",
    # Batch Operations - Config
    "BatchConfig",
    "DEFAULT_BATCH_CONFIG",
    "PARALLEL_BATCH_CONFIG",
    "SEQUENTIAL_BATCH_CONFIG",
    "FAIL_FAST_BATCH_CONFIG",
    "LARGE_DATA_BATCH_CONFIG",
    # Batch Operations - Strategies
    "ExecutionStrategy",
    "AggregationStrategy",
    "ChunkingStrategy",
    # Batch Operations - Executors
    "BatchExecutor",
    "AsyncBatchExecutor",
    "create_batch_executor",
    "create_async_batch_executor",
    # Batch Operations - Results
    "BatchResult",
    "BatchItemResult",
    # Batch Operations - Hooks
    "BatchHook",
    "LoggingBatchHook",
    "MetricsBatchHook",
    "CompositeBatchHook",
    # Batch Operations - Chunkers
    "RowCountChunker",
    "PolarsChunker",
    "DatasetListChunker",
    # Batch Operations - Exceptions
    "BatchOperationError",
    "BatchExecutionError",
    "ChunkingError",
    "AggregationError",
    # Configuration System
    "ConfigBuilder",
    "ConfigLoader",
    "ConfigValidator",
    "ConfigRegistry",
    "EnvironmentConfig",
    "ConfigSource",
    "ConfigEnvironment",
    "MergeStrategy",
    "FieldConstraint",
    "ValidationError",
    "ValidationResult",
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigLoadError",
    "load_config",
    "create_config_for_environment",
    # Sync Lifecycle Manager
    "EngineLifecycleManager",
    # Async Lifecycle Manager
    "AsyncEngineLifecycleManager",
    # Sync Health Checker
    "EngineHealthChecker",
    # Async Health Checker
    "AsyncEngineHealthChecker",
    # Sync Hooks
    "LoggingLifecycleHook",
    "MetricsLifecycleHook",
    "CompositeLifecycleHook",
    # Async Hooks
    "AsyncLoggingLifecycleHook",
    "AsyncMetricsLifecycleHook",
    "AsyncCompositeLifecycleHook",
    "SyncToAsyncLifecycleHookAdapter",
    # Sync Mixin
    "ManagedEngineMixin",
    # Async Mixin
    "AsyncManagedEngineMixin",
    # Adapter
    "SyncEngineAsyncAdapter",
    # Utilities
    "create_managed_engine",
    "create_engine_health_checker",
    "register_engine_health_check",
    # Exceptions
    "EngineLifecycleError",
    "EngineNotStartedError",
    "EngineAlreadyStartedError",
    "EngineStoppedError",
    "EngineInitializationError",
    "EngineShutdownError",
    # Truthound
    "TruthoundEngine",
    "TruthoundEngineConfig",
    "DEFAULT_TRUTHOUND_CONFIG",
    "PARALLEL_TRUTHOUND_CONFIG",
    "PRODUCTION_TRUTHOUND_CONFIG",
    # Great Expectations
    "GreatExpectationsAdapter",
    "GreatExpectationsConfig",
    "DEFAULT_GE_CONFIG",
    "PRODUCTION_GE_CONFIG",
    "DEVELOPMENT_GE_CONFIG",
    # Pandera
    "PanderaAdapter",
    "PanderaConfig",
    "DEFAULT_PANDERA_CONFIG",
    "PRODUCTION_PANDERA_CONFIG",
    "DEVELOPMENT_PANDERA_CONFIG",
    # Registry
    "EngineRegistry",
    "get_engine_registry",
    "get_engine",
    "register_engine",
    "list_engines",
    "get_default_engine",
    "set_default_engine",
    # Registry - Plugin Discovery Integration
    "enable_plugin_discovery",
    "disable_plugin_discovery",
    "is_plugin_discovery_enabled",
    "discover_and_register_plugins",
    # Plugin Discovery - Constants
    "ENTRY_POINT_GROUP",
    "DEFAULT_PLUGIN_PRIORITY",
    # Plugin Discovery - Enums
    "PluginState",
    "PluginType",
    "LoadStrategy",
    # Plugin Discovery - Exceptions
    "PluginError",
    "PluginDiscoveryError",
    "PluginLoadError",
    "PluginValidationError",
    "PluginConflictError",
    "PluginNotFoundError",
    # Plugin Discovery - Data Types
    "PluginMetadata",
    "PluginSpec",
    "PluginInstance",
    "PluginValidationResult",
    "DiscoveryConfig",
    "DEFAULT_DISCOVERY_CONFIG",
    # Plugin Discovery - Protocols
    "PluginSource",
    "PluginHook",
    "PluginValidator",
    "PluginFactory",
    # Plugin Discovery - Hook Implementations
    "BasePluginHook",
    "LoggingPluginHook",
    "MetricsPluginHook",
    "CompositePluginHook",
    # Plugin Discovery - Sources
    "EntryPointPluginSource",
    "BuiltinPluginSource",
    "DictPluginSource",
    # Plugin Discovery - Validators
    "DataQualityEngineValidator",
    "CompositeValidator",
    "DEFAULT_VALIDATOR",
    # Plugin Discovery - Factory
    "DefaultPluginFactory",
    "DEFAULT_FACTORY",
    # Plugin Discovery - Loader
    "PluginLoader",
    # Plugin Discovery - Registry
    "PluginRegistry",
    "get_plugin_registry",
    "reset_plugin_registry",
    # Plugin Discovery - Convenience Functions
    "discover_plugins",
    "load_plugins",
    "get_plugin_engine",
    "validate_plugin",
    "auto_discover_engines",
    "register_plugin",
    # Engine Metrics - Protocols
    "EngineMetricsHook",
    "AsyncEngineMetricsHook",
    # Engine Metrics - Configuration
    "EngineMetricsConfig",
    "DEFAULT_ENGINE_METRICS_CONFIG",
    "DISABLED_ENGINE_METRICS_CONFIG",
    "MINIMAL_ENGINE_METRICS_CONFIG",
    "FULL_ENGINE_METRICS_CONFIG",
    # Engine Metrics - Enums
    "EngineOperation",
    "OperationStatus",
    # Engine Metrics - Base Hooks
    "BaseEngineMetricsHook",
    "AsyncBaseEngineMetricsHook",
    # Engine Metrics - Hook Implementations
    "MetricsEngineHook",
    "LoggingEngineHook",
    "TracingEngineHook",
    "StatsCollectorHook",
    "CompositeEngineHook",
    "AsyncCompositeEngineHook",
    # Engine Metrics - Adapters
    "SyncToAsyncEngineHookAdapter",
    # Engine Metrics - Instrumented Engines
    "InstrumentedEngine",
    "AsyncInstrumentedEngine",
    # Engine Metrics - Factory Functions
    "create_instrumented_engine",
    "create_async_instrumented_engine",
    # Engine Metrics - Statistics
    "EngineOperationStats",
    # Engine Metrics - Exceptions
    "EngineMetricsError",
    # Context Management - Configuration
    "ContextConfig",
    "DEFAULT_CONTEXT_CONFIG",
    "LIGHTWEIGHT_CONTEXT_CONFIG",
    "STRICT_CONTEXT_CONFIG",
    "TESTING_CONTEXT_CONFIG",
    # Context Management - Enums
    "ContextState",
    "SessionState",
    "CleanupStrategy",
    # Context Management - Resource Tracking
    "TrackedResource",
    "ResourceTracker",
    # Context Management - Savepoint Support
    "Savepoint",
    "SavepointManager",
    # Context Management - Protocols
    "ContextHook",
    # Context Management - Hooks
    "LoggingContextHook",
    "MetricsContextHook",
    "CompositeContextHook",
    # Context Management - Main Classes
    "EngineContext",
    "EngineSession",
    "MultiEngineContext",
    "ContextStack",
    "AsyncEngineContext",
    # Context Management - Exceptions
    "ContextError",
    "ContextNotActiveError",
    "ContextAlreadyActiveError",
    "SavepointError",
    "ResourceCleanupError",
    # Context Management - Factory Functions
    "create_engine_context",
    "create_engine_session",
    "create_multi_engine_context",
    "engine_context",
    "async_engine_context",
    # Result Aggregation - Enums
    "ResultAggregationStrategy",
    "ConflictResolution",
    "StatusPriority",
    # Result Aggregation - Configuration
    "AggregationConfig",
    "DEFAULT_AGGREGATION_CONFIG",
    "STRICT_AGGREGATION_CONFIG",
    "LENIENT_AGGREGATION_CONFIG",
    "CONSENSUS_AGGREGATION_CONFIG",
    "WORST_CASE_AGGREGATION_CONFIG",
    # Result Aggregation - Result Types
    "EngineResultEntry",
    "AggregatedResult",
    "ComparisonResult",
    # Result Aggregation - Protocols
    "ResultAggregator",
    "EngineResultAggregator",
    "ResultComparator",
    "AggregationHook",
    # Result Aggregation - Base Classes
    "BaseResultAggregator",
    # Result Aggregation - Aggregator Implementations
    "CheckResultMergeAggregator",
    "CheckResultWeightedAggregator",
    "ProfileResultMergeAggregator",
    "LearnResultMergeAggregator",
    # Result Aggregation - Multi-Engine
    "MultiEngineAggregator",
    # Result Aggregation - Hooks
    "BaseAggregationHook",
    "LoggingAggregationHook",
    "MetricsAggregationHook",
    "CompositeAggregationHook",
    # Result Aggregation - Registry
    "AggregatorRegistry",
    "get_aggregator_registry",
    "get_aggregator",
    "register_aggregator",
    "list_aggregators",
    # Result Aggregation - Convenience Functions
    "aggregate_check_results",
    "aggregate_profile_results",
    "aggregate_learn_results",
    "compare_check_results",
    "create_multi_engine_aggregator",
    # Result Aggregation - Exceptions
    "ResultAggregationError",
    "NoResultsError",
    "ConflictError",
    "AggregatorNotFoundError",
    "InvalidWeightError",
    # Version Management - Core Types
    "SemanticVersion",
    "VersionConstraint",
    "VersionRange",
    "VersionCompatibilityResult",
    "EngineVersionRequirement",
    # Version Management - Enums
    "VersionOperator",
    "CompatibilityLevel",
    "CompatibilityStrategy",
    # Version Management - Checker
    "VersionCompatibilityChecker",
    "VersionCompatibilityConfig",
    "DEFAULT_VERSION_COMPATIBILITY_CONFIG",
    "STRICT_VERSION_COMPATIBILITY_CONFIG",
    "LENIENT_VERSION_COMPATIBILITY_CONFIG",
    # Version Management - Registry
    "VersionRegistry",
    "get_version_registry",
    "reset_version_registry",
    # Version Management - Convenience Functions
    "parse_version",
    "parse_constraint",
    "parse_range",
    "check_version_compatibility",
    "require_version",
    "is_version_compatible",
    "compare_versions",
    "versions_compatible",
    # Version Management - Decorator
    "version_required",
    # Version Management - Exceptions
    "VersionError",
    "VersionParseError",
    "VersionConstraintError",
    "VersionIncompatibleError",
    # Engine Chain - Exceptions
    "EngineChainError",
    "AllEnginesFailedError",
    "NoEngineSelectedError",
    "EngineChainConfigError",
    # Engine Chain - Enums
    "FallbackStrategy",
    "ChainExecutionMode",
    "FailureReason",
    # Engine Chain - Result Types
    "ChainExecutionAttempt",
    "ChainExecutionResult",
    # Engine Chain - Configuration
    "FallbackConfig",
    "DEFAULT_FALLBACK_CONFIG",
    "RETRY_FALLBACK_CONFIG",
    "HEALTH_AWARE_FALLBACK_CONFIG",
    "LOAD_BALANCED_CONFIG",
    "WEIGHTED_CONFIG",
    # Engine Chain - Protocols
    "ChainHook",
    "EngineSelector",
    "EngineCondition",
    # Engine Chain - Hooks
    "LoggingChainHook",
    "MetricsChainHook",
    "CompositeChainHook",
    # Engine Chain - Iterators/Selectors
    "SequentialEngineIterator",
    "RoundRobinEngineIterator",
    "WeightedRandomSelector",
    "PrioritySelector",
    # Engine Chain - Main Classes
    "EngineChain",
    "ConditionalEngineChain",
    "ConditionalRoute",
    "SelectorEngineChain",
    "AsyncEngineChain",
    # Engine Chain - Factory Functions
    "create_fallback_chain",
    "create_load_balanced_chain",
    "create_async_fallback_chain",
]
