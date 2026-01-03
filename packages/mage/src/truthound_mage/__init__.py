"""Mage AI Integration for Data Quality Operations.

truthound-mage provides blocks and utilities for data quality operations
in Mage AI pipelines. The package is engine-agnostic, supporting Truthound,
Great Expectations, Pandera, and custom engines.

Key Components:
    - Blocks: Transformer, Sensor, and Condition blocks for data quality
    - IO: Integration with Mage's io_config.yaml
    - SLA: Monitor and alert on SLA violations
    - Utils: Serialization and helper utilities

Example:
    >>> from truthound_mage import (
    ...     DataQualityTransformer,
    ...     CheckBlockConfig,
    ... )
    >>>
    >>> config = CheckBlockConfig(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     fail_on_error=True,
    ... )
    >>> transformer = DataQualityTransformer(config=config)
    >>> result = transformer.execute(df)
"""

from truthound_mage.version import __version__

# =============================================================================
# Blocks - Base
# =============================================================================

from truthound_mage.blocks.base import (
    BlockConfig,
    CheckBlockConfig,
    ProfileBlockConfig,
    LearnBlockConfig,
    BlockExecutionContext,
    BlockResult,
)

# =============================================================================
# Blocks - Transformer
# =============================================================================

from truthound_mage.blocks.transformer import (
    BaseDataQualityTransformer,
    DataQualityTransformer,
    CheckTransformer,
    ProfileTransformer,
    LearnTransformer,
    create_check_transformer,
    create_profile_transformer,
    create_learn_transformer,
)

# =============================================================================
# Blocks - Sensor
# =============================================================================

from truthound_mage.blocks.sensor import (
    BaseSensorBlock,
    DataQualitySensor,
    QualityGateSensor,
    SensorBlockConfig,
    create_quality_sensor,
)

# =============================================================================
# Blocks - Condition
# =============================================================================

from truthound_mage.blocks.condition import (
    BaseConditionBlock,
    DataQualityCondition,
    ConditionBlockConfig,
    ConditionResult,
    create_quality_condition,
)

# =============================================================================
# IO Configuration
# =============================================================================

from truthound_mage.io.config import (
    IOConfig,
    DataSourceConfig,
    DataSinkConfig,
    load_io_config,
)

# =============================================================================
# SLA
# =============================================================================

from truthound_mage.sla import (
    # Config
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
    # Presets
    DEFAULT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    # Monitor
    SLAMonitor,
    SLARegistry,
    # Hooks
    BaseSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
)

# =============================================================================
# Utils
# =============================================================================

from truthound_mage.utils import (
    # Exceptions
    DataQualityBlockError,
    BlockConfigurationError,
    BlockExecutionError,
    DataLoadError,
    SLAViolationError,
    # Types
    DataQualityOutput,
    BlockMetadata,
    # Serialization
    serialize_result,
    deserialize_result,
    # Helpers
    format_check_result,
    format_violations,
    create_block_metadata,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Blocks - Base
    "BlockConfig",
    "CheckBlockConfig",
    "ProfileBlockConfig",
    "LearnBlockConfig",
    "BlockExecutionContext",
    "BlockResult",
    # Blocks - Transformer
    "BaseDataQualityTransformer",
    "DataQualityTransformer",
    "CheckTransformer",
    "ProfileTransformer",
    "LearnTransformer",
    "create_check_transformer",
    "create_profile_transformer",
    "create_learn_transformer",
    # Blocks - Sensor
    "BaseSensorBlock",
    "DataQualitySensor",
    "QualityGateSensor",
    "SensorBlockConfig",
    "create_quality_sensor",
    # Blocks - Condition
    "BaseConditionBlock",
    "DataQualityCondition",
    "ConditionBlockConfig",
    "ConditionResult",
    "create_quality_condition",
    # IO
    "IOConfig",
    "DataSourceConfig",
    "DataSinkConfig",
    "load_io_config",
    # SLA - Config
    "AlertLevel",
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    "SLAViolationType",
    # SLA - Presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
    # SLA - Monitor
    "SLAMonitor",
    "SLARegistry",
    # SLA - Hooks
    "BaseSLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CompositeSLAHook",
    # Utils - Exceptions
    "DataQualityBlockError",
    "BlockConfigurationError",
    "BlockExecutionError",
    "DataLoadError",
    "SLAViolationError",
    # Utils - Types
    "DataQualityOutput",
    "BlockMetadata",
    # Utils - Serialization
    "serialize_result",
    "deserialize_result",
    # Utils - Helpers
    "format_check_result",
    "format_violations",
    "create_block_metadata",
]
