"""Mage blocks for data quality operations.

This module provides blocks for integrating data quality operations
into Mage AI pipelines.

Components:
    - Base: Configuration types and base abstractions
    - Transformer: Data quality check, profile, and learn transformers
    - Sensor: Quality gate sensors for conditional execution
    - Condition: Routing conditions based on quality results
"""

from truthound_mage.blocks.base import (
    BlockConfig,
    CheckBlockConfig,
    ProfileBlockConfig,
    LearnBlockConfig,
    BlockExecutionContext,
    BlockResult,
    DEFAULT_BLOCK_CONFIG,
    STRICT_BLOCK_CONFIG,
    LENIENT_BLOCK_CONFIG,
)

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

from truthound_mage.blocks.sensor import (
    BaseSensorBlock,
    DataQualitySensor,
    QualityGateSensor,
    SensorBlockConfig,
    create_quality_sensor,
)

from truthound_mage.blocks.condition import (
    BaseConditionBlock,
    DataQualityCondition,
    ConditionBlockConfig,
    ConditionResult,
    create_quality_condition,
)

__all__ = [
    # Base
    "BlockConfig",
    "CheckBlockConfig",
    "ProfileBlockConfig",
    "LearnBlockConfig",
    "BlockExecutionContext",
    "BlockResult",
    "DEFAULT_BLOCK_CONFIG",
    "STRICT_BLOCK_CONFIG",
    "LENIENT_BLOCK_CONFIG",
    # Transformer
    "BaseDataQualityTransformer",
    "DataQualityTransformer",
    "CheckTransformer",
    "ProfileTransformer",
    "LearnTransformer",
    "create_check_transformer",
    "create_profile_transformer",
    "create_learn_transformer",
    # Sensor
    "BaseSensorBlock",
    "DataQualitySensor",
    "QualityGateSensor",
    "SensorBlockConfig",
    "create_quality_sensor",
    # Condition
    "BaseConditionBlock",
    "DataQualityCondition",
    "ConditionBlockConfig",
    "ConditionResult",
    "create_quality_condition",
]
