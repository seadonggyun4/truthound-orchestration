"""Mage blocks for data quality operations.

This module provides blocks for integrating data quality operations
into Mage AI pipelines.

Components:
    - Base: Configuration types and base abstractions
    - Transformer: Data quality check, profile, and learn transformers
    - Drift: Drift detection between baseline and current datasets
    - Anomaly: Anomaly detection in datasets
    - Sensor: Quality gate sensors for conditional execution
    - Condition: Routing conditions based on quality results
"""

from truthound_mage.blocks.anomaly import (
    AnomalyTransformer,
    create_anomaly_transformer,
)
from truthound_mage.blocks.base import (
    DEFAULT_ANOMALY_BLOCK_CONFIG,
    DEFAULT_BLOCK_CONFIG,
    DEFAULT_DRIFT_BLOCK_CONFIG,
    LENIENT_ANOMALY_BLOCK_CONFIG,
    LENIENT_BLOCK_CONFIG,
    LENIENT_DRIFT_BLOCK_CONFIG,
    STRICT_ANOMALY_BLOCK_CONFIG,
    STRICT_BLOCK_CONFIG,
    STRICT_DRIFT_BLOCK_CONFIG,
    AnomalyBlockConfig,
    BlockConfig,
    BlockExecutionContext,
    BlockResult,
    CheckBlockConfig,
    DriftBlockConfig,
    LearnBlockConfig,
    ProfileBlockConfig,
)
from truthound_mage.blocks.condition import (
    BaseConditionBlock,
    ConditionBlockConfig,
    ConditionResult,
    DataQualityCondition,
    create_quality_condition,
)
from truthound_mage.blocks.depot import scheduled_sync, sync_asset, validate_branch
from truthound_mage.blocks.drift import (
    BaseDriftTransformer,
    DriftTransformer,
    create_drift_transformer,
)
from truthound_mage.blocks.sensor import (
    BaseSensorBlock,
    DataQualitySensor,
    QualityGateSensor,
    SensorBlockConfig,
    create_quality_sensor,
)
from truthound_mage.blocks.transformer import (
    BaseDataQualityTransformer,
    CheckTransformer,
    DataQualityTransformer,
    LearnTransformer,
    ProfileTransformer,
    create_check_transformer,
    create_learn_transformer,
    create_profile_transformer,
)

__all__ = [
    # Base
    "BlockConfig",
    "CheckBlockConfig",
    "ProfileBlockConfig",
    "LearnBlockConfig",
    "DriftBlockConfig",
    "AnomalyBlockConfig",
    "BlockExecutionContext",
    "BlockResult",
    "DEFAULT_BLOCK_CONFIG",
    "STRICT_BLOCK_CONFIG",
    "LENIENT_BLOCK_CONFIG",
    "DEFAULT_DRIFT_BLOCK_CONFIG",
    "STRICT_DRIFT_BLOCK_CONFIG",
    "LENIENT_DRIFT_BLOCK_CONFIG",
    "DEFAULT_ANOMALY_BLOCK_CONFIG",
    "STRICT_ANOMALY_BLOCK_CONFIG",
    "LENIENT_ANOMALY_BLOCK_CONFIG",
    # Transformer
    "BaseDataQualityTransformer",
    "DataQualityTransformer",
    "CheckTransformer",
    "ProfileTransformer",
    "LearnTransformer",
    "create_check_transformer",
    "create_profile_transformer",
    "create_learn_transformer",
    # Drift
    "BaseDriftTransformer",
    "DriftTransformer",
    "create_drift_transformer",
    # Anomaly
    "AnomalyTransformer",
    "create_anomaly_transformer",
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
    "sync_asset",
    "validate_branch",
    "scheduled_sync",
]
