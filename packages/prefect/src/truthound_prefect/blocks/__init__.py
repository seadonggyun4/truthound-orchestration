"""Prefect Blocks for data quality operations.

This package provides Prefect Block implementations for integrating
data quality engines into Prefect workflows.

Blocks:
    - DataQualityBlock: High-level Prefect-native block (can be saved/loaded)
    - EngineBlock: Low-level engine wrapper with direct access

Example:
    >>> # Using DataQualityBlock (recommended)
    >>> from truthound_prefect.blocks import DataQualityBlock
    >>> block = DataQualityBlock(engine_name="truthound", auto_schema=True)
    >>> result = block.check(data)
    >>>
    >>> # Using EngineBlock for more control
    >>> from truthound_prefect.blocks import EngineBlock, EngineBlockConfig
    >>> config = EngineBlockConfig(engine_name="truthound", parallel=True)
    >>> with EngineBlock(config) as engine:
    ...     result = engine.check(data, auto_schema=True)
"""

from truthound_prefect.blocks.base import (
    BaseBlock,
    BlockConfig,
    DEFAULT_BLOCK_CONFIG,
    DEVELOPMENT_BLOCK_CONFIG,
    PRODUCTION_BLOCK_CONFIG,
)
from truthound_prefect.blocks.engine import (
    AUTO_SCHEMA_ENGINE_CONFIG,
    DEFAULT_ENGINE_CONFIG,
    DEVELOPMENT_ENGINE_CONFIG,
    DataQualityBlock,
    EngineBlock,
    EngineBlockConfig,
    PARALLEL_ENGINE_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
)

__all__ = [
    # Base
    "BlockConfig",
    "BaseBlock",
    "DEFAULT_BLOCK_CONFIG",
    "PRODUCTION_BLOCK_CONFIG",
    "DEVELOPMENT_BLOCK_CONFIG",
    # Engine
    "EngineBlockConfig",
    "EngineBlock",
    "DataQualityBlock",
    "DEFAULT_ENGINE_CONFIG",
    "PARALLEL_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    "DEVELOPMENT_ENGINE_CONFIG",
    "AUTO_SCHEMA_ENGINE_CONFIG",
]
