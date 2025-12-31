"""Data Quality Resources for Dagster.

This module provides Dagster resources for data quality operations.
Resources are the Dagster-native way to share state and connections
across ops and assets.

Available Resources:
    - DataQualityResource: Unified resource for all data quality operations
    - EngineResource: Low-level engine access resource

Configuration:
    - ResourceConfig: Base resource configuration
    - DataQualityResourceConfig: Full configuration with engine options

Example:
    >>> from dagster import Definitions, asset
    >>> from truthound_dagster.resources import DataQualityResource
    >>>
    >>> @asset
    ... def my_asset(data_quality: DataQualityResource):
    ...     result = data_quality.check(
    ...         data=my_data,
    ...         rules=[{"column": "id", "type": "not_null"}],
    ...     )
    ...     return result
    >>>
    >>> defs = Definitions(
    ...     assets=[my_asset],
    ...     resources={"data_quality": DataQualityResource()},
    ... )
"""

from truthound_dagster.resources.base import (
    BaseResource,
    ResourceConfig,
)
from truthound_dagster.resources.engine import (
    DataQualityResource,
    DataQualityResourceConfig,
    EngineResource,
    EngineResourceConfig,
    # Presets
    DEFAULT_ENGINE_CONFIG,
    PARALLEL_ENGINE_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
    DEFAULT_DQ_CONFIG,
    STRICT_DQ_CONFIG,
    LENIENT_DQ_CONFIG,
)

__all__ = [
    # Base
    "BaseResource",
    "ResourceConfig",
    # Engine resources
    "DataQualityResource",
    "DataQualityResourceConfig",
    "EngineResource",
    "EngineResourceConfig",
    # Presets
    "DEFAULT_ENGINE_CONFIG",
    "PARALLEL_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    "DEFAULT_DQ_CONFIG",
    "STRICT_DQ_CONFIG",
    "LENIENT_DQ_CONFIG",
]
