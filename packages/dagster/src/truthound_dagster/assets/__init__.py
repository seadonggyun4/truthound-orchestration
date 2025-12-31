"""Data Quality Assets for Dagster.

This module provides asset decorators and utilities for integrating
data quality operations into Dagster's Software-Defined Assets.

Available Decorators:
    - quality_checked_asset: Asset with automatic quality checks
    - profiled_asset: Asset with automatic profiling

Asset Factories:
    - create_quality_asset: Create asset with quality configuration
    - create_quality_check_asset: Create standalone quality check asset

Configuration:
    - QualityAssetConfig: Configuration for quality-checked assets

Example:
    >>> from dagster import Definitions
    >>> from truthound_dagster.assets import quality_checked_asset
    >>> from truthound_dagster.resources import DataQualityResource
    >>>
    >>> @quality_checked_asset(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ... )
    ... def users(context):
    ...     return load_users()
    >>>
    >>> defs = Definitions(
    ...     assets=[users],
    ...     resources={"data_quality": DataQualityResource()},
    ... )
"""

from truthound_dagster.assets.decorators import (
    profiled_asset,
    quality_checked_asset,
)
from truthound_dagster.assets.factories import (
    create_quality_asset,
    create_quality_check_asset,
)
from truthound_dagster.assets.config import (
    QualityAssetConfig,
    ProfileAssetConfig,
    QualityCheckMode,
)

__all__ = [
    # Decorators
    "quality_checked_asset",
    "profiled_asset",
    # Factories
    "create_quality_asset",
    "create_quality_check_asset",
    # Configuration
    "QualityAssetConfig",
    "ProfileAssetConfig",
    "QualityCheckMode",
]
