"""Prefect Flows for data quality operations.

This package provides flow decorators and factory functions for
integrating data quality checks into Prefect workflows.

Decorators:
    - quality_checked_flow: Add quality checks to any flow
    - profiled_flow: Add data profiling to any flow
    - validated_flow: Simple input/output validation

Factory Functions:
    - create_quality_flow: Create a complete quality flow
    - create_validation_flow: Create a standalone validation flow
    - create_pipeline_flow: Create multi-stage pipeline with checks
    - create_multi_table_quality_flows: Create flows for multiple tables

Example:
    >>> from prefect import flow
    >>> from truthound_prefect.flows import quality_checked_flow
    >>>
    >>> @quality_checked_flow(
    ...     rules=[{"type": "not_null", "column": "id"}],
    ...     fail_on_error=True,
    ... )
    ... async def process_users():
    ...     data = load_users()
    ...     return transform(data)
"""

from truthound_prefect.flows.config import (
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
)
from truthound_prefect.flows.decorators import (
    profiled_flow,
    quality_checked_flow,
    validated_flow,
)
from truthound_prefect.flows.factories import (
    create_multi_table_quality_flows,
    create_pipeline_flow,
    create_quality_flow,
    create_validation_flow,
)

__all__ = [
    # Configs
    "FlowConfig",
    "QualityFlowConfig",
    "PipelineFlowConfig",
    # Config presets
    "DEFAULT_FLOW_CONFIG",
    "DEFAULT_QUALITY_FLOW_CONFIG",
    "STRICT_QUALITY_FLOW_CONFIG",
    "LENIENT_QUALITY_FLOW_CONFIG",
    "AUTO_SCHEMA_FLOW_CONFIG",
    "DEFAULT_PIPELINE_CONFIG",
    "FULL_PIPELINE_CONFIG",
    # Decorators
    "quality_checked_flow",
    "profiled_flow",
    "validated_flow",
    # Factories
    "create_quality_flow",
    "create_validation_flow",
    "create_pipeline_flow",
    "create_multi_table_quality_flows",
]
