"""Flow modules for Kestra data quality integration.

This package provides configuration classes and template generators
for creating Kestra flow definitions programmatically.

Configuration Classes:
    FlowConfig: Complete flow configuration.
    TaskConfig: Task configuration.
    TriggerConfig: Trigger configuration.
    InputConfig: Input variable configuration.
    OutputConfig: Output variable configuration.
    RetryConfig: Retry policy configuration.

Template Functions:
    generate_flow_yaml: Generate YAML from FlowConfig.
    generate_check_flow: Generate a check quality flow.
    generate_profile_flow: Generate a profiling flow.
    generate_learn_flow: Generate a schema learning flow.
    generate_quality_pipeline: Generate a complete pipeline.

Example:
    >>> from truthound_kestra.flows import (
    ...     FlowConfig,
    ...     TaskConfig,
    ...     generate_flow_yaml,
    ...     generate_check_flow,
    ... )
    >>>
    >>> # Quick flow generation
    >>> yaml_content = generate_check_flow(
    ...     flow_id="check_users",
    ...     namespace="production",
    ...     input_uri="s3://bucket/users.parquet",
    ...     rules=[{"type": "not_null", "column": "id"}]
    ... )
    >>>
    >>> # Custom flow configuration
    >>> flow = FlowConfig(
    ...     id="custom_flow",
    ...     namespace="production",
    ...     tasks=[TaskConfig(id="check", task_type=TaskType.CHECK)]
    ... )
    >>> yaml_content = generate_flow_yaml(flow)
"""

from truthound_kestra.flows.config import (
    DEFAULT_FLOW_CONFIG,
    PRODUCTION_FLOW_CONFIG,
    FlowConfig,
    InputConfig,
    OutputConfig,
    RetryConfig,
    RetryPolicy,
    TaskConfig,
    TaskType,
    TriggerConfig,
    TriggerType,
)
from truthound_kestra.flows.templates import (
    FlowGenerator,
    generate_check_flow,
    generate_flow_yaml,
    generate_learn_flow,
    generate_profile_flow,
    generate_quality_pipeline,
)

__all__ = [
    # Enums
    "TaskType",
    "TriggerType",
    "RetryPolicy",
    # Config classes
    "FlowConfig",
    "TaskConfig",
    "TriggerConfig",
    "InputConfig",
    "OutputConfig",
    "RetryConfig",
    # Presets
    "DEFAULT_FLOW_CONFIG",
    "PRODUCTION_FLOW_CONFIG",
    # Generator
    "FlowGenerator",
    # Template functions
    "generate_flow_yaml",
    "generate_check_flow",
    "generate_profile_flow",
    "generate_learn_flow",
    "generate_quality_pipeline",
]
