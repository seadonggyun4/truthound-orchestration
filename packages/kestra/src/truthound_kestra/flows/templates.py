"""YAML flow template generators for Kestra data quality integration.

This module provides functions to generate Kestra flow YAML files
from configuration objects.

Example:
    >>> from truthound_kestra.flows.templates import (
    ...     generate_flow_yaml,
    ...     generate_check_flow,
    ... )
    >>>
    >>> yaml_content = generate_check_flow(
    ...     flow_id="validate_users",
    ...     namespace="production",
    ...     input_uri="s3://bucket/users.parquet",
    ...     rules=[{"type": "not_null", "column": "id"}],
    ... )
    >>> print(yaml_content)
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from truthound_kestra.flows.config import (
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
from truthound_kestra.utils.types import RuleDict

__all__ = [
    "generate_flow_yaml",
    "generate_check_flow",
    "generate_profile_flow",
    "generate_learn_flow",
    "generate_quality_pipeline",
    "FlowGenerator",
]


class FlowGenerator:
    """Generator for Kestra flow YAML files.

    This class provides methods to generate complete Kestra flow
    definitions as YAML strings or dictionaries.

    Example:
        >>> generator = FlowGenerator()
        >>> yaml_content = generator.generate(flow_config)
    """

    def __init__(self, python_version: str = "3.11") -> None:
        """Initialize the generator.

        Args:
            python_version: Python version for script tasks.
        """
        self.python_version = python_version

    def generate(self, config: FlowConfig) -> str:
        """Generate YAML content from flow configuration.

        Args:
            config: Flow configuration.

        Returns:
            YAML string representation of the flow.
        """
        flow_dict = self._build_flow_dict(config)
        return yaml.dump(
            flow_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def generate_dict(self, config: FlowConfig) -> dict[str, Any]:
        """Generate dictionary from flow configuration.

        Args:
            config: Flow configuration.

        Returns:
            Dictionary representation of the flow.
        """
        return self._build_flow_dict(config)

    def _build_flow_dict(self, config: FlowConfig) -> dict[str, Any]:
        """Build complete flow dictionary."""
        flow: dict[str, Any] = {
            "id": config.id,
            "namespace": config.namespace,
        }

        if config.description:
            flow["description"] = config.description

        if config.disabled:
            flow["disabled"] = True

        if config.labels:
            flow["labels"] = list(config.labels)

        if config.variables:
            flow["variables"] = config.variables

        if config.concurrency is not None:
            flow["concurrency"] = {"limit": config.concurrency}

        if config.inputs:
            flow["inputs"] = [self._build_input(i) for i in config.inputs]

        if config.triggers:
            flow["triggers"] = [self._build_trigger(t) for t in config.triggers]

        if config.tasks:
            flow["tasks"] = [self._build_task(t, config.retry) for t in config.tasks]

        if config.outputs:
            flow["outputs"] = [self._build_output(o) for o in config.outputs]

        return flow

    def _build_input(self, input_cfg: InputConfig) -> dict[str, Any]:
        """Build input dictionary."""
        return input_cfg.to_dict()

    def _build_output(self, output_cfg: OutputConfig) -> dict[str, Any]:
        """Build output dictionary."""
        return output_cfg.to_dict()

    def _build_trigger(self, trigger: TriggerConfig) -> dict[str, Any]:
        """Build trigger dictionary."""
        return trigger.to_dict()

    def _build_task(
        self,
        task: TaskConfig,
        default_retry: RetryConfig | None,
    ) -> dict[str, Any]:
        """Build task dictionary."""
        task_dict: dict[str, Any] = {
            "id": task.id,
            "type": "io.kestra.plugin.scripts.python.Script",
        }

        if task.description:
            task_dict["description"] = task.description

        # Set up Docker image with Python
        task_dict["docker"] = {
            "image": f"python:{self.python_version}-slim",
        }

        # Install dependencies
        task_dict["beforeCommands"] = [
            "pip install truthound-kestra polars",
        ]

        # Generate script
        task_dict["script"] = task.generate_script()

        # Add timeout
        if task.timeout_seconds:
            task_dict["timeout"] = f"PT{int(task.timeout_seconds)}S"

        # Add retry configuration
        retry = task.retry or default_retry
        if retry:
            retry_dict = retry.to_dict()
            if retry_dict:
                task_dict["retry"] = retry_dict

        # Add dependencies
        if task.depends_on:
            task_dict["dependsOn"] = list(task.depends_on)

        # Add conditions
        if task.conditions:
            task_dict["runIf"] = list(task.conditions)

        return task_dict


# Module-level generator instance
_generator = FlowGenerator()


def generate_flow_yaml(config: FlowConfig) -> str:
    """Generate YAML content from flow configuration.

    Args:
        config: Flow configuration.

    Returns:
        YAML string representation of the flow.

    Example:
        >>> flow = FlowConfig(
        ...     id="my_flow",
        ...     namespace="production",
        ...     tasks=[TaskConfig(id="task1")]
        ... )
        >>> yaml_content = generate_flow_yaml(flow)
    """
    return _generator.generate(config)


def generate_check_flow(
    flow_id: str,
    namespace: str = "default",
    input_uri: str | None = None,
    rules: list[RuleDict] | None = None,
    engine_name: str = "truthound",
    fail_on_error: bool = True,
    schedule: str | None = None,
    description: str = "",
) -> str:
    """Generate a simple check quality flow.

    This is a convenience function to quickly generate a flow
    with a single check task.

    Args:
        flow_id: Flow identifier.
        namespace: Kestra namespace.
        input_uri: URI of data to check.
        rules: Validation rules.
        engine_name: Data quality engine.
        fail_on_error: Whether to fail on check failure.
        schedule: Cron expression for scheduling.
        description: Flow description.

    Returns:
        YAML string representation of the flow.

    Example:
        >>> yaml_content = generate_check_flow(
        ...     flow_id="check_users",
        ...     namespace="production",
        ...     input_uri="s3://bucket/users.parquet",
        ...     rules=[{"type": "not_null", "column": "id"}],
        ...     schedule="0 0 * * *"  # Daily at midnight
        ... )
    """
    # Build inputs
    inputs = []
    if input_uri is None:
        inputs.append(InputConfig(
            name="data_uri",
            type="STRING",
            required=True,
            description="URI of data to validate",
        ))

    # Build task
    task = TaskConfig(
        id="check_quality",
        task_type=TaskType.CHECK,
        input_uri=input_uri,
        rules=tuple(rules or []),
        engine_name=engine_name,
        fail_on_error=fail_on_error,
    )

    # Build triggers
    triggers = []
    if schedule:
        triggers.append(TriggerConfig(
            id="schedule",
            type=TriggerType.SCHEDULE,
            cron=schedule,
        ))

    # Build outputs
    outputs = [
        OutputConfig(
            name="check_result",
            value="{{ outputs.check_quality.result }}",
        ),
        OutputConfig(
            name="status",
            value="{{ outputs.check_quality.status }}",
        ),
    ]

    # Build flow
    config = FlowConfig(
        id=flow_id,
        namespace=namespace,
        description=description or "Data quality check flow",
        inputs=tuple(inputs),
        tasks=(task,),
        triggers=tuple(triggers),
        outputs=tuple(outputs),
    )

    return generate_flow_yaml(config)


def generate_profile_flow(
    flow_id: str,
    namespace: str = "default",
    input_uri: str | None = None,
    engine_name: str = "truthound",
    schedule: str | None = None,
    description: str = "",
) -> str:
    """Generate a data profiling flow.

    Args:
        flow_id: Flow identifier.
        namespace: Kestra namespace.
        input_uri: URI of data to profile.
        engine_name: Data quality engine.
        schedule: Cron expression for scheduling.
        description: Flow description.

    Returns:
        YAML string representation of the flow.

    Example:
        >>> yaml_content = generate_profile_flow(
        ...     flow_id="profile_users",
        ...     namespace="production",
        ...     input_uri="s3://bucket/users.parquet"
        ... )
    """
    inputs = []
    if input_uri is None:
        inputs.append(InputConfig(
            name="data_uri",
            type="STRING",
            required=True,
            description="URI of data to profile",
        ))

    task = TaskConfig(
        id="profile_data",
        task_type=TaskType.PROFILE,
        input_uri=input_uri,
        engine_name=engine_name,
    )

    triggers = []
    if schedule:
        triggers.append(TriggerConfig(
            id="schedule",
            type=TriggerType.SCHEDULE,
            cron=schedule,
        ))

    outputs = [
        OutputConfig(
            name="profile_result",
            value="{{ outputs.profile_data.result }}",
        ),
    ]

    config = FlowConfig(
        id=flow_id,
        namespace=namespace,
        description=description or "Data profiling flow",
        inputs=tuple(inputs),
        tasks=(task,),
        triggers=tuple(triggers),
        outputs=tuple(outputs),
    )

    return generate_flow_yaml(config)


def generate_learn_flow(
    flow_id: str,
    namespace: str = "default",
    input_uri: str | None = None,
    engine_name: str = "truthound",
    min_confidence: float = 0.8,
    schedule: str | None = None,
    description: str = "",
) -> str:
    """Generate a schema learning flow.

    Args:
        flow_id: Flow identifier.
        namespace: Kestra namespace.
        input_uri: URI of data to learn from.
        engine_name: Data quality engine.
        min_confidence: Minimum confidence for rules.
        schedule: Cron expression for scheduling.
        description: Flow description.

    Returns:
        YAML string representation of the flow.

    Example:
        >>> yaml_content = generate_learn_flow(
        ...     flow_id="learn_schema",
        ...     namespace="production",
        ...     input_uri="s3://bucket/baseline.parquet"
        ... )
    """
    inputs = []
    if input_uri is None:
        inputs.append(InputConfig(
            name="data_uri",
            type="STRING",
            required=True,
            description="URI of baseline data",
        ))

    task = TaskConfig(
        id="learn_schema",
        task_type=TaskType.LEARN,
        input_uri=input_uri,
        engine_name=engine_name,
        extra_args={"min_confidence": min_confidence},
    )

    triggers = []
    if schedule:
        triggers.append(TriggerConfig(
            id="schedule",
            type=TriggerType.SCHEDULE,
            cron=schedule,
        ))

    outputs = [
        OutputConfig(
            name="learn_result",
            value="{{ outputs.learn_schema.result }}",
        ),
        OutputConfig(
            name="learned_rules",
            value="{{ outputs.learn_schema.learned_rules }}",
        ),
    ]

    config = FlowConfig(
        id=flow_id,
        namespace=namespace,
        description=description or "Schema learning flow",
        inputs=tuple(inputs),
        tasks=(task,),
        triggers=tuple(triggers),
        outputs=tuple(outputs),
    )

    return generate_flow_yaml(config)


def generate_quality_pipeline(
    flow_id: str,
    namespace: str = "default",
    input_uri: str | None = None,
    rules: list[RuleDict] | None = None,
    engine_name: str = "truthound",
    include_profile: bool = True,
    include_learn: bool = False,
    fail_on_error: bool = True,
    schedule: str | None = None,
    description: str = "",
) -> str:
    """Generate a complete data quality pipeline.

    This function generates a flow with multiple tasks:
    1. Profile (optional) - Generate data profile
    2. Check - Run quality checks
    3. Learn (optional) - Learn rules from data

    Args:
        flow_id: Flow identifier.
        namespace: Kestra namespace.
        input_uri: URI of data to process.
        rules: Validation rules for check task.
        engine_name: Data quality engine.
        include_profile: Whether to include profiling.
        include_learn: Whether to include learning.
        fail_on_error: Whether to fail on check failure.
        schedule: Cron expression for scheduling.
        description: Flow description.

    Returns:
        YAML string representation of the flow.

    Example:
        >>> yaml_content = generate_quality_pipeline(
        ...     flow_id="data_quality_pipeline",
        ...     namespace="production",
        ...     input_uri="s3://bucket/data.parquet",
        ...     rules=[{"type": "not_null", "column": "id"}],
        ...     include_profile=True,
        ...     schedule="0 6 * * *"  # Daily at 6 AM
        ... )
    """
    inputs = []
    if input_uri is None:
        inputs.append(InputConfig(
            name="data_uri",
            type="STRING",
            required=True,
            description="URI of data to process",
        ))

    tasks = []
    outputs = []

    # Profile task
    if include_profile:
        profile_task = TaskConfig(
            id="profile",
            task_type=TaskType.PROFILE,
            input_uri=input_uri,
            engine_name=engine_name,
            description="Generate data profile",
        )
        tasks.append(profile_task)
        outputs.append(OutputConfig(
            name="profile_result",
            value="{{ outputs.profile.result }}",
        ))

    # Check task
    check_task = TaskConfig(
        id="check",
        task_type=TaskType.CHECK,
        input_uri=input_uri,
        rules=tuple(rules or []),
        engine_name=engine_name,
        fail_on_error=fail_on_error,
        depends_on=("profile",) if include_profile else (),
        description="Run quality checks",
    )
    tasks.append(check_task)
    outputs.extend([
        OutputConfig(
            name="check_result",
            value="{{ outputs.check.result }}",
        ),
        OutputConfig(
            name="check_status",
            value="{{ outputs.check.status }}",
        ),
    ])

    # Learn task
    if include_learn:
        learn_task = TaskConfig(
            id="learn",
            task_type=TaskType.LEARN,
            input_uri=input_uri,
            engine_name=engine_name,
            depends_on=("check",),
            description="Learn schema from data",
        )
        tasks.append(learn_task)
        outputs.append(OutputConfig(
            name="learned_rules",
            value="{{ outputs.learn.learned_rules }}",
        ))

    # Triggers
    triggers = []
    if schedule:
        triggers.append(TriggerConfig(
            id="schedule",
            type=TriggerType.SCHEDULE,
            cron=schedule,
        ))

    # Build flow
    config = FlowConfig(
        id=flow_id,
        namespace=namespace,
        description=description or "Data quality pipeline",
        inputs=tuple(inputs),
        tasks=tuple(tasks),
        triggers=tuple(triggers),
        outputs=tuple(outputs),
        retry=RetryConfig(
            max_attempts=3,
            initial_delay_seconds=30.0,
            policy=RetryPolicy.EXPONENTIAL,
        ),
    )

    return generate_flow_yaml(config)
