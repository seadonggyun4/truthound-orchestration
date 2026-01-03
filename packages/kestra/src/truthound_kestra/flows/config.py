"""Flow configuration for Kestra data quality integration.

This module provides configuration classes for generating Kestra
flow definitions programmatically.

Example:
    >>> from truthound_kestra.flows.config import (
    ...     FlowConfig,
    ...     TaskConfig,
    ...     TriggerConfig,
    ... )
    >>>
    >>> flow = FlowConfig(
    ...     id="data_quality_pipeline",
    ...     namespace="production",
    ...     description="Daily data quality checks",
    ...     tasks=[
    ...         TaskConfig(
    ...             id="check_users",
    ...             script_type="check",
    ...             input_uri="s3://bucket/users.parquet"
    ...         )
    ...     ]
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from truthound_kestra.utils.exceptions import ConfigurationError

__all__ = [
    # Enums
    "TaskType",
    "TriggerType",
    "RetryPolicy",
    # Config classes
    "FlowConfig",
    "TaskConfig",
    "TriggerConfig",
    "RetryConfig",
    "InputConfig",
    "OutputConfig",
    # Presets
    "DEFAULT_FLOW_CONFIG",
    "PRODUCTION_FLOW_CONFIG",
]


class TaskType(str, Enum):
    """Type of data quality task.

    Values:
        CHECK: Data quality check task.
        PROFILE: Data profiling task.
        LEARN: Schema learning task.
        PYTHON_SCRIPT: Generic Python script task.
    """

    CHECK = "check"
    PROFILE = "profile"
    LEARN = "learn"
    PYTHON_SCRIPT = "python_script"


class TriggerType(str, Enum):
    """Type of flow trigger.

    Values:
        SCHEDULE: Cron-based scheduling.
        FLOW: Triggered by another flow.
        WEBHOOK: HTTP webhook trigger.
        POLLING: Polling for changes.
    """

    SCHEDULE = "schedule"
    FLOW = "flow"
    WEBHOOK = "webhook"
    POLLING = "polling"


class RetryPolicy(str, Enum):
    """Retry policy for failed tasks.

    Values:
        NONE: No retries.
        CONSTANT: Fixed delay between retries.
        EXPONENTIAL: Exponential backoff.
        LINEAR: Linear increase in delay.
    """

    NONE = "none"
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for task retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        initial_delay_seconds: Initial delay before first retry.
        max_delay_seconds: Maximum delay between retries.
        policy: Retry policy to use.
        multiplier: Multiplier for exponential backoff.

    Example:
        >>> config = RetryConfig(
        ...     max_attempts=3,
        ...     initial_delay_seconds=10.0,
        ...     policy=RetryPolicy.EXPONENTIAL
        ... )
    """

    max_attempts: int = 3
    initial_delay_seconds: float = 10.0
    max_delay_seconds: float = 300.0
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_attempts < 0:
            raise ConfigurationError(
                message="max_attempts must be non-negative",
                field="max_attempts",
                value=self.max_attempts,
            )
        if self.initial_delay_seconds < 0:
            raise ConfigurationError(
                message="initial_delay_seconds must be non-negative",
                field="initial_delay_seconds",
                value=self.initial_delay_seconds,
            )

    def with_max_attempts(self, max_attempts: int) -> RetryConfig:
        """Return new config with updated max_attempts."""
        return RetryConfig(
            max_attempts=max_attempts,
            initial_delay_seconds=self.initial_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            policy=self.policy,
            multiplier=self.multiplier,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML generation."""
        if self.policy == RetryPolicy.NONE:
            return {}

        return {
            "maxAttempt": self.max_attempts,
            "type": self.policy.value,
            "interval": f"PT{int(self.initial_delay_seconds)}S",
            "maxDuration": f"PT{int(self.max_delay_seconds)}S",
        }


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Configuration for flow input variables.

    Attributes:
        name: Input variable name.
        type: Input type (STRING, INT, BOOLEAN, FILE, etc.).
        required: Whether the input is required.
        default: Default value if not provided.
        description: Description of the input.

    Example:
        >>> input_cfg = InputConfig(
        ...     name="data_uri",
        ...     type="STRING",
        ...     required=True,
        ...     description="URI of data to validate"
        ... )
    """

    name: str
    type: str = "STRING"
    required: bool = True
    default: Any = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML generation."""
        result: dict[str, Any] = {
            "id": self.name,
            "type": self.type,
        }
        if not self.required:
            result["required"] = False
        if self.default is not None:
            result["defaults"] = self.default
        if self.description:
            result["description"] = self.description
        return result


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Configuration for task output variables.

    Attributes:
        name: Output variable name.
        value: Value expression (e.g., "{{ outputs.task.result }}").
        description: Description of the output.

    Example:
        >>> output_cfg = OutputConfig(
        ...     name="check_result",
        ...     value="{{ outputs.check_users.result }}",
        ... )
    """

    name: str
    value: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML generation."""
        result: dict[str, Any] = {
            "id": self.name,
            "value": self.value,
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    """Configuration for flow triggers.

    Attributes:
        id: Unique trigger identifier.
        type: Type of trigger (schedule, flow, webhook).
        cron: Cron expression for schedule triggers.
        flow_id: Flow ID for flow triggers.
        namespace: Namespace for flow triggers.
        conditions: Optional conditions for trigger.

    Example:
        >>> # Schedule trigger (daily at midnight)
        >>> trigger = TriggerConfig(
        ...     id="daily",
        ...     type=TriggerType.SCHEDULE,
        ...     cron="0 0 * * *"
        ... )
        >>>
        >>> # Flow trigger
        >>> trigger = TriggerConfig(
        ...     id="after_ingestion",
        ...     type=TriggerType.FLOW,
        ...     flow_id="data_ingestion",
        ...     namespace="production"
        ... )
    """

    id: str
    type: TriggerType
    cron: str | None = None
    flow_id: str | None = None
    namespace: str | None = None
    conditions: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.type == TriggerType.SCHEDULE and not self.cron:
            raise ConfigurationError(
                message="cron expression required for schedule triggers",
                field="cron",
            )
        if self.type == TriggerType.FLOW and not self.flow_id:
            raise ConfigurationError(
                message="flow_id required for flow triggers",
                field="flow_id",
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML generation."""
        result: dict[str, Any] = {"id": self.id}

        if self.type == TriggerType.SCHEDULE:
            result["type"] = "io.kestra.core.models.triggers.types.Schedule"
            result["cron"] = self.cron
        elif self.type == TriggerType.FLOW:
            result["type"] = "io.kestra.core.models.triggers.types.Flow"
            result["inputs"] = {
                "flowId": self.flow_id,
            }
            if self.namespace:
                result["inputs"]["namespace"] = self.namespace
        elif self.type == TriggerType.WEBHOOK:
            result["type"] = "io.kestra.core.models.triggers.types.Webhook"
        elif self.type == TriggerType.POLLING:
            result["type"] = "io.kestra.core.models.triggers.types.Polling"

        if self.conditions:
            result["conditions"] = list(self.conditions)

        return result


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """Configuration for a data quality task.

    Attributes:
        id: Unique task identifier.
        task_type: Type of task (check, profile, learn).
        input_uri: URI of input data.
        rules: Rules for check tasks.
        engine_name: Data quality engine to use.
        fail_on_error: Whether to fail flow on check failure.
        timeout_seconds: Task timeout.
        retry: Retry configuration.
        depends_on: List of task IDs this task depends on.
        conditions: Conditions for task execution.
        description: Task description.
        labels: Labels for the task.
        extra_args: Additional arguments for the script.

    Example:
        >>> task = TaskConfig(
        ...     id="check_users",
        ...     task_type=TaskType.CHECK,
        ...     input_uri="s3://bucket/users.parquet",
        ...     rules=(
        ...         {"type": "not_null", "column": "id"},
        ...         {"type": "unique", "column": "email"},
        ...     ),
        ...     fail_on_error=True
        ... )
    """

    id: str
    task_type: TaskType = TaskType.CHECK
    input_uri: str | None = None
    rules: tuple[dict[str, Any], ...] = ()
    engine_name: str = "truthound"
    fail_on_error: bool = True
    timeout_seconds: float = 300.0
    retry: RetryConfig | None = None
    depends_on: tuple[str, ...] = ()
    conditions: tuple[dict[str, Any], ...] = ()
    description: str = ""
    labels: frozenset[str] = field(default_factory=frozenset)
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.id:
            raise ConfigurationError(
                message="Task id is required",
                field="id",
            )

    def with_rules(self, rules: list[dict[str, Any]]) -> TaskConfig:
        """Return new config with updated rules."""
        return TaskConfig(
            id=self.id,
            task_type=self.task_type,
            input_uri=self.input_uri,
            rules=tuple(rules),
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
            depends_on=self.depends_on,
            conditions=self.conditions,
            description=self.description,
            labels=self.labels,
            extra_args=self.extra_args,
        )

    def with_depends_on(self, *task_ids: str) -> TaskConfig:
        """Return new config with updated dependencies."""
        return TaskConfig(
            id=self.id,
            task_type=self.task_type,
            input_uri=self.input_uri,
            rules=self.rules,
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            retry=self.retry,
            depends_on=tuple(task_ids),
            conditions=self.conditions,
            description=self.description,
            labels=self.labels,
            extra_args=self.extra_args,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "input_uri": self.input_uri,
            "rules": list(self.rules),
            "engine_name": self.engine_name,
            "fail_on_error": self.fail_on_error,
            "timeout_seconds": self.timeout_seconds,
            "retry": self.retry.to_dict() if self.retry else None,
            "depends_on": list(self.depends_on),
            "description": self.description,
            "labels": list(self.labels),
            "extra_args": self.extra_args,
        }

    def generate_script(self) -> str:
        """Generate Python script content for this task."""
        if self.task_type == TaskType.CHECK:
            return self._generate_check_script()
        elif self.task_type == TaskType.PROFILE:
            return self._generate_profile_script()
        elif self.task_type == TaskType.LEARN:
            return self._generate_learn_script()
        else:
            return self._generate_generic_script()

    def _generate_check_script(self) -> str:
        """Generate check script content."""
        lines = [
            "from truthound_kestra.scripts import check_quality_script",
            "",
            "check_quality_script(",
        ]

        if self.input_uri:
            lines.append(f'    input_uri="{self.input_uri}",')
        else:
            lines.append('    input_uri="{{ inputs.data_uri }}",')

        if self.rules:
            import json
            rules_str = json.dumps(list(self.rules), indent=8)
            lines.append(f"    rules={rules_str},")

        lines.append(f'    engine_name="{self.engine_name}",')
        lines.append(f"    fail_on_error={self.fail_on_error},")
        lines.append(")")

        return "\n".join(lines)

    def _generate_profile_script(self) -> str:
        """Generate profile script content."""
        lines = [
            "from truthound_kestra.scripts import profile_data_script",
            "",
            "profile_data_script(",
        ]

        if self.input_uri:
            lines.append(f'    input_uri="{self.input_uri}",')
        else:
            lines.append('    input_uri="{{ inputs.data_uri }}",')

        lines.append(f'    engine_name="{self.engine_name}",')
        lines.append(")")

        return "\n".join(lines)

    def _generate_learn_script(self) -> str:
        """Generate learn script content."""
        lines = [
            "from truthound_kestra.scripts import learn_schema_script",
            "",
            "learn_schema_script(",
        ]

        if self.input_uri:
            lines.append(f'    input_uri="{self.input_uri}",')
        else:
            lines.append('    input_uri="{{ inputs.data_uri }}",')

        lines.append(f'    engine_name="{self.engine_name}",')
        lines.append(")")

        return "\n".join(lines)

    def _generate_generic_script(self) -> str:
        """Generate generic script content."""
        return "# Add your custom script here\npass"


@dataclass(frozen=True, slots=True)
class FlowConfig:
    """Configuration for a complete Kestra flow.

    Attributes:
        id: Unique flow identifier.
        namespace: Kestra namespace.
        description: Flow description.
        tasks: List of task configurations.
        inputs: Flow input configurations.
        outputs: Flow output configurations.
        triggers: Flow trigger configurations.
        retry: Default retry configuration for tasks.
        labels: Labels for the flow.
        disabled: Whether the flow is disabled.
        variables: Flow variables.
        concurrency: Maximum concurrent executions.

    Example:
        >>> flow = FlowConfig(
        ...     id="data_quality_pipeline",
        ...     namespace="production",
        ...     description="Daily data quality checks",
        ...     tasks=[
        ...         TaskConfig(
        ...             id="check_users",
        ...             task_type=TaskType.CHECK,
        ...             input_uri="s3://bucket/users.parquet",
        ...             rules=({"type": "not_null", "column": "id"},)
        ...         )
        ...     ],
        ...     triggers=[
        ...         TriggerConfig(
        ...             id="daily",
        ...             type=TriggerType.SCHEDULE,
        ...             cron="0 0 * * *"
        ...         )
        ...     ]
        ... )
    """

    id: str
    namespace: str = "default"
    description: str = ""
    tasks: tuple[TaskConfig, ...] = ()
    inputs: tuple[InputConfig, ...] = ()
    outputs: tuple[OutputConfig, ...] = ()
    triggers: tuple[TriggerConfig, ...] = ()
    retry: RetryConfig | None = None
    labels: frozenset[str] = field(default_factory=frozenset)
    disabled: bool = False
    variables: dict[str, Any] = field(default_factory=dict)
    concurrency: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.id:
            raise ConfigurationError(
                message="Flow id is required",
                field="id",
            )

    def with_namespace(self, namespace: str) -> FlowConfig:
        """Return new config with updated namespace."""
        return FlowConfig(
            id=self.id,
            namespace=namespace,
            description=self.description,
            tasks=self.tasks,
            inputs=self.inputs,
            outputs=self.outputs,
            triggers=self.triggers,
            retry=self.retry,
            labels=self.labels,
            disabled=self.disabled,
            variables=self.variables,
            concurrency=self.concurrency,
        )

    def with_tasks(self, *tasks: TaskConfig) -> FlowConfig:
        """Return new config with updated tasks."""
        return FlowConfig(
            id=self.id,
            namespace=self.namespace,
            description=self.description,
            tasks=tuple(tasks),
            inputs=self.inputs,
            outputs=self.outputs,
            triggers=self.triggers,
            retry=self.retry,
            labels=self.labels,
            disabled=self.disabled,
            variables=self.variables,
            concurrency=self.concurrency,
        )

    def add_task(self, task: TaskConfig) -> FlowConfig:
        """Return new config with added task."""
        return FlowConfig(
            id=self.id,
            namespace=self.namespace,
            description=self.description,
            tasks=self.tasks + (task,),
            inputs=self.inputs,
            outputs=self.outputs,
            triggers=self.triggers,
            retry=self.retry,
            labels=self.labels,
            disabled=self.disabled,
            variables=self.variables,
            concurrency=self.concurrency,
        )

    def with_triggers(self, *triggers: TriggerConfig) -> FlowConfig:
        """Return new config with updated triggers."""
        return FlowConfig(
            id=self.id,
            namespace=self.namespace,
            description=self.description,
            tasks=self.tasks,
            inputs=self.inputs,
            outputs=self.outputs,
            triggers=tuple(triggers),
            retry=self.retry,
            labels=self.labels,
            disabled=self.disabled,
            variables=self.variables,
            concurrency=self.concurrency,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "namespace": self.namespace,
            "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "triggers": [t.to_dict() for t in self.triggers],
            "retry": self.retry.to_dict() if self.retry else None,
            "labels": list(self.labels),
            "disabled": self.disabled,
            "variables": self.variables,
            "concurrency": self.concurrency,
        }


# Preset configurations
DEFAULT_FLOW_CONFIG = FlowConfig(
    id="data_quality",
    namespace="default",
)

PRODUCTION_FLOW_CONFIG = FlowConfig(
    id="data_quality",
    namespace="production",
    retry=RetryConfig(
        max_attempts=3,
        initial_delay_seconds=30.0,
        policy=RetryPolicy.EXPONENTIAL,
    ),
    concurrency=1,
)
