"""Tests for truthound_kestra.flows module."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from truthound_kestra.flows import (
    # Configuration
    FlowConfig,
    TaskConfig,
    TriggerConfig,
    InputConfig,
    OutputConfig,
    RetryConfig,
    # Enums
    TaskType,
    TriggerType,
    RetryPolicy,
    # Generator
    FlowGenerator,
    # Functions
    generate_flow_yaml,
    generate_check_flow,
    generate_profile_flow,
    generate_learn_flow,
    generate_quality_pipeline,
)


class TestEnums:
    """Tests for flow enums."""

    def test_task_type_values(self) -> None:
        """Test TaskType enum values."""
        assert TaskType.CHECK.value == "check"
        assert TaskType.PROFILE.value == "profile"
        assert TaskType.LEARN.value == "learn"
        assert TaskType.PYTHON_SCRIPT.value == "python_script"

    def test_trigger_type_values(self) -> None:
        """Test TriggerType enum values."""
        assert TriggerType.SCHEDULE.value == "schedule"
        assert TriggerType.FLOW.value == "flow"
        assert TriggerType.WEBHOOK.value == "webhook"
        assert TriggerType.POLLING.value == "polling"

    def test_retry_policy_values(self) -> None:
        """Test RetryPolicy enum values."""
        assert RetryPolicy.NONE.value == "none"
        assert RetryPolicy.CONSTANT.value == "constant"
        assert RetryPolicy.EXPONENTIAL.value == "exponential"
        assert RetryPolicy.LINEAR.value == "linear"


class TestFlowConfig:
    """Tests for FlowConfig."""

    def test_flow_config_defaults(self) -> None:
        """Test FlowConfig default values."""
        config = FlowConfig(id="test_flow")

        assert config.id == "test_flow"
        assert config.namespace == "default"
        assert config.description == ""
        assert config.disabled is False
        assert config.tasks == ()
        assert config.triggers == ()
        assert config.inputs == ()
        assert config.outputs == ()

    def test_flow_config_with_all_fields(self) -> None:
        """Test FlowConfig with all fields."""
        task = TaskConfig(
            id="check_task",
            task_type=TaskType.CHECK,
            input_uri="s3://bucket/data.parquet",
        )
        trigger = TriggerConfig(
            id="daily",
            type=TriggerType.SCHEDULE,
            cron="0 0 * * *",
        )
        input_config = InputConfig(
            name="input_uri",
            type="STRING",
            required=True,
        )
        output_config = OutputConfig(
            name="result",
            value="{{ outputs.check_task.result }}",
        )

        config = FlowConfig(
            id="full_flow",
            namespace="company.data",
            description="Full test flow",
            disabled=False,
            tasks=(task,),
            triggers=(trigger,),
            inputs=(input_config,),
            outputs=(output_config,),
            labels=frozenset(["production"]),
        )

        assert config.id == "full_flow"
        assert len(config.tasks) == 1
        assert len(config.triggers) == 1
        assert len(config.inputs) == 1
        assert len(config.outputs) == 1
        assert "production" in config.labels

    def test_flow_config_builder(self) -> None:
        """Test FlowConfig builder pattern."""
        config = FlowConfig(id="test", namespace="ns")
        config = config.with_namespace("new_namespace")

        assert config.namespace == "new_namespace"

    def test_flow_config_add_task(self) -> None:
        """Test FlowConfig add_task method."""
        config = FlowConfig(id="test", namespace="ns")
        task = TaskConfig(id="task1", task_type=TaskType.CHECK)
        config = config.add_task(task)

        assert len(config.tasks) == 1
        assert config.tasks[0].id == "task1"

    def test_flow_config_with_triggers(self) -> None:
        """Test FlowConfig with_triggers method."""
        config = FlowConfig(id="test", namespace="ns")
        trigger = TriggerConfig(id="trig1", type=TriggerType.SCHEDULE, cron="0 * * * *")
        config = config.with_triggers(trigger)

        assert len(config.triggers) == 1
        assert config.triggers[0].id == "trig1"

    def test_flow_config_to_dict(self) -> None:
        """Test FlowConfig serialization."""
        config = FlowConfig(
            id="test_flow",
            namespace="company.data",
            description="Test",
        )
        d = config.to_dict()

        assert d["id"] == "test_flow"
        assert d["namespace"] == "company.data"
        assert d["description"] == "Test"

    def test_flow_config_immutability(self) -> None:
        """Test that FlowConfig is immutable."""
        config = FlowConfig(id="test", namespace="ns")
        with pytest.raises(AttributeError):
            config.id = "new_id"  # type: ignore


class TestTaskConfig:
    """Tests for TaskConfig."""

    def test_task_config_check(self) -> None:
        """Test TaskConfig for check task."""
        config = TaskConfig(
            id="check_task",
            task_type=TaskType.CHECK,
            input_uri="s3://bucket/data.parquet",
            rules=({"type": "not_null", "column": "id"},),
        )

        assert config.id == "check_task"
        assert config.task_type == TaskType.CHECK
        assert len(config.rules) == 1

    def test_task_config_with_depends_on(self) -> None:
        """Test TaskConfig with dependencies."""
        config = TaskConfig(
            id="task",
            task_type=TaskType.CHECK,
            depends_on=("upstream_task",),
        )

        assert "upstream_task" in config.depends_on

    def test_task_config_with_retry(self) -> None:
        """Test TaskConfig with retry configuration."""
        retry = RetryConfig(
            max_attempts=3,
            policy=RetryPolicy.EXPONENTIAL,
        )
        config = TaskConfig(
            id="task",
            task_type=TaskType.CHECK,
            retry=retry,
        )

        assert config.retry is not None
        assert config.retry.max_attempts == 3

    def test_task_config_builder(self) -> None:
        """Test TaskConfig builder pattern."""
        config = TaskConfig(id="task", task_type=TaskType.CHECK)
        config = config.with_rules([{"type": "not_null", "column": "id"}])
        config = config.with_depends_on("other_task")

        assert len(config.rules) == 1
        assert "other_task" in config.depends_on

    def test_task_config_to_dict(self) -> None:
        """Test TaskConfig serialization."""
        config = TaskConfig(
            id="task",
            task_type=TaskType.CHECK,
            input_uri="s3://bucket/data.parquet",
        )
        d = config.to_dict()

        assert d["id"] == "task"
        assert d["task_type"] == "check"
        assert d["input_uri"] == "s3://bucket/data.parquet"


class TestTriggerConfig:
    """Tests for TriggerConfig."""

    def test_schedule_trigger(self) -> None:
        """Test TriggerConfig for schedule trigger."""
        config = TriggerConfig(
            id="daily",
            type=TriggerType.SCHEDULE,
            cron="0 0 * * *",
        )

        assert config.id == "daily"
        assert config.type == TriggerType.SCHEDULE
        assert config.cron == "0 0 * * *"

    def test_flow_trigger(self) -> None:
        """Test TriggerConfig for flow trigger."""
        config = TriggerConfig(
            id="on_upstream",
            type=TriggerType.FLOW,
            flow_id="upstream_flow",
            namespace="production",
        )

        assert config.type == TriggerType.FLOW
        assert config.flow_id == "upstream_flow"

    def test_webhook_trigger(self) -> None:
        """Test TriggerConfig for webhook trigger."""
        config = TriggerConfig(
            id="webhook",
            type=TriggerType.WEBHOOK,
        )

        assert config.type == TriggerType.WEBHOOK

    def test_trigger_config_to_dict(self) -> None:
        """Test TriggerConfig serialization."""
        config = TriggerConfig(
            id="daily",
            type=TriggerType.SCHEDULE,
            cron="0 0 * * *",
        )
        d = config.to_dict()

        assert d["id"] == "daily"
        assert "cron" in d


class TestInputConfig:
    """Tests for InputConfig."""

    def test_string_input(self) -> None:
        """Test InputConfig for string input."""
        config = InputConfig(
            name="input_uri",
            type="STRING",
            required=True,
            description="Input data URI",
        )

        assert config.name == "input_uri"
        assert config.type == "STRING"
        assert config.required is True

    def test_input_with_defaults(self) -> None:
        """Test InputConfig with defaults."""
        config = InputConfig(
            name="threshold",
            type="FLOAT",
            required=False,
            default="0.95",
        )

        assert config.default == "0.95"
        assert config.required is False

    def test_input_to_dict(self) -> None:
        """Test InputConfig serialization."""
        config = InputConfig(name="test", type="STRING")
        d = config.to_dict()

        assert d["id"] == "test"
        assert d["type"] == "STRING"


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_output_config(self) -> None:
        """Test OutputConfig creation."""
        config = OutputConfig(
            name="result",
            value="{{ outputs.check_task.result }}",
        )

        assert config.name == "result"
        assert "outputs.check_task" in config.value

    def test_output_to_dict(self) -> None:
        """Test OutputConfig serialization."""
        config = OutputConfig(name="result", value="value")
        d = config.to_dict()

        assert d["id"] == "result"
        assert d["value"] == "value"


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_retry_config_defaults(self) -> None:
        """Test RetryConfig defaults."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.policy == RetryPolicy.EXPONENTIAL

    def test_retry_config_custom(self) -> None:
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            policy=RetryPolicy.CONSTANT,
            initial_delay_seconds=60.0,
            max_delay_seconds=3600.0,
        )

        assert config.max_attempts == 5
        assert config.policy == RetryPolicy.CONSTANT
        assert config.initial_delay_seconds == 60.0
        assert config.max_delay_seconds == 3600.0

    def test_retry_config_builder(self) -> None:
        """Test RetryConfig builder pattern."""
        config = RetryConfig()
        config = config.with_max_attempts(10)

        assert config.max_attempts == 10


class TestFlowGenerator:
    """Tests for FlowGenerator."""

    def test_generator_creation(self) -> None:
        """Test FlowGenerator creation."""
        generator = FlowGenerator()
        assert generator is not None

    def test_generator_generate_simple_flow(self) -> None:
        """Test generating a simple flow."""
        config = FlowConfig(
            id="simple_flow",
            namespace="company.data",
            description="Simple test flow",
        )
        task = TaskConfig(
            id="task1",
            task_type=TaskType.CHECK,
            input_uri="s3://bucket/data.parquet",
        )
        config = config.add_task(task)

        generator = FlowGenerator()
        yaml_content = generator.generate(config)

        assert isinstance(yaml_content, str)
        # Parse to verify valid YAML
        parsed = yaml.safe_load(yaml_content)
        assert parsed["id"] == "simple_flow"
        assert parsed["namespace"] == "company.data"

    def test_generator_generate_with_trigger(self) -> None:
        """Test generating flow with trigger."""
        config = FlowConfig(id="triggered", namespace="ns")
        trigger = TriggerConfig(
            id="daily",
            type=TriggerType.SCHEDULE,
            cron="0 0 * * *",
        )
        config = config.with_triggers(trigger)

        generator = FlowGenerator()
        yaml_content = generator.generate(config)
        parsed = yaml.safe_load(yaml_content)

        assert "triggers" in parsed
        assert len(parsed["triggers"]) == 1

    def test_generator_generate_with_inputs(self) -> None:
        """Test generating flow with inputs."""
        config = FlowConfig(id="with_inputs", namespace="ns")
        input_config = InputConfig(name="uri", type="STRING", required=True)
        config = FlowConfig(
            id="with_inputs",
            namespace="ns",
            inputs=(input_config,),
        )

        generator = FlowGenerator()
        yaml_content = generator.generate(config)
        parsed = yaml.safe_load(yaml_content)

        assert "inputs" in parsed
        assert len(parsed["inputs"]) == 1


class TestGeneratorFunctions:
    """Tests for generator convenience functions."""

    def test_generate_flow_yaml(self) -> None:
        """Test generate_flow_yaml function."""
        config = FlowConfig(id="test", namespace="ns")
        yaml_content = generate_flow_yaml(config)

        assert isinstance(yaml_content, str)
        parsed = yaml.safe_load(yaml_content)
        assert parsed["id"] == "test"

    def test_generate_check_flow(self) -> None:
        """Test generate_check_flow function."""
        yaml_content = generate_check_flow(
            flow_id="check_flow",
            namespace="company.data",
            input_uri="{{ inputs.uri }}",
            rules=[{"type": "not_null", "column": "id"}],
        )

        assert isinstance(yaml_content, str)
        parsed = yaml.safe_load(yaml_content)
        assert parsed["id"] == "check_flow"
        assert len(parsed["tasks"]) > 0

    def test_generate_profile_flow(self) -> None:
        """Test generate_profile_flow function."""
        yaml_content = generate_profile_flow(
            flow_id="profile_flow",
            namespace="company.data",
            input_uri="{{ inputs.uri }}",
        )

        assert isinstance(yaml_content, str)
        parsed = yaml.safe_load(yaml_content)
        assert parsed["id"] == "profile_flow"

    def test_generate_learn_flow(self) -> None:
        """Test generate_learn_flow function."""
        yaml_content = generate_learn_flow(
            flow_id="learn_flow",
            namespace="company.data",
            input_uri="{{ inputs.uri }}",
        )

        assert isinstance(yaml_content, str)
        parsed = yaml.safe_load(yaml_content)
        assert parsed["id"] == "learn_flow"

    def test_generate_quality_pipeline(self) -> None:
        """Test generate_quality_pipeline function."""
        yaml_content = generate_quality_pipeline(
            flow_id="pipeline",
            namespace="company.data",
            input_uri="{{ inputs.uri }}",
            include_profile=True,
            include_learn=True,
        )

        assert isinstance(yaml_content, str)
        parsed = yaml.safe_load(yaml_content)
        assert parsed["id"] == "pipeline"
        # Should have multiple tasks for check, profile, learn
        assert len(parsed["tasks"]) >= 1
