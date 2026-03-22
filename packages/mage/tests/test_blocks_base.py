"""Tests for Mage blocks base configurations."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from truthound_mage.blocks.base import (
    BlockConfig,
    CheckBlockConfig,
    ProfileBlockConfig,
    LearnBlockConfig,
    BlockExecutionContext,
    BlockResult,
    BlockType,
    ExecutionMode,
    DEFAULT_BLOCK_CONFIG,
    STRICT_BLOCK_CONFIG,
    LENIENT_BLOCK_CONFIG,
)


class TestBlockConfig:
    """Tests for BlockConfig base class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BlockConfig()
        assert config.engine_name is None
        assert config.fail_on_error is True
        assert config.timeout_seconds == 300
        assert config.tags == frozenset()
        assert config.output_key == "data_quality_result"
        assert config.log_results is True

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = BlockConfig()
        with pytest.raises(AttributeError):
            config.fail_on_error = False  # type: ignore[misc]

    def test_with_engine_name(self) -> None:
        """Test engine_name builder method."""
        config = BlockConfig()
        new_config = config.with_engine_name("truthound")
        assert new_config.engine_name == "truthound"
        assert config.engine_name is None  # Original unchanged

    def test_with_fail_on_error(self) -> None:
        """Test fail_on_error builder method."""
        config = BlockConfig()
        new_config = config.with_fail_on_error(False)
        assert new_config.fail_on_error is False
        assert config.fail_on_error is True  # Original unchanged

    def test_with_timeout(self) -> None:
        """Test timeout builder method."""
        config = BlockConfig()
        new_config = config.with_timeout(60)
        assert new_config.timeout_seconds == 60

    def test_with_timeout_validation(self) -> None:
        """Test timeout validation."""
        with pytest.raises(ValueError, match="positive"):
            BlockConfig(timeout_seconds=0)

    def test_with_tags(self) -> None:
        """Test tags builder method."""
        config = BlockConfig()
        new_config = config.with_tags(frozenset({"prod", "critical"}))
        assert new_config.tags == frozenset({"prod", "critical"})

    def test_with_extra(self) -> None:
        """Test extra options builder method."""
        config = BlockConfig()
        new_config = config.with_extra(custom_key="custom_value")
        assert new_config.extra["custom_key"] == "custom_value"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = BlockConfig(engine_name="test", fail_on_error=False)
        data = config.to_dict()
        assert data["engine_name"] == "test"
        assert data["fail_on_error"] is False
        assert data["timeout_seconds"] == 300

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"engine_name": "test", "timeout_seconds": 60}
        config = BlockConfig.from_dict(data)
        assert config.engine_name == "test"
        assert config.timeout_seconds == 60


class TestCheckBlockConfig:
    """Tests for CheckBlockConfig."""

    def test_default_values(self) -> None:
        """Test default check configuration."""
        config = CheckBlockConfig()
        assert config.rules == ()
        assert config.warning_threshold is None
        assert config.parallel is True
        assert config.auto_schema is False
        assert config.sample_size is None

    def test_with_rules(self) -> None:
        """Test rules builder method."""
        config = CheckBlockConfig()
        rules = [{"type": "not_null", "column": "id"}]
        new_config = config.with_rules(rules)
        assert len(new_config.rules) == 1
        assert new_config.rules[0]["type"] == "not_null"

    def test_with_warning_threshold(self) -> None:
        """Test warning threshold builder."""
        config = CheckBlockConfig()
        new_config = config.with_warning_threshold(0.9)
        assert new_config.warning_threshold == 0.9

    def test_with_warning_threshold_validation(self) -> None:
        """Test warning threshold validation."""
        with pytest.raises(ValueError):
            CheckBlockConfig(warning_threshold=1.5)

    def test_with_auto_schema(self) -> None:
        """Test auto schema builder."""
        config = CheckBlockConfig()
        new_config = config.with_auto_schema(True)
        assert new_config.auto_schema is True

    def test_with_parallel(self) -> None:
        """Test parallel builder method."""
        config = CheckBlockConfig()
        new_config = config.with_parallel(False)
        assert new_config.parallel is False

    def test_with_sample_size(self) -> None:
        """Test sample size builder method."""
        config = CheckBlockConfig()
        new_config = config.with_sample_size(1000)
        assert new_config.sample_size == 1000

    def test_sample_size_validation(self) -> None:
        """Test sample size validation."""
        with pytest.raises(ValueError):
            CheckBlockConfig(sample_size=-1)

    def test_min_severity_validation(self) -> None:
        """Test min_severity validation."""
        # Valid severities
        config = CheckBlockConfig(min_severity="critical")
        assert config.min_severity == "critical"

        # Invalid severity
        with pytest.raises(ValueError):
            CheckBlockConfig(min_severity="invalid")

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
            parallel=True,
        )
        data = config.to_dict()
        assert len(data["rules"]) == 1
        assert data["parallel"] is True

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "rules": [{"type": "not_null", "column": "id"}],
            "parallel": False,
        }
        config = CheckBlockConfig.from_dict(data)
        assert len(config.rules) == 1
        assert config.parallel is False


class TestProfileBlockConfig:
    """Tests for ProfileBlockConfig."""

    def test_default_values(self) -> None:
        """Test default profile configuration."""
        config = ProfileBlockConfig()
        assert config.include_statistics is True
        assert config.include_patterns is True
        assert config.include_distributions is True
        assert config.sample_size is None
        assert config.columns is None

    def test_with_columns(self) -> None:
        """Test columns builder method."""
        config = ProfileBlockConfig()
        new_config = config.with_columns(["col1", "col2"])
        assert new_config.columns == frozenset({"col1", "col2"})

    def test_with_statistics(self) -> None:
        """Test statistics builder method."""
        config = ProfileBlockConfig()
        new_config = config.with_statistics(False)
        assert new_config.include_statistics is False

    def test_sample_size_validation(self) -> None:
        """Test sample size validation."""
        with pytest.raises(ValueError):
            ProfileBlockConfig(sample_size=-1)

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = ProfileBlockConfig(
            columns=frozenset({"col1", "col2"}),
            include_statistics=True,
        )
        data = config.to_dict()
        assert set(data["columns"]) == {"col1", "col2"}
        assert data["include_statistics"] is True


class TestLearnBlockConfig:
    """Tests for LearnBlockConfig."""

    def test_default_values(self) -> None:
        """Test default learn configuration."""
        config = LearnBlockConfig()
        assert config.output_path is None
        assert config.strictness == "moderate"
        assert config.infer_constraints is True
        assert config.categorical_threshold == 20

    def test_with_output_path(self) -> None:
        """Test output path builder method."""
        config = LearnBlockConfig()
        new_config = config.with_output_path("/tmp/schema.json")
        assert new_config.output_path == "/tmp/schema.json"

    def test_with_strictness(self) -> None:
        """Test strictness builder method."""
        config = LearnBlockConfig()
        new_config = config.with_strictness("strict")
        assert new_config.strictness == "strict"

    def test_strictness_validation(self) -> None:
        """Test strictness validation."""
        with pytest.raises(ValueError):
            LearnBlockConfig(strictness="invalid")

    def test_categorical_threshold_validation(self) -> None:
        """Test categorical threshold validation."""
        with pytest.raises(ValueError):
            LearnBlockConfig(categorical_threshold=0)

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = LearnBlockConfig(
            output_path="/tmp/schema.json",
            strictness="strict",
        )
        data = config.to_dict()
        assert data["output_path"] == "/tmp/schema.json"
        assert data["strictness"] == "strict"


class TestBlockExecutionContext:
    """Tests for BlockExecutionContext."""

    def test_default_values(self) -> None:
        """Test default context values."""
        context = BlockExecutionContext()
        assert context.block_uuid == ""
        assert context.pipeline_uuid == ""
        assert context.partition is None
        assert context.run_id is None

    def test_with_block_uuid(self) -> None:
        """Test context with block_uuid."""
        context = BlockExecutionContext(block_uuid="test_block")
        assert context.block_uuid == "test_block"

    def test_with_pipeline_uuid(self) -> None:
        """Test context with pipeline_uuid."""
        context = BlockExecutionContext(pipeline_uuid="test_pipeline")
        assert context.pipeline_uuid == "test_pipeline"

    def test_get_variable(self) -> None:
        """Test get_variable method."""
        context = BlockExecutionContext(variables={"key": "value"})
        assert context.get_variable("key") == "value"
        assert context.get_variable("missing", "default") == "default"

    def test_get_upstream_output(self) -> None:
        """Test get_upstream_output method."""
        context = BlockExecutionContext(upstream_outputs={"block1": "output1"})
        assert context.get_upstream_output("block1") == "output1"
        assert context.get_upstream_output("missing") is None

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        context = BlockExecutionContext(
            block_uuid="test_block",
            pipeline_uuid="test_pipeline",
        )
        data = context.to_dict()
        assert data["block_uuid"] == "test_block"
        assert data["pipeline_uuid"] == "test_pipeline"

    def test_from_mage_context(self) -> None:
        """Test creation from Mage context."""
        context = BlockExecutionContext.from_mage_context(
            block_uuid="test_block",
            pipeline_uuid="test_pipeline",
            variables={"env": "prod"},
        )
        assert context.block_uuid == "test_block"
        assert context.variables["env"] == "prod"


class TestBlockResult:
    """Tests for BlockResult."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = BlockResult(
            success=True,
            result_dict={"status": "PASSED"},
            execution_time_ms=100.0,
        )
        assert result.success is True
        assert result.execution_time_ms == 100.0

    def test_is_success_property(self) -> None:
        """Test is_success property."""
        result = BlockResult(success=True)
        assert result.is_success is True

    def test_has_error_property(self) -> None:
        """Test has_error property."""
        result_no_error = BlockResult(success=True)
        assert result_no_error.has_error is False

        result_with_error = BlockResult(success=False, error=ValueError("test"))
        assert result_with_error.has_error is True

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        result = BlockResult(
            success=True,
            result_dict={"status": "PASSED"},
            execution_time_ms=150.5,
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["execution_time_ms"] == 150.5
        assert data["error"] is None

    def test_to_dict_with_error(self) -> None:
        """Test dictionary conversion with error."""
        result = BlockResult(
            success=False,
            error=ValueError("Something went wrong"),
        )
        data = result.to_dict()
        assert data["success"] is False
        assert "Something went wrong" in data["error"]


class TestBlockEnums:
    """Tests for block enums."""

    def test_block_type_values(self) -> None:
        """Test BlockType enum values."""
        assert BlockType.TRANSFORMER.value == "transformer"
        assert BlockType.SENSOR.value == "sensor"
        assert BlockType.CONDITION.value == "condition"

    def test_execution_mode_values(self) -> None:
        """Test ExecutionMode enum values."""
        assert ExecutionMode.SYNC.value == "sync"
        assert ExecutionMode.ASYNC.value == "async"
        assert ExecutionMode.STREAMING.value == "streaming"


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_block_config(self) -> None:
        """Test DEFAULT_BLOCK_CONFIG."""
        assert DEFAULT_BLOCK_CONFIG.fail_on_error is True
        assert DEFAULT_BLOCK_CONFIG.timeout_seconds == 300

    def test_strict_block_config(self) -> None:
        """Test STRICT_BLOCK_CONFIG."""
        assert STRICT_BLOCK_CONFIG.fail_on_error is True
        assert STRICT_BLOCK_CONFIG.timeout_seconds == 120

    def test_lenient_block_config(self) -> None:
        """Test LENIENT_BLOCK_CONFIG."""
        assert LENIENT_BLOCK_CONFIG.fail_on_error is False
        assert LENIENT_BLOCK_CONFIG.timeout_seconds == 600
