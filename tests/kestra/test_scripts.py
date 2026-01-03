"""Tests for truthound_kestra.scripts module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.testing import MockDataQualityEngine

from truthound_kestra.scripts import (
    # Entry points
    check_quality_script,
    profile_data_script,
    learn_schema_script,
    # Executors
    CheckScriptExecutor,
    ProfileScriptExecutor,
    LearnScriptExecutor,
    # Results
    CheckScriptResult,
    ProfileScriptResult,
    LearnScriptResult,
    # Configuration
    ScriptConfig,
    CheckScriptConfig,
    ProfileScriptConfig,
    LearnScriptConfig,
    # Presets
    DEFAULT_SCRIPT_CONFIG,
    STRICT_SCRIPT_CONFIG,
    LENIENT_SCRIPT_CONFIG,
    PRODUCTION_SCRIPT_CONFIG,
    # Protocols
    DataQualityEngineProtocol,
    ScriptExecutorProtocol,
    # Utilities
    get_engine,
    create_script_config,
)
from truthound_kestra.utils.types import (
    CheckStatus,
    OperationType,
    ScriptOutput,
    Severity,
)


class TestScriptConfig:
    """Tests for script configuration classes."""

    def test_script_config_defaults(self) -> None:
        """Test ScriptConfig default values."""
        config = ScriptConfig()
        assert config.engine_name == "truthound"
        assert config.enabled is True
        assert config.timeout_seconds == 300.0
        assert config.tags == frozenset()
        assert config.description == ""

    def test_script_config_builder(self) -> None:
        """Test ScriptConfig builder pattern."""
        config = ScriptConfig()
        config = config.with_engine("great_expectations")
        config = config.with_timeout(600.0)
        config = config.with_tags("production", "users")

        assert config.engine_name == "great_expectations"
        assert config.timeout_seconds == 600.0
        assert "production" in config.tags
        assert "users" in config.tags

    def test_script_config_immutability(self) -> None:
        """Test that ScriptConfig is immutable."""
        config = ScriptConfig()
        with pytest.raises(AttributeError):
            config.engine_name = "new_engine"  # type: ignore

    def test_script_config_to_dict(self) -> None:
        """Test ScriptConfig serialization."""
        config = ScriptConfig(engine_name="truthound", timeout_seconds=120.0)
        d = config.to_dict()
        assert d["engine_name"] == "truthound"
        assert d["timeout_seconds"] == 120.0
        assert d["enabled"] is True

    def test_script_config_from_dict(self) -> None:
        """Test ScriptConfig deserialization."""
        data = {
            "engine_name": "great_expectations",
            "timeout_seconds": 600.0,
            "tags": ["test"],
        }
        config = ScriptConfig.from_dict(data)
        assert config.engine_name == "great_expectations"
        assert config.timeout_seconds == 600.0
        assert "test" in config.tags


class TestCheckScriptConfig:
    """Tests for CheckScriptConfig."""

    def test_check_script_config_defaults(self) -> None:
        """Test CheckScriptConfig default values."""
        config = CheckScriptConfig()
        assert config.engine_name == "truthound"
        assert config.fail_on_error is True
        assert config.auto_schema is False
        assert config.min_severity == Severity.LOW
        assert config.sample_failures == 100
        assert config.parallel is False
        assert config.rules == ()

    def test_check_script_config_creation(self) -> None:
        """Test CheckScriptConfig creation."""
        config = CheckScriptConfig(
            rules=({"type": "not_null", "column": "id"},),
            auto_schema=True,
        )
        assert config.rules == ({"type": "not_null", "column": "id"},)
        assert config.auto_schema is True

    def test_check_script_config_builder(self) -> None:
        """Test CheckScriptConfig builder pattern."""
        config = CheckScriptConfig()
        config = config.with_rules([{"type": "unique", "column": "email"}])
        config = config.with_auto_schema(True)
        config = config.with_fail_on_error(False)

        assert len(config.rules) == 1
        assert config.auto_schema is True
        assert config.fail_on_error is False

    def test_check_script_config_with_parallel(self) -> None:
        """Test CheckScriptConfig with parallel settings."""
        config = CheckScriptConfig()
        config = config.with_parallel(True, max_workers=4)

        assert config.parallel is True
        assert config.max_workers == 4

    def test_check_script_config_with_min_severity(self) -> None:
        """Test CheckScriptConfig with min_severity."""
        config = CheckScriptConfig()
        config = config.with_min_severity(Severity.HIGH)

        assert config.min_severity == Severity.HIGH

    def test_check_script_config_with_min_severity_string(self) -> None:
        """Test CheckScriptConfig with min_severity as string."""
        config = CheckScriptConfig()
        config = config.with_min_severity("medium")

        assert config.min_severity == Severity.MEDIUM


class TestProfileScriptConfig:
    """Tests for ProfileScriptConfig."""

    def test_profile_script_config_defaults(self) -> None:
        """Test ProfileScriptConfig default values."""
        config = ProfileScriptConfig()
        assert config.engine_name == "truthound"
        assert config.include_stats is True
        assert config.include_histograms is False
        assert config.sample_size == 0
        assert config.top_n == 10

    def test_profile_script_config_creation(self) -> None:
        """Test ProfileScriptConfig creation."""
        config = ProfileScriptConfig(
            include_stats=True,
            include_histograms=True,
            sample_size=1000,
        )
        assert config.include_stats is True
        assert config.include_histograms is True
        assert config.sample_size == 1000

    def test_profile_script_config_builder(self) -> None:
        """Test ProfileScriptConfig builder pattern."""
        config = ProfileScriptConfig()
        config = config.with_stats(False)
        config = config.with_sample_size(5000)

        assert config.include_stats is False
        assert config.sample_size == 5000


class TestLearnScriptConfig:
    """Tests for LearnScriptConfig."""

    def test_learn_script_config_defaults(self) -> None:
        """Test LearnScriptConfig default values."""
        config = LearnScriptConfig()
        assert config.engine_name == "truthound"
        assert config.min_confidence == 0.8
        assert config.include_patterns is True
        assert config.include_ranges is True
        assert config.include_categories is True
        assert config.categorical_threshold == 50
        assert config.sample_size == 0

    def test_learn_script_config_creation(self) -> None:
        """Test LearnScriptConfig creation."""
        config = LearnScriptConfig(
            min_confidence=0.9,
            include_patterns=True,
            categorical_threshold=100,
        )
        assert config.min_confidence == 0.9
        assert config.include_patterns is True
        assert config.categorical_threshold == 100

    def test_learn_script_config_builder(self) -> None:
        """Test LearnScriptConfig builder pattern."""
        config = LearnScriptConfig()
        config = config.with_min_confidence(0.95)
        config = config.with_sample_size(10000)

        assert config.min_confidence == 0.95
        assert config.sample_size == 10000


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_script_config(self) -> None:
        """Test DEFAULT_SCRIPT_CONFIG is a ScriptConfig."""
        assert isinstance(DEFAULT_SCRIPT_CONFIG, ScriptConfig)
        assert DEFAULT_SCRIPT_CONFIG.engine_name == "truthound"

    def test_strict_script_config(self) -> None:
        """Test STRICT_SCRIPT_CONFIG."""
        assert isinstance(STRICT_SCRIPT_CONFIG, CheckScriptConfig)
        assert STRICT_SCRIPT_CONFIG.fail_on_error is True
        assert STRICT_SCRIPT_CONFIG.min_severity == Severity.LOW

    def test_lenient_script_config(self) -> None:
        """Test LENIENT_SCRIPT_CONFIG."""
        assert isinstance(LENIENT_SCRIPT_CONFIG, CheckScriptConfig)
        assert LENIENT_SCRIPT_CONFIG.fail_on_error is False
        assert LENIENT_SCRIPT_CONFIG.min_severity == Severity.HIGH

    def test_production_script_config(self) -> None:
        """Test PRODUCTION_SCRIPT_CONFIG."""
        assert isinstance(PRODUCTION_SCRIPT_CONFIG, CheckScriptConfig)
        assert PRODUCTION_SCRIPT_CONFIG.timeout_seconds >= 300.0
        assert PRODUCTION_SCRIPT_CONFIG.parallel is True


class TestCreateScriptConfig:
    """Tests for create_script_config factory function."""

    def test_create_check_config(self) -> None:
        """Test creating check configuration."""
        config = create_script_config("check", fail_on_error=False)
        assert isinstance(config, CheckScriptConfig)
        assert config.fail_on_error is False

    def test_create_profile_config(self) -> None:
        """Test creating profile configuration."""
        config = create_script_config("profile", sample_size=1000)
        assert isinstance(config, ProfileScriptConfig)
        assert config.sample_size == 1000

    def test_create_learn_config(self) -> None:
        """Test creating learn configuration."""
        config = create_script_config("learn", min_confidence=0.9)
        assert isinstance(config, LearnScriptConfig)
        assert config.min_confidence == 0.9

    def test_create_config_with_operation_type(self) -> None:
        """Test creating config with OperationType enum."""
        config = create_script_config(OperationType.CHECK, auto_schema=True)
        assert isinstance(config, CheckScriptConfig)
        assert config.auto_schema is True


class TestCheckScriptExecutor:
    """Tests for CheckScriptExecutor."""

    def test_executor_creation(self, mock_engine: MockDataQualityEngine) -> None:
        """Test CheckScriptExecutor creation."""
        config = CheckScriptConfig()
        executor = CheckScriptExecutor(config, engine=mock_engine)

        assert executor.config == config
        assert executor.engine == mock_engine

    def test_executor_execute_success(
        self,
        mock_engine: MockDataQualityEngine,
        sample_data: dict[str, list[Any]],
    ) -> None:
        """Test CheckScriptExecutor execution with success."""
        config = CheckScriptConfig(
            rules=({"type": "not_null", "column": "id"},),
            auto_schema=True,
        )
        executor = CheckScriptExecutor(config, engine=mock_engine)

        result = executor.execute(sample_data)

        assert isinstance(result, CheckScriptResult)
        assert result.is_success is True

    def test_executor_execute_with_rules(
        self,
        mock_engine: MockDataQualityEngine,
        sample_data: dict[str, list[Any]],
        sample_rules: tuple[dict[str, Any], ...],
    ) -> None:
        """Test CheckScriptExecutor execution with rules."""
        config = CheckScriptConfig(rules=sample_rules)
        executor = CheckScriptExecutor(config, engine=mock_engine)

        result = executor.execute(sample_data)

        assert result.is_success is True

    def test_executor_execute_failure(
        self,
        failing_mock_engine: MockDataQualityEngine,
        sample_data: dict[str, list[Any]],
    ) -> None:
        """Test CheckScriptExecutor execution with failure."""
        config = CheckScriptConfig(auto_schema=True, fail_on_error=False)
        executor = CheckScriptExecutor(config, engine=failing_mock_engine)

        result = executor.execute(sample_data)

        # Either success is False or status is not PASSED
        assert result.is_success is False or result.status != CheckStatus.PASSED


class TestProfileScriptExecutor:
    """Tests for ProfileScriptExecutor."""

    def test_executor_creation(self, mock_engine: MockDataQualityEngine) -> None:
        """Test ProfileScriptExecutor creation."""
        config = ProfileScriptConfig()
        executor = ProfileScriptExecutor(config, engine=mock_engine)

        assert executor.config == config
        assert executor.engine == mock_engine

    def test_executor_execute(
        self,
        mock_engine: MockDataQualityEngine,
        sample_data: dict[str, list[Any]],
    ) -> None:
        """Test ProfileScriptExecutor execution."""
        config = ProfileScriptConfig()
        executor = ProfileScriptExecutor(config, engine=mock_engine)

        result = executor.execute(sample_data)

        assert isinstance(result, ProfileScriptResult)
        assert result.is_success is True


class TestLearnScriptExecutor:
    """Tests for LearnScriptExecutor."""

    def test_executor_creation(self, mock_engine: MockDataQualityEngine) -> None:
        """Test LearnScriptExecutor creation."""
        config = LearnScriptConfig()
        executor = LearnScriptExecutor(config, engine=mock_engine)

        assert executor.config == config
        assert executor.engine == mock_engine

    def test_executor_execute(
        self,
        mock_engine: MockDataQualityEngine,
        sample_data: dict[str, list[Any]],
    ) -> None:
        """Test LearnScriptExecutor execution."""
        config = LearnScriptConfig()
        executor = LearnScriptExecutor(config, engine=mock_engine)

        result = executor.execute(sample_data)

        assert isinstance(result, LearnScriptResult)
        assert result.is_success is True


class TestScriptResults:
    """Tests for script result classes."""

    def test_check_script_result_creation(self) -> None:
        """Test CheckScriptResult creation."""
        output = ScriptOutput(
            status=CheckStatus.PASSED,
            operation=OperationType.CHECK,
            passed_count=10,
            failed_count=0,
            execution_time_ms=150.0,
        )
        result = CheckScriptResult(output=output)

        assert result.is_success is True
        assert result.passed_count == 10
        assert result.failed_count == 0
        assert result.execution_time_ms == 150.0

    def test_check_script_result_to_dict(self) -> None:
        """Test CheckScriptResult serialization."""
        output = ScriptOutput(
            status=CheckStatus.PASSED,
            operation=OperationType.CHECK,
            passed_count=10,
            failed_count=0,
            execution_time_ms=150.0,
        )
        result = CheckScriptResult(output=output)

        d = result.to_dict()
        assert "output" in d
        assert d["output"]["status"] == "passed"

    def test_profile_script_result_creation(self) -> None:
        """Test ProfileScriptResult creation."""
        output = ScriptOutput(
            status=CheckStatus.PASSED,
            operation=OperationType.PROFILE,
            execution_time_ms=200.0,
        )
        result = ProfileScriptResult(output=output)

        assert result.is_success is True
        assert result.execution_time_ms == 200.0

    def test_learn_script_result_creation(self) -> None:
        """Test LearnScriptResult creation."""
        output = ScriptOutput(
            status=CheckStatus.PASSED,
            operation=OperationType.LEARN,
            execution_time_ms=180.0,
        )
        result = LearnScriptResult(output=output)

        assert result.is_success is True
        assert result.execution_time_ms == 180.0


class TestProtocols:
    """Tests for protocol compliance."""

    def test_mock_engine_protocol_compliance(
        self,
        mock_engine: MockDataQualityEngine,
    ) -> None:
        """Test that MockDataQualityEngine follows DataQualityEngineProtocol."""
        assert hasattr(mock_engine, "check")
        assert hasattr(mock_engine, "profile")
        assert hasattr(mock_engine, "learn")
        assert callable(mock_engine.check)
        assert callable(mock_engine.profile)
        assert callable(mock_engine.learn)


class TestGetEngine:
    """Tests for get_engine function."""

    def test_get_engine_truthound(self) -> None:
        """Test get_engine with truthound."""
        # This may raise if truthound is not installed
        try:
            engine = get_engine("truthound")
            assert engine is not None
        except ImportError:
            pytest.skip("truthound not installed")

    def test_get_engine_invalid(self) -> None:
        """Test get_engine with invalid engine name."""
        from truthound_kestra.utils.exceptions import ConfigurationError

        with pytest.raises((ValueError, ImportError, ConfigurationError)):
            get_engine("nonexistent_engine")
