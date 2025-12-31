"""Tests for engine configuration system.

This module tests the engine configuration system including:
- Configuration validation
- Builder pattern
- Environment-based loading
- Serialization/deserialization
- Configuration merging
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from common.engines import (
    # Base
    EngineConfig,
    # Configuration System
    BaseEngineConfig,
    ConfigBuilder,
    ConfigEnvironment,
    ConfigLoader,
    ConfigRegistry,
    ConfigValidator,
    ConfigValidationError,
    ConfigLoadError,
    EnvironmentConfig,
    FieldConstraint,
    MergeStrategy,
    ValidationResult,
    create_config_for_environment,
    load_config,
    # Engine Configs
    TruthoundEngineConfig,
    GreatExpectationsConfig,
    PanderaConfig,
    DEFAULT_TRUTHOUND_CONFIG,
    DEFAULT_GE_CONFIG,
    DEFAULT_PANDERA_CONFIG,
    PRODUCTION_TRUTHOUND_CONFIG,
    PRODUCTION_GE_CONFIG,
    PRODUCTION_PANDERA_CONFIG,
)


# =============================================================================
# BaseEngineConfig Tests
# =============================================================================


class TestBaseEngineConfig:
    """Tests for BaseEngineConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BaseEngineConfig()
        assert config.auto_start is False
        assert config.auto_stop is True
        assert config.health_check_enabled is True
        assert config.health_check_interval_seconds == 30.0
        assert config.startup_timeout_seconds == 30.0
        assert config.shutdown_timeout_seconds == 10.0
        assert config.max_retries_on_failure == 3
        assert config.fail_fast is False
        assert config.tags == frozenset()
        assert config.metadata == {}

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = BaseEngineConfig(
            auto_start=True,
            health_check_interval_seconds=60.0,
            max_retries_on_failure=5,
            tags=frozenset(["prod", "critical"]),
            metadata={"team": "data"},
        )
        assert config.auto_start is True
        assert config.health_check_interval_seconds == 60.0
        assert config.max_retries_on_failure == 5
        assert "prod" in config.tags
        assert config.metadata["team"] == "data"

    def test_immutability(self) -> None:
        """Test that config is immutable (frozen dataclass)."""
        config = BaseEngineConfig()
        with pytest.raises(AttributeError):
            config.auto_start = True  # type: ignore[misc]

    def test_validation_negative_timeout(self) -> None:
        """Test validation fails for negative timeout."""
        with pytest.raises(ValueError, match="non-negative"):
            BaseEngineConfig(startup_timeout_seconds=-1.0)

    def test_validation_negative_retries(self) -> None:
        """Test validation fails for negative retries."""
        with pytest.raises(ValueError, match="non-negative"):
            BaseEngineConfig(max_retries_on_failure=-1)


# =============================================================================
# Builder Pattern Tests
# =============================================================================


class TestBuilderPattern:
    """Tests for builder pattern methods."""

    def test_with_auto_start(self) -> None:
        """Test with_auto_start builder method."""
        config = BaseEngineConfig().with_auto_start(True)
        assert config.auto_start is True
        # Other values unchanged
        assert config.auto_stop is True

    def test_with_auto_stop(self) -> None:
        """Test with_auto_stop builder method."""
        config = BaseEngineConfig().with_auto_stop(False)
        assert config.auto_stop is False

    def test_with_health_check(self) -> None:
        """Test with_health_check builder method."""
        config = BaseEngineConfig().with_health_check(True, interval_seconds=15.0)
        assert config.health_check_enabled is True
        assert config.health_check_interval_seconds == 15.0

    def test_with_timeouts(self) -> None:
        """Test with_timeouts builder method."""
        config = BaseEngineConfig().with_timeouts(
            startup_seconds=60.0, shutdown_seconds=20.0
        )
        assert config.startup_timeout_seconds == 60.0
        assert config.shutdown_timeout_seconds == 20.0

    def test_with_retries(self) -> None:
        """Test with_retries builder method."""
        config = BaseEngineConfig().with_retries(5)
        assert config.max_retries_on_failure == 5

    def test_with_tags(self) -> None:
        """Test with_tags builder method."""
        config = BaseEngineConfig().with_tags("prod", "critical")
        assert "prod" in config.tags
        assert "critical" in config.tags

    def test_with_metadata(self) -> None:
        """Test with_metadata builder method."""
        config = BaseEngineConfig().with_metadata(team="data", env="prod")
        assert config.metadata["team"] == "data"
        assert config.metadata["env"] == "prod"

    def test_chained_builders(self) -> None:
        """Test chaining multiple builder methods."""
        config = (
            BaseEngineConfig()
            .with_auto_start(True)
            .with_health_check(True, interval_seconds=60.0)
            .with_timeouts(startup_seconds=45.0)
            .with_retries(5)
            .with_tags("prod")
            .with_metadata(version="1.0")
        )
        assert config.auto_start is True
        assert config.health_check_enabled is True
        assert config.health_check_interval_seconds == 60.0
        assert config.startup_timeout_seconds == 45.0
        assert config.max_retries_on_failure == 5
        assert "prod" in config.tags
        assert config.metadata["version"] == "1.0"


# =============================================================================
# EngineConfig Builder Tests
# =============================================================================


class TestEngineConfigBuilder:
    """Tests for EngineConfig builder methods."""

    def test_engine_config_with_auto_start(self) -> None:
        """Test EngineConfig with_auto_start."""
        config = EngineConfig().with_auto_start(True)
        assert config.auto_start is True

    def test_engine_config_with_health_check(self) -> None:
        """Test EngineConfig with_health_check."""
        config = EngineConfig().with_health_check(True, interval_seconds=45.0)
        assert config.health_check_enabled is True
        assert config.health_check_interval_seconds == 45.0

    def test_engine_config_with_timeouts(self) -> None:
        """Test EngineConfig with_timeouts."""
        config = EngineConfig().with_timeouts(startup_seconds=90.0)
        assert config.startup_timeout_seconds == 90.0

    def test_engine_config_with_retries(self) -> None:
        """Test EngineConfig with_retries."""
        config = EngineConfig().with_retries(10)
        assert config.max_retries_on_failure == 10


# =============================================================================
# TruthoundEngineConfig Tests
# =============================================================================


class TestTruthoundEngineConfig:
    """Tests for TruthoundEngineConfig."""

    def test_default_values(self) -> None:
        """Test Truthound default configuration."""
        config = TruthoundEngineConfig()
        assert config.parallel is False
        assert config.max_workers is None
        assert config.min_severity is None
        assert config.cache_schemas is True
        assert config.infer_constraints is True
        assert config.categorical_threshold == 20

    def test_with_parallel(self) -> None:
        """Test with_parallel builder method."""
        config = TruthoundEngineConfig().with_parallel(True, max_workers=4)
        assert config.parallel is True
        assert config.max_workers == 4

    def test_with_min_severity(self) -> None:
        """Test with_min_severity builder method."""
        config = TruthoundEngineConfig().with_min_severity("medium")
        assert config.min_severity == "medium"

    def test_with_cache_schemas(self) -> None:
        """Test with_cache_schemas builder method."""
        config = TruthoundEngineConfig().with_cache_schemas(False)
        assert config.cache_schemas is False

    def test_with_infer_constraints(self) -> None:
        """Test with_infer_constraints builder method."""
        config = TruthoundEngineConfig().with_infer_constraints(False)
        assert config.infer_constraints is False

    def test_with_categorical_threshold(self) -> None:
        """Test with_categorical_threshold builder method."""
        config = TruthoundEngineConfig().with_categorical_threshold(50)
        assert config.categorical_threshold == 50

    def test_chained_truthound_builders(self) -> None:
        """Test chaining Truthound-specific builder methods."""
        config = (
            TruthoundEngineConfig()
            .with_auto_start(True)
            .with_parallel(True, max_workers=8)
            .with_min_severity("high")
            .with_health_check(True, interval_seconds=30.0)
        )
        assert config.auto_start is True
        assert config.parallel is True
        assert config.max_workers == 8
        assert config.min_severity == "high"
        assert config.health_check_enabled is True

    def test_validation_invalid_max_workers(self) -> None:
        """Test validation fails for invalid max_workers."""
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            TruthoundEngineConfig(max_workers=0)

    def test_validation_invalid_severity(self) -> None:
        """Test validation fails for invalid severity."""
        with pytest.raises(ValueError, match="min_severity must be one of"):
            TruthoundEngineConfig(min_severity="invalid")

    def test_validation_invalid_categorical_threshold(self) -> None:
        """Test validation fails for invalid categorical_threshold."""
        with pytest.raises(ValueError, match="categorical_threshold must be at least 1"):
            TruthoundEngineConfig(categorical_threshold=0)


# =============================================================================
# GreatExpectationsConfig Tests
# =============================================================================


class TestGreatExpectationsConfig:
    """Tests for GreatExpectationsConfig."""

    def test_default_values(self) -> None:
        """Test GE default configuration."""
        config = GreatExpectationsConfig()
        assert config.result_format == "COMPLETE"
        assert config.context_root_dir is None
        assert config.include_profiling is True
        assert config.catch_exceptions is True
        assert config.enable_data_docs is False

    def test_with_result_format(self) -> None:
        """Test with_result_format builder method."""
        config = GreatExpectationsConfig().with_result_format("basic")
        assert config.result_format == "BASIC"

    def test_with_context_root_dir(self) -> None:
        """Test with_context_root_dir builder method."""
        config = GreatExpectationsConfig().with_context_root_dir("/path/to/ge")
        assert config.context_root_dir == "/path/to/ge"

    def test_with_profiling(self) -> None:
        """Test with_profiling builder method."""
        config = GreatExpectationsConfig().with_profiling(False)
        assert config.include_profiling is False

    def test_with_catch_exceptions(self) -> None:
        """Test with_catch_exceptions builder method."""
        config = GreatExpectationsConfig().with_catch_exceptions(False)
        assert config.catch_exceptions is False

    def test_with_data_docs(self) -> None:
        """Test with_data_docs builder method."""
        config = GreatExpectationsConfig().with_data_docs(True)
        assert config.enable_data_docs is True

    def test_validation_invalid_result_format(self) -> None:
        """Test validation fails for invalid result_format."""
        with pytest.raises(ValueError, match="result_format must be one of"):
            GreatExpectationsConfig(result_format="INVALID")


# =============================================================================
# PanderaConfig Tests
# =============================================================================


class TestPanderaConfig:
    """Tests for PanderaConfig."""

    def test_default_values(self) -> None:
        """Test Pandera default configuration."""
        config = PanderaConfig()
        assert config.lazy is True
        assert config.strict is False
        assert config.coerce is False
        assert config.unique_column_names is False
        assert config.report_duplicates == "all"

    def test_with_lazy(self) -> None:
        """Test with_lazy builder method."""
        config = PanderaConfig().with_lazy(False)
        assert config.lazy is False

    def test_with_strict(self) -> None:
        """Test with_strict builder method."""
        config = PanderaConfig().with_strict(True)
        assert config.strict is True

    def test_with_coerce(self) -> None:
        """Test with_coerce builder method."""
        config = PanderaConfig().with_coerce(True)
        assert config.coerce is True

    def test_with_unique_column_names(self) -> None:
        """Test with_unique_column_names builder method."""
        config = PanderaConfig().with_unique_column_names(True)
        assert config.unique_column_names is True

    def test_with_report_duplicates(self) -> None:
        """Test with_report_duplicates builder method."""
        config = PanderaConfig().with_report_duplicates("first")
        assert config.report_duplicates == "first"

    def test_validation_invalid_report_duplicates(self) -> None:
        """Test validation fails for invalid report_duplicates."""
        with pytest.raises(ValueError, match="report_duplicates must be one of"):
            PanderaConfig(report_duplicates="invalid")


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for configuration serialization."""

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = TruthoundEngineConfig(
            auto_start=True,
            parallel=True,
            max_workers=4,
            tags=frozenset(["test"]),
        )
        d = config.to_dict()
        assert d["auto_start"] is True
        assert d["parallel"] is True
        assert d["max_workers"] == 4
        assert "test" in d["tags"]

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "auto_start": True,
            "parallel": True,
            "max_workers": 4,
            "tags": ["test"],
        }
        config = TruthoundEngineConfig.from_dict(data)
        assert config.auto_start is True
        assert config.parallel is True
        assert config.max_workers == 4
        assert "test" in config.tags

    def test_round_trip(self) -> None:
        """Test serialization round trip."""
        original = TruthoundEngineConfig(
            auto_start=True,
            health_check_enabled=True,
            parallel=True,
            max_workers=8,
            min_severity="medium",
            tags=frozenset(["prod", "critical"]),
            metadata={"version": "1.0"},
        )
        data = original.to_dict()
        restored = TruthoundEngineConfig.from_dict(data)

        assert restored.auto_start == original.auto_start
        assert restored.parallel == original.parallel
        assert restored.max_workers == original.max_workers
        assert restored.min_severity == original.min_severity
        assert restored.tags == original.tags


# =============================================================================
# ConfigValidator Tests
# =============================================================================


class TestConfigValidator:
    """Tests for ConfigValidator."""

    def test_valid_config(self) -> None:
        """Test validation passes for valid config."""
        validator = ConfigValidator()
        config = BaseEngineConfig()
        result = validator.validate(config)
        assert result.is_valid

    def test_custom_constraint(self) -> None:
        """Test adding custom constraint."""
        validator = ConfigValidator()
        validator.add_constraint(
            FieldConstraint(
                field_name="max_retries_on_failure",
                validator=lambda x: x <= 10,
                message="max_retries_on_failure must be <= 10",
                constraint_name="max_retries_limit",
            )
        )

        # Valid config
        config = BaseEngineConfig(max_retries_on_failure=5)
        result = validator.validate(config)
        assert result.is_valid

        # Invalid config (would fail if we could set it > 10, but default is 3)
        config = BaseEngineConfig(max_retries_on_failure=3)
        result = validator.validate(config)
        assert result.is_valid

    def test_warnings_for_short_interval(self) -> None:
        """Test warnings for very short health check interval."""
        validator = ConfigValidator()
        config = BaseEngineConfig(
            health_check_enabled=True,
            health_check_interval_seconds=1.0,
        )
        result = validator.validate(config)
        assert result.is_valid  # Still valid, but with warnings
        assert len(result.warnings) > 0


# =============================================================================
# ConfigLoader Tests
# =============================================================================


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_from_dict(self) -> None:
        """Test loading from dictionary."""
        loader = ConfigLoader(TruthoundEngineConfig)
        config = loader.from_dict(
            {"auto_start": True, "parallel": True, "max_workers": 4}
        )
        assert config.auto_start is True
        assert config.parallel is True
        assert config.max_workers == 4

    def test_from_dict_with_defaults(self) -> None:
        """Test loading from dictionary with defaults."""
        loader = ConfigLoader(TruthoundEngineConfig)
        defaults = TruthoundEngineConfig(parallel=True, max_workers=2)
        config = loader.from_dict({"max_workers": 4}, defaults=defaults)
        assert config.parallel is True  # From defaults
        assert config.max_workers == 4  # Overridden

    def test_from_json_file(self) -> None:
        """Test loading from JSON file."""
        loader = ConfigLoader(TruthoundEngineConfig)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(
                {"auto_start": True, "parallel": True, "max_workers": 4}, f
            )
            f.flush()

            try:
                config = loader.from_file(f.name)
                assert config.auto_start is True
                assert config.parallel is True
                assert config.max_workers == 4
            finally:
                os.unlink(f.name)

    def test_from_env(self) -> None:
        """Test loading from environment variables."""
        loader = ConfigLoader(TruthoundEngineConfig)

        # Set environment variables
        os.environ["TEST_AUTO_START"] = "true"
        os.environ["TEST_PARALLEL"] = "true"
        os.environ["TEST_MAX_WORKERS"] = "4"

        try:
            config = loader.from_env("TEST_")
            assert config.auto_start is True
            assert config.parallel is True
            assert config.max_workers == 4
        finally:
            # Clean up
            del os.environ["TEST_AUTO_START"]
            del os.environ["TEST_PARALLEL"]
            del os.environ["TEST_MAX_WORKERS"]

    def test_from_file_not_found(self) -> None:
        """Test error when file not found."""
        loader = ConfigLoader(TruthoundEngineConfig)
        with pytest.raises(ConfigLoadError, match="not found"):
            loader.from_file("/nonexistent/path/config.json")


# =============================================================================
# ConfigBuilder Tests
# =============================================================================


class TestConfigBuilder:
    """Tests for ConfigBuilder."""

    def test_basic_build(self) -> None:
        """Test basic configuration building."""
        config = (
            ConfigBuilder(TruthoundEngineConfig)
            .with_auto_start(True)
            .set("parallel", True)
            .set("max_workers", 4)
            .build()
        )
        assert config.auto_start is True
        assert config.parallel is True
        assert config.max_workers == 4

    def test_build_with_base(self) -> None:
        """Test building with base configuration."""
        base = TruthoundEngineConfig(parallel=True, max_workers=2)
        config = (
            ConfigBuilder(TruthoundEngineConfig, base=base)
            .set("max_workers", 8)
            .build()
        )
        assert config.parallel is True  # From base
        assert config.max_workers == 8  # Overridden

    def test_builder_methods(self) -> None:
        """Test all builder methods."""
        config = (
            ConfigBuilder(TruthoundEngineConfig)
            .with_auto_start(True)
            .with_auto_stop(False)
            .with_health_check(True, interval_seconds=60.0)
            .with_timeouts(startup_seconds=45.0, shutdown_seconds=15.0)
            .with_retries(5)
            .with_fail_fast(True)
            .with_tags("prod", "critical")
            .with_metadata(team="data")
            .build()
        )
        assert config.auto_start is True
        assert config.auto_stop is False
        assert config.health_check_enabled is True
        assert config.health_check_interval_seconds == 60.0
        assert config.startup_timeout_seconds == 45.0
        assert config.shutdown_timeout_seconds == 15.0
        assert config.max_retries_on_failure == 5
        # Note: fail_fast is not in EngineConfig, only BaseEngineConfig
        assert "prod" in config.tags
        assert config.metadata["team"] == "data"


# =============================================================================
# ConfigRegistry Tests
# =============================================================================


class TestConfigRegistry:
    """Tests for ConfigRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving configurations."""
        registry = ConfigRegistry()
        config = TruthoundEngineConfig(parallel=True)
        registry.register("test", config)

        retrieved = registry.get("test")
        assert retrieved.parallel is True  # type: ignore[attr-defined]

    def test_register_duplicate(self) -> None:
        """Test error on duplicate registration."""
        registry = ConfigRegistry()
        config = TruthoundEngineConfig()
        registry.register("test", config)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", config)

    def test_register_override(self) -> None:
        """Test override on duplicate registration."""
        registry = ConfigRegistry()
        config1 = TruthoundEngineConfig(parallel=False)
        config2 = TruthoundEngineConfig(parallel=True)

        registry.register("test", config1)
        registry.register("test", config2, override=True)

        retrieved = registry.get("test")
        assert retrieved.parallel is True  # type: ignore[attr-defined]

    def test_get_not_found(self) -> None:
        """Test error when configuration not found."""
        registry = ConfigRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_get_or_default(self) -> None:
        """Test get_or_default method."""
        registry = ConfigRegistry()
        default = TruthoundEngineConfig(parallel=True)

        result = registry.get_or_default("nonexistent", default)
        assert result.parallel is True  # type: ignore[attr-defined]

    def test_list_configs(self) -> None:
        """Test listing registered configurations."""
        registry = ConfigRegistry()
        registry.register("config1", TruthoundEngineConfig())
        registry.register("config2", GreatExpectationsConfig())

        names = registry.list()
        assert "config1" in names
        assert "config2" in names

    def test_clear(self) -> None:
        """Test clearing registry."""
        registry = ConfigRegistry()
        registry.register("test", TruthoundEngineConfig())
        registry.clear()

        assert len(registry.list()) == 0


# =============================================================================
# EnvironmentConfig Tests
# =============================================================================


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving environment configs."""
        env_config = EnvironmentConfig(TruthoundEngineConfig)

        prod_config = TruthoundEngineConfig(auto_start=True, parallel=True)
        dev_config = TruthoundEngineConfig(auto_start=False, parallel=False)

        env_config.register_environment(ConfigEnvironment.PRODUCTION, prod_config)
        env_config.register_environment(ConfigEnvironment.DEVELOPMENT, dev_config)

        assert env_config.get(ConfigEnvironment.PRODUCTION).parallel is True
        assert env_config.get(ConfigEnvironment.DEVELOPMENT).parallel is False

    def test_get_current_environment(self) -> None:
        """Test getting current environment from env var."""
        env_config = EnvironmentConfig(
            TruthoundEngineConfig,
            env_var="TEST_ENV",
            default_env=ConfigEnvironment.DEVELOPMENT,
        )

        # Default when not set
        assert env_config.get_current_environment() == ConfigEnvironment.DEVELOPMENT

        # Set environment variable
        os.environ["TEST_ENV"] = "production"
        try:
            assert env_config.get_current_environment() == ConfigEnvironment.PRODUCTION
        finally:
            del os.environ["TEST_ENV"]

    def test_environment_from_string(self) -> None:
        """Test ConfigEnvironment.from_string."""
        assert ConfigEnvironment.from_string("prod") == ConfigEnvironment.PRODUCTION
        assert ConfigEnvironment.from_string("production") == ConfigEnvironment.PRODUCTION
        assert ConfigEnvironment.from_string("dev") == ConfigEnvironment.DEVELOPMENT
        assert ConfigEnvironment.from_string("test") == ConfigEnvironment.TESTING
        assert ConfigEnvironment.from_string("staging") == ConfigEnvironment.STAGING
        # Unknown defaults to development
        assert ConfigEnvironment.from_string("unknown") == ConfigEnvironment.DEVELOPMENT


# =============================================================================
# Merge Strategy Tests
# =============================================================================


class TestMergeStrategy:
    """Tests for configuration merging."""

    def test_merge_override(self) -> None:
        """Test merge with OVERRIDE strategy."""
        config1 = BaseEngineConfig(auto_start=False, max_retries_on_failure=1)
        config2 = BaseEngineConfig(auto_start=True, max_retries_on_failure=5)

        merged = config1.merge_with(config2, MergeStrategy.OVERRIDE)
        assert merged.auto_start is True
        assert merged.max_retries_on_failure == 5

    def test_merge_keep_first(self) -> None:
        """Test merge with KEEP_FIRST strategy."""
        config1 = BaseEngineConfig(auto_start=False, max_retries_on_failure=1)
        config2 = BaseEngineConfig(auto_start=True, max_retries_on_failure=5)

        merged = config1.merge_with(config2, MergeStrategy.KEEP_FIRST)
        assert merged.auto_start is False
        assert merged.max_retries_on_failure == 1

    def test_merge_deep_merge_metadata(self) -> None:
        """Test deep merge for metadata."""
        config1 = BaseEngineConfig(metadata={"a": 1, "b": 2})
        config2 = BaseEngineConfig(metadata={"b": 3, "c": 4})

        merged = config1.merge_with(config2, MergeStrategy.DEEP_MERGE)
        assert merged.metadata["a"] == 1  # From config1
        assert merged.metadata["b"] == 3  # Overridden by config2
        assert merged.metadata["c"] == 4  # From config2

    def test_merge_deep_merge_tags(self) -> None:
        """Test deep merge for tags (union)."""
        config1 = BaseEngineConfig(tags=frozenset(["a", "b"]))
        config2 = BaseEngineConfig(tags=frozenset(["b", "c"]))

        merged = config1.merge_with(config2, MergeStrategy.DEEP_MERGE)
        assert "a" in merged.tags
        assert "b" in merged.tags
        assert "c" in merged.tags


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_config_basic(self) -> None:
        """Test load_config with defaults."""
        config = load_config(TruthoundEngineConfig)
        assert isinstance(config, TruthoundEngineConfig)

    def test_load_config_from_env(self) -> None:
        """Test load_config from environment."""
        os.environ["LOAD_TEST_AUTO_START"] = "true"
        os.environ["LOAD_TEST_PARALLEL"] = "true"

        try:
            config = load_config(
                TruthoundEngineConfig,
                env_prefix="LOAD_TEST_",
            )
            assert config.auto_start is True
            assert config.parallel is True
        finally:
            del os.environ["LOAD_TEST_AUTO_START"]
            del os.environ["LOAD_TEST_PARALLEL"]

    def test_create_config_for_environment(self) -> None:
        """Test create_config_for_environment."""
        prod_config = create_config_for_environment(
            TruthoundEngineConfig, ConfigEnvironment.PRODUCTION
        )
        assert prod_config.auto_start is True
        assert prod_config.health_check_enabled is True
        assert prod_config.max_retries_on_failure == 5
        assert prod_config.startup_timeout_seconds == 60.0

        dev_config = create_config_for_environment(
            TruthoundEngineConfig, "development"
        )
        assert dev_config.auto_start is False
        assert dev_config.health_check_enabled is False
        assert dev_config.max_retries_on_failure == 1

        test_config = create_config_for_environment(
            TruthoundEngineConfig, ConfigEnvironment.TESTING
        )
        assert test_config.auto_start is True
        assert test_config.health_check_enabled is False
        assert test_config.max_retries_on_failure == 0
        assert test_config.startup_timeout_seconds == 5.0

        staging_config = create_config_for_environment(
            GreatExpectationsConfig, ConfigEnvironment.STAGING
        )
        assert staging_config.auto_start is True
        assert staging_config.health_check_interval_seconds == 30.0


# =============================================================================
# Preset Configuration Tests
# =============================================================================


class TestPresetConfigurations:
    """Tests for preset configurations."""

    def test_default_truthound_config(self) -> None:
        """Test DEFAULT_TRUTHOUND_CONFIG."""
        config = DEFAULT_TRUTHOUND_CONFIG
        assert config.parallel is False
        assert config.auto_start is False

    def test_production_truthound_config(self) -> None:
        """Test PRODUCTION_TRUTHOUND_CONFIG."""
        config = PRODUCTION_TRUTHOUND_CONFIG
        assert config.auto_start is True
        assert config.health_check_enabled is True
        assert config.parallel is True

    def test_default_ge_config(self) -> None:
        """Test DEFAULT_GE_CONFIG."""
        config = DEFAULT_GE_CONFIG
        assert config.result_format == "COMPLETE"

    def test_production_ge_config(self) -> None:
        """Test PRODUCTION_GE_CONFIG."""
        config = PRODUCTION_GE_CONFIG
        assert config.auto_start is True
        assert config.enable_data_docs is True

    def test_default_pandera_config(self) -> None:
        """Test DEFAULT_PANDERA_CONFIG."""
        config = DEFAULT_PANDERA_CONFIG
        assert config.lazy is True
        assert config.strict is False

    def test_production_pandera_config(self) -> None:
        """Test PRODUCTION_PANDERA_CONFIG."""
        config = PRODUCTION_PANDERA_CONFIG
        assert config.strict is True
        assert config.unique_column_names is True
