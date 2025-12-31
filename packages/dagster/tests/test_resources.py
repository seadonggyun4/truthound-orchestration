"""Tests for Resources module."""

from __future__ import annotations

import pytest

from truthound_dagster.resources.base import ResourceConfig
from truthound_dagster.resources.engine import (
    DEFAULT_ENGINE_CONFIG,
    PARALLEL_ENGINE_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
    EngineResourceConfig,
    DataQualityResourceConfig,
)


class TestResourceConfig:
    """Tests for ResourceConfig base class."""

    def test_default_creation(self) -> None:
        config = ResourceConfig()
        assert config.enabled is True
        assert config.timeout_seconds == 300.0
        assert config.tags == frozenset()

    def test_with_tags(self) -> None:
        config = ResourceConfig(tags=frozenset({"production", "critical"}))
        assert "production" in config.tags
        assert "critical" in config.tags

    def test_with_enabled(self) -> None:
        config = ResourceConfig().with_enabled(False)
        assert config.enabled is False

    def test_with_timeout(self) -> None:
        config = ResourceConfig().with_timeout(600.0)
        assert config.timeout_seconds == 600.0

    def test_immutability(self) -> None:
        config = ResourceConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore


class TestEngineResourceConfig:
    """Tests for EngineResourceConfig."""

    def test_default_creation(self) -> None:
        config = EngineResourceConfig()
        assert config.engine_name == "truthound"
        assert config.parallel is False
        assert config.auto_start is True

    def test_with_engine(self) -> None:
        config = EngineResourceConfig().with_engine("great_expectations")
        assert config.engine_name == "great_expectations"

    def test_with_parallel(self) -> None:
        config = EngineResourceConfig().with_parallel(True, max_workers=8)
        assert config.parallel is True
        assert config.max_workers == 8

    def test_to_dict(self) -> None:
        config = EngineResourceConfig(
            engine_name="truthound",
            parallel=True,
            max_workers=4,
        )
        data = config.to_dict()
        assert data["engine_name"] == "truthound"
        assert data["parallel"] is True
        assert data["max_workers"] == 4


class TestDataQualityResourceConfig:
    """Tests for DataQualityResourceConfig."""

    def test_default_creation(self) -> None:
        config = DataQualityResourceConfig()
        assert config.engine_name == "truthound"
        assert config.fail_on_error is True
        assert config.warning_threshold is None

    def test_with_fail_on_error(self) -> None:
        config = DataQualityResourceConfig().with_fail_on_error(False)
        assert config.fail_on_error is False

    def test_with_warning_threshold(self) -> None:
        config = DataQualityResourceConfig().with_warning_threshold(0.05)
        assert config.warning_threshold == 0.05

    def test_with_sample_size(self) -> None:
        config = DataQualityResourceConfig().with_sample_size(1000)
        assert config.sample_size == 1000

    def test_invalid_warning_threshold(self) -> None:
        with pytest.raises(ValueError):
            DataQualityResourceConfig(warning_threshold=1.5)

    def test_invalid_sample_size(self) -> None:
        with pytest.raises(ValueError):
            DataQualityResourceConfig(sample_size=0)

    def test_to_dict(self) -> None:
        config = DataQualityResourceConfig(
            engine_name="truthound",
            fail_on_error=True,
            warning_threshold=0.05,
        )
        data = config.to_dict()
        assert data["engine_name"] == "truthound"
        assert data["fail_on_error"] is True
        assert data["warning_threshold"] == 0.05


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_config(self) -> None:
        assert DEFAULT_ENGINE_CONFIG.engine_name == "truthound"
        assert DEFAULT_ENGINE_CONFIG.parallel is False

    def test_parallel_config(self) -> None:
        assert PARALLEL_ENGINE_CONFIG.parallel is True
        assert PARALLEL_ENGINE_CONFIG.max_workers == 4

    def test_production_config(self) -> None:
        assert PRODUCTION_ENGINE_CONFIG.parallel is True
        assert PRODUCTION_ENGINE_CONFIG.max_workers == 8
        assert PRODUCTION_ENGINE_CONFIG.timeout_seconds == 600.0


class TestConfigChaining:
    """Tests for config method chaining."""

    def test_engine_config_chaining(self) -> None:
        config = (
            EngineResourceConfig()
            .with_engine("great_expectations")
            .with_parallel(True, max_workers=8)
        )
        assert config.engine_name == "great_expectations"
        assert config.parallel is True
        assert config.max_workers == 8

    def test_dq_config_chaining(self) -> None:
        config = (
            DataQualityResourceConfig()
            .with_fail_on_error(False)
            .with_warning_threshold(0.05)
            .with_sample_size(1000)
        )
        assert config.fail_on_error is False
        assert config.warning_threshold == 0.05
        assert config.sample_size == 1000


class TestConfigRoundtrip:
    """Tests for config serialization roundtrip."""

    def test_engine_resource_config_roundtrip(self) -> None:
        original = EngineResourceConfig(
            engine_name="pandera",
            parallel=True,
            max_workers=8,
            timeout_seconds=600.0,
        )
        data = original.to_dict()

        assert data["engine_name"] == original.engine_name
        assert data["parallel"] == original.parallel
        assert data["max_workers"] == original.max_workers

    def test_dq_resource_config_roundtrip(self) -> None:
        original = DataQualityResourceConfig(
            engine_name="truthound",
            fail_on_error=False,
            warning_threshold=0.05,
            sample_size=1000,
        )
        data = original.to_dict()

        assert data["fail_on_error"] == original.fail_on_error
        assert data["warning_threshold"] == original.warning_threshold
        assert data["sample_size"] == original.sample_size


class TestEngineSelection:
    """Tests for engine selection configuration."""

    def test_supported_engines(self) -> None:
        # Should not raise for supported engines
        EngineResourceConfig().with_engine("truthound")
        EngineResourceConfig().with_engine("great_expectations")
        EngineResourceConfig().with_engine("pandera")

    def test_custom_engine(self) -> None:
        # Custom engine names should work
        config = EngineResourceConfig().with_engine("custom_engine")
        assert config.engine_name == "custom_engine"
