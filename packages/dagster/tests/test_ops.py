"""Tests for Ops module."""

from typing import Any

import pytest

from truthound_dagster.ops.base import (
    LENIENT_CHECK_CONFIG,
    STRICT_CHECK_CONFIG,
    CheckOpConfig,
    LearnOpConfig,
    ProfileOpConfig,
)


class TestCheckOpConfig:
    """Tests for CheckOpConfig."""

    def test_default_creation(self) -> None:
        config = CheckOpConfig()
        assert config.fail_on_error is True
        assert config.auto_schema is False
        assert config.timeout_seconds == 300.0

    def test_with_rules(self) -> None:
        rules = (
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
        )
        config = CheckOpConfig(rules=rules)
        assert len(config.rules) == 2

    def test_with_fail_on_error(self) -> None:
        config = CheckOpConfig().with_fail_on_error(False)
        assert config.fail_on_error is False

    def test_with_warning_threshold(self) -> None:
        config = CheckOpConfig().with_warning_threshold(0.05)
        assert config.warning_threshold == 0.05

    def test_with_auto_schema(self) -> None:
        config = CheckOpConfig().with_auto_schema(True)
        assert config.auto_schema is True

    def test_with_timeout(self) -> None:
        config = CheckOpConfig().with_timeout(600.0)
        assert config.timeout_seconds == 600.0

    def test_to_dict(self) -> None:
        config = CheckOpConfig(
            rules=({"type": "not_null", "column": "id"},),
            fail_on_error=True,
        )
        data = config.to_dict()
        assert "rules" in data
        assert data["fail_on_error"] is True

    def test_immutability(self) -> None:
        config = CheckOpConfig()
        with pytest.raises(AttributeError):
            config.fail_on_error = False  # type: ignore


class TestProfileOpConfig:
    """Tests for ProfileOpConfig."""

    def test_default_creation(self) -> None:
        config = ProfileOpConfig()
        assert config.include_histograms is True
        assert config.include_samples is True
        assert config.sample_size == 10

    def test_with_histograms(self) -> None:
        config = ProfileOpConfig().with_histograms(include=False)
        assert config.include_histograms is False

    def test_with_samples(self) -> None:
        config = ProfileOpConfig().with_samples(include=True, sample_size=20)
        assert config.include_samples is True
        assert config.sample_size == 20

    def test_to_dict(self) -> None:
        config = ProfileOpConfig(sample_size=1000)
        data = config.to_dict()
        assert data["sample_size"] == 1000
        assert data["include_histograms"] is True


class TestLearnOpConfig:
    """Tests for LearnOpConfig."""

    def test_default_creation(self) -> None:
        config = LearnOpConfig()
        assert config.min_confidence == 0.8
        assert config.infer_constraints is True
        assert config.categorical_threshold == 20

    def test_with_min_confidence(self) -> None:
        config = LearnOpConfig().with_min_confidence(0.95)
        assert config.min_confidence == 0.95

    def test_with_categorical_threshold(self) -> None:
        config = LearnOpConfig().with_categorical_threshold(50)
        assert config.categorical_threshold == 50

    def test_to_dict(self) -> None:
        config = LearnOpConfig(min_confidence=0.95)
        data = config.to_dict()
        assert data["min_confidence"] == 0.95


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_strict_check_config(self) -> None:
        assert STRICT_CHECK_CONFIG.fail_on_error is True
        assert STRICT_CHECK_CONFIG.warning_threshold is None

    def test_lenient_check_config(self) -> None:
        assert LENIENT_CHECK_CONFIG.fail_on_error is False
        assert LENIENT_CHECK_CONFIG.warning_threshold == 0.10


class TestConfigChaining:
    """Tests for config method chaining."""

    def test_check_config_chaining(self) -> None:
        # Note: with_timeout returns BaseOpConfig, so call subclass methods last
        config = (
            CheckOpConfig(timeout_seconds=600.0)
            .with_fail_on_error(False)
            .with_warning_threshold(0.05)
            .with_auto_schema(True)
        )
        assert config.fail_on_error is False
        assert config.warning_threshold == 0.05
        assert config.auto_schema is True
        assert config.timeout_seconds == 600.0

    def test_profile_config_chaining(self) -> None:
        config = (
            ProfileOpConfig()
            .with_samples(include=True, sample_size=5000)
            .with_histograms(include=False)
        )
        assert config.sample_size == 5000
        assert config.include_samples is True
        assert config.include_histograms is False

    def test_learn_config_chaining(self) -> None:
        config = (
            LearnOpConfig()
            .with_min_confidence(0.95)
            .with_categorical_threshold(30)
        )
        assert config.min_confidence == 0.95
        assert config.categorical_threshold == 30
