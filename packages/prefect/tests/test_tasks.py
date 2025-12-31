"""Tests for truthound_prefect.tasks module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from truthound_prefect.tasks.base import (
    AUTO_SCHEMA_CHECK_CONFIG,
    DEFAULT_CHECK_CONFIG,
    DEFAULT_LEARN_CONFIG,
    DEFAULT_PROFILE_CONFIG,
    FULL_PROFILE_CONFIG,
    LENIENT_CHECK_CONFIG,
    MINIMAL_PROFILE_CONFIG,
    STRICT_CHECK_CONFIG,
    STRICT_LEARN_CONFIG,
    BaseTaskConfig,
    CheckTaskConfig,
    LearnTaskConfig,
    ProfileTaskConfig,
)
from truthound_prefect.tasks.check import (
    create_check_task,
    data_quality_check_task,
)
from truthound_prefect.tasks.profile import (
    create_profile_task,
    data_quality_profile_task,
)
from truthound_prefect.tasks.learn import (
    create_learn_task,
    data_quality_learn_task,
)


class TestBaseTaskConfig:
    """Tests for BaseTaskConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BaseTaskConfig()
        assert config.enabled is True
        assert config.timeout_seconds == 300.0
        assert config.retries == 0
        assert config.retry_delay_seconds == 10.0
        assert config.tags == frozenset()
        assert config.cache_key is None
        assert config.cache_expiration_seconds is None

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = BaseTaskConfig()
        with pytest.raises(AttributeError):
            config.timeout_seconds = 600.0  # type: ignore

    def test_with_timeout(self) -> None:
        """Test with_timeout builder method."""
        config = BaseTaskConfig()
        new_config = config.with_timeout(600.0)
        assert config.timeout_seconds == 300.0  # Original unchanged
        assert new_config.timeout_seconds == 600.0

    def test_with_retries(self) -> None:
        """Test with_retries builder method."""
        config = BaseTaskConfig()
        new_config = config.with_retries(3, retry_delay_seconds=30.0)
        assert new_config.retries == 3
        assert new_config.retry_delay_seconds == 30.0

    def test_with_tags(self) -> None:
        """Test with_tags builder method."""
        config = BaseTaskConfig()
        new_config = config.with_tags("production", "critical")
        assert "production" in new_config.tags
        assert "critical" in new_config.tags

    def test_with_cache(self) -> None:
        """Test with_cache builder method."""
        config = BaseTaskConfig()
        new_config = config.with_cache("my_cache_key", expiration_seconds=3600.0)
        assert new_config.cache_key == "my_cache_key"
        assert new_config.cache_expiration_seconds == 3600.0


class TestCheckTaskConfig:
    """Tests for CheckTaskConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CheckTaskConfig()
        assert config.fail_on_error is True
        assert config.auto_schema is False
        assert config.rules == ()
        assert config.warning_threshold is None
        assert config.store_result is True
        assert config.result_key == "check_result"

    def test_with_fail_on_error(self) -> None:
        """Test with_fail_on_error builder method."""
        config = CheckTaskConfig()
        new_config = config.with_fail_on_error(False)
        assert new_config.fail_on_error is False

    def test_with_auto_schema(self) -> None:
        """Test with_auto_schema builder method."""
        config = CheckTaskConfig()
        new_config = config.with_auto_schema(True)
        assert new_config.auto_schema is True

    def test_with_warning_threshold(self) -> None:
        """Test with_warning_threshold builder method."""
        config = CheckTaskConfig()
        new_config = config.with_warning_threshold(0.05)
        assert new_config.warning_threshold == 0.05

    def test_with_rules(self) -> None:
        """Test with_rules builder method."""
        rules = [{"type": "not_null", "column": "id"}]
        config = CheckTaskConfig()
        new_config = config.with_rules(rules)
        assert len(new_config.rules) == 1
        assert new_config.rules[0]["type"] == "not_null"


class TestProfileTaskConfig:
    """Tests for ProfileTaskConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ProfileTaskConfig()
        assert config.include_histograms is False
        assert config.sample_size is None
        assert config.store_result is True
        assert config.result_key == "profile_result"

    def test_with_histograms(self) -> None:
        """Test with_histograms builder method."""
        config = ProfileTaskConfig()
        new_config = config.with_histograms(True)
        assert new_config.include_histograms is True

    def test_with_sample_size(self) -> None:
        """Test with_sample_size builder method."""
        config = ProfileTaskConfig()
        new_config = config.with_sample_size(1000)
        assert new_config.sample_size == 1000


class TestLearnTaskConfig:
    """Tests for LearnTaskConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LearnTaskConfig()
        assert config.infer_constraints is True
        assert config.min_confidence == 0.9
        assert config.categorical_threshold == 20
        assert config.store_result is True

    def test_with_min_confidence(self) -> None:
        """Test with_min_confidence builder method."""
        config = LearnTaskConfig()
        new_config = config.with_min_confidence(0.8)
        assert new_config.min_confidence == 0.8

    def test_with_categorical_threshold(self) -> None:
        """Test with_categorical_threshold builder method."""
        config = LearnTaskConfig()
        new_config = config.with_categorical_threshold(50)
        assert new_config.categorical_threshold == 50


class TestPresetConfigs:
    """Tests for preset task configurations."""

    def test_default_check_config(self) -> None:
        """Test DEFAULT_CHECK_CONFIG preset."""
        assert DEFAULT_CHECK_CONFIG.fail_on_error is True
        assert DEFAULT_CHECK_CONFIG.auto_schema is False

    def test_strict_check_config(self) -> None:
        """Test STRICT_CHECK_CONFIG preset."""
        assert STRICT_CHECK_CONFIG.fail_on_error is True
        assert STRICT_CHECK_CONFIG.warning_threshold is None
        assert "strict" in STRICT_CHECK_CONFIG.tags

    def test_lenient_check_config(self) -> None:
        """Test LENIENT_CHECK_CONFIG preset."""
        assert LENIENT_CHECK_CONFIG.fail_on_error is False
        assert LENIENT_CHECK_CONFIG.warning_threshold == 0.10

    def test_auto_schema_check_config(self) -> None:
        """Test AUTO_SCHEMA_CHECK_CONFIG preset."""
        assert AUTO_SCHEMA_CHECK_CONFIG.auto_schema is True
        assert "auto-schema" in AUTO_SCHEMA_CHECK_CONFIG.tags

    def test_default_profile_config(self) -> None:
        """Test DEFAULT_PROFILE_CONFIG preset."""
        assert DEFAULT_PROFILE_CONFIG.include_histograms is False

    def test_minimal_profile_config(self) -> None:
        """Test MINIMAL_PROFILE_CONFIG preset."""
        assert MINIMAL_PROFILE_CONFIG.include_histograms is False
        assert MINIMAL_PROFILE_CONFIG.sample_size == 10000
        assert "minimal" in MINIMAL_PROFILE_CONFIG.tags

    def test_full_profile_config(self) -> None:
        """Test FULL_PROFILE_CONFIG preset."""
        assert FULL_PROFILE_CONFIG.include_histograms is True
        assert FULL_PROFILE_CONFIG.sample_size is None
        assert "full" in FULL_PROFILE_CONFIG.tags

    def test_default_learn_config(self) -> None:
        """Test DEFAULT_LEARN_CONFIG preset."""
        assert DEFAULT_LEARN_CONFIG.infer_constraints is True

    def test_strict_learn_config(self) -> None:
        """Test STRICT_LEARN_CONFIG preset."""
        assert STRICT_LEARN_CONFIG.min_confidence == 0.95
        assert "strict" in STRICT_LEARN_CONFIG.tags


class TestTaskFactories:
    """Tests for task factory functions."""

    def test_create_check_task(self) -> None:
        """Test create_check_task factory."""
        task = create_check_task(
            name="custom_check",
            config=STRICT_CHECK_CONFIG,
        )
        assert task is not None
        assert callable(task)

    def test_create_profile_task(self) -> None:
        """Test create_profile_task factory."""
        task = create_profile_task(
            name="custom_profile",
            config=FULL_PROFILE_CONFIG,
        )
        assert task is not None
        assert callable(task)

    def test_create_learn_task(self) -> None:
        """Test create_learn_task factory."""
        task = create_learn_task(
            name="custom_learn",
            config=STRICT_LEARN_CONFIG,
        )
        assert task is not None
        assert callable(task)
