"""Tests for Assets module."""

import pytest

from truthound_dagster.assets.config import (
    ProfileAssetConfig,
    QualityAssetConfig,
    QualityCheckMode,
)


class TestQualityCheckMode:
    """Tests for QualityCheckMode enum."""

    def test_modes(self) -> None:
        assert QualityCheckMode.BEFORE.value == "before"
        assert QualityCheckMode.AFTER.value == "after"
        assert QualityCheckMode.BOTH.value == "both"
        assert QualityCheckMode.NONE.value == "none"


class TestQualityAssetConfig:
    """Tests for QualityAssetConfig."""

    def test_default_creation(self) -> None:
        config = QualityAssetConfig()
        assert config.rules == ()
        assert config.check_mode == QualityCheckMode.AFTER
        assert config.fail_on_error is True
        assert config.store_result is True

    def test_with_rules(self) -> None:
        rules = (
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
        )
        config = QualityAssetConfig(rules=rules)
        assert len(config.rules) == 2

    def test_with_check_mode(self) -> None:
        config = QualityAssetConfig().with_check_mode(QualityCheckMode.BEFORE)
        assert config.check_mode == QualityCheckMode.BEFORE

    def test_with_fail_on_error(self) -> None:
        config = QualityAssetConfig().with_fail_on_error(False)
        assert config.fail_on_error is False

    def test_with_auto_schema(self) -> None:
        config = QualityAssetConfig().with_auto_schema(True)
        assert config.auto_schema is True

    def test_to_dict(self) -> None:
        config = QualityAssetConfig(
            rules=({"type": "not_null", "column": "id"},),
            check_mode=QualityCheckMode.AFTER,
            fail_on_error=True,
        )
        data = config.to_dict()
        assert "rules" in data
        assert data["check_mode"] == "after"
        assert data["fail_on_error"] is True

    def test_immutability(self) -> None:
        config = QualityAssetConfig()
        with pytest.raises(AttributeError):
            config.fail_on_error = False  # type: ignore


class TestProfileAssetConfig:
    """Tests for ProfileAssetConfig."""

    def test_default_creation(self) -> None:
        config = ProfileAssetConfig()
        assert config.include_histograms is True
        assert config.include_samples is True
        assert config.store_result is True

    def test_with_histograms(self) -> None:
        config = ProfileAssetConfig().with_histograms(False)
        assert config.include_histograms is False

    def test_with_samples(self) -> None:
        config = ProfileAssetConfig().with_samples(True, sample_size=100)
        assert config.include_samples is True
        assert config.sample_size == 100

    def test_to_dict(self) -> None:
        config = ProfileAssetConfig(
            include_histograms=True,
            sample_size=1000,
        )
        data = config.to_dict()
        assert data["include_histograms"] is True
        assert data["sample_size"] == 1000


class TestConfigChaining:
    """Tests for config method chaining."""

    def test_quality_config_chaining(self) -> None:
        config = (
            QualityAssetConfig()
            .with_check_mode(QualityCheckMode.BOTH)
            .with_fail_on_error(False)
            .with_auto_schema(True)
        )
        assert config.check_mode == QualityCheckMode.BOTH
        assert config.fail_on_error is False
        assert config.auto_schema is True

    def test_profile_config_chaining(self) -> None:
        config = (
            ProfileAssetConfig()
            .with_histograms(False)
            .with_samples(True, sample_size=5000)
        )
        assert config.include_histograms is False
        assert config.include_samples is True
        assert config.sample_size == 5000


class TestConfigRoundtrip:
    """Tests for config serialization."""

    def test_quality_config_to_dict(self) -> None:
        original = QualityAssetConfig(
            rules=({"type": "not_null", "column": "id"},),
            check_mode=QualityCheckMode.BOTH,
            fail_on_error=False,
            warning_threshold=0.05,
            auto_schema=True,
            store_result=True,
        )
        data = original.to_dict()

        assert data["check_mode"] == "both"
        assert data["fail_on_error"] is False
        assert data["warning_threshold"] == 0.05
        assert data["auto_schema"] is True
        assert data["store_result"] is True

    def test_profile_config_to_dict(self) -> None:
        original = ProfileAssetConfig(
            include_histograms=True,
            include_samples=True,
            sample_size=1000,
            store_result=False,
        )
        data = original.to_dict()

        assert data["include_histograms"] is True
        assert data["include_samples"] is True
        assert data["sample_size"] == 1000
        assert data["store_result"] is False
