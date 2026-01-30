"""Tests for DriftConfig, AnomalyConfig, StreamConfig (TASK 0-3)."""

from __future__ import annotations

import pytest

from common.base import AnomalyConfig, DriftConfig, StreamConfig


# =============================================================================
# DriftConfig Tests
# =============================================================================


class TestDriftConfig:
    def test_default_creation(self) -> None:
        config = DriftConfig()
        assert config.method == "auto"
        assert config.columns is None
        assert config.threshold is None
        assert config.extra == {}

    def test_with_method(self) -> None:
        config = DriftConfig().with_method("ks")
        assert config.method == "ks"

    def test_with_columns(self) -> None:
        config = DriftConfig().with_columns("age", "income")
        assert config.columns == ("age", "income")

    def test_with_threshold(self) -> None:
        config = DriftConfig().with_threshold(0.05)
        assert config.threshold == 0.05

    def test_with_extra(self) -> None:
        config = DriftConfig().with_extra(custom_param="value")
        assert config.extra == {"custom_param": "value"}

    def test_builder_chaining(self) -> None:
        config = (
            DriftConfig()
            .with_method("psi")
            .with_columns("col1")
            .with_threshold(0.1)
        )
        assert config.method == "psi"
        assert config.columns == ("col1",)
        assert config.threshold == 0.1

    def test_immutable(self) -> None:
        config = DriftConfig()
        with pytest.raises(AttributeError):
            config.method = "ks"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        config = DriftConfig(method="ks", columns=("a", "b"), threshold=0.05)
        d = config.to_dict()
        assert d["method"] == "ks"
        assert d["columns"] == ["a", "b"]
        assert d["threshold"] == 0.05

    def test_to_dict_none_columns(self) -> None:
        d = DriftConfig().to_dict()
        assert d["columns"] is None

    def test_roundtrip(self) -> None:
        config = DriftConfig(
            method="psi",
            columns=("x", "y"),
            threshold=0.1,
            min_severity="warning",
            timeout_seconds=60,
        )
        restored = DriftConfig.from_dict(config.to_dict())
        assert restored.method == config.method
        assert restored.columns == config.columns
        assert restored.threshold == config.threshold
        assert restored.min_severity == config.min_severity
        assert restored.timeout_seconds == config.timeout_seconds

    def test_from_dict_defaults(self) -> None:
        config = DriftConfig.from_dict({})
        assert config.method == "auto"
        assert config.columns is None


# =============================================================================
# AnomalyConfig Tests
# =============================================================================


class TestAnomalyConfig:
    def test_default_creation(self) -> None:
        config = AnomalyConfig()
        assert config.detector == "isolation_forest"
        assert config.contamination == 0.05
        assert config.columns is None

    def test_contamination_validation_lower(self) -> None:
        with pytest.raises(ValueError, match="contamination"):
            AnomalyConfig(contamination=0.0)

    def test_contamination_validation_upper(self) -> None:
        with pytest.raises(ValueError, match="contamination"):
            AnomalyConfig(contamination=0.5)

    def test_contamination_validation_negative(self) -> None:
        with pytest.raises(ValueError, match="contamination"):
            AnomalyConfig(contamination=-0.1)

    def test_valid_contamination_edge(self) -> None:
        config = AnomalyConfig(contamination=0.01)
        assert config.contamination == 0.01
        config = AnomalyConfig(contamination=0.49)
        assert config.contamination == 0.49

    def test_with_detector(self) -> None:
        config = AnomalyConfig().with_detector("z_score")
        assert config.detector == "z_score"

    def test_with_columns(self) -> None:
        config = AnomalyConfig().with_columns("col1", "col2")
        assert config.columns == ("col1", "col2")

    def test_with_contamination(self) -> None:
        config = AnomalyConfig().with_contamination(0.1)
        assert config.contamination == 0.1

    def test_with_contamination_invalid(self) -> None:
        with pytest.raises(ValueError):
            AnomalyConfig().with_contamination(0.0)

    def test_with_extra(self) -> None:
        config = AnomalyConfig().with_extra(n_estimators=100)
        assert config.extra == {"n_estimators": 100}

    def test_roundtrip(self) -> None:
        config = AnomalyConfig(
            detector="lof",
            columns=("a",),
            contamination=0.1,
            threshold=0.8,
            timeout_seconds=30,
        )
        restored = AnomalyConfig.from_dict(config.to_dict())
        assert restored.detector == config.detector
        assert restored.columns == config.columns
        assert restored.contamination == config.contamination
        assert restored.threshold == config.threshold

    def test_from_dict_defaults(self) -> None:
        config = AnomalyConfig.from_dict({})
        assert config.detector == "isolation_forest"
        assert config.contamination == 0.05

    def test_immutable(self) -> None:
        config = AnomalyConfig()
        with pytest.raises(AttributeError):
            config.detector = "z_score"  # type: ignore[misc]


# =============================================================================
# StreamConfig Tests
# =============================================================================


class TestStreamConfig:
    def test_default_creation(self) -> None:
        config = StreamConfig()
        assert config.batch_size == 1000
        assert config.max_batches is None
        assert config.fail_fast is False

    def test_batch_size_validation(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            StreamConfig(batch_size=0)

    def test_batch_size_negative(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            StreamConfig(batch_size=-1)

    def test_with_batch_size(self) -> None:
        config = StreamConfig().with_batch_size(500)
        assert config.batch_size == 500

    def test_with_batch_size_invalid(self) -> None:
        with pytest.raises(ValueError):
            StreamConfig().with_batch_size(0)

    def test_with_fail_fast(self) -> None:
        config = StreamConfig().with_fail_fast()
        assert config.fail_fast is True

    def test_with_extra(self) -> None:
        config = StreamConfig().with_extra(source="kafka")
        assert config.extra == {"source": "kafka"}

    def test_roundtrip(self) -> None:
        config = StreamConfig(
            batch_size=500,
            max_batches=10,
            timeout_per_batch_seconds=5.0,
            fail_fast=True,
        )
        restored = StreamConfig.from_dict(config.to_dict())
        assert restored.batch_size == config.batch_size
        assert restored.max_batches == config.max_batches
        assert restored.timeout_per_batch_seconds == config.timeout_per_batch_seconds
        assert restored.fail_fast == config.fail_fast

    def test_from_dict_defaults(self) -> None:
        config = StreamConfig.from_dict({})
        assert config.batch_size == 1000
        assert config.fail_fast is False

    def test_immutable(self) -> None:
        config = StreamConfig()
        with pytest.raises(AttributeError):
            config.batch_size = 500  # type: ignore[misc]
