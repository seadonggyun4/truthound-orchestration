"""Tests for drift and anomaly Result types (TASK 0-1)."""

from __future__ import annotations

import json

import pytest

from common.base import (
    AnomalyResult,
    AnomalyScore,
    AnomalyStatus,
    ColumnDrift,
    DriftMethod,
    DriftResult,
    DriftStatus,
    Severity,
)


# =============================================================================
# DriftStatus Enum Tests
# =============================================================================


class TestDriftStatus:
    def test_is_drifted_true(self) -> None:
        assert DriftStatus.DRIFT_DETECTED.is_drifted() is True
        assert DriftStatus.WARNING.is_drifted() is True

    def test_is_drifted_false(self) -> None:
        assert DriftStatus.NO_DRIFT.is_drifted() is False
        assert DriftStatus.ERROR.is_drifted() is False

    def test_all_values_exist(self) -> None:
        expected = {"NO_DRIFT", "DRIFT_DETECTED", "WARNING", "ERROR"}
        assert {s.name for s in DriftStatus} == expected


# =============================================================================
# AnomalyStatus Enum Tests
# =============================================================================


class TestAnomalyStatus:
    def test_has_anomalies_true(self) -> None:
        assert AnomalyStatus.ANOMALY_DETECTED.has_anomalies() is True
        assert AnomalyStatus.WARNING.has_anomalies() is True

    def test_has_anomalies_false(self) -> None:
        assert AnomalyStatus.NORMAL.has_anomalies() is False
        assert AnomalyStatus.ERROR.has_anomalies() is False


# =============================================================================
# DriftMethod Enum Tests
# =============================================================================


class TestDriftMethod:
    def test_all_methods_present(self) -> None:
        expected = {
            "ks", "psi", "chi2", "kl", "js", "wasserstein", "hellinger",
            "bhattacharyya", "tv", "energy", "mmd", "cvm", "anderson_darling", "auto",
        }
        assert {m.value for m in DriftMethod} == expected

    def test_auto_method(self) -> None:
        assert DriftMethod.AUTO.value == "auto"

    def test_from_value(self) -> None:
        assert DriftMethod("ks") == DriftMethod.KS


# =============================================================================
# ColumnDrift Tests
# =============================================================================


class TestColumnDrift:
    @pytest.fixture()
    def sample_drift(self) -> ColumnDrift:
        return ColumnDrift(
            column="age",
            method=DriftMethod.KS,
            statistic=0.35,
            p_value=0.001,
            threshold=0.05,
            is_drifted=True,
            severity=Severity.WARNING,
            baseline_stats={"mean": 30.0, "std": 5.0},
            current_stats={"mean": 45.0, "std": 8.0},
            metadata={"sample_size": 1000},
        )

    def test_creation(self, sample_drift: ColumnDrift) -> None:
        assert sample_drift.column == "age"
        assert sample_drift.method == DriftMethod.KS
        assert sample_drift.is_drifted is True

    def test_to_dict(self, sample_drift: ColumnDrift) -> None:
        d = sample_drift.to_dict()
        assert d["column"] == "age"
        assert d["method"] == "ks"
        assert d["severity"] == "WARNING"
        assert d["is_drifted"] is True

    def test_roundtrip(self, sample_drift: ColumnDrift) -> None:
        d = sample_drift.to_dict()
        restored = ColumnDrift.from_dict(d)
        assert restored.column == sample_drift.column
        assert restored.method == sample_drift.method
        assert restored.statistic == sample_drift.statistic
        assert restored.p_value == sample_drift.p_value
        assert restored.is_drifted == sample_drift.is_drifted
        assert restored.severity == sample_drift.severity

    def test_from_dict_defaults(self) -> None:
        d = {
            "column": "x",
            "method": "psi",
            "statistic": 0.1,
            "threshold": 0.2,
            "is_drifted": False,
        }
        cd = ColumnDrift.from_dict(d)
        assert cd.p_value is None
        assert cd.severity == Severity.INFO
        assert cd.baseline_stats == {}

    def test_immutable(self, sample_drift: ColumnDrift) -> None:
        with pytest.raises(AttributeError):
            sample_drift.column = "new"  # type: ignore[misc]


# =============================================================================
# DriftResult Tests
# =============================================================================


class TestDriftResult:
    @pytest.fixture()
    def drifted_result(self) -> DriftResult:
        col = ColumnDrift(
            column="age",
            method=DriftMethod.KS,
            statistic=0.35,
            p_value=0.001,
            threshold=0.05,
            is_drifted=True,
            severity=Severity.WARNING,
        )
        return DriftResult(
            status=DriftStatus.DRIFT_DETECTED,
            drifted_columns=(col,),
            total_columns=5,
            drifted_count=1,
            method=DriftMethod.KS,
            execution_time_ms=123.4,
        )

    def test_is_drifted(self, drifted_result: DriftResult) -> None:
        assert drifted_result.is_drifted is True

    def test_drift_rate(self, drifted_result: DriftResult) -> None:
        assert drifted_result.drift_rate == pytest.approx(20.0)

    def test_drift_rate_zero_columns(self) -> None:
        result = DriftResult(status=DriftStatus.NO_DRIFT, total_columns=0)
        assert result.drift_rate == 0.0

    def test_no_drift(self) -> None:
        result = DriftResult(status=DriftStatus.NO_DRIFT, total_columns=5)
        assert result.is_drifted is False

    def test_to_dict(self, drifted_result: DriftResult) -> None:
        d = drifted_result.to_dict()
        assert d["status"] == "DRIFT_DETECTED"
        assert d["is_drifted"] is True
        assert d["drift_rate"] == pytest.approx(20.0)
        assert len(d["drifted_columns"]) == 1

    def test_to_json(self, drifted_result: DriftResult) -> None:
        j = drifted_result.to_json()
        parsed = json.loads(j)
        assert parsed["status"] == "DRIFT_DETECTED"

    def test_roundtrip(self, drifted_result: DriftResult) -> None:
        d = drifted_result.to_dict()
        restored = DriftResult.from_dict(d)
        assert restored.status == drifted_result.status
        assert restored.total_columns == drifted_result.total_columns
        assert restored.drifted_count == drifted_result.drifted_count
        assert restored.method == drifted_result.method
        assert len(restored.drifted_columns) == 1

    def test_from_dict_defaults(self) -> None:
        d = {"status": "NO_DRIFT"}
        result = DriftResult.from_dict(d)
        assert result.total_columns == 0
        assert result.method == DriftMethod.AUTO
        assert result.drifted_columns == ()


# =============================================================================
# AnomalyScore Tests
# =============================================================================


class TestAnomalyScore:
    @pytest.fixture()
    def sample_score(self) -> AnomalyScore:
        return AnomalyScore(
            column="amount",
            score=0.85,
            threshold=0.7,
            is_anomaly=True,
            detector="isolation_forest",
            metadata={"contamination": 0.05},
        )

    def test_creation(self, sample_score: AnomalyScore) -> None:
        assert sample_score.column == "amount"
        assert sample_score.is_anomaly is True

    def test_roundtrip(self, sample_score: AnomalyScore) -> None:
        d = sample_score.to_dict()
        restored = AnomalyScore.from_dict(d)
        assert restored.column == sample_score.column
        assert restored.score == sample_score.score
        assert restored.is_anomaly == sample_score.is_anomaly
        assert restored.detector == sample_score.detector

    def test_from_dict_defaults(self) -> None:
        d = {"column": "x", "score": 0.5, "threshold": 0.7, "is_anomaly": False}
        score = AnomalyScore.from_dict(d)
        assert score.detector == "isolation_forest"
        assert score.metadata == {}


# =============================================================================
# AnomalyResult Tests
# =============================================================================


class TestAnomalyResult:
    @pytest.fixture()
    def anomaly_result(self) -> AnomalyResult:
        score = AnomalyScore(
            column="amount",
            score=0.85,
            threshold=0.7,
            is_anomaly=True,
        )
        return AnomalyResult(
            status=AnomalyStatus.ANOMALY_DETECTED,
            anomalies=(score,),
            anomalous_row_count=50,
            total_row_count=1000,
            detector="isolation_forest",
            execution_time_ms=456.7,
        )

    def test_has_anomalies(self, anomaly_result: AnomalyResult) -> None:
        assert anomaly_result.has_anomalies is True

    def test_anomaly_rate(self, anomaly_result: AnomalyResult) -> None:
        assert anomaly_result.anomaly_rate == pytest.approx(5.0)

    def test_anomaly_rate_zero_rows(self) -> None:
        result = AnomalyResult(status=AnomalyStatus.NORMAL, total_row_count=0)
        assert result.anomaly_rate == 0.0

    def test_no_anomalies(self) -> None:
        result = AnomalyResult(status=AnomalyStatus.NORMAL, total_row_count=1000)
        assert result.has_anomalies is False

    def test_to_dict(self, anomaly_result: AnomalyResult) -> None:
        d = anomaly_result.to_dict()
        assert d["status"] == "ANOMALY_DETECTED"
        assert d["has_anomalies"] is True
        assert d["anomaly_rate"] == pytest.approx(5.0)
        assert len(d["anomalies"]) == 1

    def test_to_json(self, anomaly_result: AnomalyResult) -> None:
        j = anomaly_result.to_json()
        parsed = json.loads(j)
        assert parsed["status"] == "ANOMALY_DETECTED"

    def test_roundtrip(self, anomaly_result: AnomalyResult) -> None:
        d = anomaly_result.to_dict()
        restored = AnomalyResult.from_dict(d)
        assert restored.status == anomaly_result.status
        assert restored.anomalous_row_count == anomaly_result.anomalous_row_count
        assert restored.detector == anomaly_result.detector
        assert len(restored.anomalies) == 1

    def test_from_dict_defaults(self) -> None:
        d = {"status": "NORMAL"}
        result = AnomalyResult.from_dict(d)
        assert result.anomalies == ()
        assert result.detector == "isolation_forest"
        assert result.total_row_count == 0
