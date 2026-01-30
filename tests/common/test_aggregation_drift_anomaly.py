"""Tests for DriftResultAggregator and AnomalyResultAggregator (TASK 4-3)."""

from __future__ import annotations

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
from common.engines.aggregation import (
    AggregationConfig,
    AnomalyResultAggregator,
    DriftResultAggregator,
    MultiEngineAggregator,
    ResultAggregationStrategy,
    StatusPriority,
    aggregate_anomaly_results,
    aggregate_drift_results,
)


# =============================================================================
# StatusPriority Extension Tests
# =============================================================================


class TestStatusPriorityDriftAnomaly:
    def test_from_drift_status_mapping(self) -> None:
        assert StatusPriority.from_drift_status(DriftStatus.ERROR) == StatusPriority.ERROR
        assert StatusPriority.from_drift_status(DriftStatus.DRIFT_DETECTED) == StatusPriority.FAILED
        assert StatusPriority.from_drift_status(DriftStatus.WARNING) == StatusPriority.WARNING
        assert StatusPriority.from_drift_status(DriftStatus.NO_DRIFT) == StatusPriority.PASSED

    def test_from_anomaly_status_mapping(self) -> None:
        assert StatusPriority.from_anomaly_status(AnomalyStatus.ERROR) == StatusPriority.ERROR
        assert (
            StatusPriority.from_anomaly_status(AnomalyStatus.ANOMALY_DETECTED)
            == StatusPriority.FAILED
        )
        assert StatusPriority.from_anomaly_status(AnomalyStatus.WARNING) == StatusPriority.WARNING
        assert StatusPriority.from_anomaly_status(AnomalyStatus.NORMAL) == StatusPriority.PASSED

    def test_from_drift_status_unknown_defaults_to_passed(self) -> None:
        """Unknown status names should default to PASSED."""

        class FakeStatus:
            name = "UNKNOWN_STATUS"

        assert StatusPriority.from_drift_status(FakeStatus()) == StatusPriority.PASSED

    def test_from_anomaly_status_unknown_defaults_to_passed(self) -> None:
        class FakeStatus:
            name = "UNKNOWN_STATUS"

        assert StatusPriority.from_anomaly_status(FakeStatus()) == StatusPriority.PASSED


# =============================================================================
# Helper Fixtures
# =============================================================================


def _make_drift_result(
    status: DriftStatus = DriftStatus.NO_DRIFT,
    columns: tuple[ColumnDrift, ...] = (),
    total_columns: int = 5,
    method: str = "ks",
) -> DriftResult:
    return DriftResult(
        status=status,
        drifted_columns=columns,
        total_columns=total_columns,
        drifted_count=sum(1 for c in columns if c.is_drifted),
        method=method,
    )


def _make_column_drift(
    column: str,
    is_drifted: bool = True,
    statistic: float = 0.5,
    p_value: float = 0.01,
    method: DriftMethod = DriftMethod.KS,
) -> ColumnDrift:
    return ColumnDrift(
        column=column,
        method=method,
        statistic=statistic,
        p_value=p_value,
        threshold=0.05,
        is_drifted=is_drifted,
        severity=Severity.ERROR if is_drifted else Severity.INFO,
    )


def _make_anomaly_result(
    status: AnomalyStatus = AnomalyStatus.NORMAL,
    anomalies: tuple[AnomalyScore, ...] = (),
    total_rows: int = 100,
    anomalous_rows: int = 0,
    detector: str = "isolation_forest",
) -> AnomalyResult:
    return AnomalyResult(
        status=status,
        anomalies=anomalies,
        anomalous_row_count=anomalous_rows,
        total_row_count=total_rows,
        detector=detector,
    )


def _make_anomaly_score(
    column: str,
    score: float = 0.9,
    threshold: float = 0.5,
    is_anomaly: bool = True,
    detector: str = "isolation_forest",
) -> AnomalyScore:
    return AnomalyScore(
        column=column,
        score=score,
        threshold=threshold,
        is_anomaly=is_anomaly,
        detector=detector,
    )


# =============================================================================
# DriftResultAggregator Tests
# =============================================================================


class TestDriftResultAggregator:
    def setup_method(self) -> None:
        self.aggregator = DriftResultAggregator()

    def test_aggregate_single_result(self) -> None:
        result = _make_drift_result(status=DriftStatus.NO_DRIFT)
        config = AggregationConfig(strategy=ResultAggregationStrategy.MERGE)
        aggregated = self.aggregator.aggregate([result], config)
        assert aggregated.status == DriftStatus.NO_DRIFT

    def test_aggregate_merge_combines_columns(self) -> None:
        col_a = _make_column_drift("col_a", is_drifted=True)
        col_b = _make_column_drift("col_b", is_drifted=False)
        r1 = _make_drift_result(
            status=DriftStatus.DRIFT_DETECTED, columns=(col_a,), total_columns=3
        )
        r2 = _make_drift_result(
            status=DriftStatus.NO_DRIFT, columns=(col_b,), total_columns=3
        )
        config = AggregationConfig(strategy=ResultAggregationStrategy.MERGE)
        aggregated = self.aggregator.aggregate([r1, r2], config)
        # Worst status wins
        assert aggregated.status == DriftStatus.DRIFT_DETECTED
        assert len(aggregated.drifted_columns) == 2

    def test_aggregate_merge_deduplicates_columns(self) -> None:
        col = _make_column_drift("col_a", is_drifted=True, statistic=0.8)
        col_dup = _make_column_drift("col_a", is_drifted=True, statistic=0.3)
        r1 = _make_drift_result(columns=(col,))
        r2 = _make_drift_result(columns=(col_dup,))
        config = AggregationConfig(strategy=ResultAggregationStrategy.MERGE)
        aggregated = self.aggregator.aggregate([r1, r2], config)
        # Should deduplicate, keeping higher statistic
        col_names = [c.column for c in aggregated.drifted_columns]
        assert col_names.count("col_a") == 1

    def test_aggregate_worst_strategy(self) -> None:
        r1 = _make_drift_result(status=DriftStatus.NO_DRIFT)
        r2 = _make_drift_result(status=DriftStatus.DRIFT_DETECTED)
        config = AggregationConfig(strategy=ResultAggregationStrategy.WORST)
        aggregated = self.aggregator.aggregate([r1, r2], config)
        assert aggregated.status == DriftStatus.DRIFT_DETECTED

    def test_aggregate_best_strategy(self) -> None:
        r1 = _make_drift_result(status=DriftStatus.ERROR)
        r2 = _make_drift_result(status=DriftStatus.NO_DRIFT)
        config = AggregationConfig(strategy=ResultAggregationStrategy.BEST)
        aggregated = self.aggregator.aggregate([r1, r2], config)
        assert aggregated.status == DriftStatus.NO_DRIFT

    def test_aggregate_empty_list_raises(self) -> None:
        config = AggregationConfig(strategy=ResultAggregationStrategy.MERGE)
        with pytest.raises(Exception):
            self.aggregator.aggregate([], config)


# =============================================================================
# AnomalyResultAggregator Tests
# =============================================================================


class TestAnomalyResultAggregator:
    def setup_method(self) -> None:
        self.aggregator = AnomalyResultAggregator()

    def test_aggregate_single_result(self) -> None:
        result = _make_anomaly_result(status=AnomalyStatus.NORMAL)
        config = AggregationConfig(strategy=ResultAggregationStrategy.MERGE)
        aggregated = self.aggregator.aggregate([result], config)
        assert aggregated.status == AnomalyStatus.NORMAL

    def test_aggregate_merge_combines_anomalies(self) -> None:
        a1 = _make_anomaly_score("col_a", is_anomaly=True)
        a2 = _make_anomaly_score("col_b", is_anomaly=False)
        r1 = _make_anomaly_result(
            status=AnomalyStatus.ANOMALY_DETECTED, anomalies=(a1,), anomalous_rows=5
        )
        r2 = _make_anomaly_result(
            status=AnomalyStatus.NORMAL, anomalies=(a2,), anomalous_rows=0
        )
        config = AggregationConfig(strategy=ResultAggregationStrategy.MERGE)
        aggregated = self.aggregator.aggregate([r1, r2], config)
        assert aggregated.status == AnomalyStatus.ANOMALY_DETECTED
        assert len(aggregated.anomalies) == 2

    def test_aggregate_worst_strategy(self) -> None:
        r1 = _make_anomaly_result(status=AnomalyStatus.NORMAL)
        r2 = _make_anomaly_result(status=AnomalyStatus.ANOMALY_DETECTED)
        config = AggregationConfig(strategy=ResultAggregationStrategy.WORST)
        aggregated = self.aggregator.aggregate([r1, r2], config)
        assert aggregated.status == AnomalyStatus.ANOMALY_DETECTED

    def test_aggregate_consensus_strategy(self) -> None:
        """CONSENSUS is custom for AnomalyResultAggregator."""
        a1 = _make_anomaly_score("col_a", is_anomaly=True)
        a2 = _make_anomaly_score("col_a", is_anomaly=False)
        a3 = _make_anomaly_score("col_a", is_anomaly=True)
        r1 = _make_anomaly_result(status=AnomalyStatus.ANOMALY_DETECTED, anomalies=(a1,))
        r2 = _make_anomaly_result(status=AnomalyStatus.NORMAL, anomalies=(a2,))
        r3 = _make_anomaly_result(status=AnomalyStatus.ANOMALY_DETECTED, anomalies=(a3,))
        config = AggregationConfig(strategy=ResultAggregationStrategy.CONSENSUS)
        aggregated = self.aggregator.aggregate([r1, r2, r3], config)
        # 2/3 say anomaly -> should be detected
        assert aggregated.status == AnomalyStatus.ANOMALY_DETECTED

    def test_aggregate_empty_list_raises(self) -> None:
        config = AggregationConfig(strategy=ResultAggregationStrategy.MERGE)
        with pytest.raises(Exception):
            self.aggregator.aggregate([], config)


# =============================================================================
# MultiEngineAggregator Tests
# =============================================================================


class TestMultiEngineAggregatorDriftAnomaly:
    def setup_method(self) -> None:
        self.aggregator = MultiEngineAggregator()

    def test_aggregate_drift_results(self) -> None:
        r1 = _make_drift_result(status=DriftStatus.NO_DRIFT)
        r2 = _make_drift_result(status=DriftStatus.DRIFT_DETECTED)
        entries = {"engine1": r1, "engine2": r2}
        aggregated = self.aggregator.aggregate_drift_results(entries)
        assert aggregated is not None

    def test_aggregate_anomaly_results(self) -> None:
        r1 = _make_anomaly_result(status=AnomalyStatus.NORMAL)
        r2 = _make_anomaly_result(status=AnomalyStatus.ANOMALY_DETECTED)
        entries = {"engine1": r1, "engine2": r2}
        aggregated = self.aggregator.aggregate_anomaly_results(entries)
        assert aggregated is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    def test_aggregate_drift_results_function(self) -> None:
        results = {
            "e1": _make_drift_result(status=DriftStatus.NO_DRIFT),
            "e2": _make_drift_result(status=DriftStatus.DRIFT_DETECTED),
        }
        aggregated = aggregate_drift_results(results)
        assert aggregated is not None

    def test_aggregate_anomaly_results_function(self) -> None:
        results = {
            "e1": _make_anomaly_result(status=AnomalyStatus.NORMAL),
            "e2": _make_anomaly_result(status=AnomalyStatus.ANOMALY_DETECTED),
        }
        aggregated = aggregate_anomaly_results(results)
        assert aggregated is not None
