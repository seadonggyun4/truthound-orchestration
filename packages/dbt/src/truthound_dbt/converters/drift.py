"""Drift and Anomaly Detection SQL Converters for dbt.

This module provides rule handlers that generate SQL for drift detection
and anomaly detection within dbt tests. Since dbt operates in SQL,
these handlers produce statistical comparison queries rather than
calling Python engines directly.

Drift Detection Approach:
    Compare statistical properties (mean, stddev, null_rate, distinct_count)
    between a baseline model and a current model. Flag columns where
    differences exceed configured thresholds.

Anomaly Detection Approach:
    Use statistical methods (z-score, IQR) to identify outlier rows
    within a single model.

Example:
    >>> from truthound_dbt.converters.drift import DriftMeanHandler
    >>> from truthound_dbt.converters.base import ConversionContext
    >>>
    >>> handler = DriftMeanHandler()
    >>> rule = {
    ...     "type": "drift_mean",
    ...     "column": "revenue",
    ...     "baseline_model": "{{ ref('baseline_revenue') }}",
    ...     "threshold": 0.1,
    ... }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from truthound_dbt.converters.base import (
    ConversionContext,
    InvalidRuleError,
    RuleSQL,
)
from truthound_dbt.converters.rules import RuleHandler

if TYPE_CHECKING:
    from truthound_dbt.adapters.base import SQLAdapter


# =============================================================================
# Drift Detection Handlers
# =============================================================================


class DriftMeanHandler(RuleHandler):
    """Generate SQL to detect drift via mean value comparison.

    Compares the mean value of a numeric column between baseline
    and current models. Drift is flagged when the relative difference
    exceeds the threshold.

    Rule format:
        {
            "type": "drift_mean",
            "column": "revenue",
            "baseline_model": "{{ ref('baseline') }}",
            "threshold": 0.1  # 10% relative difference
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["drift_mean", "drift_mean_check"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        baseline_model = rule.get("baseline_model")
        threshold = rule.get("threshold", 0.1)

        if not baseline_model:
            raise InvalidRuleError("drift_mean", "baseline_model is required")

        col = adapter.quote_column(column)
        where_clause = (
            f"abs(avg({col}) - "
            f"(select avg({col}) from {baseline_model})) / "
            f"nullif((select avg({col}) from {baseline_model}), 0) "
            f"> {threshold}"
        )

        return RuleSQL(
            where_clause=where_clause,
            rule_type="drift_mean",
            column=column,
            select_clause=(
                f"'drift_mean' as rule_type, "
                f"'{column}' as column_name, "
                f"avg({col}) as current_value, "
                f"(select avg({col}) from {baseline_model}) as baseline_value, "
                f"{threshold} as threshold"
            ),
            metadata={
                "baseline_model": baseline_model,
                "threshold": threshold,
                "method": "mean",
            },
        )


class DriftStddevHandler(RuleHandler):
    """Generate SQL to detect drift via standard deviation comparison.

    Rule format:
        {
            "type": "drift_stddev",
            "column": "amount",
            "baseline_model": "{{ ref('baseline') }}",
            "threshold": 0.2
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["drift_stddev", "drift_stddev_check"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        baseline_model = rule.get("baseline_model")
        threshold = rule.get("threshold", 0.2)

        if not baseline_model:
            raise InvalidRuleError("drift_stddev", "baseline_model is required")

        col = adapter.quote_column(column)
        where_clause = (
            f"abs(stddev({col}) - "
            f"(select stddev({col}) from {baseline_model})) / "
            f"nullif((select stddev({col}) from {baseline_model}), 0) "
            f"> {threshold}"
        )

        return RuleSQL(
            where_clause=where_clause,
            rule_type="drift_stddev",
            column=column,
            select_clause=(
                f"'drift_stddev' as rule_type, "
                f"'{column}' as column_name, "
                f"stddev({col}) as current_value, "
                f"(select stddev({col}) from {baseline_model}) as baseline_value, "
                f"{threshold} as threshold"
            ),
            metadata={
                "baseline_model": baseline_model,
                "threshold": threshold,
                "method": "stddev",
            },
        )


class DriftNullRateHandler(RuleHandler):
    """Generate SQL to detect drift via null rate comparison.

    Rule format:
        {
            "type": "drift_null_rate",
            "column": "email",
            "baseline_model": "{{ ref('baseline') }}",
            "threshold": 0.05
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["drift_null_rate", "drift_nulls"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        baseline_model = rule.get("baseline_model")
        threshold = rule.get("threshold", 0.05)

        if not baseline_model:
            raise InvalidRuleError("drift_null_rate", "baseline_model is required")

        col = adapter.quote_column(column)
        null_rate_expr = (
            f"sum(case when {col} is null then 1 else 0 end)::float "
            f"/ nullif(count(*), 0)"
        )
        baseline_null_rate = f"(select {null_rate_expr} from {baseline_model})"

        where_clause = (
            f"abs(({null_rate_expr}) - {baseline_null_rate}) > {threshold}"
        )

        return RuleSQL(
            where_clause=where_clause,
            rule_type="drift_null_rate",
            column=column,
            select_clause=(
                f"'drift_null_rate' as rule_type, "
                f"'{column}' as column_name, "
                f"({null_rate_expr}) as current_value, "
                f"{baseline_null_rate} as baseline_value, "
                f"{threshold} as threshold"
            ),
            metadata={
                "baseline_model": baseline_model,
                "threshold": threshold,
                "method": "null_rate",
            },
        )


class DriftDistinctCountHandler(RuleHandler):
    """Generate SQL to detect drift via distinct value count comparison.

    Rule format:
        {
            "type": "drift_distinct_count",
            "column": "category",
            "baseline_model": "{{ ref('baseline') }}",
            "threshold": 0.15
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["drift_distinct_count", "drift_cardinality"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        baseline_model = rule.get("baseline_model")
        threshold = rule.get("threshold", 0.15)

        if not baseline_model:
            raise InvalidRuleError(
                "drift_distinct_count", "baseline_model is required"
            )

        col = adapter.quote_column(column)
        where_clause = (
            f"abs(count(distinct {col})::float - "
            f"(select count(distinct {col})::float from {baseline_model})) / "
            f"nullif((select count(distinct {col})::float from {baseline_model}), 0) "
            f"> {threshold}"
        )

        return RuleSQL(
            where_clause=where_clause,
            rule_type="drift_distinct_count",
            column=column,
            select_clause=(
                f"'drift_distinct_count' as rule_type, "
                f"'{column}' as column_name, "
                f"count(distinct {col}) as current_value, "
                f"(select count(distinct {col}) from {baseline_model}) as baseline_value, "
                f"{threshold} as threshold"
            ),
            metadata={
                "baseline_model": baseline_model,
                "threshold": threshold,
                "method": "distinct_count",
            },
        )


class DriftRowCountHandler(RuleHandler):
    """Generate SQL to detect drift via row count comparison.

    Rule format:
        {
            "type": "drift_row_count",
            "baseline_model": "{{ ref('baseline') }}",
            "threshold": 0.1
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["drift_row_count", "drift_volume"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        baseline_model = rule.get("baseline_model")
        threshold = rule.get("threshold", 0.1)

        if not baseline_model:
            raise InvalidRuleError("drift_row_count", "baseline_model is required")

        where_clause = (
            f"abs(count(*)::float - "
            f"(select count(*)::float from {baseline_model})) / "
            f"nullif((select count(*)::float from {baseline_model}), 0) "
            f"> {threshold}"
        )

        return RuleSQL(
            where_clause=where_clause,
            rule_type="drift_row_count",
            column=None,
            select_clause=(
                f"'drift_row_count' as rule_type, "
                f"'*' as column_name, "
                f"count(*) as current_value, "
                f"(select count(*) from {baseline_model}) as baseline_value, "
                f"{threshold} as threshold"
            ),
            metadata={
                "baseline_model": baseline_model,
                "threshold": threshold,
                "method": "row_count",
            },
        )


# =============================================================================
# Anomaly Detection Handlers
# =============================================================================


class AnomalyZScoreHandler(RuleHandler):
    """Generate SQL to detect anomalous rows via z-score.

    Flags rows where the column value deviates from the mean by more
    than the configured number of standard deviations.

    Rule format:
        {
            "type": "anomaly_zscore",
            "column": "amount",
            "threshold": 3.0  # standard deviations
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["anomaly_zscore", "anomaly_z_score", "outlier_zscore"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        threshold = rule.get("threshold", 3.0)

        col = adapter.quote_column(column)
        where_clause = (
            f"abs({col} - avg({col}) over()) / "
            f"nullif(stddev({col}) over(), 0) > {threshold}"
        )

        return RuleSQL(
            where_clause=where_clause,
            rule_type="anomaly_zscore",
            column=column,
            select_clause=(
                f"*, "
                f"abs({col} - avg({col}) over()) / "
                f"nullif(stddev({col}) over(), 0) as z_score"
            ),
            metadata={
                "threshold": threshold,
                "method": "zscore",
            },
        )


class AnomalyIQRHandler(RuleHandler):
    """Generate SQL to detect anomalous rows via IQR method.

    Flags rows where the column value falls outside Q1 - factor*IQR
    or Q3 + factor*IQR.

    Rule format:
        {
            "type": "anomaly_iqr",
            "column": "price",
            "factor": 1.5  # IQR multiplier
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["anomaly_iqr", "outlier_iqr"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        factor = rule.get("factor", 1.5)

        col = adapter.quote_column(column)
        q1 = f"percentile_cont(0.25) within group (order by {col}) over()"
        q3 = f"percentile_cont(0.75) within group (order by {col}) over()"
        iqr = f"({q3} - {q1})"

        where_clause = (
            f"{col} < ({q1} - {factor} * {iqr}) "
            f"or {col} > ({q3} + {factor} * {iqr})"
        )

        return RuleSQL(
            where_clause=where_clause,
            rule_type="anomaly_iqr",
            column=column,
            metadata={
                "factor": factor,
                "method": "iqr",
            },
        )


class AnomalyRangeHandler(RuleHandler):
    """Generate SQL to detect anomalous rows via static range deviation.

    Flags rows where the column value falls outside expected_min or
    expected_max (learned from historical data).

    Rule format:
        {
            "type": "anomaly_range",
            "column": "temperature",
            "expected_min": -10.0,
            "expected_max": 50.0
        }
    """

    @property
    def rule_types(self) -> list[str]:
        return ["anomaly_range", "outlier_range"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        expected_min = rule.get("expected_min")
        expected_max = rule.get("expected_max")

        if expected_min is None and expected_max is None:
            raise InvalidRuleError(
                "anomaly_range",
                "at least one of expected_min or expected_max is required",
            )

        col = adapter.quote_column(column)
        conditions = []
        if expected_min is not None:
            conditions.append(f"{col} < {expected_min}")
        if expected_max is not None:
            conditions.append(f"{col} > {expected_max}")

        where_clause = " or ".join(conditions)

        return RuleSQL(
            where_clause=where_clause,
            rule_type="anomaly_range",
            column=column,
            metadata={
                "expected_min": expected_min,
                "expected_max": expected_max,
                "method": "range",
            },
        )


# =============================================================================
# Handler Registration
# =============================================================================

DRIFT_HANDLERS: tuple[type[RuleHandler], ...] = (
    DriftMeanHandler,
    DriftStddevHandler,
    DriftNullRateHandler,
    DriftDistinctCountHandler,
    DriftRowCountHandler,
)

ANOMALY_HANDLERS: tuple[type[RuleHandler], ...] = (
    AnomalyZScoreHandler,
    AnomalyIQRHandler,
    AnomalyRangeHandler,
)

ALL_HANDLERS: tuple[type[RuleHandler], ...] = DRIFT_HANDLERS + ANOMALY_HANDLERS


def register_drift_anomaly_handlers() -> None:
    """Register all drift and anomaly handlers with the global registry."""
    from truthound_dbt.converters.rules import get_handler_registry

    registry = get_handler_registry()
    for handler_cls in ALL_HANDLERS:
        registry.register(handler_cls())


def get_drift_handler(method: str) -> RuleHandler:
    """Get a drift handler by method name.

    Args:
        method: Drift detection method
            (mean, stddev, null_rate, distinct_count, row_count).

    Returns:
        Appropriate RuleHandler.

    Raises:
        ValueError: If method not recognized.
    """
    method_map: dict[str, type[RuleHandler]] = {
        "mean": DriftMeanHandler,
        "stddev": DriftStddevHandler,
        "null_rate": DriftNullRateHandler,
        "distinct_count": DriftDistinctCountHandler,
        "row_count": DriftRowCountHandler,
    }
    handler_cls = method_map.get(method)
    if handler_cls is None:
        raise ValueError(
            f"Unknown drift method: {method}. "
            f"Available: {', '.join(method_map.keys())}"
        )
    return handler_cls()


def get_anomaly_handler(method: str) -> RuleHandler:
    """Get an anomaly handler by method name.

    Args:
        method: Anomaly detection method (zscore, iqr, range).

    Returns:
        Appropriate RuleHandler.

    Raises:
        ValueError: If method not recognized.
    """
    method_map: dict[str, type[RuleHandler]] = {
        "zscore": AnomalyZScoreHandler,
        "iqr": AnomalyIQRHandler,
        "range": AnomalyRangeHandler,
    }
    handler_cls = method_map.get(method)
    if handler_cls is None:
        raise ValueError(
            f"Unknown anomaly method: {method}. "
            f"Available: {', '.join(method_map.keys())}"
        )
    return handler_cls()
