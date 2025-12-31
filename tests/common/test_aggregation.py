"""Tests for Result Aggregation module.

This module tests the comprehensive result aggregation system including:
- AggregationConfig and builder pattern
- Various aggregation strategies (MERGE, WORST, BEST, MAJORITY, etc.)
- CheckResult, ProfileResult, LearnResult aggregators
- Multi-engine aggregation
- Weighted aggregation
- Comparison functionality
- Hooks and registry
"""

from __future__ import annotations

import pytest
from typing import Any

from common.base import (
    CheckResult,
    CheckStatus,
    ColumnProfile,
    LearnedRule,
    LearnResult,
    LearnStatus,
    ProfileResult,
    ProfileStatus,
    Severity,
    ValidationFailure,
)
from common.engines.aggregation import (
    # Enums
    ResultAggregationStrategy,
    ConflictResolution,
    StatusPriority,
    # Configuration
    AggregationConfig,
    DEFAULT_AGGREGATION_CONFIG,
    STRICT_AGGREGATION_CONFIG,
    LENIENT_AGGREGATION_CONFIG,
    CONSENSUS_AGGREGATION_CONFIG,
    WORST_CASE_AGGREGATION_CONFIG,
    # Result Types
    EngineResultEntry,
    AggregatedResult,
    ComparisonResult,
    # Aggregators
    CheckResultMergeAggregator,
    CheckResultWeightedAggregator,
    ProfileResultAggregator,
    LearnResultAggregator,
    MultiEngineAggregator,
    # Hooks
    BaseAggregationHook,
    LoggingAggregationHook,
    MetricsAggregationHook,
    CompositeAggregationHook,
    # Registry
    AggregatorRegistry,
    get_aggregator_registry,
    get_aggregator,
    register_aggregator,
    list_aggregators,
    # Convenience Functions
    aggregate_check_results,
    aggregate_profile_results,
    aggregate_learn_results,
    compare_check_results,
    create_multi_engine_aggregator,
    # Exceptions
    AggregationError,
    NoResultsError,
    ConflictError,
    AggregatorNotFoundError,
    InvalidWeightError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def passed_check_result() -> CheckResult:
    """Create a passed CheckResult."""
    return CheckResult(
        status=CheckStatus.PASSED,
        passed_count=10,
        failed_count=0,
        warning_count=0,
        execution_time_ms=100.0,
        metadata={"engine": "test1"},
    )


@pytest.fixture
def failed_check_result() -> CheckResult:
    """Create a failed CheckResult."""
    return CheckResult(
        status=CheckStatus.FAILED,
        passed_count=5,
        failed_count=5,
        warning_count=0,
        failures=(
            ValidationFailure(
                rule_name="not_null",
                column="id",
                message="Column 'id' contains null values",
                severity=Severity.ERROR,
                failed_count=5,
            ),
        ),
        execution_time_ms=150.0,
        metadata={"engine": "test2"},
    )


@pytest.fixture
def warning_check_result() -> CheckResult:
    """Create a warning CheckResult."""
    return CheckResult(
        status=CheckStatus.WARNING,
        passed_count=8,
        failed_count=0,
        warning_count=2,
        execution_time_ms=120.0,
        metadata={"engine": "test3"},
    )


@pytest.fixture
def profile_result1() -> ProfileResult:
    """Create a ProfileResult."""
    return ProfileResult(
        status=ProfileStatus.COMPLETED,
        row_count=1000,
        column_count=5,
        columns=(
            ColumnProfile(
                column_name="id",
                dtype="int64",
                null_count=0,
                unique_count=1000,
            ),
            ColumnProfile(
                column_name="name",
                dtype="string",
                null_count=10,
                unique_count=990,
            ),
        ),
        execution_time_ms=200.0,
    )


@pytest.fixture
def profile_result2() -> ProfileResult:
    """Create another ProfileResult."""
    return ProfileResult(
        status=ProfileStatus.COMPLETED,
        row_count=500,
        column_count=3,
        columns=(
            ColumnProfile(
                column_name="age",
                dtype="int64",
                null_count=5,
                mean=35.5,
                min_value=18,
                max_value=80,
            ),
        ),
        execution_time_ms=100.0,
    )


@pytest.fixture
def learn_result1() -> LearnResult:
    """Create a LearnResult."""
    return LearnResult(
        status=LearnStatus.COMPLETED,
        rules=(
            LearnedRule(
                rule_type="not_null",
                column="id",
                confidence=0.95,
                sample_size=1000,
            ),
            LearnedRule(
                rule_type="unique",
                column="email",
                confidence=0.99,
                sample_size=1000,
            ),
        ),
        columns_analyzed=5,
        execution_time_ms=300.0,
    )


@pytest.fixture
def learn_result2() -> LearnResult:
    """Create another LearnResult."""
    return LearnResult(
        status=LearnStatus.COMPLETED,
        rules=(
            LearnedRule(
                rule_type="not_null",
                column="id",
                confidence=0.90,
                sample_size=500,
            ),
            LearnedRule(
                rule_type="in_range",
                column="age",
                parameters={"min": 0, "max": 150},
                confidence=0.85,
                sample_size=500,
            ),
        ),
        columns_analyzed=3,
        execution_time_ms=150.0,
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestAggregationConfig:
    """Tests for AggregationConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AggregationConfig()
        assert config.strategy == ResultAggregationStrategy.MERGE
        assert config.conflict_resolution == ConflictResolution.PREFER_FAILURE
        assert config.consensus_threshold == 0.5
        assert config.include_metadata is True
        assert config.preserve_individual_results is False

    def test_with_strategy(self) -> None:
        """Test with_strategy builder method."""
        config = AggregationConfig()
        new_config = config.with_strategy(ResultAggregationStrategy.WORST)
        assert new_config.strategy == ResultAggregationStrategy.WORST
        assert config.strategy == ResultAggregationStrategy.MERGE  # Original unchanged

    def test_with_weights(self) -> None:
        """Test with_weights builder method."""
        weights = {"engine1": 2.0, "engine2": 1.0}
        config = AggregationConfig(
            strategy=ResultAggregationStrategy.WEIGHTED,
            weights=weights,
        )
        new_weights = {"engine1": 3.0, "engine2": 1.0}
        new_config = config.with_weights(new_weights)
        assert new_config.weights == new_weights
        assert config.weights == weights  # Original unchanged

    def test_with_conflict_resolution(self) -> None:
        """Test with_conflict_resolution builder method."""
        config = AggregationConfig()
        new_config = config.with_conflict_resolution(ConflictResolution.PREFER_SUCCESS)
        assert new_config.conflict_resolution == ConflictResolution.PREFER_SUCCESS

    def test_with_consensus_threshold(self) -> None:
        """Test with_consensus_threshold builder method."""
        config = AggregationConfig()
        new_config = config.with_consensus_threshold(0.75)
        assert new_config.consensus_threshold == 0.75

    def test_invalid_consensus_threshold(self) -> None:
        """Test that invalid consensus threshold raises error."""
        with pytest.raises(ValueError, match="consensus_threshold"):
            AggregationConfig(consensus_threshold=1.5)

    def test_weighted_strategy_requires_weights(self) -> None:
        """Test that WEIGHTED strategy requires weights."""
        with pytest.raises(ValueError, match="weights must be provided"):
            AggregationConfig(strategy=ResultAggregationStrategy.WEIGHTED)

    def test_preset_configs(self) -> None:
        """Test preset configurations."""
        assert DEFAULT_AGGREGATION_CONFIG.strategy == ResultAggregationStrategy.MERGE
        assert STRICT_AGGREGATION_CONFIG.strategy == ResultAggregationStrategy.STRICT_ALL
        assert LENIENT_AGGREGATION_CONFIG.strategy == ResultAggregationStrategy.LENIENT_ANY
        assert CONSENSUS_AGGREGATION_CONFIG.strategy == ResultAggregationStrategy.CONSENSUS
        assert WORST_CASE_AGGREGATION_CONFIG.strategy == ResultAggregationStrategy.WORST


# =============================================================================
# CheckResult Aggregation Tests
# =============================================================================


class TestCheckResultMergeAggregator:
    """Tests for CheckResultMergeAggregator."""

    def test_merge_passed_results(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test merging multiple passed results."""
        aggregator = CheckResultMergeAggregator()
        results = [passed_check_result, passed_check_result]
        config = DEFAULT_AGGREGATION_CONFIG

        merged = aggregator.aggregate(results, config)

        assert merged.status == CheckStatus.PASSED
        assert merged.passed_count == 20  # 10 + 10
        assert merged.failed_count == 0
        assert merged.execution_time_ms == 200.0  # 100 + 100

    def test_merge_mixed_results(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test merging passed and failed results."""
        aggregator = CheckResultMergeAggregator()
        results = [passed_check_result, failed_check_result]
        config = DEFAULT_AGGREGATION_CONFIG

        merged = aggregator.aggregate(results, config)

        assert merged.status == CheckStatus.FAILED
        assert merged.passed_count == 15  # 10 + 5
        assert merged.failed_count == 5
        assert len(merged.failures) == 1

    def test_worst_strategy(
        self,
        passed_check_result: CheckResult,
        warning_check_result: CheckResult,
    ) -> None:
        """Test WORST aggregation strategy."""
        aggregator = CheckResultMergeAggregator()
        results = [passed_check_result, warning_check_result]
        config = AggregationConfig(strategy=ResultAggregationStrategy.WORST)

        result = aggregator.aggregate(results, config)

        assert result.status == CheckStatus.WARNING

    def test_best_strategy(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test BEST aggregation strategy."""
        aggregator = CheckResultMergeAggregator()
        results = [passed_check_result, failed_check_result]
        config = AggregationConfig(strategy=ResultAggregationStrategy.BEST)

        result = aggregator.aggregate(results, config)

        assert result.status == CheckStatus.PASSED

    def test_majority_strategy(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test MAJORITY aggregation strategy."""
        aggregator = CheckResultMergeAggregator()
        # 2 passed vs 1 failed
        results = [passed_check_result, passed_check_result, failed_check_result]
        config = AggregationConfig(strategy=ResultAggregationStrategy.MAJORITY)

        result = aggregator.aggregate(results, config)

        assert result.status == CheckStatus.PASSED

    def test_first_failure_strategy(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test FIRST_FAILURE aggregation strategy."""
        aggregator = CheckResultMergeAggregator()
        results = [passed_check_result, failed_check_result, passed_check_result]
        config = AggregationConfig(strategy=ResultAggregationStrategy.FIRST_FAILURE)

        result = aggregator.aggregate(results, config)

        assert result.status == CheckStatus.FAILED

    def test_strict_all_strategy(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test STRICT_ALL aggregation strategy."""
        aggregator = CheckResultMergeAggregator()
        results = [passed_check_result, failed_check_result]
        config = AggregationConfig(strategy=ResultAggregationStrategy.STRICT_ALL)

        result = aggregator.aggregate(results, config)

        assert result.status == CheckStatus.FAILED

    def test_lenient_any_strategy(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test LENIENT_ANY aggregation strategy."""
        aggregator = CheckResultMergeAggregator()
        results = [failed_check_result, passed_check_result]
        config = AggregationConfig(strategy=ResultAggregationStrategy.LENIENT_ANY)

        result = aggregator.aggregate(results, config)

        assert result.status == CheckStatus.PASSED

    def test_empty_results_raises_error(self) -> None:
        """Test that empty results raises NoResultsError."""
        aggregator = CheckResultMergeAggregator()
        config = DEFAULT_AGGREGATION_CONFIG

        with pytest.raises(NoResultsError):
            aggregator.aggregate([], config)

    def test_deduplicate_failures(
        self,
        failed_check_result: CheckResult,
    ) -> None:
        """Test failure deduplication."""
        aggregator = CheckResultMergeAggregator()
        # Same failure twice
        results = [failed_check_result, failed_check_result]
        config = AggregationConfig(deduplicate_failures=True)

        merged = aggregator.aggregate(results, config)

        assert len(merged.failures) == 1  # Deduplicated

    def test_no_deduplicate_failures(
        self,
        failed_check_result: CheckResult,
    ) -> None:
        """Test without failure deduplication."""
        aggregator = CheckResultMergeAggregator()
        results = [failed_check_result, failed_check_result]
        config = AggregationConfig(deduplicate_failures=False)

        merged = aggregator.aggregate(results, config)

        assert len(merged.failures) == 2  # Not deduplicated


class TestCheckResultWeightedAggregator:
    """Tests for CheckResultWeightedAggregator."""

    def test_weighted_aggregation(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test weighted aggregation."""
        aggregator = CheckResultWeightedAggregator()

        engine_results = {
            "engine1": passed_check_result,
            "engine2": failed_check_result,
        }
        config = AggregationConfig(
            strategy=ResultAggregationStrategy.WEIGHTED,
            weights={"engine1": 3.0, "engine2": 1.0},  # Weight passed higher
        )

        result = aggregator.aggregate_with_weights(engine_results, config)

        # With weights 3:1 for passed:failed, normalized score is 3.0
        # Score 3.0 is between 1.5 and 3.5, so WARNING is expected
        assert "weighted_score" in result.metadata
        assert result.metadata["weighted_score"] == 3.0  # (4.0*3 + 0.0*1) / 4 = 3.0
        assert result.metadata["total_weight"] == 4.0

    def test_weighted_with_equal_weights(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test weighted aggregation with equal weights."""
        aggregator = CheckResultWeightedAggregator()

        engine_results = {
            "engine1": passed_check_result,
            "engine2": failed_check_result,
        }
        config = AggregationConfig(
            strategy=ResultAggregationStrategy.WEIGHTED,
            weights={"engine1": 1.0, "engine2": 1.0},
        )

        result = aggregator.aggregate_with_weights(engine_results, config)

        # Equal weights, should result in WARNING (middle ground)
        assert result.metadata["total_weight"] == 2.0


# =============================================================================
# ProfileResult Aggregation Tests
# =============================================================================


class TestProfileResultAggregator:
    """Tests for ProfileResultAggregator."""

    def test_merge_profile_results(
        self,
        profile_result1: ProfileResult,
        profile_result2: ProfileResult,
    ) -> None:
        """Test merging ProfileResults."""
        aggregator = ProfileResultAggregator()
        results = [profile_result1, profile_result2]
        config = DEFAULT_AGGREGATION_CONFIG

        merged = aggregator.aggregate(results, config)

        assert merged.status == ProfileStatus.COMPLETED
        assert merged.row_count == 1500  # 1000 + 500
        # Should have columns from both: id, name, age
        assert merged.column_count == 3
        column_names = {c.column_name for c in merged.columns}
        assert "id" in column_names
        assert "name" in column_names
        assert "age" in column_names

    def test_merge_failed_profile(
        self,
        profile_result1: ProfileResult,
    ) -> None:
        """Test merging with failed profile."""
        failed = ProfileResult(
            status=ProfileStatus.FAILED,
            row_count=0,
            column_count=0,
        )
        aggregator = ProfileResultAggregator()
        results = [profile_result1, failed]
        config = DEFAULT_AGGREGATION_CONFIG

        merged = aggregator.aggregate(results, config)

        assert merged.status == ProfileStatus.FAILED

    def test_worst_strategy_profile(
        self,
        profile_result1: ProfileResult,
    ) -> None:
        """Test WORST strategy for ProfileResult."""
        partial = ProfileResult(
            status=ProfileStatus.PARTIAL,
            row_count=100,
            column_count=1,
        )
        aggregator = ProfileResultAggregator()
        results = [profile_result1, partial]
        config = AggregationConfig(strategy=ResultAggregationStrategy.WORST)

        result = aggregator.aggregate(results, config)

        assert result.status == ProfileStatus.PARTIAL


# =============================================================================
# LearnResult Aggregation Tests
# =============================================================================


class TestLearnResultAggregator:
    """Tests for LearnResultAggregator."""

    def test_merge_learn_results(
        self,
        learn_result1: LearnResult,
        learn_result2: LearnResult,
    ) -> None:
        """Test merging LearnResults."""
        aggregator = LearnResultAggregator()
        results = [learn_result1, learn_result2]
        config = DEFAULT_AGGREGATION_CONFIG

        merged = aggregator.aggregate(results, config)

        assert merged.status == LearnStatus.COMPLETED
        # Should have merged rules: not_null (merged), unique, in_range
        assert len(merged.rules) == 3
        rule_types = {r.rule_type for r in merged.rules}
        assert "not_null" in rule_types
        assert "unique" in rule_types
        assert "in_range" in rule_types

    def test_merge_same_rule_confidence(
        self,
        learn_result1: LearnResult,
        learn_result2: LearnResult,
    ) -> None:
        """Test that same rule type/column merges confidence."""
        aggregator = LearnResultAggregator()
        results = [learn_result1, learn_result2]
        config = DEFAULT_AGGREGATION_CONFIG

        merged = aggregator.aggregate(results, config)

        # Find the merged not_null rule for id
        not_null_rule = next(
            (r for r in merged.rules if r.rule_type == "not_null" and r.column == "id"),
            None,
        )
        assert not_null_rule is not None
        # Weighted average: (0.95 * 1000 + 0.90 * 500) / 1500
        expected_confidence = (0.95 * 1000 + 0.90 * 500) / 1500
        assert abs(not_null_rule.confidence - expected_confidence) < 0.001
        assert not_null_rule.sample_size == 1500


# =============================================================================
# Multi-Engine Aggregation Tests
# =============================================================================


class TestMultiEngineAggregator:
    """Tests for MultiEngineAggregator."""

    def test_aggregate_check_results(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test aggregating CheckResults from multiple engines."""
        aggregator = MultiEngineAggregator()

        engine_results = {
            "truthound": passed_check_result,
            "ge": failed_check_result,
        }

        aggregated = aggregator.aggregate_check_results(engine_results)

        assert aggregated.source_count == 2
        assert set(aggregated.source_engines) == {"truthound", "ge"}
        assert aggregated.result.status == CheckStatus.FAILED
        assert aggregated.aggregation_time_ms > 0

    def test_aggregate_profile_results(
        self,
        profile_result1: ProfileResult,
        profile_result2: ProfileResult,
    ) -> None:
        """Test aggregating ProfileResults from multiple engines."""
        aggregator = MultiEngineAggregator()

        engine_results = {
            "truthound": profile_result1,
            "pandera": profile_result2,
        }

        aggregated = aggregator.aggregate_profile_results(engine_results)

        assert aggregated.source_count == 2
        assert aggregated.result.status == ProfileStatus.COMPLETED

    def test_aggregate_learn_results(
        self,
        learn_result1: LearnResult,
        learn_result2: LearnResult,
    ) -> None:
        """Test aggregating LearnResults from multiple engines."""
        aggregator = MultiEngineAggregator()

        engine_results = {
            "truthound": learn_result1,
            "ge": learn_result2,
        }

        aggregated = aggregator.aggregate_learn_results(engine_results)

        assert aggregated.source_count == 2
        assert aggregated.result.status == LearnStatus.COMPLETED

    def test_with_config(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test creating aggregator with different config."""
        config = STRICT_AGGREGATION_CONFIG
        aggregator = MultiEngineAggregator(config=config)
        new_aggregator = aggregator.with_config(LENIENT_AGGREGATION_CONFIG)

        assert new_aggregator.config.strategy == ResultAggregationStrategy.LENIENT_ANY

    def test_preserve_individual_results(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test preserving individual results."""
        config = AggregationConfig(preserve_individual_results=True)
        aggregator = MultiEngineAggregator(config=config)

        engine_results = {
            "engine1": passed_check_result,
            "engine2": failed_check_result,
        }

        aggregated = aggregator.aggregate_check_results(engine_results)

        assert aggregated.individual_results is not None
        assert "engine1" in aggregated.individual_results
        assert "engine2" in aggregated.individual_results


class TestCompareCheckResults:
    """Tests for result comparison."""

    def test_unanimous_agreement(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test comparison when all engines agree."""
        aggregator = MultiEngineAggregator()

        engine_results = {
            "engine1": passed_check_result,
            "engine2": passed_check_result,
        }

        comparison = aggregator.compare_check_results(engine_results)

        assert comparison.unanimous is True
        assert comparison.agreement_ratio == 1.0
        assert comparison.majority_status == "PASSED"
        assert comparison.has_discrepancies is False

    def test_disagreement(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test comparison when engines disagree."""
        aggregator = MultiEngineAggregator()

        engine_results = {
            "engine1": passed_check_result,
            "engine2": failed_check_result,
        }

        comparison = aggregator.compare_check_results(engine_results)

        assert comparison.unanimous is False
        assert comparison.agreement_ratio == 0.5
        assert comparison.has_discrepancies is True
        assert len(comparison.discrepancies) == 1

    def test_majority_with_discrepancy(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test comparison with majority agreement."""
        aggregator = MultiEngineAggregator()

        engine_results = {
            "engine1": passed_check_result,
            "engine2": passed_check_result,
            "engine3": failed_check_result,
        }

        comparison = aggregator.compare_check_results(engine_results)

        assert comparison.unanimous is False
        assert comparison.agreement_ratio == pytest.approx(2 / 3)
        assert comparison.majority_status == "PASSED"
        assert len(comparison.discrepancies) == 1
        assert comparison.discrepancies[0]["engine"] == "engine3"


# =============================================================================
# Hooks Tests
# =============================================================================


class TestAggregationHooks:
    """Tests for aggregation hooks."""

    def test_metrics_hook(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test MetricsAggregationHook."""
        hook = MetricsAggregationHook()
        aggregator = CheckResultMergeAggregator(hooks=[hook])

        results = [passed_check_result, passed_check_result]
        aggregator.aggregate(results, DEFAULT_AGGREGATION_CONFIG)

        stats = hook.get_stats()
        assert stats["aggregation_count"] == 1
        assert stats["total_results_aggregated"] == 2
        assert stats["average_duration_ms"] > 0

    def test_metrics_hook_reset(self) -> None:
        """Test MetricsAggregationHook reset."""
        hook = MetricsAggregationHook()
        hook.on_aggregation_complete(None, 5, 0, 100.0)

        assert hook.aggregation_count == 1

        hook.reset()

        assert hook.aggregation_count == 0

    def test_composite_hook(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test CompositeAggregationHook."""
        metrics_hook = MetricsAggregationHook()
        composite = CompositeAggregationHook([metrics_hook])
        aggregator = CheckResultMergeAggregator(hooks=[composite])

        results = [passed_check_result]
        aggregator.aggregate(results, DEFAULT_AGGREGATION_CONFIG)

        assert metrics_hook.aggregation_count == 1

    def test_hook_error_isolation(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test that hook errors don't break aggregation."""

        class BrokenHook(BaseAggregationHook):
            def on_aggregation_start(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Hook error!")

        aggregator = CheckResultMergeAggregator(hooks=[BrokenHook()])
        results = [passed_check_result]

        # Should not raise despite broken hook
        result = aggregator.aggregate(results, DEFAULT_AGGREGATION_CONFIG)
        assert result.status == CheckStatus.PASSED


# =============================================================================
# Registry Tests
# =============================================================================


class TestAggregatorRegistry:
    """Tests for AggregatorRegistry."""

    def test_default_aggregators(self) -> None:
        """Test that default aggregators are registered."""
        registry = AggregatorRegistry()

        assert registry.has("check")
        assert registry.has("check_weighted")
        assert registry.has("profile")
        assert registry.has("learn")

    def test_register_and_get(self) -> None:
        """Test registering and getting aggregator."""
        registry = AggregatorRegistry()

        class CustomAggregator(CheckResultMergeAggregator):
            pass

        custom = CustomAggregator()
        registry.register("custom", custom)

        retrieved = registry.get("custom")
        assert retrieved is custom

    def test_register_duplicate_raises(self) -> None:
        """Test that duplicate registration raises error."""
        registry = AggregatorRegistry()

        with pytest.raises(ValueError, match="already registered"):
            registry.register("check", CheckResultMergeAggregator())

    def test_register_with_override(self) -> None:
        """Test registration with override."""
        registry = AggregatorRegistry()

        new_aggregator = CheckResultMergeAggregator()
        registry.register("check", new_aggregator, allow_override=True)

        assert registry.get("check") is new_aggregator

    def test_unregister(self) -> None:
        """Test unregistering aggregator."""
        registry = AggregatorRegistry()
        registry.register("temp", CheckResultMergeAggregator())

        removed = registry.unregister("temp")
        assert removed is not None
        assert not registry.has("temp")

    def test_get_not_found(self) -> None:
        """Test getting non-existent aggregator."""
        registry = AggregatorRegistry()

        with pytest.raises(AggregatorNotFoundError):
            registry.get("nonexistent")

    def test_list_aggregators(self) -> None:
        """Test listing aggregators."""
        registry = AggregatorRegistry()

        names = registry.list()

        assert "check" in names
        assert "profile" in names
        assert "learn" in names

    def test_clear_and_re_register_defaults(self) -> None:
        """Test clearing and re-registering defaults."""
        registry = AggregatorRegistry()
        registry.register("custom", CheckResultMergeAggregator())

        registry.clear()

        assert registry.has("check")  # Defaults re-registered
        assert not registry.has("custom")  # Custom removed


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_aggregator_registry(self) -> None:
        """Test getting global registry."""
        registry = get_aggregator_registry()
        assert isinstance(registry, AggregatorRegistry)

    def test_get_aggregator(self) -> None:
        """Test getting aggregator from global registry."""
        aggregator = get_aggregator("check")
        assert isinstance(aggregator, CheckResultMergeAggregator)

    def test_list_aggregators(self) -> None:
        """Test listing aggregators from global registry."""
        names = list_aggregators()
        assert "check" in names


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_aggregate_check_results_function(
        self,
        passed_check_result: CheckResult,
        failed_check_result: CheckResult,
    ) -> None:
        """Test aggregate_check_results convenience function."""
        engine_results = {
            "engine1": passed_check_result,
            "engine2": failed_check_result,
        }

        aggregated = aggregate_check_results(engine_results)

        assert aggregated.source_count == 2
        assert aggregated.result.status == CheckStatus.FAILED

    def test_aggregate_profile_results_function(
        self,
        profile_result1: ProfileResult,
    ) -> None:
        """Test aggregate_profile_results convenience function."""
        engine_results = {"engine1": profile_result1}

        aggregated = aggregate_profile_results(engine_results)

        assert aggregated.source_count == 1

    def test_aggregate_learn_results_function(
        self,
        learn_result1: LearnResult,
    ) -> None:
        """Test aggregate_learn_results convenience function."""
        engine_results = {"engine1": learn_result1}

        aggregated = aggregate_learn_results(engine_results)

        assert aggregated.source_count == 1

    def test_compare_check_results_function(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test compare_check_results convenience function."""
        engine_results = {
            "engine1": passed_check_result,
            "engine2": passed_check_result,
        }

        comparison = compare_check_results(engine_results)

        assert comparison.unanimous is True

    def test_create_multi_engine_aggregator_function(self) -> None:
        """Test create_multi_engine_aggregator convenience function."""
        aggregator = create_multi_engine_aggregator(
            strategy=ResultAggregationStrategy.WORST,
        )

        assert aggregator.config.strategy == ResultAggregationStrategy.WORST


# =============================================================================
# Result Type Tests
# =============================================================================


class TestAggregatedResult:
    """Tests for AggregatedResult."""

    def test_to_dict(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test AggregatedResult serialization."""
        aggregated = AggregatedResult(
            result=passed_check_result,
            strategy=ResultAggregationStrategy.MERGE,
            source_engines=("engine1", "engine2"),
            source_count=2,
            conflict_count=0,
            aggregation_time_ms=10.0,
        )

        data = aggregated.to_dict()

        assert data["strategy"] == "MERGE"
        assert data["source_count"] == 2
        assert data["source_engines"] == ["engine1", "engine2"]

    def test_has_conflicts(self) -> None:
        """Test has_conflicts property."""
        with_conflicts = AggregatedResult(
            result=None,
            strategy=ResultAggregationStrategy.MERGE,
            source_engines=(),
            source_count=0,
            conflict_count=1,
        )
        without_conflicts = AggregatedResult(
            result=None,
            strategy=ResultAggregationStrategy.MERGE,
            source_engines=(),
            source_count=0,
            conflict_count=0,
        )

        assert with_conflicts.has_conflicts is True
        assert without_conflicts.has_conflicts is False


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_to_dict(self) -> None:
        """Test ComparisonResult serialization."""
        comparison = ComparisonResult(
            agreement_ratio=0.67,
            unanimous=False,
            majority_status="PASSED",
            engine_statuses={"engine1": "PASSED", "engine2": "FAILED"},
            discrepancies=({"engine": "engine2", "status": "FAILED"},),
        )

        data = comparison.to_dict()

        assert data["agreement_ratio"] == 0.67
        assert data["unanimous"] is False
        assert len(data["discrepancies"]) == 1


class TestEngineResultEntry:
    """Tests for EngineResultEntry."""

    def test_creation(
        self,
        passed_check_result: CheckResult,
    ) -> None:
        """Test EngineResultEntry creation."""
        entry = EngineResultEntry(
            engine_name="truthound",
            result=passed_check_result,
            weight=2.0,
        )

        assert entry.engine_name == "truthound"
        assert entry.weight == 2.0
        assert entry.result is passed_check_result


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for aggregation exceptions."""

    def test_no_results_error(self) -> None:
        """Test NoResultsError."""
        error = NoResultsError()
        assert "No results provided" in str(error)

    def test_conflict_error(self) -> None:
        """Test ConflictError."""
        conflicts = {"engine1": "PASSED", "engine2": "FAILED"}
        error = ConflictError("Conflict detected", conflicts)

        assert "Conflict detected" in str(error)
        assert error.conflicts == conflicts

    def test_aggregator_not_found_error(self) -> None:
        """Test AggregatorNotFoundError."""
        error = AggregatorNotFoundError("custom")

        assert "custom" in str(error)
        assert error.aggregator_name == "custom"

    def test_invalid_weight_error(self) -> None:
        """Test InvalidWeightError."""
        weights = {"engine1": -1.0}
        error = InvalidWeightError("Negative weight", weights)

        assert "Negative weight" in str(error)
        assert error.weights == weights
