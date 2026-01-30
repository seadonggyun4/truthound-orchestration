"""Tests for TASK 2-2: Rule Validation schema extensions (drift, anomaly, extended rules)."""

from __future__ import annotations

import pytest

from common.rule_validation import (
    COMMON_RULE_SCHEMAS,
    RuleCategory,
    RuleRegistry,
    TruthoundRuleValidator,
    get_rule_registry,
    validate_rule,
    validate_rules,
)


# =============================================================================
# RuleCategory Enum Extensions
# =============================================================================


class TestRuleCategoryExtensions:
    def test_drift_category_exists(self) -> None:
        assert hasattr(RuleCategory, "DRIFT")
        assert RuleCategory.DRIFT is not None

    def test_anomaly_category_exists(self) -> None:
        assert hasattr(RuleCategory, "ANOMALY")
        assert RuleCategory.ANOMALY is not None

    def test_existing_categories_preserved(self) -> None:
        for name in ("COMPLETENESS", "UNIQUENESS", "VALIDITY", "CONSISTENCY", "ACCURACY", "TIMELINESS"):
            assert hasattr(RuleCategory, name)


# =============================================================================
# New Rule Schemas in COMMON_RULE_SCHEMAS
# =============================================================================


NEW_DRIFT_RULES = ("statistical_drift", "distribution_change")
NEW_ANOMALY_RULES = ("outlier", "z_score_outlier")
NEW_EXTENDED_RULES = (
    "completeness_ratio",
    "referential_integrity",
    "cross_table_row_count",
    "conditional_null",
    "expression",
    "distribution",
)
ALL_NEW_RULES = NEW_DRIFT_RULES + NEW_ANOMALY_RULES + NEW_EXTENDED_RULES


class TestNewRuleSchemasExist:
    @pytest.mark.parametrize("rule_type", ALL_NEW_RULES)
    def test_schema_registered(self, rule_type: str) -> None:
        assert rule_type in COMMON_RULE_SCHEMAS, f"{rule_type} not in COMMON_RULE_SCHEMAS"

    @pytest.mark.parametrize("rule_type", NEW_DRIFT_RULES)
    def test_drift_category(self, rule_type: str) -> None:
        assert COMMON_RULE_SCHEMAS[rule_type].category == RuleCategory.DRIFT

    @pytest.mark.parametrize("rule_type", NEW_ANOMALY_RULES)
    def test_anomaly_category(self, rule_type: str) -> None:
        assert COMMON_RULE_SCHEMAS[rule_type].category == RuleCategory.ANOMALY


class TestExistingRulesPreserved:
    @pytest.mark.parametrize(
        "rule_type",
        ["not_null", "unique", "in_set", "in_range", "regex", "dtype",
         "min_length", "max_length", "greater_than", "less_than", "column_exists"],
    )
    def test_existing_rule_still_present(self, rule_type: str) -> None:
        assert rule_type in COMMON_RULE_SCHEMAS


# =============================================================================
# Drift Rule Validation
# =============================================================================


class TestStatisticalDriftValidation:
    def test_valid_minimal(self) -> None:
        result = validate_rule({"type": "statistical_drift", "column": "age"})
        assert result.is_valid

    def test_valid_full(self) -> None:
        result = validate_rule({
            "type": "statistical_drift",
            "column": "age",
            "method": "ks",
            "threshold": 0.05,
        })
        assert result.is_valid

    def test_missing_column(self) -> None:
        result = validate_rule({"type": "statistical_drift"})
        assert not result.is_valid

    def test_alias_drift(self) -> None:
        schema = COMMON_RULE_SCHEMAS["statistical_drift"]
        assert "drift" in schema.aliases or "stat_drift" in schema.aliases


class TestDistributionChangeValidation:
    def test_valid(self) -> None:
        result = validate_rule({
            "type": "distribution_change",
            "column": "income",
            "baseline_profile": {"mean": 50000, "std": 15000},
        })
        assert result.is_valid

    def test_missing_baseline(self) -> None:
        result = validate_rule({
            "type": "distribution_change",
            "column": "income",
        })
        assert not result.is_valid


# =============================================================================
# Anomaly Rule Validation
# =============================================================================


class TestOutlierValidation:
    def test_valid_minimal(self) -> None:
        result = validate_rule({"type": "outlier", "column": "amount"})
        assert result.is_valid

    def test_valid_full(self) -> None:
        result = validate_rule({
            "type": "outlier",
            "column": "amount",
            "detector": "isolation_forest",
            "contamination": 0.1,
        })
        assert result.is_valid

    def test_missing_column(self) -> None:
        result = validate_rule({"type": "outlier"})
        assert not result.is_valid


class TestZScoreOutlierValidation:
    def test_valid_minimal(self) -> None:
        result = validate_rule({"type": "z_score_outlier", "column": "price"})
        assert result.is_valid

    def test_valid_with_threshold(self) -> None:
        result = validate_rule({
            "type": "z_score_outlier",
            "column": "price",
            "threshold": 2.5,
        })
        assert result.is_valid


# =============================================================================
# Extended Rule Validation
# =============================================================================


class TestCompletenessRatioValidation:
    def test_valid(self) -> None:
        result = validate_rule({
            "type": "completeness_ratio",
            "column": "email",
            "min_ratio": 0.95,
        })
        assert result.is_valid

    def test_missing_min_ratio(self) -> None:
        result = validate_rule({
            "type": "completeness_ratio",
            "column": "email",
        })
        assert not result.is_valid


class TestReferentialIntegrityValidation:
    def test_valid(self) -> None:
        result = validate_rule({
            "type": "referential_integrity",
            "column": "user_id",
            "reference_table": "users",
            "reference_column": "id",
        })
        assert result.is_valid

    def test_missing_reference_table(self) -> None:
        result = validate_rule({
            "type": "referential_integrity",
            "column": "user_id",
            "reference_column": "id",
        })
        assert not result.is_valid


class TestCrossTableRowCountValidation:
    def test_valid(self) -> None:
        result = validate_rule({
            "type": "cross_table_row_count",
            "table1": "orders",
            "table2": "order_items",
        })
        assert result.is_valid

    def test_missing_table2(self) -> None:
        result = validate_rule({
            "type": "cross_table_row_count",
            "table1": "orders",
        })
        assert not result.is_valid


class TestConditionalNullValidation:
    def test_valid(self) -> None:
        result = validate_rule({
            "type": "conditional_null",
            "column": "email",
            "condition": "status = 'active'",
        })
        assert result.is_valid

    def test_missing_condition(self) -> None:
        result = validate_rule({
            "type": "conditional_null",
            "column": "email",
        })
        assert not result.is_valid


class TestExpressionValidation:
    def test_valid(self) -> None:
        result = validate_rule({
            "type": "expression",
            "expression": "price * quantity == total",
        })
        assert result.is_valid

    def test_missing_expression(self) -> None:
        result = validate_rule({"type": "expression"})
        assert not result.is_valid


class TestDistributionValidation:
    def test_valid(self) -> None:
        result = validate_rule({
            "type": "distribution",
            "column": "height",
            "distribution_type": "normal",
        })
        assert result.is_valid

    def test_valid_with_params(self) -> None:
        result = validate_rule({
            "type": "distribution",
            "column": "height",
            "distribution_type": "normal",
            "parameters": {"mean": 170, "std": 10},
        })
        assert result.is_valid

    def test_missing_distribution_type(self) -> None:
        result = validate_rule({
            "type": "distribution",
            "column": "height",
        })
        assert not result.is_valid


# =============================================================================
# Batch Validation of New Rules
# =============================================================================


class TestBatchValidation:
    def test_validate_mixed_rules(self) -> None:
        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "statistical_drift", "column": "age", "method": "ks"},
            {"type": "outlier", "column": "amount"},
            {"type": "completeness_ratio", "column": "email", "min_ratio": 0.9},
        ]
        result = validate_rules(rules)
        assert result.is_valid
        assert result.valid_count == 4


# =============================================================================
# TruthoundRuleValidator with New Rules
# =============================================================================


class TestTruthoundValidatorExtended:
    def test_validates_drift_rule(self) -> None:
        validator = TruthoundRuleValidator()
        result = validator.validate({
            "type": "statistical_drift",
            "column": "age",
            "method": "psi",
        })
        assert result.is_valid

    def test_validates_anomaly_rule(self) -> None:
        validator = TruthoundRuleValidator()
        result = validator.validate({
            "type": "outlier",
            "column": "amount",
            "detector": "isolation_forest",
        })
        assert result.is_valid

    def test_validates_distribution_rule(self) -> None:
        validator = TruthoundRuleValidator()
        result = validator.validate({
            "type": "distribution",
            "column": "height",
            "distribution_type": "normal",
        })
        assert result.is_valid

    def test_truthound_warning_still_present(self) -> None:
        validator = TruthoundRuleValidator()
        result = validator.validate({"type": "statistical_drift", "column": "x"})
        assert any("schema-based" in w for w in result.warnings)


# =============================================================================
# Registry Integration
# =============================================================================


class TestRegistryIntegration:
    def test_new_rules_in_global_registry(self) -> None:
        registry = get_rule_registry()
        for rule_type in ALL_NEW_RULES:
            schema = registry.get(rule_type)
            assert schema is not None, f"{rule_type} not found in registry"

    def test_aliases_work(self) -> None:
        registry = get_rule_registry()
        # statistical_drift has alias "drift"
        schema = COMMON_RULE_SCHEMAS["statistical_drift"]
        assert len(schema.aliases) > 0
