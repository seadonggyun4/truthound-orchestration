"""Tests for Mage condition blocks."""

from __future__ import annotations

import pytest

from truthound_mage.blocks.condition import (
    ConditionBlockConfig,
    RouteDecision,
    ConditionResult,
    BaseConditionBlock,
    DataQualityCondition,
    create_quality_condition,
)
from truthound_mage.blocks.base import BlockExecutionContext


class TestConditionBlockConfig:
    """Tests for ConditionBlockConfig."""

    def test_default_values(self) -> None:
        """Test default condition configuration."""
        config = ConditionBlockConfig()
        assert config.pass_threshold == 0.95
        assert config.warning_threshold is None
        assert config.route_on_pass is None
        assert config.route_on_fail is None

    def test_with_thresholds(self) -> None:
        """Test thresholds builder."""
        config = ConditionBlockConfig()
        new_config = config.with_thresholds(
            pass_threshold=0.99,
            warning_threshold=0.90,
        )
        assert new_config.pass_threshold == 0.99
        assert new_config.warning_threshold == 0.90

    def test_threshold_validation(self) -> None:
        """Test threshold validation."""
        with pytest.raises(ValueError):
            ConditionBlockConfig(pass_threshold=1.5)

    def test_warning_must_be_less_than_pass(self) -> None:
        """Test warning must be less than pass threshold."""
        with pytest.raises(ValueError):
            ConditionBlockConfig(pass_threshold=0.9, warning_threshold=0.95)

    def test_with_routes(self) -> None:
        """Test routes builder."""
        config = ConditionBlockConfig()
        new_config = config.with_routes(
            on_pass="success",
            on_fail="error",
            on_warning="review",
        )
        assert new_config.route_on_pass == "success"
        assert new_config.route_on_fail == "error"
        assert new_config.route_on_warning == "review"

    def test_with_custom_route(self) -> None:
        """Test custom route builder."""
        config = ConditionBlockConfig()
        new_config = config.with_custom_route("high_volume", "special_processing")
        assert new_config.custom_routes["high_volume"] == "special_processing"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = ConditionBlockConfig(
            pass_threshold=0.99,
            route_on_pass="success",
        )
        data = config.to_dict()
        assert data["pass_threshold"] == 0.99
        assert data["route_on_pass"] == "success"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "pass_threshold": 0.99,
            "route_on_pass": "success",
            "route_on_fail": "error",
        }
        config = ConditionBlockConfig.from_dict(data)
        assert config.pass_threshold == 0.99
        assert config.route_on_pass == "success"
        assert config.route_on_fail == "error"


class TestRouteDecision:
    """Tests for RouteDecision enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert RouteDecision.PASS.value == "pass"
        assert RouteDecision.FAIL.value == "fail"
        assert RouteDecision.WARNING.value == "warning"
        assert RouteDecision.SKIP.value == "skip"
        assert RouteDecision.ERROR.value == "error"


class TestConditionResult:
    """Tests for ConditionResult."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = ConditionResult(
            decision=RouteDecision.PASS,
            route="success",
            metrics={"pass_rate": 0.98},
        )
        assert result.decision == RouteDecision.PASS
        assert result.route == "success"
        assert result.metrics["pass_rate"] == 0.98

    def test_is_pass_property(self) -> None:
        """Test is_pass property."""
        result = ConditionResult(decision=RouteDecision.PASS)
        assert result.is_pass is True

        result_fail = ConditionResult(decision=RouteDecision.FAIL)
        assert result_fail.is_pass is False

    def test_is_fail_property(self) -> None:
        """Test is_fail property."""
        result = ConditionResult(decision=RouteDecision.FAIL)
        assert result.is_fail is True

    def test_is_warning_property(self) -> None:
        """Test is_warning property."""
        result = ConditionResult(decision=RouteDecision.WARNING)
        assert result.is_warning is True

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        result = ConditionResult(
            decision=RouteDecision.PASS,
            route="success",
            metrics={"pass_rate": 0.98},
            reasons=("Pass rate above threshold",),
        )
        data = result.to_dict()
        assert data["decision"] == "pass"
        assert data["route"] == "success"
        assert data["metrics"]["pass_rate"] == 0.98


class TestDataQualityCondition:
    """Tests for DataQualityCondition."""

    def test_creation(self) -> None:
        """Test condition creation."""
        condition = DataQualityCondition()
        assert condition.config is not None

    def test_evaluate_pass(self) -> None:
        """Test evaluation that passes."""
        config = ConditionBlockConfig(
            pass_threshold=0.90,
            route_on_pass="success",
        )
        condition = DataQualityCondition(config=config)

        check_result = {
            "status": "PASSED",
            "pass_rate": 0.95,
            "failure_rate": 0.05,
            "passed_count": 95,
            "failed_count": 5,
        }
        result = condition.evaluate(check_result)

        assert result.decision == RouteDecision.PASS
        assert result.route == "success"

    def test_evaluate_fail(self) -> None:
        """Test evaluation that fails."""
        config = ConditionBlockConfig(
            pass_threshold=0.99,
            route_on_fail="quarantine",
        )
        condition = DataQualityCondition(config=config)

        check_result = {
            "status": "PASSED",
            "pass_rate": 0.90,
            "failure_rate": 0.10,
            "passed_count": 90,
            "failed_count": 10,
        }
        result = condition.evaluate(check_result)

        assert result.decision == RouteDecision.FAIL
        assert result.route == "quarantine"

    def test_evaluate_warning(self) -> None:
        """Test evaluation with warning threshold."""
        config = ConditionBlockConfig(
            pass_threshold=0.99,
            warning_threshold=0.90,
            route_on_warning="review",
        )
        condition = DataQualityCondition(config=config)

        check_result = {
            "status": "PASSED",
            "pass_rate": 0.95,
            "failure_rate": 0.05,
            "passed_count": 95,
            "failed_count": 5,
        }
        result = condition.evaluate(check_result)

        assert result.decision == RouteDecision.WARNING
        assert result.route == "review"

    def test_evaluate_error_status(self) -> None:
        """Test evaluation with ERROR status."""
        config = ConditionBlockConfig(route_on_error="error_handler")
        condition = DataQualityCondition(config=config)

        check_result = {
            "status": "ERROR",
            "pass_rate": 0.0,
            "failure_rate": 1.0,
        }
        result = condition.evaluate(check_result)

        assert result.decision == RouteDecision.ERROR
        assert result.route == "error_handler"

    def test_evaluate_with_context(self) -> None:
        """Test evaluation with context."""
        context = BlockExecutionContext(
            block_uuid="condition_1",
            pipeline_uuid="pipeline_1",
        )
        condition = DataQualityCondition()

        check_result = {
            "status": "PASSED",
            "pass_rate": 1.0,
            "failure_rate": 0.0,
            "passed_count": 100,
            "failed_count": 0,
        }
        result = condition.evaluate(check_result, context=context)

        assert result.decision == RouteDecision.PASS

    def test_evaluate_with_custom_conditions(self) -> None:
        """Test evaluation with custom conditions."""

        def high_volume_condition(metrics: dict) -> bool:
            return metrics.get("passed_count", 0) > 1000

        condition = DataQualityCondition(
            config=ConditionBlockConfig(pass_threshold=0.90),
            custom_conditions={"high_volume": high_volume_condition},
        )
        condition.config = condition.config.with_custom_route(
            "high_volume", "high_volume_processing"
        )

        check_result = {
            "status": "PASSED",
            "pass_rate": 0.95,
            "failure_rate": 0.05,
            "passed_count": 1500,
            "failed_count": 80,
        }
        result = condition.evaluate(check_result)

        assert result.route == "high_volume_processing"

    def test_get_route(self) -> None:
        """Test get_route convenience method."""
        config = ConditionBlockConfig(
            pass_threshold=0.90,
            route_on_pass="success",
        )
        condition = DataQualityCondition(config=config)

        check_result = {
            "status": "PASSED",
            "pass_rate": 0.95,
            "failure_rate": 0.05,
            "passed_count": 95,
            "failed_count": 5,
        }
        route = condition.get_route(check_result)

        assert route == "success"


class TestCreateQualityCondition:
    """Tests for create_quality_condition factory function."""

    def test_basic_creation(self) -> None:
        """Test basic condition creation."""
        condition = create_quality_condition(pass_threshold=0.95)
        assert condition.config.pass_threshold == 0.95

    def test_creation_with_all_options(self) -> None:
        """Test condition creation with all options."""
        condition = create_quality_condition(
            pass_threshold=0.99,
            warning_threshold=0.90,
            route_on_pass="success",
            route_on_fail="error",
            route_on_warning="review",
            default_route="fallback",
        )
        assert condition.config.pass_threshold == 0.99
        assert condition.config.warning_threshold == 0.90
        assert condition.config.route_on_pass == "success"
        assert condition.config.route_on_fail == "error"
        assert condition.config.route_on_warning == "review"
        assert condition.config.default_route == "fallback"

    def test_creation_with_custom_conditions(self) -> None:
        """Test condition creation with custom conditions."""

        def custom_check(metrics: dict) -> bool:
            return metrics.get("passed_count", 0) > 50

        condition = create_quality_condition(
            pass_threshold=0.90,
            custom_conditions={"custom": custom_check},
        )
        assert "custom" in condition._custom_conditions
