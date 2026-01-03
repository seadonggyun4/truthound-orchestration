"""Tests for Mage sensor blocks."""

from __future__ import annotations

import pytest

from truthound_mage.blocks.sensor import (
    SensorBlockConfig,
    SensorResult,
    BaseSensorBlock,
    DataQualitySensor,
    QualityGateSensor,
    create_quality_sensor,
)
from truthound_mage.blocks.base import BlockExecutionContext


class TestSensorBlockConfig:
    """Tests for SensorBlockConfig."""

    def test_default_values(self) -> None:
        """Test default sensor configuration."""
        config = SensorBlockConfig()
        assert config.poke_interval_seconds == 60.0
        assert config.min_pass_rate is None
        assert config.max_failure_rate is None
        assert config.mode == "poke"
        assert config.soft_fail is False

    def test_with_pass_rate(self) -> None:
        """Test pass rate builder."""
        config = SensorBlockConfig()
        new_config = config.with_pass_rate(0.95)
        assert new_config.min_pass_rate == 0.95

    def test_with_failure_rate(self) -> None:
        """Test failure rate builder."""
        config = SensorBlockConfig()
        new_config = config.with_failure_rate(0.05)
        assert new_config.max_failure_rate == 0.05

    def test_with_row_count_range(self) -> None:
        """Test row count range builder."""
        config = SensorBlockConfig()
        new_config = config.with_row_count_range(min_count=100, max_count=10000)
        assert new_config.min_row_count == 100
        assert new_config.max_row_count == 10000

    def test_with_poke_settings(self) -> None:
        """Test poke settings builder."""
        config = SensorBlockConfig()
        new_config = config.with_poke_settings(
            interval_seconds=30.0,
            max_attempts=5,
            exponential_backoff=True,
        )
        assert new_config.poke_interval_seconds == 30.0
        assert new_config.max_poke_attempts == 5
        assert new_config.exponential_backoff is True

    def test_pass_rate_validation(self) -> None:
        """Test pass rate validation."""
        with pytest.raises(ValueError):
            SensorBlockConfig(min_pass_rate=1.5)

    def test_failure_rate_validation(self) -> None:
        """Test failure rate validation."""
        with pytest.raises(ValueError):
            SensorBlockConfig(max_failure_rate=-0.1)

    def test_mode_validation(self) -> None:
        """Test mode validation."""
        with pytest.raises(ValueError):
            SensorBlockConfig(mode="invalid")

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = SensorBlockConfig(
            min_pass_rate=0.95,
            poke_interval_seconds=30.0,
        )
        data = config.to_dict()
        assert data["min_pass_rate"] == 0.95
        assert data["poke_interval_seconds"] == 30.0

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "min_pass_rate": 0.95,
            "max_failure_rate": 0.05,
        }
        config = SensorBlockConfig.from_dict(data)
        assert config.min_pass_rate == 0.95
        assert config.max_failure_rate == 0.05


class TestSensorResult:
    """Tests for SensorResult."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = SensorResult(
            passed=True,
            message="All conditions met",
            metrics={"pass_rate": 0.98},
        )
        assert result.passed is True
        assert result.message == "All conditions met"
        assert result.metrics["pass_rate"] == 0.98

    def test_is_passed_property(self) -> None:
        """Test is_passed property."""
        result = SensorResult(passed=True)
        assert result.is_passed is True

    def test_has_violations_property(self) -> None:
        """Test has_violations property."""
        result_no_violations = SensorResult(passed=True)
        assert result_no_violations.has_violations is False

        result_with_violations = SensorResult(
            passed=False,
            violations=("Pass rate below minimum",),
        )
        assert result_with_violations.has_violations is True

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        result = SensorResult(
            passed=True,
            message="Success",
            poke_count=3,
            total_time_seconds=90.5,
        )
        data = result.to_dict()
        assert data["passed"] is True
        assert data["message"] == "Success"
        assert data["poke_count"] == 3
        assert data["total_time_seconds"] == 90.5


class TestDataQualitySensor:
    """Tests for DataQualitySensor."""

    def test_creation(self) -> None:
        """Test sensor creation."""
        sensor = DataQualitySensor()
        assert sensor.config is not None

    def test_poke_with_pass_rate_met(self) -> None:
        """Test poke when pass rate condition is met."""
        config = SensorBlockConfig(min_pass_rate=0.90)
        sensor = DataQualitySensor(config=config)

        check_result = {
            "pass_rate": 0.95,
            "failure_rate": 0.05,
            "passed_count": 95,
            "failed_count": 5,
        }
        assert sensor.poke(check_result) is True

    def test_poke_with_pass_rate_not_met(self) -> None:
        """Test poke when pass rate condition is not met."""
        config = SensorBlockConfig(min_pass_rate=0.99)
        sensor = DataQualitySensor(config=config)

        check_result = {
            "pass_rate": 0.95,
            "failure_rate": 0.05,
            "passed_count": 95,
            "failed_count": 5,
        }
        assert sensor.poke(check_result) is False

    def test_poke_with_failure_rate_check(self) -> None:
        """Test poke with failure rate check."""
        config = SensorBlockConfig(max_failure_rate=0.05)
        sensor = DataQualitySensor(config=config)

        # Failure rate within limit
        result_ok = {
            "pass_rate": 0.97,
            "failure_rate": 0.03,
            "passed_count": 97,
            "failed_count": 3,
        }
        assert sensor.poke(result_ok) is True

        # Failure rate exceeds limit
        result_fail = {
            "pass_rate": 0.90,
            "failure_rate": 0.10,
            "passed_count": 90,
            "failed_count": 10,
        }
        assert sensor.poke(result_fail) is False

    def test_poke_with_row_count_check(self) -> None:
        """Test poke with row count validation."""
        config = SensorBlockConfig(min_row_count=100, max_row_count=10000)
        sensor = DataQualitySensor(config=config)

        # Valid row count
        result_valid = {
            "pass_rate": 1.0,
            "failure_rate": 0.0,
            "passed_count": 500,
            "failed_count": 0,
            "total_count": 500,
        }
        assert sensor.poke(result_valid) is True

        # Too few rows
        result_too_few = {
            "pass_rate": 1.0,
            "failure_rate": 0.0,
            "passed_count": 50,
            "failed_count": 0,
            "total_count": 50,
        }
        assert sensor.poke(result_too_few) is False

    def test_poke_with_custom_condition(self) -> None:
        """Test poke with custom condition function."""

        def custom_condition(result: dict) -> bool:
            return result.get("passed_count", 0) >= 100

        sensor = DataQualitySensor(condition_fn=custom_condition)

        assert sensor.poke({"passed_count": 100}) is True
        assert sensor.poke({"passed_count": 99}) is False

    def test_check_single_poke(self) -> None:
        """Test check method (single poke without waiting)."""
        config = SensorBlockConfig(min_pass_rate=0.90)
        sensor = DataQualitySensor(config=config)

        check_result = {
            "pass_rate": 0.95,
            "failure_rate": 0.05,
            "passed_count": 95,
            "failed_count": 5,
            "total_count": 100,
        }
        result = sensor.check(check_result)
        assert result.passed is True
        assert result.poke_count == 1

    def test_check_with_context(self) -> None:
        """Test check with execution context."""
        sensor = DataQualitySensor()
        context = BlockExecutionContext(
            block_uuid="sensor_1",
            pipeline_uuid="pipeline_1",
        )
        check_result = {
            "pass_rate": 1.0,
            "failure_rate": 0.0,
            "passed_count": 100,
            "failed_count": 0,
        }
        result = sensor.check(check_result, context=context)
        assert result.passed is True


class TestQualityGateSensor:
    """Tests for QualityGateSensor."""

    def test_poke_with_passed_status(self) -> None:
        """Test poke with PASSED status."""
        config = SensorBlockConfig(min_pass_rate=0.95)
        sensor = QualityGateSensor(config=config)

        check_result = {
            "status": "PASSED",
            "pass_rate": 0.98,
            "failure_rate": 0.02,
            "passed_count": 98,
            "failed_count": 2,
        }
        assert sensor.poke(check_result) is True

    def test_poke_with_failed_status(self) -> None:
        """Test poke with FAILED status."""
        sensor = QualityGateSensor()

        check_result = {
            "status": "FAILED",
            "pass_rate": 0.50,
            "failure_rate": 0.50,
            "passed_count": 50,
            "failed_count": 50,
        }
        assert sensor.poke(check_result) is False

    def test_poke_with_error_status(self) -> None:
        """Test poke with ERROR status."""
        sensor = QualityGateSensor()

        check_result = {
            "status": "ERROR",
            "pass_rate": 0.0,
            "failure_rate": 1.0,
        }
        assert sensor.poke(check_result) is False

    def test_poke_with_row_count_limits(self) -> None:
        """Test poke with row count validation."""
        config = SensorBlockConfig(min_row_count=100, max_row_count=10000)
        sensor = QualityGateSensor(config=config)

        # Valid row count
        result_valid = {
            "status": "PASSED",
            "pass_rate": 1.0,
            "failure_rate": 0.0,
            "total_count": 500,
        }
        assert sensor.poke(result_valid) is True

        # Too few rows
        result_too_few = {
            "status": "PASSED",
            "pass_rate": 1.0,
            "failure_rate": 0.0,
            "total_count": 50,
        }
        assert sensor.poke(result_too_few) is False


class TestCreateQualitySensor:
    """Tests for create_quality_sensor factory function."""

    def test_basic_creation(self) -> None:
        """Test basic sensor creation."""
        sensor = create_quality_sensor(min_pass_rate=0.95)
        assert sensor.config.min_pass_rate == 0.95

    def test_creation_with_all_options(self) -> None:
        """Test sensor creation with all options."""
        sensor = create_quality_sensor(
            min_pass_rate=0.95,
            max_failure_rate=0.05,
            min_row_count=100,
            poke_interval_seconds=30.0,
            timeout_seconds=120,
            soft_fail=True,
        )
        assert sensor.config.min_pass_rate == 0.95
        assert sensor.config.max_failure_rate == 0.05
        assert sensor.config.min_row_count == 100
        assert sensor.config.poke_interval_seconds == 30.0
        assert sensor.config.timeout_seconds == 120
        assert sensor.config.soft_fail is True

    def test_creation_with_custom_condition(self) -> None:
        """Test sensor creation with custom condition."""

        def custom_condition(result: dict) -> bool:
            return result.get("passed_count", 0) > 0

        sensor = create_quality_sensor(condition_fn=custom_condition)
        assert sensor.poke({"passed_count": 1}) is True
        assert sensor.poke({"passed_count": 0}) is False
