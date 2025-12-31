"""Tests for SLA configuration types."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_values(self) -> None:
        """Test all alert level values exist."""
        from truthound_airflow.sla.config import AlertLevel

        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestSLAViolationType:
    """Tests for SLAViolationType enum."""

    def test_values(self) -> None:
        """Test all violation type values exist."""
        from truthound_airflow.sla.config import SLAViolationType

        assert SLAViolationType.FAILURE_RATE_EXCEEDED.value == "failure_rate_exceeded"
        assert SLAViolationType.PASS_RATE_BELOW_MINIMUM.value == "pass_rate_below_minimum"
        assert SLAViolationType.EXECUTION_TIME_EXCEEDED.value == "execution_time_exceeded"
        assert SLAViolationType.ROW_COUNT_BELOW_MINIMUM.value == "row_count_below_minimum"
        assert SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM.value == "row_count_above_maximum"
        assert SLAViolationType.CONSECUTIVE_FAILURES.value == "consecutive_failures"
        assert SLAViolationType.CUSTOM.value == "custom"


class TestSLAConfig:
    """Tests for SLAConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from truthound_airflow.sla.config import AlertLevel, SLAConfig

        config = SLAConfig()

        assert config.max_failure_rate is None
        assert config.min_pass_rate is None
        assert config.max_execution_time_seconds is None
        assert config.min_row_count is None
        assert config.max_row_count is None
        assert config.max_consecutive_failures == 3
        assert config.alert_on_warning is False
        assert config.alert_level == AlertLevel.ERROR
        assert config.enabled is True
        assert config.tags == frozenset()

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        from truthound_airflow.sla.config import AlertLevel, SLAConfig

        config = SLAConfig(
            max_failure_rate=0.05,
            min_pass_rate=0.95,
            max_execution_time_seconds=300.0,
            min_row_count=100,
            max_row_count=10000,
            max_consecutive_failures=5,
            alert_on_warning=True,
            alert_level=AlertLevel.CRITICAL,
            tags=frozenset({"production"}),
        )

        assert config.max_failure_rate == 0.05
        assert config.min_pass_rate == 0.95
        assert config.max_execution_time_seconds == 300.0
        assert config.min_row_count == 100
        assert config.max_row_count == 10000

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        from truthound_airflow.sla.config import SLAConfig

        config = SLAConfig()

        with pytest.raises(AttributeError):
            config.max_failure_rate = 0.1  # type: ignore

    def test_validation_failure_rate_range(self) -> None:
        """Test failure rate validation."""
        from truthound_airflow.sla.config import SLAConfig

        with pytest.raises(ValueError, match="max_failure_rate must be between 0 and 1"):
            SLAConfig(max_failure_rate=1.5)

        with pytest.raises(ValueError, match="max_failure_rate must be between 0 and 1"):
            SLAConfig(max_failure_rate=-0.1)

    def test_validation_pass_rate_range(self) -> None:
        """Test pass rate validation."""
        from truthound_airflow.sla.config import SLAConfig

        with pytest.raises(ValueError, match="min_pass_rate must be between 0 and 1"):
            SLAConfig(min_pass_rate=1.5)

    def test_validation_execution_time_positive(self) -> None:
        """Test execution time validation."""
        from truthound_airflow.sla.config import SLAConfig

        with pytest.raises(ValueError, match="max_execution_time_seconds must be positive"):
            SLAConfig(max_execution_time_seconds=-1.0)

    def test_validation_row_count_range(self) -> None:
        """Test row count range validation."""
        from truthound_airflow.sla.config import SLAConfig

        with pytest.raises(ValueError, match="min_row_count cannot exceed max_row_count"):
            SLAConfig(min_row_count=1000, max_row_count=100)

    def test_with_failure_rate(self) -> None:
        """Test builder method for failure rate."""
        from truthound_airflow.sla.config import SLAConfig

        config = SLAConfig()
        new_config = config.with_failure_rate(0.05)

        assert config.max_failure_rate is None
        assert new_config.max_failure_rate == 0.05

    def test_with_pass_rate(self) -> None:
        """Test builder method for pass rate."""
        from truthound_airflow.sla.config import SLAConfig

        config = SLAConfig()
        new_config = config.with_pass_rate(0.95)

        assert config.min_pass_rate is None
        assert new_config.min_pass_rate == 0.95

    def test_with_execution_time(self) -> None:
        """Test builder method for execution time."""
        from truthound_airflow.sla.config import SLAConfig

        config = SLAConfig()
        new_config = config.with_execution_time(300.0)

        assert new_config.max_execution_time_seconds == 300.0

    def test_with_row_count_range(self) -> None:
        """Test builder method for row count range."""
        from truthound_airflow.sla.config import SLAConfig

        config = SLAConfig()
        new_config = config.with_row_count_range(min_count=100, max_count=5000)

        assert new_config.min_row_count == 100
        assert new_config.max_row_count == 5000

    def test_with_alert_level(self) -> None:
        """Test builder method for alert level."""
        from truthound_airflow.sla.config import AlertLevel, SLAConfig

        config = SLAConfig()
        new_config = config.with_alert_level(AlertLevel.CRITICAL)

        assert new_config.alert_level == AlertLevel.CRITICAL

    def test_with_enabled(self) -> None:
        """Test builder method for enabled status."""
        from truthound_airflow.sla.config import SLAConfig

        config = SLAConfig()
        new_config = config.with_enabled(False)

        assert new_config.enabled is False


class TestSLAMetrics:
    """Tests for SLAMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default metrics values."""
        from truthound_airflow.sla.config import SLAMetrics

        metrics = SLAMetrics()

        assert metrics.passed_count == 0
        assert metrics.failed_count == 0
        assert metrics.warning_count == 0
        assert metrics.execution_time_ms == 0.0
        assert metrics.row_count is None

    def test_total_count_property(self) -> None:
        """Test total_count property."""
        from truthound_airflow.sla.config import SLAMetrics

        metrics = SLAMetrics(passed_count=8, failed_count=2)

        assert metrics.total_count == 10

    def test_pass_rate_property(self) -> None:
        """Test pass_rate property."""
        from truthound_airflow.sla.config import SLAMetrics

        metrics = SLAMetrics(passed_count=8, failed_count=2)

        assert metrics.pass_rate == 0.8

    def test_pass_rate_zero_total(self) -> None:
        """Test pass_rate with zero total count."""
        from truthound_airflow.sla.config import SLAMetrics

        metrics = SLAMetrics()

        assert metrics.pass_rate == 1.0

    def test_failure_rate_property(self) -> None:
        """Test failure_rate property."""
        from truthound_airflow.sla.config import SLAMetrics

        metrics = SLAMetrics(passed_count=8, failed_count=2)

        assert metrics.failure_rate == 0.2

    def test_execution_time_seconds_property(self) -> None:
        """Test execution_time_seconds property."""
        from truthound_airflow.sla.config import SLAMetrics

        metrics = SLAMetrics(execution_time_ms=1500.0)

        assert metrics.execution_time_seconds == 1.5

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        from truthound_airflow.sla.config import SLAMetrics

        metrics = SLAMetrics(
            passed_count=8,
            failed_count=2,
            warning_count=1,
            execution_time_ms=1500.0,
            row_count=1000,
        )

        result = metrics.to_dict()

        assert result["passed_count"] == 8
        assert result["failed_count"] == 2
        assert result["total_count"] == 10
        assert result["pass_rate"] == 0.8
        assert result["failure_rate"] == 0.2
        assert result["row_count"] == 1000

    def test_from_check_result(self) -> None:
        """Test from_check_result class method."""
        from truthound_airflow.sla.config import SLAMetrics

        result_dict = {
            "passed_count": 95,
            "failed_count": 5,
            "warning_count": 2,
            "execution_time_ms": 1000.0,
            "_metadata": {"row_count": 10000},
        }

        metrics = SLAMetrics.from_check_result(
            result_dict,
            task_id="test_task",
            dag_id="test_dag",
        )

        assert metrics.passed_count == 95
        assert metrics.failed_count == 5
        assert metrics.task_id == "test_task"
        assert metrics.dag_id == "test_dag"


class TestSLAViolation:
    """Tests for SLAViolation dataclass."""

    def test_creation(self) -> None:
        """Test violation creation."""
        from truthound_airflow.sla.config import AlertLevel, SLAViolation, SLAViolationType

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Failure rate 10% exceeds threshold 5%",
            threshold=0.05,
            actual=0.10,
            alert_level=AlertLevel.ERROR,
        )

        assert violation.violation_type == SLAViolationType.FAILURE_RATE_EXCEEDED
        assert violation.threshold == 0.05
        assert violation.actual == 0.10

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        from truthound_airflow.sla.config import AlertLevel, SLAViolation, SLAViolationType

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test message",
            threshold=0.05,
            actual=0.10,
        )

        result = violation.to_dict()

        assert result["violation_type"] == "failure_rate_exceeded"
        assert result["message"] == "Test message"
        assert result["threshold"] == 0.05
        assert result["actual"] == 0.10


class TestPresetConfigs:
    """Tests for preset SLA configurations."""

    def test_default_config(self) -> None:
        """Test default SLA config preset."""
        from truthound_airflow.sla.config import DEFAULT_SLA_CONFIG

        assert DEFAULT_SLA_CONFIG.enabled is True

    def test_strict_config(self) -> None:
        """Test strict SLA config preset."""
        from truthound_airflow.sla.config import AlertLevel, STRICT_SLA_CONFIG

        assert STRICT_SLA_CONFIG.max_failure_rate == 0.01
        assert STRICT_SLA_CONFIG.min_pass_rate == 0.99
        assert STRICT_SLA_CONFIG.max_consecutive_failures == 1
        assert STRICT_SLA_CONFIG.alert_level == AlertLevel.CRITICAL

    def test_lenient_config(self) -> None:
        """Test lenient SLA config preset."""
        from truthound_airflow.sla.config import AlertLevel, LENIENT_SLA_CONFIG

        assert LENIENT_SLA_CONFIG.max_failure_rate == 0.10
        assert LENIENT_SLA_CONFIG.min_pass_rate == 0.90
        assert LENIENT_SLA_CONFIG.alert_level == AlertLevel.WARNING

    def test_production_config(self) -> None:
        """Test production SLA config preset."""
        from truthound_airflow.sla.config import PRODUCTION_SLA_CONFIG

        assert PRODUCTION_SLA_CONFIG.max_failure_rate == 0.05
        assert PRODUCTION_SLA_CONFIG.min_pass_rate == 0.95
        assert PRODUCTION_SLA_CONFIG.alert_on_warning is True
