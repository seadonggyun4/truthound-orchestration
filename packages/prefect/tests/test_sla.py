"""Tests for truthound_prefect.sla module."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound_prefect.sla.config import (
    DEFAULT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)
from truthound_prefect.sla.monitor import (
    SLAMonitor,
    SLARegistry,
    get_sla_registry,
    reset_sla_registry,
)
from truthound_prefect.sla.hooks import (
    CallbackSLAHook,
    CompositeSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    SLAHook,
    SLAHookStats,
)
from truthound_prefect.sla.block import SLABlock


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_levels_exist(self) -> None:
        """Test that all alert levels exist."""
        assert AlertLevel.DEBUG is not None
        assert AlertLevel.INFO is not None
        assert AlertLevel.WARNING is not None
        assert AlertLevel.ERROR is not None
        assert AlertLevel.CRITICAL is not None

    def test_alert_level_values(self) -> None:
        """Test alert level string values."""
        assert AlertLevel.DEBUG.value == "debug"
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestSLAViolationType:
    """Tests for SLAViolationType enum."""

    def test_violation_types_exist(self) -> None:
        """Test that all violation types exist."""
        assert SLAViolationType.FAILURE_RATE_EXCEEDED is not None
        assert SLAViolationType.PASS_RATE_BELOW_MINIMUM is not None
        assert SLAViolationType.EXECUTION_TIME_EXCEEDED is not None
        assert SLAViolationType.ROW_COUNT_BELOW_MINIMUM is not None
        assert SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM is not None
        assert SLAViolationType.CONSECUTIVE_FAILURES is not None


class TestSLAConfig:
    """Tests for SLAConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SLAConfig()
        assert config.max_failure_rate is None
        assert config.min_pass_rate is None
        assert config.max_execution_time_seconds is None
        assert config.min_row_count is None
        assert config.max_row_count is None
        assert config.max_consecutive_failures == 3
        assert config.alert_level == AlertLevel.ERROR
        assert config.enabled is True

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = SLAConfig()
        with pytest.raises(AttributeError):
            config.max_failure_rate = 0.1  # type: ignore

    def test_with_failure_rate(self) -> None:
        """Test with_failure_rate builder method."""
        config = SLAConfig()
        new_config = config.with_failure_rate(0.05)
        assert new_config.max_failure_rate == 0.05

    def test_with_pass_rate(self) -> None:
        """Test with_pass_rate builder method."""
        config = SLAConfig()
        new_config = config.with_pass_rate(0.95)
        assert new_config.min_pass_rate == 0.95

    def test_with_execution_time(self) -> None:
        """Test with_execution_time builder method."""
        config = SLAConfig()
        new_config = config.with_execution_time(300.0)
        assert new_config.max_execution_time_seconds == 300.0

    def test_with_row_count_bounds(self) -> None:
        """Test with_row_count_bounds builder method."""
        config = SLAConfig()
        new_config = config.with_row_count_bounds(min_count=100, max_count=10000)
        assert new_config.min_row_count == 100
        assert new_config.max_row_count == 10000

    def test_with_consecutive_failures(self) -> None:
        """Test with_consecutive_failures builder method."""
        config = SLAConfig()
        new_config = config.with_consecutive_failures(5)
        assert new_config.max_consecutive_failures == 5

    def test_with_alert_settings(self) -> None:
        """Test with_alert_settings builder method."""
        config = SLAConfig()
        new_config = config.with_alert_settings(
            alert_level=AlertLevel.CRITICAL, alert_on_warning=True
        )
        assert new_config.alert_level == AlertLevel.CRITICAL
        assert new_config.alert_on_warning is True

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        config = SLAConfig(
            max_failure_rate=0.05,
            min_pass_rate=0.95,
            max_execution_time_seconds=300.0,
        )
        d = config.to_dict()
        assert d["max_failure_rate"] == 0.05
        assert d["min_pass_rate"] == 0.95
        assert d["max_execution_time_seconds"] == 300.0


class TestSLAMetrics:
    """Tests for SLAMetrics."""

    def test_default_values(self) -> None:
        """Test default values."""
        metrics = SLAMetrics()
        assert metrics.passed_count == 0
        assert metrics.failed_count == 0
        assert metrics.warning_count == 0
        assert metrics.execution_time_ms == 0.0
        assert metrics.row_count is None

    def test_total_count(self) -> None:
        """Test total_count computed property."""
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        assert metrics.total_count == 100

    def test_pass_rate_computation(self) -> None:
        """Test pass_rate computed property."""
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        assert metrics.pass_rate == 0.95

    def test_pass_rate_zero_total(self) -> None:
        """Test pass_rate with zero total returns 1.0."""
        metrics = SLAMetrics(passed_count=0, failed_count=0)
        assert metrics.pass_rate == 1.0

    def test_failure_rate_computation(self) -> None:
        """Test failure_rate computed property."""
        metrics = SLAMetrics(passed_count=90, failed_count=10)
        assert metrics.failure_rate == 0.10

    def test_execution_time_seconds(self) -> None:
        """Test execution_time_seconds property."""
        metrics = SLAMetrics(execution_time_ms=1500.0)
        assert metrics.execution_time_seconds == 1.5

    def test_from_check_result(self) -> None:
        """Test creating metrics from check result dict."""
        result_dict = {
            "passed_count": 95,
            "failed_count": 5,
            "execution_time_ms": 1500.0,
            "metadata": {"row_count": 1000},
        }
        metrics = SLAMetrics.from_check_result(result_dict)
        assert metrics.passed_count == 95
        assert metrics.failed_count == 5
        assert metrics.execution_time_ms == 1500.0
        assert metrics.row_count == 1000


class TestSLAViolation:
    """Tests for SLAViolation."""

    def test_creation(self) -> None:
        """Test violation creation."""
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Failure rate 10% exceeds maximum 5%",
            threshold=0.05,
            actual=0.10,
            alert_level=AlertLevel.WARNING,
        )
        assert violation.violation_type == SLAViolationType.FAILURE_RATE_EXCEEDED
        assert violation.actual == 0.10
        assert violation.threshold == 0.05

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        violation = SLAViolation(
            violation_type=SLAViolationType.EXECUTION_TIME_EXCEEDED,
            message="Execution time exceeded",
            actual=400.0,
            threshold=300.0,
            alert_level=AlertLevel.ERROR,
        )
        d = violation.to_dict()
        assert d["violation_type"] == "execution_time_exceeded"
        assert d["actual"] == 400.0
        assert d["threshold"] == 300.0


class TestPresetSLAConfigs:
    """Tests for preset SLA configurations."""

    def test_default_sla_config(self) -> None:
        """Test DEFAULT_SLA_CONFIG preset."""
        assert DEFAULT_SLA_CONFIG.max_failure_rate is None
        assert DEFAULT_SLA_CONFIG.max_consecutive_failures == 3

    def test_strict_sla_config(self) -> None:
        """Test STRICT_SLA_CONFIG preset."""
        assert STRICT_SLA_CONFIG.max_failure_rate == 0.01
        assert STRICT_SLA_CONFIG.min_pass_rate == 0.99
        assert STRICT_SLA_CONFIG.max_consecutive_failures == 1

    def test_lenient_sla_config(self) -> None:
        """Test LENIENT_SLA_CONFIG preset."""
        assert LENIENT_SLA_CONFIG.max_failure_rate == 0.10
        assert LENIENT_SLA_CONFIG.min_pass_rate == 0.90

    def test_production_sla_config(self) -> None:
        """Test PRODUCTION_SLA_CONFIG preset."""
        assert PRODUCTION_SLA_CONFIG.max_failure_rate == 0.05
        assert PRODUCTION_SLA_CONFIG.alert_level == AlertLevel.ERROR


class TestSLAMonitor:
    """Tests for SLAMonitor."""

    def test_creation(self) -> None:
        """Test monitor creation."""
        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config, name="test_monitor")
        assert monitor.name == "test_monitor"
        assert monitor.config == config

    def test_check_no_violations(self) -> None:
        """Test check with no violations."""
        config = SLAConfig(max_failure_rate=0.10)
        monitor = SLAMonitor(config, name="test")
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        violations = monitor.check(metrics)
        assert len(violations) == 0

    def test_check_failure_rate_violation(self) -> None:
        """Test check with failure rate violation."""
        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config, name="test")
        metrics = SLAMetrics(passed_count=90, failed_count=10)
        violations = monitor.check(metrics)
        assert len(violations) >= 1
        assert any(
            v.violation_type == SLAViolationType.FAILURE_RATE_EXCEEDED
            for v in violations
        )

    def test_check_pass_rate_violation(self) -> None:
        """Test check with pass rate violation."""
        config = SLAConfig(min_pass_rate=0.95)
        monitor = SLAMonitor(config, name="test")
        metrics = SLAMetrics(passed_count=90, failed_count=10)
        violations = monitor.check(metrics)
        assert len(violations) >= 1
        assert any(
            v.violation_type == SLAViolationType.PASS_RATE_BELOW_MINIMUM
            for v in violations
        )

    def test_check_execution_time_violation(self) -> None:
        """Test check with execution time violation."""
        config = SLAConfig(max_execution_time_seconds=60.0)
        monitor = SLAMonitor(config, name="test")
        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            execution_time_ms=120000.0,  # 120 seconds in ms
        )
        violations = monitor.check(metrics)
        assert len(violations) >= 1
        assert any(
            v.violation_type == SLAViolationType.EXECUTION_TIME_EXCEEDED
            for v in violations
        )

    def test_check_row_count_below_minimum(self) -> None:
        """Test check with row count below minimum."""
        config = SLAConfig(min_row_count=100)
        monitor = SLAMonitor(config, name="test")
        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            row_count=50,
        )
        violations = monitor.check(metrics)
        assert len(violations) >= 1
        assert any(
            v.violation_type == SLAViolationType.ROW_COUNT_BELOW_MINIMUM
            for v in violations
        )

    def test_consecutive_failure_tracking(self) -> None:
        """Test consecutive failure tracking."""
        config = SLAConfig(max_consecutive_failures=2)
        monitor = SLAMonitor(config, name="test")

        # First failure
        metrics1 = SLAMetrics(passed_count=0, failed_count=10)
        monitor.check(metrics1)

        # Second failure
        monitor.check(metrics1)

        # Third failure (should trigger consecutive violation)
        violations3 = monitor.check(metrics1)
        assert any(
            v.violation_type == SLAViolationType.CONSECUTIVE_FAILURES
            for v in violations3
        )

    def test_reset(self) -> None:
        """Test monitor reset."""
        config = SLAConfig(max_consecutive_failures=3)
        monitor = SLAMonitor(config, name="test")

        # Create some failures
        metrics = SLAMetrics(passed_count=0, failed_count=10)
        monitor.check(metrics)
        monitor.check(metrics)

        # Reset
        monitor.reset()

        # Verify reset
        assert monitor._consecutive_failures == 0


class TestSLARegistry:
    """Tests for SLARegistry."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_sla_registry()

    def test_singleton(self) -> None:
        """Test registry singleton behavior."""
        registry1 = get_sla_registry()
        registry2 = get_sla_registry()
        assert registry1 is registry2

    def test_register_monitor(self) -> None:
        """Test registering a monitor."""
        registry = get_sla_registry()
        config = SLAConfig(max_failure_rate=0.05)
        monitor = registry.register("test_monitor", config)
        assert monitor is not None
        assert monitor.name == "test_monitor"

    def test_get_monitor(self) -> None:
        """Test getting a registered monitor."""
        registry = get_sla_registry()
        config = SLAConfig(max_failure_rate=0.05)
        registry.register("test_monitor", config)
        monitor = registry.get("test_monitor")
        assert monitor is not None
        assert monitor.name == "test_monitor"

    def test_get_nonexistent_monitor(self) -> None:
        """Test getting a non-existent monitor."""
        registry = get_sla_registry()
        monitor = registry.get("nonexistent")
        assert monitor is None

    def test_list_monitors(self) -> None:
        """Test listing all monitors."""
        registry = get_sla_registry()
        registry.register("monitor1", SLAConfig())
        registry.register("monitor2", SLAConfig())
        names = registry.list_names()
        assert "monitor1" in names
        assert "monitor2" in names

    def test_unregister_monitor(self) -> None:
        """Test unregistering a monitor."""
        registry = get_sla_registry()
        registry.register("test_monitor", SLAConfig())
        result = registry.unregister("test_monitor")
        assert result is True
        assert registry.get("test_monitor") is None


class TestSLAHooks:
    """Tests for SLA hooks."""

    def test_logging_hook(self) -> None:
        """Test LoggingSLAHook."""
        hook = LoggingSLAHook()
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test violation",
            actual=0.10,
            threshold=0.05,
            alert_level=AlertLevel.WARNING,
        )
        # Should not raise - on_violation takes (violation, context)
        hook.on_violation(violation)

    def test_metrics_hook(self) -> None:
        """Test MetricsSLAHook."""
        hook = MetricsSLAHook()
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test violation",
            actual=0.10,
            threshold=0.05,
            alert_level=AlertLevel.WARNING,
        )
        hook.on_violation(violation)
        # Check stats using the property
        assert hook.violation_count == 1
        assert hook.stats.violations_by_type.get("failure_rate_exceeded", 0) == 1

    def test_callback_hook(self) -> None:
        """Test CallbackSLAHook."""
        violations_received: list[SLAViolation] = []

        def callback(violation: SLAViolation, context: dict | None = None) -> None:
            violations_received.append(violation)

        hook = CallbackSLAHook(on_violation_callback=callback)
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test violation",
            actual=0.10,
            threshold=0.05,
            alert_level=AlertLevel.WARNING,
        )
        hook.on_violation(violation)
        assert len(violations_received) == 1

    def test_composite_hook(self) -> None:
        """Test CompositeSLAHook."""
        logging_hook = LoggingSLAHook()
        metrics_hook = MetricsSLAHook()
        composite = CompositeSLAHook([logging_hook, metrics_hook])

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test violation",
            actual=0.10,
            threshold=0.05,
            alert_level=AlertLevel.WARNING,
        )
        composite.on_violation(violation)
        # Verify metrics hook received it
        assert metrics_hook.violation_count == 1


class TestSLABlock:
    """Tests for SLABlock."""

    def test_creation(self) -> None:
        """Test SLA block creation."""
        block = SLABlock(
            max_failure_rate=0.05,
            min_pass_rate=0.95,
            max_execution_time_seconds=300.0,
        )
        assert block.max_failure_rate == 0.05
        assert block.min_pass_rate == 0.95
        assert block.max_execution_time_seconds == 300.0

    def test_get_config(self) -> None:
        """Test getting SLA config from block (private method)."""
        block = SLABlock(
            max_failure_rate=0.05,
            max_consecutive_failures=5,
        )
        # _get_config is a private method
        config = block._get_config()
        assert isinstance(config, SLAConfig)
        assert config.max_failure_rate == 0.05
        assert config.max_consecutive_failures == 5

    def test_check(self) -> None:
        """Test checking metrics against SLA using check()."""
        block = SLABlock(max_failure_rate=0.05)
        metrics = SLAMetrics(passed_count=90, failed_count=10)
        violations = block.check(metrics)
        assert isinstance(violations, list)
        # Should have violation since 10% > 5%
        assert len(violations) >= 1

    def test_check_result(self) -> None:
        """Test checking result dict against SLA."""
        block = SLABlock(max_failure_rate=0.05)
        result_dict = {"passed_count": 90, "failed_count": 10}
        violations = block.check_result(result_dict)
        assert isinstance(violations, list)
        # Should have violation since 10% > 5%
        assert len(violations) >= 1
