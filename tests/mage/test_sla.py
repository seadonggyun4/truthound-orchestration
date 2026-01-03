"""Tests for Mage SLA monitoring system."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from truthound_mage.sla import (
    # Config
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
    # Presets
    DEFAULT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    # Monitor
    SLAMonitor,
    SLARegistry,
    # Hooks
    BaseSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
)


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.INFO.value == "info"


class TestSLAConfig:
    """Tests for SLAConfig."""

    def test_default_values(self) -> None:
        """Test default SLA configuration."""
        config = SLAConfig()
        assert config.min_pass_rate is None
        assert config.max_failure_rate is None
        assert config.max_execution_time_seconds is None
        assert config.max_consecutive_failures == 3
        assert config.enabled is True

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = SLAConfig()
        with pytest.raises(AttributeError):
            config.min_pass_rate = 0.5  # type: ignore[misc]

    def test_with_pass_rate(self) -> None:
        """Test pass rate builder."""
        config = SLAConfig()
        new_config = config.with_pass_rate(0.99)
        assert new_config.min_pass_rate == 0.99
        assert config.min_pass_rate is None  # Original unchanged

    def test_with_failure_rate(self) -> None:
        """Test failure rate builder."""
        config = SLAConfig()
        new_config = config.with_failure_rate(0.05)
        assert new_config.max_failure_rate == 0.05

    def test_with_execution_time(self) -> None:
        """Test execution time builder."""
        config = SLAConfig()
        new_config = config.with_execution_time(60.0)
        assert new_config.max_execution_time_seconds == 60.0

    def test_with_row_count_range(self) -> None:
        """Test row count range builder."""
        config = SLAConfig()
        new_config = config.with_row_count_range(min_count=100, max_count=10000)
        assert new_config.min_row_count == 100
        assert new_config.max_row_count == 10000

    def test_with_alert_level(self) -> None:
        """Test alert level builder."""
        config = SLAConfig()
        new_config = config.with_alert_level(AlertLevel.CRITICAL)
        assert new_config.alert_level == AlertLevel.CRITICAL

    def test_with_enabled(self) -> None:
        """Test enabled builder."""
        config = SLAConfig()
        new_config = config.with_enabled(False)
        assert new_config.enabled is False

    def test_pass_rate_validation(self) -> None:
        """Test pass rate validation."""
        with pytest.raises(ValueError):
            SLAConfig(min_pass_rate=1.5)

    def test_failure_rate_validation(self) -> None:
        """Test failure rate validation."""
        with pytest.raises(ValueError):
            SLAConfig(max_failure_rate=-0.1)

    def test_execution_time_validation(self) -> None:
        """Test execution time validation."""
        with pytest.raises(ValueError):
            SLAConfig(max_execution_time_seconds=-1.0)

    def test_row_count_validation(self) -> None:
        """Test row count validation."""
        with pytest.raises(ValueError):
            SLAConfig(min_row_count=100, max_row_count=50)

    def test_preset_configs(self) -> None:
        """Test preset configurations."""
        # DEFAULT has no thresholds set
        assert DEFAULT_SLA_CONFIG.min_pass_rate is None

        # STRICT has tight thresholds
        assert STRICT_SLA_CONFIG.min_pass_rate == 0.99
        assert STRICT_SLA_CONFIG.max_failure_rate == 0.01
        assert STRICT_SLA_CONFIG.alert_level == AlertLevel.CRITICAL

        # LENIENT has loose thresholds
        assert LENIENT_SLA_CONFIG.min_pass_rate == 0.90
        assert LENIENT_SLA_CONFIG.max_failure_rate == 0.10

        # PRODUCTION has balanced thresholds
        assert PRODUCTION_SLA_CONFIG.min_pass_rate == 0.95
        assert PRODUCTION_SLA_CONFIG.max_failure_rate == 0.05

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = SLAConfig(min_pass_rate=0.9)
        data = config.to_dict()
        assert data["min_pass_rate"] == 0.9

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "min_pass_rate": 0.95,
            "max_failure_rate": 0.05,
            "alert_level": "error",
        }
        config = SLAConfig.from_dict(data)
        assert config.min_pass_rate == 0.95
        assert config.max_failure_rate == 0.05
        assert config.alert_level == AlertLevel.ERROR


class TestSLAMetrics:
    """Tests for SLAMetrics."""

    def test_creation(self) -> None:
        """Test metrics creation."""
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=1000.0,
        )
        assert metrics.passed_count == 95
        assert metrics.failed_count == 5
        assert metrics.execution_time_ms == 1000.0

    def test_computed_properties(self) -> None:
        """Test computed properties."""
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
        )
        assert metrics.total_count == 100
        assert metrics.pass_rate == 0.95
        assert metrics.failure_rate == 0.05

    def test_execution_time_seconds(self) -> None:
        """Test execution time in seconds."""
        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            execution_time_ms=1500.0,
        )
        assert metrics.execution_time_seconds == 1.5

    def test_empty_metrics(self) -> None:
        """Test metrics with zero counts."""
        metrics = SLAMetrics(
            passed_count=0,
            failed_count=0,
        )
        assert metrics.total_count == 0
        assert metrics.pass_rate == 1.0  # Default to 1.0 for empty
        assert metrics.failure_rate == 0.0

    def test_with_context(self) -> None:
        """Test metrics with context info."""
        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            block_uuid="block_1",
            pipeline_uuid="pipeline_1",
            run_id="run_123",
        )
        assert metrics.block_uuid == "block_1"
        assert metrics.pipeline_uuid == "pipeline_1"
        assert metrics.run_id == "run_123"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            block_uuid="test",
        )
        data = metrics.to_dict()
        assert data["passed_count"] == 95
        assert data["failed_count"] == 5
        assert data["total_count"] == 100
        assert data["pass_rate"] == 0.95
        assert data["block_uuid"] == "test"

    def test_from_check_result(self) -> None:
        """Test creation from check result."""
        result = {
            "passed_count": 90,
            "failed_count": 10,
            "execution_time_ms": 500.0,
        }
        metrics = SLAMetrics.from_check_result(
            result,
            block_uuid="block_1",
        )
        assert metrics.passed_count == 90
        assert metrics.failed_count == 10
        assert metrics.block_uuid == "block_1"


class TestSLAViolation:
    """Tests for SLAViolation."""

    def test_creation(self) -> None:
        """Test violation creation."""
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            alert_level=AlertLevel.CRITICAL,
            message="Pass rate below threshold",
            threshold=0.95,
            actual=0.85,
        )
        assert violation.violation_type == SLAViolationType.PASS_RATE_BELOW_MINIMUM
        assert violation.alert_level == AlertLevel.CRITICAL
        assert violation.threshold == 0.95
        assert violation.actual == 0.85

    def test_with_context(self) -> None:
        """Test violation with context."""
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test",
            block_uuid="block_1",
            pipeline_uuid="pipeline_1",
        )
        assert violation.block_uuid == "block_1"
        assert violation.pipeline_uuid == "pipeline_1"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            alert_level=AlertLevel.ERROR,
            message="Test violation",
            threshold=0.95,
            actual=0.90,
        )
        data = violation.to_dict()

        assert data["violation_type"] == "pass_rate_below_minimum"
        assert data["alert_level"] == "error"
        assert data["threshold"] == 0.95
        assert data["actual"] == 0.90


class TestSLAViolationType:
    """Tests for SLAViolationType enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert SLAViolationType.FAILURE_RATE_EXCEEDED.value == "failure_rate_exceeded"
        assert SLAViolationType.PASS_RATE_BELOW_MINIMUM.value == "pass_rate_below_minimum"
        assert SLAViolationType.EXECUTION_TIME_EXCEEDED.value == "execution_time_exceeded"
        assert SLAViolationType.ROW_COUNT_BELOW_MINIMUM.value == "row_count_below_minimum"
        assert SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM.value == "row_count_above_maximum"
        assert SLAViolationType.CONSECUTIVE_FAILURES.value == "consecutive_failures"


class TestSLAMonitor:
    """Tests for SLAMonitor."""

    def test_creation(self) -> None:
        """Test monitor creation."""
        monitor = SLAMonitor(config=DEFAULT_SLA_CONFIG)
        assert monitor.config == DEFAULT_SLA_CONFIG

    def test_check_passing(self) -> None:
        """Test check with passing metrics."""
        config = SLAConfig(min_pass_rate=0.90)
        monitor = SLAMonitor(config=config)
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=1000.0,
        )

        violations = monitor.check(metrics)

        assert len(violations) == 0

    def test_check_pass_rate_violation(self) -> None:
        """Test check with pass rate violation."""
        config = SLAConfig(min_pass_rate=0.95)
        monitor = SLAMonitor(config=config)
        metrics = SLAMetrics(
            passed_count=80,
            failed_count=20,
        )

        violations = monitor.check(metrics)

        assert len(violations) > 0
        assert any(
            v.violation_type == SLAViolationType.PASS_RATE_BELOW_MINIMUM
            for v in violations
        )

    def test_check_failure_rate_violation(self) -> None:
        """Test check with failure rate violation."""
        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config=config)
        metrics = SLAMetrics(
            passed_count=90,
            failed_count=10,  # 10% failure rate
        )

        violations = monitor.check(metrics)

        assert any(
            v.violation_type == SLAViolationType.FAILURE_RATE_EXCEEDED
            for v in violations
        )

    def test_check_execution_time_violation(self) -> None:
        """Test check with execution time violation."""
        config = SLAConfig(max_execution_time_seconds=1.0)
        monitor = SLAMonitor(config=config)
        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            execution_time_ms=5000.0,  # 5 seconds - exceeds limit
        )

        violations = monitor.check(metrics)

        assert any(
            v.violation_type == SLAViolationType.EXECUTION_TIME_EXCEEDED
            for v in violations
        )

    def test_check_row_count_violations(self) -> None:
        """Test check with row count violations."""
        config = SLAConfig(min_row_count=100, max_row_count=10000)
        monitor = SLAMonitor(config=config)

        # Below minimum
        metrics_low = SLAMetrics(
            passed_count=50,
            failed_count=0,
            row_count=50,
        )
        violations = monitor.check(metrics_low)
        assert any(
            v.violation_type == SLAViolationType.ROW_COUNT_BELOW_MINIMUM
            for v in violations
        )

        # Reset consecutive failures
        monitor.reset_failures()

        # Above maximum
        metrics_high = SLAMetrics(
            passed_count=100,
            failed_count=0,
            row_count=20000,
        )
        violations = monitor.check(metrics_high)
        assert any(
            v.violation_type == SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM
            for v in violations
        )

    def test_consecutive_failures(self) -> None:
        """Test consecutive failure tracking."""
        config = SLAConfig(min_pass_rate=0.99, max_consecutive_failures=2)
        monitor = SLAMonitor(config=config)

        metrics = SLAMetrics(passed_count=90, failed_count=10)

        # First failure
        monitor.check(metrics)
        assert monitor.consecutive_failures == 1

        # Second failure
        monitor.check(metrics)
        assert monitor.consecutive_failures == 2

        # Third failure - should trigger consecutive failure violation
        violations = monitor.check(metrics)
        assert any(
            v.violation_type == SLAViolationType.CONSECUTIVE_FAILURES
            for v in violations
        )

    def test_disabled_config(self) -> None:
        """Test disabled SLA config."""
        config = SLAConfig(min_pass_rate=0.99, enabled=False)
        monitor = SLAMonitor(config=config)
        metrics = SLAMetrics(passed_count=50, failed_count=50)

        violations = monitor.check(metrics)

        assert len(violations) == 0

    def test_get_stats(self) -> None:
        """Test getting monitor stats."""
        monitor = SLAMonitor(config=SLAConfig())
        metrics = SLAMetrics(passed_count=90, failed_count=10)

        monitor.check(metrics)
        stats = monitor.get_stats()

        assert stats["total_checks"] == 1
        assert "avg_pass_rate" in stats


class TestSLARegistry:
    """Tests for SLARegistry."""

    def test_register(self) -> None:
        """Test registering monitor."""
        registry = SLARegistry()
        config = SLAConfig(min_pass_rate=0.95)

        monitor = registry.register("test", config=config)

        assert registry.get("test") is monitor

    def test_get_or_create(self) -> None:
        """Test get or create monitor."""
        registry = SLARegistry()

        monitor = registry.get_or_create("test", config=DEFAULT_SLA_CONFIG)
        monitor2 = registry.get_or_create("test", config=STRICT_SLA_CONFIG)

        # Should return same monitor
        assert monitor is monitor2

    def test_list_monitors(self) -> None:
        """Test listing monitors."""
        registry = SLARegistry()
        registry.register("a", config=DEFAULT_SLA_CONFIG)
        registry.register("b", config=DEFAULT_SLA_CONFIG)

        names = registry.list_monitors()

        assert "a" in names
        assert "b" in names

    def test_check(self) -> None:
        """Test checking via registry."""
        registry = SLARegistry()
        config = SLAConfig(min_pass_rate=0.95)
        registry.register("test", config=config)

        metrics = SLAMetrics(passed_count=80, failed_count=20)
        violations = registry.check("test", metrics)

        assert len(violations) > 0

    def test_remove(self) -> None:
        """Test removing monitor."""
        registry = SLARegistry()
        registry.register("test", config=DEFAULT_SLA_CONFIG)

        assert registry.remove("test") is True
        assert registry.get("test") is None
        assert registry.remove("test") is False

    def test_reset_all(self) -> None:
        """Test resetting all monitors."""
        registry = SLARegistry()
        config = SLAConfig(min_pass_rate=0.99)
        registry.register("test", config=config)

        # Cause some failures
        metrics = SLAMetrics(passed_count=80, failed_count=20)
        registry.check("test", metrics)

        monitor = registry.get("test")
        assert monitor is not None
        assert monitor.consecutive_failures > 0

        registry.reset_all()
        assert monitor.consecutive_failures == 0


class TestSLAHooks:
    """Tests for SLA hooks."""

    def test_logging_hook(self) -> None:
        """Test logging hook."""
        hook = LoggingSLAHook()
        metrics = SLAMetrics(passed_count=95, failed_count=5)

        # Should not raise
        hook.on_check_start(DEFAULT_SLA_CONFIG, metrics)
        hook.on_check_complete(DEFAULT_SLA_CONFIG, metrics, [])

    def test_metrics_hook(self) -> None:
        """Test metrics hook."""
        hook = MetricsSLAHook()
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            alert_level=AlertLevel.WARNING,
            message="Test",
        )

        hook.on_check_start(DEFAULT_SLA_CONFIG, metrics)
        hook.on_violation(violation)
        hook.on_check_complete(DEFAULT_SLA_CONFIG, metrics, [violation])

        assert hook.check_count == 1
        assert hook.violation_count == 1

        stats = hook.get_stats()
        assert stats["check_count"] == 1
        assert stats["violation_count"] == 1

    def test_metrics_hook_reset(self) -> None:
        """Test metrics hook reset."""
        hook = MetricsSLAHook()
        metrics = SLAMetrics(passed_count=95, failed_count=5)

        hook.on_check_start(DEFAULT_SLA_CONFIG, metrics)
        hook.on_check_complete(DEFAULT_SLA_CONFIG, metrics, [])
        hook.reset()

        assert hook.check_count == 0

    def test_composite_hook(self) -> None:
        """Test composite hook."""
        hook1 = MetricsSLAHook()
        hook2 = MetricsSLAHook()
        composite = CompositeSLAHook([hook1, hook2])

        metrics = SLAMetrics(passed_count=95, failed_count=5)
        composite.on_check_start(DEFAULT_SLA_CONFIG, metrics)
        composite.on_check_complete(DEFAULT_SLA_CONFIG, metrics, [])

        # Both hooks should be called
        assert hook1.check_count == 1
        assert hook2.check_count == 1

    def test_composite_hook_add_remove(self) -> None:
        """Test adding/removing hooks from composite."""
        hook1 = MetricsSLAHook()
        hook2 = MetricsSLAHook()
        composite = CompositeSLAHook([hook1])

        composite.add_hook(hook2)
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        composite.on_check_complete(DEFAULT_SLA_CONFIG, metrics, [])

        assert hook1.check_count == 1
        assert hook2.check_count == 1

        # Remove hook2
        result = composite.remove_hook(hook2)
        assert result is True

        composite.on_check_complete(DEFAULT_SLA_CONFIG, metrics, [])
        assert hook1.check_count == 2
        assert hook2.check_count == 1  # Not called again
