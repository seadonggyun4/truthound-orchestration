"""Tests for SLA module."""

from __future__ import annotations

import pytest

from truthound_dagster.sla.config import (
    DEFAULT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)
from truthound_dagster.sla.hooks import (
    CompositeSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    SLAHook,
)
from truthound_dagster.sla.monitor import (
    SLAMonitor,
    SLARegistry,
    get_sla_registry,
    reset_sla_registry,
)


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_values(self) -> None:
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestSLAViolationType:
    """Tests for SLAViolationType enum."""

    def test_values(self) -> None:
        assert SLAViolationType.FAILURE_RATE_EXCEEDED.value == "failure_rate_exceeded"
        assert SLAViolationType.PASS_RATE_BELOW_MINIMUM.value == "pass_rate_below_minimum"
        assert SLAViolationType.EXECUTION_TIME_EXCEEDED.value == "execution_time_exceeded"
        assert SLAViolationType.ROW_COUNT_BELOW_MINIMUM.value == "row_count_below_minimum"
        assert SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM.value == "row_count_above_maximum"
        assert SLAViolationType.CONSECUTIVE_FAILURES.value == "consecutive_failures"
        assert SLAViolationType.CUSTOM.value == "custom"


class TestSLAConfig:
    """Tests for SLAConfig."""

    def test_default_creation(self) -> None:
        config = SLAConfig()
        assert config.max_failure_rate is None
        assert config.min_pass_rate is None
        assert config.max_execution_time_seconds is None
        assert config.enabled is True
        assert config.alert_level == AlertLevel.ERROR

    def test_with_failure_rate(self) -> None:
        config = SLAConfig().with_failure_rate(0.05)
        assert config.max_failure_rate == 0.05

    def test_with_pass_rate(self) -> None:
        config = SLAConfig().with_pass_rate(0.95)
        assert config.min_pass_rate == 0.95

    def test_with_execution_time(self) -> None:
        config = SLAConfig().with_execution_time(300.0)
        assert config.max_execution_time_seconds == 300.0

    def test_with_row_count_range(self) -> None:
        config = SLAConfig().with_row_count_range(min_count=100, max_count=10000)
        assert config.min_row_count == 100
        assert config.max_row_count == 10000

    def test_with_alert_level(self) -> None:
        config = SLAConfig().with_alert_level(AlertLevel.CRITICAL)
        assert config.alert_level == AlertLevel.CRITICAL

    def test_with_enabled(self) -> None:
        config = SLAConfig().with_enabled(False)
        assert config.enabled is False

    def test_to_dict(self) -> None:
        config = SLAConfig(max_failure_rate=0.05)
        data = config.to_dict()
        assert data["max_failure_rate"] == 0.05
        assert data["enabled"] is True
        assert data["alert_level"] == "error"

    def test_invalid_failure_rate(self) -> None:
        with pytest.raises(ValueError):
            SLAConfig(max_failure_rate=1.5)

    def test_invalid_pass_rate(self) -> None:
        with pytest.raises(ValueError):
            SLAConfig(min_pass_rate=-0.1)

    def test_invalid_execution_time(self) -> None:
        with pytest.raises(ValueError):
            SLAConfig(max_execution_time_seconds=-1.0)

    def test_invalid_row_count_range(self) -> None:
        with pytest.raises(ValueError):
            SLAConfig(min_row_count=100, max_row_count=50)

    def test_immutability(self) -> None:
        config = SLAConfig()
        with pytest.raises(AttributeError):
            config.max_failure_rate = 0.1  # type: ignore

    def test_method_chaining(self) -> None:
        config = (
            SLAConfig()
            .with_failure_rate(0.05)
            .with_pass_rate(0.95)
            .with_execution_time(300.0)
            .with_enabled(True)
        )
        assert config.max_failure_rate == 0.05
        assert config.min_pass_rate == 0.95
        assert config.max_execution_time_seconds == 300.0


class TestSLAMetrics:
    """Tests for SLAMetrics."""

    def test_creation(self) -> None:
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            warning_count=2,
            execution_time_ms=1000.0,
        )
        assert metrics.passed_count == 95
        assert metrics.failed_count == 5
        assert metrics.warning_count == 2

    def test_total_count(self) -> None:
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        assert metrics.total_count == 100

    def test_pass_rate(self) -> None:
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        assert metrics.pass_rate == 0.95

    def test_failure_rate(self) -> None:
        metrics = SLAMetrics(passed_count=95, failed_count=5)
        assert metrics.failure_rate == 0.05

    def test_execution_time_seconds(self) -> None:
        metrics = SLAMetrics(execution_time_ms=1500.0)
        assert metrics.execution_time_seconds == 1.5

    def test_empty_metrics_rates(self) -> None:
        metrics = SLAMetrics()
        assert metrics.pass_rate == 1.0
        assert metrics.failure_rate == 0.0

    def test_from_check_result(self) -> None:
        result = {
            "passed_count": 95,
            "failed_count": 5,
            "warning_count": 2,
            "execution_time_ms": 1000.0,
            "row_count": 10000,
        }
        metrics = SLAMetrics.from_check_result(result, asset_key="test_asset")
        assert metrics.passed_count == 95
        assert metrics.failed_count == 5
        assert metrics.row_count == 10000
        assert metrics.asset_key == "test_asset"

    def test_to_dict(self) -> None:
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=1000.0,
        )
        data = metrics.to_dict()
        assert data["passed_count"] == 95
        assert data["total_count"] == 100
        assert data["pass_rate"] == 0.95
        assert data["failure_rate"] == 0.05


class TestSLAViolation:
    """Tests for SLAViolation."""

    def test_creation(self) -> None:
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Failure rate exceeded",
            threshold=0.05,
            actual=0.10,
            alert_level=AlertLevel.CRITICAL,
        )
        assert violation.violation_type == SLAViolationType.FAILURE_RATE_EXCEEDED
        assert violation.alert_level == AlertLevel.CRITICAL
        assert violation.threshold == 0.05
        assert violation.actual == 0.10

    def test_to_dict(self) -> None:
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test",
            threshold=0.05,
            actual=0.10,
            alert_level=AlertLevel.WARNING,
        )
        data = violation.to_dict()
        assert data["violation_type"] == "failure_rate_exceeded"
        assert data["alert_level"] == "warning"
        assert data["threshold"] == 0.05
        assert data["actual"] == 0.10

    def test_with_metadata(self) -> None:
        violation = SLAViolation(
            violation_type=SLAViolationType.CUSTOM,
            message="Custom violation",
            metadata={"key": "value"},
        )
        assert violation.metadata["key"] == "value"


class TestSLAPresets:
    """Tests for SLA preset configurations."""

    def test_default_config(self) -> None:
        assert DEFAULT_SLA_CONFIG.enabled is True
        assert DEFAULT_SLA_CONFIG.max_failure_rate is None

    def test_strict_config(self) -> None:
        assert STRICT_SLA_CONFIG.max_failure_rate == 0.01
        assert STRICT_SLA_CONFIG.min_pass_rate == 0.99
        assert STRICT_SLA_CONFIG.max_execution_time_seconds == 60.0
        assert STRICT_SLA_CONFIG.alert_level == AlertLevel.CRITICAL

    def test_lenient_config(self) -> None:
        assert LENIENT_SLA_CONFIG.max_failure_rate == 0.10
        assert LENIENT_SLA_CONFIG.min_pass_rate == 0.90
        assert LENIENT_SLA_CONFIG.alert_level == AlertLevel.WARNING


class TestSLAMonitor:
    """Tests for SLAMonitor."""

    def test_creation(self) -> None:
        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config)
        assert monitor.config == config

    def test_creation_with_name(self) -> None:
        config = SLAConfig()
        monitor = SLAMonitor(config, name="test_monitor")
        assert monitor.name == "test_monitor"

    def test_check_metrics_pass(self) -> None:
        config = SLAConfig(max_failure_rate=0.10)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=100.0,
        )

        violations = monitor.check(metrics)
        assert len(violations) == 0

    def test_check_failure_rate_violation(self) -> None:
        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(
            passed_count=90,
            failed_count=10,
            execution_time_ms=100.0,
        )

        violations = monitor.check(metrics)
        assert len(violations) > 0
        assert any(
            v.violation_type == SLAViolationType.FAILURE_RATE_EXCEEDED
            for v in violations
        )

    def test_check_pass_rate_violation(self) -> None:
        config = SLAConfig(min_pass_rate=0.95)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(
            passed_count=90,
            failed_count=10,
            execution_time_ms=100.0,
        )

        violations = monitor.check(metrics)
        assert len(violations) > 0
        assert any(
            v.violation_type == SLAViolationType.PASS_RATE_BELOW_MINIMUM
            for v in violations
        )

    def test_check_execution_time_violation(self) -> None:
        config = SLAConfig(max_execution_time_seconds=1.0)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            execution_time_ms=2000.0,  # 2 seconds
        )

        violations = monitor.check(metrics)
        assert len(violations) > 0
        assert any(
            v.violation_type == SLAViolationType.EXECUTION_TIME_EXCEEDED
            for v in violations
        )

    def test_check_row_count_violations(self) -> None:
        config = SLAConfig(min_row_count=100, max_row_count=1000)
        monitor = SLAMonitor(config)

        # Below minimum
        metrics_low = SLAMetrics(passed_count=100, failed_count=0, row_count=50)
        violations_low = monitor.check(metrics_low)
        assert any(
            v.violation_type == SLAViolationType.ROW_COUNT_BELOW_MINIMUM
            for v in violations_low
        )

        # Above maximum
        monitor.reset()
        metrics_high = SLAMetrics(passed_count=100, failed_count=0, row_count=2000)
        violations_high = monitor.check(metrics_high)
        assert any(
            v.violation_type == SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM
            for v in violations_high
        )

    def test_disabled_monitor(self) -> None:
        config = SLAConfig(max_failure_rate=0.05, enabled=False)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(
            passed_count=50,
            failed_count=50,
            execution_time_ms=100.0,
        )

        violations = monitor.check(metrics)
        assert len(violations) == 0

    def test_consecutive_failure_tracking(self) -> None:
        config = SLAConfig(max_consecutive_failures=3)
        monitor = SLAMonitor(config)

        assert monitor.consecutive_failures == 0

        monitor.record_failure()
        assert monitor.consecutive_failures == 1

        monitor.record_failure()
        assert monitor.consecutive_failures == 2

        monitor.record_success()
        assert monitor.consecutive_failures == 0

    def test_consecutive_failure_violation(self) -> None:
        config = SLAConfig(max_consecutive_failures=2)
        monitor = SLAMonitor(config)

        monitor.record_failure()
        violations = monitor.check_consecutive_failures()
        assert len(violations) == 0

        monitor.record_failure()
        violations = monitor.check_consecutive_failures()
        assert len(violations) > 0
        assert violations[0].violation_type == SLAViolationType.CONSECUTIVE_FAILURES

    def test_reset(self) -> None:
        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config)

        monitor.record_failure()
        monitor.record_failure()
        assert monitor.consecutive_failures == 2

        monitor.reset()
        assert monitor.consecutive_failures == 0

    def test_get_history(self) -> None:
        config = SLAConfig()
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(passed_count=100, failed_count=0)
        monitor.check(metrics)

        history = monitor.get_history()
        assert len(history) == 1
        assert history[0].passed_count == 100

    def test_get_summary(self) -> None:
        config = SLAConfig()
        monitor = SLAMonitor(config, name="test")

        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            execution_time_ms=100.0,
        )
        monitor.check(metrics)

        summary = monitor.get_summary()
        assert summary["name"] == "test"
        assert summary["history_count"] == 1
        assert summary["average_pass_rate"] == 1.0


class TestSLARegistry:
    """Tests for SLARegistry."""

    def setup_method(self) -> None:
        reset_sla_registry()

    def test_register_and_get(self) -> None:
        registry = SLARegistry()
        config = SLAConfig(max_failure_rate=0.05)

        monitor = registry.register("test", config)
        assert monitor is not None

        retrieved = registry.get("test")
        assert retrieved == monitor

    def test_get_nonexistent(self) -> None:
        registry = SLARegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_register_duplicate_raises(self) -> None:
        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test", config)
        with pytest.raises(ValueError):
            registry.register("test", config)

    def test_register_replace(self) -> None:
        registry = SLARegistry()
        config1 = SLAConfig(max_failure_rate=0.05)
        config2 = SLAConfig(max_failure_rate=0.10)

        registry.register("test", config1)
        registry.register("test", config2, replace=True)

        monitor = registry.get("test")
        assert monitor is not None
        assert monitor.config.max_failure_rate == 0.10

    def test_get_or_create(self) -> None:
        registry = SLARegistry()
        config = SLAConfig(max_failure_rate=0.05)

        monitor1 = registry.get_or_create("test", config)
        monitor2 = registry.get_or_create("test", config)

        assert monitor1 is monitor2

    def test_unregister(self) -> None:
        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test", config)
        removed = registry.unregister("test")
        assert removed is True
        assert registry.get("test") is None

    def test_unregister_nonexistent(self) -> None:
        registry = SLARegistry()
        removed = registry.unregister("nonexistent")
        assert removed is False

    def test_list_names(self) -> None:
        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test1", config)
        registry.register("test2", config)

        names = registry.list_names()
        assert "test1" in names
        assert "test2" in names

    def test_get_all(self) -> None:
        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test1", config)
        registry.register("test2", config)

        all_monitors = registry.get_all()
        assert "test1" in all_monitors
        assert "test2" in all_monitors

    def test_check_all(self) -> None:
        registry = SLARegistry()
        config = SLAConfig(max_failure_rate=0.05)

        registry.register("test1", config)
        registry.register("test2", config)

        metrics_by_name = {
            "test1": SLAMetrics(passed_count=100, failed_count=0),
            "test2": SLAMetrics(passed_count=80, failed_count=20),
        }

        results = registry.check_all(metrics_by_name)
        assert len(results["test1"]) == 0  # No violations
        assert len(results["test2"]) > 0  # Has violations

    def test_reset_all(self) -> None:
        registry = SLARegistry()
        config = SLAConfig()

        monitor = registry.register("test", config)
        monitor.record_failure()
        assert monitor.consecutive_failures == 1

        registry.reset_all()
        assert monitor.consecutive_failures == 0

    def test_global_registry(self) -> None:
        registry = get_sla_registry()
        assert isinstance(registry, SLARegistry)

        registry2 = get_sla_registry()
        assert registry is registry2

    def test_reset_global_registry(self) -> None:
        registry = get_sla_registry()
        config = SLAConfig()
        registry.register("test", config)

        reset_sla_registry()
        new_registry = get_sla_registry()
        assert new_registry.get("test") is None


class TestSLAHooks:
    """Tests for SLA hooks."""

    def test_logging_hook_on_check(self) -> None:
        hook = LoggingSLAHook()

        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=100.0,
        )

        # Should not raise
        hook.on_check(metrics, [])
        hook.on_success(metrics)

    def test_logging_hook_on_violation(self) -> None:
        hook = LoggingSLAHook()

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test violation",
            threshold=0.05,
            actual=0.10,
        )

        # Should not raise
        hook.on_violation(violation)

    def test_metrics_hook_counts(self) -> None:
        hook = MetricsSLAHook()

        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=100.0,
        )

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test",
            threshold=0.05,
            actual=0.10,
        )

        hook.on_check(metrics, [violation])
        hook.on_violation(violation)

        assert hook.check_count == 1
        assert hook.violation_count == 1

    def test_metrics_hook_success_rate(self) -> None:
        hook = MetricsSLAHook()

        metrics = SLAMetrics(passed_count=100, failed_count=0)

        # Check with no violations = success
        hook.on_check(metrics, [])
        assert hook.success_count == 1
        assert hook.success_rate == 1.0

        # Check with violations
        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test",
        )
        hook.on_check(metrics, [violation])
        assert hook.check_count == 2
        assert hook.success_count == 1
        assert hook.success_rate == 0.5

    def test_metrics_hook_get_stats(self) -> None:
        hook = MetricsSLAHook()

        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            execution_time_ms=100.0,
        )

        hook.on_check(metrics, [])

        stats = hook.get_stats()
        assert stats["check_count"] == 1
        assert stats["success_count"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["average_execution_time_ms"] == 100.0

    def test_metrics_hook_reset(self) -> None:
        hook = MetricsSLAHook()

        metrics = SLAMetrics(passed_count=100, failed_count=0)
        hook.on_check(metrics, [])
        assert hook.check_count == 1

        hook.reset()
        assert hook.check_count == 0
        assert hook.violation_count == 0
        assert hook.success_count == 0

    def test_composite_hook(self) -> None:
        logging_hook = LoggingSLAHook()
        metrics_hook = MetricsSLAHook()

        composite = CompositeSLAHook([logging_hook, metrics_hook])

        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=100.0,
        )

        composite.on_check(metrics, [])
        assert metrics_hook.check_count == 1

    def test_composite_hook_add_remove(self) -> None:
        hook1 = MetricsSLAHook()
        hook2 = MetricsSLAHook()

        composite = CompositeSLAHook([hook1])
        assert len(composite.hooks) == 1

        composite.add_hook(hook2)
        assert len(composite.hooks) == 2

        removed = composite.remove_hook(hook2)
        assert removed is True
        assert len(composite.hooks) == 1

        removed = composite.remove_hook(hook2)
        assert removed is False

    def test_composite_hook_isolates_failures(self) -> None:
        """Test that hook failures are isolated."""

        class FailingHook(SLAHook):
            def on_check(self, metrics, violations, context=None):
                raise RuntimeError("Intentional failure")

            def on_violation(self, violation, context=None):
                raise RuntimeError("Intentional failure")

        failing_hook = FailingHook()
        metrics_hook = MetricsSLAHook()

        composite = CompositeSLAHook([failing_hook, metrics_hook])

        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=100.0,
        )

        # Should not raise, and metrics_hook should still be called
        composite.on_check(metrics, [])
        assert metrics_hook.check_count == 1

    def test_composite_on_success(self) -> None:
        metrics_hook = MetricsSLAHook()
        composite = CompositeSLAHook([metrics_hook])

        metrics = SLAMetrics(passed_count=100, failed_count=0)
        composite.on_success(metrics)
        # on_success doesn't increment counters in MetricsSLAHook

    def test_composite_on_consecutive_failure(self) -> None:
        logging_hook = LoggingSLAHook()
        composite = CompositeSLAHook([logging_hook])

        # Should not raise
        composite.on_consecutive_failure(count=3, threshold=5)
