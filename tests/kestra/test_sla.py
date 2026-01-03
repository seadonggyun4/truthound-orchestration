"""Tests for truthound_kestra.sla module."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound_kestra.sla import (
    # Enums
    AlertLevel,
    SLAViolationType,
    # Configuration
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    # Presets
    DEFAULT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    # Monitor
    SLAMonitor,
    SLAEvaluationResult,
    SLARegistry,
    # Hooks
    SLAHookProtocol,
    BaseSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CallbackSLAHook,
    CompositeSLAHook,
    KestraNotificationHook,
    # Functions
    get_sla_registry,
    reset_sla_registry,
    register_sla,
    evaluate_sla,
)


class TestEnums:
    """Tests for SLA enums."""

    def test_alert_level_values(self) -> None:
        """Test AlertLevel enum values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_alert_level_comparison(self) -> None:
        """Test AlertLevel comparison."""
        assert AlertLevel.CRITICAL > AlertLevel.ERROR
        assert AlertLevel.ERROR > AlertLevel.WARNING
        assert AlertLevel.WARNING > AlertLevel.INFO

    def test_sla_violation_type_values(self) -> None:
        """Test SLAViolationType enum values."""
        assert SLAViolationType.FAILURE_RATE_EXCEEDED.value == "failure_rate_exceeded"
        assert SLAViolationType.PASS_RATE_BELOW_MINIMUM.value == "pass_rate_below_minimum"
        assert SLAViolationType.EXECUTION_TIME_EXCEEDED.value == "execution_time_exceeded"
        assert SLAViolationType.ROW_COUNT_BELOW_MINIMUM.value == "row_count_below_minimum"
        assert SLAViolationType.CONSECUTIVE_FAILURES.value == "consecutive_failures"


class TestSLAConfig:
    """Tests for SLAConfig."""

    def test_sla_config_defaults(self) -> None:
        """Test SLAConfig default values."""
        config = SLAConfig()

        assert config.min_pass_rate is None
        assert config.max_failure_rate is None
        assert config.max_execution_time_seconds is None
        assert config.max_consecutive_failures == 3

    def test_sla_config_custom(self) -> None:
        """Test SLAConfig with custom values."""
        config = SLAConfig(
            min_pass_rate=0.99,
            max_failure_rate=0.01,
            max_execution_time_seconds=600.0,
            min_row_count=1000,
            max_consecutive_failures=5,
        )

        assert config.min_pass_rate == 0.99
        assert config.max_failure_rate == 0.01
        assert config.max_execution_time_seconds == 600.0
        assert config.min_row_count == 1000
        assert config.max_consecutive_failures == 5

    def test_sla_config_builder(self) -> None:
        """Test SLAConfig builder pattern."""
        config = SLAConfig()
        config = config.with_pass_rate(0.98)
        config = config.with_execution_time(120.0)
        config = config.with_failure_rate(0.02)

        assert config.min_pass_rate == 0.98
        assert config.max_execution_time_seconds == 120.0
        assert config.max_failure_rate == 0.02

    def test_sla_config_immutability(self) -> None:
        """Test that SLAConfig is immutable."""
        config = SLAConfig()
        with pytest.raises(AttributeError):
            config.min_pass_rate = 0.5  # type: ignore

    def test_sla_config_to_dict(self) -> None:
        """Test SLAConfig serialization."""
        config = SLAConfig(min_pass_rate=0.99)
        d = config.to_dict()

        assert d["min_pass_rate"] == 0.99

    def test_sla_config_from_dict(self) -> None:
        """Test SLAConfig deserialization."""
        data = {"min_pass_rate": 0.99, "max_execution_time_seconds": 600.0}
        config = SLAConfig.from_dict(data)

        assert config.min_pass_rate == 0.99

    def test_preset_configs(self) -> None:
        """Test preset SLA configurations."""
        # DEFAULT has no thresholds set
        assert DEFAULT_SLA_CONFIG.min_pass_rate is None
        # STRICT has high requirements
        assert STRICT_SLA_CONFIG.min_pass_rate == 0.99
        assert STRICT_SLA_CONFIG.max_failure_rate == 0.01
        # LENIENT has lower requirements
        assert LENIENT_SLA_CONFIG.min_pass_rate == 0.90
        assert LENIENT_SLA_CONFIG.max_failure_rate == 0.10
        # PRODUCTION has balanced settings
        assert PRODUCTION_SLA_CONFIG.min_pass_rate == 0.95
        assert PRODUCTION_SLA_CONFIG.max_consecutive_failures == 3

    def test_sla_config_validation(self) -> None:
        """Test SLAConfig validation."""
        # Invalid pass rate
        with pytest.raises(ValueError):
            SLAConfig(min_pass_rate=1.5)

        with pytest.raises(ValueError):
            SLAConfig(max_failure_rate=-0.1)


class TestSLAMetrics:
    """Tests for SLAMetrics."""

    def test_sla_metrics_creation(self) -> None:
        """Test SLAMetrics creation."""
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=120000.0,
            row_count=10000,
        )

        assert metrics.passed_count == 95
        assert metrics.failed_count == 5
        assert metrics.execution_time_ms == 120000.0
        assert metrics.row_count == 10000

    def test_sla_metrics_computed_properties(self) -> None:
        """Test SLAMetrics computed properties."""
        metrics = SLAMetrics(
            passed_count=95,
            failed_count=5,
            execution_time_ms=120000.0,
        )

        assert metrics.pass_rate == pytest.approx(0.95)
        assert metrics.failure_rate == pytest.approx(0.05)
        assert metrics.execution_time_seconds == pytest.approx(120.0)
        assert metrics.total_count == 100

    def test_sla_metrics_with_timestamp(self) -> None:
        """Test SLAMetrics with timestamp."""
        now = datetime.now(timezone.utc)
        metrics = SLAMetrics(
            passed_count=100,
            failed_count=0,
            execution_time_ms=60000.0,
            timestamp=now,
        )

        assert metrics.timestamp == now

    def test_sla_metrics_to_dict(self) -> None:
        """Test SLAMetrics serialization."""
        metrics = SLAMetrics(passed_count=95, failed_count=5, execution_time_ms=120000.0)
        d = metrics.to_dict()

        assert d["passed_count"] == 95
        assert d["failed_count"] == 5
        assert d["execution_time_ms"] == 120000.0
        assert d["pass_rate"] == 0.95

    def test_sla_metrics_from_dict(self) -> None:
        """Test SLAMetrics deserialization."""
        data = {"passed_count": 95, "failed_count": 5, "execution_time_ms": 120000.0}
        metrics = SLAMetrics.from_dict(data)

        assert metrics.passed_count == 95
        assert metrics.pass_rate == 0.95

    def test_sla_metrics_from_check_result(self) -> None:
        """Test SLAMetrics.from_check_result."""
        result = {
            "passed_count": 95,
            "failed_count": 5,
            "warning_count": 2,
            "execution_time_ms": 150.0,
            "total_rows": 1000,
        }
        metrics = SLAMetrics.from_check_result(result, flow_id="test_flow")

        assert metrics.passed_count == 95
        assert metrics.failed_count == 5
        assert metrics.flow_id == "test_flow"


class TestSLAViolation:
    """Tests for SLAViolation."""

    def test_sla_violation_creation(self) -> None:
        """Test SLAViolation creation."""
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Pass rate below threshold",
            threshold=0.95,
            actual=0.90,
            alert_level=AlertLevel.ERROR,
        )

        assert violation.violation_type == SLAViolationType.PASS_RATE_BELOW_MINIMUM
        assert violation.threshold == 0.95
        assert violation.actual == 0.90
        assert violation.alert_level == AlertLevel.ERROR

    def test_sla_violation_to_dict(self) -> None:
        """Test SLAViolation serialization."""
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Test",
            threshold=0.95,
            actual=0.90,
        )
        d = violation.to_dict()

        assert d["violation_type"] == "pass_rate_below_minimum"
        assert d["threshold"] == 0.95
        assert d["actual"] == 0.90


class TestSLAMonitor:
    """Tests for SLAMonitor."""

    def test_monitor_creation(self) -> None:
        """Test SLAMonitor creation."""
        config = SLAConfig()
        monitor = SLAMonitor(config)

        assert monitor.config == config
        assert monitor.consecutive_failures == 0

    def test_monitor_evaluate_pass(self) -> None:
        """Test SLAMonitor evaluation with passing metrics."""
        config = SLAConfig(
            min_pass_rate=0.95,
            max_execution_time_seconds=300.0,
        )
        monitor = SLAMonitor(config)
        metrics = SLAMetrics(
            passed_count=98,
            failed_count=2,
            execution_time_ms=120000.0,
        )

        result = monitor.evaluate(metrics)

        assert result.is_compliant is True
        assert len(result.violations) == 0

    def test_monitor_evaluate_fail_pass_rate(self) -> None:
        """Test SLAMonitor evaluation with failing pass rate."""
        config = SLAConfig(min_pass_rate=0.95)
        monitor = SLAMonitor(config)
        metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=60000.0)

        result = monitor.evaluate(metrics)

        assert result.is_compliant is False
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == SLAViolationType.PASS_RATE_BELOW_MINIMUM

    def test_monitor_evaluate_fail_execution_time(self) -> None:
        """Test SLAMonitor evaluation with failing execution time."""
        config = SLAConfig(max_execution_time_seconds=60.0)
        monitor = SLAMonitor(config)
        metrics = SLAMetrics(passed_count=100, failed_count=0, execution_time_ms=120000.0)

        result = monitor.evaluate(metrics)

        assert result.is_compliant is False
        assert any(
            v.violation_type == SLAViolationType.EXECUTION_TIME_EXCEEDED
            for v in result.violations
        )

    def test_monitor_consecutive_failures_tracking(self) -> None:
        """Test SLAMonitor tracks consecutive failures."""
        config = SLAConfig(min_pass_rate=0.95, max_consecutive_failures=3)
        monitor = SLAMonitor(config)

        # First failure
        metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=60000.0)
        monitor.evaluate(metrics)
        assert monitor.consecutive_failures == 1

        # Second failure
        monitor.evaluate(metrics)
        assert monitor.consecutive_failures == 2

        # Pass resets counter
        passing_metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)
        monitor.evaluate(passing_metrics)
        assert monitor.consecutive_failures == 0

    def test_monitor_with_hooks(self) -> None:
        """Test SLAMonitor with hooks."""
        config = SLAConfig(min_pass_rate=0.95)
        hook = MetricsSLAHook()
        monitor = SLAMonitor(config, hooks=[hook])

        # Pass
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)
        monitor.evaluate(metrics)
        assert hook.pass_count == 1

        # Fail
        failing_metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=60000.0)
        monitor.evaluate(failing_metrics)
        assert hook.violation_count == 1

    def test_monitor_reset(self) -> None:
        """Test SLAMonitor reset."""
        config = SLAConfig(min_pass_rate=0.95)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=60000.0)
        monitor.evaluate(metrics)
        assert monitor.consecutive_failures == 1

        monitor.reset_consecutive_failures()
        assert monitor.consecutive_failures == 0

    def test_monitor_clear_history(self) -> None:
        """Test SLAMonitor clear_history."""
        config = SLAConfig(min_pass_rate=0.95)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)
        monitor.evaluate(metrics)
        assert len(monitor.history) == 1

        monitor.clear_history()
        assert len(monitor.history) == 0


class TestSLAEvaluationResult:
    """Tests for SLAEvaluationResult."""

    def test_result_creation_passed(self) -> None:
        """Test SLAEvaluationResult for passed evaluation."""
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)
        result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[],
            consecutive_failures=0,
        )

        assert result.is_compliant is True
        assert len(result.violations) == 0
        assert result.consecutive_failures == 0

    def test_result_creation_failed(self) -> None:
        """Test SLAEvaluationResult for failed evaluation."""
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=60000.0)
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Pass rate below threshold",
            threshold=0.95,
            actual=0.90,
        )
        result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[violation],
            consecutive_failures=1,
        )

        assert result.is_compliant is False
        assert len(result.violations) == 1

    def test_result_max_alert_level(self) -> None:
        """Test SLAEvaluationResult max_alert_level property."""
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=400000.0)
        violations = [
            SLAViolation(
                violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
                message="Pass rate violation",
                threshold=0.95,
                actual=0.90,
                alert_level=AlertLevel.WARNING,
            ),
            SLAViolation(
                violation_type=SLAViolationType.EXECUTION_TIME_EXCEEDED,
                message="Execution time violation",
                threshold=300.0,
                actual=400.0,
                alert_level=AlertLevel.ERROR,
            ),
        ]
        result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=violations,
            consecutive_failures=1,
        )

        assert result.max_alert_level == AlertLevel.ERROR

    def test_result_has_critical_violations(self) -> None:
        """Test SLAEvaluationResult has_critical_violations property."""
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=60000.0)

        # No critical violations
        result1 = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[
                SLAViolation(
                    violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
                    message="Test",
                    alert_level=AlertLevel.WARNING,
                ),
            ],
            consecutive_failures=0,
        )
        assert result1.has_critical_violations is False

        # Has critical violation
        result2 = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[
                SLAViolation(
                    violation_type=SLAViolationType.CONSECUTIVE_FAILURES,
                    message="Test",
                    alert_level=AlertLevel.CRITICAL,
                ),
            ],
            consecutive_failures=0,
        )
        assert result2.has_critical_violations is True


class TestSLAHooks:
    """Tests for SLA hooks."""

    def test_logging_hook(self) -> None:
        """Test LoggingSLAHook."""
        hook = LoggingSLAHook(log_level="info")

        # Should not raise
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)
        result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[],
            consecutive_failures=0,
        )
        hook.on_sla_pass(result)

    def test_metrics_hook(self) -> None:
        """Test MetricsSLAHook."""
        hook = MetricsSLAHook()
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)

        # Pass
        pass_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[],
            consecutive_failures=0,
        )
        hook.on_sla_pass(pass_result)
        assert hook.pass_count == 1

        # Violation
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Test",
            threshold=0.95,
            actual=0.90,
        )
        fail_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[violation],
            consecutive_failures=1,
        )
        hook.on_sla_violation(fail_result)
        assert hook.violation_count == 1
        assert hook.total_violations == 1

    def test_metrics_hook_pass_rate(self) -> None:
        """Test MetricsSLAHook pass_rate property."""
        hook = MetricsSLAHook()
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)

        pass_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[],
            consecutive_failures=0,
        )

        hook.on_sla_pass(pass_result)
        hook.on_sla_pass(pass_result)

        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Test",
            threshold=0.95,
            actual=0.90,
        )
        fail_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[violation],
            consecutive_failures=1,
        )
        hook.on_sla_violation(fail_result)

        # 2 passes, 1 failure = 2/3 ≈ 0.67
        assert 0.6 < hook.pass_rate < 0.7

    def test_metrics_hook_reset(self) -> None:
        """Test MetricsSLAHook reset."""
        hook = MetricsSLAHook()
        config = SLAConfig()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)

        pass_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[],
            consecutive_failures=0,
        )
        hook.on_sla_pass(pass_result)
        assert hook.pass_count == 1

        hook.reset()
        assert hook.pass_count == 0
        assert hook.violation_count == 0

    def test_callback_hook(self) -> None:
        """Test CallbackSLAHook."""
        pass_called = []
        violation_called = []

        hook = CallbackSLAHook(
            on_pass=lambda r: pass_called.append(r),
            on_violation=lambda r: violation_called.append(r),
        )

        config = SLAConfig()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)

        pass_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[],
            consecutive_failures=0,
        )
        hook.on_sla_pass(pass_result)
        assert len(pass_called) == 1

        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Test",
            threshold=0.95,
            actual=0.90,
        )
        fail_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[violation],
            consecutive_failures=1,
        )
        hook.on_sla_violation(fail_result)
        assert len(violation_called) == 1

    def test_composite_hook(self) -> None:
        """Test CompositeSLAHook."""
        hook1 = MetricsSLAHook()
        hook2 = MetricsSLAHook()
        composite = CompositeSLAHook([hook1, hook2])

        config = SLAConfig()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)

        pass_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[],
            consecutive_failures=0,
        )
        composite.on_sla_pass(pass_result)

        assert hook1.pass_count == 1
        assert hook2.pass_count == 1

    def test_composite_hook_add_remove(self) -> None:
        """Test CompositeSLAHook add/remove."""
        hook1 = MetricsSLAHook()
        composite = CompositeSLAHook([hook1])

        hook2 = LoggingSLAHook()  # Use different hook type for identity
        composite.add_hook(hook2)

        assert composite.remove_hook(hook2) is True
        assert composite.remove_hook(hook2) is False  # Already removed

    def test_kestra_notification_hook(self) -> None:
        """Test KestraNotificationHook."""
        hook = KestraNotificationHook(
            channel="slack",
            min_alert_level=AlertLevel.WARNING,
        )

        config = SLAConfig()
        metrics = SLAMetrics(passed_count=90, failed_count=10, execution_time_ms=60000.0)

        # Should not raise even without Kestra SDK
        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Test",
            threshold=0.95,
            actual=0.90,
            alert_level=AlertLevel.ERROR,
        )
        fail_result = SLAEvaluationResult(
            config=config,
            metrics=metrics,
            violations=[violation],
            consecutive_failures=1,
        )
        hook.on_sla_violation(fail_result)


class TestSLARegistry:
    """Tests for SLARegistry."""

    def test_registry_creation(self) -> None:
        """Test SLARegistry creation."""
        registry = SLARegistry()
        assert len(registry.list_names()) == 0

    def test_registry_register(self) -> None:
        """Test SLARegistry register."""
        registry = SLARegistry()
        config = SLAConfig()

        monitor = registry.register("test_monitor", config)
        assert "test_monitor" in registry.list_names()
        assert isinstance(monitor, SLAMonitor)

    def test_registry_get(self) -> None:
        """Test SLARegistry get."""
        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test", config)
        retrieved = registry.get("test")

        assert retrieved is not None
        assert isinstance(retrieved, SLAMonitor)

    def test_registry_get_not_found(self) -> None:
        """Test SLARegistry get with non-existent monitor."""
        registry = SLARegistry()

        # Returns None instead of raising KeyError
        result = registry.get("nonexistent")
        assert result is None

    def test_registry_remove(self) -> None:
        """Test SLARegistry remove."""
        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test", config)
        assert registry.remove("test") is True
        assert registry.remove("test") is False  # Already removed

    def test_registry_evaluate(self) -> None:
        """Test SLARegistry evaluate."""
        registry = SLARegistry()
        config = SLAConfig(min_pass_rate=0.95)

        registry.register("test", config)
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)

        result = registry.evaluate("test", metrics)
        assert result is not None
        assert result.is_compliant is True

    def test_registry_evaluate_not_found(self) -> None:
        """Test SLARegistry evaluate with non-existent SLA."""
        registry = SLARegistry()
        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)

        result = registry.evaluate("nonexistent", metrics)
        assert result is None

    def test_registry_clear(self) -> None:
        """Test SLARegistry clear."""
        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test1", config)
        registry.register("test2", config)
        assert len(registry.list_names()) == 2

        registry.clear()
        assert len(registry.list_names()) == 0


class TestGlobalFunctions:
    """Tests for global SLA functions."""

    def test_get_sla_registry(self) -> None:
        """Test get_sla_registry returns singleton."""
        registry1 = get_sla_registry()
        registry2 = get_sla_registry()

        assert registry1 is registry2

    def test_reset_sla_registry(self) -> None:
        """Test reset_sla_registry."""
        registry = get_sla_registry()
        config = SLAConfig()
        registry.register("test", config)

        reset_sla_registry()

        new_registry = get_sla_registry()
        assert len(new_registry.list_names()) == 0

    def test_register_sla(self) -> None:
        """Test register_sla convenience function."""
        reset_sla_registry()
        config = SLAConfig()

        register_sla("test", config)

        registry = get_sla_registry()
        assert "test" in registry.list_names()

    def test_evaluate_sla(self) -> None:
        """Test evaluate_sla convenience function."""
        reset_sla_registry()
        config = SLAConfig(min_pass_rate=0.95)
        register_sla("test", config)

        metrics = SLAMetrics(passed_count=98, failed_count=2, execution_time_ms=60000.0)
        result = evaluate_sla("test", metrics)

        assert result is not None
        assert result.is_compliant is True
