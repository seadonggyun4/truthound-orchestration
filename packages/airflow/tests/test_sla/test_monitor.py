"""Tests for SLA monitoring."""

from __future__ import annotations

from typing import Any

import pytest


class TestSLAMonitor:
    """Tests for SLAMonitor class."""

    def test_initialization(self) -> None:
        """Test monitor initialization."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config, name="test_monitor")

        assert monitor.name == "test_monitor"
        assert monitor.config == config
        assert monitor.consecutive_failures == 0

    def test_default_name_is_none(self) -> None:
        """Test default name is None if not provided."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig()
        monitor = SLAMonitor(config)

        assert monitor.name is None

    def test_check_no_violations(self) -> None:
        """Test check with no violations."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(max_failure_rate=0.10, min_pass_rate=0.90)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(passed_count=95, failed_count=5)

        violations = monitor.check(metrics)

        assert len(violations) == 0

    def test_check_failure_rate_exceeded(self) -> None:
        """Test check with failure rate exceeded."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics, SLAViolationType
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(max_failure_rate=0.05)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(passed_count=80, failed_count=20)  # 20% failure rate

        violations = monitor.check(metrics)

        assert len(violations) == 1
        assert violations[0].violation_type == SLAViolationType.FAILURE_RATE_EXCEEDED
        assert violations[0].threshold == 0.05
        assert violations[0].actual == 0.2

    def test_check_pass_rate_below_minimum(self) -> None:
        """Test check with pass rate below minimum."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics, SLAViolationType
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(min_pass_rate=0.95)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(passed_count=80, failed_count=20)  # 80% pass rate

        violations = monitor.check(metrics)

        assert len(violations) == 1
        assert violations[0].violation_type == SLAViolationType.PASS_RATE_BELOW_MINIMUM

    def test_check_execution_time_exceeded(self) -> None:
        """Test check with execution time exceeded."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics, SLAViolationType
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(max_execution_time_seconds=60.0)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(execution_time_ms=120000.0)  # 120 seconds

        violations = monitor.check(metrics)

        assert len(violations) == 1
        assert violations[0].violation_type == SLAViolationType.EXECUTION_TIME_EXCEEDED

    def test_check_row_count_below_minimum(self) -> None:
        """Test check with row count below minimum."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics, SLAViolationType
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(min_row_count=1000)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(row_count=500)

        violations = monitor.check(metrics)

        assert len(violations) == 1
        assert violations[0].violation_type == SLAViolationType.ROW_COUNT_BELOW_MINIMUM

    def test_check_row_count_above_maximum(self) -> None:
        """Test check with row count above maximum."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics, SLAViolationType
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(max_row_count=1000)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(row_count=5000)

        violations = monitor.check(metrics)

        assert len(violations) == 1
        assert violations[0].violation_type == SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM

    def test_check_multiple_violations(self) -> None:
        """Test check with multiple violations."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(
            max_failure_rate=0.05,
            min_pass_rate=0.95,
            max_execution_time_seconds=60.0,
        )
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(
            passed_count=80,
            failed_count=20,
            execution_time_ms=120000.0,
        )

        violations = monitor.check(metrics)

        assert len(violations) == 3

    def test_check_disabled_sla(self) -> None:
        """Test check with disabled SLA returns no violations."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(max_failure_rate=0.05, enabled=False)
        monitor = SLAMonitor(config)

        metrics = SLAMetrics(passed_count=50, failed_count=50)  # 50% failure

        violations = monitor.check(metrics)

        assert len(violations) == 0

    def test_record_failure(self) -> None:
        """Test recording a failure."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig()
        monitor = SLAMonitor(config)

        assert monitor.consecutive_failures == 0

        count = monitor.record_failure()
        assert count == 1
        assert monitor.consecutive_failures == 1

        count = monitor.record_failure()
        assert count == 2
        assert monitor.consecutive_failures == 2

    def test_record_success_resets_failures(self) -> None:
        """Test recording success resets consecutive failures."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig()
        monitor = SLAMonitor(config)

        monitor.record_failure()
        monitor.record_failure()
        assert monitor.consecutive_failures == 2

        monitor.record_success()
        assert monitor.consecutive_failures == 0

    def test_check_consecutive_failures(self) -> None:
        """Test check with consecutive failures threshold."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics, SLAViolationType
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig(max_consecutive_failures=3)
        monitor = SLAMonitor(config)

        # Record 2 failures first (below threshold)
        monitor.record_failure()
        monitor.record_failure()

        # Now check with metrics that have failed_count > 0
        # This will call record_failure() and increment to 3
        metrics = SLAMetrics(passed_count=90, failed_count=10)

        violations = monitor.check(metrics)

        # Should have consecutive failures violation (count reached 3)
        failure_violations = [
            v for v in violations if v.violation_type == SLAViolationType.CONSECUTIVE_FAILURES
        ]
        assert len(failure_violations) == 1

    def test_reset(self) -> None:
        """Test reset method."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLAMonitor

        config = SLAConfig()
        monitor = SLAMonitor(config)

        monitor.record_failure()
        monitor.record_failure()

        monitor.reset()

        assert monitor.consecutive_failures == 0


class TestSLARegistry:
    """Tests for SLARegistry class."""

    def test_register(self) -> None:
        """Test registering a monitor."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()
        config = SLAConfig(max_failure_rate=0.05)

        monitor = registry.register("test_sla", config)

        assert monitor is not None
        assert monitor.name == "test_sla"

    def test_register_duplicate_raises(self) -> None:
        """Test registering duplicate name raises error."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test_sla", config)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_sla", config)

    def test_register_replace(self) -> None:
        """Test registering with replace=True."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()
        config1 = SLAConfig(max_failure_rate=0.05)
        config2 = SLAConfig(max_failure_rate=0.10)

        registry.register("test_sla", config1)
        monitor = registry.register("test_sla", config2, replace=True)

        assert monitor.config.max_failure_rate == 0.10

    def test_get(self) -> None:
        """Test getting a monitor."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test_sla", config)
        monitor = registry.get("test_sla")

        assert monitor is not None
        assert monitor.name == "test_sla"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent monitor returns None."""
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()

        assert registry.get("nonexistent") is None

    def test_unregister(self) -> None:
        """Test unregistering a monitor."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()
        config = SLAConfig()

        registry.register("test_sla", config)
        result = registry.unregister("test_sla")

        assert result is True
        assert registry.get("test_sla") is None

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent monitor returns False."""
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()

        assert registry.unregister("nonexistent") is False

    def test_list_names(self) -> None:
        """Test listing all monitor names."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()

        registry.register("sla1", SLAConfig())
        registry.register("sla2", SLAConfig())
        registry.register("sla3", SLAConfig())

        names = registry.list_names()

        assert len(names) == 3
        assert "sla1" in names
        assert "sla2" in names
        assert "sla3" in names

    def test_check_all(self) -> None:
        """Test checking all monitors."""
        from truthound_airflow.sla.config import SLAConfig, SLAMetrics
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()

        registry.register("sla1", SLAConfig(max_failure_rate=0.05))
        registry.register("sla2", SLAConfig(min_pass_rate=0.95))

        metrics_by_name = {
            "sla1": SLAMetrics(passed_count=90, failed_count=10),
            "sla2": SLAMetrics(passed_count=90, failed_count=10),
        }

        results = registry.check_all(metrics_by_name)

        assert "sla1" in results
        assert "sla2" in results
        assert len(results["sla1"]) > 0
        assert len(results["sla2"]) > 0

    def test_reset_all(self) -> None:
        """Test resetting all monitors."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()

        registry.register("sla1", SLAConfig())
        registry.register("sla2", SLAConfig())

        # Record some failures
        registry.get("sla1").record_failure()
        registry.get("sla2").record_failure()

        registry.reset_all()

        assert registry.get("sla1").consecutive_failures == 0
        assert registry.get("sla2").consecutive_failures == 0

    def test_get_all(self) -> None:
        """Test getting all monitors."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import SLARegistry

        registry = SLARegistry()

        registry.register("sla1", SLAConfig())
        registry.register("sla2", SLAConfig())

        monitors = registry.get_all()

        assert len(monitors) == 2
        assert "sla1" in monitors
        assert "sla2" in monitors


class TestGlobalRegistry:
    """Tests for global SLA registry functions."""

    def test_get_sla_registry(self) -> None:
        """Test getting global SLA registry."""
        from truthound_airflow.sla.monitor import SLARegistry, get_sla_registry

        registry = get_sla_registry()

        assert isinstance(registry, SLARegistry)

    def test_reset_sla_registry(self) -> None:
        """Test resetting global SLA registry."""
        from truthound_airflow.sla.config import SLAConfig
        from truthound_airflow.sla.monitor import get_sla_registry, reset_sla_registry

        registry = get_sla_registry()
        registry.register("test", SLAConfig())

        reset_sla_registry()

        new_registry = get_sla_registry()
        assert len(new_registry.list_names()) == 0
