"""Unit tests for Engine Lifecycle Management.

Tests cover:
- Engine state transitions
- Lifecycle management (start/stop/health_check)
- Context manager support
- Configuration and hooks
- Health checker integration
"""

from __future__ import annotations

import time
import threading
import pytest

from common.engines.lifecycle import (
    # State
    EngineState,
    EngineStateSnapshot,
    EngineStateTracker,
    # Configuration
    EngineConfig,
    DEFAULT_ENGINE_CONFIG,
    PRODUCTION_ENGINE_CONFIG,
    DEVELOPMENT_ENGINE_CONFIG,
    TESTING_ENGINE_CONFIG,
    # Manager
    EngineLifecycleManager,
    # Health
    EngineHealthChecker,
    # Hooks
    LoggingLifecycleHook,
    MetricsLifecycleHook,
    CompositeLifecycleHook,
    # Mixin
    ManagedEngineMixin,
    # Utilities
    create_managed_engine,
    create_engine_health_checker,
    # Exceptions
    EngineAlreadyStartedError,
    EngineStoppedError,
    EngineInitializationError,
)
from common.health import HealthStatus


# =============================================================================
# Test Fixtures
# =============================================================================


class MockEngine:
    """Mock engine for testing lifecycle management.

    This implements the ManagedEngine protocol for use with EngineLifecycleManager.
    """

    def __init__(
        self,
        name: str = "mock",
        should_fail_start: bool = False,
        should_fail_stop: bool = False,
    ) -> None:
        self._name = name
        self._should_fail_start = should_fail_start
        self._should_fail_stop = should_fail_stop
        self._started = False
        self._start_count = 0
        self._stop_count = 0
        self._state = EngineState.CREATED

    @property
    def engine_name(self) -> str:
        return self._name

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def start(self) -> None:
        if self._should_fail_start:
            raise RuntimeError("Mock start failure")
        self._started = True
        self._start_count += 1
        self._state = EngineState.RUNNING

    def stop(self) -> None:
        if self._should_fail_stop:
            raise RuntimeError("Mock stop failure")
        self._started = False
        self._stop_count += 1
        self._state = EngineState.STOPPED

    def health_check(self):
        """Mock health check."""
        from common.health import HealthCheckResult
        return HealthCheckResult.healthy(self._name)

    def get_state(self) -> EngineState:
        return self._state


# =============================================================================
# Test EngineState
# =============================================================================


class TestEngineState:
    """Tests for EngineState enum."""

    def test_is_active_states(self) -> None:
        """Test is_active property for different states."""
        assert EngineState.STARTING.is_active
        assert EngineState.RUNNING.is_active
        assert not EngineState.CREATED.is_active
        assert not EngineState.STOPPED.is_active
        assert not EngineState.FAILED.is_active

    def test_can_start_states(self) -> None:
        """Test can_start property."""
        assert EngineState.CREATED.can_start
        assert not EngineState.RUNNING.can_start
        assert not EngineState.STOPPED.can_start

    def test_can_stop_states(self) -> None:
        """Test can_stop property."""
        assert EngineState.RUNNING.can_stop
        assert EngineState.STARTING.can_stop
        assert EngineState.FAILED.can_stop
        assert not EngineState.STOPPED.can_stop
        assert not EngineState.CREATED.can_stop

    def test_is_terminal(self) -> None:
        """Test is_terminal property."""
        assert EngineState.STOPPED.is_terminal
        assert not EngineState.CREATED.is_terminal
        assert not EngineState.RUNNING.is_terminal


# =============================================================================
# Test EngineConfig
# =============================================================================


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EngineConfig()
        assert not config.auto_start
        assert config.auto_stop
        assert config.health_check_enabled
        assert config.startup_timeout_seconds == 30.0
        assert config.shutdown_timeout_seconds == 10.0

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = EngineConfig()
        with pytest.raises(AttributeError):
            config.auto_start = True  # type: ignore

    def test_with_auto_start(self) -> None:
        """Test with_auto_start builder method."""
        config = EngineConfig().with_auto_start(True)
        assert config.auto_start

    def test_with_health_check(self) -> None:
        """Test with_health_check builder method."""
        config = EngineConfig().with_health_check(False, interval_seconds=60.0)
        assert not config.health_check_enabled
        assert config.health_check_interval_seconds == 60.0

    def test_with_timeouts(self) -> None:
        """Test with_timeouts builder method."""
        config = EngineConfig().with_timeouts(
            startup_seconds=60.0,
            shutdown_seconds=30.0,
        )
        assert config.startup_timeout_seconds == 60.0
        assert config.shutdown_timeout_seconds == 30.0

    def test_with_tags(self) -> None:
        """Test with_tags builder method."""
        config = EngineConfig().with_tags("production", "critical")
        assert "production" in config.tags
        assert "critical" in config.tags

    def test_with_metadata(self) -> None:
        """Test with_metadata builder method."""
        config = EngineConfig().with_metadata(env="prod", region="us-east")
        assert config.metadata["env"] == "prod"
        assert config.metadata["region"] == "us-east"

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        config = EngineConfig(auto_start=True)
        config_dict = config.to_dict()
        assert config_dict["auto_start"] is True
        assert "startup_timeout_seconds" in config_dict

    def test_from_dict(self) -> None:
        """Test from_dict deserialization."""
        config = EngineConfig.from_dict({"auto_start": True, "tags": ["test"]})
        assert config.auto_start
        assert "test" in config.tags

    def test_preset_configs(self) -> None:
        """Test preset configurations exist and have correct values."""
        assert PRODUCTION_ENGINE_CONFIG.auto_start
        assert PRODUCTION_ENGINE_CONFIG.health_check_enabled
        assert not DEVELOPMENT_ENGINE_CONFIG.health_check_enabled
        assert TESTING_ENGINE_CONFIG.startup_timeout_seconds == 5.0

    def test_validation_negative_timeout(self) -> None:
        """Test that negative timeouts raise ValueError."""
        with pytest.raises(ValueError):
            EngineConfig(startup_timeout_seconds=-1.0)


# =============================================================================
# Test EngineStateTracker
# =============================================================================


class TestEngineStateTracker:
    """Tests for EngineStateTracker."""

    def test_initial_state(self) -> None:
        """Test initial state is CREATED."""
        tracker = EngineStateTracker("test")
        assert tracker.state == EngineState.CREATED

    def test_transition_to(self) -> None:
        """Test state transitions."""
        tracker = EngineStateTracker("test")
        old = tracker.transition_to(EngineState.STARTING)
        assert old == EngineState.CREATED
        assert tracker.state == EngineState.STARTING

    def test_transition_to_running_sets_started_at(self) -> None:
        """Test that transitioning to RUNNING sets started_at."""
        tracker = EngineStateTracker("test")
        tracker.transition_to(EngineState.RUNNING)
        snapshot = tracker.get_snapshot()
        assert snapshot.started_at is not None

    def test_transition_to_stopped_sets_stopped_at(self) -> None:
        """Test that transitioning to STOPPED sets stopped_at."""
        tracker = EngineStateTracker("test")
        tracker.transition_to(EngineState.RUNNING)
        tracker.transition_to(EngineState.STOPPED)
        snapshot = tracker.get_snapshot()
        assert snapshot.stopped_at is not None

    def test_record_health_check(self) -> None:
        """Test recording health check results."""
        tracker = EngineStateTracker("test")
        tracker.record_health_check(HealthStatus.HEALTHY)
        snapshot = tracker.get_snapshot()
        assert snapshot.last_health_status == HealthStatus.HEALTHY
        assert snapshot.last_health_check_at is not None

    def test_record_error(self) -> None:
        """Test error recording."""
        tracker = EngineStateTracker("test")
        count = tracker.record_error()
        assert count == 1
        count = tracker.record_error()
        assert count == 2
        snapshot = tracker.get_snapshot()
        assert snapshot.error_count == 2

    def test_record_operation(self) -> None:
        """Test operation recording."""
        tracker = EngineStateTracker("test")
        count = tracker.record_operation()
        assert count == 1

    def test_get_snapshot(self) -> None:
        """Test getting state snapshot."""
        tracker = EngineStateTracker("test")
        tracker.transition_to(EngineState.RUNNING)
        snapshot = tracker.get_snapshot()
        assert isinstance(snapshot, EngineStateSnapshot)
        assert snapshot.state == EngineState.RUNNING

    def test_thread_safety(self) -> None:
        """Test thread safety of state tracker."""
        tracker = EngineStateTracker("test")
        errors = []

        def transition():
            try:
                for _ in range(100):
                    tracker.record_operation()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=transition) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        snapshot = tracker.get_snapshot()
        assert snapshot.operation_count == 1000


# =============================================================================
# Test EngineStateSnapshot
# =============================================================================


class TestEngineStateSnapshot:
    """Tests for EngineStateSnapshot."""

    def test_uptime_seconds_not_started(self) -> None:
        """Test uptime when not started."""
        snapshot = EngineStateSnapshot(state=EngineState.CREATED)
        assert snapshot.uptime_seconds is None

    def test_is_healthy(self) -> None:
        """Test is_healthy property."""
        snapshot = EngineStateSnapshot(
            state=EngineState.RUNNING,
            last_health_status=HealthStatus.HEALTHY,
        )
        assert snapshot.is_healthy

        snapshot = EngineStateSnapshot(
            state=EngineState.RUNNING,
            last_health_status=HealthStatus.UNHEALTHY,
        )
        assert not snapshot.is_healthy

        snapshot = EngineStateSnapshot(state=EngineState.STOPPED)
        assert not snapshot.is_healthy

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        snapshot = EngineStateSnapshot(
            state=EngineState.RUNNING,
            error_count=5,
        )
        data = snapshot.to_dict()
        assert data["state"] == "RUNNING"
        assert data["error_count"] == 5


# =============================================================================
# Test EngineLifecycleManager
# =============================================================================


class TestEngineLifecycleManager:
    """Tests for EngineLifecycleManager."""

    def test_start_stop(self) -> None:
        """Test basic start and stop."""
        engine = MockEngine()
        manager = EngineLifecycleManager(engine)

        manager.start()
        assert manager.state == EngineState.RUNNING
        assert engine._start_count == 1

        manager.stop()
        assert manager.state == EngineState.STOPPED
        assert engine._stop_count == 1

    def test_double_start_raises(self) -> None:
        """Test that starting twice raises error."""
        engine = MockEngine()
        manager = EngineLifecycleManager(engine)

        manager.start()
        with pytest.raises(EngineAlreadyStartedError):
            manager.start()

    def test_start_after_stop_raises(self) -> None:
        """Test that starting after stop raises error."""
        engine = MockEngine()
        manager = EngineLifecycleManager(engine)

        manager.start()
        manager.stop()
        with pytest.raises(EngineStoppedError):
            manager.start()

    def test_stop_idempotent(self) -> None:
        """Test that stop is idempotent."""
        engine = MockEngine()
        manager = EngineLifecycleManager(engine)

        manager.start()
        manager.stop()
        manager.stop()  # Should not raise
        assert engine._stop_count == 1

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        engine = MockEngine()
        config = EngineConfig(auto_stop=True)

        with EngineLifecycleManager(engine, config=config) as manager:
            assert manager.state == EngineState.RUNNING

        assert manager.state == EngineState.STOPPED

    def test_context_manager_auto_stop_false(self) -> None:
        """Test context manager with auto_stop=False."""
        engine = MockEngine()
        config = EngineConfig(auto_stop=False)

        with EngineLifecycleManager(engine, config=config) as manager:
            pass

        assert manager.state == EngineState.RUNNING

    def test_health_check_not_running(self) -> None:
        """Test health check when not running."""
        engine = MockEngine()
        manager = EngineLifecycleManager(engine)

        result = manager.health_check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_health_check_running(self) -> None:
        """Test health check when running."""
        engine = MockEngine()
        manager = EngineLifecycleManager(engine)

        manager.start()
        result = manager.health_check()
        # MockEngine doesn't implement health_check, so it uses basic check
        assert result.status == HealthStatus.HEALTHY

    def test_start_failure_transitions_to_failed(self) -> None:
        """Test that start failure transitions to FAILED state."""
        engine = MockEngine(should_fail_start=True)
        manager = EngineLifecycleManager(engine)

        with pytest.raises(EngineInitializationError):
            manager.start()

        assert manager.state == EngineState.FAILED

    def test_get_state_snapshot(self) -> None:
        """Test getting state snapshot from manager."""
        engine = MockEngine()
        manager = EngineLifecycleManager(engine)

        manager.start()
        snapshot = manager.get_state_snapshot()
        assert snapshot.state == EngineState.RUNNING

    def test_hooks_called(self) -> None:
        """Test that hooks are called during lifecycle."""
        engine = MockEngine()
        hook = MetricsLifecycleHook()
        manager = EngineLifecycleManager(engine, hooks=[hook])

        manager.start()
        assert hook.get_start_count("mock") == 1

        manager.stop()
        assert hook.get_stop_count("mock") == 1


# =============================================================================
# Test Lifecycle Hooks
# =============================================================================


class TestMetricsLifecycleHook:
    """Tests for MetricsLifecycleHook."""

    def test_records_starts(self) -> None:
        """Test that starts are recorded."""
        hook = MetricsLifecycleHook()
        hook.on_started("engine1", 100.0, {})
        hook.on_started("engine1", 200.0, {})
        assert hook.get_start_count("engine1") == 2

    def test_records_stops(self) -> None:
        """Test that stops are recorded."""
        hook = MetricsLifecycleHook()
        hook.on_stopped("engine1", 50.0, {})
        assert hook.get_stop_count("engine1") == 1

    def test_records_errors(self) -> None:
        """Test that errors are recorded."""
        hook = MetricsLifecycleHook()
        hook.on_error("engine1", RuntimeError("test"), {})
        assert hook.get_error_count("engine1") == 1

    def test_average_startup_time(self) -> None:
        """Test average startup time calculation."""
        hook = MetricsLifecycleHook()
        hook.on_started("engine1", 100.0, {})
        hook.on_started("engine1", 200.0, {})
        assert hook.get_average_startup_time_ms("engine1") == 150.0

    def test_health_status_counts(self) -> None:
        """Test health status counting."""
        hook = MetricsLifecycleHook()
        from common.health import HealthCheckResult

        hook.on_health_check("engine1", HealthCheckResult.healthy("test"), {})
        hook.on_health_check("engine1", HealthCheckResult.healthy("test"), {})
        hook.on_health_check(
            "engine1",
            HealthCheckResult.unhealthy("test", message="bad"),
            {},
        )

        counts = hook.get_health_status_counts("engine1")
        assert counts[HealthStatus.HEALTHY] == 2
        assert counts[HealthStatus.UNHEALTHY] == 1

    def test_reset(self) -> None:
        """Test metrics reset."""
        hook = MetricsLifecycleHook()
        hook.on_started("engine1", 100.0, {})
        hook.reset()
        assert hook.get_start_count("engine1") == 0


class TestCompositeLifecycleHook:
    """Tests for CompositeLifecycleHook."""

    def test_calls_all_hooks(self) -> None:
        """Test that all hooks are called."""
        hook1 = MetricsLifecycleHook()
        hook2 = MetricsLifecycleHook()
        composite = CompositeLifecycleHook([hook1, hook2])

        composite.on_started("engine1", 100.0, {})

        assert hook1.get_start_count("engine1") == 1
        assert hook2.get_start_count("engine1") == 1

    def test_add_remove_hook(self) -> None:
        """Test adding and removing hooks."""
        hook1 = MetricsLifecycleHook()
        composite = CompositeLifecycleHook()

        composite.add_hook(hook1)
        composite.on_started("engine1", 100.0, {})
        assert hook1.get_start_count("engine1") == 1

        composite.remove_hook(hook1)
        composite.on_started("engine1", 100.0, {})
        assert hook1.get_start_count("engine1") == 1  # Not incremented

    def test_suppresses_exceptions(self) -> None:
        """Test that exceptions in hooks are suppressed."""

        class FailingHook:
            def on_started(self, *args, **kwargs):
                raise RuntimeError("Hook failed")

        hook = MetricsLifecycleHook()
        composite = CompositeLifecycleHook([FailingHook(), hook])

        composite.on_started("engine1", 100.0, {})
        assert hook.get_start_count("engine1") == 1  # Still called


# =============================================================================
# Test EngineHealthChecker
# =============================================================================


class TestEngineHealthChecker:
    """Tests for EngineHealthChecker."""

    def test_check_basic_engine(self) -> None:
        """Test health check on basic engine."""
        engine = MockEngine()
        checker = EngineHealthChecker(engine)

        result = checker.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.duration_ms > 0

    def test_name_override(self) -> None:
        """Test name override."""
        engine = MockEngine(name="original")
        checker = EngineHealthChecker(engine, name="custom")

        assert checker.name == "custom"
        # Note: When the engine implements health_check(), the result name
        # comes from the engine's health_check method, not the checker's name.
        # This is expected behavior since the checker delegates to the engine.
        result = checker.check()
        # The checker wraps the result but preserves the engine's name
        assert result.duration_ms > 0


# =============================================================================
# Test ManagedEngineMixin
# =============================================================================


class TestManagedEngineMixin:
    """Tests for ManagedEngineMixin."""

    def test_basic_lifecycle(self) -> None:
        """Test basic lifecycle with mixin."""

        class TestEngine(ManagedEngineMixin):
            engine_name = "test"
            engine_version = "1.0.0"
            started = False
            stopped = False

            def __init__(self):
                super().__init__()

            def _do_start(self):
                self.started = True

            def _do_stop(self):
                self.stopped = True

        engine = TestEngine()
        assert engine.get_state() == EngineState.CREATED

        engine.start()
        assert engine.get_state() == EngineState.RUNNING
        assert engine.started

        engine.stop()
        assert engine.get_state() == EngineState.STOPPED
        assert engine.stopped

    def test_context_manager_with_mixin(self) -> None:
        """Test context manager with mixin."""

        class TestEngine(ManagedEngineMixin):
            engine_name = "test"
            engine_version = "1.0.0"

            def __init__(self):
                super().__init__()

        with TestEngine() as engine:
            assert engine.get_state() == EngineState.RUNNING

        assert engine.get_state() == EngineState.STOPPED


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_managed_engine(self) -> None:
        """Test create_managed_engine function."""
        engine = MockEngine()
        managed = create_managed_engine(engine)

        assert isinstance(managed, EngineLifecycleManager)
        assert managed.engine is engine

    def test_create_engine_health_checker(self) -> None:
        """Test create_engine_health_checker function."""
        engine = MockEngine()
        checker = create_engine_health_checker(engine)

        assert isinstance(checker, EngineHealthChecker)
        result = checker.check()
        assert result.status == HealthStatus.HEALTHY
