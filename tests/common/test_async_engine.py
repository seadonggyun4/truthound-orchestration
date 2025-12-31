"""Tests for async engine lifecycle management.

This module tests the async engine components including:
- AsyncManagedEngineMixin
- AsyncEngineLifecycleManager
- AsyncLifecycleHook implementations
- SyncEngineAsyncAdapter
- Async mock engines

These tests ensure the async engine infrastructure works correctly
for integration with async frameworks like Prefect and FastAPI.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from common.base import CheckStatus
from common.engines.base import EngineInfoMixin
from common.engines.lifecycle import (
    AsyncCompositeLifecycleHook,
    AsyncEngineHealthChecker,
    AsyncEngineLifecycleManager,
    AsyncLoggingLifecycleHook,
    AsyncManagedEngineMixin,
    AsyncMetricsLifecycleHook,
    EngineAlreadyStartedError,
    EngineInitializationError,
    EngineShutdownError,
    EngineState,
    EngineStoppedError,
    SyncEngineAsyncAdapter,
    SyncToAsyncLifecycleHookAdapter,
)
from common.health import HealthCheckResult, HealthStatus
from common.testing import (
    AsyncMockDataQualityEngine,
    AsyncMockLifecycleHook,
    AsyncMockManagedEngine,
    MockDataQualityEngine,
    create_async_mock_engine,
    create_async_mock_lifecycle_hook,
    create_async_mock_managed_engine,
)


if TYPE_CHECKING:
    from common.engines.lifecycle import EngineConfig


# =============================================================================
# AsyncMockDataQualityEngine Tests
# =============================================================================


class TestAsyncMockDataQualityEngine:
    """Tests for AsyncMockDataQualityEngine."""

    @pytest.mark.asyncio
    async def test_basic_check(self) -> None:
        """Test basic async check operation."""
        engine = AsyncMockDataQualityEngine()
        engine.configure_check(success=True, passed_count=10)

        result = await engine.check({"data": "test"}, rules=[])

        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 10
        assert engine.check_call_count == 1

    @pytest.mark.asyncio
    async def test_check_with_delay(self) -> None:
        """Test async check with configured delay."""
        engine = AsyncMockDataQualityEngine()
        engine.configure_check(success=True, delay_seconds=0.1)

        start = asyncio.get_event_loop().time()
        await engine.check({"data": "test"}, rules=[])
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_check_failure(self) -> None:
        """Test async check failure configuration."""
        engine = AsyncMockDataQualityEngine()
        engine.configure_check(success=False, failed_count=5)

        result = await engine.check({"data": "test"}, rules=[])

        assert result.status == CheckStatus.FAILED
        assert result.failed_count == 5

    @pytest.mark.asyncio
    async def test_check_raises_error(self) -> None:
        """Test async check raises configured error."""
        engine = AsyncMockDataQualityEngine()
        engine.configure_check(raise_error=ValueError("Test error"))

        with pytest.raises(ValueError, match="Test error"):
            await engine.check({"data": "test"}, rules=[])

    @pytest.mark.asyncio
    async def test_profile(self) -> None:
        """Test async profile operation."""
        engine = AsyncMockDataQualityEngine()
        engine.configure_profile(success=True, row_count=1000)

        result = await engine.profile({"data": "test"})

        assert result.row_count == 1000
        assert engine.profile_call_count == 1

    @pytest.mark.asyncio
    async def test_learn(self) -> None:
        """Test async learn operation."""
        engine = AsyncMockDataQualityEngine()
        engine.configure_learn(success=True, columns_analyzed=5)

        result = await engine.learn({"data": "test"})

        assert result.columns_analyzed == 5
        assert engine.learn_call_count == 1

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test reset clears all state."""
        engine = AsyncMockDataQualityEngine()
        await engine.check({}, [])
        await engine.profile({})
        await engine.learn({})

        engine.reset()

        assert engine.check_call_count == 0
        assert engine.profile_call_count == 0
        assert engine.learn_call_count == 0


# =============================================================================
# AsyncMockManagedEngine Tests
# =============================================================================


class TestAsyncMockManagedEngine:
    """Tests for AsyncMockManagedEngine."""

    @pytest.mark.asyncio
    async def test_lifecycle_start_stop(self) -> None:
        """Test basic lifecycle start/stop."""
        engine = AsyncMockManagedEngine()

        assert engine.get_state() == EngineState.CREATED

        await engine.start()
        assert engine.get_state() == EngineState.RUNNING

        await engine.stop()
        assert engine.get_state() == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager protocol."""
        engine = AsyncMockManagedEngine()

        async with engine:
            assert engine.get_state() == EngineState.RUNNING
            result = await engine.check({}, [])
            assert result.status == CheckStatus.PASSED

        assert engine.get_state() == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_start_with_delay(self) -> None:
        """Test start with configured delay."""
        engine = AsyncMockManagedEngine()
        engine.configure_lifecycle(start_delay_seconds=0.1)

        start = asyncio.get_event_loop().time()
        await engine.start()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.1
        await engine.stop()

    @pytest.mark.asyncio
    async def test_start_failure(self) -> None:
        """Test start failure with configured error."""
        engine = AsyncMockManagedEngine()
        engine.configure_lifecycle(start_raise_error=RuntimeError("Init failed"))

        with pytest.raises(EngineInitializationError):
            await engine.start()

        assert engine.get_state() == EngineState.FAILED

    @pytest.mark.asyncio
    async def test_stop_failure(self) -> None:
        """Test stop failure with configured error."""
        engine = AsyncMockManagedEngine()
        await engine.start()
        engine.configure_lifecycle(stop_raise_error=RuntimeError("Shutdown failed"))

        with pytest.raises(EngineShutdownError):
            await engine.stop()

        assert engine.get_state() == EngineState.FAILED

    @pytest.mark.asyncio
    async def test_health_check_healthy(self) -> None:
        """Test health check returns healthy status."""
        engine = AsyncMockManagedEngine()
        engine.configure_lifecycle(health_status="healthy")

        await engine.start()
        result = await engine.health_check()

        assert result.status == HealthStatus.HEALTHY
        await engine.stop()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self) -> None:
        """Test health check returns unhealthy status."""
        engine = AsyncMockManagedEngine()
        engine.configure_lifecycle(health_status="unhealthy")

        await engine.start()
        result = await engine.health_check()

        assert result.status == HealthStatus.UNHEALTHY
        await engine.stop()

    @pytest.mark.asyncio
    async def test_health_check_not_running(self) -> None:
        """Test health check when engine not running."""
        engine = AsyncMockManagedEngine()
        result = await engine.health_check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "not running" in result.message.lower()

    @pytest.mark.asyncio
    async def test_double_start_raises_error(self) -> None:
        """Test starting already running engine raises error."""
        engine = AsyncMockManagedEngine()
        await engine.start()

        with pytest.raises(EngineAlreadyStartedError):
            await engine.start()

        await engine.stop()

    @pytest.mark.asyncio
    async def test_start_after_stop_raises_error(self) -> None:
        """Test starting stopped engine raises error."""
        engine = AsyncMockManagedEngine()
        await engine.start()
        await engine.stop()

        with pytest.raises(EngineStoppedError):
            await engine.start()


# =============================================================================
# AsyncManagedEngineMixin Tests
# =============================================================================


class SampleAsyncEngine(AsyncManagedEngineMixin, EngineInfoMixin):
    """Sample async engine for testing the mixin."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        super().__init__(config)
        self.start_called = False
        self.stop_called = False
        self.connection = None

    @property
    def engine_name(self) -> str:
        return "sample_async"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    async def _do_start(self) -> None:
        self.start_called = True
        self.connection = "connected"

    async def _do_stop(self) -> None:
        self.stop_called = True
        self.connection = None

    async def _do_health_check(self) -> HealthCheckResult:
        if self.connection:
            return HealthCheckResult.healthy(self.engine_name)
        return HealthCheckResult.unhealthy(self.engine_name)

    async def check(self, data, rules, **kwargs):
        from common.base import CheckResult, CheckStatus

        return CheckResult(status=CheckStatus.PASSED, passed_count=1)

    async def profile(self, data, **kwargs):
        from common.base import ProfileResult, ProfileStatus

        return ProfileResult(status=ProfileStatus.COMPLETED, row_count=100, column_count=5)

    async def learn(self, data, **kwargs):
        from common.base import LearnResult, LearnStatus

        return LearnResult(status=LearnStatus.COMPLETED, columns_analyzed=5)


class TestAsyncManagedEngineMixin:
    """Tests for AsyncManagedEngineMixin."""

    @pytest.mark.asyncio
    async def test_lifecycle(self) -> None:
        """Test full lifecycle with mixin."""
        engine = SampleAsyncEngine()

        assert not engine.start_called
        assert engine.get_state() == EngineState.CREATED

        await engine.start()
        assert engine.start_called
        assert engine.connection == "connected"
        assert engine.get_state() == EngineState.RUNNING

        await engine.stop()
        assert engine.stop_called
        assert engine.connection is None
        assert engine.get_state() == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        engine = SampleAsyncEngine()

        async with engine:
            assert engine.start_called
            result = await engine.check({}, [])
            assert result.status == CheckStatus.PASSED

        assert engine.stop_called

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check through mixin."""
        engine = SampleAsyncEngine()

        await engine.start()
        result = await engine.health_check()
        assert result.status == HealthStatus.HEALTHY

        await engine.stop()


class FailingAsyncEngine(AsyncManagedEngineMixin, EngineInfoMixin):
    """Async engine that fails during lifecycle operations."""

    @property
    def engine_name(self) -> str:
        return "failing_async"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    async def _do_start(self) -> None:
        raise RuntimeError("Start failed")

    async def check(self, data, rules, **kwargs):
        pass

    async def profile(self, data, **kwargs):
        pass

    async def learn(self, data, **kwargs):
        pass


class TestAsyncManagedEngineMixinFailure:
    """Tests for failure scenarios with AsyncManagedEngineMixin."""

    @pytest.mark.asyncio
    async def test_start_failure_sets_failed_state(self) -> None:
        """Test that start failure sets FAILED state."""
        engine = FailingAsyncEngine()

        with pytest.raises(EngineInitializationError):
            await engine.start()

        assert engine.get_state() == EngineState.FAILED


# =============================================================================
# AsyncEngineLifecycleManager Tests
# =============================================================================


class TestAsyncEngineLifecycleManager:
    """Tests for AsyncEngineLifecycleManager."""

    @pytest.mark.asyncio
    async def test_basic_lifecycle(self) -> None:
        """Test basic lifecycle management."""
        engine = AsyncMockDataQualityEngine()
        manager = AsyncEngineLifecycleManager(engine)

        assert manager.state == EngineState.CREATED

        await manager.start()
        assert manager.state == EngineState.RUNNING

        await manager.stop()
        assert manager.state == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        engine = AsyncMockDataQualityEngine()
        manager = AsyncEngineLifecycleManager(engine)

        async with manager:
            assert manager.state == EngineState.RUNNING

        assert manager.state == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_with_hooks(self) -> None:
        """Test lifecycle manager with hooks."""
        engine = AsyncMockDataQualityEngine()
        hook = AsyncMockLifecycleHook()
        manager = AsyncEngineLifecycleManager(engine, hooks=[hook])

        await manager.start()
        assert hook.start_count == 1
        assert engine.engine_name in hook.started_engines

        await manager.health_check()
        assert hook.health_check_count == 1

        await manager.stop()
        assert hook.stop_count == 1

    @pytest.mark.asyncio
    async def test_health_check_not_running(self) -> None:
        """Test health check when not running."""
        engine = AsyncMockDataQualityEngine()
        manager = AsyncEngineLifecycleManager(engine)

        result = await manager.health_check()
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_get_state_snapshot(self) -> None:
        """Test state snapshot."""
        engine = AsyncMockDataQualityEngine()
        manager = AsyncEngineLifecycleManager(engine)

        await manager.start()
        snapshot = manager.get_state_snapshot()

        assert snapshot.state == EngineState.RUNNING
        assert snapshot.started_at is not None
        assert snapshot.uptime_seconds >= 0

        await manager.stop()


# =============================================================================
# AsyncLifecycleHook Tests
# =============================================================================


class TestAsyncLoggingLifecycleHook:
    """Tests for AsyncLoggingLifecycleHook."""

    @pytest.mark.asyncio
    async def test_logs_lifecycle_events(self) -> None:
        """Test that logging hook logs events without errors."""
        from common.logging import LogContext

        # Initialize logging context for the test
        with LogContext(test="async_logging"):
            hook = AsyncLoggingLifecycleHook()

            # Should not raise
            await hook.on_starting("test_engine", {})
            await hook.on_started("test_engine", 100.0, {})
            await hook.on_stopping("test_engine", {})
            await hook.on_stopped("test_engine", 50.0, {})
            await hook.on_error("test_engine", RuntimeError("test"), {})
            await hook.on_health_check(
                "test_engine",
                HealthCheckResult.healthy("test_engine"),
                {},
            )


class TestAsyncMetricsLifecycleHook:
    """Tests for AsyncMetricsLifecycleHook."""

    @pytest.mark.asyncio
    async def test_collects_metrics(self) -> None:
        """Test metrics collection."""
        hook = AsyncMetricsLifecycleHook()

        await hook.on_started("engine1", 100.0, {})
        await hook.on_started("engine1", 200.0, {})
        await hook.on_stopped("engine1", 50.0, {})

        assert hook.get_start_count("engine1") == 2
        assert hook.get_stop_count("engine1") == 1
        assert hook.get_average_startup_time_ms("engine1") == 150.0

    @pytest.mark.asyncio
    async def test_error_counting(self) -> None:
        """Test error counting."""
        hook = AsyncMetricsLifecycleHook()

        await hook.on_error("engine1", RuntimeError("err1"), {})
        await hook.on_error("engine1", RuntimeError("err2"), {})

        assert hook.get_error_count("engine1") == 2

    @pytest.mark.asyncio
    async def test_health_status_counts(self) -> None:
        """Test health status counting."""
        hook = AsyncMetricsLifecycleHook()

        await hook.on_health_check(
            "engine1",
            HealthCheckResult.healthy("engine1"),
            {},
        )
        await hook.on_health_check(
            "engine1",
            HealthCheckResult.healthy("engine1"),
            {},
        )
        await hook.on_health_check(
            "engine1",
            HealthCheckResult.unhealthy("engine1"),
            {},
        )

        counts = hook.get_health_status_counts("engine1")
        assert counts[HealthStatus.HEALTHY] == 2
        assert counts[HealthStatus.UNHEALTHY] == 1

    def test_reset(self) -> None:
        """Test reset clears all metrics."""
        hook = AsyncMetricsLifecycleHook()
        hook._start_counts["engine1"] = 5

        hook.reset()

        assert hook.get_start_count("engine1") == 0


class TestAsyncCompositeLifecycleHook:
    """Tests for AsyncCompositeLifecycleHook."""

    @pytest.mark.asyncio
    async def test_calls_all_hooks(self) -> None:
        """Test composite calls all hooks."""
        hook1 = AsyncMockLifecycleHook()
        hook2 = AsyncMockLifecycleHook()
        composite = AsyncCompositeLifecycleHook([hook1, hook2])

        await composite.on_started("engine", 100.0, {})

        assert hook1.start_count == 1
        assert hook2.start_count == 1

    @pytest.mark.asyncio
    async def test_suppresses_hook_exceptions(self) -> None:
        """Test exceptions from individual hooks are suppressed."""

        class FailingHook(AsyncMockLifecycleHook):
            async def on_started(self, *args, **kwargs):
                raise RuntimeError("Hook failed")

        hook1 = FailingHook()
        hook2 = AsyncMockLifecycleHook()
        composite = AsyncCompositeLifecycleHook([hook1, hook2])

        # Should not raise, and hook2 should still be called
        await composite.on_started("engine", 100.0, {})
        assert hook2.start_count == 1


# =============================================================================
# SyncToAsyncLifecycleHookAdapter Tests
# =============================================================================


class TestSyncToAsyncLifecycleHookAdapter:
    """Tests for SyncToAsyncLifecycleHookAdapter."""

    @pytest.mark.asyncio
    async def test_adapts_sync_hook(self) -> None:
        """Test adapter wraps sync hook correctly."""
        from common.engines.lifecycle import MetricsLifecycleHook

        # Use MetricsLifecycleHook instead of LoggingLifecycleHook
        # to avoid logging context issues in async tests
        sync_hook = MetricsLifecycleHook()
        async_hook = SyncToAsyncLifecycleHookAdapter(sync_hook)

        # Should not raise and should record metrics
        await async_hook.on_started("engine", 100.0, {})
        assert sync_hook.get_start_count("engine") == 1


# =============================================================================
# SyncEngineAsyncAdapter Tests
# =============================================================================


class TestSyncEngineAsyncAdapter:
    """Tests for SyncEngineAsyncAdapter."""

    @pytest.mark.asyncio
    async def test_wraps_sync_engine(self) -> None:
        """Test adapter wraps sync engine."""
        sync_engine = MockDataQualityEngine()
        sync_engine.configure_check(success=True, passed_count=10)

        async_adapter = SyncEngineAsyncAdapter(sync_engine)

        assert async_adapter.engine_name == "mock"
        assert async_adapter.wrapped_engine is sync_engine

    @pytest.mark.asyncio
    async def test_async_check(self) -> None:
        """Test async check delegates to sync engine."""
        sync_engine = MockDataQualityEngine()
        sync_engine.configure_check(success=True, passed_count=15)

        async_adapter = SyncEngineAsyncAdapter(sync_engine)
        await async_adapter.start()

        result = await async_adapter.check({}, [])

        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 15
        assert sync_engine.check_call_count == 1

        await async_adapter.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        sync_engine = MockDataQualityEngine()
        async_adapter = SyncEngineAsyncAdapter(sync_engine)

        async with async_adapter:
            assert async_adapter.get_state() == EngineState.RUNNING
            await async_adapter.check({}, [])

        assert async_adapter.get_state() == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check through adapter."""
        sync_engine = MockDataQualityEngine()
        async_adapter = SyncEngineAsyncAdapter(sync_engine)

        await async_adapter.start()
        result = await async_adapter.health_check()

        assert result.status == HealthStatus.HEALTHY

        await async_adapter.stop()


# =============================================================================
# AsyncEngineHealthChecker Tests
# =============================================================================


class TestAsyncEngineHealthChecker:
    """Tests for AsyncEngineHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_managed_engine(self) -> None:
        """Test health check with managed engine."""
        engine = AsyncMockManagedEngine()
        await engine.start()

        checker = AsyncEngineHealthChecker(engine)
        result = await checker.check()

        assert result.status == HealthStatus.HEALTHY
        assert result.duration_ms > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_check_lifecycle_manager(self) -> None:
        """Test health check with lifecycle manager."""
        engine = AsyncMockDataQualityEngine()
        manager = AsyncEngineLifecycleManager(engine)
        await manager.start()

        checker = AsyncEngineHealthChecker(manager)
        result = await checker.check()

        assert result.status == HealthStatus.HEALTHY

        await manager.stop()

    @pytest.mark.asyncio
    async def test_custom_name(self) -> None:
        """Test health checker with custom name."""
        engine = AsyncMockDataQualityEngine()
        checker = AsyncEngineHealthChecker(engine, name="custom_name")

        assert checker.name == "custom_name"


# =============================================================================
# AsyncMockLifecycleHook Tests
# =============================================================================


class TestAsyncMockLifecycleHook:
    """Tests for AsyncMockLifecycleHook."""

    @pytest.mark.asyncio
    async def test_records_events(self) -> None:
        """Test hook records all events."""
        hook = AsyncMockLifecycleHook()

        await hook.on_starting("engine1", {"key": "value"})
        await hook.on_started("engine1", 100.0, {})
        await hook.on_stopping("engine1", {})
        await hook.on_stopped("engine1", 50.0, {})
        await hook.on_error("engine1", RuntimeError("test"), {})
        await hook.on_health_check("engine1", HealthCheckResult.healthy("engine1"), {})

        assert hook.start_count == 1
        assert hook.stop_count == 1
        assert hook.error_count == 1
        assert hook.health_check_count == 1
        assert "engine1" in hook.started_engines
        assert "engine1" in hook.stopped_engines

    @pytest.mark.asyncio
    async def test_get_errors(self) -> None:
        """Test get_errors returns error list."""
        hook = AsyncMockLifecycleHook()
        error = RuntimeError("test error")

        await hook.on_error("engine1", error, {})

        errors = hook.get_errors()
        assert len(errors) == 1
        assert errors[0] == ("engine1", error)

    def test_reset(self) -> None:
        """Test reset clears all events."""
        hook = AsyncMockLifecycleHook()
        hook._started_events.append(("engine", 100.0, {}))

        hook.reset()

        assert hook.start_count == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_async_mock_engine(self) -> None:
        """Test create_async_mock_engine factory."""
        engine = create_async_mock_engine(
            name="test_engine",
            version="2.0.0",
            check_success=False,
        )

        assert engine.engine_name == "test_engine"
        assert engine.engine_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_create_async_mock_engine_check(self) -> None:
        """Test created engine check behavior."""
        engine = create_async_mock_engine(check_success=False)
        result = await engine.check({}, [])

        assert result.status == CheckStatus.FAILED

    def test_create_async_mock_managed_engine(self) -> None:
        """Test create_async_mock_managed_engine factory."""
        engine = create_async_mock_managed_engine(
            name="managed_test",
            health_status="degraded",
        )

        assert engine.engine_name == "managed_test"

    @pytest.mark.asyncio
    async def test_create_async_mock_managed_engine_health(self) -> None:
        """Test created managed engine health status."""
        engine = create_async_mock_managed_engine(health_status="degraded")
        await engine.start()
        result = await engine.health_check()

        assert result.status == HealthStatus.DEGRADED
        await engine.stop()

    def test_create_async_mock_lifecycle_hook(self) -> None:
        """Test create_async_mock_lifecycle_hook factory."""
        hook = create_async_mock_lifecycle_hook()

        assert isinstance(hook, AsyncMockLifecycleHook)
        assert hook.start_count == 0


# =============================================================================
# Concurrent Operation Tests
# =============================================================================


class TestConcurrentOperations:
    """Tests for concurrent async operations."""

    @pytest.mark.asyncio
    async def test_concurrent_checks(self) -> None:
        """Test multiple concurrent check operations."""
        engine = AsyncMockDataQualityEngine()
        engine.configure_check(success=True, delay_seconds=0.1)

        # Run 5 checks concurrently
        tasks = [engine.check({}, []) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.status == CheckStatus.PASSED for r in results)
        assert engine.check_call_count == 5

    @pytest.mark.asyncio
    async def test_concurrent_engines(self) -> None:
        """Test multiple engines running concurrently."""
        engines = [
            AsyncMockManagedEngine(name=f"engine_{i}")
            for i in range(3)
        ]

        # Start all engines concurrently
        await asyncio.gather(*[e.start() for e in engines])

        assert all(e.get_state() == EngineState.RUNNING for e in engines)

        # Stop all engines concurrently
        await asyncio.gather(*[e.stop() for e in engines])

        assert all(e.get_state() == EngineState.STOPPED for e in engines)
