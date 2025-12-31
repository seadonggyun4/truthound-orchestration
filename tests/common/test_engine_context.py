"""Unit tests for Engine Context Management.

Tests cover:
- EngineContext: Basic context management, resource tracking
- EngineSession: Transaction-like sessions, savepoints
- MultiEngineContext: Multi-engine coordination
- ContextStack: Nested context management
- AsyncEngineContext: Async context support
- Hooks: Logging and metrics hooks
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from common.engines.context import (
    # Configuration
    ContextConfig,
    DEFAULT_CONTEXT_CONFIG,
    LIGHTWEIGHT_CONTEXT_CONFIG,
    STRICT_CONTEXT_CONFIG,
    TESTING_CONTEXT_CONFIG,
    # Enums
    ContextState,
    SessionState,
    CleanupStrategy,
    # Resource Tracking
    TrackedResource,
    ResourceTracker,
    # Savepoint Support
    Savepoint,
    SavepointManager,
    # Hooks
    LoggingContextHook,
    MetricsContextHook,
    CompositeContextHook,
    # Main Context Classes
    EngineContext,
    EngineSession,
    MultiEngineContext,
    ContextStack,
    AsyncEngineContext,
    # Exceptions
    ContextError,
    ContextNotActiveError,
    ContextAlreadyActiveError,
    SavepointError,
    ResourceCleanupError,
    # Factory Functions
    create_engine_context,
    create_engine_session,
    create_multi_engine_context,
    engine_context,
)
from common.health import HealthCheckResult, HealthStatus


# =============================================================================
# Test Fixtures
# =============================================================================


class MockEngine:
    """Mock engine for testing context management."""

    def __init__(
        self,
        name: str = "mock",
        should_fail_check: bool = False,
    ) -> None:
        self._name = name
        self._should_fail_check = should_fail_check
        self._check_count = 0
        self._profile_count = 0
        self._started = False
        self._stopped = False

    @property
    def engine_name(self) -> str:
        return self._name

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def check(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        if self._should_fail_check:
            raise RuntimeError("Check failed")
        self._check_count += 1
        return {"status": "passed", "data_size": len(data) if hasattr(data, "__len__") else 0}

    def profile(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        self._profile_count += 1
        return {"profile": "completed"}

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._stopped = True

    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult.healthy(self._name)


class ManagedMockEngine(MockEngine):
    """Mock engine that implements ManagedEngine protocol."""

    from common.engines.lifecycle import EngineState

    def __init__(self, name: str = "managed_mock") -> None:
        super().__init__(name)
        self._state = self.EngineState.CREATED

    def start(self) -> None:
        super().start()
        self._state = self.EngineState.RUNNING

    def stop(self) -> None:
        super().stop()
        self._state = self.EngineState.STOPPED

    def get_state(self):
        return self._state


# =============================================================================
# Test ContextState
# =============================================================================


class TestContextState:
    """Tests for ContextState enum."""

    def test_is_active(self) -> None:
        """Test is_active property."""
        assert ContextState.ACTIVE.is_active
        assert not ContextState.CREATED.is_active
        assert not ContextState.EXITED.is_active

    def test_can_enter(self) -> None:
        """Test can_enter property."""
        assert ContextState.CREATED.can_enter
        assert not ContextState.ACTIVE.can_enter
        assert not ContextState.EXITED.can_enter

    def test_can_exit(self) -> None:
        """Test can_exit property."""
        assert ContextState.ACTIVE.can_exit
        assert ContextState.ENTERING.can_exit
        assert ContextState.FAILED.can_exit
        assert not ContextState.EXITED.can_exit
        assert not ContextState.CREATED.can_exit

    def test_is_terminal(self) -> None:
        """Test is_terminal property."""
        assert ContextState.EXITED.is_terminal
        assert ContextState.FAILED.is_terminal
        assert not ContextState.ACTIVE.is_terminal


# =============================================================================
# Test ContextConfig
# =============================================================================


class TestContextConfig:
    """Tests for ContextConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ContextConfig()
        assert config.auto_start_engine
        assert config.auto_stop_engine
        assert config.cleanup_strategy == CleanupStrategy.ALWAYS
        assert config.track_resources
        assert config.enable_savepoints

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = ContextConfig()
        with pytest.raises(AttributeError):
            config.auto_start_engine = False  # type: ignore

    def test_with_auto_start(self) -> None:
        """Test with_auto_start builder method."""
        config = ContextConfig().with_auto_start(False)
        assert not config.auto_start_engine

    def test_with_cleanup_strategy(self) -> None:
        """Test with_cleanup_strategy builder method."""
        config = ContextConfig().with_cleanup_strategy(CleanupStrategy.ON_FAILURE)
        assert config.cleanup_strategy == CleanupStrategy.ON_FAILURE

    def test_with_timeout(self) -> None:
        """Test with_timeout builder method."""
        config = ContextConfig().with_timeout(30.0)
        assert config.timeout_seconds == 30.0

    def test_with_savepoints(self) -> None:
        """Test with_savepoints builder method."""
        config = ContextConfig().with_savepoints(False)
        assert not config.enable_savepoints

    def test_preset_configs(self) -> None:
        """Test preset configurations."""
        assert LIGHTWEIGHT_CONTEXT_CONFIG.auto_start_engine is False
        assert not LIGHTWEIGHT_CONTEXT_CONFIG.track_resources
        assert STRICT_CONTEXT_CONFIG.propagate_exceptions
        assert TESTING_CONTEXT_CONFIG.timeout_seconds == 5.0


# =============================================================================
# Test ResourceTracker
# =============================================================================


class TestResourceTracker:
    """Tests for ResourceTracker."""

    def test_register_resource(self) -> None:
        """Test registering a resource."""
        tracker = ResourceTracker()
        resource = tracker.register("conn1", "connection", name="Database connection")

        assert resource.resource_id == "conn1"
        assert resource.resource_type == "connection"
        assert resource.name == "Database connection"
        assert "conn1" in tracker

    def test_unregister_resource(self) -> None:
        """Test unregistering a resource."""
        tracker = ResourceTracker()
        tracker.register("conn1", "connection")

        resource = tracker.unregister("conn1")
        assert resource is not None
        assert "conn1" not in tracker

    def test_get_resource(self) -> None:
        """Test getting a resource by ID."""
        tracker = ResourceTracker()
        tracker.register("conn1", "connection")

        resource = tracker.get("conn1")
        assert resource is not None
        assert resource.resource_id == "conn1"

        assert tracker.get("nonexistent") is None

    def test_list_resources(self) -> None:
        """Test listing resources."""
        tracker = ResourceTracker()
        tracker.register("conn1", "connection")
        tracker.register("file1", "file")
        tracker.register("conn2", "connection")

        all_resources = tracker.list_resources()
        assert len(all_resources) == 3

        connections = tracker.list_resources("connection")
        assert len(connections) == 2

    def test_cleanup_all(self) -> None:
        """Test cleaning up all resources."""
        cleanup_called = []

        def cleanup1():
            cleanup_called.append("conn1")

        def cleanup2():
            cleanup_called.append("conn2")

        tracker = ResourceTracker()
        tracker.register("conn1", "connection", cleanup_func=cleanup1)
        tracker.register("conn2", "connection", cleanup_func=cleanup2)

        failed = tracker.cleanup_all()

        assert len(failed) == 0
        assert "conn1" in cleanup_called
        assert "conn2" in cleanup_called
        # LIFO order
        assert cleanup_called.index("conn2") < cleanup_called.index("conn1")

    def test_cleanup_handles_errors(self) -> None:
        """Test that cleanup handles errors gracefully."""

        def failing_cleanup():
            raise RuntimeError("Cleanup failed")

        tracker = ResourceTracker()
        tracker.register("conn1", "connection", cleanup_func=failing_cleanup)
        tracker.register("conn2", "connection", cleanup_func=lambda: None)

        failed = tracker.cleanup_all()
        assert "conn1" in failed

    def test_len(self) -> None:
        """Test __len__ method."""
        tracker = ResourceTracker()
        assert len(tracker) == 0

        tracker.register("conn1", "connection")
        assert len(tracker) == 1


# =============================================================================
# Test SavepointManager
# =============================================================================


class TestSavepointManager:
    """Tests for SavepointManager."""

    def test_create_savepoint(self) -> None:
        """Test creating a savepoint."""
        manager = SavepointManager()
        savepoint = manager.create("sp1", {"data": [1, 2, 3]})

        assert savepoint.name == "sp1"
        assert savepoint.state_snapshot == {"data": [1, 2, 3]}
        assert savepoint.sequence == 1

    def test_duplicate_savepoint_raises(self) -> None:
        """Test that duplicate savepoint names raise error."""
        manager = SavepointManager()
        manager.create("sp1")

        with pytest.raises(SavepointError):
            manager.create("sp1")

    def test_get_savepoint(self) -> None:
        """Test getting a savepoint."""
        manager = SavepointManager()
        manager.create("sp1")

        assert manager.get("sp1") is not None
        assert manager.get("nonexistent") is None

    def test_exists(self) -> None:
        """Test exists method."""
        manager = SavepointManager()
        manager.create("sp1")

        assert manager.exists("sp1")
        assert not manager.exists("sp2")

    def test_release_savepoint(self) -> None:
        """Test releasing a savepoint."""
        manager = SavepointManager()
        manager.create("sp1")

        released = manager.release("sp1")
        assert released is not None
        assert not manager.exists("sp1")

    def test_rollback_to_savepoint(self) -> None:
        """Test rolling back to a savepoint."""
        manager = SavepointManager()
        manager.create("sp1", {"count": 1})
        manager.create("sp2", {"count": 2})
        manager.create("sp3", {"count": 3})

        snapshot = manager.rollback_to("sp1")

        assert snapshot == {"count": 1}
        assert manager.exists("sp1")
        assert not manager.exists("sp2")
        assert not manager.exists("sp3")

    def test_rollback_to_nonexistent_raises(self) -> None:
        """Test that rollback to nonexistent savepoint raises error."""
        manager = SavepointManager()

        with pytest.raises(SavepointError):
            manager.rollback_to("nonexistent")

    def test_list_savepoints(self) -> None:
        """Test listing savepoints."""
        manager = SavepointManager()
        manager.create("sp1")
        manager.create("sp2")

        savepoints = manager.list_savepoints()
        assert len(savepoints) == 2
        assert savepoints[0].name == "sp1"
        assert savepoints[1].name == "sp2"

    def test_clear(self) -> None:
        """Test clearing all savepoints."""
        manager = SavepointManager()
        manager.create("sp1")
        manager.create("sp2")

        count = manager.clear()
        assert count == 2
        assert len(manager.list_savepoints()) == 0


# =============================================================================
# Test EngineContext
# =============================================================================


class TestEngineContext:
    """Tests for EngineContext."""

    def test_basic_context_manager(self) -> None:
        """Test basic context manager usage."""
        engine = MockEngine()

        with EngineContext(engine) as ctx:
            assert ctx.is_active
            assert ctx.state == ContextState.ACTIVE
            result = ctx.execute(lambda e: e.check([1, 2, 3]))
            assert result["status"] == "passed"

        assert ctx.state == ContextState.EXITED

    def test_explicit_enter_exit(self) -> None:
        """Test explicit enter and exit."""
        engine = MockEngine()
        ctx = EngineContext(engine)

        ctx.enter()
        assert ctx.is_active

        ctx.exit()
        assert ctx.state == ContextState.EXITED

    def test_double_enter_raises(self) -> None:
        """Test that double enter raises error."""
        engine = MockEngine()
        ctx = EngineContext(engine)

        ctx.enter()
        with pytest.raises(ContextAlreadyActiveError):
            ctx.enter()

    def test_execute_outside_context_raises(self) -> None:
        """Test that execute outside context raises error."""
        engine = MockEngine()
        ctx = EngineContext(engine)

        with pytest.raises(ContextNotActiveError):
            ctx.execute(lambda e: e.check([]))

    def test_engine_property_outside_context_raises(self) -> None:
        """Test that accessing engine outside context raises error."""
        engine = MockEngine()
        ctx = EngineContext(engine)

        with pytest.raises(ContextNotActiveError):
            _ = ctx.engine

    def test_context_id_and_name(self) -> None:
        """Test context_id and name properties."""
        engine = MockEngine()
        ctx = EngineContext(engine, name="my_context")

        assert ctx.context_id is not None
        assert len(ctx.context_id) == 8
        assert ctx.name == "my_context"

    def test_duration_tracking(self) -> None:
        """Test duration tracking."""
        engine = MockEngine()

        with EngineContext(engine) as ctx:
            time.sleep(0.01)  # 10ms
            assert ctx.duration_ms is not None
            assert ctx.duration_ms >= 10

    def test_resource_tracking(self) -> None:
        """Test resource registration and tracking."""
        engine = MockEngine()
        config = ContextConfig(track_resources=True)

        with EngineContext(engine, config=config) as ctx:
            resource = ctx.register_resource("conn1", "connection")
            assert resource is not None
            assert resource.resource_id == "conn1"

            resources = ctx.list_resources()
            assert len(resources) == 1

            ctx.unregister_resource("conn1")
            assert len(ctx.list_resources()) == 0

    def test_resource_cleanup_on_exit(self) -> None:
        """Test resource cleanup on context exit."""
        cleanup_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        engine = MockEngine()
        config = ContextConfig(track_resources=True, cleanup_strategy=CleanupStrategy.ALWAYS)

        with EngineContext(engine, config=config) as ctx:
            ctx.register_resource("conn1", "connection", cleanup_func=cleanup)

        assert cleanup_called

    def test_cleanup_strategy_on_success(self) -> None:
        """Test cleanup strategy ON_SUCCESS."""
        cleanup_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        config = ContextConfig(
            track_resources=True,
            cleanup_strategy=CleanupStrategy.ON_SUCCESS,
        )
        engine = MockEngine()

        with EngineContext(engine, config=config) as ctx:
            ctx.register_resource("conn1", "connection", cleanup_func=cleanup)

        assert cleanup_called

    def test_cleanup_strategy_on_failure(self) -> None:
        """Test cleanup strategy ON_FAILURE."""
        cleanup_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        config = ContextConfig(
            track_resources=True,
            cleanup_strategy=CleanupStrategy.ON_FAILURE,
        )
        engine = MockEngine()

        # Success case - should not cleanup
        with EngineContext(engine, config=config) as ctx:
            ctx.register_resource("conn1", "connection", cleanup_func=cleanup)

        assert not cleanup_called

        # Failure case - should cleanup
        cleanup_called = False
        try:
            with EngineContext(engine, config=config) as ctx:
                ctx.register_resource("conn2", "connection", cleanup_func=cleanup)
                raise ValueError("Error")
        except ValueError:
            pass

        assert cleanup_called

    def test_resource_tracking_disabled(self) -> None:
        """Test with resource tracking disabled."""
        config = ContextConfig(track_resources=False)
        engine = MockEngine()

        with EngineContext(engine, config=config) as ctx:
            resource = ctx.register_resource("conn1", "connection")
            assert resource is None
            assert len(ctx.list_resources()) == 0

    def test_auto_start_engine(self) -> None:
        """Test auto-start engine."""
        config = ContextConfig(auto_start_engine=True)
        engine = ManagedMockEngine()

        with EngineContext(engine, config=config):
            assert engine._started

    def test_auto_stop_engine(self) -> None:
        """Test auto-stop engine."""
        config = ContextConfig(auto_stop_engine=True)
        engine = ManagedMockEngine()

        with EngineContext(engine, config=config):
            pass

        assert engine._stopped

    def test_hooks_called(self) -> None:
        """Test that hooks are called."""
        engine = MockEngine()
        hook = MetricsContextHook()

        def named_check(e):
            return e.check([1, 2, 3])

        with EngineContext(engine, hooks=[hook]) as ctx:
            ctx.execute(named_check, name="check")

        assert hook.get_enter_count("mock") == 1
        ops = hook.get_operation_counts("mock")
        assert ops.get("check", 0) == 1


# =============================================================================
# Test EngineSession
# =============================================================================


class TestEngineSession:
    """Tests for EngineSession."""

    def test_basic_session(self) -> None:
        """Test basic session usage."""
        engine = MockEngine()

        with EngineSession(engine) as session:
            assert session.is_active
            result = session.execute(lambda e: e.check([1, 2, 3]))
            assert result["status"] == "passed"
            session.commit()

        assert session.state == SessionState.COMMITTED

    def test_session_rollback(self) -> None:
        """Test session rollback."""
        engine = MockEngine()
        rollback_called = False

        def rollback_handler():
            nonlocal rollback_called
            rollback_called = True

        with EngineSession(engine) as session:
            session.execute(
                lambda e: e.check([1, 2, 3]),
                rollback_handler=rollback_handler,
            )
            session.rollback()

        assert session.state == SessionState.ROLLED_BACK
        assert rollback_called

    def test_auto_rollback_on_exception(self) -> None:
        """Test auto-rollback on exception."""
        engine = MockEngine(should_fail_check=True)
        rollback_called = False

        def rollback_handler():
            nonlocal rollback_called
            rollback_called = True

        try:
            with EngineSession(engine) as session:
                session.register_rollback_handler(rollback_handler)
                session.execute(lambda e: e.check([]))  # Will fail
        except RuntimeError:
            pass

        assert session.state == SessionState.ROLLED_BACK
        assert rollback_called

    def test_auto_commit(self) -> None:
        """Test auto-commit on success."""
        engine = MockEngine()

        with EngineSession(engine, auto_commit=True) as session:
            session.execute(lambda e: e.check([1, 2, 3]))

        assert session.state == SessionState.COMMITTED

    def test_savepoint_create_rollback(self) -> None:
        """Test savepoint creation and rollback."""
        engine = MockEngine()

        with EngineSession(engine) as session:
            session.execute(lambda e: e.check([1]))
            session.create_savepoint("sp1")

            session.execute(lambda e: e.check([2]))
            session.execute(lambda e: e.check([3]))

            # Operations before rollback: 3
            assert len(session.operations) == 3

            session.rollback_to_savepoint("sp1")

            # Operations after rollback: 1 (only before savepoint)
            assert len(session.operations) == 1

            session.commit()

    def test_savepoint_release(self) -> None:
        """Test savepoint release."""
        engine = MockEngine()

        with EngineSession(engine) as session:
            session.create_savepoint("sp1")
            assert len(session.list_savepoints()) == 1

            released = session.release_savepoint("sp1")
            assert released is not None
            assert len(session.list_savepoints()) == 0

            session.commit()

    def test_operations_recorded(self) -> None:
        """Test that operations are recorded."""
        engine = MockEngine()

        with EngineSession(engine) as session:
            session.execute(lambda e: e.check([1]), name="check_1")
            session.execute(lambda e: e.profile([1]), name="profile_1")
            session.commit()

        ops = session.operations
        assert len(ops) == 2
        assert ops[0].name == "check_1"
        assert ops[1].name == "profile_1"
        assert all(op.success for op in ops)

    def test_session_id(self) -> None:
        """Test session_id property."""
        engine = MockEngine()

        with EngineSession(engine) as session:
            assert session.session_id is not None
            assert len(session.session_id) == 8
            session.commit()


# =============================================================================
# Test MultiEngineContext
# =============================================================================


class TestMultiEngineContext:
    """Tests for MultiEngineContext."""

    def test_basic_multi_engine(self) -> None:
        """Test basic multi-engine context."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2")

        with MultiEngineContext(engine1, engine2) as ctx:
            assert ctx.is_active
            assert "engine1" in ctx.engine_names
            assert "engine2" in ctx.engine_names

    def test_execute_on_specific_engine(self) -> None:
        """Test executing on a specific engine."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2")

        with MultiEngineContext(engine1, engine2) as ctx:
            result = ctx.execute_on("engine1", lambda e: e.check([1, 2, 3]))
            assert result["status"] == "passed"
            assert engine1._check_count == 1
            assert engine2._check_count == 0

    def test_execute_all(self) -> None:
        """Test executing on all engines."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2")

        with MultiEngineContext(engine1, engine2) as ctx:
            results = ctx.execute_all(lambda e: e.check([1, 2, 3]))

            assert "engine1" in results
            assert "engine2" in results
            assert engine1._check_count == 1
            assert engine2._check_count == 1

    def test_execute_all_parallel(self) -> None:
        """Test parallel execution on all engines."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2")

        with MultiEngineContext(engine1, engine2, parallel=True) as ctx:
            results = ctx.execute_all(lambda e: e.check([1, 2, 3]))

            assert len(results) == 2
            assert engine1._check_count == 1
            assert engine2._check_count == 1

    def test_execute_all_with_failure(self) -> None:
        """Test execute_all with one engine failing."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2", should_fail_check=True)

        with MultiEngineContext(engine1, engine2) as ctx:
            results = ctx.execute_all(lambda e: e.check([1, 2, 3]))

            assert results["engine1"]["status"] == "passed"
            assert isinstance(results["engine2"], Exception)

    def test_fail_fast(self) -> None:
        """Test fail_fast mode."""
        engine1 = MockEngine("engine1", should_fail_check=True)
        engine2 = MockEngine("engine2")

        with MultiEngineContext(engine1, engine2, fail_fast=True) as ctx:
            results = ctx.execute_all(lambda e: e.check([1, 2, 3]))

            # engine1 fails, engine2 should not be executed
            assert isinstance(results["engine1"], Exception)
            assert "engine2" not in results

    def test_get_engine(self) -> None:
        """Test getting engine by name."""
        engine1 = MockEngine("engine1")

        with MultiEngineContext(engine1) as ctx:
            assert ctx.get_engine("engine1") is engine1

            with pytest.raises(KeyError):
                ctx.get_engine("nonexistent")

    def test_health_check(self) -> None:
        """Test aggregated health check."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2")

        with MultiEngineContext(engine1, engine2) as ctx:
            health = ctx.health_check()

            assert "engine1" in health
            assert "engine2" in health
            assert health["engine1"].status == HealthStatus.HEALTHY

    def test_duplicate_engine_name_raises(self) -> None:
        """Test that duplicate engine names raise error."""
        engine1 = MockEngine("same_name")
        engine2 = MockEngine("same_name")

        with pytest.raises(ValueError, match="Duplicate engine name"):
            MultiEngineContext(engine1, engine2)


# =============================================================================
# Test ContextStack
# =============================================================================


class TestContextStack:
    """Tests for ContextStack."""

    def test_push_and_pop(self) -> None:
        """Test push and pop operations."""
        stack = ContextStack()
        engine = MockEngine()
        ctx = EngineContext(engine)

        stack.push(ctx)
        assert stack.depth == 1
        assert stack.current is ctx

        popped = stack.pop()
        assert popped is ctx
        assert stack.is_empty

    def test_enter_exit_all(self) -> None:
        """Test entering and exiting all contexts."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2")

        stack = ContextStack()
        stack.push(EngineContext(engine1))
        stack.push(EngineContext(engine2))

        with stack:
            for ctx in [stack.current]:
                assert ctx is not None and ctx.is_active

        # All contexts should be exited
        for ctx in [stack._stack[0], stack._stack[1]]:
            assert ctx.state == ContextState.EXITED

    def test_max_depth(self) -> None:
        """Test max depth enforcement."""
        stack = ContextStack(max_depth=2)

        stack.push(EngineContext(MockEngine("e1")))
        stack.push(EngineContext(MockEngine("e2")))

        with pytest.raises(ContextError, match="Maximum context depth"):
            stack.push(EngineContext(MockEngine("e3")))


# =============================================================================
# Test Hooks
# =============================================================================


class TestMetricsContextHook:
    """Tests for MetricsContextHook."""

    def test_records_enters_and_exits(self) -> None:
        """Test that enters and exits are recorded."""
        hook = MetricsContextHook()
        engine = MockEngine()

        with EngineContext(engine, hooks=[hook]):
            pass

        assert hook.get_enter_count("mock") == 1
        assert hook.get_success_rate("mock") == 1.0

    def test_records_failures(self) -> None:
        """Test that failures are recorded."""
        hook = MetricsContextHook()
        engine = MockEngine(should_fail_check=True)

        try:
            with EngineContext(engine, hooks=[hook]) as ctx:
                ctx.execute(lambda e: e.check([]))
        except RuntimeError:
            pass

        assert hook.get_success_rate("mock") == 0.0

    def test_average_duration(self) -> None:
        """Test average duration calculation."""
        hook = MetricsContextHook()
        engine = MockEngine()

        with EngineContext(engine, hooks=[hook]):
            time.sleep(0.01)

        avg = hook.get_average_duration_ms("mock")
        assert avg >= 10

    def test_reset(self) -> None:
        """Test metrics reset."""
        hook = MetricsContextHook()
        engine = MockEngine()

        with EngineContext(engine, hooks=[hook]):
            pass

        hook.reset()
        assert hook.get_enter_count("mock") == 0


class TestCompositeContextHook:
    """Tests for CompositeContextHook."""

    def test_calls_all_hooks(self) -> None:
        """Test that all hooks are called."""
        hook1 = MetricsContextHook()
        hook2 = MetricsContextHook()
        composite = CompositeContextHook([hook1, hook2])

        engine = MockEngine()

        with EngineContext(engine, hooks=[composite]):
            pass

        assert hook1.get_enter_count("mock") == 1
        assert hook2.get_enter_count("mock") == 1

    def test_add_remove_hook(self) -> None:
        """Test adding and removing hooks."""
        hook = MetricsContextHook()
        composite = CompositeContextHook()

        composite.add_hook(hook)
        assert hook in composite._hooks

        composite.remove_hook(hook)
        assert hook not in composite._hooks


# =============================================================================
# Test AsyncEngineContext
# =============================================================================


class TestAsyncEngineContext:
    """Tests for AsyncEngineContext."""

    @pytest.mark.asyncio
    async def test_basic_async_context(self) -> None:
        """Test basic async context manager."""
        engine = MockEngine()

        async with AsyncEngineContext(engine) as ctx:
            assert ctx.is_active
            result = await ctx.execute(lambda e: e.check([1, 2, 3]))
            assert result["status"] == "passed"

        assert ctx.state == ContextState.EXITED

    @pytest.mark.asyncio
    async def test_explicit_enter_exit(self) -> None:
        """Test explicit async enter and exit."""
        engine = MockEngine()
        ctx = AsyncEngineContext(engine)

        await ctx.enter()
        assert ctx.is_active

        await ctx.exit()
        assert ctx.state == ContextState.EXITED

    @pytest.mark.asyncio
    async def test_resource_tracking(self) -> None:
        """Test resource tracking in async context."""
        cleanup_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        engine = MockEngine()
        config = ContextConfig(
            track_resources=True,
            cleanup_strategy=CleanupStrategy.ALWAYS,
        )

        async with AsyncEngineContext(engine, config=config) as ctx:
            ctx.register_resource("conn1", "connection", cleanup_func=cleanup)

        assert cleanup_called


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_engine_context(self) -> None:
        """Test create_engine_context."""
        engine = MockEngine()
        ctx = create_engine_context(engine, name="test_ctx")

        assert isinstance(ctx, EngineContext)
        assert ctx.name == "test_ctx"

    def test_create_engine_session(self) -> None:
        """Test create_engine_session."""
        engine = MockEngine()
        session = create_engine_session(engine, auto_commit=True)

        assert isinstance(session, EngineSession)

    def test_create_multi_engine_context(self) -> None:
        """Test create_multi_engine_context."""
        engine1 = MockEngine("e1")
        engine2 = MockEngine("e2")
        ctx = create_multi_engine_context(engine1, engine2, parallel=True)

        assert isinstance(ctx, MultiEngineContext)

    def test_engine_context_function(self) -> None:
        """Test engine_context convenience function."""
        engine = MockEngine()

        with engine_context(engine) as ctx:
            assert ctx.is_active
            result = ctx.execute(lambda e: e.check([1]))
            assert result["status"] == "passed"


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_resource_tracker_thread_safe(self) -> None:
        """Test ResourceTracker thread safety."""
        tracker = ResourceTracker()
        errors = []

        def worker(i: int):
            try:
                for j in range(100):
                    tracker.register(f"r{i}_{j}", "test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracker) == 1000

    def test_savepoint_manager_thread_safe(self) -> None:
        """Test SavepointManager thread safety."""
        manager = SavepointManager()
        errors = []

        def worker(i: int):
            try:
                for j in range(100):
                    manager.create(f"sp{i}_{j}")
            except SavepointError:
                pass  # Duplicate is expected
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
