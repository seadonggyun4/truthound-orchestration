"""Tests for common.health module."""

from __future__ import annotations

import asyncio
import importlib.util
import time

import pytest

from common.health import (
    CACHED_HEALTH_CHECK_CONFIG,
    DEFAULT_HEALTH_CHECK_CONFIG,
    FAST_HEALTH_CHECK_CONFIG,
    THOROUGH_HEALTH_CHECK_CONFIG,
    AggregationStrategy,
    AsyncSimpleHealthChecker,
    CompositeHealthChecker,
    CompositeHealthCheckHook,
    HealthCheckConfig,
    HealthCheckError,
    HealthCheckExecutor,
    HealthCheckRegistry,
    HealthCheckResult,
    HealthCheckTimeoutError,
    HealthStatus,
    LoggingHealthCheckHook,
    MetricsHealthCheckHook,
    SimpleHealthChecker,
    check_all_health,
    check_health,
    create_async_health_checker,
    create_composite_checker,
    create_health_checker,
    get_health_registry,
    health_check,
    register_health_check,
)


# Check if pytest-asyncio is available
HAS_PYTEST_ASYNCIO = importlib.util.find_spec("pytest_asyncio") is not None

asyncio_test = pytest.mark.skipif(
    not HAS_PYTEST_ASYNCIO,
    reason="pytest-asyncio not installed"
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert HealthStatus.HEALTHY
        assert HealthStatus.DEGRADED
        assert HealthStatus.UNHEALTHY
        assert HealthStatus.UNKNOWN

    def test_is_healthy(self):
        """Test is_healthy property."""
        assert HealthStatus.HEALTHY.is_healthy is True
        assert HealthStatus.DEGRADED.is_healthy is False
        assert HealthStatus.UNHEALTHY.is_healthy is False
        assert HealthStatus.UNKNOWN.is_healthy is False

    def test_is_operational(self):
        """Test is_operational property."""
        assert HealthStatus.HEALTHY.is_operational is True
        assert HealthStatus.DEGRADED.is_operational is True
        assert HealthStatus.UNHEALTHY.is_operational is False
        assert HealthStatus.UNKNOWN.is_operational is False

    def test_weight(self):
        """Test weight property."""
        assert HealthStatus.HEALTHY.weight == 100
        assert HealthStatus.DEGRADED.weight == 50
        assert HealthStatus.UNKNOWN.weight == 25
        assert HealthStatus.UNHEALTHY.weight == 0

    def test_comparison(self):
        """Test status comparison operators."""
        assert HealthStatus.HEALTHY > HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED > HealthStatus.UNKNOWN
        assert HealthStatus.UNKNOWN > HealthStatus.UNHEALTHY
        assert HealthStatus.UNHEALTHY < HealthStatus.HEALTHY
        assert HealthStatus.HEALTHY >= HealthStatus.HEALTHY
        assert HealthStatus.UNHEALTHY <= HealthStatus.DEGRADED


class TestHealthCheckConfig:
    """Tests for HealthCheckConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HealthCheckConfig()
        assert config.timeout_seconds == 5.0
        assert config.cache_ttl_seconds == 0.0
        assert config.include_details is True
        assert config.fail_on_timeout is True
        assert config.tags == frozenset()
        assert config.metadata == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HealthCheckConfig(
            timeout_seconds=10.0,
            cache_ttl_seconds=30.0,
            include_details=False,
            fail_on_timeout=False,
            tags=frozenset(["critical", "database"]),
            metadata={"version": "1.0"},
        )
        assert config.timeout_seconds == 10.0
        assert config.cache_ttl_seconds == 30.0
        assert config.include_details is False
        assert config.fail_on_timeout is False
        assert "critical" in config.tags
        assert config.metadata["version"] == "1.0"

    def test_validation_timeout(self):
        """Test validation of timeout_seconds."""
        with pytest.raises(ValueError, match="timeout_seconds must be non-negative"):
            HealthCheckConfig(timeout_seconds=-1.0)

    def test_validation_cache_ttl(self):
        """Test validation of cache_ttl_seconds."""
        with pytest.raises(ValueError, match="cache_ttl_seconds must be non-negative"):
            HealthCheckConfig(cache_ttl_seconds=-1.0)

    def test_with_timeout(self):
        """Test with_timeout builder method."""
        config = HealthCheckConfig(timeout_seconds=5.0)
        new_config = config.with_timeout(10.0)
        assert new_config.timeout_seconds == 10.0
        assert config.timeout_seconds == 5.0  # Original unchanged

    def test_with_cache_ttl(self):
        """Test with_cache_ttl builder method."""
        config = HealthCheckConfig()
        new_config = config.with_cache_ttl(60.0)
        assert new_config.cache_ttl_seconds == 60.0

    def test_with_tags(self):
        """Test with_tags builder method."""
        config = HealthCheckConfig(tags=frozenset(["existing"]))
        new_config = config.with_tags("new", "another")
        assert "existing" in new_config.tags
        assert "new" in new_config.tags
        assert "another" in new_config.tags

    def test_with_metadata(self):
        """Test with_metadata builder method."""
        config = HealthCheckConfig(metadata={"a": 1})
        new_config = config.with_metadata(b=2)
        assert new_config.metadata["a"] == 1
        assert new_config.metadata["b"] == 2

    def test_to_dict(self):
        """Test to_dict method."""
        config = HealthCheckConfig(
            timeout_seconds=10.0,
            tags=frozenset(["test"]),
            metadata={"key": "value"},
        )
        result = config.to_dict()
        assert result["timeout_seconds"] == 10.0
        assert "test" in result["tags"]
        assert result["metadata"]["key"] == "value"

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "timeout_seconds": 10.0,
            "cache_ttl_seconds": 30.0,
            "tags": ["test"],
            "metadata": {"key": "value"},
        }
        config = HealthCheckConfig.from_dict(data)
        assert config.timeout_seconds == 10.0
        assert config.cache_ttl_seconds == 30.0
        assert "test" in config.tags
        assert config.metadata["key"] == "value"

    def test_immutability(self):
        """Test that config is immutable."""
        config = HealthCheckConfig()
        with pytest.raises(AttributeError):
            config.timeout_seconds = 10.0  # type: ignore


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_config(self):
        """Test DEFAULT_HEALTH_CHECK_CONFIG."""
        assert DEFAULT_HEALTH_CHECK_CONFIG.timeout_seconds == 5.0
        assert DEFAULT_HEALTH_CHECK_CONFIG.cache_ttl_seconds == 0.0

    def test_fast_config(self):
        """Test FAST_HEALTH_CHECK_CONFIG."""
        assert FAST_HEALTH_CHECK_CONFIG.timeout_seconds == 1.0
        assert FAST_HEALTH_CHECK_CONFIG.cache_ttl_seconds == 10.0

    def test_thorough_config(self):
        """Test THOROUGH_HEALTH_CHECK_CONFIG."""
        assert THOROUGH_HEALTH_CHECK_CONFIG.timeout_seconds == 30.0
        assert THOROUGH_HEALTH_CHECK_CONFIG.include_details is True

    def test_cached_config(self):
        """Test CACHED_HEALTH_CHECK_CONFIG."""
        assert CACHED_HEALTH_CHECK_CONFIG.cache_ttl_seconds == 60.0


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == ""
        assert result.duration_ms == 0.0
        assert result.details == {}
        assert result.dependencies == ()

    def test_is_healthy(self):
        """Test is_healthy property."""
        healthy = HealthCheckResult(name="test", status=HealthStatus.HEALTHY)
        degraded = HealthCheckResult(name="test", status=HealthStatus.DEGRADED)
        unhealthy = HealthCheckResult(name="test", status=HealthStatus.UNHEALTHY)

        assert healthy.is_healthy is True
        assert degraded.is_healthy is False
        assert unhealthy.is_healthy is False

    def test_is_operational(self):
        """Test is_operational property."""
        healthy = HealthCheckResult(name="test", status=HealthStatus.HEALTHY)
        degraded = HealthCheckResult(name="test", status=HealthStatus.DEGRADED)
        unhealthy = HealthCheckResult(name="test", status=HealthStatus.UNHEALTHY)

        assert healthy.is_operational is True
        assert degraded.is_operational is True
        assert unhealthy.is_operational is False

    def test_with_status(self):
        """Test with_status method."""
        result = HealthCheckResult(name="test", status=HealthStatus.HEALTHY)
        new_result = result.with_status(HealthStatus.DEGRADED)
        assert new_result.status == HealthStatus.DEGRADED
        assert result.status == HealthStatus.HEALTHY  # Original unchanged

    def test_with_message(self):
        """Test with_message method."""
        result = HealthCheckResult(name="test", status=HealthStatus.HEALTHY)
        new_result = result.with_message("Updated message")
        assert new_result.message == "Updated message"

    def test_with_details(self):
        """Test with_details method."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            details={"a": 1},
        )
        new_result = result.with_details(b=2)
        assert new_result.details["a"] == 1
        assert new_result.details["b"] == 2

    def test_to_dict(self):
        """Test to_dict method."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            duration_ms=45.5,
        )
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "HEALTHY"
        assert data["message"] == "All good"
        assert data["duration_ms"] == 45.5
        assert data["is_healthy"] is True
        assert data["is_operational"] is True

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "name": "test",
            "status": "HEALTHY",
            "message": "All good",
            "duration_ms": 45.5,
        }
        result = HealthCheckResult.from_dict(data)
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.duration_ms == 45.5

    def test_healthy_factory(self):
        """Test healthy factory method."""
        result = HealthCheckResult.healthy("test", message="OK", extra="value")
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "OK"
        assert result.details["extra"] == "value"

    def test_degraded_factory(self):
        """Test degraded factory method."""
        result = HealthCheckResult.degraded("test", message="Slow")
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "Slow"

    def test_unhealthy_factory(self):
        """Test unhealthy factory method."""
        result = HealthCheckResult.unhealthy("test", message="Down")
        assert result.status == HealthStatus.UNHEALTHY
        assert result.message == "Down"

    def test_unknown_factory(self):
        """Test unknown factory method."""
        result = HealthCheckResult.unknown("test")
        assert result.status == HealthStatus.UNKNOWN


class TestHealthCheckExceptions:
    """Tests for health check exceptions."""

    def test_health_check_error(self):
        """Test HealthCheckError."""
        error = HealthCheckError(
            "Test error",
            check_name="test",
            status=HealthStatus.UNHEALTHY,
        )
        assert "Test error" in str(error)
        assert error.check_name == "test"
        assert error.status == HealthStatus.UNHEALTHY

    def test_health_check_timeout_error(self):
        """Test HealthCheckTimeoutError."""
        error = HealthCheckTimeoutError(
            "Timeout",
            check_name="test",
            timeout_seconds=5.0,
        )
        assert error.timeout_seconds == 5.0
        assert error.check_name == "test"
        assert error.status == HealthStatus.UNHEALTHY


class TestHealthCheckHooks:
    """Tests for health check hooks."""

    def test_logging_hook_on_check_start(self):
        """Test LoggingHealthCheckHook on_check_start."""
        hook = LoggingHealthCheckHook()
        # Should not raise
        hook.on_check_start("test", {})

    def test_logging_hook_on_check_complete(self):
        """Test LoggingHealthCheckHook on_check_complete."""
        hook = LoggingHealthCheckHook()
        result = HealthCheckResult.healthy("test")
        # Should not raise
        hook.on_check_complete("test", result, 45.0, {})

    def test_logging_hook_on_check_error(self):
        """Test LoggingHealthCheckHook on_check_error."""
        hook = LoggingHealthCheckHook()
        # Should not raise
        hook.on_check_error("test", ValueError("error"), {})

    def test_metrics_hook(self):
        """Test MetricsHealthCheckHook."""
        hook = MetricsHealthCheckHook()

        hook.on_check_start("test", {})
        assert hook.get_check_count("test") == 0

        result = HealthCheckResult.healthy("test")
        hook.on_check_complete("test", result, 45.0, {})
        assert hook.get_check_count("test") == 1
        assert hook.get_average_duration_ms("test") == 45.0

        unhealthy = HealthCheckResult.unhealthy("test")
        hook.on_check_complete("test", unhealthy, 100.0, {})
        assert hook.get_check_count("test") == 2
        assert hook.get_average_duration_ms("test") == 72.5

        status_counts = hook.get_status_counts("test")
        assert status_counts[HealthStatus.HEALTHY] == 1
        assert status_counts[HealthStatus.UNHEALTHY] == 1

    def test_metrics_hook_errors(self):
        """Test MetricsHealthCheckHook error tracking."""
        hook = MetricsHealthCheckHook()
        hook.on_check_error("test", ValueError("error"), {})
        assert hook.get_error_count("test") == 1

    def test_metrics_hook_last_result(self):
        """Test MetricsHealthCheckHook last result tracking."""
        hook = MetricsHealthCheckHook()
        result = HealthCheckResult.healthy("test")
        hook.on_check_complete("test", result, 45.0, {})
        assert hook.get_last_result("test") == result

    def test_metrics_hook_reset(self):
        """Test MetricsHealthCheckHook reset."""
        hook = MetricsHealthCheckHook()
        result = HealthCheckResult.healthy("test")
        hook.on_check_complete("test", result, 45.0, {})
        hook.reset()
        assert hook.get_check_count("test") == 0

    def test_composite_hook(self):
        """Test CompositeHealthCheckHook."""
        hook1 = MetricsHealthCheckHook()
        hook2 = MetricsHealthCheckHook()
        composite = CompositeHealthCheckHook([hook1, hook2])

        result = HealthCheckResult.healthy("test")
        composite.on_check_complete("test", result, 45.0, {})

        assert hook1.get_check_count("test") == 1
        assert hook2.get_check_count("test") == 1

    def test_composite_hook_add_remove(self):
        """Test adding and removing hooks."""
        hook1 = MetricsHealthCheckHook()
        composite = CompositeHealthCheckHook([])

        composite.add_hook(hook1)
        result = HealthCheckResult.healthy("test")
        composite.on_check_complete("test", result, 45.0, {})
        assert hook1.get_check_count("test") == 1

        composite.remove_hook(hook1)
        composite.on_check_complete("test", result, 45.0, {})
        assert hook1.get_check_count("test") == 1  # Not incremented


class TestHealthCheckExecutor:
    """Tests for HealthCheckExecutor."""

    def test_execute_successful(self):
        """Test successful health check execution."""
        executor = HealthCheckExecutor()

        def check_func():
            return HealthCheckResult.healthy("test")

        result = executor.execute("test", check_func)
        assert result.status == HealthStatus.HEALTHY

    def test_execute_bool_true(self):
        """Test execution with boolean True return."""
        executor = HealthCheckExecutor()
        result = executor.execute("test", lambda: True)
        assert result.status == HealthStatus.HEALTHY

    def test_execute_bool_false(self):
        """Test execution with boolean False return."""
        executor = HealthCheckExecutor()
        result = executor.execute("test", lambda: False)
        assert result.status == HealthStatus.UNHEALTHY

    def test_execute_none(self):
        """Test execution with None return."""
        executor = HealthCheckExecutor()
        result = executor.execute("test", lambda: None)
        assert result.status == HealthStatus.HEALTHY

    def test_execute_exception(self):
        """Test execution with exception."""
        executor = HealthCheckExecutor()

        def check_func():
            raise ValueError("Test error")

        result = executor.execute("test", check_func)
        assert result.status == HealthStatus.UNHEALTHY
        assert "Test error" in result.message

    def test_execute_timeout(self):
        """Test execution timeout."""
        config = HealthCheckConfig(timeout_seconds=0.1)
        executor = HealthCheckExecutor(config=config)

        def slow_check():
            time.sleep(0.5)
            return True

        result = executor.execute("test", slow_check)
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message

    def test_execute_with_hooks(self):
        """Test execution with hooks."""
        hook = MetricsHealthCheckHook()
        executor = HealthCheckExecutor(hooks=[hook])

        executor.execute("test", lambda: True)
        assert hook.get_check_count("test") == 1
        assert hook.get_last_result("test") is not None

    def test_caching(self):
        """Test result caching."""
        config = HealthCheckConfig(cache_ttl_seconds=10.0)
        executor = HealthCheckExecutor(config=config)

        call_count = [0]

        def check_func():
            call_count[0] += 1
            return HealthCheckResult.healthy("test")

        # First call
        executor.execute("test", check_func)
        assert call_count[0] == 1

        # Second call - should use cache
        result2 = executor.execute("test", check_func)
        assert call_count[0] == 1  # Not incremented
        assert result2.status == HealthStatus.HEALTHY

    def test_clear_cache(self):
        """Test clearing cache."""
        config = HealthCheckConfig(cache_ttl_seconds=10.0)
        executor = HealthCheckExecutor(config=config)

        call_count = [0]

        def check_func():
            call_count[0] += 1
            return HealthCheckResult.healthy("test")

        executor.execute("test", check_func)
        assert call_count[0] == 1

        executor.clear_cache("test")
        executor.execute("test", check_func)
        assert call_count[0] == 2

    @asyncio_test
    @pytest.mark.asyncio
    async def test_execute_async(self):
        """Test async health check execution."""
        executor = HealthCheckExecutor()

        async def check_func():
            return HealthCheckResult.healthy("test")

        result = await executor.execute_async("test", check_func)
        assert result.status == HealthStatus.HEALTHY

    @asyncio_test
    @pytest.mark.asyncio
    async def test_execute_async_timeout(self):
        """Test async execution timeout."""
        config = HealthCheckConfig(timeout_seconds=0.1)
        executor = HealthCheckExecutor(config=config)

        async def slow_check():
            await asyncio.sleep(0.5)
            return True

        result = await executor.execute_async("test", slow_check)
        assert result.status == HealthStatus.UNHEALTHY


class TestSimpleHealthChecker:
    """Tests for SimpleHealthChecker."""

    def test_basic_check(self):
        """Test basic health check."""
        checker = SimpleHealthChecker(
            name="test",
            check_func=lambda: True,
        )
        assert checker.name == "test"
        result = checker.check()
        assert result.status == HealthStatus.HEALTHY

    def test_check_with_result(self):
        """Test check returning HealthCheckResult."""
        checker = SimpleHealthChecker(
            name="test",
            check_func=lambda: HealthCheckResult.degraded("test", "Slow"),
        )
        result = checker.check()
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "Slow"


class TestAsyncSimpleHealthChecker:
    """Tests for AsyncSimpleHealthChecker."""

    @asyncio_test
    @pytest.mark.asyncio
    async def test_basic_async_check(self):
        """Test basic async health check."""

        async def check_func():
            return True

        checker = AsyncSimpleHealthChecker(
            name="test",
            check_func=check_func,
        )
        assert checker.name == "test"
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY


class TestAggregationStrategy:
    """Tests for AggregationStrategy."""

    def test_worst_strategy(self):
        """Test WORST aggregation strategy."""
        strategy = AggregationStrategy.WORST
        statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert strategy.aggregate(statuses) == HealthStatus.UNHEALTHY

    def test_best_strategy(self):
        """Test BEST aggregation strategy."""
        strategy = AggregationStrategy.BEST
        statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert strategy.aggregate(statuses) == HealthStatus.HEALTHY

    def test_majority_strategy(self):
        """Test MAJORITY aggregation strategy."""
        strategy = AggregationStrategy.MAJORITY
        statuses = [
            HealthStatus.HEALTHY,
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
        ]
        assert strategy.aggregate(statuses) == HealthStatus.HEALTHY

    def test_all_healthy_strategy(self):
        """Test ALL_HEALTHY aggregation strategy."""
        strategy = AggregationStrategy.ALL_HEALTHY

        all_healthy = [HealthStatus.HEALTHY, HealthStatus.HEALTHY]
        assert strategy.aggregate(all_healthy) == HealthStatus.HEALTHY

        mixed = [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert strategy.aggregate(mixed) == HealthStatus.DEGRADED

        with_unhealthy = [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        assert strategy.aggregate(with_unhealthy) == HealthStatus.UNHEALTHY

    def test_any_healthy_strategy(self):
        """Test ANY_HEALTHY aggregation strategy."""
        strategy = AggregationStrategy.ANY_HEALTHY

        with_healthy = [HealthStatus.UNHEALTHY, HealthStatus.HEALTHY]
        assert strategy.aggregate(with_healthy) == HealthStatus.HEALTHY

        no_healthy = [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        assert strategy.aggregate(no_healthy) == HealthStatus.UNHEALTHY

    def test_empty_statuses(self):
        """Test aggregation with empty statuses."""
        strategy = AggregationStrategy.WORST
        assert strategy.aggregate([]) == HealthStatus.UNKNOWN


class TestCompositeHealthChecker:
    """Tests for CompositeHealthChecker."""

    def test_composite_check(self):
        """Test composite health check."""
        checker1 = SimpleHealthChecker("db", lambda: True)
        checker2 = SimpleHealthChecker("cache", lambda: True)

        composite = CompositeHealthChecker(
            checkers=[checker1, checker2],
            name="system",
        )

        result = composite.check()
        assert result.status == HealthStatus.HEALTHY
        assert len(result.dependencies) == 2

    def test_composite_with_failure(self):
        """Test composite with one failing check."""
        checker1 = SimpleHealthChecker("db", lambda: True)
        checker2 = SimpleHealthChecker("cache", lambda: False)

        composite = CompositeHealthChecker(
            checkers=[checker1, checker2],
            name="system",
        )

        result = composite.check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_composite_parallel(self):
        """Test parallel execution."""
        start_times = []

        def slow_check():
            start_times.append(time.time())
            time.sleep(0.1)
            return True

        checker1 = SimpleHealthChecker("db", slow_check)
        checker2 = SimpleHealthChecker("cache", slow_check)

        composite = CompositeHealthChecker(
            checkers=[checker1, checker2],
            name="system",
            parallel=True,
        )

        start = time.time()
        composite.check()
        duration = time.time() - start

        # Parallel should be faster than sequential
        assert duration < 0.2  # Would be 0.2+ if sequential

    def test_composite_sequential(self):
        """Test sequential execution."""

        def slow_check():
            time.sleep(0.05)
            return True

        checker1 = SimpleHealthChecker("db", slow_check)
        checker2 = SimpleHealthChecker("cache", slow_check)

        composite = CompositeHealthChecker(
            checkers=[checker1, checker2],
            name="system",
            parallel=False,
        )

        start = time.time()
        composite.check()
        duration = time.time() - start

        # Sequential should take longer
        assert duration >= 0.1

    def test_add_remove_checker(self):
        """Test adding and removing checkers."""
        checker1 = SimpleHealthChecker("db", lambda: True)
        composite = CompositeHealthChecker([], name="system")

        composite.add_checker(checker1)
        result = composite.check()
        assert len(result.dependencies) == 1

        composite.remove_checker("db")
        result = composite.check()
        assert len(result.dependencies) == 0

    @asyncio_test
    @pytest.mark.asyncio
    async def test_composite_async(self):
        """Test async composite check."""
        checker1 = SimpleHealthChecker("db", lambda: True)
        checker2 = SimpleHealthChecker("cache", lambda: True)

        composite = CompositeHealthChecker(
            checkers=[checker1, checker2],
            name="system",
        )

        result = await composite.check_async()
        assert result.status == HealthStatus.HEALTHY


class TestHealthCheckRegistry:
    """Tests for HealthCheckRegistry."""

    def test_register_and_check(self):
        """Test registering and checking."""
        registry = HealthCheckRegistry()
        registry.register("test", lambda: True)

        result = registry.check("test")
        assert result.status == HealthStatus.HEALTHY

    def test_register_checker(self):
        """Test registering a HealthChecker."""
        registry = HealthCheckRegistry()
        checker = SimpleHealthChecker("test", lambda: True)
        registry.register("test", checker)

        result = registry.check("test")
        assert result.status == HealthStatus.HEALTHY

    def test_unregister(self):
        """Test unregistering a checker."""
        registry = HealthCheckRegistry()
        registry.register("test", lambda: True)
        assert registry.unregister("test") is True
        assert registry.unregister("nonexistent") is False

    def test_get(self):
        """Test getting a checker."""
        registry = HealthCheckRegistry()
        registry.register("test", lambda: True)

        checker = registry.get("test")
        assert checker is not None
        assert registry.get("nonexistent") is None

    def test_check_not_found(self):
        """Test checking a non-existent checker."""
        registry = HealthCheckRegistry()
        result = registry.check("nonexistent")
        assert result.status == HealthStatus.UNKNOWN

    def test_check_all(self):
        """Test checking all registered checkers."""
        registry = HealthCheckRegistry()
        registry.register("db", lambda: True)
        registry.register("cache", lambda: True)

        result = registry.check_all()
        assert result.status == HealthStatus.HEALTHY
        assert len(result.dependencies) == 2

    def test_check_all_empty(self):
        """Test check_all with no registrations."""
        registry = HealthCheckRegistry()
        result = registry.check_all()
        assert result.status == HealthStatus.UNKNOWN

    def test_names(self):
        """Test getting all names."""
        registry = HealthCheckRegistry()
        registry.register("db", lambda: True)
        registry.register("cache", lambda: True)

        names = registry.names
        assert "db" in names
        assert "cache" in names

    def test_clear(self):
        """Test clearing all registrations."""
        registry = HealthCheckRegistry()
        registry.register("test", lambda: True)
        registry.clear()
        assert registry.names == []


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_health_registry(self):
        """Test getting the global registry."""
        registry = get_health_registry()
        assert isinstance(registry, HealthCheckRegistry)

    def test_register_and_check_global(self):
        """Test global register and check functions."""
        # Clean up first
        registry = get_health_registry()
        registry.clear()

        register_health_check("global_test", lambda: True)
        result = check_health("global_test")
        assert result.status == HealthStatus.HEALTHY

        # Clean up
        registry.unregister("global_test")

    def test_check_all_global(self):
        """Test global check_all function."""
        registry = get_health_registry()
        registry.clear()

        register_health_check("test1", lambda: True)
        register_health_check("test2", lambda: True)

        result = check_all_health()
        assert result.status == HealthStatus.HEALTHY

        # Clean up
        registry.clear()


class TestHealthCheckDecorator:
    """Tests for health_check decorator."""

    def test_basic_decorator(self):
        """Test basic decorator usage."""

        @health_check(name="decorated_test")
        def my_check():
            return True

        result = my_check()
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "decorated_test"

    def test_decorator_with_exception(self):
        """Test decorator with exception."""

        @health_check(name="failing_test")
        def my_check():
            raise ValueError("Test error")

        result = my_check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_decorator_with_config(self):
        """Test decorator with custom config."""
        config = HealthCheckConfig(timeout_seconds=1.0)

        @health_check(name="configured_test", config=config)
        def my_check():
            return True

        result = my_check()
        assert result.status == HealthStatus.HEALTHY

    @asyncio_test
    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test decorator with async function."""

        @health_check(name="async_test")
        async def my_check():
            return True

        result = await my_check()
        assert result.status == HealthStatus.HEALTHY


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_health_checker(self):
        """Test create_health_checker function."""
        checker = create_health_checker(
            name="test",
            check_func=lambda: True,
        )
        assert checker.name == "test"
        result = checker.check()
        assert result.status == HealthStatus.HEALTHY

    @asyncio_test
    @pytest.mark.asyncio
    async def test_create_async_health_checker(self):
        """Test create_async_health_checker function."""

        async def check_func():
            return True

        checker = create_async_health_checker(
            name="test",
            check_func=check_func,
        )
        assert checker.name == "test"
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY

    def test_create_composite_checker(self):
        """Test create_composite_checker function."""
        checker1 = SimpleHealthChecker("db", lambda: True)
        checker2 = SimpleHealthChecker("cache", lambda: True)

        composite = create_composite_checker(
            name="system",
            checkers=[checker1, checker2],
        )
        assert composite.name == "system"
        result = composite.check()
        assert result.status == HealthStatus.HEALTHY


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_database_health_check_pattern(self):
        """Test typical database health check pattern."""

        def check_database():
            # Simulate database ping
            return HealthCheckResult.healthy(
                "database",
                message="Database is responsive",
                latency_ms=5.2,
                connections_active=10,
            )

        checker = create_health_checker("database", check_database)
        result = checker.check()

        assert result.is_healthy
        assert result.details["latency_ms"] == 5.2
        assert result.details["connections_active"] == 10

    def test_degraded_service_pattern(self):
        """Test degraded service pattern."""

        def check_cache():
            # Simulate degraded cache
            return HealthCheckResult.degraded(
                "cache",
                message="Cache hit rate below threshold",
                hit_rate=0.65,
                threshold=0.80,
            )

        checker = create_health_checker("cache", check_cache)
        result = checker.check()

        assert result.status == HealthStatus.DEGRADED
        assert result.is_operational

    def test_multi_component_system(self):
        """Test multi-component system health check."""
        db_checker = create_health_checker(
            "database",
            lambda: HealthCheckResult.healthy("database"),
        )
        cache_checker = create_health_checker(
            "cache",
            lambda: HealthCheckResult.healthy("cache"),
        )
        api_checker = create_health_checker(
            "api",
            lambda: HealthCheckResult.healthy("api"),
        )

        system_checker = create_composite_checker(
            name="system",
            checkers=[db_checker, cache_checker, api_checker],
        )

        result = system_checker.check()

        assert result.is_healthy
        assert len(result.dependencies) == 3
        assert result.details["healthy_count"] == 3
        assert result.details["total_count"] == 3

    def test_fallback_on_unhealthy(self):
        """Test fallback logic when service is unhealthy."""
        primary_healthy = True

        def check_primary():
            if primary_healthy:
                return HealthCheckResult.healthy("primary")
            return HealthCheckResult.unhealthy("primary", "Service down")

        checker = create_health_checker("primary", check_primary)
        result = checker.check()
        assert result.is_healthy

        # Simulate failure
        primary_healthy = False
        result = checker.check()
        assert not result.is_healthy

        # Application would use fallback here
        if not result.is_healthy:
            # Use fallback service
            pass
