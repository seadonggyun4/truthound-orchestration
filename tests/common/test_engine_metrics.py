"""Tests for Engine Metrics Integration.

Tests cover:
- EngineMetricsHook protocol compliance
- MetricsEngineHook metrics collection
- LoggingEngineHook structured logging
- TracingEngineHook span creation
- CompositeEngineHook hook composition
- InstrumentedEngine wrapper functionality
- AsyncInstrumentedEngine async functionality
- Statistics collection
- Error handling and hook isolation
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

import pytest

from common.base import (
    CheckResult,
    CheckStatus,
    LearnResult,
    LearnStatus,
    ProfileResult,
    ProfileStatus,
)
from common.engines.base import EngineCapabilities, EngineInfoMixin
from common.engines.metrics import (
    DEFAULT_ENGINE_METRICS_CONFIG,
    DISABLED_ENGINE_METRICS_CONFIG,
    FULL_ENGINE_METRICS_CONFIG,
    MINIMAL_ENGINE_METRICS_CONFIG,
    AsyncBaseEngineMetricsHook,
    AsyncCompositeEngineHook,
    AsyncEngineMetricsHook,
    AsyncInstrumentedEngine,
    # Base Hooks
    BaseEngineMetricsHook,
    CompositeEngineHook,
    # Configuration
    EngineMetricsConfig,
    # Exceptions
    EngineMetricsError,
    # Protocols
    EngineMetricsHook,
    # Enums
    EngineOperation,
    # Statistics
    EngineOperationStats,
    # Instrumented Engines
    InstrumentedEngine,
    LoggingEngineHook,
    # Hook Implementations
    MetricsEngineHook,
    OperationStatus,
    StatsCollectorHook,
    # Adapters
    SyncToAsyncEngineHookAdapter,
    TracingEngineHook,
    create_async_instrumented_engine,
    # Factory Functions
    create_instrumented_engine,
)


# =============================================================================
# Mock Engine for Testing
# =============================================================================


class MockEngine(EngineInfoMixin):
    """Mock engine for testing."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        should_raise: bool = False,
        delay_ms: float = 0,
    ) -> None:
        """Initialize mock engine.

        Args:
            should_fail: Whether check/profile/learn should return failure status.
            should_raise: Whether to raise an exception.
            delay_ms: Delay to simulate work.
        """
        self.should_fail = should_fail
        self.should_raise = should_raise
        self.delay_ms = delay_ms
        self.call_history: list[dict[str, Any]] = []

    @property
    def engine_name(self) -> str:
        return "mock_engine"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def _get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
        )

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> CheckResult:
        """Execute check."""
        self.call_history.append({
            "method": "check",
            "data": data,
            "rules": rules,
            "kwargs": kwargs,
        })

        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)

        if self.should_raise:
            raise ValueError("Mock check error")

        if self.should_fail:
            return CheckResult(
                status=CheckStatus.FAILED,
                passed_count=5,
                failed_count=3,
            )

        return CheckResult(
            status=CheckStatus.PASSED,
            passed_count=10,
            failed_count=0,
        )

    def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        """Execute profile."""
        self.call_history.append({
            "method": "profile",
            "data": data,
            "kwargs": kwargs,
        })

        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)

        if self.should_raise:
            raise ValueError("Mock profile error")

        if self.should_fail:
            return ProfileResult(
                status=ProfileStatus.FAILED,
                row_count=100,
                column_count=5,
            )

        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            row_count=1000,
            column_count=10,
        )

    def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        """Execute learn."""
        self.call_history.append({
            "method": "learn",
            "data": data,
            "kwargs": kwargs,
        })

        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)

        if self.should_raise:
            raise ValueError("Mock learn error")

        if self.should_fail:
            return LearnResult(
                status=LearnStatus.FAILED,
                columns_analyzed=3,
            )

        return LearnResult(
            status=LearnStatus.COMPLETED,
            columns_analyzed=10,
        )


class AsyncMockEngine:
    """Async mock engine for testing."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        should_raise: bool = False,
        delay_ms: float = 0,
    ) -> None:
        self.should_fail = should_fail
        self.should_raise = should_raise
        self.delay_ms = delay_ms
        self.call_history: list[dict[str, Any]] = []

    @property
    def engine_name(self) -> str:
        return "async_mock_engine"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    async def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> CheckResult:
        """Execute async check."""
        self.call_history.append({"method": "check", "data": data})

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        if self.should_raise:
            raise ValueError("Async mock check error")

        status = CheckStatus.FAILED if self.should_fail else CheckStatus.PASSED
        return CheckResult(status=status, passed_count=10 if not self.should_fail else 5)

    async def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        """Execute async profile."""
        self.call_history.append({"method": "profile", "data": data})

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        if self.should_raise:
            raise ValueError("Async mock profile error")

        status = ProfileStatus.FAILED if self.should_fail else ProfileStatus.COMPLETED
        return ProfileResult(status=status, row_count=1000)

    async def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        """Execute async learn."""
        self.call_history.append({"method": "learn", "data": data})

        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)

        if self.should_raise:
            raise ValueError("Async mock learn error")

        status = LearnStatus.FAILED if self.should_fail else LearnStatus.COMPLETED
        return LearnResult(status=status, columns_analyzed=10)


# =============================================================================
# Test Configuration
# =============================================================================


class TestEngineMetricsConfig:
    """Tests for EngineMetricsConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EngineMetricsConfig()
        assert config.enabled is True
        assert config.prefix == "engine"
        assert config.include_data_size is True
        assert config.include_result_counts is True
        assert config.include_tracing is True
        assert len(config.histogram_buckets) > 0

    def test_disabled_config_preset(self) -> None:
        """Test disabled preset."""
        config = DISABLED_ENGINE_METRICS_CONFIG
        assert config.enabled is False

    def test_minimal_config_preset(self) -> None:
        """Test minimal preset."""
        config = MINIMAL_ENGINE_METRICS_CONFIG
        assert config.include_data_size is False
        assert config.include_result_counts is False
        assert config.include_tracing is False

    def test_full_config_preset(self) -> None:
        """Test full preset."""
        config = FULL_ENGINE_METRICS_CONFIG
        assert config.include_data_size is True
        assert config.include_result_counts is True
        assert config.include_tracing is True
        assert len(config.histogram_buckets) > len(DEFAULT_ENGINE_METRICS_CONFIG.histogram_buckets)

    def test_with_enabled(self) -> None:
        """Test with_enabled builder method."""
        config = EngineMetricsConfig()
        new_config = config.with_enabled(False)
        assert new_config.enabled is False
        assert config.enabled is True  # Original unchanged

    def test_with_prefix(self) -> None:
        """Test with_prefix builder method."""
        config = EngineMetricsConfig()
        new_config = config.with_prefix("custom")
        assert new_config.prefix == "custom"
        assert config.prefix == "engine"

    def test_with_default_labels(self) -> None:
        """Test with_default_labels builder method."""
        config = EngineMetricsConfig()
        new_config = config.with_default_labels(env="test", service="data_quality")
        assert new_config.default_labels == {"env": "test", "service": "data_quality"}
        assert config.default_labels == {}

    def test_with_histogram_buckets(self) -> None:
        """Test with_histogram_buckets builder method."""
        config = EngineMetricsConfig()
        new_config = config.with_histogram_buckets(0.1, 1.0, 10.0)
        assert new_config.histogram_buckets == (0.1, 1.0, 10.0)

    def test_to_dict_from_dict(self) -> None:
        """Test serialization round-trip."""
        config = EngineMetricsConfig(
            enabled=True,
            prefix="test",
            include_data_size=False,
            default_labels={"env": "prod"},
        )
        data = config.to_dict()
        restored = EngineMetricsConfig.from_dict(data)
        assert restored.enabled == config.enabled
        assert restored.prefix == config.prefix
        assert restored.include_data_size == config.include_data_size
        assert restored.default_labels == config.default_labels


# =============================================================================
# Test Base Hooks
# =============================================================================


class TestBaseEngineMetricsHook:
    """Tests for BaseEngineMetricsHook."""

    def test_all_methods_no_op(self) -> None:
        """Test that base hook methods are no-ops."""
        hook = BaseEngineMetricsHook()
        result = CheckResult(status=CheckStatus.PASSED)

        # All should run without error
        hook.on_check_start("test", None, {})
        hook.on_check_end("test", result, 100.0, {})
        hook.on_profile_start("test", None, {})
        hook.on_profile_end("test", ProfileResult(status=ProfileStatus.COMPLETED), 100.0, {})
        hook.on_learn_start("test", None, {})
        hook.on_learn_end("test", LearnResult(status=LearnStatus.COMPLETED), 100.0, {})
        hook.on_error("test", EngineOperation.CHECK, ValueError(), 100.0, {})


class TestAsyncBaseEngineMetricsHook:
    """Tests for AsyncBaseEngineMetricsHook."""

    @pytest.mark.asyncio
    async def test_all_methods_no_op(self) -> None:
        """Test that async base hook methods are no-ops."""
        hook = AsyncBaseEngineMetricsHook()
        result = CheckResult(status=CheckStatus.PASSED)

        # All should run without error
        await hook.on_check_start("test", None, {})
        await hook.on_check_end("test", result, 100.0, {})
        await hook.on_profile_start("test", None, {})
        await hook.on_profile_end("test", ProfileResult(status=ProfileStatus.COMPLETED), 100.0, {})
        await hook.on_learn_start("test", None, {})
        await hook.on_learn_end("test", LearnResult(status=LearnStatus.COMPLETED), 100.0, {})
        await hook.on_error("test", EngineOperation.CHECK, ValueError(), 100.0, {})


# =============================================================================
# Test MetricsEngineHook
# =============================================================================


class TestMetricsEngineHook:
    """Tests for MetricsEngineHook."""

    def test_protocol_compliance(self) -> None:
        """Test that MetricsEngineHook implements the protocol."""
        hook = MetricsEngineHook()
        assert isinstance(hook, EngineMetricsHook)

    def test_check_metrics_collection(self) -> None:
        """Test metrics are collected for check operations."""
        hook = MetricsEngineHook()
        result = CheckResult(
            status=CheckStatus.PASSED,
            passed_count=10,
            failed_count=0,
        )

        hook.on_check_start("test_engine", 1000, {"data_size": 1000})
        hook.on_check_end("test_engine", result, 150.5, {"data_size": 1000})

        # Metrics should be collected (no exception)
        # Actual metric values are internal to the registry

    def test_disabled_config(self) -> None:
        """Test that disabled config prevents metrics collection."""
        config = EngineMetricsConfig(enabled=False)
        hook = MetricsEngineHook(config=config)

        # Should not raise even though metrics aren't initialized
        hook.on_check_start("test", 100, {})
        hook.on_check_end(
            "test",
            CheckResult(status=CheckStatus.PASSED),
            100.0,
            {},
        )

    def test_error_metrics(self) -> None:
        """Test error metrics collection."""
        hook = MetricsEngineHook()
        exception = ValueError("test error")

        hook.on_check_start("test_engine", 1000, {})
        hook.on_error(
            "test_engine",
            EngineOperation.CHECK,
            exception,
            50.0,
            {},
        )


# =============================================================================
# Test LoggingEngineHook
# =============================================================================


class TestLoggingEngineHook:
    """Tests for LoggingEngineHook."""

    def test_protocol_compliance(self) -> None:
        """Test that LoggingEngineHook implements the protocol."""
        hook = LoggingEngineHook()
        assert isinstance(hook, EngineMetricsHook)

    def test_check_logging(self) -> None:
        """Test logging for check operations."""
        hook = LoggingEngineHook()

        with patch.object(hook._logger, 'debug') as mock_debug:
            hook.on_check_start("test_engine", 1000, {})
            mock_debug.assert_called_once()

        with patch.object(hook._logger, 'info') as mock_info:
            result = CheckResult(status=CheckStatus.PASSED, passed_count=10)
            hook.on_check_end("test_engine", result, 100.0, {})
            mock_info.assert_called_once()

    def test_warning_on_failure(self) -> None:
        """Test warning log level on failure result."""
        hook = LoggingEngineHook()

        with patch.object(hook._logger, 'warning') as mock_warning:
            result = CheckResult(status=CheckStatus.FAILED, failed_count=5)
            hook.on_check_end("test_engine", result, 100.0, {})
            mock_warning.assert_called_once()

    def test_error_logging(self) -> None:
        """Test error logging."""
        hook = LoggingEngineHook()

        with patch.object(hook._logger, 'error') as mock_error:
            hook.on_error(
                "test_engine",
                EngineOperation.CHECK,
                ValueError("test"),
                100.0,
                {},
            )
            mock_error.assert_called_once()

    def test_disable_start_logging(self) -> None:
        """Test disabling start event logging."""
        hook = LoggingEngineHook(log_start=False)

        with patch.object(hook._logger, 'debug') as mock_debug:
            hook.on_check_start("test_engine", 1000, {})
            mock_debug.assert_not_called()

    def test_disable_end_logging(self) -> None:
        """Test disabling end event logging."""
        hook = LoggingEngineHook(log_end=False)

        with patch.object(hook._logger, 'info') as mock_info:
            result = CheckResult(status=CheckStatus.PASSED)
            hook.on_check_end("test_engine", result, 100.0, {})
            mock_info.assert_not_called()


# =============================================================================
# Test TracingEngineHook
# =============================================================================


class TestTracingEngineHook:
    """Tests for TracingEngineHook."""

    def test_protocol_compliance(self) -> None:
        """Test that TracingEngineHook implements the protocol."""
        hook = TracingEngineHook()
        assert isinstance(hook, EngineMetricsHook)

    def test_disabled_tracing(self) -> None:
        """Test with tracing disabled in config."""
        config = EngineMetricsConfig(include_tracing=False)
        hook = TracingEngineHook(config=config)

        # Should not create spans when disabled
        hook.on_check_start("test", 100, {})
        hook.on_check_end("test", CheckResult(status=CheckStatus.PASSED), 100.0, {})


# =============================================================================
# Test CompositeEngineHook
# =============================================================================


class TestCompositeEngineHook:
    """Tests for CompositeEngineHook."""

    def test_protocol_compliance(self) -> None:
        """Test that CompositeEngineHook implements the protocol."""
        hook = CompositeEngineHook()
        assert isinstance(hook, EngineMetricsHook)

    def test_calls_all_hooks(self) -> None:
        """Test that composite calls all hooks."""
        mock_hook1 = MagicMock(spec=EngineMetricsHook)
        mock_hook2 = MagicMock(spec=EngineMetricsHook)

        composite = CompositeEngineHook([mock_hook1, mock_hook2])
        composite.on_check_start("test", 100, {})

        mock_hook1.on_check_start.assert_called_once_with("test", 100, {})
        mock_hook2.on_check_start.assert_called_once_with("test", 100, {})

    def test_hook_isolation(self) -> None:
        """Test that one hook error doesn't affect others."""
        failing_hook = MagicMock(spec=EngineMetricsHook)
        failing_hook.on_check_start.side_effect = ValueError("Hook error")
        successful_hook = MagicMock(spec=EngineMetricsHook)

        composite = CompositeEngineHook([failing_hook, successful_hook])
        composite.on_check_start("test", 100, {})

        # Second hook should still be called
        successful_hook.on_check_start.assert_called_once()

    def test_add_hook(self) -> None:
        """Test adding hooks dynamically."""
        composite = CompositeEngineHook()
        assert len(composite.hooks) == 0

        mock_hook = MagicMock(spec=EngineMetricsHook)
        composite.add_hook(mock_hook)
        assert len(composite.hooks) == 1

    def test_remove_hook(self) -> None:
        """Test removing hooks."""
        mock_hook = MagicMock(spec=EngineMetricsHook)
        composite = CompositeEngineHook([mock_hook])

        result = composite.remove_hook(mock_hook)
        assert result is True
        assert len(composite.hooks) == 0

        result = composite.remove_hook(mock_hook)
        assert result is False


class TestAsyncCompositeEngineHook:
    """Tests for AsyncCompositeEngineHook."""

    @pytest.mark.asyncio
    async def test_calls_all_hooks(self) -> None:
        """Test that async composite calls all hooks."""
        mock_hook1 = MagicMock(spec=AsyncEngineMetricsHook)
        mock_hook1.on_check_start = AsyncMock()
        mock_hook2 = MagicMock(spec=AsyncEngineMetricsHook)
        mock_hook2.on_check_start = AsyncMock()

        composite = AsyncCompositeEngineHook([mock_hook1, mock_hook2])
        await composite.on_check_start("test", 100, {})

        mock_hook1.on_check_start.assert_called_once()
        mock_hook2.on_check_start.assert_called_once()


# AsyncMock helper for older Python versions
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


# =============================================================================
# Test StatsCollectorHook
# =============================================================================


class TestStatsCollectorHook:
    """Tests for StatsCollectorHook."""

    def test_protocol_compliance(self) -> None:
        """Test that StatsCollectorHook implements the protocol."""
        hook = StatsCollectorHook()
        assert isinstance(hook, EngineMetricsHook)

    def test_collects_check_stats(self) -> None:
        """Test statistics collection for check operations."""
        hook = StatsCollectorHook()
        result = CheckResult(status=CheckStatus.PASSED, passed_count=10)

        hook.on_check_end("test", result, 100.0, {"data_size": 1000})

        stats = hook.stats
        assert stats.total_operations == 1
        assert stats.successful_operations == 1
        assert stats.failed_operations == 0
        assert stats.total_duration_ms == 100.0
        assert stats.total_rows_processed == 1000
        assert stats.operation_counts["check"] == 1

    def test_collects_failure_stats(self) -> None:
        """Test statistics collection for failures."""
        hook = StatsCollectorHook()
        result = CheckResult(status=CheckStatus.FAILED, failed_count=5)

        hook.on_check_end("test", result, 100.0, {})

        stats = hook.stats
        assert stats.failed_operations == 1
        assert stats.successful_operations == 0

    def test_collects_error_stats(self) -> None:
        """Test statistics collection for errors."""
        hook = StatsCollectorHook()

        hook.on_error(
            "test",
            EngineOperation.CHECK,
            ValueError("test"),
            50.0,
            {},
        )

        stats = hook.stats
        assert stats.error_operations == 1
        assert stats.error_counts["ValueError"] == 1

    def test_reset_stats(self) -> None:
        """Test statistics reset."""
        hook = StatsCollectorHook()
        hook.on_check_end("test", CheckResult(status=CheckStatus.PASSED), 100.0, {})

        assert hook.stats.total_operations == 1

        hook.reset()
        assert hook.stats.total_operations == 0

    def test_stats_properties(self) -> None:
        """Test computed properties."""
        hook = StatsCollectorHook()

        # 3 successes, 1 failure, 1 error
        hook.on_check_end("test", CheckResult(status=CheckStatus.PASSED), 100.0, {})
        hook.on_check_end("test", CheckResult(status=CheckStatus.PASSED), 100.0, {})
        hook.on_check_end("test", CheckResult(status=CheckStatus.PASSED), 100.0, {})
        hook.on_check_end("test", CheckResult(status=CheckStatus.FAILED), 100.0, {})
        hook.on_error("test", EngineOperation.CHECK, ValueError(), 100.0, {})

        stats = hook.stats
        assert stats.total_operations == 5
        assert stats.average_duration_ms == 100.0
        assert stats.success_rate == 60.0  # 3/5
        assert stats.error_rate == 20.0  # 1/5


# =============================================================================
# Test InstrumentedEngine
# =============================================================================


class TestInstrumentedEngine:
    """Tests for InstrumentedEngine wrapper."""

    def test_proxies_engine_properties(self) -> None:
        """Test that engine properties are proxied."""
        base_engine = MockEngine()
        engine = InstrumentedEngine(base_engine)

        assert engine.engine_name == "mock_engine"
        assert engine.engine_version == "1.0.0"

    def test_check_calls_hooks(self) -> None:
        """Test that check operation calls hooks."""
        base_engine = MockEngine()
        stats_hook = StatsCollectorHook()
        engine = InstrumentedEngine(base_engine, hooks=[stats_hook])

        result = engine.check([1, 2, 3], rules=[])

        assert result.status == CheckStatus.PASSED
        assert stats_hook.stats.total_operations == 1
        assert stats_hook.stats.operation_counts["check"] == 1

    def test_profile_calls_hooks(self) -> None:
        """Test that profile operation calls hooks."""
        base_engine = MockEngine()
        stats_hook = StatsCollectorHook()
        engine = InstrumentedEngine(base_engine, hooks=[stats_hook])

        result = engine.profile([1, 2, 3])

        assert result.status == ProfileStatus.COMPLETED
        assert stats_hook.stats.operation_counts["profile"] == 1

    def test_learn_calls_hooks(self) -> None:
        """Test that learn operation calls hooks."""
        base_engine = MockEngine()
        stats_hook = StatsCollectorHook()
        engine = InstrumentedEngine(base_engine, hooks=[stats_hook])

        result = engine.learn([1, 2, 3])

        assert result.status == LearnStatus.COMPLETED
        assert stats_hook.stats.operation_counts["learn"] == 1

    def test_error_calls_hooks(self) -> None:
        """Test that errors call hooks."""
        base_engine = MockEngine(should_raise=True)
        stats_hook = StatsCollectorHook()
        engine = InstrumentedEngine(base_engine, hooks=[stats_hook])

        with pytest.raises(ValueError):
            engine.check([1, 2, 3])

        assert stats_hook.stats.error_operations == 1

    def test_disabled_config(self) -> None:
        """Test with disabled configuration."""
        base_engine = MockEngine()
        stats_hook = StatsCollectorHook()
        config = EngineMetricsConfig(enabled=False)
        engine = InstrumentedEngine(base_engine, hooks=[stats_hook], config=config)

        engine.check([1, 2, 3])

        # Hooks should not be called when disabled
        assert stats_hook.stats.total_operations == 0

    def test_hook_exception_isolation(self) -> None:
        """Test that hook exceptions don't affect engine."""
        base_engine = MockEngine()
        failing_hook = MagicMock(spec=EngineMetricsHook)
        failing_hook.on_check_start.side_effect = ValueError("Hook error")
        failing_hook.on_check_end.side_effect = ValueError("Hook error")

        engine = InstrumentedEngine(base_engine, hooks=[failing_hook])

        # Should not raise despite hook errors
        result = engine.check([1, 2, 3])
        assert result.status == CheckStatus.PASSED

    def test_add_hook(self) -> None:
        """Test adding hooks dynamically."""
        base_engine = MockEngine()
        engine = InstrumentedEngine(base_engine)

        stats_hook = StatsCollectorHook()
        engine.add_hook(stats_hook)

        engine.check([1, 2, 3])
        assert stats_hook.stats.total_operations == 1

    def test_context_manager(self) -> None:
        """Test context manager support."""
        base_engine = MockEngine()

        with InstrumentedEngine(base_engine) as engine:
            result = engine.check([1, 2, 3])
            assert result.status == CheckStatus.PASSED

    def test_proxies_lifecycle_methods(self) -> None:
        """Test that lifecycle methods are proxied."""
        base_engine = MockEngine()
        base_engine.start = MagicMock()
        base_engine.stop = MagicMock()

        engine = InstrumentedEngine(base_engine)
        engine.start()
        engine.stop()

        base_engine.start.assert_called_once()
        base_engine.stop.assert_called_once()


# =============================================================================
# Test AsyncInstrumentedEngine
# =============================================================================


class TestAsyncInstrumentedEngine:
    """Tests for AsyncInstrumentedEngine wrapper."""

    @pytest.mark.asyncio
    async def test_check_calls_hooks(self) -> None:
        """Test that async check operation calls hooks."""
        base_engine = AsyncMockEngine()
        sync_hook = StatsCollectorHook()
        async_hook = SyncToAsyncEngineHookAdapter(sync_hook)
        engine = AsyncInstrumentedEngine(base_engine, hooks=[async_hook])

        result = await engine.check([1, 2, 3])

        assert result.status == CheckStatus.PASSED
        assert sync_hook.stats.total_operations == 1

    @pytest.mark.asyncio
    async def test_error_calls_hooks(self) -> None:
        """Test that async errors call hooks."""
        base_engine = AsyncMockEngine(should_raise=True)
        sync_hook = StatsCollectorHook()
        async_hook = SyncToAsyncEngineHookAdapter(sync_hook)
        engine = AsyncInstrumentedEngine(base_engine, hooks=[async_hook])

        with pytest.raises(ValueError):
            await engine.check([1, 2, 3])

        assert sync_hook.stats.error_operations == 1


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_instrumented_engine_defaults(self) -> None:
        """Test create_instrumented_engine with defaults."""
        base_engine = MockEngine()
        engine = create_instrumented_engine(base_engine)

        assert isinstance(engine, InstrumentedEngine)
        result = engine.check([1, 2, 3])
        assert result.status == CheckStatus.PASSED

    def test_create_instrumented_engine_custom(self) -> None:
        """Test create_instrumented_engine with custom options."""
        base_engine = MockEngine()
        engine = create_instrumented_engine(
            base_engine,
            enable_metrics=True,
            enable_logging=False,
            enable_tracing=True,
        )

        assert isinstance(engine, InstrumentedEngine)

    def test_create_async_instrumented_engine(self) -> None:
        """Test create_async_instrumented_engine."""
        base_engine = AsyncMockEngine()
        engine = create_async_instrumented_engine(base_engine)

        assert isinstance(engine, AsyncInstrumentedEngine)


# =============================================================================
# Test SyncToAsyncEngineHookAdapter
# =============================================================================


class TestSyncToAsyncEngineHookAdapter:
    """Tests for SyncToAsyncEngineHookAdapter."""

    @pytest.mark.asyncio
    async def test_wraps_sync_hook(self) -> None:
        """Test adapter wraps sync hook correctly."""
        sync_hook = StatsCollectorHook()
        async_hook = SyncToAsyncEngineHookAdapter(sync_hook)

        await async_hook.on_check_start("test", 100, {})
        await async_hook.on_check_end(
            "test",
            CheckResult(status=CheckStatus.PASSED),
            100.0,
            {},
        )

        assert sync_hook.stats.total_operations == 1


# =============================================================================
# Test EngineOperationStats
# =============================================================================


class TestEngineOperationStats:
    """Tests for EngineOperationStats."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = EngineOperationStats()
        assert stats.total_operations == 0
        assert stats.average_duration_ms == 0.0
        assert stats.success_rate == 100.0  # Default to 100% when no operations
        assert stats.error_rate == 0.0

    def test_computed_properties(self) -> None:
        """Test computed properties."""
        stats = EngineOperationStats(
            total_operations=10,
            successful_operations=7,
            failed_operations=2,
            error_operations=1,
            total_duration_ms=1000.0,
        )

        assert stats.average_duration_ms == 100.0
        assert stats.success_rate == 70.0
        assert stats.error_rate == 10.0


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_engine_operation(self) -> None:
        """Test EngineOperation enum."""
        assert EngineOperation.CHECK.name == "CHECK"
        assert EngineOperation.PROFILE.name == "PROFILE"
        assert EngineOperation.LEARN.name == "LEARN"
        assert EngineOperation.HEALTH_CHECK.name == "HEALTH_CHECK"

    def test_operation_status(self) -> None:
        """Test OperationStatus enum."""
        assert OperationStatus.SUCCESS.name == "SUCCESS"
        assert OperationStatus.FAILURE.name == "FAILURE"
        assert OperationStatus.ERROR.name == "ERROR"


# =============================================================================
# Test Exception
# =============================================================================


class TestEngineMetricsError:
    """Tests for EngineMetricsError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = EngineMetricsError("Test error")
        assert str(error) == "Test error"

    def test_error_with_context(self) -> None:
        """Test error with engine context."""
        error = EngineMetricsError(
            "Test error",
            engine_name="test_engine",
            operation="check",
        )
        assert error.engine_name == "test_engine"
        assert error.operation == "check"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for engine metrics."""

    def test_full_workflow(self) -> None:
        """Test complete workflow with multiple hooks."""
        base_engine = MockEngine()
        stats_hook = StatsCollectorHook()
        logging_hook = LoggingEngineHook(log_start=False)

        engine = InstrumentedEngine(
            base_engine,
            hooks=[stats_hook, logging_hook],
        )

        # Run multiple operations
        for _ in range(5):
            engine.check([1, 2, 3])
        for _ in range(3):
            engine.profile([1, 2, 3])
        for _ in range(2):
            engine.learn([1, 2, 3])

        stats = stats_hook.stats
        assert stats.total_operations == 10
        assert stats.operation_counts["check"] == 5
        assert stats.operation_counts["profile"] == 3
        assert stats.operation_counts["learn"] == 2

    def test_mixed_success_and_failure(self) -> None:
        """Test with mixed success and failure results."""
        stats_hook = StatsCollectorHook()

        # Success engine
        success_engine = MockEngine()
        instrumented_success = InstrumentedEngine(success_engine, hooks=[stats_hook])
        instrumented_success.check([1, 2, 3])

        # Failure engine
        failure_engine = MockEngine(should_fail=True)
        instrumented_failure = InstrumentedEngine(failure_engine, hooks=[stats_hook])
        instrumented_failure.check([1, 2, 3])

        stats = stats_hook.stats
        assert stats.successful_operations == 1
        assert stats.failed_operations == 1

    def test_duration_tracking(self) -> None:
        """Test that durations are tracked correctly."""
        base_engine = MockEngine(delay_ms=50)  # 50ms delay
        stats_hook = StatsCollectorHook()
        engine = InstrumentedEngine(base_engine, hooks=[stats_hook])

        engine.check([1, 2, 3])

        stats = stats_hook.stats
        assert stats.total_duration_ms >= 50  # At least 50ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
