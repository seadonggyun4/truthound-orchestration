"""Tests for Engine Chain and Fallback patterns.

This module contains comprehensive tests for the chain.py module including:
- EngineChain with various fallback strategies
- ConditionalEngineChain for condition-based routing
- SelectorEngineChain for custom selection logic
- AsyncEngineChain for async engines
- Hook implementations
- Factory functions
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from common.base import (
    CheckResult,
    CheckStatus,
    LearnResult,
    LearnStatus,
    ProfileResult,
    ProfileStatus,
)
from common.engines.base import DataQualityEngine, EngineCapabilities, EngineInfoMixin
from common.engines.chain import (
    DEFAULT_FALLBACK_CONFIG,
    HEALTH_AWARE_FALLBACK_CONFIG,
    LOAD_BALANCED_CONFIG,
    RETRY_FALLBACK_CONFIG,
    WEIGHTED_CONFIG,
    AllEnginesFailedError,
    AsyncEngineChain,
    ChainExecutionAttempt,
    ChainExecutionMode,
    ChainExecutionResult,
    CompositeChainHook,
    ConditionalEngineChain,
    EngineChain,
    EngineChainConfigError,
    FailureReason,
    FallbackConfig,
    FallbackStrategy,
    LoggingChainHook,
    MetricsChainHook,
    NoEngineSelectedError,
    PrioritySelector,
    RoundRobinEngineIterator,
    SelectorEngineChain,
    SequentialEngineIterator,
    WeightedRandomSelector,
    create_async_fallback_chain,
    create_fallback_chain,
    create_load_balanced_chain,
)


# =============================================================================
# Test Fixtures and Mock Engines
# =============================================================================


class MockEngine(EngineInfoMixin):
    """Mock engine for testing."""

    def __init__(
        self,
        name: str = "mock_engine",
        should_fail: bool = False,
        fail_times: int = 0,
    ) -> None:
        """Initialize mock engine.

        Args:
            name: Engine name.
            should_fail: Whether engine should always fail.
            fail_times: Number of times to fail before succeeding.
        """
        self._name = name
        self._should_fail = should_fail
        self._fail_times = fail_times
        self._call_count = 0

    @property
    def engine_name(self) -> str:
        return self._name

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def _get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
            supported_data_types=("polars", "pandas"),
            supported_rule_types=("not_null", "unique"),
        )

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        self._call_count += 1
        if self._should_fail or self._call_count <= self._fail_times:
            raise RuntimeError(f"Engine {self._name} failed")
        return CheckResult(
            status=CheckStatus.PASSED,
            passed_count=len(rules),
            failed_count=0,
            metadata={"engine": self._name},
        )

    def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        self._call_count += 1
        if self._should_fail or self._call_count <= self._fail_times:
            raise RuntimeError(f"Engine {self._name} failed")
        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            row_count=100,
            column_count=10,
            columns=(),
            metadata={"engine": self._name},
        )

    def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        self._call_count += 1
        if self._should_fail or self._call_count <= self._fail_times:
            raise RuntimeError(f"Engine {self._name} failed")
        return LearnResult(
            status=LearnStatus.COMPLETED,
            rules=(),
            metadata={"engine": self._name},
        )


class AsyncMockEngine:
    """Async mock engine for testing."""

    def __init__(
        self,
        name: str = "async_mock_engine",
        should_fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        self._name = name
        self._should_fail = should_fail
        self._delay = delay
        self._call_count = 0

    @property
    def engine_name(self) -> str:
        return self._name

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    async def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        self._call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise RuntimeError(f"Async engine {self._name} failed")
        return CheckResult(
            status=CheckStatus.PASSED,
            passed_count=len(rules),
            metadata={"engine": self._name},
        )

    async def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        self._call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise RuntimeError(f"Async engine {self._name} failed")
        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            row_count=100,
            column_count=10,
            columns=(),
            metadata={"engine": self._name},
        )

    async def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        self._call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._should_fail:
            raise RuntimeError(f"Async engine {self._name} failed")
        return LearnResult(
            status=LearnStatus.COMPLETED,
            rules=(),
            metadata={"engine": self._name},
        )


@pytest.fixture
def primary_engine() -> MockEngine:
    """Create primary engine."""
    return MockEngine(name="primary")


@pytest.fixture
def backup_engine() -> MockEngine:
    """Create backup engine."""
    return MockEngine(name="backup")


@pytest.fixture
def failing_engine() -> MockEngine:
    """Create engine that always fails."""
    return MockEngine(name="failing", should_fail=True)


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Sample data for testing."""
    return {"id": [1, 2, 3], "name": ["a", "b", "c"]}


@pytest.fixture
def sample_rules() -> list[dict[str, Any]]:
    """Sample rules for testing."""
    return [{"type": "not_null", "column": "id"}]


# =============================================================================
# Test FallbackConfig
# =============================================================================


class TestFallbackConfig:
    """Tests for FallbackConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = FallbackConfig()
        assert config.strategy == FallbackStrategy.SEQUENTIAL
        assert config.mode == ChainExecutionMode.FALLBACK
        assert config.retry_count == 1
        assert config.retry_delay_seconds == 0.0
        assert config.timeout_seconds is None
        assert config.check_health is False

    def test_with_strategy(self) -> None:
        """Test with_strategy builder method."""
        config = FallbackConfig()
        new_config = config.with_strategy(FallbackStrategy.ROUND_ROBIN)
        assert new_config.strategy == FallbackStrategy.ROUND_ROBIN
        assert config.strategy == FallbackStrategy.SEQUENTIAL  # Original unchanged

    def test_with_retry(self) -> None:
        """Test with_retry builder method."""
        config = FallbackConfig()
        new_config = config.with_retry(count=3, delay_seconds=1.0)
        assert new_config.retry_count == 3
        assert new_config.retry_delay_seconds == 1.0

    def test_with_health_check(self) -> None:
        """Test with_health_check builder method."""
        config = FallbackConfig()
        new_config = config.with_health_check(enabled=True, skip_unhealthy=True)
        assert new_config.check_health is True
        assert new_config.skip_unhealthy is True

    def test_with_weights(self) -> None:
        """Test with_weights builder method."""
        config = FallbackConfig()
        new_config = config.with_weights(engine1=2.0, engine2=1.0)
        assert new_config.weights == {"engine1": 2.0, "engine2": 1.0}

    def test_validation_retry_count(self) -> None:
        """Test validation of retry_count."""
        with pytest.raises(ValueError, match="retry_count must be non-negative"):
            FallbackConfig(retry_count=-1)

    def test_validation_retry_delay(self) -> None:
        """Test validation of retry_delay_seconds."""
        with pytest.raises(ValueError, match="retry_delay_seconds must be non-negative"):
            FallbackConfig(retry_delay_seconds=-1.0)

    def test_validation_timeout(self) -> None:
        """Test validation of timeout_seconds."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            FallbackConfig(timeout_seconds=0)

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        config = FallbackConfig(retry_count=3, check_health=True)
        d = config.to_dict()
        assert d["retry_count"] == 3
        assert d["check_health"] is True
        assert d["strategy"] == "SEQUENTIAL"

    def test_preset_configs(self) -> None:
        """Test preset configurations exist and are valid."""
        assert DEFAULT_FALLBACK_CONFIG.retry_count == 1
        assert RETRY_FALLBACK_CONFIG.retry_count == 3
        assert HEALTH_AWARE_FALLBACK_CONFIG.check_health is True
        assert LOAD_BALANCED_CONFIG.strategy == FallbackStrategy.ROUND_ROBIN
        assert WEIGHTED_CONFIG.strategy == FallbackStrategy.WEIGHTED


# =============================================================================
# Test EngineChain
# =============================================================================


class TestEngineChain:
    """Tests for EngineChain."""

    def test_create_chain(
        self, primary_engine: MockEngine, backup_engine: MockEngine
    ) -> None:
        """Test basic chain creation."""
        chain = EngineChain([primary_engine, backup_engine])
        assert chain.engine_name == "engine_chain"
        assert len(chain.engines) == 2

    def test_empty_engines_raises_error(self) -> None:
        """Test that empty engines list raises error."""
        with pytest.raises(EngineChainConfigError, match="at least one engine"):
            EngineChain([])

    def test_check_success(
        self,
        primary_engine: MockEngine,
        backup_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test successful check with primary engine."""
        chain = EngineChain([primary_engine, backup_engine])
        result = chain.check(sample_data, sample_rules)

        assert result.status == CheckStatus.PASSED
        assert result.metadata["engine"] == "primary"
        assert primary_engine._call_count == 1
        assert backup_engine._call_count == 0

    def test_fallback_on_failure(
        self,
        failing_engine: MockEngine,
        backup_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test fallback to backup when primary fails."""
        chain = EngineChain([failing_engine, backup_engine])
        result = chain.check(sample_data, sample_rules)

        assert result.status == CheckStatus.PASSED
        assert result.metadata["engine"] == "backup"

    def test_all_engines_fail(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test error when all engines fail."""
        engine1 = MockEngine(name="engine1", should_fail=True)
        engine2 = MockEngine(name="engine2", should_fail=True)
        chain = EngineChain([engine1, engine2])

        with pytest.raises(AllEnginesFailedError) as exc_info:
            chain.check(sample_data, sample_rules)

        assert "engine1" in exc_info.value.attempted_engines
        assert "engine2" in exc_info.value.attempted_engines

    def test_profile(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
    ) -> None:
        """Test profile operation."""
        chain = EngineChain([primary_engine])
        result = chain.profile(sample_data)

        assert result.row_count == 100
        assert result.metadata["engine"] == "primary"

    def test_learn(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
    ) -> None:
        """Test learn operation."""
        chain = EngineChain([primary_engine])
        result = chain.learn(sample_data)

        assert result.metadata["engine"] == "primary"

    def test_last_execution_result(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test that last execution result is tracked."""
        chain = EngineChain([primary_engine])
        chain.check(sample_data, sample_rules)

        exec_result = chain.last_execution_result
        assert exec_result is not None
        assert exec_result.success is True
        assert exec_result.final_engine == "primary"
        assert exec_result.attempt_count == 1

    def test_retry_logic(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test retry before fallback."""
        engine = MockEngine(name="flaky", fail_times=2)
        config = FallbackConfig(retry_count=3)
        chain = EngineChain([engine], config=config)

        result = chain.check(sample_data, sample_rules)
        assert result.status == CheckStatus.PASSED
        assert engine._call_count == 3

    def test_fail_fast_mode(
        self,
        failing_engine: MockEngine,
        backup_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test FAIL_FAST mode stops on first failure."""
        config = FallbackConfig(mode=ChainExecutionMode.FAIL_FAST)
        chain = EngineChain([failing_engine, backup_engine], config=config)

        with pytest.raises(AllEnginesFailedError):
            chain.check(sample_data, sample_rules)

        assert backup_engine._call_count == 0

    def test_add_engine(
        self,
        primary_engine: MockEngine,
        backup_engine: MockEngine,
    ) -> None:
        """Test adding engine to chain."""
        chain = EngineChain([primary_engine])
        assert len(chain.engines) == 1

        chain.add_engine(backup_engine)
        assert len(chain.engines) == 2

    def test_remove_engine(
        self,
        primary_engine: MockEngine,
        backup_engine: MockEngine,
    ) -> None:
        """Test removing engine from chain."""
        chain = EngineChain([primary_engine, backup_engine])

        removed = chain.remove_engine("backup")
        assert removed is True
        assert len(chain.engines) == 1

        removed = chain.remove_engine("nonexistent")
        assert removed is False

    def test_context_manager(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test context manager usage."""
        with EngineChain([primary_engine]) as chain:
            result = chain.check(sample_data, sample_rules)
            assert result.status == CheckStatus.PASSED


# =============================================================================
# Test Engine Iterators
# =============================================================================


class TestEngineIterators:
    """Tests for engine iterators and selectors."""

    def test_sequential_iterator(self) -> None:
        """Test SequentialEngineIterator."""
        engines = [MockEngine(name=f"engine{i}") for i in range(3)]
        iterator = SequentialEngineIterator(engines)

        assert iterator.next().engine_name == "engine0"
        assert iterator.next().engine_name == "engine1"
        assert iterator.next().engine_name == "engine2"
        assert iterator.next() is None

        iterator.reset()
        assert iterator.next().engine_name == "engine0"

    def test_round_robin_iterator(self) -> None:
        """Test RoundRobinEngineIterator."""
        engines = [MockEngine(name=f"engine{i}") for i in range(3)]
        iterator = RoundRobinEngineIterator(engines)

        assert iterator.next().engine_name == "engine0"
        assert iterator.next().engine_name == "engine1"
        assert iterator.next().engine_name == "engine2"
        assert iterator.next().engine_name == "engine0"  # Wraps around

    def test_weighted_random_selector(self) -> None:
        """Test WeightedRandomSelector."""
        engines = [MockEngine(name=f"engine{i}") for i in range(3)]
        weights = {"engine0": 10.0, "engine1": 0.0, "engine2": 0.0}
        selector = WeightedRandomSelector(engines, weights)

        # With engine0 having all the weight, it should always be selected
        for _ in range(10):
            assert selector.select().engine_name == "engine0"

    def test_priority_selector(self) -> None:
        """Test PrioritySelector."""
        engines = [
            MockEngine(name="low"),
            MockEngine(name="high"),
            MockEngine(name="medium"),
        ]
        priorities = {"low": 1, "medium": 5, "high": 10}
        selector = PrioritySelector(engines, priorities)

        assert selector.next().engine_name == "high"
        assert selector.next().engine_name == "medium"
        assert selector.next().engine_name == "low"
        assert selector.next() is None


# =============================================================================
# Test ConditionalEngineChain
# =============================================================================


class TestConditionalEngineChain:
    """Tests for ConditionalEngineChain."""

    def test_create_conditional_chain(self) -> None:
        """Test basic creation."""
        chain = ConditionalEngineChain(name="test_conditional")
        assert chain.engine_name == "test_conditional"

    def test_condition_routing(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test routing based on conditions."""
        small_engine = MockEngine(name="small")
        large_engine = MockEngine(name="large")

        chain = ConditionalEngineChain()
        chain.add_route(
            lambda data, rules: len(rules) > 5,
            large_engine,
            name="large_rules",
        )
        chain.add_route(
            lambda data, rules: True,  # Default
            small_engine,
            name="default",
        )

        # With 1 rule, should use small_engine
        result = chain.check(sample_data, sample_rules)
        assert result.metadata["engine"] == "small"

        # With many rules, should use large_engine
        many_rules = sample_rules * 10
        result = chain.check(sample_data, many_rules)
        assert result.metadata["engine"] == "large"

    def test_priority_routing(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test that higher priority routes are evaluated first."""
        high_priority_engine = MockEngine(name="high")
        low_priority_engine = MockEngine(name="low")

        chain = ConditionalEngineChain()
        chain.add_route(
            lambda data, rules: True,
            low_priority_engine,
            priority=1,
            name="low",
        )
        chain.add_route(
            lambda data, rules: True,
            high_priority_engine,
            priority=10,
            name="high",
        )

        result = chain.check(sample_data, sample_rules)
        assert result.metadata["engine"] == "high"

    def test_no_matching_route_without_default(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test error when no route matches and no default."""
        chain = ConditionalEngineChain()
        chain.add_route(
            lambda data, rules: False,  # Never matches
            MockEngine(name="never"),
        )

        with pytest.raises(NoEngineSelectedError):
            chain.check(sample_data, sample_rules)

    def test_default_engine(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test default engine is used when no condition matches."""
        default_engine = MockEngine(name="default")
        chain = ConditionalEngineChain()
        chain.add_route(
            lambda data, rules: False,
            MockEngine(name="never"),
        )
        chain.set_default_engine(default_engine)

        result = chain.check(sample_data, sample_rules)
        assert result.metadata["engine"] == "default"

    def test_remove_route(self) -> None:
        """Test removing a route by name."""
        chain = ConditionalEngineChain()
        chain.add_route(
            lambda data, rules: True,
            MockEngine(name="test"),
            name="test_route",
        )
        assert len(chain.routes) == 1

        removed = chain.remove_route("test_route")
        assert removed is True
        assert len(chain.routes) == 0

        removed = chain.remove_route("nonexistent")
        assert removed is False


# =============================================================================
# Test SelectorEngineChain
# =============================================================================


class TestSelectorEngineChain:
    """Tests for SelectorEngineChain."""

    def test_custom_selector(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test with custom selector."""

        class AlwaysFirstSelector:
            def select_engine(
                self,
                data: Any,
                rules: Sequence[Mapping[str, Any]],
                available_engines: Sequence[DataQualityEngine],
                context: dict[str, Any],
            ) -> DataQualityEngine | None:
                return available_engines[0] if available_engines else None

        engines = [
            MockEngine(name="first"),
            MockEngine(name="second"),
        ]
        chain = SelectorEngineChain(
            engines=engines,
            selector=AlwaysFirstSelector(),
        )

        result = chain.check(sample_data, sample_rules)
        assert result.metadata["engine"] == "first"

    def test_selector_returns_none(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test when selector returns None without fallback."""

        class NoneSelector:
            def select_engine(
                self,
                data: Any,
                rules: Sequence[Mapping[str, Any]],
                available_engines: Sequence[DataQualityEngine],
                context: dict[str, Any],
            ) -> DataQualityEngine | None:
                return None

        chain = SelectorEngineChain(
            engines=[MockEngine()],
            selector=NoneSelector(),
        )

        with pytest.raises(NoEngineSelectedError):
            chain.check(sample_data, sample_rules)


# =============================================================================
# Test Chain Hooks
# =============================================================================


class TestChainHooks:
    """Tests for chain execution hooks."""

    def test_metrics_chain_hook(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test MetricsChainHook collects metrics."""
        hook = MetricsChainHook()
        chain = EngineChain([primary_engine], hooks=[hook], name="test_chain")

        chain.check(sample_data, sample_rules)

        assert hook.get_chain_success_rate("test_chain") == 1.0
        assert hook.get_fallback_rate("test_chain") == 0.0
        assert hook.get_average_duration_ms("test_chain") > 0

        stats = hook.get_engine_stats("test_chain")
        assert "primary" in stats["attempts"]

    def test_metrics_hook_with_fallback(
        self,
        failing_engine: MockEngine,
        backup_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test MetricsChainHook tracks fallbacks."""
        hook = MetricsChainHook()
        chain = EngineChain(
            [failing_engine, backup_engine],
            hooks=[hook],
            name="fallback_chain",
        )

        chain.check(sample_data, sample_rules)

        assert hook.get_chain_success_rate("fallback_chain") == 1.0
        assert hook.get_fallback_rate("fallback_chain") == 1.0

    def test_composite_chain_hook(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test CompositeChainHook combines hooks."""
        hook1 = MetricsChainHook()
        hook2 = MetricsChainHook()
        composite = CompositeChainHook([hook1, hook2])

        chain = EngineChain([primary_engine], hooks=[composite], name="composite_test")
        chain.check(sample_data, sample_rules)

        # Both hooks should have recorded the execution
        assert hook1.get_chain_success_rate("composite_test") == 1.0
        assert hook2.get_chain_success_rate("composite_test") == 1.0

    def test_logging_chain_hook(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test LoggingChainHook doesn't raise exceptions."""
        hook = LoggingChainHook()
        chain = EngineChain([primary_engine], hooks=[hook])

        # Should not raise even if logger isn't configured
        result = chain.check(sample_data, sample_rules)
        assert result.status == CheckStatus.PASSED

    def test_hook_exception_isolation(
        self,
        primary_engine: MockEngine,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test that hook exceptions don't affect chain execution."""

        class BrokenHook:
            def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Hook error")

            def on_engine_attempt(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Hook error")

            def on_engine_success(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Hook error")

            def on_engine_failure(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Hook error")

            def on_fallback(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Hook error")

            def on_chain_complete(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Hook error")

        composite = CompositeChainHook([BrokenHook()])
        chain = EngineChain([primary_engine], hooks=[composite])

        # Should succeed despite hook failures
        result = chain.check(sample_data, sample_rules)
        assert result.status == CheckStatus.PASSED


# =============================================================================
# Test AsyncEngineChain
# =============================================================================


class TestAsyncEngineChain:
    """Tests for AsyncEngineChain."""

    @pytest.mark.asyncio
    async def test_async_check_success(self) -> None:
        """Test async check with success."""
        engine = AsyncMockEngine(name="async_primary")
        chain = AsyncEngineChain([engine])

        result = await chain.check({}, [{"type": "test"}])
        assert result.status == CheckStatus.PASSED
        assert result.metadata["engine"] == "async_primary"

    @pytest.mark.asyncio
    async def test_async_fallback(self) -> None:
        """Test async fallback on failure."""
        failing = AsyncMockEngine(name="failing", should_fail=True)
        backup = AsyncMockEngine(name="backup")
        chain = AsyncEngineChain([failing, backup])

        result = await chain.check({}, [])
        assert result.metadata["engine"] == "backup"

    @pytest.mark.asyncio
    async def test_async_all_fail(self) -> None:
        """Test async chain when all engines fail."""
        engine1 = AsyncMockEngine(name="fail1", should_fail=True)
        engine2 = AsyncMockEngine(name="fail2", should_fail=True)
        chain = AsyncEngineChain([engine1, engine2])

        with pytest.raises(AllEnginesFailedError):
            await chain.check({}, [])

    @pytest.mark.asyncio
    async def test_async_profile(self) -> None:
        """Test async profile operation."""
        engine = AsyncMockEngine(name="async_engine")
        chain = AsyncEngineChain([engine])

        result = await chain.profile({})
        assert result.row_count == 100

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager."""
        engine = AsyncMockEngine(name="async_engine")
        chain = AsyncEngineChain([engine])

        async with chain:
            result = await chain.check({}, [])
            assert result.status == CheckStatus.PASSED


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_fallback_chain(
        self,
        primary_engine: MockEngine,
        backup_engine: MockEngine,
    ) -> None:
        """Test create_fallback_chain factory."""
        chain = create_fallback_chain(
            primary_engine,
            backup_engine,
            retry_count=2,
            name="fallback_test",
        )

        assert chain.engine_name == "fallback_test"
        assert len(chain.engines) == 2
        assert chain.config.retry_count == 2
        assert chain.config.strategy == FallbackStrategy.SEQUENTIAL

    def test_create_load_balanced_chain(
        self,
        primary_engine: MockEngine,
        backup_engine: MockEngine,
    ) -> None:
        """Test create_load_balanced_chain factory."""
        chain = create_load_balanced_chain(
            primary_engine,
            backup_engine,
            strategy=FallbackStrategy.ROUND_ROBIN,
            name="lb_test",
        )

        assert chain.engine_name == "lb_test"
        assert chain.config.strategy == FallbackStrategy.ROUND_ROBIN
        assert chain.config.check_health is True

    def test_create_async_fallback_chain(self) -> None:
        """Test create_async_fallback_chain factory."""
        engine1 = AsyncMockEngine(name="async1")
        engine2 = AsyncMockEngine(name="async2")

        chain = create_async_fallback_chain(
            engine1,
            engine2,
            retry_count=3,
            name="async_fallback",
        )

        assert chain.engine_name == "async_fallback"
        assert len(chain.engines) == 2


# =============================================================================
# Test ChainExecutionResult
# =============================================================================


class TestChainExecutionResult:
    """Tests for ChainExecutionResult."""

    def test_result_properties(self) -> None:
        """Test result properties."""
        attempts = (
            ChainExecutionAttempt(
                engine_name="engine1",
                success=False,
                failure_reason=FailureReason.EXCEPTION,
            ),
            ChainExecutionAttempt(
                engine_name="engine2",
                success=True,
                duration_ms=50.0,
            ),
        )
        result = ChainExecutionResult(
            chain_name="test",
            success=True,
            final_engine="engine2",
            attempts=attempts,
            total_duration_ms=100.0,
        )

        assert result.attempt_count == 2
        assert result.failed_engines == ("engine1",)

    def test_result_to_dict(self) -> None:
        """Test serialization to dict."""
        result = ChainExecutionResult(
            chain_name="test",
            success=True,
            final_engine="engine1",
            attempts=(),
            total_duration_ms=50.0,
        )

        d = result.to_dict()
        assert d["chain_name"] == "test"
        assert d["success"] is True
        assert d["final_engine"] == "engine1"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for chain functionality."""

    def test_chain_as_engine(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test that chains can be used as engines (composability)."""
        # Create inner chain
        inner_chain = EngineChain(
            [MockEngine(name="inner1"), MockEngine(name="inner2")],
            name="inner",
        )

        # Create outer chain containing inner chain
        outer_chain = EngineChain(
            [inner_chain, MockEngine(name="outer_backup")],
            name="outer",
        )

        result = outer_chain.check(sample_data, sample_rules)
        assert result.status == CheckStatus.PASSED

    def test_conditional_with_fallback(
        self,
        sample_data: dict[str, Any],
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test conditional chain with fallback chain as route."""
        primary = MockEngine(name="primary", should_fail=True)
        backup = MockEngine(name="backup")
        fallback_chain = EngineChain([primary, backup], name="fallback")

        conditional = ConditionalEngineChain()
        conditional.add_route(
            lambda data, rules: True,
            fallback_chain,
        )

        result = conditional.check(sample_data, sample_rules)
        assert result.status == CheckStatus.PASSED

    def test_complete_workflow(
        self,
        sample_data: dict[str, Any],
    ) -> None:
        """Test complete workflow with metrics and logging."""
        # Setup engines
        engine1 = MockEngine(name="engine1", fail_times=1)
        engine2 = MockEngine(name="engine2")

        # Setup hooks
        metrics_hook = MetricsChainHook()
        logging_hook = LoggingChainHook()

        # Create chain with configuration
        config = FallbackConfig(
            strategy=FallbackStrategy.SEQUENTIAL,
            retry_count=2,
        )
        chain = EngineChain(
            [engine1, engine2],
            config=config,
            hooks=[metrics_hook, logging_hook],
            name="workflow_chain",
        )

        # Execute operations
        rules = [{"type": "not_null", "column": "id"}]
        check_result = chain.check(sample_data, rules)
        profile_result = chain.profile(sample_data)

        # Verify results
        assert check_result.status == CheckStatus.PASSED
        assert profile_result.row_count == 100

        # Verify metrics
        assert metrics_hook.get_chain_success_rate("workflow_chain") == 1.0
