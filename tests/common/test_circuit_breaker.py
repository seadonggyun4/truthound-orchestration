"""Tests for common.circuit_breaker module."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from common.circuit_breaker import (
    AGGRESSIVE_CIRCUIT_BREAKER_CONFIG,
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
    RESILIENT_CIRCUIT_BREAKER_CONFIG,
    SENSITIVE_CIRCUIT_BREAKER_CONFIG,
    CallableFailureDetector,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    CompositeCircuitBreakerHook,
    CompositeFailureDetector,
    LoggingCircuitBreakerHook,
    MetricsCircuitBreakerHook,
    TypeBasedFailureDetector,
    circuit_breaker,
    circuit_breaker_call,
    circuit_breaker_call_async,
    get_circuit_breaker,
    get_registry,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert CircuitState.CLOSED
        assert CircuitState.OPEN
        assert CircuitState.HALF_OPEN


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 1
        assert config.recovery_timeout_seconds == 30.0
        assert config.half_open_max_calls == 1
        assert config.exceptions == (Exception,)
        assert config.ignored_exceptions == ()
        assert config.name is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            recovery_timeout_seconds=60.0,
            half_open_max_calls=5,
            exceptions=(ValueError, ConnectionError),
            ignored_exceptions=(KeyError,),
            name="test_circuit",
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.recovery_timeout_seconds == 60.0
        assert config.half_open_max_calls == 5
        assert config.exceptions == (ValueError, ConnectionError)
        assert config.ignored_exceptions == (KeyError,)
        assert config.name == "test_circuit"

    def test_validation_failure_threshold(self):
        """Test validation of failure_threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            CircuitBreakerConfig(failure_threshold=0)

    def test_validation_success_threshold(self):
        """Test validation of success_threshold."""
        with pytest.raises(ValueError, match="success_threshold must be at least 1"):
            CircuitBreakerConfig(success_threshold=0)

    def test_validation_recovery_timeout(self):
        """Test validation of recovery_timeout_seconds."""
        with pytest.raises(
            ValueError, match="recovery_timeout_seconds must be non-negative"
        ):
            CircuitBreakerConfig(recovery_timeout_seconds=-1.0)

    def test_validation_half_open_max_calls(self):
        """Test validation of half_open_max_calls."""
        with pytest.raises(ValueError, match="half_open_max_calls must be at least 1"):
            CircuitBreakerConfig(half_open_max_calls=0)

    def test_with_failure_threshold(self):
        """Test with_failure_threshold builder method."""
        config = CircuitBreakerConfig(failure_threshold=5)
        new_config = config.with_failure_threshold(10)
        assert new_config.failure_threshold == 10
        assert config.failure_threshold == 5  # Original unchanged

    def test_with_success_threshold(self):
        """Test with_success_threshold builder method."""
        config = CircuitBreakerConfig()
        new_config = config.with_success_threshold(3)
        assert new_config.success_threshold == 3

    def test_with_recovery_timeout(self):
        """Test with_recovery_timeout builder method."""
        config = CircuitBreakerConfig()
        new_config = config.with_recovery_timeout(60.0)
        assert new_config.recovery_timeout_seconds == 60.0

    def test_with_exceptions(self):
        """Test with_exceptions builder method."""
        config = CircuitBreakerConfig()
        new_config = config.with_exceptions(
            exceptions=(ValueError,),
            ignored=(KeyError,),
        )
        assert new_config.exceptions == (ValueError,)
        assert new_config.ignored_exceptions == (KeyError,)

    def test_with_name(self):
        """Test with_name builder method."""
        config = CircuitBreakerConfig()
        new_config = config.with_name("my_circuit")
        assert new_config.name == "my_circuit"

    def test_to_dict(self):
        """Test to_dict method."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            exceptions=(ValueError,),
            name="test",
        )
        result = config.to_dict()
        assert result["failure_threshold"] == 10
        assert "ValueError" in result["exceptions"]
        assert result["name"] == "test"

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "failure_threshold": 10,
            "success_threshold": 3,
            "recovery_timeout_seconds": 60.0,
            "name": "test",
        }
        config = CircuitBreakerConfig.from_dict(data)
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.recovery_timeout_seconds == 60.0
        assert config.name == "test"

    def test_immutability(self):
        """Test that config is immutable."""
        config = CircuitBreakerConfig()
        with pytest.raises(AttributeError):
            config.failure_threshold = 10  # type: ignore


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_config(self):
        """Test DEFAULT_CIRCUIT_BREAKER_CONFIG."""
        assert DEFAULT_CIRCUIT_BREAKER_CONFIG.failure_threshold == 5
        assert DEFAULT_CIRCUIT_BREAKER_CONFIG.success_threshold == 1
        assert DEFAULT_CIRCUIT_BREAKER_CONFIG.recovery_timeout_seconds == 30.0

    def test_sensitive_config(self):
        """Test SENSITIVE_CIRCUIT_BREAKER_CONFIG."""
        assert SENSITIVE_CIRCUIT_BREAKER_CONFIG.failure_threshold == 3
        assert SENSITIVE_CIRCUIT_BREAKER_CONFIG.success_threshold == 2
        assert SENSITIVE_CIRCUIT_BREAKER_CONFIG.recovery_timeout_seconds == 60.0

    def test_resilient_config(self):
        """Test RESILIENT_CIRCUIT_BREAKER_CONFIG."""
        assert RESILIENT_CIRCUIT_BREAKER_CONFIG.failure_threshold == 10
        assert RESILIENT_CIRCUIT_BREAKER_CONFIG.success_threshold == 1
        assert RESILIENT_CIRCUIT_BREAKER_CONFIG.recovery_timeout_seconds == 15.0

    def test_aggressive_config(self):
        """Test AGGRESSIVE_CIRCUIT_BREAKER_CONFIG."""
        assert AGGRESSIVE_CIRCUIT_BREAKER_CONFIG.failure_threshold == 2
        assert AGGRESSIVE_CIRCUIT_BREAKER_CONFIG.success_threshold == 3
        assert AGGRESSIVE_CIRCUIT_BREAKER_CONFIG.recovery_timeout_seconds == 120.0


class TestFailureDetectors:
    """Tests for failure detectors."""

    def test_type_based_detector_default(self):
        """Test TypeBasedFailureDetector with default settings."""
        detector = TypeBasedFailureDetector()
        assert detector.is_failure(ValueError("test")) is True
        assert detector.is_failure(KeyError("test")) is True
        assert detector.is_success("result") is True

    def test_type_based_detector_specific_exceptions(self):
        """Test TypeBasedFailureDetector with specific exceptions."""
        detector = TypeBasedFailureDetector(
            failure_exceptions=(ValueError, TypeError),
        )
        assert detector.is_failure(ValueError("test")) is True
        assert detector.is_failure(TypeError("test")) is True
        assert detector.is_failure(KeyError("test")) is False

    def test_type_based_detector_ignored_exceptions(self):
        """Test TypeBasedFailureDetector with ignored exceptions."""
        detector = TypeBasedFailureDetector(
            failure_exceptions=(Exception,),
            ignored_exceptions=(KeyError,),
        )
        assert detector.is_failure(ValueError("test")) is True
        assert detector.is_failure(KeyError("test")) is False

    def test_callable_detector_failure(self):
        """Test CallableFailureDetector for failure detection."""

        def is_critical(exc: Exception) -> bool:
            return "critical" in str(exc).lower()

        detector = CallableFailureDetector(failure_predicate=is_critical)
        assert detector.is_failure(ValueError("Critical error")) is True
        assert detector.is_failure(ValueError("Minor issue")) is False

    def test_callable_detector_success(self):
        """Test CallableFailureDetector for success detection."""

        def is_valid(result: object) -> bool:
            return result is not None

        detector = CallableFailureDetector(success_predicate=is_valid)
        assert detector.is_success("valid") is True
        assert detector.is_success(None) is False

    def test_composite_detector_any(self):
        """Test CompositeFailureDetector with any mode."""
        detector1 = TypeBasedFailureDetector(failure_exceptions=(ValueError,))
        detector2 = TypeBasedFailureDetector(failure_exceptions=(TypeError,))
        composite = CompositeFailureDetector([detector1, detector2])

        assert composite.is_failure(ValueError()) is True
        assert composite.is_failure(TypeError()) is True
        assert composite.is_failure(KeyError()) is False

    def test_composite_detector_all(self):
        """Test CompositeFailureDetector with all mode."""
        detector1 = TypeBasedFailureDetector(failure_exceptions=(Exception,))

        def attempt_check(exc: Exception) -> bool:
            return "retry" in str(exc).lower()

        detector2 = CallableFailureDetector(failure_predicate=attempt_check)
        composite = CompositeFailureDetector(
            [detector1, detector2],
            require_all_for_failure=True,
        )

        assert composite.is_failure(ValueError("Please retry")) is True
        assert composite.is_failure(ValueError("No retries")) is False


class TestCircuitBreakerHooks:
    """Tests for circuit breaker hooks."""

    def test_metrics_hook_state_changes(self):
        """Test MetricsCircuitBreakerHook tracks state changes."""
        hook = MetricsCircuitBreakerHook()

        hook.on_state_change(CircuitState.CLOSED, CircuitState.OPEN, {})
        hook.on_state_change(CircuitState.OPEN, CircuitState.HALF_OPEN, {})
        hook.on_state_change(CircuitState.HALF_OPEN, CircuitState.CLOSED, {})

        assert len(hook.state_changes) == 3
        assert hook.times_opened == 1

    def test_metrics_hook_success_failure(self):
        """Test MetricsCircuitBreakerHook tracks success/failure."""
        hook = MetricsCircuitBreakerHook()

        hook.on_success(CircuitState.CLOSED, {})
        hook.on_success(CircuitState.CLOSED, {})
        hook.on_failure(ValueError("test"), CircuitState.CLOSED, 1, {})

        assert hook.success_count == 2
        assert hook.failure_count == 1

    def test_metrics_hook_rejected(self):
        """Test MetricsCircuitBreakerHook tracks rejected requests."""
        hook = MetricsCircuitBreakerHook()

        hook.on_rejected(CircuitState.OPEN, {})
        hook.on_rejected(CircuitState.OPEN, {})

        assert hook.rejected_count == 2

    def test_metrics_hook_reset(self):
        """Test MetricsCircuitBreakerHook reset."""
        hook = MetricsCircuitBreakerHook()
        hook.on_success(CircuitState.CLOSED, {})
        hook.on_failure(ValueError("test"), CircuitState.CLOSED, 1, {})
        hook.on_rejected(CircuitState.OPEN, {})
        hook.on_state_change(CircuitState.CLOSED, CircuitState.OPEN, {})

        hook.reset()

        assert hook.success_count == 0
        assert hook.failure_count == 0
        assert hook.rejected_count == 0
        assert len(hook.state_changes) == 0

    def test_composite_hook(self):
        """Test CompositeCircuitBreakerHook calls all hooks."""
        hook1 = MagicMock()
        hook2 = MagicMock()
        composite = CompositeCircuitBreakerHook([hook1, hook2])

        composite.on_state_change(CircuitState.CLOSED, CircuitState.OPEN, {})
        hook1.on_state_change.assert_called_once()
        hook2.on_state_change.assert_called_once()

        composite.on_success(CircuitState.CLOSED, {})
        hook1.on_success.assert_called_once()
        hook2.on_success.assert_called_once()

        composite.on_failure(ValueError("test"), CircuitState.CLOSED, 1, {})
        hook1.on_failure.assert_called_once()
        hook2.on_failure.assert_called_once()

        composite.on_rejected(CircuitState.OPEN, {})
        hook1.on_rejected.assert_called_once()
        hook2.on_rejected.assert_called_once()

    def test_composite_hook_add_remove(self):
        """Test CompositeCircuitBreakerHook add/remove."""
        hook1 = MagicMock()
        hook2 = MagicMock()
        composite = CompositeCircuitBreakerHook([hook1])

        composite.add_hook(hook2)
        composite.on_state_change(CircuitState.CLOSED, CircuitState.OPEN, {})
        assert hook2.on_state_change.called

        composite.remove_hook(hook2)
        hook2.reset_mock()
        composite.on_state_change(CircuitState.OPEN, CircuitState.HALF_OPEN, {})
        assert not hook2.on_state_change.called

    def test_composite_hook_exception_resilience(self):
        """Test CompositeCircuitBreakerHook handles hook exceptions."""
        failing_hook = MagicMock()
        failing_hook.on_state_change.side_effect = RuntimeError("Hook failed")
        working_hook = MagicMock()
        composite = CompositeCircuitBreakerHook([failing_hook, working_hook])

        composite.on_state_change(CircuitState.CLOSED, CircuitState.OPEN, {})
        assert working_hook.on_state_change.called


class TestCircuitBreakerExceptions:
    """Tests for circuit breaker exceptions."""

    def test_circuit_breaker_error(self):
        """Test CircuitBreakerError."""
        exc = CircuitBreakerError(
            "Test error",
            state=CircuitState.OPEN,
            failure_count=5,
        )
        assert exc.state == CircuitState.OPEN
        assert exc.failure_count == 5
        assert "failure_count" in exc.details
        assert "state" in exc.details

    def test_circuit_open_error(self):
        """Test CircuitOpenError."""
        exc = CircuitOpenError(
            remaining_seconds=10.0,
            failure_count=5,
        )
        assert exc.state == CircuitState.OPEN
        assert exc.remaining_seconds == 10.0
        assert exc.failure_count == 5
        assert "remaining_seconds" in exc.details


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test circuit starts in closed state."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open
        assert cb.failure_count == 0

    def test_success_in_closed_state(self):
        """Test successful call in closed state."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        def success_func() -> str:
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb.is_closed
        assert cb.failure_count == 0

    def test_failure_increments_count(self):
        """Test failures increment failure count."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        def failing_func() -> None:
            raise ValueError("error")

        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.failure_count == 3
        assert cb.is_closed  # Not yet at threshold

    def test_opens_at_threshold(self):
        """Test circuit opens when failure threshold is reached."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

        def failing_func() -> None:
            raise ValueError("error")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.is_open
        assert cb.failure_count == 3

    def test_rejects_when_open(self):
        """Test circuit rejects calls when open."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=60.0,
            )
        )

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.is_open

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: "success")

        assert exc_info.value.remaining_seconds > 0

    def test_transitions_to_half_open(self):
        """Test circuit transitions to half-open after recovery timeout."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.05,
            )
        )

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.is_open

        time.sleep(0.06)
        assert cb.is_half_open

    def test_closes_after_success_in_half_open(self):
        """Test circuit closes after success in half-open state."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=1,
                recovery_timeout_seconds=0.01,
            )
        )

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        time.sleep(0.02)

        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.is_closed
        assert cb.failure_count == 0

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.01,
            )
        )

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        time.sleep(0.02)
        assert cb.is_half_open

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.is_open

    def test_success_threshold_multiple(self):
        """Test multiple successes needed to close circuit."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=3,
                recovery_timeout_seconds=0.01,
                half_open_max_calls=10,
            )
        )

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        time.sleep(0.02)

        # First two successes - still half-open
        cb.call(lambda: "success")
        assert cb.is_half_open
        cb.call(lambda: "success")
        assert cb.is_half_open

        # Third success - closes circuit
        cb.call(lambda: "success")
        assert cb.is_closed

    def test_half_open_limits_concurrent_calls(self):
        """Test half-open state limits concurrent calls."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.01,
                half_open_max_calls=1,
            )
        )

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        time.sleep(0.02)
        assert cb.is_half_open

        # First call allowed in half-open
        result = cb.call(lambda: "success")
        assert result == "success"

    def test_reset(self):
        """Test manual reset of circuit breaker."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.is_open

        cb.reset()
        assert cb.is_closed
        assert cb.failure_count == 0

    def test_success_resets_failure_count(self):
        """Test success in closed state resets failure count."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        def failing_func() -> None:
            raise ValueError("error")

        # Fail twice
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.failure_count == 2

        # Success resets count
        cb.call(lambda: "success")
        assert cb.failure_count == 0

    def test_ignored_exception_not_counted(self):
        """Test ignored exceptions don't count as failures."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                ignored_exceptions=(KeyError,),
            )
        )

        def raises_keyerror() -> None:
            raise KeyError("ignored")

        # KeyError should not count as failure
        for _ in range(5):
            with pytest.raises(KeyError):
                cb.call(raises_keyerror)

        assert cb.failure_count == 0
        assert cb.is_closed

    def test_hooks_called_on_state_change(self):
        """Test hooks are called on state changes."""
        hook = MetricsCircuitBreakerHook()
        cb = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.01),
            hooks=[hook],
        )

        def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert hook.times_opened == 1
        assert hook.failure_count == 1

    def test_remaining_recovery_time(self):
        """Test remaining_recovery_time calculation."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=1.0,
            )
        )

        assert cb.remaining_recovery_time() == 0.0  # Closed, no timeout

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        remaining = cb.remaining_recovery_time()
        assert remaining > 0.0
        assert remaining <= 1.0


class TestCircuitBreakerAsync:
    """Tests for CircuitBreaker async functionality."""

    def test_async_success(self):
        """Test async successful call."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        async def success_func() -> str:
            return "success"

        result = asyncio.get_event_loop().run_until_complete(
            cb.call_async(success_func)
        )
        assert result == "success"

    def test_async_failure_opens_circuit(self):
        """Test async failure opens circuit."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))

        async def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(cb.call_async(failing_func))

        assert cb.is_open

    def test_async_rejects_when_open(self):
        """Test async rejects when circuit is open."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=60.0,
            )
        )

        async def failing_func() -> None:
            raise ValueError("error")

        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(cb.call_async(failing_func))

        async def success_func() -> str:
            return "success"

        with pytest.raises(CircuitOpenError):
            asyncio.get_event_loop().run_until_complete(cb.call_async(success_func))


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create(self):
        """Test get_or_create returns same instance."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("test_cb")
        cb2 = registry.get_or_create("test_cb")

        assert cb1 is cb2

    def test_get_returns_none_if_not_exists(self):
        """Test get returns None if circuit breaker doesn't exist."""
        registry = CircuitBreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_get_returns_existing(self):
        """Test get returns existing circuit breaker."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("test_cb")

        cb = registry.get("test_cb")
        assert cb is not None

    def test_remove(self):
        """Test remove removes circuit breaker."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("test_cb")

        assert registry.remove("test_cb") is True
        assert registry.get("test_cb") is None
        assert registry.remove("test_cb") is False

    def test_reset_all(self):
        """Test reset_all resets all circuit breakers."""
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create(
            "cb1",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        cb2 = registry.get_or_create(
            "cb2",
            config=CircuitBreakerConfig(failure_threshold=1),
        )

        # Open both circuits
        with pytest.raises(ValueError):
            cb1.call(lambda: (_ for _ in ()).throw(ValueError()))
        with pytest.raises(ValueError):
            cb2.call(lambda: (_ for _ in ()).throw(ValueError()))

        assert cb1.is_open
        assert cb2.is_open

        registry.reset_all()

        assert cb1.is_closed
        assert cb2.is_closed

    def test_get_all_states(self):
        """Test get_all_states returns all states."""
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create(
            "cb1",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        registry.get_or_create("cb2")

        with pytest.raises(ValueError):
            cb1.call(lambda: (_ for _ in ()).throw(ValueError()))

        states = registry.get_all_states()
        assert states["cb1"] == CircuitState.OPEN
        assert states["cb2"] == CircuitState.CLOSED

    def test_names(self):
        """Test names property."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("cb1")
        registry.get_or_create("cb2")
        registry.get_or_create("cb3")

        names = registry.names
        assert len(names) == 3
        assert "cb1" in names
        assert "cb2" in names
        assert "cb3" in names


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker decorator."""

    def test_decorator_basic(self):
        """Test basic decorator usage."""

        @circuit_breaker(
            name="test_decorator_basic",
            failure_threshold=3,
            use_registry=False,
        )
        def success_func() -> str:
            return "success"

        result = success_func()
        assert result == "success"

    def test_decorator_opens_on_failures(self):
        """Test decorator opens circuit on failures."""
        call_count = 0

        @circuit_breaker(
            name="test_decorator_opens",
            failure_threshold=2,
            recovery_timeout_seconds=60.0,
            use_registry=False,
        )
        def failing_func() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        # First two calls fail
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()

        # Third call should be rejected
        with pytest.raises(CircuitOpenError):
            failing_func()

        assert call_count == 2  # Only 2 actual calls made

    def test_decorator_with_config(self):
        """Test decorator with CircuitBreakerConfig."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=60.0,
        )

        @circuit_breaker(config=config, name="test_decorator_config", use_registry=False)
        def success_func() -> str:
            return "success"

        result = success_func()
        assert result == "success"

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @circuit_breaker(name="test_metadata", use_registry=False)
        def documented_func() -> str:
            """This is a docstring."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    def test_decorator_with_arguments(self):
        """Test decorator with function arguments."""

        @circuit_breaker(name="test_args", use_registry=False)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_decorator_async(self):
        """Test decorator with async function."""

        @circuit_breaker(name="test_async_decorator", use_registry=False)
        async def async_func() -> str:
            return "async_success"

        result = asyncio.get_event_loop().run_until_complete(async_func())
        assert result == "async_success"

    def test_decorator_async_opens_on_failures(self):
        """Test async decorator opens circuit on failures."""

        @circuit_breaker(
            name="test_async_opens",
            failure_threshold=2,
            recovery_timeout_seconds=60.0,
            use_registry=False,
        )
        async def failing_func() -> None:
            raise ValueError("error")

        for _ in range(2):
            with pytest.raises(ValueError):
                asyncio.get_event_loop().run_until_complete(failing_func())

        with pytest.raises(CircuitOpenError):
            asyncio.get_event_loop().run_until_complete(failing_func())


class TestCircuitBreakerCallFunctions:
    """Tests for circuit_breaker_call and circuit_breaker_call_async."""

    def test_circuit_breaker_call_basic(self):
        """Test circuit_breaker_call function."""

        def success_func() -> str:
            return "success"

        result = circuit_breaker_call(
            success_func,
            name="test_cb_call_basic",
            config=CircuitBreakerConfig(),
        )
        assert result == "success"

    def test_circuit_breaker_call_with_args(self):
        """Test circuit_breaker_call with function arguments."""

        def add(a: int, b: int) -> int:
            return a + b

        result = circuit_breaker_call(
            add,
            1,
            2,
            name="test_cb_call_args",
        )
        assert result == 3

    def test_circuit_breaker_call_async_basic(self):
        """Test circuit_breaker_call_async function."""

        async def success_func() -> str:
            return "async_success"

        result = asyncio.get_event_loop().run_until_complete(
            circuit_breaker_call_async(
                success_func,
                name="test_cb_call_async_basic",
            )
        )
        assert result == "async_success"


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_circuit_breaker(self):
        """Test get_circuit_breaker function."""
        cb = get_circuit_breaker("global_test_cb")
        assert cb is not None
        assert cb.config.name == "global_test_cb"

    def test_get_registry(self):
        """Test get_registry function."""
        registry = get_registry()
        assert isinstance(registry, CircuitBreakerRegistry)


class TestLoggingCircuitBreakerHook:
    """Tests for LoggingCircuitBreakerHook."""

    def test_logging_hook_on_state_change_open(self):
        """Test LoggingCircuitBreakerHook logs when circuit opens."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingCircuitBreakerHook()
            hook.on_state_change(CircuitState.CLOSED, CircuitState.OPEN, {})

            mock_logger.warning.assert_called_once()

    def test_logging_hook_on_state_change_closed(self):
        """Test LoggingCircuitBreakerHook logs when circuit closes."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingCircuitBreakerHook()
            hook.on_state_change(CircuitState.HALF_OPEN, CircuitState.CLOSED, {})

            mock_logger.info.assert_called_once()

    def test_logging_hook_on_failure(self):
        """Test LoggingCircuitBreakerHook logs failures."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingCircuitBreakerHook()
            hook.on_failure(ValueError("test"), CircuitState.CLOSED, 1, {})

            mock_logger.warning.assert_called_once()

    def test_logging_hook_on_rejected(self):
        """Test LoggingCircuitBreakerHook logs rejected requests."""
        with patch("common.logging.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            hook = LoggingCircuitBreakerHook()
            hook.on_rejected(CircuitState.OPEN, {})

            mock_logger.warning.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_failure_threshold(self):
        """Test with single failure threshold."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))

        assert cb.is_open

    def test_zero_recovery_timeout(self):
        """Test with zero recovery timeout (immediate half-open)."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout_seconds=0.0,
            )
        )

        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))

        # Should immediately transition to half-open
        assert cb.is_half_open

    def test_custom_failure_detector(self):
        """Test with custom failure detector."""

        class CustomDetector:
            def is_failure(self, exc: Exception) -> bool:
                return "critical" in str(exc).lower()

            def is_success(self, result: object) -> bool:
                return True

        cb = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=1),
            failure_detector=CustomDetector(),
        )

        # Non-critical error should not count
        for _ in range(5):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("minor")))

        assert cb.is_closed

        # Critical error should count
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("CRITICAL")))

        assert cb.is_open

    def test_unsuccessful_result_counted_as_failure(self):
        """Test that unsuccessful results are counted as failures."""

        class ResultChecker:
            def is_failure(self, exc: Exception) -> bool:
                return True

            def is_success(self, result: object) -> bool:
                return result is not None

        cb = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=2),
            failure_detector=ResultChecker(),
        )

        # Return None twice - should open circuit
        cb.call(lambda: None)
        cb.call(lambda: None)

        assert cb.is_open
