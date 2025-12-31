"""dbt Lifecycle Hook Protocols and Implementations.

This module provides protocols and implementations for handling dbt
test lifecycle events.

Example:
    >>> from truthound_dbt.hooks import LoggingDbtHook, MetricsDbtHook
    >>>
    >>> hook = LoggingDbtHook()
    >>> hook.on_test_start(TestStartEvent(test_name="not_null"))
    >>> hook.on_test_end(TestEndEvent(test_name="not_null", passed=True))
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable


# =============================================================================
# Exceptions
# =============================================================================


class HookError(Exception):
    """Base exception for hook errors."""

    pass


class HookExecutionError(HookError):
    """Raised when hook execution fails."""

    def __init__(self, hook_name: str, event: str, cause: Exception) -> None:
        self.hook_name = hook_name
        self.event = event
        self.cause = cause
        super().__init__(f"Hook '{hook_name}' failed on '{event}': {cause}")


# =============================================================================
# Event Types
# =============================================================================


class EventType(str, Enum):
    """dbt lifecycle event types."""

    TEST_START = "test_start"
    TEST_END = "test_end"
    PARSE_START = "parse_start"
    PARSE_END = "parse_end"
    CONVERSION_START = "conversion_start"
    CONVERSION_END = "conversion_end"
    GENERATION_START = "generation_start"
    GENERATION_END = "generation_end"


@dataclass(frozen=True, slots=True)
class DbtEvent:
    """Base class for dbt events.

    Attributes:
        event_type: Type of the event.
        timestamp: Event timestamp.
        metadata: Additional event metadata.
    """

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TestStartEvent(DbtEvent):
    """Event fired when a test starts.

    Attributes:
        test_name: Name of the test.
        model: Associated model.
        column: Associated column.
        rule_type: Type of rule being tested.
    """

    test_name: str = ""
    model: str | None = None
    column: str | None = None
    rule_type: str | None = None
    event_type: EventType = field(default=EventType.TEST_START, init=False)


@dataclass(frozen=True, slots=True)
class TestEndEvent(DbtEvent):
    """Event fired when a test ends.

    Attributes:
        test_name: Name of the test.
        passed: Whether the test passed.
        duration_ms: Duration in milliseconds.
        failures: Number of failures.
        message: Result message.
    """

    test_name: str = ""
    passed: bool = True
    duration_ms: float = 0.0
    failures: int = 0
    message: str | None = None
    event_type: EventType = field(default=EventType.TEST_END, init=False)


@dataclass(frozen=True, slots=True)
class ParseStartEvent(DbtEvent):
    """Event fired when manifest parsing starts.

    Attributes:
        manifest_path: Path to manifest file.
    """

    manifest_path: str = ""
    event_type: EventType = field(default=EventType.PARSE_START, init=False)


@dataclass(frozen=True, slots=True)
class ParseEndEvent(DbtEvent):
    """Event fired when manifest parsing ends.

    Attributes:
        manifest_path: Path to manifest file.
        success: Whether parsing succeeded.
        duration_ms: Duration in milliseconds.
        test_count: Number of tests found.
        model_count: Number of models found.
    """

    manifest_path: str = ""
    success: bool = True
    duration_ms: float = 0.0
    test_count: int = 0
    model_count: int = 0
    event_type: EventType = field(default=EventType.PARSE_END, init=False)


@dataclass(frozen=True, slots=True)
class ConversionStartEvent(DbtEvent):
    """Event fired when rule conversion starts.

    Attributes:
        rule_count: Number of rules to convert.
        adapter_name: Name of the adapter.
    """

    rule_count: int = 0
    adapter_name: str = ""
    event_type: EventType = field(default=EventType.CONVERSION_START, init=False)


@dataclass(frozen=True, slots=True)
class ConversionEndEvent(DbtEvent):
    """Event fired when rule conversion ends.

    Attributes:
        success: Whether conversion succeeded.
        duration_ms: Duration in milliseconds.
        converted_count: Number of rules converted.
        error_count: Number of conversion errors.
    """

    success: bool = True
    duration_ms: float = 0.0
    converted_count: int = 0
    error_count: int = 0
    event_type: EventType = field(default=EventType.CONVERSION_END, init=False)


@dataclass(frozen=True, slots=True)
class GenerationStartEvent(DbtEvent):
    """Event fired when SQL/test generation starts.

    Attributes:
        model: Model being generated for.
        generation_type: Type of generation (sql, test, schema).
    """

    model: str = ""
    generation_type: str = ""
    event_type: EventType = field(default=EventType.GENERATION_START, init=False)


@dataclass(frozen=True, slots=True)
class GenerationEndEvent(DbtEvent):
    """Event fired when SQL/test generation ends.

    Attributes:
        model: Model generated for.
        generation_type: Type of generation.
        success: Whether generation succeeded.
        duration_ms: Duration in milliseconds.
    """

    model: str = ""
    generation_type: str = ""
    success: bool = True
    duration_ms: float = 0.0
    event_type: EventType = field(default=EventType.GENERATION_END, init=False)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class DbtHook(Protocol):
    """Protocol for dbt lifecycle hooks.

    Implementations handle various lifecycle events during dbt operations.
    """

    def on_test_start(self, event: TestStartEvent) -> None:
        """Called when a test starts."""
        ...

    def on_test_end(self, event: TestEndEvent) -> None:
        """Called when a test ends."""
        ...

    def on_parse_start(self, event: ParseStartEvent) -> None:
        """Called when manifest parsing starts."""
        ...

    def on_parse_end(self, event: ParseEndEvent) -> None:
        """Called when manifest parsing ends."""
        ...

    def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Called when rule conversion starts."""
        ...

    def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Called when rule conversion ends."""
        ...

    def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Called when generation starts."""
        ...

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Called when generation ends."""
        ...


@runtime_checkable
class AsyncDbtHook(Protocol):
    """Async protocol for dbt lifecycle hooks."""

    async def on_test_start(self, event: TestStartEvent) -> None:
        """Called when a test starts."""
        ...

    async def on_test_end(self, event: TestEndEvent) -> None:
        """Called when a test ends."""
        ...

    async def on_parse_start(self, event: ParseStartEvent) -> None:
        """Called when manifest parsing starts."""
        ...

    async def on_parse_end(self, event: ParseEndEvent) -> None:
        """Called when manifest parsing ends."""
        ...

    async def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Called when rule conversion starts."""
        ...

    async def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Called when rule conversion ends."""
        ...

    async def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Called when generation starts."""
        ...

    async def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Called when generation ends."""
        ...


# =============================================================================
# Base Implementations
# =============================================================================


class BaseDbtHook:
    """Base implementation of DbtHook with no-op methods."""

    def on_test_start(self, event: TestStartEvent) -> None:
        """Called when a test starts."""
        pass

    def on_test_end(self, event: TestEndEvent) -> None:
        """Called when a test ends."""
        pass

    def on_parse_start(self, event: ParseStartEvent) -> None:
        """Called when manifest parsing starts."""
        pass

    def on_parse_end(self, event: ParseEndEvent) -> None:
        """Called when manifest parsing ends."""
        pass

    def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Called when rule conversion starts."""
        pass

    def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Called when rule conversion ends."""
        pass

    def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Called when generation starts."""
        pass

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Called when generation ends."""
        pass


class AsyncBaseDbtHook:
    """Base implementation of AsyncDbtHook with no-op methods."""

    async def on_test_start(self, event: TestStartEvent) -> None:
        """Called when a test starts."""
        pass

    async def on_test_end(self, event: TestEndEvent) -> None:
        """Called when a test ends."""
        pass

    async def on_parse_start(self, event: ParseStartEvent) -> None:
        """Called when manifest parsing starts."""
        pass

    async def on_parse_end(self, event: ParseEndEvent) -> None:
        """Called when manifest parsing ends."""
        pass

    async def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Called when rule conversion starts."""
        pass

    async def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Called when rule conversion ends."""
        pass

    async def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Called when generation starts."""
        pass

    async def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Called when generation ends."""
        pass


# =============================================================================
# Logging Hook
# =============================================================================


class LoggingDbtHook(BaseDbtHook):
    """Hook that logs all lifecycle events.

    Example:
        >>> hook = LoggingDbtHook()
        >>> hook.on_test_start(TestStartEvent(test_name="not_null"))
        INFO:truthound_dbt.hooks:Test started: not_null
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Initialize logging hook.

        Args:
            logger: Logger to use (defaults to module logger).
            level: Log level for events.
        """
        self._logger = logger or logging.getLogger("truthound_dbt.hooks")
        self._level = level

    def on_test_start(self, event: TestStartEvent) -> None:
        """Log test start."""
        self._logger.log(
            self._level,
            "Test started: %s (model=%s, column=%s)",
            event.test_name,
            event.model,
            event.column,
        )

    def on_test_end(self, event: TestEndEvent) -> None:
        """Log test end."""
        status = "PASSED" if event.passed else "FAILED"
        self._logger.log(
            self._level,
            "Test ended: %s [%s] duration=%.2fms failures=%d",
            event.test_name,
            status,
            event.duration_ms,
            event.failures,
        )

    def on_parse_start(self, event: ParseStartEvent) -> None:
        """Log parse start."""
        self._logger.log(
            self._level,
            "Parsing manifest: %s",
            event.manifest_path,
        )

    def on_parse_end(self, event: ParseEndEvent) -> None:
        """Log parse end."""
        status = "SUCCESS" if event.success else "FAILED"
        self._logger.log(
            self._level,
            "Parsing completed: %s [%s] duration=%.2fms tests=%d models=%d",
            event.manifest_path,
            status,
            event.duration_ms,
            event.test_count,
            event.model_count,
        )

    def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Log conversion start."""
        self._logger.log(
            self._level,
            "Converting %d rules with adapter '%s'",
            event.rule_count,
            event.adapter_name,
        )

    def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Log conversion end."""
        status = "SUCCESS" if event.success else "FAILED"
        self._logger.log(
            self._level,
            "Conversion completed: [%s] duration=%.2fms converted=%d errors=%d",
            status,
            event.duration_ms,
            event.converted_count,
            event.error_count,
        )

    def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Log generation start."""
        self._logger.log(
            self._level,
            "Generating %s for model '%s'",
            event.generation_type,
            event.model,
        )

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Log generation end."""
        status = "SUCCESS" if event.success else "FAILED"
        self._logger.log(
            self._level,
            "Generation completed: %s for '%s' [%s] duration=%.2fms",
            event.generation_type,
            event.model,
            status,
            event.duration_ms,
        )


# =============================================================================
# Metrics Hook
# =============================================================================


@dataclass
class DbtMetrics:
    """Collected metrics from dbt operations.

    Attributes:
        tests_started: Number of tests started.
        tests_completed: Number of tests completed.
        tests_passed: Number of tests passed.
        tests_failed: Number of tests failed.
        total_test_duration_ms: Total test duration.
        parse_count: Number of parse operations.
        parse_duration_ms: Total parse duration.
        conversion_count: Number of conversion operations.
        rules_converted: Number of rules converted.
        conversion_errors: Number of conversion errors.
        generation_count: Number of generation operations.
    """

    tests_started: int = 0
    tests_completed: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    total_test_duration_ms: float = 0.0
    parse_count: int = 0
    parse_duration_ms: float = 0.0
    conversion_count: int = 0
    rules_converted: int = 0
    conversion_errors: int = 0
    generation_count: int = 0
    generation_duration_ms: float = 0.0

    @property
    def test_success_rate(self) -> float:
        """Calculate test success rate."""
        if self.tests_completed == 0:
            return 0.0
        return (self.tests_passed / self.tests_completed) * 100

    @property
    def average_test_duration_ms(self) -> float:
        """Calculate average test duration."""
        if self.tests_completed == 0:
            return 0.0
        return self.total_test_duration_ms / self.tests_completed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tests_started": self.tests_started,
            "tests_completed": self.tests_completed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "test_success_rate": self.test_success_rate,
            "average_test_duration_ms": self.average_test_duration_ms,
            "parse_count": self.parse_count,
            "parse_duration_ms": self.parse_duration_ms,
            "conversion_count": self.conversion_count,
            "rules_converted": self.rules_converted,
            "conversion_errors": self.conversion_errors,
            "generation_count": self.generation_count,
            "generation_duration_ms": self.generation_duration_ms,
        }


class MetricsDbtHook(BaseDbtHook):
    """Hook that collects metrics from lifecycle events.

    Example:
        >>> hook = MetricsDbtHook()
        >>> hook.on_test_end(TestEndEvent(test_name="test", passed=True))
        >>> print(hook.metrics.test_success_rate)
        100.0
    """

    def __init__(self) -> None:
        """Initialize metrics hook."""
        self._metrics = DbtMetrics()

    @property
    def metrics(self) -> DbtMetrics:
        """Return collected metrics."""
        return self._metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = DbtMetrics()

    def on_test_start(self, event: TestStartEvent) -> None:
        """Track test start."""
        self._metrics.tests_started += 1

    def on_test_end(self, event: TestEndEvent) -> None:
        """Track test end."""
        self._metrics.tests_completed += 1
        self._metrics.total_test_duration_ms += event.duration_ms

        if event.passed:
            self._metrics.tests_passed += 1
        else:
            self._metrics.tests_failed += 1

    def on_parse_start(self, event: ParseStartEvent) -> None:
        """Track parse start."""
        pass

    def on_parse_end(self, event: ParseEndEvent) -> None:
        """Track parse end."""
        self._metrics.parse_count += 1
        self._metrics.parse_duration_ms += event.duration_ms

    def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Track conversion start."""
        pass

    def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Track conversion end."""
        self._metrics.conversion_count += 1
        self._metrics.rules_converted += event.converted_count
        self._metrics.conversion_errors += event.error_count

    def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Track generation start."""
        pass

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Track generation end."""
        self._metrics.generation_count += 1
        self._metrics.generation_duration_ms += event.duration_ms


# =============================================================================
# Composite Hook
# =============================================================================


class CompositeDbtHook(BaseDbtHook):
    """Hook that combines multiple hooks.

    Example:
        >>> logging_hook = LoggingDbtHook()
        >>> metrics_hook = MetricsDbtHook()
        >>> composite = CompositeDbtHook([logging_hook, metrics_hook])
        >>> composite.on_test_start(event)  # Calls both hooks
    """

    def __init__(
        self,
        hooks: Sequence[DbtHook],
        *,
        ignore_errors: bool = True,
    ) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to combine.
            ignore_errors: If True, continue if a hook fails.
        """
        self._hooks = list(hooks)
        self._ignore_errors = ignore_errors

    @property
    def hooks(self) -> list[DbtHook]:
        """Return the list of hooks."""
        return self._hooks

    def add_hook(self, hook: DbtHook) -> None:
        """Add a hook to the composite."""
        self._hooks.append(hook)

    def remove_hook(self, hook: DbtHook) -> None:
        """Remove a hook from the composite."""
        self._hooks.remove(hook)

    def _call_hooks(self, method_name: str, event: DbtEvent) -> None:
        """Call a method on all hooks."""
        for hook in self._hooks:
            try:
                method = getattr(hook, method_name, None)
                if method:
                    method(event)
            except Exception as e:
                if not self._ignore_errors:
                    raise HookExecutionError(
                        hook_name=type(hook).__name__,
                        event=method_name,
                        cause=e,
                    ) from e

    def on_test_start(self, event: TestStartEvent) -> None:
        """Call on_test_start on all hooks."""
        self._call_hooks("on_test_start", event)

    def on_test_end(self, event: TestEndEvent) -> None:
        """Call on_test_end on all hooks."""
        self._call_hooks("on_test_end", event)

    def on_parse_start(self, event: ParseStartEvent) -> None:
        """Call on_parse_start on all hooks."""
        self._call_hooks("on_parse_start", event)

    def on_parse_end(self, event: ParseEndEvent) -> None:
        """Call on_parse_end on all hooks."""
        self._call_hooks("on_parse_end", event)

    def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Call on_conversion_start on all hooks."""
        self._call_hooks("on_conversion_start", event)

    def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Call on_conversion_end on all hooks."""
        self._call_hooks("on_conversion_end", event)

    def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Call on_generation_start on all hooks."""
        self._call_hooks("on_generation_start", event)

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Call on_generation_end on all hooks."""
        self._call_hooks("on_generation_end", event)


# =============================================================================
# Async Implementations
# =============================================================================


class AsyncLoggingDbtHook(AsyncBaseDbtHook):
    """Async hook that logs all lifecycle events."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Initialize async logging hook."""
        self._logger = logger or logging.getLogger("truthound_dbt.hooks")
        self._level = level

    async def on_test_start(self, event: TestStartEvent) -> None:
        """Log test start."""
        self._logger.log(
            self._level,
            "Test started: %s (model=%s, column=%s)",
            event.test_name,
            event.model,
            event.column,
        )

    async def on_test_end(self, event: TestEndEvent) -> None:
        """Log test end."""
        status = "PASSED" if event.passed else "FAILED"
        self._logger.log(
            self._level,
            "Test ended: %s [%s] duration=%.2fms failures=%d",
            event.test_name,
            status,
            event.duration_ms,
            event.failures,
        )

    async def on_parse_start(self, event: ParseStartEvent) -> None:
        """Log parse start."""
        self._logger.log(self._level, "Parsing manifest: %s", event.manifest_path)

    async def on_parse_end(self, event: ParseEndEvent) -> None:
        """Log parse end."""
        status = "SUCCESS" if event.success else "FAILED"
        self._logger.log(
            self._level,
            "Parsing completed: %s [%s] duration=%.2fms",
            event.manifest_path,
            status,
            event.duration_ms,
        )

    async def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Log conversion start."""
        self._logger.log(
            self._level,
            "Converting %d rules with adapter '%s'",
            event.rule_count,
            event.adapter_name,
        )

    async def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Log conversion end."""
        status = "SUCCESS" if event.success else "FAILED"
        self._logger.log(
            self._level,
            "Conversion completed: [%s] converted=%d errors=%d",
            status,
            event.converted_count,
            event.error_count,
        )

    async def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Log generation start."""
        self._logger.log(
            self._level,
            "Generating %s for model '%s'",
            event.generation_type,
            event.model,
        )

    async def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Log generation end."""
        status = "SUCCESS" if event.success else "FAILED"
        self._logger.log(
            self._level,
            "Generation completed: %s [%s] duration=%.2fms",
            event.generation_type,
            status,
            event.duration_ms,
        )


class AsyncCompositeDbtHook(AsyncBaseDbtHook):
    """Async hook that combines multiple async hooks."""

    def __init__(
        self,
        hooks: Sequence[AsyncDbtHook],
        *,
        ignore_errors: bool = True,
    ) -> None:
        """Initialize async composite hook."""
        self._hooks = list(hooks)
        self._ignore_errors = ignore_errors

    async def _call_hooks(self, method_name: str, event: DbtEvent) -> None:
        """Call a method on all hooks concurrently."""
        tasks = []
        for hook in self._hooks:
            method = getattr(hook, method_name, None)
            if method:
                tasks.append(method(event))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if not self._ignore_errors:
                for result in results:
                    if isinstance(result, Exception):
                        raise result

    async def on_test_start(self, event: TestStartEvent) -> None:
        """Call on_test_start on all hooks."""
        await self._call_hooks("on_test_start", event)

    async def on_test_end(self, event: TestEndEvent) -> None:
        """Call on_test_end on all hooks."""
        await self._call_hooks("on_test_end", event)

    async def on_parse_start(self, event: ParseStartEvent) -> None:
        """Call on_parse_start on all hooks."""
        await self._call_hooks("on_parse_start", event)

    async def on_parse_end(self, event: ParseEndEvent) -> None:
        """Call on_parse_end on all hooks."""
        await self._call_hooks("on_parse_end", event)

    async def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Call on_conversion_start on all hooks."""
        await self._call_hooks("on_conversion_start", event)

    async def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Call on_conversion_end on all hooks."""
        await self._call_hooks("on_conversion_end", event)

    async def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Call on_generation_start on all hooks."""
        await self._call_hooks("on_generation_start", event)

    async def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Call on_generation_end on all hooks."""
        await self._call_hooks("on_generation_end", event)


# =============================================================================
# Sync to Async Adapter
# =============================================================================


class SyncToAsyncDbtHookAdapter(AsyncBaseDbtHook):
    """Adapter to use sync hooks in async context.

    Example:
        >>> sync_hook = LoggingDbtHook()
        >>> async_hook = SyncToAsyncDbtHookAdapter(sync_hook)
        >>> await async_hook.on_test_start(event)
    """

    def __init__(self, hook: DbtHook) -> None:
        """Initialize adapter.

        Args:
            hook: Sync hook to wrap.
        """
        self._hook = hook

    async def on_test_start(self, event: TestStartEvent) -> None:
        """Call sync hook."""
        self._hook.on_test_start(event)

    async def on_test_end(self, event: TestEndEvent) -> None:
        """Call sync hook."""
        self._hook.on_test_end(event)

    async def on_parse_start(self, event: ParseStartEvent) -> None:
        """Call sync hook."""
        self._hook.on_parse_start(event)

    async def on_parse_end(self, event: ParseEndEvent) -> None:
        """Call sync hook."""
        self._hook.on_parse_end(event)

    async def on_conversion_start(self, event: ConversionStartEvent) -> None:
        """Call sync hook."""
        self._hook.on_conversion_start(event)

    async def on_conversion_end(self, event: ConversionEndEvent) -> None:
        """Call sync hook."""
        self._hook.on_conversion_end(event)

    async def on_generation_start(self, event: GenerationStartEvent) -> None:
        """Call sync hook."""
        self._hook.on_generation_start(event)

    async def on_generation_end(self, event: GenerationEndEvent) -> None:
        """Call sync hook."""
        self._hook.on_generation_end(event)
