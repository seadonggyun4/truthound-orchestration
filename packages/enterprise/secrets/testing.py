"""Testing utilities for secret management.

This module provides mock implementations and test fixtures
for testing code that uses secret providers.

Example:
    >>> from packages.enterprise.secrets.testing import (
    ...     MockSecretProvider,
    ...     create_test_provider,
    ... )
    >>>
    >>> # Create a mock provider with preset secrets
    >>> provider = create_test_provider({
    ...     "database/password": "test-secret",
    ...     "api/key": "test-api-key",
    ... })
    >>> secret = provider.get("database/password")
    >>> assert secret.value == "test-secret"
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from .backends.memory import InMemorySecretProvider
from .base import (
    AsyncSecretProvider,
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
    SecretMetadata,
    SecretProvider,
    SecretType,
    SecretValue,
)
from .hooks import SecretHook, SecretOperationContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# =============================================================================
# Mock Providers
# =============================================================================


@dataclass
class MockSecretProvider:
    """Mock secret provider for testing.

    Allows configuring behavior and tracking calls.

    Attributes:
        secrets: Pre-configured secrets.
        call_history: List of method calls.
        should_fail: Whether to raise errors.
        failure_exception: Exception to raise on failure.
        delay_seconds: Artificial delay for each operation.

    Example:
        >>> provider = MockSecretProvider()
        >>> provider.secrets["db/pass"] = "secret123"
        >>> secret = provider.get("db/pass")
        >>> assert secret.value == "secret123"
        >>> assert len(provider.call_history) == 1
    """

    secrets: dict[str, str] = field(default_factory=dict)
    call_history: list[dict[str, Any]] = field(default_factory=list)
    should_fail: bool = False
    failure_exception: Exception | None = None
    delay_seconds: float = 0.0

    def _record_call(self, method: str, **kwargs: Any) -> None:
        """Record a method call."""
        self.call_history.append({
            "method": method,
            "timestamp": datetime.now(timezone.utc),
            **kwargs,
        })

    def _maybe_delay(self) -> None:
        """Apply artificial delay if configured."""
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

    def _maybe_fail(self) -> None:
        """Raise exception if configured to fail."""
        if self.should_fail:
            raise self.failure_exception or RuntimeError("Mock failure")

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a mock secret."""
        self._record_call("get", path=path, version=version)
        self._maybe_delay()
        self._maybe_fail()

        if path not in self.secrets:
            return None

        return SecretValue(
            value=self.secrets[path],
            version=version or "mock-1",
            created_at=datetime.now(timezone.utc),
            secret_type=SecretType.STRING,
            metadata={"mock": True},
        )

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set a mock secret."""
        self._record_call(
            "set",
            path=path,
            value=value,
            secret_type=secret_type,
            expires_at=expires_at,
            metadata=metadata,
        )
        self._maybe_delay()
        self._maybe_fail()

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        self.secrets[path] = value

        return SecretValue(
            value=value,
            version="mock-1",
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            secret_type=secret_type,
            metadata={"mock": True},
        )

    def delete(self, path: str) -> bool:
        """Delete a mock secret."""
        self._record_call("delete", path=path)
        self._maybe_delay()
        self._maybe_fail()

        if path in self.secrets:
            del self.secrets[path]
            return True
        return False

    def exists(self, path: str) -> bool:
        """Check if a mock secret exists."""
        self._record_call("exists", path=path)
        self._maybe_delay()
        self._maybe_fail()
        return path in self.secrets

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List mock secrets."""
        self._record_call("list", prefix=prefix, limit=limit, offset=offset)
        self._maybe_delay()
        self._maybe_fail()

        results = [
            SecretMetadata(
                path=path,
                version="mock-1",
                secret_type=SecretType.STRING,
            )
            for path in sorted(self.secrets.keys())
            if path.startswith(prefix)
        ]

        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return results

    def reset(self) -> None:
        """Reset all state."""
        self.secrets.clear()
        self.call_history.clear()
        self.should_fail = False
        self.failure_exception = None


@dataclass
class AsyncMockSecretProvider:
    """Async mock secret provider for testing.

    Attributes:
        secrets: Pre-configured secrets.
        call_history: List of method calls.
        should_fail: Whether to raise errors.
        failure_exception: Exception to raise on failure.
        delay_seconds: Artificial delay for each operation.
    """

    secrets: dict[str, str] = field(default_factory=dict)
    call_history: list[dict[str, Any]] = field(default_factory=list)
    should_fail: bool = False
    failure_exception: Exception | None = None
    delay_seconds: float = 0.0

    def _record_call(self, method: str, **kwargs: Any) -> None:
        """Record a method call."""
        self.call_history.append({
            "method": method,
            "timestamp": datetime.now(timezone.utc),
            **kwargs,
        })

    async def _maybe_delay(self) -> None:
        """Apply artificial delay if configured."""
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

    def _maybe_fail(self) -> None:
        """Raise exception if configured to fail."""
        if self.should_fail:
            raise self.failure_exception or RuntimeError("Mock failure")

    async def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a mock secret."""
        self._record_call("get", path=path, version=version)
        await self._maybe_delay()
        self._maybe_fail()

        if path not in self.secrets:
            return None

        return SecretValue(
            value=self.secrets[path],
            version=version or "mock-1",
            created_at=datetime.now(timezone.utc),
            secret_type=SecretType.STRING,
            metadata={"mock": True},
        )

    async def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Set a mock secret."""
        self._record_call("set", path=path, value=value)
        await self._maybe_delay()
        self._maybe_fail()

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        self.secrets[path] = value

        return SecretValue(
            value=value,
            version="mock-1",
            created_at=datetime.now(timezone.utc),
            secret_type=secret_type,
            metadata={"mock": True},
        )

    async def delete(self, path: str) -> bool:
        """Delete a mock secret."""
        self._record_call("delete", path=path)
        await self._maybe_delay()
        self._maybe_fail()

        if path in self.secrets:
            del self.secrets[path]
            return True
        return False

    async def exists(self, path: str) -> bool:
        """Check if a mock secret exists."""
        self._record_call("exists", path=path)
        await self._maybe_delay()
        self._maybe_fail()
        return path in self.secrets

    async def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List mock secrets."""
        self._record_call("list", prefix=prefix)
        await self._maybe_delay()
        self._maybe_fail()

        results = [
            SecretMetadata(path=path, version="mock-1", secret_type=SecretType.STRING)
            for path in sorted(self.secrets.keys())
            if path.startswith(prefix)
        ]

        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return results

    def reset(self) -> None:
        """Reset all state."""
        self.secrets.clear()
        self.call_history.clear()
        self.should_fail = False
        self.failure_exception = None


# =============================================================================
# Mock Hooks
# =============================================================================


@dataclass
class MockSecretHook:
    """Mock hook for testing hook behavior.

    Tracks all hook calls for verification. Uses the same method names
    as the SecretHook protocol (on_before_get, on_after_get, etc.).

    Attributes:
        before_get_calls: List of before_get calls.
        after_get_calls: List of after_get calls.
        before_set_calls: List of before_set calls.
        after_set_calls: List of after_set calls.
        before_delete_calls: List of before_delete calls.
        after_delete_calls: List of after_delete calls.
    """

    before_get_calls: list[dict[str, Any]] = field(default_factory=list)
    after_get_calls: list[dict[str, Any]] = field(default_factory=list)
    before_set_calls: list[dict[str, Any]] = field(default_factory=list)
    after_set_calls: list[dict[str, Any]] = field(default_factory=list)
    before_delete_calls: list[dict[str, Any]] = field(default_factory=list)
    after_delete_calls: list[dict[str, Any]] = field(default_factory=list)

    def on_before_get(self, context: SecretOperationContext) -> None:
        """Record before_get call."""
        self.before_get_calls.append({"context": context})

    def on_after_get(
        self,
        context: SecretOperationContext,
        result: SecretValue | None,
        duration_ms: float,
    ) -> None:
        """Record after_get call."""
        self.after_get_calls.append({
            "context": context,
            "result": result,
            "duration_ms": duration_ms,
        })

    def on_before_set(self, context: SecretOperationContext) -> None:
        """Record before_set call."""
        self.before_set_calls.append({"context": context})

    def on_after_set(
        self,
        context: SecretOperationContext,
        result: SecretValue,
        duration_ms: float,
    ) -> None:
        """Record after_set call."""
        self.after_set_calls.append({
            "context": context,
            "result": result,
            "duration_ms": duration_ms,
        })

    def on_before_delete(self, context: SecretOperationContext) -> None:
        """Record before_delete call."""
        self.before_delete_calls.append({"context": context})

    def on_after_delete(
        self,
        context: SecretOperationContext,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record after_delete call."""
        self.after_delete_calls.append({
            "context": context,
            "success": success,
            "duration_ms": duration_ms,
        })

    def on_error(
        self,
        context: SecretOperationContext,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """Record error."""
        pass  # Could add error tracking if needed

    @property
    def total_calls(self) -> int:
        """Get total number of calls."""
        return (
            len(self.before_get_calls)
            + len(self.after_get_calls)
            + len(self.before_set_calls)
            + len(self.after_set_calls)
            + len(self.before_delete_calls)
            + len(self.after_delete_calls)
        )

    def reset(self) -> None:
        """Reset all recorded calls."""
        self.before_get_calls.clear()
        self.after_get_calls.clear()
        self.before_set_calls.clear()
        self.after_set_calls.clear()
        self.before_delete_calls.clear()
        self.after_delete_calls.clear()


# =============================================================================
# Factory Functions
# =============================================================================


def create_test_provider(
    secrets: Mapping[str, str] | None = None,
) -> InMemorySecretProvider:
    """Create a test provider with pre-populated secrets.

    Args:
        secrets: Optional mapping of path to value.

    Returns:
        Configured InMemorySecretProvider.

    Example:
        >>> provider = create_test_provider({
        ...     "database/password": "secret123",
        ...     "api/key": "key-abc",
        ... })
        >>> secret = provider.get("database/password")
        >>> assert secret.value == "secret123"
    """
    provider = InMemorySecretProvider()

    if secrets:
        for path, value in secrets.items():
            provider.set(path, value)

    return provider


def create_mock_provider(
    secrets: Mapping[str, str] | None = None,
    should_fail: bool = False,
    failure_exception: Exception | None = None,
    delay_seconds: float = 0.0,
) -> MockSecretProvider:
    """Create a mock provider with configurable behavior.

    Args:
        secrets: Optional mapping of path to value.
        should_fail: Whether operations should fail.
        failure_exception: Exception to raise on failure.
        delay_seconds: Artificial delay for operations.

    Returns:
        Configured MockSecretProvider.

    Example:
        >>> provider = create_mock_provider(should_fail=True)
        >>> try:
        ...     provider.get("any/path")
        ... except RuntimeError:
        ...     print("Failed as expected")
    """
    provider = MockSecretProvider(
        secrets=dict(secrets) if secrets else {},
        should_fail=should_fail,
        failure_exception=failure_exception,
        delay_seconds=delay_seconds,
    )
    return provider


def create_async_mock_provider(
    secrets: Mapping[str, str] | None = None,
    should_fail: bool = False,
    failure_exception: Exception | None = None,
    delay_seconds: float = 0.0,
) -> AsyncMockSecretProvider:
    """Create an async mock provider with configurable behavior.

    Args:
        secrets: Optional mapping of path to value.
        should_fail: Whether operations should fail.
        failure_exception: Exception to raise on failure.
        delay_seconds: Artificial delay for operations.

    Returns:
        Configured AsyncMockSecretProvider.
    """
    provider = AsyncMockSecretProvider(
        secrets=dict(secrets) if secrets else {},
        should_fail=should_fail,
        failure_exception=failure_exception,
        delay_seconds=delay_seconds,
    )
    return provider


def create_mock_hook() -> MockSecretHook:
    """Create a mock hook for testing.

    Returns:
        MockSecretHook instance.

    Example:
        >>> hook = create_mock_hook()
        >>> # Use with provider...
        >>> assert len(hook.before_get_calls) == 1
    """
    return MockSecretHook()


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class SecretTestCase:
    """Test case for secret operations.

    Attributes:
        name: Test case name.
        path: Secret path.
        value: Secret value.
        expected_value: Expected retrieved value.
        should_exist: Whether secret should exist.
        expected_error: Expected exception type.
    """

    name: str
    path: str
    value: str = ""
    expected_value: str | None = None
    should_exist: bool = True
    expected_error: type[Exception] | None = None


# Common test cases
COMMON_TEST_CASES: tuple[SecretTestCase, ...] = (
    SecretTestCase(
        name="simple_string",
        path="test/simple",
        value="hello",
        expected_value="hello",
    ),
    SecretTestCase(
        name="nested_path",
        path="a/b/c/d/secret",
        value="nested",
        expected_value="nested",
    ),
    SecretTestCase(
        name="special_characters",
        path="test/special-chars_123",
        value="value!@#$%",
        expected_value="value!@#$%",
    ),
    SecretTestCase(
        name="unicode",
        path="test/unicode",
        value="こんにちは世界",
        expected_value="こんにちは世界",
    ),
    SecretTestCase(
        name="empty_value",
        path="test/empty",
        value="",
        expected_value="",
    ),
    SecretTestCase(
        name="json_value",
        path="test/json",
        value='{"key": "value", "number": 42}',
        expected_value='{"key": "value", "number": 42}',
    ),
)


@contextmanager
def temporary_secrets(
    provider: SecretProvider,
    secrets: Mapping[str, str],
):
    """Context manager that sets up and tears down secrets.

    Args:
        provider: Secret provider.
        secrets: Secrets to set up.

    Yields:
        The provider.

    Example:
        >>> provider = InMemorySecretProvider()
        >>> with temporary_secrets(provider, {"test": "value"}) as p:
        ...     secret = p.get("test")
        ...     assert secret.value == "value"
        >>> # Secrets are cleaned up after context
    """
    # Set up secrets
    paths = list(secrets.keys())
    for path, value in secrets.items():
        provider.set(path, value)

    try:
        yield provider
    finally:
        # Clean up secrets
        for path in paths:
            provider.delete(path)


# =============================================================================
# Assertions
# =============================================================================


def assert_secret_equal(
    secret: SecretValue | None,
    expected_value: str,
    msg: str | None = None,
) -> None:
    """Assert that a secret has the expected value.

    Args:
        secret: Secret value (may be None).
        expected_value: Expected value.
        msg: Optional assertion message.

    Raises:
        AssertionError: If assertion fails.
    """
    if secret is None:
        raise AssertionError(msg or "Secret is None")
    if secret.value != expected_value:
        raise AssertionError(
            msg or f"Secret value mismatch: {secret.value!r} != {expected_value!r}"
        )


def assert_secret_exists(
    provider: SecretProvider,
    path: str,
    msg: str | None = None,
) -> None:
    """Assert that a secret exists.

    Args:
        provider: Secret provider.
        path: Secret path.
        msg: Optional assertion message.

    Raises:
        AssertionError: If assertion fails.
    """
    if not provider.exists(path):
        raise AssertionError(msg or f"Secret does not exist: {path}")


def assert_secret_not_exists(
    provider: SecretProvider,
    path: str,
    msg: str | None = None,
) -> None:
    """Assert that a secret does not exist.

    Args:
        provider: Secret provider.
        path: Secret path.
        msg: Optional assertion message.

    Raises:
        AssertionError: If assertion fails.
    """
    if provider.exists(path):
        raise AssertionError(msg or f"Secret exists: {path}")


def assert_hook_called(
    hook: MockSecretHook,
    method: str,
    times: int | None = None,
    msg: str | None = None,
) -> None:
    """Assert that a hook method was called.

    Args:
        hook: Mock hook.
        method: Method name (e.g., "before_get").
        times: Expected number of calls (None for any).
        msg: Optional assertion message.

    Raises:
        AssertionError: If assertion fails.
    """
    calls = getattr(hook, f"{method}_calls", [])
    if times is not None and len(calls) != times:
        raise AssertionError(
            msg or f"Hook {method} called {len(calls)} times, expected {times}"
        )
    if not calls:
        raise AssertionError(msg or f"Hook {method} was not called")


# =============================================================================
# Health Check Testing
# =============================================================================


class AlwaysHealthyProvider(InMemorySecretProvider, HealthCheckable):
    """Provider that always reports healthy status."""

    def health_check(self) -> HealthCheckResult:
        """Return healthy status."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Always healthy",
            latency_ms=0.1,
        )


class AlwaysUnhealthyProvider(InMemorySecretProvider, HealthCheckable):
    """Provider that always reports unhealthy status."""

    def health_check(self) -> HealthCheckResult:
        """Return unhealthy status."""
        return HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message="Always unhealthy",
            latency_ms=0.1,
        )


class ConfigurableHealthProvider(InMemorySecretProvider, HealthCheckable):
    """Provider with configurable health status."""

    def __init__(
        self,
        status: HealthStatus = HealthStatus.HEALTHY,
        message: str = "Test provider",
        latency_ms: float = 1.0,
    ) -> None:
        """Initialize the provider.

        Args:
            status: Health status to return.
            message: Health message.
            latency_ms: Simulated latency.
        """
        super().__init__()
        self._health_status = status
        self._health_message = message
        self._health_latency = latency_ms

    def set_health(
        self,
        status: HealthStatus,
        message: str | None = None,
    ) -> None:
        """Update health status.

        Args:
            status: New status.
            message: Optional new message.
        """
        self._health_status = status
        if message:
            self._health_message = message

    def health_check(self) -> HealthCheckResult:
        """Return configured health status."""
        return HealthCheckResult(
            status=self._health_status,
            message=self._health_message,
            latency_ms=self._health_latency,
        )
