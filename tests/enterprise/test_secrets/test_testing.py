"""Tests for testing utilities."""

from __future__ import annotations

import pytest

from packages.enterprise.secrets.base import HealthStatus, SecretType
from packages.enterprise.secrets.hooks import SecretOperationContext
from packages.enterprise.secrets.testing import (
    AlwaysHealthyProvider,
    AlwaysUnhealthyProvider,
    AsyncMockSecretProvider,
    ConfigurableHealthProvider,
    MockSecretHook,
    MockSecretProvider,
    assert_hook_called,
    assert_secret_equal,
    assert_secret_exists,
    assert_secret_not_exists,
    create_async_mock_provider,
    create_mock_hook,
    create_mock_provider,
    create_test_provider,
    temporary_secrets,
)


class TestMockSecretProvider:
    """Tests for MockSecretProvider."""

    def test_set_and_get(self):
        """Test basic set and get."""
        provider = MockSecretProvider()
        provider.set("test/secret", "value")

        result = provider.get("test/secret")
        assert result is not None
        assert result.value == "value"

    def test_get_nonexistent(self):
        """Test get for nonexistent secret."""
        provider = MockSecretProvider()
        result = provider.get("nonexistent")
        assert result is None

    def test_delete(self):
        """Test delete."""
        provider = MockSecretProvider()
        provider.set("test/secret", "value")

        assert provider.delete("test/secret") is True
        assert provider.get("test/secret") is None

    def test_exists(self):
        """Test exists."""
        provider = MockSecretProvider()
        provider.set("test/secret", "value")

        assert provider.exists("test/secret") is True
        assert provider.exists("nonexistent") is False

    def test_list(self):
        """Test list."""
        provider = MockSecretProvider()
        provider.set("a/secret", "value1")
        provider.set("b/secret", "value2")

        secrets = provider.list()
        assert len(secrets) == 2

    def test_call_history(self):
        """Test call history is recorded."""
        provider = MockSecretProvider()
        provider.set("test/secret", "value")
        provider.get("test/secret")
        provider.delete("test/secret")

        assert len(provider.call_history) == 3
        assert provider.call_history[0]["method"] == "set"
        assert provider.call_history[1]["method"] == "get"
        assert provider.call_history[2]["method"] == "delete"

    def test_should_fail(self):
        """Test failure mode."""
        provider = MockSecretProvider(should_fail=True)

        with pytest.raises(RuntimeError):
            provider.get("test/secret")

    def test_custom_exception(self):
        """Test custom failure exception."""
        provider = MockSecretProvider(
            should_fail=True,
            failure_exception=ValueError("Custom error"),
        )

        with pytest.raises(ValueError, match="Custom error"):
            provider.get("test/secret")

    def test_reset(self):
        """Test reset clears state."""
        provider = MockSecretProvider()
        provider.set("test/secret", "value")
        provider.should_fail = True

        provider.reset()

        assert len(provider.secrets) == 0
        assert len(provider.call_history) == 0
        assert provider.should_fail is False


class TestAsyncMockSecretProvider:
    """Tests for AsyncMockSecretProvider."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic async set and get."""
        provider = AsyncMockSecretProvider()
        await provider.set("test/secret", "value")

        result = await provider.get("test/secret")
        assert result is not None
        assert result.value == "value"

    @pytest.mark.asyncio
    async def test_should_fail(self):
        """Test async failure mode."""
        provider = AsyncMockSecretProvider(should_fail=True)

        with pytest.raises(RuntimeError):
            await provider.get("test/secret")


class TestMockSecretHook:
    """Tests for MockSecretHook."""

    def test_before_get_recorded(self):
        """Test on_before_get is recorded."""
        hook = MockSecretHook()
        context = SecretOperationContext(
            path="test/secret",
            operation="get",
            provider_name="mock",
        )

        hook.on_before_get(context)

        assert len(hook.before_get_calls) == 1

    def test_after_get_recorded(self):
        """Test on_after_get is recorded."""
        hook = MockSecretHook()
        context = SecretOperationContext(
            path="test/secret",
            operation="get",
            provider_name="mock",
        )

        hook.on_after_get(context, result=None, duration_ms=0.0)

        assert len(hook.after_get_calls) == 1

    def test_total_calls(self):
        """Test total_calls property."""
        hook = MockSecretHook()
        context = SecretOperationContext(
            path="test",
            operation="get",
            provider_name="mock",
        )

        hook.on_before_get(context)
        hook.on_after_get(context, result=None, duration_ms=0.0)

        assert hook.total_calls == 2

    def test_reset(self):
        """Test reset clears all calls."""
        hook = MockSecretHook()
        context = SecretOperationContext(
            path="test",
            operation="get",
            provider_name="mock",
        )

        hook.on_before_get(context)
        hook.reset()

        assert hook.total_calls == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_test_provider(self):
        """Test create_test_provider."""
        provider = create_test_provider({
            "db/password": "secret123",
            "api/key": "key-abc",
        })

        assert provider.get("db/password").value == "secret123"
        assert provider.get("api/key").value == "key-abc"

    def test_create_mock_provider(self):
        """Test create_mock_provider."""
        provider = create_mock_provider(
            secrets={"test": "value"},
            should_fail=False,
        )

        assert provider.get("test").value == "value"

    def test_create_mock_provider_failing(self):
        """Test create_mock_provider with failure."""
        provider = create_mock_provider(should_fail=True)

        with pytest.raises(RuntimeError):
            provider.get("test")

    def test_create_async_mock_provider(self):
        """Test create_async_mock_provider."""
        provider = create_async_mock_provider(
            secrets={"test": "value"},
        )

        assert provider.secrets["test"] == "value"

    def test_create_mock_hook(self):
        """Test create_mock_hook."""
        hook = create_mock_hook()
        assert isinstance(hook, MockSecretHook)


class TestTemporarySecrets:
    """Tests for temporary_secrets context manager."""

    def test_secrets_set_up(self):
        """Test secrets are set up."""
        provider = create_test_provider()

        with temporary_secrets(provider, {"test": "value"}) as p:
            assert p.get("test").value == "value"

    def test_secrets_cleaned_up(self):
        """Test secrets are cleaned up."""
        provider = create_test_provider()

        with temporary_secrets(provider, {"test": "value"}):
            pass

        assert provider.get("test") is None


class TestAssertions:
    """Tests for assertion functions."""

    def test_assert_secret_equal_success(self):
        """Test assert_secret_equal with matching value."""
        provider = create_test_provider({"test": "value"})
        secret = provider.get("test")

        # Should not raise
        assert_secret_equal(secret, "value")

    def test_assert_secret_equal_failure(self):
        """Test assert_secret_equal with mismatch."""
        provider = create_test_provider({"test": "value"})
        secret = provider.get("test")

        with pytest.raises(AssertionError):
            assert_secret_equal(secret, "different")

    def test_assert_secret_equal_none(self):
        """Test assert_secret_equal with None."""
        with pytest.raises(AssertionError):
            assert_secret_equal(None, "value")

    def test_assert_secret_exists_success(self):
        """Test assert_secret_exists with existing secret."""
        provider = create_test_provider({"test": "value"})

        # Should not raise
        assert_secret_exists(provider, "test")

    def test_assert_secret_exists_failure(self):
        """Test assert_secret_exists with missing secret."""
        provider = create_test_provider()

        with pytest.raises(AssertionError):
            assert_secret_exists(provider, "nonexistent")

    def test_assert_secret_not_exists_success(self):
        """Test assert_secret_not_exists with missing secret."""
        provider = create_test_provider()

        # Should not raise
        assert_secret_not_exists(provider, "nonexistent")

    def test_assert_secret_not_exists_failure(self):
        """Test assert_secret_not_exists with existing secret."""
        provider = create_test_provider({"test": "value"})

        with pytest.raises(AssertionError):
            assert_secret_not_exists(provider, "test")

    def test_assert_hook_called(self):
        """Test assert_hook_called."""
        hook = MockSecretHook()
        context = SecretOperationContext(
            path="test",
            operation="get",
            provider_name="mock",
        )
        hook.on_before_get(context)

        # Should not raise
        assert_hook_called(hook, "before_get", times=1)

    def test_assert_hook_called_wrong_count(self):
        """Test assert_hook_called with wrong count."""
        hook = MockSecretHook()

        with pytest.raises(AssertionError):
            assert_hook_called(hook, "before_get", times=1)


class TestHealthProviders:
    """Tests for health check test providers."""

    def test_always_healthy(self):
        """Test AlwaysHealthyProvider."""
        provider = AlwaysHealthyProvider()
        result = provider.health_check()

        assert result.status == HealthStatus.HEALTHY

    def test_always_unhealthy(self):
        """Test AlwaysUnhealthyProvider."""
        provider = AlwaysUnhealthyProvider()
        result = provider.health_check()

        assert result.status == HealthStatus.UNHEALTHY

    def test_configurable_health(self):
        """Test ConfigurableHealthProvider."""
        provider = ConfigurableHealthProvider(
            status=HealthStatus.DEGRADED,
            message="Test message",
        )
        result = provider.health_check()

        assert result.status == HealthStatus.DEGRADED
        assert result.message == "Test message"

    def test_configurable_health_update(self):
        """Test updating ConfigurableHealthProvider."""
        provider = ConfigurableHealthProvider(status=HealthStatus.HEALTHY)

        provider.set_health(HealthStatus.UNHEALTHY, "New message")
        result = provider.health_check()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.message == "New message"
