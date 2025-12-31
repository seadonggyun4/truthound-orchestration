"""Tests for secret middleware wrappers."""

from __future__ import annotations

import pytest

from packages.enterprise.secrets.backends import InMemorySecretProvider
from packages.enterprise.secrets.base import SecretType
from packages.enterprise.secrets.middleware import (
    CachingProviderWrapper,
    HookedProviderWrapper,
    NamespacedProviderWrapper,
    ValidatingProviderWrapper,
    create_wrapped_provider,
)
from packages.enterprise.secrets.testing import MockSecretHook


class TestNamespacedProviderWrapper:
    """Tests for NamespacedProviderWrapper."""

    def test_namespace_prefix(self):
        """Test namespace is prefixed to path."""
        provider = InMemorySecretProvider()
        wrapped = NamespacedProviderWrapper(provider, namespace="prod")

        wrapped.set("database/password", "secret123")

        # Check actual storage path
        actual = provider.get("prod/database/password")
        assert actual is not None
        assert actual.value == "secret123"

    def test_get_with_namespace(self):
        """Test get with namespace."""
        provider = InMemorySecretProvider()
        wrapped = NamespacedProviderWrapper(provider, namespace="prod")

        wrapped.set("database/password", "secret123")
        result = wrapped.get("database/password")

        assert result is not None
        assert result.value == "secret123"

    def test_delete_with_namespace(self):
        """Test delete with namespace."""
        provider = InMemorySecretProvider()
        wrapped = NamespacedProviderWrapper(provider, namespace="prod")

        wrapped.set("test/secret", "value")
        assert wrapped.delete("test/secret") is True
        assert wrapped.get("test/secret") is None

    def test_exists_with_namespace(self):
        """Test exists with namespace."""
        provider = InMemorySecretProvider()
        wrapped = NamespacedProviderWrapper(provider, namespace="prod")

        wrapped.set("test/secret", "value")
        assert wrapped.exists("test/secret") is True
        assert wrapped.exists("nonexistent") is False

    def test_list_with_namespace(self):
        """Test list filters by namespace."""
        provider = InMemorySecretProvider()

        # Set up secrets in different namespaces
        provider.set("prod/a", "value1")
        provider.set("prod/b", "value2")
        provider.set("dev/c", "value3")

        wrapped = NamespacedProviderWrapper(provider, namespace="prod")
        secrets = wrapped.list()

        paths = [s.path for s in secrets]
        assert "a" in paths
        assert "b" in paths
        assert "c" not in paths


class TestHookedProviderWrapper:
    """Tests for HookedProviderWrapper."""

    def test_hook_before_get(self):
        """Test on_before_get hook is called."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()
        wrapped = HookedProviderWrapper(provider, hooks=[hook])

        provider.set("test/secret", "value")
        wrapped.get("test/secret")

        assert len(hook.before_get_calls) == 1

    def test_hook_after_get(self):
        """Test on_after_get hook is called."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()
        wrapped = HookedProviderWrapper(provider, hooks=[hook])

        provider.set("test/secret", "value")
        wrapped.get("test/secret")

        assert len(hook.after_get_calls) == 1
        assert hook.after_get_calls[0]["result"] is not None

    def test_hook_before_set(self):
        """Test on_before_set hook is called."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()
        wrapped = HookedProviderWrapper(provider, hooks=[hook])

        wrapped.set("test/secret", "value")

        assert len(hook.before_set_calls) == 1

    def test_hook_after_set(self):
        """Test on_after_set hook is called."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()
        wrapped = HookedProviderWrapper(provider, hooks=[hook])

        wrapped.set("test/secret", "value")

        assert len(hook.after_set_calls) == 1

    def test_hook_before_delete(self):
        """Test on_before_delete hook is called."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()
        wrapped = HookedProviderWrapper(provider, hooks=[hook])

        provider.set("test/secret", "value")
        wrapped.delete("test/secret")

        assert len(hook.before_delete_calls) == 1

    def test_hook_after_delete(self):
        """Test on_after_delete hook is called."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()
        wrapped = HookedProviderWrapper(provider, hooks=[hook])

        provider.set("test/secret", "value")
        wrapped.delete("test/secret")

        assert len(hook.after_delete_calls) == 1

    def test_multiple_hooks(self):
        """Test multiple hooks are called."""
        provider = InMemorySecretProvider()
        hook1 = MockSecretHook()
        hook2 = MockSecretHook()
        wrapped = HookedProviderWrapper(provider, hooks=[hook1, hook2])

        wrapped.set("test/secret", "value")

        assert len(hook1.before_set_calls) == 1
        assert len(hook2.before_set_calls) == 1


class TestCachingProviderWrapper:
    """Tests for CachingProviderWrapper."""

    def test_cache_hit(self):
        """Test cache hit returns cached value."""
        provider = InMemorySecretProvider()
        wrapped = CachingProviderWrapper(provider, ttl_seconds=60.0)

        # Set and get to populate cache
        wrapped.set("test/secret", "value")
        wrapped.get("test/secret")

        # Modify underlying value directly
        provider.set("test/secret", "modified")

        # Should still get cached value
        result = wrapped.get("test/secret")
        assert result.value == "value"

    def test_cache_invalidation_on_set(self):
        """Test cache is invalidated on set."""
        provider = InMemorySecretProvider()
        wrapped = CachingProviderWrapper(provider, ttl_seconds=60.0)

        # Set and get to populate cache
        wrapped.set("test/secret", "value1")
        wrapped.get("test/secret")

        # Set new value through wrapper (should invalidate cache)
        wrapped.set("test/secret", "value2")

        # Should get new value
        result = wrapped.get("test/secret")
        assert result.value == "value2"

    def test_cache_invalidation_on_delete(self):
        """Test cache is invalidated on delete."""
        provider = InMemorySecretProvider()
        wrapped = CachingProviderWrapper(provider, ttl_seconds=60.0)

        # Set and get to populate cache
        wrapped.set("test/secret", "value")
        wrapped.get("test/secret")

        # Delete (should invalidate cache)
        wrapped.delete("test/secret")

        # Should return None
        result = wrapped.get("test/secret")
        assert result is None


class TestValidatingProviderWrapper:
    """Tests for ValidatingProviderWrapper."""

    def test_valid_path(self):
        """Test valid path passes validation."""
        provider = InMemorySecretProvider()
        wrapped = ValidatingProviderWrapper(provider)

        # Should not raise
        wrapped.set("valid/path", "value")

    def test_reject_expired(self):
        """Test expired secrets are rejected."""
        from datetime import datetime, timedelta, timezone

        from packages.enterprise.secrets.exceptions import SecretExpiredError

        provider = InMemorySecretProvider()
        wrapped = ValidatingProviderWrapper(provider, reject_expired=True)

        # Set secret with past expiration
        past = datetime.now(timezone.utc) - timedelta(days=1)
        provider.set("test/secret", "value", expires_at=past)

        # Should raise SecretExpiredError
        with pytest.raises(SecretExpiredError):
            wrapped.get("test/secret")

    def test_accept_expired_when_disabled(self):
        """Test expired secrets are accepted when reject_expired=False."""
        from datetime import datetime, timedelta, timezone

        provider = InMemorySecretProvider()
        wrapped = ValidatingProviderWrapper(provider, reject_expired=False)

        # Set secret with past expiration
        past = datetime.now(timezone.utc) - timedelta(days=1)
        provider.set("test/secret", "value", expires_at=past)

        # Should not raise
        result = wrapped.get("test/secret")
        assert result is not None


class TestCreateWrappedProvider:
    """Tests for create_wrapped_provider factory."""

    def test_with_namespace(self):
        """Test creating with namespace."""
        provider = InMemorySecretProvider()
        wrapped = create_wrapped_provider(
            provider,
            namespace="prod",
        )

        wrapped.set("test", "value")
        assert provider.get("prod/test") is not None

    def test_with_hooks(self):
        """Test creating with hooks."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()
        wrapped = create_wrapped_provider(
            provider,
            hooks=[hook],
        )

        wrapped.set("test", "value")
        assert len(hook.before_set_calls) == 1

    def test_combined_wrappers(self):
        """Test combining namespace and hooks."""
        provider = InMemorySecretProvider()
        hook = MockSecretHook()

        wrapped = create_wrapped_provider(
            provider,
            namespace="prod",
            hooks=[hook],
        )

        wrapped.set("test", "value")

        # Check namespace
        assert provider.get("prod/test") is not None

        # Check hooks
        assert len(hook.before_set_calls) == 1
