"""Tests for secret provider registry."""

from __future__ import annotations

import pytest

from packages.enterprise.secrets.backends import InMemorySecretProvider
from packages.enterprise.secrets.exceptions import (
    ProviderNotFoundError,
    SecretConfigurationError,
)
from packages.enterprise.secrets.registry import (
    SecretProviderRegistry,
    delete_secret,
    get_secret,
    get_secret_registry,
    list_secrets,
    reset_secret_registry,
    secret_exists,
    set_secret,
)


class TestSecretProviderRegistry:
    """Tests for SecretProviderRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_secret_registry()

    def teardown_method(self):
        """Reset registry after each test."""
        reset_secret_registry()

    def test_singleton(self):
        """Test singleton pattern."""
        registry1 = get_secret_registry()
        registry2 = get_secret_registry()
        assert registry1 is registry2

    def test_register_provider(self):
        """Test registering a provider."""
        registry = get_secret_registry()
        provider = InMemorySecretProvider()

        registry.register("memory", provider)

        assert registry.exists("memory")
        assert registry.get("memory") is provider

    def test_register_sets_default(self):
        """Test first registered provider becomes default."""
        registry = get_secret_registry()
        provider = InMemorySecretProvider()

        registry.register("memory", provider)

        assert registry.default_provider_name == "memory"

    def test_register_with_set_default(self):
        """Test registering with set_default flag."""
        registry = get_secret_registry()
        provider1 = InMemorySecretProvider()
        provider2 = InMemorySecretProvider()

        registry.register("first", provider1)
        registry.register("second", provider2, set_default=True)

        assert registry.default_provider_name == "second"

    def test_register_empty_name(self):
        """Test registering with empty name fails."""
        registry = get_secret_registry()
        provider = InMemorySecretProvider()

        with pytest.raises(SecretConfigurationError):
            registry.register("", provider)

    def test_unregister(self):
        """Test unregistering a provider."""
        registry = get_secret_registry()
        provider = InMemorySecretProvider()

        registry.register("memory", provider)
        assert registry.unregister("memory") is True
        assert not registry.exists("memory")

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent provider."""
        registry = get_secret_registry()
        assert registry.unregister("nonexistent") is False

    def test_get_nonexistent(self):
        """Test getting nonexistent provider."""
        registry = get_secret_registry()

        with pytest.raises(ProviderNotFoundError):
            registry.get("nonexistent")

    def test_get_default(self):
        """Test getting default provider."""
        registry = get_secret_registry()
        provider = InMemorySecretProvider()

        registry.register("memory", provider)

        assert registry.get() is provider

    def test_get_no_providers(self):
        """Test getting when no providers registered."""
        registry = get_secret_registry()

        with pytest.raises(ProviderNotFoundError):
            registry.get()

    def test_set_default(self):
        """Test setting default provider."""
        registry = get_secret_registry()
        provider1 = InMemorySecretProvider()
        provider2 = InMemorySecretProvider()

        registry.register("first", provider1)
        registry.register("second", provider2)

        registry.set_default("second")
        assert registry.default_provider_name == "second"

    def test_set_default_nonexistent(self):
        """Test setting nonexistent provider as default."""
        registry = get_secret_registry()

        with pytest.raises(ProviderNotFoundError):
            registry.set_default("nonexistent")

    def test_list_providers(self):
        """Test listing providers."""
        registry = get_secret_registry()
        registry.register("first", InMemorySecretProvider())
        registry.register("second", InMemorySecretProvider())

        providers = registry.list_providers()
        assert "first" in providers
        assert "second" in providers

    def test_clear(self):
        """Test clearing all providers."""
        registry = get_secret_registry()
        registry.register("first", InMemorySecretProvider())
        registry.register("second", InMemorySecretProvider())

        registry.clear()

        assert len(registry.list_providers()) == 0
        assert registry.default_provider_name is None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Set up registry with a provider."""
        reset_secret_registry()
        registry = get_secret_registry()
        registry.register("memory", InMemorySecretProvider())

    def teardown_method(self):
        """Reset registry after each test."""
        reset_secret_registry()

    def test_get_secret(self):
        """Test get_secret function."""
        set_secret("test/path", "value")
        secret = get_secret("test/path")

        assert secret is not None
        assert secret.value == "value"

    def test_set_secret(self):
        """Test set_secret function."""
        result = set_secret("test/path", "value")

        assert result.value == "value"

    def test_delete_secret(self):
        """Test delete_secret function."""
        set_secret("test/path", "value")
        result = delete_secret("test/path")

        assert result is True
        assert get_secret("test/path") is None

    def test_secret_exists(self):
        """Test secret_exists function."""
        set_secret("test/path", "value")

        assert secret_exists("test/path") is True
        assert secret_exists("nonexistent") is False

    def test_list_secrets(self):
        """Test list_secrets function."""
        set_secret("a/secret", "value1")
        set_secret("b/secret", "value2")

        secrets = list_secrets()
        paths = [s.path for s in secrets]

        assert "a/secret" in paths
        assert "b/secret" in paths

    def test_list_secrets_with_prefix(self):
        """Test list_secrets with prefix."""
        set_secret("a/secret", "value1")
        set_secret("b/secret", "value2")
        set_secret("a/other", "value3")

        secrets = list_secrets("a/")
        paths = [s.path for s in secrets]

        assert len(paths) == 2
        assert "a/secret" in paths
        assert "a/other" in paths
