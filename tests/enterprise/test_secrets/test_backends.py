"""Tests for secret backends."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from packages.enterprise.secrets.backends import InMemorySecretProvider
from packages.enterprise.secrets.backends.env import EnvironmentSecretProvider
from packages.enterprise.secrets.base import HealthStatus, SecretType


class TestInMemorySecretProvider:
    """Tests for InMemorySecretProvider."""

    def test_set_and_get(self):
        """Test basic set and get."""
        provider = InMemorySecretProvider()
        result = provider.set("test/secret", "value123")

        assert result.value == "value123"
        assert result.version == "1"

        retrieved = provider.get("test/secret")
        assert retrieved is not None
        assert retrieved.value == "value123"

    def test_get_nonexistent(self):
        """Test get for nonexistent secret."""
        provider = InMemorySecretProvider()
        result = provider.get("nonexistent")
        assert result is None

    def test_delete(self):
        """Test delete."""
        provider = InMemorySecretProvider()
        provider.set("test/secret", "value")

        assert provider.delete("test/secret") is True
        assert provider.get("test/secret") is None

    def test_delete_nonexistent(self):
        """Test delete for nonexistent secret."""
        provider = InMemorySecretProvider()
        assert provider.delete("nonexistent") is False

    def test_exists(self):
        """Test exists."""
        provider = InMemorySecretProvider()
        provider.set("test/secret", "value")

        assert provider.exists("test/secret") is True
        assert provider.exists("nonexistent") is False

    def test_list_all(self):
        """Test list all secrets."""
        provider = InMemorySecretProvider()
        provider.set("a/secret", "value1")
        provider.set("b/secret", "value2")
        provider.set("a/other", "value3")

        secrets = provider.list()
        assert len(secrets) == 3

    def test_list_with_prefix(self):
        """Test list with prefix."""
        provider = InMemorySecretProvider()
        provider.set("a/secret", "value1")
        provider.set("b/secret", "value2")
        provider.set("a/other", "value3")

        secrets = provider.list("a/")
        assert len(secrets) == 2

    def test_list_with_limit_and_offset(self):
        """Test list with limit and offset."""
        provider = InMemorySecretProvider()
        for i in range(10):
            provider.set(f"secret/{i}", f"value{i}")

        # Test limit
        secrets = provider.list(limit=5)
        assert len(secrets) == 5

        # Test offset
        secrets = provider.list(offset=5)
        assert len(secrets) == 5

        # Test both
        secrets = provider.list(limit=3, offset=3)
        assert len(secrets) == 3

    def test_versioning(self):
        """Test version tracking."""
        provider = InMemorySecretProvider()
        result1 = provider.set("test/secret", "value1")
        result2 = provider.set("test/secret", "value2")
        result3 = provider.set("test/secret", "value3")

        assert result1.version == "1"
        assert result2.version == "2"
        assert result3.version == "3"

        # Get latest
        latest = provider.get("test/secret")
        assert latest.value == "value3"
        assert latest.version == "3"

        # Get specific version
        v1 = provider.get("test/secret", version="1")
        assert v1.value == "value1"
        assert v1.version == "1"

    def test_list_versions(self):
        """Test listing all versions."""
        provider = InMemorySecretProvider()
        provider.set("test/secret", "value1")
        provider.set("test/secret", "value2")
        provider.set("test/secret", "value3")

        versions = provider.list_versions("test/secret")
        assert len(versions) == 3

    def test_health_check(self):
        """Test health check."""
        provider = InMemorySecretProvider()
        result = provider.health_check()

        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None

    def test_metadata(self):
        """Test storing metadata."""
        provider = InMemorySecretProvider()
        result = provider.set(
            "test/secret",
            "value",
            metadata={"key": "value", "tags": ["prod"]},
        )

        assert "key" in result.metadata
        assert result.metadata["key"] == "value"

    def test_expiration(self):
        """Test expiration handling."""
        provider = InMemorySecretProvider()
        past = datetime.now(timezone.utc) - timedelta(days=1)

        provider.set("test/secret", "value", expires_at=past)

        retrieved = provider.get("test/secret")
        assert retrieved is not None
        assert retrieved.is_expired is True


class TestEnvironmentSecretProvider:
    """Tests for EnvironmentSecretProvider."""

    def test_get_existing_var(self):
        """Test getting an existing environment variable."""
        # Set up
        os.environ["SECRET_TEST_VAR"] = "test-value"
        try:
            provider = EnvironmentSecretProvider(prefix="SECRET_")
            secret = provider.get("TEST_VAR")

            assert secret is not None
            assert secret.value == "test-value"
        finally:
            del os.environ["SECRET_TEST_VAR"]

    def test_get_nonexistent_var(self):
        """Test getting a nonexistent variable."""
        provider = EnvironmentSecretProvider(prefix="SECRET_")
        secret = provider.get("NONEXISTENT_VAR")
        assert secret is None

    def test_set_with_allow(self):
        """Test setting when allowed."""
        provider = EnvironmentSecretProvider(prefix="SECRET_", allow_set=True)
        try:
            result = provider.set("NEW_VAR", "new-value")

            assert result.value == "new-value"
            assert os.environ.get("SECRET_NEW_VAR") == "new-value"
        finally:
            if "SECRET_NEW_VAR" in os.environ:
                del os.environ["SECRET_NEW_VAR"]

    def test_set_without_allow(self):
        """Test setting when not allowed."""
        provider = EnvironmentSecretProvider(prefix="SECRET_", allow_set=False)
        with pytest.raises(PermissionError):
            provider.set("NEW_VAR", "new-value")

    def test_delete_with_allow(self):
        """Test deleting when allowed."""
        os.environ["SECRET_TO_DELETE"] = "value"
        try:
            provider = EnvironmentSecretProvider(prefix="SECRET_", allow_set=True)
            result = provider.delete("TO_DELETE")

            assert result is True
            assert "SECRET_TO_DELETE" not in os.environ
        finally:
            if "SECRET_TO_DELETE" in os.environ:
                del os.environ["SECRET_TO_DELETE"]

    def test_list(self):
        """Test listing environment variables."""
        os.environ["SECRET_VAR1"] = "value1"
        os.environ["SECRET_VAR2"] = "value2"
        try:
            provider = EnvironmentSecretProvider(prefix="SECRET_")
            secrets = provider.list()

            paths = [s.path for s in secrets]
            assert "VAR1" in paths
            assert "VAR2" in paths
        finally:
            del os.environ["SECRET_VAR1"]
            del os.environ["SECRET_VAR2"]

    def test_exists(self):
        """Test exists check."""
        os.environ["SECRET_EXISTS_VAR"] = "value"
        try:
            provider = EnvironmentSecretProvider(prefix="SECRET_")
            assert provider.exists("EXISTS_VAR") is True
            assert provider.exists("NOT_EXISTS") is False
        finally:
            del os.environ["SECRET_EXISTS_VAR"]

    def test_health_check(self):
        """Test health check."""
        provider = EnvironmentSecretProvider(prefix="SECRET_")
        result = provider.health_check()

        assert result.status == HealthStatus.HEALTHY


class TestFileSecretProvider:
    """Tests for FileSecretProvider."""

    def test_set_and_get(self):
        """Test basic set and get."""
        from packages.enterprise.secrets.backends.file import FileSecretProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretProvider(
                directory=tmpdir,
                encryption_key=None,  # No encryption for simple test
            )

            result = provider.set("test/secret", "value123")
            assert result.value == "value123"

            retrieved = provider.get("test/secret")
            assert retrieved is not None
            assert retrieved.value == "value123"

    def test_delete(self):
        """Test delete."""
        from packages.enterprise.secrets.backends.file import FileSecretProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretProvider(directory=tmpdir)

            provider.set("test/secret", "value")
            assert provider.delete("test/secret") is True
            assert provider.get("test/secret") is None

    def test_list(self):
        """Test listing secrets."""
        from packages.enterprise.secrets.backends.file import FileSecretProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretProvider(directory=tmpdir)

            provider.set("a/secret", "value1")
            provider.set("b/secret", "value2")

            secrets = provider.list()
            assert len(secrets) >= 2

    def test_versioning(self):
        """Test version tracking."""
        from packages.enterprise.secrets.backends.file import FileSecretProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretProvider(directory=tmpdir)

            provider.set("test/secret", "value1")
            provider.set("test/secret", "value2")

            # Get latest
            latest = provider.get("test/secret")
            assert latest.value == "value2"

            # Get specific version
            v1 = provider.get("test/secret", version="1")
            assert v1.value == "value1"

    def test_health_check(self):
        """Test health check."""
        from packages.enterprise.secrets.backends.file import FileSecretProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FileSecretProvider(directory=tmpdir)
            result = provider.health_check()

            assert result.status == HealthStatus.HEALTHY
