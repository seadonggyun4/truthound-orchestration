"""Tests for base types and protocols."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from packages.enterprise.secrets.base import (
    HealthCheckResult,
    HealthStatus,
    SecretMetadata,
    SecretType,
    SecretValue,
    SecretVersion,
)


class TestSecretType:
    """Tests for SecretType enum."""

    def test_string_type(self):
        """Test STRING type exists."""
        assert SecretType.STRING is not None
        assert SecretType.STRING.name == "STRING"

    def test_binary_type(self):
        """Test BINARY type exists."""
        assert SecretType.BINARY is not None
        assert SecretType.BINARY.name == "BINARY"

    def test_json_type(self):
        """Test JSON type exists."""
        assert SecretType.JSON is not None
        assert SecretType.JSON.name == "JSON"

    def test_certificate_type(self):
        """Test CERTIFICATE type exists."""
        assert SecretType.CERTIFICATE is not None
        assert SecretType.CERTIFICATE.name == "CERTIFICATE"

    def test_api_key_type(self):
        """Test API_KEY type exists."""
        assert SecretType.API_KEY is not None
        assert SecretType.API_KEY.name == "API_KEY"


class TestSecretValue:
    """Tests for SecretValue dataclass."""

    def test_creation_minimal(self):
        """Test creation with minimal fields."""
        secret = SecretValue(
            value="test-secret",
            version="1",
            secret_type=SecretType.STRING,
        )
        assert secret.value == "test-secret"
        assert secret.version == "1"
        assert secret.secret_type == SecretType.STRING
        # created_at has default factory, so it's set automatically
        assert secret.created_at is not None
        assert secret.expires_at is None
        assert secret.metadata == {}

    def test_creation_full(self):
        """Test creation with all fields."""
        now = datetime.now(timezone.utc)
        secret = SecretValue(
            value="test-secret",
            version="2",
            created_at=now,
            expires_at=now,
            secret_type=SecretType.JSON,
            metadata={"key": "value"},
        )
        assert secret.value == "test-secret"
        assert secret.version == "2"
        assert secret.created_at == now
        assert secret.expires_at == now
        assert secret.secret_type == SecretType.JSON
        assert secret.metadata == {"key": "value"}

    def test_is_expired_no_expiry(self):
        """Test is_expired with no expiry."""
        secret = SecretValue(
            value="test",
            version="1",
            secret_type=SecretType.STRING,
        )
        assert not secret.is_expired

    def test_is_expired_future(self):
        """Test is_expired with future expiry."""
        from datetime import timedelta

        future = datetime.now(timezone.utc) + timedelta(days=1)
        secret = SecretValue(
            value="test",
            version="1",
            secret_type=SecretType.STRING,
            expires_at=future,
        )
        assert not secret.is_expired

    def test_is_expired_past(self):
        """Test is_expired with past expiry."""
        from datetime import timedelta

        past = datetime.now(timezone.utc) - timedelta(days=1)
        secret = SecretValue(
            value="test",
            version="1",
            secret_type=SecretType.STRING,
            expires_at=past,
        )
        assert secret.is_expired

    def test_immutability(self):
        """Test that SecretValue is immutable."""
        secret = SecretValue(
            value="test",
            version="1",
            secret_type=SecretType.STRING,
        )
        with pytest.raises(AttributeError):
            secret.value = "modified"


class TestSecretMetadata:
    """Tests for SecretMetadata dataclass."""

    def test_creation_minimal(self):
        """Test creation with minimal fields."""
        meta = SecretMetadata(
            path="test/secret",
            version="1",
            secret_type=SecretType.STRING,
        )
        assert meta.path == "test/secret"
        assert meta.version == "1"
        assert meta.secret_type == SecretType.STRING

    def test_creation_full(self):
        """Test creation with all fields."""
        now = datetime.now(timezone.utc)
        meta = SecretMetadata(
            path="test/secret",
            version="2",
            created_at=now,
            updated_at=now,
            expires_at=now,
            secret_type=SecretType.JSON,
            tags=frozenset(["prod"]),
            description="Test secret",
        )
        assert meta.path == "test/secret"
        assert meta.version == "2"
        assert meta.created_at == now
        assert meta.updated_at == now
        assert meta.expires_at == now
        assert meta.tags == frozenset(["prod"])
        assert meta.description == "Test secret"


class TestSecretVersion:
    """Tests for SecretVersion dataclass."""

    def test_creation(self):
        """Test creation."""
        now = datetime.now(timezone.utc)
        version = SecretVersion(
            version="v1",
            created_at=now,
            is_current=True,
        )
        assert version.version == "v1"
        assert version.created_at == now
        assert version.is_current is True


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_healthy(self):
        """Test HEALTHY status."""
        assert HealthStatus.HEALTHY is not None
        assert HealthStatus.HEALTHY.name == "HEALTHY"

    def test_unhealthy(self):
        """Test UNHEALTHY status."""
        assert HealthStatus.UNHEALTHY is not None
        assert HealthStatus.UNHEALTHY.name == "UNHEALTHY"

    def test_degraded(self):
        """Test DEGRADED status."""
        assert HealthStatus.DEGRADED is not None
        assert HealthStatus.DEGRADED.name == "DEGRADED"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_creation_minimal(self):
        """Test creation with minimal fields."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="OK",
        )
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "OK"
        assert result.details == {}
        # latency_ms has a default value of 0.0
        assert result.latency_ms == 0.0

    def test_creation_full(self):
        """Test creation with all fields."""
        result = HealthCheckResult(
            status=HealthStatus.DEGRADED,
            message="Slow response",
            details={"latency": "high"},
            latency_ms=500.0,
        )
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "Slow response"
        assert result.details == {"latency": "high"}
        assert result.latency_ms == 500.0
