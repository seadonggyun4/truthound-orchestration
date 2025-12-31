"""Tests for Enterprise Engine Base module.

This module tests the base abstractions including:
- EnterpriseEngineConfig
- EnterpriseEngineAdapter
- Protocols (RuleTranslator, ResultConverter, ConnectionManager)
- Base implementations
- Exception classes
"""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from packages.enterprise.engines.base import (
    # Config
    EnterpriseEngineConfig,
    DEFAULT_ENTERPRISE_CONFIG,
    PRODUCTION_ENTERPRISE_CONFIG,
    DEVELOPMENT_ENTERPRISE_CONFIG,
    HIGH_THROUGHPUT_CONFIG,
    # Enums
    AuthType,
    ConnectionMode,
    DataTransferMode,
    # Base implementations
    BaseRuleTranslator,
    BaseResultConverter,
    BaseConnectionManager,
    # Exceptions
    EnterpriseEngineError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    VendorSDKError,
    RuleTranslationError,
)


# =============================================================================
# EnterpriseEngineConfig Tests
# =============================================================================


class TestEnterpriseEngineConfig:
    """Tests for EnterpriseEngineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EnterpriseEngineConfig()

        assert config.auto_start is False
        assert config.auto_stop is True
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.auth_type == AuthType.NONE
        assert config.verify_ssl is True
        assert config.pool_size == 5

    def test_frozen_immutability(self):
        """Test that config is immutable."""
        config = EnterpriseEngineConfig()

        with pytest.raises(FrozenInstanceError):
            config.timeout_seconds = 60.0

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = EnterpriseEngineConfig(
            api_endpoint="https://api.example.com",
            api_key="secret-key",
            auth_type=AuthType.API_KEY,
            timeout_seconds=60.0,
            max_retries=5,
        )

        assert config.api_endpoint == "https://api.example.com"
        assert config.api_key == "secret-key"
        assert config.auth_type == AuthType.API_KEY
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 5

    def test_with_api_endpoint(self):
        """Test builder method for API endpoint."""
        config = EnterpriseEngineConfig()
        new_config = config.with_api_endpoint("https://new.api.com")

        assert new_config.api_endpoint == "https://new.api.com"
        assert config.api_endpoint == ""  # Original unchanged

    def test_with_api_key(self):
        """Test builder method for API key auth."""
        config = EnterpriseEngineConfig()
        new_config = config.with_api_key("my-api-key")

        assert new_config.api_key == "my-api-key"
        assert new_config.auth_type == AuthType.API_KEY

    def test_with_basic_auth(self):
        """Test builder method for basic auth."""
        config = EnterpriseEngineConfig()
        new_config = config.with_basic_auth("user", "pass")

        assert new_config.username == "user"
        assert new_config.password == "pass"
        assert new_config.auth_type == AuthType.BASIC

    def test_with_oauth2(self):
        """Test builder method for OAuth2."""
        config = EnterpriseEngineConfig()
        new_config = config.with_oauth2(
            client_id="client",
            client_secret="secret",
            token_url="https://auth.example.com/token",
        )

        # OAuth2 config is stored in vendor_options
        assert new_config.auth_type == AuthType.OAUTH2
        assert "oauth" in new_config.vendor_options
        assert new_config.vendor_options["oauth"]["client_id"] == "client"

    def test_with_timeout(self):
        """Test builder method for timeouts."""
        config = EnterpriseEngineConfig()
        new_config = config.with_timeout(
            timeout_seconds=120.0,
            connect_timeout_seconds=30.0,
        )

        assert new_config.timeout_seconds == 120.0
        assert new_config.connect_timeout_seconds == 30.0

    def test_with_retry(self):
        """Test builder method for retry settings."""
        config = EnterpriseEngineConfig()
        new_config = config.with_retry(
            max_retries=10,
            delay_seconds=5.0,
        )

        assert new_config.max_retries == 10
        assert new_config.retry_delay_seconds == 5.0

    def test_with_pool(self):
        """Test builder method for connection pool."""
        config = EnterpriseEngineConfig()
        new_config = config.with_pool(
            size=20,
            timeout_seconds=60.0,
        )

        assert new_config.pool_size == 20
        assert new_config.pool_timeout_seconds == 60.0

    def test_with_ssl(self):
        """Test builder method for SSL settings."""
        config = EnterpriseEngineConfig()
        new_config = config.with_ssl(
            verify=False,
            proxy_url="http://proxy.example.com:8080",
        )

        assert new_config.verify_ssl is False
        assert new_config.proxy_url == "http://proxy.example.com:8080"

    def test_validation_negative_timeout(self):
        """Test validation rejects negative timeout."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            EnterpriseEngineConfig(timeout_seconds=-1.0)

    def test_validation_negative_retries(self):
        """Test validation rejects negative retries."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            EnterpriseEngineConfig(max_retries=-1)

    def test_validation_negative_pool_size(self):
        """Test validation rejects non-positive pool size."""
        with pytest.raises(ValueError, match="pool_size must be at least 1"):
            EnterpriseEngineConfig(pool_size=0)

    def test_builder_chaining(self):
        """Test chaining multiple builder methods."""
        config = (
            EnterpriseEngineConfig()
            .with_api_endpoint("https://api.example.com")
            .with_api_key("secret")
            .with_timeout(timeout_seconds=60.0)
            .with_retry(max_retries=5)
            .with_pool(size=10)
        )

        assert config.api_endpoint == "https://api.example.com"
        assert config.api_key == "secret"
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 5
        assert config.pool_size == 10


class TestPresetConfigurations:
    """Tests for preset configuration constants."""

    def test_default_config(self):
        """Test default configuration preset."""
        assert DEFAULT_ENTERPRISE_CONFIG.auto_start is False
        assert DEFAULT_ENTERPRISE_CONFIG.verify_ssl is True

    def test_production_config(self):
        """Test production configuration preset."""
        assert PRODUCTION_ENTERPRISE_CONFIG.auto_start is True
        assert PRODUCTION_ENTERPRISE_CONFIG.health_check_enabled is True
        assert PRODUCTION_ENTERPRISE_CONFIG.verify_ssl is True
        assert PRODUCTION_ENTERPRISE_CONFIG.max_retries >= 3

    def test_development_config(self):
        """Test development configuration preset."""
        assert DEVELOPMENT_ENTERPRISE_CONFIG.verify_ssl is False
        assert DEVELOPMENT_ENTERPRISE_CONFIG.health_check_enabled is False

    def test_high_throughput_config(self):
        """Test high throughput configuration preset."""
        assert HIGH_THROUGHPUT_CONFIG.pool_size >= 10
        # HIGH_THROUGHPUT_CONFIG doesn't have max_overflow - it has pool_size
        assert HIGH_THROUGHPUT_CONFIG.batch_size >= 10000


# =============================================================================
# Enum Tests
# =============================================================================


class TestAuthType:
    """Tests for AuthType enum."""

    def test_auth_types(self):
        """Test all auth types are defined."""
        assert AuthType.NONE is not None
        assert AuthType.API_KEY is not None
        assert AuthType.BASIC is not None
        assert AuthType.OAUTH2 is not None
        assert AuthType.JWT is not None
        assert AuthType.CERTIFICATE is not None


class TestConnectionMode:
    """Tests for ConnectionMode enum."""

    def test_connection_modes(self):
        """Test all connection modes are defined."""
        assert ConnectionMode.REST is not None
        assert ConnectionMode.SOAP is not None
        assert ConnectionMode.JDBC is not None
        assert ConnectionMode.NATIVE is not None


class TestDataTransferMode:
    """Tests for DataTransferMode enum."""

    def test_data_transfer_modes(self):
        """Test all data transfer modes are defined."""
        assert DataTransferMode.INLINE is not None
        assert DataTransferMode.REFERENCE is not None
        assert DataTransferMode.STREAMING is not None


# =============================================================================
# Exception Tests
# =============================================================================


class TestEnterpriseEngineError:
    """Tests for EnterpriseEngineError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = EnterpriseEngineError("Test error", engine_name="test")

        assert "Test error" in str(error)
        assert error.engine_name == "test"

    def test_error_with_details(self):
        """Test error with details."""
        error = EnterpriseEngineError(
            "Wrapper error",
            details={"engine_name": "test", "operation": "check"},
        )

        assert "Wrapper error" in str(error)
        assert error.details["operation"] == "check"


class TestConnectionError:
    """Tests for ConnectionError."""

    def test_connection_error(self):
        """Test connection error creation."""
        error = ConnectionError(
            "Failed to connect",
            engine_name="informatica",
            details={"endpoint": "https://api.example.com"},
        )

        assert "Failed to connect" in str(error)
        assert error.engine_name == "informatica"
        assert error.details["endpoint"] == "https://api.example.com"


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_authentication_error(self):
        """Test authentication error creation."""
        error = AuthenticationError(
            "Invalid credentials",
            engine_name="talend",
            details={"auth_type": "API_KEY"},
        )

        assert "Invalid credentials" in str(error)
        assert error.details["auth_type"] == "API_KEY"


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error(self):
        """Test rate limit error creation."""
        error = RateLimitError(
            "Rate limit exceeded",
            engine_name="informatica",
            retry_after_seconds=60,
        )

        assert "Rate limit exceeded" in str(error)
        assert error.retry_after_seconds == 60


class TestVendorSDKError:
    """Tests for VendorSDKError."""

    def test_vendor_sdk_error(self):
        """Test vendor SDK error creation."""
        error = VendorSDKError(
            "SDK not installed",
            engine_name="informatica",
            details={"sdk": "informatica-sdk"},
        )

        assert "SDK not installed" in str(error)
        assert error.engine_name == "informatica"


class TestRuleTranslationError:
    """Tests for RuleTranslationError."""

    def test_rule_translation_error(self):
        """Test rule translation error creation."""
        error = RuleTranslationError(
            "Unknown rule type",
            engine_name="talend",
            rule_type="invalid_rule",
        )

        assert "Unknown rule type" in str(error)
        assert error.rule_type == "invalid_rule"


# =============================================================================
# BaseRuleTranslator Tests
# =============================================================================


class TestBaseRuleTranslator:
    """Tests for BaseRuleTranslator."""

    def test_get_supported_rule_types(self):
        """Test getting supported rule types."""

        class TestTranslator(BaseRuleTranslator):
            def _get_rule_mapping(self) -> dict[str, str]:
                return {
                    "not_null": "NULL_CHECK",
                    "unique": "UNIQUE_CHECK",
                }

        translator = TestTranslator()
        types = translator.get_supported_rule_types()

        assert "not_null" in types
        assert "unique" in types

    def test_translate_single_rule(self):
        """Test translating a single rule."""

        class TestTranslator(BaseRuleTranslator):
            def _get_rule_mapping(self) -> dict[str, str]:
                return {"not_null": "NULL_CHECK"}

            def _translate_rule_params(
                self,
                rule_type: str,
                rule: dict,
            ) -> dict:
                return {"column": rule.get("column")}

        translator = TestTranslator()
        rule = {"type": "not_null", "column": "id"}

        translated = translator.translate(rule)

        assert translated["type"] == "NULL_CHECK"
        assert translated["column"] == "id"

    def test_translate_batch(self):
        """Test translating multiple rules."""

        class TestTranslator(BaseRuleTranslator):
            def _get_rule_mapping(self) -> dict[str, str]:
                return {"not_null": "NULL_CHECK"}

            def _translate_rule_params(
                self,
                rule_type: str,
                rule: dict,
            ) -> dict:
                return {"column": rule.get("column")}

        translator = TestTranslator()
        rules = [{"type": "not_null", "column": "id"}]

        translated = translator.translate_batch(rules)

        assert len(translated) == 1
        assert translated[0]["type"] == "NULL_CHECK"
        assert translated[0]["column"] == "id"

    def test_translate_unknown_rule_raises(self):
        """Test that unknown rule type raises error."""

        class TestTranslator(BaseRuleTranslator):
            def _get_rule_mapping(self) -> dict[str, str]:
                return {"not_null": "NULL_CHECK"}

        translator = TestTranslator()
        rule = {"type": "unknown_rule", "column": "id"}

        with pytest.raises(RuleTranslationError, match="unknown_rule"):
            translator.translate(rule)


# =============================================================================
# BaseResultConverter Tests
# =============================================================================


class TestBaseResultConverter:
    """Tests for BaseResultConverter."""

    def test_convert_check_result_success(self):
        """Test converting successful check result."""
        import time

        class TestConverter(BaseResultConverter):
            def _extract_check_items(self, vendor_result):
                return [
                    {"rule_name": "test", "passed": True, "failed_count": 0},
                ]

        converter = TestConverter()
        start_time = time.perf_counter()

        # Mock vendor result
        vendor_result = {"status": "success"}

        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        from common.base import CheckStatus
        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 1
        assert result.failed_count == 0

    def test_convert_check_result_failure(self):
        """Test converting failed check result."""
        import time

        class TestConverter(BaseResultConverter):
            def _extract_check_items(self, vendor_result):
                return [
                    {
                        "rule_name": "not_null",
                        "column": "id",
                        "passed": False,
                        "failed_count": 5,
                        "message": "Null values found",
                        "severity": "high",
                    },
                ]

        converter = TestConverter()
        start_time = time.perf_counter()

        result = converter.convert_check_result(
            {"status": "failed"},
            start_time=start_time,
        )

        from common.base import CheckStatus
        assert result.status == CheckStatus.FAILED
        assert result.failed_count == 1  # 1 rule failed, not 5 records
        assert len(result.failures) == 1
        assert result.failures[0].column == "id"
        assert result.failures[0].failed_count == 5  # 5 records failed


# =============================================================================
# BaseConnectionManager Tests
# =============================================================================


class TestBaseConnectionManager:
    """Tests for BaseConnectionManager."""

    def test_connection_lifecycle(self):
        """Test connection connect/disconnect lifecycle."""

        class TestConnectionManager(BaseConnectionManager):
            def _do_connect(self):
                return {"connected": True}

            def _do_disconnect(self):
                pass

            def _do_health_check(self) -> bool:
                return self._connected

        config = EnterpriseEngineConfig()
        manager = TestConnectionManager(config, name="test")

        # Initially not connected
        assert manager.is_connected() is False

        # Connect
        manager.connect()
        assert manager.is_connected() is True
        # Connection object is stored internally
        conn = manager.get_connection()
        assert conn == {"connected": True}

        # Disconnect
        manager.disconnect()
        assert manager.is_connected() is False

    def test_health_check(self):
        """Test health check."""
        from common.health import HealthStatus

        class TestConnectionManager(BaseConnectionManager):
            def _do_connect(self):
                return {"connected": True}

            def _do_disconnect(self):
                pass

            def _do_health_check(self) -> bool:
                return self._connected

        config = EnterpriseEngineConfig()
        manager = TestConnectionManager(config, name="test")

        # Health check when not connected
        result = manager.health_check()
        assert result.status == HealthStatus.UNHEALTHY

        # Health check when connected
        manager.connect()
        result = manager.health_check()
        assert result.status == HealthStatus.HEALTHY

    def test_context_manager(self):
        """Test context manager protocol."""

        class TestConnectionManager(BaseConnectionManager):
            def _do_connect(self):
                return {"connected": True}

            def _do_disconnect(self):
                pass

            def _do_health_check(self) -> bool:
                return self._connected

        config = EnterpriseEngineConfig()
        manager = TestConnectionManager(config, name="test")

        with manager:
            assert manager.is_connected() is True

        assert manager.is_connected() is False
