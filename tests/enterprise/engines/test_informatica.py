"""Tests for Informatica Data Quality Adapter.

This module tests the Informatica adapter including:
- InformaticaConfig
- InformaticaAdapter
- InformaticaRuleTranslator
- InformaticaResultConverter
- InformaticaConnectionManager
- Factory function
"""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch, MagicMock

from packages.enterprise.engines.informatica import (
    # Main classes
    InformaticaAdapter,
    InformaticaConfig,
    # Components
    InformaticaRuleTranslator,
    InformaticaResultConverter,
    InformaticaConnectionManager,
    # Factory
    create_informatica_adapter,
    # Presets
    DEFAULT_INFORMATICA_CONFIG,
    PRODUCTION_INFORMATICA_CONFIG,
    DEVELOPMENT_INFORMATICA_CONFIG,
)
from packages.enterprise.engines.base import (
    AuthType,
    EnterpriseEngineError,
    VendorSDKError,
)
from common.base import CheckStatus, Severity


# =============================================================================
# InformaticaConfig Tests
# =============================================================================


class TestInformaticaConfig:
    """Tests for InformaticaConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InformaticaConfig()

        assert config.domain == "Default"
        assert config.project == ""
        assert config.scorecard_name == ""
        assert config.async_execution is True
        assert config.poll_interval_seconds == 2.0

    def test_frozen_immutability(self):
        """Test that config is immutable."""
        config = InformaticaConfig()

        with pytest.raises(FrozenInstanceError):
            config.domain = "new_domain"

    def test_with_domain(self):
        """Test builder method for domain."""
        config = InformaticaConfig()
        new_config = config.with_domain("Production")

        assert new_config.domain == "Production"
        assert config.domain == "Default"  # Original unchanged

    def test_with_project(self):
        """Test builder method for project."""
        config = InformaticaConfig()
        new_config = config.with_project("DQ_Project", folder="/Profiles")

        assert new_config.project == "DQ_Project"
        assert new_config.folder == "/Profiles"

    def test_with_scorecard(self):
        """Test builder method for scorecard."""
        config = InformaticaConfig()
        new_config = config.with_scorecard("MainScorecard")

        assert new_config.scorecard_name == "MainScorecard"

    def test_with_async_execution(self):
        """Test builder method for async execution."""
        config = InformaticaConfig()
        new_config = config.with_async_execution(
            enabled=True,
            poll_interval=10.0,
            max_attempts=600,
        )

        assert new_config.async_execution is True
        assert new_config.poll_interval_seconds == 10.0
        assert new_config.max_poll_attempts == 600

    def test_builder_chaining(self):
        """Test chaining multiple builder methods."""
        config = (
            InformaticaConfig()
            .with_api_endpoint("https://idq.example.com/api/v2")
            .with_api_key("secret")
            .with_domain("Production")
            .with_project("DataQuality")
            .with_scorecard("CustomerScorecard")
        )

        assert config.api_endpoint == "https://idq.example.com/api/v2"
        assert config.api_key == "secret"
        assert config.domain == "Production"
        assert config.project == "DataQuality"
        assert config.scorecard_name == "CustomerScorecard"


class TestInformaticaPresetConfigs:
    """Tests for Informatica preset configurations."""

    def test_default_config(self):
        """Test default Informatica config."""
        # Default has async_execution=True by default
        assert DEFAULT_INFORMATICA_CONFIG.async_execution is True

    def test_production_config(self):
        """Test production Informatica config."""
        assert PRODUCTION_INFORMATICA_CONFIG.auto_start is True
        assert PRODUCTION_INFORMATICA_CONFIG.health_check_enabled is True
        assert PRODUCTION_INFORMATICA_CONFIG.verify_ssl is True

    def test_development_config(self):
        """Test development Informatica config."""
        assert DEVELOPMENT_INFORMATICA_CONFIG.verify_ssl is False
        assert DEVELOPMENT_INFORMATICA_CONFIG.health_check_enabled is False


# =============================================================================
# InformaticaRuleTranslator Tests
# =============================================================================


class TestInformaticaRuleTranslator:
    """Tests for InformaticaRuleTranslator."""

    def test_supported_rule_types(self):
        """Test getting supported rule types."""
        translator = InformaticaRuleTranslator()
        types = translator.get_supported_rule_types()

        assert "not_null" in types
        assert "unique" in types
        assert "in_set" in types
        assert "in_range" in types
        assert "regex" in types

    def test_translate_not_null(self):
        """Test translating not_null rule."""
        translator = InformaticaRuleTranslator()
        rule = {"type": "not_null", "column": "customer_id"}

        translated = translator.translate(rule)

        assert translated["type"] == "NULL_CHECK"
        assert translated["fieldName"] == "customer_id"

    def test_translate_unique(self):
        """Test translating unique rule."""
        translator = InformaticaRuleTranslator()
        rule = {"type": "unique", "column": "email"}

        translated = translator.translate(rule)

        assert translated["type"] == "UNIQUE_CHECK"
        assert translated["fieldName"] == "email"

    def test_translate_in_set(self):
        """Test translating in_set rule."""
        translator = InformaticaRuleTranslator()
        rule = {
            "type": "in_set",
            "column": "status",
            "values": ["active", "inactive"],
        }

        translated = translator.translate(rule)

        assert translated["type"] == "DOMAIN_CHECK"
        assert translated["validValues"] == ["active", "inactive"]

    def test_translate_in_range(self):
        """Test translating in_range rule."""
        translator = InformaticaRuleTranslator()
        rule = {
            "type": "in_range",
            "column": "age",
            "min": 0,
            "max": 150,
        }

        translated = translator.translate(rule)

        assert translated["type"] == "RANGE_CHECK"
        assert translated["minValue"] == 0
        assert translated["maxValue"] == 150

    def test_translate_regex(self):
        """Test translating regex rule."""
        translator = InformaticaRuleTranslator()
        rule = {
            "type": "regex",
            "column": "phone",
            "pattern": r"^\d{3}-\d{4}$",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "PATTERN_CHECK"
        assert translated["pattern"] == r"^\d{3}-\d{4}$"

    def test_translate_multiple_rules(self):
        """Test translating multiple rules."""
        translator = InformaticaRuleTranslator()
        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
            {"type": "in_range", "column": "age", "min": 0, "max": 120},
        ]

        translated = translator.translate_batch(rules)

        assert len(translated) == 3
        assert translated[0]["type"] == "NULL_CHECK"
        assert translated[1]["type"] == "UNIQUE_CHECK"
        assert translated[2]["type"] == "RANGE_CHECK"


# =============================================================================
# InformaticaResultConverter Tests
# =============================================================================


class TestInformaticaResultConverter:
    """Tests for InformaticaResultConverter."""

    def test_convert_successful_result(self):
        """Test converting successful check result."""
        import time

        converter = InformaticaResultConverter()
        start_time = time.perf_counter()

        # Mock IDQ result format - uses "ruleResults" key
        vendor_result = {
            "ruleResults": [
                {
                    "ruleName": "NullCheck_customer_id",
                    "fieldName": "customer_id",
                    "passed": True,
                    "failedRecordCount": 0,
                    "message": "All values non-null",
                    "severity": "MEDIUM",
                }
            ],
        }

        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.PASSED

    def test_convert_failed_result(self):
        """Test converting failed check result."""
        import time

        converter = InformaticaResultConverter()
        start_time = time.perf_counter()

        # Mock IDQ result format - uses "ruleResults" key
        vendor_result = {
            "ruleResults": [
                {
                    "ruleName": "NullCheck_email",
                    "fieldName": "email",
                    "passed": False,
                    "failedRecordCount": 10,
                    "message": "10 null values found",
                    "severity": "HIGH",
                }
            ],
        }

        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.FAILED
        assert len(result.failures) == 1
        assert result.failures[0].column == "email"

    def test_severity_mapping(self):
        """Test severity mapping from IDQ to common."""
        converter = InformaticaResultConverter()

        # SEVERITY_MAPPING uses uppercase keys
        assert converter.SEVERITY_MAPPING.get("CRITICAL") == Severity.CRITICAL
        assert converter.SEVERITY_MAPPING.get("HIGH") == Severity.ERROR
        assert converter.SEVERITY_MAPPING.get("MEDIUM") == Severity.WARNING


# =============================================================================
# InformaticaAdapter Tests
# =============================================================================


class TestInformaticaAdapter:
    """Tests for InformaticaAdapter."""

    def test_engine_name(self):
        """Test engine name property."""
        adapter = InformaticaAdapter()
        assert adapter.engine_name == "informatica"

    def test_engine_version(self):
        """Test engine version property."""
        adapter = InformaticaAdapter()
        assert adapter.engine_version  # Should return a version string

    def test_capabilities(self):
        """Test engine capabilities."""
        adapter = InformaticaAdapter()
        caps = adapter.get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is False  # IDQ doesn't support learn
        assert "polars" in caps.supported_data_types
        assert "pandas" in caps.supported_data_types

    def test_get_info(self):
        """Test getting engine info."""
        adapter = InformaticaAdapter()
        info = adapter.get_info()

        assert info.name == "informatica"
        assert "informatica" in info.description.lower()

    def test_config_assignment(self):
        """Test that config is properly assigned."""
        config = InformaticaConfig(
            api_endpoint="https://test.example.com",
            domain="TestDomain",
        )
        adapter = InformaticaAdapter(config=config)

        assert adapter._config.api_endpoint == "https://test.example.com"
        assert adapter._config.domain == "TestDomain"

    def test_default_config(self):
        """Test adapter uses default config when none provided."""
        adapter = InformaticaAdapter()

        assert adapter._config is not None
        assert isinstance(adapter._config, InformaticaConfig)


class TestInformaticaAdapterLifecycle:
    """Tests for InformaticaAdapter lifecycle management."""

    def test_context_manager(self):
        """Test context manager usage with mocked connection."""
        adapter = InformaticaAdapter()

        # Mock the start and stop methods to avoid actual network calls
        with patch.object(adapter, "start") as mock_start, \
             patch.object(adapter, "stop") as mock_stop:
            with adapter:
                # In context - start was called
                mock_start.assert_called_once()
            # After context - stop was called
            mock_stop.assert_called_once()

    def test_start_stop(self):
        """Test start and stop methods."""
        from common.engines.lifecycle import EngineState

        adapter = InformaticaAdapter()

        # Initially CREATED
        assert adapter.get_state() == EngineState.CREATED


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateInformaticaAdapter:
    """Tests for create_informatica_adapter factory function."""

    def test_create_with_api_key(self):
        """Test creating adapter with API key."""
        adapter = create_informatica_adapter(
            api_endpoint="https://idq.example.com/api/v2",
            api_key="secret-key",
            domain="Production",
        )

        assert isinstance(adapter, InformaticaAdapter)
        assert adapter._config.api_endpoint == "https://idq.example.com/api/v2"
        assert adapter._config.api_key == "secret-key"
        assert adapter._config.domain == "Production"
        assert adapter._config.auth_type == AuthType.API_KEY

    def test_create_with_username_password(self):
        """Test creating adapter with username/password."""
        adapter = create_informatica_adapter(
            api_endpoint="https://idq.example.com/api/v2",
            username="admin",
            password="secret",
        )

        assert isinstance(adapter, InformaticaAdapter)
        assert adapter._config.username == "admin"
        assert adapter._config.password == "secret"
        assert adapter._config.auth_type == AuthType.BASIC

    def test_create_with_additional_kwargs(self):
        """Test creating adapter with additional kwargs."""
        adapter = create_informatica_adapter(
            api_endpoint="https://idq.example.com/api/v2",
            api_key="secret",
            project="MyProject",
            scorecard_name="MyScorecard",
        )

        assert adapter._config.project == "MyProject"
        assert adapter._config.scorecard_name == "MyScorecard"


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestInformaticaAdapterIntegration:
    """Integration tests with mocked API calls."""

    def test_check_with_mock_api(self):
        """Test check operation with mocked API."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = InformaticaAdapter(
            config=InformaticaConfig(
                api_endpoint="https://mock.example.com",
                api_key="test-key",
            )
        )

        # Create test data
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", None],
        })

        # Mock the internal methods
        with patch.object(adapter, "_ensure_connected"), \
             patch.object(adapter, "_execute_check") as mock_check:

            # Mock check result
            mock_check.return_value = CheckStatus.FAILED, 2, 1, [
                {
                    "rule_name": "NullCheck",
                    "column": "name",
                    "passed": False,
                    "failed_count": 1,
                    "message": "1 null value found",
                    "severity": "high",
                }
            ]

            # Need to mock state as well
            from common.engines.lifecycle import EngineState
            adapter._state_tracker._state = EngineState.RUNNING

            rules = [{"type": "not_null", "column": "name"}]
            result = adapter.check(df, rules=rules)

            assert result is not None
