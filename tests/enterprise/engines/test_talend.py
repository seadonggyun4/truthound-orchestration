"""Tests for Talend Data Quality Adapter.

This module tests the Talend adapter including:
- TalendConfig
- TalendAdapter
- TalendRuleTranslator
- TalendResultConverter
- TalendConnectionManager
- Factory function
"""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch, MagicMock

from packages.enterprise.engines.talend import (
    # Main classes
    TalendAdapter,
    TalendConfig,
    # Enums
    TalendExecutionMode,
    TalendIndicatorType,
    # Components
    TalendRuleTranslator,
    TalendResultConverter,
    TalendConnectionManager,
    # Factory
    create_talend_adapter,
    # Presets
    DEFAULT_TALEND_CONFIG,
    PRODUCTION_TALEND_CONFIG,
    DEVELOPMENT_TALEND_CONFIG,
    EMBEDDED_TALEND_CONFIG,
)
from packages.enterprise.engines.base import (
    AuthType,
    VendorSDKError,
)
from common.base import CheckStatus, LearnStatus, Severity


# =============================================================================
# TalendConfig Tests
# =============================================================================


class TestTalendConfig:
    """Tests for TalendConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TalendConfig()

        assert config.execution_mode == TalendExecutionMode.API
        assert config.workspace == ""
        assert config.project == ""
        assert config.use_metadata is True
        assert config.cache_metadata is True
        assert config.default_row_limit == 100000
        assert config.sample_percentage == 100.0

    def test_frozen_immutability(self):
        """Test that config is immutable."""
        config = TalendConfig()

        with pytest.raises(FrozenInstanceError):
            config.workspace = "new_workspace"

    def test_with_execution_mode(self):
        """Test builder method for execution mode."""
        config = TalendConfig()
        new_config = config.with_execution_mode(TalendExecutionMode.EMBEDDED)

        assert new_config.execution_mode == TalendExecutionMode.EMBEDDED
        assert config.execution_mode == TalendExecutionMode.API  # Original unchanged

    def test_with_workspace(self):
        """Test builder method for workspace."""
        config = TalendConfig()
        new_config = config.with_workspace("Production")

        assert new_config.workspace == "Production"

    def test_with_project(self):
        """Test builder method for project."""
        config = TalendConfig()
        new_config = config.with_project("DQ_Project", repository_path="/repo")

        assert new_config.project == "DQ_Project"
        assert new_config.repository_path == "/repo"

    def test_with_analysis(self):
        """Test builder method for analysis."""
        config = TalendConfig()
        new_config = config.with_analysis("CustomerAnalysis")

        assert new_config.analysis_name == "CustomerAnalysis"

    def test_with_embedded_mode(self):
        """Test builder method for embedded mode."""
        config = TalendConfig()
        new_config = config.with_embedded_mode("/opt/talend/lib")

        assert new_config.execution_mode == TalendExecutionMode.EMBEDDED
        assert new_config.embedded_lib_path == "/opt/talend/lib"

    def test_with_sampling(self):
        """Test builder method for sampling."""
        config = TalendConfig()
        new_config = config.with_sampling(row_limit=50000, percentage=50.0)

        assert new_config.default_row_limit == 50000
        assert new_config.sample_percentage == 50.0

    def test_validation_sample_percentage_bounds(self):
        """Test validation of sample percentage bounds."""
        with pytest.raises(ValueError, match="sample_percentage"):
            TalendConfig(sample_percentage=0)

        with pytest.raises(ValueError, match="sample_percentage"):
            TalendConfig(sample_percentage=101)

    def test_validation_row_limit(self):
        """Test validation of row limit."""
        with pytest.raises(ValueError, match="default_row_limit"):
            TalendConfig(default_row_limit=0)

    def test_builder_chaining(self):
        """Test chaining multiple builder methods."""
        config = (
            TalendConfig()
            .with_api_endpoint("https://talend.example.com/api/v1")
            .with_api_key("secret")
            .with_workspace("Production")
            .with_project("DataQuality")
            .with_sampling(row_limit=10000, percentage=10.0)
        )

        assert config.api_endpoint == "https://talend.example.com/api/v1"
        assert config.api_key == "secret"
        assert config.workspace == "Production"
        assert config.project == "DataQuality"
        assert config.default_row_limit == 10000
        assert config.sample_percentage == 10.0


class TestTalendPresetConfigs:
    """Tests for Talend preset configurations."""

    def test_default_config(self):
        """Test default Talend config."""
        assert DEFAULT_TALEND_CONFIG.execution_mode == TalendExecutionMode.API

    def test_production_config(self):
        """Test production Talend config."""
        assert PRODUCTION_TALEND_CONFIG.auto_start is True
        assert PRODUCTION_TALEND_CONFIG.health_check_enabled is True
        assert PRODUCTION_TALEND_CONFIG.verify_ssl is True
        assert PRODUCTION_TALEND_CONFIG.default_row_limit >= 100000

    def test_development_config(self):
        """Test development Talend config."""
        assert DEVELOPMENT_TALEND_CONFIG.verify_ssl is False
        assert DEVELOPMENT_TALEND_CONFIG.sample_percentage < 100

    def test_embedded_config(self):
        """Test embedded mode config."""
        assert EMBEDDED_TALEND_CONFIG.execution_mode == TalendExecutionMode.EMBEDDED
        assert EMBEDDED_TALEND_CONFIG.use_metadata is False


# =============================================================================
# TalendExecutionMode Tests
# =============================================================================


class TestTalendExecutionMode:
    """Tests for TalendExecutionMode enum."""

    def test_all_modes_exist(self):
        """Test all execution modes are defined."""
        assert TalendExecutionMode.API is not None
        assert TalendExecutionMode.EMBEDDED is not None
        assert TalendExecutionMode.STUDIO is not None


# =============================================================================
# TalendIndicatorType Tests
# =============================================================================


class TestTalendIndicatorType:
    """Tests for TalendIndicatorType enum."""

    def test_common_indicators(self):
        """Test common indicator types exist."""
        assert TalendIndicatorType.ROW_COUNT.value == "rowCount"
        assert TalendIndicatorType.NULL_COUNT.value == "nullCount"
        assert TalendIndicatorType.DISTINCT_COUNT.value == "distinctCount"
        assert TalendIndicatorType.UNIQUE_COUNT.value == "uniqueCount"


# =============================================================================
# TalendRuleTranslator Tests
# =============================================================================


class TestTalendRuleTranslator:
    """Tests for TalendRuleTranslator."""

    def test_supported_rule_types(self):
        """Test getting supported rule types."""
        translator = TalendRuleTranslator()
        types = translator.get_supported_rule_types()

        assert "not_null" in types
        assert "unique" in types
        assert "in_set" in types
        assert "in_range" in types
        assert "regex" in types
        assert "email" in types
        assert "phone" in types

    def test_translate_not_null(self):
        """Test translating not_null rule."""
        translator = TalendRuleTranslator()
        rule = {"type": "not_null", "column": "customer_id"}

        translated = translator.translate(rule)

        assert translated["type"] == "NOT_NULL_INDICATOR"
        assert translated["analyzedColumn"] == "customer_id"

    def test_translate_unique(self):
        """Test translating unique rule."""
        translator = TalendRuleTranslator()
        rule = {"type": "unique", "column": "email"}

        translated = translator.translate(rule)

        assert translated["type"] == "UNIQUE_COUNT_INDICATOR"

    def test_translate_in_set(self):
        """Test translating in_set rule."""
        translator = TalendRuleTranslator()
        rule = {
            "type": "in_set",
            "column": "status",
            "values": ["active", "inactive"],
            "ignore_case": True,
        }

        translated = translator.translate(rule)

        assert translated["type"] == "VALUE_IN_LIST_INDICATOR"
        assert translated["validValues"] == ["active", "inactive"]
        assert translated["ignoreCase"] is True

    def test_translate_in_range(self):
        """Test translating in_range rule."""
        translator = TalendRuleTranslator()
        rule = {
            "type": "in_range",
            "column": "age",
            "min": 0,
            "max": 150,
            "include_min": True,
            "include_max": False,
        }

        translated = translator.translate(rule)

        assert translated["type"] == "VALUE_IN_RANGE_INDICATOR"
        assert translated["minValue"] == 0
        assert translated["maxValue"] == 150
        assert translated["includeMin"] is True
        assert translated["includeMax"] is False

    def test_translate_regex(self):
        """Test translating regex rule."""
        translator = TalendRuleTranslator()
        rule = {
            "type": "regex",
            "column": "code",
            "pattern": r"^[A-Z]{3}\d{4}$",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "PATTERN_MATCHING_INDICATOR"
        assert translated["pattern"] == r"^[A-Z]{3}\d{4}$"
        assert translated["patternType"] == "REGEX"

    def test_translate_email(self):
        """Test translating email rule."""
        translator = TalendRuleTranslator()
        rule = {"type": "email", "column": "email_addr"}

        translated = translator.translate(rule)

        assert translated["type"] == "EMAIL_PATTERN_INDICATOR"

    def test_translate_dtype(self):
        """Test translating dtype rule."""
        translator = TalendRuleTranslator()
        rule = {"type": "dtype", "column": "value", "dtype": "float64"}

        translated = translator.translate(rule)

        assert translated["type"] == "DATA_TYPE_INDICATOR"
        assert translated["expectedType"] == "DOUBLE"

    def test_translate_outlier(self):
        """Test translating outlier rule."""
        translator = TalendRuleTranslator()
        rule = {
            "type": "outlier",
            "column": "value",
            "method": "IQR",
            "multiplier": 2.0,
        }

        translated = translator.translate(rule)

        assert translated["type"] == "OUTLIER_INDICATOR"
        assert translated["method"] == "IQR"
        assert translated["multiplier"] == 2.0


# =============================================================================
# TalendResultConverter Tests
# =============================================================================


class TestTalendResultConverter:
    """Tests for TalendResultConverter."""

    def test_convert_successful_result(self):
        """Test converting successful check result."""
        import time

        converter = TalendResultConverter()
        start_time = time.perf_counter()

        # Mock Talend result format
        vendor_result = {
            "indicators": [
                {
                    "name": "NOT_NULL_INDICATOR",
                    "analyzedColumn": "customer_id",
                    "matchCount": 100,
                    "nonMatchCount": 0,
                    "matchPercentage": 100.0,
                    "threshold": 100.0,
                }
            ]
        }

        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 1
        assert result.failed_count == 0

    def test_convert_failed_result(self):
        """Test converting failed check result."""
        import time

        converter = TalendResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "indicators": [
                {
                    "name": "NOT_NULL_INDICATOR",
                    "analyzedColumn": "name",
                    "matchCount": 90,
                    "nonMatchCount": 10,
                    "matchPercentage": 90.0,
                    "threshold": 100.0,
                }
            ]
        }

        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.FAILED
        # failed_count is number of rules that failed (1), not nonMatchCount (10)
        assert result.failed_count == 1
        assert len(result.failures) == 1
        assert result.failures[0].failed_count == 10  # This is the record count

    def test_convert_profile_result(self):
        """Test converting profile result."""
        import time

        converter = TalendResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "rowCount": 1000,
            "columnAnalyses": [
                {
                    "columnName": "id",
                    "dataType": "INTEGER",
                    "indicators": {
                        "nullCount": 0,
                        "nullPercentage": 0.0,
                        "distinctCount": 1000,
                        "minValue": 1,
                        "maxValue": 1000,
                    }
                },
                {
                    "columnName": "name",
                    "dataType": "STRING",
                    "indicators": {
                        "nullCount": 5,
                        "nullPercentage": 0.5,
                        "distinctCount": 950,
                        "blankCount": 2,
                    }
                }
            ]
        }

        result = converter.convert_profile_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.row_count == 1000
        assert len(result.columns) == 2
        assert result.columns[0].column_name == "id"
        assert result.columns[0].null_count == 0
        assert result.columns[0].unique_count == 1000  # Uses unique_count not distinct_count
        assert result.columns[1].column_name == "name"
        assert result.columns[1].null_count == 5

    def test_convert_learn_result(self):
        """Test converting learn result."""
        import time

        converter = TalendResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "discoveries": [
                {
                    "column": "email",
                    "ruleType": "email_pattern",
                    "parameters": {"pattern": "EMAIL"},
                    "confidence": 0.95,
                    "sampleMatches": 950,
                    "description": "Email pattern detected",
                },
                {
                    "column": "phone",
                    "ruleType": "phone_pattern",
                    "parameters": {"pattern": "US_PHONE"},
                    "confidence": 0.88,
                    "sampleMatches": 880,
                    "description": "US phone pattern detected",
                }
            ]
        }

        result = converter.convert_learn_result(
            vendor_result,
            start_time=start_time,
        )

        assert len(result.rules) == 2
        assert result.rules[0].column == "email"
        assert result.rules[0].confidence == 0.95
        assert result.rules[0].sample_size == 950  # Uses sample_size not sample_matches
        assert result.rules[1].column == "phone"


# =============================================================================
# TalendAdapter Tests
# =============================================================================


class TestTalendAdapter:
    """Tests for TalendAdapter."""

    def test_engine_name(self):
        """Test engine name property."""
        adapter = TalendAdapter()
        assert adapter.engine_name == "talend"

    def test_engine_version(self):
        """Test engine version property."""
        adapter = TalendAdapter()
        assert adapter.engine_version  # Should return a version string

    def test_capabilities(self):
        """Test engine capabilities."""
        adapter = TalendAdapter()
        caps = adapter.get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is True  # Talend supports learn
        assert "polars" in caps.supported_data_types
        assert "pandas" in caps.supported_data_types

    def test_get_info(self):
        """Test getting engine info."""
        adapter = TalendAdapter()
        info = adapter.get_info()

        assert info.name == "talend"
        assert "talend" in info.description.lower()
        assert "talend" in info.homepage.lower()

    def test_config_assignment(self):
        """Test that config is properly assigned."""
        config = TalendConfig(
            api_endpoint="https://test.example.com",
            workspace="TestWorkspace",
        )
        adapter = TalendAdapter(config=config)

        assert adapter._config.api_endpoint == "https://test.example.com"
        assert adapter._config.workspace == "TestWorkspace"

    def test_default_config(self):
        """Test adapter uses default config when none provided."""
        adapter = TalendAdapter()

        assert adapter._config is not None
        assert isinstance(adapter._config, TalendConfig)


class TestTalendAdapterLifecycle:
    """Tests for TalendAdapter lifecycle management."""

    def test_context_manager(self):
        """Test context manager usage with mocked connection."""
        adapter = TalendAdapter()

        # Mock the start and stop methods to avoid actual network calls
        with patch.object(adapter, "start") as mock_start, \
             patch.object(adapter, "stop") as mock_stop:
            with adapter:
                # In context - start was called
                mock_start.assert_called_once()
            # After context - stop was called
            mock_stop.assert_called_once()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTalendAdapter:
    """Tests for create_talend_adapter factory function."""

    def test_create_api_mode(self):
        """Test creating adapter in API mode."""
        adapter = create_talend_adapter(
            api_endpoint="https://talend.example.com/api/v1",
            api_key="secret-key",
        )

        assert isinstance(adapter, TalendAdapter)
        assert adapter._config.api_endpoint == "https://talend.example.com/api/v1"
        assert adapter._config.api_key == "secret-key"
        assert adapter._config.execution_mode == TalendExecutionMode.API
        assert adapter._config.auth_type == AuthType.API_KEY

    def test_create_embedded_mode(self):
        """Test creating adapter in embedded mode."""
        adapter = create_talend_adapter(
            execution_mode=TalendExecutionMode.EMBEDDED,
            embedded_lib_path="/opt/talend/lib",
        )

        assert isinstance(adapter, TalendAdapter)
        assert adapter._config.execution_mode == TalendExecutionMode.EMBEDDED
        assert adapter._config.embedded_lib_path == "/opt/talend/lib"

    def test_create_with_additional_kwargs(self):
        """Test creating adapter with additional kwargs."""
        adapter = create_talend_adapter(
            api_endpoint="https://talend.example.com/api/v1",
            api_key="secret",
            workspace="Production",
            project="DQ",
        )

        assert adapter._config.workspace == "Production"
        assert adapter._config.project == "DQ"


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestTalendAdapterIntegration:
    """Integration tests with mocked API calls."""

    def test_check_with_mock_api(self):
        """Test check operation with mocked API."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = TalendAdapter(
            config=TalendConfig(
                api_endpoint="https://mock.example.com",
                api_key="test-key",
            )
        )

        # Create test data
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", None],
        })

        # Mock the connection and API call
        with patch.object(adapter, "_get_connection_manager") as mock_conn:
            mock_manager = MagicMock()
            mock_conn.return_value = mock_manager

            # Mock API response
            mock_manager.execute_api_call.return_value = {
                "indicators": [
                    {
                        "name": "NOT_NULL_INDICATOR",
                        "analyzedColumn": "name",
                        "matchCount": 2,
                        "nonMatchCount": 1,
                        "matchPercentage": 66.7,
                        "threshold": 100.0,
                    }
                ]
            }

            # Need to mock start as well
            with patch.object(adapter, "start"):
                adapter._state_tracker._state = (
                    __import__(
                        "common.engines.lifecycle", fromlist=["EngineState"]
                    ).EngineState.RUNNING
                )

                rules = [{"type": "not_null", "column": "name"}]
                result = adapter.check(df, rules=rules)

                assert result.status == CheckStatus.FAILED
                assert result.failed_count == 1

    def test_profile_with_mock_api(self):
        """Test profile operation with mocked API."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = TalendAdapter(
            config=TalendConfig(
                api_endpoint="https://mock.example.com",
                api_key="test-key",
            )
        )

        df = pl.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
        })

        with patch.object(adapter, "_get_connection_manager") as mock_conn:
            mock_manager = MagicMock()
            mock_conn.return_value = mock_manager

            mock_manager.execute_api_call.return_value = {
                "rowCount": 3,
                "columnAnalyses": [
                    {
                        "columnName": "id",
                        "dataType": "INTEGER",
                        "indicators": {
                            "nullCount": 0,
                            "distinctCount": 3,
                        }
                    },
                    {
                        "columnName": "value",
                        "dataType": "DOUBLE",
                        "indicators": {
                            "nullCount": 0,
                            "distinctCount": 3,
                            "mean": 20.0,
                        }
                    }
                ]
            }

            with patch.object(adapter, "start"):
                adapter._state_tracker._state = (
                    __import__(
                        "common.engines.lifecycle", fromlist=["EngineState"]
                    ).EngineState.RUNNING
                )

                result = adapter.profile(df)

                assert result.row_count == 3
                assert len(result.columns) == 2
