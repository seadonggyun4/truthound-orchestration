"""Tests for SAP Data Services Adapter.

This module tests the SAP Data Services adapter including:
- SAPDataServicesConfig
- SAPDataServicesAdapter
- SAPDataServicesRuleTranslator
- SAPDataServicesResultConverter
- SAPDataServicesConnectionManager
- Factory function
"""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, Mock, patch

import pytest

from common.base import CheckStatus, Severity
from packages.enterprise.engines.base import (
    AuthType,
    VendorSDKError,
)
from packages.enterprise.engines.sap_data_services import (
    # Main classes
    SAPDataServicesAdapter,
    SAPDataServicesConfig,
    # Enums
    SAPExecutionMode,
    SAPRuleType,
    SAPDataType,
    SAPJobStatus,
    # Components
    SAPDataServicesConnectionManager,
    SAPDataServicesResultConverter,
    SAPDataServicesRuleTranslator,
    # Factory
    create_sap_data_services_adapter,
    # Presets
    DEFAULT_SAP_DS_CONFIG,
    PRODUCTION_SAP_DS_CONFIG,
    DEVELOPMENT_SAP_DS_CONFIG,
    REALTIME_SAP_DS_CONFIG,
    ADDRESS_CLEANSING_CONFIG,
)


# =============================================================================
# SAPDataServicesConfig Tests
# =============================================================================


class TestSAPDataServicesConfig:
    """Tests for SAPDataServicesConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SAPDataServicesConfig()

        assert config.repository == "Central"
        assert config.job_server == ""
        assert config.access_server == ""
        assert config.execution_mode == SAPExecutionMode.REALTIME
        assert config.enable_address_cleansing is False
        assert config.enable_geocoding is False
        assert config.enable_text_analysis is False
        assert config.locale == "en_US"

    def test_frozen_immutability(self):
        """Test that config is immutable."""
        config = SAPDataServicesConfig()

        with pytest.raises(FrozenInstanceError):
            config.repository = "new_repository"

    def test_with_repository(self):
        """Test builder method for repository."""
        config = SAPDataServicesConfig()
        new_config = config.with_repository(
            "DataQuality",
            datastore="MainStore",
            project="MyProject",
        )

        assert new_config.repository == "DataQuality"
        assert new_config.datastore == "MainStore"
        assert new_config.project == "MyProject"
        assert config.repository == "Central"  # Original unchanged

    def test_with_servers(self):
        """Test builder method for servers."""
        config = SAPDataServicesConfig()
        new_config = config.with_servers(
            job_server="JobServer1",
            access_server="AccessServer1",
            cmc_endpoint="https://cmc.example.com",
        )

        assert new_config.job_server == "JobServer1"
        assert new_config.access_server == "AccessServer1"
        assert new_config.cmc_endpoint == "https://cmc.example.com"

    def test_with_execution_mode(self):
        """Test builder method for execution mode."""
        config = SAPDataServicesConfig()
        new_config = config.with_execution_mode(SAPExecutionMode.BATCH)

        assert new_config.execution_mode == SAPExecutionMode.BATCH

    def test_with_address_cleansing(self):
        """Test builder method for address cleansing."""
        config = SAPDataServicesConfig()
        new_config = config.with_address_cleansing(
            enabled=True,
            directories=("USA", "CAN"),
            locale="en_CA",
        )

        assert new_config.enable_address_cleansing is True
        assert new_config.address_directories == ("USA", "CAN")
        assert new_config.locale == "en_CA"

    def test_with_geocoding(self):
        """Test builder method for geocoding."""
        config = SAPDataServicesConfig()
        new_config = config.with_geocoding(enabled=True)

        assert new_config.enable_geocoding is True

    def test_with_text_analysis(self):
        """Test builder method for text analysis."""
        config = SAPDataServicesConfig()
        new_config = config.with_text_analysis(enabled=True)

        assert new_config.enable_text_analysis is True

    def test_with_async_settings(self):
        """Test builder method for async settings."""
        config = SAPDataServicesConfig()
        new_config = config.with_async_settings(poll_interval=5.0, max_wait=600.0)

        assert new_config.async_poll_interval == 5.0
        assert new_config.max_async_wait == 600.0

    def test_with_substitution_parameters(self):
        """Test builder method for substitution parameters."""
        config = SAPDataServicesConfig()
        new_config = config.with_substitution_parameters(
            SOURCE_TABLE="customers",
            TARGET_TABLE="customers_clean",
        )

        assert new_config.substitution_parameters["SOURCE_TABLE"] == "customers"
        assert new_config.substitution_parameters["TARGET_TABLE"] == "customers_clean"

    def test_builder_chaining(self):
        """Test chaining multiple builder methods."""
        config = (
            SAPDataServicesConfig()
            .with_api_endpoint("https://sap-ds.example.com/api")
            .with_repository("DataQuality", datastore="MainStore")
            .with_servers(job_server="JobServer1")
            .with_execution_mode(SAPExecutionMode.BATCH)
            .with_address_cleansing(enabled=True, locale="en_US")
            .with_async_settings(poll_interval=3.0, max_wait=900.0)
        )

        assert config.api_endpoint == "https://sap-ds.example.com/api"
        assert config.repository == "DataQuality"
        assert config.datastore == "MainStore"
        assert config.job_server == "JobServer1"
        assert config.execution_mode == SAPExecutionMode.BATCH
        assert config.enable_address_cleansing is True
        assert config.locale == "en_US"
        assert config.async_poll_interval == 3.0
        assert config.max_async_wait == 900.0


class TestSAPDataServicesPresetConfigs:
    """Tests for SAP Data Services preset configurations."""

    def test_default_config(self):
        """Test default SAP DS config."""
        assert DEFAULT_SAP_DS_CONFIG.timeout_seconds == 120.0
        assert DEFAULT_SAP_DS_CONFIG.max_retries == 3
        assert DEFAULT_SAP_DS_CONFIG.auth_type == AuthType.BASIC
        assert DEFAULT_SAP_DS_CONFIG.execution_mode == SAPExecutionMode.REALTIME

    def test_production_config(self):
        """Test production SAP DS config."""
        assert PRODUCTION_SAP_DS_CONFIG.execution_mode == SAPExecutionMode.BATCH
        assert PRODUCTION_SAP_DS_CONFIG.verify_ssl is True
        assert PRODUCTION_SAP_DS_CONFIG.pool_size == 10
        assert PRODUCTION_SAP_DS_CONFIG.max_async_wait == 600.0

    def test_development_config(self):
        """Test development SAP DS config."""
        assert DEVELOPMENT_SAP_DS_CONFIG.verify_ssl is False
        assert DEVELOPMENT_SAP_DS_CONFIG.max_retries == 1
        assert DEVELOPMENT_SAP_DS_CONFIG.execution_mode == SAPExecutionMode.REALTIME

    def test_realtime_config(self):
        """Test real-time SAP DS config."""
        assert REALTIME_SAP_DS_CONFIG.execution_mode == SAPExecutionMode.REALTIME
        assert REALTIME_SAP_DS_CONFIG.timeout_seconds == 30.0
        assert REALTIME_SAP_DS_CONFIG.pool_size == 20

    def test_address_cleansing_config(self):
        """Test address cleansing config."""
        assert ADDRESS_CLEANSING_CONFIG.enable_address_cleansing is True
        assert ADDRESS_CLEANSING_CONFIG.enable_geocoding is True
        assert ADDRESS_CLEANSING_CONFIG.locale == "en_US"


# =============================================================================
# Enums Tests
# =============================================================================


class TestSAPDataServicesEnums:
    """Tests for SAP Data Services enums."""

    def test_execution_mode_values(self):
        """Test SAPExecutionMode enum values."""
        assert SAPExecutionMode.REALTIME.value == "REALTIME"
        assert SAPExecutionMode.BATCH.value == "BATCH"
        assert SAPExecutionMode.EMBEDDED.value == "EMBEDDED"

    def test_rule_type_values(self):
        """Test SAPRuleType enum values."""
        assert SAPRuleType.VALIDATION.value == "VALIDATION"
        assert SAPRuleType.ADDRESS_CLEANSE.value == "ADDRESS_CLEANSE"
        assert SAPRuleType.GEOCODE.value == "GEOCODE"
        assert SAPRuleType.TEXT_ANALYSIS.value == "TEXT_ANALYSIS"
        assert SAPRuleType.NULL_CHECK.value == "NULL_CHECK"
        assert SAPRuleType.UNIQUE_CHECK.value == "UNIQUE_CHECK"

    def test_data_type_values(self):
        """Test SAPDataType enum values."""
        assert SAPDataType.VARCHAR.value == "VARCHAR"
        assert SAPDataType.INTEGER.value == "INTEGER"
        assert SAPDataType.DECIMAL.value == "DECIMAL"
        assert SAPDataType.DATE.value == "DATE"
        assert SAPDataType.DATETIME.value == "DATETIME"

    def test_job_status_values(self):
        """Test SAPJobStatus enum values."""
        assert SAPJobStatus.PENDING.value == "PENDING"
        assert SAPJobStatus.RUNNING.value == "RUNNING"
        assert SAPJobStatus.SUCCEEDED.value == "SUCCEEDED"
        assert SAPJobStatus.FAILED.value == "FAILED"
        assert SAPJobStatus.CANCELLED.value == "CANCELLED"


# =============================================================================
# SAPDataServicesRuleTranslator Tests
# =============================================================================


class TestSAPDataServicesRuleTranslator:
    """Tests for SAPDataServicesRuleTranslator."""

    def test_supported_rule_types(self):
        """Test getting supported rule types."""
        translator = SAPDataServicesRuleTranslator()
        types = translator.get_supported_rule_types()

        assert "not_null" in types
        assert "unique" in types
        assert "in_set" in types
        assert "in_range" in types
        assert "regex" in types
        assert "dtype" in types
        assert "address" in types
        assert "match" in types

    def test_translate_not_null(self):
        """Test translating not_null rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {"type": "not_null", "column": "customer_id"}

        translated = translator.translate(rule)

        assert translated["type"] == "NULL_CHECK"
        assert translated["binding"]["column"] == "customer_id"
        assert translated["parameters"]["allowEmpty"] is False
        assert translated["parameters"]["treatWhitespaceAsNull"] is True

    def test_translate_unique(self):
        """Test translating unique rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {"type": "unique", "column": "email"}

        translated = translator.translate(rule)

        assert translated["type"] == "UNIQUE_CHECK"
        assert translated["binding"]["column"] == "email"
        assert translated["parameters"]["caseSensitive"] is True
        assert translated["parameters"]["includeNulls"] is False

    def test_translate_in_set(self):
        """Test translating in_set rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "in_set",
            "column": "status",
            "values": ["active", "inactive", "pending"],
        }

        translated = translator.translate(rule)

        assert translated["type"] == "DOMAIN_CHECK"
        assert translated["binding"]["column"] == "status"
        assert translated["parameters"]["validValues"] == [
            "active",
            "inactive",
            "pending",
        ]
        assert translated["parameters"]["caseSensitive"] is True

    def test_translate_in_range(self):
        """Test translating in_range rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "in_range",
            "column": "age",
            "min": 0,
            "max": 150,
        }

        translated = translator.translate(rule)

        assert translated["type"] == "RANGE_CHECK"
        assert translated["binding"]["column"] == "age"
        assert translated["parameters"]["minValue"] == 0
        assert translated["parameters"]["maxValue"] == 150
        assert translated["parameters"]["minInclusive"] is True
        assert translated["parameters"]["maxInclusive"] is True

    def test_translate_greater_than(self):
        """Test translating greater_than rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "greater_than",
            "column": "amount",
            "value": 0,
        }

        translated = translator.translate(rule)

        assert translated["type"] == "RANGE_CHECK"
        assert translated["parameters"]["minValue"] == 0
        assert translated["parameters"]["minInclusive"] is False

    def test_translate_regex(self):
        """Test translating regex rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "regex",
            "column": "phone",
            "pattern": r"^\d{3}-\d{4}$",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "FORMAT_CHECK"
        assert translated["binding"]["column"] == "phone"
        assert translated["parameters"]["pattern"] == r"^\d{3}-\d{4}$"
        assert translated["parameters"]["patternType"] == "REGEX"

    def test_translate_dtype(self):
        """Test translating dtype rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "dtype",
            "column": "score",
            "dtype": "float64",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "DATA_TYPE_CHECK"
        assert translated["parameters"]["expectedType"] == "DOUBLE"

    def test_translate_length_check(self):
        """Test translating length rules."""
        translator = SAPDataServicesRuleTranslator()

        # Min length
        rule = {"type": "min_length", "column": "name", "min": 2}
        translated = translator.translate(rule)
        assert translated["type"] == "LENGTH_CHECK"
        assert translated["parameters"]["minLength"] == 2

        # Max length
        rule = {"type": "max_length", "column": "name", "max": 100}
        translated = translator.translate(rule)
        assert translated["type"] == "LENGTH_CHECK"
        assert translated["parameters"]["maxLength"] == 100

        # Exact length
        rule = {"type": "length", "column": "name", "length": 10}
        translated = translator.translate(rule)
        assert translated["type"] == "LENGTH_CHECK"
        assert translated["parameters"]["exactLength"] == 10

    def test_translate_foreign_key(self):
        """Test translating foreign_key rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "foreign_key",
            "column": "customer_id",
            "reference_table": "customers",
            "reference_column": "id",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "REFERENCE_CHECK"
        assert translated["parameters"]["referenceTable"] == "customers"
        assert translated["parameters"]["referenceColumn"] == "id"

    def test_translate_expression(self):
        """Test translating expression rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "expression",
            "column": "total",
            "expression": "quantity * price == total",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "EXPRESSION_CHECK"
        assert translated["parameters"]["expression"] == "quantity * price == total"
        assert translated["parameters"]["language"] == "SAP_DS"

    def test_translate_sql_rule(self):
        """Test translating SQL rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "sql",
            "column": "id",
            "sql": "SELECT id FROM table WHERE status = 'active'",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "SQL_CHECK"
        assert (
            translated["parameters"]["sqlQuery"]
            == "SELECT id FROM table WHERE status = 'active'"
        )

    def test_translate_address_cleanse(self):
        """Test translating address_cleanse rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "address",
            "column": "address",
            "mode": "CERTIFY",
            "locale": "en_US",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "ADDRESS_CLEANSE"
        assert translated["parameters"]["mode"] == "CERTIFY"
        assert translated["parameters"]["locale"] == "en_US"

    def test_translate_geocode(self):
        """Test translating geocode rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "geocode",
            "column": "address",
            "precision": "HIGH",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "GEOCODE"
        assert translated["parameters"]["precision"] == "HIGH"

    def test_translate_match_transform(self):
        """Test translating match transform rule."""
        translator = SAPDataServicesRuleTranslator()
        rule = {
            "type": "match",
            "column": "name",
            "match_strategy": "FUZZY",
            "threshold": 0.9,
            "algorithm": "WEIGHTED",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "MATCH_TRANSFORM"
        assert translated["parameters"]["matchStrategy"] == "FUZZY"
        assert translated["parameters"]["threshold"] == 0.9
        assert translated["parameters"]["algorithm"] == "WEIGHTED"

    def test_translate_multiple_rules(self):
        """Test translating multiple rules."""
        translator = SAPDataServicesRuleTranslator()
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

    def test_severity_mapping(self):
        """Test severity mapping."""
        translator = SAPDataServicesRuleTranslator()

        assert translator._map_severity("critical") == "CRITICAL"
        assert translator._map_severity("high") == "ERROR"
        assert translator._map_severity("error") == "ERROR"
        assert translator._map_severity("medium") == "WARNING"
        assert translator._map_severity("warning") == "WARNING"
        assert translator._map_severity("low") == "INFO"
        assert translator._map_severity("info") == "INFO"
        assert translator._map_severity("unknown") == "WARNING"  # Default

    def test_dtype_mapping(self):
        """Test data type mapping."""
        translator = SAPDataServicesRuleTranslator()

        assert translator._map_dtype("string") == "VARCHAR"
        assert translator._map_dtype("int") == "INTEGER"
        assert translator._map_dtype("int64") == "LONG"
        assert translator._map_dtype("float64") == "DOUBLE"
        assert translator._map_dtype("bool") == "INTEGER"  # SAP uses 0/1
        assert translator._map_dtype("datetime") == "DATETIME"
        assert translator._map_dtype("unknown") == "VARCHAR"  # Default


# =============================================================================
# SAPDataServicesResultConverter Tests
# =============================================================================


class TestSAPDataServicesResultConverter:
    """Tests for SAPDataServicesResultConverter."""

    def test_convert_successful_result(self):
        """Test converting successful check result."""
        converter = SAPDataServicesResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "validationResults": [
                {
                    "ruleName": "NullCheck_customer_id",
                    "column": "customer_id",
                    "passed": True,
                    "failedCount": 0,
                    "message": "All values non-null",
                    "severity": "WARNING",
                }
            ],
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
        converter = SAPDataServicesResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "validationResults": [
                {
                    "ruleName": "NullCheck_email",
                    "column": "email",
                    "passed": False,
                    "failedCount": 10,
                    "message": "10 null values found",
                    "severity": "ERROR",
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
        assert result.failures[0].message == "10 null values found"

    def test_convert_rule_executions_format(self):
        """Test converting ruleExecutions format."""
        converter = SAPDataServicesResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "ruleExecutions": [
                {
                    "ruleName": "UniqueCheck",
                    "binding": {"column": "email"},
                    "passed": False,
                    "failedRecordCount": 5,
                    "message": "Duplicate values found",
                    "severity": "ERROR",
                }
            ],
        }

        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.FAILED
        assert len(result.failures) == 1

    def test_convert_quality_score_format(self):
        """Test converting quality score format."""
        converter = SAPDataServicesResultConverter()
        start_time = time.perf_counter()

        # Passing quality score
        vendor_result = {
            "qualityScore": 95,
            "threshold": 90,
            "ruleName": "OverallQuality",
            "exceptionCount": 0,
        }

        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.PASSED

        # Failing quality score
        vendor_result["qualityScore"] = 80
        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.FAILED

    def test_severity_mapping(self):
        """Test severity mapping from SAP to common."""
        converter = SAPDataServicesResultConverter()

        assert converter.SEVERITY_MAPPING.get("CRITICAL") == Severity.CRITICAL
        assert converter.SEVERITY_MAPPING.get("ERROR") == Severity.ERROR
        assert converter.SEVERITY_MAPPING.get("WARNING") == Severity.WARNING
        assert converter.SEVERITY_MAPPING.get("INFO") == Severity.INFO
        assert converter.SEVERITY_MAPPING.get("INFORMATIONAL") == Severity.INFO

    def test_convert_profile_result(self):
        """Test converting profile result."""
        converter = SAPDataServicesResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "rowCount": 1000,
            "columnProfiles": [
                {
                    "columnName": "id",
                    "dataType": "INTEGER",
                    "nullCount": 0,
                    "nullPercentage": 0.0,
                    "distinctCount": 1000,
                    "minValue": 1,
                    "maxValue": 1000,
                },
                {
                    "columnName": "name",
                    "dataType": "VARCHAR",
                    "nullCount": 10,
                    "nullPercentage": 1.0,
                    "distinctCount": 950,
                },
            ],
        }

        result = converter.convert_profile_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.row_count == 1000
        assert len(result.columns) == 2
        assert result.columns[0].column_name == "id"
        assert result.columns[0].dtype == "INTEGER"
        assert result.columns[0].null_count == 0
        assert result.columns[0].unique_count == 1000
        assert result.columns[1].column_name == "name"
        assert result.columns[1].null_percentage == 1.0


# =============================================================================
# SAPDataServicesConnectionManager Tests
# =============================================================================


class TestSAPDataServicesConnectionManager:
    """Tests for SAPDataServicesConnectionManager."""

    def test_init(self):
        """Test connection manager initialization."""
        config = SAPDataServicesConfig(
            api_endpoint="https://sap-ds.example.com/api",
        )
        manager = SAPDataServicesConnectionManager(config)

        assert manager._config == config
        assert manager._session is None

    def test_connect_with_basic_auth(self):
        """Test connection with basic authentication."""
        config = SAPDataServicesConfig(
            api_endpoint="https://sap-ds.example.com/api",
            username="admin",
            password="secret",
            auth_type=AuthType.BASIC,
        )
        manager = SAPDataServicesConnectionManager(config)

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            # Mock login response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"logonToken": "test-token"}
            mock_session.post.return_value = mock_response

            manager._do_connect()

            assert manager._session is not None
            assert manager._auth_token == "test-token"

    def test_execute_api_call(self):
        """Test executing API call."""
        config = SAPDataServicesConfig(
            api_endpoint="https://sap-ds.example.com/api",
        )
        manager = SAPDataServicesConnectionManager(config)

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_session.request.return_value = mock_response

        manager._session = mock_session

        result = manager.execute_api_call("GET", "/serverInfo")

        assert result == {"status": "success"}
        mock_session.request.assert_called_once()


# =============================================================================
# SAPDataServicesAdapter Tests
# =============================================================================


class TestSAPDataServicesAdapter:
    """Tests for SAPDataServicesAdapter."""

    def test_engine_name(self):
        """Test engine name property."""
        adapter = SAPDataServicesAdapter()
        assert adapter.engine_name == "sap_data_services"

    def test_engine_version(self):
        """Test engine version property."""
        adapter = SAPDataServicesAdapter()
        assert adapter.engine_version == "4.3"

    def test_capabilities(self):
        """Test engine capabilities."""
        adapter = SAPDataServicesAdapter()
        caps = adapter.get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is False
        assert caps.supports_async is True
        assert "polars" in caps.supported_data_types
        assert "pandas" in caps.supported_data_types

    def test_capabilities_with_address_cleansing(self):
        """Test capabilities include address when enabled."""
        config = SAPDataServicesConfig(enable_address_cleansing=True)
        adapter = SAPDataServicesAdapter(config=config)
        caps = adapter.get_capabilities()

        assert "address" in caps.supported_rule_types
        assert "address_cleanse" in caps.supported_rule_types

    def test_get_info(self):
        """Test getting engine info."""
        adapter = SAPDataServicesAdapter()
        info = adapter.get_info()

        assert info.name == "sap_data_services"

    def test_config_assignment(self):
        """Test that config is properly assigned."""
        config = SAPDataServicesConfig(
            api_endpoint="https://test.example.com",
            repository="TestRepo",
        )
        adapter = SAPDataServicesAdapter(config=config)

        assert adapter._config.api_endpoint == "https://test.example.com"
        assert adapter._config.repository == "TestRepo"

    def test_default_config(self):
        """Test adapter uses default config when none provided."""
        adapter = SAPDataServicesAdapter()

        assert adapter._config is not None
        assert isinstance(adapter._config, SAPDataServicesConfig)


class TestSAPDataServicesAdapterLifecycle:
    """Tests for SAPDataServicesAdapter lifecycle management."""

    def test_context_manager(self):
        """Test context manager usage with mocked connection."""
        adapter = SAPDataServicesAdapter()

        with (
            patch.object(adapter, "start") as mock_start,
            patch.object(adapter, "stop") as mock_stop,
        ):
            with adapter:
                mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_start_stop(self):
        """Test start and stop methods."""
        from common.engines.lifecycle import EngineState

        adapter = SAPDataServicesAdapter()

        # Initially CREATED
        assert adapter.get_state() == EngineState.CREATED


class TestSAPDataServicesAdapterMethods:
    """Tests for SAPDataServicesAdapter specific methods."""

    def test_build_check_payload(self):
        """Test building check payload."""
        pytest.importorskip("pandas")
        import pandas as pd

        config = SAPDataServicesConfig(
            repository="TestRepo",
            datastore="MainStore",
            project="MyProject",
            job_server="JobServer1",
            execution_mode=SAPExecutionMode.BATCH,
        )
        adapter = SAPDataServicesAdapter(config=config)

        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        rules = [{"type": "NULL_CHECK", "binding": {"column": "id"}}]

        payload = adapter._build_check_payload(df, rules)

        assert payload["repository"] == "TestRepo"
        assert payload["datastore"] == "MainStore"
        assert payload["project"] == "MyProject"
        assert payload["rules"] == rules
        assert payload["executionMode"] == "BATCH"
        assert payload["jobServer"] == "JobServer1"

    def test_build_profile_payload(self):
        """Test building profile payload."""
        pytest.importorskip("pandas")
        import pandas as pd

        config = SAPDataServicesConfig(
            repository="TestRepo",
            datastore="MainStore",
        )
        adapter = SAPDataServicesAdapter(config=config)

        df = pd.DataFrame({"id": [1, 2, 3]})

        payload = adapter._build_profile_payload(df)

        assert payload["repository"] == "TestRepo"
        assert payload["datastore"] == "MainStore"
        assert payload["options"]["computeStatistics"] is True
        assert payload["options"]["inferDataTypes"] is True

    def test_serialize_data(self):
        """Test data serialization."""
        pytest.importorskip("pandas")
        import pandas as pd

        adapter = SAPDataServicesAdapter()
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        serialized = adapter._serialize_data(df)

        assert serialized["rowCount"] == 2
        assert len(serialized["columns"]) == 2
        assert serialized["columns"][0]["name"] == "id"
        assert len(serialized["rows"]) == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateSAPDataServicesAdapter:
    """Tests for create_sap_data_services_adapter factory function."""

    def test_create_with_api_key(self):
        """Test creating adapter with API key."""
        adapter = create_sap_data_services_adapter(
            api_endpoint="https://sap-ds.example.com/api",
            api_key="secret-key",
            repository="Production",
        )

        assert isinstance(adapter, SAPDataServicesAdapter)
        assert adapter._config.api_endpoint == "https://sap-ds.example.com/api"
        assert adapter._config.api_key == "secret-key"
        assert adapter._config.repository == "Production"
        assert adapter._config.auth_type == AuthType.API_KEY

    def test_create_with_username_password(self):
        """Test creating adapter with username/password."""
        adapter = create_sap_data_services_adapter(
            api_endpoint="https://sap-ds.example.com/api",
            username="admin",
            password="secret",
        )

        assert isinstance(adapter, SAPDataServicesAdapter)
        assert adapter._config.username == "admin"
        assert adapter._config.password == "secret"
        assert adapter._config.auth_type == AuthType.BASIC

    def test_create_with_execution_mode(self):
        """Test creating adapter with execution mode."""
        adapter = create_sap_data_services_adapter(
            api_endpoint="https://sap-ds.example.com/api",
            username="admin",
            password="secret",
            execution_mode=SAPExecutionMode.BATCH,
        )

        assert adapter._config.execution_mode == SAPExecutionMode.BATCH


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestSAPDataServicesAdapterIntegration:
    """Integration tests with mocked API calls."""

    def test_check_with_mock_api(self):
        """Test check operation with mocked API."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = SAPDataServicesAdapter(
            config=SAPDataServicesConfig(
                api_endpoint="https://mock.example.com",
                username="test",
                password="test",
            )
        )

        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", None],
            }
        )

        with (
            patch.object(adapter, "_ensure_connected"),
            patch.object(adapter, "_execute_check") as mock_check,
        ):
            mock_check.return_value = {
                "validationResults": [
                    {
                        "ruleName": "NullCheck",
                        "column": "name",
                        "passed": False,
                        "failedCount": 1,
                        "message": "1 null value found",
                        "severity": "ERROR",
                    }
                ]
            }

            from common.engines.lifecycle import EngineState

            adapter._state_tracker._state = EngineState.RUNNING

            rules = [{"type": "not_null", "column": "name"}]
            result = adapter.check(df, rules=rules)

            assert result is not None

    def test_profile_with_mock_api(self):
        """Test profile operation with mocked API."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = SAPDataServicesAdapter(
            config=SAPDataServicesConfig(
                api_endpoint="https://mock.example.com",
                username="test",
                password="test",
            )
        )

        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10.5, 20.3, 30.1],
            }
        )

        with (
            patch.object(adapter, "_ensure_connected"),
            patch.object(adapter, "_execute_profile") as mock_profile,
        ):
            mock_profile.return_value = {
                "rowCount": 3,
                "columnProfiles": [
                    {
                        "columnName": "id",
                        "dataType": "INTEGER",
                        "nullCount": 0,
                        "nullPercentage": 0.0,
                        "distinctCount": 3,
                    },
                    {
                        "columnName": "value",
                        "dataType": "DOUBLE",
                        "nullCount": 0,
                        "nullPercentage": 0.0,
                        "distinctCount": 3,
                    },
                ],
            }

            from common.engines.lifecycle import EngineState

            adapter._state_tracker._state = EngineState.RUNNING

            result = adapter.profile(df)

            assert result is not None
            assert result.row_count == 3


class TestSAPDataServicesBatchExecution:
    """Tests for batch execution mode."""

    def test_batch_job_execution(self):
        """Test batch job submission and polling."""
        config = SAPDataServicesConfig(
            api_endpoint="https://sap-ds.example.com",
            execution_mode=SAPExecutionMode.BATCH,
            async_poll_interval=0.05,
            max_async_wait=10.0,
        )
        adapter = SAPDataServicesAdapter(config=config)

        mock_connection = MagicMock()
        mock_connection.execute_api_call.side_effect = [
            {"jobId": "job123"},
            {"status": "SUCCEEDED"},
            {"validationResults": []},
        ]

        result = adapter._execute_batch_job(
            mock_connection, "validation/execute", {"rules": []}
        )

        assert result == {"validationResults": []}

    def test_batch_job_timeout(self):
        """Test batch job timeout."""
        config = SAPDataServicesConfig(
            api_endpoint="https://sap-ds.example.com",
            execution_mode=SAPExecutionMode.BATCH,
            async_poll_interval=0.1,
            max_async_wait=0.3,
        )
        adapter = SAPDataServicesAdapter(config=config)

        mock_connection = MagicMock()
        mock_connection.execute_api_call.side_effect = [
            {"jobId": "job123"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
        ]

        with pytest.raises(VendorSDKError) as exc_info:
            adapter._execute_batch_job(
                mock_connection, "validation/execute", {"rules": []}
            )

        assert "timed out" in str(exc_info.value).lower()

    def test_batch_job_failure(self):
        """Test batch job failure handling."""
        config = SAPDataServicesConfig(
            api_endpoint="https://sap-ds.example.com",
            execution_mode=SAPExecutionMode.BATCH,
            async_poll_interval=0.1,
        )
        adapter = SAPDataServicesAdapter(config=config)

        mock_connection = MagicMock()
        mock_connection.execute_api_call.side_effect = [
            {"jobId": "job123"},
            {"status": "FAILED", "errorMessage": "Validation error"},
        ]

        with pytest.raises(VendorSDKError) as exc_info:
            adapter._execute_batch_job(
                mock_connection, "validation/execute", {"rules": []}
            )

        assert "failed" in str(exc_info.value).lower()


# =============================================================================
# SAP-Specific Features Tests
# =============================================================================


class TestSAPDataServicesSpecificFeatures:
    """Tests for SAP-specific features."""

    def test_address_cleansing_not_enabled(self):
        """Test address cleansing raises error when not enabled."""
        adapter = SAPDataServicesAdapter()

        with pytest.raises(VendorSDKError) as exc_info:
            adapter.cleanse_address({}, {"street": "street_col"})

        assert "not enabled" in str(exc_info.value).lower()

    def test_geocoding_not_enabled(self):
        """Test geocoding raises error when not enabled."""
        adapter = SAPDataServicesAdapter()

        with pytest.raises(VendorSDKError) as exc_info:
            adapter.geocode({}, {"street": "street_col"})

        assert "not enabled" in str(exc_info.value).lower()

    def test_text_analysis_not_enabled(self):
        """Test text analysis raises error when not enabled."""
        adapter = SAPDataServicesAdapter()

        with pytest.raises(VendorSDKError) as exc_info:
            adapter.analyze_text({}, "text_col")

        assert "not enabled" in str(exc_info.value).lower()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestSAPDataServicesEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_rules_list(self):
        """Test handling empty rules list."""
        translator = SAPDataServicesRuleTranslator()
        translated = translator.translate_batch([])
        assert translated == []

    def test_unknown_rule_type(self):
        """Test handling unknown rule type."""
        from packages.enterprise.engines.base import RuleTranslationError

        translator = SAPDataServicesRuleTranslator()
        rule = {"type": "unknown_type", "column": "id"}

        with pytest.raises(RuleTranslationError):
            translator.translate(rule)

    def test_empty_vendor_result(self):
        """Test handling empty vendor result."""
        converter = SAPDataServicesResultConverter()
        start_time = time.perf_counter()

        vendor_result = {}
        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 0
        assert result.failed_count == 0

    def test_serialization_with_polars(self):
        """Test data serialization with Polars DataFrame."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = SAPDataServicesAdapter()
        df = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})

        serialized = adapter._serialize_data(df)

        assert serialized["rowCount"] == 2
        assert len(serialized["rows"]) == 2

    def test_serialization_with_dict(self):
        """Test data serialization with dictionary."""
        adapter = SAPDataServicesAdapter()
        data = {"key": "value"}

        serialized = adapter._serialize_data(data)

        assert serialized == {"key": "value"}

    def test_serialization_with_list(self):
        """Test data serialization with list."""
        adapter = SAPDataServicesAdapter()
        data = [{"id": 1}, {"id": 2}]

        serialized = adapter._serialize_data(data)

        assert serialized["rowCount"] == 2
        assert serialized["rows"] == data
