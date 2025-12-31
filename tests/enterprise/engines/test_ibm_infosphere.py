"""Tests for IBM InfoSphere Information Server Adapter.

This module tests the IBM InfoSphere adapter including:
- IBMInfoSphereConfig
- IBMInfoSphereAdapter
- IBMInfoSphereRuleTranslator
- IBMInfoSphereResultConverter
- IBMInfoSphereConnectionManager
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
from packages.enterprise.engines.ibm_infosphere import (
    # Main classes
    IBMInfoSphereAdapter,
    IBMInfoSphereConfig,
    # Enums
    InfoSphereAnalysisType,
    InfoSphereExecutionMode,
    InfoSphereRuleType,
    # Components
    IBMInfoSphereConnectionManager,
    IBMInfoSphereResultConverter,
    IBMInfoSphereRuleTranslator,
    # Factory
    create_ibm_infosphere_adapter,
    # Presets
    BATCH_INFOSPHERE_CONFIG,
    DEFAULT_INFOSPHERE_CONFIG,
    DEVELOPMENT_INFOSPHERE_CONFIG,
    PRODUCTION_INFOSPHERE_CONFIG,
)


# =============================================================================
# IBMInfoSphereConfig Tests
# =============================================================================


class TestIBMInfoSphereConfig:
    """Tests for IBMInfoSphereConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IBMInfoSphereConfig()

        assert config.project == "default"
        assert config.host_name == ""
        assert config.execution_mode == InfoSphereExecutionMode.SYNCHRONOUS
        assert config.sampling_enabled is True
        assert config.sample_size == 10000
        assert config.enable_lineage is False

    def test_frozen_immutability(self):
        """Test that config is immutable."""
        config = IBMInfoSphereConfig()

        with pytest.raises(FrozenInstanceError):
            config.project = "new_project"

    def test_with_host_name(self):
        """Test builder method for host name."""
        config = IBMInfoSphereConfig()
        new_config = config.with_host_name("iis-server.example.com")

        assert new_config.host_name == "iis-server.example.com"
        assert config.host_name == ""  # Original unchanged

    def test_with_project(self):
        """Test builder method for project."""
        config = IBMInfoSphereConfig()
        new_config = config.with_project("DataQuality", folder="Analysis")

        assert new_config.project == "DataQuality"
        assert new_config.folder == "Analysis"

    def test_with_data_store(self):
        """Test builder method for data store."""
        config = IBMInfoSphereConfig()
        new_config = config.with_data_store("MainStore", schema_name="dbo")

        assert new_config.data_store == "MainStore"
        assert new_config.schema_name == "dbo"

    def test_with_execution_mode(self):
        """Test builder method for execution mode."""
        config = IBMInfoSphereConfig()
        new_config = config.with_execution_mode(InfoSphereExecutionMode.ASYNCHRONOUS)

        assert new_config.execution_mode == InfoSphereExecutionMode.ASYNCHRONOUS

    def test_with_qualitystage(self):
        """Test builder method for QualityStage settings."""
        config = IBMInfoSphereConfig()
        new_config = config.with_qualitystage("DataCleaningSuite", job_name="CleanJob")

        assert new_config.suite_name == "DataCleaningSuite"
        assert new_config.job_name == "CleanJob"

    def test_with_sampling(self):
        """Test builder method for sampling settings."""
        config = IBMInfoSphereConfig()
        new_config = config.with_sampling(enabled=False)

        assert new_config.sampling_enabled is False

        new_config_with_size = config.with_sampling(enabled=True, sample_size=50000)
        assert new_config_with_size.sampling_enabled is True
        assert new_config_with_size.sample_size == 50000

    def test_with_async_settings(self):
        """Test builder method for async settings."""
        config = IBMInfoSphereConfig()
        new_config = config.with_async_settings(poll_interval=5.0, max_wait=600.0)

        assert new_config.async_poll_interval == 5.0
        assert new_config.max_async_wait == 600.0

    def test_builder_chaining(self):
        """Test chaining multiple builder methods."""
        config = (
            IBMInfoSphereConfig()
            .with_api_endpoint("https://iis.example.com/ibm/iis/ia/api/v1")
            .with_host_name("iis-server.example.com")
            .with_project("DataQuality", folder="Production")
            .with_data_store("MainStore", schema_name="analytics")
            .with_execution_mode(InfoSphereExecutionMode.BATCH)
            .with_sampling(enabled=True, sample_size=25000)
        )

        assert config.api_endpoint == "https://iis.example.com/ibm/iis/ia/api/v1"
        assert config.host_name == "iis-server.example.com"
        assert config.project == "DataQuality"
        assert config.folder == "Production"
        assert config.data_store == "MainStore"
        assert config.schema_name == "analytics"
        assert config.execution_mode == InfoSphereExecutionMode.BATCH
        assert config.sampling_enabled is True
        assert config.sample_size == 25000


class TestIBMInfoSpherePresetConfigs:
    """Tests for IBM InfoSphere preset configurations."""

    def test_default_config(self):
        """Test default IBM InfoSphere config."""
        assert DEFAULT_INFOSPHERE_CONFIG.timeout_seconds == 120.0
        assert DEFAULT_INFOSPHERE_CONFIG.max_retries == 3
        assert DEFAULT_INFOSPHERE_CONFIG.auth_type == AuthType.BASIC

    def test_production_config(self):
        """Test production IBM InfoSphere config."""
        assert PRODUCTION_INFOSPHERE_CONFIG.execution_mode == (
            InfoSphereExecutionMode.ASYNCHRONOUS
        )
        assert PRODUCTION_INFOSPHERE_CONFIG.verify_ssl is True
        assert PRODUCTION_INFOSPHERE_CONFIG.enable_lineage is True
        assert PRODUCTION_INFOSPHERE_CONFIG.sampling_enabled is False
        assert PRODUCTION_INFOSPHERE_CONFIG.pool_size == 10

    def test_development_config(self):
        """Test development IBM InfoSphere config."""
        assert DEVELOPMENT_INFOSPHERE_CONFIG.verify_ssl is False
        assert DEVELOPMENT_INFOSPHERE_CONFIG.max_retries == 1
        assert DEVELOPMENT_INFOSPHERE_CONFIG.sampling_enabled is True
        assert DEVELOPMENT_INFOSPHERE_CONFIG.sample_size == 1000
        assert DEVELOPMENT_INFOSPHERE_CONFIG.execution_mode == (
            InfoSphereExecutionMode.SYNCHRONOUS
        )

    def test_batch_config(self):
        """Test batch IBM InfoSphere config."""
        assert BATCH_INFOSPHERE_CONFIG.execution_mode == InfoSphereExecutionMode.BATCH
        assert BATCH_INFOSPHERE_CONFIG.timeout_seconds == 600.0
        assert BATCH_INFOSPHERE_CONFIG.max_async_wait == 1800.0  # 30 minutes
        assert BATCH_INFOSPHERE_CONFIG.sampling_enabled is False


# =============================================================================
# Enums Tests
# =============================================================================


class TestInfoSphereEnums:
    """Tests for IBM InfoSphere enums."""

    def test_analysis_type_values(self):
        """Test InfoSphereAnalysisType enum values."""
        assert InfoSphereAnalysisType.COLUMN_ANALYSIS.value == "COLUMN_ANALYSIS"
        assert InfoSphereAnalysisType.KEY_ANALYSIS.value == "KEY_ANALYSIS"
        assert (
            InfoSphereAnalysisType.REFERENTIAL_INTEGRITY_ANALYSIS.value
            == "REFERENTIAL_INTEGRITY_ANALYSIS"
        )

    def test_rule_type_values(self):
        """Test InfoSphereRuleType enum values."""
        assert InfoSphereRuleType.NULL_CHECK.value == "NULL_CHECK"
        assert InfoSphereRuleType.UNIQUENESS.value == "UNIQUENESS"
        assert InfoSphereRuleType.DOMAIN_CHECK.value == "DOMAIN_CHECK"
        assert InfoSphereRuleType.PATTERN_CHECK.value == "PATTERN_CHECK"

    def test_execution_mode_values(self):
        """Test InfoSphereExecutionMode enum values."""
        assert InfoSphereExecutionMode.SYNCHRONOUS.value == "SYNCHRONOUS"
        assert InfoSphereExecutionMode.ASYNCHRONOUS.value == "ASYNCHRONOUS"
        assert InfoSphereExecutionMode.BATCH.value == "BATCH"


# =============================================================================
# IBMInfoSphereRuleTranslator Tests
# =============================================================================


class TestIBMInfoSphereRuleTranslator:
    """Tests for IBMInfoSphereRuleTranslator."""

    def test_supported_rule_types(self):
        """Test getting supported rule types."""
        translator = IBMInfoSphereRuleTranslator()
        types = translator.get_supported_rule_types()

        assert "not_null" in types
        assert "unique" in types
        assert "in_set" in types
        assert "in_range" in types
        assert "regex" in types
        assert "dtype" in types
        assert "min_length" in types
        assert "max_length" in types
        assert "foreign_key" in types

    def test_translate_not_null(self):
        """Test translating not_null rule."""
        translator = IBMInfoSphereRuleTranslator()
        rule = {"type": "not_null", "column": "customer_id"}

        translated = translator.translate(rule)

        assert translated["type"] == "NULL_CHECK"
        assert translated["binding"]["column"] == "customer_id"
        assert translated["parameters"]["allowEmpty"] is False
        assert translated["parameters"]["treatWhitespaceAsNull"] is True

    def test_translate_unique(self):
        """Test translating unique rule."""
        translator = IBMInfoSphereRuleTranslator()
        rule = {"type": "unique", "column": "email"}

        translated = translator.translate(rule)

        assert translated["type"] == "UNIQUENESS"
        assert translated["binding"]["column"] == "email"
        assert translated["parameters"]["caseSensitive"] is True
        assert translated["parameters"]["includeNulls"] is False

    def test_translate_in_set(self):
        """Test translating in_set rule."""
        translator = IBMInfoSphereRuleTranslator()
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
        translator = IBMInfoSphereRuleTranslator()
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
        translator = IBMInfoSphereRuleTranslator()
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
        translator = IBMInfoSphereRuleTranslator()
        rule = {
            "type": "regex",
            "column": "phone",
            "pattern": r"^\d{3}-\d{4}$",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "PATTERN_CHECK"
        assert translated["binding"]["column"] == "phone"
        assert translated["parameters"]["pattern"] == r"^\d{3}-\d{4}$"
        assert translated["parameters"]["patternType"] == "REGEX"

    def test_translate_dtype(self):
        """Test translating dtype rule."""
        translator = IBMInfoSphereRuleTranslator()
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
        translator = IBMInfoSphereRuleTranslator()

        # length rule with min_length specified
        rule = {"type": "length", "column": "name", "min": 2}
        translated = translator.translate(rule)
        assert translated["type"] == "LENGTH_CHECK"
        assert translated["parameters"]["minLength"] == 2

        # length rule with max_length specified
        rule = {"type": "length", "column": "name", "max": 100}
        translated = translator.translate(rule)
        assert translated["type"] == "LENGTH_CHECK"
        assert translated["parameters"]["maxLength"] == 100

        # length rule with exact length
        rule = {"type": "length", "column": "name", "length": 10}
        translated = translator.translate(rule)
        assert translated["type"] == "LENGTH_CHECK"
        assert translated["parameters"]["exactLength"] == 10

    def test_translate_foreign_key(self):
        """Test translating foreign_key rule."""
        translator = IBMInfoSphereRuleTranslator()
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
        translator = IBMInfoSphereRuleTranslator()
        rule = {
            "type": "expression",
            "column": "total",
            "expression": "quantity * price == total",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "EXPRESSION_RULE"
        assert translated["parameters"]["expression"] == "quantity * price == total"
        assert translated["parameters"]["language"] == "INFOSPHERE"

    def test_translate_sql_rule(self):
        """Test translating SQL rule."""
        translator = IBMInfoSphereRuleTranslator()
        rule = {
            "type": "sql",
            "column": "id",
            "sql": "SELECT id FROM table WHERE status = 'active'",
        }

        translated = translator.translate(rule)

        assert translated["type"] == "SQL_RULE"
        assert (
            translated["parameters"]["sqlQuery"]
            == "SELECT id FROM table WHERE status = 'active'"
        )

    def test_translate_multiple_rules(self):
        """Test translating multiple rules."""
        translator = IBMInfoSphereRuleTranslator()
        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
            {"type": "in_range", "column": "age", "min": 0, "max": 120},
        ]

        translated = translator.translate_batch(rules)

        assert len(translated) == 3
        assert translated[0]["type"] == "NULL_CHECK"
        assert translated[1]["type"] == "UNIQUENESS"
        assert translated[2]["type"] == "RANGE_CHECK"

    def test_severity_mapping(self):
        """Test severity mapping."""
        translator = IBMInfoSphereRuleTranslator()

        assert translator._map_severity("critical") == "CRITICAL"
        assert translator._map_severity("high") == "HIGH"
        assert translator._map_severity("error") == "HIGH"
        assert translator._map_severity("medium") == "MEDIUM"
        assert translator._map_severity("warning") == "MEDIUM"
        assert translator._map_severity("low") == "LOW"
        assert translator._map_severity("info") == "INFORMATIONAL"
        assert translator._map_severity("unknown") == "MEDIUM"  # Default

    def test_dtype_mapping(self):
        """Test data type mapping."""
        translator = IBMInfoSphereRuleTranslator()

        assert translator._map_dtype("string") == "VARCHAR"
        assert translator._map_dtype("int") == "INTEGER"
        assert translator._map_dtype("int64") == "BIGINT"
        assert translator._map_dtype("float64") == "DOUBLE"
        assert translator._map_dtype("bool") == "BOOLEAN"
        assert translator._map_dtype("datetime") == "TIMESTAMP"
        assert translator._map_dtype("unknown") == "VARCHAR"  # Default


# =============================================================================
# IBMInfoSphereResultConverter Tests
# =============================================================================


class TestIBMInfoSphereResultConverter:
    """Tests for IBMInfoSphereResultConverter."""

    def test_convert_successful_result(self):
        """Test converting successful check result."""
        converter = IBMInfoSphereResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "ruleExecutions": [
                {
                    "ruleName": "NullCheck_customer_id",
                    "binding": {"column": "customer_id"},
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
        assert result.passed_count == 1
        assert result.failed_count == 0

    def test_convert_failed_result(self):
        """Test converting failed check result."""
        converter = IBMInfoSphereResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "ruleExecutions": [
                {
                    "ruleName": "NullCheck_email",
                    "binding": {"column": "email"},
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
        assert result.failures[0].message == "10 null values found"

    def test_convert_analysis_results_format(self):
        """Test converting Information Analyzer results format."""
        converter = IBMInfoSphereResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "analysisResults": [
                {
                    "analysisType": "COLUMN_ANALYSIS",
                    "ruleResults": [
                        {
                            "ruleName": "CompletenessCheck",
                            "columnName": "name",
                            "qualityScore": 85,  # Below 90 = failed
                            "exceptionCount": 15,
                            "description": "Completeness check failed",
                            "severity": "HIGH",
                        }
                    ],
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
        converter = IBMInfoSphereResultConverter()
        start_time = time.perf_counter()

        # Passing quality score
        vendor_result = {
            "qualityScore": 95,
            "threshold": 90,
            "ruleName": "OverallQuality",
            "exceptionCount": 0,
            "message": "Quality threshold met",
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
        """Test severity mapping from InfoSphere to common."""
        converter = IBMInfoSphereResultConverter()

        assert converter.SEVERITY_MAPPING.get("CRITICAL") == Severity.CRITICAL
        assert converter.SEVERITY_MAPPING.get("HIGH") == Severity.ERROR
        assert converter.SEVERITY_MAPPING.get("MEDIUM") == Severity.WARNING
        assert converter.SEVERITY_MAPPING.get("LOW") == Severity.INFO
        assert converter.SEVERITY_MAPPING.get("INFORMATIONAL") == Severity.INFO

    def test_convert_profile_result(self):
        """Test converting profile result."""
        converter = IBMInfoSphereResultConverter()
        start_time = time.perf_counter()

        vendor_result = {
            "rowCount": 1000,
            "columnAnalysis": [
                {
                    "columnName": "id",
                    "inferredDataType": "INTEGER",
                    "nullCount": 0,
                    "nullPercentage": 0.0,
                    "distinctCount": 1000,
                    "minValue": 1,
                    "maxValue": 1000,
                },
                {
                    "columnName": "name",
                    "inferredDataType": "VARCHAR",
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
# IBMInfoSphereConnectionManager Tests
# =============================================================================


class TestIBMInfoSphereConnectionManager:
    """Tests for IBMInfoSphereConnectionManager."""

    def test_init(self):
        """Test connection manager initialization."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
        )
        manager = IBMInfoSphereConnectionManager(config)

        assert manager._config == config
        assert manager._session is None

    def test_connect_with_basic_auth(self):
        """Test connection with basic authentication."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
            username="admin",
            password="secret",
            auth_type=AuthType.BASIC,
        )
        manager = IBMInfoSphereConnectionManager(config)

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            # Mock login response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_session.post.return_value = mock_response
            mock_session.get.return_value = mock_response

            manager._do_connect()

            assert manager._session is not None

    def test_execute_api_call(self):
        """Test executing API call."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
        )
        manager = IBMInfoSphereConnectionManager(config)

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_session.request.return_value = mock_response

        manager._session = mock_session

        with patch.object(manager, "get_connection", return_value=mock_session):
            result = manager.execute_api_call("GET", "/projects")

        assert result == {"status": "success"}
        mock_session.request.assert_called_once()


# =============================================================================
# IBMInfoSphereAdapter Tests
# =============================================================================


class TestIBMInfoSphereAdapter:
    """Tests for IBMInfoSphereAdapter."""

    def test_engine_name(self):
        """Test engine name property."""
        adapter = IBMInfoSphereAdapter()
        assert adapter.engine_name == "ibm_infosphere"

    def test_engine_version(self):
        """Test engine version property."""
        adapter = IBMInfoSphereAdapter()
        assert adapter.engine_version  # Should return a version string

    def test_capabilities(self):
        """Test engine capabilities."""
        adapter = IBMInfoSphereAdapter()
        caps = adapter.get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is False  # InfoSphere doesn't support learn
        assert "polars" in caps.supported_data_types
        assert "pandas" in caps.supported_data_types
        assert "db" in caps.supported_data_types

    def test_get_info(self):
        """Test getting engine info."""
        adapter = IBMInfoSphereAdapter()
        info = adapter.get_info()

        assert info.name == "ibm_infosphere"
        assert "ibm" in info.description.lower() or "infosphere" in info.description.lower()

    def test_config_assignment(self):
        """Test that config is properly assigned."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://test.example.com",
            project="TestProject",
        )
        adapter = IBMInfoSphereAdapter(config=config)

        assert adapter._config.api_endpoint == "https://test.example.com"
        assert adapter._config.project == "TestProject"

    def test_default_config(self):
        """Test adapter uses default config when none provided."""
        adapter = IBMInfoSphereAdapter()

        assert adapter._config is not None
        assert isinstance(adapter._config, IBMInfoSphereConfig)


class TestIBMInfoSphereAdapterLifecycle:
    """Tests for IBMInfoSphereAdapter lifecycle management."""

    def test_context_manager(self):
        """Test context manager usage with mocked connection."""
        adapter = IBMInfoSphereAdapter()

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

        adapter = IBMInfoSphereAdapter()

        # Initially CREATED
        assert adapter.get_state() == EngineState.CREATED


class TestIBMInfoSphereAdapterMethods:
    """Tests for IBMInfoSphereAdapter specific methods."""

    def test_build_check_payload(self):
        """Test building check payload."""
        pytest.importorskip("pandas")
        import pandas as pd

        config = IBMInfoSphereConfig(
            project="TestProject",
            folder="Production",
            data_store="MainStore",
            schema_name="dbo",
            suite_name="QualitySuite",
            sampling_enabled=True,
            sample_size=5000,
        )
        adapter = IBMInfoSphereAdapter(config=config)

        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        rules = [{"type": "NULL_CHECK", "binding": {"column": "id"}}]

        payload = adapter._build_check_payload(df, rules)

        assert payload["project"] == "TestProject"
        assert payload["folder"] == "Production"
        assert payload["dataStore"] == "MainStore"
        assert payload["schema"] == "dbo"
        assert payload["suiteName"] == "QualitySuite"
        assert payload["rules"] == rules
        assert payload["sampling"]["enabled"] is True
        assert payload["sampling"]["sampleSize"] == 5000

    def test_build_profile_payload(self):
        """Test building profile payload."""
        pytest.importorskip("pandas")
        import pandas as pd

        config = IBMInfoSphereConfig(
            project="TestProject",
            analysis_project="AnalysisProject",
            sampling_enabled=True,
            sample_size=10000,
        )
        adapter = IBMInfoSphereAdapter(config=config)

        df = pd.DataFrame({"id": [1, 2, 3]})

        payload = adapter._build_profile_payload(df)

        assert payload["project"] == "AnalysisProject"
        assert payload["options"]["computeStatistics"] is True
        assert payload["options"]["inferDataTypes"] is True
        assert payload["sampling"]["enabled"] is True
        assert payload["sampling"]["sampleSize"] == 10000

    def test_serialize_data(self):
        """Test data serialization."""
        pytest.importorskip("pandas")
        import pandas as pd

        adapter = IBMInfoSphereAdapter()
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        serialized = adapter._serialize_data(df)

        assert serialized["rowCount"] == 2
        assert len(serialized["columns"]) == 2
        assert serialized["columns"][0]["name"] == "id"
        assert len(serialized["rows"]) == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateIBMInfoSphereAdapter:
    """Tests for create_ibm_infosphere_adapter factory function."""

    def test_create_with_api_key(self):
        """Test creating adapter with API key."""
        adapter = create_ibm_infosphere_adapter(
            api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
            api_key="secret-key",
            project="Production",
        )

        assert isinstance(adapter, IBMInfoSphereAdapter)
        assert (
            adapter._config.api_endpoint
            == "https://iis.example.com/ibm/iis/ia/api/v1"
        )
        assert adapter._config.api_key == "secret-key"
        assert adapter._config.project == "Production"
        assert adapter._config.auth_type == AuthType.API_KEY

    def test_create_with_username_password(self):
        """Test creating adapter with username/password."""
        adapter = create_ibm_infosphere_adapter(
            api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
            username="admin",
            password="secret",
        )

        assert isinstance(adapter, IBMInfoSphereAdapter)
        assert adapter._config.username == "admin"
        assert adapter._config.password == "secret"
        assert adapter._config.auth_type == AuthType.BASIC

    def test_create_with_additional_kwargs(self):
        """Test creating adapter with additional kwargs."""
        adapter = create_ibm_infosphere_adapter(
            api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
            username="admin",
            password="secret",
            project="MyProject",
            host_name="iis-server.example.com",
            data_store="MainStore",
        )

        assert adapter._config.project == "MyProject"
        assert adapter._config.host_name == "iis-server.example.com"
        assert adapter._config.data_store == "MainStore"


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestIBMInfoSphereAdapterIntegration:
    """Integration tests with mocked API calls."""

    def test_check_with_mock_api(self):
        """Test check operation with mocked API."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = IBMInfoSphereAdapter(
            config=IBMInfoSphereConfig(
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
                "ruleExecutions": [
                    {
                        "ruleName": "NullCheck",
                        "binding": {"column": "name"},
                        "passed": False,
                        "failedRecordCount": 1,
                        "message": "1 null value found",
                        "severity": "HIGH",
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

        adapter = IBMInfoSphereAdapter(
            config=IBMInfoSphereConfig(
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
                "columnAnalysis": [
                    {
                        "columnName": "id",
                        "inferredDataType": "INTEGER",
                        "nullCount": 0,
                        "nullPercentage": 0.0,
                        "distinctCount": 3,
                    },
                    {
                        "columnName": "value",
                        "inferredDataType": "DOUBLE",
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


class TestIBMInfoSphereAsyncExecution:
    """Tests for async execution mode."""

    def test_async_job_execution(self):
        """Test async job submission and polling."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://iis.example.com",
            execution_mode=InfoSphereExecutionMode.ASYNCHRONOUS,
            async_poll_interval=0.1,  # Fast polling for test
            max_async_wait=1.0,
        )
        adapter = IBMInfoSphereAdapter(config=config)

        mock_connection = MagicMock()

        # First call returns job ID, subsequent calls return status/result
        mock_connection.execute_api_call.side_effect = [
            {"jobId": "job123"},  # Submit
            {"status": "RUNNING"},  # First poll
            {"status": "COMPLETED"},  # Second poll
            {"ruleExecutions": []},  # Get result
        ]

        result = adapter._execute_async_job(
            mock_connection, "dataRule/execute", {"rules": []}
        )

        assert result == {"ruleExecutions": []}
        assert mock_connection.execute_api_call.call_count == 4

    def test_async_job_timeout(self):
        """Test async job timeout."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://iis.example.com",
            execution_mode=InfoSphereExecutionMode.ASYNCHRONOUS,
            async_poll_interval=0.1,
            max_async_wait=0.3,  # Short timeout
        )
        adapter = IBMInfoSphereAdapter(config=config)

        mock_connection = MagicMock()
        mock_connection.execute_api_call.side_effect = [
            {"jobId": "job123"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
        ]

        with pytest.raises(VendorSDKError) as exc_info:
            adapter._execute_async_job(
                mock_connection, "dataRule/execute", {"rules": []}
            )

        assert "timed out" in str(exc_info.value).lower()

    def test_async_job_failure(self):
        """Test async job failure handling."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://iis.example.com",
            execution_mode=InfoSphereExecutionMode.ASYNCHRONOUS,
            async_poll_interval=0.1,
        )
        adapter = IBMInfoSphereAdapter(config=config)

        mock_connection = MagicMock()
        mock_connection.execute_api_call.side_effect = [
            {"jobId": "job123"},
            {"status": "FAILED", "errorMessage": "Rule execution error"},
        ]

        with pytest.raises(VendorSDKError) as exc_info:
            adapter._execute_async_job(
                mock_connection, "dataRule/execute", {"rules": []}
            )

        assert "failed" in str(exc_info.value).lower()


class TestIBMInfoSphereBatchExecution:
    """Tests for batch execution mode."""

    def test_batch_job_execution(self):
        """Test batch job submission and polling."""
        config = IBMInfoSphereConfig(
            api_endpoint="https://iis.example.com",
            execution_mode=InfoSphereExecutionMode.BATCH,
            async_poll_interval=0.05,  # Very short for testing
            max_async_wait=10.0,  # Longer to accommodate polling
        )
        adapter = IBMInfoSphereAdapter(config=config)

        mock_connection = MagicMock()
        mock_connection.execute_api_call.side_effect = [
            {"batchId": "batch123"},
            {"status": "COMPLETED"},  # Completes on first poll
            {"ruleExecutions": []},  # Get result
        ]

        result = adapter._execute_batch_job(
            mock_connection, "dataRule/executeBatch", {"rules": []}
        )

        assert result == {"ruleExecutions": []}


# =============================================================================
# Additional Analysis Methods Tests
# =============================================================================


class TestIBMInfoSphereAnalysisMethods:
    """Tests for InfoSphere-specific analysis methods."""

    def test_run_column_analysis(self):
        """Test column analysis method."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = IBMInfoSphereAdapter()
        df = pl.DataFrame({"id": [1, 2, 3]})

        with patch.object(adapter, "profile") as mock_profile:
            mock_profile.return_value = MagicMock()

            adapter.run_column_analysis(df, columns=["id"])

            # Verify profile was called with COLUMN_ANALYSIS type
            call_kwargs = mock_profile.call_args[1]
            assert call_kwargs.get("analysis_type") == (
                InfoSphereAnalysisType.COLUMN_ANALYSIS
            )
            assert call_kwargs.get("columns") == ["id"]

    def test_run_key_analysis(self):
        """Test key analysis method."""
        pytest.importorskip("polars")
        import polars as pl

        adapter = IBMInfoSphereAdapter()
        df = pl.DataFrame({"id": [1, 2, 3], "code": ["A", "B", "C"]})

        with patch.object(adapter, "profile") as mock_profile:
            mock_profile.return_value = MagicMock()

            adapter.run_key_analysis(df, candidate_keys=[["id"], ["id", "code"]])

            call_kwargs = mock_profile.call_args[1]
            assert call_kwargs.get("analysis_type") == InfoSphereAnalysisType.KEY_ANALYSIS
            assert call_kwargs.get("candidateKeys") == [["id"], ["id", "code"]]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestIBMInfoSphereEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_rules_list(self):
        """Test handling empty rules list."""
        translator = IBMInfoSphereRuleTranslator()
        translated = translator.translate_batch([])
        assert translated == []

    def test_unknown_rule_type(self):
        """Test handling unknown rule type."""
        from packages.enterprise.engines.base import RuleTranslationError

        translator = IBMInfoSphereRuleTranslator()
        rule = {"type": "unknown_type", "column": "id"}

        # Should raise RuleTranslationError for unknown types
        with pytest.raises(RuleTranslationError):
            translator.translate(rule)

    def test_missing_column_in_rule(self):
        """Test handling rule with missing column."""
        translator = IBMInfoSphereRuleTranslator()
        rule = {"type": "not_null"}  # Missing column

        translated = translator.translate(rule)
        # Should handle gracefully with empty column
        assert translated["binding"]["column"] == ""

    def test_empty_vendor_result(self):
        """Test handling empty vendor result."""
        converter = IBMInfoSphereResultConverter()
        start_time = time.perf_counter()

        vendor_result = {}
        result = converter.convert_check_result(
            vendor_result,
            start_time=start_time,
        )

        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 0
        assert result.failed_count == 0
