"""IBM InfoSphere Information Server Data Quality Adapter.

This module provides an adapter for IBM InfoSphere Information Server's
DataStage and QualityStage components for enterprise data quality operations.

IBM InfoSphere Information Server is a comprehensive platform for:
- Data Quality analysis and validation
- Data Profiling with statistical analysis
- Data Standardization and matching
- Data Integration through DataStage

The adapter translates common data quality rules to InfoSphere-specific
rule specifications and converts results back to the common format.

Example:
    >>> from packages.enterprise.engines import (
    ...     IBMInfoSphereAdapter,
    ...     IBMInfoSphereConfig,
    ... )
    >>>
    >>> config = IBMInfoSphereConfig(
    ...     api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
    ...     username="admin",
    ...     password="secret",
    ...     project="DataQuality",
    ... )
    >>> engine = IBMInfoSphereAdapter(config=config)
    >>>
    >>> with engine:
    ...     result = engine.check(data, rules)
    ...     if result.status == CheckStatus.FAILED:
    ...         for failure in result.failures:
    ...             print(f"{failure.column}: {failure.message}")

Note:
    This adapter requires either the ibm-information-server-sdk package
    or the requests library for HTTP API communication.

    Install with: pip install ibm-information-server-sdk
    Or fallback: pip install requests
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from common.base import (
    CheckStatus,
    ColumnProfile,
    ProfileResult,
    ProfileStatus,
    Severity,
)
from packages.enterprise.engines.base import (
    AuthType,
    BaseConnectionManager,
    BaseResultConverter,
    BaseRuleTranslator,
    ConnectionMode,
    DataTransferMode,
    EnterpriseEngineAdapter,
    EnterpriseEngineConfig,
    VendorSDKError,
)

# For type hints
try:
    from common.engines.base import EngineCapabilities
except ImportError:
    EngineCapabilities = Any  # type: ignore


# =============================================================================
# Enums
# =============================================================================


class InfoSphereAnalysisType(str, Enum):
    """IBM InfoSphere analysis types.

    These represent the different types of analysis that can be performed
    using InfoSphere Information Analyzer.
    """

    COLUMN_ANALYSIS = "COLUMN_ANALYSIS"
    KEY_ANALYSIS = "KEY_ANALYSIS"
    CROSS_DOMAIN_ANALYSIS = "CROSS_DOMAIN_ANALYSIS"
    REFERENTIAL_INTEGRITY_ANALYSIS = "REFERENTIAL_INTEGRITY_ANALYSIS"
    BASELINE_COMPARISON = "BASELINE_COMPARISON"
    DATA_RULE_ANALYSIS = "DATA_RULE_ANALYSIS"


class InfoSphereRuleType(str, Enum):
    """IBM InfoSphere data rule types.

    These represent the rule types supported by InfoSphere QualityStage.
    """

    # Completeness
    COMPLETENESS = "COMPLETENESS"
    NULL_CHECK = "NULL_CHECK"

    # Uniqueness
    UNIQUENESS = "UNIQUENESS"
    DUPLICATE_CHECK = "DUPLICATE_CHECK"

    # Validity
    DOMAIN_CHECK = "DOMAIN_CHECK"
    RANGE_CHECK = "RANGE_CHECK"
    PATTERN_CHECK = "PATTERN_CHECK"
    FORMAT_CHECK = "FORMAT_CHECK"
    DATA_TYPE_CHECK = "DATA_TYPE_CHECK"

    # Consistency
    CROSS_FIELD_VALIDATION = "CROSS_FIELD_VALIDATION"
    REFERENCE_CHECK = "REFERENCE_CHECK"

    # Length
    LENGTH_CHECK = "LENGTH_CHECK"

    # Custom
    EXPRESSION_RULE = "EXPRESSION_RULE"
    SQL_RULE = "SQL_RULE"


class InfoSphereExecutionMode(str, Enum):
    """InfoSphere job execution modes."""

    SYNCHRONOUS = "SYNCHRONOUS"
    ASYNCHRONOUS = "ASYNCHRONOUS"
    BATCH = "BATCH"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class IBMInfoSphereConfig(EnterpriseEngineConfig):
    """Configuration for IBM InfoSphere Information Server adapter.

    Extends EnterpriseEngineConfig with InfoSphere-specific settings.

    Attributes:
        host_name: InfoSphere server hostname.
        project: InfoSphere project name.
        folder: Folder within the project.
        data_store: Data store name for analysis.
        schema_name: Database schema for data operations.
        analysis_project: Project for Information Analyzer.
        execution_mode: Job execution mode.
        suite_name: QualityStage suite name.
        job_name: DataStage job name for execution.
        enable_lineage: Whether to enable data lineage tracking.
        sampling_enabled: Whether to use sampling for large datasets.
        sample_size: Sample size when sampling is enabled.
        async_poll_interval: Polling interval for async jobs.
        max_async_wait: Maximum wait time for async jobs.

    Example:
        >>> config = IBMInfoSphereConfig(
        ...     api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
        ...     username="admin",
        ...     password="secret",
        ...     project="DataQuality",
        ...     host_name="iis-server.example.com",
        ... )
        >>>
        >>> # Using builder pattern
        >>> config = config.with_project("NewProject")
        >>> config = config.with_execution_mode(InfoSphereExecutionMode.BATCH)
    """

    # Server settings
    host_name: str = ""

    # Project settings
    project: str = "default"
    folder: str = ""
    data_store: str = ""
    schema_name: str = ""

    # Information Analyzer settings
    analysis_project: str = ""

    # Execution settings
    execution_mode: InfoSphereExecutionMode = InfoSphereExecutionMode.SYNCHRONOUS

    # QualityStage settings
    suite_name: str = ""
    job_name: str = ""

    # Feature flags
    enable_lineage: bool = False
    sampling_enabled: bool = True
    sample_size: int = 10000

    # Async settings
    async_poll_interval: float = 2.0
    max_async_wait: float = 300.0

    # ==========================================================================
    # Builder Methods
    # ==========================================================================

    def with_host_name(self, host_name: str) -> "IBMInfoSphereConfig":
        """Return new config with updated host name.

        Args:
            host_name: InfoSphere server hostname.

        Returns:
            New configuration instance.
        """
        return IBMInfoSphereConfig(
            **{**self._to_dict(), "host_name": host_name}
        )

    def with_project(self, project: str, folder: str = "") -> "IBMInfoSphereConfig":
        """Return new config with updated project settings.

        Args:
            project: Project name.
            folder: Optional folder within project.

        Returns:
            New configuration instance.
        """
        updates = {"project": project}
        if folder:
            updates["folder"] = folder
        return IBMInfoSphereConfig(**{**self._to_dict(), **updates})

    def with_data_store(
        self,
        data_store: str,
        schema_name: str = "",
    ) -> "IBMInfoSphereConfig":
        """Return new config with updated data store settings.

        Args:
            data_store: Data store name.
            schema_name: Optional schema name.

        Returns:
            New configuration instance.
        """
        updates = {"data_store": data_store}
        if schema_name:
            updates["schema_name"] = schema_name
        return IBMInfoSphereConfig(**{**self._to_dict(), **updates})

    def with_execution_mode(
        self,
        mode: InfoSphereExecutionMode,
    ) -> "IBMInfoSphereConfig":
        """Return new config with updated execution mode.

        Args:
            mode: Execution mode.

        Returns:
            New configuration instance.
        """
        return IBMInfoSphereConfig(
            **{**self._to_dict(), "execution_mode": mode}
        )

    def with_qualitystage(
        self,
        suite_name: str,
        job_name: str = "",
    ) -> "IBMInfoSphereConfig":
        """Return new config with QualityStage settings.

        Args:
            suite_name: QualityStage suite name.
            job_name: Optional job name.

        Returns:
            New configuration instance.
        """
        updates = {"suite_name": suite_name}
        if job_name:
            updates["job_name"] = job_name
        return IBMInfoSphereConfig(**{**self._to_dict(), **updates})

    def with_sampling(
        self,
        enabled: bool = True,
        sample_size: int = 10000,
    ) -> "IBMInfoSphereConfig":
        """Return new config with sampling settings.

        Args:
            enabled: Whether to enable sampling.
            sample_size: Sample size when enabled.

        Returns:
            New configuration instance.
        """
        return IBMInfoSphereConfig(
            **{
                **self._to_dict(),
                "sampling_enabled": enabled,
                "sample_size": sample_size,
            }
        )

    def with_async_settings(
        self,
        poll_interval: float = 2.0,
        max_wait: float = 300.0,
    ) -> "IBMInfoSphereConfig":
        """Return new config with async execution settings.

        Args:
            poll_interval: Polling interval in seconds.
            max_wait: Maximum wait time in seconds.

        Returns:
            New configuration instance.
        """
        return IBMInfoSphereConfig(
            **{
                **self._to_dict(),
                "async_poll_interval": poll_interval,
                "max_async_wait": max_wait,
            }
        )

    def _to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for updates.

        Returns:
            Dictionary representation of config.
        """
        return {
            # Base config fields
            "api_endpoint": self.api_endpoint,
            "api_key": self.api_key,
            "username": self.username,
            "password": self.password,
            "auth_type": self.auth_type,
            "connection_mode": self.connection_mode,
            "data_transfer_mode": self.data_transfer_mode,
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "pool_size": self.pool_size,
            "verify_ssl": self.verify_ssl,
            "proxy_url": self.proxy_url,
            "vendor_options": self.vendor_options,
            "auto_start": self.auto_start,
            "auto_stop": self.auto_stop,
            # InfoSphere-specific fields
            "host_name": self.host_name,
            "project": self.project,
            "folder": self.folder,
            "data_store": self.data_store,
            "schema_name": self.schema_name,
            "analysis_project": self.analysis_project,
            "execution_mode": self.execution_mode,
            "suite_name": self.suite_name,
            "job_name": self.job_name,
            "enable_lineage": self.enable_lineage,
            "sampling_enabled": self.sampling_enabled,
            "sample_size": self.sample_size,
            "async_poll_interval": self.async_poll_interval,
            "max_async_wait": self.max_async_wait,
        }


# =============================================================================
# Preset Configurations
# =============================================================================


DEFAULT_INFOSPHERE_CONFIG = IBMInfoSphereConfig(
    timeout_seconds=120.0,
    connect_timeout_seconds=30.0,
    max_retries=3,
    retry_delay_seconds=1.0,
    auth_type=AuthType.BASIC,
)
"""Default configuration for IBM InfoSphere."""


PRODUCTION_INFOSPHERE_CONFIG = IBMInfoSphereConfig(
    timeout_seconds=300.0,
    connect_timeout_seconds=60.0,
    max_retries=5,
    retry_delay_seconds=2.0,
    pool_size=10,
    auth_type=AuthType.BASIC,
    verify_ssl=True,
    execution_mode=InfoSphereExecutionMode.ASYNCHRONOUS,
    enable_lineage=True,
    sampling_enabled=False,
)
"""Production-optimized configuration with better reliability."""


DEVELOPMENT_INFOSPHERE_CONFIG = IBMInfoSphereConfig(
    timeout_seconds=60.0,
    connect_timeout_seconds=15.0,
    max_retries=1,
    retry_delay_seconds=0.5,
    pool_size=2,
    auth_type=AuthType.BASIC,
    verify_ssl=False,
    execution_mode=InfoSphereExecutionMode.SYNCHRONOUS,
    sampling_enabled=True,
    sample_size=1000,
)
"""Development configuration with faster feedback."""


BATCH_INFOSPHERE_CONFIG = IBMInfoSphereConfig(
    timeout_seconds=600.0,
    connect_timeout_seconds=60.0,
    max_retries=5,
    retry_delay_seconds=5.0,
    pool_size=5,
    auth_type=AuthType.BASIC,
    execution_mode=InfoSphereExecutionMode.BATCH,
    sampling_enabled=False,
    async_poll_interval=5.0,
    max_async_wait=1800.0,  # 30 minutes
)
"""Batch processing configuration for large-scale operations."""


# =============================================================================
# Rule Translator
# =============================================================================


class IBMInfoSphereRuleTranslator(BaseRuleTranslator):
    """Translates common rules to IBM InfoSphere format.

    InfoSphere uses a rule-based specification format with typed rules
    and structured parameters.

    Example rule translation:
        Common: {"type": "not_null", "column": "id"}
        InfoSphere: {
            "ruleType": "NULL_CHECK",
            "binding": {"column": "id"},
            "severity": "HIGH",
            "parameters": {"allowEmpty": False}
        }
    """

    # Maps common rule types to InfoSphere rule types
    RULE_MAPPING: dict[str, str] = {
        # Completeness rules
        "not_null": "NULL_CHECK",
        "null": "NULL_CHECK",
        "notnull": "NULL_CHECK",
        "completeness": "COMPLETENESS",

        # Uniqueness rules
        "unique": "UNIQUENESS",
        "duplicate": "DUPLICATE_CHECK",

        # Domain rules
        "in_set": "DOMAIN_CHECK",
        "in_list": "DOMAIN_CHECK",
        "domain": "DOMAIN_CHECK",

        # Range rules
        "in_range": "RANGE_CHECK",
        "between": "RANGE_CHECK",
        "range": "RANGE_CHECK",
        "greater_than": "RANGE_CHECK",
        "less_than": "RANGE_CHECK",

        # Pattern rules
        "regex": "PATTERN_CHECK",
        "pattern": "PATTERN_CHECK",
        "format": "FORMAT_CHECK",

        # Type rules
        "dtype": "DATA_TYPE_CHECK",
        "type": "DATA_TYPE_CHECK",

        # Length rules
        "min_length": "LENGTH_CHECK",
        "max_length": "LENGTH_CHECK",
        "length": "LENGTH_CHECK",

        # Referential rules
        "foreign_key": "REFERENCE_CHECK",
        "reference": "REFERENCE_CHECK",

        # Cross-field rules
        "cross_field": "CROSS_FIELD_VALIDATION",
        "consistency": "CROSS_FIELD_VALIDATION",

        # Custom rules
        "expression": "EXPRESSION_RULE",
        "sql": "SQL_RULE",
        "custom": "EXPRESSION_RULE",
    }

    def _get_rule_mapping(self) -> dict[str, str]:
        """Return rule type mapping.

        Returns:
            Mapping from common to InfoSphere rule types.
        """
        return self.RULE_MAPPING

    def _translate_rule_params(
        self,
        rule_type: str,
        rule: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate rule parameters to InfoSphere format.

        Args:
            rule_type: InfoSphere rule type.
            rule: Common rule dictionary.

        Returns:
            InfoSphere-format parameters dictionary.
        """
        params: dict[str, Any] = {}

        # Common parameters
        column = rule.get("column", rule.get("field", ""))
        params["binding"] = {"column": column}

        # Map severity
        severity = rule.get("severity", "medium")
        params["severity"] = self._map_severity(severity)

        # Rule-specific parameters
        if rule_type == "NULL_CHECK":
            params["parameters"] = {
                "allowEmpty": rule.get("allow_empty", False),
                "treatWhitespaceAsNull": rule.get("treat_whitespace_as_null", True),
            }

        elif rule_type == "UNIQUENESS":
            params["parameters"] = {
                "caseSensitive": rule.get("case_sensitive", True),
                "includeNulls": rule.get("include_nulls", False),
            }

        elif rule_type == "DOMAIN_CHECK":
            values = rule.get("values", rule.get("allowed_values", []))
            params["parameters"] = {
                "validValues": list(values),
                "caseSensitive": rule.get("case_sensitive", True),
            }

        elif rule_type == "RANGE_CHECK":
            params["parameters"] = {}
            if "min" in rule:
                params["parameters"]["minValue"] = rule["min"]
            if "max" in rule:
                params["parameters"]["maxValue"] = rule["max"]
            if rule.get("type") == "greater_than":
                params["parameters"]["minValue"] = rule.get("value")
                params["parameters"]["minInclusive"] = False
            elif rule.get("type") == "less_than":
                params["parameters"]["maxValue"] = rule.get("value")
                params["parameters"]["maxInclusive"] = False
            else:
                params["parameters"]["minInclusive"] = rule.get("inclusive", True)
                params["parameters"]["maxInclusive"] = rule.get("inclusive", True)

        elif rule_type == "PATTERN_CHECK":
            params["parameters"] = {
                "pattern": rule.get("pattern", ""),
                "patternType": "REGEX",
                "caseSensitive": rule.get("case_sensitive", True),
            }

        elif rule_type == "FORMAT_CHECK":
            params["parameters"] = {
                "format": rule.get("format", ""),
                "formatType": rule.get("format_type", "CUSTOM"),
            }

        elif rule_type == "DATA_TYPE_CHECK":
            params["parameters"] = {
                "expectedType": self._map_dtype(rule.get("dtype", "string")),
            }

        elif rule_type == "LENGTH_CHECK":
            params["parameters"] = {}
            if "min" in rule or "min_length" in rule:
                params["parameters"]["minLength"] = rule.get(
                    "min", rule.get("min_length", 0)
                )
            if "max" in rule or "max_length" in rule:
                params["parameters"]["maxLength"] = rule.get(
                    "max", rule.get("max_length", 255)
                )
            if "length" in rule:
                params["parameters"]["exactLength"] = rule["length"]

        elif rule_type == "REFERENCE_CHECK":
            params["parameters"] = {
                "referenceTable": rule.get("reference_table", ""),
                "referenceColumn": rule.get("reference_column", ""),
                "referenceDataStore": rule.get("reference_data_store", ""),
            }

        elif rule_type == "CROSS_FIELD_VALIDATION":
            params["parameters"] = {
                "columns": rule.get("columns", []),
                "expression": rule.get("expression", ""),
            }
            params["binding"] = {"columns": rule.get("columns", [])}

        elif rule_type == "EXPRESSION_RULE":
            params["parameters"] = {
                "expression": rule.get("expression", ""),
                "language": rule.get("language", "INFOSPHERE"),
            }

        elif rule_type == "SQL_RULE":
            params["parameters"] = {
                "sqlQuery": rule.get("sql", rule.get("query", "")),
            }

        return params

    def _map_severity(self, severity: str) -> str:
        """Map common severity to InfoSphere severity.

        Args:
            severity: Common severity string.

        Returns:
            InfoSphere severity string.
        """
        mapping = {
            "critical": "CRITICAL",
            "high": "HIGH",
            "error": "HIGH",
            "medium": "MEDIUM",
            "warning": "MEDIUM",
            "low": "LOW",
            "info": "INFORMATIONAL",
        }
        return mapping.get(severity.lower(), "MEDIUM")

    def _map_dtype(self, dtype: str) -> str:
        """Map common dtype to InfoSphere data type.

        Args:
            dtype: Common data type string.

        Returns:
            InfoSphere data type string.
        """
        mapping = {
            "string": "VARCHAR",
            "str": "VARCHAR",
            "int": "INTEGER",
            "int32": "INTEGER",
            "int64": "BIGINT",
            "integer": "INTEGER",
            "float": "DECIMAL",
            "float32": "FLOAT",
            "float64": "DOUBLE",
            "double": "DOUBLE",
            "decimal": "DECIMAL",
            "bool": "BOOLEAN",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "datetime": "TIMESTAMP",
            "timestamp": "TIMESTAMP",
            "time": "TIME",
        }
        return mapping.get(dtype.lower(), "VARCHAR")


# =============================================================================
# Result Converter
# =============================================================================


class IBMInfoSphereResultConverter(BaseResultConverter):
    """Converts IBM InfoSphere results to common format.

    InfoSphere returns results in Information Analyzer format with
    rule execution details and quality scores.
    """

    # Maps InfoSphere severity to common Severity enum
    SEVERITY_MAPPING = {
        "CRITICAL": Severity.CRITICAL,
        "HIGH": Severity.ERROR,
        "MEDIUM": Severity.WARNING,
        "LOW": Severity.INFO,
        "INFORMATIONAL": Severity.INFO,
    }

    def _extract_check_items(
        self,
        vendor_result: Any,
    ) -> list[dict[str, Any]]:
        """Extract check items from InfoSphere result.

        Args:
            vendor_result: Result from InfoSphere API.

        Returns:
            List of normalized check items.
        """
        items: list[dict[str, Any]] = []

        # Handle mock/dict results for testing
        if isinstance(vendor_result, dict):
            return self._extract_from_dict(vendor_result)

        # Handle actual InfoSphere result objects
        if hasattr(vendor_result, "ruleExecutions"):
            for rule_exec in vendor_result.ruleExecutions:
                items.append({
                    "rule_name": getattr(rule_exec, "ruleName", "unknown"),
                    "column": self._get_column_from_binding(
                        getattr(rule_exec, "binding", None)
                    ),
                    "passed": getattr(rule_exec, "passed", False),
                    "failed_count": getattr(rule_exec, "failedRecordCount", 0),
                    "message": getattr(rule_exec, "message", ""),
                    "severity": getattr(rule_exec, "severity", "MEDIUM"),
                    "details": {
                        "qualityScore": getattr(rule_exec, "qualityScore", 0),
                        "totalRecords": getattr(rule_exec, "totalRecordCount", 0),
                        "passedRecords": getattr(rule_exec, "passedRecordCount", 0),
                        "executionTime": getattr(rule_exec, "executionTimeMs", 0),
                    },
                })

        return items

    def _get_column_from_binding(self, binding: Any) -> str | None:
        """Extract column name from binding.

        Args:
            binding: InfoSphere binding object or dict.

        Returns:
            Column name or None.
        """
        if binding is None:
            return None
        if isinstance(binding, dict):
            return binding.get("column")
        return getattr(binding, "column", None)

    def _extract_from_dict(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract items from dictionary result.

        Args:
            result: Dictionary result.

        Returns:
            List of check items.
        """
        items: list[dict[str, Any]] = []

        # Handle ruleExecutions format
        if "ruleExecutions" in result:
            for rule in result["ruleExecutions"]:
                binding = rule.get("binding", {})
                items.append({
                    "rule_name": rule.get("ruleName", "unknown"),
                    "column": binding.get("column") if isinstance(binding, dict) else None,
                    "passed": rule.get("passed", False),
                    "failed_count": rule.get("failedRecordCount", 0),
                    "message": rule.get("message", ""),
                    "severity": rule.get("severity", "MEDIUM"),
                    "details": rule,
                })

        # Handle analysisResults format (Information Analyzer)
        elif "analysisResults" in result:
            for analysis in result["analysisResults"]:
                for rule_result in analysis.get("ruleResults", []):
                    items.append({
                        "rule_name": rule_result.get("ruleName", "unknown"),
                        "column": rule_result.get("columnName"),
                        "passed": rule_result.get("qualityScore", 0) >= 90,
                        "failed_count": rule_result.get("exceptionCount", 0),
                        "message": rule_result.get("description", ""),
                        "severity": rule_result.get("severity", "MEDIUM"),
                        "details": rule_result,
                    })

        # Handle qualityScore format
        elif "qualityScore" in result:
            passed = result.get("qualityScore", 0) >= result.get("threshold", 90)
            items.append({
                "rule_name": result.get("ruleName", "overall"),
                "column": None,
                "passed": passed,
                "failed_count": result.get("exceptionCount", 0),
                "message": result.get("message", ""),
                "severity": "HIGH" if not passed else "LOW",
                "details": result,
            })

        return items

    def convert_profile_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
    ) -> ProfileResult:
        """Convert InfoSphere profile result to ProfileResult.

        Args:
            vendor_result: Result from InfoSphere API.
            start_time: Start time for duration calculation.

        Returns:
            Common ProfileResult.
        """
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        columns: list[ColumnProfile] = []

        # Handle mock/dict results
        if isinstance(vendor_result, dict):
            row_count = vendor_result.get("rowCount", 0)
            for col_data in vendor_result.get("columnAnalysis", []):
                columns.append(self._convert_column_profile(col_data))
        elif hasattr(vendor_result, "columnAnalysis"):
            row_count = getattr(vendor_result, "rowCount", 0)
            for col_profile in vendor_result.columnAnalysis:
                columns.append(self._convert_column_profile_obj(col_profile))
        else:
            row_count = 0

        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            columns=tuple(columns),
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            metadata={"engine": "ibm_infosphere"},
        )

    def _convert_column_profile(self, col_data: dict[str, Any]) -> ColumnProfile:
        """Convert column data dict to ColumnProfile.

        Args:
            col_data: Column analysis data.

        Returns:
            ColumnProfile instance.
        """
        return ColumnProfile(
            column_name=col_data.get("columnName", ""),
            dtype=col_data.get("inferredDataType", "unknown"),
            null_count=col_data.get("nullCount", 0),
            null_percentage=col_data.get("nullPercentage", 0.0),
            unique_count=col_data.get("distinctCount", 0),
            min_value=col_data.get("minValue"),
            max_value=col_data.get("maxValue"),
            mean=col_data.get("meanValue"),
            std=col_data.get("standardDeviation"),
            metadata={
                **col_data,
                "frequencyDistribution": col_data.get("frequencyDistribution", []),
                "patternDistribution": col_data.get("patternDistribution", []),
            },
        )

    def _convert_column_profile_obj(self, col_profile: Any) -> ColumnProfile:
        """Convert column profile object to ColumnProfile.

        Args:
            col_profile: InfoSphere column profile object.

        Returns:
            ColumnProfile instance.
        """
        return ColumnProfile(
            column_name=getattr(col_profile, "columnName", ""),
            dtype=getattr(col_profile, "inferredDataType", "unknown"),
            null_count=getattr(col_profile, "nullCount", 0),
            null_percentage=getattr(col_profile, "nullPercentage", 0.0),
            unique_count=getattr(col_profile, "distinctCount", 0),
            min_value=getattr(col_profile, "minValue", None),
            max_value=getattr(col_profile, "maxValue", None),
            mean=getattr(col_profile, "meanValue", None),
            std=getattr(col_profile, "standardDeviation", None),
        )


# =============================================================================
# Connection Manager
# =============================================================================


class IBMInfoSphereConnectionManager(BaseConnectionManager):
    """Manages connections to IBM InfoSphere Information Server API.

    Handles authentication, session management, and API calls.
    Supports both synchronous and asynchronous execution modes.
    """

    def __init__(
        self,
        config: IBMInfoSphereConfig,
    ) -> None:
        """Initialize connection manager.

        Args:
            config: InfoSphere configuration.
        """
        super().__init__(config, name="ibm_infosphere")
        self._config: IBMInfoSphereConfig = config
        self._session: Any = None
        self._cookies: dict[str, str] = {}
        self._xsrf_token: str | None = None

    def _do_connect(self) -> Any:
        """Establish connection to InfoSphere API.

        Returns:
            Session object.
        """
        import requests

        session = requests.Session()

        # Configure session
        session.verify = self._config.verify_ssl
        if self._config.proxy_url:
            session.proxies = {
                "http": self._config.proxy_url,
                "https": self._config.proxy_url,
            }

        # Set up authentication
        if self._config.auth_type == AuthType.API_KEY:
            session.headers["X-IBM-API-Key"] = self._config.api_key or ""
        elif self._config.auth_type == AuthType.BASIC:
            self._authenticate_basic(session)
        elif self._config.auth_type == AuthType.OAUTH2:
            self._authenticate_oauth2(session)

        # Test connection
        self._test_connection(session)

        self._session = session
        return session

    def _authenticate_basic(self, session: Any) -> None:
        """Authenticate using basic auth.

        InfoSphere uses form-based login that returns session cookies.

        Args:
            session: Requests session.
        """
        # Try form-based login first (common for IIS)
        login_url = f"{self._config.api_endpoint}/login"
        try:
            response = session.post(
                login_url,
                data={
                    "username": self._config.username,
                    "password": self._config.password,
                },
                timeout=self._config.connect_timeout_seconds,
            )
            response.raise_for_status()

            # Extract XSRF token if present
            if "X-XSRF-TOKEN" in response.headers:
                self._xsrf_token = response.headers["X-XSRF-TOKEN"]
                session.headers["X-XSRF-TOKEN"] = self._xsrf_token

            self._cookies = dict(session.cookies)

        except Exception:
            # Fall back to HTTP Basic Auth
            session.auth = (self._config.username, self._config.password)

    def _authenticate_oauth2(self, session: Any) -> None:
        """Authenticate using OAuth2.

        Args:
            session: Requests session.
        """
        oauth_config = self._config.vendor_options.get("oauth", {})
        token_url = oauth_config.get(
            "token_url",
            f"{self._config.api_endpoint}/oauth2/token"
        )
        client_id = oauth_config.get("client_id", "")
        client_secret = oauth_config.get("client_secret", "")

        response = session.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=self._config.connect_timeout_seconds,
        )
        response.raise_for_status()

        token_data = response.json()
        access_token = token_data["access_token"]
        session.headers["Authorization"] = f"Bearer {access_token}"

    def _test_connection(self, session: Any) -> None:
        """Test connection to InfoSphere API.

        Args:
            session: Requests session.

        Raises:
            ConnectionError: If connection test fails.
        """
        try:
            # Try health endpoint
            response = session.get(
                f"{self._config.api_endpoint}/health",
                timeout=self._config.connect_timeout_seconds,
            )
            if response.status_code == 200:
                return

            # Try version endpoint
            response = session.get(
                f"{self._config.api_endpoint}/version",
                timeout=self._config.connect_timeout_seconds,
            )
            if response.status_code == 200:
                return

            # Try projects endpoint (common in IIS)
            response = session.get(
                f"{self._config.api_endpoint}/projects",
                timeout=self._config.connect_timeout_seconds,
            )
            response.raise_for_status()

        except Exception as e:
            from packages.enterprise.engines.base import ConnectionError
            raise ConnectionError(
                f"Failed to connect to IBM InfoSphere API: {e}",
                engine_name="ibm_infosphere",
                cause=e,
            ) from e

    def _do_disconnect(self) -> None:
        """Close connection."""
        if self._session:
            # Try to logout gracefully
            try:
                self._session.post(
                    f"{self._config.api_endpoint}/logout",
                    timeout=5.0,
                )
            except Exception:
                pass
            self._session.close()
            self._session = None
            self._cookies = {}
            self._xsrf_token = None

    def _do_health_check(self) -> bool:
        """Check connection health.

        Returns:
            True if healthy.
        """
        if not self._session:
            return False

        try:
            response = self._session.get(
                f"{self._config.api_endpoint}/health",
                timeout=self._config.timeout_seconds,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_session(self) -> Any:
        """Get the requests session.

        Returns:
            Requests session object.
        """
        return self.get_connection()

    def execute_api_call(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """Execute an API call.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            **kwargs: Additional request arguments.

        Returns:
            Response JSON.

        Raises:
            ConnectionError: If API call fails.
        """
        session = self.get_session()
        url = f"{self._config.api_endpoint}{endpoint}"

        # Set default timeout
        kwargs.setdefault("timeout", self._config.timeout_seconds)

        # Add XSRF token if available
        if self._xsrf_token and method.upper() in ("POST", "PUT", "DELETE"):
            headers = kwargs.get("headers", {})
            headers["X-XSRF-TOKEN"] = self._xsrf_token
            kwargs["headers"] = headers

        response = session.request(method, url, **kwargs)
        response.raise_for_status()

        return response.json()


# =============================================================================
# IBM InfoSphere Adapter
# =============================================================================


class IBMInfoSphereAdapter(EnterpriseEngineAdapter):
    """Adapter for IBM InfoSphere Information Server.

    Provides integration with IBM InfoSphere through its REST API,
    supporting both Information Analyzer and QualityStage components.

    Features:
        - Data quality validation using QualityStage rules
        - Data profiling using Information Analyzer
        - Column and key analysis
        - Referential integrity analysis
        - Async job execution support
        - Connection pooling

    Example:
        >>> config = IBMInfoSphereConfig(
        ...     api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
        ...     username="admin",
        ...     password="secret",
        ...     project="DataQuality",
        ... )
        >>> engine = IBMInfoSphereAdapter(config=config)
        >>> with engine:
        ...     result = engine.check(df, rules)
        ...     if result.status == CheckStatus.FAILED:
        ...         for failure in result.failures:
        ...             print(f"{failure.column}: {failure.message}")
    """

    def __init__(
        self,
        config: IBMInfoSphereConfig | None = None,
    ) -> None:
        """Initialize IBM InfoSphere adapter.

        Args:
            config: InfoSphere configuration.
        """
        super().__init__(config or DEFAULT_INFOSPHERE_CONFIG)
        self._config: IBMInfoSphereConfig = self._config  # type: ignore

    @property
    def engine_name(self) -> str:
        """Return engine name."""
        return "ibm_infosphere"

    @property
    def engine_version(self) -> str:
        """Return engine version."""
        # Try to get from vendor library
        if self._vendor_library:
            return getattr(self._vendor_library, "__version__", "11.7.0")
        return "11.7.0"

    def _get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities."""
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=False,  # InfoSphere doesn't have auto-learn
            supports_async=True,
            supports_streaming=False,
            supported_data_types=("polars", "pandas", "csv", "parquet", "db"),
            supported_rule_types=self._get_rule_translator().get_supported_rule_types(),
        )

    def _get_description(self) -> str:
        """Return engine description."""
        return (
            "IBM InfoSphere Information Server - "
            "Enterprise data quality and integration platform"
        )

    def _get_homepage(self) -> str:
        """Return engine homepage."""
        return "https://www.ibm.com/products/infosphere-information-server"

    def _ensure_vendor_library(self) -> Any:
        """Ensure IBM InfoSphere SDK is loaded.

        Returns:
            InfoSphere SDK module or requests.

        Raises:
            VendorSDKError: If SDK cannot be loaded.
        """
        if self._vendor_library is not None:
            return self._vendor_library

        try:
            # Try to import the official SDK
            import ibm_information_server_sdk as iis
            self._vendor_library = iis
        except ImportError:
            # Fall back to requests-based implementation
            try:
                import requests
                self._vendor_library = requests
            except ImportError as e:
                raise VendorSDKError(
                    "Neither ibm-information-server-sdk nor requests is installed. "
                    "Install with: pip install ibm-information-server-sdk "
                    "or pip install requests",
                    engine_name="ibm_infosphere",
                    cause=e,
                ) from e

        return self._vendor_library

    def _create_connection_manager(self) -> BaseConnectionManager:
        """Create InfoSphere connection manager.

        Returns:
            Connection manager instance.
        """
        return IBMInfoSphereConnectionManager(self._config)

    def _create_rule_translator(self) -> BaseRuleTranslator:
        """Create InfoSphere rule translator.

        Returns:
            Rule translator instance.
        """
        return IBMInfoSphereRuleTranslator()

    def _create_result_converter(self) -> BaseResultConverter:
        """Create InfoSphere result converter.

        Returns:
            Result converter instance.
        """
        return IBMInfoSphereResultConverter()

    def _execute_check(
        self,
        data: Any,
        translated_rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute check using InfoSphere API.

        Args:
            data: Converted data (Pandas DataFrame).
            translated_rules: InfoSphere-format rules.
            **kwargs: Additional parameters.

        Returns:
            InfoSphere result object.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, IBMInfoSphereConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        # Build request payload
        payload = self._build_check_payload(data, translated_rules, **kwargs)

        # Execute validation based on mode
        if self._config.execution_mode == InfoSphereExecutionMode.ASYNCHRONOUS:
            return self._execute_async_job(
                connection,
                "dataRule/execute",
                payload,
            )
        elif self._config.execution_mode == InfoSphereExecutionMode.BATCH:
            return self._execute_batch_job(
                connection,
                "dataRule/executeBatch",
                payload,
            )
        else:
            return connection.execute_api_call(
                "POST",
                "/dataRule/execute",
                json=payload,
            )

    def _execute_profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute profiling using InfoSphere API.

        Args:
            data: Converted data (Pandas DataFrame).
            **kwargs: Additional parameters.

        Returns:
            InfoSphere result object.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, IBMInfoSphereConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        # Build request payload
        payload = self._build_profile_payload(data, **kwargs)

        # Execute profiling
        analysis_type = kwargs.get(
            "analysis_type",
            InfoSphereAnalysisType.COLUMN_ANALYSIS,
        )

        if self._config.execution_mode == InfoSphereExecutionMode.ASYNCHRONOUS:
            return self._execute_async_job(
                connection,
                f"analysis/{analysis_type.value}",
                payload,
            )
        else:
            return connection.execute_api_call(
                "POST",
                f"/analysis/{analysis_type.value}",
                json=payload,
            )

    def _build_check_payload(
        self,
        data: Any,
        rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build validation request payload.

        Args:
            data: Data to validate.
            rules: Translated rules.
            **kwargs: Additional parameters.

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "project": self._config.project,
            "rules": rules,
        }

        # Add folder if specified
        if self._config.folder:
            payload["folder"] = self._config.folder

        # Add data store reference
        if self._config.data_store:
            payload["dataStore"] = self._config.data_store
            if self._config.schema_name:
                payload["schema"] = self._config.schema_name

        # Add QualityStage settings
        if self._config.suite_name:
            payload["suiteName"] = self._config.suite_name
        if self._config.job_name:
            payload["jobName"] = self._config.job_name

        # Handle data transfer based on mode
        if self._config.data_transfer_mode == DataTransferMode.INLINE:
            payload["data"] = self._serialize_data(data)
        elif self._config.data_transfer_mode == DataTransferMode.REFERENCE:
            payload["dataSource"] = kwargs.get("data_source", {})

        # Sampling settings
        if self._config.sampling_enabled:
            payload["sampling"] = {
                "enabled": True,
                "sampleSize": self._config.sample_size,
            }

        return payload

    def _build_profile_payload(
        self,
        data: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build profiling request payload.

        Args:
            data: Data to profile.
            **kwargs: Additional parameters.

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "project": self._config.analysis_project or self._config.project,
        }

        # Add folder if specified
        if self._config.folder:
            payload["folder"] = self._config.folder

        # Add data store reference
        if self._config.data_store:
            payload["dataStore"] = self._config.data_store
            if self._config.schema_name:
                payload["schema"] = self._config.schema_name

        # Handle data transfer
        if self._config.data_transfer_mode == DataTransferMode.INLINE:
            payload["data"] = self._serialize_data(data)
        elif self._config.data_transfer_mode == DataTransferMode.REFERENCE:
            payload["dataSource"] = kwargs.get("data_source", {})

        # Profile options
        payload["options"] = {
            "computeFrequencyDistribution": kwargs.get(
                "compute_frequency", True
            ),
            "computePatternDistribution": kwargs.get("compute_patterns", False),
            "computeStatistics": kwargs.get("compute_statistics", True),
            "inferDataTypes": kwargs.get("infer_types", True),
        }

        # Sampling settings
        if self._config.sampling_enabled:
            payload["sampling"] = {
                "enabled": True,
                "sampleSize": self._config.sample_size,
            }

        return payload

    def _serialize_data(self, data: Any) -> dict[str, Any]:
        """Serialize data for API payload.

        Args:
            data: Pandas DataFrame.

        Returns:
            Serialized data dictionary.
        """
        # Convert to records format
        records = data.to_dict(orient="records")
        columns = [
            {
                "name": col,
                "type": str(data[col].dtype),
            }
            for col in data.columns
        ]

        return {
            "columns": columns,
            "rows": records,
            "rowCount": len(records),
        }

    def _execute_async_job(
        self,
        connection: IBMInfoSphereConnectionManager,
        endpoint: str,
        payload: dict[str, Any],
    ) -> Any:
        """Execute async job and wait for completion.

        Args:
            connection: Connection manager.
            endpoint: API endpoint.
            payload: Request payload.

        Returns:
            Job result.
        """
        # Submit job
        submit_response = connection.execute_api_call(
            "POST",
            f"/{endpoint}/submit",
            json=payload,
        )

        job_id = submit_response.get("jobId") or submit_response.get("executionId")
        if not job_id:
            return submit_response

        # Poll for completion
        max_attempts = int(
            self._config.max_async_wait / self._config.async_poll_interval
        )
        for _ in range(max_attempts):
            time.sleep(self._config.async_poll_interval)

            status_response = connection.execute_api_call(
                "GET",
                f"/jobs/{job_id}/status",
            )

            status = status_response.get("status", "")
            if status in ("COMPLETED", "SUCCESS", "FINISHED"):
                return connection.execute_api_call(
                    "GET",
                    f"/jobs/{job_id}/result",
                )
            elif status in ("FAILED", "ERROR", "CANCELLED"):
                error_msg = status_response.get(
                    "errorMessage",
                    status_response.get("message", "Job failed"),
                )
                raise VendorSDKError(
                    f"IBM InfoSphere job failed: {error_msg}",
                    engine_name="ibm_infosphere",
                )

        raise VendorSDKError(
            f"IBM InfoSphere job timed out after {self._config.max_async_wait}s",
            engine_name="ibm_infosphere",
        )

    def _execute_batch_job(
        self,
        connection: IBMInfoSphereConnectionManager,
        endpoint: str,
        payload: dict[str, Any],
    ) -> Any:
        """Execute batch job.

        Batch jobs are submitted and monitored differently from async jobs.

        Args:
            connection: Connection manager.
            endpoint: API endpoint.
            payload: Request payload.

        Returns:
            Job result.
        """
        # Add batch-specific settings
        payload["executionMode"] = "BATCH"
        payload["notifyOnCompletion"] = False

        # Submit batch job
        submit_response = connection.execute_api_call(
            "POST",
            f"/{endpoint}",
            json=payload,
        )

        batch_id = submit_response.get("batchId") or submit_response.get("jobId")
        if not batch_id:
            return submit_response

        # Poll for batch completion (with longer intervals)
        poll_interval = max(self._config.async_poll_interval * 2, 5.0)
        max_attempts = int(self._config.max_async_wait / poll_interval)

        for _ in range(max_attempts):
            time.sleep(poll_interval)

            status_response = connection.execute_api_call(
                "GET",
                f"/batch/{batch_id}/status",
            )

            status = status_response.get("status", "")
            if status in ("COMPLETED", "SUCCESS"):
                return connection.execute_api_call(
                    "GET",
                    f"/batch/{batch_id}/result",
                )
            elif status in ("FAILED", "ERROR", "CANCELLED"):
                error_msg = status_response.get("errorMessage", "Batch job failed")
                raise VendorSDKError(
                    f"IBM InfoSphere batch job failed: {error_msg}",
                    engine_name="ibm_infosphere",
                )

        raise VendorSDKError(
            f"IBM InfoSphere batch job timed out after {self._config.max_async_wait}s",
            engine_name="ibm_infosphere",
        )

    # ==========================================================================
    # Additional InfoSphere-specific methods
    # ==========================================================================

    def run_column_analysis(
        self,
        data: Any,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> ProfileResult:
        """Run column analysis using Information Analyzer.

        Args:
            data: Data to analyze.
            columns: Specific columns to analyze (all if None).
            **kwargs: Additional options.

        Returns:
            ProfileResult with column analysis.
        """
        kwargs["analysis_type"] = InfoSphereAnalysisType.COLUMN_ANALYSIS
        if columns:
            kwargs["columns"] = columns
        return self.profile(data, **kwargs)

    def run_key_analysis(
        self,
        data: Any,
        candidate_keys: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> ProfileResult:
        """Run key analysis to discover primary keys.

        Args:
            data: Data to analyze.
            candidate_keys: Candidate key column combinations.
            **kwargs: Additional options.

        Returns:
            ProfileResult with key analysis.
        """
        kwargs["analysis_type"] = InfoSphereAnalysisType.KEY_ANALYSIS
        if candidate_keys:
            kwargs["candidateKeys"] = candidate_keys
        return self.profile(data, **kwargs)

    def run_referential_integrity_analysis(
        self,
        source_data: Any,
        target_data: Any,
        source_columns: list[str],
        target_columns: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run referential integrity analysis between datasets.

        Args:
            source_data: Source dataset.
            target_data: Target dataset.
            source_columns: Foreign key columns in source.
            target_columns: Primary key columns in target.
            **kwargs: Additional options.

        Returns:
            Analysis result dictionary.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, IBMInfoSphereConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        payload = {
            "project": self._config.project,
            "analysisType": "REFERENTIAL_INTEGRITY_ANALYSIS",
            "source": {
                "data": self._serialize_data(source_data),
                "columns": source_columns,
            },
            "target": {
                "data": self._serialize_data(target_data),
                "columns": target_columns,
            },
            **kwargs,
        }

        return connection.execute_api_call(
            "POST",
            "/analysis/referentialIntegrity",
            json=payload,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_ibm_infosphere_adapter(
    api_endpoint: str,
    username: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> IBMInfoSphereAdapter:
    """Create an IBM InfoSphere adapter with configuration.

    Factory function for convenient adapter creation.

    Args:
        api_endpoint: InfoSphere API endpoint URL.
        username: Username for basic auth.
        password: Password for basic auth.
        api_key: API key for authentication.
        **kwargs: Additional configuration options.

    Returns:
        Configured IBMInfoSphereAdapter instance.

    Example:
        >>> adapter = create_ibm_infosphere_adapter(
        ...     api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
        ...     username="admin",
        ...     password="secret",
        ...     project="DataQuality",
        ... )
    """
    auth_type = AuthType.API_KEY if api_key else AuthType.BASIC

    config = IBMInfoSphereConfig(
        api_endpoint=api_endpoint,
        api_key=api_key,
        username=username,
        password=password,
        auth_type=auth_type,
        **kwargs,
    )
    return IBMInfoSphereAdapter(config=config)
