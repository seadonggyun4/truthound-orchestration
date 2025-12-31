"""SAP Data Services Adapter for Enterprise Data Quality.

This module provides an adapter for SAP Data Services (formerly SAP Data Quality),
enabling integration with SAP's enterprise data management platform.

SAP Data Services provides:
- Data profiling and quality analysis
- Data validation rules (match transforms, validation transforms)
- Address cleansing and geocoding
- Text analysis and entity extraction
- Integration with SAP Business Suite

Key Features:
    - REST API integration with SAP Data Services server
    - Rule translation from common format to SAP format
    - Support for real-time and batch execution modes
    - Connection pool management with CMC authentication
    - Support for Address Cleansing, Geocoding, and Text Analysis

Example:
    >>> from packages.enterprise.engines import SAPDataServicesAdapter
    >>> config = SAPDataServicesConfig(
    ...     api_endpoint="https://sap-ds.example.com/dswsbobje/rest/v1",
    ...     username="admin",
    ...     password="secret",
    ...     repository="DataQuality",
    ... )
    >>> with SAPDataServicesAdapter(config=config) as engine:
    ...     result = engine.check(data, rules)
    ...     profile = engine.profile(data)

Entry Point Registration:
    Add to pyproject.toml:
    ```toml
    [project.entry-points."truthound.engines"]
    sap_data_services = "packages.enterprise.engines:SAPDataServicesAdapter"
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

from common.base import (
    CheckResult,
    CheckStatus,
    ColumnProfile,
    ProfileResult,
    ProfileStatus,
    Severity,
    ValidationFailure,
)
from common.engines.base import EngineCapabilities
from common.health import HealthCheckResult, HealthStatus
from packages.enterprise.engines.base import (
    AuthType,
    BaseConnectionManager,
    BaseResultConverter,
    BaseRuleTranslator,
    EnterpriseEngineAdapter,
    EnterpriseEngineConfig,
    RuleTranslationError,
    VendorSDKError,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# =============================================================================
# Enums
# =============================================================================


class SAPExecutionMode(Enum):
    """Execution mode for SAP Data Services operations.

    Attributes:
        REALTIME: Real-time execution via REST API (low latency).
        BATCH: Batch execution via job submission (high throughput).
        EMBEDDED: Embedded execution in SAP BW/HANA context.
    """

    REALTIME = "REALTIME"
    BATCH = "BATCH"
    EMBEDDED = "EMBEDDED"


class SAPRuleType(Enum):
    """SAP Data Services rule types.

    Maps to SAP Data Services transform types and validation functions.
    """

    # Data Quality Rules
    VALIDATION = "VALIDATION"
    MATCH_TRANSFORM = "MATCH_TRANSFORM"
    DATA_CLEANSE = "DATA_CLEANSE"

    # Address Rules
    ADDRESS_CLEANSE = "ADDRESS_CLEANSE"
    GEOCODE = "GEOCODE"

    # Text Analysis
    TEXT_ANALYSIS = "TEXT_ANALYSIS"
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"

    # Profiling
    COLUMN_PROFILE = "COLUMN_PROFILE"
    PATTERN_ANALYSIS = "PATTERN_ANALYSIS"
    RELATIONSHIP_ANALYSIS = "RELATIONSHIP_ANALYSIS"

    # Data Validation
    NULL_CHECK = "NULL_CHECK"
    UNIQUE_CHECK = "UNIQUE_CHECK"
    DOMAIN_CHECK = "DOMAIN_CHECK"
    RANGE_CHECK = "RANGE_CHECK"
    FORMAT_CHECK = "FORMAT_CHECK"
    REFERENCE_CHECK = "REFERENCE_CHECK"
    EXPRESSION_CHECK = "EXPRESSION_CHECK"


class SAPDataType(Enum):
    """SAP Data Services data types."""

    VARCHAR = "VARCHAR"
    INTEGER = "INTEGER"
    DECIMAL = "DECIMAL"
    DATE = "DATE"
    DATETIME = "DATETIME"
    BLOB = "BLOB"
    DOUBLE = "DOUBLE"
    LONG = "LONG"


class SAPJobStatus(Enum):
    """SAP Data Services job status values."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    WARNING = "WARNING"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class SAPDataServicesConfig(EnterpriseEngineConfig):
    """Configuration for SAP Data Services adapter.

    Attributes:
        repository: Repository name (Data Services object store).
        job_server: Job Server name for batch execution.
        access_server: Access Server for real-time execution.
        cmc_endpoint: Central Management Console endpoint for auth.
        execution_mode: Execution mode (REALTIME, BATCH, EMBEDDED).
        datastore: Default datastore name.
        project: Default project name.
        enable_address_cleansing: Enable address cleansing features.
        enable_geocoding: Enable geocoding features.
        enable_text_analysis: Enable text analysis features.
        locale: Default locale for data cleansing (e.g., "en_US").
        address_directories: Address directories for cleansing.
        async_poll_interval: Poll interval for async jobs (seconds).
        max_async_wait: Maximum wait time for async jobs (seconds).
        substitution_parameters: Default substitution parameters.

    Example:
        >>> config = SAPDataServicesConfig(
        ...     api_endpoint="https://sap-ds.example.com/dswsbobje/rest/v1",
        ...     username="admin",
        ...     password="secret",
        ...     repository="DataQuality",
        ...     execution_mode=SAPExecutionMode.REALTIME,
        ... )
    """

    # SAP-specific settings
    repository: str = "Central"
    job_server: str = ""
    access_server: str = ""
    cmc_endpoint: str = ""

    # Execution settings
    execution_mode: SAPExecutionMode = SAPExecutionMode.REALTIME

    # Project settings
    datastore: str = ""
    project: str = ""

    # Feature flags
    enable_address_cleansing: bool = False
    enable_geocoding: bool = False
    enable_text_analysis: bool = False

    # Localization
    locale: str = "en_US"
    address_directories: tuple[str, ...] = ()

    # Async settings
    async_poll_interval: float = 2.0
    max_async_wait: float = 300.0

    # SAP-specific parameters
    substitution_parameters: dict[str, str] = field(default_factory=dict)

    # Builder methods for SAP-specific settings

    def with_repository(
        self,
        repository: str,
        datastore: str | None = None,
        project: str | None = None,
    ) -> Self:
        """Create config with repository settings.

        Args:
            repository: Repository name.
            datastore: Optional datastore name.
            project: Optional project name.

        Returns:
            New configuration with repository settings.
        """
        updates: dict[str, Any] = {"repository": repository}
        if datastore is not None:
            updates["datastore"] = datastore
        if project is not None:
            updates["project"] = project
        return self._copy_with(**updates)

    def with_servers(
        self,
        job_server: str | None = None,
        access_server: str | None = None,
        cmc_endpoint: str | None = None,
    ) -> Self:
        """Create config with server settings.

        Args:
            job_server: Job Server name.
            access_server: Access Server name.
            cmc_endpoint: CMC endpoint.

        Returns:
            New configuration with server settings.
        """
        updates: dict[str, Any] = {}
        if job_server is not None:
            updates["job_server"] = job_server
        if access_server is not None:
            updates["access_server"] = access_server
        if cmc_endpoint is not None:
            updates["cmc_endpoint"] = cmc_endpoint
        return self._copy_with(**updates)

    def with_execution_mode(self, mode: SAPExecutionMode) -> Self:
        """Create config with execution mode.

        Args:
            mode: Execution mode (REALTIME, BATCH, EMBEDDED).

        Returns:
            New configuration with execution mode.
        """
        return self._copy_with(execution_mode=mode)

    def with_address_cleansing(
        self,
        enabled: bool = True,
        directories: tuple[str, ...] | None = None,
        locale: str | None = None,
    ) -> Self:
        """Create config with address cleansing settings.

        Args:
            enabled: Enable address cleansing.
            directories: Address directories.
            locale: Locale for cleansing.

        Returns:
            New configuration with address cleansing.
        """
        updates: dict[str, Any] = {"enable_address_cleansing": enabled}
        if directories is not None:
            updates["address_directories"] = directories
        if locale is not None:
            updates["locale"] = locale
        return self._copy_with(**updates)

    def with_geocoding(self, enabled: bool = True) -> Self:
        """Create config with geocoding enabled.

        Args:
            enabled: Enable geocoding.

        Returns:
            New configuration with geocoding.
        """
        return self._copy_with(enable_geocoding=enabled)

    def with_text_analysis(self, enabled: bool = True) -> Self:
        """Create config with text analysis enabled.

        Args:
            enabled: Enable text analysis.

        Returns:
            New configuration with text analysis.
        """
        return self._copy_with(enable_text_analysis=enabled)

    def with_async_settings(
        self,
        poll_interval: float | None = None,
        max_wait: float | None = None,
    ) -> Self:
        """Create config with async job settings.

        Args:
            poll_interval: Poll interval in seconds.
            max_wait: Maximum wait time in seconds.

        Returns:
            New configuration with async settings.
        """
        updates: dict[str, Any] = {}
        if poll_interval is not None:
            updates["async_poll_interval"] = poll_interval
        if max_wait is not None:
            updates["max_async_wait"] = max_wait
        return self._copy_with(**updates)

    def with_substitution_parameters(self, **params: str) -> Self:
        """Create config with substitution parameters.

        Args:
            **params: Key-value substitution parameters.

        Returns:
            New configuration with substitution parameters.
        """
        merged = {**self.substitution_parameters, **params}
        return self._copy_with(substitution_parameters=merged)


# Preset configurations

DEFAULT_SAP_DS_CONFIG = SAPDataServicesConfig(
    auth_type=AuthType.BASIC,
    timeout_seconds=120.0,
    max_retries=3,
    execution_mode=SAPExecutionMode.REALTIME,
)

PRODUCTION_SAP_DS_CONFIG = SAPDataServicesConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    health_check_interval_seconds=60.0,
    auth_type=AuthType.BASIC,
    timeout_seconds=180.0,
    connect_timeout_seconds=30.0,
    max_retries=5,
    retry_delay_seconds=2.0,
    verify_ssl=True,
    pool_size=10,
    execution_mode=SAPExecutionMode.BATCH,
    async_poll_interval=5.0,
    max_async_wait=600.0,
)

DEVELOPMENT_SAP_DS_CONFIG = SAPDataServicesConfig(
    auto_start=False,
    auto_stop=True,
    health_check_enabled=False,
    auth_type=AuthType.BASIC,
    timeout_seconds=60.0,
    max_retries=1,
    verify_ssl=False,
    execution_mode=SAPExecutionMode.REALTIME,
)

REALTIME_SAP_DS_CONFIG = SAPDataServicesConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    auth_type=AuthType.BASIC,
    timeout_seconds=30.0,
    connect_timeout_seconds=5.0,
    max_retries=2,
    pool_size=20,
    execution_mode=SAPExecutionMode.REALTIME,
)

ADDRESS_CLEANSING_CONFIG = SAPDataServicesConfig(
    auto_start=True,
    auto_stop=True,
    auth_type=AuthType.BASIC,
    timeout_seconds=60.0,
    execution_mode=SAPExecutionMode.REALTIME,
    enable_address_cleansing=True,
    enable_geocoding=True,
    locale="en_US",
)


# =============================================================================
# Rule Translator
# =============================================================================


class SAPDataServicesRuleTranslator(BaseRuleTranslator):
    """Translate common rules to SAP Data Services format.

    Converts common rule types to SAP Data Services validation transforms,
    match transforms, and data quality functions.

    Supported Rule Types:
        - not_null -> NULL_CHECK validation
        - unique -> UNIQUE_CHECK validation
        - in_set -> DOMAIN_CHECK validation
        - in_range -> RANGE_CHECK validation
        - regex -> FORMAT_CHECK validation
        - dtype -> Data type validation
        - foreign_key -> REFERENCE_CHECK validation
        - expression -> EXPRESSION_CHECK validation
        - address -> ADDRESS_CLEANSE transform
        - match -> MATCH_TRANSFORM
    """

    # Mapping from common rule types to SAP rule types
    RULE_TYPE_MAPPING: dict[str, str] = {
        "not_null": "NULL_CHECK",
        "notnull": "NULL_CHECK",
        "unique": "UNIQUE_CHECK",
        "in_set": "DOMAIN_CHECK",
        "in_list": "DOMAIN_CHECK",
        "in_range": "RANGE_CHECK",
        "between": "RANGE_CHECK",
        "regex": "FORMAT_CHECK",
        "pattern": "FORMAT_CHECK",
        "dtype": "DATA_TYPE_CHECK",
        "data_type": "DATA_TYPE_CHECK",
        "min_length": "LENGTH_CHECK",
        "max_length": "LENGTH_CHECK",
        "length": "LENGTH_CHECK",
        "greater_than": "RANGE_CHECK",
        "less_than": "RANGE_CHECK",
        "foreign_key": "REFERENCE_CHECK",
        "reference": "REFERENCE_CHECK",
        "expression": "EXPRESSION_CHECK",
        "sql": "SQL_CHECK",
        "address": "ADDRESS_CLEANSE",
        "address_cleanse": "ADDRESS_CLEANSE",
        "geocode": "GEOCODE",
        "match": "MATCH_TRANSFORM",
    }

    # Data type mapping from common to SAP
    DTYPE_MAPPING: dict[str, str] = {
        "string": "VARCHAR",
        "str": "VARCHAR",
        "int": "INTEGER",
        "int32": "INTEGER",
        "int64": "LONG",
        "float": "DOUBLE",
        "float32": "DOUBLE",
        "float64": "DOUBLE",
        "decimal": "DECIMAL",
        "bool": "INTEGER",  # SAP uses 0/1 for boolean
        "boolean": "INTEGER",
        "date": "DATE",
        "datetime": "DATETIME",
        "timestamp": "DATETIME",
    }

    # Severity mapping from common to SAP
    SEVERITY_MAPPING: dict[str, str] = {
        "critical": "CRITICAL",
        "high": "ERROR",
        "error": "ERROR",
        "medium": "WARNING",
        "warning": "WARNING",
        "low": "INFO",
        "info": "INFO",
    }

    def _get_rule_mapping(self) -> dict[str, str]:
        """Return rule type mapping."""
        return self.RULE_TYPE_MAPPING

    def _map_severity(self, severity: str) -> str:
        """Map common severity to SAP severity.

        Args:
            severity: Common severity string.

        Returns:
            SAP severity string.
        """
        return self.SEVERITY_MAPPING.get(severity.lower(), "WARNING")

    def _map_dtype(self, dtype: str) -> str:
        """Map common data type to SAP data type.

        Args:
            dtype: Common data type string.

        Returns:
            SAP data type string.
        """
        return self.DTYPE_MAPPING.get(dtype.lower(), "VARCHAR")

    def translate(
        self,
        rule: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate a common rule to SAP Data Services format.

        Args:
            rule: Common rule dictionary.

        Returns:
            SAP-specific rule dictionary.

        Raises:
            RuleTranslationError: If rule cannot be translated.
        """
        rule_type = rule.get("type", "")
        column = rule.get("column", "")

        # Check for custom translator
        if rule_type in self._custom_translators:
            return self._custom_translators[rule_type](rule)

        # Get SAP rule type
        sap_type = self.RULE_TYPE_MAPPING.get(rule_type.lower())
        if sap_type is None:
            raise RuleTranslationError(
                f"Unsupported rule type for SAP Data Services: {rule_type}",
                rule_type=rule_type,
            )

        # Build base translated rule
        translated: dict[str, Any] = {
            "type": sap_type,
            "binding": {"column": column},
            "parameters": {},
            "options": {},
        }

        # Add severity if provided
        severity = rule.get("severity", "medium")
        translated["options"]["severity"] = self._map_severity(severity)

        # Add rule name if provided
        if "name" in rule:
            translated["name"] = rule["name"]
        else:
            translated["name"] = f"{sap_type}_{column}"

        # Translate parameters based on rule type
        params = translated["parameters"]

        if sap_type == "NULL_CHECK":
            params["allowEmpty"] = rule.get("allow_empty", False)
            params["treatWhitespaceAsNull"] = rule.get(
                "treat_whitespace_as_null", True
            )

        elif sap_type == "UNIQUE_CHECK":
            params["caseSensitive"] = rule.get("case_sensitive", True)
            params["includeNulls"] = rule.get("include_nulls", False)

        elif sap_type == "DOMAIN_CHECK":
            values = rule.get("values", [])
            params["validValues"] = list(values) if values else []
            params["caseSensitive"] = rule.get("case_sensitive", True)
            params["allowNull"] = rule.get("allow_null", False)

        elif sap_type == "RANGE_CHECK":
            if rule_type in ("greater_than", "gt"):
                params["minValue"] = rule.get("value")
                params["minInclusive"] = False
            elif rule_type in ("less_than", "lt"):
                params["maxValue"] = rule.get("value")
                params["maxInclusive"] = False
            else:
                params["minValue"] = rule.get("min")
                params["maxValue"] = rule.get("max")
                params["minInclusive"] = rule.get("min_inclusive", True)
                params["maxInclusive"] = rule.get("max_inclusive", True)

        elif sap_type == "FORMAT_CHECK":
            params["pattern"] = rule.get("pattern", "")
            params["patternType"] = "REGEX"
            params["caseSensitive"] = rule.get("case_sensitive", True)

        elif sap_type == "DATA_TYPE_CHECK":
            dtype = rule.get("dtype", "string")
            params["expectedType"] = self._map_dtype(dtype)

        elif sap_type == "LENGTH_CHECK":
            if "min" in rule or "min_length" in rule:
                params["minLength"] = rule.get("min") or rule.get("min_length")
            if "max" in rule or "max_length" in rule:
                params["maxLength"] = rule.get("max") or rule.get("max_length")
            if "length" in rule:
                params["exactLength"] = rule.get("length")

        elif sap_type == "REFERENCE_CHECK":
            params["referenceTable"] = rule.get("reference_table", "")
            params["referenceColumn"] = rule.get("reference_column", "")
            params["referenceDatastore"] = rule.get("reference_datastore", "")

        elif sap_type == "EXPRESSION_CHECK":
            params["expression"] = rule.get("expression", "")
            params["language"] = "SAP_DS"

        elif sap_type == "SQL_CHECK":
            params["sqlQuery"] = rule.get("sql", "")

        elif sap_type == "ADDRESS_CLEANSE":
            params["mode"] = rule.get("mode", "CERTIFY")
            params["outputFields"] = rule.get(
                "output_fields",
                ["address_line", "city", "region", "postal_code", "country"],
            )
            params["locale"] = rule.get("locale", "en_US")

        elif sap_type == "GEOCODE":
            params["outputFields"] = rule.get(
                "output_fields",
                ["latitude", "longitude", "accuracy"],
            )
            params["precision"] = rule.get("precision", "HIGH")

        elif sap_type == "MATCH_TRANSFORM":
            params["matchStrategy"] = rule.get("match_strategy", "FUZZY")
            params["threshold"] = rule.get("threshold", 0.8)
            params["algorithm"] = rule.get("algorithm", "WEIGHTED")
            params["groupByFields"] = rule.get("group_by", [])

        return translated

    def get_supported_rule_types(self) -> tuple[str, ...]:
        """Return supported rule types."""
        return tuple(self.RULE_TYPE_MAPPING.keys())


# =============================================================================
# Result Converter
# =============================================================================


class SAPDataServicesResultConverter(BaseResultConverter):
    """Convert SAP Data Services results to common format.

    Handles conversion of SAP Data Services validation results,
    profiling results, and cleansing results to common types.
    """

    # Severity mapping from SAP to common
    SEVERITY_MAPPING: dict[str, Severity] = {
        "CRITICAL": Severity.CRITICAL,
        "ERROR": Severity.ERROR,
        "WARNING": Severity.WARNING,
        "INFO": Severity.INFO,
        "INFORMATIONAL": Severity.INFO,
    }

    def _map_severity(self, vendor_severity: str) -> Severity:
        """Map SAP severity to common Severity.

        Args:
            vendor_severity: SAP severity string.

        Returns:
            Common Severity enum value.
        """
        return self.SEVERITY_MAPPING.get(
            vendor_severity.upper(), Severity.WARNING
        )

    def _extract_check_items(
        self,
        vendor_result: Any,
    ) -> list[dict[str, Any]]:
        """Extract check items from SAP Data Services result.

        Handles multiple SAP result formats:
        1. Validation results (validationResults)
        2. Rule execution results (ruleExecutions)
        3. Quality score format (qualityScore)

        Args:
            vendor_result: SAP Data Services result dictionary.

        Returns:
            List of normalized check items.
        """
        items: list[dict[str, Any]] = []

        if not isinstance(vendor_result, dict):
            return items

        # Format 1: validationResults (standard validation output)
        validation_results = vendor_result.get("validationResults", [])
        for result in validation_results:
            passed = result.get("passed", result.get("valid", True))
            items.append(
                {
                    "rule_name": result.get("ruleName", "Unknown"),
                    "column": result.get("column", result.get("columnName", "")),
                    "passed": passed,
                    "failed_count": result.get(
                        "failedCount", result.get("exceptionCount", 0 if passed else 1)
                    ),
                    "message": result.get("message", result.get("description", "")),
                    "severity": result.get("severity", "WARNING"),
                    "details": {
                        "score": result.get("qualityScore"),
                        "passedCount": result.get("passedCount"),
                        "totalCount": result.get("totalCount"),
                    },
                }
            )

        # Format 2: ruleExecutions (job execution output)
        rule_executions = vendor_result.get("ruleExecutions", [])
        for execution in rule_executions:
            passed = execution.get("passed", execution.get("status") == "SUCCEEDED")
            binding = execution.get("binding", {})
            items.append(
                {
                    "rule_name": execution.get("ruleName", "Unknown"),
                    "column": binding.get("column", ""),
                    "passed": passed,
                    "failed_count": execution.get("failedRecordCount", 0 if passed else 1),
                    "message": execution.get("message", ""),
                    "severity": execution.get("severity", "WARNING"),
                    "details": execution.get("details", {}),
                }
            )

        # Format 3: qualityScore (simple score-based result)
        if "qualityScore" in vendor_result and not items:
            score = vendor_result.get("qualityScore", 100)
            threshold = vendor_result.get("threshold", 90)
            passed = score >= threshold
            items.append(
                {
                    "rule_name": vendor_result.get("ruleName", "QualityScore"),
                    "column": "",
                    "passed": passed,
                    "failed_count": vendor_result.get("exceptionCount", 0 if passed else 1),
                    "message": vendor_result.get(
                        "message", f"Quality score: {score}% (threshold: {threshold}%)"
                    ),
                    "severity": "ERROR" if not passed else "INFO",
                    "details": {"score": score, "threshold": threshold},
                }
            )

        return items

    def convert_profile_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
    ) -> ProfileResult:
        """Convert SAP Data Services profile result to ProfileResult.

        Args:
            vendor_result: SAP profile result dictionary.
            start_time: Start time for duration calculation.

        Returns:
            Common ProfileResult.
        """
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        if not isinstance(vendor_result, dict):
            return ProfileResult(
                columns=(),
                row_count=0,
                execution_time_ms=execution_time_ms,
                metadata={"engine": "SAPDataServices"},
            )

        row_count = vendor_result.get("rowCount", vendor_result.get("recordCount", 0))
        columns: list[ColumnProfile] = []

        # Parse column profiles
        column_profiles = vendor_result.get(
            "columnProfiles", vendor_result.get("columns", [])
        )
        for col_data in column_profiles:
            col_name = col_data.get("columnName", col_data.get("name", ""))
            dtype = col_data.get("dataType", col_data.get("inferredType", "VARCHAR"))

            # Extract statistics
            null_count = col_data.get("nullCount", 0)
            null_pct = col_data.get(
                "nullPercentage",
                (null_count / row_count * 100) if row_count > 0 else 0.0,
            )
            unique_count = col_data.get(
                "distinctCount", col_data.get("uniqueCount", 0)
            )

            # Additional profile info
            min_value = col_data.get("minValue")
            max_value = col_data.get("maxValue")
            avg_value = col_data.get("avgValue")
            min_length = col_data.get("minLength")
            max_length = col_data.get("maxLength")
            avg_length = col_data.get("avgLength")

            # Pattern info
            patterns = col_data.get("patterns", [])
            top_values = col_data.get("topValues", col_data.get("frequentValues", []))

            columns.append(
                ColumnProfile(
                    column_name=col_name,
                    dtype=dtype,
                    null_count=null_count,
                    null_percentage=null_pct,
                    unique_count=unique_count,
                    min_value=min_value,
                    max_value=max_value,
                    metadata={
                        "avg_value": avg_value,
                        "min_length": min_length,
                        "max_length": max_length,
                        "avg_length": avg_length,
                        "patterns": patterns,
                        "top_values": top_values,
                        "completeness": 100.0 - null_pct,
                        "quality_score": col_data.get("qualityScore"),
                    },
                )
            )

        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            columns=tuple(columns),
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            metadata={
                "engine": "SAPDataServices",
                "profile_type": vendor_result.get("profileType", "STANDARD"),
                "repository": vendor_result.get("repository"),
            },
        )


# =============================================================================
# Connection Manager
# =============================================================================


class SAPDataServicesConnectionManager(BaseConnectionManager):
    """Manage connections to SAP Data Services.

    Handles:
    - CMC (Central Management Console) authentication
    - REST API session management
    - Connection pooling
    - Health checks
    """

    def __init__(self, config: SAPDataServicesConfig) -> None:
        """Initialize connection manager.

        Args:
            config: SAP Data Services configuration.
        """
        super().__init__(config, name="sap_data_services")
        self._config: SAPDataServicesConfig = config
        self._session: Any = None
        self._auth_token: str | None = None
        self._token_expiry: float = 0

    def _do_connect(self) -> Any:
        """Establish connection to SAP Data Services.

        Returns:
            Session object for API calls.

        Raises:
            ConnectionError: If connection fails.
        """
        try:
            import requests
        except ImportError as e:
            raise VendorSDKError(
                "requests library required for SAP Data Services. "
                "Install with: pip install requests",
                cause=e,
            ) from e

        # Create session with pooling
        session = requests.Session()
        session.verify = self._config.verify_ssl

        if self._config.proxy_url:
            session.proxies = {
                "http": self._config.proxy_url,
                "https": self._config.proxy_url,
            }

        # Authenticate based on auth type
        if self._config.auth_type == AuthType.BASIC:
            self._authenticate_basic(session)
        elif self._config.auth_type == AuthType.API_KEY:
            self._authenticate_api_key(session)
        elif self._config.auth_type == AuthType.OAUTH2:
            self._authenticate_oauth2(session)

        self._session = session
        return session

    def _authenticate_basic(self, session: Any) -> None:
        """Authenticate using basic authentication.

        Args:
            session: Requests session.

        Raises:
            AuthenticationError: If authentication fails.
        """
        from packages.enterprise.engines.base import AuthenticationError

        if not self._config.username or not self._config.password:
            raise AuthenticationError(
                "Username and password required for basic authentication",
                engine_name="sap_data_services",
            )

        # SAP Data Services uses CMC for authentication
        cmc_endpoint = self._config.cmc_endpoint or self._config.api_endpoint
        login_url = f"{cmc_endpoint}/logon/long"

        try:
            response = session.post(
                login_url,
                json={
                    "userName": self._config.username,
                    "password": self._config.password,
                    "auth": "secEnterprise",
                },
                timeout=self._config.connect_timeout_seconds,
            )
            response.raise_for_status()

            # Extract token from response
            data = response.json()
            self._auth_token = data.get("logonToken", data.get("token"))

            # Set token in session headers
            if self._auth_token:
                session.headers["X-SAP-LogonToken"] = self._auth_token

        except Exception as e:
            raise AuthenticationError(
                f"SAP Data Services authentication failed: {e}",
                engine_name="sap_data_services",
                cause=e,
            ) from e

    def _authenticate_api_key(self, session: Any) -> None:
        """Authenticate using API key.

        Args:
            session: Requests session.
        """
        if self._config.api_key:
            session.headers["X-API-Key"] = self._config.api_key

    def _authenticate_oauth2(self, session: Any) -> None:
        """Authenticate using OAuth2.

        Args:
            session: Requests session.

        Raises:
            AuthenticationError: If authentication fails.
        """
        from packages.enterprise.engines.base import AuthenticationError

        oauth_config = self._config.vendor_options.get("oauth", {})
        token_url = oauth_config.get("token_url")
        client_id = oauth_config.get("client_id")
        client_secret = oauth_config.get("client_secret")

        if not all([token_url, client_id, client_secret]):
            raise AuthenticationError(
                "OAuth2 requires token_url, client_id, and client_secret",
                engine_name="sap_data_services",
            )

        try:
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

            data = response.json()
            self._auth_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)
            self._token_expiry = time.time() + expires_in - 60  # Buffer

            if self._auth_token:
                session.headers["Authorization"] = f"Bearer {self._auth_token}"

        except Exception as e:
            raise AuthenticationError(
                f"OAuth2 authentication failed: {e}",
                engine_name="sap_data_services",
                cause=e,
            ) from e

    def _do_disconnect(self) -> None:
        """Disconnect from SAP Data Services."""
        if self._session and self._auth_token:
            try:
                cmc_endpoint = self._config.cmc_endpoint or self._config.api_endpoint
                self._session.post(
                    f"{cmc_endpoint}/logoff",
                    timeout=self._config.connect_timeout_seconds,
                )
            except Exception:
                pass  # Best effort logout

        if self._session:
            self._session.close()
            self._session = None

        self._auth_token = None

    def _do_health_check(self) -> bool:
        """Check connection health.

        Returns:
            True if connection is healthy.
        """
        if not self._session:
            return False

        try:
            # Call server info endpoint
            response = self._session.get(
                f"{self._config.api_endpoint}/serverInfo",
                timeout=self._config.connect_timeout_seconds,
            )
            return response.status_code == 200
        except Exception:
            return False

    def execute_api_call(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an API call to SAP Data Services.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            endpoint: API endpoint path.
            data: Request body data.
            params: Query parameters.

        Returns:
            Response data as dictionary.

        Raises:
            ConnectionError: If API call fails.
        """
        from packages.enterprise.engines.base import ConnectionError as ConnError

        if not self._session:
            raise ConnError(
                "Not connected to SAP Data Services",
                engine_name="sap_data_services",
            )

        url = f"{self._config.api_endpoint}/{endpoint.lstrip('/')}"

        try:
            response = self._session.request(
                method=method.upper(),
                url=url,
                json=data,
                params=params,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
            return response.json() if response.content else {}

        except Exception as e:
            raise ConnError(
                f"SAP Data Services API call failed: {e}",
                engine_name="sap_data_services",
                cause=e,
            ) from e


# =============================================================================
# Main Adapter
# =============================================================================


class SAPDataServicesAdapter(EnterpriseEngineAdapter):
    """SAP Data Services adapter for enterprise data quality.

    Provides integration with SAP Data Services for:
    - Data validation using SAP validation transforms
    - Data profiling and quality analysis
    - Address cleansing and geocoding
    - Text analysis and entity extraction

    Example:
        >>> config = SAPDataServicesConfig(
        ...     api_endpoint="https://sap-ds.example.com/dswsbobje/rest/v1",
        ...     username="admin",
        ...     password="secret",
        ...     repository="DataQuality",
        ... )
        >>> with SAPDataServicesAdapter(config=config) as engine:
        ...     result = engine.check(
        ...         data,
        ...         rules=[
        ...             {"type": "not_null", "column": "id"},
        ...             {"type": "unique", "column": "email"},
        ...         ],
        ...     )
        ...     print(f"Status: {result.status.name}")
        ...     print(f"Passed: {result.passed_count}, Failed: {result.failed_count}")
    """

    def __init__(
        self,
        config: SAPDataServicesConfig | None = None,
    ) -> None:
        """Initialize SAP Data Services adapter.

        Args:
            config: SAP Data Services configuration.
        """
        super().__init__(config or DEFAULT_SAP_DS_CONFIG)
        self._config: SAPDataServicesConfig = self._config  # type: ignore

    @property
    def engine_name(self) -> str:
        """Return engine name."""
        return "sap_data_services"

    @property
    def engine_version(self) -> str:
        """Return engine version."""
        return "4.3"  # SAP Data Services 4.3

    def _get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities."""
        rule_types = list(SAPDataServicesRuleTranslator.RULE_TYPE_MAPPING.keys())

        # Add address/geocode capabilities if enabled
        if self._config.enable_address_cleansing:
            rule_types.extend(["address", "address_cleanse"])
        if self._config.enable_geocoding:
            rule_types.append("geocode")
        if self._config.enable_text_analysis:
            rule_types.extend(["text_analysis", "entity_extraction"])

        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=False,  # SAP DS doesn't support rule learning
            supports_async=True,  # Batch mode is async
            supports_streaming=False,
            supported_data_types=("polars", "pandas", "dict", "db"),
            supported_rule_types=tuple(rule_types),
        )

    def _ensure_vendor_library(self) -> Any:
        """Ensure required libraries are available.

        Returns:
            The requests library module.

        Raises:
            VendorSDKError: If library cannot be loaded.
        """
        if self._vendor_library is None:
            try:
                import requests

                self._vendor_library = requests
            except ImportError as e:
                raise VendorSDKError(
                    "requests library required for SAP Data Services. "
                    "Install with: pip install requests",
                    cause=e,
                ) from e
        return self._vendor_library

    def _create_connection_manager(self) -> SAPDataServicesConnectionManager:
        """Create SAP Data Services connection manager.

        Returns:
            Connection manager instance.
        """
        return SAPDataServicesConnectionManager(self._config)

    def _create_rule_translator(self) -> SAPDataServicesRuleTranslator:
        """Create SAP Data Services rule translator.

        Returns:
            Rule translator instance.
        """
        return SAPDataServicesRuleTranslator()

    def _create_result_converter(self) -> SAPDataServicesResultConverter:
        """Create SAP Data Services result converter.

        Returns:
            Result converter instance.
        """
        return SAPDataServicesResultConverter()

    def _ensure_connected(self) -> None:
        """Ensure connection is established."""
        manager = self._get_connection_manager()
        if not manager.is_connected():
            manager.connect()

    def _serialize_data(self, data: Any) -> dict[str, Any]:
        """Serialize data for API transmission.

        Args:
            data: Input data (DataFrame or dict).

        Returns:
            Serialized data dictionary.
        """
        # Handle Polars DataFrame
        if hasattr(data, "to_pandas"):
            data = data.to_pandas()

        # Handle Pandas DataFrame
        if hasattr(data, "to_dict"):
            columns = [{"name": col, "type": str(data[col].dtype)} for col in data.columns]
            rows = data.to_dict(orient="records")
            return {
                "rowCount": len(rows),
                "columns": columns,
                "rows": rows,
            }

        # Handle dict/list directly
        if isinstance(data, (dict, list)):
            if isinstance(data, list):
                return {"rowCount": len(data), "rows": data}
            return data

        return {"data": str(data)}

    def _build_check_payload(
        self,
        data: Any,
        rules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build validation request payload.

        Args:
            data: Serialized data.
            rules: Translated rules.

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "repository": self._config.repository,
            "datastore": self._config.datastore,
            "project": self._config.project,
            "rules": rules,
            "data": self._serialize_data(data),
        }

        # Add execution mode settings
        if self._config.execution_mode == SAPExecutionMode.BATCH:
            payload["executionMode"] = "BATCH"
            if self._config.job_server:
                payload["jobServer"] = self._config.job_server
        elif self._config.execution_mode == SAPExecutionMode.REALTIME:
            payload["executionMode"] = "REALTIME"
            if self._config.access_server:
                payload["accessServer"] = self._config.access_server

        # Add substitution parameters
        if self._config.substitution_parameters:
            payload["substitutionParameters"] = self._config.substitution_parameters

        return payload

    def _build_profile_payload(self, data: Any) -> dict[str, Any]:
        """Build profiling request payload.

        Args:
            data: Serialized data.

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "repository": self._config.repository,
            "datastore": self._config.datastore,
            "project": self._config.project,
            "data": self._serialize_data(data),
            "options": {
                "computeStatistics": True,
                "computePatterns": True,
                "computeDistributions": True,
                "inferDataTypes": True,
            },
        }

        # Add profiling-specific settings
        if self._config.execution_mode == SAPExecutionMode.BATCH:
            payload["executionMode"] = "BATCH"

        return payload

    def _execute_check(
        self,
        data: Any,
        translated_rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute validation check via SAP Data Services API.

        Args:
            data: Converted data.
            translated_rules: SAP-format rules.
            **kwargs: Additional parameters.

        Returns:
            Vendor result dictionary.
        """
        manager = self._get_connection_manager()
        self._ensure_connected()

        payload = self._build_check_payload(data, translated_rules)

        # Execute based on mode
        if self._config.execution_mode == SAPExecutionMode.BATCH:
            return self._execute_batch_job(manager, "validation/execute", payload)
        else:
            return manager.execute_api_call("POST", "validation/execute", data=payload)

    def _execute_profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute profiling via SAP Data Services API.

        Args:
            data: Converted data.
            **kwargs: Additional parameters.

        Returns:
            Vendor result dictionary.
        """
        manager = self._get_connection_manager()
        self._ensure_connected()

        payload = self._build_profile_payload(data)

        # Execute based on mode
        if self._config.execution_mode == SAPExecutionMode.BATCH:
            return self._execute_batch_job(manager, "profiling/execute", payload)
        else:
            return manager.execute_api_call("POST", "profiling/execute", data=payload)

    def _execute_batch_job(
        self,
        manager: SAPDataServicesConnectionManager,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a batch job and wait for results.

        Args:
            manager: Connection manager.
            endpoint: API endpoint.
            payload: Request payload.

        Returns:
            Job result dictionary.

        Raises:
            VendorSDKError: If job fails or times out.
        """
        # Submit job
        submit_response = manager.execute_api_call("POST", endpoint, data=payload)
        job_id = submit_response.get("jobId", submit_response.get("id"))

        if not job_id:
            raise VendorSDKError(
                "Failed to submit batch job: no job ID returned",
                engine_name="sap_data_services",
            )

        # Poll for completion
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > self._config.max_async_wait:
                raise VendorSDKError(
                    f"Batch job {job_id} timed out after {elapsed:.0f} seconds",
                    engine_name="sap_data_services",
                )

            # Check job status
            status_response = manager.execute_api_call(
                "GET", f"jobs/{job_id}/status"
            )
            status = status_response.get("status", "PENDING")

            if status == "SUCCEEDED":
                # Get results
                return manager.execute_api_call("GET", f"jobs/{job_id}/result")
            elif status == "FAILED":
                error_msg = status_response.get("errorMessage", "Unknown error")
                raise VendorSDKError(
                    f"Batch job {job_id} failed: {error_msg}",
                    engine_name="sap_data_services",
                )
            elif status == "CANCELLED":
                raise VendorSDKError(
                    f"Batch job {job_id} was cancelled",
                    engine_name="sap_data_services",
                )

            time.sleep(self._config.async_poll_interval)

    # SAP-specific methods

    def cleanse_address(
        self,
        data: Any,
        input_columns: dict[str, str],
        output_columns: dict[str, str] | None = None,
        locale: str | None = None,
    ) -> dict[str, Any]:
        """Cleanse and standardize addresses.

        Args:
            data: Input data with address fields.
            input_columns: Mapping of address components to column names.
            output_columns: Optional output column mapping.
            locale: Locale for address cleansing.

        Returns:
            Cleansed address data.

        Raises:
            VendorSDKError: If address cleansing is not enabled.
        """
        if not self._config.enable_address_cleansing:
            raise VendorSDKError(
                "Address cleansing is not enabled. "
                "Use config.with_address_cleansing(enabled=True)",
                engine_name="sap_data_services",
            )

        manager = self._get_connection_manager()
        self._ensure_connected()

        payload = {
            "repository": self._config.repository,
            "data": self._serialize_data(data),
            "inputColumns": input_columns,
            "outputColumns": output_columns or {},
            "locale": locale or self._config.locale,
            "mode": "CERTIFY",
        }

        if self._config.address_directories:
            payload["directories"] = list(self._config.address_directories)

        return manager.execute_api_call("POST", "address/cleanse", data=payload)

    def geocode(
        self,
        data: Any,
        address_columns: dict[str, str],
        precision: str = "HIGH",
    ) -> dict[str, Any]:
        """Geocode addresses to coordinates.

        Args:
            data: Input data with address fields.
            address_columns: Mapping of address components to column names.
            precision: Geocoding precision (HIGH, MEDIUM, LOW).

        Returns:
            Geocoded data with latitude/longitude.

        Raises:
            VendorSDKError: If geocoding is not enabled.
        """
        if not self._config.enable_geocoding:
            raise VendorSDKError(
                "Geocoding is not enabled. Use config.with_geocoding(enabled=True)",
                engine_name="sap_data_services",
            )

        manager = self._get_connection_manager()
        self._ensure_connected()

        payload = {
            "repository": self._config.repository,
            "data": self._serialize_data(data),
            "addressColumns": address_columns,
            "precision": precision,
        }

        return manager.execute_api_call("POST", "geocode/execute", data=payload)

    def analyze_text(
        self,
        data: Any,
        text_column: str,
        analysis_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze text content for entities and sentiment.

        Args:
            data: Input data with text field.
            text_column: Column containing text to analyze.
            analysis_types: Types of analysis (ENTITY, SENTIMENT, LANGUAGE).

        Returns:
            Text analysis results.

        Raises:
            VendorSDKError: If text analysis is not enabled.
        """
        if not self._config.enable_text_analysis:
            raise VendorSDKError(
                "Text analysis is not enabled. Use config.with_text_analysis(enabled=True)",
                engine_name="sap_data_services",
            )

        manager = self._get_connection_manager()
        self._ensure_connected()

        payload = {
            "repository": self._config.repository,
            "data": self._serialize_data(data),
            "textColumn": text_column,
            "analysisTypes": analysis_types or ["ENTITY", "SENTIMENT"],
        }

        return manager.execute_api_call("POST", "text/analyze", data=payload)


# =============================================================================
# Factory Function
# =============================================================================


def create_sap_data_services_adapter(
    api_endpoint: str,
    *,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    repository: str = "Central",
    execution_mode: SAPExecutionMode = SAPExecutionMode.REALTIME,
    **kwargs: Any,
) -> SAPDataServicesAdapter:
    """Create a configured SAP Data Services adapter.

    Convenience factory function for creating adapters with common configurations.

    Args:
        api_endpoint: SAP Data Services API endpoint URL.
        api_key: API key for authentication.
        username: Username for basic authentication.
        password: Password for basic authentication.
        repository: Repository name.
        execution_mode: Execution mode (REALTIME, BATCH, EMBEDDED).
        **kwargs: Additional configuration options.

    Returns:
        Configured SAPDataServicesAdapter instance.

    Example:
        >>> adapter = create_sap_data_services_adapter(
        ...     api_endpoint="https://sap-ds.example.com/dswsbobje/rest/v1",
        ...     username="admin",
        ...     password="secret",
        ...     repository="DataQuality",
        ... )
    """
    # Determine auth type
    if api_key:
        auth_type = AuthType.API_KEY
    elif username and password:
        auth_type = AuthType.BASIC
    else:
        auth_type = AuthType.NONE

    config = SAPDataServicesConfig(
        api_endpoint=api_endpoint,
        api_key=api_key,
        username=username,
        password=password,
        auth_type=auth_type,
        repository=repository,
        execution_mode=execution_mode,
        **kwargs,
    )

    return SAPDataServicesAdapter(config=config)
