"""Talend Data Quality Engine Adapter.

This module provides an adapter for Talend Data Quality, enabling
integration with the truthound-orchestration framework.

Talend Data Quality provides:
- Data profiling and discovery
- Data quality rules and indicators
- Data cleansing and standardization
- Reference data management

This adapter communicates with Talend DQ via its REST API or embedded mode,
translating common rule formats to Talend-specific rules and converting results.

Example:
    >>> from packages.enterprise.engines import TalendAdapter, TalendConfig
    >>> config = TalendConfig(
    ...     api_endpoint="https://talend.example.com/api/v1",
    ...     api_key="your-api-key",
    ... )
    >>> engine = TalendAdapter(config=config)
    >>> with engine:
    ...     result = engine.check(data, rules)
    ...     profile = engine.profile(data)

Notes:
    - Requires `talend-data-quality` package for embedded mode
    - Supports both Talend Cloud and on-premise deployments
    - API endpoint typically: https://{host}/api/v1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Self

from packages.enterprise.engines.base import (
    AuthType,
    BaseConnectionManager,
    BaseResultConverter,
    BaseRuleTranslator,
    ConnectionMode,
    DataTransferMode,
    EnterpriseEngineAdapter,
    EnterpriseEngineConfig,
    RuleTranslationError,
    VendorSDKError,
)
from common.base import (
    CheckResult,
    CheckStatus,
    ColumnProfile,
    LearnResult,
    LearnedRule,
    LearnStatus,
    ProfileResult,
    ProfileStatus,
    Severity,
    ValidationFailure,
)
from common.engines.base import EngineCapabilities
from common.health import HealthCheckResult, HealthStatus


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# =============================================================================
# Enums
# =============================================================================


class TalendExecutionMode(Enum):
    """Talend execution mode."""

    API = auto()  # Use Talend REST API
    EMBEDDED = auto()  # Use embedded Talend libraries
    STUDIO = auto()  # Connect to Talend Studio


class TalendIndicatorType(Enum):
    """Talend indicator types for profiling."""

    ROW_COUNT = "rowCount"
    NULL_COUNT = "nullCount"
    BLANK_COUNT = "blankCount"
    DISTINCT_COUNT = "distinctCount"
    UNIQUE_COUNT = "uniqueCount"
    DUPLICATE_COUNT = "duplicateCount"
    MIN_VALUE = "minValue"
    MAX_VALUE = "maxValue"
    MEAN = "mean"
    MEDIAN = "median"
    LOWER_QUARTILE = "lowerQuartile"
    UPPER_QUARTILE = "upperQuartile"
    PATTERN_FREQUENCY = "patternFrequency"
    PATTERN_LOW_FREQUENCY = "patternLowFrequency"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class TalendConfig(EnterpriseEngineConfig):
    """Configuration for Talend Data Quality adapter.

    Extends EnterpriseEngineConfig with Talend-specific settings.

    Attributes:
        execution_mode: Talend execution mode (API, EMBEDDED, STUDIO).
        workspace: Talend workspace name.
        project: Talend project name.
        repository_path: Path to Talend repository.
        analysis_name: Default analysis name.
        pattern_catalog: Pattern catalog name.
        indicator_catalog: Indicator catalog name.
        semantic_model: Semantic model name.
        embedded_lib_path: Path to embedded Talend libraries.
        use_metadata: Whether to use Talend metadata repository.
        cache_metadata: Whether to cache metadata.

    Example:
        >>> config = TalendConfig(
        ...     api_endpoint="https://talend.example.com/api/v1",
        ...     api_key="secret",
        ...     workspace="Production",
        ...     project="DataQuality",
        ... )
    """

    # Talend-specific settings
    execution_mode: TalendExecutionMode = TalendExecutionMode.API
    workspace: str = ""
    project: str = ""
    repository_path: str = ""
    analysis_name: str = ""
    pattern_catalog: str = ""
    indicator_catalog: str = ""
    semantic_model: str = ""

    # Embedded mode settings
    embedded_lib_path: str = ""

    # Metadata settings
    use_metadata: bool = True
    cache_metadata: bool = True

    # Analysis settings
    default_row_limit: int = 100000
    sample_percentage: float = 100.0  # 100% = full analysis
    enable_patterns: bool = True
    enable_semantic: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Call validation methods directly instead of super().__post_init__()
        # to avoid issues with frozen dataclasses and slots
        self._validate_base_config()
        self._validate_enterprise_config()
        self._validate_talend_config()

    def _validate_talend_config(self) -> None:
        """Validate Talend-specific configuration values."""
        if self.sample_percentage <= 0 or self.sample_percentage > 100:
            raise ValueError("sample_percentage must be between 0 and 100")
        if self.default_row_limit < 1:
            raise ValueError("default_row_limit must be at least 1")

    def with_execution_mode(self, mode: TalendExecutionMode) -> Self:
        """Create config with execution mode.

        Args:
            mode: Talend execution mode.

        Returns:
            New configuration with mode.
        """
        return self._copy_with(execution_mode=mode)

    def with_workspace(self, workspace: str) -> Self:
        """Create config with workspace.

        Args:
            workspace: Talend workspace name.

        Returns:
            New configuration with workspace.
        """
        return self._copy_with(workspace=workspace)

    def with_project(self, project: str, repository_path: str = "") -> Self:
        """Create config with project settings.

        Args:
            project: Talend project name.
            repository_path: Repository path.

        Returns:
            New configuration with project.
        """
        return self._copy_with(project=project, repository_path=repository_path)

    def with_analysis(self, name: str) -> Self:
        """Create config with default analysis name.

        Args:
            name: Analysis name.

        Returns:
            New configuration with analysis.
        """
        return self._copy_with(analysis_name=name)

    def with_embedded_mode(self, lib_path: str) -> Self:
        """Create config for embedded mode.

        Args:
            lib_path: Path to Talend libraries.

        Returns:
            New configuration for embedded mode.
        """
        return self._copy_with(
            execution_mode=TalendExecutionMode.EMBEDDED,
            embedded_lib_path=lib_path,
        )

    def with_sampling(
        self,
        row_limit: int | None = None,
        percentage: float | None = None,
    ) -> Self:
        """Create config with sampling settings.

        Args:
            row_limit: Maximum rows to analyze.
            percentage: Sample percentage (0-100).

        Returns:
            New configuration with sampling.
        """
        updates: dict[str, Any] = {}
        if row_limit is not None:
            updates["default_row_limit"] = row_limit
        if percentage is not None:
            updates["sample_percentage"] = percentage
        return self._copy_with(**updates)


# Preset configurations
DEFAULT_TALEND_CONFIG = TalendConfig()

PRODUCTION_TALEND_CONFIG = TalendConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    health_check_interval_seconds=60.0,
    timeout_seconds=120.0,
    connect_timeout_seconds=30.0,
    max_retries=5,
    retry_delay_seconds=5.0,
    verify_ssl=True,
    pool_size=10,
    execution_mode=TalendExecutionMode.API,
    cache_metadata=True,
    default_row_limit=1000000,
)

DEVELOPMENT_TALEND_CONFIG = TalendConfig(
    auto_start=False,
    auto_stop=True,
    health_check_enabled=False,
    timeout_seconds=60.0,
    max_retries=2,
    verify_ssl=False,
    execution_mode=TalendExecutionMode.API,
    default_row_limit=10000,
    sample_percentage=10.0,
)

EMBEDDED_TALEND_CONFIG = TalendConfig(
    auto_start=True,
    auto_stop=True,
    execution_mode=TalendExecutionMode.EMBEDDED,
    use_metadata=False,
)


# =============================================================================
# Rule Translator
# =============================================================================


class TalendRuleTranslator(BaseRuleTranslator):
    """Translates common rules to Talend DQ rule format.

    Talend uses:
    - Indicators: Metrics computed on data (null count, pattern match, etc.)
    - Patterns: Regular expression patterns for validation
    - Rules: Business rules that combine indicators and thresholds
    """

    # Mapping from common rule types to Talend indicators/patterns
    RULE_MAPPING = {
        # Simple Indicators
        "not_null": "NOT_NULL_INDICATOR",
        "not_empty": "NOT_BLANK_INDICATOR",
        "required": "NOT_NULL_INDICATOR",

        # Count Indicators
        "unique": "UNIQUE_COUNT_INDICATOR",
        "duplicate": "DUPLICATE_COUNT_INDICATOR",
        "distinct": "DISTINCT_COUNT_INDICATOR",

        # Value Indicators
        "in_set": "VALUE_IN_LIST_INDICATOR",
        "in_range": "VALUE_IN_RANGE_INDICATOR",
        "greater_than": "VALUE_GREATER_THAN_INDICATOR",
        "less_than": "VALUE_LESS_THAN_INDICATOR",

        # Pattern Indicators
        "regex": "PATTERN_MATCHING_INDICATOR",
        "email": "EMAIL_PATTERN_INDICATOR",
        "phone": "PHONE_PATTERN_INDICATOR",
        "date": "DATE_PATTERN_INDICATOR",
        "url": "URL_PATTERN_INDICATOR",

        # Type Indicators
        "dtype": "DATA_TYPE_INDICATOR",
        "length": "TEXT_LENGTH_INDICATOR",
        "min_length": "MIN_LENGTH_INDICATOR",
        "max_length": "MAX_LENGTH_INDICATOR",

        # Statistical Indicators
        "outlier": "OUTLIER_INDICATOR",
        "distribution": "BENFORD_LAW_INDICATOR",

        # Referential Indicators
        "lookup": "REFERENCE_DATA_INDICATOR",
        "foreign_key": "REFERENTIAL_INTEGRITY_INDICATOR",
    }

    def _get_rule_mapping(self) -> dict[str, str]:
        """Return rule type mapping."""
        return self.RULE_MAPPING.copy()

    def _translate_rule_params(
        self,
        rule_type: str,
        rule: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate rule parameters to Talend format.

        Args:
            rule_type: Talend indicator type.
            rule: Original rule dictionary.

        Returns:
            Talend-specific parameter dictionary.
        """
        params: dict[str, Any] = {}

        # Common fields
        column = rule.get("column")
        if column:
            params["analyzedColumn"] = column

        # Threshold (default 100% for quality indicators)
        threshold = rule.get("threshold", 100.0)
        params["threshold"] = threshold

        # Rule-specific parameters
        if rule_type == "VALUE_IN_LIST_INDICATOR":
            values = rule.get("values", [])
            params["validValues"] = list(values)
            params["ignoreCase"] = rule.get("ignore_case", False)

        elif rule_type == "VALUE_IN_RANGE_INDICATOR":
            params["minValue"] = rule.get("min")
            params["maxValue"] = rule.get("max")
            params["includeMin"] = rule.get("include_min", True)
            params["includeMax"] = rule.get("include_max", True)

        elif rule_type in ("VALUE_GREATER_THAN_INDICATOR", "VALUE_LESS_THAN_INDICATOR"):
            params["comparisonValue"] = rule.get("value")
            params["inclusive"] = rule.get("inclusive", False)

        elif rule_type == "PATTERN_MATCHING_INDICATOR":
            params["pattern"] = rule.get("pattern", "")
            params["patternType"] = "REGEX"

        elif rule_type in ("EMAIL_PATTERN_INDICATOR", "PHONE_PATTERN_INDICATOR",
                          "DATE_PATTERN_INDICATOR", "URL_PATTERN_INDICATOR"):
            # Built-in patterns use default settings
            params["strictMode"] = rule.get("strict", True)

        elif rule_type == "DATA_TYPE_INDICATOR":
            params["expectedType"] = self._map_dtype(rule.get("dtype", "string"))

        elif rule_type in ("MIN_LENGTH_INDICATOR", "TEXT_LENGTH_INDICATOR"):
            params["minLength"] = rule.get("length", rule.get("min", 0))

        elif rule_type == "MAX_LENGTH_INDICATOR":
            params["maxLength"] = rule.get("length", rule.get("max", 255))

        elif rule_type == "OUTLIER_INDICATOR":
            params["method"] = rule.get("method", "IQR")
            params["multiplier"] = rule.get("multiplier", 1.5)

        elif rule_type == "REFERENCE_DATA_INDICATOR":
            params["referenceDataSource"] = rule.get("reference_source", "")
            params["referenceColumn"] = rule.get("reference_column", "")
            params["matchType"] = rule.get("match_type", "EXACT")

        return params

    def _map_dtype(self, dtype: str) -> str:
        """Map common dtype to Talend data type."""
        mapping = {
            "string": "STRING",
            "str": "STRING",
            "int": "INTEGER",
            "int32": "INTEGER",
            "int64": "LONG",
            "integer": "INTEGER",
            "float": "DOUBLE",
            "float32": "FLOAT",
            "float64": "DOUBLE",
            "double": "DOUBLE",
            "decimal": "BIGDECIMAL",
            "bool": "BOOLEAN",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "datetime": "DATETIME",
            "timestamp": "DATETIME",
        }
        return mapping.get(dtype.lower(), "STRING")


# =============================================================================
# Result Converter
# =============================================================================


class TalendResultConverter(BaseResultConverter):
    """Converts Talend DQ results to common format.

    Talend returns results in an indicator-based structure with
    match/non-match counts and percentages.
    """

    # Maps Talend severity to common Severity enum
    # Note: Severity enum has CRITICAL, ERROR, WARNING, INFO (not HIGH/MEDIUM/LOW)
    SEVERITY_MAPPING = {
        "CRITICAL": Severity.CRITICAL,
        "ERROR": Severity.ERROR,
        "WARNING": Severity.WARNING,
        "INFO": Severity.INFO,
        "OK": Severity.INFO,
    }

    def _extract_check_items(
        self,
        vendor_result: Any,
    ) -> list[dict[str, Any]]:
        """Extract check items from Talend result.

        Args:
            vendor_result: Result from Talend API.

        Returns:
            List of normalized check items.
        """
        items: list[dict[str, Any]] = []

        # Handle mock/dict results for testing
        if isinstance(vendor_result, dict):
            return self._extract_from_dict(vendor_result)

        # Handle actual Talend result objects
        if hasattr(vendor_result, "indicators"):
            for indicator in vendor_result.indicators:
                items.append({
                    "rule_name": getattr(indicator, "name", "unknown"),
                    "column": getattr(indicator, "analyzedColumn", None),
                    "passed": self._is_indicator_passed(indicator),
                    "failed_count": getattr(indicator, "nonMatchCount", 0),
                    "message": getattr(indicator, "message", ""),
                    "severity": self._determine_severity(indicator),
                    "details": {
                        "matchCount": getattr(indicator, "matchCount", 0),
                        "matchPercentage": getattr(indicator, "matchPercentage", 100),
                        "threshold": getattr(indicator, "threshold", 100),
                    },
                })

        return items

    def _extract_from_dict(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract items from dictionary result.

        Args:
            result: Dictionary result.

        Returns:
            List of check items.
        """
        items: list[dict[str, Any]] = []

        # Handle analysis result format
        if "analysisResult" in result:
            analysis = result["analysisResult"]
            for indicator in analysis.get("indicators", []):
                match_pct = indicator.get("matchPercentage", 100)
                threshold = indicator.get("threshold", 100)
                items.append({
                    "rule_name": indicator.get("name", "unknown"),
                    "column": indicator.get("analyzedColumn"),
                    "passed": match_pct >= threshold,
                    "failed_count": indicator.get("nonMatchCount", 0),
                    "message": indicator.get("message", f"Match: {match_pct:.1f}%"),
                    "severity": self._compute_severity(match_pct, threshold),
                    "details": indicator,
                })

        # Handle indicator results format
        elif "indicators" in result:
            for indicator in result["indicators"]:
                match_pct = indicator.get("matchPercentage", 100)
                threshold = indicator.get("threshold", 100)
                items.append({
                    "rule_name": indicator.get("name", "unknown"),
                    "column": indicator.get("analyzedColumn"),
                    "passed": match_pct >= threshold,
                    "failed_count": indicator.get("nonMatchCount", 0),
                    "message": indicator.get("message", f"Match: {match_pct:.1f}%"),
                    "severity": self._compute_severity(match_pct, threshold),
                    "details": indicator,
                })

        return items

    def _is_indicator_passed(self, indicator: Any) -> bool:
        """Determine if an indicator passed.

        Args:
            indicator: Talend indicator object.

        Returns:
            True if passed.
        """
        match_pct = getattr(indicator, "matchPercentage", 100)
        threshold = getattr(indicator, "threshold", 100)
        return match_pct >= threshold

    def _determine_severity(self, indicator: Any) -> str:
        """Determine severity from indicator result.

        Args:
            indicator: Talend indicator object.

        Returns:
            Severity string.
        """
        match_pct = getattr(indicator, "matchPercentage", 100)
        threshold = getattr(indicator, "threshold", 100)
        return self._compute_severity(match_pct, threshold)

    def _compute_severity(self, match_pct: float, threshold: float) -> str:
        """Compute severity from match percentage.

        Args:
            match_pct: Match percentage.
            threshold: Required threshold.

        Returns:
            Severity string.
        """
        if match_pct >= threshold:
            return "OK"
        gap = threshold - match_pct
        if gap > 20:
            return "CRITICAL"
        if gap > 10:
            return "ERROR"
        if gap > 5:
            return "WARNING"
        return "INFO"

    def convert_profile_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
    ) -> ProfileResult:
        """Convert Talend profile result to ProfileResult.

        Args:
            vendor_result: Result from Talend API.
            start_time: Start time for duration calculation.

        Returns:
            Common ProfileResult.
        """
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        columns: list[ColumnProfile] = []

        # Handle mock/dict results
        if isinstance(vendor_result, dict):
            row_count = vendor_result.get("rowCount", 0)
            for col_data in vendor_result.get("columnAnalyses", []):
                columns.append(self._convert_column_profile(col_data))
        elif hasattr(vendor_result, "columnAnalyses"):
            row_count = getattr(vendor_result, "rowCount", 0)
            for col_analysis in vendor_result.columnAnalyses:
                columns.append(self._convert_column_analysis(col_analysis))
        else:
            row_count = 0

        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            columns=tuple(columns),
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            metadata={"engine": "talend"},
        )

    def _convert_column_profile(self, col_data: dict[str, Any]) -> ColumnProfile:
        """Convert column data dictionary to ColumnProfile.

        Args:
            col_data: Column data dictionary.

        Returns:
            ColumnProfile instance.
        """
        indicators = col_data.get("indicators", {})

        return ColumnProfile(
            column_name=col_data.get("columnName", ""),
            dtype=col_data.get("dataType", "unknown"),
            null_count=indicators.get("nullCount", 0),
            null_percentage=indicators.get("nullPercentage", 0.0),
            unique_count=indicators.get("distinctCount", 0),
            min_value=indicators.get("minValue"),
            max_value=indicators.get("maxValue"),
            mean=indicators.get("mean"),
            std=indicators.get("stdDev"),
            metadata={
                "blankCount": indicators.get("blankCount"),
                "uniqueCount": indicators.get("uniqueCount"),
                "duplicateCount": indicators.get("duplicateCount"),
                "patterns": col_data.get("patterns", []),
                "sampleValues": col_data.get("sampleValues", []),
            },
        )

    def _convert_column_analysis(self, col_analysis: Any) -> ColumnProfile:
        """Convert Talend column analysis object to ColumnProfile.

        Args:
            col_analysis: Talend column analysis object.

        Returns:
            ColumnProfile instance.
        """
        return ColumnProfile(
            column_name=getattr(col_analysis, "columnName", ""),
            dtype=getattr(col_analysis, "dataType", "unknown"),
            null_count=getattr(col_analysis, "nullCount", 0),
            null_percentage=getattr(col_analysis, "nullPercentage", 0.0),
            unique_count=getattr(col_analysis, "distinctCount", 0),
            min_value=getattr(col_analysis, "minValue", None),
            max_value=getattr(col_analysis, "maxValue", None),
            mean=getattr(col_analysis, "mean", None),
            std=getattr(col_analysis, "stdDev", None),
        )

    def convert_learn_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
    ) -> LearnResult:
        """Convert Talend discovery result to LearnResult.

        Args:
            vendor_result: Result from Talend pattern discovery.
            start_time: Start time for duration calculation.

        Returns:
            Common LearnResult.
        """
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        rules: list[LearnedRule] = []

        # Handle mock/dict results
        if isinstance(vendor_result, dict):
            for discovery in vendor_result.get("discoveries", []):
                rules.append(LearnedRule(
                    column=discovery.get("column", ""),
                    rule_type=discovery.get("ruleType", "pattern"),
                    parameters=discovery.get("parameters", {}),
                    confidence=discovery.get("confidence", 0.0),
                    sample_size=discovery.get("sampleMatches", 0),
                    metadata={
                        "description": discovery.get("description", ""),
                    },
                ))

        return LearnResult(
            status=LearnStatus.COMPLETED,
            rules=tuple(rules),
            execution_time_ms=execution_time_ms,
            metadata={"engine": "talend"},
        )


# =============================================================================
# Connection Manager
# =============================================================================


class TalendConnectionManager(BaseConnectionManager):
    """Manages connections to Talend Data Quality.

    Supports both API mode and embedded mode connections.
    """

    def __init__(
        self,
        config: TalendConfig,
    ) -> None:
        """Initialize connection manager.

        Args:
            config: Talend configuration.
        """
        super().__init__(config, name="talend")
        self._config: TalendConfig = config
        self._session: Any = None
        self._embedded_engine: Any = None

    def _do_connect(self) -> Any:
        """Establish connection to Talend.

        Returns:
            Connection object (session or embedded engine).
        """
        if self._config.execution_mode == TalendExecutionMode.EMBEDDED:
            return self._connect_embedded()
        else:
            return self._connect_api()

    def _connect_api(self) -> Any:
        """Connect via REST API.

        Returns:
            Requests session.
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
            session.headers["Authorization"] = f"Bearer {self._config.api_key}"
        elif self._config.auth_type == AuthType.BASIC:
            session.auth = (self._config.username, self._config.password)

        # Set common headers
        session.headers["Content-Type"] = "application/json"
        session.headers["Accept"] = "application/json"

        # Test connection
        self._test_api_connection(session)

        self._session = session
        return session

    def _connect_embedded(self) -> Any:
        """Connect in embedded mode.

        Returns:
            Embedded Talend engine.
        """
        try:
            # Try to import Talend libraries
            import sys
            if self._config.embedded_lib_path:
                sys.path.insert(0, self._config.embedded_lib_path)

            from talend.dataquality import DataQualityEngine
            self._embedded_engine = DataQualityEngine()
            self._embedded_engine.initialize()
            return self._embedded_engine

        except ImportError as e:
            raise VendorSDKError(
                "Talend Data Quality libraries not found. "
                "Ensure embedded_lib_path is correct.",
                engine_name="talend",
                cause=e,
            ) from e

    def _test_api_connection(self, session: Any) -> None:
        """Test API connection.

        Args:
            session: Requests session.
        """
        try:
            response = session.get(
                f"{self._config.api_endpoint}/system/info",
                timeout=self._config.connect_timeout_seconds,
            )
            response.raise_for_status()
        except Exception:
            # Try alternative endpoint
            try:
                response = session.get(
                    f"{self._config.api_endpoint}/health",
                    timeout=self._config.connect_timeout_seconds,
                )
                response.raise_for_status()
            except Exception as e:
                from packages.enterprise.engines.base import ConnectionError
                raise ConnectionError(
                    f"Failed to connect to Talend API: {e}",
                    engine_name="talend",
                    cause=e,
                ) from e

    def _do_disconnect(self) -> None:
        """Close connection."""
        if self._session:
            self._session.close()
            self._session = None

        if self._embedded_engine:
            try:
                self._embedded_engine.shutdown()
            except Exception:
                pass
            self._embedded_engine = None

    def _do_health_check(self) -> bool:
        """Check connection health.

        Returns:
            True if healthy.
        """
        if self._config.execution_mode == TalendExecutionMode.EMBEDDED:
            return self._embedded_engine is not None

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

    def get_embedded_engine(self) -> Any:
        """Get the embedded engine.

        Returns:
            Embedded Talend engine.
        """
        if self._config.execution_mode != TalendExecutionMode.EMBEDDED:
            raise VendorSDKError("Not in embedded mode")
        return self._embedded_engine

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
        """
        session = self.get_session()
        url = f"{self._config.api_endpoint}{endpoint}"

        kwargs.setdefault("timeout", self._config.timeout_seconds)

        response = session.request(method, url, **kwargs)
        response.raise_for_status()

        return response.json()


# =============================================================================
# Talend Adapter
# =============================================================================


class TalendAdapter(EnterpriseEngineAdapter):
    """Adapter for Talend Data Quality.

    Provides integration with Talend DQ through REST API or embedded mode.

    Features:
        - Data quality validation using Talend indicators
        - Data profiling with pattern discovery
        - Rule learning from data patterns
        - Support for embedded execution
        - Connection pooling for API mode

    Example:
        >>> config = TalendConfig(
        ...     api_endpoint="https://talend.example.com/api/v1",
        ...     api_key="secret",
        ...     workspace="Production",
        ... )
        >>> engine = TalendAdapter(config=config)
        >>> with engine:
        ...     result = engine.check(df, rules)
        ...     profile = engine.profile(df)
        ...     learned = engine.learn(df)
    """

    def __init__(
        self,
        config: TalendConfig | None = None,
    ) -> None:
        """Initialize Talend adapter.

        Args:
            config: Talend configuration.
        """
        super().__init__(config or DEFAULT_TALEND_CONFIG)
        self._config: TalendConfig = self._config  # Type narrowing

    @property
    def engine_name(self) -> str:
        """Return engine name."""
        return "talend"

    @property
    def engine_version(self) -> str:
        """Return engine version."""
        if self._vendor_library:
            return getattr(self._vendor_library, "__version__", "1.0.0")
        return "1.0.0"

    def _get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities."""
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,  # Talend supports pattern discovery
            supports_async=False,
            supports_streaming=False,
            supported_data_types=("polars", "pandas", "csv", "parquet", "jdbc"),
            supported_rule_types=self._get_rule_translator().get_supported_rule_types(),
        )

    def _get_description(self) -> str:
        """Return engine description."""
        return "Talend Data Quality - Open source and enterprise data quality"

    def _get_homepage(self) -> str:
        """Return engine homepage."""
        return "https://www.talend.com/products/data-quality/"

    def _ensure_vendor_library(self) -> Any:
        """Ensure Talend SDK is loaded.

        Returns:
            Vendor library module.

        Raises:
            VendorSDKError: If library cannot be loaded.
        """
        if self._vendor_library is not None:
            return self._vendor_library

        try:
            # Try embedded Talend library
            if self._config.execution_mode == TalendExecutionMode.EMBEDDED:
                import talend.dataquality as tdq
                self._vendor_library = tdq
            else:
                # Use requests for API mode
                import requests
                self._vendor_library = requests
        except ImportError as e:
            # Fall back to requests for API mode
            try:
                import requests
                self._vendor_library = requests
            except ImportError as e2:
                raise VendorSDKError(
                    "Talend libraries or requests not found. "
                    "Install with: pip install requests",
                    engine_name="talend",
                    cause=e2,
                ) from e2

        return self._vendor_library

    def _create_connection_manager(self) -> BaseConnectionManager:
        """Create Talend connection manager.

        Returns:
            Connection manager instance.
        """
        return TalendConnectionManager(self._config)

    def _create_rule_translator(self) -> BaseRuleTranslator:
        """Create Talend rule translator.

        Returns:
            Rule translator instance.
        """
        return TalendRuleTranslator()

    def _create_result_converter(self) -> BaseResultConverter:
        """Create Talend result converter.

        Returns:
            Result converter instance.
        """
        return TalendResultConverter()

    def _execute_check(
        self,
        data: Any,
        translated_rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute check using Talend.

        Args:
            data: Converted data (Pandas DataFrame).
            translated_rules: Talend-format indicators.
            **kwargs: Additional parameters.

        Returns:
            Talend result object.
        """
        if self._config.execution_mode == TalendExecutionMode.EMBEDDED:
            return self._execute_check_embedded(data, translated_rules, **kwargs)
        else:
            return self._execute_check_api(data, translated_rules, **kwargs)

    def _execute_check_api(
        self,
        data: Any,
        rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute check via API.

        Args:
            data: Data to validate.
            rules: Translated indicators.
            **kwargs: Additional parameters.

        Returns:
            API response.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, TalendConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        # Build request payload
        payload = {
            "workspace": self._config.workspace,
            "project": self._config.project,
            "data": self._serialize_data(data),
            "indicators": rules,
            "options": {
                "rowLimit": self._config.default_row_limit,
                "samplePercentage": self._config.sample_percentage,
            },
        }

        # Add analysis name if specified
        analysis = kwargs.get("analysis", self._config.analysis_name)
        if analysis:
            payload["analysisName"] = analysis

        return connection.execute_api_call(
            "POST",
            "/analysis/execute",
            json=payload,
        )

    def _execute_check_embedded(
        self,
        data: Any,
        rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute check in embedded mode.

        Args:
            data: Data to validate.
            rules: Translated indicators.
            **kwargs: Additional parameters.

        Returns:
            Embedded engine result.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, TalendConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        engine = connection.get_embedded_engine()

        # Configure analysis
        analysis = engine.createAnalysis()
        for rule in rules:
            indicator = engine.createIndicator(rule["type"])
            for key, value in rule.items():
                if key != "type":
                    indicator.setParameter(key, value)
            analysis.addIndicator(indicator)

        # Set data source
        analysis.setData(data)

        # Execute
        return analysis.execute()

    def _execute_profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute profiling using Talend.

        Args:
            data: Converted data (Pandas DataFrame).
            **kwargs: Additional parameters.

        Returns:
            Talend result object.
        """
        if self._config.execution_mode == TalendExecutionMode.EMBEDDED:
            return self._execute_profile_embedded(data, **kwargs)
        else:
            return self._execute_profile_api(data, **kwargs)

    def _execute_profile_api(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute profiling via API.

        Args:
            data: Data to profile.
            **kwargs: Additional parameters.

        Returns:
            API response.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, TalendConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        # Build request payload
        payload = {
            "workspace": self._config.workspace,
            "project": self._config.project,
            "data": self._serialize_data(data),
            "options": {
                "rowLimit": self._config.default_row_limit,
                "samplePercentage": self._config.sample_percentage,
                "computePatterns": self._config.enable_patterns,
                "computeSemantic": self._config.enable_semantic,
            },
        }

        # Standard indicators for profiling
        payload["indicators"] = [
            {"type": "ROW_COUNT"},
            {"type": "NULL_COUNT"},
            {"type": "BLANK_COUNT"},
            {"type": "DISTINCT_COUNT"},
            {"type": "UNIQUE_COUNT"},
            {"type": "DUPLICATE_COUNT"},
            {"type": "MIN_VALUE"},
            {"type": "MAX_VALUE"},
            {"type": "MEAN"},
        ]

        if self._config.enable_patterns:
            payload["indicators"].extend([
                {"type": "PATTERN_FREQUENCY"},
                {"type": "PATTERN_LOW_FREQUENCY"},
            ])

        return connection.execute_api_call(
            "POST",
            "/profile/execute",
            json=payload,
        )

    def _execute_profile_embedded(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute profiling in embedded mode.

        Args:
            data: Data to profile.
            **kwargs: Additional parameters.

        Returns:
            Embedded engine result.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, TalendConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        engine = connection.get_embedded_engine()

        # Create profiling analysis
        profile = engine.createProfile()
        profile.setData(data)
        profile.setRowLimit(self._config.default_row_limit)
        profile.enablePatterns(self._config.enable_patterns)

        return profile.execute()

    def _execute_learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute rule learning using Talend pattern discovery.

        Args:
            data: Data to learn from.
            **kwargs: Additional parameters.

        Returns:
            Talend discovery result.
        """
        if self._config.execution_mode == TalendExecutionMode.EMBEDDED:
            return self._execute_learn_embedded(data, **kwargs)
        else:
            return self._execute_learn_api(data, **kwargs)

    def _execute_learn_api(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute learning via API.

        Args:
            data: Data to learn from.
            **kwargs: Additional parameters.

        Returns:
            API response.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, TalendConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        payload = {
            "workspace": self._config.workspace,
            "project": self._config.project,
            "data": self._serialize_data(data),
            "options": {
                "discoverPatterns": True,
                "discoverSemanticTypes": self._config.enable_semantic,
                "sampleSize": kwargs.get("sample_size", 10000),
                "minConfidence": kwargs.get("min_confidence", 0.8),
            },
        }

        return connection.execute_api_call(
            "POST",
            "/discovery/execute",
            json=payload,
        )

    def _execute_learn_embedded(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute learning in embedded mode.

        Args:
            data: Data to learn from.
            **kwargs: Additional parameters.

        Returns:
            Embedded engine result.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, TalendConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        engine = connection.get_embedded_engine()

        discovery = engine.createDiscovery()
        discovery.setData(data)
        discovery.enablePatternDiscovery(True)
        discovery.setMinConfidence(kwargs.get("min_confidence", 0.8))

        return discovery.execute()

    def _serialize_data(self, data: Any) -> dict[str, Any]:
        """Serialize data for API payload.

        Args:
            data: Pandas DataFrame.

        Returns:
            Serialized data dictionary.
        """
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

    def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Learn validation rules from the data.

        Talend supports pattern discovery and semantic type detection.

        Args:
            data: Data to learn from.
            **kwargs: Additional parameters:
                - sample_size: Number of rows to sample
                - min_confidence: Minimum confidence threshold

        Returns:
            LearnResult with discovered rules.
        """
        start_time = time.perf_counter()

        # Ensure engine is ready
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            if self._config.auto_start and state == EngineState.CREATED:
                self.start()
            else:
                from common.engines.lifecycle import EngineNotStartedError
                raise EngineNotStartedError(self.engine_name)

        # Convert data
        converted_data = self._convert_data(data)

        # Execute learn
        vendor_result = self._execute_learn(converted_data, **kwargs)

        # Convert result
        converter = self._get_result_converter()
        if isinstance(converter, TalendResultConverter):
            return converter.convert_learn_result(
                vendor_result,
                start_time=start_time,
            )
        else:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return LearnResult(
                rules=(),
                execution_time_ms=execution_time_ms,
                metadata={"engine": "talend"},
            )


# Need to import EngineState for the learn method
from common.engines.lifecycle import EngineState


# =============================================================================
# Factory Function
# =============================================================================


def create_talend_adapter(
    api_endpoint: str | None = None,
    api_key: str | None = None,
    execution_mode: TalendExecutionMode = TalendExecutionMode.API,
    embedded_lib_path: str | None = None,
    **kwargs: Any,
) -> TalendAdapter:
    """Create a Talend adapter with configuration.

    Factory function for convenient adapter creation.

    Args:
        api_endpoint: Talend API endpoint URL (for API mode).
        api_key: API key for authentication.
        execution_mode: Execution mode (API or EMBEDDED).
        embedded_lib_path: Path to Talend libraries (for embedded mode).
        **kwargs: Additional configuration options.

    Returns:
        Configured TalendAdapter instance.

    Example:
        >>> # API mode
        >>> adapter = create_talend_adapter(
        ...     api_endpoint="https://talend.example.com/api/v1",
        ...     api_key="secret",
        ...     workspace="Production",
        ... )

        >>> # Embedded mode
        >>> adapter = create_talend_adapter(
        ...     execution_mode=TalendExecutionMode.EMBEDDED,
        ...     embedded_lib_path="/opt/talend/lib",
        ... )
    """
    config = TalendConfig(
        api_endpoint=api_endpoint or "",
        api_key=api_key,
        auth_type=AuthType.API_KEY if api_key else AuthType.NONE,
        execution_mode=execution_mode,
        embedded_lib_path=embedded_lib_path or "",
        **kwargs,
    )
    return TalendAdapter(config=config)
