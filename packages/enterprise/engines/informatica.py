"""Informatica Data Quality (IDQ) Engine Adapter.

This module provides an adapter for Informatica Data Quality, enabling
integration with the truthound-orchestration framework.

Informatica IDQ is an enterprise data quality platform that provides:
- Data profiling and discovery
- Data quality rules and scorecards
- Data cleansing and standardization
- Match and merge capabilities

This adapter communicates with IDQ via its REST API, translating common
rule formats to IDQ-specific rules and converting results back.

Example:
    >>> from packages.enterprise.engines import InformaticaAdapter, InformaticaConfig
    >>> config = InformaticaConfig(
    ...     api_endpoint="https://idq.example.com/api/v2",
    ...     api_key="your-api-key",
    ... )
    >>> engine = InformaticaAdapter(config=config)
    >>> with engine:
    ...     result = engine.check(data, rules)
    ...     profile = engine.profile(data)

Notes:
    - Requires `informatica-dq-sdk` package (pip install informatica-dq-sdk)
    - API endpoint typically: https://{host}/api/v2
    - Supports both cloud and on-premise IDQ deployments
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class InformaticaConfig(EnterpriseEngineConfig):
    """Configuration for Informatica Data Quality adapter.

    Extends EnterpriseEngineConfig with Informatica-specific settings.

    Attributes:
        domain: IDQ domain name.
        project: IDQ project name.
        folder: IDQ folder path.
        scorecard_name: Default scorecard for validation.
        profile_name: Default profile for profiling.
        rule_set_name: Default rule set name.
        use_streaming: Whether to use streaming data transfer.
        async_execution: Whether to execute jobs asynchronously.
        poll_interval_seconds: Polling interval for async jobs.
        max_poll_attempts: Maximum polling attempts.

    Example:
        >>> config = InformaticaConfig(
        ...     api_endpoint="https://idq.example.com/api/v2",
        ...     api_key="secret",
        ...     domain="Default",
        ...     project="DataQuality",
        ... )
    """

    # Informatica-specific settings
    domain: str = "Default"
    project: str = ""
    folder: str = "/"
    scorecard_name: str = ""
    profile_name: str = ""
    rule_set_name: str = ""

    # Execution settings
    use_streaming: bool = False
    async_execution: bool = True
    poll_interval_seconds: float = 2.0
    max_poll_attempts: int = 300  # 10 minutes with 2s interval

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Call validation methods directly instead of super().__post_init__()
        # to avoid issues with frozen dataclasses and slots
        self._validate_base_config()
        self._validate_enterprise_config()
        self._validate_informatica_config()

    def _validate_informatica_config(self) -> None:
        """Validate Informatica-specific configuration values."""
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.max_poll_attempts < 1:
            raise ValueError("max_poll_attempts must be at least 1")

    def with_domain(self, domain: str) -> Self:
        """Create config with domain.

        Args:
            domain: IDQ domain name.

        Returns:
            New configuration with domain.
        """
        return self._copy_with(domain=domain)

    def with_project(self, project: str, folder: str = "/") -> Self:
        """Create config with project settings.

        Args:
            project: IDQ project name.
            folder: IDQ folder path.

        Returns:
            New configuration with project.
        """
        return self._copy_with(project=project, folder=folder)

    def with_scorecard(self, name: str) -> Self:
        """Create config with default scorecard.

        Args:
            name: Scorecard name.

        Returns:
            New configuration with scorecard.
        """
        return self._copy_with(scorecard_name=name)

    def with_profile(self, name: str) -> Self:
        """Create config with default profile.

        Args:
            name: Profile name.

        Returns:
            New configuration with profile.
        """
        return self._copy_with(profile_name=name)

    def with_async_execution(
        self,
        enabled: bool = True,
        poll_interval: float = 2.0,
        max_attempts: int = 300,
    ) -> Self:
        """Create config with async execution settings.

        Args:
            enabled: Whether to use async execution.
            poll_interval: Polling interval in seconds.
            max_attempts: Maximum polling attempts.

        Returns:
            New configuration with async settings.
        """
        return self._copy_with(
            async_execution=enabled,
            poll_interval_seconds=poll_interval,
            max_poll_attempts=max_attempts,
        )


# Preset configurations
DEFAULT_INFORMATICA_CONFIG = InformaticaConfig()

PRODUCTION_INFORMATICA_CONFIG = InformaticaConfig(
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
    async_execution=True,
    poll_interval_seconds=5.0,
)

DEVELOPMENT_INFORMATICA_CONFIG = InformaticaConfig(
    auto_start=False,
    auto_stop=True,
    health_check_enabled=False,
    timeout_seconds=60.0,
    max_retries=2,
    verify_ssl=False,
    async_execution=False,
)


# =============================================================================
# Rule Translator
# =============================================================================


class InformaticaRuleTranslator(BaseRuleTranslator):
    """Translates common rules to Informatica IDQ rule format.

    Informatica uses a different terminology:
    - Rules are organized into Rule Sets
    - Each rule has a Rule Specification
    - Rules reference Data Quality Dimensions (Completeness, Accuracy, etc.)
    """

    # Mapping from common rule types to Informatica rule specifications
    RULE_MAPPING = {
        # Completeness rules
        "not_null": "NULL_CHECK",
        "required": "REQUIRED_CHECK",
        "not_empty": "EMPTY_CHECK",

        # Uniqueness rules
        "unique": "UNIQUE_CHECK",
        "primary_key": "PRIMARY_KEY_CHECK",
        "duplicate": "DUPLICATE_CHECK",

        # Validity rules
        "in_set": "DOMAIN_CHECK",
        "in_range": "RANGE_CHECK",
        "regex": "PATTERN_CHECK",
        "dtype": "DATA_TYPE_CHECK",
        "length": "LENGTH_CHECK",
        "min_length": "MIN_LENGTH_CHECK",
        "max_length": "MAX_LENGTH_CHECK",

        # Accuracy rules
        "lookup": "REFERENCE_CHECK",
        "foreign_key": "REFERENTIAL_INTEGRITY_CHECK",

        # Format rules
        "email": "EMAIL_FORMAT_CHECK",
        "phone": "PHONE_FORMAT_CHECK",
        "date": "DATE_FORMAT_CHECK",

        # Statistical rules
        "outlier": "OUTLIER_CHECK",
        "distribution": "DISTRIBUTION_CHECK",
    }

    def _get_rule_mapping(self) -> dict[str, str]:
        """Return rule type mapping."""
        return self.RULE_MAPPING.copy()

    def _translate_rule_params(
        self,
        rule_type: str,
        rule: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate rule parameters to Informatica format.

        Args:
            rule_type: Informatica rule type.
            rule: Original rule dictionary.

        Returns:
            Informatica-specific parameter dictionary.
        """
        params: dict[str, Any] = {}

        # Common fields
        column = rule.get("column")
        if column:
            params["fieldName"] = column

        # Map severity
        severity = rule.get("severity", "medium")
        params["severity"] = self._map_severity(severity)

        # Rule-specific parameters
        if rule_type == "DOMAIN_CHECK":
            values = rule.get("values", [])
            params["validValues"] = list(values)

        elif rule_type == "RANGE_CHECK":
            if "min" in rule:
                params["minValue"] = rule["min"]
            if "max" in rule:
                params["maxValue"] = rule["max"]
            params["inclusive"] = rule.get("inclusive", True)

        elif rule_type == "PATTERN_CHECK":
            params["pattern"] = rule.get("pattern", "")
            params["matchType"] = rule.get("match_type", "REGEX")

        elif rule_type == "DATA_TYPE_CHECK":
            params["expectedType"] = self._map_dtype(rule.get("dtype", "string"))

        elif rule_type in ("MIN_LENGTH_CHECK", "LENGTH_CHECK"):
            params["minLength"] = rule.get("length", rule.get("min", 0))

        elif rule_type == "MAX_LENGTH_CHECK":
            params["maxLength"] = rule.get("length", rule.get("max", 255))

        elif rule_type == "REFERENCE_CHECK":
            params["referenceTable"] = rule.get("reference_table", "")
            params["referenceColumn"] = rule.get("reference_column", "")

        elif rule_type == "OUTLIER_CHECK":
            params["method"] = rule.get("method", "IQR")
            params["threshold"] = rule.get("threshold", 1.5)

        return params

    def _map_severity(self, severity: str) -> str:
        """Map common severity to Informatica severity."""
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
        """Map common dtype to Informatica data type."""
        mapping = {
            "string": "STRING",
            "str": "STRING",
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
            "datetime": "DATETIME",
            "timestamp": "TIMESTAMP",
        }
        return mapping.get(dtype.lower(), "STRING")


# =============================================================================
# Result Converter
# =============================================================================


class InformaticaResultConverter(BaseResultConverter):
    """Converts Informatica IDQ results to common format.

    Informatica returns results in a scorecard/rule-based structure.
    """

    # Maps Informatica severity to common Severity enum
    # Note: Severity enum has CRITICAL, ERROR, WARNING, INFO (not HIGH/MEDIUM/LOW)
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
        """Extract check items from Informatica result.

        Args:
            vendor_result: Result from Informatica API.

        Returns:
            List of normalized check items.
        """
        items: list[dict[str, Any]] = []

        # Handle mock/dict results for testing
        if isinstance(vendor_result, dict):
            return self._extract_from_dict(vendor_result)

        # Handle actual Informatica result objects
        if hasattr(vendor_result, "ruleResults"):
            for rule_result in vendor_result.ruleResults:
                items.append({
                    "rule_name": getattr(rule_result, "ruleName", "unknown"),
                    "column": getattr(rule_result, "fieldName", None),
                    "passed": getattr(rule_result, "passed", False),
                    "failed_count": getattr(rule_result, "failedRecordCount", 0),
                    "message": getattr(rule_result, "message", ""),
                    "severity": getattr(rule_result, "severity", "MEDIUM"),
                    "details": {
                        "score": getattr(rule_result, "score", 0),
                        "totalRecords": getattr(rule_result, "totalRecordCount", 0),
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

        # Handle scorecard format
        if "scorecard" in result:
            scorecard = result["scorecard"]
            for dimension in scorecard.get("dimensions", []):
                for rule in dimension.get("rules", []):
                    items.append({
                        "rule_name": rule.get("name", "unknown"),
                        "column": rule.get("field"),
                        "passed": rule.get("score", 0) >= rule.get("threshold", 90),
                        "failed_count": rule.get("failedCount", 0),
                        "message": rule.get("message", ""),
                        "severity": rule.get("severity", "MEDIUM"),
                        "details": rule,
                    })

        # Handle rule results format
        elif "ruleResults" in result:
            for rule in result["ruleResults"]:
                items.append({
                    "rule_name": rule.get("ruleName", "unknown"),
                    "column": rule.get("fieldName"),
                    "passed": rule.get("passed", False),
                    "failed_count": rule.get("failedRecordCount", 0),
                    "message": rule.get("message", ""),
                    "severity": rule.get("severity", "MEDIUM"),
                    "details": rule,
                })

        return items

    def convert_profile_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
    ) -> ProfileResult:
        """Convert Informatica profile result to ProfileResult.

        Args:
            vendor_result: Result from Informatica API.
            start_time: Start time for duration calculation.

        Returns:
            Common ProfileResult.
        """
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        columns: list[ColumnProfile] = []

        # Handle mock/dict results
        if isinstance(vendor_result, dict):
            row_count = vendor_result.get("rowCount", 0)
            for col_data in vendor_result.get("columns", []):
                columns.append(
                    ColumnProfile(
                        column_name=col_data.get("name", ""),
                        dtype=col_data.get("dataType", "unknown"),
                        null_count=col_data.get("nullCount", 0),
                        null_percentage=col_data.get("nullPercentage", 0.0),
                        unique_count=col_data.get("distinctCount", 0),
                        min_value=col_data.get("minValue"),
                        max_value=col_data.get("maxValue"),
                        mean=col_data.get("mean"),
                        std=col_data.get("stdDev"),
                        metadata={
                            **col_data,
                            "sampleValues": col_data.get("sampleValues", []),
                        },
                    )
                )
        elif hasattr(vendor_result, "columnProfiles"):
            row_count = getattr(vendor_result, "rowCount", 0)
            for col_profile in vendor_result.columnProfiles:
                columns.append(
                    ColumnProfile(
                        column_name=getattr(col_profile, "columnName", ""),
                        dtype=getattr(col_profile, "dataType", "unknown"),
                        null_count=getattr(col_profile, "nullCount", 0),
                        null_percentage=getattr(col_profile, "nullPercentage", 0.0),
                        unique_count=getattr(col_profile, "distinctCount", 0),
                        min_value=getattr(col_profile, "minValue", None),
                        max_value=getattr(col_profile, "maxValue", None),
                        mean=getattr(col_profile, "mean", None),
                        std=getattr(col_profile, "stdDev", None),
                    )
                )
        else:
            row_count = 0

        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            columns=tuple(columns),
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            metadata={"engine": "informatica"},
        )


# =============================================================================
# Connection Manager
# =============================================================================


class InformaticaConnectionManager(BaseConnectionManager):
    """Manages connections to Informatica Data Quality API.

    Handles authentication, session management, and API calls.
    """

    def __init__(
        self,
        config: InformaticaConfig,
    ) -> None:
        """Initialize connection manager.

        Args:
            config: Informatica configuration.
        """
        super().__init__(config, name="informatica")
        self._config: InformaticaConfig = config
        self._session: Any = None
        self._auth_token: str | None = None
        self._token_expires_at: float = 0

    def _do_connect(self) -> Any:
        """Establish connection to Informatica API.

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
            session.headers["X-API-Key"] = self._config.api_key
        elif self._config.auth_type == AuthType.BASIC:
            session.auth = (self._config.username, self._config.password)
        elif self._config.auth_type == AuthType.OAUTH2:
            self._authenticate_oauth2(session)

        # Test connection
        self._test_connection(session)

        self._session = session
        return session

    def _authenticate_oauth2(self, session: Any) -> None:
        """Authenticate using OAuth2.

        Args:
            session: Requests session.
        """
        oauth_config = self._config.vendor_options.get("oauth", {})
        token_url = oauth_config.get("token_url", f"{self._config.api_endpoint}/oauth/token")
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
        self._auth_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60  # Refresh 1 min early

        session.headers["Authorization"] = f"Bearer {self._auth_token}"

    def _test_connection(self, session: Any) -> None:
        """Test connection to Informatica API.

        Args:
            session: Requests session.

        Raises:
            ConnectionError: If connection test fails.
        """
        try:
            response = session.get(
                f"{self._config.api_endpoint}/health",
                timeout=self._config.connect_timeout_seconds,
            )
            response.raise_for_status()
        except Exception:
            # Try alternative health endpoint
            try:
                response = session.get(
                    f"{self._config.api_endpoint}/version",
                    timeout=self._config.connect_timeout_seconds,
                )
                response.raise_for_status()
            except Exception as e:
                from packages.enterprise.engines.base import ConnectionError
                raise ConnectionError(
                    f"Failed to connect to Informatica API: {e}",
                    engine_name="informatica",
                    cause=e,
                ) from e

    def _do_disconnect(self) -> None:
        """Close connection."""
        if self._session:
            self._session.close()
            self._session = None
            self._auth_token = None

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
            Response object.

        Raises:
            ConnectionError: If API call fails.
        """
        session = self.get_session()
        url = f"{self._config.api_endpoint}{endpoint}"

        # Set default timeout
        kwargs.setdefault("timeout", self._config.timeout_seconds)

        response = session.request(method, url, **kwargs)
        response.raise_for_status()

        return response.json()


# =============================================================================
# Informatica Adapter
# =============================================================================


class InformaticaAdapter(EnterpriseEngineAdapter):
    """Adapter for Informatica Data Quality.

    Provides integration with Informatica IDQ through its REST API.

    Features:
        - Data quality validation using IDQ rules
        - Data profiling using IDQ profiler
        - Scorecard-based quality measurement
        - Async job execution support
        - Connection pooling

    Example:
        >>> config = InformaticaConfig(
        ...     api_endpoint="https://idq.example.com/api/v2",
        ...     api_key="secret",
        ...     domain="Production",
        ...     project="DataQuality",
        ... )
        >>> engine = InformaticaAdapter(config=config)
        >>> with engine:
        ...     result = engine.check(df, rules)
        ...     if result.status == CheckStatus.FAILED:
        ...         for failure in result.failures:
        ...             print(f"{failure.column}: {failure.message}")
    """

    def __init__(
        self,
        config: InformaticaConfig | None = None,
    ) -> None:
        """Initialize Informatica adapter.

        Args:
            config: Informatica configuration.
        """
        super().__init__(config or DEFAULT_INFORMATICA_CONFIG)
        self._config: InformaticaConfig = self._config  # Type narrowing

    @property
    def engine_name(self) -> str:
        """Return engine name."""
        return "informatica"

    @property
    def engine_version(self) -> str:
        """Return engine version."""
        # Try to get from vendor library
        if self._vendor_library:
            return getattr(self._vendor_library, "__version__", "1.0.0")
        return "1.0.0"

    def _get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities."""
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=False,  # IDQ doesn't have auto-learn
            supports_async=True,
            supports_streaming=self._config.use_streaming,
            supported_data_types=("polars", "pandas", "csv", "parquet"),
            supported_rule_types=self._get_rule_translator().get_supported_rule_types(),
        )

    def _get_description(self) -> str:
        """Return engine description."""
        return "Informatica Data Quality - Enterprise data quality platform"

    def _get_homepage(self) -> str:
        """Return engine homepage."""
        return "https://www.informatica.com/products/data-quality.html"

    def _ensure_vendor_library(self) -> Any:
        """Ensure Informatica SDK is loaded.

        Returns:
            Informatica SDK module.

        Raises:
            VendorSDKError: If SDK cannot be loaded.
        """
        if self._vendor_library is not None:
            return self._vendor_library

        try:
            # Try to import the official SDK
            import informatica_dq_sdk as idq
            self._vendor_library = idq
        except ImportError:
            # Fall back to requests-based implementation
            try:
                import requests
                self._vendor_library = requests
            except ImportError as e:
                raise VendorSDKError(
                    "Neither informatica-dq-sdk nor requests is installed. "
                    "Install with: pip install informatica-dq-sdk or pip install requests",
                    engine_name="informatica",
                    cause=e,
                ) from e

        return self._vendor_library

    def _create_connection_manager(self) -> BaseConnectionManager:
        """Create Informatica connection manager.

        Returns:
            Connection manager instance.
        """
        return InformaticaConnectionManager(self._config)

    def _create_rule_translator(self) -> BaseRuleTranslator:
        """Create Informatica rule translator.

        Returns:
            Rule translator instance.
        """
        return InformaticaRuleTranslator()

    def _create_result_converter(self) -> BaseResultConverter:
        """Create Informatica result converter.

        Returns:
            Result converter instance.
        """
        return InformaticaResultConverter()

    def _execute_check(
        self,
        data: Any,
        translated_rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute check using Informatica API.

        Args:
            data: Converted data (Pandas DataFrame).
            translated_rules: Informatica-format rules.
            **kwargs: Additional parameters.

        Returns:
            Informatica result object.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, InformaticaConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        # Build request payload
        payload = self._build_check_payload(data, translated_rules, **kwargs)

        # Execute validation
        if self._config.async_execution:
            return self._execute_async_job(
                connection,
                "validation",
                payload,
            )
        else:
            return connection.execute_api_call(
                "POST",
                "/validation/execute",
                json=payload,
            )

    def _execute_profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute profiling using Informatica API.

        Args:
            data: Converted data (Pandas DataFrame).
            **kwargs: Additional parameters.

        Returns:
            Informatica result object.
        """
        connection = self._get_connection_manager()
        if not isinstance(connection, InformaticaConnectionManager):
            raise VendorSDKError("Invalid connection manager type")

        # Build request payload
        payload = self._build_profile_payload(data, **kwargs)

        # Execute profiling
        if self._config.async_execution:
            return self._execute_async_job(
                connection,
                "profile",
                payload,
            )
        else:
            return connection.execute_api_call(
                "POST",
                "/profile/execute",
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
            "domain": self._config.domain,
            "project": self._config.project,
            "folder": self._config.folder,
            "rules": rules,
        }

        # Add scorecard if specified
        scorecard = kwargs.get("scorecard", self._config.scorecard_name)
        if scorecard:
            payload["scorecard"] = scorecard

        # Handle data transfer based on mode
        if self._config.data_transfer_mode == DataTransferMode.INLINE:
            payload["data"] = self._serialize_data(data)
        elif self._config.data_transfer_mode == DataTransferMode.REFERENCE:
            payload["dataSource"] = kwargs.get("data_source", {})

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
            "domain": self._config.domain,
            "project": self._config.project,
            "folder": self._config.folder,
        }

        # Add profile name if specified
        profile = kwargs.get("profile", self._config.profile_name)
        if profile:
            payload["profileName"] = profile

        # Handle data transfer
        if self._config.data_transfer_mode == DataTransferMode.INLINE:
            payload["data"] = self._serialize_data(data)
        elif self._config.data_transfer_mode == DataTransferMode.REFERENCE:
            payload["dataSource"] = kwargs.get("data_source", {})

        # Profile options
        payload["options"] = {
            "computeStatistics": kwargs.get("compute_statistics", True),
            "sampleSize": kwargs.get("sample_size", 10000),
            "computePatterns": kwargs.get("compute_patterns", False),
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
        columns = list(data.columns)

        return {
            "columns": columns,
            "rows": records,
            "rowCount": len(records),
        }

    def _execute_async_job(
        self,
        connection: InformaticaConnectionManager,
        job_type: str,
        payload: dict[str, Any],
    ) -> Any:
        """Execute async job and wait for completion.

        Args:
            connection: Connection manager.
            job_type: Type of job (validation, profile).
            payload: Request payload.

        Returns:
            Job result.
        """
        # Submit job
        submit_response = connection.execute_api_call(
            "POST",
            f"/{job_type}/submit",
            json=payload,
        )

        job_id = submit_response.get("jobId")
        if not job_id:
            return submit_response

        # Poll for completion
        for _ in range(self._config.max_poll_attempts):
            time.sleep(self._config.poll_interval_seconds)

            status_response = connection.execute_api_call(
                "GET",
                f"/jobs/{job_id}/status",
            )

            status = status_response.get("status", "")
            if status in ("COMPLETED", "SUCCESS"):
                return connection.execute_api_call(
                    "GET",
                    f"/jobs/{job_id}/result",
                )
            elif status in ("FAILED", "ERROR"):
                error_msg = status_response.get("errorMessage", "Job failed")
                raise VendorSDKError(
                    f"Informatica job failed: {error_msg}",
                    engine_name="informatica",
                )

        raise VendorSDKError(
            f"Informatica job timed out after {self._config.max_poll_attempts} attempts",
            engine_name="informatica",
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_informatica_adapter(
    api_endpoint: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    **kwargs: Any,
) -> InformaticaAdapter:
    """Create an Informatica adapter with configuration.

    Factory function for convenient adapter creation.

    Args:
        api_endpoint: Informatica API endpoint URL.
        api_key: API key for authentication.
        username: Username for basic auth.
        password: Password for basic auth.
        **kwargs: Additional configuration options.

    Returns:
        Configured InformaticaAdapter instance.

    Example:
        >>> adapter = create_informatica_adapter(
        ...     api_endpoint="https://idq.example.com/api/v2",
        ...     api_key="secret",
        ...     domain="Production",
        ... )
    """
    config = InformaticaConfig(
        api_endpoint=api_endpoint,
        api_key=api_key,
        username=username,
        password=password,
        auth_type=AuthType.API_KEY if api_key else AuthType.BASIC,
        **kwargs,
    )
    return InformaticaAdapter(config=config)
