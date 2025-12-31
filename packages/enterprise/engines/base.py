"""Base abstractions for Enterprise Data Quality Engine Adapters.

This module provides base classes and protocols for enterprise data quality
engines like Informatica, Talend, SAP Data Services, and IBM InfoSphere.

Design Principles:
    1. Protocol-first: Extend common.engines protocols for enterprise features
    2. Vendor-agnostic: Abstract patterns that work across enterprise tools
    3. API-first: Most enterprise tools are accessed via REST/SOAP APIs
    4. Lazy loading: Vendor SDKs loaded only when needed
    5. Configuration-driven: Support complex enterprise configurations

Key Abstractions:
    - EnterpriseEngineConfig: Extended configuration for enterprise engines
    - EnterpriseEngineAdapter: Base adapter with common patterns
    - ConnectionManager: Connection pooling and management
    - RuleTranslator: Translate common rules to vendor-specific format

Example:
    >>> from packages.enterprise.engines import InformaticaAdapter
    >>> config = InformaticaConfig(
    ...     api_endpoint="https://idq.example.com/api",
    ...     api_key="secret",
    ... )
    >>> engine = InformaticaAdapter(config=config)
    >>> with engine:
    ...     result = engine.check(data, rules)
"""

from __future__ import annotations

import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

from common.base import CheckResult, CheckStatus, LearnResult, ProfileResult, Severity
from common.engines.base import EngineCapabilities, EngineInfoMixin
from common.engines.lifecycle import (
    DEFAULT_ENGINE_CONFIG,
    EngineConfig,
    EngineInitializationError,
    EngineShutdownError,
    EngineState,
    EngineStateTracker,
    ManagedEngineMixin,
)
from common.exceptions import TruthoundIntegrationError
from common.health import HealthCheckResult, HealthStatus


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class EnterpriseEngineError(TruthoundIntegrationError):
    """Base exception for enterprise engine errors."""

    def __init__(
        self,
        message: str,
        *,
        engine_name: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize enterprise engine error.

        Args:
            message: Human-readable error description.
            engine_name: Name of the engine.
            details: Additional error context.
            cause: Original exception.
        """
        details = details or {}
        if engine_name:
            details["engine_name"] = engine_name
        super().__init__(message, details=details, cause=cause)
        self.engine_name = engine_name


class ConnectionError(EnterpriseEngineError):
    """Error connecting to enterprise engine API."""

    pass


class AuthenticationError(EnterpriseEngineError):
    """Authentication failed with enterprise engine."""

    pass


class RateLimitError(EnterpriseEngineError):
    """Rate limit exceeded on enterprise engine API."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message.
            retry_after_seconds: Seconds to wait before retry.
            **kwargs: Additional error context.
        """
        super().__init__(message, **kwargs)
        self.retry_after_seconds = retry_after_seconds


class VendorSDKError(EnterpriseEngineError):
    """Error from vendor SDK."""

    pass


class RuleTranslationError(EnterpriseEngineError):
    """Error translating rules to vendor format."""

    def __init__(
        self,
        message: str,
        *,
        rule_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize rule translation error.

        Args:
            message: Error message.
            rule_type: The rule type that failed translation.
            **kwargs: Additional error context.
        """
        super().__init__(message, **kwargs)
        self.rule_type = rule_type


# =============================================================================
# Enums
# =============================================================================


class AuthType(Enum):
    """Authentication type for enterprise APIs."""

    NONE = auto()
    API_KEY = auto()
    BASIC = auto()
    OAUTH2 = auto()
    JWT = auto()
    CERTIFICATE = auto()


class ConnectionMode(Enum):
    """Connection mode for enterprise engines."""

    REST = auto()
    SOAP = auto()
    JDBC = auto()
    NATIVE = auto()


class DataTransferMode(Enum):
    """How data is transferred to the engine."""

    INLINE = auto()  # Data sent directly in API payload
    REFERENCE = auto()  # Data reference (path, URL) sent
    STREAMING = auto()  # Data streamed to engine


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class RuleTranslator(Protocol):
    """Protocol for translating common rules to vendor-specific format.

    Each enterprise engine has its own rule definition format. This protocol
    defines the interface for translating between common and vendor formats.
    """

    def translate(
        self,
        rule: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate a common rule to vendor format.

        Args:
            rule: Common rule dictionary with 'type', 'column', etc.

        Returns:
            Vendor-specific rule dictionary.

        Raises:
            RuleTranslationError: If rule cannot be translated.
        """
        ...

    def translate_batch(
        self,
        rules: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Translate multiple rules to vendor format.

        Args:
            rules: Sequence of common rule dictionaries.

        Returns:
            List of vendor-specific rule dictionaries.
        """
        ...

    def get_supported_rule_types(self) -> tuple[str, ...]:
        """Return supported common rule types.

        Returns:
            Tuple of supported rule type names.
        """
        ...


@runtime_checkable
class ResultConverter(Protocol):
    """Protocol for converting vendor results to common format.

    Each enterprise engine returns results in its own format. This protocol
    defines the interface for converting to common CheckResult, ProfileResult, etc.
    """

    def convert_check_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
        rules: Sequence[Mapping[str, Any]] | None = None,
    ) -> CheckResult:
        """Convert vendor check result to CheckResult.

        Args:
            vendor_result: Result from vendor API.
            start_time: Start time for duration calculation.
            rules: Original rules for context.

        Returns:
            Common CheckResult.
        """
        ...

    def convert_profile_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
    ) -> ProfileResult:
        """Convert vendor profile result to ProfileResult.

        Args:
            vendor_result: Result from vendor API.
            start_time: Start time for duration calculation.

        Returns:
            Common ProfileResult.
        """
        ...


@runtime_checkable
class ConnectionManager(Protocol):
    """Protocol for managing connections to enterprise engines.

    Handles connection pooling, authentication, and reconnection.
    """

    def connect(self) -> None:
        """Establish connection to the engine.

        Raises:
            ConnectionError: If connection fails.
            AuthenticationError: If authentication fails.
        """
        ...

    def disconnect(self) -> None:
        """Close connection to the engine."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to the engine.

        Returns:
            True if connected.
        """
        ...

    def health_check(self) -> HealthCheckResult:
        """Check connection health.

        Returns:
            HealthCheckResult with connection status.
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


ConfigT = TypeVar("ConfigT", bound="EnterpriseEngineConfig")


@dataclass(frozen=True, slots=True)
class EnterpriseEngineConfig(EngineConfig):
    """Base configuration for enterprise data quality engines.

    Extends EngineConfig with enterprise-specific settings like API endpoints,
    authentication, timeouts, and retry policies.

    Attributes:
        api_endpoint: Base URL for the engine API.
        api_key: API key for authentication.
        username: Username for basic auth.
        password: Password for basic auth.
        auth_type: Authentication type.
        connection_mode: Connection mode (REST, SOAP, etc.).
        timeout_seconds: Request timeout in seconds.
        connect_timeout_seconds: Connection timeout in seconds.
        max_retries: Maximum API retry attempts.
        retry_delay_seconds: Base delay between retries.
        verify_ssl: Whether to verify SSL certificates.
        proxy_url: Proxy URL if needed.
        pool_size: Connection pool size.
        pool_timeout_seconds: Pool acquisition timeout.
        data_transfer_mode: How data is sent to engine.
        max_payload_size_mb: Maximum payload size in MB.
        batch_size: Default batch size for operations.

    Example:
        >>> config = EnterpriseEngineConfig(
        ...     api_endpoint="https://api.example.com",
        ...     api_key="secret-key",
        ...     auth_type=AuthType.API_KEY,
        ...     timeout_seconds=60.0,
        ... )
    """

    # API Configuration
    api_endpoint: str = ""
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    auth_type: AuthType = AuthType.NONE
    connection_mode: ConnectionMode = ConnectionMode.REST

    # Timeout Configuration
    timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0

    # Retry Configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # SSL/Proxy Configuration
    verify_ssl: bool = True
    proxy_url: str | None = None

    # Connection Pool Configuration
    pool_size: int = 5
    pool_timeout_seconds: float = 30.0

    # Data Transfer Configuration
    data_transfer_mode: DataTransferMode = DataTransferMode.INLINE
    max_payload_size_mb: float = 100.0
    batch_size: int = 10000

    # Additional vendor-specific options
    vendor_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self._validate_base_config()
        self._validate_enterprise_config()

    def _validate_enterprise_config(self) -> None:
        """Validate enterprise-specific configuration values."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if self.pool_size < 1:
            raise ValueError("pool_size must be at least 1")
        if self.max_payload_size_mb <= 0:
            raise ValueError("max_payload_size_mb must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    # Builder methods
    def with_api_endpoint(self, endpoint: str) -> Self:
        """Create config with API endpoint.

        Args:
            endpoint: API endpoint URL.

        Returns:
            New configuration with endpoint.
        """
        return self._copy_with(api_endpoint=endpoint)

    def with_api_key(self, key: str) -> Self:
        """Create config with API key authentication.

        Args:
            key: API key.

        Returns:
            New configuration with API key auth.
        """
        return self._copy_with(api_key=key, auth_type=AuthType.API_KEY)

    def with_basic_auth(self, username: str, password: str) -> Self:
        """Create config with basic authentication.

        Args:
            username: Username.
            password: Password.

        Returns:
            New configuration with basic auth.
        """
        return self._copy_with(
            username=username,
            password=password,
            auth_type=AuthType.BASIC,
        )

    def with_oauth2(self, **oauth_config: Any) -> Self:
        """Create config with OAuth2 authentication.

        Args:
            **oauth_config: OAuth2 configuration options.

        Returns:
            New configuration with OAuth2 auth.
        """
        return self._copy_with(
            auth_type=AuthType.OAUTH2,
            vendor_options={**self.vendor_options, "oauth": oauth_config},
        )

    def with_timeout(
        self,
        timeout_seconds: float,
        connect_timeout_seconds: float | None = None,
    ) -> Self:
        """Create config with timeout settings.

        Args:
            timeout_seconds: Request timeout.
            connect_timeout_seconds: Connection timeout.

        Returns:
            New configuration with timeouts.
        """
        updates: dict[str, Any] = {"timeout_seconds": timeout_seconds}
        if connect_timeout_seconds is not None:
            updates["connect_timeout_seconds"] = connect_timeout_seconds
        return self._copy_with(**updates)

    def with_retry(
        self,
        max_retries: int,
        delay_seconds: float | None = None,
    ) -> Self:
        """Create config with retry settings.

        Args:
            max_retries: Maximum retry attempts.
            delay_seconds: Base delay between retries.

        Returns:
            New configuration with retry settings.
        """
        updates: dict[str, Any] = {"max_retries": max_retries}
        if delay_seconds is not None:
            updates["retry_delay_seconds"] = delay_seconds
        return self._copy_with(**updates)

    def with_ssl(self, verify: bool = True, proxy_url: str | None = None) -> Self:
        """Create config with SSL/proxy settings.

        Args:
            verify: Whether to verify SSL certificates.
            proxy_url: Proxy URL if needed.

        Returns:
            New configuration with SSL settings.
        """
        return self._copy_with(verify_ssl=verify, proxy_url=proxy_url)

    def with_pool(
        self,
        size: int,
        timeout_seconds: float | None = None,
    ) -> Self:
        """Create config with connection pool settings.

        Args:
            size: Connection pool size.
            timeout_seconds: Pool acquisition timeout.

        Returns:
            New configuration with pool settings.
        """
        updates: dict[str, Any] = {"pool_size": size}
        if timeout_seconds is not None:
            updates["pool_timeout_seconds"] = timeout_seconds
        return self._copy_with(**updates)

    def with_data_transfer(
        self,
        mode: DataTransferMode,
        max_payload_mb: float | None = None,
    ) -> Self:
        """Create config with data transfer settings.

        Args:
            mode: Data transfer mode.
            max_payload_mb: Maximum payload size in MB.

        Returns:
            New configuration with transfer settings.
        """
        updates: dict[str, Any] = {"data_transfer_mode": mode}
        if max_payload_mb is not None:
            updates["max_payload_size_mb"] = max_payload_mb
        return self._copy_with(**updates)

    def with_vendor_options(self, **options: Any) -> Self:
        """Create config with vendor-specific options.

        Args:
            **options: Vendor-specific key-value options.

        Returns:
            New configuration with vendor options.
        """
        return self._copy_with(
            vendor_options={**self.vendor_options, **options}
        )


# Preset configurations
DEFAULT_ENTERPRISE_CONFIG = EnterpriseEngineConfig()

PRODUCTION_ENTERPRISE_CONFIG = EnterpriseEngineConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    health_check_interval_seconds=60.0,
    timeout_seconds=60.0,
    connect_timeout_seconds=15.0,
    max_retries=5,
    retry_delay_seconds=2.0,
    verify_ssl=True,
    pool_size=10,
)

DEVELOPMENT_ENTERPRISE_CONFIG = EnterpriseEngineConfig(
    auto_start=False,
    auto_stop=True,
    health_check_enabled=False,
    timeout_seconds=30.0,
    connect_timeout_seconds=5.0,
    max_retries=1,
    retry_delay_seconds=0.5,
    verify_ssl=False,  # Allow self-signed certs in dev
)

HIGH_THROUGHPUT_CONFIG = EnterpriseEngineConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    timeout_seconds=120.0,
    connect_timeout_seconds=10.0,
    max_retries=3,
    pool_size=20,
    batch_size=50000,
    max_payload_size_mb=500.0,
)


# =============================================================================
# Base Rule Translator
# =============================================================================


class BaseRuleTranslator:
    """Base implementation of RuleTranslator with common patterns.

    Subclasses should override _get_rule_mapping() and _translate_rule_params()
    to provide vendor-specific translation.
    """

    def __init__(self) -> None:
        """Initialize rule translator."""
        self._rule_mapping: dict[str, str] = {}
        self._custom_translators: dict[str, Any] = {}

    def _get_rule_mapping(self) -> dict[str, str]:
        """Return mapping from common rule types to vendor rule types.

        Override in subclass.

        Returns:
            Dictionary mapping common types to vendor types.
        """
        return {}

    def _translate_rule_params(
        self,
        rule_type: str,
        rule: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate rule parameters to vendor format.

        Override in subclass for vendor-specific parameter translation.

        Args:
            rule_type: Vendor rule type.
            rule: Original rule dictionary.

        Returns:
            Vendor-specific parameter dictionary.
        """
        params = dict(rule)
        params.pop("type", None)
        return params

    def register_custom_translator(
        self,
        rule_type: str,
        translator: Any,
    ) -> None:
        """Register a custom translator for a rule type.

        Args:
            rule_type: Common rule type.
            translator: Callable that translates the rule.
        """
        self._custom_translators[rule_type] = translator

    def translate(
        self,
        rule: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate a common rule to vendor format.

        Args:
            rule: Common rule dictionary with 'type', 'column', etc.

        Returns:
            Vendor-specific rule dictionary.

        Raises:
            RuleTranslationError: If rule cannot be translated.
        """
        rule_type = rule.get("type", "")

        # Check for custom translator
        if rule_type in self._custom_translators:
            return self._custom_translators[rule_type](rule)

        # Get rule mapping
        mapping = self._get_rule_mapping()
        if rule_type not in mapping:
            raise RuleTranslationError(
                f"Unsupported rule type: {rule_type}",
                rule_type=rule_type,
            )

        vendor_type = mapping[rule_type]
        params = self._translate_rule_params(vendor_type, rule)

        return {
            "type": vendor_type,
            **params,
        }

    def translate_batch(
        self,
        rules: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Translate multiple rules to vendor format.

        Args:
            rules: Sequence of common rule dictionaries.

        Returns:
            List of vendor-specific rule dictionaries.
        """
        return [self.translate(rule) for rule in rules]

    def get_supported_rule_types(self) -> tuple[str, ...]:
        """Return supported common rule types.

        Returns:
            Tuple of supported rule type names.
        """
        mapping = self._get_rule_mapping()
        custom = set(self._custom_translators.keys())
        return tuple(set(mapping.keys()) | custom)


# =============================================================================
# Base Result Converter
# =============================================================================


class BaseResultConverter:
    """Base implementation of ResultConverter with common patterns.

    Subclasses should override _extract_check_items() and _map_severity()
    to provide vendor-specific result conversion.
    """

    # Severity mapping from vendor to common
    # Maps vendor severity strings to common Severity enum values
    # Note: Severity enum has CRITICAL, ERROR, WARNING, INFO (not HIGH/MEDIUM/LOW)
    SEVERITY_MAPPING: dict[str, Severity] = {
        "critical": Severity.CRITICAL,
        "high": Severity.ERROR,
        "error": Severity.ERROR,
        "medium": Severity.WARNING,
        "warning": Severity.WARNING,
        "low": Severity.INFO,
        "info": Severity.INFO,
    }

    def _map_severity(self, vendor_severity: str) -> Severity:
        """Map vendor severity to common Severity.

        Args:
            vendor_severity: Vendor severity string.

        Returns:
            Common Severity enum value.
        """
        return self.SEVERITY_MAPPING.get(
            vendor_severity.lower(),
            Severity.WARNING,
        )

    def _extract_check_items(
        self,
        vendor_result: Any,
    ) -> list[dict[str, Any]]:
        """Extract check items from vendor result.

        Override in subclass to parse vendor-specific result format.

        Args:
            vendor_result: Result from vendor API.

        Returns:
            List of normalized check items with fields:
                - rule_name: str
                - column: str | None
                - passed: bool
                - failed_count: int
                - message: str
                - severity: str
        """
        return []

    def _determine_status(
        self,
        items: list[dict[str, Any]],
    ) -> CheckStatus:
        """Determine overall check status from items.

        Args:
            items: List of check items.

        Returns:
            Overall CheckStatus.
        """
        if not items:
            return CheckStatus.PASSED

        has_critical = any(
            not item["passed"] and item.get("severity", "").lower() in ("critical", "high")
            for item in items
        )
        has_failures = any(not item["passed"] for item in items)
        has_warnings = any(
            item.get("severity", "").lower() in ("low", "info")
            for item in items
            if not item["passed"]
        )

        if has_critical:
            return CheckStatus.FAILED
        if has_failures and not has_warnings:
            return CheckStatus.FAILED
        if has_warnings:
            return CheckStatus.WARNING
        return CheckStatus.PASSED

    def convert_check_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
        rules: Sequence[Mapping[str, Any]] | None = None,
    ) -> CheckResult:
        """Convert vendor check result to CheckResult.

        Args:
            vendor_result: Result from vendor API.
            start_time: Start time for duration calculation.
            rules: Original rules for context.

        Returns:
            Common CheckResult.
        """
        from common.base import ValidationFailure

        items = self._extract_check_items(vendor_result)
        status = self._determine_status(items)

        failures: list[ValidationFailure] = []
        passed_count = 0
        failed_count = 0

        for item in items:
            if item["passed"]:
                passed_count += 1
            else:
                failed_count += 1
                failures.append(
                    ValidationFailure(
                        rule_name=item["rule_name"],
                        column=item.get("column"),
                        message=item.get("message", "Validation failed"),
                        severity=self._map_severity(item.get("severity", "medium")),
                        failed_count=item.get("failed_count", 1),
                        sample_values=tuple(item.get("sample_values", ())),
                        metadata=item.get("details", {}),
                    )
                )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return CheckResult(
            status=status,
            passed_count=passed_count,
            failed_count=failed_count,
            failures=tuple(failures),
            execution_time_ms=execution_time_ms,
            metadata={
                "engine": self.__class__.__name__.replace("ResultConverter", ""),
                "rule_count": len(rules) if rules else len(items),
            },
        )

    def convert_profile_result(
        self,
        vendor_result: Any,
        *,
        start_time: float,
    ) -> ProfileResult:
        """Convert vendor profile result to ProfileResult.

        Override in subclass for vendor-specific profile conversion.

        Args:
            vendor_result: Result from vendor API.
            start_time: Start time for duration calculation.

        Returns:
            Common ProfileResult.
        """
        from common.base import ColumnProfile

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return ProfileResult(
            columns=(),
            row_count=0,
            execution_time_ms=execution_time_ms,
            metadata={"engine": self.__class__.__name__},
        )


# =============================================================================
# Base Connection Manager
# =============================================================================


class BaseConnectionManager:
    """Base implementation of ConnectionManager.

    Provides common connection management patterns for enterprise APIs.
    Subclasses should override _do_connect(), _do_disconnect(), and
    _do_health_check() for vendor-specific behavior.
    """

    def __init__(
        self,
        config: EnterpriseEngineConfig,
        *,
        name: str = "connection",
    ) -> None:
        """Initialize connection manager.

        Args:
            config: Enterprise engine configuration.
            name: Connection name for logging.
        """
        self._config = config
        self._name = name
        self._connected = False
        self._connection: Any = None
        self._lock = threading.RLock()
        self._last_health_check: float = 0
        self._consecutive_failures = 0

    @property
    def name(self) -> str:
        """Return connection name."""
        return self._name

    def _do_connect(self) -> Any:
        """Establish connection. Override in subclass.

        Returns:
            Connection object.

        Raises:
            ConnectionError: If connection fails.
        """
        raise NotImplementedError

    def _do_disconnect(self) -> None:
        """Close connection. Override in subclass."""
        pass

    def _do_health_check(self) -> bool:
        """Check connection health. Override in subclass.

        Returns:
            True if healthy.
        """
        return self._connected

    def connect(self) -> None:
        """Establish connection to the engine.

        Raises:
            ConnectionError: If connection fails.
            AuthenticationError: If authentication fails.
        """
        with self._lock:
            if self._connected:
                return

            try:
                self._connection = self._do_connect()
                self._connected = True
                self._consecutive_failures = 0
            except Exception as e:
                self._connected = False
                self._consecutive_failures += 1
                raise ConnectionError(
                    f"Failed to connect to {self._name}: {e}",
                    cause=e,
                ) from e

    def disconnect(self) -> None:
        """Close connection to the engine."""
        with self._lock:
            if not self._connected:
                return

            try:
                self._do_disconnect()
            finally:
                self._connected = False
                self._connection = None

    def is_connected(self) -> bool:
        """Check if connected to the engine.

        Returns:
            True if connected.
        """
        with self._lock:
            return self._connected

    def get_connection(self) -> Any:
        """Get the connection object.

        Returns:
            Connection object.

        Raises:
            ConnectionError: If not connected.
        """
        with self._lock:
            if not self._connected:
                raise ConnectionError(f"Not connected to {self._name}")
            return self._connection

    def health_check(self) -> HealthCheckResult:
        """Check connection health.

        Returns:
            HealthCheckResult with connection status.
        """
        start_time = time.perf_counter()

        if not self._connected:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message="Not connected",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            is_healthy = self._do_health_check()
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._last_health_check = time.time()

            if is_healthy:
                return HealthCheckResult(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                    message="Connection is healthy",
                    duration_ms=duration_ms,
                )
            else:
                return HealthCheckResult(
                    name=self._name,
                    status=HealthStatus.DEGRADED,
                    message="Connection degraded",
                    duration_ms=duration_ms,
                )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                duration_ms=duration_ms,
                details={"error": str(e)},
            )

    def __enter__(self) -> Self:
        """Enter context manager."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        self.disconnect()


# =============================================================================
# Base Enterprise Engine Adapter
# =============================================================================


class EnterpriseEngineAdapter(EngineInfoMixin):
    """Base adapter for enterprise data quality engines.

    Provides common patterns for enterprise engine integration:
    - Lazy vendor library loading
    - Connection management
    - Rule translation
    - Result conversion
    - Retry logic
    - Health checking

    Subclasses must implement:
    - engine_name property
    - engine_version property
    - _get_capabilities()
    - _create_connection_manager()
    - _create_rule_translator()
    - _create_result_converter()
    - _execute_check()
    - _execute_profile()
    - _execute_learn() (optional)

    Example:
        >>> class MyEnterpriseAdapter(EnterpriseEngineAdapter):
        ...     @property
        ...     def engine_name(self) -> str:
        ...         return "my_enterprise"
        ...
        ...     def _execute_check(self, data, translated_rules, **kwargs):
        ...         # Call vendor API
        ...         return vendor_result
    """

    def __init__(
        self,
        config: EnterpriseEngineConfig | None = None,
    ) -> None:
        """Initialize enterprise engine adapter.

        Args:
            config: Enterprise engine configuration.
        """
        self._config = config or DEFAULT_ENTERPRISE_CONFIG
        self._vendor_library: Any = None
        self._connection_manager: BaseConnectionManager | None = None
        self._rule_translator: BaseRuleTranslator | None = None
        self._result_converter: BaseResultConverter | None = None
        self._state_tracker = EngineStateTracker(self.engine_name)
        self._lifecycle_lock = threading.RLock()
        self._lifecycle_config = self._config

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the unique name of this engine."""
        ...

    @property
    @abstractmethod
    def engine_version(self) -> str:
        """Return the version of this engine."""
        ...

    @property
    def config(self) -> EnterpriseEngineConfig:
        """Return the configuration."""
        return self._config

    def _get_capabilities(self) -> EngineCapabilities:
        """Return engine capabilities. Override in subclass."""
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=False,
            supports_async=False,
            supports_streaming=False,
            supported_data_types=("polars", "pandas"),
            supported_rule_types=self._get_rule_translator().get_supported_rule_types(),
        )

    @abstractmethod
    def _ensure_vendor_library(self) -> Any:
        """Ensure vendor library is loaded.

        Implement lazy import of vendor SDK.

        Returns:
            Vendor library module.

        Raises:
            VendorSDKError: If library cannot be loaded.
        """
        ...

    def _create_connection_manager(self) -> BaseConnectionManager:
        """Create connection manager. Override in subclass.

        Returns:
            Connection manager instance.
        """
        raise NotImplementedError

    def _create_rule_translator(self) -> BaseRuleTranslator:
        """Create rule translator. Override in subclass.

        Returns:
            Rule translator instance.
        """
        raise NotImplementedError

    def _create_result_converter(self) -> BaseResultConverter:
        """Create result converter. Override in subclass.

        Returns:
            Result converter instance.
        """
        raise NotImplementedError

    def _get_connection_manager(self) -> BaseConnectionManager:
        """Get or create connection manager.

        Returns:
            Connection manager instance.
        """
        if self._connection_manager is None:
            self._connection_manager = self._create_connection_manager()
        return self._connection_manager

    def _get_rule_translator(self) -> BaseRuleTranslator:
        """Get or create rule translator.

        Returns:
            Rule translator instance.
        """
        if self._rule_translator is None:
            self._rule_translator = self._create_rule_translator()
        return self._rule_translator

    def _get_result_converter(self) -> BaseResultConverter:
        """Get or create result converter.

        Returns:
            Result converter instance.
        """
        if self._result_converter is None:
            self._result_converter = self._create_result_converter()
        return self._result_converter

    def _convert_data(self, data: Any) -> Any:
        """Convert data to vendor-compatible format.

        Default implementation handles Polars -> Pandas conversion.
        Override for vendor-specific needs.

        Args:
            data: Input data.

        Returns:
            Vendor-compatible data.
        """
        # Convert Polars to Pandas if needed
        if hasattr(data, "to_pandas"):
            return data.to_pandas()
        return data

    @abstractmethod
    def _execute_check(
        self,
        data: Any,
        translated_rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Execute check using vendor API.

        Args:
            data: Converted data.
            translated_rules: Vendor-format rules.
            **kwargs: Additional parameters.

        Returns:
            Vendor result object.
        """
        ...

    @abstractmethod
    def _execute_profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute profiling using vendor API.

        Args:
            data: Converted data.
            **kwargs: Additional parameters.

        Returns:
            Vendor result object.
        """
        ...

    def _execute_learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute rule learning using vendor API.

        Override in subclass if supported.

        Args:
            data: Converted data.
            **kwargs: Additional parameters.

        Returns:
            Vendor result object.
        """
        raise NotImplementedError(
            f"{self.engine_name} does not support rule learning"
        )

    def _do_start(self) -> None:
        """Start the engine."""
        self._ensure_vendor_library()
        self._get_connection_manager().connect()

    def _do_stop(self) -> None:
        """Stop the engine."""
        if self._connection_manager:
            self._connection_manager.disconnect()

    def _do_health_check(self) -> HealthCheckResult:
        """Perform health check."""
        if self._connection_manager:
            return self._connection_manager.health_check()
        return HealthCheckResult.healthy(
            self.engine_name,
            message="Engine is available",
        )

    def start(self) -> None:
        """Start the engine."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state.is_terminal:
                from common.engines.lifecycle import EngineStoppedError
                raise EngineStoppedError(self.engine_name)
            if state.is_active:
                from common.engines.lifecycle import EngineAlreadyStartedError
                raise EngineAlreadyStartedError(self.engine_name)

            self._state_tracker.transition_to(EngineState.STARTING)
            try:
                self._do_start()
                self._state_tracker.transition_to(EngineState.RUNNING)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineInitializationError(
                    f"Failed to start engine: {e}",
                    engine_name=self.engine_name,
                    cause=e,
                ) from e

    def stop(self) -> None:
        """Stop the engine."""
        with self._lifecycle_lock:
            state = self._state_tracker.state
            if state == EngineState.STOPPED:
                return
            if not state.can_stop:
                return

            self._state_tracker.transition_to(EngineState.STOPPING)
            try:
                self._do_stop()
                self._state_tracker.transition_to(EngineState.STOPPED)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineShutdownError(
                    f"Failed to stop engine: {e}",
                    engine_name=self.engine_name,
                    cause=e,
                ) from e

    def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            return HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
            )

        result = self._do_health_check()
        self._state_tracker.record_health_check(result.status)
        return result

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self._state_tracker.state

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation checks on the data.

        Args:
            data: Data to validate.
            rules: Validation rules in common format.
            **kwargs: Engine-specific parameters.

        Returns:
            CheckResult with validation outcomes.
        """
        start_time = time.perf_counter()

        # Ensure engine is ready
        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            # Auto-start if configured
            if self._config.auto_start and state == EngineState.CREATED:
                self.start()
            else:
                from common.engines.lifecycle import EngineNotStartedError
                raise EngineNotStartedError(self.engine_name)

        # Convert data
        converted_data = self._convert_data(data)

        # Translate rules
        translator = self._get_rule_translator()
        translated_rules = translator.translate_batch(rules)

        # Execute check
        vendor_result = self._execute_check(converted_data, translated_rules, **kwargs)

        # Convert result
        converter = self._get_result_converter()
        return converter.convert_check_result(
            vendor_result,
            start_time=start_time,
            rules=rules,
        )

    def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Profile the data.

        Args:
            data: Data to profile.
            **kwargs: Engine-specific parameters.

        Returns:
            ProfileResult with profiling outcomes.
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

        # Execute profile
        vendor_result = self._execute_profile(converted_data, **kwargs)

        # Convert result
        converter = self._get_result_converter()
        return converter.convert_profile_result(
            vendor_result,
            start_time=start_time,
        )

    def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Learn validation rules from the data.

        Args:
            data: Data to learn from.
            **kwargs: Engine-specific parameters.

        Returns:
            LearnResult with learned rules.

        Raises:
            NotImplementedError: If engine doesn't support learning.
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

        # Check if learning is supported
        caps = self.get_capabilities()
        if not caps.supports_learn:
            raise NotImplementedError(
                f"{self.engine_name} does not support rule learning"
            )

        # Convert data
        converted_data = self._convert_data(data)

        # Execute learn
        vendor_result = self._execute_learn(converted_data, **kwargs)

        # Convert result
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return LearnResult(
            rules=(),
            execution_time_ms=execution_time_ms,
            metadata={"engine": self.engine_name},
        )

    def __enter__(self) -> Self:
        """Enter context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if self._config.auto_stop:
            self.stop()
