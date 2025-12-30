"""Prometheus metric exporter for Truthound Integrations.

This module provides a comprehensive Prometheus integration with:
- Text format serialization (OpenMetrics compatible)
- Push Gateway client for batch jobs
- HTTP endpoint server for scraping
- Configurable label handling and metric naming
- Multi-tenant support with tenant-aware metric isolation
- Authentication support for Push Gateway and HTTP Server
- Async support for high-throughput environments
- Distributed environment features (instance auto-detection, cluster labels)

Design Principles:
    1. Protocol-based: Implements MetricExporter protocol
    2. Immutable Config: Thread-safe configuration using frozen dataclass
    3. Extensible: Strategy pattern for formatting and transport
    4. Observable: Hook system for export events
    5. Production-ready: Retry, timeout, and error handling
    6. Enterprise-ready: Multi-tenant, auth, HA patterns

Example:
    >>> from common.exporters import PrometheusExporter, PrometheusConfig
    >>> config = PrometheusConfig(job_name="data_quality")
    >>> exporter = PrometheusExporter(config)
    >>> # Use with MetricsRegistry
    >>> registry.export()  # Exports to Prometheus format

    >>> # Push Gateway usage
    >>> from common.exporters import create_pushgateway_exporter
    >>> exporter = create_pushgateway_exporter(
    ...     gateway_url="http://pushgateway:9091",
    ...     job_name="batch_job",
    ... )
    >>> registry.export()

    >>> # HTTP Server for scraping
    >>> from common.exporters import create_prometheus_http_server
    >>> server = create_prometheus_http_server(port=9090, registry=registry)
    >>> server.start()  # Non-blocking

    >>> # Multi-tenant usage
    >>> from common.exporters import TenantAwarePrometheusExporter
    >>> exporter = TenantAwarePrometheusExporter(config)
    >>> exporter.export(metrics)  # Automatically adds tenant_id label

    >>> # With authentication
    >>> config = PrometheusConfig().with_auth(
    ...     username="prometheus",
    ...     password="secret",
    ... )
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import http.server
import os
import platform
import re
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

from common.exceptions import TruthoundIntegrationError
from common.metrics import MetricData, MetricType


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from common.metrics import MetricsRegistry


# =============================================================================
# Exceptions
# =============================================================================


class PrometheusExporterError(TruthoundIntegrationError):
    """Base exception for Prometheus exporter errors.

    Attributes:
        exporter_type: Type of exporter that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        exporter_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize Prometheus exporter error.

        Args:
            message: Human-readable error description.
            exporter_type: Type of exporter that failed.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if exporter_type:
            details["exporter_type"] = exporter_type
        super().__init__(message, details=details, cause=cause)
        self.exporter_type = exporter_type


class PushGatewayError(PrometheusExporterError):
    """Error when pushing metrics to Push Gateway.

    Attributes:
        gateway_url: URL of the Push Gateway.
        status_code: HTTP status code (if available).
    """

    def __init__(
        self,
        message: str,
        *,
        gateway_url: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize Push Gateway error.

        Args:
            message: Human-readable error description.
            gateway_url: URL of the Push Gateway.
            status_code: HTTP status code (if available).
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if gateway_url:
            details["gateway_url"] = gateway_url
        if status_code is not None:
            details["status_code"] = status_code
        super().__init__(
            message,
            exporter_type="pushgateway",
            details=details,
            cause=cause,
        )
        self.gateway_url = gateway_url
        self.status_code = status_code


class HttpServerError(PrometheusExporterError):
    """Error when running HTTP server for scraping.

    Attributes:
        host: Server host.
        port: Server port.
    """

    def __init__(
        self,
        message: str,
        *,
        host: str | None = None,
        port: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize HTTP server error.

        Args:
            message: Human-readable error description.
            host: Server host.
            port: Server port.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if host:
            details["host"] = host
        if port is not None:
            details["port"] = port
        super().__init__(
            message,
            exporter_type="http_server",
            details=details,
            cause=cause,
        )
        self.host = host
        self.port = port


# =============================================================================
# Enums
# =============================================================================


class ExportMode(Enum):
    """Export mode for Prometheus metrics.

    Attributes:
        PULL: Expose metrics via HTTP endpoint for scraping.
        PUSH: Push metrics to Push Gateway.
        BOTH: Both pull and push modes.
    """

    PULL = auto()
    PUSH = auto()
    BOTH = auto()


class MetricNamingStrategy(Enum):
    """Strategy for naming metrics in Prometheus format.

    Attributes:
        SNAKE_CASE: Convert to snake_case (default).
        PRESERVE: Keep original name.
        PREFIX_ONLY: Only add prefix, preserve rest.
    """

    SNAKE_CASE = auto()
    PRESERVE = auto()
    PREFIX_ONLY = auto()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class MetricFormatter(Protocol):
    """Protocol for metric formatters.

    Implement this to customize metric serialization.
    """

    @abstractmethod
    def format(self, metrics: Sequence[MetricData]) -> str:
        """Format metrics for export.

        Args:
            metrics: Metrics to format.

        Returns:
            Formatted string representation.
        """
        ...

    @abstractmethod
    def content_type(self) -> str:
        """Get content type for the formatted output.

        Returns:
            MIME content type string.
        """
        ...


@runtime_checkable
class MetricTransport(Protocol):
    """Protocol for metric transport mechanisms.

    Implement this to customize how metrics are sent.
    """

    @abstractmethod
    def send(self, data: str, content_type: str) -> None:
        """Send formatted metrics.

        Args:
            data: Formatted metric data.
            content_type: MIME content type.

        Raises:
            PrometheusExporterError: If sending fails.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the transport."""
        ...


@runtime_checkable
class ExportHook(Protocol):
    """Protocol for export event hooks.

    Implement this to receive notifications about export events.
    """

    @abstractmethod
    def on_export_start(
        self,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Called when export starts.

        Args:
            metric_count: Number of metrics being exported.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_export_success(
        self,
        metric_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when export succeeds.

        Args:
            metric_count: Number of metrics exported.
            duration_ms: Export duration in milliseconds.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_export_error(
        self,
        error: Exception,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Called when export fails.

        Args:
            error: The exception that occurred.
            metric_count: Number of metrics attempted.
            context: Additional context.
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class PrometheusConfig:
    """Configuration for Prometheus exporter.

    Immutable configuration object for Prometheus operations.
    Use builder methods to create modified copies.

    Attributes:
        enabled: Whether export is enabled.
        job_name: Job name for Push Gateway grouping.
        instance: Instance label value (auto-detected if empty).
        namespace: Metric namespace prefix.
        subsystem: Metric subsystem prefix.
        const_labels: Constant labels added to all metrics.
        naming_strategy: Strategy for metric naming.
        include_timestamp: Whether to include timestamps in output.
        include_help: Whether to include HELP comments.
        include_type: Whether to include TYPE comments.
        gateway_url: Push Gateway URL (for push mode).
        gateway_timeout_seconds: Timeout for Push Gateway requests.
        gateway_retry_count: Number of retries for Push Gateway.
        gateway_retry_delay_seconds: Delay between retries.
        http_host: HTTP server host (for pull mode).
        http_port: HTTP server port (for pull mode).
        http_path: HTTP endpoint path (default: /metrics).
        auth_username: Username for Basic Auth (Push Gateway).
        auth_password: Password for Basic Auth (Push Gateway).
        auth_token: Bearer token for authentication.
        auto_instance: Whether to auto-detect instance name.
        cluster_name: Cluster name for distributed deployments.
        enable_tenant_isolation: Whether to enable multi-tenant metric isolation.
        default_tenant_label: Label name for tenant ID (default: tenant_id).
        batch_size: Number of metrics to batch before pushing.
        async_push: Whether to use async push for better throughput.

    Example:
        >>> config = PrometheusConfig(
        ...     job_name="data_quality",
        ...     namespace="truthound",
        ...     const_labels={"env": "production"},
        ... )
        >>> push_config = config.with_gateway(
        ...     url="http://pushgateway:9091",
        ...     timeout_seconds=10.0,
        ... )
        >>> # With authentication
        >>> auth_config = config.with_auth(
        ...     username="prometheus",
        ...     password="secret",
        ... )
    """

    enabled: bool = True
    job_name: str = "truthound"
    instance: str = ""
    namespace: str = ""
    subsystem: str = ""
    const_labels: dict[str, str] = field(default_factory=dict)
    naming_strategy: MetricNamingStrategy = MetricNamingStrategy.SNAKE_CASE
    include_timestamp: bool = False
    include_help: bool = True
    include_type: bool = True
    gateway_url: str = ""
    gateway_timeout_seconds: float = 10.0
    gateway_retry_count: int = 3
    gateway_retry_delay_seconds: float = 1.0
    http_host: str = "0.0.0.0"
    http_port: int = 9090
    http_path: str = "/metrics"
    # Authentication
    auth_username: str = ""
    auth_password: str = ""
    auth_token: str = ""
    # Distributed/Enterprise
    auto_instance: bool = True
    cluster_name: str = ""
    enable_tenant_isolation: bool = False
    default_tenant_label: str = "tenant_id"
    # Performance
    batch_size: int = 0  # 0 = no batching
    async_push: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.gateway_timeout_seconds < 0:
            raise ValueError("gateway_timeout_seconds must be non-negative")
        if self.gateway_retry_count < 0:
            raise ValueError("gateway_retry_count must be non-negative")
        if self.gateway_retry_delay_seconds < 0:
            raise ValueError("gateway_retry_delay_seconds must be non-negative")
        if self.http_port < 0 or self.http_port > 65535:
            raise ValueError("http_port must be between 0 and 65535")
        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative")

    def _copy_with(self, **overrides: Any) -> PrometheusConfig:
        """Create a copy with specified field overrides.

        Internal helper to ensure all fields are preserved.
        """
        return PrometheusConfig(
            enabled=overrides.get("enabled", self.enabled),
            job_name=overrides.get("job_name", self.job_name),
            instance=overrides.get("instance", self.instance),
            namespace=overrides.get("namespace", self.namespace),
            subsystem=overrides.get("subsystem", self.subsystem),
            const_labels=overrides.get("const_labels", self.const_labels),
            naming_strategy=overrides.get("naming_strategy", self.naming_strategy),
            include_timestamp=overrides.get("include_timestamp", self.include_timestamp),
            include_help=overrides.get("include_help", self.include_help),
            include_type=overrides.get("include_type", self.include_type),
            gateway_url=overrides.get("gateway_url", self.gateway_url),
            gateway_timeout_seconds=overrides.get(
                "gateway_timeout_seconds", self.gateway_timeout_seconds
            ),
            gateway_retry_count=overrides.get(
                "gateway_retry_count", self.gateway_retry_count
            ),
            gateway_retry_delay_seconds=overrides.get(
                "gateway_retry_delay_seconds", self.gateway_retry_delay_seconds
            ),
            http_host=overrides.get("http_host", self.http_host),
            http_port=overrides.get("http_port", self.http_port),
            http_path=overrides.get("http_path", self.http_path),
            auth_username=overrides.get("auth_username", self.auth_username),
            auth_password=overrides.get("auth_password", self.auth_password),
            auth_token=overrides.get("auth_token", self.auth_token),
            auto_instance=overrides.get("auto_instance", self.auto_instance),
            cluster_name=overrides.get("cluster_name", self.cluster_name),
            enable_tenant_isolation=overrides.get(
                "enable_tenant_isolation", self.enable_tenant_isolation
            ),
            default_tenant_label=overrides.get(
                "default_tenant_label", self.default_tenant_label
            ),
            batch_size=overrides.get("batch_size", self.batch_size),
            async_push=overrides.get("async_push", self.async_push),
        )

    def with_enabled(self, enabled: bool) -> PrometheusConfig:
        """Create config with new enabled state."""
        return self._copy_with(enabled=enabled)

    def with_job(self, job_name: str, instance: str = "") -> PrometheusConfig:
        """Create config with new job name and instance."""
        return self._copy_with(
            job_name=job_name,
            instance=instance if instance else self.instance,
        )

    def with_namespace(
        self,
        namespace: str,
        subsystem: str = "",
    ) -> PrometheusConfig:
        """Create config with new namespace and subsystem."""
        return self._copy_with(
            namespace=namespace,
            subsystem=subsystem if subsystem else self.subsystem,
        )

    def with_const_labels(self, **labels: str) -> PrometheusConfig:
        """Create config with additional constant labels."""
        return self._copy_with(const_labels={**self.const_labels, **labels})

    def with_gateway(
        self,
        url: str,
        timeout_seconds: float | None = None,
        retry_count: int | None = None,
        retry_delay_seconds: float | None = None,
    ) -> PrometheusConfig:
        """Create config with Push Gateway settings."""
        overrides: dict[str, Any] = {"gateway_url": url}
        if timeout_seconds is not None:
            overrides["gateway_timeout_seconds"] = timeout_seconds
        if retry_count is not None:
            overrides["gateway_retry_count"] = retry_count
        if retry_delay_seconds is not None:
            overrides["gateway_retry_delay_seconds"] = retry_delay_seconds
        return self._copy_with(**overrides)

    def with_http_server(
        self,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
    ) -> PrometheusConfig:
        """Create config with HTTP server settings."""
        overrides: dict[str, Any] = {}
        if host is not None:
            overrides["http_host"] = host
        if port is not None:
            overrides["http_port"] = port
        if path is not None:
            overrides["http_path"] = path
        return self._copy_with(**overrides)

    def with_naming_strategy(
        self,
        strategy: MetricNamingStrategy,
    ) -> PrometheusConfig:
        """Create config with new naming strategy."""
        return self._copy_with(naming_strategy=strategy)

    def with_output_options(
        self,
        include_timestamp: bool | None = None,
        include_help: bool | None = None,
        include_type: bool | None = None,
    ) -> PrometheusConfig:
        """Create config with output formatting options."""
        overrides: dict[str, Any] = {}
        if include_timestamp is not None:
            overrides["include_timestamp"] = include_timestamp
        if include_help is not None:
            overrides["include_help"] = include_help
        if include_type is not None:
            overrides["include_type"] = include_type
        return self._copy_with(**overrides)

    def with_auth(
        self,
        username: str = "",
        password: str = "",
        token: str = "",
    ) -> PrometheusConfig:
        """Create config with authentication settings.

        Args:
            username: Basic auth username.
            password: Basic auth password.
            token: Bearer token (alternative to basic auth).

        Returns:
            New PrometheusConfig with auth settings.

        Example:
            >>> config = PrometheusConfig().with_auth(
            ...     username="prometheus",
            ...     password="secret",
            ... )
            >>> # Or with bearer token
            >>> config = PrometheusConfig().with_auth(token="my-token")
        """
        return self._copy_with(
            auth_username=username,
            auth_password=password,
            auth_token=token,
        )

    def with_tenant_isolation(
        self,
        enabled: bool = True,
        label_name: str = "tenant_id",
    ) -> PrometheusConfig:
        """Create config with multi-tenant isolation settings.

        Args:
            enabled: Whether to enable tenant isolation.
            label_name: Label name for tenant ID.

        Returns:
            New PrometheusConfig with tenant settings.

        Example:
            >>> config = PrometheusConfig().with_tenant_isolation(
            ...     enabled=True,
            ...     label_name="organization_id",
            ... )
        """
        return self._copy_with(
            enable_tenant_isolation=enabled,
            default_tenant_label=label_name,
        )

    def with_distributed(
        self,
        cluster_name: str = "",
        auto_instance: bool = True,
    ) -> PrometheusConfig:
        """Create config for distributed environments.

        Args:
            cluster_name: Name of the cluster.
            auto_instance: Whether to auto-detect instance name.

        Returns:
            New PrometheusConfig with distributed settings.

        Example:
            >>> config = PrometheusConfig().with_distributed(
            ...     cluster_name="production-us-east",
            ...     auto_instance=True,
            ... )
        """
        return self._copy_with(
            cluster_name=cluster_name,
            auto_instance=auto_instance,
        )

    def with_performance(
        self,
        batch_size: int = 0,
        async_push: bool = False,
    ) -> PrometheusConfig:
        """Create config with performance settings.

        Args:
            batch_size: Number of metrics to batch (0 = no batching).
            async_push: Whether to use async push.

        Returns:
            New PrometheusConfig with performance settings.

        Example:
            >>> config = PrometheusConfig().with_performance(
            ...     batch_size=100,
            ...     async_push=True,
            ... )
        """
        return self._copy_with(
            batch_size=batch_size,
            async_push=async_push,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "job_name": self.job_name,
            "instance": self.instance,
            "namespace": self.namespace,
            "subsystem": self.subsystem,
            "const_labels": self.const_labels,
            "naming_strategy": self.naming_strategy.name,
            "include_timestamp": self.include_timestamp,
            "include_help": self.include_help,
            "include_type": self.include_type,
            "gateway_url": self.gateway_url,
            "gateway_timeout_seconds": self.gateway_timeout_seconds,
            "gateway_retry_count": self.gateway_retry_count,
            "gateway_retry_delay_seconds": self.gateway_retry_delay_seconds,
            "http_host": self.http_host,
            "http_port": self.http_port,
            "http_path": self.http_path,
            "auth_username": self.auth_username,
            "auth_password": self.auth_password,
            "auth_token": self.auth_token,
            "auto_instance": self.auto_instance,
            "cluster_name": self.cluster_name,
            "enable_tenant_isolation": self.enable_tenant_isolation,
            "default_tenant_label": self.default_tenant_label,
            "batch_size": self.batch_size,
            "async_push": self.async_push,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create PrometheusConfig from dictionary.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New PrometheusConfig instance.
        """
        naming_strategy_str = data.get("naming_strategy", "SNAKE_CASE")
        naming_strategy = (
            MetricNamingStrategy[naming_strategy_str]
            if isinstance(naming_strategy_str, str)
            else naming_strategy_str
        )

        return cls(
            enabled=data.get("enabled", True),
            job_name=data.get("job_name", "truthound"),
            instance=data.get("instance", ""),
            namespace=data.get("namespace", ""),
            subsystem=data.get("subsystem", ""),
            const_labels=data.get("const_labels", {}),
            naming_strategy=naming_strategy,
            include_timestamp=data.get("include_timestamp", False),
            include_help=data.get("include_help", True),
            include_type=data.get("include_type", True),
            gateway_url=data.get("gateway_url", ""),
            gateway_timeout_seconds=data.get("gateway_timeout_seconds", 10.0),
            gateway_retry_count=data.get("gateway_retry_count", 3),
            gateway_retry_delay_seconds=data.get("gateway_retry_delay_seconds", 1.0),
            http_host=data.get("http_host", "0.0.0.0"),
            http_port=data.get("http_port", 9090),
            http_path=data.get("http_path", "/metrics"),
            auth_username=data.get("auth_username", ""),
            auth_password=data.get("auth_password", ""),
            auth_token=data.get("auth_token", ""),
            auto_instance=data.get("auto_instance", True),
            cluster_name=data.get("cluster_name", ""),
            enable_tenant_isolation=data.get("enable_tenant_isolation", False),
            default_tenant_label=data.get("default_tenant_label", "tenant_id"),
            batch_size=data.get("batch_size", 0),
            async_push=data.get("async_push", False),
        )


# Default configurations
DEFAULT_PROMETHEUS_CONFIG = PrometheusConfig()

PUSHGATEWAY_PROMETHEUS_CONFIG = PrometheusConfig(
    gateway_url="http://localhost:9091",
    gateway_timeout_seconds=10.0,
    gateway_retry_count=3,
)

HTTP_SERVER_PROMETHEUS_CONFIG = PrometheusConfig(
    http_host="0.0.0.0",
    http_port=9090,
    http_path="/metrics",
)

MINIMAL_PROMETHEUS_CONFIG = PrometheusConfig(
    include_help=False,
    include_type=False,
    include_timestamp=False,
)


# =============================================================================
# Metric Naming Utilities
# =============================================================================


def _to_snake_case(name: str) -> str:
    """Convert a name to snake_case.

    Args:
        name: Original name.

    Returns:
        Snake case version.
    """
    # Replace common separators with underscore
    name = re.sub(r"[-.\s]+", "_", name)
    # Insert underscore before uppercase letters
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Convert to lowercase
    name = name.lower()
    # Remove consecutive underscores
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    return name.strip("_")


def _sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for Prometheus.

    Prometheus metric names must match [a-zA-Z_:][a-zA-Z0-9_:]*

    Args:
        name: Original metric name.

    Returns:
        Sanitized metric name.
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    # Ensure doesn't start with digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "metric"


def _sanitize_label_name(name: str) -> str:
    """Sanitize label name for Prometheus.

    Prometheus label names must match [a-zA-Z_][a-zA-Z0-9_]*
    Labels starting with __ are reserved.

    Args:
        name: Original label name.

    Returns:
        Sanitized label name.
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure doesn't start with digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    # Remove __ prefix (reserved)
    while sanitized.startswith("__"):
        sanitized = sanitized[1:]
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "label"


def _escape_label_value(value: str) -> str:
    """Escape label value for Prometheus text format.

    Args:
        value: Original label value.

    Returns:
        Escaped label value.
    """
    # Escape backslash, newline, and double quote
    value = value.replace("\\", "\\\\")
    value = value.replace("\n", "\\n")
    value = value.replace('"', '\\"')
    return value


def _build_metric_name(
    name: str,
    namespace: str,
    subsystem: str,
    strategy: MetricNamingStrategy,
) -> str:
    """Build full metric name with namespace and subsystem.

    Args:
        name: Base metric name.
        namespace: Namespace prefix.
        subsystem: Subsystem prefix.
        strategy: Naming strategy.

    Returns:
        Full metric name.
    """
    if strategy == MetricNamingStrategy.SNAKE_CASE:
        name = _to_snake_case(name)
        namespace = _to_snake_case(namespace) if namespace else ""
        subsystem = _to_snake_case(subsystem) if subsystem else ""

    parts = [p for p in [namespace, subsystem, name] if p]
    full_name = "_".join(parts)

    return _sanitize_metric_name(full_name)


# =============================================================================
# Prometheus Text Format
# =============================================================================


class PrometheusFormatter:
    """Formatter for Prometheus text exposition format.

    Implements OpenMetrics-compatible text format serialization.

    Example:
        >>> formatter = PrometheusFormatter(config)
        >>> text = formatter.format(metrics)
        >>> print(text)
        # HELP requests_total Total requests
        # TYPE requests_total counter
        requests_total{method="POST"} 42 1609459200000
    """

    # Prometheus type names
    _TYPE_MAP: dict[MetricType, str] = {
        MetricType.COUNTER: "counter",
        MetricType.GAUGE: "gauge",
        MetricType.HISTOGRAM: "histogram",
        MetricType.SUMMARY: "summary",
    }

    def __init__(self, config: PrometheusConfig | None = None) -> None:
        """Initialize formatter.

        Args:
            config: Prometheus configuration.
        """
        self._config = config or DEFAULT_PROMETHEUS_CONFIG

    def format(self, metrics: Sequence[MetricData]) -> str:
        """Format metrics in Prometheus text format.

        Args:
            metrics: Metrics to format.

        Returns:
            Prometheus text format string.
        """
        if not metrics:
            return ""

        # Group metrics by name (for HELP/TYPE comments)
        grouped: dict[str, list[MetricData]] = {}
        for metric in metrics:
            full_name = _build_metric_name(
                metric.name,
                self._config.namespace,
                self._config.subsystem,
                self._config.naming_strategy,
            )
            if full_name not in grouped:
                grouped[full_name] = []
            grouped[full_name].append(metric)

        lines: list[str] = []
        for name, group in grouped.items():
            # Use first metric for metadata
            first = group[0]

            # Add HELP comment
            if self._config.include_help and first.description:
                escaped_help = first.description.replace("\\", "\\\\").replace(
                    "\n", "\\n"
                )
                lines.append(f"# HELP {name} {escaped_help}")

            # Add TYPE comment
            if self._config.include_type:
                prom_type = self._TYPE_MAP.get(first.metric_type, "untyped")
                lines.append(f"# TYPE {name} {prom_type}")

            # Add metric lines
            for metric in group:
                line = self._format_metric_line(name, metric)
                lines.append(line)

        # Add trailing newline
        return "\n".join(lines) + "\n" if lines else ""

    def _format_metric_line(self, name: str, metric: MetricData) -> str:
        """Format a single metric line.

        Args:
            name: Full metric name.
            metric: Metric data.

        Returns:
            Formatted metric line.
        """
        # Merge labels with const_labels
        labels = {**self._config.const_labels, **metric.labels}

        # Format labels
        label_str = self._format_labels(labels)

        # Format value
        value_str = self._format_value(metric.value)

        # Build line
        if label_str:
            line = f"{name}{{{label_str}}} {value_str}"
        else:
            line = f"{name} {value_str}"

        # Add timestamp if configured
        if self._config.include_timestamp and metric.timestamp:
            timestamp_ms = self._parse_timestamp_ms(metric.timestamp)
            if timestamp_ms is not None:
                line = f"{line} {timestamp_ms}"

        return line

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus text format.

        Args:
            labels: Label dict.

        Returns:
            Formatted label string.
        """
        if not labels:
            return ""

        parts = []
        for key, value in sorted(labels.items()):
            sanitized_key = _sanitize_label_name(key)
            escaped_value = _escape_label_value(str(value))
            parts.append(f'{sanitized_key}="{escaped_value}"')

        return ",".join(parts)

    def _format_value(self, value: float) -> str:
        """Format metric value.

        Args:
            value: Metric value.

        Returns:
            Formatted value string.
        """
        if value == float("inf"):
            return "+Inf"
        if value == float("-inf"):
            return "-Inf"
        if value != value:  # NaN check
            return "NaN"

        # Use general format to avoid unnecessary decimal places
        if value == int(value):
            return str(int(value))
        return f"{value:g}"

    def _parse_timestamp_ms(self, timestamp: str) -> int | None:
        """Parse ISO timestamp to milliseconds since epoch.

        Args:
            timestamp: ISO format timestamp.

        Returns:
            Milliseconds since epoch, or None if parsing fails.
        """
        try:
            from datetime import datetime

            # Handle various ISO formats
            if timestamp.endswith("Z"):
                timestamp = timestamp[:-1] + "+00:00"

            dt = datetime.fromisoformat(timestamp)
            return int(dt.timestamp() * 1000)
        except (ValueError, AttributeError):
            return None

    def content_type(self) -> str:
        """Get content type for Prometheus text format.

        Returns:
            MIME content type.
        """
        return "text/plain; version=0.0.4; charset=utf-8"


# =============================================================================
# Push Gateway Client
# =============================================================================


class PrometheusPushGatewayClient:
    """Client for pushing metrics to Prometheus Push Gateway.

    Supports retry, timeout, and job/instance grouping.

    Example:
        >>> client = PrometheusPushGatewayClient(
        ...     gateway_url="http://pushgateway:9091",
        ...     job_name="batch_job",
        ... )
        >>> client.push(formatted_metrics)
        >>> client.delete()  # Remove job from gateway
    """

    def __init__(
        self,
        gateway_url: str,
        job_name: str,
        instance: str = "",
        grouping_key: dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
        retry_count: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize Push Gateway client.

        Args:
            gateway_url: Push Gateway base URL.
            job_name: Job name for grouping.
            instance: Optional instance label.
            grouping_key: Additional grouping labels.
            timeout_seconds: Request timeout.
            retry_count: Number of retries on failure.
            retry_delay_seconds: Delay between retries.
        """
        self._gateway_url = gateway_url.rstrip("/")
        self._job_name = job_name
        self._instance = instance
        self._grouping_key = grouping_key or {}
        self._timeout_seconds = timeout_seconds
        self._retry_count = retry_count
        self._retry_delay_seconds = retry_delay_seconds
        self._lock = threading.Lock()

    def _build_url(self) -> str:
        """Build Push Gateway URL with job and grouping key.

        Returns:
            Full URL for pushing metrics.
        """
        # URL encode job name
        encoded_job = urllib.parse.quote(self._job_name, safe="")
        path = f"/metrics/job/{encoded_job}"

        # Add instance if specified
        if self._instance:
            encoded_instance = urllib.parse.quote(self._instance, safe="")
            path = f"{path}/instance/{encoded_instance}"

        # Add grouping key labels
        for key, value in sorted(self._grouping_key.items()):
            encoded_key = urllib.parse.quote(key, safe="")
            encoded_value = urllib.parse.quote(value, safe="")
            path = f"{path}/{encoded_key}/{encoded_value}"

        return f"{self._gateway_url}{path}"

    def push(
        self,
        data: str,
        content_type: str = "text/plain; version=0.0.4; charset=utf-8",
    ) -> None:
        """Push metrics to Push Gateway.

        Args:
            data: Formatted metric data.
            content_type: MIME content type.

        Raises:
            PushGatewayError: If push fails after retries.
        """
        url = self._build_url()
        last_error: Exception | None = None

        for attempt in range(self._retry_count + 1):
            try:
                self._do_push(url, data, content_type)
                return
            except Exception as e:
                last_error = e
                if attempt < self._retry_count:
                    time.sleep(self._retry_delay_seconds)

        raise PushGatewayError(
            f"Failed to push metrics after {self._retry_count + 1} attempts",
            gateway_url=url,
            cause=last_error,
        )

    def _do_push(self, url: str, data: str, content_type: str) -> None:
        """Execute single push request.

        Args:
            url: Target URL.
            data: Metric data.
            content_type: Content type.

        Raises:
            PushGatewayError: If request fails.
        """
        request = urllib.request.Request(
            url,
            data=data.encode("utf-8"),
            method="POST",
            headers={"Content-Type": content_type},
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                if response.status not in (200, 202):
                    raise PushGatewayError(
                        f"Push Gateway returned status {response.status}",
                        gateway_url=url,
                        status_code=response.status,
                    )
        except urllib.error.HTTPError as e:
            raise PushGatewayError(
                f"HTTP error: {e.reason}",
                gateway_url=url,
                status_code=e.code,
                cause=e,
            ) from e
        except urllib.error.URLError as e:
            raise PushGatewayError(
                f"URL error: {e.reason}",
                gateway_url=url,
                cause=e,
            ) from e

    def delete(self) -> None:
        """Delete all metrics for this job from Push Gateway.

        Raises:
            PushGatewayError: If delete fails.
        """
        url = self._build_url()

        request = urllib.request.Request(url, method="DELETE")

        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                if response.status not in (200, 202, 204):
                    raise PushGatewayError(
                        f"Delete returned status {response.status}",
                        gateway_url=url,
                        status_code=response.status,
                    )
        except urllib.error.HTTPError as e:
            raise PushGatewayError(
                f"HTTP error: {e.reason}",
                gateway_url=url,
                status_code=e.code,
                cause=e,
            ) from e
        except urllib.error.URLError as e:
            raise PushGatewayError(
                f"URL error: {e.reason}",
                gateway_url=url,
                cause=e,
            ) from e

    def replace(
        self,
        data: str,
        content_type: str = "text/plain; version=0.0.4; charset=utf-8",
    ) -> None:
        """Replace all metrics for this job (PUT instead of POST).

        Args:
            data: Formatted metric data.
            content_type: MIME content type.

        Raises:
            PushGatewayError: If replace fails.
        """
        url = self._build_url()

        request = urllib.request.Request(
            url,
            data=data.encode("utf-8"),
            method="PUT",
            headers={"Content-Type": content_type},
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                if response.status not in (200, 202):
                    raise PushGatewayError(
                        f"Replace returned status {response.status}",
                        gateway_url=url,
                        status_code=response.status,
                    )
        except urllib.error.HTTPError as e:
            raise PushGatewayError(
                f"HTTP error: {e.reason}",
                gateway_url=url,
                status_code=e.code,
                cause=e,
            ) from e
        except urllib.error.URLError as e:
            raise PushGatewayError(
                f"URL error: {e.reason}",
                gateway_url=url,
                cause=e,
            ) from e


# =============================================================================
# HTTP Server for Scraping
# =============================================================================


class PrometheusHttpServer:
    """HTTP server for exposing metrics to Prometheus scraper.

    Provides a non-blocking server that exposes metrics at configurable endpoint.

    Example:
        >>> def get_metrics():
        ...     return registry.collect()
        >>> server = PrometheusHttpServer(
        ...     host="0.0.0.0",
        ...     port=9090,
        ...     metrics_callback=get_metrics,
        ... )
        >>> server.start()  # Non-blocking
        >>> # ... application runs ...
        >>> server.stop()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        path: str = "/metrics",
        metrics_callback: Callable[[], Sequence[MetricData]] | None = None,
        formatter: PrometheusFormatter | None = None,
    ) -> None:
        """Initialize HTTP server.

        Args:
            host: Server host.
            port: Server port.
            path: Metrics endpoint path.
            metrics_callback: Callback to get current metrics.
            formatter: Metric formatter.
        """
        self._host = host
        self._port = port
        self._path = path
        self._metrics_callback = metrics_callback
        self._formatter = formatter or PrometheusFormatter()
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False

    def _create_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        """Create HTTP request handler class.

        Returns:
            Handler class.
        """
        metrics_callback = self._metrics_callback
        formatter = self._formatter
        path = self._path

        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            """Handler for /metrics endpoint."""

            def log_message(self, format: str, *args: Any) -> None:
                """Suppress default logging."""
                pass

            def do_GET(self) -> None:
                """Handle GET requests."""
                if self.path == path or self.path == path + "/":
                    self._serve_metrics()
                else:
                    self.send_error(404, "Not Found")

            def _serve_metrics(self) -> None:
                """Serve metrics in Prometheus format."""
                try:
                    if metrics_callback:
                        metrics = metrics_callback()
                        body = formatter.format(metrics)
                    else:
                        body = ""

                    self.send_response(200)
                    self.send_header("Content-Type", formatter.content_type())
                    self.send_header("Content-Length", str(len(body.encode("utf-8"))))
                    self.end_headers()
                    self.wfile.write(body.encode("utf-8"))
                except Exception:
                    self.send_error(500, "Internal Server Error")

        return MetricsHandler

    def start(self) -> None:
        """Start the HTTP server in a background thread.

        Raises:
            HttpServerError: If server fails to start.
        """
        with self._lock:
            if self._running:
                return

            try:
                handler_class = self._create_handler()

                # Allow address reuse
                socketserver.TCPServer.allow_reuse_address = True

                self._server = socketserver.TCPServer(
                    (self._host, self._port),
                    handler_class,
                )

                self._thread = threading.Thread(
                    target=self._server.serve_forever,
                    daemon=True,
                    name="prometheus-http-server",
                )
                self._thread.start()
                self._running = True
            except OSError as e:
                raise HttpServerError(
                    f"Failed to start server: {e}",
                    host=self._host,
                    port=self._port,
                    cause=e,
                ) from e

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the HTTP server.

        Args:
            timeout: Timeout for thread join.
        """
        with self._lock:
            if not self._running:
                return

            if self._server:
                self._server.shutdown()
                self._server.server_close()
                self._server = None

            if self._thread:
                self._thread.join(timeout=timeout)
                self._thread = None

            self._running = False

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def address(self) -> tuple[str, int]:
        """Get server address."""
        return (self._host, self._port)

    def __enter__(self) -> PrometheusHttpServer:
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
        self.stop()


# =============================================================================
# Export Hooks
# =============================================================================


class BaseExportHook:
    """Base implementation of ExportHook with no-op methods."""

    def on_export_start(
        self,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Called when export starts."""
        pass

    def on_export_success(
        self,
        metric_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when export succeeds."""
        pass

    def on_export_error(
        self,
        error: Exception,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Called when export fails."""
        pass


class LoggingExportHook(BaseExportHook):
    """Hook that logs export events."""

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name.
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.exporters.prometheus")

    def on_export_start(
        self,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Log export start."""
        self._logger.debug(
            "Prometheus export starting",
            metric_count=metric_count,
            **context,
        )

    def on_export_success(
        self,
        metric_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log export success."""
        self._logger.info(
            "Prometheus export completed",
            metric_count=metric_count,
            duration_ms=round(duration_ms, 2),
            **context,
        )

    def on_export_error(
        self,
        error: Exception,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Log export error."""
        self._logger.error(
            "Prometheus export failed",
            metric_count=metric_count,
            error_type=type(error).__name__,
            error_message=str(error),
            **context,
        )


class MetricsExportHook(BaseExportHook):
    """Hook that collects export statistics."""

    def __init__(self) -> None:
        """Initialize metrics hook."""
        self._lock = threading.Lock()
        self._export_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_metrics_exported = 0
        self._total_duration_ms = 0.0

    @property
    def export_count(self) -> int:
        """Total number of exports attempted."""
        with self._lock:
            return self._export_count

    @property
    def success_count(self) -> int:
        """Number of successful exports."""
        with self._lock:
            return self._success_count

    @property
    def error_count(self) -> int:
        """Number of failed exports."""
        with self._lock:
            return self._error_count

    @property
    def total_metrics_exported(self) -> int:
        """Total number of metrics exported."""
        with self._lock:
            return self._total_metrics_exported

    @property
    def average_duration_ms(self) -> float:
        """Average export duration in milliseconds."""
        with self._lock:
            if self._success_count == 0:
                return 0.0
            return self._total_duration_ms / self._success_count

    def on_export_start(
        self,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Track export start."""
        with self._lock:
            self._export_count += 1

    def on_export_success(
        self,
        metric_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Track export success."""
        with self._lock:
            self._success_count += 1
            self._total_metrics_exported += metric_count
            self._total_duration_ms += duration_ms

    def on_export_error(
        self,
        error: Exception,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Track export error."""
        with self._lock:
            self._error_count += 1

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._export_count = 0
            self._success_count = 0
            self._error_count = 0
            self._total_metrics_exported = 0
            self._total_duration_ms = 0.0


class CompositeExportHook(BaseExportHook):
    """Hook that delegates to multiple hooks."""

    def __init__(self, hooks: Sequence[ExportHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to delegate to.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: ExportHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def on_export_start(
        self,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_start(metric_count, context)

    def on_export_success(
        self,
        metric_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_success(metric_count, duration_ms, context)

    def on_export_error(
        self,
        error: Exception,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_error(error, metric_count, context)


# =============================================================================
# Main Prometheus Exporter
# =============================================================================


class PrometheusExporter:
    """Prometheus metric exporter implementing MetricExporter protocol.

    Provides flexible export to Prometheus via:
    - Push Gateway (for batch jobs)
    - HTTP endpoint (for scraping)
    - Direct text format output

    Thread-safe and production-ready with retry, timeout, and hooks.

    Example:
        >>> exporter = PrometheusExporter(
        ...     config=PrometheusConfig(
        ...         namespace="truthound",
        ...         job_name="data_quality",
        ...     ),
        ... )
        >>> exporter.export(metrics)

        >>> # With Push Gateway
        >>> exporter = PrometheusExporter(
        ...     config=PrometheusConfig(
        ...         job_name="batch_job",
        ...     ).with_gateway("http://pushgateway:9091"),
        ... )
        >>> exporter.export(metrics)  # Pushes to gateway
    """

    def __init__(
        self,
        config: PrometheusConfig | None = None,
        formatter: PrometheusFormatter | None = None,
        hooks: Sequence[ExportHook] | None = None,
    ) -> None:
        """Initialize exporter.

        Args:
            config: Prometheus configuration.
            formatter: Custom formatter (uses config-based formatter if None).
            hooks: Export event hooks.
        """
        self._config = config or DEFAULT_PROMETHEUS_CONFIG
        self._formatter = formatter or PrometheusFormatter(self._config)
        self._hooks = list(hooks) if hooks else []
        self._push_client: PrometheusPushGatewayClient | None = None
        self._lock = threading.Lock()
        self._last_export: str = ""

        # Initialize push client if gateway URL is configured
        if self._config.gateway_url:
            self._push_client = PrometheusPushGatewayClient(
                gateway_url=self._config.gateway_url,
                job_name=self._config.job_name,
                instance=self._config.instance,
                grouping_key=dict(self._config.const_labels),
                timeout_seconds=self._config.gateway_timeout_seconds,
                retry_count=self._config.gateway_retry_count,
                retry_delay_seconds=self._config.gateway_retry_delay_seconds,
            )

    @property
    def config(self) -> PrometheusConfig:
        """Get exporter configuration."""
        return self._config

    @property
    def formatter(self) -> PrometheusFormatter:
        """Get metric formatter."""
        return self._formatter

    @property
    def last_export(self) -> str:
        """Get last exported text (for debugging)."""
        with self._lock:
            return self._last_export

    def export(self, metrics: Sequence[MetricData]) -> None:
        """Export metrics to Prometheus.

        If Push Gateway is configured, metrics are pushed.
        Otherwise, metrics are formatted and stored for HTTP endpoint.

        Args:
            metrics: Metrics to export.
        """
        if not self._config.enabled:
            return

        metric_count = len(metrics)
        context = {
            "job_name": self._config.job_name,
            "has_gateway": bool(self._config.gateway_url),
        }

        # Notify hooks
        self._notify_start(metric_count, context)

        start_time = time.perf_counter()
        try:
            # Format metrics
            text = self._formatter.format(metrics)

            with self._lock:
                self._last_export = text

            # Push to gateway if configured
            if self._push_client:
                self._push_client.push(
                    data=text,
                    content_type=self._formatter.content_type(),
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._notify_success(metric_count, duration_ms, context)

        except Exception as e:
            self._notify_error(e, metric_count, context)
            raise

    def get_metrics_text(self, metrics: Sequence[MetricData]) -> str:
        """Format metrics to Prometheus text format without exporting.

        Args:
            metrics: Metrics to format.

        Returns:
            Prometheus text format string.
        """
        return self._formatter.format(metrics)

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass

    def _notify_start(self, metric_count: int, context: dict[str, Any]) -> None:
        """Notify hooks about export start."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_start(metric_count, context)

    def _notify_success(
        self,
        metric_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Notify hooks about export success."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_success(metric_count, duration_ms, context)

    def _notify_error(
        self,
        error: Exception,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Notify hooks about export error."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_error(error, metric_count, context)


# =============================================================================
# Factory Functions
# =============================================================================


def create_prometheus_exporter(
    namespace: str = "",
    job_name: str = "truthound",
    const_labels: dict[str, str] | None = None,
    include_help: bool = True,
    include_type: bool = True,
    hooks: Sequence[ExportHook] | None = None,
) -> PrometheusExporter:
    """Create a Prometheus exporter with common settings.

    Args:
        namespace: Metric namespace prefix.
        job_name: Job name for grouping.
        const_labels: Constant labels for all metrics.
        include_help: Whether to include HELP comments.
        include_type: Whether to include TYPE comments.
        hooks: Export event hooks.

    Returns:
        Configured PrometheusExporter.

    Example:
        >>> exporter = create_prometheus_exporter(
        ...     namespace="myapp",
        ...     job_name="data_processor",
        ...     const_labels={"env": "production"},
        ... )
    """
    config = PrometheusConfig(
        job_name=job_name,
        namespace=namespace,
        const_labels=const_labels or {},
        include_help=include_help,
        include_type=include_type,
    )
    return PrometheusExporter(config=config, hooks=hooks)


def create_pushgateway_exporter(
    gateway_url: str,
    job_name: str,
    instance: str = "",
    namespace: str = "",
    const_labels: dict[str, str] | None = None,
    timeout_seconds: float = 10.0,
    retry_count: int = 3,
    hooks: Sequence[ExportHook] | None = None,
) -> PrometheusExporter:
    """Create a Prometheus exporter that pushes to Push Gateway.

    Args:
        gateway_url: Push Gateway URL.
        job_name: Job name for grouping.
        instance: Optional instance label.
        namespace: Metric namespace prefix.
        const_labels: Constant labels for all metrics.
        timeout_seconds: Request timeout.
        retry_count: Number of retries.
        hooks: Export event hooks.

    Returns:
        Configured PrometheusExporter for Push Gateway.

    Example:
        >>> exporter = create_pushgateway_exporter(
        ...     gateway_url="http://pushgateway:9091",
        ...     job_name="batch_job",
        ...     instance="worker-1",
        ... )
    """
    config = PrometheusConfig(
        job_name=job_name,
        instance=instance,
        namespace=namespace,
        const_labels=const_labels or {},
        gateway_url=gateway_url,
        gateway_timeout_seconds=timeout_seconds,
        gateway_retry_count=retry_count,
    )
    return PrometheusExporter(config=config, hooks=hooks)


def create_prometheus_http_server(
    host: str = "0.0.0.0",
    port: int = 9090,
    path: str = "/metrics",
    registry: MetricsRegistry | None = None,
    metrics_callback: Callable[[], Sequence[MetricData]] | None = None,
    config: PrometheusConfig | None = None,
) -> PrometheusHttpServer:
    """Create an HTTP server for Prometheus scraping.

    Args:
        host: Server host.
        port: Server port.
        path: Metrics endpoint path.
        registry: MetricsRegistry to expose (uses its collect method).
        metrics_callback: Custom callback to get metrics.
        config: Prometheus configuration for formatting.

    Returns:
        Configured PrometheusHttpServer.

    Example:
        >>> server = create_prometheus_http_server(
        ...     port=9090,
        ...     registry=my_registry,
        ... )
        >>> server.start()
    """
    if registry is not None:
        callback = registry.collect
    else:
        callback = metrics_callback

    formatter = PrometheusFormatter(config) if config else PrometheusFormatter()

    return PrometheusHttpServer(
        host=host,
        port=port,
        path=path,
        metrics_callback=callback,
        formatter=formatter,
    )


# =============================================================================
# Instance Auto-Detection Utilities
# =============================================================================


def get_auto_instance_name() -> str:
    """Auto-detect instance name for distributed environments.

    Returns:
        Instance name in format 'hostname:pid' or from environment.
    """
    # Check environment variables first (Kubernetes, Docker, etc.)
    for env_var in [
        "HOSTNAME",
        "POD_NAME",
        "INSTANCE_ID",
        "CONTAINER_ID",
        "DYNO",  # Heroku
    ]:
        value = os.environ.get(env_var)
        if value:
            return value

    # Fallback to hostname:pid
    try:
        hostname = socket.gethostname()
        pid = os.getpid()
        return f"{hostname}:{pid}"
    except Exception:
        return f"unknown:{os.getpid()}"


def get_cluster_labels() -> dict[str, str]:
    """Get cluster-related labels from environment.

    Returns:
        Dictionary of cluster labels.
    """
    labels: dict[str, str] = {}

    # Kubernetes labels
    k8s_namespace = os.environ.get("POD_NAMESPACE")
    if k8s_namespace:
        labels["k8s_namespace"] = k8s_namespace

    k8s_node = os.environ.get("NODE_NAME")
    if k8s_node:
        labels["k8s_node"] = k8s_node

    # Cloud provider labels
    cloud_region = os.environ.get("CLOUD_REGION") or os.environ.get("AWS_REGION")
    if cloud_region:
        labels["cloud_region"] = cloud_region

    # Platform info
    labels["platform"] = platform.system().lower()

    return labels


# =============================================================================
# Authentication Support
# =============================================================================


class AuthenticatedPushGatewayClient(PrometheusPushGatewayClient):
    """Push Gateway client with authentication support.

    Supports Basic Auth and Bearer token authentication.

    Example:
        >>> client = AuthenticatedPushGatewayClient(
        ...     gateway_url="http://pushgateway:9091",
        ...     job_name="batch_job",
        ...     auth_username="prometheus",
        ...     auth_password="secret",
        ... )
        >>> client.push(formatted_metrics)

        >>> # Or with bearer token
        >>> client = AuthenticatedPushGatewayClient(
        ...     gateway_url="http://pushgateway:9091",
        ...     job_name="batch_job",
        ...     auth_token="my-bearer-token",
        ... )
    """

    def __init__(
        self,
        gateway_url: str,
        job_name: str,
        instance: str = "",
        grouping_key: dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
        retry_count: int = 3,
        retry_delay_seconds: float = 1.0,
        auth_username: str = "",
        auth_password: str = "",
        auth_token: str = "",
    ) -> None:
        """Initialize authenticated Push Gateway client.

        Args:
            gateway_url: Push Gateway base URL.
            job_name: Job name for grouping.
            instance: Optional instance label.
            grouping_key: Additional grouping labels.
            timeout_seconds: Request timeout.
            retry_count: Number of retries on failure.
            retry_delay_seconds: Delay between retries.
            auth_username: Basic auth username.
            auth_password: Basic auth password.
            auth_token: Bearer token (alternative to basic auth).
        """
        super().__init__(
            gateway_url=gateway_url,
            job_name=job_name,
            instance=instance,
            grouping_key=grouping_key,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_delay_seconds=retry_delay_seconds,
        )
        self._auth_username = auth_username
        self._auth_password = auth_password
        self._auth_token = auth_token

    def _get_auth_header(self) -> dict[str, str]:
        """Get authentication header.

        Returns:
            Dictionary with Authorization header if auth is configured.
        """
        if self._auth_token:
            return {"Authorization": f"Bearer {self._auth_token}"}
        if self._auth_username and self._auth_password:
            credentials = f"{self._auth_username}:{self._auth_password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        return {}

    def _do_push(self, url: str, data: str, content_type: str) -> None:
        """Execute single push request with authentication."""
        headers = {"Content-Type": content_type}
        headers.update(self._get_auth_header())

        request = urllib.request.Request(
            url,
            data=data.encode("utf-8"),
            method="POST",
            headers=headers,
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                if response.status not in (200, 202):
                    raise PushGatewayError(
                        f"Push Gateway returned status {response.status}",
                        gateway_url=url,
                        status_code=response.status,
                    )
        except urllib.error.HTTPError as e:
            raise PushGatewayError(
                f"HTTP error: {e.reason}",
                gateway_url=url,
                status_code=e.code,
                cause=e,
            ) from e
        except urllib.error.URLError as e:
            raise PushGatewayError(
                f"URL error: {e.reason}",
                gateway_url=url,
                cause=e,
            ) from e


# =============================================================================
# Multi-Tenant Support
# =============================================================================


class TenantAwarePrometheusExporter(PrometheusExporter):
    """Prometheus exporter with multi-tenant support.

    Automatically injects tenant_id label into all metrics when
    a tenant context is active.

    Example:
        >>> exporter = TenantAwarePrometheusExporter(
        ...     config=PrometheusConfig().with_tenant_isolation(True),
        ... )
        >>> # When used within a tenant context:
        >>> with tenant_context("tenant_123"):
        ...     exporter.export(metrics)  # All metrics get tenant_id="tenant_123"
    """

    def __init__(
        self,
        config: PrometheusConfig | None = None,
        formatter: PrometheusFormatter | None = None,
        hooks: Sequence[ExportHook] | None = None,
        tenant_getter: Callable[[], str | None] | None = None,
    ) -> None:
        """Initialize tenant-aware exporter.

        Args:
            config: Prometheus configuration.
            formatter: Custom formatter.
            hooks: Export event hooks.
            tenant_getter: Custom function to get current tenant ID.
                If None, tries to use enterprise.multi_tenant.context.
        """
        super().__init__(config=config, formatter=formatter, hooks=hooks)
        self._tenant_getter = tenant_getter or self._default_tenant_getter

    def _default_tenant_getter(self) -> str | None:
        """Default tenant getter using enterprise multi_tenant module."""
        try:
            # Try to import from enterprise module
            from packages.enterprise.multi_tenant.context import get_current_tenant_id

            return get_current_tenant_id()
        except ImportError:
            return None

    def export(self, metrics: Sequence[MetricData]) -> None:
        """Export metrics with tenant label injection.

        Args:
            metrics: Metrics to export.
        """
        if not self._config.enable_tenant_isolation:
            return super().export(metrics)

        tenant_id = self._tenant_getter()
        if not tenant_id:
            return super().export(metrics)

        # Inject tenant label into all metrics
        tenant_label = self._config.default_tenant_label
        enriched_metrics = []
        for metric in metrics:
            enriched_labels = {**metric.labels, tenant_label: tenant_id}
            enriched_metric = MetricData(
                name=metric.name,
                description=metric.description,
                metric_type=metric.metric_type,
                value=metric.value,
                labels=enriched_labels,
                timestamp=metric.timestamp,
            )
            enriched_metrics.append(enriched_metric)

        return super().export(enriched_metrics)


# =============================================================================
# Async Support
# =============================================================================


class AsyncPrometheusExporter:
    """Async Prometheus exporter for high-throughput environments.

    Provides async export methods for use with asyncio.

    Example:
        >>> exporter = AsyncPrometheusExporter(config)
        >>> await exporter.export(metrics)
    """

    def __init__(
        self,
        config: PrometheusConfig | None = None,
        formatter: PrometheusFormatter | None = None,
        hooks: Sequence[ExportHook] | None = None,
    ) -> None:
        """Initialize async exporter.

        Args:
            config: Prometheus configuration.
            formatter: Custom formatter.
            hooks: Export event hooks.
        """
        self._config = config or DEFAULT_PROMETHEUS_CONFIG
        self._formatter = formatter or PrometheusFormatter(self._config)
        self._hooks = list(hooks) if hooks else []
        self._lock = asyncio.Lock()
        self._last_export: str = ""
        self._push_client: AuthenticatedPushGatewayClient | None = None

        if self._config.gateway_url:
            self._push_client = AuthenticatedPushGatewayClient(
                gateway_url=self._config.gateway_url,
                job_name=self._config.job_name,
                instance=self._config.instance
                or (get_auto_instance_name() if self._config.auto_instance else ""),
                grouping_key=dict(self._config.const_labels),
                timeout_seconds=self._config.gateway_timeout_seconds,
                retry_count=self._config.gateway_retry_count,
                retry_delay_seconds=self._config.gateway_retry_delay_seconds,
                auth_username=self._config.auth_username,
                auth_password=self._config.auth_password,
                auth_token=self._config.auth_token,
            )

    async def export(self, metrics: Sequence[MetricData]) -> None:
        """Export metrics asynchronously.

        Args:
            metrics: Metrics to export.
        """
        if not self._config.enabled:
            return

        metric_count = len(metrics)
        context = {
            "job_name": self._config.job_name,
            "has_gateway": bool(self._config.gateway_url),
            "async": True,
        }

        self._notify_start(metric_count, context)
        start_time = time.perf_counter()

        try:
            text = self._formatter.format(metrics)

            async with self._lock:
                self._last_export = text

            if self._push_client:
                # Run sync push in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._push_client.push,
                    text,
                    self._formatter.content_type(),
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._notify_success(metric_count, duration_ms, context)

        except Exception as e:
            self._notify_error(e, metric_count, context)
            raise

    def _notify_start(self, metric_count: int, context: dict[str, Any]) -> None:
        """Notify hooks about export start."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_start(metric_count, context)

    def _notify_success(
        self,
        metric_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Notify hooks about export success."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_success(metric_count, duration_ms, context)

    def _notify_error(
        self,
        error: Exception,
        metric_count: int,
        context: dict[str, Any],
    ) -> None:
        """Notify hooks about export error."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_export_error(error, metric_count, context)

    @property
    def last_export(self) -> str:
        """Get last exported text."""
        return self._last_export

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


# =============================================================================
# Enterprise Configuration Presets
# =============================================================================


# Enterprise preset with multi-tenant isolation
ENTERPRISE_PROMETHEUS_CONFIG = PrometheusConfig(
    enable_tenant_isolation=True,
    auto_instance=True,
    gateway_retry_count=5,
    gateway_timeout_seconds=15.0,
)

# High-availability preset with aggressive retry
HA_PROMETHEUS_CONFIG = PrometheusConfig(
    gateway_retry_count=5,
    gateway_retry_delay_seconds=2.0,
    gateway_timeout_seconds=30.0,
    auto_instance=True,
)

# Async/high-throughput preset
ASYNC_PROMETHEUS_CONFIG = PrometheusConfig(
    async_push=True,
    batch_size=100,
    auto_instance=True,
)
