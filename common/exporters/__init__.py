"""Metric exporters for Truthound Integrations.

This module provides various metric exporters for different observability backends.

Features:
    - Prometheus text format serialization
    - Push Gateway client with retry and authentication
    - HTTP endpoint server for scraping
    - Multi-tenant support with tenant-aware metric isolation
    - Async support for high-throughput environments
    - Distributed environment features (instance auto-detection, cluster labels)
"""

from __future__ import annotations

from common.exporters.prometheus import (
    # Configuration
    PrometheusConfig,
    DEFAULT_PROMETHEUS_CONFIG,
    PUSHGATEWAY_PROMETHEUS_CONFIG,
    ENTERPRISE_PROMETHEUS_CONFIG,
    HA_PROMETHEUS_CONFIG,
    ASYNC_PROMETHEUS_CONFIG,
    # Enums
    ExportMode,
    MetricNamingStrategy,
    # Protocols
    MetricFormatter,
    MetricTransport,
    ExportHook,
    # Core Exporter
    PrometheusExporter,
    PrometheusFormatter,
    # Push Gateway
    PrometheusPushGatewayClient,
    AuthenticatedPushGatewayClient,
    # HTTP Server
    PrometheusHttpServer,
    # Multi-Tenant
    TenantAwarePrometheusExporter,
    # Async
    AsyncPrometheusExporter,
    # Hooks
    BaseExportHook,
    LoggingExportHook,
    MetricsExportHook,
    CompositeExportHook,
    # Utilities
    get_auto_instance_name,
    get_cluster_labels,
    # Factory functions
    create_prometheus_exporter,
    create_pushgateway_exporter,
    create_prometheus_http_server,
    # Exceptions
    PrometheusExporterError,
    PushGatewayError,
    HttpServerError,
)

__all__ = [
    # Configuration
    "PrometheusConfig",
    "DEFAULT_PROMETHEUS_CONFIG",
    "PUSHGATEWAY_PROMETHEUS_CONFIG",
    "ENTERPRISE_PROMETHEUS_CONFIG",
    "HA_PROMETHEUS_CONFIG",
    "ASYNC_PROMETHEUS_CONFIG",
    # Enums
    "ExportMode",
    "MetricNamingStrategy",
    # Protocols
    "MetricFormatter",
    "MetricTransport",
    "ExportHook",
    # Core Exporter
    "PrometheusExporter",
    "PrometheusFormatter",
    # Push Gateway
    "PrometheusPushGatewayClient",
    "AuthenticatedPushGatewayClient",
    # HTTP Server
    "PrometheusHttpServer",
    # Multi-Tenant
    "TenantAwarePrometheusExporter",
    # Async
    "AsyncPrometheusExporter",
    # Hooks
    "BaseExportHook",
    "LoggingExportHook",
    "MetricsExportHook",
    "CompositeExportHook",
    # Utilities
    "get_auto_instance_name",
    "get_cluster_labels",
    # Factory functions
    "create_prometheus_exporter",
    "create_pushgateway_exporter",
    "create_prometheus_http_server",
    # Exceptions
    "PrometheusExporterError",
    "PushGatewayError",
    "HttpServerError",
]
