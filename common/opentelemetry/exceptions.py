"""OpenTelemetry-specific exceptions for truthound-orchestration.

This module defines the exception hierarchy for OpenTelemetry integration errors.
All exceptions inherit from the base TruthoundIntegrationError for consistency.
"""

from typing import Any

from common.exceptions import TruthoundIntegrationError

__all__ = [
    "OTelError",
    "OTelConfigurationError",
    "OTelExporterError",
    "OTelProviderError",
    "OTelBridgeError",
    "OTelSamplingError",
    "OTelContextError",
    "OTelNotInstalledError",
]


class OTelError(TruthoundIntegrationError):
    """Base exception for all OpenTelemetry-related errors.

    All OpenTelemetry exceptions inherit from this class, allowing
    users to catch all OTel-related errors with a single except clause.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize OTelError.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.details = details or {}


class OTelConfigurationError(OTelError):
    """Raised when OpenTelemetry configuration is invalid.

    This includes invalid endpoint URLs, unsupported protocols,
    missing required configuration, etc.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OTelConfigurationError.

        Args:
            message: Human-readable error message.
            config_key: The configuration key that caused the error.
            config_value: The invalid configuration value.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class OTelExporterError(OTelError):
    """Raised when an exporter fails to export telemetry data.

    This includes connection failures, serialization errors,
    and backend rejection of data.
    """

    def __init__(
        self,
        message: str,
        exporter_type: str | None = None,
        endpoint: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OTelExporterError.

        Args:
            message: Human-readable error message.
            exporter_type: The type of exporter that failed.
            endpoint: The endpoint the exporter was trying to reach.
            original_error: The underlying exception that caused this error.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.exporter_type = exporter_type
        self.endpoint = endpoint
        self.original_error = original_error


class OTelProviderError(OTelError):
    """Raised when a meter or tracer provider encounters an error.

    This includes initialization failures, shutdown errors,
    and resource creation failures.
    """

    def __init__(
        self,
        message: str,
        provider_type: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OTelProviderError.

        Args:
            message: Human-readable error message.
            provider_type: The type of provider (meter/tracer) that failed.
            original_error: The underlying exception that caused this error.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.provider_type = provider_type
        self.original_error = original_error


class OTelBridgeError(OTelError):
    """Raised when bridging between internal and OTel systems fails.

    This includes metric conversion failures, span translation errors,
    and context propagation failures.
    """

    def __init__(
        self,
        message: str,
        bridge_type: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OTelBridgeError.

        Args:
            message: Human-readable error message.
            bridge_type: The type of bridge (metrics/tracing/context).
            source_type: The source data type being converted.
            target_type: The target data type.
            original_error: The underlying exception that caused this error.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.bridge_type = bridge_type
        self.source_type = source_type
        self.target_type = target_type
        self.original_error = original_error


class OTelSamplingError(OTelError):
    """Raised when a sampling decision cannot be made.

    This includes sampler initialization failures and
    invalid sampling configuration.
    """

    def __init__(
        self,
        message: str,
        sampler_type: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OTelSamplingError.

        Args:
            message: Human-readable error message.
            sampler_type: The type of sampler that failed.
            original_error: The underlying exception that caused this error.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.sampler_type = sampler_type
        self.original_error = original_error


class OTelContextError(OTelError):
    """Raised when context propagation fails.

    This includes invalid context format, propagation failures,
    and context injection/extraction errors.
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        propagator: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OTelContextError.

        Args:
            message: Human-readable error message.
            operation: The operation that failed (inject/extract).
            propagator: The propagator being used.
            original_error: The underlying exception that caused this error.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.operation = operation
        self.propagator = propagator
        self.original_error = original_error


class OTelNotInstalledError(OTelError):
    """Raised when OpenTelemetry SDK is not installed.

    This exception is raised when attempting to use OpenTelemetry
    features without the optional opentelemetry dependencies installed.
    """

    def __init__(
        self,
        feature: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OTelNotInstalledError.

        Args:
            feature: The feature that requires OpenTelemetry.
            details: Optional dictionary with additional error context.
        """
        message = (
            "OpenTelemetry SDK is not installed. "
            "Install with: pip install truthound-orchestration[opentelemetry]"
        )
        if feature:
            message = f"Feature '{feature}' requires OpenTelemetry. {message}"
        super().__init__(message, details)
        self.feature = feature
