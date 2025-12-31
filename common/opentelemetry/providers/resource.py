"""OpenTelemetry Resource provider for truthound-orchestration.

This module provides utilities for creating OpenTelemetry Resource
objects that identify the service producing telemetry.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING, Any

from common.opentelemetry.config import ResourceConfig
from common.opentelemetry.exceptions import OTelNotInstalledError, OTelProviderError

if TYPE_CHECKING:
    from opentelemetry.sdk.resources import Resource

__all__ = [
    "create_resource",
    "get_default_resource",
    "merge_resources",
    "ResourceFactory",
]


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.sdk.resources  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="resource provider") from e


def create_resource(config: ResourceConfig | None = None) -> Resource:
    """Create an OpenTelemetry Resource from configuration.

    Args:
        config: Resource configuration. If None, uses defaults.

    Returns:
        OpenTelemetry Resource instance.

    Example:
        config = ResourceConfig(
            service_name="my-service",
            deployment_environment="production",
        )
        resource = create_resource(config)
    """
    _check_otel_installed()

    from opentelemetry.sdk.resources import Resource

    config = config or ResourceConfig()

    # Build attributes from config
    attributes = config.to_attributes()

    # Add auto-detected attributes
    attributes.update(_get_auto_detected_attributes())

    return Resource.create(attributes)


def get_default_resource() -> Resource:
    """Get a default Resource with auto-detected attributes.

    Returns:
        OpenTelemetry Resource with default attributes.
    """
    return create_resource(ResourceConfig())


def merge_resources(*resources: Resource) -> Resource:
    """Merge multiple resources into one.

    Later resources take precedence over earlier ones for conflicting keys.

    Args:
        resources: Resources to merge.

    Returns:
        Merged Resource.
    """
    _check_otel_installed()

    from opentelemetry.sdk.resources import Resource

    if not resources:
        return Resource.create({})

    result = resources[0]
    for resource in resources[1:]:
        result = result.merge(resource)

    return result


def _get_auto_detected_attributes() -> dict[str, Any]:
    """Get auto-detected resource attributes.

    Returns:
        Dictionary of auto-detected attributes.
    """
    attributes: dict[str, Any] = {
        # Host attributes
        "host.name": platform.node(),
        "host.arch": platform.machine(),
        # OS attributes
        "os.type": platform.system().lower(),
        "os.description": platform.platform(),
        # Process attributes
        "process.runtime.name": platform.python_implementation(),
        "process.runtime.version": platform.python_version(),
        "process.pid": os.getpid(),
    }

    # Add process executable info if available
    if sys.executable:
        attributes["process.executable.name"] = os.path.basename(sys.executable)
        attributes["process.executable.path"] = sys.executable

    # Add telemetry SDK info
    try:
        import opentelemetry.sdk
        attributes["telemetry.sdk.name"] = "opentelemetry"
        attributes["telemetry.sdk.language"] = "python"
        attributes["telemetry.sdk.version"] = getattr(opentelemetry.sdk, "__version__", "unknown")
    except Exception:
        pass

    # Add container info if available
    container_id = _get_container_id()
    if container_id:
        attributes["container.id"] = container_id

    return attributes


def _get_container_id() -> str | None:
    """Attempt to get container ID from cgroup.

    Returns:
        Container ID or None if not in a container.
    """
    try:
        # Try to read from cgroup file (Linux containers)
        cgroup_path = "/proc/self/cgroup"
        if os.path.exists(cgroup_path):
            with open(cgroup_path) as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) >= 3:
                        path = parts[2]
                        # Docker container ID is 64 hex characters
                        if "/docker/" in path:
                            container_id = path.split("/docker/")[-1]
                            if len(container_id) >= 64:
                                return container_id[:64]
                        # Kubernetes pod ID
                        if "/kubepods/" in path:
                            # Extract container ID from path
                            parts = path.split("/")
                            for part in reversed(parts):
                                if len(part) >= 64 and all(c in "0123456789abcdef" for c in part[:64]):
                                    return part[:64]
    except Exception:
        pass

    # Try environment variables
    for env_var in ("CONTAINER_ID", "HOSTNAME"):
        value = os.environ.get(env_var)
        if value and len(value) >= 12:
            return value

    return None


class ResourceFactory:
    """Factory for creating OpenTelemetry Resources with customization.

    This class provides a builder-style interface for creating
    resources with various attributes.

    Example:
        factory = ResourceFactory()
        resource = (
            factory
            .with_service("my-service", "1.0.0")
            .with_environment("production")
            .with_attributes(custom_key="custom_value")
            .create()
        )
    """

    def __init__(self) -> None:
        """Initialize ResourceFactory."""
        _check_otel_installed()
        self._config = ResourceConfig()
        self._extra_attributes: dict[str, Any] = {}
        self._include_auto_detect: bool = True

    def with_service(
        self,
        name: str,
        version: str | None = None,
        namespace: str | None = None,
    ) -> "ResourceFactory":
        """Set service information.

        Args:
            name: Service name.
            version: Service version.
            namespace: Service namespace.

        Returns:
            Self for chaining.
        """
        self._config = self._config.with_service_name(name)
        if version:
            self._config = self._config.with_service_version(version)
        if namespace:
            self._config = ResourceConfig(
                service_name=self._config.service_name,
                service_version=self._config.service_version,
                service_namespace=namespace,
                deployment_environment=self._config.deployment_environment,
                service_instance_id=self._config.service_instance_id,
                additional_attributes=self._config.additional_attributes,
            )
        return self

    def with_environment(self, environment: str) -> "ResourceFactory":
        """Set deployment environment.

        Args:
            environment: Deployment environment (development, staging, production).

        Returns:
            Self for chaining.
        """
        self._config = self._config.with_environment(environment)
        return self

    def with_instance_id(self, instance_id: str) -> "ResourceFactory":
        """Set service instance ID.

        Args:
            instance_id: Unique instance identifier.

        Returns:
            Self for chaining.
        """
        self._config = self._config.with_instance_id(instance_id)
        return self

    def with_attributes(self, **attributes: Any) -> "ResourceFactory":
        """Add custom attributes.

        Args:
            **attributes: Key-value pairs to add.

        Returns:
            Self for chaining.
        """
        self._extra_attributes.update(attributes)
        return self

    def without_auto_detect(self) -> "ResourceFactory":
        """Disable auto-detection of resource attributes.

        Returns:
            Self for chaining.
        """
        self._include_auto_detect = False
        return self

    def create(self) -> Resource:
        """Create the Resource.

        Returns:
            Configured OpenTelemetry Resource.
        """
        from opentelemetry.sdk.resources import Resource

        # Start with config attributes
        attributes = self._config.to_attributes()

        # Add extra attributes
        attributes.update(self._extra_attributes)

        # Add auto-detected attributes
        if self._include_auto_detect:
            auto_attrs = _get_auto_detected_attributes()
            # Don't override explicit attributes
            for key, value in auto_attrs.items():
                if key not in attributes:
                    attributes[key] = value

        return Resource.create(attributes)
