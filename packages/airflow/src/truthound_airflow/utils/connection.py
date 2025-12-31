"""Connection Utilities for Data Quality Operations.

This module provides utilities for working with Airflow connections,
including parsing, building, and managing connection configurations.

Example:
    >>> from truthound_airflow.utils import get_connection_config
    >>>
    >>> config = get_connection_config("my_s3_connection")
    >>> print(config["conn_type"])  # "s3"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

if TYPE_CHECKING:
    from airflow.models import Connection


# =============================================================================
# Connection Helper
# =============================================================================


class ConnectionHelper:
    """Helper class for working with Airflow connections.

    Provides utilities for extracting configuration from connections,
    building connection URIs, and handling sensitive data.

    Example:
        >>> helper = ConnectionHelper()
        >>> config = helper.get_config(connection)
        >>> uri = helper.build_uri(config)
    """

    # Sensitive keys that should be masked in logs
    SENSITIVE_KEYS = frozenset({
        "password",
        "secret",
        "secret_key",
        "api_key",
        "token",
        "access_key",
        "private_key",
        "credentials",
        "auth",
    })

    def get_config(self, connection: Connection) -> dict[str, Any]:
        """Extract configuration from Airflow Connection.

        Args:
            connection: Airflow Connection object.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        extra = {}
        if hasattr(connection, "extra_dejson"):
            extra = connection.extra_dejson or {}

        return {
            "conn_id": connection.conn_id,
            "conn_type": extra.get("conn_type", connection.conn_type or "filesystem"),
            "host": connection.host,
            "port": connection.port,
            "schema": connection.schema,
            "login": connection.login,
            "password": connection.password,
            "extra": extra,
        }

    def build_uri(
        self,
        config: dict[str, Any],
        include_password: bool = False,
    ) -> str:
        """Build connection URI from configuration.

        Args:
            config: Connection configuration.
            include_password: Whether to include password.

        Returns:
            str: Connection URI.
        """
        conn_type = config.get("conn_type", "filesystem")
        scheme = self._get_scheme_for_type(conn_type)

        host = config.get("host", "")
        port = config.get("port")
        schema = config.get("schema", "")
        login = config.get("login", "")
        password = config.get("password", "")

        # Build auth part
        auth = ""
        if login:
            auth = quote(login, safe="")
            if password and include_password:
                auth += f":{quote(password, safe='')}"
            auth += "@"

        # Build host:port
        host_part = host
        if port:
            host_part = f"{host}:{port}"

        # Build path
        path = f"/{schema}" if schema else ""

        # Build query string from extra
        extra = config.get("extra", {})
        query = urlencode(extra) if extra else ""

        uri = f"{scheme}://{auth}{host_part}{path}"
        if query:
            uri += f"?{query}"

        return uri

    def _get_scheme_for_type(self, conn_type: str) -> str:
        """Get URI scheme for connection type."""
        schemes = {
            "postgres": "postgresql",
            "postgresql": "postgresql",
            "mysql": "mysql",
            "s3": "s3",
            "gcs": "gs",
            "bigquery": "bigquery",
            "filesystem": "file",
            "truthound": "truthound",
        }
        return schemes.get(conn_type, conn_type)

    def mask_sensitive(self, config: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive values in configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            dict[str, Any]: Configuration with masked values.
        """
        return mask_sensitive_values(config, self.SENSITIVE_KEYS)

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate connection configuration.

        Args:
            config: Configuration to validate.

        Returns:
            list[str]: List of validation errors.
        """
        errors = []
        conn_type = config.get("conn_type", "")

        if conn_type == "s3":
            if not config.get("host"):
                errors.append("S3 connection requires 'host' (bucket name)")

        elif conn_type == "gcs":
            if not config.get("host"):
                errors.append("GCS connection requires 'host' (bucket name)")

        elif conn_type in ("postgres", "postgresql", "mysql"):
            if not config.get("host"):
                errors.append(f"{conn_type} connection requires 'host'")
            if not config.get("schema"):
                errors.append(f"{conn_type} connection requires 'schema' (database)")

        elif conn_type == "bigquery":
            extra = config.get("extra", {})
            if not extra.get("project"):
                errors.append("BigQuery connection requires 'project' in extra")

        return errors


# =============================================================================
# Convenience Functions
# =============================================================================

_default_helper = ConnectionHelper()


def get_connection_config(connection_id: str) -> dict[str, Any]:
    """Get configuration from Airflow Connection by ID.

    Args:
        connection_id: Airflow Connection ID.

    Returns:
        dict[str, Any]: Connection configuration.

    Raises:
        AirflowNotFoundException: If connection not found.
    """
    from airflow.hooks.base import BaseHook

    connection = BaseHook.get_connection(connection_id)
    return _default_helper.get_config(connection)


def parse_connection_uri(uri: str) -> dict[str, Any]:
    """Parse connection URI into configuration dictionary.

    Args:
        uri: Connection URI string.

    Returns:
        dict[str, Any]: Parsed configuration.

    Example:
        >>> config = parse_connection_uri(
        ...     "postgresql://user:pass@host:5432/mydb?sslmode=require"
        ... )
        >>> print(config["host"])  # "host"
        >>> print(config["port"])  # 5432
    """
    parsed = urlparse(uri)

    # Parse query string
    extra = {}
    if parsed.query:
        for key, values in parse_qs(parsed.query).items():
            extra[key] = values[0] if len(values) == 1 else values

    # Map scheme to conn_type
    scheme_to_type = {
        "postgresql": "postgres",
        "postgres": "postgres",
        "mysql": "mysql",
        "s3": "s3",
        "gs": "gcs",
        "bigquery": "bigquery",
        "file": "filesystem",
        "truthound": "truthound",
    }
    conn_type = scheme_to_type.get(parsed.scheme, parsed.scheme)

    return {
        "conn_type": conn_type,
        "host": parsed.hostname or "",
        "port": parsed.port,
        "schema": parsed.path.lstrip("/") if parsed.path else "",
        "login": unquote(parsed.username) if parsed.username else "",
        "password": unquote(parsed.password) if parsed.password else "",
        "extra": extra,
    }


def build_connection_uri(
    config: dict[str, Any],
    include_password: bool = False,
) -> str:
    """Build connection URI from configuration.

    Args:
        config: Connection configuration.
        include_password: Whether to include password.

    Returns:
        str: Connection URI.
    """
    return _default_helper.build_uri(config, include_password)


def mask_sensitive_values(
    data: dict[str, Any],
    sensitive_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Mask sensitive values in a dictionary.

    Args:
        data: Dictionary to mask.
        sensitive_keys: Keys to mask.

    Returns:
        dict[str, Any]: Dictionary with masked values.
    """
    if sensitive_keys is None:
        sensitive_keys = ConnectionHelper.SENSITIVE_KEYS

    result = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Check if key matches any sensitive pattern
        is_sensitive = any(
            sensitive in key_lower for sensitive in sensitive_keys
        )

        if is_sensitive and value:
            result[key] = "***MASKED***"
        elif isinstance(value, dict):
            result[key] = mask_sensitive_values(value, sensitive_keys)
        else:
            result[key] = value

    return result


def validate_connection_config(config: dict[str, Any]) -> list[str]:
    """Validate connection configuration.

    Args:
        config: Configuration to validate.

    Returns:
        list[str]: List of validation errors.
    """
    return _default_helper.validate_config(config)


# =============================================================================
# Connection Type Helpers
# =============================================================================


def get_s3_config(
    bucket: str,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
    endpoint_url: str | None = None,
) -> dict[str, Any]:
    """Build S3 connection configuration.

    Args:
        bucket: S3 bucket name.
        access_key: AWS access key.
        secret_key: AWS secret key.
        region: AWS region.
        endpoint_url: Custom endpoint URL.

    Returns:
        dict[str, Any]: S3 connection configuration.
    """
    extra: dict[str, Any] = {
        "conn_type": "s3",
        "region": region,
    }
    if endpoint_url:
        extra["endpoint_url"] = endpoint_url

    return {
        "conn_type": "s3",
        "host": bucket,
        "login": access_key,
        "password": secret_key,
        "extra": extra,
    }


def get_gcs_config(
    bucket: str,
    credentials_path: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Build GCS connection configuration.

    Args:
        bucket: GCS bucket name.
        credentials_path: Path to credentials file.
        project: GCP project ID.

    Returns:
        dict[str, Any]: GCS connection configuration.
    """
    extra: dict[str, Any] = {"conn_type": "gcs"}
    if credentials_path:
        extra["credentials_path"] = credentials_path
    if project:
        extra["project"] = project

    return {
        "conn_type": "gcs",
        "host": bucket,
        "extra": extra,
    }


def get_postgres_config(
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5432,
    sslmode: str | None = None,
) -> dict[str, Any]:
    """Build PostgreSQL connection configuration.

    Args:
        host: Database host.
        database: Database name.
        user: Database user.
        password: Database password.
        port: Database port.
        sslmode: SSL mode.

    Returns:
        dict[str, Any]: PostgreSQL connection configuration.
    """
    extra: dict[str, Any] = {"conn_type": "postgres"}
    if sslmode:
        extra["sslmode"] = sslmode

    return {
        "conn_type": "postgres",
        "host": host,
        "port": port,
        "schema": database,
        "login": user,
        "password": password,
        "extra": extra,
    }


def get_bigquery_config(
    project: str,
    credentials_path: str | None = None,
    location: str = "US",
) -> dict[str, Any]:
    """Build BigQuery connection configuration.

    Args:
        project: GCP project ID.
        credentials_path: Path to credentials file.
        location: BigQuery location.

    Returns:
        dict[str, Any]: BigQuery connection configuration.
    """
    extra: dict[str, Any] = {
        "conn_type": "bigquery",
        "project": project,
        "location": location,
    }
    if credentials_path:
        extra["credentials_path"] = credentials_path

    return {
        "conn_type": "bigquery",
        "extra": extra,
    }
