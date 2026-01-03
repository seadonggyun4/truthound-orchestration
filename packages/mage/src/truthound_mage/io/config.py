"""I/O configuration for Mage data quality operations.

This module provides integration with Mage's io_config.yaml for
configuring data sources and sinks.

Example io_config.yaml structure:
    default:
      TRUTHOUND_ENGINE: truthound
      TRUTHOUND_TIMEOUT: 300

    data_sources:
      warehouse:
        type: postgres
        host: localhost
        port: 5432
        database: analytics

Example:
    >>> from truthound_mage.io import load_io_config, IOConfig
    >>>
    >>> config = load_io_config("io_config.yaml")
    >>> source_config = config.get_source("warehouse")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Enums
# =============================================================================


class DataSourceType(str, Enum):
    """Supported data source types."""

    FILE = "file"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    API = "api"
    CUSTOM = "custom"


class DataFormat(str, Enum):
    """Supported data formats."""

    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    AVRO = "avro"
    ORC = "orc"


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class DataSourceConfig:
    """Configuration for a data source.

    Attributes:
        name: Source name identifier.
        source_type: Type of data source (postgres, s3, etc.).
        connection_string: Full connection string if applicable.
        host: Database host.
        port: Database port.
        database: Database name.
        schema: Database schema.
        username: Database username.
        password: Database password (or env var reference).
        bucket: Cloud storage bucket.
        path: File path or prefix.
        format: Data format for file sources.
        options: Additional source-specific options.

    Example:
        >>> config = DataSourceConfig(
        ...     name="warehouse",
        ...     source_type=DataSourceType.POSTGRES,
        ...     host="localhost",
        ...     database="analytics",
        ... )
    """

    name: str
    source_type: DataSourceType = DataSourceType.FILE
    connection_string: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    schema: str | None = None
    username: str | None = None
    password: str | None = None
    bucket: str | None = None
    path: str | None = None
    format: DataFormat = DataFormat.PARQUET
    options: dict[str, Any] = field(default_factory=dict)

    def get_connection_string(self) -> str:
        """Build connection string from components.

        Returns:
            Full connection string.

        Raises:
            ValueError: If required components are missing.
        """
        if self.connection_string:
            return self._resolve_env_vars(self.connection_string)

        if self.source_type == DataSourceType.POSTGRES:
            return self._build_postgres_url()
        elif self.source_type == DataSourceType.MYSQL:
            return self._build_mysql_url()
        elif self.source_type == DataSourceType.S3:
            return self._build_s3_url()
        elif self.source_type == DataSourceType.GCS:
            return self._build_gcs_url()
        else:
            msg = f"Cannot build connection string for source type: {self.source_type}"
            raise ValueError(msg)

    def _build_postgres_url(self) -> str:
        """Build PostgreSQL connection URL."""
        if not all([self.host, self.database]):
            msg = "host and database required for PostgreSQL"
            raise ValueError(msg)

        username = self._resolve_env_vars(self.username or "")
        password = self._resolve_env_vars(self.password or "")
        port = self.port or 5432

        auth = f"{username}:{password}@" if username else ""
        return f"postgresql://{auth}{self.host}:{port}/{self.database}"

    def _build_mysql_url(self) -> str:
        """Build MySQL connection URL."""
        if not all([self.host, self.database]):
            msg = "host and database required for MySQL"
            raise ValueError(msg)

        username = self._resolve_env_vars(self.username or "")
        password = self._resolve_env_vars(self.password or "")
        port = self.port or 3306

        auth = f"{username}:{password}@" if username else ""
        return f"mysql://{auth}{self.host}:{port}/{self.database}"

    def _build_s3_url(self) -> str:
        """Build S3 URL."""
        if not self.bucket:
            msg = "bucket required for S3"
            raise ValueError(msg)

        path = self.path or ""
        return f"s3://{self.bucket}/{path.lstrip('/')}"

    def _build_gcs_url(self) -> str:
        """Build GCS URL."""
        if not self.bucket:
            msg = "bucket required for GCS"
            raise ValueError(msg)

        path = self.path or ""
        return f"gs://{self.bucket}/{path.lstrip('/')}"

    @staticmethod
    def _resolve_env_vars(value: str) -> str:
        """Resolve environment variable references in value.

        Supports ${VAR_NAME} and $VAR_NAME syntax.

        Args:
            value: String potentially containing env var references.

        Returns:
            String with env vars resolved.
        """
        if not value:
            return value

        import re

        # Match ${VAR_NAME} pattern
        pattern = r"\$\{([^}]+)\}"
        result = re.sub(pattern, lambda m: os.environ.get(m.group(1), m.group(0)), value)

        # Match $VAR_NAME pattern (word boundary)
        pattern = r"\$([A-Za-z_][A-Za-z0-9_]*)"
        result = re.sub(pattern, lambda m: os.environ.get(m.group(1), m.group(0)), result)

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.source_type.value,
            "connection_string": self.connection_string,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "schema": self.schema,
            "username": self.username,
            "password": self.password,
            "bucket": self.bucket,
            "path": self.path,
            "format": self.format.value,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> DataSourceConfig:
        """Create from dictionary."""
        source_type = data.get("type", "file")
        if isinstance(source_type, str):
            source_type = DataSourceType(source_type.lower())

        data_format = data.get("format", "parquet")
        if isinstance(data_format, str):
            data_format = DataFormat(data_format.lower())

        return cls(
            name=name,
            source_type=source_type,
            connection_string=data.get("connection_string"),
            host=data.get("host"),
            port=data.get("port"),
            database=data.get("database"),
            schema=data.get("schema"),
            username=data.get("username"),
            password=data.get("password"),
            bucket=data.get("bucket"),
            path=data.get("path"),
            format=data_format,
            options=data.get("options", {}),
        )


@dataclass(frozen=True, slots=True)
class DataSinkConfig:
    """Configuration for a data sink.

    Attributes:
        name: Sink name identifier.
        sink_type: Type of data sink.
        path: Output path or table name.
        format: Output format.
        partition_by: Partition columns.
        mode: Write mode (overwrite, append, etc.).
        options: Additional sink-specific options.

    Example:
        >>> config = DataSinkConfig(
        ...     name="quality_results",
        ...     sink_type=DataSourceType.S3,
        ...     path="s3://bucket/quality/",
        ...     format=DataFormat.PARQUET,
        ... )
    """

    name: str
    sink_type: DataSourceType = DataSourceType.FILE
    path: str | None = None
    format: DataFormat = DataFormat.PARQUET
    partition_by: tuple[str, ...] = field(default_factory=tuple)
    mode: str = "overwrite"
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.sink_type.value,
            "path": self.path,
            "format": self.format.value,
            "partition_by": list(self.partition_by),
            "mode": self.mode,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> DataSinkConfig:
        """Create from dictionary."""
        sink_type = data.get("type", "file")
        if isinstance(sink_type, str):
            sink_type = DataSourceType(sink_type.lower())

        data_format = data.get("format", "parquet")
        if isinstance(data_format, str):
            data_format = DataFormat(data_format.lower())

        partition_by = data.get("partition_by", [])
        if isinstance(partition_by, list):
            partition_by = tuple(partition_by)

        return cls(
            name=name,
            sink_type=sink_type,
            path=data.get("path"),
            format=data_format,
            partition_by=partition_by,
            mode=data.get("mode", "overwrite"),
            options=data.get("options", {}),
        )


@dataclass(frozen=True, slots=True)
class IOConfig:
    """Main I/O configuration for data quality operations.

    This class aggregates data source and sink configurations,
    as well as global settings for data quality operations.

    Attributes:
        sources: Dictionary of data source configurations.
        sinks: Dictionary of data sink configurations.
        engine_name: Default data quality engine to use.
        timeout_seconds: Default operation timeout.
        profile: Configuration profile name.
        variables: Global variables.

    Example:
        >>> config = IOConfig(
        ...     sources={"warehouse": warehouse_config},
        ...     engine_name="truthound",
        ... )
        >>> source = config.get_source("warehouse")
    """

    sources: dict[str, DataSourceConfig] = field(default_factory=dict)
    sinks: dict[str, DataSinkConfig] = field(default_factory=dict)
    engine_name: str | None = None
    timeout_seconds: int = 300
    profile: str = "default"
    variables: dict[str, Any] = field(default_factory=dict)

    def get_source(self, name: str) -> DataSourceConfig:
        """Get a data source configuration by name.

        Args:
            name: Source name.

        Returns:
            DataSourceConfig for the named source.

        Raises:
            KeyError: If source not found.
        """
        if name not in self.sources:
            msg = f"Data source '{name}' not found in configuration"
            raise KeyError(msg)
        return self.sources[name]

    def get_sink(self, name: str) -> DataSinkConfig:
        """Get a data sink configuration by name.

        Args:
            name: Sink name.

        Returns:
            DataSinkConfig for the named sink.

        Raises:
            KeyError: If sink not found.
        """
        if name not in self.sinks:
            msg = f"Data sink '{name}' not found in configuration"
            raise KeyError(msg)
        return self.sinks[name]

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable value.

        Args:
            key: Variable key.
            default: Default value if not found.

        Returns:
            Variable value or default.
        """
        return self.variables.get(key, default)

    def list_sources(self) -> list[str]:
        """List all source names."""
        return list(self.sources.keys())

    def list_sinks(self) -> list[str]:
        """List all sink names."""
        return list(self.sinks.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
            "sinks": {k: v.to_dict() for k, v in self.sinks.items()},
            "engine_name": self.engine_name,
            "timeout_seconds": self.timeout_seconds,
            "profile": self.profile,
            "variables": self.variables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IOConfig:
        """Create from dictionary."""
        sources = {}
        for name, source_data in data.get("sources", {}).items():
            sources[name] = DataSourceConfig.from_dict(name, source_data)

        sinks = {}
        for name, sink_data in data.get("sinks", {}).items():
            sinks[name] = DataSinkConfig.from_dict(name, sink_data)

        return cls(
            sources=sources,
            sinks=sinks,
            engine_name=data.get("engine_name") or data.get("TRUTHOUND_ENGINE"),
            timeout_seconds=data.get("timeout_seconds")
            or data.get("TRUTHOUND_TIMEOUT", 300),
            profile=data.get("profile", "default"),
            variables=data.get("variables", {}),
        )

    @classmethod
    def from_mage_config(cls, mage_config: dict[str, Any], profile: str = "default") -> IOConfig:
        """Create from Mage's io_config.yaml structure.

        Mage's io_config.yaml has a specific structure with profiles:
            default:
              KEY: value
            dev:
              KEY: dev_value

        Args:
            mage_config: Parsed Mage io_config.yaml content.
            profile: Profile name to use.

        Returns:
            IOConfig instance.
        """
        # Get profile data
        profile_data = mage_config.get(profile, mage_config.get("default", {}))

        # Extract data quality specific settings
        engine_name = profile_data.get("TRUTHOUND_ENGINE")
        timeout_seconds = profile_data.get("TRUTHOUND_TIMEOUT", 300)

        # Extract data sources (Mage uses flat keys like POSTGRES_HOST, etc.)
        sources = {}
        sinks = {}

        # Check for data_sources section (custom extension)
        if "data_sources" in profile_data:
            for name, source_data in profile_data["data_sources"].items():
                sources[name] = DataSourceConfig.from_dict(name, source_data)

        if "data_sinks" in profile_data:
            for name, sink_data in profile_data["data_sinks"].items():
                sinks[name] = DataSinkConfig.from_dict(name, sink_data)

        # Also check for standard Mage connection prefixes
        for source_type in ["POSTGRES", "MYSQL", "SNOWFLAKE", "BIGQUERY", "REDSHIFT"]:
            host_key = f"{source_type}_HOST"
            if host_key in profile_data:
                name = source_type.lower()
                sources[name] = DataSourceConfig(
                    name=name,
                    source_type=DataSourceType(source_type.lower())
                    if source_type.lower() in [e.value for e in DataSourceType]
                    else DataSourceType.CUSTOM,
                    host=profile_data.get(host_key),
                    port=profile_data.get(f"{source_type}_PORT"),
                    database=profile_data.get(f"{source_type}_DATABASE")
                    or profile_data.get(f"{source_type}_DB"),
                    schema=profile_data.get(f"{source_type}_SCHEMA"),
                    username=profile_data.get(f"{source_type}_USER")
                    or profile_data.get(f"{source_type}_USERNAME"),
                    password=profile_data.get(f"{source_type}_PASSWORD"),
                )

        # Check for cloud storage
        if "AWS_ACCESS_KEY_ID" in profile_data or "S3_BUCKET" in profile_data:
            sources["s3"] = DataSourceConfig(
                name="s3",
                source_type=DataSourceType.S3,
                bucket=profile_data.get("S3_BUCKET"),
                path=profile_data.get("S3_PATH", ""),
                options={
                    "aws_access_key_id": profile_data.get("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": profile_data.get("AWS_SECRET_ACCESS_KEY"),
                    "region": profile_data.get("AWS_REGION"),
                },
            )

        if "GOOGLE_SERVICE_ACC_KEY_FILEPATH" in profile_data or "GCS_BUCKET" in profile_data:
            sources["gcs"] = DataSourceConfig(
                name="gcs",
                source_type=DataSourceType.GCS,
                bucket=profile_data.get("GCS_BUCKET"),
                path=profile_data.get("GCS_PATH", ""),
                options={
                    "service_account_key_path": profile_data.get(
                        "GOOGLE_SERVICE_ACC_KEY_FILEPATH"
                    ),
                },
            )

        return cls(
            sources=sources,
            sinks=sinks,
            engine_name=engine_name,
            timeout_seconds=int(timeout_seconds),
            profile=profile,
            variables=profile_data,
        )


# =============================================================================
# Loading Functions
# =============================================================================


def load_io_config(
    path: str | Path | None = None,
    profile: str = "default",
) -> IOConfig:
    """Load I/O configuration from file.

    Args:
        path: Path to io_config.yaml. If None, searches default locations.
        profile: Configuration profile to use.

    Returns:
        IOConfig instance.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If config file is invalid.

    Example:
        >>> config = load_io_config("io_config.yaml", profile="production")
        >>> source = config.get_source("warehouse")
    """
    if path is None:
        # Search default Mage locations
        search_paths = [
            Path.cwd() / "io_config.yaml",
            Path.cwd() / "io_config.yml",
            Path.home() / ".mage_ai" / "io_config.yaml",
        ]

        for search_path in search_paths:
            if search_path.exists():
                path = search_path
                break
        else:
            msg = "io_config.yaml not found in default locations"
            raise FileNotFoundError(msg)

    path = Path(path)
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    return IOConfig.from_mage_config(raw_config, profile=profile)


# =============================================================================
# Preset Configurations
# =============================================================================


DEFAULT_IO_CONFIG = IOConfig()
"""Default I/O configuration with no sources configured."""
