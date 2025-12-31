"""Data Quality Hook for Apache Airflow.

This module provides hooks for data loading and connection management
in data quality operations. The hook abstracts data source access,
supporting files (S3, GCS, local), databases, and other sources.

Example:
    >>> from truthound_airflow.hooks import DataQualityHook
    >>>
    >>> hook = DataQualityHook(connection_id="my_s3")
    >>> data = hook.load_data("s3://bucket/data.parquet")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from airflow.hooks.base import BaseHook

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Data Loader Protocol
# =============================================================================


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for data loading implementations.

    Implement this protocol to add support for custom data sources.

    Example:
        >>> class CustomLoader:
        ...     def load(self, path: str, **kwargs) -> pl.DataFrame:
        ...         return custom_load(path)
        ...
        ...     def supports(self, path: str) -> bool:
        ...         return path.startswith("custom://")
    """

    def load(self, path: str, **kwargs: Any) -> pl.DataFrame:
        """Load data from path.

        Args:
            path: Data path.
            **kwargs: Additional options.

        Returns:
            pl.DataFrame: Loaded data.
        """
        ...

    def supports(self, path: str) -> bool:
        """Check if this loader supports the given path.

        Args:
            path: Data path.

        Returns:
            bool: True if supported.
        """
        ...


@runtime_checkable
class DataWriter(Protocol):
    """Protocol for data writing implementations."""

    def write(self, data: Any, path: str, **kwargs: Any) -> None:
        """Write data to path.

        Args:
            data: Data to write.
            path: Destination path.
            **kwargs: Additional options.
        """
        ...

    def supports(self, path: str) -> bool:
        """Check if this writer supports the given path."""
        ...


# =============================================================================
# Connection Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConnectionConfig:
    """Configuration for data source connection.

    Attributes:
        conn_type: Connection type (s3, gcs, postgres, bigquery, filesystem).
        host: Host or bucket name.
        port: Port number (for databases).
        schema: Database schema/catalog.
        login: Username or access key.
        password: Password or secret key.
        extra: Additional connection-specific options.

    Example:
        >>> config = ConnectionConfig(
        ...     conn_type="s3",
        ...     host="my-bucket",
        ...     extra={"region": "us-east-1"},
        ... )
    """

    conn_type: str = "filesystem"
    host: str | None = None
    port: int | None = None
    schema: str | None = None
    login: str | None = None
    password: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_airflow_connection(cls, conn: Any) -> ConnectionConfig:
        """Create from Airflow Connection object.

        Args:
            conn: Airflow Connection instance.

        Returns:
            ConnectionConfig: Parsed configuration.
        """
        extra = conn.extra_dejson if hasattr(conn, "extra_dejson") else {}

        return cls(
            conn_type=extra.get("conn_type", "filesystem"),
            host=conn.host,
            port=conn.port,
            schema=conn.schema,
            login=conn.login,
            password=conn.password,
            extra=extra,
        )


# =============================================================================
# Hook Implementation
# =============================================================================


class DataQualityHook(BaseHook):
    """Hook for data loading and connection management.

    This hook provides unified data access across different sources:
    - File systems (local, S3, GCS)
    - Databases (PostgreSQL, BigQuery)
    - Custom data sources

    The hook uses Airflow Connections for credential management.

    Parameters
    ----------
    connection_id : str
        Airflow Connection ID. Default: "truthound_default"

    Attributes
    ----------
    conn_type : str
        Connection type name.
    hook_name : str
        Display name for the hook.

    Examples
    --------
    Load from S3:

    >>> hook = DataQualityHook(connection_id="my_s3")
    >>> data = hook.load_data("s3://bucket/data.parquet")

    Load from PostgreSQL:

    >>> hook = DataQualityHook(connection_id="my_postgres")
    >>> data = hook.query("SELECT * FROM users WHERE active = true")

    Save schema to file:

    >>> hook.save_json(schema_dict, "s3://bucket/schemas/v1.json")
    """

    conn_type = "truthound"
    conn_name_attr = "connection_id"
    hook_name = "Data Quality"

    def __init__(
        self,
        connection_id: str = "truthound_default",
    ) -> None:
        """Initialize hook.

        Args:
            connection_id: Airflow Connection ID.
        """
        super().__init__()
        self.connection_id = connection_id
        self._config: ConnectionConfig | None = None
        self._loaders: list[DataLoader] = []

    @property
    def config(self) -> ConnectionConfig:
        """Get connection configuration.

        Returns:
            ConnectionConfig: Parsed connection config.

        Note:
            Lazily loads and caches the configuration.
        """
        if self._config is None:
            try:
                conn = self.get_connection(self.connection_id)
                self._config = ConnectionConfig.from_airflow_connection(conn)
            except Exception:
                # Default to filesystem if connection not found
                self._config = ConnectionConfig()
        return self._config

    def get_conn(self) -> ConnectionConfig:
        """Get connection configuration (Airflow hook interface).

        Returns:
            ConnectionConfig: Connection configuration.
        """
        return self.config

    def load_data(
        self,
        path: str,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Load data from path.

        Automatically detects file format and handles remote paths.

        Args:
            path: Data path (local, s3://, gs://, etc.).
            **kwargs: Additional loading options.

        Returns:
            pl.DataFrame: Loaded data.

        Raises:
            FileNotFoundError: If path doesn't exist.
            ValueError: If format is not supported.
        """
        import polars as pl

        self.log.info(f"Loading data from: {path}")

        # Try custom loaders first
        for loader in self._loaders:
            if loader.supports(path):
                return loader.load(path, **kwargs)

        # Handle different path types
        if path.startswith("s3://"):
            return self._load_s3(path, **kwargs)
        elif path.startswith("gs://"):
            return self._load_gcs(path, **kwargs)
        else:
            return self._load_local(path, **kwargs)

    def _load_s3(self, path: str, **kwargs: Any) -> pl.DataFrame:
        """Load data from S3.

        Args:
            path: S3 path (s3://bucket/key).
            **kwargs: Additional options.

        Returns:
            pl.DataFrame: Loaded data.
        """
        import polars as pl

        # Build storage options from config
        storage_options = {}
        if self.config.login:
            storage_options["aws_access_key_id"] = self.config.login
        if self.config.password:
            storage_options["aws_secret_access_key"] = self.config.password
        if self.config.extra.get("region"):
            storage_options["aws_region"] = self.config.extra["region"]
        if self.config.extra.get("endpoint_url"):
            storage_options["aws_endpoint_url"] = self.config.extra["endpoint_url"]

        # Merge with any provided options
        storage_options.update(kwargs.pop("storage_options", {}))

        # Detect format and load
        if path.endswith(".parquet"):
            return pl.read_parquet(path, storage_options=storage_options, **kwargs)
        elif path.endswith(".csv"):
            return pl.read_csv(path, storage_options=storage_options, **kwargs)
        elif path.endswith(".json") or path.endswith(".jsonl"):
            return pl.read_ndjson(path, storage_options=storage_options, **kwargs)
        else:
            # Default to parquet
            return pl.read_parquet(path, storage_options=storage_options, **kwargs)

    def _load_gcs(self, path: str, **kwargs: Any) -> pl.DataFrame:
        """Load data from Google Cloud Storage.

        Args:
            path: GCS path (gs://bucket/key).
            **kwargs: Additional options.

        Returns:
            pl.DataFrame: Loaded data.
        """
        import polars as pl

        # Build storage options
        storage_options = {}
        if self.config.extra.get("credentials_path"):
            storage_options["service_account_path"] = self.config.extra["credentials_path"]

        storage_options.update(kwargs.pop("storage_options", {}))

        if path.endswith(".parquet"):
            return pl.read_parquet(path, storage_options=storage_options, **kwargs)
        elif path.endswith(".csv"):
            return pl.read_csv(path, storage_options=storage_options, **kwargs)
        else:
            return pl.read_parquet(path, storage_options=storage_options, **kwargs)

    def _load_local(self, path: str, **kwargs: Any) -> pl.DataFrame:
        """Load data from local filesystem.

        Args:
            path: Local file path.
            **kwargs: Additional options.

        Returns:
            pl.DataFrame: Loaded data.
        """
        import polars as pl

        if path.endswith(".parquet"):
            return pl.read_parquet(path, **kwargs)
        elif path.endswith(".csv"):
            return pl.read_csv(path, **kwargs)
        elif path.endswith(".json"):
            return pl.read_json(path, **kwargs)
        elif path.endswith(".jsonl") or path.endswith(".ndjson"):
            return pl.read_ndjson(path, **kwargs)
        else:
            # Try parquet by default
            return pl.read_parquet(path, **kwargs)

    def query(
        self,
        sql: str,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Execute SQL query and return results.

        Args:
            sql: SQL query string.
            **kwargs: Additional query options.

        Returns:
            pl.DataFrame: Query results.

        Raises:
            ValueError: If connection type doesn't support SQL.
        """
        import polars as pl

        self.log.info(f"Executing query: {sql[:100]}...")

        conn_type = self.config.conn_type

        if conn_type == "postgres":
            return self._query_postgres(sql, **kwargs)
        elif conn_type == "bigquery":
            return self._query_bigquery(sql, **kwargs)
        elif conn_type == "mysql":
            return self._query_mysql(sql, **kwargs)
        else:
            msg = f"SQL queries not supported for connection type: {conn_type}"
            raise ValueError(msg)

    def _query_postgres(self, sql: str, **kwargs: Any) -> pl.DataFrame:
        """Execute PostgreSQL query.

        Args:
            sql: SQL query.
            **kwargs: Additional options.

        Returns:
            pl.DataFrame: Query results.
        """
        import polars as pl

        # Build connection URI
        host = self.config.host or "localhost"
        port = self.config.port or 5432
        database = self.config.schema or "postgres"
        user = self.config.login or "postgres"
        password = self.config.password or ""

        uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        return pl.read_database(sql, uri, **kwargs)

    def _query_bigquery(self, sql: str, **kwargs: Any) -> pl.DataFrame:
        """Execute BigQuery query.

        Args:
            sql: SQL query.
            **kwargs: Additional options.

        Returns:
            pl.DataFrame: Query results.
        """
        import polars as pl
        from google.cloud import bigquery

        project = self.config.extra.get("project")
        credentials_path = self.config.extra.get("credentials_path")

        if credentials_path:
            client = bigquery.Client.from_service_account_json(
                credentials_path, project=project
            )
        else:
            client = bigquery.Client(project=project)

        query_job = client.query(sql)
        result = query_job.result()

        # Convert to Polars
        return pl.from_arrow(result.to_arrow())

    def _query_mysql(self, sql: str, **kwargs: Any) -> pl.DataFrame:
        """Execute MySQL query.

        Args:
            sql: SQL query.
            **kwargs: Additional options.

        Returns:
            pl.DataFrame: Query results.
        """
        import polars as pl

        host = self.config.host or "localhost"
        port = self.config.port or 3306
        database = self.config.schema or ""
        user = self.config.login or "root"
        password = self.config.password or ""

        uri = f"mysql://{user}:{password}@{host}:{port}/{database}"

        return pl.read_database(sql, uri, **kwargs)

    def save_json(
        self,
        data: dict[str, Any],
        path: str,
        **kwargs: Any,
    ) -> None:
        """Save dictionary as JSON to path.

        Args:
            data: Dictionary to save.
            path: Destination path.
            **kwargs: Additional options.
        """
        self.log.info(f"Saving JSON to: {path}")

        json_str = json.dumps(data, indent=2, default=str)

        if path.startswith("s3://"):
            self._write_s3(json_str, path)
        elif path.startswith("gs://"):
            self._write_gcs(json_str, path)
        else:
            self._write_local(json_str, path)

    def _write_s3(self, content: str, path: str) -> None:
        """Write content to S3.

        Args:
            content: String content to write.
            path: S3 path.
        """
        import boto3

        # Parse S3 path
        bucket, key = path[5:].split("/", 1)

        # Create client
        client_kwargs = {}
        if self.config.extra.get("region"):
            client_kwargs["region_name"] = self.config.extra["region"]
        if self.config.login and self.config.password:
            client_kwargs["aws_access_key_id"] = self.config.login
            client_kwargs["aws_secret_access_key"] = self.config.password

        s3 = boto3.client("s3", **client_kwargs)
        s3.put_object(Bucket=bucket, Key=key, Body=content.encode())

    def _write_gcs(self, content: str, path: str) -> None:
        """Write content to GCS.

        Args:
            content: String content to write.
            path: GCS path.
        """
        from google.cloud import storage

        # Parse GCS path
        bucket_name, blob_name = path[5:].split("/", 1)

        credentials_path = self.config.extra.get("credentials_path")
        if credentials_path:
            client = storage.Client.from_service_account_json(credentials_path)
        else:
            client = storage.Client()

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content)

    def _write_local(self, content: str, path: str) -> None:
        """Write content to local file.

        Args:
            content: String content.
            path: Local path.
        """
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    def register_loader(self, loader: DataLoader) -> None:
        """Register a custom data loader.

        Args:
            loader: DataLoader implementation.

        Example:
            >>> class HDFSLoader:
            ...     def load(self, path, **kwargs):
            ...         return hdfs_load(path)
            ...     def supports(self, path):
            ...         return path.startswith("hdfs://")
            >>>
            >>> hook.register_loader(HDFSLoader())
        """
        self._loaders.append(loader)

    def test_connection(self) -> tuple[bool, str]:
        """Test the connection.

        Returns:
            tuple[bool, str]: (success, message)
        """
        try:
            _ = self.config
            return True, "Connection configuration loaded successfully"
        except Exception as e:
            return False, str(e)


# Alias for backwards compatibility
TruthoundHook = DataQualityHook
