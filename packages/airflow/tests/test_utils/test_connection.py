"""Tests for connection utilities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestConnectionHelper:
    """Tests for ConnectionHelper class."""

    def test_get_config_from_connection(self, mock_connection: MagicMock) -> None:
        """Test getting config from Airflow connection."""
        from truthound_airflow.utils.connection import ConnectionHelper

        helper = ConnectionHelper()

        config = helper.get_config(mock_connection)

        assert config["host"] == mock_connection.host
        assert config["login"] == mock_connection.login
        assert config["extra"]["conn_type"] == "s3"

    def test_build_uri_postgres(self) -> None:
        """Test building PostgreSQL URI."""
        from truthound_airflow.utils.connection import ConnectionHelper

        helper = ConnectionHelper()

        config = {
            "conn_type": "postgres",
            "host": "localhost",
            "port": 5432,
            "schema": "testdb",
            "login": "user",
            "password": "pass",
        }

        uri = helper.build_uri(config, include_password=True)

        assert "postgresql://" in uri
        assert "localhost" in uri
        assert "5432" in uri
        assert "testdb" in uri

    def test_build_uri_without_password(self) -> None:
        """Test building URI without password."""
        from truthound_airflow.utils.connection import ConnectionHelper

        helper = ConnectionHelper()

        config = {
            "conn_type": "postgres",
            "host": "localhost",
            "port": 5432,
            "schema": "testdb",
            "login": "user",
            "password": "secret",
        }

        uri = helper.build_uri(config, include_password=False)

        assert "secret" not in uri

    def test_mask_sensitive(self) -> None:
        """Test masking sensitive data."""
        from truthound_airflow.utils.connection import ConnectionHelper

        helper = ConnectionHelper()

        config = {
            "host": "localhost",
            "login": "access_key",
            "password": "secret_password",
            "api_key": "api_secret",
            "token": "auth_token",
        }

        masked = helper.mask_sensitive(config)

        assert masked["host"] == "localhost"
        assert masked["password"] == "***MASKED***"
        assert masked["api_key"] == "***MASKED***"
        assert masked["token"] == "***MASKED***"

    def test_does_not_modify_original(self) -> None:
        """Test that masking doesn't modify original."""
        from truthound_airflow.utils.connection import ConnectionHelper

        helper = ConnectionHelper()

        config = {"password": "secret"}

        helper.mask_sensitive(config)

        assert config["password"] == "secret"


class TestGetConnectionConfig:
    """Tests for get_connection_config function."""

    @patch("airflow.hooks.base.BaseHook.get_connection")
    def test_get_connection_config(
        self,
        mock_get_connection: MagicMock,
        mock_connection: MagicMock,
    ) -> None:
        """Test getting connection config by ID."""
        from truthound_airflow.utils.connection import get_connection_config

        mock_get_connection.return_value = mock_connection

        config = get_connection_config("test_conn")

        assert config is not None
        mock_get_connection.assert_called_once_with("test_conn")


class TestParseConnectionUri:
    """Tests for parse_connection_uri function."""

    def test_parse_postgres_uri(self) -> None:
        """Test parsing PostgreSQL URI."""
        from truthound_airflow.utils.connection import parse_connection_uri

        uri = "postgresql://user:pass@localhost:5432/mydb"

        config = parse_connection_uri(uri)

        assert config["conn_type"] == "postgres"
        assert config["host"] == "localhost"
        assert config["port"] == 5432
        assert config["schema"] == "mydb"

    def test_parse_s3_uri(self) -> None:
        """Test parsing S3 URI."""
        from truthound_airflow.utils.connection import parse_connection_uri

        uri = "s3://my-bucket/path/to/data"

        config = parse_connection_uri(uri)

        assert config["conn_type"] == "s3"
        assert config["host"] == "my-bucket"

    def test_parse_uri_with_query_params(self) -> None:
        """Test parsing URI with query parameters."""
        from truthound_airflow.utils.connection import parse_connection_uri

        uri = "postgresql://localhost:5432/mydb?sslmode=require"

        config = parse_connection_uri(uri)

        assert "extra" in config
        assert config["extra"].get("sslmode") == "require"


class TestGetS3Config:
    """Tests for get_s3_config function."""

    def test_get_s3_config_basic(self) -> None:
        """Test getting basic S3 config."""
        from truthound_airflow.utils.connection import get_s3_config

        config = get_s3_config(
            bucket="my-bucket",
            access_key="AKID123",
            secret_key="secret123",
        )

        assert config["host"] == "my-bucket"
        assert config["login"] == "AKID123"
        assert config["password"] == "secret123"
        assert config["conn_type"] == "s3"

    def test_get_s3_config_with_region(self) -> None:
        """Test getting S3 config with region."""
        from truthound_airflow.utils.connection import get_s3_config

        config = get_s3_config(
            bucket="my-bucket",
            region="us-west-2",
        )

        assert config["extra"]["region"] == "us-west-2"

    def test_get_s3_config_with_endpoint(self) -> None:
        """Test getting S3 config with custom endpoint."""
        from truthound_airflow.utils.connection import get_s3_config

        config = get_s3_config(
            bucket="my-bucket",
            endpoint_url="http://localhost:9000",
        )

        assert config["extra"]["endpoint_url"] == "http://localhost:9000"


class TestGetPostgresConfig:
    """Tests for get_postgres_config function."""

    def test_get_postgres_config_basic(self) -> None:
        """Test getting basic PostgreSQL config."""
        from truthound_airflow.utils.connection import get_postgres_config

        config = get_postgres_config(
            host="localhost",
            database="mydb",
            user="admin",
            password="secret",
        )

        assert config["host"] == "localhost"
        assert config["schema"] == "mydb"
        assert config["port"] == 5432  # default
        assert config["login"] == "admin"
        assert config["password"] == "secret"

    def test_get_postgres_config_with_port(self) -> None:
        """Test getting PostgreSQL config with custom port."""
        from truthound_airflow.utils.connection import get_postgres_config

        config = get_postgres_config(
            host="localhost",
            database="mydb",
            user="admin",
            password="secret",
            port=5433,
        )

        assert config["port"] == 5433


class TestGetBigQueryConfig:
    """Tests for get_bigquery_config function."""

    def test_get_bigquery_config_basic(self) -> None:
        """Test getting basic BigQuery config."""
        from truthound_airflow.utils.connection import get_bigquery_config

        config = get_bigquery_config(project="my-project")

        assert config["extra"]["project"] == "my-project"
        assert config["conn_type"] == "bigquery"

    def test_get_bigquery_config_with_credentials(self) -> None:
        """Test getting BigQuery config with credentials path."""
        from truthound_airflow.utils.connection import get_bigquery_config

        config = get_bigquery_config(
            project="my-project",
            credentials_path="/path/to/creds.json",
        )

        assert config["extra"]["credentials_path"] == "/path/to/creds.json"

    def test_get_bigquery_config_with_location(self) -> None:
        """Test getting BigQuery config with location."""
        from truthound_airflow.utils.connection import get_bigquery_config

        config = get_bigquery_config(
            project="my-project",
            location="EU",
        )

        assert config["extra"]["location"] == "EU"


class TestGetGcsConfig:
    """Tests for get_gcs_config function."""

    def test_get_gcs_config_basic(self) -> None:
        """Test getting basic GCS config."""
        from truthound_airflow.utils.connection import get_gcs_config

        config = get_gcs_config(bucket="my-bucket")

        assert config["host"] == "my-bucket"
        assert config["conn_type"] == "gcs"

    def test_get_gcs_config_with_credentials(self) -> None:
        """Test getting GCS config with credentials."""
        from truthound_airflow.utils.connection import get_gcs_config

        config = get_gcs_config(
            bucket="my-bucket",
            credentials_path="/path/to/creds.json",
        )

        assert config["extra"]["credentials_path"] == "/path/to/creds.json"
