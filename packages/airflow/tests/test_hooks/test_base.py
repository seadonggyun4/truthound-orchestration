"""Tests for DataQualityHook."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestConnectionConfig:
    """Tests for ConnectionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from truthound_airflow.hooks.base import ConnectionConfig

        config = ConnectionConfig()

        assert config.conn_type == "filesystem"
        assert config.host is None
        assert config.port is None
        assert config.schema is None
        assert config.login is None
        assert config.password is None
        assert config.extra == {}

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        from truthound_airflow.hooks.base import ConnectionConfig

        config = ConnectionConfig(
            conn_type="s3",
            host="my-bucket",
            login="access_key",
            password="secret_key",
            extra={"region": "us-east-1"},
        )

        assert config.conn_type == "s3"
        assert config.host == "my-bucket"
        assert config.login == "access_key"
        assert config.extra["region"] == "us-east-1"

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        from truthound_airflow.hooks.base import ConnectionConfig

        config = ConnectionConfig()

        with pytest.raises(AttributeError):
            config.conn_type = "postgres"  # type: ignore

    def test_from_airflow_connection(self, mock_connection: MagicMock) -> None:
        """Test creating config from Airflow Connection."""
        from truthound_airflow.hooks.base import ConnectionConfig

        config = ConnectionConfig.from_airflow_connection(mock_connection)

        assert config.host == mock_connection.host
        assert config.login == mock_connection.login
        assert config.password == mock_connection.password


class TestDataQualityHook:
    """Tests for DataQualityHook."""

    def test_initialization(self) -> None:
        """Test hook initialization."""
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook(connection_id="test_conn")

        assert hook.connection_id == "test_conn"

    def test_default_connection_id(self) -> None:
        """Test default connection ID."""
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook()

        assert hook.connection_id == "truthound_default"

    def test_conn_type_attribute(self) -> None:
        """Test conn_type class attribute."""
        from truthound_airflow.hooks.base import DataQualityHook

        assert DataQualityHook.conn_type == "truthound"

    def test_hook_name_attribute(self) -> None:
        """Test hook_name class attribute."""
        from truthound_airflow.hooks.base import DataQualityHook

        assert DataQualityHook.hook_name == "Data Quality"

    @patch("truthound_airflow.hooks.base.BaseHook.get_connection")
    def test_config_property(
        self,
        mock_get_connection: MagicMock,
        mock_connection: MagicMock,
    ) -> None:
        """Test config property loads connection."""
        from truthound_airflow.hooks.base import DataQualityHook

        mock_get_connection.return_value = mock_connection

        hook = DataQualityHook(connection_id="test_conn")
        config = hook.config

        assert config is not None
        mock_get_connection.assert_called_once_with("test_conn")

    @patch("truthound_airflow.hooks.base.BaseHook.get_connection")
    def test_config_caching(
        self,
        mock_get_connection: MagicMock,
        mock_connection: MagicMock,
    ) -> None:
        """Test config is cached after first access."""
        from truthound_airflow.hooks.base import DataQualityHook

        mock_get_connection.return_value = mock_connection

        hook = DataQualityHook(connection_id="test_conn")

        # Access config twice
        _ = hook.config
        _ = hook.config

        # Should only call get_connection once due to caching
        assert mock_get_connection.call_count == 1

    @patch("truthound_airflow.hooks.base.BaseHook.get_connection")
    def test_config_fallback_on_error(
        self,
        mock_get_connection: MagicMock,
    ) -> None:
        """Test config falls back to default on connection error."""
        from truthound_airflow.hooks.base import DataQualityHook

        mock_get_connection.side_effect = Exception("Connection not found")

        hook = DataQualityHook(connection_id="nonexistent")
        config = hook.config

        assert config.conn_type == "filesystem"

    @patch("truthound_airflow.hooks.base.BaseHook.get_connection")
    def test_get_conn_returns_config(
        self,
        mock_get_connection: MagicMock,
        mock_connection: MagicMock,
    ) -> None:
        """Test get_conn returns ConnectionConfig."""
        from truthound_airflow.hooks.base import ConnectionConfig, DataQualityHook

        mock_get_connection.return_value = mock_connection

        hook = DataQualityHook(connection_id="test_conn")
        conn = hook.get_conn()

        assert isinstance(conn, ConnectionConfig)

    def test_register_loader(self) -> None:
        """Test registering a custom loader."""
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook()

        # Create a mock loader
        mock_loader = MagicMock()
        mock_loader.supports.return_value = True
        mock_loader.load.return_value = MagicMock()

        hook.register_loader(mock_loader)

        assert mock_loader in hook._loaders

    def test_test_connection(self) -> None:
        """Test test_connection method."""
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook()

        # Default should work (filesystem)
        success, message = hook.test_connection()

        assert isinstance(success, bool)
        assert isinstance(message, str)


class TestDataQualityHookDataLoading:
    """Tests for DataQualityHook data loading methods."""

    @patch("truthound_airflow.hooks.base.BaseHook.get_connection")
    @pytest.mark.requires_polars
    def test_load_data_local_parquet(
        self,
        mock_get_connection: MagicMock,
        mock_connection: MagicMock,
        temp_parquet_file: Any,
    ) -> None:
        """Test loading local parquet file."""
        from truthound_airflow.hooks.base import DataQualityHook

        mock_get_connection.return_value = mock_connection

        hook = DataQualityHook()
        data = hook.load_data(str(temp_parquet_file))

        assert data is not None
        assert len(data) > 0

    @patch("truthound_airflow.hooks.base.BaseHook.get_connection")
    @pytest.mark.requires_polars
    def test_load_data_local_csv(
        self,
        mock_get_connection: MagicMock,
        mock_connection: MagicMock,
        temp_csv_file: Any,
    ) -> None:
        """Test loading local CSV file."""
        from truthound_airflow.hooks.base import DataQualityHook

        mock_get_connection.return_value = mock_connection

        hook = DataQualityHook()
        data = hook.load_data(str(temp_csv_file))

        assert data is not None
        assert len(data) > 0


class TestTruthoundHookAlias:
    """Tests for TruthoundHook legacy alias."""

    def test_alias_exists(self) -> None:
        """Test that legacy alias exists."""
        from truthound_airflow.hooks.base import DataQualityHook, TruthoundHook

        assert TruthoundHook is DataQualityHook


class TestDataLoaderProtocol:
    """Tests for DataLoader protocol."""

    def test_protocol_runtime_checkable(self) -> None:
        """Test that DataLoader is runtime checkable."""
        from truthound_airflow.hooks.base import DataLoader

        class ValidLoader:
            def load(self, path: str, **kwargs: Any) -> Any:
                return None

            def supports(self, path: str) -> bool:
                return True

        loader = ValidLoader()
        assert isinstance(loader, DataLoader)

    def test_protocol_missing_method(self) -> None:
        """Test that incomplete implementation is not a DataLoader."""
        from truthound_airflow.hooks.base import DataLoader

        class IncompleteLoader:
            def load(self, path: str, **kwargs: Any) -> Any:
                return None

            # Missing supports method

        loader = IncompleteLoader()
        assert not isinstance(loader, DataLoader)


class TestDataWriterProtocol:
    """Tests for DataWriter protocol."""

    def test_protocol_runtime_checkable(self) -> None:
        """Test that DataWriter is runtime checkable."""
        from truthound_airflow.hooks.base import DataWriter

        class ValidWriter:
            def write(self, data: Any, path: str, **kwargs: Any) -> None:
                pass

            def supports(self, path: str) -> bool:
                return True

        writer = ValidWriter()
        assert isinstance(writer, DataWriter)
