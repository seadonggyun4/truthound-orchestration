"""Tests for Mage I/O configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from truthound_mage.io.config import (
    IOConfig,
    DataSourceConfig,
    DataSinkConfig,
    DataSourceType,
    DataFormat,
    load_io_config,
    DEFAULT_IO_CONFIG,
)


class TestDataSourceConfig:
    """Tests for DataSourceConfig."""

    def test_default_values(self) -> None:
        """Test default source configuration."""
        config = DataSourceConfig(name="test")
        assert config.name == "test"
        assert config.source_type == DataSourceType.FILE
        assert config.connection_string is None
        assert config.options == {}

    def test_with_postgres_config(self) -> None:
        """Test PostgreSQL configuration."""
        config = DataSourceConfig(
            name="warehouse",
            source_type=DataSourceType.POSTGRES,
            host="localhost",
            port=5432,
            database="analytics",
            username="user",
            password="pass",
        )
        assert config.source_type == DataSourceType.POSTGRES
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "analytics"

    def test_with_s3_config(self) -> None:
        """Test S3 configuration."""
        config = DataSourceConfig(
            name="data_lake",
            source_type=DataSourceType.S3,
            bucket="my-bucket",
            path="data/",
            format=DataFormat.PARQUET,
        )
        assert config.source_type == DataSourceType.S3
        assert config.bucket == "my-bucket"
        assert config.format == DataFormat.PARQUET

    def test_get_connection_string_postgres(self) -> None:
        """Test building PostgreSQL connection string."""
        config = DataSourceConfig(
            name="db",
            source_type=DataSourceType.POSTGRES,
            host="localhost",
            port=5432,
            database="test",
            username="user",
            password="pass",
        )
        conn_str = config.get_connection_string()
        assert "postgresql://" in conn_str
        assert "localhost" in conn_str
        assert "test" in conn_str

    def test_get_connection_string_s3(self) -> None:
        """Test building S3 URL."""
        config = DataSourceConfig(
            name="s3",
            source_type=DataSourceType.S3,
            bucket="my-bucket",
            path="data/files",
        )
        url = config.get_connection_string()
        assert url == "s3://my-bucket/data/files"

    def test_get_connection_string_gcs(self) -> None:
        """Test building GCS URL."""
        config = DataSourceConfig(
            name="gcs",
            source_type=DataSourceType.GCS,
            bucket="my-bucket",
            path="data/files",
        )
        url = config.get_connection_string()
        assert url == "gs://my-bucket/data/files"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = DataSourceConfig(
            name="test",
            source_type=DataSourceType.POSTGRES,
            host="localhost",
            database="test",
        )
        data = config.to_dict()

        assert data["name"] == "test"
        assert data["type"] == "postgres"
        assert data["host"] == "localhost"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "type": "postgres",
            "host": "localhost",
            "database": "test",
        }
        config = DataSourceConfig.from_dict("test", data)

        assert config.name == "test"
        assert config.source_type == DataSourceType.POSTGRES
        assert config.host == "localhost"


class TestDataSinkConfig:
    """Tests for DataSinkConfig."""

    def test_default_values(self) -> None:
        """Test default sink configuration."""
        config = DataSinkConfig(name="test")
        assert config.name == "test"
        assert config.sink_type == DataSourceType.FILE
        assert config.mode == "overwrite"
        assert config.format == DataFormat.PARQUET

    def test_with_s3_sink(self) -> None:
        """Test S3 sink configuration."""
        config = DataSinkConfig(
            name="output",
            sink_type=DataSourceType.S3,
            path="s3://bucket/output/",
            format=DataFormat.PARQUET,
            partition_by=("date", "region"),
            mode="append",
        )
        assert config.sink_type == DataSourceType.S3
        assert config.partition_by == ("date", "region")
        assert config.mode == "append"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = DataSinkConfig(
            name="test",
            sink_type=DataSourceType.S3,
            path="s3://bucket/output/",
            mode="overwrite",
        )
        data = config.to_dict()

        assert data["name"] == "test"
        assert data["type"] == "s3"
        assert data["mode"] == "overwrite"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "type": "s3",
            "path": "s3://bucket/output/",
            "mode": "append",
        }
        config = DataSinkConfig.from_dict("test", data)

        assert config.name == "test"
        assert config.sink_type == DataSourceType.S3
        assert config.mode == "append"


class TestIOConfig:
    """Tests for IOConfig."""

    def test_default_values(self) -> None:
        """Test default I/O configuration."""
        config = IOConfig()
        assert config.sources == {}
        assert config.sinks == {}
        assert config.engine_name is None
        assert config.timeout_seconds == 300
        assert config.profile == "default"

    def test_with_sources(self) -> None:
        """Test configuration with sources."""
        source = DataSourceConfig(name="db", source_type=DataSourceType.POSTGRES)
        config = IOConfig(sources={"db": source})

        assert len(config.sources) == 1
        assert "db" in config.sources

    def test_with_sinks(self) -> None:
        """Test configuration with sinks."""
        sink = DataSinkConfig(name="output", sink_type=DataSourceType.S3)
        config = IOConfig(sinks={"output": sink})

        assert len(config.sinks) == 1
        assert "output" in config.sinks

    def test_get_source(self) -> None:
        """Test getting source by name."""
        source = DataSourceConfig(name="db", source_type=DataSourceType.POSTGRES)
        config = IOConfig(sources={"db": source})

        assert config.get_source("db") is source

    def test_get_source_not_found(self) -> None:
        """Test getting nonexistent source."""
        config = IOConfig()
        with pytest.raises(KeyError):
            config.get_source("missing")

    def test_get_sink(self) -> None:
        """Test getting sink by name."""
        sink = DataSinkConfig(name="output")
        config = IOConfig(sinks={"output": sink})

        assert config.get_sink("output") is sink

    def test_get_sink_not_found(self) -> None:
        """Test getting nonexistent sink."""
        config = IOConfig()
        with pytest.raises(KeyError):
            config.get_sink("missing")

    def test_get_variable(self) -> None:
        """Test getting variable."""
        config = IOConfig(variables={"key": "value"})
        assert config.get_variable("key") == "value"
        assert config.get_variable("missing") is None
        assert config.get_variable("missing", "default") == "default"

    def test_list_sources(self) -> None:
        """Test listing source names."""
        source1 = DataSourceConfig(name="db1")
        source2 = DataSourceConfig(name="db2")
        config = IOConfig(sources={"db1": source1, "db2": source2})

        names = config.list_sources()
        assert "db1" in names
        assert "db2" in names

    def test_list_sinks(self) -> None:
        """Test listing sink names."""
        sink1 = DataSinkConfig(name="out1")
        sink2 = DataSinkConfig(name="out2")
        config = IOConfig(sinks={"out1": sink1, "out2": sink2})

        names = config.list_sinks()
        assert "out1" in names
        assert "out2" in names

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        source = DataSourceConfig(name="db", source_type=DataSourceType.POSTGRES)
        sink = DataSinkConfig(name="output", sink_type=DataSourceType.S3)
        config = IOConfig(
            sources={"db": source},
            sinks={"output": sink},
            engine_name="truthound",
        )

        data = config.to_dict()

        assert "db" in data["sources"]
        assert "output" in data["sinks"]
        assert data["engine_name"] == "truthound"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "sources": {
                "db": {"type": "postgres", "host": "localhost"},
            },
            "sinks": {
                "output": {"type": "s3", "path": "s3://bucket/"},
            },
            "engine_name": "truthound",
        }
        config = IOConfig.from_dict(data)

        assert "db" in config.sources
        assert "output" in config.sinks
        assert config.engine_name == "truthound"

    def test_from_mage_config(self) -> None:
        """Test creation from Mage io_config.yaml structure."""
        mage_config = {
            "default": {
                "TRUTHOUND_ENGINE": "truthound",
                "TRUTHOUND_TIMEOUT": 600,
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": 5432,
                "POSTGRES_DATABASE": "analytics",
            },
        }
        config = IOConfig.from_mage_config(mage_config, profile="default")

        assert config.engine_name == "truthound"
        assert config.timeout_seconds == 600
        assert "postgres" in config.sources

    def test_from_mage_config_with_data_sources(self) -> None:
        """Test creation with explicit data_sources section."""
        mage_config = {
            "default": {
                "data_sources": {
                    "warehouse": {
                        "type": "postgres",
                        "host": "localhost",
                        "database": "analytics",
                    },
                },
            },
        }
        config = IOConfig.from_mage_config(mage_config, profile="default")

        assert "warehouse" in config.sources
        assert config.sources["warehouse"].source_type == DataSourceType.POSTGRES


class TestLoadIOConfig:
    """Tests for load_io_config function."""

    def test_load_yaml(self) -> None:
        """Test loading from YAML file."""
        config_data = {
            "default": {
                "TRUTHOUND_ENGINE": "truthound",
                "data_sources": {
                    "db": {
                        "type": "postgres",
                        "host": "localhost",
                        "database": "test",
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = load_io_config(temp_path)

            assert "db" in config.sources
            assert config.sources["db"].source_type == DataSourceType.POSTGRES
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_io_config(Path("/nonexistent/path/io_config.yaml"))

    def test_default_config(self) -> None:
        """Test default I/O configuration."""
        assert DEFAULT_IO_CONFIG.sources == {}
        assert DEFAULT_IO_CONFIG.sinks == {}
