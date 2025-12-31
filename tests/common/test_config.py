"""Tests for common.config module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from common.config import (
    DEFAULT_ENV_PREFIX,
    EnvReader,
    PlatformConfig,
    TruthoundConfig,
    find_config_file,
    load_config_file,
    require_valid_config,
    validate_config,
)
from common.exceptions import (
    ConfigurationError,
    InvalidConfigValueError,
    MissingConfigError,
)


class TestEnvReader:
    """Tests for EnvReader class."""

    def test_default_prefix(self):
        """Test default prefix is applied."""
        reader = EnvReader()
        assert reader.prefix == DEFAULT_ENV_PREFIX

    def test_custom_prefix(self):
        """Test custom prefix."""
        reader = EnvReader(prefix="MYAPP")
        assert reader.prefix == "MYAPP"

    def test_get_existing_var(self):
        """Test getting an existing environment variable."""
        with patch.dict(os.environ, {"TRUTHOUND_TEST": "value"}):
            reader = EnvReader()
            assert reader.get("TEST") == "value"

    def test_get_nonexistent_var(self):
        """Test getting a nonexistent variable returns default."""
        reader = EnvReader()
        assert reader.get("NONEXISTENT") is None
        assert reader.get("NONEXISTENT", default="fallback") == "fallback"

    def test_get_required_existing(self):
        """Test get_required with existing variable."""
        with patch.dict(os.environ, {"TRUTHOUND_KEY": "secret"}):
            reader = EnvReader()
            assert reader.get_required("KEY") == "secret"

    def test_get_required_missing(self):
        """Test get_required raises for missing variable."""
        reader = EnvReader()
        with pytest.raises(MissingConfigError):
            reader.get_required("DEFINITELY_NOT_SET")

    def test_get_int(self):
        """Test getting integer values."""
        with patch.dict(os.environ, {"TRUTHOUND_COUNT": "42"}):
            reader = EnvReader()
            assert reader.get_int("COUNT") == 42
            assert reader.get_int("MISSING") is None
            assert reader.get_int("MISSING", default=10) == 10

    def test_get_int_invalid(self):
        """Test get_int raises for invalid value."""
        with patch.dict(os.environ, {"TRUTHOUND_BAD": "not_a_number"}):
            reader = EnvReader()
            with pytest.raises(InvalidConfigValueError) as exc_info:
                reader.get_int("BAD")
            assert "integer" in str(exc_info.value.expected)

    def test_get_float(self):
        """Test getting float values."""
        with patch.dict(os.environ, {"TRUTHOUND_RATE": "3.14"}):
            reader = EnvReader()
            assert reader.get_float("RATE") == 3.14

    def test_get_float_invalid(self):
        """Test get_float raises for invalid value."""
        with patch.dict(os.environ, {"TRUTHOUND_BAD": "not_float"}):
            reader = EnvReader()
            with pytest.raises(InvalidConfigValueError):
                reader.get_float("BAD")

    def test_get_bool_true_values(self):
        """Test get_bool with truthy values."""
        reader = EnvReader()
        true_values = ["1", "true", "TRUE", "yes", "YES", "on", "ON"]

        for value in true_values:
            with patch.dict(os.environ, {"TRUTHOUND_FLAG": value}):
                assert reader.get_bool("FLAG") is True

    def test_get_bool_false_values(self):
        """Test get_bool with falsy values."""
        reader = EnvReader()
        false_values = ["0", "false", "FALSE", "no", "NO", "off", "OFF"]

        for value in false_values:
            with patch.dict(os.environ, {"TRUTHOUND_FLAG": value}):
                assert reader.get_bool("FLAG") is False

    def test_get_bool_invalid(self):
        """Test get_bool raises for invalid value."""
        with patch.dict(os.environ, {"TRUTHOUND_BAD": "maybe"}):
            reader = EnvReader()
            with pytest.raises(InvalidConfigValueError):
                reader.get_bool("BAD")

    def test_get_list(self):
        """Test getting list values."""
        with patch.dict(os.environ, {"TRUTHOUND_ITEMS": "a,b,c"}):
            reader = EnvReader()
            result = reader.get_list("ITEMS")
            assert result == ["a", "b", "c"]

    def test_get_list_custom_separator(self):
        """Test list with custom separator."""
        with patch.dict(os.environ, {"TRUTHOUND_ITEMS": "a;b;c"}):
            reader = EnvReader()
            result = reader.get_list("ITEMS", separator=";")
            assert result == ["a", "b", "c"]

    def test_get_list_empty(self):
        """Test empty list."""
        with patch.dict(os.environ, {"TRUTHOUND_ITEMS": ""}):
            reader = EnvReader()
            assert reader.get_list("ITEMS") == []

    def test_get_json(self):
        """Test getting JSON values."""
        data = {"key": "value", "count": 42}
        with patch.dict(os.environ, {"TRUTHOUND_DATA": json.dumps(data)}):
            reader = EnvReader()
            result = reader.get_json("DATA")
            assert result == data

    def test_get_json_invalid(self):
        """Test get_json raises for invalid JSON."""
        with patch.dict(os.environ, {"TRUTHOUND_BAD": "not json"}):
            reader = EnvReader()
            with pytest.raises(InvalidConfigValueError):
                reader.get_json("BAD")


class TestPlatformConfig:
    """Tests for PlatformConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PlatformConfig(platform="test")
        assert config.platform == "test"
        assert config.enabled is True
        assert config.connection_id is None
        assert config.timeout_seconds == 300
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 10

    def test_with_extra(self):
        """Test with_extra builder method."""
        config = PlatformConfig(platform="airflow", extra={"a": 1})
        new_config = config.with_extra(b=2)

        assert config.extra == {"a": 1}
        assert new_config.extra == {"a": 1, "b": 2}

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        config = PlatformConfig(
            platform="dagster",
            enabled=True,
            connection_id="conn_123",
            timeout_seconds=60,
        )
        data = config.to_dict()
        restored = PlatformConfig.from_dict(data, "dagster")

        assert restored.platform == "dagster"
        assert restored.connection_id == "conn_123"
        assert restored.timeout_seconds == 60


class TestTruthoundConfig:
    """Tests for TruthoundConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TruthoundConfig()
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.default_timeout_seconds == 300
        assert config.fail_fast is True

    def test_get_platform_config_existing(self):
        """Test getting existing platform config."""
        platform_config = PlatformConfig(platform="airflow", timeout_seconds=120)
        config = TruthoundConfig(platforms={"airflow": platform_config})

        result = config.get_platform_config("airflow")
        assert result.timeout_seconds == 120

    def test_get_platform_config_default(self):
        """Test getting default platform config."""
        config = TruthoundConfig()
        result = config.get_platform_config("nonexistent")

        assert result.platform == "nonexistent"
        assert result.enabled is True

    def test_is_platform_enabled(self):
        """Test platform enabled check."""
        disabled = PlatformConfig(platform="dagster", enabled=False)
        config = TruthoundConfig(platforms={"dagster": disabled})

        assert config.is_platform_enabled("dagster") is False
        assert config.is_platform_enabled("airflow") is True  # default

    def test_with_platform(self):
        """Test with_platform builder method."""
        config = TruthoundConfig()
        platform = PlatformConfig(platform="prefect", timeout_seconds=60)
        new_config = config.with_platform(platform)

        assert "prefect" not in config.platforms
        assert new_config.get_platform_config("prefect").timeout_seconds == 60

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        config = TruthoundConfig(
            debug=True,
            log_level="DEBUG",
            default_timeout_seconds=120,
            platforms={
                "airflow": PlatformConfig(platform="airflow", enabled=True),
            },
        )
        data = config.to_dict()
        restored = TruthoundConfig.from_dict(data)

        assert restored.debug is True
        assert restored.log_level == "DEBUG"
        assert "airflow" in restored.platforms

    def test_from_env(self):
        """Test loading from environment variables."""
        env_vars = {
            "TRUTHOUND_DEBUG": "true",
            "TRUTHOUND_LOG_LEVEL": "DEBUG",
            "TRUTHOUND_TIMEOUT": "120",
            "TRUTHOUND_AIRFLOW_ENABLED": "true",
            "TRUTHOUND_AIRFLOW_CONNECTION_ID": "truthound_conn",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = TruthoundConfig.from_env()

            assert config.debug is True
            assert config.log_level == "DEBUG"
            assert config.default_timeout_seconds == 120
            assert "airflow" in config.platforms
            assert config.platforms["airflow"].connection_id == "truthound_conn"


class TestConfigFile:
    """Tests for configuration file loading."""

    def test_load_json_config(self, tmp_path: Path):
        """Test loading JSON configuration file."""
        config_data = {
            "debug": True,
            "log_level": "WARNING",
            "platforms": {
                "airflow": {"enabled": True, "timeout_seconds": 60},
            },
        }
        config_file = tmp_path / "truthound.json"
        config_file.write_text(json.dumps(config_data))

        result = load_config_file(config_file)
        assert result["debug"] is True
        assert result["platforms"]["airflow"]["timeout_seconds"] == 60

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test loading nonexistent file raises error."""
        with pytest.raises(ConfigurationError):
            load_config_file(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, tmp_path: Path):
        """Test loading invalid JSON raises error."""
        config_file = tmp_path / "bad.json"
        config_file.write_text("not valid json")

        with pytest.raises(ConfigurationError):
            load_config_file(config_file)

    def test_load_unsupported_format(self, tmp_path: Path):
        """Test loading unsupported format raises error."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some text")

        with pytest.raises(ConfigurationError):
            load_config_file(config_file)

    def test_find_config_file(self, tmp_path: Path):
        """Test finding configuration file."""
        config_file = tmp_path / "truthound.json"
        config_file.write_text("{}")

        result = find_config_file(start_dir=tmp_path)
        assert result == config_file

    def test_find_config_file_not_found(self, tmp_path: Path):
        """Test find_config_file returns None when not found."""
        result = find_config_file(start_dir=tmp_path, max_depth=1)
        assert result is None

    def test_config_from_file(self, tmp_path: Path):
        """Test creating config from file."""
        config_data = {
            "debug": True,
            "default_timeout_seconds": 180,
        }
        config_file = tmp_path / "truthound.json"
        config_file.write_text(json.dumps(config_data))

        config = TruthoundConfig.from_file(config_file)
        assert config.debug is True
        assert config.default_timeout_seconds == 180


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        config = TruthoundConfig()
        issues = validate_config(config)
        assert issues == []

    def test_validate_invalid_log_level(self):
        """Test validation catches invalid log level."""
        config = TruthoundConfig(log_level="INVALID")
        issues = validate_config(config)

        assert len(issues) == 1
        assert "log_level" in issues[0].lower()

    def test_validate_invalid_timeout(self):
        """Test validation catches invalid timeout."""
        config = TruthoundConfig(default_timeout_seconds=-1)
        issues = validate_config(config)

        assert len(issues) == 1
        assert "timeout" in issues[0].lower()

    def test_validate_invalid_sample_size(self):
        """Test validation catches invalid sample size."""
        config = TruthoundConfig(default_sample_size=-100)
        issues = validate_config(config)

        assert len(issues) == 1
        assert "sample_size" in issues[0].lower()

    def test_validate_platform_invalid_timeout(self):
        """Test validation catches invalid platform timeout."""
        platform = PlatformConfig(platform="test", timeout_seconds=0)
        config = TruthoundConfig(platforms={"test": platform})
        issues = validate_config(config)

        assert len(issues) == 1
        assert "timeout" in issues[0].lower()

    def test_require_valid_config_valid(self):
        """Test require_valid_config passes for valid config."""
        config = TruthoundConfig()
        require_valid_config(config)  # Should not raise

    def test_require_valid_config_invalid(self):
        """Test require_valid_config raises for invalid config."""
        config = TruthoundConfig(log_level="BAD")
        with pytest.raises(ConfigurationError):
            require_valid_config(config)


class TestConfigMerging:
    """Tests for configuration merging."""

    def test_load_with_env_override(self, tmp_path: Path):
        """Test that environment variables override file config."""
        config_data = {
            "debug": False,
            "log_level": "INFO",
        }
        config_file = tmp_path / "truthound.json"
        config_file.write_text(json.dumps(config_data))

        env_vars = {
            "TRUTHOUND_DEBUG": "true",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = TruthoundConfig.load(config_file=config_file)

            # Environment overrides file
            assert config.debug is True
            # File value used when env not set
            assert config.log_level == "INFO"
