"""Configuration management for Truthound Integrations.

This module provides a unified configuration system that supports:
- Environment variable loading with prefix support
- File-based configuration (JSON/YAML)
- Platform-specific configuration extraction
- Configuration merging with proper precedence

Configuration Precedence (highest to lowest):
    1. Explicit parameters
    2. Environment variables
    3. Configuration file
    4. Default values

Example:
    >>> from common.config import TruthoundConfig
    >>> config = TruthoundConfig.from_env()
    >>> airflow_config = config.get_platform_config("airflow")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from common.exceptions import ConfigurationError, InvalidConfigValueError, MissingConfigError


# =============================================================================
# Constants
# =============================================================================

DEFAULT_ENV_PREFIX = "TRUTHOUND"
CONFIG_FILE_NAMES = ("truthound.json", "truthound.yaml", "truthound.yml", ".truthound.json")


# =============================================================================
# Environment Variable Utilities
# =============================================================================


class EnvReader:
    """Utility class for reading environment variables with prefix support.

    Provides typed accessors for environment variables with optional
    prefix handling and default values.

    Example:
        >>> reader = EnvReader(prefix="TRUTHOUND")
        >>> timeout = reader.get_int("TIMEOUT", default=30)
        >>> debug = reader.get_bool("DEBUG", default=False)
    """

    def __init__(self, prefix: str = DEFAULT_ENV_PREFIX) -> None:
        """Initialize the environment reader.

        Args:
            prefix: Prefix for environment variable names.
        """
        self.prefix = prefix

    def _make_key(self, name: str) -> str:
        """Create full environment variable key with prefix."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def get(self, name: str, default: str | None = None) -> str | None:
        """Get a string environment variable.

        Args:
            name: Variable name (without prefix).
            default: Default value if not set.

        Returns:
            Environment variable value or default.
        """
        return os.environ.get(self._make_key(name), default)

    def get_required(self, name: str) -> str:
        """Get a required string environment variable.

        Args:
            name: Variable name (without prefix).

        Returns:
            Environment variable value.

        Raises:
            MissingConfigError: If variable is not set.
        """
        key = self._make_key(name)
        value = os.environ.get(key)
        if value is None:
            raise MissingConfigError(key)
        return value

    def get_int(self, name: str, default: int | None = None) -> int | None:
        """Get an integer environment variable.

        Args:
            name: Variable name (without prefix).
            default: Default value if not set.

        Returns:
            Parsed integer value or default.

        Raises:
            InvalidConfigValueError: If value cannot be parsed as int.
        """
        value = self.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError as e:
            raise InvalidConfigValueError(
                f"Invalid integer value for {self._make_key(name)}",
                config_key=self._make_key(name),
                value=value,
                expected="integer",
                cause=e,
            ) from e

    def get_float(self, name: str, default: float | None = None) -> float | None:
        """Get a float environment variable.

        Args:
            name: Variable name (without prefix).
            default: Default value if not set.

        Returns:
            Parsed float value or default.

        Raises:
            InvalidConfigValueError: If value cannot be parsed as float.
        """
        value = self.get(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError as e:
            raise InvalidConfigValueError(
                f"Invalid float value for {self._make_key(name)}",
                config_key=self._make_key(name),
                value=value,
                expected="float",
                cause=e,
            ) from e

    def get_bool(self, name: str, default: bool | None = None) -> bool | None:
        """Get a boolean environment variable.

        Truthy values: "1", "true", "yes", "on" (case-insensitive)
        Falsy values: "0", "false", "no", "off" (case-insensitive)

        Args:
            name: Variable name (without prefix).
            default: Default value if not set.

        Returns:
            Parsed boolean value or default.

        Raises:
            InvalidConfigValueError: If value cannot be parsed as bool.
        """
        value = self.get(name)
        if value is None:
            return default
        lower_value = value.lower()
        if lower_value in ("1", "true", "yes", "on"):
            return True
        if lower_value in ("0", "false", "no", "off"):
            return False
        raise InvalidConfigValueError(
            f"Invalid boolean value for {self._make_key(name)}",
            config_key=self._make_key(name),
            value=value,
            expected="boolean (1/0, true/false, yes/no, on/off)",
        )

    def get_list(
        self,
        name: str,
        separator: str = ",",
        default: list[str] | None = None,
    ) -> list[str] | None:
        """Get a list environment variable (comma-separated by default).

        Args:
            name: Variable name (without prefix).
            separator: List item separator.
            default: Default value if not set.

        Returns:
            Parsed list of strings or default.
        """
        value = self.get(name)
        if value is None:
            return default
        if not value.strip():
            return []
        return [item.strip() for item in value.split(separator)]

    def get_json(self, name: str, default: Any = None) -> Any:
        """Get a JSON-encoded environment variable.

        Args:
            name: Variable name (without prefix).
            default: Default value if not set.

        Returns:
            Parsed JSON value or default.

        Raises:
            InvalidConfigValueError: If value cannot be parsed as JSON.
        """
        value = self.get(name)
        if value is None:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise InvalidConfigValueError(
                f"Invalid JSON value for {self._make_key(name)}",
                config_key=self._make_key(name),
                value=value,
                expected="valid JSON",
                cause=e,
            ) from e


# =============================================================================
# File Configuration Utilities
# =============================================================================


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If YAML parsing fails.
    """
    try:
        import yaml
    except ImportError as e:
        raise ConfigurationError(
            "PyYAML is required for YAML configuration files. "
            "Install with: pip install pyyaml",
            cause=e,
        ) from e

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML configuration: {path}",
            details={"path": str(path)},
            cause=e,
        ) from e


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON configuration file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If JSON parsing fails.
    """
    try:
        with path.open() as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Failed to parse JSON configuration: {path}",
            details={"path": str(path)},
            cause=e,
        ) from e


def load_config_file(path: Path) -> dict[str, Any]:
    """Load configuration from a file (JSON or YAML).

    Args:
        path: Path to configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If file cannot be loaded.
    """
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}",
            details={"path": str(path)},
        )

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    elif suffix == ".json":
        return _load_json(path)
    else:
        raise ConfigurationError(
            f"Unsupported configuration file format: {suffix}",
            details={"path": str(path), "suffix": suffix},
        )


def find_config_file(
    start_dir: Path | None = None,
    max_depth: int = 5,
) -> Path | None:
    """Find configuration file by searching up the directory tree.

    Searches for files named in CONFIG_FILE_NAMES, starting from
    start_dir and moving up the directory tree.

    Args:
        start_dir: Directory to start search from (default: cwd).
        max_depth: Maximum directories to traverse up.

    Returns:
        Path to configuration file if found, None otherwise.
    """
    current = start_dir or Path.cwd()
    for _ in range(max_depth):
        for name in CONFIG_FILE_NAMES:
            config_path = current / name
            if config_path.is_file():
                return config_path
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# =============================================================================
# Platform Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    """Configuration for a specific platform integration.

    Attributes:
        platform: Platform name (e.g., 'airflow', 'dagster').
        enabled: Whether the platform integration is enabled.
        connection_id: Platform-specific connection identifier.
        timeout_seconds: Default timeout for operations.
        retry_count: Number of retries for failed operations.
        retry_delay_seconds: Delay between retries.
        extra: Platform-specific extra configuration.
    """

    platform: str
    enabled: bool = True
    connection_id: str | None = None
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay_seconds: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "platform": self.platform,
            "enabled": self.enabled,
            "connection_id": self.connection_id,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], platform: str) -> Self:
        """Create a PlatformConfig from a dictionary.

        Args:
            data: Dictionary containing platform configuration.
            platform: Platform name.

        Returns:
            New PlatformConfig instance.
        """
        return cls(
            platform=platform,
            enabled=data.get("enabled", True),
            connection_id=data.get("connection_id"),
            timeout_seconds=data.get("timeout_seconds", 300),
            retry_count=data.get("retry_count", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 10),
            extra=data.get("extra", {}),
        )

    def with_extra(self, **kwargs: Any) -> PlatformConfig:
        """Create a new config with additional extra parameters.

        Args:
            **kwargs: Additional platform-specific parameters.

        Returns:
            New PlatformConfig with merged extra parameters.
        """
        return PlatformConfig(
            platform=self.platform,
            enabled=self.enabled,
            connection_id=self.connection_id,
            timeout_seconds=self.timeout_seconds,
            retry_count=self.retry_count,
            retry_delay_seconds=self.retry_delay_seconds,
            extra={**self.extra, **kwargs},
        )


# =============================================================================
# Main Configuration Class
# =============================================================================


@dataclass(frozen=True, slots=True)
class TruthoundConfig:
    """Main configuration class for Truthound Integrations.

    Provides a unified interface for accessing configuration from
    environment variables, files, and explicit parameters.

    Attributes:
        debug: Enable debug mode.
        log_level: Logging level.
        default_timeout_seconds: Default operation timeout.
        default_sample_size: Default sample size for large datasets.
        fail_fast: Whether to fail fast on first error.
        platforms: Platform-specific configurations.
        extra: Additional configuration.

    Example:
        >>> config = TruthoundConfig.from_env()
        >>> airflow_config = config.get_platform_config("airflow")
        >>> if airflow_config.enabled:
        ...     # Use airflow configuration
        ...     pass
    """

    debug: bool = False
    log_level: str = "INFO"
    default_timeout_seconds: int = 300
    default_sample_size: int | None = None
    fail_fast: bool = True
    platforms: dict[str, PlatformConfig] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def get_platform_config(self, platform: str) -> PlatformConfig:
        """Get configuration for a specific platform.

        Args:
            platform: Platform name (e.g., 'airflow', 'dagster').

        Returns:
            PlatformConfig for the specified platform.
            Returns a default config if platform not explicitly configured.
        """
        if platform in self.platforms:
            return self.platforms[platform]
        return PlatformConfig(platform=platform)

    def is_platform_enabled(self, platform: str) -> bool:
        """Check if a platform integration is enabled.

        Args:
            platform: Platform name.

        Returns:
            True if platform is enabled, False otherwise.
        """
        return self.get_platform_config(platform).enabled

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "debug": self.debug,
            "log_level": self.log_level,
            "default_timeout_seconds": self.default_timeout_seconds,
            "default_sample_size": self.default_sample_size,
            "fail_fast": self.fail_fast,
            "platforms": {k: v.to_dict() for k, v in self.platforms.items()},
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a TruthoundConfig from a dictionary.

        Args:
            data: Dictionary containing configuration.

        Returns:
            New TruthoundConfig instance.
        """
        platforms_data = data.get("platforms", {})
        platforms = {
            name: PlatformConfig.from_dict(config, name)
            for name, config in platforms_data.items()
        }
        return cls(
            debug=data.get("debug", False),
            log_level=data.get("log_level", "INFO"),
            default_timeout_seconds=data.get("default_timeout_seconds", 300),
            default_sample_size=data.get("default_sample_size"),
            fail_fast=data.get("fail_fast", True),
            platforms=platforms,
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_env(cls, prefix: str = DEFAULT_ENV_PREFIX) -> Self:
        """Create configuration from environment variables.

        Environment Variables:
            {PREFIX}_DEBUG: Enable debug mode (bool)
            {PREFIX}_LOG_LEVEL: Logging level (string)
            {PREFIX}_TIMEOUT: Default timeout in seconds (int)
            {PREFIX}_SAMPLE_SIZE: Default sample size (int)
            {PREFIX}_FAIL_FAST: Fail fast mode (bool)
            {PREFIX}_{PLATFORM}_ENABLED: Enable platform (bool)
            {PREFIX}_{PLATFORM}_CONNECTION_ID: Connection ID (string)
            {PREFIX}_{PLATFORM}_TIMEOUT: Platform timeout (int)

        Args:
            prefix: Environment variable prefix.

        Returns:
            New TruthoundConfig instance.
        """
        env = EnvReader(prefix)

        # Load platform-specific configs
        platforms: dict[str, PlatformConfig] = {}
        for platform in ("airflow", "dagster", "prefect", "dbt"):
            platform_upper = platform.upper()
            platform_env = EnvReader(f"{prefix}_{platform_upper}")

            enabled = platform_env.get_bool("ENABLED")
            if enabled is not None or platform_env.get("CONNECTION_ID") is not None:
                platforms[platform] = PlatformConfig(
                    platform=platform,
                    enabled=enabled if enabled is not None else True,
                    connection_id=platform_env.get("CONNECTION_ID"),
                    timeout_seconds=platform_env.get_int("TIMEOUT") or 300,
                    retry_count=platform_env.get_int("RETRY_COUNT") or 3,
                    retry_delay_seconds=platform_env.get_int("RETRY_DELAY") or 10,
                )

        return cls(
            debug=env.get_bool("DEBUG") or False,
            log_level=env.get("LOG_LEVEL") or "INFO",
            default_timeout_seconds=env.get_int("TIMEOUT") or 300,
            default_sample_size=env.get_int("SAMPLE_SIZE"),
            fail_fast=env.get_bool("FAIL_FAST") if env.get("FAIL_FAST") else True,
            platforms=platforms,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Create configuration from a file.

        Args:
            path: Path to configuration file.

        Returns:
            New TruthoundConfig instance.
        """
        config_path = Path(path)
        data = load_config_file(config_path)
        return cls.from_dict(data)

    @classmethod
    def load(
        cls,
        config_file: Path | str | None = None,
        env_prefix: str = DEFAULT_ENV_PREFIX,
        search_config: bool = True,
    ) -> Self:
        """Load configuration with automatic discovery and merging.

        This method provides the most convenient way to load configuration,
        automatically discovering config files and merging with environment
        variables.

        Precedence (highest to lowest):
            1. Environment variables
            2. Specified config file
            3. Auto-discovered config file
            4. Default values

        Args:
            config_file: Explicit config file path.
            env_prefix: Environment variable prefix.
            search_config: Whether to search for config file.

        Returns:
            Merged TruthoundConfig instance.
        """
        # Start with defaults
        base_config: dict[str, Any] = {}

        # Load from file if specified or discovered
        file_path: Path | None = None
        if config_file:
            file_path = Path(config_file)
        elif search_config:
            file_path = find_config_file()

        if file_path and file_path.exists():
            base_config = load_config_file(file_path)

        # Create config from file data
        file_config = cls.from_dict(base_config)

        # Load environment config
        env_config = cls.from_env(env_prefix)

        # Merge configs (env takes precedence)
        return cls._merge_configs(file_config, env_config)

    @classmethod
    def _merge_configs(cls, base: TruthoundConfig, override: TruthoundConfig) -> Self:
        """Merge two configurations, with override taking precedence.

        Args:
            base: Base configuration.
            override: Override configuration (takes precedence).

        Returns:
            Merged configuration.
        """
        # Merge platforms
        merged_platforms = dict(base.platforms)
        for name, platform_config in override.platforms.items():
            if name in merged_platforms:
                # Merge platform extras
                base_platform = merged_platforms[name]
                merged_platforms[name] = PlatformConfig(
                    platform=name,
                    enabled=platform_config.enabled,
                    connection_id=platform_config.connection_id or base_platform.connection_id,
                    timeout_seconds=platform_config.timeout_seconds,
                    retry_count=platform_config.retry_count,
                    retry_delay_seconds=platform_config.retry_delay_seconds,
                    extra={**base_platform.extra, **platform_config.extra},
                )
            else:
                merged_platforms[name] = platform_config

        return cls(
            debug=override.debug or base.debug,
            log_level=override.log_level if override.log_level != "INFO" else base.log_level,
            default_timeout_seconds=(
                override.default_timeout_seconds
                if override.default_timeout_seconds != 300
                else base.default_timeout_seconds
            ),
            default_sample_size=override.default_sample_size or base.default_sample_size,
            fail_fast=override.fail_fast and base.fail_fast,
            platforms=merged_platforms,
            extra={**base.extra, **override.extra},
        )

    def with_platform(self, platform_config: PlatformConfig) -> TruthoundConfig:
        """Create a new config with an updated platform configuration.

        Args:
            platform_config: Platform configuration to add/update.

        Returns:
            New TruthoundConfig with updated platform.
        """
        new_platforms = dict(self.platforms)
        new_platforms[platform_config.platform] = platform_config
        return TruthoundConfig(
            debug=self.debug,
            log_level=self.log_level,
            default_timeout_seconds=self.default_timeout_seconds,
            default_sample_size=self.default_sample_size,
            fail_fast=self.fail_fast,
            platforms=new_platforms,
            extra=self.extra,
        )


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_config(config: TruthoundConfig) -> list[str]:
    """Validate configuration and return list of issues.

    Args:
        config: Configuration to validate.

    Returns:
        List of validation issue messages (empty if valid).
    """
    issues: list[str] = []

    # Validate log level
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if config.log_level.upper() not in valid_log_levels:
        issues.append(
            f"Invalid log_level: {config.log_level}. "
            f"Must be one of: {', '.join(valid_log_levels)}"
        )

    # Validate timeout
    if config.default_timeout_seconds <= 0:
        issues.append(
            f"Invalid default_timeout_seconds: {config.default_timeout_seconds}. "
            "Must be positive."
        )

    # Validate sample size if set
    if config.default_sample_size is not None and config.default_sample_size <= 0:
        issues.append(
            f"Invalid default_sample_size: {config.default_sample_size}. "
            "Must be positive."
        )

    # Validate platform configs
    for name, platform in config.platforms.items():
        if platform.timeout_seconds <= 0:
            issues.append(
                f"Invalid timeout_seconds for platform {name}: "
                f"{platform.timeout_seconds}. Must be positive."
            )
        if platform.retry_count < 0:
            issues.append(
                f"Invalid retry_count for platform {name}: "
                f"{platform.retry_count}. Must be non-negative."
            )
        if platform.retry_delay_seconds < 0:
            issues.append(
                f"Invalid retry_delay_seconds for platform {name}: "
                f"{platform.retry_delay_seconds}. Must be non-negative."
            )

    return issues


def require_valid_config(config: TruthoundConfig) -> None:
    """Validate configuration and raise if invalid.

    Args:
        config: Configuration to validate.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    issues = validate_config(config)
    if issues:
        raise ConfigurationError(
            "Invalid configuration",
            details={"issues": issues},
        )
