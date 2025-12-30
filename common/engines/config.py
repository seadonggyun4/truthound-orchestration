"""Engine Configuration System.

This module provides an extensible, enterprise-grade configuration system
for data quality engines with support for:
- Type-safe configuration with validation
- Environment-based configuration loading
- Multi-source configuration merging
- Serialization/deserialization (JSON, YAML, TOML)
- Builder pattern for fluent configuration

Design Principles:
    1. Immutable configurations (frozen dataclasses)
    2. Type-safe with runtime validation
    3. Environment-aware loading (dev, staging, prod)
    4. Extensible for custom engine configurations
    5. Serializable for persistence and transport

Example:
    >>> from common.engines.config import (
    ...     ConfigBuilder,
    ...     ConfigLoader,
    ...     ConfigValidator,
    ... )
    >>>
    >>> # Build configuration
    >>> config = (
    ...     ConfigBuilder(TruthoundEngineConfig)
    ...     .with_parallel(True, max_workers=4)
    ...     .with_health_check(enabled=True, interval_seconds=30.0)
    ...     .with_timeouts(startup_seconds=60.0, shutdown_seconds=30.0)
    ...     .build()
    ... )
    >>>
    >>> # Load from environment
    >>> config = ConfigLoader.from_env("TRUTHOUND_", TruthoundEngineConfig)
    >>>
    >>> # Validate configuration
    >>> validator = ConfigValidator()
    >>> errors = validator.validate(config)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# =============================================================================
# Type Variables
# =============================================================================

ConfigT = TypeVar("ConfigT", bound="BaseEngineConfig")


# =============================================================================
# Enums
# =============================================================================


class ConfigSource(Enum):
    """Source of configuration values."""

    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    PROGRAMMATIC = "programmatic"
    OVERRIDE = "override"


class ConfigEnvironment(Enum):
    """Deployment environment for configuration."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def from_string(cls, value: str) -> ConfigEnvironment:
        """Create from string value."""
        value_lower = value.lower()
        mapping = {
            "dev": cls.DEVELOPMENT,
            "development": cls.DEVELOPMENT,
            "test": cls.TESTING,
            "testing": cls.TESTING,
            "stage": cls.STAGING,
            "staging": cls.STAGING,
            "prod": cls.PRODUCTION,
            "production": cls.PRODUCTION,
        }
        return mapping.get(value_lower, cls.DEVELOPMENT)


class MergeStrategy(Enum):
    """Strategy for merging configuration values."""

    OVERRIDE = "override"  # Later values override earlier
    DEEP_MERGE = "deep_merge"  # Merge nested dicts
    KEEP_FIRST = "keep_first"  # Keep first non-None value
    KEEP_LAST = "keep_last"  # Keep last non-None value


# =============================================================================
# Exceptions
# =============================================================================


class ConfigurationError(Exception):
    """Base exception for configuration errors."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        config_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Human-readable error description.
            field_name: Name of the invalid field.
            config_type: Type of configuration.
            details: Additional error context.
        """
        super().__init__(message)
        self.message = message
        self.field_name = field_name
        self.config_type = config_type
        self.details = details or {}


class ConfigValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""

    def __init__(
        self,
        message: str,
        *,
        errors: Sequence[ValidationError] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Human-readable error description.
            errors: List of validation errors.
            **kwargs: Additional arguments.
        """
        super().__init__(message, **kwargs)
        self.errors = list(errors) if errors else []


class ConfigLoadError(ConfigurationError):
    """Exception raised when configuration loading fails."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize load error.

        Args:
            message: Human-readable error description.
            source: Configuration source that failed.
            **kwargs: Additional arguments.
        """
        super().__init__(message, **kwargs)
        self.source = source


# =============================================================================
# Validation Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class ValidationError:
    """Represents a single validation error.

    Attributes:
        field: Name of the field that failed validation.
        message: Human-readable error message.
        value: The invalid value.
        constraint: The constraint that was violated.
    """

    field: str
    message: str
    value: Any = None
    constraint: str | None = None

    def __str__(self) -> str:
        """Return string representation."""
        if self.constraint:
            return f"{self.field}: {self.message} (constraint: {self.constraint})"
        return f"{self.field}: {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        is_valid: Whether the configuration is valid.
        errors: List of validation errors.
        warnings: List of validation warnings.
    """

    is_valid: bool
    errors: tuple[ValidationError, ...] = field(default_factory=tuple)
    warnings: tuple[ValidationError, ...] = field(default_factory=tuple)

    @classmethod
    def valid(cls) -> ValidationResult:
        """Create a valid result."""
        return cls(is_valid=True)

    @classmethod
    def invalid(
        cls,
        errors: Sequence[ValidationError],
        warnings: Sequence[ValidationError] | None = None,
    ) -> ValidationResult:
        """Create an invalid result."""
        return cls(
            is_valid=False,
            errors=tuple(errors),
            warnings=tuple(warnings or []),
        )


# =============================================================================
# Base Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class BaseEngineConfig:
    """Base configuration for all data quality engines.

    This is the foundation for all engine configurations. It provides
    common settings for lifecycle management, health checking, and
    operational parameters.

    All configurations are immutable (frozen dataclass). Use builder
    methods to create modified copies.

    Attributes:
        auto_start: Whether to automatically start engine on creation.
        auto_stop: Whether to automatically stop on context exit.
        health_check_enabled: Whether to enable health checks.
        health_check_interval_seconds: Interval between health checks.
        startup_timeout_seconds: Maximum time for engine startup.
        shutdown_timeout_seconds: Maximum time for engine shutdown.
        max_retries_on_failure: Max retries for failed operations.
        fail_fast: Whether to fail immediately on first error.
        tags: Tags for categorization and filtering.
        metadata: Additional configuration metadata.

    Example:
        >>> config = BaseEngineConfig(
        ...     auto_start=True,
        ...     health_check_enabled=True,
        ...     startup_timeout_seconds=30.0,
        ... )
    """

    auto_start: bool = False
    auto_stop: bool = True
    health_check_enabled: bool = True
    health_check_interval_seconds: float = 30.0
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    max_retries_on_failure: int = 3
    fail_fast: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If validation fails.
        """
        if self.startup_timeout_seconds < 0:
            raise ValueError("startup_timeout_seconds must be non-negative")
        if self.shutdown_timeout_seconds < 0:
            raise ValueError("shutdown_timeout_seconds must be non-negative")
        if self.health_check_interval_seconds < 0:
            raise ValueError("health_check_interval_seconds must be non-negative")
        if self.max_retries_on_failure < 0:
            raise ValueError("max_retries_on_failure must be non-negative")

    # =========================================================================
    # Builder Methods
    # =========================================================================

    def with_auto_start(self: ConfigT, enabled: bool) -> ConfigT:
        """Create config with auto_start setting.

        Args:
            enabled: Whether to auto-start.

        Returns:
            New configuration with updated setting.
        """
        return self._copy_with(auto_start=enabled)

    def with_auto_stop(self: ConfigT, enabled: bool) -> ConfigT:
        """Create config with auto_stop setting.

        Args:
            enabled: Whether to auto-stop.

        Returns:
            New configuration with updated setting.
        """
        return self._copy_with(auto_stop=enabled)

    def with_health_check(
        self: ConfigT,
        enabled: bool = True,
        interval_seconds: float | None = None,
    ) -> ConfigT:
        """Create config with health check settings.

        Args:
            enabled: Whether health checks are enabled.
            interval_seconds: Interval between health checks.

        Returns:
            New configuration with updated settings.
        """
        updates: dict[str, Any] = {"health_check_enabled": enabled}
        if interval_seconds is not None:
            updates["health_check_interval_seconds"] = interval_seconds
        return self._copy_with(**updates)

    def with_timeouts(
        self: ConfigT,
        startup_seconds: float | None = None,
        shutdown_seconds: float | None = None,
    ) -> ConfigT:
        """Create config with timeout settings.

        Args:
            startup_seconds: Maximum startup time.
            shutdown_seconds: Maximum shutdown time.

        Returns:
            New configuration with updated settings.
        """
        updates: dict[str, Any] = {}
        if startup_seconds is not None:
            updates["startup_timeout_seconds"] = startup_seconds
        if shutdown_seconds is not None:
            updates["shutdown_timeout_seconds"] = shutdown_seconds
        return self._copy_with(**updates)

    def with_retries(self: ConfigT, max_retries: int) -> ConfigT:
        """Create config with retry settings.

        Args:
            max_retries: Maximum number of retries.

        Returns:
            New configuration with updated setting.
        """
        return self._copy_with(max_retries_on_failure=max_retries)

    def with_fail_fast(self: ConfigT, enabled: bool) -> ConfigT:
        """Create config with fail_fast setting.

        Args:
            enabled: Whether to fail fast.

        Returns:
            New configuration with updated setting.
        """
        return self._copy_with(fail_fast=enabled)

    def with_tags(self: ConfigT, *tags: str) -> ConfigT:
        """Create config with additional tags.

        Args:
            *tags: Tags to add.

        Returns:
            New configuration with updated tags.
        """
        return self._copy_with(tags=self.tags | frozenset(tags))

    def with_metadata(self: ConfigT, **metadata: Any) -> ConfigT:
        """Create config with additional metadata.

        Args:
            **metadata: Metadata key-value pairs.

        Returns:
            New configuration with updated metadata.
        """
        return self._copy_with(metadata={**self.metadata, **metadata})

    def _copy_with(self: ConfigT, **updates: Any) -> ConfigT:
        """Create a copy with updated fields.

        Args:
            **updates: Fields to update.

        Returns:
            New configuration with updated fields.
        """
        current_values = {}
        for f in fields(self):
            current_values[f.name] = getattr(self, f.name)
        current_values.update(updates)
        return self.__class__(**current_values)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, frozenset):
                value = list(value)
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls: type[ConfigT], data: dict[str, Any]) -> ConfigT:
        """Create from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            Configuration instance.
        """
        # Convert list back to frozenset for tags
        if "tags" in data and isinstance(data["tags"], list):
            data = dict(data)
            data["tags"] = frozenset(data["tags"])

        # Filter to only known fields
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    def merge_with(
        self: ConfigT,
        other: ConfigT,
        strategy: MergeStrategy = MergeStrategy.OVERRIDE,
    ) -> ConfigT:
        """Merge with another configuration.

        Args:
            other: Configuration to merge with.
            strategy: Merge strategy to use.

        Returns:
            Merged configuration.
        """
        if strategy == MergeStrategy.OVERRIDE:
            # Other overrides self completely
            return other
        elif strategy == MergeStrategy.KEEP_FIRST:
            # Self takes precedence
            return self
        elif strategy == MergeStrategy.KEEP_LAST:
            # Other takes precedence (same as override for two configs)
            return other
        elif strategy == MergeStrategy.DEEP_MERGE:
            # Merge field by field
            merged: dict[str, Any] = {}
            for f in fields(self):
                self_value = getattr(self, f.name)
                other_value = getattr(other, f.name)

                # For dicts, deep merge
                if isinstance(self_value, dict) and isinstance(other_value, dict):
                    merged[f.name] = {**self_value, **other_value}
                # For frozensets, union
                elif isinstance(self_value, frozenset) and isinstance(
                    other_value, frozenset
                ):
                    merged[f.name] = self_value | other_value
                # For other types, use other's value if not default
                else:
                    # Get default value
                    default_value = f.default if f.default is not f.default_factory else None  # type: ignore[attr-defined, comparison-overlap]
                    if other_value != default_value:
                        merged[f.name] = other_value
                    else:
                        merged[f.name] = self_value

            return self.__class__(**merged)

        return other  # Default to override


# =============================================================================
# Validation Protocol
# =============================================================================


class ConfigValidatorProtocol(ABC):
    """Protocol for configuration validators."""

    @abstractmethod
    def validate(self, config: BaseEngineConfig) -> ValidationResult:
        """Validate a configuration.

        Args:
            config: Configuration to validate.

        Returns:
            Validation result.
        """
        ...

    @abstractmethod
    def validate_field(
        self,
        field_name: str,
        value: Any,
        config_type: type[BaseEngineConfig],
    ) -> list[ValidationError]:
        """Validate a single field.

        Args:
            field_name: Name of the field.
            value: Value to validate.
            config_type: Type of configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        ...


# =============================================================================
# Configuration Validator
# =============================================================================


@dataclass
class FieldConstraint:
    """Constraint for a configuration field.

    Attributes:
        field_name: Name of the field.
        validator: Validation function.
        message: Error message template.
        constraint_name: Name of the constraint.
    """

    field_name: str
    validator: Callable[[Any], bool]
    message: str
    constraint_name: str | None = None


class ConfigValidator:
    """Validates engine configurations.

    Provides comprehensive validation for configuration objects with
    support for:
    - Built-in constraints for common validations
    - Custom constraints
    - Field-level and configuration-level validation
    - Warnings for non-critical issues

    Example:
        >>> validator = ConfigValidator()
        >>> validator.add_constraint(FieldConstraint(
        ...     field_name="max_workers",
        ...     validator=lambda x: x is None or x > 0,
        ...     message="max_workers must be positive",
        ... ))
        >>> result = validator.validate(config)
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(error)
    """

    def __init__(self) -> None:
        """Initialize validator with default constraints."""
        self._constraints: list[FieldConstraint] = []
        self._config_validators: list[Callable[[BaseEngineConfig], list[ValidationError]]] = []
        self._register_default_constraints()

    def _register_default_constraints(self) -> None:
        """Register default validation constraints."""
        # Timeout constraints
        self.add_constraint(
            FieldConstraint(
                field_name="startup_timeout_seconds",
                validator=lambda x: isinstance(x, (int, float)) and x >= 0,
                message="startup_timeout_seconds must be a non-negative number",
                constraint_name="non_negative_timeout",
            )
        )
        self.add_constraint(
            FieldConstraint(
                field_name="shutdown_timeout_seconds",
                validator=lambda x: isinstance(x, (int, float)) and x >= 0,
                message="shutdown_timeout_seconds must be a non-negative number",
                constraint_name="non_negative_timeout",
            )
        )
        self.add_constraint(
            FieldConstraint(
                field_name="health_check_interval_seconds",
                validator=lambda x: isinstance(x, (int, float)) and x >= 0,
                message="health_check_interval_seconds must be a non-negative number",
                constraint_name="non_negative_interval",
            )
        )

        # Retry constraints
        self.add_constraint(
            FieldConstraint(
                field_name="max_retries_on_failure",
                validator=lambda x: isinstance(x, int) and x >= 0,
                message="max_retries_on_failure must be a non-negative integer",
                constraint_name="non_negative_retries",
            )
        )

        # Boolean constraints
        for field_name in ["auto_start", "auto_stop", "health_check_enabled", "fail_fast"]:
            self.add_constraint(
                FieldConstraint(
                    field_name=field_name,
                    validator=lambda x: isinstance(x, bool),
                    message=f"{field_name} must be a boolean",
                    constraint_name="boolean_type",
                )
            )

    def add_constraint(self, constraint: FieldConstraint) -> None:
        """Add a field constraint.

        Args:
            constraint: Constraint to add.
        """
        self._constraints.append(constraint)

    def add_config_validator(
        self,
        validator: Callable[[BaseEngineConfig], list[ValidationError]],
    ) -> None:
        """Add a configuration-level validator.

        Args:
            validator: Validation function.
        """
        self._config_validators.append(validator)

    def validate(self, config: BaseEngineConfig) -> ValidationResult:
        """Validate a configuration.

        Args:
            config: Configuration to validate.

        Returns:
            Validation result.
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # Validate fields
        for f in fields(config):
            value = getattr(config, f.name)
            field_errors = self.validate_field(f.name, value, type(config))
            errors.extend(field_errors)

        # Run config-level validators
        for validator in self._config_validators:
            try:
                validator_errors = validator(config)
                errors.extend(validator_errors)
            except Exception as e:
                errors.append(
                    ValidationError(
                        field="__config__",
                        message=f"Config validator failed: {e}",
                    )
                )

        # Add warnings for potentially problematic configurations
        warnings.extend(self._check_warnings(config))

        if errors:
            return ValidationResult.invalid(errors, warnings)
        return ValidationResult(is_valid=True, warnings=tuple(warnings))

    def validate_field(
        self,
        field_name: str,
        value: Any,
        config_type: type[BaseEngineConfig],
    ) -> list[ValidationError]:
        """Validate a single field.

        Args:
            field_name: Name of the field.
            value: Value to validate.
            config_type: Type of configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[ValidationError] = []

        for constraint in self._constraints:
            if constraint.field_name != field_name:
                continue

            try:
                if not constraint.validator(value):
                    errors.append(
                        ValidationError(
                            field=field_name,
                            message=constraint.message,
                            value=value,
                            constraint=constraint.constraint_name,
                        )
                    )
            except Exception as e:
                errors.append(
                    ValidationError(
                        field=field_name,
                        message=f"Validation error: {e}",
                        value=value,
                        constraint=constraint.constraint_name,
                    )
                )

        return errors

    def _check_warnings(self, config: BaseEngineConfig) -> list[ValidationError]:
        """Check for warning conditions.

        Args:
            config: Configuration to check.

        Returns:
            List of warning validation errors.
        """
        warnings: list[ValidationError] = []

        # Warning: very short health check interval
        if (
            config.health_check_enabled
            and config.health_check_interval_seconds < 5.0
        ):
            warnings.append(
                ValidationError(
                    field="health_check_interval_seconds",
                    message="Health check interval is very short (<5s), may cause performance issues",
                    value=config.health_check_interval_seconds,
                )
            )

        # Warning: very long startup timeout
        if config.startup_timeout_seconds > 300:
            warnings.append(
                ValidationError(
                    field="startup_timeout_seconds",
                    message="Startup timeout is very long (>5min), consider reducing",
                    value=config.startup_timeout_seconds,
                )
            )

        # Warning: no retries but auto_start enabled
        if config.auto_start and config.max_retries_on_failure == 0:
            warnings.append(
                ValidationError(
                    field="max_retries_on_failure",
                    message="Auto-start is enabled but no retries configured",
                    value=config.max_retries_on_failure,
                )
            )

        return warnings


# =============================================================================
# Configuration Loader
# =============================================================================


class ConfigLoader(Generic[ConfigT]):
    """Loads configuration from various sources.

    Supports loading from:
    - Environment variables
    - JSON files
    - YAML files
    - TOML files
    - Dictionaries

    Example:
        >>> loader = ConfigLoader(TruthoundEngineConfig)
        >>> config = loader.from_env("TRUTHOUND_")
        >>> config = loader.from_file("config.yaml")
        >>> config = loader.from_dict({"parallel": True})
    """

    def __init__(
        self,
        config_class: type[ConfigT],
        validator: ConfigValidator | None = None,
    ) -> None:
        """Initialize loader.

        Args:
            config_class: Configuration class to load.
            validator: Optional validator for loaded configurations.
        """
        self._config_class = config_class
        self._validator = validator or ConfigValidator()

    def from_env(
        self,
        prefix: str,
        *,
        defaults: ConfigT | None = None,
        validate: bool = True,
    ) -> ConfigT:
        """Load configuration from environment variables.

        Environment variables should be named: {prefix}{FIELD_NAME}
        For example, with prefix "TRUTHOUND_":
        - TRUTHOUND_AUTO_START=true
        - TRUTHOUND_MAX_WORKERS=4
        - TRUTHOUND_HEALTH_CHECK_INTERVAL_SECONDS=30

        Args:
            prefix: Prefix for environment variables.
            defaults: Default configuration to use as base.
            validate: Whether to validate loaded configuration.

        Returns:
            Loaded configuration.

        Raises:
            ConfigLoadError: If loading fails.
            ConfigValidationError: If validation fails.
        """
        data: dict[str, Any] = {}

        for f in fields(self._config_class):
            env_name = f"{prefix}{f.name.upper()}"
            env_value = os.environ.get(env_name)

            if env_value is not None:
                try:
                    data[f.name] = self._parse_env_value(env_value, f.type)
                except Exception as e:
                    raise ConfigLoadError(
                        f"Failed to parse environment variable {env_name}: {e}",
                        source="environment",
                        field_name=f.name,
                    ) from e

        # Merge with defaults
        if defaults:
            default_dict = defaults.to_dict()
            data = {**default_dict, **data}

        try:
            config = self._config_class.from_dict(data) if data else self._config_class()
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to create configuration: {e}",
                source="environment",
            ) from e

        if validate:
            self._validate_or_raise(config)

        return config

    def from_file(
        self,
        path: str | Path,
        *,
        defaults: ConfigT | None = None,
        validate: bool = True,
    ) -> ConfigT:
        """Load configuration from a file.

        Supports JSON, YAML, and TOML files based on extension.

        Args:
            path: Path to configuration file.
            defaults: Default configuration to use as base.
            validate: Whether to validate loaded configuration.

        Returns:
            Loaded configuration.

        Raises:
            ConfigLoadError: If loading fails.
            ConfigValidationError: If validation fails.
        """
        path = Path(path)

        if not path.exists():
            raise ConfigLoadError(
                f"Configuration file not found: {path}",
                source=str(path),
            )

        try:
            data = self._load_file(path)
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to load configuration file: {e}",
                source=str(path),
            ) from e

        # Merge with defaults
        if defaults:
            default_dict = defaults.to_dict()
            data = {**default_dict, **data}

        try:
            config = self._config_class.from_dict(data)
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to create configuration from file: {e}",
                source=str(path),
            ) from e

        if validate:
            self._validate_or_raise(config)

        return config

    def from_dict(
        self,
        data: dict[str, Any],
        *,
        defaults: ConfigT | None = None,
        validate: bool = True,
    ) -> ConfigT:
        """Load configuration from a dictionary.

        Args:
            data: Configuration dictionary.
            defaults: Default configuration to use as base.
            validate: Whether to validate loaded configuration.

        Returns:
            Loaded configuration.

        Raises:
            ConfigLoadError: If loading fails.
            ConfigValidationError: If validation fails.
        """
        # Merge with defaults
        if defaults:
            default_dict = defaults.to_dict()
            data = {**default_dict, **data}

        try:
            config = self._config_class.from_dict(data)
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to create configuration from dictionary: {e}",
                source="dict",
            ) from e

        if validate:
            self._validate_or_raise(config)

        return config

    def _parse_env_value(self, value: str, type_hint: Any) -> Any:
        """Parse environment variable value to correct type.

        Args:
            value: String value from environment.
            type_hint: Expected type.

        Returns:
            Parsed value.
        """
        # Get actual type from string representation
        type_str = str(type_hint).lower()

        # Boolean
        if "bool" in type_str:
            return value.lower() in ("true", "1", "yes", "on")

        # Integer
        if "int" in type_str:
            return int(value)

        # Float
        if "float" in type_str:
            return float(value)

        # Frozenset: comma-separated values
        if "frozenset" in type_str:
            if not value:
                return frozenset()
            return frozenset(v.strip() for v in value.split(","))

        # Dictionary from JSON-encoded string
        if "dict" in type_str:
            import json

            return json.loads(value)

        # Default: return as string
        return value

    def _load_file(self, path: Path) -> dict[str, Any]:
        """Load configuration from file based on extension.

        Args:
            path: Path to file.

        Returns:
            Configuration dictionary.
        """
        suffix = path.suffix.lower()

        if suffix == ".json":
            import json

            with path.open() as f:
                return dict(json.load(f))

        elif suffix in (".yaml", ".yml"):
            try:
                import yaml

                with path.open() as f:
                    return dict(yaml.safe_load(f))
            except ImportError as e:
                raise ConfigLoadError(
                    "PyYAML is required for YAML files. Install with: pip install pyyaml",
                    source=str(path),
                ) from e

        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[import-not-found, no-redef]
                except ImportError as e:
                    raise ConfigLoadError(
                        "tomli is required for TOML files on Python < 3.11. "
                        "Install with: pip install tomli",
                        source=str(path),
                    ) from e

            with path.open("rb") as f:
                return dict(tomllib.load(f))

        else:
            raise ConfigLoadError(
                f"Unsupported file format: {suffix}",
                source=str(path),
            )

    def _validate_or_raise(self, config: ConfigT) -> None:
        """Validate configuration and raise if invalid.

        Args:
            config: Configuration to validate.

        Raises:
            ConfigValidationError: If validation fails.
        """
        result = self._validator.validate(config)
        if not result.is_valid:
            raise ConfigValidationError(
                f"Configuration validation failed with {len(result.errors)} error(s)",
                errors=result.errors,
                config_type=self._config_class.__name__,
            )


# =============================================================================
# Configuration Builder
# =============================================================================


class ConfigBuilder(Generic[ConfigT]):
    """Fluent builder for engine configurations.

    Provides a chainable API for building configuration objects.

    Example:
        >>> config = (
        ...     ConfigBuilder(TruthoundEngineConfig)
        ...     .with_auto_start(True)
        ...     .with_parallel(True, max_workers=4)
        ...     .with_health_check(enabled=True, interval_seconds=30.0)
        ...     .with_timeouts(startup_seconds=60.0)
        ...     .with_tags("production", "critical")
        ...     .build()
        ... )
    """

    def __init__(
        self,
        config_class: type[ConfigT],
        base: ConfigT | None = None,
    ) -> None:
        """Initialize builder.

        Args:
            config_class: Configuration class to build.
            base: Optional base configuration to start from.
        """
        self._config_class = config_class
        self._values: dict[str, Any] = {}
        if base:
            self._values = base.to_dict()

    def set(self, field_name: str, value: Any) -> ConfigBuilder[ConfigT]:
        """Set a configuration field value.

        Args:
            field_name: Name of the field.
            value: Value to set.

        Returns:
            Self for chaining.
        """
        self._values[field_name] = value
        return self

    def with_auto_start(self, enabled: bool) -> ConfigBuilder[ConfigT]:
        """Set auto_start.

        Args:
            enabled: Whether to auto-start.

        Returns:
            Self for chaining.
        """
        return self.set("auto_start", enabled)

    def with_auto_stop(self, enabled: bool) -> ConfigBuilder[ConfigT]:
        """Set auto_stop.

        Args:
            enabled: Whether to auto-stop.

        Returns:
            Self for chaining.
        """
        return self.set("auto_stop", enabled)

    def with_health_check(
        self,
        enabled: bool = True,
        interval_seconds: float | None = None,
    ) -> ConfigBuilder[ConfigT]:
        """Set health check settings.

        Args:
            enabled: Whether health checks are enabled.
            interval_seconds: Interval between health checks.

        Returns:
            Self for chaining.
        """
        self._values["health_check_enabled"] = enabled
        if interval_seconds is not None:
            self._values["health_check_interval_seconds"] = interval_seconds
        return self

    def with_timeouts(
        self,
        startup_seconds: float | None = None,
        shutdown_seconds: float | None = None,
    ) -> ConfigBuilder[ConfigT]:
        """Set timeout settings.

        Args:
            startup_seconds: Maximum startup time.
            shutdown_seconds: Maximum shutdown time.

        Returns:
            Self for chaining.
        """
        if startup_seconds is not None:
            self._values["startup_timeout_seconds"] = startup_seconds
        if shutdown_seconds is not None:
            self._values["shutdown_timeout_seconds"] = shutdown_seconds
        return self

    def with_retries(self, max_retries: int) -> ConfigBuilder[ConfigT]:
        """Set retry settings.

        Args:
            max_retries: Maximum number of retries.

        Returns:
            Self for chaining.
        """
        return self.set("max_retries_on_failure", max_retries)

    def with_fail_fast(self, enabled: bool) -> ConfigBuilder[ConfigT]:
        """Set fail_fast.

        Args:
            enabled: Whether to fail fast.

        Returns:
            Self for chaining.
        """
        return self.set("fail_fast", enabled)

    def with_tags(self, *tags: str) -> ConfigBuilder[ConfigT]:
        """Add tags.

        Args:
            *tags: Tags to add.

        Returns:
            Self for chaining.
        """
        existing = self._values.get("tags", frozenset())
        if isinstance(existing, (list, set)):
            existing = frozenset(existing)
        self._values["tags"] = existing | frozenset(tags)
        return self

    def with_metadata(self, **metadata: Any) -> ConfigBuilder[ConfigT]:
        """Add metadata.

        Args:
            **metadata: Metadata key-value pairs.

        Returns:
            Self for chaining.
        """
        existing = self._values.get("metadata", {})
        self._values["metadata"] = {**existing, **metadata}
        return self

    def build(self, *, validate: bool = True) -> ConfigT:
        """Build the configuration.

        Args:
            validate: Whether to validate the configuration.

        Returns:
            Built configuration.

        Raises:
            ConfigValidationError: If validation fails.
        """
        config = self._config_class.from_dict(self._values)

        if validate:
            validator = ConfigValidator()
            result = validator.validate(config)
            if not result.is_valid:
                raise ConfigValidationError(
                    f"Configuration validation failed with {len(result.errors)} error(s)",
                    errors=result.errors,
                    config_type=self._config_class.__name__,
                )

        return config


# =============================================================================
# Configuration Registry
# =============================================================================


class ConfigRegistry:
    """Registry for named configurations.

    Provides a central place to store and retrieve configuration presets.

    Example:
        >>> registry = ConfigRegistry()
        >>> registry.register("production", PRODUCTION_CONFIG)
        >>> registry.register("development", DEVELOPMENT_CONFIG)
        >>> config = registry.get("production")
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._configs: dict[str, BaseEngineConfig] = {}

    def register(
        self,
        name: str,
        config: BaseEngineConfig,
        *,
        override: bool = False,
    ) -> None:
        """Register a configuration.

        Args:
            name: Name for the configuration.
            config: Configuration to register.
            override: Whether to override existing configuration.

        Raises:
            ValueError: If name exists and override is False.
        """
        if name in self._configs and not override:
            raise ValueError(f"Configuration '{name}' already registered")
        self._configs[name] = config

    def get(self, name: str) -> BaseEngineConfig:
        """Get a configuration by name.

        Args:
            name: Name of the configuration.

        Returns:
            Configuration.

        Raises:
            KeyError: If configuration not found.
        """
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        return self._configs[name]

    def get_or_default(
        self,
        name: str,
        default: BaseEngineConfig,
    ) -> BaseEngineConfig:
        """Get a configuration by name or return default.

        Args:
            name: Name of the configuration.
            default: Default configuration if not found.

        Returns:
            Configuration.
        """
        return self._configs.get(name, default)

    def list(self) -> list[str]:
        """List all registered configuration names.

        Returns:
            List of configuration names.
        """
        return list(self._configs.keys())

    def clear(self) -> None:
        """Clear all registered configurations."""
        self._configs.clear()


# =============================================================================
# Environment-Aware Configuration
# =============================================================================


class EnvironmentConfig(Generic[ConfigT]):
    """Environment-aware configuration management.

    Loads different configurations based on deployment environment.

    Example:
        >>> env_config = EnvironmentConfig(TruthoundEngineConfig)
        >>> env_config.register_environment(
        ...     ConfigEnvironment.PRODUCTION,
        ...     PRODUCTION_CONFIG,
        ... )
        >>> config = env_config.get_current()  # Based on ENV_NAME
    """

    def __init__(
        self,
        config_class: type[ConfigT],
        *,
        env_var: str = "ENV_NAME",
        default_env: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT,
    ) -> None:
        """Initialize environment configuration.

        Args:
            config_class: Configuration class.
            env_var: Environment variable to read environment from.
            default_env: Default environment if not set.
        """
        self._config_class = config_class
        self._env_var = env_var
        self._default_env = default_env
        self._configs: dict[ConfigEnvironment, ConfigT] = {}

    def register_environment(
        self,
        environment: ConfigEnvironment,
        config: ConfigT,
    ) -> None:
        """Register configuration for an environment.

        Args:
            environment: Target environment.
            config: Configuration for the environment.
        """
        self._configs[environment] = config

    def get_current_environment(self) -> ConfigEnvironment:
        """Get the current environment from environment variable.

        Returns:
            Current environment.
        """
        env_value = os.environ.get(self._env_var, "")
        if env_value:
            return ConfigEnvironment.from_string(env_value)
        return self._default_env

    def get_current(self) -> ConfigT:
        """Get configuration for the current environment.

        Returns:
            Configuration.

        Raises:
            KeyError: If no configuration for current environment.
        """
        env = self.get_current_environment()
        return self.get(env)

    def get(self, environment: ConfigEnvironment) -> ConfigT:
        """Get configuration for a specific environment.

        Args:
            environment: Target environment.

        Returns:
            Configuration.

        Raises:
            KeyError: If no configuration for environment.
        """
        if environment not in self._configs:
            raise KeyError(f"No configuration for environment: {environment.value}")
        return self._configs[environment]

    def get_or_default(
        self,
        environment: ConfigEnvironment,
    ) -> ConfigT:
        """Get configuration or default for environment.

        Args:
            environment: Target environment.

        Returns:
            Configuration.
        """
        if environment in self._configs:
            return self._configs[environment]
        return self._config_class()


# =============================================================================
# Convenience Functions
# =============================================================================


def load_config(
    config_class: type[ConfigT],
    *,
    env_prefix: str | None = None,
    file_path: str | Path | None = None,
    defaults: ConfigT | None = None,
    validate: bool = True,
) -> ConfigT:
    """Convenience function to load configuration from multiple sources.

    Loads configuration with priority (highest to lowest):
    1. Environment variables (if env_prefix provided)
    2. File (if file_path provided)
    3. Defaults (if provided)
    4. Class defaults

    Args:
        config_class: Configuration class to load.
        env_prefix: Prefix for environment variables.
        file_path: Path to configuration file.
        defaults: Default configuration.
        validate: Whether to validate loaded configuration.

    Returns:
        Loaded configuration.
    """
    loader = ConfigLoader(config_class)
    config = defaults

    # Load from file first (lower priority)
    if file_path:
        config = loader.from_file(file_path, defaults=config, validate=False)

    # Load from environment (higher priority)
    if env_prefix:
        config = loader.from_env(env_prefix, defaults=config, validate=False)

    # If no config yet, use class defaults
    if config is None:
        config = config_class()

    # Validate final configuration
    if validate:
        validator = ConfigValidator()
        result = validator.validate(config)
        if not result.is_valid:
            raise ConfigValidationError(
                f"Configuration validation failed with {len(result.errors)} error(s)",
                errors=result.errors,
                config_type=config_class.__name__,
            )

    return config


def create_config_for_environment(
    config_class: type[ConfigT],
    environment: ConfigEnvironment | str,
) -> ConfigT:
    """Create appropriate configuration for an environment.

    Creates a configuration with environment-appropriate defaults.
    Works with any configuration class that extends EngineConfig.

    Args:
        config_class: Configuration class.
        environment: Target environment.

    Returns:
        Configuration appropriate for the environment.
    """
    if isinstance(environment, str):
        environment = ConfigEnvironment.from_string(environment)

    # Common base parameters that all EngineConfig subclasses have
    if environment == ConfigEnvironment.PRODUCTION:
        return config_class(
            auto_start=True,
            auto_stop=True,
            health_check_enabled=True,
            health_check_interval_seconds=60.0,
            startup_timeout_seconds=60.0,
            shutdown_timeout_seconds=30.0,
            max_retries_on_failure=5,
        )
    elif environment == ConfigEnvironment.STAGING:
        return config_class(
            auto_start=True,
            auto_stop=True,
            health_check_enabled=True,
            health_check_interval_seconds=30.0,
            startup_timeout_seconds=45.0,
            shutdown_timeout_seconds=20.0,
            max_retries_on_failure=3,
        )
    elif environment == ConfigEnvironment.TESTING:
        return config_class(
            auto_start=True,
            auto_stop=True,
            health_check_enabled=False,
            startup_timeout_seconds=5.0,
            shutdown_timeout_seconds=2.0,
            max_retries_on_failure=0,
        )
    else:  # DEVELOPMENT
        return config_class(
            auto_start=False,
            auto_stop=True,
            health_check_enabled=False,
            startup_timeout_seconds=10.0,
            shutdown_timeout_seconds=5.0,
            max_retries_on_failure=1,
        )
