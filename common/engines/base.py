"""Base types and protocols for Data Quality Engines.

This module defines the core abstractions for data quality engines:
- DataQualityEngine: Protocol for synchronous engines
- AsyncDataQualityEngine: Protocol for asynchronous engines
- EngineCapabilities: Describes what an engine can do
- EngineInfo: Metadata about an engine

Design Principles:
    1. Protocol-based: Use structural typing for flexible implementations
    2. Engine-agnostic: No dependency on specific data quality libraries
    3. Capability-aware: Engines declare what operations they support
    4. Type-safe: Full type annotations for static analysis

Example:
    >>> class MyEngine:
    ...     @property
    ...     def engine_name(self) -> str:
    ...         return "my_engine"
    ...     @property
    ...     def engine_version(self) -> str:
    ...         return "1.0.0"
    ...     def check(self, data, rules, **kwargs):
    ...         # Implementation
    ...         return CheckResult(status=CheckStatus.PASSED)
    ...
    >>> # Verify it satisfies the protocol
    >>> assert isinstance(MyEngine(), DataQualityEngine)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from common.base import CheckResult, LearnResult, ProfileResult


# =============================================================================
# Engine Capabilities
# =============================================================================


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Describes the capabilities of a data quality engine.

    This allows consumers to check what operations an engine supports
    before attempting to use them.

    Attributes:
        supports_check: Whether the engine supports validation checks.
        supports_profile: Whether the engine supports data profiling.
        supports_learn: Whether the engine supports rule learning.
        supports_async: Whether the engine supports async operations.
        supports_streaming: Whether the engine supports streaming data.
        supported_data_types: List of supported data types (e.g., "polars", "pandas").
        supported_rule_types: List of supported rule types.
        extra: Additional capability flags.

    Example:
        >>> caps = EngineCapabilities(
        ...     supports_check=True,
        ...     supports_profile=True,
        ...     supports_learn=False,
        ...     supported_data_types=("polars", "pandas"),
        ... )
        >>> if caps.supports_profile:
        ...     result = engine.profile(data)
    """

    supports_check: bool = True
    supports_profile: bool = True
    supports_learn: bool = True
    supports_async: bool = False
    supports_streaming: bool = False
    supported_data_types: tuple[str, ...] = ("polars",)
    supported_rule_types: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def supports_data_type(self, data_type: str) -> bool:
        """Check if a data type is supported.

        Args:
            data_type: Data type name (e.g., "polars", "pandas").

        Returns:
            True if supported, False otherwise.
        """
        return data_type.lower() in [dt.lower() for dt in self.supported_data_types]

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if a rule type is supported.

        Args:
            rule_type: Rule type name (e.g., "not_null", "unique").

        Returns:
            True if supported or if supported_rule_types is empty (all supported).
        """
        if not self.supported_rule_types:
            return True
        return rule_type.lower() in [rt.lower() for rt in self.supported_rule_types]


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """Metadata about a data quality engine.

    Provides detailed information about an engine for introspection
    and documentation purposes.

    Attributes:
        name: Engine name.
        version: Engine version.
        description: Human-readable description.
        homepage: URL to engine documentation/homepage.
        capabilities: Engine capabilities.
        extra: Additional metadata.

    Example:
        >>> info = engine.get_info()
        >>> print(f"Using {info.name} v{info.version}")
    """

    name: str
    version: str
    description: str = ""
    homepage: str = ""
    capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "homepage": self.homepage,
            "capabilities": {
                "supports_check": self.capabilities.supports_check,
                "supports_profile": self.capabilities.supports_profile,
                "supports_learn": self.capabilities.supports_learn,
                "supports_async": self.capabilities.supports_async,
                "supports_streaming": self.capabilities.supports_streaming,
                "supported_data_types": list(self.capabilities.supported_data_types),
                "supported_rule_types": list(self.capabilities.supported_rule_types),
            },
            "extra": self.extra,
        }


# =============================================================================
# Data Quality Engine Protocol
# =============================================================================


@runtime_checkable
class DataQualityEngine(Protocol):
    """Protocol for synchronous data quality engines.

    Any class that implements these methods can be used as a data quality
    engine, regardless of the underlying library (Truthound, Great Expectations,
    Pandera, custom, etc.).

    Required Properties:
        engine_name: Unique name identifying the engine.
        engine_version: Version string of the engine.

    Required Methods:
        check: Execute validation checks on data.
        profile: Profile data characteristics.
        learn: Learn validation rules from data.

    Optional Methods:
        get_info: Return detailed engine information.
        get_capabilities: Return engine capabilities.

    Example:
        >>> class MyEngine:
        ...     @property
        ...     def engine_name(self) -> str:
        ...         return "my_engine"
        ...
        ...     @property
        ...     def engine_version(self) -> str:
        ...         return "1.0.0"
        ...
        ...     def check(self, data, rules, **kwargs) -> CheckResult:
        ...         # Validate data against rules
        ...         ...
        ...
        ...     def profile(self, data, **kwargs) -> ProfileResult:
        ...         # Profile data
        ...         ...
        ...
        ...     def learn(self, data, **kwargs) -> LearnResult:
        ...         # Learn rules from data
        ...         ...
    """

    @property
    def engine_name(self) -> str:
        """Return the unique name of this engine.

        This should be a short, lowercase identifier that uniquely
        identifies the engine (e.g., "truthound", "great_expectations").

        Returns:
            Engine name string.
        """
        ...

    @property
    def engine_version(self) -> str:
        """Return the version of this engine.

        Should follow semantic versioning (e.g., "1.0.0").

        Returns:
            Version string.
        """
        ...

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation checks on the data.

        Args:
            data: Data to validate (typically a DataFrame).
            rules: Sequence of validation rule dictionaries.
            **kwargs: Engine-specific parameters.

        Returns:
            CheckResult with validation outcomes.

        Raises:
            ValidationExecutionError: If validation execution fails.
            UnsupportedOperationError: If check is not supported.
        """
        ...

    def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Profile the data to understand its characteristics.

        Args:
            data: Data to profile (typically a DataFrame).
            **kwargs: Engine-specific parameters.

        Returns:
            ProfileResult with profiling outcomes.

        Raises:
            ValidationExecutionError: If profiling execution fails.
            UnsupportedOperationError: If profile is not supported.
        """
        ...

    def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Learn validation rules from the data.

        Args:
            data: Data to learn from (typically a DataFrame).
            **kwargs: Engine-specific parameters.

        Returns:
            LearnResult with learned rules.

        Raises:
            ValidationExecutionError: If learning execution fails.
            UnsupportedOperationError: If learn is not supported.
        """
        ...


@runtime_checkable
class AsyncDataQualityEngine(Protocol):
    """Protocol for asynchronous data quality engines.

    Async version of DataQualityEngine for engines that support
    asynchronous operations.

    Example:
        >>> async def validate_data():
        ...     engine = AsyncTruthoundEngine()
        ...     result = await engine.check(data, rules)
        ...     return result
    """

    @property
    def engine_name(self) -> str:
        """Return the unique name of this engine."""
        ...

    @property
    def engine_version(self) -> str:
        """Return the version of this engine."""
        ...

    async def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation checks asynchronously.

        Args:
            data: Data to validate.
            rules: Sequence of validation rule dictionaries.
            **kwargs: Engine-specific parameters.

        Returns:
            CheckResult with validation outcomes.
        """
        ...

    async def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Profile the data asynchronously.

        Args:
            data: Data to profile.
            **kwargs: Engine-specific parameters.

        Returns:
            ProfileResult with profiling outcomes.
        """
        ...

    async def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Learn validation rules asynchronously.

        Args:
            data: Data to learn from.
            **kwargs: Engine-specific parameters.

        Returns:
            LearnResult with learned rules.
        """
        ...


# =============================================================================
# Mixin Classes
# =============================================================================


class EngineInfoMixin:
    """Mixin that provides get_info and get_capabilities methods.

    Use this mixin to add standard info/capabilities methods to your engine.

    Example:
        >>> class MyEngine(EngineInfoMixin):
        ...     @property
        ...     def engine_name(self) -> str:
        ...         return "my_engine"
        ...
        ...     @property
        ...     def engine_version(self) -> str:
        ...         return "1.0.0"
        ...
        ...     def _get_capabilities(self) -> EngineCapabilities:
        ...         return EngineCapabilities(supports_learn=False)
    """

    engine_name: str
    engine_version: str

    def _get_capabilities(self) -> EngineCapabilities:
        """Override to provide custom capabilities.

        Returns:
            EngineCapabilities instance.
        """
        return EngineCapabilities()

    def _get_description(self) -> str:
        """Override to provide custom description.

        Returns:
            Description string.
        """
        return f"{self.engine_name} data quality engine"

    def _get_homepage(self) -> str:
        """Override to provide custom homepage.

        Returns:
            Homepage URL string.
        """
        return ""

    def get_capabilities(self) -> EngineCapabilities:
        """Return the capabilities of this engine.

        Returns:
            EngineCapabilities instance.
        """
        return self._get_capabilities()

    def get_info(self) -> EngineInfo:
        """Return detailed information about this engine.

        Returns:
            EngineInfo instance.
        """
        return EngineInfo(
            name=self.engine_name,
            version=self.engine_version,
            description=self._get_description(),
            homepage=self._get_homepage(),
            capabilities=self._get_capabilities(),
        )
