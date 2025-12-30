"""Engine Version Management System.

This module provides comprehensive version handling for data quality engines:
- SemanticVersion: Parsed semantic version with comparison support
- VersionConstraint: Single version constraint (e.g., ">=1.0.0")
- VersionRange: Complex version requirements (e.g., ">=1.0.0,<2.0.0")
- VersionCompatibility: Version compatibility checking
- VersionedEngine: Version-aware engine wrapper
- VersionRegistry: Track engine versions and requirements

Design Principles:
    1. Semantic Versioning: Full SemVer 2.0.0 specification support
    2. Immutable Types: All version types are frozen dataclasses
    3. Composable Constraints: Build complex version requirements from simple parts
    4. Engine-Agnostic: Works with any DataQualityEngine implementation
    5. Extensible: Support for custom compatibility strategies

Example:
    >>> from common.engines.version import SemanticVersion, VersionConstraint
    >>> version = SemanticVersion.parse("1.2.3")
    >>> constraint = VersionConstraint.parse(">=1.0.0")
    >>> constraint.satisfied_by(version)
    True

Version Requirements:
    >>> from common.engines.version import VersionRange, check_version_compatibility
    >>> requirements = VersionRange.parse(">=1.0.0,<2.0.0")
    >>> engine = TruthoundEngine()
    >>> check_version_compatibility(engine, requirements)
    VersionCompatibilityResult(compatible=True, ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from common.engines.base import DataQualityEngine


# =============================================================================
# Exceptions
# =============================================================================


class VersionError(TruthoundIntegrationError):
    """Base exception for version-related errors."""

    pass


class VersionParseError(VersionError):
    """Exception raised when version string parsing fails.

    Attributes:
        version_string: The invalid version string.
        reason: Explanation of why parsing failed.
    """

    def __init__(
        self,
        version_string: str,
        reason: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize version parse error.

        Args:
            version_string: The invalid version string.
            reason: Explanation of why parsing failed.
            details: Optional additional context.
            cause: Optional original exception.
        """
        details = details or {}
        details["version_string"] = version_string
        details["reason"] = reason
        message = f"Failed to parse version '{version_string}': {reason}"
        super().__init__(message, details=details, cause=cause)
        self.version_string = version_string
        self.reason = reason


class VersionConstraintError(VersionError):
    """Exception raised when version constraint parsing fails.

    Attributes:
        constraint_string: The invalid constraint string.
        reason: Explanation of why parsing failed.
    """

    def __init__(
        self,
        constraint_string: str,
        reason: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize version constraint error.

        Args:
            constraint_string: The invalid constraint string.
            reason: Explanation of why parsing failed.
            details: Optional additional context.
            cause: Optional original exception.
        """
        details = details or {}
        details["constraint_string"] = constraint_string
        details["reason"] = reason
        message = f"Failed to parse constraint '{constraint_string}': {reason}"
        super().__init__(message, details=details, cause=cause)
        self.constraint_string = constraint_string
        self.reason = reason


class VersionIncompatibleError(VersionError):
    """Exception raised when version compatibility check fails.

    Attributes:
        engine_name: Name of the engine.
        engine_version: Version of the engine.
        required: Required version constraint.
    """

    def __init__(
        self,
        engine_name: str,
        engine_version: str,
        required: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize version incompatible error.

        Args:
            engine_name: Name of the engine.
            engine_version: Version of the engine.
            required: Required version constraint.
            details: Optional additional context.
            cause: Optional original exception.
        """
        details = details or {}
        details["engine_name"] = engine_name
        details["engine_version"] = engine_version
        details["required"] = required
        message = (
            f"Engine '{engine_name}' version {engine_version} "
            f"is incompatible with requirement '{required}'"
        )
        super().__init__(message, details=details, cause=cause)
        self.engine_name = engine_name
        self.engine_version = engine_version
        self.required = required


# =============================================================================
# Enums
# =============================================================================


class VersionOperator(Enum):
    """Version comparison operators.

    These operators are used in version constraints to specify
    version requirements.
    """

    EQ = "=="  # Exact match
    NE = "!="  # Not equal
    GT = ">"  # Greater than
    GE = ">="  # Greater than or equal
    LT = "<"  # Less than
    LE = "<="  # Less than or equal
    CARET = "^"  # Compatible with (same major, minor can change)
    TILDE = "~"  # Approximately (same major.minor, patch can change)
    WILDCARD = "*"  # Any version

    def __str__(self) -> str:
        """Return the operator symbol."""
        return self.value


class CompatibilityLevel(Enum):
    """Level of version compatibility.

    Used to describe how compatible two versions are.
    """

    IDENTICAL = "identical"  # Exact same version
    COMPATIBLE = "compatible"  # Fully compatible (same major)
    MINOR_DIFF = "minor_diff"  # Minor version difference
    PATCH_DIFF = "patch_diff"  # Only patch version differs
    INCOMPATIBLE = "incompatible"  # Major version differs
    UNKNOWN = "unknown"  # Cannot determine compatibility


class CompatibilityStrategy(Enum):
    """Strategy for determining version compatibility.

    Different strategies for checking version compatibility:
    - STRICT: Exact version match required
    - SEMVER: Semantic versioning rules (major must match)
    - MAJOR: Only major version must match
    - MINOR: Major and minor must match
    - ANY: Any version is acceptable
    """

    STRICT = "strict"
    SEMVER = "semver"
    MAJOR = "major"
    MINOR = "minor"
    ANY = "any"


# =============================================================================
# Semantic Version
# =============================================================================


# Regex for semantic versioning (SemVer 2.0.0 specification)
SEMVER_PATTERN = re.compile(
    r"^"
    r"(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?"
    r"$"
)

# Relaxed pattern for common version formats
RELAXED_VERSION_PATTERN = re.compile(
    r"^"
    r"v?"  # Optional 'v' prefix
    r"(?P<major>\d+)"
    r"(?:\.(?P<minor>\d+))?"
    r"(?:\.(?P<patch>\d+))?"
    r"(?:[-._]?(?P<prerelease>[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*))?"
    r"(?:\+(?P<build>[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*))?"
    r"$"
)


def _parse_prerelease_part(part: str) -> tuple[int, str]:
    """Parse a prerelease part for comparison.

    Numeric parts are compared as integers, string parts lexically.
    Returns tuple (is_numeric, value) for sorting.
    """
    try:
        return (0, str(int(part)))  # Numeric parts come first
    except ValueError:
        return (1, part)  # String parts come after


def _compare_prerelease(pre1: str | None, pre2: str | None) -> int:
    """Compare prerelease strings according to SemVer.

    Per SemVer:
    - No prerelease > any prerelease (1.0.0 > 1.0.0-alpha)
    - Prerelease parts compared: numeric < alpha, shorter < longer
    """
    if pre1 is None and pre2 is None:
        return 0
    if pre1 is None:
        return 1  # No prerelease > any prerelease
    if pre2 is None:
        return -1

    parts1 = pre1.split(".")
    parts2 = pre2.split(".")

    for p1, p2 in zip(parts1, parts2, strict=False):
        parsed1 = _parse_prerelease_part(p1)
        parsed2 = _parse_prerelease_part(p2)

        if parsed1[0] != parsed2[0]:
            return parsed1[0] - parsed2[0]  # Numeric < alpha

        if parsed1[0] == 0:  # Both numeric
            n1, n2 = int(parsed1[1]), int(parsed2[1])
            if n1 != n2:
                return n1 - n2
        elif parsed1[1] != parsed2[1]:
            return -1 if parsed1[1] < parsed2[1] else 1

    # Shorter prerelease < longer
    return len(parts1) - len(parts2)


@total_ordering
@dataclass(frozen=True, slots=True)
class SemanticVersion:
    """Semantic version representation.

    Implements full SemVer 2.0.0 specification with comparison support.
    Versions are immutable and comparable.

    Attributes:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
        prerelease: Optional prerelease identifier (e.g., "alpha.1").
        build: Optional build metadata (ignored in comparisons).

    Example:
        >>> v1 = SemanticVersion(1, 2, 3)
        >>> v2 = SemanticVersion.parse("1.2.4")
        >>> v1 < v2
        True
        >>> v1.with_patch(4) == v2
        True
    """

    major: int
    minor: int = 0
    patch: int = 0
    prerelease: str | None = None
    build: str | None = None

    def __post_init__(self) -> None:
        """Validate version components."""
        if self.major < 0:
            raise ValueError("Major version must be non-negative")
        if self.minor < 0:
            raise ValueError("Minor version must be non-negative")
        if self.patch < 0:
            raise ValueError("Patch version must be non-negative")

    def __str__(self) -> str:
        """Return the version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"SemanticVersion({self!s})"

    def __eq__(self, other: object) -> bool:
        """Check equality (ignores build metadata per SemVer)."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __hash__(self) -> int:
        """Hash (ignores build metadata per SemVer)."""
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def __lt__(self, other: object) -> bool:
        """Compare versions according to SemVer precedence."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented

        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (
            other.major,
            other.minor,
            other.patch,
        ):
            return (self.major, self.minor, self.patch) < (
                other.major,
                other.minor,
                other.patch,
            )

        # Compare prerelease
        return _compare_prerelease(self.prerelease, other.prerelease) < 0

    @classmethod
    def parse(cls, version_string: str, *, strict: bool = False) -> SemanticVersion:
        """Parse a version string into a SemanticVersion.

        Args:
            version_string: Version string to parse.
            strict: If True, require strict SemVer format.
                   If False, accept relaxed formats (e.g., "1.0").

        Returns:
            SemanticVersion instance.

        Raises:
            VersionParseError: If parsing fails.

        Example:
            >>> SemanticVersion.parse("1.2.3-alpha.1+build.123")
            SemanticVersion(1.2.3-alpha.1+build.123)
            >>> SemanticVersion.parse("1.0")  # Relaxed
            SemanticVersion(1.0.0)
        """
        if not version_string:
            raise VersionParseError(version_string, "Empty version string")

        version_string = version_string.strip()

        # Try strict SemVer pattern first
        match = SEMVER_PATTERN.match(version_string)
        if match:
            return cls(
                major=int(match.group("major")),
                minor=int(match.group("minor")),
                patch=int(match.group("patch")),
                prerelease=match.group("prerelease"),
                build=match.group("build"),
            )

        if strict:
            raise VersionParseError(
                version_string,
                "Does not match strict SemVer format (MAJOR.MINOR.PATCH)",
            )

        # Try relaxed pattern
        match = RELAXED_VERSION_PATTERN.match(version_string)
        if match:
            return cls(
                major=int(match.group("major")),
                minor=int(match.group("minor") or 0),
                patch=int(match.group("patch") or 0),
                prerelease=match.group("prerelease"),
                build=match.group("build"),
            )

        raise VersionParseError(
            version_string,
            "Does not match any recognized version format",
        )

    @classmethod
    def try_parse(cls, version_string: str) -> SemanticVersion | None:
        """Try to parse a version string, returning None on failure.

        Args:
            version_string: Version string to parse.

        Returns:
            SemanticVersion or None if parsing fails.
        """
        try:
            return cls.parse(version_string, strict=False)
        except VersionParseError:
            return None

    @classmethod
    def zero(cls) -> SemanticVersion:
        """Return version 0.0.0.

        Returns:
            SemanticVersion(0, 0, 0)
        """
        return cls(0, 0, 0)

    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version.

        Returns:
            True if version has prerelease identifier.
        """
        return self.prerelease is not None

    def is_stable(self) -> bool:
        """Check if this is a stable release.

        Returns:
            True if major >= 1 and no prerelease.
        """
        return self.major >= 1 and not self.is_prerelease()

    def is_compatible_with(
        self,
        other: SemanticVersion,
        *,
        strategy: CompatibilityStrategy = CompatibilityStrategy.SEMVER,
    ) -> bool:
        """Check if this version is compatible with another.

        Args:
            other: Version to check compatibility with.
            strategy: Compatibility checking strategy.

        Returns:
            True if versions are compatible.
        """
        if strategy == CompatibilityStrategy.ANY:
            return True
        if strategy == CompatibilityStrategy.STRICT:
            return self == other
        if strategy == CompatibilityStrategy.MAJOR:
            return self.major == other.major
        if strategy == CompatibilityStrategy.MINOR:
            return self.major == other.major and self.minor == other.minor
        # SEMVER: same major, this >= other
        if strategy == CompatibilityStrategy.SEMVER:
            if self.major != other.major:
                return False
            return self >= other
        return False

    def get_compatibility_level(self, other: SemanticVersion) -> CompatibilityLevel:
        """Get the compatibility level between this version and another.

        Args:
            other: Version to compare with.

        Returns:
            CompatibilityLevel describing the relationship.
        """
        if self == other:
            return CompatibilityLevel.IDENTICAL
        if self.major != other.major:
            return CompatibilityLevel.INCOMPATIBLE
        if self.minor != other.minor:
            return CompatibilityLevel.MINOR_DIFF
        if self.patch != other.patch:
            return CompatibilityLevel.PATCH_DIFF
        return CompatibilityLevel.COMPATIBLE

    def next_major(self) -> SemanticVersion:
        """Return the next major version.

        Returns:
            Version with major incremented, minor and patch reset to 0.
        """
        return SemanticVersion(self.major + 1, 0, 0)

    def next_minor(self) -> SemanticVersion:
        """Return the next minor version.

        Returns:
            Version with minor incremented, patch reset to 0.
        """
        return SemanticVersion(self.major, self.minor + 1, 0)

    def next_patch(self) -> SemanticVersion:
        """Return the next patch version.

        Returns:
            Version with patch incremented.
        """
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def with_major(self, major: int) -> SemanticVersion:
        """Return a new version with different major.

        Args:
            major: New major version number.

        Returns:
            New SemanticVersion with updated major.
        """
        return SemanticVersion(
            major, self.minor, self.patch, self.prerelease, self.build
        )

    def with_minor(self, minor: int) -> SemanticVersion:
        """Return a new version with different minor.

        Args:
            minor: New minor version number.

        Returns:
            New SemanticVersion with updated minor.
        """
        return SemanticVersion(
            self.major, minor, self.patch, self.prerelease, self.build
        )

    def with_patch(self, patch: int) -> SemanticVersion:
        """Return a new version with different patch.

        Args:
            patch: New patch version number.

        Returns:
            New SemanticVersion with updated patch.
        """
        return SemanticVersion(
            self.major, self.minor, patch, self.prerelease, self.build
        )

    def with_prerelease(self, prerelease: str | None) -> SemanticVersion:
        """Return a new version with different prerelease.

        Args:
            prerelease: New prerelease identifier or None.

        Returns:
            New SemanticVersion with updated prerelease.
        """
        return SemanticVersion(
            self.major, self.minor, self.patch, prerelease, self.build
        )

    def with_build(self, build: str | None) -> SemanticVersion:
        """Return a new version with different build metadata.

        Args:
            build: New build metadata or None.

        Returns:
            New SemanticVersion with updated build.
        """
        return SemanticVersion(
            self.major, self.minor, self.patch, self.prerelease, build
        )

    def to_tuple(self) -> tuple[int, int, int]:
        """Return version as (major, minor, patch) tuple.

        Returns:
            Tuple of version components.
        """
        return (self.major, self.minor, self.patch)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "build": self.build,
            "string": str(self),
        }


# =============================================================================
# Version Constraint
# =============================================================================


# Pattern for parsing version constraints
CONSTRAINT_PATTERN = re.compile(
    r"^"
    r"(?P<operator>==|!=|>=|<=|>|<|\^|~|\*)"
    r"\s*"
    r"(?P<version>.+)?"
    r"$"
)

# Pattern for implicit equality (just a version)
IMPLICIT_EQ_PATTERN = re.compile(r"^[\dv]")


@dataclass(frozen=True, slots=True)
class VersionConstraint:
    """A single version constraint.

    Represents a constraint like ">=1.0.0" or "^2.0.0".

    Attributes:
        operator: The comparison operator.
        version: The version to compare against (None for wildcard).

    Example:
        >>> c = VersionConstraint.parse(">=1.0.0")
        >>> c.satisfied_by(SemanticVersion(1, 2, 0))
        True
        >>> c.satisfied_by(SemanticVersion(0, 9, 0))
        False
    """

    operator: VersionOperator
    version: SemanticVersion | None = None

    def __str__(self) -> str:
        """Return constraint as string."""
        if self.operator == VersionOperator.WILDCARD:
            return "*"
        if self.version is None:
            return str(self.operator)
        return f"{self.operator}{self.version}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"VersionConstraint({self!s})"

    @classmethod
    def parse(cls, constraint_string: str) -> VersionConstraint:
        """Parse a constraint string.

        Args:
            constraint_string: Constraint string (e.g., ">=1.0.0", "^2.0").

        Returns:
            VersionConstraint instance.

        Raises:
            VersionConstraintError: If parsing fails.

        Example:
            >>> VersionConstraint.parse(">=1.0.0")
            VersionConstraint(>=1.0.0)
            >>> VersionConstraint.parse("*")
            VersionConstraint(*)
        """
        if not constraint_string:
            raise VersionConstraintError(constraint_string, "Empty constraint string")

        constraint_string = constraint_string.strip()

        # Handle wildcard
        if constraint_string == "*":
            return cls(VersionOperator.WILDCARD, None)

        # Try explicit operator
        match = CONSTRAINT_PATTERN.match(constraint_string)
        if match:
            op_str = match.group("operator")
            version_str = match.group("version")

            try:
                operator = VersionOperator(op_str)
            except ValueError as exc:
                raise VersionConstraintError(
                    constraint_string, f"Unknown operator '{op_str}'"
                ) from exc

            if operator == VersionOperator.WILDCARD:
                return cls(operator, None)

            if not version_str:
                raise VersionConstraintError(
                    constraint_string,
                    f"Operator '{op_str}' requires a version",
                )

            try:
                version = SemanticVersion.parse(version_str.strip())
            except VersionParseError as e:
                raise VersionConstraintError(
                    constraint_string,
                    f"Invalid version: {e.reason}",
                    cause=e,
                ) from e

            return cls(operator, version)

        # Try implicit equality (just a version number)
        if IMPLICIT_EQ_PATTERN.match(constraint_string):
            try:
                version = SemanticVersion.parse(constraint_string)
                return cls(VersionOperator.EQ, version)
            except VersionParseError as e:
                raise VersionConstraintError(
                    constraint_string,
                    f"Invalid version: {e.reason}",
                    cause=e,
                ) from e

        raise VersionConstraintError(
            constraint_string,
            "Does not match constraint format (OPERATOR VERSION)",
        )

    @classmethod
    def any_version(cls) -> VersionConstraint:
        """Create a constraint that matches any version.

        Returns:
            Wildcard constraint.
        """
        return cls(VersionOperator.WILDCARD, None)

    @classmethod
    def exact(cls, version: SemanticVersion | str) -> VersionConstraint:
        """Create an exact version constraint.

        Args:
            version: Version to match exactly.

        Returns:
            Equality constraint.
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return cls(VersionOperator.EQ, version)

    @classmethod
    def at_least(cls, version: SemanticVersion | str) -> VersionConstraint:
        """Create a minimum version constraint.

        Args:
            version: Minimum version (inclusive).

        Returns:
            Greater-than-or-equal constraint.
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return cls(VersionOperator.GE, version)

    @classmethod
    def less_than(cls, version: SemanticVersion | str) -> VersionConstraint:
        """Create a maximum version constraint.

        Args:
            version: Maximum version (exclusive).

        Returns:
            Less-than constraint.
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return cls(VersionOperator.LT, version)

    @classmethod
    def compatible_with(cls, version: SemanticVersion | str) -> VersionConstraint:
        """Create a caret compatibility constraint.

        The caret constraint allows changes that do not modify the
        leftmost non-zero digit:
        - ^1.2.3 := >=1.2.3 <2.0.0
        - ^0.2.3 := >=0.2.3 <0.3.0
        - ^0.0.3 := >=0.0.3 <0.0.4

        Args:
            version: Base version.

        Returns:
            Caret constraint.
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return cls(VersionOperator.CARET, version)

    @classmethod
    def approximately(cls, version: SemanticVersion | str) -> VersionConstraint:
        """Create a tilde approximate constraint.

        The tilde constraint allows patch-level changes:
        - ~1.2.3 := >=1.2.3 <1.3.0
        - ~1.2 := >=1.2.0 <1.3.0

        Args:
            version: Base version.

        Returns:
            Tilde constraint.
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return cls(VersionOperator.TILDE, version)

    def satisfied_by(self, version: SemanticVersion) -> bool:  # noqa: PLR0911
        """Check if a version satisfies this constraint.

        Args:
            version: Version to check.

        Returns:
            True if the version satisfies this constraint.

        Example:
            >>> c = VersionConstraint.parse(">=1.0.0")
            >>> c.satisfied_by(SemanticVersion(1, 2, 0))
            True
        """
        if self.operator == VersionOperator.WILDCARD:
            return True

        if self.version is None:
            return False

        v = self.version

        if self.operator == VersionOperator.EQ:
            return version == v
        if self.operator == VersionOperator.NE:
            return version != v
        if self.operator == VersionOperator.GT:
            return version > v
        if self.operator == VersionOperator.GE:
            return version >= v
        if self.operator == VersionOperator.LT:
            return version < v
        if self.operator == VersionOperator.LE:
            return version <= v

        if self.operator == VersionOperator.CARET:
            # ^X.Y.Z allows changes that do not modify leftmost non-zero
            if version < v:
                return False
            if v.major != 0:
                return version.major == v.major
            if v.minor != 0:
                return version.major == 0 and version.minor == v.minor
            return (
                version.major == 0
                and version.minor == 0
                and version.patch == v.patch
            )

        if self.operator == VersionOperator.TILDE:
            # ~X.Y.Z allows patch-level changes only
            if version < v:
                return False
            return version.major == v.major and version.minor == v.minor

        return False

    def allows_prerelease(self) -> bool:
        """Check if this constraint allows prerelease versions.

        Returns:
            True if the constraint version is a prerelease.
        """
        return self.version is not None and self.version.is_prerelease()


# =============================================================================
# Version Range
# =============================================================================


@dataclass(frozen=True, slots=True)
class VersionRange:
    """A combination of version constraints.

    Represents complex version requirements like ">=1.0.0,<2.0.0".
    All constraints must be satisfied for a version to match.

    Attributes:
        constraints: Tuple of constraints (all must be satisfied).

    Example:
        >>> r = VersionRange.parse(">=1.0.0,<2.0.0")
        >>> r.satisfied_by(SemanticVersion(1, 5, 0))
        True
        >>> r.satisfied_by(SemanticVersion(2, 0, 0))
        False
    """

    constraints: tuple[VersionConstraint, ...] = ()

    def __str__(self) -> str:
        """Return range as string."""
        if not self.constraints:
            return "*"
        return ",".join(str(c) for c in self.constraints)

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"VersionRange({self!s})"

    def __bool__(self) -> bool:
        """Return True if there are any constraints."""
        return bool(self.constraints)

    @classmethod
    def parse(cls, range_string: str) -> VersionRange:
        """Parse a version range string.

        Args:
            range_string: Range string with comma-separated constraints.

        Returns:
            VersionRange instance.

        Raises:
            VersionConstraintError: If parsing fails.

        Example:
            >>> VersionRange.parse(">=1.0.0,<2.0.0")
            VersionRange(>=1.0.0,<2.0.0)
            >>> VersionRange.parse("*")
            VersionRange(*)
        """
        if not range_string:
            return cls(())

        range_string = range_string.strip()

        if range_string == "*":
            return cls((VersionConstraint.any_version(),))

        # Split by comma and parse each constraint
        parts = [p.strip() for p in range_string.split(",")]
        constraints = [
            VersionConstraint.parse(part) for part in parts if part
        ]

        return cls(tuple(constraints))

    @classmethod
    def any_version(cls) -> VersionRange:
        """Create a range that matches any version.

        Returns:
            Range with wildcard constraint.
        """
        return cls((VersionConstraint.any_version(),))

    @classmethod
    def exact(cls, version: SemanticVersion | str) -> VersionRange:
        """Create a range that matches exactly one version.

        Args:
            version: Version to match.

        Returns:
            Range with exact constraint.
        """
        return cls((VersionConstraint.exact(version),))

    @classmethod
    def at_least(cls, version: SemanticVersion | str) -> VersionRange:
        """Create a range that matches version or higher.

        Args:
            version: Minimum version.

        Returns:
            Range with at-least constraint.
        """
        return cls((VersionConstraint.at_least(version),))

    @classmethod
    def between(
        cls,
        min_version: SemanticVersion | str,
        max_version: SemanticVersion | str,
        *,
        include_max: bool = False,
    ) -> VersionRange:
        """Create a range between two versions.

        Args:
            min_version: Minimum version (inclusive).
            max_version: Maximum version.
            include_max: If True, include max version (<=). Otherwise (<).

        Returns:
            Range between versions.
        """
        constraints = [VersionConstraint.at_least(min_version)]
        if include_max:
            if isinstance(max_version, str):
                max_version = SemanticVersion.parse(max_version)
            constraints.append(VersionConstraint(VersionOperator.LE, max_version))
        else:
            constraints.append(VersionConstraint.less_than(max_version))
        return cls(tuple(constraints))

    @classmethod
    def compatible_with(cls, version: SemanticVersion | str) -> VersionRange:
        """Create a SemVer-compatible range.

        Equivalent to ^version (caret constraint).

        Args:
            version: Base version.

        Returns:
            Range with caret constraint.
        """
        return cls((VersionConstraint.compatible_with(version),))

    def satisfied_by(
        self,
        version: SemanticVersion,
        *,
        allow_prerelease: bool = False,
    ) -> bool:
        """Check if a version satisfies all constraints.

        Args:
            version: Version to check.
            allow_prerelease: If False, reject prereleases unless
                             the constraint itself specifies one.

        Returns:
            True if all constraints are satisfied.
        """
        if not self.constraints:
            return True

        # Handle prerelease versions: reject if not allowed and no constraint permits it
        if (
            version.is_prerelease()
            and not allow_prerelease
            and not any(c.allows_prerelease() for c in self.constraints)
        ):
            return False

        return all(c.satisfied_by(version) for c in self.constraints)

    def is_empty(self) -> bool:
        """Check if this range has no constraints.

        Returns:
            True if no constraints.
        """
        return len(self.constraints) == 0

    def is_any(self) -> bool:
        """Check if this range accepts any version.

        Returns:
            True if only constraint is wildcard.
        """
        return (
            len(self.constraints) == 1
            and self.constraints[0].operator == VersionOperator.WILDCARD
        )

    def and_constraint(self, constraint: VersionConstraint) -> VersionRange:
        """Add a constraint to this range.

        Args:
            constraint: Constraint to add.

        Returns:
            New range with additional constraint.
        """
        return VersionRange((*self.constraints, constraint))

    def and_range(self, other: VersionRange) -> VersionRange:
        """Combine with another range (AND).

        Args:
            other: Range to combine with.

        Returns:
            New range with all constraints from both.
        """
        return VersionRange((*self.constraints, *other.constraints))


# =============================================================================
# Version Compatibility Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class VersionCompatibilityResult:
    """Result of a version compatibility check.

    Attributes:
        compatible: Whether the version is compatible.
        engine_name: Name of the engine checked.
        engine_version: Version of the engine.
        required: The version requirement.
        level: Compatibility level (if compatible).
        message: Human-readable message.
        warnings: List of compatibility warnings.
    """

    compatible: bool
    engine_name: str
    engine_version: SemanticVersion
    required: VersionRange
    level: CompatibilityLevel = CompatibilityLevel.UNKNOWN
    message: str = ""
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Return True if compatible."""
        return self.compatible

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "compatible": self.compatible,
            "engine_name": self.engine_name,
            "engine_version": str(self.engine_version),
            "required": str(self.required),
            "level": self.level.value,
            "message": self.message,
            "warnings": list(self.warnings),
        }


# =============================================================================
# Version Checker Protocol
# =============================================================================


@runtime_checkable
class VersionChecker(Protocol):
    """Protocol for version compatibility checkers.

    Implement this protocol to create custom compatibility checking logic.
    """

    def check(
        self,
        engine_version: SemanticVersion,
        required: VersionRange,
        *,
        engine_name: str = "",
    ) -> VersionCompatibilityResult:
        """Check version compatibility.

        Args:
            engine_version: The engine's version.
            required: Required version range.
            engine_name: Optional engine name for messages.

        Returns:
            Compatibility result.
        """
        ...


# =============================================================================
# Version Compatibility Checker
# =============================================================================


@dataclass(frozen=True, slots=True)
class VersionCompatibilityConfig:
    """Configuration for version compatibility checking.

    Attributes:
        strategy: Default compatibility strategy.
        allow_prerelease: Allow prerelease versions by default.
        strict: Raise exceptions on incompatibility.
        warn_on_prerelease: Emit warning for prerelease versions.
        warn_on_patch_diff: Emit warning for patch differences.
    """

    strategy: CompatibilityStrategy = CompatibilityStrategy.SEMVER
    allow_prerelease: bool = False
    strict: bool = False
    warn_on_prerelease: bool = True
    warn_on_patch_diff: bool = False


# Default configuration
DEFAULT_VERSION_COMPATIBILITY_CONFIG = VersionCompatibilityConfig()

# Strict configuration - requires exact version match
STRICT_VERSION_COMPATIBILITY_CONFIG = VersionCompatibilityConfig(
    strategy=CompatibilityStrategy.STRICT,
    strict=True,
)

# Lenient configuration - accepts any version
LENIENT_VERSION_COMPATIBILITY_CONFIG = VersionCompatibilityConfig(
    strategy=CompatibilityStrategy.ANY,
    allow_prerelease=True,
    warn_on_prerelease=False,
)


class VersionCompatibilityChecker:
    """Check version compatibility with configurable strategies.

    This class provides flexible version compatibility checking with
    support for different strategies and warning generation.

    Example:
        >>> checker = VersionCompatibilityChecker()
        >>> result = checker.check(
        ...     SemanticVersion(1, 2, 3),
        ...     VersionRange.parse(">=1.0.0"),
        ...     engine_name="truthound",
        ... )
        >>> result.compatible
        True
    """

    def __init__(
        self,
        config: VersionCompatibilityConfig | None = None,
    ) -> None:
        """Initialize the checker.

        Args:
            config: Configuration for compatibility checking.
        """
        self._config = config or DEFAULT_VERSION_COMPATIBILITY_CONFIG

    @property
    def config(self) -> VersionCompatibilityConfig:
        """Get the current configuration."""
        return self._config

    def check(
        self,
        engine_version: SemanticVersion,
        required: VersionRange,
        *,
        engine_name: str = "",
    ) -> VersionCompatibilityResult:
        """Check if an engine version satisfies requirements.

        Args:
            engine_version: The engine's version.
            required: Required version range.
            engine_name: Optional engine name for messages.

        Returns:
            VersionCompatibilityResult with check outcome.

        Raises:
            VersionIncompatibleError: If strict mode and incompatible.
        """
        warnings: list[str] = []

        # Handle prerelease versions
        if engine_version.is_prerelease():
            if self._config.warn_on_prerelease:
                warnings.append(
                    f"Engine version {engine_version} is a prerelease"
                )
            # Check if range explicitly allows it when prereleases not allowed
            if (
                not self._config.allow_prerelease
                and not required.satisfied_by(engine_version, allow_prerelease=True)
            ):
                return self._make_result(
                    compatible=False,
                    engine_name=engine_name,
                    engine_version=engine_version,
                    required=required,
                    message=f"Prerelease version {engine_version} not allowed",
                    warnings=tuple(warnings),
                )

        # Check satisfaction
        compatible = required.satisfied_by(
            engine_version,
            allow_prerelease=self._config.allow_prerelease,
        )

        if not compatible:
            message = (
                f"Version {engine_version} does not satisfy {required}"
            )
            if self._config.strict:
                raise VersionIncompatibleError(
                    engine_name or "unknown",
                    str(engine_version),
                    str(required),
                )
            return self._make_result(
                compatible=False,
                engine_name=engine_name,
                engine_version=engine_version,
                required=required,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=message,
                warnings=tuple(warnings),
            )

        # Determine compatibility level
        level = self._determine_level(engine_version, required)

        # Generate warnings based on level
        if level == CompatibilityLevel.PATCH_DIFF and self._config.warn_on_patch_diff:
            warnings.append("Minor version difference detected")

        message = f"Version {engine_version} satisfies {required}"

        return self._make_result(
            compatible=True,
            engine_name=engine_name,
            engine_version=engine_version,
            required=required,
            level=level,
            message=message,
            warnings=tuple(warnings),
        )

    def _determine_level(
        self,
        version: SemanticVersion,
        required: VersionRange,
    ) -> CompatibilityLevel:
        """Determine the compatibility level.

        Args:
            version: The actual version.
            required: The required range.

        Returns:
            CompatibilityLevel.
        """
        # Find the primary constraint's version for comparison
        for constraint in required.constraints:
            if constraint.version is not None:
                return version.get_compatibility_level(constraint.version)

        return CompatibilityLevel.COMPATIBLE

    def _make_result(
        self,
        *,
        compatible: bool,
        engine_name: str,
        engine_version: SemanticVersion,
        required: VersionRange,
        level: CompatibilityLevel = CompatibilityLevel.UNKNOWN,
        message: str = "",
        warnings: tuple[str, ...] = (),
    ) -> VersionCompatibilityResult:
        """Create a compatibility result.

        Args:
            compatible: Whether compatible.
            engine_name: Engine name.
            engine_version: Engine version.
            required: Required range.
            level: Compatibility level.
            message: Result message.
            warnings: Warning list.

        Returns:
            VersionCompatibilityResult.
        """
        return VersionCompatibilityResult(
            compatible=compatible,
            engine_name=engine_name,
            engine_version=engine_version,
            required=required,
            level=level,
            message=message,
            warnings=warnings,
        )


# =============================================================================
# Version Requirements
# =============================================================================


@dataclass(frozen=True, slots=True)
class EngineVersionRequirement:
    """Version requirement for a specific engine.

    Attributes:
        engine_name: Name of the required engine.
        version_range: Required version range.
        optional: If True, missing engine is not an error.
        reason: Human-readable reason for requirement.
    """

    engine_name: str
    version_range: VersionRange
    optional: bool = False
    reason: str = ""

    def __str__(self) -> str:
        """Return requirement as string."""
        opt = " (optional)" if self.optional else ""
        return f"{self.engine_name}{self.version_range}{opt}"

    @classmethod
    def parse(cls, requirement_string: str) -> EngineVersionRequirement:
        """Parse a requirement string.

        Format: "engine_name OP VERSION" or "engine_name OP VERSION (optional)"

        Args:
            requirement_string: Requirement string.

        Returns:
            EngineVersionRequirement instance.

        Example:
            >>> EngineVersionRequirement.parse("truthound >=1.0.0")
            EngineVersionRequirement(truthound>=1.0.0)
        """
        if not requirement_string:
            raise VersionConstraintError(requirement_string, "Empty requirement")

        requirement_string = requirement_string.strip()

        # Check for optional marker
        optional = False
        if requirement_string.endswith("(optional)"):
            optional = True
            requirement_string = requirement_string[:-10].strip()

        # Split on first operator
        for op in (">=", "<=", "!=", "==", ">", "<", "^", "~", "*"):
            if op in requirement_string:
                parts = requirement_string.split(op, 1)
                engine_name = parts[0].strip()
                version_part = op + parts[1] if len(parts) > 1 else op
                version_range = VersionRange.parse(version_part)
                return cls(engine_name, version_range, optional)

        # No operator found - might be just engine name with *
        if " " in requirement_string:
            parts = requirement_string.split(None, 1)
            engine_name = parts[0]
            version_range = VersionRange.parse(parts[1])
            return cls(engine_name, version_range, optional)

        # Just engine name - any version
        return cls(requirement_string, VersionRange.any_version(), optional)

    def satisfied_by(
        self,
        engine_version: SemanticVersion,
        *,
        allow_prerelease: bool = False,
    ) -> bool:
        """Check if a version satisfies this requirement.

        Args:
            engine_version: Version to check.
            allow_prerelease: Allow prerelease versions.

        Returns:
            True if satisfied.
        """
        return self.version_range.satisfied_by(
            engine_version,
            allow_prerelease=allow_prerelease,
        )


# =============================================================================
# Version Registry
# =============================================================================


class VersionRegistry:
    """Registry for tracking engine versions and requirements.

    This registry maintains information about registered engines
    and their version requirements. It can be used to validate
    that all version requirements are satisfied.

    Example:
        >>> registry = VersionRegistry()
        >>> registry.register_engine("truthound", SemanticVersion(1, 2, 3))
        >>> registry.add_requirement(
        ...     EngineVersionRequirement.parse("truthound >=1.0.0")
        ... )
        >>> registry.validate_all()
        {'truthound': VersionCompatibilityResult(compatible=True, ...)}
    """

    def __init__(
        self,
        checker: VersionCompatibilityChecker | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            checker: Version compatibility checker.
        """
        self._checker = checker or VersionCompatibilityChecker()
        self._engines: dict[str, SemanticVersion] = {}
        self._requirements: list[EngineVersionRequirement] = []
        self._engine_info: dict[str, dict[str, Any]] = {}

    def register_engine(
        self,
        name: str,
        version: SemanticVersion | str,
        *,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Register an engine with its version.

        Args:
            name: Engine name.
            version: Engine version.
            info: Optional additional engine info.
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        self._engines[name] = version
        if info:
            self._engine_info[name] = info

    def register_from_engine(self, engine: DataQualityEngine) -> None:
        """Register an engine instance.

        Args:
            engine: Engine to register.
        """
        version = SemanticVersion.try_parse(engine.engine_version)
        if version is None:
            version = SemanticVersion.zero()

        info: dict[str, Any] = {}
        if hasattr(engine, "get_info"):
            engine_info = engine.get_info()
            info = engine_info.to_dict()

        self.register_engine(engine.engine_name, version, info=info)

    def unregister_engine(self, name: str) -> bool:
        """Remove an engine from the registry.

        Args:
            name: Engine name.

        Returns:
            True if engine was removed.
        """
        if name in self._engines:
            del self._engines[name]
            self._engine_info.pop(name, None)
            return True
        return False

    def add_requirement(self, requirement: EngineVersionRequirement) -> None:
        """Add a version requirement.

        Args:
            requirement: Requirement to add.
        """
        self._requirements.append(requirement)

    def add_requirements(
        self,
        requirements: Sequence[EngineVersionRequirement | str],
    ) -> None:
        """Add multiple requirements.

        Args:
            requirements: Requirements to add (strings are parsed).
        """
        for requirement in requirements:
            parsed_req: EngineVersionRequirement
            if isinstance(requirement, str):
                parsed_req = EngineVersionRequirement.parse(requirement)
            else:
                parsed_req = requirement
            self.add_requirement(parsed_req)

    def clear_requirements(self) -> None:
        """Remove all requirements."""
        self._requirements.clear()

    def get_engine_version(self, name: str) -> SemanticVersion | None:
        """Get a registered engine's version.

        Args:
            name: Engine name.

        Returns:
            Engine version or None if not registered.
        """
        return self._engines.get(name)

    def get_engine_info(self, name: str) -> dict[str, Any]:
        """Get engine info.

        Args:
            name: Engine name.

        Returns:
            Engine info dictionary.
        """
        return self._engine_info.get(name, {})

    def list_engines(self) -> list[str]:
        """List registered engine names.

        Returns:
            List of engine names.
        """
        return list(self._engines.keys())

    def list_requirements(self) -> list[EngineVersionRequirement]:
        """List all requirements.

        Returns:
            List of requirements.
        """
        return list(self._requirements)

    def check_requirement(
        self,
        requirement: EngineVersionRequirement,
    ) -> VersionCompatibilityResult | None:
        """Check a single requirement.

        Args:
            requirement: Requirement to check.

        Returns:
            Compatibility result or None if engine not registered
            and requirement is optional.
        """
        version = self._engines.get(requirement.engine_name)

        if version is None:
            if requirement.optional:
                return None
            # Create a failed result for missing engine
            return VersionCompatibilityResult(
                compatible=False,
                engine_name=requirement.engine_name,
                engine_version=SemanticVersion.zero(),
                required=requirement.version_range,
                level=CompatibilityLevel.INCOMPATIBLE,
                message=f"Engine '{requirement.engine_name}' not registered",
            )

        return self._checker.check(
            version,
            requirement.version_range,
            engine_name=requirement.engine_name,
        )

    def validate_all(self) -> dict[str, VersionCompatibilityResult]:
        """Validate all requirements.

        Returns:
            Dictionary mapping engine names to compatibility results.
        """
        results: dict[str, VersionCompatibilityResult] = {}

        for req in self._requirements:
            result = self.check_requirement(req)
            if result is not None:
                results[req.engine_name] = result

        return results

    def are_all_compatible(self) -> bool:
        """Check if all requirements are satisfied.

        Returns:
            True if all requirements are compatible.
        """
        results = self.validate_all()
        return all(r.compatible for r in results.values())

    def get_incompatible(self) -> list[VersionCompatibilityResult]:
        """Get list of incompatible results.

        Returns:
            List of failed compatibility results.
        """
        results = self.validate_all()
        return [r for r in results.values() if not r.compatible]

    def to_dict(self) -> dict[str, Any]:
        """Convert registry state to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "engines": {
                name: str(version) for name, version in self._engines.items()
            },
            "requirements": [str(req) for req in self._requirements],
            "engine_info": self._engine_info,
        }


# =============================================================================
# Global Registry
# =============================================================================


_global_version_registry: VersionRegistry | None = None


def get_version_registry() -> VersionRegistry:
    """Get the global version registry.

    Returns:
        Global VersionRegistry instance.
    """
    global _global_version_registry
    if _global_version_registry is None:
        _global_version_registry = VersionRegistry()
    return _global_version_registry


def reset_version_registry() -> None:
    """Reset the global version registry."""
    global _global_version_registry
    _global_version_registry = None


# =============================================================================
# Convenience Functions
# =============================================================================


def parse_version(version_string: str) -> SemanticVersion:
    """Parse a version string.

    Args:
        version_string: Version string to parse.

    Returns:
        SemanticVersion instance.

    Raises:
        VersionParseError: If parsing fails.
    """
    return SemanticVersion.parse(version_string)


def parse_constraint(constraint_string: str) -> VersionConstraint:
    """Parse a version constraint string.

    Args:
        constraint_string: Constraint string to parse.

    Returns:
        VersionConstraint instance.

    Raises:
        VersionConstraintError: If parsing fails.
    """
    return VersionConstraint.parse(constraint_string)


def parse_range(range_string: str) -> VersionRange:
    """Parse a version range string.

    Args:
        range_string: Range string to parse.

    Returns:
        VersionRange instance.

    Raises:
        VersionConstraintError: If parsing fails.
    """
    return VersionRange.parse(range_string)


def check_version_compatibility(
    engine: DataQualityEngine,
    required: VersionRange | str,
    *,
    config: VersionCompatibilityConfig | None = None,
) -> VersionCompatibilityResult:
    """Check if an engine's version satisfies requirements.

    Args:
        engine: Engine to check.
        required: Required version range (string or VersionRange).
        config: Optional compatibility configuration.

    Returns:
        VersionCompatibilityResult.

    Example:
        >>> result = check_version_compatibility(engine, ">=1.0.0")
        >>> if not result.compatible:
        ...     print(f"Incompatible: {result.message}")
    """
    if isinstance(required, str):
        required = VersionRange.parse(required)

    version = SemanticVersion.try_parse(engine.engine_version)
    if version is None:
        return VersionCompatibilityResult(
            compatible=False,
            engine_name=engine.engine_name,
            engine_version=SemanticVersion.zero(),
            required=required,
            level=CompatibilityLevel.UNKNOWN,
            message=f"Could not parse engine version: {engine.engine_version}",
        )

    checker = VersionCompatibilityChecker(config)
    return checker.check(version, required, engine_name=engine.engine_name)


def require_version(
    engine: DataQualityEngine,
    required: VersionRange | str,
    *,
    config: VersionCompatibilityConfig | None = None,
) -> None:
    """Require an engine to meet version requirements.

    Args:
        engine: Engine to check.
        required: Required version range.
        config: Optional compatibility configuration.

    Raises:
        VersionIncompatibleError: If version is incompatible.

    Example:
        >>> require_version(engine, ">=1.0.0")  # Raises if incompatible
    """
    if isinstance(required, str):
        required = VersionRange.parse(required)

    # Use strict config to raise exception
    strict_config = VersionCompatibilityConfig(
        strategy=(config or DEFAULT_VERSION_COMPATIBILITY_CONFIG).strategy,
        allow_prerelease=(config or DEFAULT_VERSION_COMPATIBILITY_CONFIG).allow_prerelease,
        strict=True,
    )

    check_version_compatibility(engine, required, config=strict_config)


def is_version_compatible(
    engine: DataQualityEngine,
    required: VersionRange | str,
) -> bool:
    """Check if an engine's version is compatible.

    Args:
        engine: Engine to check.
        required: Required version range.

    Returns:
        True if compatible.

    Example:
        >>> if is_version_compatible(engine, ">=1.0.0"):
        ...     print("Compatible!")
    """
    result = check_version_compatibility(engine, required)
    return result.compatible


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Args:
        v1: First version string.
        v2: Second version string.

    Returns:
        -1 if v1 < v2, 0 if equal, 1 if v1 > v2.

    Example:
        >>> compare_versions("1.0.0", "2.0.0")
        -1
    """
    version1 = SemanticVersion.parse(v1)
    version2 = SemanticVersion.parse(v2)

    if version1 < version2:
        return -1
    if version1 > version2:
        return 1
    return 0


def versions_compatible(
    v1: str,
    v2: str,
    *,
    strategy: CompatibilityStrategy = CompatibilityStrategy.SEMVER,
) -> bool:
    """Check if two versions are compatible.

    Args:
        v1: First version string.
        v2: Second version string.
        strategy: Compatibility strategy.

    Returns:
        True if compatible.

    Example:
        >>> versions_compatible("1.2.0", "1.5.0")
        True
        >>> versions_compatible("1.2.0", "2.0.0")
        False
    """
    version1 = SemanticVersion.parse(v1)
    version2 = SemanticVersion.parse(v2)
    return version1.is_compatible_with(version2, strategy=strategy)


# =============================================================================
# Decorators
# =============================================================================


def version_required(
    required: VersionRange | str,
    *,
    engine_param: str = "engine",
    config: VersionCompatibilityConfig | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to require specific engine version.

    Use this decorator to ensure functions receive engines with
    compatible versions.

    Args:
        required: Required version range.
        engine_param: Name of the engine parameter.
        config: Optional compatibility configuration.

    Returns:
        Decorator function.

    Example:
        >>> @version_required(">=1.0.0")
        ... def process_data(engine: DataQualityEngine, data):
        ...     return engine.check(data, [])
    """
    if isinstance(required, str):
        required = VersionRange.parse(required)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find engine in args or kwargs
            import inspect

            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            engine = None
            if engine_param in kwargs:
                engine = kwargs[engine_param]
            elif params and engine_param in params:
                idx = params.index(engine_param)
                if idx < len(args):
                    engine = args[idx]

            if engine is not None:
                require_version(engine, required, config=config)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Type Aliases
# =============================================================================

# Version specification types that can be parsed
VersionSpec = str | SemanticVersion
ConstraintSpec = str | VersionConstraint
RangeSpec = str | VersionRange
