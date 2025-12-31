"""Tests for engine version management system.

This module tests all version-related functionality:
- SemanticVersion parsing and comparison
- VersionConstraint parsing and satisfaction
- VersionRange combining multiple constraints
- VersionCompatibilityChecker with strategies
- VersionRegistry for tracking requirements
"""

from __future__ import annotations

import pytest

from common.engines.version import (
    LENIENT_VERSION_COMPATIBILITY_CONFIG,
    STRICT_VERSION_COMPATIBILITY_CONFIG,
    # Enums
    CompatibilityLevel,
    CompatibilityStrategy,
    EngineVersionRequirement,
    # Types
    SemanticVersion,
    # Checker
    VersionCompatibilityChecker,
    VersionConstraint,
    VersionConstraintError,
    # Exceptions
    VersionIncompatibleError,
    VersionOperator,
    VersionParseError,
    VersionRange,
    # Registry
    VersionRegistry,
    check_version_compatibility,
    compare_versions,
    get_version_registry,
    is_version_compatible,
    parse_constraint,
    parse_range,
    # Convenience functions
    parse_version,
    require_version,
    reset_version_registry,
    # Decorators
    version_required,
    versions_compatible,
)


# =============================================================================
# SemanticVersion Tests
# =============================================================================


class TestSemanticVersionParsing:
    """Tests for SemanticVersion.parse()."""

    def test_parse_simple_version(self) -> None:
        """Test parsing simple X.Y.Z versions."""
        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease is None
        assert v.build is None

    def test_parse_version_with_prerelease(self) -> None:
        """Test parsing versions with prerelease identifiers."""
        v = SemanticVersion.parse("1.2.3-alpha.1")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == "alpha.1"
        assert v.build is None

    def test_parse_version_with_build(self) -> None:
        """Test parsing versions with build metadata."""
        v = SemanticVersion.parse("1.2.3+build.123")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease is None
        assert v.build == "build.123"

    def test_parse_version_with_prerelease_and_build(self) -> None:
        """Test parsing versions with both prerelease and build."""
        v = SemanticVersion.parse("1.2.3-beta.2+build.456")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == "beta.2"
        assert v.build == "build.456"

    def test_parse_relaxed_version_without_patch(self) -> None:
        """Test parsing relaxed versions without patch number."""
        v = SemanticVersion.parse("1.2")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 0

    def test_parse_relaxed_version_without_minor(self) -> None:
        """Test parsing relaxed versions with only major."""
        v = SemanticVersion.parse("1")
        assert v.major == 1
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_version_with_v_prefix(self) -> None:
        """Test parsing versions with 'v' prefix."""
        v = SemanticVersion.parse("v1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_empty_string_raises(self) -> None:
        """Test that empty string raises VersionParseError."""
        with pytest.raises(VersionParseError) as exc_info:
            SemanticVersion.parse("")
        assert exc_info.value.version_string == ""

    def test_parse_invalid_string_raises(self) -> None:
        """Test that invalid strings raise VersionParseError."""
        with pytest.raises(VersionParseError):
            SemanticVersion.parse("not-a-version")

    def test_parse_strict_mode_rejects_relaxed(self) -> None:
        """Test strict mode rejects relaxed formats."""
        with pytest.raises(VersionParseError):
            SemanticVersion.parse("1.2", strict=True)

    def test_try_parse_returns_none_on_failure(self) -> None:
        """Test try_parse returns None instead of raising."""
        result = SemanticVersion.try_parse("invalid")
        assert result is None

    def test_try_parse_returns_version_on_success(self) -> None:
        """Test try_parse returns version on success."""
        result = SemanticVersion.try_parse("1.2.3")
        assert result is not None
        assert result.major == 1


class TestSemanticVersionComparison:
    """Tests for SemanticVersion comparison operations."""

    def test_equal_versions(self) -> None:
        """Test equality of identical versions."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        assert v1 == v2

    def test_equal_ignores_build_metadata(self) -> None:
        """Test that build metadata is ignored in equality."""
        v1 = SemanticVersion(1, 2, 3, build="build1")
        v2 = SemanticVersion(1, 2, 3, build="build2")
        assert v1 == v2

    def test_not_equal_different_major(self) -> None:
        """Test inequality with different major versions."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(2, 0, 0)
        assert v1 != v2

    def test_less_than_major(self) -> None:
        """Test less than comparison by major."""
        v1 = SemanticVersion(1, 9, 9)
        v2 = SemanticVersion(2, 0, 0)
        assert v1 < v2

    def test_less_than_minor(self) -> None:
        """Test less than comparison by minor."""
        v1 = SemanticVersion(1, 2, 9)
        v2 = SemanticVersion(1, 3, 0)
        assert v1 < v2

    def test_less_than_patch(self) -> None:
        """Test less than comparison by patch."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 4)
        assert v1 < v2

    def test_prerelease_less_than_release(self) -> None:
        """Test that prerelease versions are less than release."""
        v1 = SemanticVersion(1, 0, 0, prerelease="alpha")
        v2 = SemanticVersion(1, 0, 0)
        assert v1 < v2

    def test_prerelease_comparison_numeric(self) -> None:
        """Test prerelease comparison with numeric parts."""
        v1 = SemanticVersion(1, 0, 0, prerelease="alpha.1")
        v2 = SemanticVersion(1, 0, 0, prerelease="alpha.2")
        assert v1 < v2

    def test_prerelease_comparison_alpha_vs_numeric(self) -> None:
        """Test that numeric prerelease parts sort before alpha."""
        v1 = SemanticVersion(1, 0, 0, prerelease="1")
        v2 = SemanticVersion(1, 0, 0, prerelease="alpha")
        assert v1 < v2

    def test_versions_are_hashable(self) -> None:
        """Test that versions can be used as dict keys."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        d = {v1: "value"}
        assert d[v2] == "value"

    def test_hash_ignores_build(self) -> None:
        """Test that hash ignores build metadata."""
        v1 = SemanticVersion(1, 2, 3, build="a")
        v2 = SemanticVersion(1, 2, 3, build="b")
        assert hash(v1) == hash(v2)


class TestSemanticVersionMethods:
    """Tests for SemanticVersion instance methods."""

    def test_is_prerelease(self) -> None:
        """Test is_prerelease method."""
        v1 = SemanticVersion(1, 0, 0, prerelease="alpha")
        v2 = SemanticVersion(1, 0, 0)
        assert v1.is_prerelease()
        assert not v2.is_prerelease()

    def test_is_stable(self) -> None:
        """Test is_stable method."""
        assert SemanticVersion(1, 0, 0).is_stable()
        assert SemanticVersion(2, 5, 3).is_stable()
        assert not SemanticVersion(0, 9, 0).is_stable()  # Major < 1
        assert not SemanticVersion(1, 0, 0, prerelease="rc1").is_stable()

    def test_next_major(self) -> None:
        """Test next_major method."""
        v = SemanticVersion(1, 2, 3)
        assert v.next_major() == SemanticVersion(2, 0, 0)

    def test_next_minor(self) -> None:
        """Test next_minor method."""
        v = SemanticVersion(1, 2, 3)
        assert v.next_minor() == SemanticVersion(1, 3, 0)

    def test_next_patch(self) -> None:
        """Test next_patch method."""
        v = SemanticVersion(1, 2, 3)
        assert v.next_patch() == SemanticVersion(1, 2, 4)

    def test_with_methods(self) -> None:
        """Test with_* builder methods."""
        v = SemanticVersion(1, 2, 3)
        assert v.with_major(5).major == 5
        assert v.with_minor(5).minor == 5
        assert v.with_patch(5).patch == 5
        assert v.with_prerelease("rc1").prerelease == "rc1"
        assert v.with_build("build1").build == "build1"

    def test_to_tuple(self) -> None:
        """Test to_tuple method."""
        v = SemanticVersion(1, 2, 3)
        assert v.to_tuple() == (1, 2, 3)

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        v = SemanticVersion(1, 2, 3, prerelease="alpha", build="build1")
        d = v.to_dict()
        assert d["major"] == 1
        assert d["minor"] == 2
        assert d["patch"] == 3
        assert d["prerelease"] == "alpha"
        assert d["build"] == "build1"
        assert d["string"] == "1.2.3-alpha+build1"

    def test_str_representation(self) -> None:
        """Test string representation."""
        assert str(SemanticVersion(1, 2, 3)) == "1.2.3"
        assert str(SemanticVersion(1, 2, 3, prerelease="alpha")) == "1.2.3-alpha"
        assert str(SemanticVersion(1, 2, 3, build="b1")) == "1.2.3+b1"
        assert str(SemanticVersion(1, 2, 3, "alpha", "b1")) == "1.2.3-alpha+b1"

    def test_zero_version(self) -> None:
        """Test zero() class method."""
        v = SemanticVersion.zero()
        assert v == SemanticVersion(0, 0, 0)

    def test_get_compatibility_level(self) -> None:
        """Test get_compatibility_level method."""
        v1 = SemanticVersion(1, 2, 3)

        assert v1.get_compatibility_level(SemanticVersion(1, 2, 3)) == CompatibilityLevel.IDENTICAL
        assert v1.get_compatibility_level(SemanticVersion(1, 2, 4)) == CompatibilityLevel.PATCH_DIFF
        assert v1.get_compatibility_level(SemanticVersion(1, 3, 0)) == CompatibilityLevel.MINOR_DIFF
        assert v1.get_compatibility_level(SemanticVersion(2, 0, 0)) == CompatibilityLevel.INCOMPATIBLE

    def test_is_compatible_with_strategies(self) -> None:
        """Test is_compatible_with with different strategies."""
        v1 = SemanticVersion(1, 2, 3)

        # ANY accepts everything
        assert v1.is_compatible_with(SemanticVersion(9, 9, 9), strategy=CompatibilityStrategy.ANY)

        # STRICT requires exact match
        assert v1.is_compatible_with(SemanticVersion(1, 2, 3), strategy=CompatibilityStrategy.STRICT)
        assert not v1.is_compatible_with(SemanticVersion(1, 2, 4), strategy=CompatibilityStrategy.STRICT)

        # MAJOR requires same major
        assert v1.is_compatible_with(SemanticVersion(1, 9, 9), strategy=CompatibilityStrategy.MAJOR)
        assert not v1.is_compatible_with(SemanticVersion(2, 0, 0), strategy=CompatibilityStrategy.MAJOR)

        # MINOR requires same major.minor
        assert v1.is_compatible_with(SemanticVersion(1, 2, 9), strategy=CompatibilityStrategy.MINOR)
        assert not v1.is_compatible_with(SemanticVersion(1, 3, 0), strategy=CompatibilityStrategy.MINOR)


# =============================================================================
# VersionConstraint Tests
# =============================================================================


class TestVersionConstraintParsing:
    """Tests for VersionConstraint.parse()."""

    def test_parse_equality(self) -> None:
        """Test parsing equality constraint."""
        c = VersionConstraint.parse("==1.2.3")
        assert c.operator == VersionOperator.EQ
        assert c.version == SemanticVersion(1, 2, 3)

    def test_parse_not_equal(self) -> None:
        """Test parsing not-equal constraint."""
        c = VersionConstraint.parse("!=1.0.0")
        assert c.operator == VersionOperator.NE
        assert c.version == SemanticVersion(1, 0, 0)

    def test_parse_greater_than(self) -> None:
        """Test parsing greater-than constraint."""
        c = VersionConstraint.parse(">1.0.0")
        assert c.operator == VersionOperator.GT

    def test_parse_greater_or_equal(self) -> None:
        """Test parsing greater-or-equal constraint."""
        c = VersionConstraint.parse(">=1.0.0")
        assert c.operator == VersionOperator.GE

    def test_parse_less_than(self) -> None:
        """Test parsing less-than constraint."""
        c = VersionConstraint.parse("<2.0.0")
        assert c.operator == VersionOperator.LT

    def test_parse_less_or_equal(self) -> None:
        """Test parsing less-or-equal constraint."""
        c = VersionConstraint.parse("<=2.0.0")
        assert c.operator == VersionOperator.LE

    def test_parse_caret(self) -> None:
        """Test parsing caret constraint."""
        c = VersionConstraint.parse("^1.2.3")
        assert c.operator == VersionOperator.CARET
        assert c.version == SemanticVersion(1, 2, 3)

    def test_parse_tilde(self) -> None:
        """Test parsing tilde constraint."""
        c = VersionConstraint.parse("~1.2.3")
        assert c.operator == VersionOperator.TILDE

    def test_parse_wildcard(self) -> None:
        """Test parsing wildcard constraint."""
        c = VersionConstraint.parse("*")
        assert c.operator == VersionOperator.WILDCARD
        assert c.version is None

    def test_parse_implicit_equality(self) -> None:
        """Test parsing version without operator implies equality."""
        c = VersionConstraint.parse("1.2.3")
        assert c.operator == VersionOperator.EQ
        assert c.version == SemanticVersion(1, 2, 3)

    def test_parse_empty_raises(self) -> None:
        """Test that empty string raises VersionConstraintError."""
        with pytest.raises(VersionConstraintError):
            VersionConstraint.parse("")

    def test_parse_invalid_raises(self) -> None:
        """Test that invalid strings raise VersionConstraintError."""
        with pytest.raises(VersionConstraintError):
            VersionConstraint.parse("???invalid")


class TestVersionConstraintSatisfaction:
    """Tests for VersionConstraint.satisfied_by()."""

    def test_equality_satisfied(self) -> None:
        """Test equality constraint satisfaction."""
        c = VersionConstraint.parse("==1.2.3")
        assert c.satisfied_by(SemanticVersion(1, 2, 3))
        assert not c.satisfied_by(SemanticVersion(1, 2, 4))

    def test_not_equal_satisfied(self) -> None:
        """Test not-equal constraint satisfaction."""
        c = VersionConstraint.parse("!=1.0.0")
        assert c.satisfied_by(SemanticVersion(1, 0, 1))
        assert not c.satisfied_by(SemanticVersion(1, 0, 0))

    def test_greater_than_satisfied(self) -> None:
        """Test greater-than constraint satisfaction."""
        c = VersionConstraint.parse(">1.0.0")
        assert c.satisfied_by(SemanticVersion(1, 0, 1))
        assert c.satisfied_by(SemanticVersion(2, 0, 0))
        assert not c.satisfied_by(SemanticVersion(1, 0, 0))
        assert not c.satisfied_by(SemanticVersion(0, 9, 9))

    def test_greater_or_equal_satisfied(self) -> None:
        """Test greater-or-equal constraint satisfaction."""
        c = VersionConstraint.parse(">=1.0.0")
        assert c.satisfied_by(SemanticVersion(1, 0, 0))
        assert c.satisfied_by(SemanticVersion(1, 0, 1))
        assert not c.satisfied_by(SemanticVersion(0, 9, 9))

    def test_less_than_satisfied(self) -> None:
        """Test less-than constraint satisfaction."""
        c = VersionConstraint.parse("<2.0.0")
        assert c.satisfied_by(SemanticVersion(1, 9, 9))
        assert not c.satisfied_by(SemanticVersion(2, 0, 0))
        assert not c.satisfied_by(SemanticVersion(2, 0, 1))

    def test_less_or_equal_satisfied(self) -> None:
        """Test less-or-equal constraint satisfaction."""
        c = VersionConstraint.parse("<=2.0.0")
        assert c.satisfied_by(SemanticVersion(2, 0, 0))
        assert c.satisfied_by(SemanticVersion(1, 9, 9))
        assert not c.satisfied_by(SemanticVersion(2, 0, 1))

    def test_caret_satisfied_major_nonzero(self) -> None:
        """Test caret constraint with non-zero major."""
        c = VersionConstraint.parse("^1.2.3")
        assert c.satisfied_by(SemanticVersion(1, 2, 3))
        assert c.satisfied_by(SemanticVersion(1, 2, 9))
        assert c.satisfied_by(SemanticVersion(1, 9, 0))
        assert not c.satisfied_by(SemanticVersion(2, 0, 0))
        assert not c.satisfied_by(SemanticVersion(1, 2, 2))

    def test_caret_satisfied_major_zero(self) -> None:
        """Test caret constraint with zero major."""
        c = VersionConstraint.parse("^0.2.3")
        assert c.satisfied_by(SemanticVersion(0, 2, 3))
        assert c.satisfied_by(SemanticVersion(0, 2, 9))
        assert not c.satisfied_by(SemanticVersion(0, 3, 0))

    def test_caret_satisfied_major_minor_zero(self) -> None:
        """Test caret constraint with zero major and minor."""
        c = VersionConstraint.parse("^0.0.3")
        assert c.satisfied_by(SemanticVersion(0, 0, 3))
        assert not c.satisfied_by(SemanticVersion(0, 0, 4))

    def test_tilde_satisfied(self) -> None:
        """Test tilde constraint satisfaction."""
        c = VersionConstraint.parse("~1.2.3")
        assert c.satisfied_by(SemanticVersion(1, 2, 3))
        assert c.satisfied_by(SemanticVersion(1, 2, 9))
        assert not c.satisfied_by(SemanticVersion(1, 3, 0))
        assert not c.satisfied_by(SemanticVersion(1, 2, 2))

    def test_wildcard_satisfied(self) -> None:
        """Test wildcard constraint satisfaction."""
        c = VersionConstraint.parse("*")
        assert c.satisfied_by(SemanticVersion(0, 0, 1))
        assert c.satisfied_by(SemanticVersion(99, 99, 99))


class TestVersionConstraintFactoryMethods:
    """Tests for VersionConstraint factory methods."""

    def test_any_version(self) -> None:
        """Test any_version factory."""
        c = VersionConstraint.any_version()
        assert c.operator == VersionOperator.WILDCARD

    def test_exact(self) -> None:
        """Test exact factory."""
        c = VersionConstraint.exact("1.2.3")
        assert c.operator == VersionOperator.EQ
        assert c.version == SemanticVersion(1, 2, 3)

    def test_at_least(self) -> None:
        """Test at_least factory."""
        c = VersionConstraint.at_least("1.0.0")
        assert c.operator == VersionOperator.GE

    def test_less_than(self) -> None:
        """Test less_than factory."""
        c = VersionConstraint.less_than("2.0.0")
        assert c.operator == VersionOperator.LT

    def test_compatible_with(self) -> None:
        """Test compatible_with factory."""
        c = VersionConstraint.compatible_with("1.2.3")
        assert c.operator == VersionOperator.CARET

    def test_approximately(self) -> None:
        """Test approximately factory."""
        c = VersionConstraint.approximately("1.2.3")
        assert c.operator == VersionOperator.TILDE


# =============================================================================
# VersionRange Tests
# =============================================================================


class TestVersionRangeParsing:
    """Tests for VersionRange.parse()."""

    def test_parse_single_constraint(self) -> None:
        """Test parsing single constraint range."""
        r = VersionRange.parse(">=1.0.0")
        assert len(r.constraints) == 1
        assert r.constraints[0].operator == VersionOperator.GE

    def test_parse_multiple_constraints(self) -> None:
        """Test parsing multiple constraint range."""
        r = VersionRange.parse(">=1.0.0,<2.0.0")
        assert len(r.constraints) == 2
        assert r.constraints[0].operator == VersionOperator.GE
        assert r.constraints[1].operator == VersionOperator.LT

    def test_parse_wildcard(self) -> None:
        """Test parsing wildcard range."""
        r = VersionRange.parse("*")
        assert len(r.constraints) == 1
        assert r.constraints[0].operator == VersionOperator.WILDCARD

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string returns empty range."""
        r = VersionRange.parse("")
        assert len(r.constraints) == 0


class TestVersionRangeSatisfaction:
    """Tests for VersionRange.satisfied_by()."""

    def test_single_constraint_satisfied(self) -> None:
        """Test single constraint range satisfaction."""
        r = VersionRange.parse(">=1.0.0")
        assert r.satisfied_by(SemanticVersion(1, 0, 0))
        assert r.satisfied_by(SemanticVersion(2, 0, 0))
        assert not r.satisfied_by(SemanticVersion(0, 9, 0))

    def test_multiple_constraints_all_satisfied(self) -> None:
        """Test multiple constraint range - all must be satisfied."""
        r = VersionRange.parse(">=1.0.0,<2.0.0")
        assert r.satisfied_by(SemanticVersion(1, 0, 0))
        assert r.satisfied_by(SemanticVersion(1, 9, 9))
        assert not r.satisfied_by(SemanticVersion(2, 0, 0))
        assert not r.satisfied_by(SemanticVersion(0, 9, 0))

    def test_empty_range_satisfied_by_any(self) -> None:
        """Test empty range is satisfied by any version."""
        r = VersionRange.parse("")
        assert r.satisfied_by(SemanticVersion(1, 2, 3))

    def test_prerelease_rejection(self) -> None:
        """Test prerelease versions are rejected by default."""
        r = VersionRange.parse(">=1.0.0")
        # 1.0.0-alpha < 1.0.0 per SemVer, so it doesn't satisfy >=1.0.0
        assert not r.satisfied_by(SemanticVersion(1, 0, 0, prerelease="alpha"))
        # Even with allow_prerelease=True, 1.0.0-alpha < 1.0.0
        assert not r.satisfied_by(
            SemanticVersion(1, 0, 0, prerelease="alpha"),
            allow_prerelease=True,
        )
        # But 1.0.1-alpha > 1.0.0 and should be allowed with allow_prerelease
        r2 = VersionRange.parse(">=1.0.0")
        assert not r2.satisfied_by(SemanticVersion(1, 0, 1, prerelease="alpha"))  # rejected by default
        assert r2.satisfied_by(
            SemanticVersion(1, 0, 1, prerelease="alpha"),
            allow_prerelease=True,
        )


class TestVersionRangeFactoryMethods:
    """Tests for VersionRange factory methods."""

    def test_any_version(self) -> None:
        """Test any_version factory."""
        r = VersionRange.any_version()
        assert r.is_any()
        assert r.satisfied_by(SemanticVersion(99, 99, 99))

    def test_exact(self) -> None:
        """Test exact factory."""
        r = VersionRange.exact("1.2.3")
        assert r.satisfied_by(SemanticVersion(1, 2, 3))
        assert not r.satisfied_by(SemanticVersion(1, 2, 4))

    def test_at_least(self) -> None:
        """Test at_least factory."""
        r = VersionRange.at_least("1.0.0")
        assert r.satisfied_by(SemanticVersion(1, 0, 0))
        assert r.satisfied_by(SemanticVersion(2, 0, 0))

    def test_between_exclusive(self) -> None:
        """Test between factory with exclusive max."""
        r = VersionRange.between("1.0.0", "2.0.0")
        assert r.satisfied_by(SemanticVersion(1, 0, 0))
        assert r.satisfied_by(SemanticVersion(1, 9, 9))
        assert not r.satisfied_by(SemanticVersion(2, 0, 0))

    def test_between_inclusive(self) -> None:
        """Test between factory with inclusive max."""
        r = VersionRange.between("1.0.0", "2.0.0", include_max=True)
        assert r.satisfied_by(SemanticVersion(2, 0, 0))
        assert not r.satisfied_by(SemanticVersion(2, 0, 1))

    def test_compatible_with(self) -> None:
        """Test compatible_with factory."""
        r = VersionRange.compatible_with("1.2.3")
        assert r.satisfied_by(SemanticVersion(1, 9, 0))
        assert not r.satisfied_by(SemanticVersion(2, 0, 0))


class TestVersionRangeMethods:
    """Tests for VersionRange instance methods."""

    def test_is_empty(self) -> None:
        """Test is_empty method."""
        assert VersionRange(()).is_empty()
        assert not VersionRange.parse(">=1.0.0").is_empty()

    def test_is_any(self) -> None:
        """Test is_any method."""
        assert VersionRange.parse("*").is_any()
        assert not VersionRange.parse(">=1.0.0").is_any()

    def test_and_constraint(self) -> None:
        """Test adding a constraint."""
        r = VersionRange.parse(">=1.0.0")
        r2 = r.and_constraint(VersionConstraint.parse("<2.0.0"))
        assert len(r2.constraints) == 2

    def test_and_range(self) -> None:
        """Test combining ranges."""
        r1 = VersionRange.parse(">=1.0.0")
        r2 = VersionRange.parse("<2.0.0")
        combined = r1.and_range(r2)
        assert len(combined.constraints) == 2

    def test_str_representation(self) -> None:
        """Test string representation."""
        r = VersionRange.parse(">=1.0.0,<2.0.0")
        assert ">=1.0.0" in str(r)
        assert "<2.0.0" in str(r)


# =============================================================================
# VersionCompatibilityChecker Tests
# =============================================================================


class TestVersionCompatibilityChecker:
    """Tests for VersionCompatibilityChecker."""

    def test_check_compatible(self) -> None:
        """Test check returns compatible result."""
        checker = VersionCompatibilityChecker()
        result = checker.check(
            SemanticVersion(1, 2, 3),
            VersionRange.parse(">=1.0.0"),
            engine_name="test",
        )
        assert result.compatible
        assert result.engine_name == "test"

    def test_check_incompatible(self) -> None:
        """Test check returns incompatible result."""
        checker = VersionCompatibilityChecker()
        result = checker.check(
            SemanticVersion(0, 9, 0),
            VersionRange.parse(">=1.0.0"),
        )
        assert not result.compatible

    def test_check_strict_raises(self) -> None:
        """Test check with strict config raises on incompatibility."""
        checker = VersionCompatibilityChecker(STRICT_VERSION_COMPATIBILITY_CONFIG)
        with pytest.raises(VersionIncompatibleError):
            checker.check(
                SemanticVersion(0, 9, 0),
                VersionRange.parse(">=1.0.0"),
                engine_name="test",
            )

    def test_check_prerelease_warning(self) -> None:
        """Test check generates warning for prerelease."""
        checker = VersionCompatibilityChecker()
        result = checker.check(
            SemanticVersion(1, 0, 0, prerelease="alpha"),
            VersionRange.parse(">=1.0.0-alpha"),
            engine_name="test",
        )
        assert len(result.warnings) > 0
        assert "prerelease" in result.warnings[0].lower()

    def test_check_lenient_config(self) -> None:
        """Test check with lenient config allows prerelease."""
        checker = VersionCompatibilityChecker(LENIENT_VERSION_COMPATIBILITY_CONFIG)
        # Use 1.0.1-alpha which is > 1.0.0 so it satisfies >=1.0.0
        result = checker.check(
            SemanticVersion(1, 0, 1, prerelease="alpha"),
            VersionRange.parse(">=1.0.0"),
        )
        assert result.compatible
        assert len(result.warnings) == 0  # No warnings in lenient mode


# =============================================================================
# EngineVersionRequirement Tests
# =============================================================================


class TestEngineVersionRequirement:
    """Tests for EngineVersionRequirement."""

    def test_parse_simple(self) -> None:
        """Test parsing simple requirement."""
        req = EngineVersionRequirement.parse("truthound >=1.0.0")
        assert req.engine_name == "truthound"
        assert not req.optional

    def test_parse_optional(self) -> None:
        """Test parsing optional requirement."""
        req = EngineVersionRequirement.parse("truthound >=1.0.0 (optional)")
        assert req.engine_name == "truthound"
        assert req.optional

    def test_parse_just_name(self) -> None:
        """Test parsing requirement with just engine name."""
        req = EngineVersionRequirement.parse("truthound")
        assert req.engine_name == "truthound"
        assert req.version_range.is_any()

    def test_satisfied_by(self) -> None:
        """Test satisfied_by method."""
        req = EngineVersionRequirement.parse("truthound >=1.0.0")
        assert req.satisfied_by(SemanticVersion(1, 0, 0))
        assert not req.satisfied_by(SemanticVersion(0, 9, 0))


# =============================================================================
# VersionRegistry Tests
# =============================================================================


class TestVersionRegistry:
    """Tests for VersionRegistry."""

    def test_register_engine(self) -> None:
        """Test registering an engine."""
        registry = VersionRegistry()
        registry.register_engine("test", "1.2.3")
        assert registry.get_engine_version("test") == SemanticVersion(1, 2, 3)

    def test_register_engine_with_semantic_version(self) -> None:
        """Test registering with SemanticVersion object."""
        registry = VersionRegistry()
        registry.register_engine("test", SemanticVersion(1, 2, 3))
        assert registry.get_engine_version("test") == SemanticVersion(1, 2, 3)

    def test_unregister_engine(self) -> None:
        """Test unregistering an engine."""
        registry = VersionRegistry()
        registry.register_engine("test", "1.0.0")
        assert registry.unregister_engine("test")
        assert registry.get_engine_version("test") is None
        assert not registry.unregister_engine("nonexistent")

    def test_add_requirement(self) -> None:
        """Test adding requirements."""
        registry = VersionRegistry()
        registry.add_requirement(EngineVersionRequirement.parse("test >=1.0.0"))
        assert len(registry.list_requirements()) == 1

    def test_add_requirements_strings(self) -> None:
        """Test adding requirements from strings."""
        registry = VersionRegistry()
        registry.add_requirements(["test >=1.0.0", "other >=2.0.0"])
        assert len(registry.list_requirements()) == 2

    def test_clear_requirements(self) -> None:
        """Test clearing requirements."""
        registry = VersionRegistry()
        registry.add_requirement(EngineVersionRequirement.parse("test >=1.0.0"))
        registry.clear_requirements()
        assert len(registry.list_requirements()) == 0

    def test_check_requirement_satisfied(self) -> None:
        """Test checking satisfied requirement."""
        registry = VersionRegistry()
        registry.register_engine("test", "1.2.3")
        req = EngineVersionRequirement.parse("test >=1.0.0")
        result = registry.check_requirement(req)
        assert result is not None
        assert result.compatible

    def test_check_requirement_missing_optional(self) -> None:
        """Test checking optional requirement for missing engine."""
        registry = VersionRegistry()
        req = EngineVersionRequirement.parse("test >=1.0.0 (optional)")
        result = registry.check_requirement(req)
        assert result is None

    def test_check_requirement_missing_required(self) -> None:
        """Test checking required requirement for missing engine."""
        registry = VersionRegistry()
        req = EngineVersionRequirement.parse("test >=1.0.0")
        result = registry.check_requirement(req)
        assert result is not None
        assert not result.compatible

    def test_validate_all(self) -> None:
        """Test validating all requirements."""
        registry = VersionRegistry()
        registry.register_engine("test", "1.2.3")
        registry.add_requirement(EngineVersionRequirement.parse("test >=1.0.0"))
        results = registry.validate_all()
        assert "test" in results
        assert results["test"].compatible

    def test_are_all_compatible_true(self) -> None:
        """Test are_all_compatible returns True when all satisfied."""
        registry = VersionRegistry()
        registry.register_engine("test", "1.2.3")
        registry.add_requirement(EngineVersionRequirement.parse("test >=1.0.0"))
        assert registry.are_all_compatible()

    def test_are_all_compatible_false(self) -> None:
        """Test are_all_compatible returns False when not satisfied."""
        registry = VersionRegistry()
        registry.register_engine("test", "0.9.0")
        registry.add_requirement(EngineVersionRequirement.parse("test >=1.0.0"))
        assert not registry.are_all_compatible()

    def test_get_incompatible(self) -> None:
        """Test getting incompatible results."""
        registry = VersionRegistry()
        registry.register_engine("test", "0.9.0")
        registry.add_requirement(EngineVersionRequirement.parse("test >=1.0.0"))
        incompatible = registry.get_incompatible()
        assert len(incompatible) == 1
        assert incompatible[0].engine_name == "test"

    def test_list_engines(self) -> None:
        """Test listing registered engines."""
        registry = VersionRegistry()
        registry.register_engine("a", "1.0.0")
        registry.register_engine("b", "2.0.0")
        engines = registry.list_engines()
        assert "a" in engines
        assert "b" in engines

    def test_to_dict(self) -> None:
        """Test converting registry to dict."""
        registry = VersionRegistry()
        registry.register_engine("test", "1.2.3", info={"key": "value"})
        d = registry.to_dict()
        assert "engines" in d
        assert "test" in d["engines"]
        assert d["engines"]["test"] == "1.2.3"


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalVersionRegistry:
    """Tests for global version registry functions."""

    def test_get_version_registry_singleton(self) -> None:
        """Test that get_version_registry returns same instance."""
        reset_version_registry()
        r1 = get_version_registry()
        r2 = get_version_registry()
        assert r1 is r2

    def test_reset_version_registry(self) -> None:
        """Test reset_version_registry creates new instance."""
        r1 = get_version_registry()
        r1.register_engine("test", "1.0.0")
        reset_version_registry()
        r2 = get_version_registry()
        assert r2.get_engine_version("test") is None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_parse_version(self) -> None:
        """Test parse_version function."""
        v = parse_version("1.2.3")
        assert v == SemanticVersion(1, 2, 3)

    def test_parse_constraint(self) -> None:
        """Test parse_constraint function."""
        c = parse_constraint(">=1.0.0")
        assert c.operator == VersionOperator.GE

    def test_parse_range(self) -> None:
        """Test parse_range function."""
        r = parse_range(">=1.0.0,<2.0.0")
        assert len(r.constraints) == 2

    def test_compare_versions(self) -> None:
        """Test compare_versions function."""
        assert compare_versions("1.0.0", "2.0.0") == -1
        assert compare_versions("2.0.0", "1.0.0") == 1
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_versions_compatible(self) -> None:
        """Test versions_compatible function."""
        # SEMVER strategy: same major AND v1 >= v2
        # 1.2.0 and 1.5.0 have same major, but 1.2.0 < 1.5.0
        assert not versions_compatible("1.2.0", "1.5.0")  # 1.2.0 < 1.5.0
        assert versions_compatible("1.5.0", "1.2.0")  # 1.5.0 >= 1.2.0
        assert not versions_compatible("1.2.0", "2.0.0")  # different major
        # With MAJOR strategy
        assert versions_compatible("1.2.0", "1.5.0", strategy=CompatibilityStrategy.MAJOR)
        assert not versions_compatible("1.2.0", "2.0.0", strategy=CompatibilityStrategy.MAJOR)


class TestCheckVersionCompatibility:
    """Tests for check_version_compatibility with mock engines."""

    def test_with_compatible_engine(self) -> None:
        """Test check_version_compatibility with compatible version."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "1.2.3"

        result = check_version_compatibility(MockEngine(), ">=1.0.0")
        assert result.compatible

    def test_with_incompatible_engine(self) -> None:
        """Test check_version_compatibility with incompatible version."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "0.9.0"

        result = check_version_compatibility(MockEngine(), ">=1.0.0")
        assert not result.compatible

    def test_with_unparseable_version(self) -> None:
        """Test check_version_compatibility with unparseable version."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "not-a-version"

        result = check_version_compatibility(MockEngine(), ">=1.0.0")
        assert not result.compatible
        assert "Could not parse" in result.message

    def test_is_version_compatible(self) -> None:
        """Test is_version_compatible convenience function."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "1.2.3"

        assert is_version_compatible(MockEngine(), ">=1.0.0")
        assert not is_version_compatible(MockEngine(), ">=2.0.0")

    def test_require_version_raises_on_incompatible(self) -> None:
        """Test require_version raises for incompatible version."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "0.9.0"

        with pytest.raises(VersionIncompatibleError) as exc_info:
            require_version(MockEngine(), ">=1.0.0")
        assert exc_info.value.engine_name == "mock"


# =============================================================================
# Decorator Tests
# =============================================================================


class TestVersionRequiredDecorator:
    """Tests for version_required decorator."""

    def test_decorator_allows_compatible(self) -> None:
        """Test decorator allows compatible engine."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "1.2.3"

        @version_required(">=1.0.0")
        def process(engine):
            return "success"

        result = process(engine=MockEngine())
        assert result == "success"

    def test_decorator_raises_on_incompatible(self) -> None:
        """Test decorator raises for incompatible engine."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "0.9.0"

        @version_required(">=1.0.0")
        def process(engine):
            return "success"

        with pytest.raises(VersionIncompatibleError):
            process(engine=MockEngine())

    def test_decorator_with_positional_arg(self) -> None:
        """Test decorator works with positional engine argument."""
        class MockEngine:
            engine_name = "mock"
            engine_version = "1.2.3"

        @version_required(">=1.0.0", engine_param="engine")
        def process(engine, data):
            return data

        result = process(MockEngine(), "test_data")
        assert result == "test_data"


# =============================================================================
# Exception Tests
# =============================================================================


class TestVersionExceptions:
    """Tests for version exception classes."""

    def test_version_parse_error(self) -> None:
        """Test VersionParseError attributes."""
        e = VersionParseError("bad", "invalid format")
        assert e.version_string == "bad"
        assert e.reason == "invalid format"
        assert "bad" in str(e)

    def test_version_constraint_error(self) -> None:
        """Test VersionConstraintError attributes."""
        e = VersionConstraintError("???", "unknown operator")
        assert e.constraint_string == "???"
        assert e.reason == "unknown operator"

    def test_version_incompatible_error(self) -> None:
        """Test VersionIncompatibleError attributes."""
        e = VersionIncompatibleError("engine", "0.9.0", ">=1.0.0")
        assert e.engine_name == "engine"
        assert e.engine_version == "0.9.0"
        assert e.required == ">=1.0.0"


# =============================================================================
# Integration Tests
# =============================================================================


class TestVersionIntegration:
    """Integration tests combining multiple version components."""

    def test_full_workflow(self) -> None:
        """Test full version checking workflow."""
        # Create registry
        registry = VersionRegistry()

        # Register engines
        registry.register_engine("truthound", "1.5.0")
        registry.register_engine("great_expectations", "0.18.0")

        # Add requirements
        registry.add_requirements([
            "truthound >=1.0.0,<2.0.0",
            "great_expectations >=0.17.0",
        ])

        # Validate
        assert registry.are_all_compatible()

        # Check individual
        truthound_result = registry.validate_all().get("truthound")
        assert truthound_result is not None
        assert truthound_result.compatible

    def test_complex_range_workflow(self) -> None:
        """Test complex version range scenarios."""
        # SemVer compatible versions
        range1 = VersionRange.compatible_with("1.2.3")
        assert range1.satisfied_by(SemanticVersion(1, 2, 3))
        assert range1.satisfied_by(SemanticVersion(1, 9, 0))
        assert not range1.satisfied_by(SemanticVersion(2, 0, 0))

        # Combined constraints
        range2 = VersionRange.between("1.0.0", "2.0.0")
        range2 = range2.and_constraint(VersionConstraint.parse("!=1.5.0"))
        assert range2.satisfied_by(SemanticVersion(1, 4, 0))
        assert not range2.satisfied_by(SemanticVersion(1, 5, 0))
