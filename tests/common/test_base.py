"""Tests for common.base module."""

import json

import pytest

from common.base import (
    CheckConfig,
    CheckResult,
    CheckResultBuilder,
    CheckStatus,
    ColumnProfile,
    FailureAction,
    LearnConfig,
    LearnedRule,
    LearnResult,
    LearnStatus,
    ProfileConfig,
    ProfileResult,
    ProfileStatus,
    Severity,
    ValidationFailure,
)


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_is_success(self):
        """Test is_success method."""
        assert CheckStatus.PASSED.is_success() is True
        assert CheckStatus.WARNING.is_success() is True
        assert CheckStatus.SKIPPED.is_success() is True
        assert CheckStatus.FAILED.is_success() is False
        assert CheckStatus.ERROR.is_success() is False

    def test_is_terminal_failure(self):
        """Test is_terminal_failure method."""
        assert CheckStatus.FAILED.is_terminal_failure() is True
        assert CheckStatus.ERROR.is_terminal_failure() is True
        assert CheckStatus.PASSED.is_terminal_failure() is False
        assert CheckStatus.WARNING.is_terminal_failure() is False


class TestSeverity:
    """Tests for Severity enum."""

    def test_weight(self):
        """Test severity weights."""
        assert Severity.CRITICAL.weight > Severity.ERROR.weight
        assert Severity.ERROR.weight > Severity.WARNING.weight
        assert Severity.WARNING.weight > Severity.INFO.weight

    def test_comparison(self):
        """Test severity comparison operators."""
        assert Severity.INFO < Severity.WARNING
        assert Severity.WARNING < Severity.ERROR
        assert Severity.ERROR < Severity.CRITICAL

        assert Severity.CRITICAL > Severity.ERROR
        assert Severity.CRITICAL >= Severity.CRITICAL
        assert Severity.INFO <= Severity.INFO


class TestValidationFailure:
    """Tests for ValidationFailure dataclass."""

    def test_basic_creation(self):
        """Test basic failure creation."""
        failure = ValidationFailure(
            rule_name="not_null",
            column="email",
            message="Found null values",
        )
        assert failure.rule_name == "not_null"
        assert failure.column == "email"
        assert failure.message == "Found null values"

    def test_failure_rate(self):
        """Test failure rate calculation."""
        failure = ValidationFailure(
            rule_name="test",
            failed_count=10,
            total_count=100,
        )
        assert failure.failure_rate == 10.0

    def test_failure_rate_zero_total(self):
        """Test failure rate with zero total."""
        failure = ValidationFailure(rule_name="test", failed_count=0, total_count=0)
        assert failure.failure_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        failure = ValidationFailure(
            rule_name="unique",
            column="id",
            message="Duplicates found",
            severity=Severity.ERROR,
            failed_count=5,
            total_count=100,
            sample_values=(1, 2, 3),
        )
        result = failure.to_dict()

        assert result["rule_name"] == "unique"
        assert result["column"] == "id"
        assert result["severity"] == "ERROR"
        assert result["sample_values"] == [1, 2, 3]
        assert result["failure_rate"] == 5.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "rule_name": "not_null",
            "column": "name",
            "message": "Null found",
            "severity": "WARNING",
            "failed_count": 2,
            "total_count": 50,
            "sample_values": [None, None],
        }
        failure = ValidationFailure.from_dict(data)

        assert failure.rule_name == "not_null"
        assert failure.column == "name"
        assert failure.severity == Severity.WARNING
        assert failure.sample_values == (None, None)

    def test_immutable(self):
        """Test that failure is immutable."""
        failure = ValidationFailure(rule_name="test")
        with pytest.raises(AttributeError):
            failure.rule_name = "modified"  # type: ignore


class TestCheckConfig:
    """Tests for CheckConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CheckConfig()
        assert config.rules == ()
        assert config.fail_on_error is True
        assert config.failure_action == FailureAction.RAISE
        assert config.sample_size is None
        assert config.parallel is False
        assert config.timeout_seconds is None
        assert config.tags == frozenset()

    def test_with_rules(self):
        """Test with_rules builder method."""
        config = CheckConfig(rules=({"type": "not_null"},))
        new_config = config.with_rules({"type": "unique"})

        # Original unchanged
        assert len(config.rules) == 1

        # New config has both rules
        assert len(new_config.rules) == 2
        assert new_config.rules[0]["type"] == "not_null"
        assert new_config.rules[1]["type"] == "unique"

    def test_with_timeout(self):
        """Test with_timeout builder method."""
        config = CheckConfig()
        new_config = config.with_timeout(60)

        assert config.timeout_seconds is None
        assert new_config.timeout_seconds == 60

    def test_with_failure_action(self):
        """Test with_failure_action builder method."""
        config = CheckConfig()
        new_config = config.with_failure_action(FailureAction.WARN)

        assert config.failure_action == FailureAction.RAISE
        assert new_config.failure_action == FailureAction.WARN

    def test_with_tags(self):
        """Test with_tags builder method."""
        config = CheckConfig(tags=frozenset(["a"]))
        new_config = config.with_tags("b", "c")

        assert config.tags == frozenset(["a"])
        assert new_config.tags == frozenset(["a", "b", "c"])

    def test_with_extra(self):
        """Test with_extra builder method."""
        config = CheckConfig(extra={"x": 1})
        new_config = config.with_extra(y=2)

        assert config.extra == {"x": 1}
        assert new_config.extra == {"x": 1, "y": 2}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CheckConfig(
            rules=({"type": "not_null"},),
            failure_action=FailureAction.WARN,
            timeout_seconds=30,
            tags=frozenset(["test"]),
        )
        result = config.to_dict()

        assert result["rules"] == [{"type": "not_null"}]
        assert result["failure_action"] == "WARN"
        assert result["timeout_seconds"] == 30
        assert "test" in result["tags"]

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "rules": [{"type": "unique", "column": "id"}],
            "failure_action": "LOG",
            "parallel": True,
            "tags": ["critical"],
        }
        config = CheckConfig.from_dict(data)

        assert len(config.rules) == 1
        assert config.failure_action == FailureAction.LOG
        assert config.parallel is True
        assert "critical" in config.tags

    def test_to_truthound_kwargs(self):
        """Test conversion to Truthound API kwargs."""
        config = CheckConfig(
            rules=({"type": "not_null"},),
            sample_size=1000,
            parallel=True,
            timeout_seconds=60,
        )
        kwargs = config.to_truthound_kwargs()

        assert kwargs["rules"] == [{"type": "not_null"}]
        assert kwargs["sample_size"] == 1000
        assert kwargs["parallel"] is True
        assert kwargs["timeout"] == 60


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = CheckResult(
            status=CheckStatus.PASSED,
            passed_count=10,
        )
        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 10
        assert result.is_success is True

    def test_is_success(self):
        """Test is_success property."""
        passed = CheckResult(status=CheckStatus.PASSED)
        failed = CheckResult(status=CheckStatus.FAILED)
        warning = CheckResult(status=CheckStatus.WARNING)

        assert passed.is_success is True
        assert failed.is_success is False
        assert warning.is_success is True

    def test_total_count(self):
        """Test total_count property."""
        result = CheckResult(
            status=CheckStatus.PASSED,
            passed_count=10,
            failed_count=2,
            warning_count=1,
            skipped_count=3,
        )
        assert result.total_count == 16

    def test_pass_rate(self):
        """Test pass_rate calculation."""
        result = CheckResult(
            status=CheckStatus.PASSED,
            passed_count=90,
            failed_count=10,
        )
        assert result.pass_rate == 90.0

    def test_pass_rate_zero_total(self):
        """Test pass_rate with zero total."""
        result = CheckResult(status=CheckStatus.SKIPPED)
        assert result.pass_rate == 100.0

    def test_failure_rate(self):
        """Test failure_rate calculation."""
        result = CheckResult(
            status=CheckStatus.FAILED,
            passed_count=80,
            failed_count=20,
        )
        assert result.failure_rate == 20.0

    def test_iter_failures(self):
        """Test failure iteration with filtering."""
        failures = (
            ValidationFailure(rule_name="r1", severity=Severity.ERROR),
            ValidationFailure(rule_name="r2", severity=Severity.WARNING),
            ValidationFailure(rule_name="r3", severity=Severity.CRITICAL),
        )
        result = CheckResult(status=CheckStatus.FAILED, failures=failures)

        # All failures
        all_failures = list(result.iter_failures())
        assert len(all_failures) == 3

        # Only ERROR and above
        error_failures = list(result.iter_failures(min_severity=Severity.ERROR))
        assert len(error_failures) == 2

        # Only CRITICAL
        critical_failures = list(result.iter_failures(min_severity=Severity.CRITICAL))
        assert len(critical_failures) == 1

    def test_get_critical_failures(self):
        """Test get_critical_failures method."""
        failures = (
            ValidationFailure(rule_name="r1", severity=Severity.CRITICAL),
            ValidationFailure(rule_name="r2", severity=Severity.ERROR),
            ValidationFailure(rule_name="r3", severity=Severity.CRITICAL),
        )
        result = CheckResult(status=CheckStatus.FAILED, failures=failures)

        critical = result.get_critical_failures()
        assert len(critical) == 2
        assert all(f.severity == Severity.CRITICAL for f in critical)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CheckResult(
            status=CheckStatus.PASSED,
            passed_count=10,
            failed_count=0,
            execution_time_ms=100.5,
        )
        data = result.to_dict()

        assert data["status"] == "PASSED"
        assert data["passed_count"] == 10
        assert data["is_success"] is True
        assert data["pass_rate"] == 100.0

    def test_to_json(self):
        """Test JSON serialization."""
        result = CheckResult(
            status=CheckStatus.PASSED,
            passed_count=5,
        )
        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["status"] == "PASSED"
        assert data["passed_count"] == 5

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "status": "FAILED",
            "passed_count": 8,
            "failed_count": 2,
            "failures": [
                {"rule_name": "not_null", "column": "id", "severity": "ERROR"}
            ],
        }
        result = CheckResult.from_dict(data)

        assert result.status == CheckStatus.FAILED
        assert result.passed_count == 8
        assert len(result.failures) == 1


class TestCheckResultBuilder:
    """Tests for CheckResultBuilder."""

    def test_basic_build(self):
        """Test basic result building."""
        builder = CheckResultBuilder()
        result = builder.with_passed(10).with_failed(2).build()

        assert result.passed_count == 10
        assert result.failed_count == 2
        assert result.status == CheckStatus.FAILED

    def test_build_with_failures(self):
        """Test building with failures."""
        failure = ValidationFailure(rule_name="test")
        builder = CheckResultBuilder()
        result = builder.with_passed(5).add_failure(failure).build()

        assert len(result.failures) == 1

    def test_auto_status_passed(self):
        """Test automatic status determination - passed."""
        result = CheckResultBuilder().with_passed(10).build()
        assert result.status == CheckStatus.PASSED

    def test_auto_status_failed(self):
        """Test automatic status determination - failed."""
        result = CheckResultBuilder().with_passed(8).with_failed(2).build()
        assert result.status == CheckStatus.FAILED

    def test_auto_status_warning(self):
        """Test automatic status determination - warning."""
        result = CheckResultBuilder().with_passed(8).with_warnings(2).build()
        assert result.status == CheckStatus.WARNING

    def test_auto_status_skipped(self):
        """Test automatic status determination - skipped."""
        result = CheckResultBuilder().build()
        assert result.status == CheckStatus.SKIPPED


class TestProfileConfig:
    """Tests for ProfileConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProfileConfig()
        assert config.columns is None
        assert config.include_histograms is True
        assert config.include_correlations is False

    def test_with_columns(self):
        """Test with_columns builder method."""
        config = ProfileConfig()
        new_config = config.with_columns("a", "b", "c")

        assert config.columns is None
        assert new_config.columns == ("a", "b", "c")

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        config = ProfileConfig(
            columns=("x", "y"),
            include_histograms=False,
            sample_size=1000,
        )
        data = config.to_dict()
        restored = ProfileConfig.from_dict(data)

        assert restored.columns == ("x", "y")
        assert restored.include_histograms is False
        assert restored.sample_size == 1000


class TestLearnConfig:
    """Tests for LearnConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LearnConfig()
        assert config.include_types is True
        assert config.include_ranges is True
        assert config.include_patterns is False
        assert config.confidence_threshold == 0.95


class TestColumnProfile:
    """Tests for ColumnProfile dataclass."""

    def test_basic_creation(self):
        """Test basic profile creation."""
        profile = ColumnProfile(
            column_name="id",
            dtype="Int64",
            null_count=0,
            unique_count=100,
        )
        assert profile.column_name == "id"
        assert profile.dtype == "Int64"

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        profile = ColumnProfile(
            column_name="value",
            dtype="Float64",
            null_count=5,
            null_percentage=5.0,
            min_value=0.0,
            max_value=100.0,
            mean=50.0,
        )
        data = profile.to_dict()
        restored = ColumnProfile.from_dict(data)

        assert restored.column_name == "value"
        assert restored.mean == 50.0


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = ProfileResult(
            status=ProfileStatus.COMPLETED,
            row_count=1000,
        )
        assert result.is_success is True
        assert result.row_count == 1000

    def test_get_column(self):
        """Test get_column method."""
        columns = (
            ColumnProfile(column_name="a", dtype="Int64"),
            ColumnProfile(column_name="b", dtype="Utf8"),
        )
        result = ProfileResult(
            status=ProfileStatus.COMPLETED,
            columns=columns,
        )

        assert result.get_column("a") is not None
        assert result.get_column("a").dtype == "Int64"
        assert result.get_column("nonexistent") is None


class TestLearnedRule:
    """Tests for LearnedRule dataclass."""

    def test_to_rule_dict(self):
        """Test conversion to rule dictionary."""
        rule = LearnedRule(
            rule_type="not_null",
            column="id",
            parameters={"allow_empty": False},
            confidence=0.99,
        )
        rule_dict = rule.to_rule_dict()

        assert rule_dict["type"] == "not_null"
        assert rule_dict["column"] == "id"
        assert rule_dict["allow_empty"] is False


class TestLearnResult:
    """Tests for LearnResult dataclass."""

    def test_to_check_config(self):
        """Test conversion to CheckConfig."""
        rules = (
            LearnedRule(rule_type="not_null", column="id"),
            LearnedRule(rule_type="unique", column="id"),
        )
        result = LearnResult(
            status=LearnStatus.COMPLETED,
            rules=rules,
        )

        config = result.to_check_config(failure_action=FailureAction.WARN)
        assert len(config.rules) == 2
        assert config.failure_action == FailureAction.WARN

    def test_filter_by_confidence(self):
        """Test filtering rules by confidence."""
        rules = (
            LearnedRule(rule_type="r1", column="a", confidence=0.99),
            LearnedRule(rule_type="r2", column="b", confidence=0.80),
            LearnedRule(rule_type="r3", column="c", confidence=0.95),
        )
        result = LearnResult(status=LearnStatus.COMPLETED, rules=rules)

        high_confidence = result.filter_by_confidence(0.90)
        assert len(high_confidence) == 2
