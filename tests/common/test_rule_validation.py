"""Tests for Rule Validation System.

This module tests the rule validation system including:
- FieldSchema validation
- RuleSchema validation
- RuleRegistry management
- StandardRuleValidator and engine-specific validators
- Rule normalization
- Engine integration utilities
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from common.rule_validation import (
    # Enums
    FieldType,
    RuleCategory,
    # Exceptions
    RuleValidationError,
    UnknownRuleTypeError,
    MissingFieldError,
    InvalidFieldTypeError,
    InvalidFieldValueError,
    MultipleRuleValidationErrors,
    # Schema Types
    FieldSchema,
    RuleSchema,
    RuleValidationResult,
    BatchValidationResult,
    # Validators
    StandardRuleValidator,
    TruthoundRuleValidator,
    GreatExpectationsRuleValidator,
    PanderaRuleValidator,
    # Registry
    RuleRegistry,
    get_rule_registry,
    reset_rule_registry,
    COMMON_RULE_SCHEMAS,
    # Normalizer
    RuleNormalizer,
    # Convenience Functions
    validate_rule,
    validate_rules,
    normalize_rules,
    get_validator_for_engine,
    register_rule_schema,
    get_supported_rule_types,
    # Engine Integration
    ValidatingEngineWrapper,
    wrap_engine_with_validation,
    validate_rules_decorator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the global registry before each test."""
    reset_rule_registry()
    yield
    reset_rule_registry()


@pytest.fixture
def sample_rules() -> list[dict[str, Any]]:
    """Sample valid rules for testing."""
    return [
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
        {"type": "in_range", "column": "age", "min": 0, "max": 150},
        {"type": "in_set", "column": "status", "values": ["active", "inactive"]},
        {"type": "regex", "column": "phone", "pattern": r"^\d{3}-\d{4}$"},
    ]


@pytest.fixture
def mock_engine():
    """Mock engine for testing wrapper."""
    from common.base import CheckResult, CheckStatus, ProfileResult, ProfileStatus, LearnResult, LearnStatus

    class MockEngine:
        @property
        def engine_name(self) -> str:
            # Use a supported engine name for testing
            return "truthound"

        @property
        def engine_version(self) -> str:
            return "1.0.0"

        def check(self, data, rules=None, **kwargs):
            return CheckResult(
                status=CheckStatus.PASSED,
                passed_count=len(rules) if rules else 0,
            )

        def profile(self, data, **kwargs):
            return ProfileResult(status=ProfileStatus.COMPLETED)

        def learn(self, data, **kwargs):
            return LearnResult(status=LearnStatus.COMPLETED)

    return MockEngine()


# =============================================================================
# FieldType Tests
# =============================================================================


class TestFieldType:
    """Tests for FieldType enum."""

    def test_all_types_defined(self):
        """Verify all expected types are defined."""
        expected = {"STRING", "INTEGER", "FLOAT", "NUMBER", "BOOLEAN", "LIST", "DICT", "ANY", "REGEX"}
        actual = {t.name for t in FieldType}
        assert actual == expected


# =============================================================================
# RuleCategory Tests
# =============================================================================


class TestRuleCategory:
    """Tests for RuleCategory enum."""

    def test_all_categories_defined(self):
        """Verify all expected categories are defined."""
        expected = {"COMPLETENESS", "UNIQUENESS", "VALIDITY", "CONSISTENCY", "ACCURACY", "TIMELINESS"}
        actual = {c.name for c in RuleCategory}
        assert actual == expected


# =============================================================================
# FieldSchema Tests
# =============================================================================


class TestFieldSchema:
    """Tests for FieldSchema validation."""

    def test_string_field_validation(self):
        """Test string field validation."""
        field = FieldSchema(
            name="column",
            field_type=FieldType.STRING,
            required=True,
        )

        is_valid, error = field.validate("test_column")
        assert is_valid
        assert error is None

        is_valid, error = field.validate(123)
        assert not is_valid
        assert "expected string" in error.lower()

    def test_integer_field_validation(self):
        """Test integer field validation."""
        field = FieldSchema(
            name="count",
            field_type=FieldType.INTEGER,
            required=True,
            min_value=0,
            max_value=100,
        )

        is_valid, error = field.validate(50)
        assert is_valid

        is_valid, error = field.validate(-1)
        assert not is_valid
        assert "must be >=" in error

        is_valid, error = field.validate(101)
        assert not is_valid
        assert "must be <=" in error

        # Boolean should not pass as integer
        is_valid, error = field.validate(True)
        assert not is_valid

    def test_float_field_validation(self):
        """Test float field validation."""
        field = FieldSchema(
            name="threshold",
            field_type=FieldType.FLOAT,
            required=True,
        )

        is_valid, error = field.validate(0.5)
        assert is_valid

        is_valid, error = field.validate(1)
        assert not is_valid
        assert "expected float" in error.lower()

    def test_number_field_validation(self):
        """Test number (int or float) field validation."""
        field = FieldSchema(
            name="value",
            field_type=FieldType.NUMBER,
            required=True,
        )

        is_valid, error = field.validate(10)
        assert is_valid

        is_valid, error = field.validate(10.5)
        assert is_valid

        is_valid, error = field.validate("10")
        assert not is_valid

    def test_boolean_field_validation(self):
        """Test boolean field validation."""
        field = FieldSchema(
            name="enabled",
            field_type=FieldType.BOOLEAN,
            required=True,
        )

        is_valid, error = field.validate(True)
        assert is_valid

        is_valid, error = field.validate(False)
        assert is_valid

        is_valid, error = field.validate(1)
        assert not is_valid

    def test_list_field_validation(self):
        """Test list field validation."""
        field = FieldSchema(
            name="values",
            field_type=FieldType.LIST,
            required=True,
            min_length=1,
            max_length=5,
        )

        is_valid, error = field.validate(["a", "b"])
        assert is_valid

        is_valid, error = field.validate([])
        assert not is_valid
        assert "length must be >=" in error

        is_valid, error = field.validate([1, 2, 3, 4, 5, 6])
        assert not is_valid
        assert "length must be <=" in error

    def test_list_item_type_validation(self):
        """Test list with item type validation."""
        field = FieldSchema(
            name="values",
            field_type=FieldType.LIST,
            required=True,
            item_type=FieldType.STRING,
        )

        is_valid, error = field.validate(["a", "b", "c"])
        assert is_valid

        is_valid, error = field.validate(["a", 1, "c"])
        assert not is_valid
        assert "item at index" in error

    def test_dict_field_validation(self):
        """Test dict field validation."""
        field = FieldSchema(
            name="metadata",
            field_type=FieldType.DICT,
            required=True,
        )

        is_valid, error = field.validate({"key": "value"})
        assert is_valid

        is_valid, error = field.validate([("key", "value")])
        assert not is_valid

    def test_choices_validation(self):
        """Test field with choices constraint."""
        field = FieldSchema(
            name="severity",
            field_type=FieldType.STRING,
            required=False,
            choices=("error", "warning", "info"),
        )

        is_valid, error = field.validate("error")
        assert is_valid

        is_valid, error = field.validate("invalid")
        assert not is_valid
        assert "must be one of" in error

    def test_pattern_validation(self):
        """Test string pattern validation."""
        field = FieldSchema(
            name="email",
            field_type=FieldType.STRING,
            required=True,
            pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        )

        is_valid, error = field.validate("test@example.com")
        assert is_valid

        is_valid, error = field.validate("invalid-email")
        assert not is_valid
        assert "must match pattern" in error

    def test_regex_field_validation(self):
        """Test regex field validation."""
        field = FieldSchema(
            name="pattern",
            field_type=FieldType.REGEX,
            required=True,
        )

        is_valid, error = field.validate(r"^\d+$")
        assert is_valid

        is_valid, error = field.validate(r"[invalid(")
        assert not is_valid
        assert "invalid regex" in error

    def test_custom_validator(self):
        """Test custom validator function."""
        def is_even(value):
            return value % 2 == 0

        field = FieldSchema(
            name="even_number",
            field_type=FieldType.INTEGER,
            required=True,
            validator=is_even,
        )

        is_valid, error = field.validate(4)
        assert is_valid

        is_valid, error = field.validate(3)
        assert not is_valid
        assert "custom validation failed" in error

    def test_aliases(self):
        """Test field aliases."""
        field = FieldSchema(
            name="min",
            field_type=FieldType.NUMBER,
            required=False,
            aliases=("min_value", "minimum"),
        )

        rule = {"min_value": 0}
        actual_name = field.get_field_name(rule)
        assert actual_name == "min_value"

        rule = {"min": 0}
        actual_name = field.get_field_name(rule)
        assert actual_name == "min"

        rule = {"max": 100}
        actual_name = field.get_field_name(rule)
        assert actual_name is None


# =============================================================================
# RuleSchema Tests
# =============================================================================


class TestRuleSchema:
    """Tests for RuleSchema validation."""

    def test_basic_schema_validation(self):
        """Test basic schema validation."""
        schema = RuleSchema(
            rule_type="not_null",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
            ),
        )

        result = schema.validate({"type": "not_null", "column": "id"})
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_required_field(self):
        """Test validation with missing required field."""
        schema = RuleSchema(
            rule_type="not_null",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
            ),
        )

        result = schema.validate({"type": "not_null"})
        assert not result.is_valid
        assert len(result.errors) == 1
        assert isinstance(result.errors[0], MissingFieldError)

    def test_invalid_field_value(self):
        """Test validation with invalid field value."""
        schema = RuleSchema(
            rule_type="in_range",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
                FieldSchema(name="min", field_type=FieldType.NUMBER, required=False),
            ),
        )

        result = schema.validate({"type": "in_range", "column": "age", "min": "zero"})
        assert not result.is_valid
        assert any("min" in str(e) for e in result.errors)

    def test_optional_field(self):
        """Test validation with optional field."""
        schema = RuleSchema(
            rule_type="not_null",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
                FieldSchema(name="severity", field_type=FieldType.STRING, required=False),
            ),
        )

        # Without optional field
        result = schema.validate({"type": "not_null", "column": "id"})
        assert result.is_valid

        # With optional field
        result = schema.validate({"type": "not_null", "column": "id", "severity": "error"})
        assert result.is_valid

    def test_strict_mode(self):
        """Test strict mode with unknown fields."""
        schema = RuleSchema(
            rule_type="not_null",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
            ),
        )

        result = schema.validate(
            {"type": "not_null", "column": "id", "unknown_field": "value"},
            strict=True,
        )
        assert result.is_valid  # Unknown fields don't fail, just warn
        assert len(result.warnings) > 0
        assert any("unknown_field" in w.lower() for w in result.warnings)

    def test_deprecated_rule(self):
        """Test deprecated rule warning."""
        schema = RuleSchema(
            rule_type="old_rule",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
            ),
            deprecated=True,
            deprecated_message="Use 'new_rule' instead.",
        )

        result = schema.validate({"type": "old_rule", "column": "id"})
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("deprecated" in w.lower() for w in result.warnings)

    def test_engine_support(self):
        """Test engine support checking."""
        schema = RuleSchema(
            rule_type="custom_rule",
            fields=(),
            engines=("pandera", "great_expectations"),
        )

        assert schema.supports_engine("pandera")
        assert schema.supports_engine("PANDERA")  # Case insensitive
        assert schema.supports_engine("great_expectations")
        assert not schema.supports_engine("truthound")

    def test_get_required_fields(self):
        """Test getting required fields."""
        schema = RuleSchema(
            rule_type="test",
            fields=(
                FieldSchema(name="required1", field_type=FieldType.STRING, required=True),
                FieldSchema(name="optional1", field_type=FieldType.STRING, required=False),
                FieldSchema(name="required2", field_type=FieldType.STRING, required=True),
            ),
        )

        required = schema.get_required_fields()
        assert len(required) == 2
        assert all(f.required for f in required)

    def test_get_field_by_name_or_alias(self):
        """Test getting field by name or alias."""
        schema = RuleSchema(
            rule_type="test",
            fields=(
                FieldSchema(
                    name="min",
                    field_type=FieldType.NUMBER,
                    aliases=("min_value", "minimum"),
                ),
            ),
        )

        assert schema.get_field("min") is not None
        assert schema.get_field("min_value") is not None
        assert schema.get_field("minimum") is not None
        assert schema.get_field("max") is None


# =============================================================================
# RuleValidationResult Tests
# =============================================================================


class TestRuleValidationResult:
    """Tests for RuleValidationResult."""

    def test_valid_result(self):
        """Test valid result."""
        result = RuleValidationResult(
            is_valid=True,
            rule_type="not_null",
        )

        assert result.is_valid
        assert result.rule_type == "not_null"
        assert len(result.errors) == 0
        result.raise_if_invalid()  # Should not raise

    def test_invalid_result_raises(self):
        """Test invalid result raises on raise_if_invalid."""
        error = MissingFieldError("column", rule_type="not_null")
        result = RuleValidationResult(
            is_valid=False,
            rule_type="not_null",
            errors=(error,),
        )

        with pytest.raises(MissingFieldError):
            result.raise_if_invalid()

    def test_multiple_errors_raises(self):
        """Test multiple errors raises MultipleRuleValidationErrors."""
        errors = (
            MissingFieldError("column", rule_type="not_null"),
            MissingFieldError("pattern", rule_type="regex"),
        )
        result = RuleValidationResult(
            is_valid=False,
            rule_type="not_null",
            errors=errors,
        )

        with pytest.raises(MultipleRuleValidationErrors) as exc_info:
            result.raise_if_invalid()

        assert len(exc_info.value.errors) == 2


# =============================================================================
# BatchValidationResult Tests
# =============================================================================


class TestBatchValidationResult:
    """Tests for BatchValidationResult."""

    def test_from_results(self):
        """Test creating batch result from individual results."""
        results = [
            RuleValidationResult(is_valid=True, rule_type="not_null"),
            RuleValidationResult(is_valid=True, rule_type="unique"),
            RuleValidationResult(is_valid=False, rule_type="invalid", errors=(
                MissingFieldError("column"),
            )),
        ]

        batch = BatchValidationResult.from_results(results)
        assert not batch.is_valid
        assert batch.total_rules == 3
        assert batch.valid_count == 2
        assert batch.invalid_count == 1

    def test_get_errors_with_indices(self):
        """Test getting errors with rule indices."""
        results = [
            RuleValidationResult(is_valid=True, rule_type="not_null"),
            RuleValidationResult(is_valid=False, rule_type="invalid", errors=(
                MissingFieldError("column"),
            )),
        ]

        batch = BatchValidationResult.from_results(results)
        errors = batch.get_errors()

        assert len(errors) == 1
        assert errors[0].rule_index == 1

    def test_get_warnings(self):
        """Test getting warnings with rule indices."""
        results = [
            RuleValidationResult(is_valid=True, rule_type="not_null", warnings=("warning1",)),
            RuleValidationResult(is_valid=True, rule_type="unique", warnings=("warning2",)),
        ]

        batch = BatchValidationResult.from_results(results)
        warnings = batch.get_warnings()

        assert len(warnings) == 2
        assert warnings[0] == (0, "warning1")
        assert warnings[1] == (1, "warning2")


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for rule validation exceptions."""

    def test_rule_validation_error(self):
        """Test RuleValidationError."""
        error = RuleValidationError(
            "Invalid rule",
            rule_index=0,
            rule_type="not_null",
            field="column",
            expected="string",
            actual=123,
        )

        assert "Invalid rule" in str(error)
        assert "rule_index=0" in str(error)
        assert "rule_type='not_null'" in str(error)

    def test_unknown_rule_type_error(self):
        """Test UnknownRuleTypeError."""
        error = UnknownRuleTypeError(
            "invalid_type",
            available_types=["not_null", "unique"],
            rule_index=0,
        )

        assert "invalid_type" in str(error)
        assert "not_null" in str(error)
        assert "unique" in str(error)

    def test_missing_field_error(self):
        """Test MissingFieldError."""
        error = MissingFieldError("column", rule_type="not_null")

        assert "column" in str(error)
        assert error.field == "column"
        assert error.rule_type == "not_null"

    def test_invalid_field_type_error(self):
        """Test InvalidFieldTypeError."""
        error = InvalidFieldTypeError(
            "column",
            "string",
            "int",
            123,
            rule_type="not_null",
        )

        assert "column" in str(error)
        assert "string" in str(error)
        assert "int" in str(error)

    def test_multiple_rule_validation_errors(self):
        """Test MultipleRuleValidationErrors."""
        errors = [
            MissingFieldError("column"),
            MissingFieldError("pattern"),
        ]

        multi_error = MultipleRuleValidationErrors(errors)

        assert len(multi_error) == 2
        assert list(multi_error) == errors
        assert "2 errors" in str(multi_error)


# =============================================================================
# RuleRegistry Tests
# =============================================================================


class TestRuleRegistry:
    """Tests for RuleRegistry."""

    def test_common_schemas_registered(self):
        """Test that common schemas are registered by default."""
        registry = RuleRegistry()

        assert registry.get("not_null") is not None
        assert registry.get("unique") is not None
        assert registry.get("in_set") is not None
        assert registry.get("in_range") is not None
        assert registry.get("regex") is not None
        assert registry.get("dtype") is not None

    def test_register_custom_schema(self):
        """Test registering a custom schema."""
        registry = RuleRegistry()

        custom_schema = RuleSchema(
            rule_type="custom_rule",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
            ),
        )

        registry.register(custom_schema)
        assert registry.get("custom_rule") is not None

    def test_register_duplicate_raises(self):
        """Test registering duplicate raises error."""
        registry = RuleRegistry()

        with pytest.raises(ValueError, match="already registered"):
            registry.register(COMMON_RULE_SCHEMAS["not_null"])

    def test_unregister_schema(self):
        """Test unregistering a schema."""
        registry = RuleRegistry()

        registry.unregister("not_null")
        assert registry.get("not_null") is None

    def test_alias_lookup(self):
        """Test looking up schema by alias."""
        registry = RuleRegistry()

        schema = RuleSchema(
            rule_type="custom_rule",
            fields=(),
            aliases=("custom", "my_rule"),
        )
        registry.register(schema)

        assert registry.get("custom_rule") is schema
        assert registry.get("custom") is schema
        assert registry.get("my_rule") is schema

    def test_list_rule_types(self):
        """Test listing rule types."""
        registry = RuleRegistry()

        types = registry.list_rule_types()
        assert "not_null" in types
        assert "unique" in types

    def test_list_by_category(self):
        """Test listing schemas by category."""
        registry = RuleRegistry()

        completeness = registry.list_by_category(RuleCategory.COMPLETENESS)
        assert any(s.rule_type == "not_null" for s in completeness)

    def test_list_by_engine(self):
        """Test listing schemas by engine."""
        registry = RuleRegistry()

        pandera_schemas = registry.list_by_engine("pandera")
        assert len(pandera_schemas) > 0
        assert all(s.supports_engine("pandera") for s in pandera_schemas)

    def test_clear(self):
        """Test clearing the registry."""
        registry = RuleRegistry()
        registry.clear()

        assert len(registry.list_rule_types()) == 0


# =============================================================================
# StandardRuleValidator Tests
# =============================================================================


class TestStandardRuleValidator:
    """Tests for StandardRuleValidator."""

    def test_validate_valid_rule(self):
        """Test validating a valid rule."""
        validator = StandardRuleValidator()

        result = validator.validate({"type": "not_null", "column": "id"})
        assert result.is_valid

    def test_validate_missing_type(self):
        """Test validating rule without type field."""
        validator = StandardRuleValidator()

        result = validator.validate({"column": "id"})
        assert not result.is_valid
        assert any(isinstance(e, MissingFieldError) for e in result.errors)

    def test_validate_unknown_type(self):
        """Test validating rule with unknown type."""
        validator = StandardRuleValidator()

        result = validator.validate({"type": "invalid_type", "column": "id"})
        assert not result.is_valid
        assert any(isinstance(e, UnknownRuleTypeError) for e in result.errors)

    def test_validate_with_engine(self):
        """Test validating rule with engine constraint."""
        validator = StandardRuleValidator(engine="pandera")

        # greater_than is Pandera-specific
        result = validator.validate({"type": "greater_than", "column": "age", "value": 0})
        assert result.is_valid

        # Try with engine that doesn't support it
        validator = StandardRuleValidator(engine="great_expectations")
        result = validator.validate({"type": "greater_than", "column": "age", "value": 0})
        assert not result.is_valid

    def test_allow_unknown_types(self):
        """Test allowing unknown rule types."""
        validator = StandardRuleValidator(allow_unknown_types=True)

        result = validator.validate({"type": "custom_unknown_type", "column": "id"})
        assert result.is_valid
        assert any("unknown rule type" in w.lower() for w in result.warnings)

    def test_validate_batch(self, sample_rules):
        """Test batch validation."""
        validator = StandardRuleValidator()

        result = validator.validate_batch(sample_rules)
        assert result.is_valid
        assert result.total_rules == 5
        assert result.valid_count == 5

    def test_validate_batch_with_invalid(self):
        """Test batch validation with invalid rules."""
        validator = StandardRuleValidator()

        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "invalid_type"},  # Invalid
            {"type": "unique", "column": "email"},
        ]

        result = validator.validate_batch(rules)
        assert not result.is_valid
        assert result.invalid_count == 1
        assert result.valid_count == 2

    def test_validate_batch_fail_fast(self):
        """Test batch validation with fail_fast."""
        validator = StandardRuleValidator()

        rules = [
            {"type": "invalid1"},
            {"type": "invalid2"},
            {"type": "invalid3"},
        ]

        result = validator.validate_batch(rules, fail_fast=True)
        assert len(result.results) == 1  # Stopped at first error

    def test_supports_rule_type(self):
        """Test checking rule type support."""
        validator = StandardRuleValidator()

        assert validator.supports_rule_type("not_null")
        assert validator.supports_rule_type("unique")
        assert not validator.supports_rule_type("invalid_type")


# =============================================================================
# Engine-Specific Validator Tests
# =============================================================================


class TestTruthoundRuleValidator:
    """Tests for TruthoundRuleValidator."""

    def test_adds_schema_warning(self):
        """Test that Truthound validator adds schema-based warning."""
        validator = TruthoundRuleValidator()

        result = validator.validate({"type": "not_null", "column": "id"})
        assert result.is_valid
        assert any("schema-based" in w.lower() for w in result.warnings)

    def test_allows_unknown_types(self):
        """Test that Truthound validator allows unknown types."""
        validator = TruthoundRuleValidator()

        result = validator.validate({"type": "custom_type", "column": "id"})
        assert result.is_valid


class TestGreatExpectationsRuleValidator:
    """Tests for GreatExpectationsRuleValidator."""

    def test_validates_common_rules(self):
        """Test validating common rules."""
        validator = GreatExpectationsRuleValidator()

        result = validator.validate({"type": "not_null", "column": "id"})
        assert result.is_valid

    def test_allows_native_expectations(self):
        """Test allowing native GE expectations."""
        validator = GreatExpectationsRuleValidator()

        result = validator.validate({
            "type": "expect_column_to_exist",
            "column": "id",
        })
        assert result.is_valid
        assert any("native ge" in w.lower() for w in result.warnings)

    def test_validates_ge_specific_rules(self):
        """Test validating GE-specific rules."""
        validator = GreatExpectationsRuleValidator()

        result = validator.validate({
            "type": "expect_column_values_to_not_be_null",
            "column": "id",
        })
        assert result.is_valid


class TestPanderaRuleValidator:
    """Tests for PanderaRuleValidator."""

    def test_validates_common_rules(self):
        """Test validating common rules."""
        validator = PanderaRuleValidator()

        result = validator.validate({"type": "not_null", "column": "id"})
        assert result.is_valid

    def test_validates_pandera_specific_rules(self):
        """Test validating Pandera-specific rules."""
        validator = PanderaRuleValidator()

        result = validator.validate({
            "type": "greater_than",
            "column": "age",
            "value": 0,
        })
        assert result.is_valid

        result = validator.validate({
            "type": "less_than",
            "column": "percentage",
            "value": 100,
        })
        assert result.is_valid


# =============================================================================
# RuleNormalizer Tests
# =============================================================================


class TestRuleNormalizer:
    """Tests for RuleNormalizer."""

    def test_normalize_aliases(self):
        """Test normalizing field aliases."""
        normalizer = RuleNormalizer()

        rule = {"type": "in_range", "column": "age", "min_value": 0, "max_value": 100}
        normalized = normalizer.normalize(rule)

        # Aliases should be converted to canonical names
        assert "min" in normalized
        assert "max" in normalized
        assert normalized["min"] == 0
        assert normalized["max"] == 100

    def test_normalize_unknown_type(self):
        """Test normalizing rule with unknown type."""
        normalizer = RuleNormalizer()

        rule = {"type": "unknown", "custom_field": "value"}
        normalized = normalizer.normalize(rule)

        # Should preserve original structure
        assert normalized == rule

    def test_normalize_preserves_extra_fields(self):
        """Test that normalization preserves extra fields."""
        normalizer = RuleNormalizer()

        rule = {"type": "not_null", "column": "id", "custom": "value"}
        normalized = normalizer.normalize(rule)

        assert "custom" in normalized
        assert normalized["custom"] == "value"

    def test_normalize_batch(self):
        """Test batch normalization."""
        normalizer = RuleNormalizer()

        rules = [
            {"type": "in_range", "column": "a", "min_value": 0},
            {"type": "in_set", "column": "b", "value_set": ["x", "y"]},
        ]
        normalized = normalizer.normalize_batch(rules)

        assert len(normalized) == 2
        assert "min" in normalized[0]
        assert "values" in normalized[1]


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_rule(self):
        """Test validate_rule function."""
        result = validate_rule({"type": "not_null", "column": "id"})
        assert result.is_valid

    def test_validate_rules(self, sample_rules):
        """Test validate_rules function."""
        result = validate_rules(sample_rules)
        assert result.is_valid

    def test_validate_rules_raise_on_error(self):
        """Test validate_rules with raise_on_error."""
        rules = [{"type": "invalid_type"}]

        with pytest.raises(MultipleRuleValidationErrors):
            validate_rules(rules, raise_on_error=True)

    def test_normalize_rules_function(self):
        """Test normalize_rules function."""
        rules = [{"type": "in_range", "column": "age", "min_value": 0}]

        normalized = normalize_rules(rules)
        assert "min" in normalized[0]

    def test_get_validator_for_engine(self):
        """Test get_validator_for_engine function."""
        truthound = get_validator_for_engine("truthound")
        assert isinstance(truthound, TruthoundRuleValidator)

        ge = get_validator_for_engine("great_expectations")
        assert isinstance(ge, GreatExpectationsRuleValidator)

        pandera = get_validator_for_engine("pandera")
        assert isinstance(pandera, PanderaRuleValidator)

        generic = get_validator_for_engine("unknown")
        assert isinstance(generic, StandardRuleValidator)

    def test_register_rule_schema(self):
        """Test register_rule_schema function."""
        schema = RuleSchema(
            rule_type="custom_test_rule",
            fields=(
                FieldSchema(name="column", field_type=FieldType.STRING, required=True),
            ),
        )

        register_rule_schema(schema)
        result = validate_rule({"type": "custom_test_rule", "column": "id"})
        assert result.is_valid

    def test_get_supported_rule_types(self):
        """Test get_supported_rule_types function."""
        all_types = get_supported_rule_types()
        assert "not_null" in all_types
        assert "unique" in all_types

        pandera_types = get_supported_rule_types(engine="pandera")
        assert "greater_than" in pandera_types


# =============================================================================
# Engine Integration Tests
# =============================================================================


class TestValidatingEngineWrapper:
    """Tests for ValidatingEngineWrapper."""

    def test_check_with_valid_rules(self, mock_engine):
        """Test check with valid rules."""
        wrapper = ValidatingEngineWrapper(mock_engine)

        result = wrapper.check(
            None,
            rules=[{"type": "not_null", "column": "id"}],
        )

        assert result.passed_count == 1

    def test_check_with_invalid_rules(self, mock_engine):
        """Test check with invalid rules raises."""
        wrapper = ValidatingEngineWrapper(mock_engine)

        with pytest.raises(RuleValidationError):
            wrapper.check(
                None,
                rules=[{"type": "invalid_type"}],
            )

    def test_check_normalizes_rules(self, mock_engine):
        """Test that rules are normalized."""
        wrapper = ValidatingEngineWrapper(mock_engine)

        # This would fail if not normalized (min_value is an alias)
        result = wrapper.check(
            None,
            rules=[{"type": "in_range", "column": "age", "min_value": 0}],
        )
        assert result.passed_count == 1

    def test_check_without_validation(self, mock_engine):
        """Test check without validation."""
        wrapper = ValidatingEngineWrapper(mock_engine, validate=False)

        # This should not raise even with invalid rules
        result = wrapper.check(
            None,
            rules=[{"type": "invalid_type"}],
        )
        assert result is not None

    def test_profile_delegation(self, mock_engine):
        """Test that profile is delegated."""
        wrapper = ValidatingEngineWrapper(mock_engine)

        result = wrapper.profile(None)
        assert result is not None

    def test_learn_delegation(self, mock_engine):
        """Test that learn is delegated."""
        wrapper = ValidatingEngineWrapper(mock_engine)

        result = wrapper.learn(None)
        assert result is not None

    def test_property_delegation(self, mock_engine):
        """Test that properties are delegated."""
        wrapper = ValidatingEngineWrapper(mock_engine)

        assert wrapper.engine_name == "truthound"  # Mock uses truthound as engine name
        assert wrapper.engine_version == "1.0.0"


class TestWrapEngineWithValidation:
    """Tests for wrap_engine_with_validation function."""

    def test_wraps_engine(self, mock_engine):
        """Test wrapping engine."""
        wrapped = wrap_engine_with_validation(mock_engine)

        assert isinstance(wrapped, ValidatingEngineWrapper)
        assert wrapped.engine is mock_engine

    def test_wraps_with_options(self, mock_engine):
        """Test wrapping with options."""
        wrapped = wrap_engine_with_validation(
            mock_engine,
            validate=True,
            normalize=False,
            strict=True,
        )

        assert wrapped._validate is True
        assert wrapped._normalize is False
        assert wrapped._strict is True


class TestValidateRulesDecorator:
    """Tests for validate_rules_decorator."""

    def test_decorator_validates(self):
        """Test that decorator validates rules."""
        from common.base import CheckResult, CheckStatus

        class TestEngine:
            @validate_rules_decorator()
            def check(self, data, rules=None, **kwargs):
                return CheckResult(
                    status=CheckStatus.PASSED,
                    passed_count=len(rules) if rules else 0,
                )

        engine = TestEngine()

        # Valid rules should work
        result = engine.check(None, rules=[{"type": "not_null", "column": "id"}])
        assert result.passed_count == 1

        # Invalid rules should raise
        with pytest.raises(RuleValidationError):
            engine.check(None, rules=[{"type": "invalid_type"}])

    def test_decorator_normalizes(self):
        """Test that decorator normalizes rules."""
        from common.base import CheckResult, CheckStatus

        class TestEngine:
            @validate_rules_decorator(normalize=True)
            def check(self, data, rules=None, **kwargs):
                # Check that rules are normalized
                if rules and len(rules) > 0:
                    assert "min" in rules[0]  # Alias should be normalized
                return CheckResult(status=CheckStatus.PASSED)

        engine = TestEngine()
        engine.check(None, rules=[{"type": "in_range", "column": "age", "min_value": 0}])


# =============================================================================
# Common Rule Schema Tests
# =============================================================================


class TestCommonRuleSchemas:
    """Tests for common rule schemas."""

    def test_not_null_schema(self):
        """Test not_null schema."""
        result = validate_rule({"type": "not_null", "column": "id"})
        assert result.is_valid

        result = validate_rule({"type": "not_null"})  # Missing column
        assert not result.is_valid

    def test_unique_schema(self):
        """Test unique schema."""
        result = validate_rule({"type": "unique", "column": "email"})
        assert result.is_valid

    def test_in_set_schema(self):
        """Test in_set schema."""
        result = validate_rule({
            "type": "in_set",
            "column": "status",
            "values": ["active", "inactive"],
        })
        assert result.is_valid

        # Empty values should fail
        result = validate_rule({
            "type": "in_set",
            "column": "status",
            "values": [],
        })
        assert not result.is_valid

    def test_in_range_schema(self):
        """Test in_range schema."""
        result = validate_rule({
            "type": "in_range",
            "column": "age",
            "min": 0,
            "max": 150,
        })
        assert result.is_valid

        # Aliases should work
        result = validate_rule({
            "type": "in_range",
            "column": "age",
            "min_value": 0,
            "max_value": 150,
        })
        assert result.is_valid

    def test_regex_schema(self):
        """Test regex schema."""
        result = validate_rule({
            "type": "regex",
            "column": "email",
            "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        })
        assert result.is_valid

        # Invalid regex should fail
        result = validate_rule({
            "type": "regex",
            "column": "email",
            "pattern": "[invalid(",
        })
        assert not result.is_valid

    def test_dtype_schema(self):
        """Test dtype schema."""
        result = validate_rule({
            "type": "dtype",
            "column": "price",
            "dtype": "float64",
        })
        assert result.is_valid

    def test_min_length_schema(self):
        """Test min_length schema."""
        result = validate_rule({
            "type": "min_length",
            "column": "name",
            "min_length": 1,
        })
        assert result.is_valid

        # Negative length should fail
        result = validate_rule({
            "type": "min_length",
            "column": "name",
            "min_length": -1,
        })
        assert not result.is_valid

    def test_max_length_schema(self):
        """Test max_length schema."""
        result = validate_rule({
            "type": "max_length",
            "column": "description",
            "max_length": 1000,
        })
        assert result.is_valid

    def test_greater_than_schema(self):
        """Test greater_than schema (Pandera-specific)."""
        result = validate_rule({
            "type": "greater_than",
            "column": "price",
            "value": 0,
        })
        assert result.is_valid

    def test_less_than_schema(self):
        """Test less_than schema (Pandera-specific)."""
        result = validate_rule({
            "type": "less_than",
            "column": "percentage",
            "value": 100,
        })
        assert result.is_valid

    def test_column_exists_schema(self):
        """Test column_exists schema."""
        result = validate_rule({
            "type": "column_exists",
            "column": "id",
        })
        assert result.is_valid


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_rule_registry(self):
        """Test getting global registry."""
        registry1 = get_rule_registry()
        registry2 = get_rule_registry()

        assert registry1 is registry2

    def test_reset_rule_registry(self):
        """Test resetting global registry."""
        registry1 = get_rule_registry()

        # Register custom schema
        schema = RuleSchema(
            rule_type="temp_schema",
            fields=(),
        )
        registry1.register(schema)

        reset_rule_registry()
        registry2 = get_rule_registry()

        assert registry1 is not registry2
        assert registry2.get("temp_schema") is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_rules_list(self):
        """Test validating empty rules list."""
        result = validate_rules([])
        assert result.is_valid
        assert result.total_rules == 0

    def test_none_rules(self, mock_engine):
        """Test wrapper with None rules."""
        wrapper = ValidatingEngineWrapper(mock_engine)
        result = wrapper.check(None, rules=None)
        assert result is not None

    def test_rule_with_extra_fields(self):
        """Test rule with extra fields."""
        result = validate_rule({
            "type": "not_null",
            "column": "id",
            "extra_field": "ignored",
        })
        assert result.is_valid

    def test_rule_with_extra_fields_strict(self):
        """Test rule with extra fields in strict mode."""
        result = validate_rule(
            {
                "type": "not_null",
                "column": "id",
                "extra_field": "warned",
            },
            strict=True,
        )
        assert result.is_valid
        assert any("extra_field" in w.lower() for w in result.warnings)

    def test_deeply_nested_rule(self):
        """Test rule with nested structures."""
        result = validate_rule({
            "type": "in_set",
            "column": "status",
            "values": ["a", "b", "c"],
        })
        assert result.is_valid

    def test_unicode_in_rules(self):
        """Test rules with unicode characters."""
        result = validate_rule({
            "type": "not_null",
            "column": "名前",  # Japanese for "name"
        })
        assert result.is_valid
