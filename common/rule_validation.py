"""Rule Validation System for Truthound Orchestration.

This module provides a comprehensive rule validation system that validates rule
dictionaries before they are passed to data quality engines. It ensures that
rules have the correct structure, required fields, and valid parameter values.

Key Components:
    - RuleSchema: Defines the expected structure of a rule type
    - RuleValidator: Protocol for rule validation implementations
    - RuleValidationResult: Result of validating a rule
    - RuleRegistry: Central registry for rule schemas
    - Engine-specific validators: Validate rules for specific engines

Design Principles:
    1. Protocol-based: Use structural typing for flexible validator implementations
    2. Extensible: Easy to add new rule types and custom validators
    3. Engine-aware: Validate rules against engine-specific capabilities
    4. Fail-fast: Detect invalid rules before execution
    5. Informative: Provide clear, actionable error messages

Example:
    >>> from common.rule_validation import validate_rules, RuleValidationError
    >>> rules = [
    ...     {"type": "not_null", "column": "id"},
    ...     {"type": "in_range", "column": "age", "min": 0, "max": 150},
    ... ]
    >>> try:
    ...     validate_rules(rules, engine="great_expectations")
    ... except RuleValidationError as e:
    ...     print(f"Invalid rules: {e}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Self,
    runtime_checkable,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# =============================================================================
# Enums
# =============================================================================


class FieldType(Enum):
    """Type of a rule field.

    Used to define and validate the expected type of rule parameters.
    """

    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    NUMBER = auto()  # int or float
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    ANY = auto()
    REGEX = auto()  # string that is a valid regex pattern


class RuleCategory(Enum):
    """Category of a validation rule.

    Used for grouping and filtering rules by their purpose.
    """

    COMPLETENESS = auto()  # not_null, required
    UNIQUENESS = auto()  # unique, primary_key
    VALIDITY = auto()  # in_set, in_range, regex, dtype
    CONSISTENCY = auto()  # foreign_key, cross_column
    ACCURACY = auto()  # statistical checks
    TIMELINESS = auto()  # freshness, recency
    DRIFT = auto()  # data drift detection
    ANOMALY = auto()  # anomaly detection


# =============================================================================
# Exceptions
# =============================================================================


class RuleValidationError(Exception):
    """Exception raised when rule validation fails.

    Provides detailed information about validation failures including
    the specific field, expected value, and actual value.

    Attributes:
        message: Human-readable error description.
        rule_index: Index of the failing rule in the rules list.
        rule_type: Type of the failing rule.
        field: Specific field that failed validation.
        expected: Description of what was expected.
        actual: The actual value that failed validation.
        details: Additional error context.
    """

    def __init__(
        self,
        message: str,
        *,
        rule_index: int | None = None,
        rule_type: str | None = None,
        field: str | None = None,
        expected: str | None = None,
        actual: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.rule_index = rule_index
        self.rule_type = rule_type
        self.field = field
        self.expected = expected
        self.actual = actual
        self.details = details or {}

    def __str__(self) -> str:
        parts = [self.message]
        if self.rule_index is not None:
            parts.append(f"rule_index={self.rule_index}")
        if self.rule_type:
            parts.append(f"rule_type='{self.rule_type}'")
        if self.field:
            parts.append(f"field='{self.field}'")
        if self.expected:
            parts.append(f"expected={self.expected}")
        if self.actual is not None:
            parts.append(f"actual={self.actual!r}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"rule_index={self.rule_index!r}, "
            f"rule_type={self.rule_type!r}, "
            f"field={self.field!r})"
        )


class UnknownRuleTypeError(RuleValidationError):
    """Exception raised when an unknown rule type is encountered.

    Attributes:
        rule_type: The unknown rule type.
        available_types: List of available rule types.
    """

    def __init__(
        self,
        rule_type: str,
        *,
        available_types: Sequence[str] | None = None,
        rule_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.available_types = list(available_types) if available_types else []
        message = f"Unknown rule type: '{rule_type}'"
        if self.available_types:
            message += f". Available types: {', '.join(sorted(self.available_types))}"
        super().__init__(
            message,
            rule_index=rule_index,
            rule_type=rule_type,
            details=details,
        )


class MissingFieldError(RuleValidationError):
    """Exception raised when a required field is missing from a rule."""

    def __init__(
        self,
        field: str,
        *,
        rule_type: str | None = None,
        rule_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Missing required field: '{field}'"
        super().__init__(
            message,
            rule_index=rule_index,
            rule_type=rule_type,
            field=field,
            expected="required field to be present",
            actual="field is missing",
            details=details,
        )


class InvalidFieldTypeError(RuleValidationError):
    """Exception raised when a field has an invalid type."""

    def __init__(
        self,
        field: str,
        expected_type: str,
        actual_type: str,
        actual_value: Any,
        *,
        rule_type: str | None = None,
        rule_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Invalid type for field '{field}': expected {expected_type}, got {actual_type}"
        super().__init__(
            message,
            rule_index=rule_index,
            rule_type=rule_type,
            field=field,
            expected=expected_type,
            actual=actual_value,
            details=details,
        )


class InvalidFieldValueError(RuleValidationError):
    """Exception raised when a field has an invalid value."""

    def __init__(
        self,
        field: str,
        message: str,
        actual_value: Any,
        *,
        expected: str | None = None,
        rule_type: str | None = None,
        rule_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            rule_index=rule_index,
            rule_type=rule_type,
            field=field,
            expected=expected,
            actual=actual_value,
            details=details,
        )


class MultipleRuleValidationErrors(RuleValidationError):
    """Exception containing multiple rule validation errors.

    Useful for batch validation where all errors should be reported together.
    """

    def __init__(
        self,
        errors: Sequence[RuleValidationError],
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.errors = list(errors)
        message = f"Multiple rule validation errors ({len(errors)} errors)"
        super().__init__(message, details=details)

    def __str__(self) -> str:
        lines = [f"Multiple rule validation errors ({len(self.errors)} errors):"]
        for i, error in enumerate(self.errors, 1):
            lines.append(f"  {i}. {error}")
        return "\n".join(lines)

    def __iter__(self):
        return iter(self.errors)

    def __len__(self) -> int:
        return len(self.errors)


# =============================================================================
# Field Schema
# =============================================================================


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """Schema definition for a rule field.

    Defines the expected type, constraints, and validation rules for a field.

    Attributes:
        name: Name of the field.
        field_type: Expected type of the field.
        required: Whether the field is required.
        default: Default value if not provided.
        description: Human-readable description.
        choices: Valid choices for the field (for enum-like fields).
        min_value: Minimum value (for numeric fields).
        max_value: Maximum value (for numeric fields).
        min_length: Minimum length (for string/list fields).
        max_length: Maximum length (for string/list fields).
        pattern: Regex pattern (for string fields).
        item_type: Type of items (for list fields).
        validator: Custom validation function.
        aliases: Alternative names for this field.

    Example:
        >>> field = FieldSchema(
        ...     name="column",
        ...     field_type=FieldType.STRING,
        ...     required=True,
        ...     description="Column name to validate",
        ... )
    """

    name: str
    field_type: FieldType
    required: bool = True
    default: Any = None
    description: str = ""
    choices: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    item_type: FieldType | None = None
    validator: Callable[[Any], bool] | None = None
    aliases: tuple[str, ...] = ()

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """Validate a value against this field schema.

        Args:
            value: The value to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        # Type validation
        is_valid, error = self._validate_type(value)
        if not is_valid:
            return False, error

        # Choices validation
        if self.choices is not None and value not in self.choices:
            return False, f"value must be one of {self.choices}, got {value!r}"

        # Range validation (for numeric types)
        if self.field_type in (FieldType.INTEGER, FieldType.FLOAT, FieldType.NUMBER):
            if self.min_value is not None and value < self.min_value:
                return False, f"value must be >= {self.min_value}, got {value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"value must be <= {self.max_value}, got {value}"

        # Length validation (for string/list types)
        if self.field_type in (FieldType.STRING, FieldType.LIST):
            if self.min_length is not None and len(value) < self.min_length:
                return False, f"length must be >= {self.min_length}, got {len(value)}"
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"length must be <= {self.max_length}, got {len(value)}"

        # Pattern validation (for string types)
        if self.field_type == FieldType.STRING and self.pattern is not None:
            if not re.match(self.pattern, value):
                return False, f"value must match pattern '{self.pattern}'"

        # Regex validation
        if self.field_type == FieldType.REGEX:
            try:
                re.compile(value)
            except re.error as e:
                return False, f"invalid regex pattern: {e}"

        # List item type validation
        if self.field_type == FieldType.LIST and self.item_type is not None:
            for i, item in enumerate(value):
                is_valid, _ = self._validate_item_type(item, self.item_type)
                if not is_valid:
                    return False, f"item at index {i} has invalid type"

        # Custom validator
        if self.validator is not None:
            try:
                if not self.validator(value):
                    return False, "custom validation failed"
            except Exception as e:
                return False, f"custom validation error: {e}"

        return True, None

    def _validate_type(self, value: Any) -> tuple[bool, str | None]:
        """Validate the type of a value."""
        type_checks = {
            FieldType.STRING: lambda v: isinstance(v, str),
            FieldType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            FieldType.FLOAT: lambda v: isinstance(v, float),
            FieldType.NUMBER: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            FieldType.BOOLEAN: lambda v: isinstance(v, bool),
            FieldType.LIST: lambda v: isinstance(v, (list, tuple)),
            FieldType.DICT: lambda v: isinstance(v, dict),
            FieldType.ANY: lambda v: True,
            FieldType.REGEX: lambda v: isinstance(v, str),
        }

        check = type_checks.get(self.field_type, lambda v: True)
        if not check(value):
            expected = self.field_type.name.lower()
            actual = type(value).__name__
            return False, f"expected {expected}, got {actual}"
        return True, None

    def _validate_item_type(self, value: Any, item_type: FieldType) -> tuple[bool, str | None]:
        """Validate the type of a list item."""
        temp_schema = FieldSchema(name="item", field_type=item_type)
        return temp_schema._validate_type(value)

    def get_field_name(self, rule: Mapping[str, Any]) -> str | None:
        """Get the actual field name used in a rule (considering aliases).

        Args:
            rule: The rule dictionary.

        Returns:
            The field name if found, None otherwise.
        """
        if self.name in rule:
            return self.name
        for alias in self.aliases:
            if alias in rule:
                return alias
        return None


# =============================================================================
# Rule Schema
# =============================================================================


@dataclass(frozen=True)
class RuleSchema:
    """Schema definition for a rule type.

    Defines the expected structure, required fields, and validation rules
    for a specific rule type.

    Attributes:
        rule_type: The type identifier for this rule.
        fields: Tuple of field schemas.
        category: Category of this rule.
        description: Human-readable description.
        engines: Tuple of engine names that support this rule.
        aliases: Alternative names for this rule type.
        examples: Example rule dictionaries.
        deprecated: Whether this rule type is deprecated.
        deprecated_message: Message explaining deprecation.

    Example:
        >>> schema = RuleSchema(
        ...     rule_type="not_null",
        ...     fields=(
        ...         FieldSchema(name="column", field_type=FieldType.STRING, required=True),
        ...     ),
        ...     category=RuleCategory.COMPLETENESS,
        ...     description="Check that column values are not null",
        ... )
    """

    rule_type: str
    fields: tuple[FieldSchema, ...] = ()
    category: RuleCategory = RuleCategory.VALIDITY
    description: str = ""
    engines: tuple[str, ...] = ("truthound", "great_expectations", "pandera")
    aliases: tuple[str, ...] = ()
    examples: tuple[dict[str, Any], ...] = ()
    deprecated: bool = False
    deprecated_message: str = ""

    def get_required_fields(self) -> tuple[FieldSchema, ...]:
        """Get all required fields for this rule type."""
        return tuple(f for f in self.fields if f.required)

    def get_optional_fields(self) -> tuple[FieldSchema, ...]:
        """Get all optional fields for this rule type."""
        return tuple(f for f in self.fields if not f.required)

    def get_field(self, name: str) -> FieldSchema | None:
        """Get a field schema by name or alias.

        Args:
            name: Field name or alias.

        Returns:
            FieldSchema if found, None otherwise.
        """
        for f in self.fields:
            if f.name == name or name in f.aliases:
                return f
        return None

    def supports_engine(self, engine: str) -> bool:
        """Check if this rule type supports a specific engine.

        Args:
            engine: Engine name to check.

        Returns:
            True if the engine supports this rule type.
        """
        return engine.lower() in [e.lower() for e in self.engines]

    def validate(
        self,
        rule: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> RuleValidationResult:
        """Validate a rule dictionary against this schema.

        Args:
            rule: The rule dictionary to validate.
            strict: If True, unknown fields cause warnings.

        Returns:
            RuleValidationResult with validation outcome.
        """
        errors: list[RuleValidationError] = []
        warnings: list[str] = []

        # Check deprecated
        if self.deprecated:
            warnings.append(
                f"Rule type '{self.rule_type}' is deprecated. {self.deprecated_message}"
            )

        # Check required fields
        for field_schema in self.get_required_fields():
            field_name = field_schema.get_field_name(rule)
            if field_name is None:
                errors.append(
                    MissingFieldError(
                        field_schema.name,
                        rule_type=self.rule_type,
                    )
                )
                continue

            value = rule[field_name]
            is_valid, error_msg = field_schema.validate(value)
            if not is_valid:
                errors.append(
                    InvalidFieldValueError(
                        field_schema.name,
                        error_msg or "validation failed",
                        value,
                        rule_type=self.rule_type,
                    )
                )

        # Check optional fields if present
        for field_schema in self.get_optional_fields():
            field_name = field_schema.get_field_name(rule)
            if field_name is not None:
                value = rule[field_name]
                is_valid, error_msg = field_schema.validate(value)
                if not is_valid:
                    errors.append(
                        InvalidFieldValueError(
                            field_schema.name,
                            error_msg or "validation failed",
                            value,
                            rule_type=self.rule_type,
                        )
                    )

        # Check for unknown fields in strict mode
        if strict:
            known_fields = {"type"}
            for f in self.fields:
                known_fields.add(f.name)
                known_fields.update(f.aliases)

            for key in rule:
                if key not in known_fields:
                    warnings.append(f"Unknown field '{key}' in rule")

        return RuleValidationResult(
            is_valid=len(errors) == 0,
            rule_type=self.rule_type,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )


# =============================================================================
# Validation Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class RuleValidationResult:
    """Result of validating a single rule.

    Attributes:
        is_valid: Whether the rule is valid.
        rule_type: Type of the validated rule.
        errors: Tuple of validation errors.
        warnings: Tuple of validation warnings.
        normalized_rule: The rule with normalized field names (optional).
    """

    is_valid: bool
    rule_type: str
    errors: tuple[RuleValidationError, ...] = ()
    warnings: tuple[str, ...] = ()
    normalized_rule: dict[str, Any] | None = None

    def raise_if_invalid(self) -> None:
        """Raise an exception if the rule is invalid.

        Raises:
            RuleValidationError: If the rule is invalid.
            MultipleRuleValidationErrors: If there are multiple errors.
        """
        if self.is_valid:
            return

        if len(self.errors) == 1:
            raise self.errors[0]
        raise MultipleRuleValidationErrors(self.errors)


@dataclass(frozen=True, slots=True)
class BatchValidationResult:
    """Result of validating multiple rules.

    Attributes:
        is_valid: Whether all rules are valid.
        results: Tuple of individual validation results.
        total_rules: Total number of rules validated.
        valid_count: Number of valid rules.
        invalid_count: Number of invalid rules.
    """

    is_valid: bool
    results: tuple[RuleValidationResult, ...]
    total_rules: int
    valid_count: int
    invalid_count: int

    @classmethod
    def from_results(cls, results: Sequence[RuleValidationResult]) -> Self:
        """Create a BatchValidationResult from individual results.

        Args:
            results: Sequence of individual validation results.

        Returns:
            BatchValidationResult instance.
        """
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count
        return cls(
            is_valid=invalid_count == 0,
            results=tuple(results),
            total_rules=len(results),
            valid_count=valid_count,
            invalid_count=invalid_count,
        )

    def raise_if_invalid(self) -> None:
        """Raise an exception if any rule is invalid.

        Raises:
            MultipleRuleValidationErrors: If any rule is invalid.
        """
        if self.is_valid:
            return

        all_errors: list[RuleValidationError] = []
        for i, result in enumerate(self.results):
            for error in result.errors:
                # Add rule index to each error
                error_with_index = RuleValidationError(
                    error.message,
                    rule_index=i,
                    rule_type=error.rule_type,
                    field=error.field,
                    expected=error.expected,
                    actual=error.actual,
                    details=error.details,
                )
                all_errors.append(error_with_index)

        raise MultipleRuleValidationErrors(all_errors)

    def get_errors(self) -> list[RuleValidationError]:
        """Get all validation errors with rule indices.

        Returns:
            List of validation errors.
        """
        errors: list[RuleValidationError] = []
        for i, result in enumerate(self.results):
            for error in result.errors:
                error_with_index = RuleValidationError(
                    error.message,
                    rule_index=i,
                    rule_type=error.rule_type,
                    field=error.field,
                    expected=error.expected,
                    actual=error.actual,
                    details=error.details,
                )
                errors.append(error_with_index)
        return errors

    def get_warnings(self) -> list[tuple[int, str]]:
        """Get all validation warnings with rule indices.

        Returns:
            List of (rule_index, warning_message) tuples.
        """
        warnings: list[tuple[int, str]] = []
        for i, result in enumerate(self.results):
            for warning in result.warnings:
                warnings.append((i, warning))
        return warnings


# =============================================================================
# Rule Validator Protocol
# =============================================================================


@runtime_checkable
class RuleValidator(Protocol):
    """Protocol for rule validators.

    Defines the interface that all rule validators must implement.
    """

    def validate(
        self,
        rule: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> RuleValidationResult:
        """Validate a single rule.

        Args:
            rule: The rule dictionary to validate.
            strict: If True, unknown fields cause warnings.

        Returns:
            RuleValidationResult with validation outcome.
        """
        ...

    def validate_batch(
        self,
        rules: Sequence[Mapping[str, Any]],
        *,
        strict: bool = False,
        fail_fast: bool = False,
    ) -> BatchValidationResult:
        """Validate multiple rules.

        Args:
            rules: Sequence of rule dictionaries to validate.
            strict: If True, unknown fields cause warnings.
            fail_fast: If True, stop on first error.

        Returns:
            BatchValidationResult with validation outcomes.
        """
        ...

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if a rule type is supported.

        Args:
            rule_type: The rule type to check.

        Returns:
            True if the rule type is supported.
        """
        ...


# =============================================================================
# Common Rule Schemas
# =============================================================================


# Column field schema (reused across many rules)
COLUMN_FIELD = FieldSchema(
    name="column",
    field_type=FieldType.STRING,
    required=True,
    description="Column name to validate",
    min_length=1,
)

# Severity field schema (optional for most rules)
SEVERITY_FIELD = FieldSchema(
    name="severity",
    field_type=FieldType.STRING,
    required=False,
    default="error",
    description="Severity level for failures",
    choices=("critical", "error", "warning", "info"),
)


# Common rule schemas
COMMON_RULE_SCHEMAS: dict[str, RuleSchema] = {
    "not_null": RuleSchema(
        rule_type="not_null",
        fields=(
            COLUMN_FIELD,
            SEVERITY_FIELD,
        ),
        category=RuleCategory.COMPLETENESS,
        description="Check that column values are not null",
        examples=({"type": "not_null", "column": "id"},),
    ),
    "unique": RuleSchema(
        rule_type="unique",
        fields=(
            COLUMN_FIELD,
            SEVERITY_FIELD,
        ),
        category=RuleCategory.UNIQUENESS,
        description="Check that column values are unique",
        examples=({"type": "unique", "column": "email"},),
    ),
    "in_set": RuleSchema(
        rule_type="in_set",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="values",
                field_type=FieldType.LIST,
                required=True,
                description="Set of valid values",
                min_length=1,
                aliases=("value_set", "allowed_values"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that column values are in a set of allowed values",
        examples=({"type": "in_set", "column": "status", "values": ["active", "inactive"]},),
    ),
    "in_range": RuleSchema(
        rule_type="in_range",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="min",
                field_type=FieldType.NUMBER,
                required=False,
                description="Minimum value (inclusive)",
                aliases=("min_value",),
            ),
            FieldSchema(
                name="max",
                field_type=FieldType.NUMBER,
                required=False,
                description="Maximum value (inclusive)",
                aliases=("max_value",),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that column values are within a range",
        examples=({"type": "in_range", "column": "age", "min": 0, "max": 150},),
    ),
    "regex": RuleSchema(
        rule_type="regex",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="pattern",
                field_type=FieldType.REGEX,
                required=True,
                description="Regular expression pattern",
                aliases=("regex",),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that column values match a regex pattern",
        examples=(
            {"type": "regex", "column": "email", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
        ),
    ),
    "dtype": RuleSchema(
        rule_type="dtype",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="dtype",
                field_type=FieldType.STRING,
                required=True,
                description="Expected data type",
                aliases=("data_type", "type_"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that column has the expected data type",
        examples=({"type": "dtype", "column": "price", "dtype": "float64"},),
    ),
    "min_length": RuleSchema(
        rule_type="min_length",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="min_length",
                field_type=FieldType.INTEGER,
                required=True,
                description="Minimum string length",
                min_value=0,
                aliases=("min_len", "length"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that string column values have minimum length",
        examples=({"type": "min_length", "column": "name", "min_length": 1},),
    ),
    "max_length": RuleSchema(
        rule_type="max_length",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="max_length",
                field_type=FieldType.INTEGER,
                required=True,
                description="Maximum string length",
                min_value=0,
                aliases=("max_len",),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that string column values have maximum length",
        examples=({"type": "max_length", "column": "description", "max_length": 1000},),
    ),
    "greater_than": RuleSchema(
        rule_type="greater_than",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="value",
                field_type=FieldType.NUMBER,
                required=True,
                description="Value to compare against",
                aliases=("min_value", "threshold"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that column values are greater than a value",
        engines=("pandera",),
        examples=({"type": "greater_than", "column": "price", "value": 0},),
    ),
    "less_than": RuleSchema(
        rule_type="less_than",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="value",
                field_type=FieldType.NUMBER,
                required=True,
                description="Value to compare against",
                aliases=("max_value", "threshold"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that column values are less than a value",
        engines=("pandera",),
        examples=({"type": "less_than", "column": "percentage", "value": 100},),
    ),
    "column_exists": RuleSchema(
        rule_type="column_exists",
        fields=(
            COLUMN_FIELD,
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Check that a column exists in the data",
        examples=({"type": "column_exists", "column": "id"},),
    ),
    # =========================================================================
    # Drift Rules
    # =========================================================================
    "statistical_drift": RuleSchema(
        rule_type="statistical_drift",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="method",
                field_type=FieldType.STRING,
                required=False,
                default="ks",
                description="Statistical test method (e.g., ks, psi, chi2, auto)",
                aliases=("test_method", "drift_method"),
            ),
            FieldSchema(
                name="threshold",
                field_type=FieldType.FLOAT,
                required=False,
                default=0.05,
                description="Drift detection threshold (p-value or score)",
                min_value=0.0,
                max_value=1.0,
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.DRIFT,
        description="Detect statistical drift in a column between baseline and current data",
        engines=("truthound",),
        aliases=("drift", "stat_drift"),
        examples=(
            {"type": "statistical_drift", "column": "age", "method": "ks", "threshold": 0.05},
        ),
    ),
    "distribution_change": RuleSchema(
        rule_type="distribution_change",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="baseline_profile",
                field_type=FieldType.DICT,
                required=True,
                description="Baseline distribution profile for comparison",
                aliases=("baseline", "reference_profile"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.DRIFT,
        description="Detect distribution changes compared to a baseline profile",
        engines=("truthound",),
        aliases=("dist_change",),
        examples=(
            {
                "type": "distribution_change",
                "column": "income",
                "baseline_profile": {"mean": 50000, "std": 15000},
            },
        ),
    ),
    # =========================================================================
    # Anomaly Rules
    # =========================================================================
    "outlier": RuleSchema(
        rule_type="outlier",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="detector",
                field_type=FieldType.STRING,
                required=False,
                default="isolation_forest",
                description="Anomaly detector algorithm",
                aliases=("algorithm", "method"),
            ),
            FieldSchema(
                name="contamination",
                field_type=FieldType.FLOAT,
                required=False,
                default=0.05,
                description="Expected proportion of anomalies",
                min_value=0.0,
                max_value=0.5,
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.ANOMALY,
        description="Detect outliers in column values using anomaly detection",
        engines=("truthound",),
        aliases=("anomaly", "outlier_detection"),
        examples=(
            {"type": "outlier", "column": "transaction_amount", "detector": "isolation_forest"},
        ),
    ),
    "z_score_outlier": RuleSchema(
        rule_type="z_score_outlier",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="threshold",
                field_type=FieldType.FLOAT,
                required=False,
                default=3.0,
                description="Z-score threshold for outlier detection",
                min_value=0.0,
                aliases=("z_threshold", "sigma"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.ANOMALY,
        description="Detect outliers using z-score (standard deviations from mean)",
        engines=("truthound",),
        aliases=("zscore", "z_score"),
        examples=(
            {"type": "z_score_outlier", "column": "price", "threshold": 3.0},
        ),
    ),
    # =========================================================================
    # Extended Rules
    # =========================================================================
    "completeness_ratio": RuleSchema(
        rule_type="completeness_ratio",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="min_ratio",
                field_type=FieldType.FLOAT,
                required=True,
                description="Minimum non-null ratio (0.0 to 1.0)",
                min_value=0.0,
                max_value=1.0,
                aliases=("ratio", "min_completeness"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.COMPLETENESS,
        description="Check that column has at least a minimum ratio of non-null values",
        aliases=("completeness",),
        examples=(
            {"type": "completeness_ratio", "column": "email", "min_ratio": 0.95},
        ),
    ),
    "referential_integrity": RuleSchema(
        rule_type="referential_integrity",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="reference_table",
                field_type=FieldType.STRING,
                required=True,
                description="Reference table name",
                aliases=("ref_table", "foreign_table"),
            ),
            FieldSchema(
                name="reference_column",
                field_type=FieldType.STRING,
                required=True,
                description="Reference column name",
                aliases=("ref_column", "foreign_column"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.CONSISTENCY,
        description="Check referential integrity between tables",
        aliases=("foreign_key", "ref_integrity"),
        examples=(
            {
                "type": "referential_integrity",
                "column": "user_id",
                "reference_table": "users",
                "reference_column": "id",
            },
        ),
    ),
    "cross_table_row_count": RuleSchema(
        rule_type="cross_table_row_count",
        fields=(
            FieldSchema(
                name="table1",
                field_type=FieldType.STRING,
                required=True,
                description="First table name",
            ),
            FieldSchema(
                name="table2",
                field_type=FieldType.STRING,
                required=True,
                description="Second table name",
            ),
            FieldSchema(
                name="tolerance",
                field_type=FieldType.FLOAT,
                required=False,
                default=0.0,
                description="Allowed row count difference ratio (0.0 to 1.0)",
                min_value=0.0,
                max_value=1.0,
                aliases=("diff_tolerance", "threshold"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.CONSISTENCY,
        description="Check that two tables have comparable row counts",
        aliases=("row_count_match",),
        examples=(
            {"type": "cross_table_row_count", "table1": "orders", "table2": "order_items"},
        ),
    ),
    "conditional_null": RuleSchema(
        rule_type="conditional_null",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="condition",
                field_type=FieldType.STRING,
                required=True,
                description="SQL-like condition expression when column must not be null",
                aliases=("when", "predicate"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.COMPLETENESS,
        description="Check that column is not null when a condition is met",
        aliases=("conditional_not_null",),
        examples=(
            {"type": "conditional_null", "column": "email", "condition": "status = 'active'"},
        ),
    ),
    "expression": RuleSchema(
        rule_type="expression",
        fields=(
            FieldSchema(
                name="expression",
                field_type=FieldType.STRING,
                required=True,
                description="Boolean expression to evaluate against each row",
                aliases=("expr", "formula"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.VALIDITY,
        description="Evaluate a custom boolean expression against data",
        aliases=("custom_expression", "eval"),
        examples=(
            {"type": "expression", "expression": "price * quantity == total"},
        ),
    ),
    "distribution": RuleSchema(
        rule_type="distribution",
        fields=(
            COLUMN_FIELD,
            FieldSchema(
                name="distribution_type",
                field_type=FieldType.STRING,
                required=True,
                description="Expected distribution type (e.g., normal, uniform, exponential)",
                aliases=("dist_type", "dist"),
            ),
            FieldSchema(
                name="parameters",
                field_type=FieldType.DICT,
                required=False,
                default=None,
                description="Distribution parameters (e.g., mean, std for normal)",
                aliases=("params", "dist_params"),
            ),
            SEVERITY_FIELD,
        ),
        category=RuleCategory.ACCURACY,
        description="Check that column values follow an expected distribution",
        engines=("truthound",),
        aliases=("dist_check", "distribution_fit"),
        examples=(
            {
                "type": "distribution",
                "column": "height",
                "distribution_type": "normal",
                "parameters": {"mean": 170, "std": 10},
            },
        ),
    ),
}


# =============================================================================
# Rule Registry
# =============================================================================


class RuleRegistry:
    """Central registry for rule schemas.

    Manages rule schemas and provides lookup by rule type or alias.
    Thread-safe for concurrent access.

    Example:
        >>> registry = RuleRegistry()
        >>> registry.register(my_custom_rule_schema)
        >>> schema = registry.get("my_custom_rule")
    """

    def __init__(self) -> None:
        """Initialize the registry with common rule schemas."""
        self._schemas: dict[str, RuleSchema] = {}
        self._aliases: dict[str, str] = {}

        # Register common schemas
        for schema in COMMON_RULE_SCHEMAS.values():
            self.register(schema)

    def register(self, schema: RuleSchema) -> None:
        """Register a rule schema.

        Args:
            schema: The rule schema to register.

        Raises:
            ValueError: If the rule type is already registered.
        """
        if schema.rule_type in self._schemas:
            raise ValueError(f"Rule type '{schema.rule_type}' is already registered")

        self._schemas[schema.rule_type] = schema

        # Register aliases
        for alias in schema.aliases:
            if alias in self._aliases:
                raise ValueError(f"Alias '{alias}' is already registered")
            self._aliases[alias] = schema.rule_type

    def unregister(self, rule_type: str) -> None:
        """Unregister a rule schema.

        Args:
            rule_type: The rule type to unregister.
        """
        if rule_type in self._schemas:
            schema = self._schemas.pop(rule_type)
            for alias in schema.aliases:
                self._aliases.pop(alias, None)

    def get(self, rule_type: str) -> RuleSchema | None:
        """Get a rule schema by type or alias.

        Args:
            rule_type: The rule type or alias.

        Returns:
            RuleSchema if found, None otherwise.
        """
        # Direct lookup
        if rule_type in self._schemas:
            return self._schemas[rule_type]

        # Alias lookup
        if rule_type in self._aliases:
            return self._schemas[self._aliases[rule_type]]

        return None

    def list_rule_types(self) -> list[str]:
        """List all registered rule types.

        Returns:
            List of rule type names.
        """
        return list(self._schemas.keys())

    def list_by_category(self, category: RuleCategory) -> list[RuleSchema]:
        """List rule schemas by category.

        Args:
            category: The category to filter by.

        Returns:
            List of rule schemas in the category.
        """
        return [s for s in self._schemas.values() if s.category == category]

    def list_by_engine(self, engine: str) -> list[RuleSchema]:
        """List rule schemas supported by an engine.

        Args:
            engine: The engine name.

        Returns:
            List of rule schemas supported by the engine.
        """
        return [s for s in self._schemas.values() if s.supports_engine(engine)]

    def clear(self) -> None:
        """Clear all registered schemas."""
        self._schemas.clear()
        self._aliases.clear()


# Global registry instance
_global_registry: RuleRegistry | None = None


def get_rule_registry() -> RuleRegistry:
    """Get the global rule registry.

    Returns:
        The global RuleRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = RuleRegistry()
    return _global_registry


def reset_rule_registry() -> None:
    """Reset the global rule registry to default state."""
    global _global_registry
    _global_registry = None


# =============================================================================
# Standard Validator Implementation
# =============================================================================


@dataclass
class StandardRuleValidator:
    """Standard implementation of rule validation.

    Validates rules against registered schemas in a RuleRegistry.

    Attributes:
        registry: The rule registry to use for validation.
        engine: Optional engine name for engine-specific validation.
        allow_unknown_types: Whether to allow unknown rule types.

    Example:
        >>> validator = StandardRuleValidator(engine="great_expectations")
        >>> result = validator.validate({"type": "not_null", "column": "id"})
        >>> result.is_valid
        True
    """

    registry: RuleRegistry = field(default_factory=get_rule_registry)
    engine: str | None = None
    allow_unknown_types: bool = False

    def validate(
        self,
        rule: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> RuleValidationResult:
        """Validate a single rule.

        Args:
            rule: The rule dictionary to validate.
            strict: If True, unknown fields cause warnings.

        Returns:
            RuleValidationResult with validation outcome.
        """
        # Check for type field
        if "type" not in rule:
            error = MissingFieldError(
                "type",
                details={"rule": dict(rule)},
            )
            return RuleValidationResult(
                is_valid=False,
                rule_type="unknown",
                errors=(error,),
            )

        rule_type = rule["type"]

        # Look up schema
        schema = self.registry.get(rule_type)
        if schema is None:
            if self.allow_unknown_types:
                return RuleValidationResult(
                    is_valid=True,
                    rule_type=rule_type,
                    warnings=(f"Unknown rule type '{rule_type}' - skipping validation",),
                )
            error = UnknownRuleTypeError(
                rule_type,
                available_types=self.registry.list_rule_types(),
            )
            return RuleValidationResult(
                is_valid=False,
                rule_type=rule_type,
                errors=(error,),
            )

        # Check engine compatibility
        if self.engine and not schema.supports_engine(self.engine):
            error = RuleValidationError(
                f"Rule type '{rule_type}' is not supported by engine '{self.engine}'",
                rule_type=rule_type,
                expected=f"one of {schema.engines}",
                actual=self.engine,
            )
            return RuleValidationResult(
                is_valid=False,
                rule_type=rule_type,
                errors=(error,),
            )

        # Validate against schema
        return schema.validate(rule, strict=strict)

    def validate_batch(
        self,
        rules: Sequence[Mapping[str, Any]],
        *,
        strict: bool = False,
        fail_fast: bool = False,
    ) -> BatchValidationResult:
        """Validate multiple rules.

        Args:
            rules: Sequence of rule dictionaries to validate.
            strict: If True, unknown fields cause warnings.
            fail_fast: If True, stop on first error.

        Returns:
            BatchValidationResult with validation outcomes.
        """
        results: list[RuleValidationResult] = []

        for i, rule in enumerate(rules):
            result = self.validate(rule, strict=strict)
            # Add rule index to errors
            if not result.is_valid:
                indexed_errors = tuple(
                    RuleValidationError(
                        e.message,
                        rule_index=i,
                        rule_type=e.rule_type,
                        field=e.field,
                        expected=e.expected,
                        actual=e.actual,
                        details=e.details,
                    )
                    for e in result.errors
                )
                result = RuleValidationResult(
                    is_valid=False,
                    rule_type=result.rule_type,
                    errors=indexed_errors,
                    warnings=result.warnings,
                )
                if fail_fast:
                    results.append(result)
                    break
            results.append(result)

        return BatchValidationResult.from_results(results)

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if a rule type is supported.

        Args:
            rule_type: The rule type to check.

        Returns:
            True if the rule type is supported.
        """
        schema = self.registry.get(rule_type)
        if schema is None:
            return False
        if self.engine:
            return schema.supports_engine(self.engine)
        return True


# =============================================================================
# Engine-Specific Validators
# =============================================================================


class TruthoundRuleValidator(StandardRuleValidator):
    """Validator for Truthound engine rules.

    Truthound uses schema-based validation, so most rules are passed through
    but with a warning that they may be ignored.
    """

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        super().__init__(
            registry=registry or get_rule_registry(),
            engine="truthound",
            allow_unknown_types=True,
        )

    def validate(
        self,
        rule: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> RuleValidationResult:
        result = super().validate(rule, strict=strict)
        # Add warning about Truthound's schema-based approach
        warnings = list(result.warnings)
        warnings.append(
            "Truthound uses schema-based validation. "
            "Rule dictionaries are converted to schema constraints."
        )
        return RuleValidationResult(
            is_valid=result.is_valid,
            rule_type=result.rule_type,
            errors=result.errors,
            warnings=tuple(warnings),
        )


class GreatExpectationsRuleValidator(StandardRuleValidator):
    """Validator for Great Expectations engine rules.

    Supports both common rule types and native GE expectations.
    """

    # GE expectation pattern
    GE_EXPECTATION_PATTERN = re.compile(r"^expect_\w+$")

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        super().__init__(
            registry=registry or get_rule_registry(),
            engine="great_expectations",
            allow_unknown_types=False,
        )

    def validate(
        self,
        rule: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> RuleValidationResult:
        if "type" not in rule:
            error = MissingFieldError("type")
            return RuleValidationResult(
                is_valid=False,
                rule_type="unknown",
                errors=(error,),
            )

        rule_type = rule["type"]

        # Allow native GE expectations
        if self.GE_EXPECTATION_PATTERN.match(rule_type):
            return RuleValidationResult(
                is_valid=True,
                rule_type=rule_type,
                warnings=(f"Native GE expectation '{rule_type}' - minimal validation",),
            )

        return super().validate(rule, strict=strict)


class PanderaRuleValidator(StandardRuleValidator):
    """Validator for Pandera engine rules."""

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        super().__init__(
            registry=registry or get_rule_registry(),
            engine="pandera",
            allow_unknown_types=False,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_rule(
    rule: Mapping[str, Any],
    *,
    engine: str | None = None,
    strict: bool = False,
    registry: RuleRegistry | None = None,
) -> RuleValidationResult:
    """Validate a single rule.

    Convenience function for validating a single rule dictionary.

    Args:
        rule: The rule dictionary to validate.
        engine: Optional engine name for engine-specific validation.
        strict: If True, unknown fields cause warnings.
        registry: Optional rule registry to use.

    Returns:
        RuleValidationResult with validation outcome.

    Example:
        >>> result = validate_rule({"type": "not_null", "column": "id"})
        >>> result.is_valid
        True
    """
    validator = StandardRuleValidator(
        registry=registry or get_rule_registry(),
        engine=engine,
    )
    return validator.validate(rule, strict=strict)


def validate_rules(
    rules: Sequence[Mapping[str, Any]],
    *,
    engine: str | None = None,
    strict: bool = False,
    fail_fast: bool = False,
    raise_on_error: bool = False,
    registry: RuleRegistry | None = None,
) -> BatchValidationResult:
    """Validate multiple rules.

    Convenience function for validating multiple rule dictionaries.

    Args:
        rules: Sequence of rule dictionaries to validate.
        engine: Optional engine name for engine-specific validation.
        strict: If True, unknown fields cause warnings.
        fail_fast: If True, stop on first error.
        raise_on_error: If True, raise exception on validation failure.
        registry: Optional rule registry to use.

    Returns:
        BatchValidationResult with validation outcomes.

    Raises:
        MultipleRuleValidationErrors: If raise_on_error is True and validation fails.

    Example:
        >>> result = validate_rules([
        ...     {"type": "not_null", "column": "id"},
        ...     {"type": "in_range", "column": "age", "min": 0},
        ... ])
        >>> result.is_valid
        True
    """
    validator = StandardRuleValidator(
        registry=registry or get_rule_registry(),
        engine=engine,
    )
    result = validator.validate_batch(rules, strict=strict, fail_fast=fail_fast)

    if raise_on_error:
        result.raise_if_invalid()

    return result


def get_validator_for_engine(engine: str) -> StandardRuleValidator:
    """Get a validator for a specific engine.

    Args:
        engine: The engine name.

    Returns:
        A validator configured for the engine.

    Example:
        >>> validator = get_validator_for_engine("great_expectations")
        >>> result = validator.validate({"type": "not_null", "column": "id"})
    """
    engine_lower = engine.lower()
    if engine_lower == "truthound":
        return TruthoundRuleValidator()
    elif engine_lower in ("great_expectations", "ge"):
        return GreatExpectationsRuleValidator()
    elif engine_lower == "pandera":
        return PanderaRuleValidator()
    else:
        return StandardRuleValidator(engine=engine)


def register_rule_schema(schema: RuleSchema) -> None:
    """Register a rule schema in the global registry.

    Args:
        schema: The rule schema to register.

    Example:
        >>> schema = RuleSchema(
        ...     rule_type="my_custom_rule",
        ...     fields=(
        ...         FieldSchema(name="column", field_type=FieldType.STRING, required=True),
        ...     ),
        ... )
        >>> register_rule_schema(schema)
    """
    get_rule_registry().register(schema)


def get_supported_rule_types(engine: str | None = None) -> list[str]:
    """Get list of supported rule types.

    Args:
        engine: Optional engine name to filter by.

    Returns:
        List of supported rule type names.
    """
    registry = get_rule_registry()
    if engine:
        return [s.rule_type for s in registry.list_by_engine(engine)]
    return registry.list_rule_types()


# =============================================================================
# Rule Normalizer
# =============================================================================


class RuleNormalizer:
    """Normalizes rules by resolving aliases and applying defaults.

    Converts rules with aliases to their canonical form and fills in
    default values for optional fields.

    Example:
        >>> normalizer = RuleNormalizer()
        >>> rule = {"type": "in_range", "column": "age", "min_value": 0}
        >>> normalized = normalizer.normalize(rule)
        >>> normalized
        {"type": "in_range", "column": "age", "min": 0}
    """

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        """Initialize the normalizer.

        Args:
            registry: Optional rule registry to use.
        """
        self.registry = registry or get_rule_registry()

    def normalize(self, rule: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize a single rule.

        Args:
            rule: The rule dictionary to normalize.

        Returns:
            Normalized rule dictionary.
        """
        if "type" not in rule:
            return dict(rule)

        rule_type = rule["type"]
        schema = self.registry.get(rule_type)
        if schema is None:
            return dict(rule)

        normalized: dict[str, Any] = {"type": rule_type}

        # Process fields
        for field_schema in schema.fields:
            # Find the field in the rule (check name and aliases)
            field_name = field_schema.get_field_name(rule)
            if field_name is not None:
                # Use canonical name
                normalized[field_schema.name] = rule[field_name]
            elif not field_schema.required and field_schema.default is not None:
                # Apply default
                normalized[field_schema.name] = field_schema.default

        # Preserve any extra fields not in schema
        known_fields = {"type"}
        for f in schema.fields:
            known_fields.add(f.name)
            known_fields.update(f.aliases)

        for key, value in rule.items():
            if key not in known_fields:
                normalized[key] = value

        return normalized

    def normalize_batch(self, rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Normalize multiple rules.

        Args:
            rules: Sequence of rule dictionaries to normalize.

        Returns:
            List of normalized rule dictionaries.
        """
        return [self.normalize(rule) for rule in rules]


def normalize_rules(
    rules: Sequence[Mapping[str, Any]],
    *,
    registry: RuleRegistry | None = None,
) -> list[dict[str, Any]]:
    """Normalize rules by resolving aliases and applying defaults.

    Convenience function for normalizing rules.

    Args:
        rules: Sequence of rule dictionaries to normalize.
        registry: Optional rule registry to use.

    Returns:
        List of normalized rule dictionaries.
    """
    normalizer = RuleNormalizer(registry=registry)
    return normalizer.normalize_batch(rules)


# =============================================================================
# Engine Integration
# =============================================================================


def create_validating_check(
    engine_check: Callable[..., Any],
    engine: str | None = None,
    *,
    validate: bool = True,
    normalize: bool = True,
    strict: bool = False,
    registry: RuleRegistry | None = None,
) -> Callable[..., Any]:
    """Create a wrapper that validates rules before calling the engine check.

    This is a decorator factory that adds rule validation to any engine's
    check method.

    Args:
        engine_check: The engine's check method to wrap.
        engine: Optional engine name for engine-specific validation.
        validate: Whether to validate rules.
        normalize: Whether to normalize rules.
        strict: If True, unknown fields cause warnings.
        registry: Optional rule registry to use.

    Returns:
        A wrapper function that validates rules before checking.

    Example:
        >>> engine = TruthoundEngine()
        >>> validating_check = create_validating_check(engine.check, "truthound")
        >>> result = validating_check(data, rules=[{"type": "not_null", "column": "id"}])
    """
    import functools

    reg = registry or get_rule_registry()
    validator = StandardRuleValidator(registry=reg, engine=engine)
    normalizer = RuleNormalizer(registry=reg) if normalize else None

    @functools.wraps(engine_check)
    def wrapper(
        data: Any,
        rules: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        if rules is not None and len(rules) > 0:
            if validate:
                result = validator.validate_batch(list(rules), strict=strict)
                result.raise_if_invalid()

            if normalize and normalizer:
                rules = normalizer.normalize_batch(rules)

        return engine_check(data, rules, **kwargs)

    return wrapper


class ValidatingEngineWrapper:
    """Wrapper that adds rule validation to any DataQualityEngine.

    Transparently wraps an engine and validates rules before passing
    them to the underlying engine's check method.

    Attributes:
        engine: The wrapped engine.
        validator: The rule validator.
        normalizer: Optional rule normalizer.

    Example:
        >>> from common.engines import TruthoundEngine
        >>> engine = TruthoundEngine()
        >>> validating_engine = ValidatingEngineWrapper(engine)
        >>> result = validating_engine.check(data, rules=[...])
    """

    def __init__(
        self,
        engine: Any,
        *,
        validate: bool = True,
        normalize: bool = True,
        strict: bool = False,
        registry: RuleRegistry | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            engine: The engine to wrap.
            validate: Whether to validate rules.
            normalize: Whether to normalize rules.
            strict: If True, unknown fields cause warnings.
            registry: Optional rule registry to use.
        """
        self._engine = engine
        self._validate = validate
        self._normalize = normalize
        self._strict = strict
        self._registry = registry or get_rule_registry()

        # Get engine name from wrapped engine
        engine_name = getattr(engine, "engine_name", None)
        self._validator = StandardRuleValidator(
            registry=self._registry,
            engine=engine_name,
        )
        self._normalizer = RuleNormalizer(registry=self._registry)

    @property
    def engine(self) -> Any:
        """Get the wrapped engine."""
        return self._engine

    @property
    def engine_name(self) -> str:
        """Get the engine name from the wrapped engine."""
        return getattr(self._engine, "engine_name", "unknown")

    @property
    def engine_version(self) -> str:
        """Get the engine version from the wrapped engine."""
        return getattr(self._engine, "engine_version", "0.0.0")

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Check data with rule validation.

        Args:
            data: The data to check.
            rules: Sequence of rule dictionaries.
            **kwargs: Additional arguments for the engine.

        Returns:
            CheckResult from the engine.

        Raises:
            RuleValidationError: If rule validation fails.
        """
        processed_rules = rules

        if rules is not None and len(rules) > 0:
            if self._validate:
                result = self._validator.validate_batch(
                    list(rules),
                    strict=self._strict,
                )
                result.raise_if_invalid()

            if self._normalize:
                processed_rules = self._normalizer.normalize_batch(rules)

        return self._engine.check(data, processed_rules, **kwargs)

    def profile(self, data: Any, **kwargs: Any) -> Any:
        """Profile data (delegated to wrapped engine)."""
        return self._engine.profile(data, **kwargs)

    def learn(self, data: Any, **kwargs: Any) -> Any:
        """Learn rules from data (delegated to wrapped engine)."""
        return self._engine.learn(data, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped engine."""
        return getattr(self._engine, name)


def wrap_engine_with_validation(
    engine: Any,
    *,
    validate: bool = True,
    normalize: bool = True,
    strict: bool = False,
    registry: RuleRegistry | None = None,
) -> ValidatingEngineWrapper:
    """Wrap an engine with rule validation.

    Convenience function for creating a ValidatingEngineWrapper.

    Args:
        engine: The engine to wrap.
        validate: Whether to validate rules.
        normalize: Whether to normalize rules.
        strict: If True, unknown fields cause warnings.
        registry: Optional rule registry to use.

    Returns:
        ValidatingEngineWrapper instance.

    Example:
        >>> from common.engines import TruthoundEngine
        >>> engine = wrap_engine_with_validation(TruthoundEngine())
        >>> result = engine.check(data, rules=[{"type": "not_null", "column": "id"}])
    """
    return ValidatingEngineWrapper(
        engine,
        validate=validate,
        normalize=normalize,
        strict=strict,
        registry=registry,
    )


# =============================================================================
# Decorator for Engine Methods
# =============================================================================


def validate_rules_decorator(
    engine: str | None = None,
    *,
    strict: bool = False,
    normalize: bool = True,
    registry: RuleRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that validates rules before calling a method.

    Apply this decorator to engine check methods to automatically
    validate rules.

    Args:
        engine: Optional engine name for engine-specific validation.
        strict: If True, unknown fields cause warnings.
        normalize: Whether to normalize rules.
        registry: Optional rule registry to use.

    Returns:
        Decorator function.

    Example:
        >>> class MyEngine:
        ...     @validate_rules_decorator(engine="my_engine")
        ...     def check(self, data, rules, **kwargs):
        ...         # Implementation
        ...         ...
    """
    import functools

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(
            self: Any,
            data: Any,
            rules: Sequence[Mapping[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            if rules is not None and len(rules) > 0:
                reg = registry or get_rule_registry()
                validator = StandardRuleValidator(registry=reg, engine=engine)
                result = validator.validate_batch(list(rules), strict=strict)
                result.raise_if_invalid()

                if normalize:
                    normalizer = RuleNormalizer(registry=reg)
                    rules = normalizer.normalize_batch(rules)

            return func(self, data, rules, **kwargs)

        return wrapper

    return decorator
