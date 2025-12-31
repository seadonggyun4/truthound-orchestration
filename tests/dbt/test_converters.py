"""Tests for rule converters."""

import pytest

from truthound_dbt.converters import (
    StandardRuleConverter,
    get_converter,
    list_converters,
    RuleSQL,
    ConversionResult,
    UnsupportedRuleError,
)
from truthound_dbt.converters.base import ConversionContext
from truthound_dbt.adapters import PostgresAdapter, SnowflakeAdapter
from truthound_dbt.testing import MockAdapter, create_sample_rules


class TestRuleSQL:
    """Tests for RuleSQL dataclass."""

    def test_creation(self):
        """Test RuleSQL creation."""
        rule_sql = RuleSQL(
            where_clause="col is null",
            rule_type="not_null",
            column="col",
        )
        assert rule_sql.where_clause == "col is null"
        assert rule_sql.rule_type == "not_null"
        assert rule_sql.column == "col"

    def test_frozen(self):
        """Test RuleSQL is immutable."""
        rule_sql = RuleSQL(where_clause="x", rule_type="test")
        with pytest.raises(AttributeError):
            rule_sql.where_clause = "y"


class TestStandardRuleConverter:
    """Tests for StandardRuleConverter."""

    def setup_method(self):
        """Create converter and adapter instances."""
        self.converter = StandardRuleConverter()
        self.adapter = MockAdapter()

    def test_convert_not_null(self):
        """Test converting not_null rule."""
        rule = {"type": "not_null", "column": "id"}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "not_null"
        assert result.column == "id"
        assert "is null" in result.where_clause.lower()

    def test_convert_unique(self):
        """Test converting unique rule."""
        rule = {"type": "unique", "column": "email"}
        context = ConversionContext(model="users")
        result = self.converter.convert(rule, self.adapter, context)
        assert result.rule_type == "unique"
        assert result.column == "email"

    def test_convert_in_set(self):
        """Test converting in_set rule."""
        rule = {"type": "in_set", "column": "status", "values": ["active", "inactive"]}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "in_set"
        assert result.column == "status"
        # where_clause checks for violations (not in set)
        assert "in (" in result.where_clause.lower() or "not" in result.where_clause.lower()

    def test_convert_in_range(self):
        """Test converting in_range rule."""
        rule = {"type": "in_range", "column": "age", "min": 0, "max": 150}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "in_range"
        assert result.column == "age"

    def test_convert_regex(self):
        """Test converting regex rule."""
        rule = {"type": "regex", "column": "email", "pattern": r"^[\w.-]+@[\w.-]+$"}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "regex"
        assert result.column == "email"

    def test_convert_email_format(self):
        """Test converting email_format rule."""
        rule = {"type": "email_format", "column": "email"}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "email_format"
        assert result.column == "email"

    def test_convert_url_format(self):
        """Test converting url_format rule."""
        rule = {"type": "url_format", "column": "website"}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "url_format"

    def test_convert_uuid_format(self):
        """Test converting uuid_format rule."""
        rule = {"type": "uuid_format", "column": "id"}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "uuid_format"

    def test_convert_min_length(self):
        """Test converting min_length rule."""
        rule = {"type": "min_length", "column": "name", "length": 2}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "min_length"
        assert "2" in result.where_clause

    def test_convert_max_length(self):
        """Test converting max_length rule."""
        rule = {"type": "max_length", "column": "name", "length": 100}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "max_length"
        assert "100" in result.where_clause

    def test_convert_referential_integrity(self):
        """Test converting referential_integrity rule."""
        rule = {
            "type": "referential_integrity",
            "column": "user_id",
            "to": "users",
            "field": "id",
        }
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "referential_integrity"

    def test_convert_expression(self):
        """Test converting expression rule."""
        rule = {"type": "expression", "expression": "total >= 0"}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "expression"
        assert "total >= 0" in result.where_clause

    def test_convert_row_count(self):
        """Test converting row_count rule."""
        rule = {"type": "row_count", "min": 1}
        result = self.converter.convert(rule, self.adapter)
        assert result.rule_type == "row_count"

    def test_convert_unsupported_rule_raises(self):
        """Test converting unsupported rule raises error."""
        rule = {"type": "unsupported_type", "column": "x"}
        with pytest.raises(UnsupportedRuleError):
            self.converter.convert(rule, self.adapter)

    def test_convert_all_success(self):
        """Test converting multiple rules."""
        rules = create_sample_rules(rule_types=["not_null", "unique", "in_set"])
        context = ConversionContext(model="test_model")
        result = self.converter.convert_all(rules, self.adapter, context)

        assert isinstance(result, ConversionResult)
        assert len(result.rules_sql) == 3
        assert not result.errors
        assert result.success_count == 3

    def test_convert_all_with_errors(self):
        """Test converting with some invalid rules."""
        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "invalid_type", "column": "x"},
            {"type": "unique", "column": "email"},
        ]
        context = ConversionContext(model="test_model")
        result = self.converter.convert_all(rules, self.adapter, context)

        assert len(result.rules_sql) == 2  # Only valid rules
        assert len(result.errors) == 1  # One error


class TestConverterRegistry:
    """Tests for converter registry."""

    def test_list_converters(self):
        """Test listing converters."""
        converters = list_converters()
        assert "standard" in converters

    def test_get_converter_standard(self):
        """Test getting standard converter."""
        converter = get_converter("standard")
        assert converter is not None

    def test_get_converter_default(self):
        """Test getting default converter."""
        converter = get_converter("default")
        assert converter is not None


class TestCrossAdapterConversion:
    """Tests for conversion across different adapters."""

    @pytest.fixture
    def converters_and_adapters(self):
        """Get converter with multiple adapters."""
        return {
            "converter": StandardRuleConverter(),
            "postgres": PostgresAdapter(),
            "snowflake": SnowflakeAdapter(),
            "mock": MockAdapter(),
        }

    def test_not_null_same_across_adapters(self, converters_and_adapters):
        """Test not_null produces consistent results."""
        rule = {"type": "not_null", "column": "id"}
        converter = converters_and_adapters["converter"]

        for name, adapter in converters_and_adapters.items():
            if name == "converter":
                continue
            result = converter.convert(rule, adapter)
            assert result.rule_type == "not_null"
            assert "is null" in result.where_clause.lower()

    def test_regex_uses_adapter_specific_syntax(self, converters_and_adapters):
        """Test regex uses adapter-specific syntax."""
        rule = {"type": "regex", "column": "email", "pattern": r"pattern"}
        converter = converters_and_adapters["converter"]

        postgres_result = converter.convert(rule, converters_and_adapters["postgres"])
        snowflake_result = converter.convert(rule, converters_and_adapters["snowflake"])

        # Different syntax
        assert "~" in postgres_result.where_clause
        assert "regexp_like" in snowflake_result.where_clause
