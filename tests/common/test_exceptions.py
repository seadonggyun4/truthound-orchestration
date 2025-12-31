"""Tests for common.exceptions module."""

import pytest

from common.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DataAccessError,
    DeserializeError,
    IntegrationTimeoutError,
    InvalidConfigValueError,
    MissingConfigError,
    PlatformConnectionError,
    QualityGateError,
    RuleExecutionError,
    SerializationError,
    SerializeError,
    ThresholdExceededError,
    TruthoundIntegrationError,
    ValidationExecutionError,
    wrap_exception,
)


class TestTruthoundIntegrationError:
    """Tests for TruthoundIntegrationError base class."""

    def test_basic_creation(self):
        """Test basic exception creation."""
        error = TruthoundIntegrationError("Test error")
        assert error.message == "Test error"
        assert error.details == {}
        assert error.cause is None

    def test_with_details(self):
        """Test exception with details."""
        error = TruthoundIntegrationError(
            "Test error",
            details={"key": "value", "count": 42},
        )
        assert error.details == {"key": "value", "count": 42}

    def test_with_cause(self):
        """Test exception with cause."""
        cause = ValueError("Original error")
        error = TruthoundIntegrationError("Wrapped error", cause=cause)
        assert error.cause is cause

    def test_str_without_details(self):
        """Test string representation without details."""
        error = TruthoundIntegrationError("Test error")
        assert str(error) == "Test error"

    def test_str_with_details(self):
        """Test string representation with details."""
        error = TruthoundIntegrationError("Test error", details={"key": "value"})
        assert "Test error" in str(error)
        assert "Details:" in str(error)

    def test_repr(self):
        """Test repr representation."""
        error = TruthoundIntegrationError("Test", details={"x": 1})
        repr_str = repr(error)
        assert "TruthoundIntegrationError" in repr_str
        assert "Test" in repr_str

    def test_with_context(self):
        """Test with_context method."""
        error = TruthoundIntegrationError("Error", details={"a": 1})
        new_error = error.with_context(b=2, c=3)

        # Original unchanged
        assert error.details == {"a": 1}

        # New error has merged details
        assert new_error.details == {"a": 1, "b": 2, "c": 3}
        assert new_error.message == error.message

    def test_exception_chain(self):
        """Test exception can be raised and caught."""
        with pytest.raises(TruthoundIntegrationError) as exc_info:
            raise TruthoundIntegrationError("Test")
        assert exc_info.value.message == "Test"


class TestConfigurationErrors:
    """Tests for configuration error classes."""

    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError("Config error", config_key="test_key")
        assert error.config_key == "test_key"
        assert error.details["config_key"] == "test_key"

    def test_invalid_config_value_error(self):
        """Test InvalidConfigValueError creation."""
        error = InvalidConfigValueError(
            "Invalid value",
            config_key="timeout",
            value=-1,
            expected="positive integer",
        )
        assert error.config_key == "timeout"
        assert error.value == -1
        assert error.expected == "positive integer"
        assert error.details["value"] == -1
        assert error.details["expected"] == "positive integer"

    def test_missing_config_error(self):
        """Test MissingConfigError creation."""
        error = MissingConfigError("API_KEY")
        assert error.config_key == "API_KEY"
        assert "API_KEY" in error.message
        assert "missing" in error.message.lower()

    def test_configuration_error_hierarchy(self):
        """Test configuration errors inherit correctly."""
        assert issubclass(ConfigurationError, TruthoundIntegrationError)
        assert issubclass(InvalidConfigValueError, ConfigurationError)
        assert issubclass(MissingConfigError, ConfigurationError)


class TestValidationErrors:
    """Tests for validation error classes."""

    def test_validation_execution_error(self):
        """Test ValidationExecutionError creation."""
        error = ValidationExecutionError("Validation failed", rule_name="not_null")
        assert error.rule_name == "not_null"
        assert error.details["rule_name"] == "not_null"

    def test_rule_execution_error(self):
        """Test RuleExecutionError creation."""
        error = RuleExecutionError(
            "Rule failed",
            rule_name="unique",
            column="id",
        )
        assert error.rule_name == "unique"
        assert error.column == "id"

    def test_data_access_error(self):
        """Test DataAccessError creation."""
        error = DataAccessError("Cannot read data", source="s3://bucket/data.parquet")
        assert error.source == "s3://bucket/data.parquet"

    def test_validation_error_hierarchy(self):
        """Test validation errors inherit correctly."""
        assert issubclass(ValidationExecutionError, TruthoundIntegrationError)
        assert issubclass(RuleExecutionError, ValidationExecutionError)
        assert issubclass(DataAccessError, ValidationExecutionError)


class TestSerializationErrors:
    """Tests for serialization error classes."""

    def test_serialization_error(self):
        """Test SerializationError creation."""
        error = SerializationError("Serialization failed", target_type="json")
        assert error.target_type == "json"

    def test_serialize_error(self):
        """Test SerializeError creation."""
        error = SerializeError("Cannot serialize", target_type="dict")
        assert isinstance(error, SerializationError)

    def test_deserialize_error(self):
        """Test DeserializeError creation."""
        error = DeserializeError("Cannot deserialize", target_type="CheckResult")
        assert isinstance(error, SerializationError)


class TestPlatformErrors:
    """Tests for platform error classes."""

    def test_platform_connection_error(self):
        """Test PlatformConnectionError creation."""
        error = PlatformConnectionError(
            "Connection failed",
            platform="airflow",
            endpoint="http://localhost:8080",
        )
        assert error.platform == "airflow"
        assert error.endpoint == "http://localhost:8080"

    def test_authentication_error(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError("Auth failed", platform="dagster")
        assert isinstance(error, PlatformConnectionError)


class TestTimeoutError:
    """Tests for IntegrationTimeoutError."""

    def test_timeout_error(self):
        """Test IntegrationTimeoutError creation."""
        error = IntegrationTimeoutError(
            "Operation timed out",
            timeout_seconds=30.0,
            operation="data_validation",
        )
        assert error.timeout_seconds == 30.0
        assert error.operation == "data_validation"


class TestQualityGateErrors:
    """Tests for quality gate error classes."""

    def test_quality_gate_error(self):
        """Test QualityGateError creation."""
        error = QualityGateError(
            "Quality gate failed",
            failed_count=5,
            pass_rate=95.0,
        )
        assert error.failed_count == 5
        assert error.pass_rate == 95.0

    def test_threshold_exceeded_error(self):
        """Test ThresholdExceededError creation."""
        error = ThresholdExceededError(
            "Threshold exceeded",
            threshold=99.0,
            actual=95.0,
            threshold_type="pass_rate",
        )
        assert error.threshold == 99.0
        assert error.actual == 95.0
        assert error.threshold_type == "pass_rate"


class TestWrapException:
    """Tests for wrap_exception utility function."""

    def test_wrap_basic_exception(self):
        """Test wrapping a basic exception."""
        original = ValueError("bad value")
        wrapped = wrap_exception(original)

        assert isinstance(wrapped, TruthoundIntegrationError)
        assert wrapped.cause is original
        assert "bad value" in wrapped.message

    def test_wrap_with_custom_class(self):
        """Test wrapping with custom exception class."""
        original = ValueError("invalid")
        wrapped = wrap_exception(
            original,
            wrapper_class=InvalidConfigValueError,
            config_key="test",
        )

        assert isinstance(wrapped, InvalidConfigValueError)
        assert wrapped.config_key == "test"
        assert wrapped.cause is original

    def test_wrap_with_custom_message(self):
        """Test wrapping with custom message."""
        original = RuntimeError("internal")
        wrapped = wrap_exception(
            original,
            message="Custom error message",
        )

        assert wrapped.message == "Custom error message"
        assert wrapped.cause is original


class TestExceptionCatching:
    """Tests for catching exceptions at different levels."""

    def test_catch_all_integration_errors(self):
        """Test catching all integration errors with base class."""
        errors = [
            ConfigurationError("config"),
            ValidationExecutionError("validation"),
            SerializationError("serialization"),
            PlatformConnectionError("connection"),
            IntegrationTimeoutError("timeout"),
            QualityGateError("quality"),
        ]

        for error in errors:
            with pytest.raises(TruthoundIntegrationError):
                raise error

    def test_catch_specific_error_type(self):
        """Test catching specific error types."""
        with pytest.raises(ConfigurationError):
            raise InvalidConfigValueError("invalid", config_key="key")

        with pytest.raises(ValidationExecutionError):
            raise RuleExecutionError("failed", rule_name="rule")
