"""Tests for semantic attribute conventions."""

import pytest

from common.opentelemetry.semantic.attributes import (
    CheckAttributes,
    CheckStatus,
    DataQualityAttributes,
    DatasetAttributes,
    EngineAttributes,
    LearnAttributes,
    OperationStatus,
    OperationType,
    ProfileAttributes,
    RuleAttributes,
    Severity,
    create_check_attributes,
    create_engine_attributes,
    create_learn_attributes,
    create_profile_attributes,
)


class TestDataQualityAttributes:
    """Tests for DataQualityAttributes namespace."""

    def test_namespace(self):
        """Test namespace prefix."""
        assert DataQualityAttributes.NAMESPACE == "dq"

    def test_engine_attributes(self):
        """Test engine attribute keys."""
        assert DataQualityAttributes.ENGINE_NAME == "dq.engine.name"
        assert DataQualityAttributes.ENGINE_VERSION == "dq.engine.version"
        assert DataQualityAttributes.ENGINE_TYPE == "dq.engine.type"

    def test_operation_attributes(self):
        """Test operation attribute keys."""
        assert DataQualityAttributes.OPERATION_TYPE == "dq.operation.type"
        assert DataQualityAttributes.OPERATION_STATUS == "dq.operation.status"
        assert DataQualityAttributes.OPERATION_DURATION_MS == "dq.operation.duration_ms"

    def test_check_attributes(self):
        """Test check attribute keys."""
        assert DataQualityAttributes.CHECK_STATUS == "dq.check.status"
        assert DataQualityAttributes.CHECK_PASSED_COUNT == "dq.check.passed_count"
        assert DataQualityAttributes.CHECK_FAILED_COUNT == "dq.check.failed_count"
        assert DataQualityAttributes.CHECK_SUCCESS_RATE == "dq.check.success_rate"

    def test_profile_attributes(self):
        """Test profile attribute keys."""
        assert DataQualityAttributes.PROFILE_ROW_COUNT == "dq.profile.row_count"
        assert DataQualityAttributes.PROFILE_COLUMN_COUNT == "dq.profile.column_count"

    def test_learn_attributes(self):
        """Test learn attribute keys."""
        assert DataQualityAttributes.LEARN_RULES_GENERATED == "dq.learn.rules_generated"
        assert DataQualityAttributes.LEARN_CONFIDENCE_AVG == "dq.learn.confidence_avg"

    def test_dataset_attributes(self):
        """Test dataset attribute keys."""
        assert DataQualityAttributes.DATASET_NAME == "dq.dataset.name"
        assert DataQualityAttributes.DATASET_SOURCE == "dq.dataset.source"
        assert DataQualityAttributes.DATASET_ROW_COUNT == "dq.dataset.row_count"

    def test_platform_attributes(self):
        """Test platform integration attribute keys."""
        assert DataQualityAttributes.PLATFORM_NAME == "dq.platform.name"
        assert DataQualityAttributes.PLATFORM_TASK_ID == "dq.platform.task_id"
        assert DataQualityAttributes.PLATFORM_DAG_ID == "dq.platform.dag_id"


class TestEnums:
    """Tests for semantic enums."""

    def test_operation_type(self):
        """Test OperationType enum."""
        assert OperationType.CHECK.value == "check"
        assert OperationType.PROFILE.value == "profile"
        assert OperationType.LEARN.value == "learn"

    def test_operation_status(self):
        """Test OperationStatus enum."""
        assert OperationStatus.STARTED.value == "started"
        assert OperationStatus.COMPLETED.value == "completed"
        assert OperationStatus.FAILED.value == "failed"

    def test_check_status(self):
        """Test CheckStatus enum."""
        assert CheckStatus.PASSED.value == "passed"
        assert CheckStatus.FAILED.value == "failed"
        assert CheckStatus.WARNING.value == "warning"
        assert CheckStatus.SKIPPED.value == "skipped"
        assert CheckStatus.ERROR.value == "error"

    def test_severity(self):
        """Test Severity enum."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"


class TestEngineAttributes:
    """Tests for EngineAttributes dataclass."""

    def test_basic_creation(self):
        """Test creating engine attributes."""
        attrs = EngineAttributes(name="truthound")
        assert attrs.name == "truthound"
        assert attrs.version is None
        assert attrs.engine_type is None

    def test_full_creation(self):
        """Test creating engine attributes with all fields."""
        attrs = EngineAttributes(
            name="truthound",
            version="1.0.0",
            engine_type="native",
        )
        assert attrs.name == "truthound"
        assert attrs.version == "1.0.0"
        assert attrs.engine_type == "native"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attrs = EngineAttributes(
            name="truthound",
            version="1.0.0",
            engine_type="native",
        )
        d = attrs.to_dict()
        assert d[DataQualityAttributes.ENGINE_NAME] == "truthound"
        assert d[DataQualityAttributes.ENGINE_VERSION] == "1.0.0"
        assert d[DataQualityAttributes.ENGINE_TYPE] == "native"

    def test_to_dict_minimal(self):
        """Test conversion with minimal fields."""
        attrs = EngineAttributes(name="truthound")
        d = attrs.to_dict()
        assert d[DataQualityAttributes.ENGINE_NAME] == "truthound"
        assert DataQualityAttributes.ENGINE_VERSION not in d
        assert DataQualityAttributes.ENGINE_TYPE not in d


class TestCheckAttributes:
    """Tests for CheckAttributes dataclass."""

    def test_basic_creation(self):
        """Test creating check attributes."""
        attrs = CheckAttributes(status=CheckStatus.PASSED)
        assert attrs.status == CheckStatus.PASSED
        assert attrs.passed_count == 0
        assert attrs.failed_count == 0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        attrs = CheckAttributes(
            status=CheckStatus.PASSED,
            passed_count=8,
            failed_count=2,
            rules_count=10,
        )
        assert attrs.success_rate == 0.8

    def test_success_rate_zero_rules(self):
        """Test success rate with zero rules."""
        attrs = CheckAttributes(status=CheckStatus.PASSED, rules_count=0)
        assert attrs.success_rate == 1.0  # Default to 100% when no rules

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attrs = CheckAttributes(
            status=CheckStatus.FAILED,
            passed_count=5,
            failed_count=3,
            warning_count=1,
            error_count=1,
            rules_count=10,
        )
        d = attrs.to_dict()
        assert d[DataQualityAttributes.CHECK_STATUS] == "failed"
        assert d[DataQualityAttributes.CHECK_PASSED_COUNT] == 5
        assert d[DataQualityAttributes.CHECK_FAILED_COUNT] == 3
        assert d[DataQualityAttributes.CHECK_SUCCESS_RATE] == 0.5


class TestProfileAttributes:
    """Tests for ProfileAttributes dataclass."""

    def test_basic_creation(self):
        """Test creating profile attributes."""
        attrs = ProfileAttributes()
        assert attrs.row_count == 0
        assert attrs.column_count == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attrs = ProfileAttributes(
            row_count=1000,
            column_count=10,
            null_percentage=5.5,
            unique_ratio=0.95,
        )
        d = attrs.to_dict()
        assert d[DataQualityAttributes.PROFILE_ROW_COUNT] == 1000
        assert d[DataQualityAttributes.PROFILE_COLUMN_COUNT] == 10
        assert d[DataQualityAttributes.PROFILE_NULL_PERCENTAGE] == 5.5
        assert d[DataQualityAttributes.PROFILE_UNIQUE_RATIO] == 0.95


class TestLearnAttributes:
    """Tests for LearnAttributes dataclass."""

    def test_basic_creation(self):
        """Test creating learn attributes."""
        attrs = LearnAttributes()
        assert attrs.rules_generated == 0
        assert attrs.confidence_avg == 0.0
        assert attrs.coverage == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attrs = LearnAttributes(
            rules_generated=15,
            confidence_avg=0.85,
            coverage=0.9,
        )
        d = attrs.to_dict()
        assert d[DataQualityAttributes.LEARN_RULES_GENERATED] == 15
        assert d[DataQualityAttributes.LEARN_CONFIDENCE_AVG] == 0.85
        assert d[DataQualityAttributes.LEARN_COVERAGE] == 0.9


class TestRuleAttributes:
    """Tests for RuleAttributes dataclass."""

    def test_basic_creation(self):
        """Test creating rule attributes."""
        attrs = RuleAttributes(rule_type="not_null")
        assert attrs.rule_type == "not_null"
        assert attrs.column is None
        assert attrs.severity == Severity.MEDIUM
        assert attrs.passed is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attrs = RuleAttributes(
            rule_type="unique",
            column="email",
            severity=Severity.HIGH,
            passed=False,
        )
        d = attrs.to_dict()
        assert d[DataQualityAttributes.RULE_TYPE] == "unique"
        assert d[DataQualityAttributes.RULE_COLUMN] == "email"
        assert d[DataQualityAttributes.RULE_SEVERITY] == "high"
        assert d[DataQualityAttributes.RULE_PASSED] is False


class TestDatasetAttributes:
    """Tests for DatasetAttributes dataclass."""

    def test_basic_creation(self):
        """Test creating dataset attributes."""
        attrs = DatasetAttributes()
        assert attrs.name is None
        assert attrs.source is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attrs = DatasetAttributes(
            name="users",
            source="s3://bucket/users.parquet",
            format="parquet",
            size_bytes=1024000,
            row_count=10000,
            column_count=20,
        )
        d = attrs.to_dict()
        assert d[DataQualityAttributes.DATASET_NAME] == "users"
        assert d[DataQualityAttributes.DATASET_SOURCE] == "s3://bucket/users.parquet"
        assert d[DataQualityAttributes.DATASET_FORMAT] == "parquet"
        assert d[DataQualityAttributes.DATASET_SIZE_BYTES] == 1024000

    def test_to_dict_minimal(self):
        """Test conversion with None values."""
        attrs = DatasetAttributes(name="test")
        d = attrs.to_dict()
        assert d[DataQualityAttributes.DATASET_NAME] == "test"
        assert DataQualityAttributes.DATASET_SOURCE not in d
        assert DataQualityAttributes.DATASET_FORMAT not in d


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_engine_attributes(self):
        """Test create_engine_attributes function."""
        attrs = create_engine_attributes(
            name="truthound",
            version="1.0.0",
            engine_type="native",
        )
        assert attrs[DataQualityAttributes.ENGINE_NAME] == "truthound"
        assert attrs[DataQualityAttributes.ENGINE_VERSION] == "1.0.0"
        assert attrs[DataQualityAttributes.ENGINE_TYPE] == "native"

    def test_create_check_attributes(self):
        """Test create_check_attributes function."""
        attrs = create_check_attributes(
            status=CheckStatus.PASSED,
            passed_count=10,
            failed_count=0,
            rules_count=10,
        )
        assert attrs[DataQualityAttributes.CHECK_STATUS] == "passed"
        assert attrs[DataQualityAttributes.CHECK_PASSED_COUNT] == 10

    def test_create_check_attributes_string_status(self):
        """Test create_check_attributes with string status."""
        attrs = create_check_attributes(
            status="failed",
            passed_count=5,
            failed_count=5,
            rules_count=10,
        )
        assert attrs[DataQualityAttributes.CHECK_STATUS] == "failed"

    def test_create_profile_attributes(self):
        """Test create_profile_attributes function."""
        attrs = create_profile_attributes(
            row_count=1000,
            column_count=20,
            null_percentage=2.5,
            unique_ratio=0.9,
        )
        assert attrs[DataQualityAttributes.PROFILE_ROW_COUNT] == 1000
        assert attrs[DataQualityAttributes.PROFILE_COLUMN_COUNT] == 20

    def test_create_learn_attributes(self):
        """Test create_learn_attributes function."""
        attrs = create_learn_attributes(
            rules_generated=12,
            confidence_avg=0.88,
            coverage=0.95,
        )
        assert attrs[DataQualityAttributes.LEARN_RULES_GENERATED] == 12
        assert attrs[DataQualityAttributes.LEARN_CONFIDENCE_AVG] == 0.88
