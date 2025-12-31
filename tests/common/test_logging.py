"""Tests for common.logging module."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.logging import (
    AirflowLoggerAdapter,
    BufferingHandler,
    ContextFilter,
    DagsterLoggerAdapter,
    JSONFormatter,
    LevelFilter,
    LogContext,
    LogContextData,
    LogLevel,
    LogRecord,
    NullHandler,
    PerformanceLogger,
    PrefectLoggerAdapter,
    RegexFilter,
    SensitiveDataMasker,
    StdlibLoggerAdapter,
    StreamHandler,
    TextFormatter,
    TimingResult,
    TruthoundLogger,
    configure_logging,
    configure_masker,
    create_platform_handler,
    get_current_context,
    get_logger,
    get_masker,
    get_performance_logger,
    log_call,
    log_errors,
    set_context,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_to_stdlib(self):
        """Test conversion to stdlib logging levels."""
        assert LogLevel.DEBUG.to_stdlib() == 10
        assert LogLevel.INFO.to_stdlib() == 20
        assert LogLevel.WARNING.to_stdlib() == 30
        assert LogLevel.ERROR.to_stdlib() == 40
        assert LogLevel.CRITICAL.to_stdlib() == 50

    def test_from_stdlib(self):
        """Test creation from stdlib logging levels."""
        assert LogLevel.from_stdlib(10) == LogLevel.DEBUG
        assert LogLevel.from_stdlib(20) == LogLevel.INFO
        assert LogLevel.from_stdlib(30) == LogLevel.WARNING
        assert LogLevel.from_stdlib(40) == LogLevel.ERROR
        assert LogLevel.from_stdlib(50) == LogLevel.CRITICAL
        assert LogLevel.from_stdlib(999) == LogLevel.INFO  # Default

    def test_from_string(self):
        """Test creation from string."""
        assert LogLevel.from_string("DEBUG") == LogLevel.DEBUG
        assert LogLevel.from_string("debug") == LogLevel.DEBUG
        assert LogLevel.from_string("INFO") == LogLevel.INFO
        assert LogLevel.from_string("invalid") == LogLevel.INFO  # Default


class TestLogContextData:
    """Tests for LogContextData."""

    def test_default_values(self):
        """Test default context values."""
        context = LogContextData()
        assert context.operation is None
        assert context.platform is None
        assert context.correlation_id is None
        assert context.extra == {}

    def test_merge(self):
        """Test context merging."""
        ctx1 = LogContextData(operation="op1", platform="airflow")
        ctx2 = LogContextData(operation="op2", extra={"key": "value"})

        merged = ctx1.merge(ctx2)
        assert merged.operation == "op2"  # Override
        assert merged.platform == "airflow"  # Preserved
        assert merged.extra == {"key": "value"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = LogContextData(
            operation="validate",
            platform="dagster",
            correlation_id="abc123",
            extra={"custom": "value"},
        )
        result = context.to_dict()

        assert result["operation"] == "validate"
        assert result["platform"] == "dagster"
        assert result["correlation_id"] == "abc123"
        assert result["custom"] == "value"

    def test_to_dict_empty_fields(self):
        """Test that empty fields are excluded from dict."""
        context = LogContextData(operation="test")
        result = context.to_dict()

        assert "operation" in result
        assert "platform" not in result
        assert "correlation_id" not in result


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_basic_context(self):
        """Test basic context setting."""
        with LogContext(operation="test_op", platform="airflow"):
            ctx = get_current_context()
            assert ctx.operation == "test_op"
            assert ctx.platform == "airflow"

        # Context should be restored
        ctx = get_current_context()
        assert ctx.operation is None

    def test_nested_context(self):
        """Test nested context managers."""
        with LogContext(operation="outer", platform="airflow"):
            with LogContext(task_id="task_1"):
                ctx = get_current_context()
                assert ctx.operation == "outer"  # Inherited
                assert ctx.extra.get("task_id") == "task_1"

            # Inner context restored
            ctx = get_current_context()
            assert "task_id" not in ctx.extra

    def test_context_override(self):
        """Test context value override."""
        with LogContext(operation="outer"):
            with LogContext(operation="inner"):
                ctx = get_current_context()
                assert ctx.operation == "inner"

            ctx = get_current_context()
            assert ctx.operation == "outer"

    def test_set_context(self):
        """Test set_context function."""
        set_context(operation="set_op", custom_field="custom")
        ctx = get_current_context()
        assert ctx.operation == "set_op"
        assert ctx.extra.get("custom_field") == "custom"


class TestLogRecord:
    """Tests for LogRecord."""

    def test_basic_creation(self):
        """Test basic record creation."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
        )
        assert record.level == LogLevel.INFO
        assert record.message == "Test message"
        assert record.logger_name == "test.logger"
        assert record.exc_info is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = LogRecord(
            level=LogLevel.ERROR,
            message="Error occurred",
            logger_name="test",
            context=LogContextData(operation="validate"),
            extra={"count": 10},
        )
        result = record.to_dict()

        assert result["level"] == "ERROR"
        assert result["message"] == "Error occurred"
        assert result["logger"] == "test"
        assert result["operation"] == "validate"
        assert result["count"] == 10
        assert "timestamp" in result

    def test_to_dict_with_exception(self):
        """Test dict conversion with exception info."""
        exc = ValueError("test error")
        record = LogRecord(
            level=LogLevel.ERROR,
            message="Error",
            logger_name="test",
            exc_info=exc,
        )
        result = record.to_dict()

        assert result["exception"] == "test error"
        assert result["exception_type"] == "ValueError"


class TestSensitiveDataMasker:
    """Tests for SensitiveDataMasker."""

    def test_mask_password_in_string(self):
        """Test masking password in connection string."""
        masker = SensitiveDataMasker()
        result = masker.mask_string("password=secret123&user=admin")
        assert "secret123" not in result
        assert "***MASKED***" in result
        assert "user=admin" in result

    def test_mask_url_credentials(self):
        """Test masking credentials in URL."""
        masker = SensitiveDataMasker()
        result = masker.mask_string("postgres://user:password123@localhost/db")
        assert "password123" not in result
        assert "***MASKED***" in result

    def test_mask_api_key(self):
        """Test masking API key."""
        masker = SensitiveDataMasker()
        result = masker.mask_string("api_key=abcdef123456")
        assert "abcdef123456" not in result
        assert "***MASKED***" in result

    def test_mask_dict_sensitive_keys(self):
        """Test masking sensitive keys in dictionary."""
        masker = SensitiveDataMasker()
        data = {
            "password": "secret",
            "api_key": "key123",
            "username": "admin",
        }
        result = masker.mask_dict(data)

        assert result["password"] == "***MASKED***"
        assert result["api_key"] == "***MASKED***"
        assert result["username"] == "admin"  # Not masked

    def test_mask_nested_dict(self):
        """Test masking nested dictionaries."""
        masker = SensitiveDataMasker()
        data = {
            "config": {
                "database": {
                    "password": "secret",
                    "host": "localhost",
                },
            },
            "name": "test",
        }
        result = masker.mask_dict(data)

        assert result["config"]["database"]["password"] == "***MASKED***"
        assert result["config"]["database"]["host"] == "localhost"
        assert result["name"] == "test"

    def test_mask_list(self):
        """Test masking lists."""
        masker = SensitiveDataMasker()
        data = {"items": [{"password": "secret"}, {"name": "test"}]}
        result = masker.mask_dict(data)

        assert result["items"][0]["password"] == "***MASKED***"
        assert result["items"][1]["name"] == "test"

    def test_disabled_masker(self):
        """Test disabled masker."""
        masker = SensitiveDataMasker(enabled=False)
        result = masker.mask_string("password=secret")
        assert result == "password=secret"

    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        masker = SensitiveDataMasker()
        masker.add_pattern(r"(custom_secret=)[^\s]+", r"\1***CUSTOM***")
        result = masker.mask_string("custom_secret=myvalue")
        assert "myvalue" not in result
        assert "***CUSTOM***" in result

    def test_add_sensitive_key(self):
        """Test adding custom sensitive key."""
        masker = SensitiveDataMasker()
        masker.add_sensitive_key("my_secret_key")
        result = masker.mask_dict({"my_secret_key": "value"})
        assert result["my_secret_key"] == "***MASKED***"


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_basic_format(self):
        """Test basic formatting."""
        formatter = TextFormatter()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
        )
        result = formatter.format(record)

        assert "[INFO]" in result
        assert "test.logger:" in result
        assert "Test message" in result

    def test_format_with_context(self):
        """Test formatting with context."""
        formatter = TextFormatter(include_context=True)
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
            context=LogContextData(operation="validate"),
        )
        result = formatter.format(record)

        assert "operation=validate" in result

    def test_format_with_extra(self):
        """Test formatting with extra fields."""
        formatter = TextFormatter(include_extra=True)
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
            extra={"count": 10},
        )
        result = formatter.format(record)

        assert "count=10" in result

    def test_format_without_context(self):
        """Test formatting without context."""
        formatter = TextFormatter(include_context=False)
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
            context=LogContextData(operation="validate"),
        )
        result = formatter.format(record)

        assert "operation=" not in result


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
        )
        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test.logger"

    def test_format_with_masking(self):
        """Test JSON formatting with masking."""
        formatter = JSONFormatter()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
            extra={"password": "secret", "name": "test"},
        )
        result = formatter.format(record)
        data = json.loads(result)

        assert data["password"] == "***MASKED***"
        assert data["name"] == "test"

    def test_format_with_indent(self):
        """Test JSON formatting with indentation."""
        formatter = JSONFormatter(indent=2)
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )
        result = formatter.format(record)

        assert "\n" in result  # Indented JSON has newlines


class TestStreamHandler:
    """Tests for StreamHandler."""

    def test_handle_writes_to_stream(self):
        """Test handler writes to stream."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
        )

        handler.handle(record)
        output = stream.getvalue()

        assert "Test message" in output
        assert "[INFO]" in output

    def test_level_filtering(self):
        """Test handler respects level filter."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, level=LogLevel.WARNING)
        info_record = LogRecord(
            level=LogLevel.INFO,
            message="Info message",
            logger_name="test",
        )
        warning_record = LogRecord(
            level=LogLevel.WARNING,
            message="Warning message",
            logger_name="test",
        )

        handler.handle(info_record)
        handler.handle(warning_record)
        output = stream.getvalue()

        assert "Info message" not in output
        assert "Warning message" in output

    def test_flush(self):
        """Test flush operation."""
        stream = MagicMock()
        handler = StreamHandler(stream=stream)
        handler.flush()
        stream.flush.assert_called_once()

    def test_close(self):
        """Test close operation."""
        stream = StringIO()
        handler = StreamHandler(stream=stream)
        handler.close()
        # Should not raise, and subsequent handles should be ignored
        record = LogRecord(level=LogLevel.INFO, message="Test", logger_name="test")
        handler.handle(record)
        assert stream.getvalue() == ""


class TestBufferingHandler:
    """Tests for BufferingHandler."""

    def test_buffers_records(self):
        """Test records are buffered."""
        flushed_records: list[LogRecord] = []

        def flush_callback(records: list[LogRecord]) -> None:
            flushed_records.extend(records)

        handler = BufferingHandler(capacity=3, flush_callback=flush_callback)

        for i in range(2):
            record = LogRecord(level=LogLevel.INFO, message=f"msg{i}", logger_name="test")
            handler.handle(record)

        assert len(flushed_records) == 0  # Not flushed yet

    def test_auto_flush_at_capacity(self):
        """Test auto-flush when capacity reached."""
        flushed_records: list[LogRecord] = []

        def flush_callback(records: list[LogRecord]) -> None:
            flushed_records.extend(records)

        handler = BufferingHandler(capacity=3, flush_callback=flush_callback)

        for i in range(3):
            record = LogRecord(level=LogLevel.INFO, message=f"msg{i}", logger_name="test")
            handler.handle(record)

        assert len(flushed_records) == 3

    def test_manual_flush(self):
        """Test manual flush."""
        flushed_records: list[LogRecord] = []

        def flush_callback(records: list[LogRecord]) -> None:
            flushed_records.extend(records)

        handler = BufferingHandler(capacity=100, flush_callback=flush_callback)
        record = LogRecord(level=LogLevel.INFO, message="test", logger_name="test")
        handler.handle(record)
        handler.flush()

        assert len(flushed_records) == 1


class TestNullHandler:
    """Tests for NullHandler."""

    def test_handle_does_nothing(self):
        """Test handle is a no-op."""
        handler = NullHandler()
        record = LogRecord(level=LogLevel.INFO, message="test", logger_name="test")
        handler.handle(record)  # Should not raise

    def test_flush_does_nothing(self):
        """Test flush is a no-op."""
        handler = NullHandler()
        handler.flush()  # Should not raise

    def test_close_does_nothing(self):
        """Test close is a no-op."""
        handler = NullHandler()
        handler.close()  # Should not raise


class TestFilters:
    """Tests for log filters."""

    def test_level_filter(self):
        """Test LevelFilter."""
        filter = LevelFilter(LogLevel.WARNING)

        info_record = LogRecord(level=LogLevel.INFO, message="info", logger_name="test")
        warning_record = LogRecord(level=LogLevel.WARNING, message="warn", logger_name="test")
        error_record = LogRecord(level=LogLevel.ERROR, message="error", logger_name="test")

        assert filter.filter(info_record) is False
        assert filter.filter(warning_record) is True
        assert filter.filter(error_record) is True

    def test_context_filter(self):
        """Test ContextFilter."""
        filter = ContextFilter(platform="airflow")

        airflow_record = LogRecord(
            level=LogLevel.INFO,
            message="test",
            logger_name="test",
            context=LogContextData(platform="airflow"),
        )
        dagster_record = LogRecord(
            level=LogLevel.INFO,
            message="test",
            logger_name="test",
            context=LogContextData(platform="dagster"),
        )

        assert filter.filter(airflow_record) is True
        assert filter.filter(dagster_record) is False

    def test_regex_filter_include(self):
        """Test RegexFilter with include mode."""
        filter = RegexFilter(r"important")

        match_record = LogRecord(level=LogLevel.INFO, message="important message", logger_name="test")
        no_match_record = LogRecord(level=LogLevel.INFO, message="regular message", logger_name="test")

        assert filter.filter(match_record) is True
        assert filter.filter(no_match_record) is False

    def test_regex_filter_exclude(self):
        """Test RegexFilter with exclude mode."""
        filter = RegexFilter(r"debug", exclude=True)

        debug_record = LogRecord(level=LogLevel.INFO, message="debug info", logger_name="test")
        normal_record = LogRecord(level=LogLevel.INFO, message="normal info", logger_name="test")

        assert filter.filter(debug_record) is False
        assert filter.filter(normal_record) is True


class TestTruthoundLogger:
    """Tests for TruthoundLogger."""

    def test_basic_logging(self):
        """Test basic logging functionality."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test.logger", handlers=[handler])

        logger.info("Test message")
        output = stream.getvalue()

        assert "Test message" in output
        assert "[INFO]" in output

    def test_log_levels(self):
        """Test all log levels."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])

        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.critical("critical")

        output = stream.getvalue()
        assert "[DEBUG]" in output
        assert "[INFO]" in output
        assert "[WARNING]" in output
        assert "[ERROR]" in output
        assert "[CRITICAL]" in output

    def test_structured_logging(self):
        """Test logging with structured fields."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter(include_extra=True))
        logger = TruthoundLogger("test", handlers=[handler])

        logger.info("Processing", count=10, status="active")
        output = stream.getvalue()

        assert "count=10" in output
        assert "status=active" in output

    def test_context_propagation(self):
        """Test log context is included."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter(include_context=True))
        logger = TruthoundLogger("test", handlers=[handler])

        with LogContext(operation="validate", platform="airflow"):
            logger.info("In context")

        output = stream.getvalue()
        # Context should include both operation and platform from the LogContext
        assert "platform=airflow" in output
        assert "In context" in output

    def test_sensitive_data_masking(self):
        """Test sensitive data is masked."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter(include_extra=True))
        logger = TruthoundLogger("test", handlers=[handler])

        logger.info("Connecting", password="secret123")
        output = stream.getvalue()

        assert "secret123" not in output
        assert "***MASKED***" in output

    def test_exception_logging(self):
        """Test exception logging."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("Error occurred")

        output = stream.getvalue()
        assert "[ERROR]" in output
        assert "Error occurred" in output

    def test_level_filtering(self):
        """Test logger level filtering."""
        stream = StringIO()
        handler = StreamHandler(stream=stream)
        logger = TruthoundLogger("test", level=LogLevel.WARNING, handlers=[handler])

        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")

        output = stream.getvalue()
        assert "debug" not in output
        assert "info" not in output
        assert "warning" in output

    def test_filter_application(self):
        """Test filters are applied."""
        stream = StringIO()
        handler = StreamHandler(stream=stream)
        logger = TruthoundLogger(
            "test",
            handlers=[handler],
            filters=[RegexFilter(r"important")],
        )

        logger.info("important message")
        logger.info("regular message")

        output = stream.getvalue()
        assert "important message" in output
        assert "regular message" not in output

    def test_add_remove_handler(self):
        """Test adding and removing handlers."""
        stream = StringIO()
        handler = StreamHandler(stream=stream)
        logger = TruthoundLogger("test")

        logger.add_handler(handler)
        logger.info("test1")
        assert "test1" in stream.getvalue()

        stream.truncate(0)
        stream.seek(0)

        logger.remove_handler(handler)
        logger.info("test2")
        assert stream.getvalue() == ""

    def test_is_enabled_for(self):
        """Test is_enabled_for method."""
        logger = TruthoundLogger("test", level=LogLevel.WARNING)

        assert logger.is_enabled_for(LogLevel.DEBUG) is False
        assert logger.is_enabled_for(LogLevel.INFO) is False
        assert logger.is_enabled_for(LogLevel.WARNING) is True
        assert logger.is_enabled_for(LogLevel.ERROR) is True


class TestLoggerRegistry:
    """Tests for logger registry and get_logger."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"

    def test_get_same_logger(self):
        """Test getting the same logger returns same instance."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2

    def test_configure_logging(self):
        """Test global configuration."""
        configure_logging(level=LogLevel.WARNING, format="json")
        logger = get_logger("test.configured")
        # Logger should have WARNING level
        assert logger.level == LogLevel.WARNING


class TestPlatformAdapters:
    """Tests for platform-specific adapters."""

    def test_create_platform_handler_airflow(self):
        """Test creating Airflow handler."""
        handler = create_platform_handler("airflow")
        assert isinstance(handler, AirflowLoggerAdapter)

    def test_create_platform_handler_dagster(self):
        """Test creating Dagster handler."""
        handler = create_platform_handler("dagster")
        assert isinstance(handler, DagsterLoggerAdapter)

    def test_create_platform_handler_prefect(self):
        """Test creating Prefect handler."""
        handler = create_platform_handler("prefect")
        assert isinstance(handler, PrefectLoggerAdapter)

    def test_create_platform_handler_stdlib(self):
        """Test creating stdlib handler."""
        handler = create_platform_handler("stdlib")
        assert isinstance(handler, StdlibLoggerAdapter)

    def test_create_platform_handler_unknown(self):
        """Test unknown platform raises error."""
        with pytest.raises(ValueError, match="Unknown platform"):
            create_platform_handler("unknown")

    def test_stdlib_adapter(self):
        """Test StdlibLoggerAdapter."""
        import logging

        stdlib_logger = logging.getLogger("test.stdlib.unique")
        stdlib_logger.setLevel(logging.DEBUG)

        # Use a real StreamHandler with StringIO instead of MagicMock
        stream = StringIO()
        real_handler = logging.StreamHandler(stream)
        real_handler.setLevel(logging.DEBUG)
        stdlib_logger.addHandler(real_handler)

        adapter = StdlibLoggerAdapter(stdlib_logger)
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
        )

        adapter.handle(record)
        output = stream.getvalue()
        assert "Test message" in output

        # Cleanup
        stdlib_logger.removeHandler(real_handler)


class TestPerformanceLogger:
    """Tests for PerformanceLogger."""

    def test_timed_context_manager(self):
        """Test timed context manager."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])
        perf = PerformanceLogger(logger)

        with perf.timed("test_operation"):
            pass  # Simulate work

        output = stream.getvalue()
        assert "test_operation completed" in output
        assert "ms" in output

    def test_timed_with_metadata(self):
        """Test timed with additional metadata."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter(include_extra=True))
        logger = TruthoundLogger("test", handlers=[handler])
        perf = PerformanceLogger(logger)

        with perf.timed("batch_process", batch_size=100):
            pass

        output = stream.getvalue()
        assert "batch_process" in output
        assert "batch_size=100" in output

    def test_timed_result(self):
        """Test accessing timing result."""
        logger = TruthoundLogger("test", handlers=[NullHandler()])
        perf = PerformanceLogger(logger)

        with perf.timed("operation") as ctx:
            pass

        assert ctx.result is not None
        assert ctx.result.operation == "operation"
        assert ctx.result.duration_ms >= 0
        assert ctx.result.success is True

    def test_timed_failure(self):
        """Test timing a failed operation."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])
        perf = PerformanceLogger(logger)

        with pytest.raises(ValueError):
            with perf.timed("failing_op"):
                raise ValueError("Test error")

        output = stream.getvalue()
        assert "FAILED" in output

    def test_slow_operation_warning(self):
        """Test slow operation generates warning."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])
        perf = PerformanceLogger(logger, slow_threshold_ms=0.0)  # Very low threshold

        with perf.timed("slow_op"):
            pass  # Even empty block takes some time

        output = stream.getvalue()
        assert "SLOW" in output
        assert "[WARNING]" in output

    def test_timed_decorator(self):
        """Test timed_decorator."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])
        perf = PerformanceLogger(logger)

        @perf.timed_decorator()
        def my_function() -> str:
            return "result"

        result = my_function()

        assert result == "result"
        output = stream.getvalue()
        assert "my_function completed" in output

    def test_get_performance_logger(self):
        """Test get_performance_logger convenience function."""
        perf = get_performance_logger("test.perf")
        assert isinstance(perf, PerformanceLogger)


class TestTimingResult:
    """Tests for TimingResult."""

    def test_basic_creation(self):
        """Test basic creation."""
        result = TimingResult(
            operation="test",
            duration_ms=123.45,
            success=True,
        )
        assert result.operation == "test"
        assert result.duration_ms == 123.45
        assert result.success is True
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test creation with metadata."""
        result = TimingResult(
            operation="test",
            duration_ms=100.0,
            metadata={"key": "value"},
        )
        assert result.metadata == {"key": "value"}


class TestDecorators:
    """Tests for logging decorators."""

    def test_log_call_basic(self):
        """Test log_call decorator."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])

        @log_call(logger=logger)
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)

        assert result == 10
        output = stream.getvalue()
        assert "Calling my_func" in output

    def test_log_call_with_result(self):
        """Test log_call with result logging."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter(include_extra=True))
        logger = TruthoundLogger("test", handlers=[handler])

        @log_call(logger=logger, include_result=True)
        def my_func() -> str:
            return "hello"

        my_func()

        output = stream.getvalue()
        assert "returned" in output
        assert "hello" in output

    def test_log_call_masks_sensitive_args(self):
        """Test log_call masks sensitive kwargs."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter(include_extra=True))
        logger = TruthoundLogger("test", handlers=[handler])

        @log_call(logger=logger)
        def connect(host: str, password: str) -> None:
            pass

        connect(host="localhost", password="secret")

        output = stream.getvalue()
        assert "secret" not in output
        assert "***MASKED***" in output

    def test_log_errors_decorator(self):
        """Test log_errors decorator."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])

        @log_errors(logger=logger)
        def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

        output = stream.getvalue()
        assert "[ERROR]" in output
        assert "Exception in failing_func" in output

    def test_log_errors_no_reraise(self):
        """Test log_errors without reraising."""
        stream = StringIO()
        handler = StreamHandler(stream=stream, formatter=TextFormatter())
        logger = TruthoundLogger("test", handlers=[handler])

        @log_errors(logger=logger, reraise=False)
        def failing_func() -> str:
            raise ValueError("Test error")

        result = failing_func()  # Should not raise
        assert result is None

        output = stream.getvalue()
        assert "[ERROR]" in output


class TestMaskerConfiguration:
    """Tests for masker configuration."""

    def test_get_masker(self):
        """Test getting default masker."""
        masker = get_masker()
        assert isinstance(masker, SensitiveDataMasker)

    def test_configure_masker(self):
        """Test configuring default masker."""
        custom_patterns = ((r"(test_secret=)[^\s]+", r"\1***TEST***"),)
        configure_masker(patterns=custom_patterns, enabled=True)

        masker = get_masker()
        result = masker.mask_string("test_secret=myvalue")
        assert "***TEST***" in result

        # Reset to defaults
        configure_masker()
