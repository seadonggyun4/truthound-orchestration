"""Logging utilities for Truthound Integrations.

This module provides a flexible, extensible logging system designed for
enterprise environments. It supports:

- Structured logging with context propagation
- Platform-specific adapters (Airflow, Dagster, Prefect)
- Sensitive data masking
- Performance timing utilities
- Log level filtering and routing

Design Principles:
    1. Protocol-based: Easy to extend with custom implementations
    2. Context-aware: Automatic context propagation across operations
    3. Platform-agnostic: Works with any logging backend
    4. Safe by default: Automatic masking of sensitive data

Example:
    >>> from common.logging import get_logger, LogContext
    >>> logger = get_logger(__name__)
    >>> with LogContext(operation="check", platform="airflow"):
    ...     logger.info("Starting validation", rules_count=10)
"""

from __future__ import annotations

import functools
import logging
import re
import sys
import time
from abc import abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    Self,
    runtime_checkable,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


# =============================================================================
# Constants and Configuration
# =============================================================================


class LogLevel(Enum):
    """Log severity levels with numeric values for comparison.

    Attributes:
        DEBUG: Detailed debugging information.
        INFO: General operational information.
        WARNING: Warning conditions that should be reviewed.
        ERROR: Error conditions that need attention.
        CRITICAL: Critical conditions requiring immediate action.
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    def to_stdlib(self) -> int:
        """Convert to stdlib logging level."""
        return self.value

    @classmethod
    def from_stdlib(cls, level: int) -> LogLevel:
        """Create from stdlib logging level."""
        for log_level in cls:
            if log_level.value == level:
                return log_level
        return cls.INFO

    @classmethod
    def from_string(cls, level: str) -> LogLevel:
        """Create from string representation."""
        try:
            return cls[level.upper()]
        except KeyError:
            return cls.INFO


# Default patterns for sensitive data masking
DEFAULT_SENSITIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Connection strings
    (r"(password=)[^&\s;]+", r"\1***MASKED***"),
    (r"(pwd=)[^&\s;]+", r"\1***MASKED***"),
    (r"(secret=)[^&\s;]+", r"\1***MASKED***"),
    (r"(api_key=)[^&\s;]+", r"\1***MASKED***"),
    (r"(apikey=)[^&\s;]+", r"\1***MASKED***"),
    (r"(token=)[^&\s;]+", r"\1***MASKED***"),
    (r"(access_token=)[^&\s;]+", r"\1***MASKED***"),
    # URLs with credentials
    (r"(://[^:]+:)[^@]+(@)", r"\1***MASKED***\2"),
    # AWS credentials
    (r"(AKIA[A-Z0-9]{16})", r"***AWS_KEY_MASKED***"),
    (r"([A-Za-z0-9/+=]{40})", r"***MASKED***"),  # AWS secret key pattern
    # Generic patterns
    (r'("password"\s*:\s*")[^"]+(")', r"\1***MASKED***\2"),
    (r"('password'\s*:\s*')[^']+(')", r"\1***MASKED***\2"),
)

# Keys that should be masked in structured data
DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
    "secret_key",
    "auth",
    "authorization",
    "credentials",
    "connection_string",
})


# =============================================================================
# Context Management
# =============================================================================


@dataclass(frozen=True, slots=True)
class LogContextData:
    """Immutable container for log context data.

    Attributes:
        operation: Current operation name.
        platform: Platform identifier.
        correlation_id: Request/transaction correlation ID.
        extra: Additional context fields.
    """

    operation: str | None = None
    platform: str | None = None
    correlation_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def merge(self, other: LogContextData) -> LogContextData:
        """Create a new context with merged data.

        Args:
            other: Context to merge with (takes precedence).

        Returns:
            New LogContextData with merged values.
        """
        return LogContextData(
            operation=other.operation or self.operation,
            platform=other.platform or self.platform,
            correlation_id=other.correlation_id or self.correlation_id,
            extra={**self.extra, **other.extra},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result: dict[str, Any] = {}
        if self.operation:
            result["operation"] = self.operation
        if self.platform:
            result["platform"] = self.platform
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        result.update(self.extra)
        return result


# Context variable for propagating log context
_log_context: ContextVar[LogContextData] = ContextVar("log_context")
_log_context.set(LogContextData())


class LogContext:
    """Context manager for log context propagation.

    Automatically adds context fields to all log messages within the scope.
    Supports nesting with context merging.

    Example:
        >>> with LogContext(operation="validate", platform="airflow"):
        ...     logger.info("Starting")  # Includes operation and platform
        ...     with LogContext(task_id="task_1"):
        ...         logger.info("Processing")  # Includes all three
    """

    def __init__(
        self,
        *,
        operation: str | None = None,
        platform: str | None = None,
        correlation_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Initialize log context.

        Args:
            operation: Operation name.
            platform: Platform identifier.
            correlation_id: Correlation ID for tracing.
            **extra: Additional context fields.
        """
        self._new_context = LogContextData(
            operation=operation,
            platform=platform,
            correlation_id=correlation_id,
            extra=extra,
        )
        self._token: Any = None

    def __enter__(self) -> Self:
        """Enter context and set new context data."""
        current = _log_context.get()
        merged = current.merge(self._new_context)
        self._token = _log_context.set(merged)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous context."""
        if self._token is not None:
            _log_context.reset(self._token)


def get_current_context() -> LogContextData:
    """Get the current log context.

    Returns:
        Current LogContextData or empty context if none set.
    """
    return _log_context.get()


def set_context(**kwargs: Any) -> None:
    """Set context fields for the current context.

    This modifies the current context in place. For scoped context,
    use LogContext context manager instead.

    Args:
        **kwargs: Context fields to set.
    """
    current = _log_context.get()
    new_extra = {**current.extra, **kwargs}
    new_context = LogContextData(
        operation=kwargs.get("operation", current.operation),
        platform=kwargs.get("platform", current.platform),
        correlation_id=kwargs.get("correlation_id", current.correlation_id),
        extra=new_extra,
    )
    _log_context.set(new_context)


# =============================================================================
# Log Record
# =============================================================================


@dataclass(slots=True)
class LogRecord:
    """Structured log record.

    Attributes:
        level: Log severity level.
        message: Log message.
        logger_name: Name of the logger.
        timestamp: When the log was created.
        context: Associated context data.
        extra: Additional structured fields.
        exc_info: Exception information if any.
    """

    level: LogLevel
    message: str
    logger_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: LogContextData = field(default_factory=LogContextData)
    extra: dict[str, Any] = field(default_factory=dict)
    exc_info: BaseException | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "level": self.level.name,
            "message": self.message,
            "logger": self.logger_name,
            "timestamp": self.timestamp.isoformat(),
            **self.context.to_dict(),
            **self.extra,
        }
        if self.exc_info:
            result["exception"] = str(self.exc_info)
            result["exception_type"] = type(self.exc_info).__name__
        return result


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class LogHandler(Protocol):
    """Protocol for log handlers.

    Handlers receive log records and process them (write to file,
    send to service, etc.).
    """

    @abstractmethod
    def handle(self, record: LogRecord) -> None:
        """Handle a log record.

        Args:
            record: Log record to process.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered output."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the handler and release resources."""
        ...


@runtime_checkable
class LogFormatter(Protocol):
    """Protocol for log formatters.

    Formatters convert LogRecord to string representation.
    """

    @abstractmethod
    def format(self, record: LogRecord) -> str:
        """Format a log record.

        Args:
            record: Log record to format.

        Returns:
            Formatted string representation.
        """
        ...


@runtime_checkable
class LogFilter(Protocol):
    """Protocol for log filters.

    Filters determine whether a log record should be processed.
    """

    @abstractmethod
    def filter(self, record: LogRecord) -> bool:
        """Determine if record should be logged.

        Args:
            record: Log record to evaluate.

        Returns:
            True if record should be logged, False otherwise.
        """
        ...


# =============================================================================
# Sensitive Data Masking
# =============================================================================


class SensitiveDataMasker:
    """Masks sensitive data in log messages and structured data.

    Thread-safe and configurable with custom patterns and keys.

    Example:
        >>> masker = SensitiveDataMasker()
        >>> masker.mask_string("password=secret123")
        'password=***MASKED***'
        >>> masker.mask_dict({"password": "secret", "name": "test"})
        {'password': '***MASKED***', 'name': 'test'}
    """

    MASK_VALUE: ClassVar[str] = "***MASKED***"

    def __init__(
        self,
        patterns: tuple[tuple[str, str], ...] | None = None,
        sensitive_keys: frozenset[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize the masker.

        Args:
            patterns: Tuple of (pattern, replacement) for regex masking.
            sensitive_keys: Keys to mask in dictionaries.
            enabled: Whether masking is enabled.
        """
        self.enabled = enabled
        self._patterns = patterns or DEFAULT_SENSITIVE_PATTERNS
        self._sensitive_keys = sensitive_keys or DEFAULT_SENSITIVE_KEYS
        self._compiled_patterns = tuple(
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self._patterns
        )

    def mask_string(self, value: str) -> str:
        """Mask sensitive data in a string.

        Args:
            value: String to mask.

        Returns:
            Masked string.
        """
        if not self.enabled or not value:
            return value

        result = value
        for pattern, replacement in self._compiled_patterns:
            result = pattern.sub(replacement, result)
        return result

    def mask_value(self, value: Any) -> Any:
        """Mask a single value based on type.

        Args:
            value: Value to mask.

        Returns:
            Masked value.
        """
        if not self.enabled:
            return value

        if isinstance(value, str):
            return self.mask_string(value)
        elif isinstance(value, dict):
            return self.mask_dict(value)
        elif isinstance(value, (list, tuple)):
            return type(value)(self.mask_value(v) for v in value)
        return value

    def mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive data in a dictionary.

        Args:
            data: Dictionary to mask.

        Returns:
            Dictionary with sensitive values masked.
        """
        if not self.enabled:
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            lower_key = key.lower()
            if lower_key in self._sensitive_keys:
                result[key] = self.MASK_VALUE
            elif isinstance(value, dict):
                # Recursively mask nested dictionaries
                result[key] = self.mask_dict(value)
            else:
                result[key] = self.mask_value(value)
        return result

    def add_pattern(self, pattern: str, replacement: str) -> None:
        """Add a custom masking pattern.

        Args:
            pattern: Regex pattern to match.
            replacement: Replacement string.
        """
        self._patterns = (*self._patterns, (pattern, replacement))
        self._compiled_patterns = (
            *self._compiled_patterns,
            (re.compile(pattern, re.IGNORECASE), replacement),
        )

    def add_sensitive_key(self, key: str) -> None:
        """Add a key to be masked in dictionaries.

        Args:
            key: Key name to mask.
        """
        self._sensitive_keys = self._sensitive_keys | {key.lower()}


# Global masker instance
_default_masker = SensitiveDataMasker()


def get_masker() -> SensitiveDataMasker:
    """Get the default sensitive data masker."""
    return _default_masker


def configure_masker(
    *,
    patterns: tuple[tuple[str, str], ...] | None = None,
    sensitive_keys: frozenset[str] | None = None,
    enabled: bool = True,
) -> None:
    """Configure the default sensitive data masker.

    Args:
        patterns: Custom regex patterns for masking.
        sensitive_keys: Custom keys to mask.
        enabled: Whether masking is enabled.
    """
    global _default_masker
    _default_masker = SensitiveDataMasker(
        patterns=patterns,
        sensitive_keys=sensitive_keys,
        enabled=enabled,
    )


# =============================================================================
# Formatters
# =============================================================================


class TextFormatter:
    """Plain text log formatter.

    Formats log records as human-readable text.

    Example output:
        2024-01-15T10:30:45.123456+00:00 [INFO] common.base: Message here
    """

    def __init__(
        self,
        include_context: bool = True,
        include_extra: bool = True,
        timestamp_format: str | None = None,
    ) -> None:
        """Initialize the formatter.

        Args:
            include_context: Whether to include context in output.
            include_extra: Whether to include extra fields.
            timestamp_format: Custom timestamp format (None for ISO).
        """
        self.include_context = include_context
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

    def format(self, record: LogRecord) -> str:
        """Format log record as text.

        Args:
            record: Log record to format.

        Returns:
            Formatted text string.
        """
        if self.timestamp_format:
            timestamp = record.timestamp.strftime(self.timestamp_format)
        else:
            timestamp = record.timestamp.isoformat()

        parts = [
            f"{timestamp}",
            f"[{record.level.name}]",
            f"{record.logger_name}:",
            record.message,
        ]

        # Add context
        if self.include_context:
            context_dict = record.context.to_dict()
            if context_dict:
                context_str = " ".join(f"{k}={v}" for k, v in context_dict.items())
                parts.append(f"| {context_str}")

        # Add extra fields
        if self.include_extra and record.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            parts.append(f"| {extra_str}")

        # Add exception info
        if record.exc_info:
            parts.append(f"| exception={record.exc_info!r}")

        return " ".join(parts)


class JSONFormatter:
    """JSON log formatter.

    Formats log records as JSON for structured logging systems.

    Example output:
        {"level": "INFO", "message": "...", "timestamp": "...", ...}
    """

    def __init__(
        self,
        masker: SensitiveDataMasker | None = None,
        indent: int | None = None,
    ) -> None:
        """Initialize the formatter.

        Args:
            masker: Sensitive data masker to use.
            indent: JSON indentation (None for compact).
        """
        self._masker = masker or _default_masker
        self._indent = indent

    def format(self, record: LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string.
        """
        import json

        data = record.to_dict()
        masked_data = self._masker.mask_dict(data)
        return json.dumps(masked_data, indent=self._indent, default=str)


# =============================================================================
# Handlers
# =============================================================================


class StreamHandler:
    """Handler that writes to a stream (stdout/stderr).

    Thread-safe stream handler with buffering support.
    """

    def __init__(
        self,
        stream: Any = None,
        formatter: LogFormatter | None = None,
        level: LogLevel = LogLevel.DEBUG,
    ) -> None:
        """Initialize the handler.

        Args:
            stream: Output stream (default: sys.stderr).
            formatter: Log formatter to use.
            level: Minimum log level to handle.
        """
        self._stream = stream or sys.stderr
        self._formatter = formatter or TextFormatter()
        self._level = level
        self._closed = False

    def handle(self, record: LogRecord) -> None:
        """Handle a log record.

        Args:
            record: Log record to process.
        """
        if self._closed or record.level.value < self._level.value:
            return

        try:
            message = self._formatter.format(record)
            self._stream.write(message + "\n")
        except Exception:
            # Fail silently to avoid logging loops
            pass

    def flush(self) -> None:
        """Flush the stream."""
        if not self._closed and hasattr(self._stream, "flush"):
            try:
                self._stream.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close the handler."""
        self.flush()
        self._closed = True


class BufferingHandler:
    """Handler that buffers log records before flushing.

    Useful for batching logs to external services.
    """

    def __init__(
        self,
        capacity: int = 100,
        flush_callback: Callable[[list[LogRecord]], None] | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            capacity: Maximum buffer size before auto-flush.
            flush_callback: Callback to process buffered records.
        """
        self._capacity = capacity
        self._flush_callback = flush_callback
        self._buffer: list[LogRecord] = []
        self._closed = False

    def handle(self, record: LogRecord) -> None:
        """Handle a log record by buffering it.

        Args:
            record: Log record to buffer.
        """
        if self._closed:
            return

        self._buffer.append(record)
        if len(self._buffer) >= self._capacity:
            self.flush()

    def flush(self) -> None:
        """Flush buffered records."""
        if self._buffer and self._flush_callback:
            try:
                self._flush_callback(list(self._buffer))
            except Exception:
                pass
        self._buffer.clear()

    def close(self) -> None:
        """Close the handler."""
        self.flush()
        self._closed = True


class NullHandler:
    """Handler that discards all records.

    Useful as a default handler to prevent 'no handler' warnings.
    """

    def handle(self, record: LogRecord) -> None:
        """Discard the record."""
        pass

    def flush(self) -> None:
        """No-op flush."""
        pass

    def close(self) -> None:
        """No-op close."""
        pass


# =============================================================================
# Filters
# =============================================================================


class LevelFilter:
    """Filter by log level.

    Only allows records at or above the specified level.
    """

    def __init__(self, min_level: LogLevel) -> None:
        """Initialize the filter.

        Args:
            min_level: Minimum level to allow.
        """
        self._min_level = min_level

    def filter(self, record: LogRecord) -> bool:
        """Filter by level.

        Args:
            record: Log record to evaluate.

        Returns:
            True if record level >= min_level.
        """
        return record.level.value >= self._min_level.value


class ContextFilter:
    """Filter by context field values.

    Only allows records matching specified context criteria.
    """

    def __init__(self, **criteria: Any) -> None:
        """Initialize the filter.

        Args:
            **criteria: Context field/value pairs to match.
        """
        self._criteria = criteria

    def filter(self, record: LogRecord) -> bool:
        """Filter by context.

        Args:
            record: Log record to evaluate.

        Returns:
            True if record matches all criteria.
        """
        context_dict = record.context.to_dict()
        for key, value in self._criteria.items():
            if context_dict.get(key) != value:
                return False
        return True


class RegexFilter:
    """Filter by message pattern.

    Only allows records whose message matches the pattern.
    """

    def __init__(self, pattern: str, exclude: bool = False) -> None:
        """Initialize the filter.

        Args:
            pattern: Regex pattern to match.
            exclude: If True, exclude matching records.
        """
        self._pattern = re.compile(pattern)
        self._exclude = exclude

    def filter(self, record: LogRecord) -> bool:
        """Filter by message pattern.

        Args:
            record: Log record to evaluate.

        Returns:
            True if record should be logged.
        """
        matches = bool(self._pattern.search(record.message))
        return not matches if self._exclude else matches


# =============================================================================
# Logger Implementation
# =============================================================================


class TruthoundLogger:
    """Main logger class for Truthound integrations.

    Provides structured logging with context propagation, filtering,
    and sensitive data masking.

    Example:
        >>> logger = TruthoundLogger("my.module")
        >>> logger.info("Processing data", rows=1000)
        >>> with LogContext(operation="validate"):
        ...     logger.warning("Validation issue", column="email")
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.DEBUG,
        handlers: list[LogHandler] | None = None,
        filters: list[LogFilter] | None = None,
        masker: SensitiveDataMasker | None = None,
        propagate: bool = True,
    ) -> None:
        """Initialize the logger.

        Args:
            name: Logger name (typically __name__).
            level: Minimum log level.
            handlers: Log handlers.
            filters: Log filters.
            masker: Sensitive data masker.
            propagate: Whether to propagate to parent loggers.
        """
        self.name = name
        self.level = level
        self._handlers: list[LogHandler] = handlers or []
        self._filters: list[LogFilter] = filters or []
        self._masker = masker or _default_masker
        self._propagate = propagate
        self._parent: TruthoundLogger | None = None
        self._disabled = False

    def add_handler(self, handler: LogHandler) -> None:
        """Add a handler to the logger.

        Args:
            handler: Handler to add.
        """
        if handler not in self._handlers:
            self._handlers.append(handler)

    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a handler from the logger.

        Args:
            handler: Handler to remove.
        """
        if handler in self._handlers:
            self._handlers.remove(handler)

    def add_filter(self, log_filter: LogFilter) -> None:
        """Add a filter to the logger.

        Args:
            log_filter: Filter to add.
        """
        if log_filter not in self._filters:
            self._filters.append(log_filter)

    def remove_filter(self, log_filter: LogFilter) -> None:
        """Remove a filter from the logger.

        Args:
            log_filter: Filter to remove.
        """
        if log_filter in self._filters:
            self._filters.remove(log_filter)

    def is_enabled_for(self, level: LogLevel) -> bool:
        """Check if logger is enabled for the given level.

        Args:
            level: Level to check.

        Returns:
            True if logging is enabled for level.
        """
        return not self._disabled and level.value >= self.level.value

    def _should_log(self, record: LogRecord) -> bool:
        """Check if record passes all filters.

        Args:
            record: Record to check.

        Returns:
            True if record should be logged.
        """
        for log_filter in self._filters:
            if not log_filter.filter(record):
                return False
        return True

    def _log(
        self,
        level: LogLevel,
        message: str,
        exc_info: BaseException | None = None,
        **kwargs: Any,
    ) -> None:
        """Internal logging method.

        Args:
            level: Log level.
            message: Log message.
            exc_info: Exception information.
            **kwargs: Additional structured fields.
        """
        if not self.is_enabled_for(level):
            return

        # Create record with current context
        context = get_current_context()
        masked_kwargs = self._masker.mask_dict(kwargs)
        masked_message = self._masker.mask_string(message)

        record = LogRecord(
            level=level,
            message=masked_message,
            logger_name=self.name,
            context=context,
            extra=masked_kwargs,
            exc_info=exc_info,
        )

        # Check filters
        if not self._should_log(record):
            return

        # Handle record
        for handler in self._handlers:
            try:
                handler.handle(record)
            except Exception:
                # Fail silently to avoid logging loops
                pass

        # Propagate to parent
        if self._propagate and self._parent:
            self._parent._handle_child_record(record)

    def _handle_child_record(self, record: LogRecord) -> None:
        """Handle a record propagated from a child logger.

        Args:
            record: Record from child logger.
        """
        if self._disabled:
            return

        if not self._should_log(record):
            return

        for handler in self._handlers:
            try:
                handler.handle(record)
            except Exception:
                pass

        if self._propagate and self._parent:
            self._parent._handle_child_record(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: BaseException | None = None, **kwargs: Any) -> None:
        """Log at ERROR level.

        Args:
            message: Log message.
            exc_info: Optional exception information.
            **kwargs: Additional structured fields.
        """
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: BaseException | None = None, **kwargs: Any) -> None:
        """Log at CRITICAL level.

        Args:
            message: Log message.
            exc_info: Optional exception information.
            **kwargs: Additional structured fields.
        """
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception at ERROR level.

        Automatically captures the current exception info.

        Args:
            message: Log message.
            **kwargs: Additional structured fields.
        """
        exc_info = sys.exc_info()[1]
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Log at arbitrary level.

        Args:
            level: Log level.
            message: Log message.
            **kwargs: Additional structured fields.
        """
        self._log(level, message, **kwargs)


# =============================================================================
# Logger Registry
# =============================================================================


class LoggerRegistry:
    """Registry for managing loggers.

    Provides singleton-like logger access with hierarchical naming.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._loggers: dict[str, TruthoundLogger] = {}
        self._root_handlers: list[LogHandler] = []
        self._root_level: LogLevel = LogLevel.INFO
        self._configured = False

    def get_logger(
        self,
        name: str,
        level: LogLevel | None = None,
    ) -> TruthoundLogger:
        """Get or create a logger by name.

        Args:
            name: Logger name.
            level: Optional level override.

        Returns:
            TruthoundLogger instance.
        """
        if name in self._loggers:
            return self._loggers[name]

        # Create new logger
        logger = TruthoundLogger(
            name=name,
            level=level or self._root_level,
            handlers=list(self._root_handlers) if not self._loggers else [],
        )

        # Set parent for hierarchical logging
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            if parent_name in self._loggers:
                logger._parent = self._loggers[parent_name]

        self._loggers[name] = logger
        return logger

    def configure(
        self,
        level: LogLevel = LogLevel.INFO,
        handlers: list[LogHandler] | None = None,
        format: str = "text",
    ) -> None:
        """Configure the root logging settings.

        Args:
            level: Default log level.
            handlers: Default handlers.
            format: Format type ('text' or 'json').
        """
        self._root_level = level

        if handlers:
            self._root_handlers = handlers
        elif not self._configured:
            # Set up default handler
            formatter: LogFormatter
            if format == "json":
                formatter = JSONFormatter()
            else:
                formatter = TextFormatter()

            self._root_handlers = [StreamHandler(formatter=formatter, level=level)]

        self._configured = True

        # Update existing loggers
        for logger in self._loggers.values():
            logger.level = level
            if not logger._handlers:
                logger._handlers = list(self._root_handlers)

    def disable(self) -> None:
        """Disable all logging."""
        for logger in self._loggers.values():
            logger._disabled = True

    def enable(self) -> None:
        """Enable all logging."""
        for logger in self._loggers.values():
            logger._disabled = False


# Global registry
_registry = LoggerRegistry()


def get_logger(name: str, level: LogLevel | None = None) -> TruthoundLogger:
    """Get a logger by name.

    This is the main entry point for obtaining loggers.

    Args:
        name: Logger name (typically __name__).
        level: Optional level override.

    Returns:
        TruthoundLogger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return _registry.get_logger(name, level)


def configure_logging(
    level: LogLevel | str = LogLevel.INFO,
    handlers: list[LogHandler] | None = None,
    format: str = "text",
) -> None:
    """Configure global logging settings.

    Args:
        level: Default log level (LogLevel or string).
        handlers: Default handlers.
        format: Format type ('text' or 'json').

    Example:
        >>> configure_logging(level="DEBUG", format="json")
    """
    if isinstance(level, str):
        level = LogLevel.from_string(level)
    _registry.configure(level=level, handlers=handlers, format=format)


# =============================================================================
# Platform Adapters
# =============================================================================


class StdlibLoggerAdapter:
    """Adapter to use stdlib logging as backend.

    Bridges TruthoundLogger to Python's standard logging module.
    """

    def __init__(
        self,
        stdlib_logger: logging.Logger | None = None,
        masker: SensitiveDataMasker | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            stdlib_logger: Python logger to use.
            masker: Sensitive data masker.
        """
        self._logger = stdlib_logger or logging.getLogger()
        self._masker = masker or _default_masker

    def handle(self, record: LogRecord) -> None:
        """Handle a log record by forwarding to stdlib logger.

        Args:
            record: Log record to forward.
        """
        # Convert to stdlib format
        stdlib_level = record.level.to_stdlib()

        # Build message with context
        context_dict = record.context.to_dict()
        masked_extra = self._masker.mask_dict({**context_dict, **record.extra})

        extra_str = " ".join(f"{k}={v}" for k, v in masked_extra.items())
        full_message = record.message
        if extra_str:
            full_message = f"{record.message} | {extra_str}"

        self._logger.log(stdlib_level, full_message, exc_info=record.exc_info)

    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self._logger.handlers:
            handler.flush()

    def close(self) -> None:
        """No-op close."""
        pass


class AirflowLoggerAdapter:
    """Adapter for Apache Airflow logging.

    Integrates with Airflow's task logging system.
    """

    def __init__(self, task_instance: Any = None) -> None:
        """Initialize the adapter.

        Args:
            task_instance: Airflow TaskInstance for context.
        """
        self._task_instance = task_instance
        self._logger = logging.getLogger("airflow.task")
        self._masker = _default_masker

    def handle(self, record: LogRecord) -> None:
        """Handle a log record for Airflow.

        Args:
            record: Log record to process.
        """
        # Add Airflow-specific context
        extra: dict[str, Any] = dict(record.extra)
        if self._task_instance:
            extra["task_id"] = getattr(self._task_instance, "task_id", None)
            extra["dag_id"] = getattr(self._task_instance, "dag_id", None)
            extra["run_id"] = getattr(self._task_instance, "run_id", None)

        # Build structured message
        context_dict = record.context.to_dict()
        all_extra = self._masker.mask_dict({**context_dict, **extra})

        extra_str = " ".join(f"{k}={v}" for k, v in all_extra.items())
        message = f"[Truthound] {record.message}"
        if extra_str:
            message = f"{message} | {extra_str}"

        self._logger.log(record.level.to_stdlib(), message)

    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self._logger.handlers:
            handler.flush()

    def close(self) -> None:
        """No-op close."""
        pass


class DagsterLoggerAdapter:
    """Adapter for Dagster logging.

    Integrates with Dagster's op/asset context logging.
    """

    def __init__(self, dagster_context: Any = None) -> None:
        """Initialize the adapter.

        Args:
            dagster_context: Dagster OpExecutionContext.
        """
        self._context = dagster_context
        self._masker = _default_masker

    def handle(self, record: LogRecord) -> None:
        """Handle a log record for Dagster.

        Args:
            record: Log record to process.
        """
        if self._context is None:
            return

        # Build structured message
        context_dict = record.context.to_dict()
        masked_extra = self._masker.mask_dict({**context_dict, **record.extra})

        extra_str = " ".join(f"{k}={v}" for k, v in masked_extra.items())
        message = record.message
        if extra_str:
            message = f"{message} | {extra_str}"

        # Use Dagster's context logger
        log = getattr(self._context, "log", None)
        if log:
            level_name = record.level.name.lower()
            log_method = getattr(log, level_name, log.info)
            log_method(message)

    def flush(self) -> None:
        """No-op flush."""
        pass

    def close(self) -> None:
        """No-op close."""
        pass


class PrefectLoggerAdapter:
    """Adapter for Prefect logging.

    Integrates with Prefect's flow/task logging.
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._masker = _default_masker

    def handle(self, record: LogRecord) -> None:
        """Handle a log record for Prefect.

        Args:
            record: Log record to process.
        """
        try:
            from prefect import get_run_logger
            logger = get_run_logger()
        except Exception:
            # Fall back to stdlib if Prefect not available
            logger = logging.getLogger("prefect")

        # Build structured message
        context_dict = record.context.to_dict()
        masked_extra = self._masker.mask_dict({**context_dict, **record.extra})

        extra_str = " ".join(f"{k}={v}" for k, v in masked_extra.items())
        message = record.message
        if extra_str:
            message = f"{message} | {extra_str}"

        level_name = record.level.name.lower()
        log_method = getattr(logger, level_name, logger.info)
        log_method(message)

    def flush(self) -> None:
        """No-op flush."""
        pass

    def close(self) -> None:
        """No-op close."""
        pass


def create_platform_handler(platform: str, context: Any = None) -> LogHandler:
    """Create a platform-specific log handler.

    Args:
        platform: Platform name ('airflow', 'dagster', 'prefect').
        context: Platform-specific context object.

    Returns:
        LogHandler for the platform.

    Example:
        >>> handler = create_platform_handler("airflow", task_instance)
        >>> logger.add_handler(handler)
    """
    handlers: dict[str, type] = {
        "airflow": AirflowLoggerAdapter,
        "dagster": DagsterLoggerAdapter,
        "prefect": PrefectLoggerAdapter,
        "stdlib": StdlibLoggerAdapter,
    }

    handler_class = handlers.get(platform.lower())
    if handler_class is None:
        raise ValueError(f"Unknown platform: {platform}")

    if context is not None:
        return handler_class(context)  # type: ignore
    return handler_class()  # type: ignore


# =============================================================================
# Performance Logging
# =============================================================================


@dataclass(slots=True)
class TimingResult:
    """Result of a timed operation.

    Attributes:
        operation: Name of the operation.
        duration_ms: Duration in milliseconds.
        success: Whether operation succeeded.
        metadata: Additional timing metadata.
    """

    operation: str
    duration_ms: float
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceLogger:
    """Logger for performance timing.

    Provides decorators and context managers for timing operations.

    Example:
        >>> perf_logger = PerformanceLogger(get_logger(__name__))
        >>> with perf_logger.timed("database_query"):
        ...     result = execute_query()
        >>> # Logs: "database_query completed in 123.45ms"
    """

    def __init__(
        self,
        logger: TruthoundLogger,
        log_level: LogLevel = LogLevel.DEBUG,
        slow_threshold_ms: float = 1000.0,
    ) -> None:
        """Initialize the performance logger.

        Args:
            logger: Logger to use.
            log_level: Level for timing logs.
            slow_threshold_ms: Threshold for slow operation warnings.
        """
        self._logger = logger
        self._log_level = log_level
        self._slow_threshold_ms = slow_threshold_ms

    class _TimedContext:
        """Context manager for timed operations."""

        def __init__(
            self,
            perf_logger: PerformanceLogger,
            operation: str,
            **metadata: Any,
        ) -> None:
            self._perf_logger = perf_logger
            self._operation = operation
            self._metadata = metadata
            self._start_time: float = 0.0
            self._result: TimingResult | None = None

        def __enter__(self) -> PerformanceLogger._TimedContext:
            self._start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            success = exc_type is None

            self._result = TimingResult(
                operation=self._operation,
                duration_ms=duration_ms,
                success=success,
                metadata=self._metadata,
            )

            self._perf_logger._log_timing(self._result)

        @property
        def result(self) -> TimingResult | None:
            """Get the timing result after context exits."""
            return self._result

    def timed(self, operation: str, **metadata: Any) -> _TimedContext:
        """Create a timing context manager.

        Args:
            operation: Name of the operation.
            **metadata: Additional metadata to log.

        Returns:
            Context manager that times the operation.

        Example:
            >>> with perf_logger.timed("process_batch", batch_size=100):
            ...     process_batch(data)
        """
        return self._TimedContext(self, operation, **metadata)

    def timed_decorator(
        self,
        operation: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Create a decorator for timing functions.

        Args:
            operation: Operation name (default: function name).

        Returns:
            Decorator function.

        Example:
            >>> @perf_logger.timed_decorator()
            ... def process_data(data):
            ...     return transform(data)
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            op_name = operation or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.timed(op_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    async def timed_async(
        self,
        operation: str,
        **metadata: Any,
    ) -> _TimedContext:
        """Create an async timing context manager.

        Same as timed() but for async operations.

        Args:
            operation: Name of the operation.
            **metadata: Additional metadata.

        Returns:
            Context manager for async timing.
        """
        return self._TimedContext(self, operation, **metadata)

    def _log_timing(self, result: TimingResult) -> None:
        """Log a timing result.

        Args:
            result: Timing result to log.
        """
        level = self._log_level
        message = f"{result.operation} completed in {result.duration_ms:.2f}ms"

        # Upgrade to WARNING for slow operations
        if result.duration_ms > self._slow_threshold_ms:
            level = LogLevel.WARNING
            message = f"{result.operation} SLOW: {result.duration_ms:.2f}ms (threshold: {self._slow_threshold_ms}ms)"

        # Upgrade to ERROR for failures
        if not result.success:
            level = LogLevel.ERROR
            message = f"{result.operation} FAILED after {result.duration_ms:.2f}ms"

        self._logger.log(
            level,
            message,
            duration_ms=result.duration_ms,
            success=result.success,
            **result.metadata,
        )


def get_performance_logger(
    name: str,
    slow_threshold_ms: float = 1000.0,
) -> PerformanceLogger:
    """Get a performance logger.

    Args:
        name: Logger name.
        slow_threshold_ms: Threshold for slow operation warnings.

    Returns:
        PerformanceLogger instance.

    Example:
        >>> perf = get_performance_logger(__name__)
        >>> with perf.timed("validation"):
        ...     validate_data(df)
    """
    logger = get_logger(name)
    return PerformanceLogger(logger, slow_threshold_ms=slow_threshold_ms)


# =============================================================================
# Convenience Decorators
# =============================================================================


def log_call(
    logger: TruthoundLogger | None = None,
    level: LogLevel = LogLevel.DEBUG,
    include_args: bool = True,
    include_result: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log function calls.

    Args:
        logger: Logger to use (default: auto-detect).
        level: Log level.
        include_args: Whether to log arguments.
        include_result: Whether to log return value.

    Returns:
        Decorator function.

    Example:
        >>> @log_call(include_result=True)
        ... def process(data):
        ...     return len(data)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            extra: dict[str, Any] = {"function": func.__name__}

            if include_args:
                # Mask kwargs that might contain sensitive data
                masked_kwargs = _default_masker.mask_dict(kwargs)
                extra["kwargs"] = masked_kwargs

            logger.log(level, f"Calling {func.__name__}", **extra)  # type: ignore

            result = func(*args, **kwargs)

            if include_result:
                logger.log(level, f"{func.__name__} returned", result=result)  # type: ignore

            return result

        return wrapper

    return decorator


def log_errors(
    logger: TruthoundLogger | None = None,
    level: LogLevel = LogLevel.ERROR,
    reraise: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log exceptions.

    Args:
        logger: Logger to use (default: auto-detect).
        level: Log level for errors.
        reraise: Whether to reraise the exception.

    Returns:
        Decorator function.

    Example:
        >>> @log_errors(reraise=True)
        ... def risky_operation():
        ...     raise ValueError("Something went wrong")
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(  # type: ignore
                    level,
                    f"Exception in {func.__name__}: {e}",
                    exc_info=e,
                    function=func.__name__,
                    exception_type=type(e).__name__,
                )
                if reraise:
                    raise

        return wrapper

    return decorator
