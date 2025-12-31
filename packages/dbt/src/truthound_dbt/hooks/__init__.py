"""dbt Lifecycle Hooks.

This module provides hook protocols and implementations for dbt test
lifecycle events, enabling logging, metrics, and custom integrations.

Components:
    - DbtHook: Protocol for lifecycle hooks
    - AsyncDbtHook: Async protocol for lifecycle hooks
    - LoggingDbtHook: Logs lifecycle events
    - MetricsDbtHook: Collects metrics from events
    - CompositeDbtHook: Combines multiple hooks

Example:
    >>> from truthound_dbt.hooks import LoggingDbtHook, MetricsDbtHook
    >>>
    >>> logging_hook = LoggingDbtHook()
    >>> metrics_hook = MetricsDbtHook()
    >>>
    >>> # Use with parser
    >>> parser = ManifestParser(path, hooks=[logging_hook, metrics_hook])
"""

from truthound_dbt.hooks.base import (
    # Protocols
    DbtHook,
    AsyncDbtHook,
    # Event types
    DbtEvent,
    TestStartEvent,
    TestEndEvent,
    ParseStartEvent,
    ParseEndEvent,
    ConversionStartEvent,
    ConversionEndEvent,
    GenerationStartEvent,
    GenerationEndEvent,
    # Implementations
    BaseDbtHook,
    LoggingDbtHook,
    MetricsDbtHook,
    CompositeDbtHook,
    # Async implementations
    AsyncBaseDbtHook,
    AsyncLoggingDbtHook,
    AsyncCompositeDbtHook,
    # Adapter
    SyncToAsyncDbtHookAdapter,
    # Exceptions
    HookError,
    HookExecutionError,
)

__all__ = [
    # Protocols
    "DbtHook",
    "AsyncDbtHook",
    # Event types
    "DbtEvent",
    "TestStartEvent",
    "TestEndEvent",
    "ParseStartEvent",
    "ParseEndEvent",
    "ConversionStartEvent",
    "ConversionEndEvent",
    "GenerationStartEvent",
    "GenerationEndEvent",
    # Implementations
    "BaseDbtHook",
    "LoggingDbtHook",
    "MetricsDbtHook",
    "CompositeDbtHook",
    # Async implementations
    "AsyncBaseDbtHook",
    "AsyncLoggingDbtHook",
    "AsyncCompositeDbtHook",
    # Adapter
    "SyncToAsyncDbtHookAdapter",
    # Exceptions
    "HookError",
    "HookExecutionError",
]
