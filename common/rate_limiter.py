"""Rate limiting utilities for Truthound Integrations.

This module provides flexible, production-ready rate limiting implementations
designed for controlling request rates in distributed environments. It supports:

- Multiple algorithms: Token Bucket, Sliding Window, Fixed Window, Leaky Bucket
- Configurable limits per time window
- Key-based rate limiting for multi-tenant scenarios
- Async and sync function support
- Hook system for monitoring and alerting

Design Principles:
    1. Protocol-based: Easy to extend with custom rate limiting strategies
    2. Immutable Config: Thread-safe configuration using frozen dataclass
    3. Observable: Hook system for monitoring rate limit events
    4. Composable: Works well with retry, circuit breaker, and other patterns

Rate Limiting Algorithms:
    - TOKEN_BUCKET: Smooth rate limiting with burst capacity
    - SLIDING_WINDOW: Precise rate limiting with sliding time window
    - FIXED_WINDOW: Simple rate limiting with fixed time windows
    - LEAKY_BUCKET: Constant output rate regardless of input burst

Example:
    >>> from common.rate_limiter import rate_limit, RateLimitConfig
    >>> @rate_limit(max_requests=100, window_seconds=60.0)
    ... def call_api():
    ...     return api.get("/data")

    >>> # With custom configuration
    >>> config = RateLimitConfig(
    ...     max_requests=1000,
    ...     window_seconds=3600.0,
    ...     algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    ...     burst_size=50,
    ... )
    >>> @rate_limit(config=config)
    ... async def async_call():
    ...     return await api.async_get("/data")
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
import time
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class RateLimitError(TruthoundIntegrationError):
    """Base exception for rate limiting errors.

    Attributes:
        limit: The rate limit that was exceeded.
        window_seconds: The time window for the limit.
        retry_after_seconds: Suggested wait time before retrying.
    """

    def __init__(
        self,
        message: str,
        *,
        limit: int | None = None,
        window_seconds: float | None = None,
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Human-readable error description.
            limit: The rate limit that was exceeded.
            window_seconds: The time window for the limit.
            retry_after_seconds: Suggested wait time before retrying.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if limit is not None:
            details["limit"] = limit
        if window_seconds is not None:
            details["window_seconds"] = window_seconds
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, details=details, cause=cause)
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after_seconds = retry_after_seconds


class RateLimitExceededError(RateLimitError):
    """Exception raised when rate limit is exceeded.

    Attributes:
        key: The rate limit key that was exceeded.
        current_count: Current request count in the window.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        key: str | None = None,
        current_count: int | None = None,
        limit: int | None = None,
        window_seconds: float | None = None,
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize rate limit exceeded error.

        Args:
            message: Human-readable error description.
            key: The rate limit key that was exceeded.
            current_count: Current request count in the window.
            limit: The rate limit that was exceeded.
            window_seconds: The time window for the limit.
            retry_after_seconds: Suggested wait time before retrying.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if key is not None:
            details["key"] = key
        if current_count is not None:
            details["current_count"] = current_count
        super().__init__(
            message,
            limit=limit,
            window_seconds=window_seconds,
            retry_after_seconds=retry_after_seconds,
            details=details,
            cause=cause,
        )
        self.key = key
        self.current_count = current_count


# =============================================================================
# Enums
# =============================================================================


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms.

    Attributes:
        TOKEN_BUCKET: Smooth rate limiting with burst capacity.
            Tokens are added at a fixed rate, requests consume tokens.
        SLIDING_WINDOW: Precise rate limiting with sliding time window.
            Tracks exact timestamps of requests within the window.
        FIXED_WINDOW: Simple rate limiting with fixed time windows.
            Counts requests in discrete time windows.
        LEAKY_BUCKET: Constant output rate regardless of input burst.
            Requests are processed at a fixed rate, excess is queued or rejected.
    """

    TOKEN_BUCKET = auto()
    SLIDING_WINDOW = auto()
    FIXED_WINDOW = auto()
    LEAKY_BUCKET = auto()


class RateLimitAction(Enum):
    """Action to take when rate limit is exceeded.

    Attributes:
        REJECT: Immediately reject the request with an exception.
        WAIT: Block and wait until tokens are available.
        WARN: Log a warning but allow the request through.
    """

    REJECT = auto()
    WAIT = auto()
    WARN = auto()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class RateLimiter(Protocol):
    """Protocol for rate limiter implementations.

    Implement this protocol to create custom rate limiting strategies.
    """

    @abstractmethod
    def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Attempt to acquire tokens for a request.

        Args:
            key: The rate limit key (for multi-tenant scenarios).
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False if rate limited.
        """
        ...

    @abstractmethod
    def get_wait_time(self, key: str = "default", tokens: int = 1) -> float:
        """Get time to wait before tokens are available.

        Args:
            key: The rate limit key.
            tokens: Number of tokens needed.

        Returns:
            Seconds to wait, or 0 if tokens are available now.
        """
        ...

    @abstractmethod
    def get_remaining(self, key: str = "default") -> int:
        """Get remaining tokens/requests in current window.

        Args:
            key: The rate limit key.

        Returns:
            Number of remaining tokens/requests.
        """
        ...

    @abstractmethod
    def reset(self, key: str | None = None) -> None:
        """Reset rate limit state.

        Args:
            key: Specific key to reset, or None for all keys.
        """
        ...


@runtime_checkable
class RateLimitHook(Protocol):
    """Protocol for rate limit event hooks.

    Implement this to receive notifications about rate limit events.
    """

    @abstractmethod
    def on_acquire(
        self,
        key: str,
        tokens: int,
        remaining: int,
        context: dict[str, Any],
    ) -> None:
        """Called when tokens are successfully acquired.

        Args:
            key: The rate limit key.
            tokens: Number of tokens acquired.
            remaining: Remaining tokens after acquisition.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_reject(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Called when a request is rejected due to rate limiting.

        Args:
            key: The rate limit key.
            tokens: Number of tokens requested.
            wait_time: Time until tokens would be available.
            context: Additional context information.
        """
        ...

    @abstractmethod
    def on_wait(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Called when a request waits for rate limit.

        Args:
            key: The rate limit key.
            tokens: Number of tokens requested.
            wait_time: Time the request will wait.
            context: Additional context information.
        """
        ...


@runtime_checkable
class KeyExtractor(Protocol):
    """Protocol for extracting rate limit keys from function calls.

    Implement this to create custom key extraction strategies.
    """

    @abstractmethod
    def extract_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Extract rate limit key from function call.

        Args:
            func: The function being rate limited.
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.

        Returns:
            The rate limit key.
        """
        ...


# =============================================================================
# Key Extractors
# =============================================================================


class DefaultKeyExtractor:
    """Extract key based on function name."""

    def extract_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Return function name as key.

        Args:
            func: The function being rate limited.
            args: Positional arguments (unused).
            kwargs: Keyword arguments (unused).

        Returns:
            Function name as the key.
        """
        return f"{func.__module__}.{func.__name__}"


class ArgumentKeyExtractor:
    """Extract key from function arguments."""

    def __init__(
        self,
        arg_name: str | None = None,
        arg_index: int | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize argument key extractor.

        Args:
            arg_name: Name of keyword argument to use as key.
            arg_index: Index of positional argument to use as key.
            prefix: Prefix to add to the key.
        """
        self.arg_name = arg_name
        self.arg_index = arg_index
        self.prefix = prefix

    def extract_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Extract key from arguments.

        Args:
            func: The function being rate limited.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Extracted key with optional prefix.
        """
        key = "default"

        if self.arg_name and self.arg_name in kwargs:
            key = str(kwargs[self.arg_name])
        elif self.arg_index is not None and len(args) > self.arg_index:
            key = str(args[self.arg_index])

        if self.prefix:
            return f"{self.prefix}:{key}"
        return key


class CallableKeyExtractor:
    """Extract key using a callable."""

    def __init__(
        self,
        extractor: Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], str],
    ) -> None:
        """Initialize callable key extractor.

        Args:
            extractor: Function to extract the key.
        """
        self._extractor = extractor

    def extract_key(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Extract key using the callable.

        Args:
            func: The function being rate limited.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Extracted key.
        """
        return self._extractor(func, args, kwargs)


# =============================================================================
# Hooks
# =============================================================================


class LoggingRateLimitHook:
    """Hook that logs rate limit events.

    Uses the Truthound logging system for structured logging.
    """

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.rate_limiter).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.rate_limiter")

    def on_acquire(
        self,
        key: str,
        tokens: int,
        remaining: int,
        context: dict[str, Any],
    ) -> None:
        """Log token acquisition.

        Args:
            key: The rate limit key.
            tokens: Number of tokens acquired.
            remaining: Remaining tokens.
            context: Additional context.
        """
        self._logger.debug(
            "Rate limit tokens acquired",
            key=key,
            tokens=tokens,
            remaining=remaining,
            **context,
        )

    def on_reject(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Log rate limit rejection.

        Args:
            key: The rate limit key.
            tokens: Number of tokens requested.
            wait_time: Time until tokens available.
            context: Additional context.
        """
        self._logger.warning(
            "Rate limit exceeded",
            key=key,
            tokens=tokens,
            wait_time_seconds=wait_time,
            **context,
        )

    def on_wait(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Log rate limit wait.

        Args:
            key: The rate limit key.
            tokens: Number of tokens requested.
            wait_time: Time to wait.
            context: Additional context.
        """
        self._logger.info(
            "Rate limit waiting",
            key=key,
            tokens=tokens,
            wait_time_seconds=wait_time,
            **context,
        )


class MetricsRateLimitHook:
    """Hook that collects rate limit metrics.

    Useful for monitoring and alerting on rate limiting patterns.
    """

    def __init__(self) -> None:
        """Initialize metrics collection."""
        self._acquired_count: int = 0
        self._rejected_count: int = 0
        self._waited_count: int = 0
        self._total_wait_time: float = 0.0
        self._key_stats: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def on_acquire(
        self,
        key: str,
        tokens: int,
        remaining: int,
        context: dict[str, Any],
    ) -> None:
        """Record token acquisition.

        Args:
            key: The rate limit key.
            tokens: Number of tokens acquired.
            remaining: Remaining tokens.
            context: Additional context.
        """
        with self._lock:
            self._acquired_count += tokens
            if key not in self._key_stats:
                self._key_stats[key] = {"acquired": 0, "rejected": 0, "waited": 0}
            self._key_stats[key]["acquired"] += tokens

    def on_reject(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Record rate limit rejection.

        Args:
            key: The rate limit key.
            tokens: Number of tokens requested.
            wait_time: Time until tokens available.
            context: Additional context.
        """
        with self._lock:
            self._rejected_count += 1
            if key not in self._key_stats:
                self._key_stats[key] = {"acquired": 0, "rejected": 0, "waited": 0}
            self._key_stats[key]["rejected"] += 1

    def on_wait(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Record rate limit wait.

        Args:
            key: The rate limit key.
            tokens: Number of tokens requested.
            wait_time: Time to wait.
            context: Additional context.
        """
        with self._lock:
            self._waited_count += 1
            self._total_wait_time += wait_time
            if key not in self._key_stats:
                self._key_stats[key] = {"acquired": 0, "rejected": 0, "waited": 0}
            self._key_stats[key]["waited"] += 1

    @property
    def acquired_count(self) -> int:
        """Get total acquired token count."""
        return self._acquired_count

    @property
    def rejected_count(self) -> int:
        """Get total rejected request count."""
        return self._rejected_count

    @property
    def waited_count(self) -> int:
        """Get total waited request count."""
        return self._waited_count

    @property
    def total_wait_time(self) -> float:
        """Get total wait time in seconds."""
        return self._total_wait_time

    @property
    def average_wait_time(self) -> float:
        """Get average wait time in seconds."""
        if self._waited_count == 0:
            return 0.0
        return self._total_wait_time / self._waited_count

    def get_key_stats(self, key: str) -> dict[str, int]:
        """Get stats for a specific key.

        Args:
            key: The rate limit key.

        Returns:
            Dictionary with acquired, rejected, waited counts.
        """
        with self._lock:
            return dict(self._key_stats.get(key, {"acquired": 0, "rejected": 0, "waited": 0}))

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._acquired_count = 0
            self._rejected_count = 0
            self._waited_count = 0
            self._total_wait_time = 0.0
            self._key_stats.clear()


class CompositeRateLimitHook:
    """Combine multiple rate limit hooks."""

    def __init__(self, hooks: Sequence[RateLimitHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: RateLimitHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: RateLimitHook) -> None:
        """Remove a hook.

        Args:
            hook: Hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_acquire(
        self,
        key: str,
        tokens: int,
        remaining: int,
        context: dict[str, Any],
    ) -> None:
        """Call on_acquire on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_acquire(key, tokens, remaining, context)

    def on_reject(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_reject on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_reject(key, tokens, wait_time, context)

    def on_wait(
        self,
        key: str,
        tokens: int,
        wait_time: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_wait on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_wait(key, tokens, wait_time, context)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """Configuration for rate limiting behavior.

    Immutable configuration object for rate limiting operations.
    Use builder methods to create modified copies.

    Attributes:
        max_requests: Maximum requests allowed in the time window.
        window_seconds: Time window in seconds.
        algorithm: Rate limiting algorithm to use.
        burst_size: Maximum burst size (for token bucket).
        on_limit: Action to take when rate limit is exceeded.
        max_wait_seconds: Maximum time to wait when action is WAIT.
        name: Optional name for the rate limiter (for logging/metrics).

    Example:
        >>> config = RateLimitConfig(
        ...     max_requests=100,
        ...     window_seconds=60.0,
        ...     algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        ... )
        >>> stricter_config = config.with_max_requests(50)
    """

    max_requests: int = 100
    window_seconds: float = 60.0
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_size: int | None = None
    on_limit: RateLimitAction = RateLimitAction.REJECT
    max_wait_seconds: float = 30.0
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.burst_size is not None and self.burst_size < 1:
            raise ValueError("burst_size must be at least 1 if specified")
        if self.max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be non-negative")

    @property
    def effective_burst_size(self) -> int:
        """Get effective burst size (defaults to max_requests)."""
        return self.burst_size if self.burst_size is not None else self.max_requests

    @property
    def tokens_per_second(self) -> float:
        """Get token refill rate per second."""
        return self.max_requests / self.window_seconds

    def with_max_requests(self, max_requests: int) -> RateLimitConfig:
        """Create config with new max_requests.

        Args:
            max_requests: New maximum requests.

        Returns:
            New RateLimitConfig with updated value.
        """
        return RateLimitConfig(
            max_requests=max_requests,
            window_seconds=self.window_seconds,
            algorithm=self.algorithm,
            burst_size=self.burst_size,
            on_limit=self.on_limit,
            max_wait_seconds=self.max_wait_seconds,
            name=self.name,
        )

    def with_window(self, window_seconds: float) -> RateLimitConfig:
        """Create config with new window_seconds.

        Args:
            window_seconds: New time window in seconds.

        Returns:
            New RateLimitConfig with updated value.
        """
        return RateLimitConfig(
            max_requests=self.max_requests,
            window_seconds=window_seconds,
            algorithm=self.algorithm,
            burst_size=self.burst_size,
            on_limit=self.on_limit,
            max_wait_seconds=self.max_wait_seconds,
            name=self.name,
        )

    def with_algorithm(self, algorithm: RateLimitAlgorithm) -> RateLimitConfig:
        """Create config with new algorithm.

        Args:
            algorithm: New rate limiting algorithm.

        Returns:
            New RateLimitConfig with updated value.
        """
        return RateLimitConfig(
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
            algorithm=algorithm,
            burst_size=self.burst_size,
            on_limit=self.on_limit,
            max_wait_seconds=self.max_wait_seconds,
            name=self.name,
        )

    def with_burst_size(self, burst_size: int | None) -> RateLimitConfig:
        """Create config with new burst_size.

        Args:
            burst_size: New burst size.

        Returns:
            New RateLimitConfig with updated value.
        """
        return RateLimitConfig(
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
            algorithm=self.algorithm,
            burst_size=burst_size,
            on_limit=self.on_limit,
            max_wait_seconds=self.max_wait_seconds,
            name=self.name,
        )

    def with_on_limit(self, on_limit: RateLimitAction) -> RateLimitConfig:
        """Create config with new on_limit action.

        Args:
            on_limit: New action when rate limit is exceeded.

        Returns:
            New RateLimitConfig with updated value.
        """
        return RateLimitConfig(
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
            algorithm=self.algorithm,
            burst_size=self.burst_size,
            on_limit=on_limit,
            max_wait_seconds=self.max_wait_seconds,
            name=self.name,
        )

    def with_name(self, name: str) -> RateLimitConfig:
        """Create config with a name.

        Args:
            name: Name for the rate limiter.

        Returns:
            New RateLimitConfig with updated value.
        """
        return RateLimitConfig(
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
            algorithm=self.algorithm,
            burst_size=self.burst_size,
            on_limit=self.on_limit,
            max_wait_seconds=self.max_wait_seconds,
            name=name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "algorithm": self.algorithm.name,
            "burst_size": self.burst_size,
            "on_limit": self.on_limit.name,
            "max_wait_seconds": self.max_wait_seconds,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create RateLimitConfig from dictionary.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New RateLimitConfig instance.
        """
        algorithm = RateLimitAlgorithm[data.get("algorithm", "TOKEN_BUCKET")]
        on_limit = RateLimitAction[data.get("on_limit", "REJECT")]
        return cls(
            max_requests=data.get("max_requests", 100),
            window_seconds=data.get("window_seconds", 60.0),
            algorithm=algorithm,
            burst_size=data.get("burst_size"),
            on_limit=on_limit,
            max_wait_seconds=data.get("max_wait_seconds", 30.0),
            name=data.get("name"),
        )


# Default configurations for common use cases
DEFAULT_RATE_LIMIT_CONFIG = RateLimitConfig()

# Strict rate limiting - low limit with immediate rejection
STRICT_RATE_LIMIT_CONFIG = RateLimitConfig(
    max_requests=10,
    window_seconds=60.0,
    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
    on_limit=RateLimitAction.REJECT,
)

# Lenient rate limiting - high limit with waiting
LENIENT_RATE_LIMIT_CONFIG = RateLimitConfig(
    max_requests=1000,
    window_seconds=60.0,
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    burst_size=100,
    on_limit=RateLimitAction.WAIT,
)

# Burst-friendly rate limiting - allows bursts
BURST_RATE_LIMIT_CONFIG = RateLimitConfig(
    max_requests=100,
    window_seconds=60.0,
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    burst_size=50,
    on_limit=RateLimitAction.REJECT,
)

# API rate limiting - typical API limits
API_RATE_LIMIT_CONFIG = RateLimitConfig(
    max_requests=100,
    window_seconds=1.0,
    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
    on_limit=RateLimitAction.REJECT,
)


# =============================================================================
# Rate Limiter State
# =============================================================================


@dataclass
class TokenBucketState:
    """State for token bucket rate limiter.

    Attributes:
        tokens: Current number of tokens.
        last_update: Timestamp of last token update.
    """

    tokens: float
    last_update: float


@dataclass
class SlidingWindowState:
    """State for sliding window rate limiter.

    Attributes:
        timestamps: Deque of request timestamps.
    """

    timestamps: deque[float] = field(default_factory=deque)


@dataclass
class FixedWindowState:
    """State for fixed window rate limiter.

    Attributes:
        count: Request count in current window.
        window_start: Start time of current window.
    """

    count: int = 0
    window_start: float = 0.0


@dataclass
class LeakyBucketState:
    """State for leaky bucket rate limiter.

    Attributes:
        water_level: Current water level (pending requests).
        last_leak: Timestamp of last leak.
    """

    water_level: float = 0.0
    last_leak: float = 0.0


# =============================================================================
# Rate Limiter Implementations
# =============================================================================


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation.

    Tokens are added at a fixed rate. Each request consumes tokens.
    Allows bursting up to the bucket capacity.

    Example:
        >>> limiter = TokenBucketRateLimiter(RateLimitConfig(
        ...     max_requests=100,
        ...     window_seconds=60.0,
        ...     burst_size=20,
        ... ))
        >>> if limiter.acquire("user_123"):
        ...     process_request()
    """

    def __init__(
        self,
        config: RateLimitConfig,
        hooks: Sequence[RateLimitHook] | None = None,
    ) -> None:
        """Initialize token bucket rate limiter.

        Args:
            config: Rate limit configuration.
            hooks: Rate limit event hooks.
        """
        self.config = config
        self._buckets: dict[str, TokenBucketState] = {}
        self._lock = threading.RLock()
        self._hook: RateLimitHook | None = None
        if hooks:
            self._hook = CompositeRateLimitHook(list(hooks))

    def _get_or_create_bucket(self, key: str) -> TokenBucketState:
        """Get or create a token bucket for the key.

        Args:
            key: The rate limit key.

        Returns:
            TokenBucketState for the key.
        """
        if key not in self._buckets:
            self._buckets[key] = TokenBucketState(
                tokens=float(self.config.effective_burst_size),
                last_update=time.monotonic(),
            )
        return self._buckets[key]

    def _refill_tokens(self, bucket: TokenBucketState) -> None:
        """Refill tokens based on elapsed time.

        Args:
            bucket: The bucket to refill.
        """
        now = time.monotonic()
        elapsed = now - bucket.last_update
        tokens_to_add = elapsed * self.config.tokens_per_second
        bucket.tokens = min(
            bucket.tokens + tokens_to_add,
            float(self.config.effective_burst_size),
        )
        bucket.last_update = now

    def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Attempt to acquire tokens.

        Args:
            key: The rate limit key.
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired.
        """
        with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._refill_tokens(bucket)

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                if self._hook:
                    self._hook.on_acquire(
                        key,
                        tokens,
                        int(bucket.tokens),
                        {"algorithm": "token_bucket"},
                    )
                return True
            return False

    def get_wait_time(self, key: str = "default", tokens: int = 1) -> float:
        """Get time to wait for tokens.

        Args:
            key: The rate limit key.
            tokens: Number of tokens needed.

        Returns:
            Seconds to wait.
        """
        with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._refill_tokens(bucket)

            if bucket.tokens >= tokens:
                return 0.0

            tokens_needed = tokens - bucket.tokens
            return tokens_needed / self.config.tokens_per_second

    def get_remaining(self, key: str = "default") -> int:
        """Get remaining tokens.

        Args:
            key: The rate limit key.

        Returns:
            Number of remaining tokens.
        """
        with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._refill_tokens(bucket)
            return int(bucket.tokens)

    def reset(self, key: str | None = None) -> None:
        """Reset rate limiter state.

        Args:
            key: Specific key to reset, or None for all.
        """
        with self._lock:
            if key is None:
                self._buckets.clear()
            elif key in self._buckets:
                del self._buckets[key]


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation.

    Tracks exact timestamps of requests within the window.
    Most accurate but uses more memory.

    Example:
        >>> limiter = SlidingWindowRateLimiter(RateLimitConfig(
        ...     max_requests=100,
        ...     window_seconds=60.0,
        ... ))
        >>> if limiter.acquire("user_123"):
        ...     process_request()
    """

    def __init__(
        self,
        config: RateLimitConfig,
        hooks: Sequence[RateLimitHook] | None = None,
    ) -> None:
        """Initialize sliding window rate limiter.

        Args:
            config: Rate limit configuration.
            hooks: Rate limit event hooks.
        """
        self.config = config
        self._windows: dict[str, SlidingWindowState] = {}
        self._lock = threading.RLock()
        self._hook: RateLimitHook | None = None
        if hooks:
            self._hook = CompositeRateLimitHook(list(hooks))

    def _get_or_create_window(self, key: str) -> SlidingWindowState:
        """Get or create a sliding window for the key.

        Args:
            key: The rate limit key.

        Returns:
            SlidingWindowState for the key.
        """
        if key not in self._windows:
            self._windows[key] = SlidingWindowState()
        return self._windows[key]

    def _cleanup_old_timestamps(self, window: SlidingWindowState) -> None:
        """Remove timestamps outside the window.

        Args:
            window: The window to clean up.
        """
        now = time.monotonic()
        window_start = now - self.config.window_seconds
        while window.timestamps and window.timestamps[0] < window_start:
            window.timestamps.popleft()

    def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Attempt to acquire tokens.

        Args:
            key: The rate limit key.
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired.
        """
        with self._lock:
            window = self._get_or_create_window(key)
            self._cleanup_old_timestamps(window)

            if len(window.timestamps) + tokens <= self.config.max_requests:
                now = time.monotonic()
                for _ in range(tokens):
                    window.timestamps.append(now)
                if self._hook:
                    self._hook.on_acquire(
                        key,
                        tokens,
                        self.config.max_requests - len(window.timestamps),
                        {"algorithm": "sliding_window"},
                    )
                return True
            return False

    def get_wait_time(self, key: str = "default", tokens: int = 1) -> float:
        """Get time to wait for tokens.

        Args:
            key: The rate limit key.
            tokens: Number of tokens needed.

        Returns:
            Seconds to wait.
        """
        with self._lock:
            window = self._get_or_create_window(key)
            self._cleanup_old_timestamps(window)

            if len(window.timestamps) + tokens <= self.config.max_requests:
                return 0.0

            # Find when enough old timestamps will expire
            excess = len(window.timestamps) + tokens - self.config.max_requests
            if excess > len(window.timestamps):
                # Would need to wait for more than current requests
                return self.config.window_seconds

            # Wait for the nth oldest timestamp to expire
            oldest_to_expire = sorted(window.timestamps)[excess - 1]
            now = time.monotonic()
            window_end = oldest_to_expire + self.config.window_seconds
            return max(0.0, window_end - now)

    def get_remaining(self, key: str = "default") -> int:
        """Get remaining requests.

        Args:
            key: The rate limit key.

        Returns:
            Number of remaining requests.
        """
        with self._lock:
            window = self._get_or_create_window(key)
            self._cleanup_old_timestamps(window)
            return self.config.max_requests - len(window.timestamps)

    def reset(self, key: str | None = None) -> None:
        """Reset rate limiter state.

        Args:
            key: Specific key to reset, or None for all.
        """
        with self._lock:
            if key is None:
                self._windows.clear()
            elif key in self._windows:
                del self._windows[key]


class FixedWindowRateLimiter:
    """Fixed window rate limiter implementation.

    Counts requests in discrete time windows.
    Simple but can allow bursts at window boundaries.

    Example:
        >>> limiter = FixedWindowRateLimiter(RateLimitConfig(
        ...     max_requests=100,
        ...     window_seconds=60.0,
        ... ))
        >>> if limiter.acquire("user_123"):
        ...     process_request()
    """

    def __init__(
        self,
        config: RateLimitConfig,
        hooks: Sequence[RateLimitHook] | None = None,
    ) -> None:
        """Initialize fixed window rate limiter.

        Args:
            config: Rate limit configuration.
            hooks: Rate limit event hooks.
        """
        self.config = config
        self._windows: dict[str, FixedWindowState] = {}
        self._lock = threading.RLock()
        self._hook: RateLimitHook | None = None
        if hooks:
            self._hook = CompositeRateLimitHook(list(hooks))

    def _get_or_create_window(self, key: str) -> FixedWindowState:
        """Get or create a fixed window for the key.

        Args:
            key: The rate limit key.

        Returns:
            FixedWindowState for the key.
        """
        if key not in self._windows:
            self._windows[key] = FixedWindowState(
                count=0,
                window_start=time.monotonic(),
            )
        return self._windows[key]

    def _check_window_reset(self, window: FixedWindowState) -> None:
        """Reset window if it has expired.

        Args:
            window: The window to check.
        """
        now = time.monotonic()
        if now - window.window_start >= self.config.window_seconds:
            window.count = 0
            window.window_start = now

    def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Attempt to acquire tokens.

        Args:
            key: The rate limit key.
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired.
        """
        with self._lock:
            window = self._get_or_create_window(key)
            self._check_window_reset(window)

            if window.count + tokens <= self.config.max_requests:
                window.count += tokens
                if self._hook:
                    self._hook.on_acquire(
                        key,
                        tokens,
                        self.config.max_requests - window.count,
                        {"algorithm": "fixed_window"},
                    )
                return True
            return False

    def get_wait_time(self, key: str = "default", tokens: int = 1) -> float:
        """Get time to wait for tokens.

        Args:
            key: The rate limit key.
            tokens: Number of tokens needed.

        Returns:
            Seconds to wait until window resets.
        """
        with self._lock:
            window = self._get_or_create_window(key)
            self._check_window_reset(window)

            if window.count + tokens <= self.config.max_requests:
                return 0.0

            now = time.monotonic()
            window_end = window.window_start + self.config.window_seconds
            return max(0.0, window_end - now)

    def get_remaining(self, key: str = "default") -> int:
        """Get remaining requests.

        Args:
            key: The rate limit key.

        Returns:
            Number of remaining requests.
        """
        with self._lock:
            window = self._get_or_create_window(key)
            self._check_window_reset(window)
            return self.config.max_requests - window.count

    def reset(self, key: str | None = None) -> None:
        """Reset rate limiter state.

        Args:
            key: Specific key to reset, or None for all.
        """
        with self._lock:
            if key is None:
                self._windows.clear()
            elif key in self._windows:
                del self._windows[key]


class LeakyBucketRateLimiter:
    """Leaky bucket rate limiter implementation.

    Requests "leak" out at a constant rate.
    Smooths out bursts to a constant rate.

    Example:
        >>> limiter = LeakyBucketRateLimiter(RateLimitConfig(
        ...     max_requests=100,
        ...     window_seconds=60.0,
        ... ))
        >>> if limiter.acquire("user_123"):
        ...     process_request()
    """

    def __init__(
        self,
        config: RateLimitConfig,
        hooks: Sequence[RateLimitHook] | None = None,
    ) -> None:
        """Initialize leaky bucket rate limiter.

        Args:
            config: Rate limit configuration.
            hooks: Rate limit event hooks.
        """
        self.config = config
        self._buckets: dict[str, LeakyBucketState] = {}
        self._lock = threading.RLock()
        self._hook: RateLimitHook | None = None
        if hooks:
            self._hook = CompositeRateLimitHook(list(hooks))

    def _get_or_create_bucket(self, key: str) -> LeakyBucketState:
        """Get or create a leaky bucket for the key.

        Args:
            key: The rate limit key.

        Returns:
            LeakyBucketState for the key.
        """
        if key not in self._buckets:
            self._buckets[key] = LeakyBucketState(
                water_level=0.0,
                last_leak=time.monotonic(),
            )
        return self._buckets[key]

    def _leak_water(self, bucket: LeakyBucketState) -> None:
        """Leak water based on elapsed time.

        Args:
            bucket: The bucket to leak.
        """
        now = time.monotonic()
        elapsed = now - bucket.last_leak
        leaked = elapsed * self.config.tokens_per_second
        bucket.water_level = max(0.0, bucket.water_level - leaked)
        bucket.last_leak = now

    def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Attempt to acquire tokens (add water).

        Args:
            key: The rate limit key.
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired (water added).
        """
        with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._leak_water(bucket)

            if bucket.water_level + tokens <= self.config.effective_burst_size:
                bucket.water_level += tokens
                remaining = int(self.config.effective_burst_size - bucket.water_level)
                if self._hook:
                    self._hook.on_acquire(
                        key,
                        tokens,
                        remaining,
                        {"algorithm": "leaky_bucket"},
                    )
                return True
            return False

    def get_wait_time(self, key: str = "default", tokens: int = 1) -> float:
        """Get time to wait for capacity.

        Args:
            key: The rate limit key.
            tokens: Number of tokens needed.

        Returns:
            Seconds to wait.
        """
        with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._leak_water(bucket)

            if bucket.water_level + tokens <= self.config.effective_burst_size:
                return 0.0

            overflow = bucket.water_level + tokens - self.config.effective_burst_size
            return overflow / self.config.tokens_per_second

    def get_remaining(self, key: str = "default") -> int:
        """Get remaining capacity.

        Args:
            key: The rate limit key.

        Returns:
            Remaining capacity.
        """
        with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._leak_water(bucket)
            return int(self.config.effective_burst_size - bucket.water_level)

    def reset(self, key: str | None = None) -> None:
        """Reset rate limiter state.

        Args:
            key: Specific key to reset, or None for all.
        """
        with self._lock:
            if key is None:
                self._buckets.clear()
            elif key in self._buckets:
                del self._buckets[key]


# =============================================================================
# Rate Limiter Factory
# =============================================================================


def create_rate_limiter(
    config: RateLimitConfig,
    hooks: Sequence[RateLimitHook] | None = None,
) -> RateLimiter:
    """Create a rate limiter based on configuration.

    Args:
        config: Rate limit configuration.
        hooks: Rate limit event hooks.

    Returns:
        RateLimiter instance.
    """
    if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
        return TokenBucketRateLimiter(config, hooks)
    elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
        return SlidingWindowRateLimiter(config, hooks)
    elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
        return FixedWindowRateLimiter(config, hooks)
    elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
        return LeakyBucketRateLimiter(config, hooks)
    else:
        return TokenBucketRateLimiter(config, hooks)


# =============================================================================
# Rate Limiter Executor
# =============================================================================


class RateLimitExecutor:
    """Executes functions with rate limiting.

    This class encapsulates the rate limiting logic and can be
    used directly or through the rate_limit decorator.

    Example:
        >>> executor = RateLimitExecutor(RateLimitConfig(max_requests=100))
        >>> result = executor.execute(my_function, "arg1", key="value")
    """

    def __init__(
        self,
        config: RateLimitConfig,
        limiter: RateLimiter | None = None,
        key_extractor: KeyExtractor | None = None,
        hooks: Sequence[RateLimitHook] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            config: Rate limit configuration.
            limiter: Custom rate limiter (created from config if None).
            key_extractor: Custom key extractor.
            hooks: Rate limit event hooks.
        """
        self.config = config
        self._limiter = limiter or create_rate_limiter(config, hooks)
        self._key_extractor = key_extractor or DefaultKeyExtractor()
        self._hook: RateLimitHook | None = None
        if hooks:
            self._hook = CompositeRateLimitHook(list(hooks))

    def _create_context(self, func: Callable[..., Any], key: str) -> dict[str, Any]:
        """Create context dictionary for hooks.

        Args:
            func: Function being rate limited.
            key: Rate limit key.

        Returns:
            Context dictionary.
        """
        return {
            "function": func.__name__,
            "module": func.__module__,
            "rate_limit_name": self.config.name or func.__name__,
            "key": key,
            "max_requests": self.config.max_requests,
            "window_seconds": self.config.window_seconds,
        }

    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with rate limiting.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value.

        Raises:
            RateLimitExceededError: When rate limit exceeded and action is REJECT.
        """
        key = self._key_extractor.extract_key(func, args, kwargs)
        context = self._create_context(func, key)

        if self._limiter.acquire(key):
            return func(*args, **kwargs)

        wait_time = self._limiter.get_wait_time(key)

        if (
            self.config.on_limit == RateLimitAction.WAIT
            and wait_time <= self.config.max_wait_seconds
        ):
            if self._hook:
                self._hook.on_wait(key, 1, wait_time, context)
            time.sleep(wait_time)
            # Try again after waiting
            if self._limiter.acquire(key):
                return func(*args, **kwargs)

        if self.config.on_limit == RateLimitAction.WARN:
            if self._hook:
                self._hook.on_reject(key, 1, wait_time, context)
            # Log warning but allow through
            from common.logging import get_logger
            logger = get_logger("common.rate_limiter")
            logger.warning(
                "Rate limit exceeded, allowing through",
                key=key,
                wait_time=wait_time,
            )
            return func(*args, **kwargs)

        # REJECT
        if self._hook:
            self._hook.on_reject(key, 1, wait_time, context)
        raise RateLimitExceededError(
            f"Rate limit exceeded for '{key}'",
            key=key,
            current_count=self.config.max_requests - self._limiter.get_remaining(key),
            limit=self.config.max_requests,
            window_seconds=self.config.window_seconds,
            retry_after_seconds=wait_time,
        )

    async def execute_async(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute async function with rate limiting.

        Args:
            func: Async function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function return value.

        Raises:
            RateLimitExceededError: When rate limit exceeded and action is REJECT.
        """
        key = self._key_extractor.extract_key(func, args, kwargs)
        context = self._create_context(func, key)

        if self._limiter.acquire(key):
            return await func(*args, **kwargs)

        wait_time = self._limiter.get_wait_time(key)

        if (
            self.config.on_limit == RateLimitAction.WAIT
            and wait_time <= self.config.max_wait_seconds
        ):
            if self._hook:
                self._hook.on_wait(key, 1, wait_time, context)
            await asyncio.sleep(wait_time)
            # Try again after waiting
            if self._limiter.acquire(key):
                return await func(*args, **kwargs)

        if self.config.on_limit == RateLimitAction.WARN:
            if self._hook:
                self._hook.on_reject(key, 1, wait_time, context)
            from common.logging import get_logger
            logger = get_logger("common.rate_limiter")
            logger.warning(
                "Rate limit exceeded, allowing through",
                key=key,
                wait_time=wait_time,
            )
            return await func(*args, **kwargs)

        # REJECT
        if self._hook:
            self._hook.on_reject(key, 1, wait_time, context)
        raise RateLimitExceededError(
            f"Rate limit exceeded for '{key}'",
            key=key,
            current_count=self.config.max_requests - self._limiter.get_remaining(key),
            limit=self.config.max_requests,
            window_seconds=self.config.window_seconds,
            retry_after_seconds=wait_time,
        )


# =============================================================================
# Rate Limiter Registry
# =============================================================================


class RateLimiterRegistry:
    """Registry for managing multiple rate limiters.

    Provides a central location to create, retrieve, and manage
    rate limiters by name.

    Example:
        >>> registry = RateLimiterRegistry()
        >>> limiter = registry.get_or_create("api_calls", config=my_config)
        >>> if limiter.acquire("user_123"):
        ...     process_request()
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._limiters: dict[str, RateLimiter] = {}
        self._configs: dict[str, RateLimitConfig] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> RateLimiter | None:
        """Get rate limiter by name.

        Args:
            name: Rate limiter name.

        Returns:
            RateLimiter if found, None otherwise.
        """
        with self._lock:
            return self._limiters.get(name)

    def get_or_create(
        self,
        name: str,
        config: RateLimitConfig | None = None,
        hooks: Sequence[RateLimitHook] | None = None,
    ) -> RateLimiter:
        """Get existing or create new rate limiter.

        Args:
            name: Rate limiter name.
            config: Configuration (uses default if None).
            hooks: Rate limit event hooks.

        Returns:
            RateLimiter instance.
        """
        with self._lock:
            if name in self._limiters:
                return self._limiters[name]

            rl_config = (config or DEFAULT_RATE_LIMIT_CONFIG).with_name(name)
            limiter = create_rate_limiter(rl_config, hooks)
            self._limiters[name] = limiter
            self._configs[name] = rl_config
            return limiter

    def remove(self, name: str) -> bool:
        """Remove rate limiter by name.

        Args:
            name: Rate limiter name.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._limiters:
                del self._limiters[name]
                del self._configs[name]
                return True
            return False

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        with self._lock:
            for limiter in self._limiters.values():
                limiter.reset()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all rate limiters.

        Returns:
            Dictionary mapping names to stats.
        """
        with self._lock:
            result = {}
            for name, limiter in self._limiters.items():
                config = self._configs[name]
                result[name] = {
                    "remaining": limiter.get_remaining(),
                    "max_requests": config.max_requests,
                    "window_seconds": config.window_seconds,
                    "algorithm": config.algorithm.name,
                }
            return result

    @property
    def names(self) -> list[str]:
        """Get all rate limiter names."""
        with self._lock:
            return list(self._limiters.keys())


# Global registry instance
_default_registry = RateLimiterRegistry()


def get_rate_limiter(
    name: str,
    config: RateLimitConfig | None = None,
    hooks: Sequence[RateLimitHook] | None = None,
) -> RateLimiter:
    """Get or create a rate limiter from the global registry.

    Args:
        name: Rate limiter name.
        config: Configuration (uses default if None).
        hooks: Rate limit event hooks.

    Returns:
        RateLimiter instance.

    Example:
        >>> limiter = get_rate_limiter("external_api")
        >>> if limiter.acquire("user_123"):
        ...     process_request()
    """
    return _default_registry.get_or_create(name, config=config, hooks=hooks)


def get_rate_limiter_registry() -> RateLimiterRegistry:
    """Get the global rate limiter registry.

    Returns:
        The global RateLimiterRegistry instance.
    """
    return _default_registry


# =============================================================================
# Rate Limit Decorator
# =============================================================================


def rate_limit(
    *,
    config: RateLimitConfig | None = None,
    name: str | None = None,
    max_requests: int | None = None,
    window_seconds: float | None = None,
    algorithm: RateLimitAlgorithm | None = None,
    burst_size: int | None = None,
    on_limit: RateLimitAction | None = None,
    hooks: Sequence[RateLimitHook] | None = None,
    key_extractor: KeyExtractor | None = None,
    use_registry: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to add rate limiting to functions.

    Can be used with a RateLimitConfig object or individual parameters.
    Supports both sync and async functions.

    Args:
        config: Complete rate limit configuration (takes precedence).
        name: Name for the rate limiter (for registry and logging).
        max_requests: Maximum requests in the time window.
        window_seconds: Time window in seconds.
        algorithm: Rate limiting algorithm.
        burst_size: Maximum burst size (for token bucket).
        on_limit: Action when rate limit is exceeded.
        hooks: Rate limit event hooks.
        key_extractor: Custom key extractor.
        use_registry: If True, use global registry (enables sharing).

    Returns:
        Decorator function.

    Example:
        >>> @rate_limit(max_requests=100, window_seconds=60.0)
        ... def call_api():
        ...     return api.get("/data")

        >>> @rate_limit(config=RateLimitConfig(max_requests=1000))
        ... async def async_call():
        ...     return await api.async_get("/data")
    """
    # Build config from parameters if not provided
    if config is None:
        config = RateLimitConfig(
            max_requests=max_requests if max_requests is not None else 100,
            window_seconds=window_seconds if window_seconds is not None else 60.0,
            algorithm=algorithm if algorithm is not None else RateLimitAlgorithm.TOKEN_BUCKET,
            burst_size=burst_size,
            on_limit=on_limit if on_limit is not None else RateLimitAction.REJECT,
            name=name,
        )
    elif name is not None:
        config = config.with_name(name)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        rl_name = config.name or func.__name__

        if use_registry:
            limiter = get_rate_limiter(rl_name, config=config, hooks=hooks)
        else:
            limiter = create_rate_limiter(config.with_name(rl_name), hooks)

        executor = RateLimitExecutor(
            config=config,
            limiter=limiter,
            key_extractor=key_extractor,
            hooks=hooks,
        )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await executor.execute_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return executor.execute(func, *args, **kwargs)

            return sync_wrapper

    return decorator


def rate_limit_call(
    func: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    config: RateLimitConfig | None = None,
    key: str | None = None,
    hooks: Sequence[RateLimitHook] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with rate limiting without using decorator.

    Useful when you can't modify the function definition.

    Args:
        func: Function to execute.
        *args: Positional arguments for function.
        name: Rate limiter name (uses function name if None).
        config: Rate limit configuration.
        key: Explicit rate limit key (overrides extractor).
        hooks: Rate limit event hooks.
        **kwargs: Keyword arguments for function.

    Returns:
        Function return value.

    Example:
        >>> result = rate_limit_call(
        ...     external_api.fetch,
        ...     endpoint="/data",
        ...     name="external_api",
        ...     config=RateLimitConfig(max_requests=100),
        ... )
    """
    rl_name = name or func.__name__
    rl_config = config or DEFAULT_RATE_LIMIT_CONFIG

    limiter = get_rate_limiter(rl_name, config=rl_config, hooks=hooks)

    # Use explicit key or extract from function
    if key is None:
        extractor = DefaultKeyExtractor()
        key = extractor.extract_key(func, args, kwargs)

    if limiter.acquire(key):
        return func(*args, **kwargs)

    wait_time = limiter.get_wait_time(key)

    if (
        rl_config.on_limit == RateLimitAction.WAIT
        and wait_time <= rl_config.max_wait_seconds
    ):
        time.sleep(wait_time)
        if limiter.acquire(key):
            return func(*args, **kwargs)

    if rl_config.on_limit == RateLimitAction.WARN:
        from common.logging import get_logger
        logger = get_logger("common.rate_limiter")
        logger.warning(
            "Rate limit exceeded, allowing through",
            key=key,
            wait_time=wait_time,
        )
        return func(*args, **kwargs)

    raise RateLimitExceededError(
        f"Rate limit exceeded for '{key}'",
        key=key,
        current_count=rl_config.max_requests - limiter.get_remaining(key),
        limit=rl_config.max_requests,
        window_seconds=rl_config.window_seconds,
        retry_after_seconds=wait_time,
    )


async def rate_limit_call_async(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    name: str | None = None,
    config: RateLimitConfig | None = None,
    key: str | None = None,
    hooks: Sequence[RateLimitHook] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an async function with rate limiting without using decorator.

    Args:
        func: Async function to execute.
        *args: Positional arguments for function.
        name: Rate limiter name (uses function name if None).
        config: Rate limit configuration.
        key: Explicit rate limit key (overrides extractor).
        hooks: Rate limit event hooks.
        **kwargs: Keyword arguments for function.

    Returns:
        Function return value.

    Example:
        >>> result = await rate_limit_call_async(
        ...     async_api.fetch,
        ...     endpoint="/data",
        ...     name="async_api",
        ... )
    """
    rl_name = name or func.__name__
    rl_config = config or DEFAULT_RATE_LIMIT_CONFIG

    limiter = get_rate_limiter(rl_name, config=rl_config, hooks=hooks)

    if key is None:
        extractor = DefaultKeyExtractor()
        key = extractor.extract_key(func, args, kwargs)

    if limiter.acquire(key):
        return await func(*args, **kwargs)

    wait_time = limiter.get_wait_time(key)

    if (
        rl_config.on_limit == RateLimitAction.WAIT
        and wait_time <= rl_config.max_wait_seconds
    ):
        await asyncio.sleep(wait_time)
        if limiter.acquire(key):
            return await func(*args, **kwargs)

    if rl_config.on_limit == RateLimitAction.WARN:
        from common.logging import get_logger
        logger = get_logger("common.rate_limiter")
        logger.warning(
            "Rate limit exceeded, allowing through",
            key=key,
            wait_time=wait_time,
        )
        return await func(*args, **kwargs)

    raise RateLimitExceededError(
        f"Rate limit exceeded for '{key}'",
        key=key,
        current_count=rl_config.max_requests - limiter.get_remaining(key),
        limit=rl_config.max_requests,
        window_seconds=rl_config.window_seconds,
        retry_after_seconds=wait_time,
    )
