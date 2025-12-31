"""Tenant context management.

This module provides context management for multi-tenant operations.
It uses Python's contextvars for thread-safe and async-safe context
propagation, enabling tenant isolation across request boundaries.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from .config import TenantConfig
from .exceptions import (
    NoTenantContextError,
    TenantContextAlreadySetError,
    TenantContextError,
)
from .types import IsolationLevel, TenantStatus, TenantTier

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# Context Variables
# =============================================================================

# Primary context variable for current tenant
_current_tenant: ContextVar[TenantContext | None] = ContextVar(
    "current_tenant", default=None
)

# Stack for nested tenant contexts (useful for admin operations)
_tenant_stack: ContextVar[list[TenantContext]] = ContextVar(
    "tenant_stack", default=[]
)


# =============================================================================
# Tenant Context
# =============================================================================


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Immutable context representing the current tenant.

    This class encapsulates all information about the current tenant
    context, including configuration, permissions, and metadata.

    Attributes:
        tenant_id: The unique identifier for the tenant.
        config: The tenant's configuration.
        user_id: Optional user ID within the tenant.
        correlation_id: Optional request correlation ID.
        metadata: Additional context metadata.
        entered_at: When this context was entered.
        is_admin: Whether this is an admin context (bypasses some checks).
    """

    tenant_id: str
    config: TenantConfig | None = None
    user_id: str | None = None
    correlation_id: str | None = None
    metadata: tuple[tuple[str, Any], ...] = ()
    entered_at: datetime = field(default_factory=datetime.now)
    is_admin: bool = False

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("Tenant ID cannot be empty")

    @property
    def status(self) -> TenantStatus | None:
        """Get the tenant status from config."""
        return self.config.status if self.config else None

    @property
    def tier(self) -> TenantTier | None:
        """Get the tenant tier from config."""
        return self.config.tier if self.config else None

    @property
    def isolation_level(self) -> IsolationLevel | None:
        """Get the isolation level from config."""
        return self.config.isolation_level if self.config else None

    @property
    def is_active(self) -> bool:
        """Check if the tenant is active."""
        return self.config.is_active if self.config else False

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value by key."""
        for k, v in self.metadata:
            if k == key:
                return v
        return default

    def with_user(self, user_id: str) -> TenantContext:
        """Create a new context with a user ID set."""
        return TenantContext(
            tenant_id=self.tenant_id,
            config=self.config,
            user_id=user_id,
            correlation_id=self.correlation_id,
            metadata=self.metadata,
            entered_at=self.entered_at,
            is_admin=self.is_admin,
        )

    def with_correlation_id(self, correlation_id: str) -> TenantContext:
        """Create a new context with a correlation ID set."""
        return TenantContext(
            tenant_id=self.tenant_id,
            config=self.config,
            user_id=self.user_id,
            correlation_id=correlation_id,
            metadata=self.metadata,
            entered_at=self.entered_at,
            is_admin=self.is_admin,
        )

    def with_metadata(self, key: str, value: Any) -> TenantContext:
        """Create a new context with additional metadata."""
        new_metadata = tuple((k, v) for k, v in self.metadata if k != key)
        new_metadata = (*new_metadata, (key, value))
        return TenantContext(
            tenant_id=self.tenant_id,
            config=self.config,
            user_id=self.user_id,
            correlation_id=self.correlation_id,
            metadata=new_metadata,
            entered_at=self.entered_at,
            is_admin=self.is_admin,
        )

    def as_admin(self) -> TenantContext:
        """Create an admin version of this context."""
        return TenantContext(
            tenant_id=self.tenant_id,
            config=self.config,
            user_id=self.user_id,
            correlation_id=self.correlation_id,
            metadata=self.metadata,
            entered_at=self.entered_at,
            is_admin=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "correlation_id": self.correlation_id,
            "is_admin": self.is_admin,
            "entered_at": self.entered_at.isoformat(),
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Context Access Functions
# =============================================================================


def get_current_tenant() -> TenantContext | None:
    """Get the current tenant context.

    Returns:
        The current tenant context, or None if not set.
    """
    return _current_tenant.get()


def get_current_tenant_required() -> TenantContext:
    """Get the current tenant context, raising if not set.

    Returns:
        The current tenant context.

    Raises:
        NoTenantContextError: If no tenant context is set.
    """
    ctx = _current_tenant.get()
    if ctx is None:
        raise NoTenantContextError()
    return ctx


def get_current_tenant_id() -> str | None:
    """Get the current tenant ID.

    Returns:
        The current tenant ID, or None if not set.
    """
    ctx = _current_tenant.get()
    return ctx.tenant_id if ctx else None


def get_current_tenant_id_required() -> str:
    """Get the current tenant ID, raising if not set.

    Returns:
        The current tenant ID.

    Raises:
        NoTenantContextError: If no tenant context is set.
    """
    ctx = get_current_tenant_required()
    return ctx.tenant_id


def is_tenant_context_set() -> bool:
    """Check if a tenant context is currently set.

    Returns:
        True if a tenant context is set, False otherwise.
    """
    return _current_tenant.get() is not None


# =============================================================================
# Context Managers
# =============================================================================


class TenantContextManager:
    """Context manager for tenant context.

    Provides a way to temporarily set the tenant context for a block of code.
    Supports both sync and async usage.

    Example:
        ```python
        with TenantContextManager("tenant_123"):
            # All operations here run in the tenant context
            do_something()

        async with TenantContextManager("tenant_123"):
            # Async operations also supported
            await async_operation()
        ```
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        config: TenantConfig | None = None,
        user_id: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        is_admin: bool = False,
        allow_nested: bool = False,
    ) -> None:
        """Initialize the context manager.

        Args:
            tenant_id: The tenant ID to set.
            config: Optional tenant configuration.
            user_id: Optional user ID.
            correlation_id: Optional correlation ID.
            metadata: Optional metadata dictionary.
            is_admin: Whether this is an admin context.
            allow_nested: Whether to allow nested contexts (pushes to stack).
        """
        self.tenant_id = tenant_id
        self.config = config
        self.user_id = user_id
        self.correlation_id = correlation_id
        self.metadata = tuple(metadata.items()) if metadata else ()
        self.is_admin = is_admin
        self.allow_nested = allow_nested
        self._token: Token[TenantContext | None] | None = None
        self._previous_context: TenantContext | None = None

    def _create_context(self) -> TenantContext:
        """Create the tenant context."""
        return TenantContext(
            tenant_id=self.tenant_id,
            config=self.config,
            user_id=self.user_id,
            correlation_id=self.correlation_id,
            metadata=self.metadata,
            is_admin=self.is_admin,
        )

    def __enter__(self) -> TenantContext:
        """Enter the context."""
        current = _current_tenant.get()
        if current is not None and not self.allow_nested:
            raise TenantContextAlreadySetError(
                current_tenant_id=current.tenant_id,
                new_tenant_id=self.tenant_id,
            )

        # Store previous context if nested
        if current is not None and self.allow_nested:
            self._previous_context = current
            # Push to stack
            stack = _tenant_stack.get()
            _tenant_stack.set([*stack, current])

        context = self._create_context()
        self._token = _current_tenant.set(context)
        return context

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context."""
        if self._token is not None:
            _current_tenant.reset(self._token)
            self._token = None

        # Restore previous context if nested
        if self._previous_context is not None:
            stack = _tenant_stack.get()
            if stack:
                _tenant_stack.set(stack[:-1])
            self._previous_context = None

    async def __aenter__(self) -> TenantContext:
        """Async enter the context."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async exit the context."""
        self.__exit__(exc_type, exc_val, exc_tb)


# Convenience alias
tenant_context = TenantContextManager


class AdminTenantContext(TenantContextManager):
    """Context manager for admin tenant operations.

    Admin contexts bypass certain restrictions and can access
    any tenant's resources. Use with caution.
    """

    def __init__(
        self,
        tenant_id: str,
        *,
        config: TenantConfig | None = None,
        user_id: str | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            tenant_id,
            config=config,
            user_id=user_id,
            correlation_id=correlation_id,
            metadata=metadata,
            is_admin=True,
            allow_nested=True,  # Admin contexts can always be nested
        )


# =============================================================================
# Decorators
# =============================================================================


def require_tenant_context(
    func: Callable[P, R],
) -> Callable[P, R]:
    """Decorator that requires a tenant context to be set.

    Raises NoTenantContextError if no context is set when the
    decorated function is called.

    Example:
        ```python
        @require_tenant_context
        def my_function():
            ctx = get_current_tenant_required()
            # Use ctx.tenant_id
        ```
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_tenant_context_set():
            raise NoTenantContextError(
                f"Function '{func.__name__}' requires a tenant context"
            )
        return func(*args, **kwargs)

    return wrapper


def require_tenant_context_async(
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Async decorator that requires a tenant context to be set.

    Example:
        ```python
        @require_tenant_context_async
        async def my_async_function():
            ctx = get_current_tenant_required()
            await some_operation()
        ```
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_tenant_context_set():
            raise NoTenantContextError(
                f"Async function '{func.__name__}' requires a tenant context"
            )
        return await func(*args, **kwargs)

    return wrapper


def with_tenant(
    tenant_id: str,
    *,
    config: TenantConfig | None = None,
    allow_nested: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that wraps a function in a tenant context.

    Example:
        ```python
        @with_tenant("tenant_123")
        def my_function():
            # Runs in tenant_123 context
            pass
        ```
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with TenantContextManager(
                tenant_id, config=config, allow_nested=allow_nested
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def with_tenant_async(
    tenant_id: str,
    *,
    config: TenantConfig | None = None,
    allow_nested: bool = False,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]:
    """Async decorator that wraps a function in a tenant context.

    Example:
        ```python
        @with_tenant_async("tenant_123")
        async def my_async_function():
            # Runs in tenant_123 context
            await some_operation()
        ```
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            async with TenantContextManager(
                tenant_id, config=config, allow_nested=allow_nested
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Utility Functions
# =============================================================================


def set_tenant_context(
    tenant_id: str,
    *,
    config: TenantConfig | None = None,
    user_id: str | None = None,
    correlation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    is_admin: bool = False,
) -> Token[TenantContext | None]:
    """Manually set the tenant context.

    Returns a token that can be used to reset the context.
    Prefer using TenantContextManager for automatic cleanup.

    Args:
        tenant_id: The tenant ID to set.
        config: Optional tenant configuration.
        user_id: Optional user ID.
        correlation_id: Optional correlation ID.
        metadata: Optional metadata dictionary.
        is_admin: Whether this is an admin context.

    Returns:
        A token that can be used to reset the context.
    """
    context = TenantContext(
        tenant_id=tenant_id,
        config=config,
        user_id=user_id,
        correlation_id=correlation_id,
        metadata=tuple(metadata.items()) if metadata else (),
        is_admin=is_admin,
    )
    return _current_tenant.set(context)


def reset_tenant_context(token: Token[TenantContext | None]) -> None:
    """Reset the tenant context using a token.

    Args:
        token: The token from set_tenant_context.
    """
    _current_tenant.reset(token)


def clear_tenant_context() -> None:
    """Clear the current tenant context.

    This sets the context to None. Use with caution as it may
    break code that expects a context to be set.
    """
    _current_tenant.set(None)


def get_tenant_stack() -> list[TenantContext]:
    """Get the current tenant context stack.

    Returns a copy of the stack for nested contexts.
    """
    return list(_tenant_stack.get())


def copy_context_to_thread(context: TenantContext) -> Token[TenantContext | None]:
    """Copy a tenant context to the current thread.

    Useful when spawning new threads that need the same context.

    Args:
        context: The context to copy.

    Returns:
        A token that can be used to reset the context.
    """
    return _current_tenant.set(context)


# =============================================================================
# Context Propagation Helpers
# =============================================================================


class TenantContextPropagator:
    """Helper for propagating tenant context across boundaries.

    Useful for propagating context to:
    - New threads
    - Background tasks
    - External services (via headers)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def capture(self) -> dict[str, Any]:
        """Capture the current tenant context for propagation.

        Returns:
            A dictionary containing the captured context.
        """
        ctx = get_current_tenant()
        if ctx is None:
            return {}
        return {
            "tenant_id": ctx.tenant_id,
            "user_id": ctx.user_id,
            "correlation_id": ctx.correlation_id,
            "is_admin": ctx.is_admin,
            "metadata": dict(ctx.metadata),
        }

    def restore(self, captured: dict[str, Any]) -> TenantContextManager | None:
        """Create a context manager from captured context.

        Args:
            captured: The captured context from capture().

        Returns:
            A context manager, or None if the captured context is empty.
        """
        if not captured or "tenant_id" not in captured:
            return None
        return TenantContextManager(
            tenant_id=captured["tenant_id"],
            user_id=captured.get("user_id"),
            correlation_id=captured.get("correlation_id"),
            metadata=captured.get("metadata"),
            is_admin=captured.get("is_admin", False),
        )

    def inject_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Inject tenant context into HTTP headers.

        Args:
            headers: The headers dictionary to modify.

        Returns:
            The modified headers dictionary.
        """
        ctx = get_current_tenant()
        if ctx is None:
            return headers

        headers["X-Tenant-ID"] = ctx.tenant_id
        if ctx.user_id:
            headers["X-User-ID"] = ctx.user_id
        if ctx.correlation_id:
            headers["X-Correlation-ID"] = ctx.correlation_id
        if ctx.is_admin:
            headers["X-Admin-Context"] = "true"

        return headers

    def extract_from_headers(
        self, headers: Mapping[str, str]
    ) -> TenantContextManager | None:
        """Extract tenant context from HTTP headers.

        Args:
            headers: The HTTP headers to extract from.

        Returns:
            A context manager, or None if no tenant ID found.
        """
        tenant_id = headers.get("X-Tenant-ID")
        if not tenant_id:
            return None

        return TenantContextManager(
            tenant_id=tenant_id,
            user_id=headers.get("X-User-ID"),
            correlation_id=headers.get("X-Correlation-ID"),
            is_admin=headers.get("X-Admin-Context", "").lower() == "true",
        )


# Global propagator instance
context_propagator = TenantContextPropagator()
