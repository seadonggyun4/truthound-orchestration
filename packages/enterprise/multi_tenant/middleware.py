"""Tenant middleware for request processing.

This module provides middleware components for resolving tenant context
from HTTP requests and other sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from .context import TenantContext, TenantContextManager, get_current_tenant
from .exceptions import TenantNotFoundError, TenantResolutionError
from .types import TenantResolutionStrategy

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from .config import TenantConfig
    from .registry import TenantRegistry

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# Tenant Resolvers
# =============================================================================


class HeaderTenantResolver:
    """Resolves tenant from HTTP headers.

    Looks for the tenant ID in a specified header.
    """

    def __init__(
        self,
        header_name: str = "X-Tenant-ID",
        *,
        required: bool = True,
    ) -> None:
        self.header_name = header_name
        self.required = required

    @property
    def strategy(self) -> TenantResolutionStrategy:
        return TenantResolutionStrategy.HEADER

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from headers."""
        headers = context.get("headers", {})
        # Check both exact case and lowercase
        tenant_id = headers.get(self.header_name) or headers.get(
            self.header_name.lower()
        )
        return tenant_id


class SubdomainTenantResolver:
    """Resolves tenant from request subdomain.

    Extracts tenant ID from the subdomain of the request host.
    For example: tenant123.example.com -> tenant123
    """

    def __init__(
        self,
        *,
        base_domain: str | None = None,
        subdomain_index: int = 0,
        required: bool = True,
    ) -> None:
        self.base_domain = base_domain
        self.subdomain_index = subdomain_index
        self.required = required

    @property
    def strategy(self) -> TenantResolutionStrategy:
        return TenantResolutionStrategy.SUBDOMAIN

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from subdomain."""
        host = context.get("host", "")
        if not host:
            return None

        # Remove port if present
        host = host.split(":")[0]

        # Extract subdomain
        parts = host.split(".")
        if len(parts) < 2:
            return None

        # Get subdomain at specified index
        if self.subdomain_index >= len(parts):
            return None

        return parts[self.subdomain_index]


class PathTenantResolver:
    """Resolves tenant from URL path.

    Extracts tenant ID from the URL path using a pattern.
    For example: /tenants/123/resources -> 123
    """

    def __init__(
        self,
        pattern: str = r"/tenants/([^/]+)",
        *,
        required: bool = True,
    ) -> None:
        self.pattern = re.compile(pattern)
        self.required = required

    @property
    def strategy(self) -> TenantResolutionStrategy:
        return TenantResolutionStrategy.PATH

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from path."""
        path = context.get("path", "")
        if not path:
            return None

        match = self.pattern.search(path)
        if match:
            return match.group(1)

        return None


class QueryParamTenantResolver:
    """Resolves tenant from query parameters.

    Extracts tenant ID from a query parameter.
    """

    def __init__(
        self,
        param_name: str = "tenant_id",
        *,
        required: bool = True,
    ) -> None:
        self.param_name = param_name
        self.required = required

    @property
    def strategy(self) -> TenantResolutionStrategy:
        return TenantResolutionStrategy.QUERY_PARAM

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from query parameters."""
        params = context.get("query_params", {})
        return params.get(self.param_name)


class JWTClaimTenantResolver:
    """Resolves tenant from JWT token claims.

    Extracts tenant ID from a JWT claim.
    """

    def __init__(
        self,
        claim_name: str = "tenant_id",
        *,
        required: bool = True,
    ) -> None:
        self.claim_name = claim_name
        self.required = required

    @property
    def strategy(self) -> TenantResolutionStrategy:
        return TenantResolutionStrategy.JWT_CLAIM

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from JWT claims."""
        claims = context.get("jwt_claims", {})
        return claims.get(self.claim_name)


class ContextTenantResolver:
    """Resolves tenant from already-set context.

    Uses the current tenant context if available.
    """

    def __init__(self, *, required: bool = True) -> None:
        self.required = required

    @property
    def strategy(self) -> TenantResolutionStrategy:
        return TenantResolutionStrategy.CONTEXT

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from current context."""
        ctx = get_current_tenant()
        return ctx.tenant_id if ctx else None


class CompositeTenantResolver:
    """Resolves tenant using multiple strategies in order.

    Tries each resolver until one succeeds.
    """

    def __init__(
        self,
        resolvers: Sequence[Any],  # TenantResolver
        *,
        required: bool = True,
    ) -> None:
        self.resolvers = list(resolvers)
        self.required = required

    @property
    def strategy(self) -> TenantResolutionStrategy:
        return TenantResolutionStrategy.COMPOSITE

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID using the first successful resolver."""
        for resolver in self.resolvers:
            tenant_id = resolver.resolve(context)
            if tenant_id:
                return tenant_id
        return None

    def add_resolver(self, resolver: Any) -> None:
        """Add a resolver to the chain."""
        self.resolvers.append(resolver)


# =============================================================================
# Middleware
# =============================================================================


@dataclass
class TenantMiddleware:
    """Middleware for resolving and setting tenant context.

    Can be used with any web framework by adapting the request context.
    """

    resolver: Any  # TenantResolver
    registry: TenantRegistry | None = None
    default_tenant_id: str | None = None
    require_tenant: bool = True
    validate_tenant: bool = True

    def resolve_tenant(
        self,
        request_context: Mapping[str, Any],
    ) -> tuple[str | None, TenantConfig | None]:
        """Resolve tenant from request context.

        Args:
            request_context: Dictionary containing request info (headers, path, etc.)

        Returns:
            Tuple of (tenant_id, tenant_config).

        Raises:
            TenantResolutionError: If tenant cannot be resolved and is required.
            TenantNotFoundError: If tenant is resolved but not found in registry.
        """
        # Try to resolve tenant ID
        tenant_id = self.resolver.resolve(request_context)

        # Fall back to default
        if not tenant_id:
            tenant_id = self.default_tenant_id

        # Check if required
        if not tenant_id:
            if self.require_tenant:
                raise TenantResolutionError(
                    resolution_strategy=self.resolver.strategy.value
                )
            return None, None

        # Validate tenant exists
        tenant_config = None
        if self.validate_tenant and self.registry:
            tenant_config = self.registry.get_tenant_or_none(tenant_id)
            if not tenant_config and self.require_tenant:
                raise TenantNotFoundError(tenant_id)

        return tenant_id, tenant_config

    def get_context_manager(
        self,
        request_context: Mapping[str, Any],
    ) -> TenantContextManager | None:
        """Get a context manager for the resolved tenant.

        Args:
            request_context: Dictionary containing request info.

        Returns:
            A TenantContextManager, or None if no tenant resolved.
        """
        tenant_id, tenant_config = self.resolve_tenant(request_context)
        if not tenant_id:
            return None

        # Extract additional context
        user_id = request_context.get("user_id")
        correlation_id = request_context.get("correlation_id") or request_context.get(
            "headers", {}
        ).get("X-Correlation-ID")

        return TenantContextManager(
            tenant_id=tenant_id,
            config=tenant_config,
            user_id=user_id,
            correlation_id=correlation_id,
        )


# =============================================================================
# Decorators
# =============================================================================


def with_tenant_middleware(
    middleware: TenantMiddleware,
    *,
    context_extractor: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that applies tenant middleware to a function.

    Args:
        middleware: The tenant middleware to use.
        context_extractor: Function to extract request context from args/kwargs.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Extract context
            if context_extractor:
                request_context = context_extractor(*args, **kwargs)
            else:
                # Try to extract from first positional arg or 'request' kwarg
                request = kwargs.get("request") or (args[0] if args else None)
                request_context = _extract_context_from_request(request)

            # Get context manager
            ctx_manager = middleware.get_context_manager(request_context)

            if ctx_manager:
                with ctx_manager:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def with_tenant_middleware_async(
    middleware: TenantMiddleware,
    *,
    context_extractor: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[
    [Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]
]:
    """Async decorator that applies tenant middleware.

    Args:
        middleware: The tenant middleware to use.
        context_extractor: Function to extract request context from args/kwargs.

    Returns:
        Decorated async function.
    """

    def decorator(
        func: Callable[P, Awaitable[R]]
    ) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Extract context
            if context_extractor:
                request_context = context_extractor(*args, **kwargs)
            else:
                request = kwargs.get("request") or (args[0] if args else None)
                request_context = _extract_context_from_request(request)

            # Get context manager
            ctx_manager = middleware.get_context_manager(request_context)

            if ctx_manager:
                async with ctx_manager:
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_context_from_request(request: Any) -> dict[str, Any]:
    """Try to extract context from common request object types."""
    context: dict[str, Any] = {}

    if request is None:
        return context

    # Try common attributes
    if hasattr(request, "headers"):
        headers = request.headers
        if hasattr(headers, "items"):
            context["headers"] = dict(headers.items())
        elif isinstance(headers, dict):
            context["headers"] = headers

    if hasattr(request, "path"):
        context["path"] = request.path
    elif hasattr(request, "url"):
        url = request.url
        if hasattr(url, "path"):
            context["path"] = url.path
        else:
            context["path"] = str(url)

    if hasattr(request, "host"):
        context["host"] = request.host
    elif hasattr(request, "headers"):
        context["host"] = context.get("headers", {}).get("host")

    if hasattr(request, "query_params"):
        qp = request.query_params
        if hasattr(qp, "items"):
            context["query_params"] = dict(qp.items())
        elif isinstance(qp, dict):
            context["query_params"] = qp

    # JWT claims if available
    if hasattr(request, "state") and hasattr(request.state, "user"):
        context["jwt_claims"] = getattr(request.state.user, "claims", {})

    return context


def create_default_middleware(
    registry: TenantRegistry | None = None,
    *,
    header_name: str = "X-Tenant-ID",
    query_param: str = "tenant_id",
    default_tenant_id: str | None = None,
    require_tenant: bool = True,
) -> TenantMiddleware:
    """Create middleware with sensible defaults.

    Uses a composite resolver that tries:
    1. Header (X-Tenant-ID)
    2. Query parameter (tenant_id)
    3. Current context

    Args:
        registry: Optional tenant registry for validation.
        header_name: Header name for tenant ID.
        query_param: Query parameter name for tenant ID.
        default_tenant_id: Default tenant ID if none resolved.
        require_tenant: Whether to require a tenant.

    Returns:
        Configured TenantMiddleware.
    """
    resolver = CompositeTenantResolver([
        HeaderTenantResolver(header_name),
        QueryParamTenantResolver(query_param),
        ContextTenantResolver(required=False),
    ])

    return TenantMiddleware(
        resolver=resolver,
        registry=registry,
        default_tenant_id=default_tenant_id,
        require_tenant=require_tenant,
    )
