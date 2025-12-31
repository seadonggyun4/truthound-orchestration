"""Tests for tenant context management."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ..config import TenantConfig
from ..context import (
    AdminTenantContext,
    TenantContext,
    TenantContextManager,
    TenantContextPropagator,
    clear_tenant_context,
    get_current_tenant,
    get_current_tenant_id,
    get_current_tenant_id_required,
    get_current_tenant_required,
    get_tenant_stack,
    is_tenant_context_set,
    require_tenant_context,
    require_tenant_context_async,
    set_tenant_context,
    reset_tenant_context,
    with_tenant,
    with_tenant_async,
)
from ..exceptions import NoTenantContextError, TenantContextAlreadySetError
from ..types import TenantStatus


class TestTenantContext:
    """Tests for TenantContext dataclass."""

    def test_create_context(self) -> None:
        """Test creating a tenant context."""
        ctx = TenantContext(tenant_id="test-tenant")
        assert ctx.tenant_id == "test-tenant"
        assert ctx.user_id is None
        assert ctx.is_admin is False

    def test_context_with_config(
        self, sample_tenant_config: TenantConfig
    ) -> None:
        """Test context with configuration."""
        ctx = TenantContext(
            tenant_id="test-tenant",
            config=sample_tenant_config,
        )
        assert ctx.config == sample_tenant_config
        assert ctx.status == TenantStatus.ACTIVE
        assert ctx.is_active is True

    def test_with_user(self) -> None:
        """Test adding user to context."""
        ctx = TenantContext(tenant_id="test-tenant")
        with_user = ctx.with_user("user-123")
        assert with_user.user_id == "user-123"
        assert ctx.user_id is None  # Original unchanged

    def test_with_correlation_id(self) -> None:
        """Test adding correlation ID."""
        ctx = TenantContext(tenant_id="test-tenant")
        with_corr = ctx.with_correlation_id("req-abc")
        assert with_corr.correlation_id == "req-abc"

    def test_with_metadata(self) -> None:
        """Test adding metadata."""
        ctx = TenantContext(tenant_id="test-tenant")
        with_meta = ctx.with_metadata("key", "value")
        assert with_meta.get_metadata("key") == "value"
        assert ctx.get_metadata("key") is None

    def test_as_admin(self) -> None:
        """Test creating admin context."""
        ctx = TenantContext(tenant_id="test-tenant")
        admin = ctx.as_admin()
        assert admin.is_admin is True
        assert ctx.is_admin is False

    def test_empty_tenant_id_raises(self) -> None:
        """Test that empty tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="Tenant ID cannot be empty"):
            TenantContext(tenant_id="")


class TestTenantContextManager:
    """Tests for TenantContextManager."""

    def test_basic_context_manager(self) -> None:
        """Test basic context manager usage."""
        assert get_current_tenant() is None

        with TenantContextManager("test-tenant") as ctx:
            assert ctx.tenant_id == "test-tenant"
            assert get_current_tenant_id() == "test-tenant"

        assert get_current_tenant() is None

    def test_context_with_config(
        self, sample_tenant_config: TenantConfig
    ) -> None:
        """Test context manager with configuration."""
        with TenantContextManager(
            "test-tenant",
            config=sample_tenant_config,
        ) as ctx:
            assert ctx.config == sample_tenant_config

    def test_context_with_user(self) -> None:
        """Test context manager with user."""
        with TenantContextManager(
            "test-tenant",
            user_id="user-123",
        ) as ctx:
            assert ctx.user_id == "user-123"

    def test_nested_context_not_allowed_by_default(self) -> None:
        """Test that nested contexts raise by default."""
        with TenantContextManager("tenant-1"):
            with pytest.raises(TenantContextAlreadySetError):
                with TenantContextManager("tenant-2"):
                    pass

    def test_nested_context_with_allow_nested(self) -> None:
        """Test nested contexts with allow_nested=True."""
        with TenantContextManager("tenant-1") as ctx1:
            assert get_current_tenant_id() == "tenant-1"
            with TenantContextManager("tenant-2", allow_nested=True) as ctx2:
                assert get_current_tenant_id() == "tenant-2"
            assert get_current_tenant_id() == "tenant-1"

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager usage."""
        assert get_current_tenant() is None

        async with TenantContextManager("test-tenant") as ctx:
            assert ctx.tenant_id == "test-tenant"
            assert get_current_tenant_id() == "test-tenant"

        assert get_current_tenant() is None


class TestAdminTenantContext:
    """Tests for AdminTenantContext."""

    def test_admin_context(self) -> None:
        """Test admin context is marked as admin."""
        with AdminTenantContext("test-tenant") as ctx:
            assert ctx.is_admin is True

    def test_admin_context_allows_nesting(self) -> None:
        """Test admin context can always be nested."""
        with TenantContextManager("tenant-1"):
            with AdminTenantContext("tenant-2") as ctx:
                assert ctx.tenant_id == "tenant-2"
                assert ctx.is_admin is True


class TestContextAccessFunctions:
    """Tests for context access functions."""

    def test_get_current_tenant_when_not_set(self) -> None:
        """Test get_current_tenant returns None when not set."""
        clear_tenant_context()
        assert get_current_tenant() is None

    def test_get_current_tenant_required_when_not_set(self) -> None:
        """Test get_current_tenant_required raises when not set."""
        clear_tenant_context()
        with pytest.raises(NoTenantContextError):
            get_current_tenant_required()

    def test_get_current_tenant_id_when_not_set(self) -> None:
        """Test get_current_tenant_id returns None when not set."""
        clear_tenant_context()
        assert get_current_tenant_id() is None

    def test_get_current_tenant_id_required_when_not_set(self) -> None:
        """Test get_current_tenant_id_required raises when not set."""
        clear_tenant_context()
        with pytest.raises(NoTenantContextError):
            get_current_tenant_id_required()

    def test_is_tenant_context_set(self) -> None:
        """Test is_tenant_context_set function."""
        assert is_tenant_context_set() is False
        with TenantContextManager("test"):
            assert is_tenant_context_set() is True
        assert is_tenant_context_set() is False


class TestManualContextControl:
    """Tests for manual context control functions."""

    def test_set_and_reset_context(self) -> None:
        """Test manual set and reset."""
        token = set_tenant_context("test-tenant")
        assert get_current_tenant_id() == "test-tenant"
        reset_tenant_context(token)
        assert get_current_tenant() is None

    def test_clear_tenant_context(self) -> None:
        """Test clearing context."""
        with TenantContextManager("test"):
            clear_tenant_context()
            assert get_current_tenant() is None


class TestContextDecorators:
    """Tests for context decorators."""

    def test_require_tenant_context(self) -> None:
        """Test require_tenant_context decorator."""

        @require_tenant_context
        def my_func() -> str:
            return get_current_tenant_id_required()

        with pytest.raises(NoTenantContextError):
            my_func()

        with TenantContextManager("test-tenant"):
            result = my_func()
            assert result == "test-tenant"

    @pytest.mark.asyncio
    async def test_require_tenant_context_async(self) -> None:
        """Test require_tenant_context_async decorator."""

        @require_tenant_context_async
        async def my_async_func() -> str:
            return get_current_tenant_id_required()

        with pytest.raises(NoTenantContextError):
            await my_async_func()

        async with TenantContextManager("test-tenant"):
            result = await my_async_func()
            assert result == "test-tenant"

    def test_with_tenant_decorator(self) -> None:
        """Test with_tenant decorator."""

        @with_tenant("fixed-tenant")
        def my_func() -> str:
            return get_current_tenant_id_required()

        result = my_func()
        assert result == "fixed-tenant"
        assert get_current_tenant() is None  # Context cleared after

    @pytest.mark.asyncio
    async def test_with_tenant_async_decorator(self) -> None:
        """Test with_tenant_async decorator."""

        @with_tenant_async("fixed-tenant")
        async def my_async_func() -> str:
            return get_current_tenant_id_required()

        result = await my_async_func()
        assert result == "fixed-tenant"
        assert get_current_tenant() is None


class TestContextPropagator:
    """Tests for TenantContextPropagator."""

    def test_capture_and_restore(self) -> None:
        """Test capturing and restoring context."""
        propagator = TenantContextPropagator()

        with TenantContextManager("test-tenant", user_id="user-1"):
            captured = propagator.capture()

        assert captured["tenant_id"] == "test-tenant"
        assert captured["user_id"] == "user-1"

        # Restore in a new context
        ctx_manager = propagator.restore(captured)
        assert ctx_manager is not None

        with ctx_manager:
            assert get_current_tenant_id() == "test-tenant"

    def test_capture_empty_context(self) -> None:
        """Test capturing when no context set."""
        propagator = TenantContextPropagator()
        clear_tenant_context()
        captured = propagator.capture()
        assert captured == {}

    def test_restore_empty_captured(self) -> None:
        """Test restoring empty captured context."""
        propagator = TenantContextPropagator()
        ctx_manager = propagator.restore({})
        assert ctx_manager is None

    def test_inject_headers(self) -> None:
        """Test injecting context into headers."""
        propagator = TenantContextPropagator()

        with TenantContextManager(
            "test-tenant",
            user_id="user-1",
            correlation_id="req-123",
        ):
            headers: dict[str, str] = {}
            propagator.inject_headers(headers)

        assert headers["X-Tenant-ID"] == "test-tenant"
        assert headers["X-User-ID"] == "user-1"
        assert headers["X-Correlation-ID"] == "req-123"

    def test_extract_from_headers(self) -> None:
        """Test extracting context from headers."""
        propagator = TenantContextPropagator()
        headers = {
            "X-Tenant-ID": "test-tenant",
            "X-User-ID": "user-1",
        }

        ctx_manager = propagator.extract_from_headers(headers)
        assert ctx_manager is not None

        with ctx_manager as ctx:
            assert ctx.tenant_id == "test-tenant"
            assert ctx.user_id == "user-1"


class TestThreadSafety:
    """Tests for thread safety of context management."""

    def test_context_isolation_between_threads(self) -> None:
        """Test that contexts are isolated between threads."""
        results: dict[str, str | None] = {}

        def thread_func(tenant_id: str) -> None:
            with TenantContextManager(tenant_id):
                results[tenant_id] = get_current_tenant_id()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(thread_func, f"tenant-{i}")
                for i in range(10)
            ]
            for f in futures:
                f.result()

        # Each thread should have seen its own tenant
        for i in range(10):
            assert results[f"tenant-{i}"] == f"tenant-{i}"

    @pytest.mark.asyncio
    async def test_context_isolation_between_tasks(self) -> None:
        """Test that contexts are isolated between async tasks."""
        results: dict[str, str | None] = {}

        async def task_func(tenant_id: str) -> None:
            async with TenantContextManager(tenant_id):
                await asyncio.sleep(0.01)  # Simulate async work
                results[tenant_id] = get_current_tenant_id()

        await asyncio.gather(
            *[task_func(f"tenant-{i}") for i in range(10)]
        )

        # Each task should have seen its own tenant
        for i in range(10):
            assert results[f"tenant-{i}"] == f"tenant-{i}"
