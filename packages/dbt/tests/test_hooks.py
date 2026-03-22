"""Tests for lifecycle hooks."""

import pytest
import logging

from truthound_dbt.hooks import (
    DbtHook,
    AsyncDbtHook,
    BaseDbtHook,
    LoggingDbtHook,
    MetricsDbtHook,
    CompositeDbtHook,
    AsyncLoggingDbtHook,
    AsyncCompositeDbtHook,
    SyncToAsyncDbtHookAdapter,
    TestStartEvent,
    TestEndEvent,
    ParseStartEvent,
    ParseEndEvent,
    ConversionStartEvent,
    ConversionEndEvent,
    GenerationStartEvent,
    GenerationEndEvent,
    HookExecutionError,
)


class TestDbtEvents:
    """Tests for dbt event types."""

    def test_test_start_event(self):
        """Test TestStartEvent creation."""
        event = TestStartEvent(
            test_name="not_null_users_id",
            model="users",
            column="id",
            rule_type="not_null",
        )
        assert event.test_name == "not_null_users_id"
        assert event.model == "users"
        assert event.column == "id"
        assert event.rule_type == "not_null"
        assert event.timestamp is not None

    def test_test_end_event(self):
        """Test TestEndEvent creation."""
        event = TestEndEvent(
            test_name="not_null_users_id",
            passed=True,
            duration_ms=150.5,
            failures=0,
        )
        assert event.test_name == "not_null_users_id"
        assert event.passed is True
        assert event.duration_ms == 150.5
        assert event.failures == 0

    def test_parse_start_event(self):
        """Test ParseStartEvent creation."""
        event = ParseStartEvent(manifest_path="/path/to/manifest.json")
        assert event.manifest_path == "/path/to/manifest.json"

    def test_parse_end_event(self):
        """Test ParseEndEvent creation."""
        event = ParseEndEvent(
            manifest_path="/path/to/manifest.json",
            success=True,
            duration_ms=500.0,
            test_count=10,
            model_count=5,
        )
        assert event.success is True
        assert event.test_count == 10

    def test_conversion_start_event(self):
        """Test ConversionStartEvent creation."""
        event = ConversionStartEvent(
            rule_count=5,
            adapter_name="postgres",
        )
        assert event.rule_count == 5
        assert event.adapter_name == "postgres"

    def test_conversion_end_event(self):
        """Test ConversionEndEvent creation."""
        event = ConversionEndEvent(
            success=True,
            duration_ms=100.0,
            converted_count=5,
            error_count=0,
        )
        assert event.converted_count == 5
        assert event.error_count == 0

    def test_generation_events(self):
        """Test generation events."""
        start = GenerationStartEvent(model="users", generation_type="sql")
        assert start.model == "users"
        assert start.generation_type == "sql"

        end = GenerationEndEvent(
            model="users",
            generation_type="sql",
            success=True,
            duration_ms=50.0,
        )
        assert end.success is True


class TestBaseDbtHook:
    """Tests for BaseDbtHook."""

    def test_all_methods_are_no_ops(self):
        """Test base hook methods do nothing."""
        hook = BaseDbtHook()

        # These should not raise
        hook.on_test_start(TestStartEvent(test_name="test"))
        hook.on_test_end(TestEndEvent(test_name="test"))
        hook.on_parse_start(ParseStartEvent())
        hook.on_parse_end(ParseEndEvent())
        hook.on_conversion_start(ConversionStartEvent())
        hook.on_conversion_end(ConversionEndEvent())
        hook.on_generation_start(GenerationStartEvent())
        hook.on_generation_end(GenerationEndEvent())


class TestLoggingDbtHook:
    """Tests for LoggingDbtHook."""

    def test_logs_test_start(self, caplog):
        """Test logging test start."""
        hook = LoggingDbtHook(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            hook.on_test_start(TestStartEvent(
                test_name="not_null_test",
                model="users",
            ))

        assert "Test started: not_null_test" in caplog.text

    def test_logs_test_end_passed(self, caplog):
        """Test logging test end (passed)."""
        hook = LoggingDbtHook(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            hook.on_test_end(TestEndEvent(
                test_name="not_null_test",
                passed=True,
                duration_ms=100.0,
            ))

        assert "PASSED" in caplog.text

    def test_logs_test_end_failed(self, caplog):
        """Test logging test end (failed)."""
        hook = LoggingDbtHook(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            hook.on_test_end(TestEndEvent(
                test_name="not_null_test",
                passed=False,
                duration_ms=100.0,
                failures=5,
            ))

        assert "FAILED" in caplog.text

    def test_logs_parse_events(self, caplog):
        """Test logging parse events."""
        hook = LoggingDbtHook(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            hook.on_parse_start(ParseStartEvent(manifest_path="test.json"))
            hook.on_parse_end(ParseEndEvent(
                manifest_path="test.json",
                success=True,
                test_count=10,
            ))

        assert "Parsing manifest" in caplog.text
        assert "Parsing completed" in caplog.text


class TestMetricsDbtHook:
    """Tests for MetricsDbtHook."""

    def setup_method(self):
        """Create hook instance."""
        self.hook = MetricsDbtHook()

    def test_tracks_test_start(self):
        """Test tracking test starts."""
        self.hook.on_test_start(TestStartEvent(test_name="test"))
        assert self.hook.metrics.tests_started == 1

    def test_tracks_test_end(self):
        """Test tracking test ends."""
        self.hook.on_test_end(TestEndEvent(
            test_name="test",
            passed=True,
            duration_ms=100.0,
        ))
        assert self.hook.metrics.tests_completed == 1
        assert self.hook.metrics.tests_passed == 1
        assert self.hook.metrics.total_test_duration_ms == 100.0

    def test_tracks_failed_tests(self):
        """Test tracking failed tests."""
        self.hook.on_test_end(TestEndEvent(
            test_name="test",
            passed=False,
        ))
        assert self.hook.metrics.tests_failed == 1

    def test_tracks_parse_events(self):
        """Test tracking parse events."""
        self.hook.on_parse_end(ParseEndEvent(duration_ms=500.0))
        assert self.hook.metrics.parse_count == 1
        assert self.hook.metrics.parse_duration_ms == 500.0

    def test_tracks_conversion_events(self):
        """Test tracking conversion events."""
        self.hook.on_conversion_end(ConversionEndEvent(
            converted_count=5,
            error_count=1,
        ))
        assert self.hook.metrics.conversion_count == 1
        assert self.hook.metrics.rules_converted == 5
        assert self.hook.metrics.conversion_errors == 1

    def test_tracks_generation_events(self):
        """Test tracking generation events."""
        self.hook.on_generation_end(GenerationEndEvent(duration_ms=50.0))
        assert self.hook.metrics.generation_count == 1
        assert self.hook.metrics.generation_duration_ms == 50.0

    def test_success_rate(self):
        """Test success rate calculation."""
        self.hook.on_test_end(TestEndEvent(test_name="t1", passed=True))
        self.hook.on_test_end(TestEndEvent(test_name="t2", passed=True))
        self.hook.on_test_end(TestEndEvent(test_name="t3", passed=False))

        assert self.hook.metrics.test_success_rate == pytest.approx(66.67, rel=0.1)

    def test_average_duration(self):
        """Test average duration calculation."""
        self.hook.on_test_end(TestEndEvent(test_name="t1", duration_ms=100.0))
        self.hook.on_test_end(TestEndEvent(test_name="t2", duration_ms=200.0))

        assert self.hook.metrics.average_test_duration_ms == 150.0

    def test_reset(self):
        """Test resetting metrics."""
        self.hook.on_test_end(TestEndEvent(test_name="t1", passed=True))
        self.hook.reset()

        assert self.hook.metrics.tests_completed == 0
        assert self.hook.metrics.tests_passed == 0

    def test_metrics_to_dict(self):
        """Test metrics to_dict method."""
        self.hook.on_test_end(TestEndEvent(test_name="t1", passed=True))
        data = self.hook.metrics.to_dict()

        assert "tests_completed" in data
        assert "test_success_rate" in data
        assert data["tests_completed"] == 1


class TestCompositeDbtHook:
    """Tests for CompositeDbtHook."""

    def test_calls_all_hooks(self):
        """Test composite calls all hooks."""
        metrics1 = MetricsDbtHook()
        metrics2 = MetricsDbtHook()
        composite = CompositeDbtHook([metrics1, metrics2])

        composite.on_test_start(TestStartEvent(test_name="test"))

        assert metrics1.metrics.tests_started == 1
        assert metrics2.metrics.tests_started == 1

    def test_add_hook(self):
        """Test adding hook to composite."""
        composite = CompositeDbtHook([])
        metrics = MetricsDbtHook()
        composite.add_hook(metrics)

        composite.on_test_start(TestStartEvent(test_name="test"))
        assert metrics.metrics.tests_started == 1

    def test_remove_hook(self):
        """Test removing hook from composite."""
        metrics = MetricsDbtHook()
        composite = CompositeDbtHook([metrics])
        composite.remove_hook(metrics)

        composite.on_test_start(TestStartEvent(test_name="test"))
        assert metrics.metrics.tests_started == 0

    def test_ignores_errors_by_default(self):
        """Test composite ignores hook errors by default."""
        class BrokenHook(BaseDbtHook):
            def on_test_start(self, event):
                raise RuntimeError("Broken!")

        metrics = MetricsDbtHook()
        composite = CompositeDbtHook([BrokenHook(), metrics])

        # Should not raise, and should call remaining hooks
        composite.on_test_start(TestStartEvent(test_name="test"))
        assert metrics.metrics.tests_started == 1

    def test_raises_errors_when_configured(self):
        """Test composite raises errors when configured."""
        class BrokenHook(BaseDbtHook):
            def on_test_start(self, event):
                raise RuntimeError("Broken!")

        composite = CompositeDbtHook([BrokenHook()], ignore_errors=False)

        with pytest.raises(HookExecutionError) as exc_info:
            composite.on_test_start(TestStartEvent(test_name="test"))

        assert "BrokenHook" in str(exc_info.value)


class TestAsyncHooks:
    """Tests for async hooks."""

    @pytest.mark.asyncio
    async def test_async_logging_hook(self, caplog):
        """Test async logging hook."""
        hook = AsyncLoggingDbtHook(level=logging.INFO)

        with caplog.at_level(logging.INFO):
            await hook.on_test_start(TestStartEvent(test_name="test"))
            await hook.on_test_end(TestEndEvent(test_name="test", passed=True))

        assert "Test started" in caplog.text
        assert "PASSED" in caplog.text

    @pytest.mark.asyncio
    async def test_async_composite_hook(self):
        """Test async composite hook."""
        hook1 = AsyncLoggingDbtHook()
        hook2 = AsyncLoggingDbtHook()
        composite = AsyncCompositeDbtHook([hook1, hook2])

        # Should not raise
        await composite.on_test_start(TestStartEvent(test_name="test"))
        await composite.on_test_end(TestEndEvent(test_name="test", passed=True))

    @pytest.mark.asyncio
    async def test_sync_to_async_adapter(self):
        """Test sync to async adapter."""
        sync_hook = MetricsDbtHook()
        async_hook = SyncToAsyncDbtHookAdapter(sync_hook)

        await async_hook.on_test_start(TestStartEvent(test_name="test"))
        await async_hook.on_test_end(TestEndEvent(test_name="test", passed=True))

        assert sync_hook.metrics.tests_started == 1
        assert sync_hook.metrics.tests_passed == 1
