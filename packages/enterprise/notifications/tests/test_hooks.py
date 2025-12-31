"""Tests for notification hooks."""

import pytest

from packages.enterprise.notifications.hooks import (
    CallbackNotificationHook,
    CompositeNotificationHook,
    FilteringNotificationHook,
    LoggingNotificationHook,
    MetricsNotificationHook,
    ThrottlingNotificationHook,
)
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
    NotificationStatus,
)


class TestLoggingNotificationHook:
    """Tests for LoggingNotificationHook."""

    def test_on_before_send(self) -> None:
        """Test before send logging."""
        hook = LoggingNotificationHook()

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "payload": NotificationPayload(message="Test"),
        }

        # Should not raise
        hook.on_before_send(context)

    def test_on_after_send_success(self) -> None:
        """Test after send logging on success."""
        hook = LoggingNotificationHook()

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "duration_ms": 100.0,
            "result": NotificationResult.success_result(
                NotificationChannel.SLACK, "test"
            ),
        }

        # Should not raise
        hook.on_after_send(context, success=True)

    def test_on_after_send_failure(self) -> None:
        """Test after send logging on failure."""
        hook = LoggingNotificationHook()

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "duration_ms": 100.0,
            "result": NotificationResult.failure_result(
                NotificationChannel.SLACK, "test", "Error"
            ),
        }

        # Should not raise
        hook.on_after_send(context, success=False)


class TestMetricsNotificationHook:
    """Tests for MetricsNotificationHook."""

    def test_stats_tracking(self) -> None:
        """Test that stats are tracked."""
        hook = MetricsNotificationHook()

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "payload": NotificationPayload(
                message="Test",
                level=NotificationLevel.INFO,
            ),
        }

        hook.on_before_send(context)

        context["result"] = NotificationResult.success_result(
            NotificationChannel.SLACK, "test"
        )

        hook.on_after_send(context, success=True)

        stats = hook.stats
        assert stats.total_sends == 1
        assert stats.successful_sends == 1
        assert stats.failed_sends == 0

    def test_failure_tracking(self) -> None:
        """Test failure tracking."""
        hook = MetricsNotificationHook()

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.WEBHOOK,
            "payload": NotificationPayload(message="Test"),
        }

        hook.on_before_send(context)

        context["result"] = NotificationResult.failure_result(
            NotificationChannel.WEBHOOK, "test", "Error", error_type="TestError"
        )

        hook.on_after_send(context, success=False)

        stats = hook.stats
        assert stats.failed_sends == 1
        assert "TestError" in stats.errors_by_type

    def test_channel_tracking(self) -> None:
        """Test tracking by channel."""
        hook = MetricsNotificationHook()

        for channel in [NotificationChannel.SLACK, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]:
            context = {
                "handler_name": "test",
                "channel": channel,
                "payload": NotificationPayload(message="Test"),
                "result": NotificationResult.success_result(channel, "test"),
            }
            hook.on_before_send(context)
            hook.on_after_send(context, success=True)

        stats = hook.stats
        assert stats.sends_by_channel["slack"] == 2
        assert stats.sends_by_channel["webhook"] == 1

    def test_level_tracking(self) -> None:
        """Test tracking by level."""
        hook = MetricsNotificationHook()

        for level in [NotificationLevel.INFO, NotificationLevel.ERROR]:
            context = {
                "handler_name": "test",
                "channel": NotificationChannel.SLACK,
                "payload": NotificationPayload(message="Test", level=level),
                "result": NotificationResult.success_result(
                    NotificationChannel.SLACK, "test"
                ),
            }
            hook.on_before_send(context)
            hook.on_after_send(context, success=True)

        stats = hook.stats
        assert stats.sends_by_level["info"] == 1
        assert stats.sends_by_level["error"] == 1

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        hook = MetricsNotificationHook()

        # 2 successes, 1 failure
        for success in [True, True, False]:
            context = {
                "handler_name": "test",
                "channel": NotificationChannel.SLACK,
                "payload": NotificationPayload(message="Test"),
            }
            hook.on_before_send(context)

            if success:
                context["result"] = NotificationResult.success_result(
                    NotificationChannel.SLACK, "test"
                )
            else:
                context["result"] = NotificationResult.failure_result(
                    NotificationChannel.SLACK, "test", "Error"
                )

            hook.on_after_send(context, success=success)

        assert hook.stats.success_rate == pytest.approx(2 / 3, 0.01)

    def test_retry_tracking(self) -> None:
        """Test retry event tracking."""
        hook = MetricsNotificationHook()

        context = {"handler_name": "test"}
        error = Exception("Test error")

        hook.on_retry(context, attempt=1, error=error)
        hook.on_retry(context, attempt=2, error=error)

        assert hook.stats.total_retries == 2

    def test_reset(self) -> None:
        """Test stats reset."""
        hook = MetricsNotificationHook()

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "payload": NotificationPayload(message="Test"),
            "result": NotificationResult.success_result(
                NotificationChannel.SLACK, "test"
            ),
        }
        hook.on_before_send(context)
        hook.on_after_send(context, success=True)

        assert hook.stats.total_sends == 1

        hook.reset()

        assert hook.stats.total_sends == 0


class TestCompositeNotificationHook:
    """Tests for CompositeNotificationHook."""

    def test_delegates_to_all_hooks(self) -> None:
        """Test that composite delegates to all hooks."""
        hook1 = MetricsNotificationHook()
        hook2 = MetricsNotificationHook()

        composite = CompositeNotificationHook([hook1, hook2])

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "payload": NotificationPayload(message="Test"),
            "result": NotificationResult.success_result(
                NotificationChannel.SLACK, "test"
            ),
        }

        composite.on_before_send(context)
        composite.on_after_send(context, success=True)

        assert hook1.stats.total_sends == 1
        assert hook2.stats.total_sends == 1

    def test_isolates_failures(self) -> None:
        """Test that hook failures are isolated."""

        class FailingHook(LoggingNotificationHook):
            def on_before_send(self, context):
                raise Exception("Hook failure")

        failing = FailingHook()
        metrics = MetricsNotificationHook()

        composite = CompositeNotificationHook([failing, metrics])

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "payload": NotificationPayload(message="Test"),
            "result": NotificationResult.success_result(
                NotificationChannel.SLACK, "test"
            ),
        }

        # Should not raise
        composite.on_before_send(context)
        composite.on_after_send(context, success=True)

        # Second hook should still be invoked
        assert metrics.stats.total_sends == 1

    def test_add_remove_hooks(self) -> None:
        """Test adding and removing hooks."""
        composite = CompositeNotificationHook()
        hook = MetricsNotificationHook()

        composite.add_hook(hook)

        context = {
            "handler_name": "test",
            "channel": NotificationChannel.SLACK,
            "payload": NotificationPayload(message="Test"),
            "result": NotificationResult.success_result(
                NotificationChannel.SLACK, "test"
            ),
        }
        composite.on_before_send(context)
        composite.on_after_send(context, success=True)

        assert hook.stats.total_sends == 1

        composite.remove_hook(hook)
        composite.on_before_send(context)
        composite.on_after_send(context, success=True)

        # Should not increase after removal
        assert hook.stats.total_sends == 1


class TestCallbackNotificationHook:
    """Tests for CallbackNotificationHook."""

    def test_before_callback(self) -> None:
        """Test before send callback."""
        calls = []

        def on_before(ctx):
            calls.append(("before", ctx))

        hook = CallbackNotificationHook(on_before=on_before)
        hook.on_before_send({"test": True})

        assert len(calls) == 1
        assert calls[0][0] == "before"

    def test_after_callback(self) -> None:
        """Test after send callback."""
        calls = []

        def on_after(ctx, success):
            calls.append(("after", success))

        hook = CallbackNotificationHook(on_after=on_after)
        hook.on_after_send({}, success=True)

        assert len(calls) == 1
        assert calls[0] == ("after", True)

    def test_retry_callback(self) -> None:
        """Test retry callback."""
        calls = []

        def on_retry(ctx, attempt, error):
            calls.append(("retry", attempt))

        hook = CallbackNotificationHook(on_retry_callback=on_retry)
        hook.on_retry({}, attempt=2, error=Exception())

        assert len(calls) == 1
        assert calls[0] == ("retry", 2)


class TestFilteringNotificationHook:
    """Tests for FilteringNotificationHook."""

    def test_filter_by_level(self) -> None:
        """Test filtering by minimum level."""
        hook = FilteringNotificationHook(min_level=NotificationLevel.ERROR)

        # Should be skipped
        info_context = {
            "payload": NotificationPayload(
                message="Test",
                level=NotificationLevel.INFO,
            ),
        }
        hook.on_before_send(info_context)
        assert info_context.get("skip")

        # Should not be skipped
        error_context = {
            "payload": NotificationPayload(
                message="Test",
                level=NotificationLevel.ERROR,
            ),
        }
        hook.on_before_send(error_context)
        assert not error_context.get("skip")

    def test_filter_by_required_tags(self) -> None:
        """Test filtering by required tags."""
        hook = FilteringNotificationHook(required_tags={"critical"})

        # Should be skipped (missing tag)
        context = {
            "payload": NotificationPayload(message="Test"),
        }
        hook.on_before_send(context)
        assert context.get("skip")

        # Should not be skipped (has tag)
        context = {
            "payload": NotificationPayload(
                message="Test"
            ).with_metadata(
                NotificationPayload(message="Test").metadata.with_tags("critical")
            ),
        }
        hook.on_before_send(context)
        # The tag check should pass now

    def test_filter_by_excluded_tags(self) -> None:
        """Test filtering by excluded tags."""
        hook = FilteringNotificationHook(excluded_tags={"test"})

        # Should be skipped (has excluded tag)
        context = {
            "payload": NotificationPayload(
                message="Test"
            ).with_metadata(
                NotificationPayload(message="Test").metadata.with_tags("test")
            ),
        }
        hook.on_before_send(context)
        assert context.get("skip")

    def test_filter_by_predicate(self) -> None:
        """Test filtering by custom predicate."""

        def is_important(ctx):
            return ctx.get("important", False)

        hook = FilteringNotificationHook(predicate=is_important)

        # Should be skipped
        context = {"payload": NotificationPayload(message="Test")}
        hook.on_before_send(context)
        assert context.get("skip")

        # Should not be skipped
        context = {
            "payload": NotificationPayload(message="Test"),
            "important": True,
        }
        hook.on_before_send(context)
        assert not context.get("skip")


class TestThrottlingNotificationHook:
    """Tests for ThrottlingNotificationHook."""

    def test_allows_within_limit(self) -> None:
        """Test that requests within limit are allowed."""
        hook = ThrottlingNotificationHook(max_per_window=5, window_seconds=60.0)

        for _ in range(5):
            context = {}
            hook.on_before_send(context)
            assert not context.get("skip")

    def test_throttles_over_limit(self) -> None:
        """Test that requests over limit are throttled."""
        hook = ThrottlingNotificationHook(max_per_window=3, window_seconds=60.0)

        # First 3 should pass
        for _ in range(3):
            context = {}
            hook.on_before_send(context)
            assert not context.get("skip")

        # 4th should be throttled
        context = {}
        hook.on_before_send(context)
        assert context.get("skip")
        assert context.get("skip_reason") == "throttled"

    def test_current_count(self) -> None:
        """Test current count property."""
        hook = ThrottlingNotificationHook(max_per_window=10, window_seconds=60.0)

        assert hook.current_count == 0

        hook.on_before_send({})
        hook.on_before_send({})

        assert hook.current_count == 2

    def test_reset(self) -> None:
        """Test reset clears throttle state."""
        hook = ThrottlingNotificationHook(max_per_window=2, window_seconds=60.0)

        hook.on_before_send({})
        hook.on_before_send({})

        assert hook.current_count == 2

        hook.reset()

        assert hook.current_count == 0
