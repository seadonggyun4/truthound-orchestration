"""Tests for notification types."""

from datetime import datetime, timezone

import pytest

from packages.enterprise.notifications.types import (
    BatchNotificationResult,
    NotificationChannel,
    NotificationLevel,
    NotificationMetadata,
    NotificationPayload,
    NotificationResult,
    NotificationStatus,
)


class TestNotificationLevel:
    """Tests for NotificationLevel enum."""

    def test_level_values(self) -> None:
        """Test that all levels have correct values."""
        assert NotificationLevel.DEBUG.value == "debug"
        assert NotificationLevel.INFO.value == "info"
        assert NotificationLevel.WARNING.value == "warning"
        assert NotificationLevel.ERROR.value == "error"
        assert NotificationLevel.CRITICAL.value == "critical"

    def test_level_priority(self) -> None:
        """Test level priority ordering."""
        assert NotificationLevel.DEBUG.priority == 0
        assert NotificationLevel.INFO.priority == 1
        assert NotificationLevel.WARNING.priority == 2
        assert NotificationLevel.ERROR.priority == 3
        assert NotificationLevel.CRITICAL.priority == 4

    def test_level_comparison(self) -> None:
        """Test level comparison operators."""
        assert NotificationLevel.DEBUG < NotificationLevel.INFO
        assert NotificationLevel.INFO < NotificationLevel.WARNING
        assert NotificationLevel.WARNING < NotificationLevel.ERROR
        assert NotificationLevel.ERROR < NotificationLevel.CRITICAL

        assert NotificationLevel.CRITICAL > NotificationLevel.ERROR
        assert NotificationLevel.ERROR >= NotificationLevel.ERROR
        assert NotificationLevel.WARNING <= NotificationLevel.WARNING


class TestNotificationChannel:
    """Tests for NotificationChannel enum."""

    def test_channel_values(self) -> None:
        """Test that all channels have correct values."""
        assert NotificationChannel.SLACK.value == "slack"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.CUSTOM.value == "custom"

    def test_channel_str(self) -> None:
        """Test channel string representation."""
        assert str(NotificationChannel.SLACK) == "slack"


class TestNotificationStatus:
    """Tests for NotificationStatus enum."""

    def test_terminal_status(self) -> None:
        """Test terminal status detection."""
        assert NotificationStatus.SENT.is_terminal
        assert NotificationStatus.DELIVERED.is_terminal
        assert NotificationStatus.FAILED.is_terminal
        assert NotificationStatus.SKIPPED.is_terminal

        assert not NotificationStatus.PENDING.is_terminal
        assert not NotificationStatus.SENDING.is_terminal
        assert not NotificationStatus.RETRYING.is_terminal

    def test_success_status(self) -> None:
        """Test success status detection."""
        assert NotificationStatus.SENT.is_success
        assert NotificationStatus.DELIVERED.is_success

        assert not NotificationStatus.FAILED.is_success
        assert not NotificationStatus.PENDING.is_success


class TestNotificationMetadata:
    """Tests for NotificationMetadata dataclass."""

    def test_default_metadata(self) -> None:
        """Test default metadata creation."""
        metadata = NotificationMetadata()
        assert metadata.source is None
        assert metadata.correlation_id is None
        assert metadata.tags == frozenset()
        assert metadata.attributes == ()
        assert isinstance(metadata.created_at, datetime)

    def test_metadata_builder_methods(self) -> None:
        """Test metadata builder methods."""
        metadata = NotificationMetadata()

        with_source = metadata.with_source("test_source")
        assert with_source.source == "test_source"
        assert metadata.source is None  # Original unchanged

        with_tags = metadata.with_tags("tag1", "tag2")
        assert "tag1" in with_tags.tags
        assert "tag2" in with_tags.tags

        with_attr = metadata.with_attribute("key", "value")
        assert ("key", "value") in with_attr.attributes

    def test_metadata_serialization(self) -> None:
        """Test metadata serialization."""
        metadata = NotificationMetadata(
            source="test",
            correlation_id="corr_123",
            tags=frozenset(["tag1"]),
        )

        data = metadata.to_dict()
        assert data["source"] == "test"
        assert data["correlation_id"] == "corr_123"
        assert "tag1" in data["tags"]

        restored = NotificationMetadata.from_dict(data)
        assert restored.source == metadata.source
        assert restored.correlation_id == metadata.correlation_id


class TestNotificationPayload:
    """Tests for NotificationPayload dataclass."""

    def test_payload_creation(self) -> None:
        """Test payload creation."""
        payload = NotificationPayload(
            message="Test message",
            level=NotificationLevel.WARNING,
            title="Test Title",
        )

        assert payload.message == "Test message"
        assert payload.level == NotificationLevel.WARNING
        assert payload.title == "Test Title"

    def test_payload_builder_methods(self) -> None:
        """Test payload builder methods."""
        payload = NotificationPayload(message="Test")

        with_title = payload.with_title("New Title")
        assert with_title.title == "New Title"
        assert payload.title is None

        with_context = payload.with_context(key1="value1", key2=42)
        assert ("key1", "value1") in with_context.context
        assert ("key2", 42) in with_context.context

        with_level = payload.with_level(NotificationLevel.ERROR)
        assert with_level.level == NotificationLevel.ERROR

    def test_payload_context_dict(self) -> None:
        """Test context dictionary conversion."""
        payload = NotificationPayload(
            message="Test",
            context=(("a", 1), ("b", 2)),
        )

        ctx = payload.context_dict
        assert ctx["a"] == 1
        assert ctx["b"] == 2

    def test_payload_serialization(self) -> None:
        """Test payload serialization."""
        payload = NotificationPayload(
            message="Test",
            level=NotificationLevel.INFO,
            title="Title",
            context=(("key", "value"),),
        )

        data = payload.to_dict()
        assert data["message"] == "Test"
        assert data["level"] == "info"
        assert data["title"] == "Title"

        restored = NotificationPayload.from_dict(data)
        assert restored.message == payload.message
        assert restored.level == payload.level


class TestNotificationResult:
    """Tests for NotificationResult dataclass."""

    def test_success_result(self) -> None:
        """Test success result creation."""
        result = NotificationResult.success_result(
            channel=NotificationChannel.SLACK,
            handler_name="test",
            message_id="msg_123",
        )

        assert result.success
        assert result.status == NotificationStatus.SENT
        assert result.channel == NotificationChannel.SLACK
        assert result.message_id == "msg_123"

    def test_failure_result(self) -> None:
        """Test failure result creation."""
        result = NotificationResult.failure_result(
            channel=NotificationChannel.WEBHOOK,
            handler_name="test",
            error="Connection failed",
            error_type="ConnectionError",
        )

        assert not result.success
        assert result.status == NotificationStatus.FAILED
        assert result.error == "Connection failed"
        assert result.error_type == "ConnectionError"

    def test_skipped_result(self) -> None:
        """Test skipped result creation."""
        result = NotificationResult.skipped_result(
            channel=NotificationChannel.SLACK,
            handler_name="test",
            reason="Level too low",
        )

        assert result.success  # Skipped is not a failure
        assert result.status == NotificationStatus.SKIPPED
        assert result.error == "Level too low"

    def test_result_serialization(self) -> None:
        """Test result serialization."""
        result = NotificationResult.success_result(
            channel=NotificationChannel.SLACK,
            handler_name="test",
        )

        data = result.to_dict()
        assert data["success"]
        assert data["channel"] == "slack"
        assert data["status"] == "sent"

        restored = NotificationResult.from_dict(data)
        assert restored.success == result.success
        assert restored.channel == result.channel


class TestBatchNotificationResult:
    """Tests for BatchNotificationResult dataclass."""

    def test_batch_result_creation(self) -> None:
        """Test batch result creation."""
        results = {
            "slack": NotificationResult.success_result(
                NotificationChannel.SLACK, "slack"
            ),
            "webhook": NotificationResult.failure_result(
                NotificationChannel.WEBHOOK, "webhook", "Failed"
            ),
        }

        batch = BatchNotificationResult.from_results(results, duration_ms=100.0)

        assert batch.total_count == 2
        assert batch.success_count == 1
        assert batch.failure_count == 1
        assert batch.duration_ms == 100.0

    def test_batch_all_success(self) -> None:
        """Test all_success property."""
        results = {
            "a": NotificationResult.success_result(NotificationChannel.SLACK, "a"),
            "b": NotificationResult.success_result(NotificationChannel.WEBHOOK, "b"),
        }

        batch = BatchNotificationResult.from_results(results)
        assert batch.all_success

    def test_batch_any_success(self) -> None:
        """Test any_success property."""
        results = {
            "a": NotificationResult.success_result(NotificationChannel.SLACK, "a"),
            "b": NotificationResult.failure_result(
                NotificationChannel.WEBHOOK, "b", "Failed"
            ),
        }

        batch = BatchNotificationResult.from_results(results)
        assert batch.any_success
        assert not batch.all_success

    def test_batch_get_failures(self) -> None:
        """Test get_failures method."""
        results = {
            "a": NotificationResult.success_result(NotificationChannel.SLACK, "a"),
            "b": NotificationResult.failure_result(
                NotificationChannel.WEBHOOK, "b", "Failed"
            ),
        }

        batch = BatchNotificationResult.from_results(results)
        failures = batch.get_failures()

        assert len(failures) == 1
        assert failures[0][0] == "b"
