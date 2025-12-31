"""Tests for message formatters."""

import json

import pytest

from packages.enterprise.notifications.exceptions import NotificationFormatterError
from packages.enterprise.notifications.formatters import (
    JsonFormatter,
    MarkdownFormatter,
    SlackBlockFormatter,
    TemplateFormatter,
    TextFormatter,
    get_formatter,
    list_formatters,
    register_formatter,
    reset_formatters,
)
from packages.enterprise.notifications.types import (
    NotificationLevel,
    NotificationMetadata,
    NotificationPayload,
)


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_basic_format(self) -> None:
        """Test basic text formatting."""
        formatter = TextFormatter(include_timestamp=False)
        payload = NotificationPayload(
            message="Test message",
            level=NotificationLevel.INFO,
        )

        result = formatter.format(payload)

        assert "[INFO]" in result
        assert "Test message" in result

    def test_with_title(self) -> None:
        """Test formatting with title."""
        formatter = TextFormatter(include_timestamp=False)
        payload = NotificationPayload(
            message="Test message",
            title="Test Title",
        )

        result = formatter.format(payload)

        assert "Title: Test Title" in result

    def test_with_context(self) -> None:
        """Test formatting with context."""
        formatter = TextFormatter(include_timestamp=False, include_context=True)
        payload = NotificationPayload(
            message="Test message",
            context=(("key1", "value1"), ("key2", "value2")),
        )

        result = formatter.format(payload)

        assert "Context:" in result
        assert "key1: value1" in result
        assert "key2: value2" in result

    def test_without_context(self) -> None:
        """Test formatting without context."""
        formatter = TextFormatter(include_timestamp=False, include_context=False)
        payload = NotificationPayload(
            message="Test message",
            context=(("key1", "value1"),),
        )

        result = formatter.format(payload)

        assert "Context:" not in result

    def test_with_metadata(self) -> None:
        """Test formatting with metadata."""
        formatter = TextFormatter(include_timestamp=False, include_metadata=True)
        payload = NotificationPayload(
            message="Test message",
        ).with_metadata(
            NotificationMetadata(source="test-source", correlation_id="123")
        )

        result = formatter.format(payload)

        assert "Source: test-source" in result
        assert "Correlation ID: 123" in result

    def test_level_indicators(self) -> None:
        """Test level indicators for all levels."""
        formatter = TextFormatter(include_timestamp=False)

        for level in NotificationLevel:
            payload = NotificationPayload(message="Test", level=level)
            result = formatter.format(payload)
            assert f"[{level.value.upper()}]" in result

    def test_formatter_name(self) -> None:
        """Test formatter name."""
        formatter = TextFormatter()
        assert formatter.name == "text"

        custom = TextFormatter(name="custom_text")
        assert custom.name == "custom_text"


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter."""

    def test_basic_format(self) -> None:
        """Test basic Markdown formatting."""
        formatter = MarkdownFormatter()
        payload = NotificationPayload(
            message="Test message",
            level=NotificationLevel.ERROR,
        )

        result = formatter.format(payload)

        assert "##" in result  # Header
        assert "Test message" in result

    def test_with_title(self) -> None:
        """Test formatting with custom title."""
        formatter = MarkdownFormatter()
        payload = NotificationPayload(
            message="Test message",
            title="Custom Title",
        )

        result = formatter.format(payload)

        assert "Custom Title" in result

    def test_with_context_table(self) -> None:
        """Test context formatted as table."""
        formatter = MarkdownFormatter(include_context=True)
        payload = NotificationPayload(
            message="Test message",
            context=(("key1", "value1"), ("key2", "value2")),
        )

        result = formatter.format(payload)

        assert "| Key | Value |" in result
        assert "`key1`" in result
        assert "value1" in result

    def test_metadata_footer(self) -> None:
        """Test metadata in footer."""
        formatter = MarkdownFormatter(include_metadata=True)
        payload = NotificationPayload(
            message="Test message",
        ).with_metadata(
            NotificationMetadata(source="test-source")
        )

        result = formatter.format(payload)

        assert "**Source:**" in result
        assert "test-source" in result

    def test_level_emojis(self) -> None:
        """Test emojis for different levels."""
        formatter = MarkdownFormatter()

        # Test a few levels
        payload_info = NotificationPayload(message="Test", level=NotificationLevel.INFO)
        payload_error = NotificationPayload(message="Test", level=NotificationLevel.ERROR)

        result_info = formatter.format(payload_info)
        result_error = formatter.format(payload_error)

        # Different emojis should be present
        assert result_info != result_error


class TestSlackBlockFormatter:
    """Tests for SlackBlockFormatter."""

    def test_basic_format(self) -> None:
        """Test basic Slack block formatting."""
        formatter = SlackBlockFormatter()
        payload = NotificationPayload(
            message="Test message",
            level=NotificationLevel.INFO,
        )

        result = formatter.format(payload)

        assert "blocks" in result
        assert "attachments" in result
        assert isinstance(result["blocks"], list)

    def test_header_block(self) -> None:
        """Test header block creation."""
        formatter = SlackBlockFormatter()
        payload = NotificationPayload(
            message="Test message",
            title="Test Title",
        )

        result = formatter.format(payload)

        header_blocks = [b for b in result["blocks"] if b.get("type") == "header"]
        assert len(header_blocks) > 0
        assert "Test Title" in header_blocks[0]["text"]["text"]

    def test_message_section(self) -> None:
        """Test message section block."""
        formatter = SlackBlockFormatter()
        payload = NotificationPayload(message="Test message content")

        result = formatter.format(payload)

        section_blocks = [b for b in result["blocks"] if b.get("type") == "section"]
        assert len(section_blocks) > 0

    def test_context_fields(self) -> None:
        """Test context as fields."""
        formatter = SlackBlockFormatter()
        payload = NotificationPayload(
            message="Test",
            context=(("key1", "value1"), ("key2", "value2")),
        )

        result = formatter.format(payload)

        # Find section with fields
        field_sections = [
            b for b in result["blocks"]
            if b.get("type") == "section" and "fields" in b
        ]
        assert len(field_sections) > 0

    def test_color_attachment(self) -> None:
        """Test color attachment for level."""
        formatter = SlackBlockFormatter()

        # Test different levels have different colors
        payload_info = NotificationPayload(message="Test", level=NotificationLevel.INFO)
        payload_error = NotificationPayload(message="Test", level=NotificationLevel.ERROR)

        result_info = formatter.format(payload_info)
        result_error = formatter.format(payload_error)

        assert result_info["attachments"][0]["color"] != result_error["attachments"][0]["color"]

    def test_fallback_text(self) -> None:
        """Test fallback text in attachment."""
        formatter = SlackBlockFormatter()
        payload = NotificationPayload(message="Test message")

        result = formatter.format(payload)

        assert result["attachments"][0]["fallback"] == "Test message"


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_basic_format(self) -> None:
        """Test basic JSON formatting."""
        formatter = JsonFormatter()
        payload = NotificationPayload(
            message="Test message",
            level=NotificationLevel.INFO,
        )

        result = formatter.format(payload)
        data = json.loads(result)

        assert data["message"] == "Test message"
        assert data["level"] == "info"

    def test_compact_format(self) -> None:
        """Test compact JSON (no indentation)."""
        formatter = JsonFormatter(indent=None)
        payload = NotificationPayload(message="Test")

        result = formatter.format(payload)

        assert "\n" not in result

    def test_indented_format(self) -> None:
        """Test indented JSON."""
        formatter = JsonFormatter(indent=2)
        payload = NotificationPayload(message="Test")

        result = formatter.format(payload)

        assert "\n" in result

    def test_include_all(self) -> None:
        """Test including all payload data."""
        formatter = JsonFormatter(include_all=True)
        payload = NotificationPayload(
            message="Test",
            title="Title",
            context=(("key", "value"),),
        )

        result = formatter.format(payload)
        data = json.loads(result)

        assert "title" in data
        assert "context" in data

    def test_minimal_format(self) -> None:
        """Test minimal JSON output."""
        formatter = JsonFormatter(include_all=False)
        payload = NotificationPayload(
            message="Test",
            title="Title",
            context=(("key", "value"),),
        )

        result = formatter.format(payload)
        data = json.loads(result)

        assert data["message"] == "Test"
        assert "timestamp" in data


class TestTemplateFormatter:
    """Tests for TemplateFormatter."""

    def test_basic_template(self) -> None:
        """Test basic template substitution."""
        formatter = TemplateFormatter(template="${level}: ${message}")
        payload = NotificationPayload(
            message="Test message",
            level=NotificationLevel.ERROR,
        )

        result = formatter.format(payload)

        assert result == "error: Test message"

    def test_level_upper(self) -> None:
        """Test level_upper variable."""
        formatter = TemplateFormatter(template="${level_upper} - ${message}")
        payload = NotificationPayload(
            message="Test",
            level=NotificationLevel.WARNING,
        )

        result = formatter.format(payload)

        assert "WARNING" in result

    def test_with_title(self) -> None:
        """Test title substitution."""
        formatter = TemplateFormatter(template="${title}: ${message}")
        payload = NotificationPayload(
            message="Test",
            title="Alert",
        )

        result = formatter.format(payload)

        assert result == "Alert: Test"

    def test_context_variables(self) -> None:
        """Test context variables."""
        formatter = TemplateFormatter(template="User: ${ctx_user_id}, Action: ${ctx_action}")
        payload = NotificationPayload(
            message="Test",
            context=(("user_id", "123"), ("action", "login")),
        )

        result = formatter.format(payload)

        assert result == "User: 123, Action: login"

    def test_missing_variable_safe(self) -> None:
        """Test that missing variables are handled safely."""
        formatter = TemplateFormatter(template="${missing_var}: ${message}")
        payload = NotificationPayload(message="Test")

        result = formatter.format(payload)

        # safe_substitute keeps missing variables
        assert "${missing_var}" in result or "missing_var" in result

    def test_metadata_variables(self) -> None:
        """Test metadata variables."""
        formatter = TemplateFormatter(template="[${source}] ${message}")
        payload = NotificationPayload(message="Test").with_metadata(
            NotificationMetadata(source="test-app")
        )

        result = formatter.format(payload)

        assert result == "[test-app] Test"


class TestFormatterRegistry:
    """Tests for formatter registry functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> None:
        """Reset formatter registry before and after each test."""
        reset_formatters()
        yield
        reset_formatters()

    def test_default_formatters_registered(self) -> None:
        """Test that default formatters are registered."""
        formatters = list_formatters()

        assert "text" in formatters
        assert "markdown" in formatters
        assert "slack_blocks" in formatters
        assert "json" in formatters

    def test_get_formatter(self) -> None:
        """Test getting a formatter by name."""
        formatter = get_formatter("text")

        assert formatter is not None
        assert formatter.name == "text"

    def test_get_nonexistent_formatter(self) -> None:
        """Test getting a non-existent formatter."""
        with pytest.raises(NotificationFormatterError) as exc_info:
            get_formatter("nonexistent")

        assert exc_info.value.formatter_name == "nonexistent"

    def test_register_custom_formatter(self) -> None:
        """Test registering a custom formatter."""
        custom = TemplateFormatter(template="${message}", name="custom")
        register_formatter(custom)

        assert "custom" in list_formatters()
        assert get_formatter("custom") is custom

    def test_reset_formatters(self) -> None:
        """Test resetting formatter registry."""
        custom = TemplateFormatter(template="${message}", name="custom")
        register_formatter(custom)

        reset_formatters()

        assert "custom" not in list_formatters()
        assert "text" in list_formatters()  # Defaults restored
