"""Tests for incident management notification handlers (PagerDuty, Opsgenie).

This module provides comprehensive tests for PagerDutyNotificationHandler and
OpsgenieNotificationHandler, including configuration, API interactions, and
incident lifecycle management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.enterprise.notifications.config import (
    OpsgenieConfig,
    OpsgeniePriority,
    OpsgenieRegion,
    OpsgenieResponder,
    PagerDutyConfig,
    PagerDutyRegion,
)
from packages.enterprise.notifications.exceptions import (
    NotificationConfigError,
    NotificationSendError,
)
from packages.enterprise.notifications.handlers.incident import (
    IncidentAction,
    IncidentDetails,
    IncidentManagementHandler,
    IncidentSeverity,
    create_incident_payload,
)
from packages.enterprise.notifications.handlers.opsgenie import (
    OpsgenieNotificationHandler,
    create_opsgenie_handler,
)
from packages.enterprise.notifications.handlers.pagerduty import (
    PagerDutyNotificationHandler,
    create_pagerduty_handler,
)
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pagerduty_config() -> PagerDutyConfig:
    """Create a basic PagerDuty configuration."""
    return PagerDutyConfig(
        routing_key="abcdef1234567890abcdef1234567890",  # 32-char key
        region=PagerDutyRegion.US,
        default_severity="warning",
        default_source="truthound-orchestration",
        default_component="data-quality",
    )


@pytest.fixture
def opsgenie_config() -> OpsgenieConfig:
    """Create a basic Opsgenie configuration."""
    return OpsgenieConfig(
        api_key="test-api-key-12345",
        region=OpsgenieRegion.US,
        default_priority=OpsgeniePriority.P3,
        default_responders=(
            OpsgenieResponder(id="data-quality-team", type="team"),
        ),
        default_tags=("data-quality", "truthound"),
    )


@pytest.fixture
def basic_payload() -> NotificationPayload:
    """Create a basic notification payload."""
    return NotificationPayload(
        message="Data quality check failed",
        level=NotificationLevel.ERROR,
        title="Quality Check Alert",
        context=(
            ("check_id", "chk_123"),
            ("failed_count", "5"),
            ("table", "users"),
        ),
    )


@pytest.fixture
def critical_payload() -> NotificationPayload:
    """Create a critical notification payload."""
    return NotificationPayload(
        message="Critical data quality violation detected!",
        level=NotificationLevel.CRITICAL,
        title="Critical DQ Alert",
        context=(
            ("check_id", "chk_456"),
            ("failed_count", "100"),
            ("table", "transactions"),
            ("dedup_key", "txn_check_001"),
        ),
    )


@pytest.fixture
def pagerduty_handler(pagerduty_config: PagerDutyConfig) -> PagerDutyNotificationHandler:
    """Create a PagerDuty handler."""
    return PagerDutyNotificationHandler(config=pagerduty_config)


@pytest.fixture
def opsgenie_handler(opsgenie_config: OpsgenieConfig) -> OpsgenieNotificationHandler:
    """Create an Opsgenie handler."""
    return OpsgenieNotificationHandler(config=opsgenie_config)


# =============================================================================
# Incident Severity Tests
# =============================================================================


class TestIncidentSeverity:
    """Tests for IncidentSeverity enum."""

    def test_severity_values(self) -> None:
        """Test severity enum values."""
        assert IncidentSeverity.CRITICAL.value == "critical"
        assert IncidentSeverity.HIGH.value == "high"
        assert IncidentSeverity.MEDIUM.value == "medium"
        assert IncidentSeverity.LOW.value == "low"
        assert IncidentSeverity.INFO.value == "info"

    def test_from_notification_level(self) -> None:
        """Test conversion from NotificationLevel."""
        assert IncidentSeverity.from_notification_level(NotificationLevel.CRITICAL) == IncidentSeverity.CRITICAL
        assert IncidentSeverity.from_notification_level(NotificationLevel.ERROR) == IncidentSeverity.HIGH
        assert IncidentSeverity.from_notification_level(NotificationLevel.WARNING) == IncidentSeverity.MEDIUM
        assert IncidentSeverity.from_notification_level(NotificationLevel.INFO) == IncidentSeverity.LOW
        assert IncidentSeverity.from_notification_level(NotificationLevel.DEBUG) == IncidentSeverity.INFO


class TestIncidentAction:
    """Tests for IncidentAction enum."""

    def test_action_values(self) -> None:
        """Test action enum values."""
        assert IncidentAction.TRIGGER.value == "trigger"
        assert IncidentAction.ACKNOWLEDGE.value == "acknowledge"
        assert IncidentAction.RESOLVE.value == "resolve"


class TestIncidentDetails:
    """Tests for IncidentDetails dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic incident details creation."""
        details = IncidentDetails()
        assert details.severity == IncidentSeverity.MEDIUM  # default
        assert details.dedup_key is None
        assert details.action == IncidentAction.TRIGGER

    def test_with_all_fields(self) -> None:
        """Test incident details with all fields."""
        details = IncidentDetails(
            severity=IncidentSeverity.CRITICAL,
            dedup_key="alert-001",
            source="test-service",
            component="database",
            group="infrastructure",
            custom_details=(("key", "value"),),
            links=(("https://example.com/dashboard", "Dashboard"),),
            images=("https://example.com/chart.png",),
        )
        assert details.severity == IncidentSeverity.CRITICAL
        assert details.dedup_key == "alert-001"
        assert details.source == "test-service"
        assert details.custom_details_dict["key"] == "value"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        details = IncidentDetails(
            severity=IncidentSeverity.HIGH,
            custom_details=(("foo", "bar"),),
        )
        data = details.to_dict()
        assert data["severity"] == "high"
        assert data["custom_details"]["foo"] == "bar"


class TestCreateIncidentPayload:
    """Tests for create_incident_payload helper."""

    def test_basic_payload(self) -> None:
        """Test creating basic incident payload."""
        payload = create_incident_payload(
            message="Something happened",
        )
        assert payload.message == "Something happened"
        assert payload.level == NotificationLevel.ERROR

    def test_with_level(self) -> None:
        """Test creating payload with level."""
        payload = create_incident_payload(
            message="Critical issue",
            level=NotificationLevel.CRITICAL,
        )
        assert payload.level == NotificationLevel.CRITICAL

    def test_with_context(self) -> None:
        """Test creating payload with context."""
        payload = create_incident_payload(
            message="Alert message",
            custom_details={"key": "value"},
        )
        context_dict = payload.context_dict
        assert context_dict["key"] == "value"


# =============================================================================
# PagerDuty Configuration Tests
# =============================================================================


class TestPagerDutyConfig:
    """Tests for PagerDutyConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PagerDutyConfig(
            routing_key="abcdef1234567890abcdef1234567890",
        )
        assert config.region == PagerDutyRegion.US
        assert config.default_severity == "warning"
        assert config.default_source == "truthound-orchestration"

    def test_region_api_url(self) -> None:
        """Test region API URLs."""
        assert "events.pagerduty.com" in PagerDutyRegion.US.events_api_url
        assert "events.eu.pagerduty.com" in PagerDutyRegion.EU.events_api_url

    def test_routing_key_validation(self) -> None:
        """Test routing key length validation."""
        # Valid 32-char key
        config = PagerDutyConfig(
            routing_key="a" * 32,
        )
        assert len(config.routing_key) == 32

        # Short key should raise during config creation
        with pytest.raises(NotificationConfigError, match="32-character"):
            PagerDutyConfig(routing_key="too-short")

    def test_builder_pattern(self) -> None:
        """Test builder pattern methods."""
        config = PagerDutyConfig(
            routing_key="a" * 32,
        )
        config = config.with_region(PagerDutyRegion.EU)
        config = config.with_defaults(severity="critical", source="new-source")

        assert config.region == PagerDutyRegion.EU
        assert config.default_severity == "critical"
        assert config.default_source == "new-source"

    def test_to_dict_masks_routing_key(self) -> None:
        """Test that to_dict masks the routing key."""
        config = PagerDutyConfig(
            routing_key="a" * 32,
        )
        config_dict = config.to_dict()
        assert config_dict["routing_key"] == "***MASKED***"

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config_dict = {
            "routing_key": "a" * 32,
            "region": "eu",
            "default_severity": "error",
        }
        config = PagerDutyConfig.from_dict(config_dict)
        assert config.region == PagerDutyRegion.EU
        assert config.default_severity == "error"


# =============================================================================
# Opsgenie Configuration Tests
# =============================================================================


class TestOpsgenieConfig:
    """Tests for OpsgenieConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = OpsgenieConfig(
            api_key="test-api-key",
        )
        assert config.region == OpsgenieRegion.US
        assert config.default_priority == OpsgeniePriority.P3
        assert config.default_responders == ()

    def test_region_api_url(self) -> None:
        """Test region API URLs."""
        assert "api.opsgenie.com" in OpsgenieRegion.US.api_url
        assert "api.eu.opsgenie.com" in OpsgenieRegion.EU.api_url

    def test_priority_levels(self) -> None:
        """Test priority levels."""
        assert OpsgeniePriority.P1.value == "P1"
        assert OpsgeniePriority.P5.value == "P5"

    def test_responder_types(self) -> None:
        """Test responder creation."""
        team_responder = OpsgenieResponder(
            id="team-123",
            type="team",
        )
        user_responder = OpsgenieResponder(
            name="john@example.com",
            type="user",
        )
        assert team_responder.type == "team"
        assert user_responder.type == "user"
        assert user_responder.name == "john@example.com"

    def test_builder_pattern(self) -> None:
        """Test builder pattern methods."""
        config = OpsgenieConfig(
            api_key="test-key",
        )
        config = config.with_region(OpsgenieRegion.EU)
        config = config.with_priority(OpsgeniePriority.P1)
        config = config.with_defaults(tags=["critical", "production"])

        assert config.region == OpsgenieRegion.EU
        assert config.default_priority == OpsgeniePriority.P1
        assert "critical" in config.default_tags
        assert "production" in config.default_tags

    def test_to_dict_masks_api_key(self) -> None:
        """Test that to_dict masks the API key."""
        config = OpsgenieConfig(
            api_key="secret-api-key",
        )
        config_dict = config.to_dict()
        assert config_dict["api_key"] == "***MASKED***"


# =============================================================================
# PagerDuty Handler Tests
# =============================================================================


class TestPagerDutyHandlerInitialization:
    """Tests for PagerDutyNotificationHandler initialization."""

    def test_init_with_config(self, pagerduty_config: PagerDutyConfig) -> None:
        """Test initialization with full config."""
        handler = PagerDutyNotificationHandler(config=pagerduty_config)
        assert handler.channel == NotificationChannel.PAGERDUTY
        assert handler._pagerduty_config == pagerduty_config

    def test_init_with_routing_key(self) -> None:
        """Test initialization with routing key only."""
        handler = PagerDutyNotificationHandler(
            routing_key="a" * 32,
        )
        assert handler._pagerduty_config.routing_key == "a" * 32

    def test_init_requires_config_or_routing_key(self) -> None:
        """Test that initialization fails without config or routing key."""
        with pytest.raises(ValueError, match="Either config or routing_key"):
            PagerDutyNotificationHandler()

    def test_invalid_routing_key_length(self) -> None:
        """Test that invalid routing key length raises error."""
        with pytest.raises(NotificationConfigError, match="32-character"):
            PagerDutyNotificationHandler(routing_key="short-key")

    def test_handler_name(self) -> None:
        """Test handler name configuration."""
        handler = PagerDutyNotificationHandler(
            routing_key="a" * 32,
            name="custom_pagerduty",
        )
        assert handler.name == "custom_pagerduty"


class TestPagerDutySeverityMapping:
    """Tests for PagerDuty severity mapping."""

    def test_severity_mapping(self, pagerduty_handler: PagerDutyNotificationHandler) -> None:
        """Test severity level mapping."""
        assert pagerduty_handler._map_severity(IncidentSeverity.CRITICAL) == "critical"
        assert pagerduty_handler._map_severity(IncidentSeverity.HIGH) == "error"
        assert pagerduty_handler._map_severity(IncidentSeverity.MEDIUM) == "warning"
        assert pagerduty_handler._map_severity(IncidentSeverity.LOW) == "info"
        assert pagerduty_handler._map_severity(IncidentSeverity.INFO) == "info"


class TestPagerDutyRequestBody:
    """Tests for PagerDuty request body building."""

    def test_build_trigger_request(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test building trigger request body."""
        incident_details = pagerduty_handler._extract_incident_details(basic_payload)
        body = pagerduty_handler._build_request_body(basic_payload, incident_details)

        assert body["routing_key"] == pagerduty_handler._pagerduty_config.routing_key
        assert body["event_action"] == "trigger"
        assert "payload" in body

        payload_data = body["payload"]
        assert payload_data["summary"] == basic_payload.title or basic_payload.message
        assert payload_data["severity"] in ("critical", "error", "warning", "info")
        assert "source" in payload_data

    def test_build_request_with_dedup_key(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        critical_payload: NotificationPayload,
    ) -> None:
        """Test building request with dedup key from context."""
        incident_details = pagerduty_handler._extract_incident_details(critical_payload)
        body = pagerduty_handler._build_request_body(critical_payload, incident_details)

        assert body.get("dedup_key") == "txn_check_001"

    def test_build_request_with_custom_details(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that context is included in custom_details."""
        incident_details = pagerduty_handler._extract_incident_details(basic_payload)
        body = pagerduty_handler._build_request_body(basic_payload, incident_details)

        custom_details = body["payload"].get("custom_details", {})
        assert "check_id" in custom_details or "context" in str(custom_details)


class TestPagerDutySendOperations:
    """Tests for PagerDuty send operations."""

    @pytest.mark.asyncio
    async def test_send_success(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test successful send operation using requests fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "status": "success",
            "message": "Event processed",
            "dedup_key": "test-dedup-key",
        }

        with patch("requests.post", return_value=mock_response):
            result = await pagerduty_handler.send(basic_payload)

            assert result.success
            assert result.channel == NotificationChannel.PAGERDUTY

    @pytest.mark.asyncio
    async def test_send_api_error(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test handling API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Invalid request"}
        mock_response.text = '{"message": "Invalid request"}'

        with patch("requests.post", return_value=mock_response):
            result = await pagerduty_handler.send(basic_payload)

            assert not result.success
            assert "400" in result.error or "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_trigger_with_details(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
    ) -> None:
        """Test trigger_with_details convenience method."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"status": "success", "dedup_key": "test-key"}

        with patch("requests.post", return_value=mock_response):
            result = await pagerduty_handler.trigger_with_details(
                message="Test alert",
                title="Test Title",
                severity=IncidentSeverity.HIGH,
                dedup_key="test-key",
                custom_details={"foo": "bar"},
            )

            assert result.success

    @pytest.mark.asyncio
    async def test_acknowledge(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
    ) -> None:
        """Test acknowledge operation."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"status": "success"}

        with patch("requests.post", return_value=mock_response):
            result = await pagerduty_handler.acknowledge("test-dedup-key")

            assert result.success

    @pytest.mark.asyncio
    async def test_resolve(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
    ) -> None:
        """Test resolve operation."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"status": "success"}

        with patch("requests.post", return_value=mock_response):
            result = await pagerduty_handler.resolve("test-dedup-key")

            assert result.success


# =============================================================================
# Opsgenie Handler Tests
# =============================================================================


class TestOpsgenieHandlerInitialization:
    """Tests for OpsgenieNotificationHandler initialization."""

    def test_init_with_config(self, opsgenie_config: OpsgenieConfig) -> None:
        """Test initialization with full config."""
        handler = OpsgenieNotificationHandler(config=opsgenie_config)
        assert handler.channel == NotificationChannel.OPSGENIE
        assert handler._opsgenie_config == opsgenie_config

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key only."""
        handler = OpsgenieNotificationHandler(
            api_key="test-api-key",
        )
        assert handler._opsgenie_config.api_key == "test-api-key"

    def test_init_requires_config_or_api_key(self) -> None:
        """Test that initialization fails without config or api_key."""
        with pytest.raises(ValueError, match="Either config or api_key"):
            OpsgenieNotificationHandler()

    def test_handler_name(self) -> None:
        """Test handler name configuration."""
        handler = OpsgenieNotificationHandler(
            api_key="test-key",
            name="custom_opsgenie",
        )
        assert handler.name == "custom_opsgenie"


class TestOpsgeniePriorityMapping:
    """Tests for Opsgenie priority mapping."""

    def test_priority_mapping(self, opsgenie_handler: OpsgenieNotificationHandler) -> None:
        """Test severity to priority mapping."""
        # Note: _map_severity returns a string like "P1", not OpsgeniePriority enum
        assert opsgenie_handler._map_severity(IncidentSeverity.CRITICAL) == "P1"
        assert opsgenie_handler._map_severity(IncidentSeverity.HIGH) == "P2"
        assert opsgenie_handler._map_severity(IncidentSeverity.MEDIUM) == "P3"
        assert opsgenie_handler._map_severity(IncidentSeverity.LOW) == "P4"
        assert opsgenie_handler._map_severity(IncidentSeverity.INFO) == "P5"


class TestOpsgenieRequestBody:
    """Tests for Opsgenie request body building."""

    def test_build_create_request(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test building create alert request body."""
        incident_details = opsgenie_handler._extract_incident_details(basic_payload)
        body = opsgenie_handler._build_request_body(basic_payload, incident_details)

        assert "message" in body
        assert "priority" in body
        assert body["priority"] in ("P1", "P2", "P3", "P4", "P5")

    def test_build_request_with_responders(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that default responders are included."""
        incident_details = opsgenie_handler._extract_incident_details(basic_payload)
        body = opsgenie_handler._build_request_body(basic_payload, incident_details)

        assert "responders" in body
        responders = body["responders"]
        assert len(responders) == 1
        assert responders[0]["type"] == "team"

    def test_build_request_with_tags(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that default tags are included."""
        incident_details = opsgenie_handler._extract_incident_details(basic_payload)
        body = opsgenie_handler._build_request_body(basic_payload, incident_details)

        assert "tags" in body
        tags = body["tags"]
        assert "data-quality" in tags
        assert "truthound" in tags


class TestOpsgenieSendOperations:
    """Tests for Opsgenie send operations."""

    @pytest.mark.asyncio
    async def test_send_success(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test successful send operation using requests fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "result": "Request will be processed",
            "took": 0.1,
            "requestId": "abc123",
        }

        with patch("requests.post", return_value=mock_response):
            result = await opsgenie_handler.send(basic_payload)

            assert result.success
            assert result.channel == NotificationChannel.OPSGENIE

    @pytest.mark.asyncio
    async def test_send_api_error(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test handling API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Invalid API key"}
        mock_response.text = '{"message": "Invalid API key"}'

        with patch("requests.post", return_value=mock_response):
            result = await opsgenie_handler.send(basic_payload)

            assert not result.success
            assert "401" in result.error or "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_create_alert(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
    ) -> None:
        """Test create_alert convenience method."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"result": "success", "requestId": "test-123"}

        with patch("requests.post", return_value=mock_response):
            result = await opsgenie_handler.create_alert(
                message="Test alert",
                description="Test description",
                priority=OpsgeniePriority.P2,
                alias="test-alias",
                tags=["test", "alert"],
            )

            assert result.success

    @pytest.mark.asyncio
    async def test_acknowledge(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
    ) -> None:
        """Test acknowledge operation."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"result": "success"}

        with patch("requests.post", return_value=mock_response):
            result = await opsgenie_handler.acknowledge("test-alias")

            assert result.success

    @pytest.mark.asyncio
    async def test_close(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
    ) -> None:
        """Test close operation."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"result": "success"}

        with patch("requests.post", return_value=mock_response):
            result = await opsgenie_handler.close("test-alias")

            assert result.success

    @pytest.mark.asyncio
    async def test_add_note(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
    ) -> None:
        """Test add_note operation."""
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {"result": "success"}

        with patch("requests.post", return_value=mock_response):
            result = await opsgenie_handler.add_note("test-alias", "Investigation ongoing")

            assert result.success


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_pagerduty_handler(self) -> None:
        """Test create_pagerduty_handler factory."""
        handler = create_pagerduty_handler(
            routing_key="a" * 32,
            source="test-source",
        )
        assert isinstance(handler, PagerDutyNotificationHandler)
        assert handler.pagerduty_config.default_source == "test-source"

    def test_create_pagerduty_handler_with_client(self) -> None:
        """Test create_pagerduty_handler with client info."""
        handler = create_pagerduty_handler(
            routing_key="a" * 32,
            client_name="TestClient",
            client_url="https://example.com",
        )
        assert handler.pagerduty_config.client_name == "TestClient"
        assert handler.pagerduty_config.client_url == "https://example.com"

    def test_create_opsgenie_handler(self) -> None:
        """Test create_opsgenie_handler factory."""
        handler = create_opsgenie_handler(
            api_key="test-api-key",
            priority=OpsgeniePriority.P2,
        )
        assert isinstance(handler, OpsgenieNotificationHandler)
        assert handler.opsgenie_config.default_priority == OpsgeniePriority.P2

    def test_create_opsgenie_handler_with_team(self) -> None:
        """Test create_opsgenie_handler with team."""
        handler = create_opsgenie_handler(
            api_key="test-api-key",
            team="platform-team",
        )
        # Should have one team responder
        assert len(handler.opsgenie_config.default_responders) == 1
        assert handler.opsgenie_config.default_responders[0].type == "team"
        assert handler.opsgenie_config.default_responders[0].name == "platform-team"

    def test_create_opsgenie_handler_with_tags(self) -> None:
        """Test create_opsgenie_handler with tags."""
        handler = create_opsgenie_handler(
            api_key="test-api-key",
            tags=["production", "critical"],
        )
        assert "production" in handler.opsgenie_config.default_tags
        assert "critical" in handler.opsgenie_config.default_tags


# =============================================================================
# Handler State Tests
# =============================================================================


class TestHandlerState:
    """Tests for handler state management."""

    def test_pagerduty_handler_enabled_by_default(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
    ) -> None:
        """Test that PagerDuty handler is enabled by default."""
        assert pagerduty_handler.enabled

    def test_opsgenie_handler_enabled_by_default(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
    ) -> None:
        """Test that Opsgenie handler is enabled by default."""
        assert opsgenie_handler.enabled

    def test_pagerduty_disable_enable(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
    ) -> None:
        """Test disabling and enabling PagerDuty handler."""
        pagerduty_handler.disable()
        assert not pagerduty_handler.enabled

        pagerduty_handler.enable()
        assert pagerduty_handler.enabled

    def test_opsgenie_disable_enable(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
    ) -> None:
        """Test disabling and enabling Opsgenie handler."""
        opsgenie_handler.disable()
        assert not opsgenie_handler.enabled

        opsgenie_handler.enable()
        assert opsgenie_handler.enabled

    @pytest.mark.asyncio
    async def test_disabled_pagerduty_skips_send(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that disabled PagerDuty handler skips sending."""
        from packages.enterprise.notifications.types import NotificationStatus

        pagerduty_handler.disable()
        result = await pagerduty_handler.send(basic_payload)
        # Skipped result has success=True but status=SKIPPED
        assert result.status == NotificationStatus.SKIPPED
        assert "Filtered" in (result.error or "")

    @pytest.mark.asyncio
    async def test_disabled_opsgenie_skips_send(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that disabled Opsgenie handler skips sending."""
        from packages.enterprise.notifications.types import NotificationStatus

        opsgenie_handler.disable()
        result = await opsgenie_handler.send(basic_payload)
        # Skipped result has success=True but status=SKIPPED
        assert result.status == NotificationStatus.SKIPPED
        assert "Filtered" in (result.error or "")


# =============================================================================
# Incident Details Extraction Tests
# =============================================================================


class TestIncidentDetailsExtraction:
    """Tests for incident details extraction from payload."""

    def test_pagerduty_extract_details(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test extracting incident details from payload for PagerDuty."""
        details = pagerduty_handler._extract_incident_details(basic_payload)

        # IncidentDetails doesn't have title/message fields - those come from payload
        # Check severity mapping: ERROR -> HIGH
        assert details.severity == IncidentSeverity.HIGH
        # Check that source is set
        assert details.source is not None

    def test_opsgenie_extract_details(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test extracting incident details from payload for Opsgenie."""
        details = opsgenie_handler._extract_incident_details(basic_payload)

        # IncidentDetails doesn't have title/message fields
        # Check severity mapping: ERROR -> HIGH
        assert details.severity == IncidentSeverity.HIGH
        # Check that source is set
        assert details.source is not None

    def test_extract_dedup_key_from_context(
        self,
        pagerduty_handler: PagerDutyNotificationHandler,
        critical_payload: NotificationPayload,
    ) -> None:
        """Test extracting dedup_key from payload context."""
        details = pagerduty_handler._extract_incident_details(critical_payload)
        assert details.dedup_key == "txn_check_001"

    def test_extract_custom_details_from_context(
        self,
        opsgenie_handler: OpsgenieNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that context is converted to custom details."""
        details = opsgenie_handler._extract_incident_details(basic_payload)

        # custom_details is a tuple of tuples, convert to dict for checking
        custom_dict = details.custom_details_dict
        assert "check_id" in custom_dict
        assert custom_dict["check_id"] == "chk_123"
