"""Tests for SLA callbacks module."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestDataQualityCallbackProtocol:
    """Tests for DataQualityCallback protocol."""

    def test_protocol_compliance(self) -> None:
        """Test that a class can implement the protocol."""
        from truthound_airflow.sla.callbacks import DataQualityCallback

        class CustomCallback:
            def on_success(self, result: dict, context: Any) -> None:
                pass

            def on_failure(self, result: dict, violations: list, context: Any) -> None:
                pass

            def on_sla_violation(self, violations: list, context: Any) -> None:
                pass

        callback = CustomCallback()
        assert isinstance(callback, DataQualityCallback)


class TestBaseDataQualityCallback:
    """Tests for BaseDataQualityCallback class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        from truthound_airflow.sla.callbacks import BaseDataQualityCallback

        class TestCallback(BaseDataQualityCallback):
            pass

        callback = TestCallback()

        assert callback.name == "TestCallback"
        assert callback.enabled is True

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        from truthound_airflow.sla.callbacks import BaseDataQualityCallback

        class TestCallback(BaseDataQualityCallback):
            pass

        callback = TestCallback(name="custom_name", enabled=False)

        assert callback.name == "custom_name"
        assert callback.enabled is False

    def test_default_handlers_are_noop(self, airflow_context: dict[str, Any]) -> None:
        """Test that default handlers do nothing."""
        from truthound_airflow.sla.callbacks import BaseDataQualityCallback

        class TestCallback(BaseDataQualityCallback):
            pass

        callback = TestCallback()

        # Should not raise
        callback.on_success({}, airflow_context)
        callback.on_failure({}, [], airflow_context)
        callback.on_sla_violation([], airflow_context)


class TestDataQualitySLACallback:
    """Tests for DataQualitySLACallback class."""

    def test_initialization(self) -> None:
        """Test callback initialization."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import SLAConfig

        config = SLAConfig(max_failure_rate=0.05)
        callback = DataQualitySLACallback(config=config)

        assert callback.config == config
        assert callback.raise_on_violation is False
        assert callback.enabled is True

    def test_initialization_with_options(self) -> None:
        """Test callback initialization with options."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import SLAConfig

        violation_handler = MagicMock()
        config = SLAConfig(max_failure_rate=0.05)

        callback = DataQualitySLACallback(
            config=config,
            on_violation=violation_handler,
            raise_on_violation=True,
            name="test_callback",
            enabled=False,
        )

        assert callback.raise_on_violation is True
        assert callback.name == "test_callback"
        assert callback.enabled is False

    def test_on_success_when_disabled(self, airflow_context: dict[str, Any]) -> None:
        """Test on_success when callback is disabled."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import SLAConfig

        callback = DataQualitySLACallback(
            config=SLAConfig(),
            enabled=False,
        )

        # Should not raise or do anything
        callback.on_success({"passed_count": 10}, airflow_context)

    def test_on_success_extracts_metrics(self, airflow_context: dict[str, Any]) -> None:
        """Test that on_success extracts metrics from result."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import SLAConfig

        callback = DataQualitySLACallback(
            config=SLAConfig(max_failure_rate=0.05),
        )

        result = {
            "passed_count": 95,
            "failed_count": 5,
            "warning_count": 0,
            "execution_time_ms": 1000.0,
        }

        # Should not raise
        callback.on_success(result, airflow_context)

    def test_on_failure_when_disabled(self, airflow_context: dict[str, Any]) -> None:
        """Test on_failure when callback is disabled."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import SLAConfig

        callback = DataQualitySLACallback(
            config=SLAConfig(),
            enabled=False,
        )

        # Should not raise or do anything
        callback.on_failure({"passed_count": 10}, [], airflow_context)

    def test_on_failure_calls_violation_handler(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test that on_failure calls the violation handler."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAConfig,
            SLAViolation,
            SLAViolationType,
        )

        violation_handler = MagicMock()
        callback = DataQualitySLACallback(
            config=SLAConfig(max_failure_rate=0.01),
            on_violation=violation_handler,
        )

        result = {
            "passed_count": 90,
            "failed_count": 10,
            "execution_time_ms": 100.0,
        }

        pre_violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Pre-existing violation",
            alert_level=AlertLevel.ERROR,
        )

        callback.on_failure(result, [pre_violation], airflow_context)

        # Handler should be called with violations
        assert violation_handler.called

    def test_on_failure_raises_when_configured(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test that on_failure raises when raise_on_violation is True."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAConfig,
            SLAViolation,
            SLAViolationType,
        )

        callback = DataQualitySLACallback(
            config=SLAConfig(),
            raise_on_violation=True,
        )

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test violation",
            alert_level=AlertLevel.ERROR,
        )

        with pytest.raises(Exception):  # AirflowException
            callback.on_failure({}, [violation], airflow_context)

    def test_on_sla_violation_when_disabled(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_sla_violation when callback is disabled."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import SLAConfig

        callback = DataQualitySLACallback(
            config=SLAConfig(),
            enabled=False,
        )

        # Should not raise or do anything
        callback.on_sla_violation([], airflow_context)

    def test_on_sla_violation_calls_handler(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test that on_sla_violation calls the violation handler."""
        from truthound_airflow.sla.callbacks import DataQualitySLACallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAConfig,
            SLAViolation,
            SLAViolationType,
        )

        violation_handler = MagicMock()
        callback = DataQualitySLACallback(
            config=SLAConfig(),
            on_violation=violation_handler,
        )

        violation = SLAViolation(
            violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
            message="Pass rate too low",
            alert_level=AlertLevel.WARNING,
        )

        callback.on_sla_violation([violation], airflow_context)

        violation_handler.assert_called_once()


class TestAlertHandlers:
    """Tests for AlertHandlers dataclass."""

    def test_default_handlers_are_none(self) -> None:
        """Test that default handlers are None."""
        from truthound_airflow.sla.callbacks import AlertHandlers

        handlers = AlertHandlers()

        assert handlers.on_info is None
        assert handlers.on_warning is None
        assert handlers.on_error is None
        assert handlers.on_critical is None

    def test_get_handler_for_level(self) -> None:
        """Test getting handler for a specific level."""
        from truthound_airflow.sla.callbacks import AlertHandlers
        from truthound_airflow.sla.config import AlertLevel

        info_handler = MagicMock()
        warning_handler = MagicMock()
        error_handler = MagicMock()
        critical_handler = MagicMock()

        handlers = AlertHandlers(
            on_info=info_handler,
            on_warning=warning_handler,
            on_error=error_handler,
            on_critical=critical_handler,
        )

        assert handlers.get_handler(AlertLevel.INFO) == info_handler
        assert handlers.get_handler(AlertLevel.WARNING) == warning_handler
        assert handlers.get_handler(AlertLevel.ERROR) == error_handler
        assert handlers.get_handler(AlertLevel.CRITICAL) == critical_handler


class TestQualityAlertCallback:
    """Tests for QualityAlertCallback class."""

    def test_initialization(self) -> None:
        """Test callback initialization."""
        from truthound_airflow.sla.callbacks import QualityAlertCallback

        callback = QualityAlertCallback()

        assert callback.enabled is True
        assert callback.handlers is not None

    def test_initialization_with_handlers(self) -> None:
        """Test callback initialization with handlers."""
        from truthound_airflow.sla.callbacks import AlertHandlers, QualityAlertCallback

        warning_handler = MagicMock()
        handlers = AlertHandlers(on_warning=warning_handler)

        callback = QualityAlertCallback(handlers=handlers)

        assert callback.handlers == handlers

    def test_on_success_when_disabled(self, airflow_context: dict[str, Any]) -> None:
        """Test on_success when callback is disabled."""
        from truthound_airflow.sla.callbacks import QualityAlertCallback

        success_handler = MagicMock()
        callback = QualityAlertCallback(
            on_success_handler=success_handler,
            enabled=False,
        )

        callback.on_success({"passed_count": 10}, airflow_context)

        success_handler.assert_not_called()

    def test_on_success_calls_handler(self, airflow_context: dict[str, Any]) -> None:
        """Test on_success calls the success handler."""
        from truthound_airflow.sla.callbacks import QualityAlertCallback

        success_handler = MagicMock()
        callback = QualityAlertCallback(on_success_handler=success_handler)

        result = {"passed_count": 10}
        callback.on_success(result, airflow_context)

        success_handler.assert_called_once_with(result, airflow_context)

    def test_on_failure_routes_to_level_handler(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_failure routes to level-specific handler."""
        from truthound_airflow.sla.callbacks import AlertHandlers, QualityAlertCallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAViolation,
            SLAViolationType,
        )

        error_handler = MagicMock()
        handlers = AlertHandlers(on_error=error_handler)
        callback = QualityAlertCallback(handlers=handlers)

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="High failure rate",
            alert_level=AlertLevel.ERROR,
        )

        callback.on_failure({}, [violation], airflow_context)

        error_handler.assert_called_once_with(violation, airflow_context)

    def test_on_failure_uses_general_handler(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_failure uses general failure handler when no level handler."""
        from truthound_airflow.sla.callbacks import QualityAlertCallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAViolation,
            SLAViolationType,
        )

        failure_handler = MagicMock()
        callback = QualityAlertCallback(on_failure=failure_handler)

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="High failure rate",
            alert_level=AlertLevel.ERROR,
        )

        callback.on_failure({}, [violation], airflow_context)

        failure_handler.assert_called_once_with(violation, airflow_context)

    def test_on_failure_uses_default_handler(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_failure uses default handler as fallback."""
        from truthound_airflow.sla.callbacks import QualityAlertCallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAViolation,
            SLAViolationType,
        )

        default_handler = MagicMock()
        callback = QualityAlertCallback(default_handler=default_handler)

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="High failure rate",
            alert_level=AlertLevel.ERROR,
        )

        callback.on_failure({}, [violation], airflow_context)

        default_handler.assert_called_once_with(violation, airflow_context)

    def test_on_sla_violation_routes_alerts(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_sla_violation routes alerts to appropriate handlers."""
        from truthound_airflow.sla.callbacks import AlertHandlers, QualityAlertCallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAViolation,
            SLAViolationType,
        )

        warning_handler = MagicMock()
        critical_handler = MagicMock()
        handlers = AlertHandlers(
            on_warning=warning_handler,
            on_critical=critical_handler,
        )
        callback = QualityAlertCallback(handlers=handlers)

        violations = [
            SLAViolation(
                violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
                message="Warning violation",
                alert_level=AlertLevel.WARNING,
            ),
            SLAViolation(
                violation_type=SLAViolationType.CONSECUTIVE_FAILURES,
                message="Critical violation",
                alert_level=AlertLevel.CRITICAL,
            ),
        ]

        callback.on_sla_violation(violations, airflow_context)

        warning_handler.assert_called_once()
        critical_handler.assert_called_once()


class TestCallbackChain:
    """Tests for CallbackChain class."""

    def test_initialization(self) -> None:
        """Test chain initialization."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TestCallback(BaseDataQualityCallback):
            pass

        callbacks = [TestCallback(), TestCallback()]
        chain = CallbackChain(callbacks)

        assert len(chain.callbacks) == 2
        assert chain.stop_on_error is False

    def test_on_success_calls_all_callbacks(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_success calls all callbacks."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TrackingCallback(BaseDataQualityCallback):
            called = False

            def on_success(self, result: dict, context: Any) -> None:
                self.called = True

        callbacks = [TrackingCallback(), TrackingCallback()]
        chain = CallbackChain(callbacks)

        chain.on_success({"passed_count": 10}, airflow_context)

        assert all(cb.called for cb in callbacks)

    def test_on_success_skips_disabled_callbacks(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_success skips disabled callbacks."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TrackingCallback(BaseDataQualityCallback):
            called = False

            def on_success(self, result: dict, context: Any) -> None:
                self.called = True

        enabled = TrackingCallback(enabled=True)
        disabled = TrackingCallback(enabled=False)
        chain = CallbackChain([enabled, disabled])

        chain.on_success({"passed_count": 10}, airflow_context)

        assert enabled.called is True
        assert disabled.called is False

    def test_on_failure_calls_all_callbacks(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_failure calls all callbacks."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TrackingCallback(BaseDataQualityCallback):
            called = False

            def on_failure(
                self, result: dict, violations: list, context: Any
            ) -> None:
                self.called = True

        callbacks = [TrackingCallback(), TrackingCallback()]
        chain = CallbackChain(callbacks)

        chain.on_failure({"failed_count": 5}, [], airflow_context)

        assert all(cb.called for cb in callbacks)

    def test_on_sla_violation_calls_all_callbacks(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_sla_violation calls all callbacks."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TrackingCallback(BaseDataQualityCallback):
            called = False

            def on_sla_violation(self, violations: list, context: Any) -> None:
                self.called = True

        callbacks = [TrackingCallback(), TrackingCallback()]
        chain = CallbackChain(callbacks)

        chain.on_sla_violation([], airflow_context)

        assert all(cb.called for cb in callbacks)

    def test_continues_on_error_by_default(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test chain continues on error by default."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class ErrorCallback(BaseDataQualityCallback):
            def on_success(self, result: dict, context: Any) -> None:
                raise ValueError("Error!")

        class TrackingCallback(BaseDataQualityCallback):
            called = False

            def on_success(self, result: dict, context: Any) -> None:
                self.called = True

        error_cb = ErrorCallback()
        tracking_cb = TrackingCallback()
        chain = CallbackChain([error_cb, tracking_cb], stop_on_error=False)

        chain.on_success({}, airflow_context)

        assert tracking_cb.called is True

    def test_stops_on_error_when_configured(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test chain stops on error when configured."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class ErrorCallback(BaseDataQualityCallback):
            def on_success(self, result: dict, context: Any) -> None:
                raise ValueError("Error!")

        class TrackingCallback(BaseDataQualityCallback):
            called = False

            def on_success(self, result: dict, context: Any) -> None:
                self.called = True

        error_cb = ErrorCallback()
        tracking_cb = TrackingCallback()
        chain = CallbackChain([error_cb, tracking_cb], stop_on_error=True)

        with pytest.raises(ValueError):
            chain.on_success({}, airflow_context)

        assert tracking_cb.called is False

    def test_add_callback(self) -> None:
        """Test adding callback to chain."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TestCallback(BaseDataQualityCallback):
            pass

        chain = CallbackChain([])
        callback = TestCallback()

        chain.add(callback)

        assert callback in chain.callbacks

    def test_remove_callback(self) -> None:
        """Test removing callback from chain."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TestCallback(BaseDataQualityCallback):
            pass

        callback = TestCallback()
        chain = CallbackChain([callback])

        result = chain.remove(callback)

        assert result is True
        assert callback not in chain.callbacks

    def test_remove_nonexistent_callback(self) -> None:
        """Test removing non-existent callback returns False."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TestCallback(BaseDataQualityCallback):
            pass

        chain = CallbackChain([])
        callback = TestCallback()

        result = chain.remove(callback)

        assert result is False

    def test_chain_disabled(self, airflow_context: dict[str, Any]) -> None:
        """Test disabled chain doesn't call callbacks."""
        from truthound_airflow.sla.callbacks import (
            BaseDataQualityCallback,
            CallbackChain,
        )

        class TrackingCallback(BaseDataQualityCallback):
            called = False

            def on_success(self, result: dict, context: Any) -> None:
                self.called = True

        callback = TrackingCallback()
        chain = CallbackChain([callback], enabled=False)

        chain.on_success({}, airflow_context)

        assert callback.called is False


class TestLoggingCallback:
    """Tests for LoggingCallback class."""

    def test_initialization(self) -> None:
        """Test callback initialization."""
        from truthound_airflow.sla.callbacks import LoggingCallback

        callback = LoggingCallback()

        assert callback.log_level == logging.INFO
        assert callback.enabled is True

    def test_custom_log_level(self) -> None:
        """Test callback with custom log level."""
        from truthound_airflow.sla.callbacks import LoggingCallback

        callback = LoggingCallback(log_level=logging.DEBUG)

        assert callback.log_level == logging.DEBUG

    def test_on_success_logs_message(
        self,
        airflow_context: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test on_success logs a message."""
        from truthound_airflow.sla.callbacks import LoggingCallback

        callback = LoggingCallback()

        result = {
            "passed_count": 95,
            "failed_count": 5,
            "execution_time_ms": 100.0,
        }

        with caplog.at_level(logging.INFO):
            callback.on_success(result, airflow_context)

        assert "SUCCESS" in caplog.text
        assert "passed=95" in caplog.text

    def test_on_failure_logs_message(
        self,
        airflow_context: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test on_failure logs a message."""
        from truthound_airflow.sla.callbacks import LoggingCallback

        callback = LoggingCallback()

        result = {
            "passed_count": 90,
            "failed_count": 10,
        }

        with caplog.at_level(logging.INFO):
            callback.on_failure(result, [], airflow_context)

        assert "FAILURE" in caplog.text
        assert "failed=10" in caplog.text

    def test_on_sla_violation_logs_violations(
        self,
        airflow_context: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test on_sla_violation logs violations."""
        from truthound_airflow.sla.callbacks import LoggingCallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAViolation,
            SLAViolationType,
        )

        callback = LoggingCallback()

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test violation message",
            alert_level=AlertLevel.ERROR,
        )

        with caplog.at_level(logging.INFO):
            callback.on_sla_violation([violation], airflow_context)

        assert "SLA VIOLATION" in caplog.text
        assert "Test violation message" in caplog.text

    def test_disabled_callback_does_not_log(
        self,
        airflow_context: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test disabled callback doesn't log."""
        from truthound_airflow.sla.callbacks import LoggingCallback

        callback = LoggingCallback(enabled=False)

        with caplog.at_level(logging.INFO):
            callback.on_success({}, airflow_context)

        assert "SUCCESS" not in caplog.text


class TestMetricsCallback:
    """Tests for MetricsCallback class."""

    def test_initialization(self) -> None:
        """Test callback initialization."""
        from truthound_airflow.sla.callbacks import MetricsCallback

        callback = MetricsCallback()

        assert callback.total_success == 0
        assert callback.total_failure == 0
        assert callback.total_violations == 0
        assert callback.success_rate == 1.0

    def test_on_success_increments_counter(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_success increments success counter."""
        from truthound_airflow.sla.callbacks import MetricsCallback

        callback = MetricsCallback()

        callback.on_success({"execution_time_ms": 100.0}, airflow_context)
        callback.on_success({"execution_time_ms": 200.0}, airflow_context)

        assert callback.total_success == 2
        assert callback.total_failure == 0

    def test_on_failure_increments_counter(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_failure increments failure counter."""
        from truthound_airflow.sla.callbacks import MetricsCallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAViolation,
            SLAViolationType,
        )

        callback = MetricsCallback()

        violation = SLAViolation(
            violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
            message="Test",
            alert_level=AlertLevel.ERROR,
        )

        callback.on_failure({"execution_time_ms": 100.0}, [violation], airflow_context)

        assert callback.total_failure == 1
        assert callback.total_violations == 1

    def test_on_sla_violation_increments_counter(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test on_sla_violation increments violation counter."""
        from truthound_airflow.sla.callbacks import MetricsCallback
        from truthound_airflow.sla.config import (
            AlertLevel,
            SLAViolation,
            SLAViolationType,
        )

        callback = MetricsCallback()

        violations = [
            SLAViolation(
                violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
                message="Test 1",
                alert_level=AlertLevel.ERROR,
            ),
            SLAViolation(
                violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
                message="Test 2",
                alert_level=AlertLevel.WARNING,
            ),
        ]

        callback.on_sla_violation(violations, airflow_context)

        assert callback.total_violations == 2

    def test_success_rate_calculation(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test success rate calculation."""
        from truthound_airflow.sla.callbacks import MetricsCallback

        callback = MetricsCallback()

        for _ in range(3):
            callback.on_success({"execution_time_ms": 100.0}, airflow_context)

        callback.on_failure({"execution_time_ms": 100.0}, [], airflow_context)

        assert callback.success_rate == 0.75

    def test_average_execution_time(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test average execution time calculation."""
        from truthound_airflow.sla.callbacks import MetricsCallback

        callback = MetricsCallback()

        callback.on_success({"execution_time_ms": 100.0}, airflow_context)
        callback.on_success({"execution_time_ms": 200.0}, airflow_context)
        callback.on_failure({"execution_time_ms": 300.0}, [], airflow_context)

        assert callback.average_execution_time_ms == 200.0

    def test_reset_clears_all_metrics(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test reset clears all metrics."""
        from truthound_airflow.sla.callbacks import MetricsCallback

        callback = MetricsCallback()

        callback.on_success({"execution_time_ms": 100.0}, airflow_context)
        callback.on_failure({"execution_time_ms": 100.0}, [], airflow_context)

        callback.reset()

        assert callback.total_success == 0
        assert callback.total_failure == 0
        assert callback.total_violations == 0
        assert callback.success_rate == 1.0
        assert callback.average_execution_time_ms == 0.0

    def test_get_stats_returns_all_metrics(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test get_stats returns all metrics."""
        from truthound_airflow.sla.callbacks import MetricsCallback

        callback = MetricsCallback()

        callback.on_success({"execution_time_ms": 100.0}, airflow_context)
        callback.on_failure({"execution_time_ms": 200.0}, [], airflow_context)

        stats = callback.get_stats()

        assert stats["total_success"] == 1
        assert stats["total_failure"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["execution_count"] == 2
        assert stats["average_execution_time_ms"] == 150.0

    def test_disabled_callback_does_not_track(
        self,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test disabled callback doesn't track metrics."""
        from truthound_airflow.sla.callbacks import MetricsCallback

        callback = MetricsCallback(enabled=False)

        callback.on_success({"execution_time_ms": 100.0}, airflow_context)
        callback.on_failure({"execution_time_ms": 100.0}, [], airflow_context)

        assert callback.total_success == 0
        assert callback.total_failure == 0
