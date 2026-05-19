"""Tests for the shared Depot API client."""

from __future__ import annotations

import pytest

from common.depot.client import DepotClient, DepotClientConfig, _is_retryable_status_code
from common.depot.failures import DepotClientError, DepotPollingTimeoutError, DepotProtocolError
from common.depot.models import DepotOperationStatus, DepotOperationType
from common.depot.polling import classify_status
from common.depot.testing import (
    FakeDepotResponse,
    FakeDepotTransport,
    build_client_payload,
    build_operation_request,
    build_operation_result,
)


def test_submit_operation_sends_auth_and_idempotency_headers() -> None:
    transport = FakeDepotTransport()
    transport.queue(FakeDepotResponse(payload=build_client_payload()))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.submit_operation(build_operation_request())

    assert result.status == DepotOperationStatus.SUCCEEDED
    sent = transport.requests[0]
    assert sent["headers"]["Authorization"] == "Bearer secret"
    assert sent["headers"]["Idempotency-Key"] == "pull_snapshot:depot-1:asset-1:main"
    assert sent["headers"]["X-Truthound-Operation-Id"] == "op-123"
    assert sent["decoded_body"]["operation_type"] == "pull_snapshot"


def test_submit_operation_normalizes_http_failure_into_result() -> None:
    transport = FakeDepotTransport()
    failed = build_client_payload(
        result=build_operation_result(
            status=DepotOperationStatus.FAILED,
            error_code="depot_api_failure",
            error_message="service down",
        ),
    )
    transport.queue(
        FakeDepotResponse(status_code=503, payload=failed),
        FakeDepotResponse(status_code=503, payload=failed),
        FakeDepotResponse(status_code=503, payload=failed),
    )
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.submit_operation(build_operation_request())

    assert result.status == DepotOperationStatus.FAILED
    assert result.error_code == "depot_api_failure"


def test_client_rejects_malformed_payload() -> None:
    transport = FakeDepotTransport()
    transport.queue(FakeDepotResponse(payload={"status": "succeeded"}))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    with pytest.raises(DepotProtocolError):
        client.get_operation("op-123")


def test_client_rejects_malformed_error_payload() -> None:
    transport = FakeDepotTransport()
    transport.queue(FakeDepotResponse(status_code=400, payload={"message": "bad request"}))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    with pytest.raises(DepotProtocolError):
        client.submit_operation(build_operation_request())


def test_client_preserves_request_id_from_headers() -> None:
    transport = FakeDepotTransport()
    transport.queue(
        FakeDepotResponse(
            payload=build_client_payload(extra_payload={"request_id": None}),
            headers={"x-request-id": "req-from-header"},
        )
    )
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.submit_operation(build_operation_request())

    assert result.metadata["request_id"] == "req-from-header"


def test_client_preserves_request_id_from_payload() -> None:
    transport = FakeDepotTransport()
    transport.queue(FakeDepotResponse(payload=build_client_payload(request_id="req-from-payload")))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.submit_operation(build_operation_request())

    assert result.metadata["request_id"] == "req-from-payload"


def test_get_operation_does_not_send_idempotency_headers() -> None:
    transport = FakeDepotTransport()
    transport.queue(FakeDepotResponse(payload=build_client_payload()))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    client.get_operation("op-123")

    sent = transport.requests[0]
    assert "Idempotency-Key" not in sent["headers"]
    assert "X-Truthound-Operation-Id" not in sent["headers"]


def test_client_ignores_authorization_override_in_extra_headers() -> None:
    transport = FakeDepotTransport()
    transport.queue(FakeDepotResponse(payload=build_client_payload()))
    client = DepotClient(
        DepotClientConfig(
            base_url="https://depot.truthound.test",
            api_token="secret",
            headers={"Authorization": "Bearer nope", "X-Extra": "1"},
        ),
        transport=transport,
    )

    client.submit_operation(build_operation_request())

    sent = transport.requests[0]
    assert sent["headers"]["Authorization"] == "Bearer secret"
    assert sent["headers"]["X-Extra"] == "1"


def test_network_failure_raises_client_error() -> None:
    transport = FakeDepotTransport()
    transport.queue_exception(
        DepotClientError("network down"),
        DepotClientError("network down"),
        DepotClientError("network down"),
    )
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    with pytest.raises(DepotClientError):
        client.submit_operation(build_operation_request())


def test_transient_503_retries_then_succeeds() -> None:
    transport = FakeDepotTransport()
    failed = build_client_payload(
        result=build_operation_result(
            status=DepotOperationStatus.FAILED,
            error_code="depot_api_failure",
            error_message="service down",
        )
    )
    transport.queue(
        FakeDepotResponse(status_code=503, payload=failed),
        FakeDepotResponse(payload=build_client_payload()),
    )
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.submit_operation(build_operation_request())

    assert result.status == DepotOperationStatus.SUCCEEDED
    assert len(transport.requests) == 2
    assert (
        transport.requests[0]["headers"]["Idempotency-Key"]
        == transport.requests[1]["headers"]["Idempotency-Key"]
    )


def test_non_retryable_409_normalizes_without_retry() -> None:
    transport = FakeDepotTransport()
    failed = build_client_payload(
        result=build_operation_result(
            status=DepotOperationStatus.FAILED,
            error_code="conflict_detected",
            error_message="conflict",
        )
    )
    transport.queue(FakeDepotResponse(status_code=409, payload=failed))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.submit_operation(build_operation_request())

    assert result.status == DepotOperationStatus.FAILED
    assert result.error_code == "conflict_detected"
    assert len(transport.requests) == 1


def test_wait_for_operation_uses_shared_polling() -> None:
    transport = FakeDepotTransport()
    running = build_client_payload(
        result=build_operation_result(status=DepotOperationStatus.RUNNING)
    )
    done = build_client_payload(
        result=build_operation_result(
            status=DepotOperationStatus.SUCCEEDED,
            operation_type=DepotOperationType.RELEASE_TAG,
        )
    )
    transport.queue(FakeDepotResponse(payload=running), FakeDepotResponse(payload=done))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.wait_for_operation("op-123", timeout_seconds=1.0, poll_interval_seconds=0.0)

    assert result.status == DepotOperationStatus.SUCCEEDED
    assert len(transport.requests) == 2


def test_waiting_then_failed_polling_surfaces_failed_result() -> None:
    transport = FakeDepotTransport()
    waiting = build_client_payload(
        result=build_operation_result(status=DepotOperationStatus.WAITING)
    )
    failed = build_client_payload(
        result=build_operation_result(
            status=DepotOperationStatus.FAILED,
            error_code="approval_missing",
            error_message="approval required",
        )
    )
    transport.queue(FakeDepotResponse(payload=waiting), FakeDepotResponse(payload=failed))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.wait_for_operation("op-123", timeout_seconds=1.0, poll_interval_seconds=0.0)

    assert result.status == DepotOperationStatus.FAILED
    assert result.error_code == "approval_missing"


def test_wait_for_operation_times_out_when_result_never_reaches_terminal_state() -> None:
    transport = FakeDepotTransport()
    running = build_client_payload(
        result=build_operation_result(status=DepotOperationStatus.RUNNING)
    )
    transport.queue(FakeDepotResponse(payload=running), FakeDepotResponse(payload=running))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    with pytest.raises(DepotPollingTimeoutError):
        client.wait_for_operation("op-123", timeout_seconds=0.0, poll_interval_seconds=0.0)


def test_no_op_is_treated_as_terminal() -> None:
    transport = FakeDepotTransport()
    noop = build_client_payload(
        result=build_operation_result(status=DepotOperationStatus.NO_OP)
    )
    transport.queue(FakeDepotResponse(payload=noop))
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.wait_for_operation("op-123", timeout_seconds=1.0, poll_interval_seconds=0.0)

    assert result.status == DepotOperationStatus.NO_OP


def test_status_classification_is_business_agnostic() -> None:
    assert classify_status(DepotOperationStatus.WAITING) == "wait"
    assert classify_status(DepotOperationStatus.RUNNING) == "wait"
    assert classify_status(DepotOperationStatus.SUCCEEDED) == "terminal"


def test_retryable_status_code_helper() -> None:
    assert _is_retryable_status_code(503) is True
    assert _is_retryable_status_code(504) is True
    assert _is_retryable_status_code(409) is False
