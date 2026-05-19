"""Tests for Depot retry classification."""

from __future__ import annotations

from urllib.error import URLError

from common.depot.client import (
    DepotClient,
    DepotClientConfig,
    DepotClientResponse,
    RetryableDepotHttpError,
    _is_retryable_status_code,
)
from common.depot.failures import DepotClientError, DepotProtocolError
from common.depot.testing import (
    FakeDepotResponse,
    FakeDepotTransport,
    build_client_payload,
    build_operation_request,
    build_operation_result,
)


def test_503_and_504_are_retryable_status_codes() -> None:
    assert _is_retryable_status_code(503) is True
    assert _is_retryable_status_code(504) is True


def test_protocol_errors_are_not_retryable() -> None:
    filter_ = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret")
    )._retry._exception_filter
    assert filter_.should_retry(DepotProtocolError("bad payload"), 1) is False


def test_transport_failures_are_retryable() -> None:
    filter_ = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret")
    )._retry._exception_filter
    assert filter_.should_retry(DepotClientError("network"), 1) is True
    assert filter_.should_retry(URLError("down"), 1) is True


def test_retryable_http_error_carries_response_for_final_normalization() -> None:
    transport = FakeDepotTransport()
    payload = build_client_payload(
        result=build_operation_result(error_code="depot_api_failure", error_message="busy")
    )
    transport.queue(
        FakeDepotResponse(status_code=503, payload=payload),
        FakeDepotResponse(status_code=503, payload=payload),
        FakeDepotResponse(status_code=503, payload=payload),
    )
    client = DepotClient(
        DepotClientConfig(base_url="https://depot.truthound.test", api_token="secret"),
        transport=transport,
    )

    result = client.submit_operation(build_operation_request())

    assert result.error_code == "depot_api_failure"
    assert len(transport.requests) == 3


def test_retryable_http_wrapper_is_retry_class_only() -> None:
    response = DepotClientResponse(status_code=503, payload=build_client_payload())
    exc = RetryableDepotHttpError("retry me", response=response)
    assert exc.response is response
