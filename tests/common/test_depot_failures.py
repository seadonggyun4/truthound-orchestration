"""Tests for Depot failure taxonomy."""

from __future__ import annotations

from common.depot.failures import (
    DepotClientError,
    DepotFailureCode,
    DepotPollingTimeoutError,
    DepotProtocolError,
    RetryDisposition,
    build_failure,
    build_failure_from_result,
    classify_exception_failure,
    classify_operation_result_failure,
    failure_code_to_retry_disposition,
    is_retryable_failure,
    is_wait_state_failure,
    normalize_failure_code,
)
from common.depot.models import DepotOperationStatus
from common.depot.testing import build_operation_result


def test_failure_code_maps_to_expected_retry_disposition() -> None:
    assert failure_code_to_retry_disposition(DepotFailureCode.CONNECTOR_FAILURE) == RetryDisposition.RETRYABLE
    assert failure_code_to_retry_disposition(DepotFailureCode.VALIDATION_FAILURE) == RetryDisposition.NON_RETRYABLE
    assert failure_code_to_retry_disposition(DepotFailureCode.APPROVAL_MISSING) == RetryDisposition.WAIT_REQUIRED


def test_runtime_and_validation_failure_have_distinct_retry_behavior() -> None:
    assert is_retryable_failure(DepotFailureCode.VALIDATION_RUNTIME_ERROR) is True
    assert is_retryable_failure(DepotFailureCode.VALIDATION_FAILURE) is False


def test_build_failure_keeps_compact_details_without_raw_cause() -> None:
    failure = build_failure(
        DepotFailureCode.DEPOT_API_FAILURE,
        "boom",
        details={"status_code": 500},
        cause=ValueError("raw"),
    )

    payload = failure.to_dict()
    assert payload["details"] == {"status_code": 500}
    assert payload["cause_type"] == "ValueError"
    assert is_wait_state_failure(DepotFailureCode.APPROVAL_MISSING) is True


def test_unknown_error_code_falls_back_to_depot_api_failure() -> None:
    failure = build_failure_from_result(
        build_operation_result(
            status=DepotOperationStatus.FAILED,
            error_code="weird_remote_code",
            error_message="remote exploded",
        )
    )

    assert failure.code == DepotFailureCode.DEPOT_API_FAILURE
    assert failure.details["original_error_code"] == "weird_remote_code"


def test_classify_exception_failure_normalizes_transport_and_protocol_errors() -> None:
    transport = classify_exception_failure(DepotClientError("network down"))
    protocol = classify_exception_failure(DepotProtocolError("bad payload"))
    timeout = classify_exception_failure(DepotPollingTimeoutError("waited too long"))

    assert transport.code == DepotFailureCode.DEPOT_API_FAILURE
    assert protocol.code == DepotFailureCode.DEPOT_API_FAILURE
    assert timeout.code == DepotFailureCode.DEPOT_API_FAILURE
    assert transport.retry_disposition == RetryDisposition.RETRYABLE


def test_classify_operation_result_failure_ignores_waiting_results() -> None:
    waiting = classify_operation_result_failure(
        build_operation_result(status=DepotOperationStatus.WAITING)
    )
    cancelled = classify_operation_result_failure(
        build_operation_result(status=DepotOperationStatus.CANCELLED, error_message="cancelled")
    )

    assert waiting is None
    assert cancelled is not None
    assert cancelled.code == DepotFailureCode.DEPOT_API_FAILURE


def test_normalize_failure_code_preserves_known_and_falls_back_for_unknown() -> None:
    assert normalize_failure_code("approval_missing") == DepotFailureCode.APPROVAL_MISSING
    assert normalize_failure_code("made_up") == DepotFailureCode.DEPOT_API_FAILURE
