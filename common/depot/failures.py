"""Depot-specific failure taxonomy layered on top of common exceptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from common.depot.models import make_json_safe
from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from common.depot.models import DepotOperationResult


class DepotFailureCode(StrEnum):
    """Canonical Depot failure codes shared across adapters."""

    CONNECTOR_FAILURE = "connector_failure"
    SNAPSHOT_READ_FAILURE = "snapshot_read_failure"
    VALIDATION_FAILURE = "validation_failure"
    VALIDATION_RUNTIME_ERROR = "validation_runtime_error"
    CONFLICT_DETECTED = "conflict_detected"
    APPROVAL_MISSING = "approval_missing"
    ROLLBACK_UNSAFE = "rollback_unsafe"
    DEPOT_API_FAILURE = "depot_api_failure"


class RetryDisposition(StrEnum):
    """Shared retry handling classes for Depot failures."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    WAIT_REQUIRED = "wait_required"
    LIMITED_RETRY = "limited_retry"


@dataclass(frozen=True, slots=True)
class DepotFailure:
    """Compact structured failure payload."""

    code: DepotFailureCode
    message: str
    retry_disposition: RetryDisposition
    details: dict[str, Any] = field(default_factory=dict)
    cause_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retry_disposition": self.retry_disposition.value,
            "details": make_json_safe(self.details),
            "cause_type": self.cause_type,
        }


class DepotIntegrationError(TruthoundIntegrationError):
    """Base error for Depot integration failures."""


class DepotClientError(DepotIntegrationError):
    """Raised when Depot HTTP communication fails."""


class DepotProtocolError(DepotIntegrationError):
    """Raised when Depot payloads are malformed or incomplete."""


class DepotPollingTimeoutError(DepotIntegrationError):
    """Raised when Depot polling exceeds the configured wait budget."""


_RETRY_BY_CODE: dict[DepotFailureCode, RetryDisposition] = {
    DepotFailureCode.CONNECTOR_FAILURE: RetryDisposition.RETRYABLE,
    DepotFailureCode.SNAPSHOT_READ_FAILURE: RetryDisposition.LIMITED_RETRY,
    DepotFailureCode.VALIDATION_FAILURE: RetryDisposition.NON_RETRYABLE,
    DepotFailureCode.VALIDATION_RUNTIME_ERROR: RetryDisposition.RETRYABLE,
    DepotFailureCode.CONFLICT_DETECTED: RetryDisposition.NON_RETRYABLE,
    DepotFailureCode.APPROVAL_MISSING: RetryDisposition.WAIT_REQUIRED,
    DepotFailureCode.ROLLBACK_UNSAFE: RetryDisposition.NON_RETRYABLE,
    DepotFailureCode.DEPOT_API_FAILURE: RetryDisposition.RETRYABLE,
}


def failure_code_to_retry_disposition(code: DepotFailureCode | str) -> RetryDisposition:
    normalized = normalize_failure_code(code)
    return _RETRY_BY_CODE[normalized]


def is_retryable_failure(code: DepotFailureCode | str) -> bool:
    disposition = failure_code_to_retry_disposition(code)
    return disposition in {RetryDisposition.RETRYABLE, RetryDisposition.LIMITED_RETRY}


def is_wait_state_failure(code: DepotFailureCode | str) -> bool:
    return failure_code_to_retry_disposition(code) == RetryDisposition.WAIT_REQUIRED


def build_failure(
    code: DepotFailureCode | str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    cause: Exception | None = None,
    retry_disposition: RetryDisposition | None = None,
) -> DepotFailure:
    normalized = normalize_failure_code(code)
    return DepotFailure(
        code=normalized,
        message=message,
        retry_disposition=retry_disposition or failure_code_to_retry_disposition(normalized),
        details=make_json_safe(details or {}),
        cause_type=type(cause).__name__ if cause is not None else None,
    )


def normalize_failure_code(
    value: DepotFailureCode | str | None,
    *,
    fallback: DepotFailureCode = DepotFailureCode.DEPOT_API_FAILURE,
) -> DepotFailureCode:
    if value is None:
        return fallback
    if isinstance(value, DepotFailureCode):
        return value
    try:
        return DepotFailureCode(str(value))
    except ValueError:
        return fallback


def retry_disposition_for_exception(exc: Exception) -> RetryDisposition:
    return classify_exception_failure(exc).retry_disposition


def build_failure_from_exception(
    exc: Exception,
    *,
    details: dict[str, Any] | None = None,
) -> DepotFailure:
    return classify_exception_failure(exc, details=details)


def classify_exception_failure(
    exc: Exception,
    *,
    details: dict[str, Any] | None = None,
) -> DepotFailure:
    merged_details = {"error_type": type(exc).__name__, **(details or {})}
    if isinstance(exc, (DepotClientError, DepotProtocolError, DepotPollingTimeoutError)):
        return build_failure(
            DepotFailureCode.DEPOT_API_FAILURE,
            str(exc),
            details=merged_details,
            cause=exc,
        )
    return build_failure(
        DepotFailureCode.DEPOT_API_FAILURE,
        str(exc),
        details=merged_details,
        cause=exc,
    )


def build_failure_from_result(
    result: DepotOperationResult,
    *,
    details: dict[str, Any] | None = None,
) -> DepotFailure:
    normalized_code = normalize_failure_code(result.error_code)
    merged_details = {"status": result.status.value, **(details or {})}
    if result.error_code is not None and result.error_code != normalized_code.value:
        merged_details["original_error_code"] = result.error_code
    message = result.error_message or (
        "Depot operation cancelled"
        if result.status.value == "cancelled"
        else "Depot operation failed"
    )
    return build_failure(
        normalized_code,
        message,
        details=merged_details,
    )


def classify_operation_result_failure(result: DepotOperationResult) -> DepotFailure | None:
    if result.status.value not in {"failed", "cancelled"}:
        return None
    return build_failure_from_result(result)
