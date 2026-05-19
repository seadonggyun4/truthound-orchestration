"""Thin synchronous Depot API client."""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass, field
from http.client import HTTPResponse
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from common.depot.failures import (
    DepotClientError,
    DepotFailure,
    DepotFailureCode,
    DepotProtocolError,
    build_failure,
)
from common.depot.idempotency import assert_mutation_request_has_ids
from common.depot.models import DepotOperationRequest, DepotOperationResult
from common.depot.polling import PollingConfig, wait_for_terminal_operation
from common.retry import (
    RetryConfig,
    RetryError,
    RetryExecutor,
    RetryExhaustedError,
)


RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
DEFAULT_REQUEST_ID_HEADER_CANDIDATES = (
    "x-request-id",
    "request-id",
    "x-trace-id",
)


class DepotTransport(Protocol):
    """Minimal transport seam for Depot client tests."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        verify_tls: bool,
    ) -> DepotClientResponse:
        ...


@dataclass(frozen=True, slots=True)
class DepotClientConfig:
    """Runtime configuration for the shared Depot client."""

    base_url: str
    api_token: str
    timeout_seconds: float = 10.0
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    user_agent: str = "truthound-orchestration/depot-client"
    headers: dict[str, str] = field(default_factory=dict)
    verify_tls: bool = True
    submit_path: str = "/operations"
    operation_path_template: str = "/operations/{operation_id}"
    request_id_header_candidates: tuple[str, ...] = DEFAULT_REQUEST_ID_HEADER_CANDIDATES


@dataclass(frozen=True, slots=True)
class DepotClientResponse:
    """Normalized HTTP response returned by the transport layer."""

    status_code: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    request_id: str | None = None


class RetryableDepotHttpError(DepotClientError):
    """Retry wrapper for transient Depot HTTP responses."""

    def __init__(
        self,
        message: str,
        *,
        response: DepotClientResponse,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        merged_details = {
            "status_code": response.status_code,
            "request_id": response.request_id,
            **(details or {}),
        }
        super().__init__(message, details=merged_details, cause=cause)
        self.response = response


class DepotExceptionFilter:
    """Retry filter scoped to transient Depot client failures."""

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        del attempt
        if isinstance(exception, RetryableDepotHttpError):
            return True
        if isinstance(exception, DepotProtocolError):
            return False
        if isinstance(exception, DepotClientError):
            return True
        return isinstance(exception, URLError)


class UrlLibDepotTransport:
    """Default transport backed by urllib.request."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        verify_tls: bool,
    ) -> DepotClientResponse:
        request = Request(url, data=body, headers=headers, method=method)
        context = None if verify_tls else ssl._create_unverified_context()
        try:
            with urlopen(request, timeout=timeout, context=context) as response:
                return _response_to_client_response(response)
        except HTTPError as exc:
            payload = _decode_http_error(exc)
            headers_dict = {str(key): str(value) for key, value in exc.headers.items()}
            return DepotClientResponse(
                status_code=exc.code,
                payload=payload,
                headers=headers_dict,
                request_id=_extract_request_id(
                    headers_dict,
                    payload,
                    request_id_header_candidates=DEFAULT_REQUEST_ID_HEADER_CANDIDATES,
                ),
            )
        except URLError as exc:
            raise DepotClientError(
                "Failed to reach Depot API",
                details={"url": url, "reason": str(exc.reason)},
                cause=exc,
            ) from exc


class DepotClient:
    """Thin sync-first Depot API client."""

    def __init__(
        self,
        config: DepotClientConfig,
        *,
        transport: DepotTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrlLibDepotTransport()
        self._retry = RetryExecutor(
            config.retry_config,
            exception_filter=DepotExceptionFilter(),
        )

    def submit_operation(self, request: DepotOperationRequest) -> DepotOperationResult:
        assert_mutation_request_has_ids(request)
        response = self._request_json(
            "POST",
            self._build_url(self.config.submit_path),
            body=request.to_dict(),
            operation_id=request.operation_id,
            idempotency_key=request.idempotency_key,
        )
        return self._normalize_response_payload(response)

    def get_operation(self, operation_id: str) -> DepotOperationResult:
        response = self._request_json(
            "GET",
            self._build_url(self.config.operation_path_template.format(operation_id=operation_id)),
            body=None,
            operation_id=None,
            idempotency_key=None,
        )
        return self._normalize_response_payload(response)

    def wait_for_operation(
        self,
        operation_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 2.0,
    ) -> DepotOperationResult:
        return wait_for_terminal_operation(
            self,
            operation_id,
            PollingConfig(
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else self.config.timeout_seconds * 30
                ),
                poll_interval_seconds=poll_interval_seconds,
            ),
        )

    def _build_url(self, path: str) -> str:
        return urljoin(f"{self.config.base_url.rstrip('/')}/", path.lstrip("/"))

    def _build_headers(
        self,
        *,
        operation_id: str | None,
        idempotency_key: str | None,
    ) -> dict[str, str]:
        extra_headers = {
            key: value
            for key, value in self.config.headers.items()
            if key.lower() != "authorization"
        }
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent,
            **extra_headers,
        }
        if operation_id:
            headers["X-Truthound-Operation-Id"] = operation_id
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None,
        operation_id: str | None,
        idempotency_key: str | None,
    ) -> DepotClientResponse:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = self._build_headers(
            operation_id=operation_id,
            idempotency_key=idempotency_key,
        )

        def _send() -> DepotClientResponse:
            response = self.transport.request(
                method,
                url,
                headers=headers,
                body=payload,
                timeout=self.config.timeout_seconds,
                verify_tls=self.config.verify_tls,
            )
            if _is_retryable_status_code(response.status_code):
                raise RetryableDepotHttpError(
                    "Depot API returned a transient retryable status",
                    response=response,
                )
            return response

        try:
            return cast(DepotClientResponse, self._retry.execute(_send))
        except RetryExhaustedError as exc:
            last_exception = exc.last_exception
            if isinstance(last_exception, RetryableDepotHttpError) and last_exception.response:
                return last_exception.response
            if isinstance(last_exception, DepotClientError):
                raise last_exception from exc
            raise
        except RetryError as exc:
            last_exception = exc.last_exception
            if isinstance(last_exception, RetryableDepotHttpError) and last_exception.response:
                return last_exception.response
            if isinstance(last_exception, DepotClientError):
                raise last_exception from exc
            raise

    def _normalize_response_payload(self, response: DepotClientResponse) -> DepotOperationResult:
        if response.status_code >= 400:
            partial_result = self._decode_error_payload(response)
            return self._coerce_failed_result(response, partial_result)
        return self._decode_success_payload(response)

    def _decode_error(self, response: DepotClientResponse) -> DepotFailure:
        payload = response.payload
        message = str(payload.get("error_message") or payload.get("message") or "Depot API request failed")
        code = payload.get("error_code") or DepotFailureCode.DEPOT_API_FAILURE.value
        details = self._extract_request_metadata(response)
        return build_failure(code, message, details=details)

    def _decode_success_payload(self, response: DepotClientResponse) -> DepotOperationResult:
        payload = dict(response.payload)
        request_metadata = self._extract_request_metadata(response)
        try:
            result = DepotOperationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise DepotProtocolError(
                "Depot API returned an invalid operation payload",
                details={
                    **request_metadata,
                    "payload_keys": sorted(payload.keys()),
                },
                cause=exc,
            ) from exc
        result_metadata = {**result.metadata, **request_metadata}
        return DepotOperationResult(
            operation_id=result.operation_id,
            operation_type=result.operation_type,
            status=result.status,
            depot_id=result.depot_id,
            asset_id=result.asset_id,
            branch_id=result.branch_id,
            snapshot_id=result.snapshot_id,
            merge_request_id=result.merge_request_id,
            quality_gate_id=result.quality_gate_id,
            release_tag=result.release_tag,
            started_at=result.started_at,
            completed_at=result.completed_at,
            error_code=result.error_code,
            error_message=result.error_message,
            artifact_refs=result.artifact_refs,
            platform_metadata=result.platform_metadata,
            metadata=result_metadata,
        )

    def _decode_error_payload(self, response: DepotClientResponse) -> DepotOperationResult | None:
        payload = dict(response.payload)
        try:
            partial_result = DepotOperationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return None
        metadata = {**partial_result.metadata, **self._extract_request_metadata(response)}
        return DepotOperationResult(
            operation_id=partial_result.operation_id,
            operation_type=partial_result.operation_type,
            status=partial_result.status,
            depot_id=partial_result.depot_id,
            asset_id=partial_result.asset_id,
            branch_id=partial_result.branch_id,
            snapshot_id=partial_result.snapshot_id,
            merge_request_id=partial_result.merge_request_id,
            quality_gate_id=partial_result.quality_gate_id,
            release_tag=partial_result.release_tag,
            started_at=partial_result.started_at,
            completed_at=partial_result.completed_at,
            error_code=partial_result.error_code,
            error_message=partial_result.error_message,
            artifact_refs=partial_result.artifact_refs,
            platform_metadata=partial_result.platform_metadata,
            metadata=metadata,
        )

    def _coerce_failed_result(
        self,
        response: DepotClientResponse,
        partial_result: DepotOperationResult | None,
    ) -> DepotOperationResult:
        failure = self._decode_error(response)
        metadata = self._extract_request_metadata(response)
        if partial_result is not None:
            return DepotOperationResult(
                operation_id=partial_result.operation_id,
                operation_type=partial_result.operation_type,
                status=partial_result.status,
                depot_id=partial_result.depot_id,
                asset_id=partial_result.asset_id,
                branch_id=partial_result.branch_id,
                snapshot_id=partial_result.snapshot_id,
                merge_request_id=partial_result.merge_request_id,
                quality_gate_id=partial_result.quality_gate_id,
                release_tag=partial_result.release_tag,
                started_at=partial_result.started_at,
                completed_at=partial_result.completed_at,
                error_code=failure.code.value,
                error_message=failure.message,
                artifact_refs=partial_result.artifact_refs,
                platform_metadata=partial_result.platform_metadata,
                metadata={**partial_result.metadata, **metadata},
            )
        raise DepotProtocolError(
            "Depot API returned an invalid error payload",
            details={
                **metadata,
                "payload_keys": sorted(response.payload.keys()),
            },
        )

    def _extract_request_metadata(self, response: DepotClientResponse) -> dict[str, Any]:
        metadata: dict[str, Any] = {"status_code": response.status_code}
        if response.request_id is not None:
            metadata["request_id"] = response.request_id
        return metadata


def _response_to_client_response(response: HTTPResponse) -> DepotClientResponse:
    body = response.read()
    payload = json.loads(body.decode("utf-8")) if body else {}
    headers = {str(key): str(value) for key, value in response.headers.items()}
    return DepotClientResponse(
        status_code=getattr(response, "status", 200),
        payload=payload,
        headers=headers,
        request_id=_extract_request_id(
            headers,
            payload,
            request_id_header_candidates=DEFAULT_REQUEST_ID_HEADER_CANDIDATES,
        ),
    )


def _decode_http_error(error: HTTPError) -> dict[str, Any]:
    raw = error.read()
    if not raw:
        return {"message": str(error.reason), "error_code": DepotFailureCode.DEPOT_API_FAILURE.value}
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DepotClientError(
            "Depot API returned a non-JSON error response",
            details={"status_code": error.code},
            cause=exc,
        ) from exc
    if not isinstance(decoded, dict):
        raise DepotProtocolError(
            "Depot API error response must be a JSON object",
            details={"status_code": error.code},
        )
    return decoded


def _extract_request_id(
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    request_id_header_candidates: tuple[str, ...],
) -> str | None:
    for key in request_id_header_candidates:
        for header_name, header_value in headers.items():
            if header_name.lower() == key:
                return header_value
    request_id = payload.get("request_id")
    return str(request_id) if request_id is not None else None


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES
