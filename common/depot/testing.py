"""Testing helpers and fake transports for Depot shared-layer tests."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from common.depot.client import DepotClientResponse
from common.depot.failures import DepotFailureCode, build_failure
from common.depot.models import (
    DepotArtifactRefs,
    DepotOperationRequest,
    DepotOperationResult,
    DepotOperationStatus,
    DepotOperationType,
    DepotPlatformMetadata,
)


@dataclass(frozen=True, slots=True)
class FakeDepotResponse:
    """In-memory HTTP-like response used by fake transports."""

    status_code: int = 200
    payload: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    request_id: str | None = None


@dataclass(slots=True)
class FakeDepotTransport:
    """Queue-driven fake transport for client tests."""

    responses: deque[FakeDepotResponse] = field(default_factory=deque)
    exceptions: deque[Exception] = field(default_factory=deque)
    requests: list[dict[str, Any]] = field(default_factory=list)
    assertions: list[RequestAssertion] = field(default_factory=list)

    def queue(self, *responses: FakeDepotResponse) -> None:
        self.responses.extend(responses)

    def queue_exception(self, *exceptions: Exception) -> None:
        self.exceptions.extend(exceptions)

    def add_assertion(self, assertion: RequestAssertion) -> None:
        self.assertions.append(assertion)

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
        decoded_body = self.decode_body(body)
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "decoded_body": decoded_body,
                "timeout": timeout,
                "verify_tls": verify_tls,
            }
        )
        for assertion in self.assertions:
            assertion(self.requests[-1])
        if self.exceptions:
            raise self.exceptions.popleft()
        if not self.responses:
            raise RuntimeError("No fake Depot response queued")
        response = self.responses.popleft()
        request_id = response.request_id
        if request_id is None:
            for header_name, header_value in response.headers.items():
                if header_name.lower() in {"x-request-id", "request-id", "x-trace-id"}:
                    request_id = header_value
                    break
        if request_id is None:
            payload_request_id = response.payload.get("request_id")
            request_id = str(payload_request_id) if payload_request_id is not None else None
        return DepotClientResponse(
            status_code=response.status_code,
            payload=dict(response.payload),
            headers=dict(response.headers),
            request_id=request_id,
        )

    @staticmethod
    def decode_body(body: bytes | None) -> dict[str, Any] | None:
        if body is None:
            return None
        decoded = json.loads(body.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise TypeError("Decoded Depot request body must be a JSON object")
        return decoded


@dataclass(slots=True)
class FakeDepotClient:
    """Minimal fake client for shared polling and adapter tests."""

    results: list[DepotOperationResult]
    seen_operation_ids: list[str] = field(default_factory=list)

    def get_operation(self, operation_id: str) -> DepotOperationResult:
        self.seen_operation_ids.append(operation_id)
        if self.results:
            return self.results.pop(0)
        raise RuntimeError(f"No fake Depot result queued for {operation_id}")


@dataclass(slots=True)
class FakeDepotRuntimeClient:
    """Fake client that supports submit/read flows for shared runtime tests."""

    submit_results: deque[DepotOperationResult] = field(default_factory=deque)
    get_results: deque[DepotOperationResult] = field(default_factory=deque)
    submitted_requests: list[DepotOperationRequest] = field(default_factory=list)
    seen_operation_ids: list[str] = field(default_factory=list)

    def queue_submit(self, *results: DepotOperationResult) -> None:
        self.submit_results.extend(results)

    def queue_get(self, *results: DepotOperationResult) -> None:
        self.get_results.extend(results)

    def submit_operation(self, request: DepotOperationRequest) -> DepotOperationResult:
        self.submitted_requests.append(request)
        if self.submit_results:
            return self.submit_results.popleft()
        raise RuntimeError("No fake Depot submit result queued")

    def get_operation(self, operation_id: str) -> DepotOperationResult:
        self.seen_operation_ids.append(operation_id)
        if self.get_results:
            return self.get_results.popleft()
        raise RuntimeError(f"No fake Depot get result queued for {operation_id}")


RequestAssertion = Callable[[dict[str, Any]], None]


def build_platform_metadata(
    *,
    platform: str = "airflow",
    platform_run_id: str | None = "run-123",
    platform_task_id: str | None = "task-123",
    platform_job_name: str | None = "quality-job",
    host_execution: dict[str, Any] | None = None,
    links: dict[str, str] | None = None,
    extras: dict[str, Any] | None = None,
) -> DepotPlatformMetadata:
    return DepotPlatformMetadata(
        platform=platform,
        platform_run_id=platform_run_id,
        platform_task_id=platform_task_id,
        platform_job_name=platform_job_name,
        host_execution=host_execution or {"run_id": platform_run_id},
        links=links or {"details": f"https://{platform}.truthound.test/runs/{platform_run_id}"},
        extras=extras or {},
    )


def build_artifact_refs(
    *,
    core_result_ref: str | None = "core://results/check-1",
    core_gate_result_ref: str | None = "core://gates/gate-1",
    core_evidence_ref: str | None = "core://evidence/evidence-1",
    depot_operation_ref: str | None = "depot://operations/op-1",
    merge_request_ref: str | None = "depot://merge-requests/mr-1",
    release_ref: str | None = "depot://releases/v1.0.0",
    extras: dict[str, Any] | None = None,
) -> DepotArtifactRefs:
    return DepotArtifactRefs(
        core_result_ref=core_result_ref,
        core_gate_result_ref=core_gate_result_ref,
        core_evidence_ref=core_evidence_ref,
        depot_operation_ref=depot_operation_ref,
        merge_request_ref=merge_request_ref,
        release_ref=release_ref,
        extras=extras or {},
    )


def build_operation_request(
    *,
    operation_id: str = "op-123",
    operation_type: DepotOperationType = DepotOperationType.PULL_SNAPSHOT,
    depot_id: str = "depot-1",
    asset_id: str = "asset-1",
    branch_id: str | None = "main",
    snapshot_id: str | None = "snapshot-1",
    target_branch_id: str | None = "release",
    merge_request_id: str | None = None,
    release_tag: str | None = None,
    source_ref: str | None = "s3://bucket/path",
    requested_by: str | None = "tester",
    idempotency_key: str | None = "pull_snapshot:depot-1:asset-1:main",
    platform_metadata: DepotPlatformMetadata | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotOperationRequest:
    return DepotOperationRequest(
        operation_id=operation_id,
        operation_type=operation_type,
        depot_id=depot_id,
        asset_id=asset_id,
        branch_id=branch_id,
        snapshot_id=snapshot_id,
        target_branch_id=target_branch_id,
        merge_request_id=merge_request_id,
        release_tag=release_tag,
        source_ref=source_ref,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        platform_metadata=platform_metadata or build_platform_metadata(),
        metadata=metadata or {},
    )


def build_operation_result(
    *,
    operation_id: str = "op-123",
    operation_type: DepotOperationType = DepotOperationType.PULL_SNAPSHOT,
    status: DepotOperationStatus = DepotOperationStatus.SUCCEEDED,
    depot_id: str = "depot-1",
    asset_id: str = "asset-1",
    branch_id: str | None = "main",
    snapshot_id: str | None = "snapshot-1",
    merge_request_id: str | None = None,
    quality_gate_id: str | None = "gate-1",
    release_tag: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    artifact_refs: DepotArtifactRefs | None = None,
    platform_metadata: DepotPlatformMetadata | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotOperationResult:
    started = started_at or datetime(2026, 5, 19, 0, 0, tzinfo=UTC)
    completed = completed_at or datetime(2026, 5, 19, 0, 1, tzinfo=UTC)
    return DepotOperationResult(
        operation_id=operation_id,
        operation_type=operation_type,
        status=status,
        depot_id=depot_id,
        asset_id=asset_id,
        branch_id=branch_id,
        snapshot_id=snapshot_id,
        merge_request_id=merge_request_id,
        quality_gate_id=quality_gate_id,
        release_tag=release_tag,
        started_at=started,
        completed_at=completed,
        error_code=error_code,
        error_message=error_message,
        artifact_refs=artifact_refs or build_default_artifact_refs(),
        platform_metadata=platform_metadata or build_platform_metadata(),
        metadata=metadata or {},
    )


def build_operation_failure(
    *,
    code: DepotFailureCode = DepotFailureCode.DEPOT_API_FAILURE,
    message: str = "Depot API unavailable",
    details: dict[str, Any] | None = None,
    cause: Exception | None = None,
) -> dict[str, Any]:
    return build_failure(code, message, details=details, cause=cause).to_dict()


def build_client_payload(
    *,
    result: DepotOperationResult | None = None,
    request_id: str = "req-123",
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = (result or build_operation_result()).to_dict()
    payload["request_id"] = request_id
    if extra_payload:
        payload.update(extra_payload)
    return payload


def queue_client_payloads(
    transport: FakeDepotTransport,
    payloads: Sequence[dict[str, Any]],
    *,
    status_code: int = 200,
) -> None:
    transport.queue(
        *(FakeDepotResponse(status_code=status_code, payload=payload) for payload in payloads)
    )


def build_default_artifact_refs() -> DepotArtifactRefs:
    return build_artifact_refs()
