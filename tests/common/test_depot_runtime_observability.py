"""Tests for Depot runtime observability events."""

from __future__ import annotations

from common.depot.models import DepotOperationStatus
from common.depot.observability import CompositeEmitter, StructuredLogEmitter
from common.depot.testing import (
    FakeDepotRuntimeClient,
    build_artifact_refs,
    build_operation_request,
    build_operation_result,
)
from common.orchestration import (
    OpenLineageEmitter,
    create_observability_emitter,
    submit_depot_operation,
    wait_for_depot_operation,
)
from common.runtime import ObservabilityBackend, ObservabilityConfig, normalize_runtime_context


def test_submit_depot_operation_emits_openlineage_depot_facet() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            artifact_refs=build_artifact_refs(core_result_ref="core://results/1"),
            status=DepotOperationStatus.SUCCEEDED,
        )
    )
    emitter = OpenLineageEmitter(namespace="truthound", job_name="depot-runtime")

    submit_depot_operation(
        build_operation_request(),
        runtime_context=normalize_runtime_context(
            platform="airflow",
            host_execution={"run_id": "run-123", "dag_id": "quality"},
        ),
        client=client,
        emitter=emitter,
    )

    assert len(emitter.emitted_events) == 2
    payload = emitter.emitted_events[-1]
    truthound_facet = payload["run"]["facets"]["truthound"]
    facet = truthound_facet["depot"]
    assert facet["operation_id"] == "op-123"
    assert facet["artifact_refs"]["core_result_ref"] == "core://results/1"
    assert truthound_facet["trace_fields"]["operation_id"] == "op-123"
    assert truthound_facet["failure"] is None
    assert payload["run"]["runId"] == "run-123"


def test_wait_for_depot_operation_emits_waiting_then_completed() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_get(
        build_operation_result(status=DepotOperationStatus.WAITING),
        build_operation_result(status=DepotOperationStatus.SUCCEEDED),
    )
    emitter = OpenLineageEmitter(namespace="truthound", job_name="depot-runtime")

    result = wait_for_depot_operation(
        "op-123",
        runtime_context=normalize_runtime_context(platform="prefect"),
        client=client,
        emitter=emitter,
    )

    assert result.status == DepotOperationStatus.SUCCEEDED
    event_types = [payload["run"]["facets"]["truthound"]["event_type"] for payload in emitter.emitted_events]
    assert "waiting" in event_types
    assert event_types[-1] == "completed"


def test_create_observability_emitter_supports_openlineage_plus_logger() -> None:
    emitter = create_observability_emitter(
        ObservabilityConfig(
            backend=ObservabilityBackend.OPENLINEAGE,
            structured_logging=True,
            logger_name="truthound.test.depot",
        )
    )

    assert isinstance(emitter, CompositeEmitter)
    assert any(isinstance(item, OpenLineageEmitter) for item in emitter.emitters)
    assert any(isinstance(item, StructuredLogEmitter) for item in emitter.emitters)


def test_structured_logger_payload_is_redacted_and_contains_trace_fields() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            status=DepotOperationStatus.FAILED,
            error_code="approval_missing",
            error_message="approval required",
            metadata={"request_id": "req-123"},
        )
    )
    emitter = create_observability_emitter(
        ObservabilityConfig(
            backend=ObservabilityBackend.NONE,
            structured_logging=True,
            logger_name="truthound.test.depot",
        )
    )

    submit_depot_operation(
        build_operation_request(
            metadata={
                "api_token": "secret-value",
                "snapshot_body": {"id": 1},
            }
        ),
        runtime_context=normalize_runtime_context(
            platform="airflow",
            host_execution={
                "run_id": "run-123",
                "authorization": "Bearer secret",
            },
        ),
        client=client,
        emitter=emitter,
    )

    assert isinstance(emitter, StructuredLogEmitter)
    payload = emitter.emitted_events[-1]
    assert payload["trace_fields"]["operation_id"] == "op-123"
    assert payload["failure"]["code"] == "approval_missing"
    assert payload["metadata"]["api_token"] == "[REDACTED]"
    assert payload["metadata"]["snapshot_body"] == "[REDACTED]"
    assert payload["host_execution"]["authorization"] == "[REDACTED]"
