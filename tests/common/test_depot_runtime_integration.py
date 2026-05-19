"""Tests for shared Depot runtime integration helpers."""

from __future__ import annotations

import pytest

from common.depot.failures import DepotClientError, DepotFailureCode
from common.depot.models import DepotArtifactRefs, DepotOperationStatus
from common.depot.testing import (
    FakeDepotRuntimeClient,
    build_artifact_refs,
    build_operation_request,
    build_operation_result,
)
from common.orchestration import (
    execute_depot_operation,
    result_from_runtime_failure,
    submit_depot_operation,
)
from common.runtime import normalize_runtime_context


def test_submit_depot_operation_normalizes_request_and_attaches_missing_refs() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            artifact_refs=DepotArtifactRefs(
                depot_operation_ref="depot://operations/op-123",
                merge_request_ref="depot://merge-requests/mr-1",
            ),
            platform_metadata=None,
            metadata={"from_depot": True},
        )
    )
    request = build_operation_request(metadata={"requested_by_team": "platform"})
    runtime_context = normalize_runtime_context(
        platform="airflow",
        host_execution={"dag_id": "quality", "task_id": "validate", "run_id": "run-123"},
        extras={"tenant": "alpha"},
    )
    caller_refs = build_artifact_refs(
        core_result_ref="core://results/from-caller",
        core_gate_result_ref="core://gates/from-caller",
        depot_operation_ref="depot://operations/from-caller",
    )

    result = submit_depot_operation(
        request,
        runtime_context=runtime_context,
        client=client,
        artifact_refs=caller_refs,
        metadata={"phase": "submit"},
    )

    submitted = client.submitted_requests[0]
    assert submitted.platform_metadata is not None
    assert submitted.platform_metadata.platform == "airflow"
    assert submitted.platform_metadata.host_execution["dag_id"] == "quality"
    assert submitted.metadata["tenant"] == "alpha"
    assert submitted.metadata["requested_by_team"] == "platform"
    assert result.artifact_refs.depot_operation_ref == "depot://operations/op-123"
    assert result.artifact_refs.core_result_ref == "core://results/from-caller"
    assert result.platform_metadata is not None
    assert result.platform_metadata.platform_run_id == "run-123"
    assert result.metadata["execution_refs"]["core_artifact_refs"]["core_result_ref"] == (
        "core://results/from-caller"
    )


def test_execute_depot_operation_waits_until_terminal_result() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.WAITING))
    client.queue_get(
        build_operation_result(status=DepotOperationStatus.WAITING),
        build_operation_result(status=DepotOperationStatus.SUCCEEDED),
    )

    result = execute_depot_operation(
        build_operation_request(),
        runtime_context=normalize_runtime_context(platform="prefect"),
        client=client,
        wait=True,
    )

    assert result.status == DepotOperationStatus.SUCCEEDED
    assert client.seen_operation_ids == ["op-123", "op-123"]


def test_result_from_runtime_failure_builds_failed_runtime_safe_result() -> None:
    request = build_operation_request()
    result = result_from_runtime_failure(
        request,
        runtime_context=normalize_runtime_context(platform="dagster"),
        artifact_refs=build_artifact_refs(core_result_ref="core://results/1"),
        exc=DepotClientError("transport down"),
    )

    assert result.status == DepotOperationStatus.FAILED
    assert result.error_code == DepotFailureCode.DEPOT_API_FAILURE.value
    assert result.artifact_refs.core_result_ref == "core://results/1"
    assert result.platform_metadata is not None
    assert result.platform_metadata.platform == "dagster"
    assert result.metadata["failure"]["code"] == DepotFailureCode.DEPOT_API_FAILURE.value


def test_submit_depot_operation_propagates_transport_error() -> None:
    request = build_operation_request()
    runtime_context = normalize_runtime_context(platform="airflow")

    class _FailingClient:
        def submit_operation(self, request: object):
            del request
            raise DepotClientError("down")

        def get_operation(self, operation_id: str):
            del operation_id
            raise AssertionError("get_operation should not be called")

    with pytest.raises(DepotClientError):
        submit_depot_operation(
            request,
            runtime_context=runtime_context,
            client=_FailingClient(),
        )
