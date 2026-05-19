"""Tests that Depot shared payloads stay compatible with runtime contracts."""

from __future__ import annotations

from pathlib import Path

from common.depot.models import DepotArtifactRefs, DepotPlatformMetadata
from common.depot.testing import build_operation_request, build_operation_result
from common.runtime import (
    PlatformRuntimeContext,
    attach_depot_artifact_refs,
    build_depot_platform_metadata,
    normalize_depot_runtime_request,
)


def test_platform_metadata_keeps_host_execution_json_safe() -> None:
    metadata = DepotPlatformMetadata(
        platform="prefect",
        host_execution={"path": Path("/tmp/depot"), "attempts": {1, 2}},
    )

    payload = metadata.to_dict()

    assert payload["host_execution"]["path"] == "/tmp/depot"
    assert payload["host_execution"]["attempts"] == [1, 2]


def test_depot_result_does_not_conflict_with_runtime_host_execution_context() -> None:
    runtime_context = PlatformRuntimeContext(
        platform="airflow",
        host_execution={"dag_id": "quality", "task_id": "sync"},
    )
    result = build_operation_result(
        platform_metadata=DepotPlatformMetadata(
            platform="airflow",
            host_execution=runtime_context.to_dict()["host_execution"],
        )
    )

    payload = result.to_dict()

    assert payload["platform_metadata"]["host_execution"]["dag_id"] == "quality"
    assert payload["platform_metadata"]["host_execution"]["task_id"] == "sync"


def test_build_depot_platform_metadata_merges_runtime_host_execution() -> None:
    runtime_context = PlatformRuntimeContext(
        platform="prefect",
        host_execution={"flow_run_id": "flow-1", "task_run_id": "task-1"},
        extras={"tenant": "alpha"},
    )
    metadata = build_depot_platform_metadata(
        runtime_context,
        platform_metadata=DepotPlatformMetadata(
            platform="prefect",
            host_execution={"attempt": 2},
            extras={"trigger": "manual"},
        ),
    )

    assert metadata.host_execution["flow_run_id"] == "flow-1"
    assert metadata.host_execution["attempt"] == 2
    assert metadata.extras["tenant"] == "alpha"
    assert metadata.extras["trigger"] == "manual"


def test_attach_depot_artifact_refs_prefers_depot_returned_refs() -> None:
    result = build_operation_result(
        artifact_refs=DepotArtifactRefs(
            core_result_ref="core://results/from-depot",
            depot_operation_ref="depot://operations/from-depot",
        )
    )
    attached = attach_depot_artifact_refs(
        result,
        DepotArtifactRefs(
            core_result_ref="core://results/from-caller",
            core_gate_result_ref="core://gates/from-caller",
            depot_operation_ref="depot://operations/from-caller",
        ),
    )

    assert attached.artifact_refs.core_result_ref == "core://results/from-depot"
    assert attached.artifact_refs.core_gate_result_ref == "core://gates/from-caller"
    assert attached.artifact_refs.depot_operation_ref == "depot://operations/from-depot"


def test_normalize_depot_runtime_request_projects_runtime_context_into_platform_metadata() -> None:
    runtime_context = PlatformRuntimeContext(
        platform="airflow",
        host_execution={"dag_id": "quality", "run_id": "run-123"},
        extras={"tenant": "alpha"},
    )
    envelope = normalize_depot_runtime_request(
        build_operation_request(metadata={"phase": "submit"}),
        runtime_context=runtime_context,
        artifact_refs=DepotArtifactRefs(core_result_ref="core://results/1"),
        metadata={"source": "runtime"},
    )

    assert envelope.platform_metadata.platform == "airflow"
    assert envelope.platform_metadata.host_execution["dag_id"] == "quality"
    assert envelope.request.operation.metadata["tenant"] == "alpha"
    assert envelope.request.operation.metadata["phase"] == "submit"
    assert envelope.execution_refs.core_artifact_refs["core_result_ref"] == "core://results/1"
