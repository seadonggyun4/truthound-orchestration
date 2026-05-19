"""Prefect tasks for Depot operations."""

from __future__ import annotations

from typing import Any

from common.depot.idempotency import build_idempotency_key
from common.depot.models import DepotOperationRequest, DepotOperationType
from common.depot.polling import PollingConfig
from common.orchestration import DepotOperationClient, execute_depot_flow, execute_depot_operation
from common.runtime import DepotFlowRequest, normalize_runtime_context
from prefect import task

from truthound_prefect.blocks.depot import DepotBlock
from truthound_prefect.utils.serialization import (
    serialize_depot_flow_result,
    serialize_depot_result,
    to_prefect_depot_artifact,
)


def _prefect_host_execution() -> dict[str, Any]:
    try:  # pragma: no cover - depends on prefect runtime
        from prefect.runtime import flow_run, task_run

        return {
            "flow_run_id": getattr(flow_run, "id", None),
            "task_run_id": getattr(task_run, "id", None),
            "deployment_id": getattr(flow_run, "deployment_id", None),
            "task_name": getattr(task_run, "name", None),
        }
    except Exception:
        return {}


def _run_depot_task(
    operation_type: DepotOperationType,
    *,
    block: DepotBlock | None,
    depot_id: str,
    asset_id: str,
    branch_id: str | None = None,
    snapshot_id: str | None = None,
    merge_request_id: str | None = None,
    release_tag: str | None = None,
    source_ref: str | None = None,
    requested_by: str | None = None,
    target_branch_id: str | None = None,
    operation_id: str | None = None,
    idempotency_key: str | None = None,
    wait: bool = False,
    store_artifact: bool = False,
    artifact_key: str = "depot_result",
    client: DepotOperationClient | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if block is None and client is None:
        raise ValueError("Depot Prefect tasks require either a DepotBlock or an explicit client")
    resolved_block = block
    host_execution = _prefect_host_execution()
    runtime_context = normalize_runtime_context(
        platform="prefect",
        host_metadata={"task_family": "depot", "operation_type": operation_type.value},
        host_execution=host_execution,
    )
    op_id = operation_id or f"{operation_type.value}:{depot_id}:{asset_id}:{host_execution.get('task_run_id') or 'manual'}"
    idem = idempotency_key or build_idempotency_key(
        operation_type,
        depot_id,
        asset_id,
        branch_id=branch_id,
        snapshot_id=snapshot_id,
        merge_request_id=merge_request_id,
        release_tag=release_tag,
        source_ref=source_ref,
        target_branch_id=target_branch_id,
    )
    request = DepotOperationRequest(
        operation_id=op_id,
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
        idempotency_key=idem,
        metadata={"prefect": host_execution, **(metadata or {})},
    )
    if client is not None:
        resolved_client = client
    else:
        assert resolved_block is not None
        resolved_client = resolved_block.create_client()
    polling = None
    if wait and resolved_block is not None:
        polling = PollingConfig(
            poll_interval_seconds=resolved_block.poll_interval_seconds,
            timeout_seconds=resolved_block.poll_timeout_seconds,
        )
    result = execute_depot_operation(
        request,
        runtime_context=runtime_context,
        client=resolved_client,
        wait=wait,
        polling=polling,
        metadata=metadata,
    )
    payload = serialize_depot_result(result, runtime_context=runtime_context)
    if store_artifact:
        to_prefect_depot_artifact(payload, artifact_key=artifact_key)
    return dict(payload)


def _run_depot_flow_task(
    flow_type: str,
    *,
    block: DepotBlock | None,
    depot_id: str,
    asset_id: str,
    branch_id: str | None = None,
    snapshot_id: str | None = None,
    release_tag: str | None = None,
    requested_by: str | None = None,
    target_branch_id: str | None = None,
    wait: bool = False,
    store_artifact: bool = False,
    artifact_key: str = "depot_flow",
    client: DepotOperationClient | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if block is None and client is None:
        raise ValueError("Depot Prefect flow tasks require either a DepotBlock or an explicit client")
    resolved_block = block
    host_execution = _prefect_host_execution()
    runtime_context = normalize_runtime_context(
        platform="prefect",
        host_metadata={"task_family": "depot_flow", "flow_type": flow_type},
        host_execution=host_execution,
    )
    flow_request = DepotFlowRequest(
        flow_type=flow_type,
        depot_id=depot_id,
        asset_id=asset_id,
        branch_id=branch_id,
        snapshot_id=snapshot_id,
        release_tag=release_tag,
        target_branch_id=target_branch_id,
        requested_by=requested_by,
        runtime_context=runtime_context,
        metadata={"prefect": host_execution, **(metadata or {})},
    )
    if client is not None:
        resolved_client = client
    else:
        assert resolved_block is not None
        resolved_client = resolved_block.create_client()
    polling = None
    if wait and resolved_block is not None:
        polling = PollingConfig(
            poll_interval_seconds=resolved_block.poll_interval_seconds,
            timeout_seconds=resolved_block.poll_timeout_seconds,
        )
    result = execute_depot_flow(
        flow_request,
        client=resolved_client,
        wait=wait,
        polling=polling,
    )
    payload = serialize_depot_flow_result(result, runtime_context=runtime_context)
    if store_artifact:
        to_prefect_depot_artifact(payload, artifact_key=artifact_key)
    return dict(payload)


@task(name="depot_pull_snapshot", tags=["depot", "pull_snapshot"])
def pull_snapshot_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_task(DepotOperationType.PULL_SNAPSHOT, **kwargs)


@task(name="depot_validate_branch", tags=["depot", "validate_branch"])
def validate_branch_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_task(DepotOperationType.VALIDATE_BRANCH, **kwargs)


@task(name="depot_merge_after_approval", tags=["depot", "merge_after_approval"])
def merge_after_approval_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_task(DepotOperationType.MERGE_AFTER_APPROVAL, **kwargs)


@task(name="depot_release_tag", tags=["depot", "release_tag"])
def release_tag_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_task(DepotOperationType.RELEASE_TAG, **kwargs)


@task(name="depot_rollback_to_snapshot", tags=["depot", "rollback_to_snapshot"])
def rollback_to_snapshot_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_task(DepotOperationType.ROLLBACK_TO_SNAPSHOT, **kwargs)


@task(name="depot_scheduled_sync", tags=["depot", "scheduled_sync"])
def scheduled_sync_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_flow_task("scheduled_sync", **kwargs)


@task(name="depot_scheduled_validation", tags=["depot", "scheduled_validation"])
def scheduled_validation_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_flow_task("scheduled_validation", **kwargs)


@task(name="depot_release_tag_flow", tags=["depot", "release_tag"])
def release_tag_flow_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_flow_task("release_tag", **kwargs)


@task(name="depot_rollback_flow", tags=["depot", "rollback"])
def rollback_flow_task(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_flow_task("rollback", **kwargs)


__all__ = [
    "pull_snapshot_task",
    "validate_branch_task",
    "merge_after_approval_task",
    "release_tag_task",
    "rollback_to_snapshot_task",
    "scheduled_sync_task",
    "scheduled_validation_task",
    "release_tag_flow_task",
    "rollback_flow_task",
]
