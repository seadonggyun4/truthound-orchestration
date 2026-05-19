"""Depot script entry points for Kestra."""

from __future__ import annotations

from typing import Any

from common.depot.client import DepotClient, DepotClientConfig
from common.depot.idempotency import build_idempotency_key
from common.depot.models import DepotOperationRequest, DepotOperationType
from common.orchestration import DepotOperationClient, execute_depot_operation
from truthound_kestra.scripts.base import build_runtime_context
from truthound_kestra.utils.helpers import get_execution_context
from truthound_kestra.utils.serialization import (
    serialize_depot_result,
)


def _execute_depot_script(
    operation_type: DepotOperationType,
    *,
    depot_id: str,
    asset_id: str,
    base_url: str | None = None,
    api_token: str | None = None,
    branch_id: str | None = None,
    snapshot_id: str | None = None,
    client: DepotOperationClient | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if client is None and (not base_url or not api_token):
        raise ValueError("Kestra Depot scripts require either a client or base_url/api_token")
    context = get_execution_context()
    runtime_context = build_runtime_context(
        operation_type.value,
        script_name="depot",
        metadata={
            "execution_id": context.execution_id,
            "flow_id": context.flow_id,
            "namespace": context.namespace,
            **(metadata or {}),
        },
    )
    request = DepotOperationRequest(
        operation_id=f"{operation_type.value}:{depot_id}:{asset_id}:{context.execution_id}",
        operation_type=operation_type,
        depot_id=depot_id,
        asset_id=asset_id,
        branch_id=branch_id,
        snapshot_id=snapshot_id,
        idempotency_key=build_idempotency_key(
            operation_type,
            depot_id,
            asset_id,
            branch_id=branch_id,
            snapshot_id=snapshot_id,
        ),
        metadata=metadata or {},
    )
    resolved_client = client or DepotClient(
        DepotClientConfig(base_url=base_url or "", api_token=api_token or "")
    )
    result = execute_depot_operation(
        request,
        runtime_context=runtime_context,
        client=resolved_client,
        wait=False,
        metadata=metadata,
    )
    return dict(serialize_depot_result(result, runtime_context=runtime_context))


def pull_snapshot_script(**kwargs: Any) -> dict[str, Any]:
    return _execute_depot_script(DepotOperationType.PULL_SNAPSHOT, **kwargs)


def validate_branch_script(**kwargs: Any) -> dict[str, Any]:
    return _execute_depot_script(DepotOperationType.VALIDATE_BRANCH, **kwargs)


def release_tag_script(**kwargs: Any) -> dict[str, Any]:
    return _execute_depot_script(DepotOperationType.RELEASE_TAG, **kwargs)


__all__ = ["pull_snapshot_script", "validate_branch_script", "release_tag_script"]
