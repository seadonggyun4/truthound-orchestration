"""Tests for Dagster Depot integration."""

from __future__ import annotations

from common.depot.models import DepotOperationStatus
from common.depot.testing import (
    FakeDepotRuntimeClient,
    build_operation_request,
    build_operation_result,
)


def test_depot_resource_executes_shared_runtime() -> None:
    from truthound_dagster.resources.depot import DepotResource

    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.SUCCEEDED))
    resource = DepotResource(client=client)

    result = resource.execute(
        build_operation_request(),
        host_execution={"run_id": "run-123"},
        host_metadata={"job": "depot"},
    )

    assert result.status == DepotOperationStatus.SUCCEEDED


def test_validate_branch_op_is_registered() -> None:
    from truthound_dagster.ops.depot import validate_branch_op

    assert validate_branch_op.name == "depot_validate_branch"
    assert validate_branch_op.required_resource_keys == {"depot"}


def test_scheduled_sync_op_is_registered() -> None:
    from truthound_dagster.ops.depot import scheduled_sync_op

    assert scheduled_sync_op.name == "depot_scheduled_sync"
    assert scheduled_sync_op.required_resource_keys == {"depot"}
