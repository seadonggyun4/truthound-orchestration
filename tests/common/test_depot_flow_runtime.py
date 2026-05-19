"""Tests for shared Depot orchestration flow runtime helpers."""

from __future__ import annotations

from common.depot.failures import DepotClientError
from common.depot.models import DepotOperationStatus, DepotOperationType
from common.depot.testing import (
    FakeDepotRuntimeClient,
    build_artifact_refs,
    build_operation_result,
)
from common.orchestration import (
    execute_depot_flow,
    run_release_tag_flow,
    run_rollback_flow,
    run_scheduled_sync_flow,
    run_scheduled_validation_flow,
)
from common.runtime import DepotFlowRequest, DepotFlowStatus, normalize_runtime_context


def _flow_request(flow_type: str) -> DepotFlowRequest:
    return DepotFlowRequest(
        flow_type=flow_type,
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        snapshot_id="snapshot-1",
        release_tag="v1.0.0",
        runtime_context=normalize_runtime_context(platform="airflow"),
        artifact_refs=build_artifact_refs(core_result_ref="core://results/1"),
        metadata={"source": "test"},
    )


def test_scheduled_sync_flow_treats_no_op_as_terminal_success() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.NO_OP))

    result = run_scheduled_sync_flow(_flow_request("scheduled_sync"), client=client, wait=False)

    assert result.status == DepotFlowStatus.NO_OP
    assert result.final_result.status == DepotOperationStatus.NO_OP


def test_scheduled_validation_flow_waits_until_success() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            operation_type=DepotOperationType.VALIDATE_BRANCH,
            status=DepotOperationStatus.WAITING,
        )
    )
    client.queue_get(
        build_operation_result(
            operation_type=DepotOperationType.VALIDATE_BRANCH,
            status=DepotOperationStatus.WAITING,
        ),
        build_operation_result(
            operation_type=DepotOperationType.VALIDATE_BRANCH,
            status=DepotOperationStatus.SUCCEEDED,
        ),
    )

    result = run_scheduled_validation_flow(
        _flow_request("scheduled_validation"),
        client=client,
        wait=True,
    )

    assert result.status == DepotFlowStatus.SUCCEEDED
    assert result.steps[0].operation_type == DepotOperationType.VALIDATE_BRANCH


def test_release_tag_flow_waiting_then_failed_is_preserved() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            operation_type=DepotOperationType.RELEASE_TAG,
            status=DepotOperationStatus.WAITING,
        )
    )
    client.queue_get(
        build_operation_result(
            operation_type=DepotOperationType.RELEASE_TAG,
            status=DepotOperationStatus.FAILED,
            error_code="approval_missing",
            error_message="approval required",
        )
    )

    result = run_release_tag_flow(_flow_request("release_tag"), client=client, wait=True)

    assert result.status == DepotFlowStatus.FAILED
    assert result.final_result.error_code == "approval_missing"


def test_rollback_flow_wraps_transport_error_as_failed_flow() -> None:
    class _FailingClient:
        def submit_operation(self, request):
            del request
            raise DepotClientError("down")

        def get_operation(self, operation_id: str):
            del operation_id
            raise AssertionError("get_operation should not be called")

    result = run_rollback_flow(_flow_request("rollback"), client=_FailingClient(), wait=False)

    assert result.status == DepotFlowStatus.FAILED
    assert result.final_result.error_code == "depot_api_failure"


def test_execute_depot_flow_dispatches_scheduled_validation() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            operation_type=DepotOperationType.VALIDATE_BRANCH,
            status=DepotOperationStatus.WAITING,
        )
    )

    result = execute_depot_flow(
        _flow_request("scheduled_validation"),
        client=client,
        wait=False,
    )

    assert result.status == DepotFlowStatus.WAITING
