"""Tests for shared Depot flow models."""

from __future__ import annotations

from common.depot.models import DepotOperationStatus, DepotOperationType
from common.depot.testing import build_artifact_refs, build_operation_result
from common.runtime import DepotFlowResult, DepotFlowStatus, DepotFlowStepResult


def test_depot_flow_result_round_trips_through_dict() -> None:
    result = DepotFlowResult(
        flow_type="scheduled_validation",
        status=DepotFlowStatus.WAITING,
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        final_result=build_operation_result(status=DepotOperationStatus.WAITING),
        steps=(
            DepotFlowStepResult(
                step_name="scheduled_validation",
                operation_type=DepotOperationType.VALIDATE_BRANCH,
                result=build_operation_result(status=DepotOperationStatus.WAITING),
            ),
        ),
        artifact_refs=build_artifact_refs(),
        metadata={"phase": "flow"},
    )

    restored = DepotFlowResult.from_dict(result.to_dict())

    assert restored.flow_type == "scheduled_validation"
    assert restored.status == DepotFlowStatus.WAITING
    assert restored.steps[0].operation_type == DepotOperationType.VALIDATE_BRANCH
    assert restored.final_result.status.value == "waiting"
