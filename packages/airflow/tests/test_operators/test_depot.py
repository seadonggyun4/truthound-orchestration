"""Tests for Depot Airflow operators."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from common.depot.models import DepotOperationStatus, DepotOperationType
from common.depot.testing import FakeDepotRuntimeClient, build_operation_result


def _airflow_context() -> dict[str, object]:
    ti = MagicMock()
    ti.try_number = 1
    ti.log_url = "https://airflow/log"
    return {
        "ti": ti,
        "task": SimpleNamespace(task_id="depot_task"),
        "dag_run": SimpleNamespace(dag_id="depot_dag"),
        "run_id": "run-123",
    }


def test_validate_branch_operator_pushes_compact_xcom() -> None:
    from truthound_airflow.operators.depot import DepotValidateBranchOperator

    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            operation_type=DepotOperationType.VALIDATE_BRANCH,
            status=DepotOperationStatus.SUCCEEDED,
        )
    )
    op = DepotValidateBranchOperator(
        task_id="validate_branch",
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        client=client,
    )

    result = op.execute(_airflow_context())

    assert result["payload_type"] == "depot_operation"
    assert result["status"] == "succeeded"
    assert result["operation_type"] == "validate_branch"


def test_waiting_then_failed_raises_when_fail_on_error() -> None:
    from airflow.exceptions import AirflowException

    from truthound_airflow.operators.depot import DepotOperatorConfig, DepotValidateBranchOperator

    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.WAITING))
    client.queue_get(
        build_operation_result(
            status=DepotOperationStatus.FAILED,
            error_code="approval_missing",
            error_message="approval required",
        )
    )
    op = DepotValidateBranchOperator(
        task_id="validate_branch",
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        client=client,
        config=DepotOperatorConfig(wait=True, fail_on_error=True),
    )

    with pytest.raises(AirflowException):
        op.execute(_airflow_context())


def test_scheduled_sync_operator_treats_no_op_as_terminal_success() -> None:
    from truthound_airflow.operators.depot import DepotScheduledSyncOperator

    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.NO_OP))
    op = DepotScheduledSyncOperator(
        task_id="scheduled_sync",
        depot_id="depot-1",
        asset_id="asset-1",
        client=client,
    )

    result = op.execute(_airflow_context())

    assert result["payload_type"] == "depot_flow"
    assert result["status"] == "no_op"
    assert result["final_result"]["status"] == "no_op"
