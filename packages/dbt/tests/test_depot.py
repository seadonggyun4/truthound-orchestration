"""Tests for dbt Depot helpers."""

from __future__ import annotations

from pathlib import Path

from common.depot.models import DepotOperationStatus, DepotOperationType
from common.depot.testing import FakeDepotRuntimeClient, build_operation_result


ROOT = Path(__file__).resolve().parents[3]


def test_validate_branch_operation_returns_shared_payload() -> None:
    from truthound_dbt.hooks.depot import validate_branch_operation

    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            operation_type=DepotOperationType.VALIDATE_BRANCH,
            status=DepotOperationStatus.SUCCEEDED,
        )
    )

    payload = validate_branch_operation(
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        client=client,
    )

    assert payload["status"] == "succeeded"
    assert payload["operation_type"] == "validate_branch"


def test_depot_macros_export_run_operation_surface() -> None:
    validate_file = ROOT / "packages" / "dbt" / "macros" / "depot" / "truthound_depot_validate_branch.sql"
    release_file = ROOT / "packages" / "dbt" / "macros" / "depot" / "truthound_depot_release_tag.sql"

    assert "macro truthound_depot_validate_branch" in validate_file.read_text()
    assert "macro truthound_depot_release_tag" in release_file.read_text()
