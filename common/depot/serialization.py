"""Compact wire serialization for Depot operation results."""

from __future__ import annotations

from typing import Any

from common.depot.models import DepotOperationResult


def serialize_operation_result_wire(result: DepotOperationResult) -> dict[str, Any]:
    """Serialize a Depot result into a compact shared wire payload."""

    return result.to_dict()


def deserialize_operation_result_wire(data: dict[str, Any]) -> DepotOperationResult:
    """Deserialize a shared wire payload back into a Depot result."""

    return DepotOperationResult.from_dict(data)


def to_platform_payload(result: DepotOperationResult) -> dict[str, Any]:
    """Create a compact cross-platform payload for host-native outputs."""

    artifact_refs = result.artifact_refs.to_dict()
    platform_metadata = result.platform_metadata.to_dict() if result.platform_metadata else None
    metadata = {
        str(key): value
        for key, value in result.metadata.items()
        if str(key) not in {"raw_dataset", "raw_rows", "evidence_blob", "sample_values"}
    }
    return {
        "operation_id": result.operation_id,
        "operation_type": result.operation_type.value,
        "status": result.status.value,
        "depot_id": result.depot_id,
        "asset_id": result.asset_id,
        "branch_id": result.branch_id,
        "snapshot_id": result.snapshot_id,
        "merge_request_id": result.merge_request_id,
        "quality_gate_id": result.quality_gate_id,
        "release_tag": result.release_tag,
        "started_at": result.to_dict()["started_at"],
        "completed_at": result.to_dict()["completed_at"],
        "error_code": result.error_code,
        "error_message": result.error_message,
        "artifact_refs": artifact_refs,
        "platform_metadata": platform_metadata,
        "metadata": metadata,
    }
