"""Canonical shared models for Depot orchestration operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any


def make_json_safe(value: Any) -> Any:
    """Normalize nested values into JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _serialize_datetime(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [make_json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "to_dict"):
        try:
            return make_json_safe(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _deserialize_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class DepotOperationType(StrEnum):
    """Canonical Depot orchestration operation kinds."""

    PULL_SNAPSHOT = "pull_snapshot"
    VALIDATE_BRANCH = "validate_branch"
    OPEN_MERGE_REQUEST = "open_merge_request"
    MERGE_AFTER_APPROVAL = "merge_after_approval"
    RELEASE_TAG = "release_tag"
    ROLLBACK_TO_SNAPSHOT = "rollback_to_snapshot"
    SCHEDULED_SYNC = "scheduled_sync"


class DepotOperationStatus(StrEnum):
    """Canonical Depot orchestration operation states."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NO_OP = "no_op"


@dataclass(frozen=True, slots=True)
class DepotArtifactRefs:
    """Compact references to Core and Depot artifacts."""

    core_result_ref: str | None = None
    core_gate_result_ref: str | None = None
    core_evidence_ref: str | None = None
    depot_operation_ref: str | None = None
    merge_request_ref: str | None = None
    release_ref: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_result_ref": self.core_result_ref,
            "core_gate_result_ref": self.core_gate_result_ref,
            "core_evidence_ref": self.core_evidence_ref,
            "depot_operation_ref": self.depot_operation_ref,
            "merge_request_ref": self.merge_request_ref,
            "release_ref": self.release_ref,
            "extras": make_json_safe(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DepotArtifactRefs:
        payload = data or {}
        return cls(
            core_result_ref=_as_optional_str(payload.get("core_result_ref")),
            core_gate_result_ref=_as_optional_str(payload.get("core_gate_result_ref")),
            core_evidence_ref=_as_optional_str(payload.get("core_evidence_ref")),
            depot_operation_ref=_as_optional_str(payload.get("depot_operation_ref")),
            merge_request_ref=_as_optional_str(payload.get("merge_request_ref")),
            release_ref=_as_optional_str(payload.get("release_ref")),
            extras=_as_metadata(payload.get("extras")),
        )


@dataclass(frozen=True, slots=True)
class DepotPlatformMetadata:
    """Platform execution metadata carried by shared Depot results."""

    platform: str
    platform_run_id: str | None = None
    platform_task_id: str | None = None
    platform_job_name: str | None = None
    host_execution: dict[str, Any] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "platform_run_id": self.platform_run_id,
            "platform_task_id": self.platform_task_id,
            "platform_job_name": self.platform_job_name,
            "host_execution": make_json_safe(self.host_execution),
            "links": {str(key): str(value) for key, value in self.links.items()},
            "extras": make_json_safe(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DepotPlatformMetadata | None:
        if data is None:
            return None
        return cls(
            platform=str(data.get("platform") or ""),
            platform_run_id=_as_optional_str(data.get("platform_run_id")),
            platform_task_id=_as_optional_str(data.get("platform_task_id")),
            platform_job_name=_as_optional_str(data.get("platform_job_name")),
            host_execution=_as_metadata(data.get("host_execution")),
            links=_as_links(data.get("links")),
            extras=_as_metadata(data.get("extras")),
        )


@dataclass(frozen=True, slots=True)
class DepotOperationRequest:
    """Canonical operation request submitted to Depot."""

    operation_id: str
    operation_type: DepotOperationType
    depot_id: str
    asset_id: str
    branch_id: str | None = None
    snapshot_id: str | None = None
    target_branch_id: str | None = None
    merge_request_id: str | None = None
    release_tag: str | None = None
    source_ref: str | None = None
    requested_by: str | None = None
    idempotency_key: str | None = None
    platform_metadata: DepotPlatformMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_type",
            _coerce_operation_type(self.operation_type),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "depot_id": self.depot_id,
            "asset_id": self.asset_id,
            "branch_id": self.branch_id,
            "snapshot_id": self.snapshot_id,
            "target_branch_id": self.target_branch_id,
            "merge_request_id": self.merge_request_id,
            "release_tag": self.release_tag,
            "source_ref": self.source_ref,
            "requested_by": self.requested_by,
            "idempotency_key": self.idempotency_key,
            "platform_metadata": (
                self.platform_metadata.to_dict() if self.platform_metadata is not None else None
            ),
            "metadata": make_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepotOperationRequest:
        return cls(
            operation_id=str(data["operation_id"]),
            operation_type=DepotOperationType(str(data["operation_type"])),
            depot_id=str(data["depot_id"]),
            asset_id=str(data["asset_id"]),
            branch_id=_as_optional_str(data.get("branch_id")),
            snapshot_id=_as_optional_str(data.get("snapshot_id")),
            target_branch_id=_as_optional_str(data.get("target_branch_id")),
            merge_request_id=_as_optional_str(data.get("merge_request_id")),
            release_tag=_as_optional_str(data.get("release_tag")),
            source_ref=_as_optional_str(data.get("source_ref")),
            requested_by=_as_optional_str(data.get("requested_by")),
            idempotency_key=_as_optional_str(data.get("idempotency_key")),
            platform_metadata=DepotPlatformMetadata.from_dict(_as_dict(data.get("platform_metadata"))),
            metadata=_as_metadata(data.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class DepotOperationResult:
    """Canonical operation result returned from Depot orchestration flows."""

    operation_id: str
    operation_type: DepotOperationType
    status: DepotOperationStatus
    depot_id: str
    asset_id: str
    branch_id: str | None = None
    snapshot_id: str | None = None
    merge_request_id: str | None = None
    quality_gate_id: str | None = None
    release_tag: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifact_refs: DepotArtifactRefs = field(default_factory=DepotArtifactRefs)
    platform_metadata: DepotPlatformMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_type",
            _coerce_operation_type(self.operation_type),
        )
        object.__setattr__(self, "status", _coerce_operation_status(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "status": self.status.value,
            "depot_id": self.depot_id,
            "asset_id": self.asset_id,
            "branch_id": self.branch_id,
            "snapshot_id": self.snapshot_id,
            "merge_request_id": self.merge_request_id,
            "quality_gate_id": self.quality_gate_id,
            "release_tag": self.release_tag,
            "started_at": _serialize_datetime(self.started_at),
            "completed_at": _serialize_datetime(self.completed_at),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "artifact_refs": self.artifact_refs.to_dict(),
            "platform_metadata": (
                self.platform_metadata.to_dict() if self.platform_metadata is not None else None
            ),
            "metadata": make_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepotOperationResult:
        return cls(
            operation_id=str(data["operation_id"]),
            operation_type=DepotOperationType(str(data["operation_type"])),
            status=DepotOperationStatus(str(data["status"])),
            depot_id=str(data["depot_id"]),
            asset_id=str(data["asset_id"]),
            branch_id=_as_optional_str(data.get("branch_id")),
            snapshot_id=_as_optional_str(data.get("snapshot_id")),
            merge_request_id=_as_optional_str(data.get("merge_request_id")),
            quality_gate_id=_as_optional_str(data.get("quality_gate_id")),
            release_tag=_as_optional_str(data.get("release_tag")),
            started_at=_deserialize_datetime(_as_optional_str(data.get("started_at"))),
            completed_at=_deserialize_datetime(_as_optional_str(data.get("completed_at"))),
            error_code=_as_optional_str(data.get("error_code")),
            error_message=_as_optional_str(data.get("error_message")),
            artifact_refs=DepotArtifactRefs.from_dict(_as_dict(data.get("artifact_refs"))),
            platform_metadata=DepotPlatformMetadata.from_dict(_as_dict(data.get("platform_metadata"))),
            metadata=_as_metadata(data.get("metadata")),
        )


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping payload, got {type(value).__name__}")
    return dict(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    stringified = str(value)
    return stringified if stringified != "" else None


def _as_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected metadata mapping, got {type(value).__name__}")
    return {str(key): make_json_safe(item) for key, item in value.items()}


def _as_links(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected links mapping, got {type(value).__name__}")
    return {str(key): str(item) for key, item in value.items()}


def _coerce_operation_type(value: DepotOperationType | str) -> DepotOperationType:
    if isinstance(value, DepotOperationType):
        return value
    return DepotOperationType(str(value))


def _coerce_operation_status(value: DepotOperationStatus | str) -> DepotOperationStatus:
    if isinstance(value, DepotOperationStatus):
        return value
    return DepotOperationStatus(str(value))
