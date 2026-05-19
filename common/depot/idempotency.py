"""Deterministic idempotency helpers for Depot mutation requests."""

from __future__ import annotations

import re
from hashlib import sha256

from common.depot.models import DepotOperationRequest, DepotOperationType
from common.exceptions import MissingConfigError


_TOKEN_SANITIZER = re.compile(r"[^a-z0-9._:-]+")
_MAX_KEY_LENGTH = 160
_MUTATION_OPERATIONS = {
    DepotOperationType.PULL_SNAPSHOT,
    DepotOperationType.OPEN_MERGE_REQUEST,
    DepotOperationType.MERGE_AFTER_APPROVAL,
    DepotOperationType.RELEASE_TAG,
    DepotOperationType.ROLLBACK_TO_SNAPSHOT,
    DepotOperationType.SCHEDULED_SYNC,
}


def normalize_operation_id(value: str) -> str:
    normalized = _normalize_token(value)
    if not normalized:
        raise ValueError("operation_id must not be empty")
    return normalized


def normalize_idempotency_key(value: str) -> str:
    normalized = _normalize_token(value)
    if not normalized:
        raise ValueError("idempotency_key must not be empty")
    return normalized


def requires_mutation_identity(operation_type: DepotOperationType | str) -> bool:
    normalized = (
        operation_type
        if isinstance(operation_type, DepotOperationType)
        else DepotOperationType(str(operation_type))
    )
    return normalized in _MUTATION_OPERATIONS


def build_idempotency_key(
    operation_type: DepotOperationType | str,
    depot_id: str,
    asset_id: str,
    *,
    branch_id: str | None = None,
    snapshot_id: str | None = None,
    merge_request_id: str | None = None,
    release_tag: str | None = None,
    source_ref: str | None = None,
    target_branch_id: str | None = None,
) -> str:
    normalized_operation = (
        operation_type
        if isinstance(operation_type, DepotOperationType)
        else DepotOperationType(str(operation_type))
    )
    parts = [
        normalized_operation.value,
        _normalize_part(depot_id),
        _normalize_part(asset_id),
        _normalize_part(branch_id),
        _normalize_part(snapshot_id),
        _normalize_part(target_branch_id),
        _normalize_part(merge_request_id),
        _normalize_part(release_tag),
        _normalize_part(source_ref),
    ]
    base = ":".join(parts)
    if len(base) <= _MAX_KEY_LENGTH:
        return base
    digest = sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{base[:_MAX_KEY_LENGTH - 17]}:{digest}"


def assert_mutation_request_has_ids(request: DepotOperationRequest) -> None:
    if not requires_mutation_identity(request.operation_type):
        return
    if not request.operation_id:
        raise MissingConfigError("operation_id")
    if not request.idempotency_key:
        raise MissingConfigError("idempotency_key")
    normalize_operation_id(request.operation_id)
    normalize_idempotency_key(request.idempotency_key)


def _normalize_token(value: str | None) -> str:
    collapsed = _normalize_part(value)
    return collapsed if collapsed != "-" else ""


def _normalize_part(value: str | None) -> str:
    if value is None:
        return "-"
    collapsed = _TOKEN_SANITIZER.sub("-", value.strip().lower())
    collapsed = re.sub(r"-{2,}", "-", collapsed).strip("-")
    return collapsed or "-"
