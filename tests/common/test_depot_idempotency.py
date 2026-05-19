"""Tests for Depot idempotency helpers."""

from __future__ import annotations

import pytest

from common.depot.idempotency import (
    assert_mutation_request_has_ids,
    build_idempotency_key,
    normalize_idempotency_key,
    normalize_operation_id,
    requires_mutation_identity,
)
from common.depot.models import DepotOperationType
from common.depot.testing import build_operation_request
from common.exceptions import MissingConfigError


def test_mutation_operations_require_operation_id_and_idempotency_key() -> None:
    request = build_operation_request(
        operation_type=DepotOperationType.MERGE_AFTER_APPROVAL,
        operation_id="",
        idempotency_key=None,
    )

    with pytest.raises(MissingConfigError):
        assert_mutation_request_has_ids(request)


def test_build_idempotency_key_is_deterministic() -> None:
    key1 = build_idempotency_key(
        DepotOperationType.OPEN_MERGE_REQUEST,
        "depot-1",
        "asset-1",
        branch_id="main",
        snapshot_id="snap-1",
    )
    key2 = build_idempotency_key(
        "open_merge_request",
        "depot-1",
        "asset-1",
        branch_id="main",
        snapshot_id="snap-1",
    )

    assert key1 == key2
    assert "open_merge_request" in key1


def test_normalizers_stabilize_whitespace_and_case() -> None:
    assert normalize_operation_id("  Merge Run  ") == "merge-run"
    assert normalize_idempotency_key("  Release:Prod  ") == "release:prod"
    assert requires_mutation_identity(DepotOperationType.SCHEDULED_SYNC) is True
