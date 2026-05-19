"""Narrow Depot hook helpers for dbt-first orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.depot.client import DepotClient, DepotClientConfig
from common.depot.idempotency import build_idempotency_key
from common.depot.models import DepotOperationRequest, DepotOperationType
from common.depot.polling import PollingConfig
from common.orchestration import DepotOperationClient, execute_depot_operation
from common.runtime import normalize_runtime_context


@dataclass(frozen=True, slots=True)
class DepotHookConfig:
    base_url: str
    api_token: str
    timeout_seconds: float = 10.0
    wait: bool = False
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 300.0
    headers: dict[str, str] = field(default_factory=dict)


def execute_depot_hook_operation(
    operation_type: DepotOperationType,
    *,
    depot_id: str,
    asset_id: str,
    config: DepotHookConfig | None = None,
    client: DepotOperationClient | None = None,
    branch_id: str | None = None,
    release_tag: str | None = None,
    operation_id: str | None = None,
    idempotency_key: str | None = None,
    requested_by: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None and client is None:
        raise ValueError("dbt Depot hooks require either a DepotHookConfig or an explicit client")
    resolved_config = config
    if resolved_config is None and client is None:
        raise ValueError("dbt Depot hooks require configuration when no client is supplied")
    runtime_context = normalize_runtime_context(
        platform="dbt",
        host_metadata={"hook": "depot", "operation_type": operation_type.value},
    )
    request = DepotOperationRequest(
        operation_id=operation_id or f"{operation_type.value}:{depot_id}:{asset_id}:dbt",
        operation_type=operation_type,
        depot_id=depot_id,
        asset_id=asset_id,
        branch_id=branch_id,
        release_tag=release_tag,
        requested_by=requested_by,
        idempotency_key=idempotency_key
        or build_idempotency_key(
            operation_type,
            depot_id,
            asset_id,
            branch_id=branch_id,
            release_tag=release_tag,
        ),
        metadata=metadata or {},
    )
    if client is not None:
        resolved_client = client
    else:
        assert resolved_config is not None
        resolved_client = DepotClient(
            DepotClientConfig(
                base_url=resolved_config.base_url,
                api_token=resolved_config.api_token,
                timeout_seconds=resolved_config.timeout_seconds,
                headers=resolved_config.headers,
            )
        )
    result = execute_depot_operation(
        request,
        runtime_context=runtime_context,
        client=resolved_client,
        wait=config.wait if config is not None else False,
        polling=(
            PollingConfig(
                poll_interval_seconds=config.poll_interval_seconds,
                timeout_seconds=config.poll_timeout_seconds,
            )
            if config is not None and config.wait
            else None
        ),
        metadata=metadata,
    )
    return result.to_dict()


def validate_branch_operation(**kwargs: Any) -> dict[str, Any]:
    return execute_depot_hook_operation(DepotOperationType.VALIDATE_BRANCH, **kwargs)


def release_tag_operation(**kwargs: Any) -> dict[str, Any]:
    return execute_depot_hook_operation(DepotOperationType.RELEASE_TAG, **kwargs)


__all__ = [
    "DepotHookConfig",
    "execute_depot_hook_operation",
    "validate_branch_operation",
    "release_tag_operation",
]
