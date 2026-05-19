"""Depot operators for Apache Airflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from common.depot.client import DepotClient, DepotClientConfig
from common.depot.idempotency import build_idempotency_key
from common.depot.models import DepotOperationRequest, DepotOperationType
from common.depot.polling import PollingConfig
from common.orchestration import DepotOperationClient, execute_depot_flow, execute_depot_operation
from common.runtime import DepotFlowRequest, DepotFlowStatus, normalize_runtime_context

from truthound_airflow.utils.serialization import (
    serialize_depot_flow_result,
    serialize_depot_result,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


@dataclass(frozen=True, slots=True)
class DepotOperatorConfig:
    """Configuration for Depot Airflow operators."""

    base_url: str | None = None
    api_token: str | None = None
    timeout_seconds: float = 10.0
    wait: bool = False
    fail_on_error: bool = True
    xcom_push_key: str = "depot_result"
    headers: dict[str, str] = field(default_factory=dict)
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 300.0


@dataclass(frozen=True, slots=True)
class DepotFlowOperatorConfig(DepotOperatorConfig):
    """Configuration for Depot orchestration flow operators."""

    xcom_push_key: str = "depot_flow"


class BaseDepotOperator(BaseOperator):  # type: ignore[misc]
    """Thin Airflow wrapper around the shared Depot runtime facade."""

    template_fields: Sequence[str] = (
        "depot_id",
        "asset_id",
        "branch_id",
        "snapshot_id",
        "merge_request_id",
        "release_tag",
        "source_ref",
    )
    ui_color = "#F28C3A"
    operation_type: ClassVar[DepotOperationType]

    def __init__(
        self,
        *,
        depot_id: str,
        asset_id: str,
        branch_id: str | None = None,
        snapshot_id: str | None = None,
        merge_request_id: str | None = None,
        release_tag: str | None = None,
        source_ref: str | None = None,
        requested_by: str | None = None,
        target_branch_id: str | None = None,
        operation_id: str | None = None,
        idempotency_key: str | None = None,
        config: DepotOperatorConfig | None = None,
        client: DepotOperationClient | None = None,
        observability: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.depot_id = depot_id
        self.asset_id = asset_id
        self.branch_id = branch_id
        self.snapshot_id = snapshot_id
        self.merge_request_id = merge_request_id
        self.release_tag = release_tag
        self.source_ref = source_ref
        self.requested_by = requested_by
        self.target_branch_id = target_branch_id
        self.operation_id = operation_id
        self.idempotency_key = idempotency_key
        self.config = config or DepotOperatorConfig()
        self._client = client
        self._observability = observability or {}
        self._metadata = metadata or {}

    def execute(self, context: Context) -> dict[str, Any]:
        runtime_context = normalize_runtime_context(
            platform="airflow",
            connection_id=None,
            host_metadata={
                "operator": type(self).__name__,
                "task_id": getattr(self, "task_id", None),
            },
            host_execution=self._host_execution(context),
        )
        request = self._build_request(runtime_context.host_execution)
        result = execute_depot_operation(
            request,
            runtime_context=runtime_context,
            client=self._resolve_client(),
            wait=self.config.wait,
            polling=PollingConfig(
                poll_interval_seconds=self.config.poll_interval_seconds,
                timeout_seconds=self.config.poll_timeout_seconds,
            ),
            observability=self._observability,
            metadata=self._metadata,
        )
        payload = serialize_depot_result(result, runtime_context=runtime_context)
        ti = context.get("ti")
        if ti is not None:
            ti.xcom_push(key=self.config.xcom_push_key, value=payload)
        if self.config.fail_on_error and result.status.value == "failed":
            raise AirflowException(
                result.error_message or f"Depot operation failed: {result.error_code}"
            )
        return payload

    def _resolve_client(self) -> DepotOperationClient:
        if self._client is not None:
            return self._client
        if not self.config.base_url or not self.config.api_token:
            raise ValueError("Depot operators require either client or base_url/api_token")
        return DepotClient(
            DepotClientConfig(
                base_url=self.config.base_url,
                api_token=self.config.api_token,
                timeout_seconds=self.config.timeout_seconds,
                headers=self.config.headers,
            )
        )

    def _host_execution(self, context: Context) -> dict[str, Any]:
        task_instance = context.get("ti")
        dag = context.get("dag")
        dag_run = context.get("dag_run")
        task = context.get("task")
        return {
            "dag_id": getattr(dag, "dag_id", getattr(dag_run, "dag_id", None)),
            "task_id": getattr(task, "task_id", getattr(self, "task_id", None)),
            "run_id": context.get("run_id"),
            "try_number": getattr(task_instance, "try_number", None),
            "log_url": getattr(task_instance, "log_url", None),
        }

    def _build_request(self, host_execution: dict[str, Any]) -> DepotOperationRequest:
        run_id = host_execution.get("run_id") or "manual"
        operation_id = (
            self.operation_id
            or f"{self.operation_type.value}:{self.depot_id}:{self.asset_id}:{run_id}"
        )
        idempotency_key = self.idempotency_key or build_idempotency_key(
            self.operation_type,
            self.depot_id,
            self.asset_id,
            branch_id=self.branch_id,
            snapshot_id=self.snapshot_id,
            merge_request_id=self.merge_request_id,
            release_tag=self.release_tag,
            source_ref=self.source_ref,
            target_branch_id=self.target_branch_id,
        )
        return DepotOperationRequest(
            operation_id=operation_id,
            operation_type=self.operation_type,
            depot_id=self.depot_id,
            asset_id=self.asset_id,
            branch_id=self.branch_id,
            snapshot_id=self.snapshot_id,
            target_branch_id=self.target_branch_id,
            merge_request_id=self.merge_request_id,
            release_tag=self.release_tag,
            source_ref=self.source_ref,
            requested_by=self.requested_by,
            idempotency_key=idempotency_key,
            metadata={"airflow": host_execution, **self._metadata},
        )


class BaseDepotFlowOperator(BaseOperator):  # type: ignore[misc]
    """Thin Airflow wrapper around the shared Depot flow facade."""

    template_fields: Sequence[str] = (
        "depot_id",
        "asset_id",
        "branch_id",
        "snapshot_id",
        "release_tag",
        "target_branch_id",
    )
    ui_color = "#D86D29"
    flow_type: ClassVar[str]

    def __init__(
        self,
        *,
        depot_id: str,
        asset_id: str,
        branch_id: str | None = None,
        snapshot_id: str | None = None,
        release_tag: str | None = None,
        requested_by: str | None = None,
        target_branch_id: str | None = None,
        config: DepotFlowOperatorConfig | None = None,
        client: DepotOperationClient | None = None,
        observability: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.depot_id = depot_id
        self.asset_id = asset_id
        self.branch_id = branch_id
        self.snapshot_id = snapshot_id
        self.release_tag = release_tag
        self.requested_by = requested_by
        self.target_branch_id = target_branch_id
        self.config = config or DepotFlowOperatorConfig()
        self._client = client
        self._observability = observability or {}
        self._metadata = metadata or {}

    def execute(self, context: Context) -> dict[str, Any]:
        runtime_context = normalize_runtime_context(
            platform="airflow",
            connection_id=None,
            host_metadata={
                "operator": type(self).__name__,
                "task_id": getattr(self, "task_id", None),
            },
            host_execution=self._host_execution(context),
        )
        flow_request = DepotFlowRequest(
            flow_type=self.flow_type,
            depot_id=self.depot_id,
            asset_id=self.asset_id,
            branch_id=self.branch_id,
            snapshot_id=self.snapshot_id,
            release_tag=self.release_tag,
            target_branch_id=self.target_branch_id,
            requested_by=self.requested_by,
            runtime_context=runtime_context,
            metadata={"airflow": runtime_context.host_execution, **self._metadata},
        )
        result = execute_depot_flow(
            flow_request,
            client=self._resolve_client(),
            wait=self.config.wait,
            polling=PollingConfig(
                poll_interval_seconds=self.config.poll_interval_seconds,
                timeout_seconds=self.config.poll_timeout_seconds,
            ),
            observability=self._observability,
        )
        payload = serialize_depot_flow_result(result, runtime_context=runtime_context)
        ti = context.get("ti")
        if ti is not None:
            ti.xcom_push(key=self.config.xcom_push_key, value=payload)
        if self.config.fail_on_error and result.status in {
            DepotFlowStatus.FAILED,
            DepotFlowStatus.CANCELLED,
        }:
            raise AirflowException(
                result.final_result.error_message
                or f"Depot flow failed: {result.final_result.error_code}"
            )
        return payload

    def _resolve_client(self) -> DepotOperationClient:
        if self._client is not None:
            return self._client
        if not self.config.base_url or not self.config.api_token:
            raise ValueError("Depot flow operators require either client or base_url/api_token")
        return DepotClient(
            DepotClientConfig(
                base_url=self.config.base_url,
                api_token=self.config.api_token,
                timeout_seconds=self.config.timeout_seconds,
                headers=self.config.headers,
            )
        )

    def _host_execution(self, context: Context) -> dict[str, Any]:
        task_instance = context.get("ti")
        dag = context.get("dag")
        dag_run = context.get("dag_run")
        task = context.get("task")
        return {
            "dag_id": getattr(dag, "dag_id", getattr(dag_run, "dag_id", None)),
            "task_id": getattr(task, "task_id", getattr(self, "task_id", None)),
            "run_id": context.get("run_id"),
            "try_number": getattr(task_instance, "try_number", None),
            "log_url": getattr(task_instance, "log_url", None),
        }


class DepotPullSnapshotOperator(BaseDepotOperator):
    operation_type = DepotOperationType.PULL_SNAPSHOT


class DepotValidateBranchOperator(BaseDepotOperator):
    operation_type = DepotOperationType.VALIDATE_BRANCH


class DepotMergeAfterApprovalOperator(BaseDepotOperator):
    operation_type = DepotOperationType.MERGE_AFTER_APPROVAL


class DepotReleaseTagOperator(BaseDepotOperator):
    operation_type = DepotOperationType.RELEASE_TAG


class DepotRollbackToSnapshotOperator(BaseDepotOperator):
    operation_type = DepotOperationType.ROLLBACK_TO_SNAPSHOT


class DepotScheduledSyncOperator(BaseDepotFlowOperator):
    flow_type = "scheduled_sync"


class DepotScheduledValidationOperator(BaseDepotFlowOperator):
    flow_type = "scheduled_validation"


class DepotReleaseTagFlowOperator(BaseDepotFlowOperator):
    flow_type = "release_tag"


class DepotRollbackFlowOperator(BaseDepotFlowOperator):
    flow_type = "rollback"


__all__ = [
    "DepotOperatorConfig",
    "DepotFlowOperatorConfig",
    "BaseDepotOperator",
    "BaseDepotFlowOperator",
    "DepotPullSnapshotOperator",
    "DepotValidateBranchOperator",
    "DepotMergeAfterApprovalOperator",
    "DepotReleaseTagOperator",
    "DepotRollbackToSnapshotOperator",
    "DepotScheduledSyncOperator",
    "DepotScheduledValidationOperator",
    "DepotReleaseTagFlowOperator",
    "DepotRollbackFlowOperator",
]
