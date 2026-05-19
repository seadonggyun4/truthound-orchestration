"""Shared runtime contracts for first-party orchestration integrations.

This module centralizes the runtime-facing primitives that platform packages
use to describe host context, zero-config policy, normalized input sources,
and compatibility/preflight reports.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from common.depot.models import (
    DepotArtifactRefs,
    DepotOperationRequest,
    DepotOperationResult,
    DepotPlatformMetadata,
)


class AutoConfigPolicy(StrEnum):
    """How aggressively orchestration integrations should auto-configure."""

    SAFE_AUTO = "safe_auto"
    EXPLICIT = "explicit"
    AGGRESSIVE_AUTO = "aggressive_auto"


def _make_json_safe(value: Any) -> Any:
    """Normalize nested runtime metadata into JSON-serializable values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_make_json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "to_dict"):
        try:
            return _make_json_safe(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


class ObservabilityBackend(StrEnum):
    """Supported shared observability backends."""

    NONE = "none"
    OPENLINEAGE = "openlineage"


class DataSourceKind(StrEnum):
    """Canonical source kinds understood by the shared runtime."""

    DATAFRAME = "dataframe"
    LOCAL_PATH = "local_path"
    REMOTE_URI = "remote_uri"
    SQL = "sql"
    SYNC_STREAM = "sync_stream"
    ASYNC_STREAM = "async_stream"
    CALLABLE = "callable"
    OBJECT = "object"


class CheckStatus(StrEnum):
    """Status for compatibility and preflight checks."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Typed shared observability configuration.

    Platform packages should pass this contract into common runtime helpers
    instead of constructing backend-specific emitters themselves.
    """

    backend: ObservabilityBackend = ObservabilityBackend.NONE
    endpoint: str | None = None
    namespace: str = "truthound"
    job_name: str = "truthound-orchestration"
    producer: str = "https://github.com/seadonggyun4/truthound-orchestration"
    timeout_seconds: float = 5.0
    retry_count: int = 0
    retry_backoff_seconds: float = 0.25
    structured_logging: bool = False
    logger_name: str = "truthound.orchestration.depot"
    log_level: str = "INFO"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "endpoint": self.endpoint,
            "namespace": self.namespace,
            "job_name": self.job_name,
            "producer": self.producer,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "structured_logging": self.structured_logging,
            "logger_name": self.logger_name,
            "log_level": self.log_level,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservabilityConfig:
        return cls(
            backend=normalize_observability_backend(data.get("backend")),
            endpoint=data.get("endpoint"),
            namespace=data.get("namespace", "truthound"),
            job_name=data.get("job_name", "truthound-orchestration"),
            producer=data.get(
                "producer",
                "https://github.com/seadonggyun4/truthound-orchestration",
            ),
            timeout_seconds=float(data.get("timeout_seconds", 5.0)),
            retry_count=int(data.get("retry_count", 0)),
            retry_backoff_seconds=float(data.get("retry_backoff_seconds", 0.25)),
            structured_logging=bool(data.get("structured_logging", False)),
            logger_name=str(data.get("logger_name", "truthound.orchestration.depot")),
            log_level=str(data.get("log_level", "INFO")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ResolvedDataSource:
    """Normalized description of a platform-provided data source."""

    kind: DataSourceKind
    value: Any
    reference: str
    requires_connection: bool = False
    format_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reference": self.reference,
            "requires_connection": self.requires_connection,
            "format_hint": self.format_hint,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PlatformRuntimeContext:
    """Shared host/runtime context passed from platform wrappers to common code."""

    platform: str
    auto_config_policy: AutoConfigPolicy = AutoConfigPolicy.SAFE_AUTO
    host_version: str | None = None
    connection_id: str | None = None
    profile_name: str | None = None
    source_name: str | None = None
    project_root: str | None = None
    host_metadata: dict[str, Any] = field(default_factory=dict)
    host_execution: dict[str, Any] = field(default_factory=dict)
    source_descriptors: tuple[ResolvedDataSource, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def with_source(self, source: ResolvedDataSource) -> PlatformRuntimeContext:
        """Return a new runtime context with an appended source descriptor."""

        return PlatformRuntimeContext(
            platform=self.platform,
            auto_config_policy=self.auto_config_policy,
            host_version=self.host_version,
            connection_id=self.connection_id,
            profile_name=self.profile_name,
            source_name=self.source_name,
            project_root=self.project_root,
            host_metadata=dict(self.host_metadata),
            host_execution=dict(self.host_execution),
            source_descriptors=(*self.source_descriptors, source),
            extras=dict(self.extras),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "auto_config_policy": self.auto_config_policy.value,
            "host_version": self.host_version,
            "connection_id": self.connection_id,
            "profile_name": self.profile_name,
            "source_name": self.source_name,
            "project_root": self.project_root,
            "host_metadata": _make_json_safe(self.host_metadata),
            "host_execution": _make_json_safe(self.host_execution),
            "source_descriptors": [source.to_dict() for source in self.source_descriptors],
            "extras": _make_json_safe(self.extras),
        }


@dataclass(frozen=True, slots=True)
class DepotRuntimeRequest:
    """Normalized Depot runtime request bound to a host runtime context."""

    operation: DepotOperationRequest
    runtime_context: PlatformRuntimeContext
    artifact_refs: DepotArtifactRefs | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.to_dict(),
            "runtime_context": self.runtime_context.to_dict(),
            "artifact_refs": self.artifact_refs.to_dict() if self.artifact_refs else None,
            "metadata": _make_json_safe(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DepotExecutionRefs:
    """Separated execution references for Core, Depot, and platform layers."""

    core_artifact_refs: dict[str, str | None] = field(default_factory=dict)
    depot_operation_refs: dict[str, str | None] = field(default_factory=dict)
    platform_execution_refs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_artifact_refs": _make_json_safe(self.core_artifact_refs),
            "depot_operation_refs": _make_json_safe(self.depot_operation_refs),
            "platform_execution_refs": _make_json_safe(self.platform_execution_refs),
        }


@dataclass(frozen=True, slots=True)
class DepotRuntimeEnvelope:
    """Canonical shared Depot runtime envelope consumed by orchestration helpers."""

    request: DepotRuntimeRequest
    platform_metadata: DepotPlatformMetadata
    execution_refs: DepotExecutionRefs
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "platform_metadata": self.platform_metadata.to_dict(),
            "execution_refs": self.execution_refs.to_dict(),
            "metadata": _make_json_safe(self.metadata),
        }


class DepotFlowStatus(StrEnum):
    """Canonical orchestration-level flow states for Depot wrappers."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    WAITING = "waiting"
    NO_OP = "no_op"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DepotFlowRequest:
    """Shared request contract for orchestration-level Depot flows."""

    flow_type: str
    depot_id: str
    asset_id: str
    runtime_context: PlatformRuntimeContext
    branch_id: str | None = None
    snapshot_id: str | None = None
    release_tag: str | None = None
    target_branch_id: str | None = None
    requested_by: str | None = None
    artifact_refs: DepotArtifactRefs | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_type": self.flow_type,
            "depot_id": self.depot_id,
            "asset_id": self.asset_id,
            "branch_id": self.branch_id,
            "snapshot_id": self.snapshot_id,
            "release_tag": self.release_tag,
            "target_branch_id": self.target_branch_id,
            "requested_by": self.requested_by,
            "runtime_context": self.runtime_context.to_dict(),
            "artifact_refs": self.artifact_refs.to_dict() if self.artifact_refs else None,
            "metadata": _make_json_safe(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DepotFlowStepResult:
    """One normalized shared step result within an orchestration flow."""

    step_name: str
    operation_type: str
    result: DepotOperationResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "operation_type": self.operation_type,
            "result": self.result.to_dict(),
            "metadata": _make_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepotFlowStepResult:
        return cls(
            step_name=str(data["step_name"]),
            operation_type=str(data["operation_type"]),
            result=DepotOperationResult.from_dict(cast(dict[str, Any], data["result"])),
            metadata=cast(dict[str, Any], _make_json_safe(data.get("metadata", {}))),
        )


@dataclass(frozen=True, slots=True)
class DepotFlowResult:
    """Compact shared result returned from a Depot orchestration-level flow."""

    flow_type: str
    status: DepotFlowStatus
    depot_id: str
    asset_id: str
    final_result: DepotOperationResult
    steps: tuple[DepotFlowStepResult, ...] = ()
    branch_id: str | None = None
    release_tag: str | None = None
    platform_metadata: DepotPlatformMetadata | None = None
    artifact_refs: DepotArtifactRefs = field(default_factory=DepotArtifactRefs)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_type": self.flow_type,
            "status": self.status.value,
            "depot_id": self.depot_id,
            "asset_id": self.asset_id,
            "branch_id": self.branch_id,
            "release_tag": self.release_tag,
            "steps": [step.to_dict() for step in self.steps],
            "final_result": self.final_result.to_dict(),
            "platform_metadata": (
                self.platform_metadata.to_dict() if self.platform_metadata is not None else None
            ),
            "artifact_refs": self.artifact_refs.to_dict(),
            "metadata": _make_json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepotFlowResult:
        return cls(
            flow_type=str(data["flow_type"]),
            status=DepotFlowStatus(str(data["status"])),
            depot_id=str(data["depot_id"]),
            asset_id=str(data["asset_id"]),
            branch_id=cast(str | None, data.get("branch_id")),
            release_tag=cast(str | None, data.get("release_tag")),
            steps=tuple(
                DepotFlowStepResult.from_dict(step)
                for step in cast(list[dict[str, Any]], data.get("steps", []))
            ),
            final_result=DepotOperationResult.from_dict(
                cast(dict[str, Any], data["final_result"])
            ),
            platform_metadata=DepotPlatformMetadata.from_dict(
                cast(dict[str, Any] | None, data.get("platform_metadata"))
            ),
            artifact_refs=DepotArtifactRefs.from_dict(
                cast(dict[str, Any] | None, data.get("artifact_refs"))
            ),
            metadata=cast(dict[str, Any], _make_json_safe(data.get("metadata", {}))),
        )


@dataclass(frozen=True, slots=True)
class CompatibilityCheck:
    """One compatibility or preflight check entry."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Compatibility report shared across platform integrations."""

    engine_name: str
    compatible: bool
    checks: tuple[CompatibilityCheck, ...]
    engine_version: str | None = None
    platform: str | None = None
    auto_config_policy: AutoConfigPolicy = AutoConfigPolicy.SAFE_AUTO
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> tuple[CompatibilityCheck, ...]:
        return tuple(check for check in self.checks if check.status == CheckStatus.FAILED)

    @property
    def warnings(self) -> tuple[CompatibilityCheck, ...]:
        return tuple(check for check in self.checks if check.status == CheckStatus.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "platform": self.platform,
            "compatible": self.compatible,
            "auto_config_policy": self.auto_config_policy.value,
            "checks": [check.to_dict() for check in self.checks],
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Execution preflight report for a platform/runtime combination."""

    compatibility: CompatibilityReport
    resolved_source: ResolvedDataSource | None = None
    serializer: str = "shared_wire"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compatible(self) -> bool:
        return self.compatibility.compatible

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility": self.compatibility.to_dict(),
            "resolved_source": (
                self.resolved_source.to_dict() if self.resolved_source is not None else None
            ),
            "serializer": self.serializer,
            "metadata": self.metadata,
        }


def normalize_auto_config_policy(policy: AutoConfigPolicy | str | None) -> AutoConfigPolicy:
    """Normalize a zero-config policy value."""

    if policy is None:
        return AutoConfigPolicy.SAFE_AUTO
    if isinstance(policy, AutoConfigPolicy):
        return policy
    return AutoConfigPolicy(str(policy).lower())


def normalize_observability_backend(
    backend: ObservabilityBackend | str | None,
) -> ObservabilityBackend:
    """Normalize an observability backend value."""

    if backend is None:
        return ObservabilityBackend.NONE
    if isinstance(backend, ObservabilityBackend):
        return backend
    return ObservabilityBackend(str(backend).lower())


def normalize_observability_config(
    config: ObservabilityConfig | dict[str, Any] | None,
) -> ObservabilityConfig:
    """Normalize an observability config payload into the shared contract."""

    if config is None:
        return ObservabilityConfig()
    if isinstance(config, ObservabilityConfig):
        return config
    return ObservabilityConfig.from_dict(config)


def normalize_runtime_context(
    context: PlatformRuntimeContext | None = None,
    *,
    platform: str | None = None,
    auto_config_policy: AutoConfigPolicy | str | None = None,
    host_version: str | None = None,
    connection_id: str | None = None,
    profile_name: str | None = None,
    source_name: str | None = None,
    project_root: str | Path | None = None,
    host_metadata: dict[str, Any] | None = None,
    host_execution: dict[str, Any] | None = None,
    source_descriptors: tuple[ResolvedDataSource, ...] | None = None,
    extras: dict[str, Any] | None = None,
) -> PlatformRuntimeContext:
    """Normalize or construct a platform runtime context."""

    if context is not None:
        if platform is None:
            platform = context.platform
        host_metadata = {**context.host_metadata, **(host_metadata or {})}
        host_execution = {**context.host_execution, **(host_execution or {})}
        extras = {**context.extras, **(extras or {})}
        source_descriptors = source_descriptors or context.source_descriptors
        auto_config_policy = normalize_auto_config_policy(
            auto_config_policy or context.auto_config_policy
        )
        host_version = host_version or context.host_version
        connection_id = connection_id if connection_id is not None else context.connection_id
        profile_name = profile_name if profile_name is not None else context.profile_name
        source_name = source_name if source_name is not None else context.source_name
        project_root = project_root if project_root is not None else context.project_root
    elif platform is None:
        platform = "common"

    return PlatformRuntimeContext(
        platform=platform or "common",
        auto_config_policy=normalize_auto_config_policy(auto_config_policy),
        host_version=host_version,
        connection_id=connection_id,
        profile_name=profile_name,
        source_name=source_name,
        project_root=str(project_root) if project_root is not None else None,
        host_metadata=_make_json_safe(host_metadata or {}),
        host_execution=_make_json_safe(host_execution or {}),
        source_descriptors=source_descriptors or (),
        extras=_make_json_safe(extras or {}),
    )


def merge_runtime_host_execution(
    runtime_context: PlatformRuntimeContext,
    host_execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge host execution metadata with runtime context as the baseline."""

    return cast(
        dict[str, Any],
        _make_json_safe({**runtime_context.host_execution, **(host_execution or {})}),
    )


def build_depot_platform_metadata(
    runtime_context: PlatformRuntimeContext,
    *,
    platform_metadata: DepotPlatformMetadata | None = None,
) -> DepotPlatformMetadata:
    """Build a Depot platform metadata object from runtime context and optional overrides."""

    merged_host_execution = merge_runtime_host_execution(
        runtime_context,
        platform_metadata.host_execution if platform_metadata is not None else None,
    )
    merged_links = {
        **(
            {
                str(key): str(value)
                for key, value in runtime_context.host_metadata.get("links", {}).items()
            }
            if isinstance(runtime_context.host_metadata.get("links"), dict)
            else {}
        ),
        **(platform_metadata.links if platform_metadata is not None else {}),
    }
    merged_extras = {
        **runtime_context.extras,
        **(platform_metadata.extras if platform_metadata is not None else {}),
    }
    platform_run_id = (
        platform_metadata.platform_run_id
        if platform_metadata is not None and platform_metadata.platform_run_id is not None
        else _first_string(
            merged_host_execution.get("run_id"),
            merged_host_execution.get("flow_run_id"),
            merged_host_execution.get("execution_id"),
            merged_host_execution.get("invocation_id"),
            merged_host_execution.get("task_run_id"),
        )
    )
    platform_task_id = (
        platform_metadata.platform_task_id
        if platform_metadata is not None and platform_metadata.platform_task_id is not None
        else _first_string(
            merged_host_execution.get("task_id"),
            merged_host_execution.get("task_run_id"),
        )
    )
    platform_job_name = (
        platform_metadata.platform_job_name
        if platform_metadata is not None and platform_metadata.platform_job_name is not None
        else _first_string(
            merged_host_execution.get("job_name"),
            merged_host_execution.get("dag_id"),
            merged_host_execution.get("flow_name"),
        )
    )
    return DepotPlatformMetadata(
        platform=runtime_context.platform,
        platform_run_id=platform_run_id,
        platform_task_id=platform_task_id,
        platform_job_name=platform_job_name,
        host_execution=merged_host_execution,
        links=merged_links,
        extras=_make_json_safe(merged_extras),
    )


def attach_depot_artifact_refs(
    result: DepotOperationResult,
    artifact_refs: DepotArtifactRefs | None = None,
) -> DepotOperationResult:
    """Attach caller-provided artifact refs without overriding Depot-returned refs."""

    if artifact_refs is None:
        return result

    merged_refs = DepotArtifactRefs(
        core_result_ref=result.artifact_refs.core_result_ref or artifact_refs.core_result_ref,
        core_gate_result_ref=(
            result.artifact_refs.core_gate_result_ref or artifact_refs.core_gate_result_ref
        ),
        core_evidence_ref=(
            result.artifact_refs.core_evidence_ref or artifact_refs.core_evidence_ref
        ),
        depot_operation_ref=(
            result.artifact_refs.depot_operation_ref or artifact_refs.depot_operation_ref
        ),
        merge_request_ref=result.artifact_refs.merge_request_ref or artifact_refs.merge_request_ref,
        release_ref=result.artifact_refs.release_ref or artifact_refs.release_ref,
        extras={
            **artifact_refs.extras,
            **result.artifact_refs.extras,
        },
    )
    return DepotOperationResult(
        operation_id=result.operation_id,
        operation_type=result.operation_type,
        status=result.status,
        depot_id=result.depot_id,
        asset_id=result.asset_id,
        branch_id=result.branch_id,
        snapshot_id=result.snapshot_id,
        merge_request_id=result.merge_request_id,
        quality_gate_id=result.quality_gate_id,
        release_tag=result.release_tag,
        started_at=result.started_at,
        completed_at=result.completed_at,
        error_code=result.error_code,
        error_message=result.error_message,
        artifact_refs=merged_refs,
        platform_metadata=result.platform_metadata,
        metadata=dict(result.metadata),
    )


def normalize_depot_runtime_request(
    operation: DepotOperationRequest,
    *,
    runtime_context: PlatformRuntimeContext | None = None,
    artifact_refs: DepotArtifactRefs | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotRuntimeEnvelope:
    """Normalize a Depot operation plus host runtime state into one shared envelope."""

    effective_runtime_context = normalize_runtime_context(
        runtime_context,
        platform=(
            operation.platform_metadata.platform
            if operation.platform_metadata is not None and operation.platform_metadata.platform
            else None
        ),
    )
    request_metadata = {
        **effective_runtime_context.extras,
        **operation.metadata,
    }
    if metadata:
        request_metadata.update(metadata)
    normalized_operation = DepotOperationRequest(
        operation_id=operation.operation_id,
        operation_type=operation.operation_type,
        depot_id=operation.depot_id,
        asset_id=operation.asset_id,
        branch_id=operation.branch_id,
        snapshot_id=operation.snapshot_id,
        target_branch_id=operation.target_branch_id,
        merge_request_id=operation.merge_request_id,
        release_tag=operation.release_tag,
        source_ref=operation.source_ref,
        requested_by=operation.requested_by,
        idempotency_key=operation.idempotency_key,
        platform_metadata=build_depot_platform_metadata(
            effective_runtime_context,
            platform_metadata=operation.platform_metadata,
        ),
        metadata=_make_json_safe(request_metadata),
    )
    runtime_request = DepotRuntimeRequest(
        operation=normalized_operation,
        runtime_context=effective_runtime_context,
        artifact_refs=artifact_refs,
        metadata=_make_json_safe(metadata or {}),
    )
    execution_refs = DepotExecutionRefs(
        core_artifact_refs=_extract_core_artifact_refs(artifact_refs),
        depot_operation_refs=_extract_depot_operation_refs(artifact_refs),
        platform_execution_refs=_make_json_safe(
            normalized_operation.platform_metadata.host_execution
            if normalized_operation.platform_metadata is not None
            else {}
        ),
    )
    return DepotRuntimeEnvelope(
        request=runtime_request,
        platform_metadata=normalized_operation.platform_metadata
        or build_depot_platform_metadata(effective_runtime_context),
        execution_refs=execution_refs,
        metadata=_make_json_safe({**(metadata or {}), "runtime_context": effective_runtime_context.to_dict()}),
    )


def _extract_core_artifact_refs(artifact_refs: DepotArtifactRefs | None) -> dict[str, str | None]:
    if artifact_refs is None:
        return {}
    return {
        "core_result_ref": artifact_refs.core_result_ref,
        "core_gate_result_ref": artifact_refs.core_gate_result_ref,
        "core_evidence_ref": artifact_refs.core_evidence_ref,
    }


def _extract_depot_operation_refs(artifact_refs: DepotArtifactRefs | None) -> dict[str, str | None]:
    if artifact_refs is None:
        return {}
    return {
        "depot_operation_ref": artifact_refs.depot_operation_ref,
        "merge_request_ref": artifact_refs.merge_request_ref,
        "release_ref": artifact_refs.release_ref,
    }


def _first_string(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        stringified = str(value)
        if stringified:
            return stringified
    return None


def resolve_data_source(
    data: Any | None = None,
    *,
    data_path: str | Path | None = None,
    sql: str | None = None,
    source_factory: Any | None = None,
) -> ResolvedDataSource | None:
    """Normalize a data source into a shared runtime descriptor."""

    if source_factory is not None:
        return ResolvedDataSource(
            kind=DataSourceKind.CALLABLE,
            value=source_factory,
            reference=getattr(source_factory, "__name__", type(source_factory).__name__),
        )

    if sql is not None:
        statement = sql.strip()
        if not statement:
            return None
        preview = " ".join(statement.split())[:120]
        return ResolvedDataSource(
            kind=DataSourceKind.SQL,
            value=statement,
            reference=preview,
            requires_connection=True,
        )

    candidate = data_path if data_path is not None else data
    if candidate is None:
        return None

    if isinstance(candidate, Path):
        return _resolve_path_source(candidate)

    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped:
            return None
        if _looks_like_sql(stripped):
            return ResolvedDataSource(
                kind=DataSourceKind.SQL,
                value=stripped,
                reference=" ".join(stripped.split())[:120],
                requires_connection=True,
            )
        if _looks_like_path_or_uri(stripped):
            return _resolve_path_source(stripped)
        return ResolvedDataSource(
            kind=DataSourceKind.OBJECT,
            value=candidate,
            reference=stripped[:120],
        )

    if callable(candidate):
        return ResolvedDataSource(
            kind=DataSourceKind.CALLABLE,
            value=candidate,
            reference=getattr(candidate, "__name__", type(candidate).__name__),
        )

    if hasattr(candidate, "__aiter__"):
        return ResolvedDataSource(
            kind=DataSourceKind.ASYNC_STREAM,
            value=candidate,
            reference=type(candidate).__name__,
        )

    if _looks_like_dataframe(candidate):
        return ResolvedDataSource(
            kind=DataSourceKind.DATAFRAME,
            value=candidate,
            reference=type(candidate).__name__,
            metadata=_frame_metadata(candidate),
        )

    if _looks_like_stream(candidate):
        return ResolvedDataSource(
            kind=DataSourceKind.SYNC_STREAM,
            value=candidate,
            reference=type(candidate).__name__,
        )

    return ResolvedDataSource(
        kind=DataSourceKind.OBJECT,
        value=candidate,
        reference=type(candidate).__name__,
    )


def _resolve_path_source(value: str | Path) -> ResolvedDataSource:
    raw_value = str(value)
    parsed = urlparse(raw_value)
    suffix = Path(parsed.path or raw_value).suffix.lower().lstrip(".") or None

    if parsed.scheme and parsed.scheme not in {"", "file"}:
        requires_connection = parsed.scheme not in {"http", "https"}
        return ResolvedDataSource(
            kind=DataSourceKind.REMOTE_URI,
            value=raw_value,
            reference=raw_value,
            requires_connection=requires_connection,
            format_hint=suffix,
            metadata={"scheme": parsed.scheme},
        )

    path = Path(raw_value).expanduser()
    return ResolvedDataSource(
        kind=DataSourceKind.LOCAL_PATH,
        value=str(path),
        reference=str(path),
        format_hint=suffix,
        metadata={"exists": path.exists()},
    )


def _looks_like_sql(value: str) -> bool:
    prefixes = ("select ", "with ", "insert ", "update ", "delete ")
    normalized = value.lstrip().lower()
    return normalized.startswith(prefixes)


def _looks_like_path_or_uri(value: str) -> bool:
    if "://" in value:
        return True
    if value.startswith(("/", "./", "../", "~/")):
        return True
    suffix = Path(value).suffix.lower()
    return suffix in {".csv", ".parquet", ".json", ".jsonl", ".ndjson", ".avro", ".orc"}


def _looks_like_dataframe(value: Any) -> bool:
    if hasattr(value, "schema") and hasattr(value, "columns"):
        return True
    return hasattr(value, "__dataframe__")


def _looks_like_stream(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return False
    return hasattr(value, "__iter__")


def _frame_metadata(value: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    columns = getattr(value, "columns", None)
    if columns is not None:
        with suppress(TypeError):
            metadata["column_count"] = len(columns)
    height = getattr(value, "height", None)
    if height is not None:
        metadata["row_count"] = int(height)
    elif hasattr(value, "__len__"):
        with suppress(TypeError):
            metadata["row_count"] = len(value)
    return metadata
