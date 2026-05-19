"""Shared orchestration contracts for first-party platform integrations.

This module centralizes first-party runtime behaviors that must remain
consistent across Airflow, Dagster, Prefect, Mage, Kestra, and dbt:

- capability inspection and enforcement
- Truthound-first zero-config check invocation
- bounded-memory streaming execution
- quality gate evaluation
- observability and OpenLineage-compatible event emission
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from common.base import CheckResult, CheckStatus
from common.depot.failures import (
    DepotFailureCode,
    DepotPollingTimeoutError,
    classify_exception_failure,
)
from common.depot.idempotency import build_idempotency_key
from common.depot.models import (
    DepotArtifactRefs,
    DepotOperationRequest,
    DepotOperationResult,
    DepotOperationStatus,
    DepotOperationType,
    DepotPlatformMetadata,
)
from common.depot.observability import (
    CompositeEmitter,
    StructuredLogEmitter,
    build_openlineage_depot_flow_truthound_facet,
    build_openlineage_depot_truthound_facet,
)
from common.depot.polling import PollingConfig, classify_status, is_terminal_status
from common.engines.base import (
    DataQualityEngine,
    supports_anomaly,
    supports_drift,
    supports_streaming,
)
from common.runtime import (
    DataSourceKind,
    DepotFlowRequest,
    DepotFlowResult,
    DepotFlowStatus,
    DepotFlowStepResult,
    DepotRuntimeEnvelope,
    ObservabilityBackend,
    ObservabilityConfig,
    PlatformRuntimeContext,
    ResolvedDataSource,
    attach_depot_artifact_refs,
    build_depot_platform_metadata,
    normalize_depot_runtime_request,
    normalize_observability_config,
    normalize_runtime_context,
    resolve_data_source,
)
from common.serializers import serialize_result_wire


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class OperationKind(StrEnum):
    """Supported first-party operation kinds."""

    CHECK = "check"
    PROFILE = "profile"
    LEARN = "learn"
    DRIFT = "drift"
    ANOMALY = "anomaly"
    STREAM = "stream"
    QUALITY_GATE = "quality_gate"
    OBSERVABILITY = "observability"


@dataclass(frozen=True, slots=True)
class OperationCapability:
    """One supported or unsupported operation on an engine/runtime pair."""

    operation: OperationKind
    supported: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "supported": self.supported,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class CapabilityMatrix:
    """Capability matrix exposed to platform integrations and preflight checks."""

    engine_name: str
    engine_version: str
    capabilities: tuple[OperationCapability, ...]

    def supports(self, operation: OperationKind | str) -> bool:
        normalized = normalize_operation_kind(operation)
        return any(
            capability.operation == normalized and capability.supported
            for capability in self.capabilities
        )

    def get(self, operation: OperationKind | str) -> OperationCapability:
        normalized = normalize_operation_kind(operation)
        for capability in self.capabilities:
            if capability.operation == normalized:
                return capability
        return OperationCapability(
            operation=normalized,
            supported=False,
            reason=f"{normalized.value} capability is unavailable",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
        }


def normalize_operation_kind(operation: OperationKind | str) -> OperationKind:
    if isinstance(operation, OperationKind):
        return operation
    return OperationKind(str(operation).lower())


def build_capability_matrix(engine: DataQualityEngine) -> CapabilityMatrix:
    """Build a first-party capability matrix for an engine."""

    capability_source = None
    if hasattr(engine, "get_capabilities"):
        capability_source = engine.get_capabilities()
    elif hasattr(engine, "capabilities"):
        capability_source = engine.capabilities

    supports_check_op = getattr(capability_source, "supports_check", True)
    supports_profile_op = getattr(capability_source, "supports_profile", True)
    supports_learn_op = getattr(capability_source, "supports_learn", True)
    supports_stream_op = getattr(capability_source, "supports_streaming", supports_streaming(engine))
    supports_drift_op = getattr(capability_source, "supports_drift", supports_drift(engine))
    supports_anomaly_op = getattr(
        capability_source,
        "supports_anomaly",
        supports_anomaly(engine),
    )

    capabilities = (
        OperationCapability(
            operation=OperationKind.CHECK,
            supported=supports_check_op,
            reason="engine exposes check operation",
        ),
        OperationCapability(
            operation=OperationKind.PROFILE,
            supported=supports_profile_op,
            reason="engine exposes profile operation",
        ),
        OperationCapability(
            operation=OperationKind.LEARN,
            supported=supports_learn_op,
            reason="engine exposes learn operation",
        ),
        OperationCapability(
            operation=OperationKind.DRIFT,
            supported=supports_drift_op,
            reason="engine exposes drift detection",
        ),
        OperationCapability(
            operation=OperationKind.ANOMALY,
            supported=supports_anomaly_op,
            reason="engine exposes anomaly detection",
        ),
        OperationCapability(
            operation=OperationKind.STREAM,
            supported=supports_stream_op or supports_check_op,
            reason=(
                "engine exposes native streaming"
                if supports_stream_op
                else "engine can be batch-streamed through repeated check calls"
            ),
        ),
        OperationCapability(
            operation=OperationKind.QUALITY_GATE,
            supported=supports_check_op,
            reason="quality gates are derived from check results",
        ),
        OperationCapability(
            operation=OperationKind.OBSERVABILITY,
            supported=True,
            reason="shared runtime owns observability and lineage emission",
        ),
    )
    return CapabilityMatrix(
        engine_name=engine.engine_name,
        engine_version=engine.engine_version,
        capabilities=capabilities,
    )


def ensure_operation_supported(
    engine: DataQualityEngine,
    operation: OperationKind | str,
) -> None:
    """Raise a clear error when the requested operation is unsupported."""

    normalized = normalize_operation_kind(operation)
    capability = build_capability_matrix(engine).get(normalized)
    if not capability.supported:
        raise ValueError(
            f"Engine '{engine.engine_name}' does not support {normalized.value}: {capability.reason}"
        )


def prepare_check_invocation(
    engine_or_name: DataQualityEngine | str,
    rules: Sequence[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[Sequence[dict[str, Any]] | None, dict[str, Any]]:
    """Apply Truthound-first zero-config behavior to check invocations."""

    engine_name = (
        engine_or_name.engine_name if hasattr(engine_or_name, "engine_name") else str(engine_or_name)
    ).lower()
    effective_kwargs = dict(kwargs)
    effective_rules = list(rules) if rules is not None else None

    if engine_name == "truthound" and effective_rules is None and "auto_schema" not in effective_kwargs:
        effective_kwargs["auto_schema"] = True
    return effective_rules, effective_kwargs


class OperationEventType(StrEnum):
    """Lifecycle event types emitted by the shared runtime."""

    STARTED = "started"
    BATCH_COMPLETED = "batch_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    RESUMED = "resumed"


@dataclass(frozen=True, slots=True)
class ObservabilityEvent:
    """A structured runtime lifecycle event."""

    event_type: OperationEventType
    operation: OperationKind
    engine_name: str
    platform: str
    timestamp: str = field(default_factory=_utc_now_iso)
    source: ResolvedDataSource | None = None
    result: dict[str, Any] | None = None
    host_execution: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "operation": self.operation.value,
            "engine_name": self.engine_name,
            "platform": self.platform,
            "timestamp": self.timestamp,
            "source": self.source.to_dict() if self.source is not None else None,
            "result": self.result,
            "host_execution": self.host_execution,
            "metadata": self.metadata,
            "error": self.error,
        }


class DepotOperationEventType(StrEnum):
    """Lifecycle event types emitted by the shared Depot runtime."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"
    RESUMED = "resumed"


class DepotFlowEventType(StrEnum):
    """Lifecycle event types emitted by shared Depot orchestration flows."""

    FLOW_STARTED = "flow_started"
    FLOW_WAITING = "flow_waiting"
    FLOW_COMPLETED = "flow_completed"
    FLOW_FAILED = "flow_failed"
    FLOW_NO_OP = "flow_no_op"


@dataclass(frozen=True, slots=True)
class DepotObservabilityEvent:
    """Structured Depot runtime event emitted through shared observability."""

    event_type: DepotOperationEventType
    operation_id: str
    operation_type: DepotOperationType
    status: DepotOperationStatus
    depot_id: str
    asset_id: str
    branch_id: str | None = None
    snapshot_id: str | None = None
    merge_request_id: str | None = None
    release_tag: str | None = None
    quality_gate_id: str | None = None
    platform: str = "common"
    timestamp: str = field(default_factory=_utc_now_iso)
    host_execution: dict[str, Any] = field(default_factory=dict)
    artifact_refs: DepotArtifactRefs = field(default_factory=DepotArtifactRefs)
    platform_metadata: DepotPlatformMetadata | None = None
    request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "status": self.status.value,
            "depot_id": self.depot_id,
            "asset_id": self.asset_id,
            "branch_id": self.branch_id,
            "snapshot_id": self.snapshot_id,
            "merge_request_id": self.merge_request_id,
            "release_tag": self.release_tag,
            "quality_gate_id": self.quality_gate_id,
            "platform": self.platform,
            "timestamp": self.timestamp,
            "host_execution": self.host_execution,
            "artifact_refs": self.artifact_refs.to_dict(),
            "platform_metadata": (
                self.platform_metadata.to_dict() if self.platform_metadata is not None else None
            ),
            "request_id": self.request_id,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class DepotFlowObservabilityEvent:
    """Structured shared event for Depot orchestration-level flows."""

    event_type: DepotFlowEventType
    flow_type: str
    status: DepotFlowStatus
    depot_id: str
    asset_id: str
    final_result: DepotOperationResult
    steps: tuple[DepotFlowStepResult, ...] = ()
    branch_id: str | None = None
    release_tag: str | None = None
    platform: str = "common"
    timestamp: str = field(default_factory=_utc_now_iso)
    host_execution: dict[str, Any] = field(default_factory=dict)
    artifact_refs: DepotArtifactRefs = field(default_factory=DepotArtifactRefs)
    platform_metadata: DepotPlatformMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "flow_type": self.flow_type,
            "status": self.status.value,
            "depot_id": self.depot_id,
            "asset_id": self.asset_id,
            "branch_id": self.branch_id,
            "release_tag": self.release_tag,
            "platform": self.platform,
            "timestamp": self.timestamp,
            "host_execution": self.host_execution,
            "artifact_refs": self.artifact_refs.to_dict(),
            "platform_metadata": (
                self.platform_metadata.to_dict() if self.platform_metadata is not None else None
            ),
            "final_result": self.final_result.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }


@runtime_checkable
class ObservabilityEmitter(Protocol):
    """Protocol for runtime observability emitters."""

    def emit(self, event: ObservabilityEvent) -> None:
        """Emit a runtime event."""

    def flush(self) -> None:
        """Flush buffered runtime events."""

    def preflight_check(self) -> tuple[bool, str]:
        """Validate that the emitter is ready for execution."""


@runtime_checkable
class DepotOperationClient(Protocol):
    """Protocol for Depot operation clients consumed by the shared runtime."""

    def submit_operation(self, request: DepotOperationRequest) -> DepotOperationResult:
        """Submit a Depot operation."""

    def get_operation(self, operation_id: str) -> DepotOperationResult:
        """Read a Depot operation by id."""


class NoOpEmitter:
    """Default emitter used when no observability sink is configured."""

    def emit(self, event: ObservabilityEvent) -> None:  # pragma: no cover - intentionally empty
        del event

    def emit_depot(self, event: DepotObservabilityEvent) -> None:  # pragma: no cover - intentionally empty
        del event

    def emit_depot_flow(  # pragma: no cover - intentionally empty
        self,
        event: DepotFlowObservabilityEvent,
    ) -> None:
        del event

    def flush(self) -> None:  # pragma: no cover - intentionally empty
        return None

    def preflight_check(self) -> tuple[bool, str]:
        return True, "no-op observability emitter is ready"


@dataclass(slots=True)
class OpenLineageEmitter:
    """OpenLineage-compatible observability emitter.

    The emitter always stores emitted events in-memory for local assertions and
    can optionally POST them to an HTTP endpoint without requiring the optional
    OpenLineage Python client.
    """

    namespace: str = "truthound"
    job_name: str = "truthound-orchestration"
    endpoint: str | None = None
    producer: str = "https://github.com/seadonggyun4/truthound-orchestration"
    timeout_seconds: float = 5.0
    retry_count: int = 0
    retry_backoff_seconds: float = 0.25
    emitted_events: list[dict[str, Any]] = field(default_factory=list)

    def preflight_check(self) -> tuple[bool, str]:
        if self.endpoint is None:
            return True, "OpenLineage emitter will operate in in-memory mode"

        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False, "OpenLineage endpoint must be an absolute http(s) URL"
        return True, "OpenLineage emitter is ready"

    def emit(self, event: ObservabilityEvent) -> None:
        payload = self._build_openlineage_payload(event)
        self.emitted_events.append(payload)
        if self.endpoint is None:
            return

        body = json.dumps(payload).encode("utf-8")
        attempts = max(self.retry_count, 0) + 1
        for attempt in range(attempts):
            request = Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    if response.status >= 400:
                        raise RuntimeError(
                            f"OpenLineage emitter received HTTP {response.status}"
                        )
                return
            except Exception:
                if attempt >= attempts - 1:
                    raise
                time.sleep(self.retry_backoff_seconds)

    def emit_depot(self, event: DepotObservabilityEvent) -> None:
        payload = self._build_openlineage_depot_payload(event)
        self.emitted_events.append(payload)
        if self.endpoint is None:
            return

        body = json.dumps(payload).encode("utf-8")
        attempts = max(self.retry_count, 0) + 1
        for attempt in range(attempts):
            request = Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    if response.status >= 400:
                        raise RuntimeError(
                            f"OpenLineage emitter received HTTP {response.status}"
                        )
                return
            except Exception:
                if attempt >= attempts - 1:
                    raise
                time.sleep(self.retry_backoff_seconds)

    def emit_depot_flow(self, event: DepotFlowObservabilityEvent) -> None:
        payload = self._build_openlineage_depot_flow_payload(event)
        self.emitted_events.append(payload)
        if self.endpoint is None:
            return

        body = json.dumps(payload).encode("utf-8")
        attempts = max(self.retry_count, 0) + 1
        for attempt in range(attempts):
            request = Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    if response.status >= 400:
                        raise RuntimeError(
                            f"OpenLineage emitter received HTTP {response.status}"
                        )
                return
            except Exception:
                if attempt >= attempts - 1:
                    raise
                time.sleep(self.retry_backoff_seconds)

    def flush(self) -> None:
        return None

    def _build_openlineage_payload(self, event: ObservabilityEvent) -> dict[str, Any]:
        inputs: list[dict[str, Any]] = []
        if event.source is not None:
            namespace = event.source.metadata.get("scheme") or self.namespace
            name = event.source.reference
            inputs.append({"namespace": namespace, "name": name, "facets": {}})

        job_namespace = f"{self.namespace}/{event.platform}"
        host_execution = {
            key: value
            for key, value in event.host_execution.items()
            if value is not None
        }
        run_id = (
            host_execution.get("run_id")
            or host_execution.get("task_run_id")
            or host_execution.get("flow_run_id")
            or host_execution.get("execution_id")
            or host_execution.get("invocation_id")
            or event.metadata.get("run_id")
            or event.metadata.get("execution_id")
            or f"{event.platform}:{event.operation.value}"
        )

        event_type = "COMPLETE"
        if event.event_type in {OperationEventType.STARTED, OperationEventType.RESUMED}:
            event_type = "START"
        elif event.event_type == OperationEventType.FAILED:
            event_type = "FAIL"

        return {
            "eventType": event_type,
            "eventTime": event.timestamp,
            "producer": self.producer,
            "job": {
                "namespace": job_namespace,
                "name": f"{self.job_name}.{event.operation.value}",
                "facets": {},
            },
            "run": {
                "runId": str(run_id),
                "facets": {
                    "truthound": {
                        "_producer": self.producer,
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/RunFacet.json",
                        "engine": event.engine_name,
                        "platform": event.platform,
                        "operation": event.operation.value,
                        "event_type": event.event_type.value,
                        "host_execution": host_execution,
                        "metadata": event.metadata,
                        "error": event.error,
                        "result": event.result,
                    }
                },
            },
            "inputs": inputs,
            "outputs": [],
        }

    def _build_openlineage_depot_payload(self, event: DepotObservabilityEvent) -> dict[str, Any]:
        host_execution = {
            key: value
            for key, value in event.host_execution.items()
            if value is not None
        }
        run_id = (
            host_execution.get("run_id")
            or host_execution.get("task_run_id")
            or host_execution.get("flow_run_id")
            or host_execution.get("execution_id")
            or host_execution.get("invocation_id")
            or event.metadata.get("run_id")
            or event.operation_id
        )
        if event.event_type in {DepotOperationEventType.STARTED, DepotOperationEventType.RESUMED}:
            event_type = "START"
        elif event.event_type == DepotOperationEventType.FAILED:
            event_type = "FAIL"
        else:
            event_type = "COMPLETE"
        truthound_facet = build_openlineage_depot_truthound_facet(event, producer=self.producer)
        return {
            "eventType": event_type,
            "eventTime": event.timestamp,
            "producer": self.producer,
            "job": {
                "namespace": f"{self.namespace}/{event.platform}",
                "name": f"{self.job_name}.{event.operation_type.value}",
                "facets": {},
            },
            "run": {
                "runId": str(run_id),
                "facets": {"truthound": truthound_facet},
            },
            "inputs": [],
            "outputs": [],
        }

    def _build_openlineage_depot_flow_payload(
        self,
        event: DepotFlowObservabilityEvent,
    ) -> dict[str, Any]:
        host_execution = {
            key: value
            for key, value in event.host_execution.items()
            if value is not None
        }
        run_id = (
            host_execution.get("run_id")
            or host_execution.get("flow_run_id")
            or host_execution.get("execution_id")
            or host_execution.get("invocation_id")
            or event.final_result.operation_id
        )
        if event.event_type == DepotFlowEventType.FLOW_STARTED:
            event_type = "START"
        elif event.event_type == DepotFlowEventType.FLOW_FAILED:
            event_type = "FAIL"
        else:
            event_type = "COMPLETE"
        truthound_facet = build_openlineage_depot_flow_truthound_facet(
            event,
            producer=self.producer,
        )
        return {
            "eventType": event_type,
            "eventTime": event.timestamp,
            "producer": self.producer,
            "job": {
                "namespace": f"{self.namespace}/{event.platform}",
                "name": f"{self.job_name}.{event.flow_type}",
                "facets": {},
            },
            "run": {"runId": str(run_id), "facets": {"truthound": truthound_facet}},
            "inputs": [],
            "outputs": [],
        }


def create_observability_emitter(
    config: ObservabilityConfig | dict[str, Any] | None = None,
) -> ObservabilityEmitter:
    """Create a shared observability emitter from typed config."""

    normalized = normalize_observability_config(config)
    emitters: list[Any] = []
    if normalized.backend == ObservabilityBackend.OPENLINEAGE:
        emitters.append(
            OpenLineageEmitter(
                namespace=normalized.namespace,
                job_name=normalized.job_name,
                endpoint=normalized.endpoint,
                producer=normalized.producer,
                timeout_seconds=normalized.timeout_seconds,
                retry_count=normalized.retry_count,
                retry_backoff_seconds=normalized.retry_backoff_seconds,
            )
        )
    elif normalized.backend != ObservabilityBackend.NONE:
        raise ValueError(f"Unsupported observability backend: {normalized.backend.value}")

    if normalized.structured_logging:
        emitters.append(
            StructuredLogEmitter(
                logger_name=normalized.logger_name,
                log_level=normalized.log_level,
            )
        )

    if not emitters:
        return NoOpEmitter()
    if len(emitters) == 1:
        return cast(ObservabilityEmitter, emitters[0])
    return cast(ObservabilityEmitter, CompositeEmitter(tuple(emitters)))


def _resolve_runtime_emitter(
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
    *,
    emitter: ObservabilityEmitter | None = None,
) -> ObservabilityEmitter | None:
    """Resolve either a typed observability config or a concrete emitter."""

    if observability is not None:
        if isinstance(observability, ObservabilityEmitter):
            return observability
        return create_observability_emitter(observability)
    return emitter


def emit_runtime_event(
    emitter: ObservabilityEmitter | None,
    *,
    event_type: OperationEventType,
    operation: OperationKind,
    engine_name: str,
    runtime_context: PlatformRuntimeContext | None = None,
    source: ResolvedDataSource | None = None,
    result: CheckResult | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    error: Exception | str | None = None,
) -> None:
    """Emit a shared runtime event through the configured emitter."""

    if emitter is None:
        return

    serialized_result: dict[str, Any] | None = None
    if isinstance(result, CheckResult):
        serialized_result = serialize_result_wire(result, include_result_type=True)
    elif isinstance(result, dict):
        serialized_result = dict(result)

    emitter.emit(
        ObservabilityEvent(
            event_type=event_type,
            operation=operation,
            engine_name=engine_name,
            platform=runtime_context.platform if runtime_context is not None else "common",
            source=source,
            result=serialized_result,
            host_execution=(
                dict(runtime_context.host_execution)
                if runtime_context is not None
                else {}
            ),
            metadata=metadata or {},
            error=str(error) if error is not None else None,
        )
    )


def emit_depot_runtime_event(
    emitter: ObservabilityEmitter | None,
    *,
    event_type: DepotOperationEventType,
    result: DepotOperationResult,
    runtime_context: PlatformRuntimeContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a shared Depot runtime event through the configured emitter."""

    if emitter is None or not hasattr(emitter, "emit_depot"):
        return
    event = DepotObservabilityEvent(
        event_type=event_type,
        operation_id=result.operation_id,
        operation_type=result.operation_type,
        status=result.status,
        depot_id=result.depot_id,
        asset_id=result.asset_id,
        branch_id=result.branch_id,
        snapshot_id=result.snapshot_id,
        merge_request_id=result.merge_request_id,
        release_tag=result.release_tag,
        quality_gate_id=result.quality_gate_id,
        platform=runtime_context.platform if runtime_context is not None else "common",
        host_execution=(
            dict(runtime_context.host_execution)
            if runtime_context is not None
            else (
                result.platform_metadata.host_execution
                if result.platform_metadata is not None
                else {}
            )
        ),
        artifact_refs=result.artifact_refs,
        platform_metadata=result.platform_metadata,
        request_id=cast(str | None, result.metadata.get("request_id")),
        error_code=result.error_code,
        error_message=result.error_message,
        metadata={**result.metadata, **(metadata or {})},
    )
    cast(Any, emitter).emit_depot(event)


def emit_depot_flow_event(
    emitter: ObservabilityEmitter | None,
    *,
    event_type: DepotFlowEventType,
    result: DepotFlowResult,
    runtime_context: PlatformRuntimeContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a shared Depot orchestration flow event through the configured emitter."""

    if emitter is None or not hasattr(emitter, "emit_depot_flow"):
        return
    event = DepotFlowObservabilityEvent(
        event_type=event_type,
        flow_type=result.flow_type,
        status=result.status,
        depot_id=result.depot_id,
        asset_id=result.asset_id,
        branch_id=result.branch_id,
        release_tag=result.release_tag,
        final_result=result.final_result,
        steps=result.steps,
        platform=runtime_context.platform if runtime_context is not None else "common",
        host_execution=(
            dict(runtime_context.host_execution)
            if runtime_context is not None
            else (
                result.platform_metadata.host_execution
                if result.platform_metadata is not None
                else {}
            )
        ),
        artifact_refs=result.artifact_refs,
        platform_metadata=result.platform_metadata,
        metadata={**result.metadata, **result.final_result.metadata, **(metadata or {})},
    )
    cast(Any, emitter).emit_depot_flow(event)


def normalize_depot_runtime_failure(
    exc: Exception,
) -> tuple[DepotFailureCode, str]:
    """Map shared Depot runtime exceptions into platform-safe failure codes."""

    failure = classify_exception_failure(exc)
    return failure.code, failure.message


def result_from_runtime_failure(
    request: DepotOperationRequest,
    *,
    runtime_context: PlatformRuntimeContext | None = None,
    artifact_refs: DepotArtifactRefs | None = None,
    exc: Exception,
) -> DepotOperationResult:
    """Build a compact failed Depot result from a runtime boundary error."""

    failure = classify_exception_failure(exc)
    effective_runtime_context = normalize_runtime_context(
        runtime_context,
        platform=(
            request.platform_metadata.platform
            if runtime_context is None and request.platform_metadata is not None
            else None
        ),
    )
    result = DepotOperationResult(
        operation_id=request.operation_id,
        operation_type=request.operation_type,
        status=DepotOperationStatus.FAILED,
        depot_id=request.depot_id,
        asset_id=request.asset_id,
        branch_id=request.branch_id,
        snapshot_id=request.snapshot_id,
        merge_request_id=request.merge_request_id,
        release_tag=request.release_tag,
        error_code=failure.code.value,
        error_message=failure.message,
        artifact_refs=artifact_refs or DepotArtifactRefs(),
        platform_metadata=build_depot_platform_metadata(
            effective_runtime_context,
            platform_metadata=request.platform_metadata,
        ),
        metadata={
            "error_type": type(exc).__name__,
            "failure": failure.to_dict(),
        },
    )
    return attach_depot_artifact_refs(result, artifact_refs)


def _finalize_depot_result(
    result: DepotOperationResult,
    *,
    envelope: DepotRuntimeEnvelope,
) -> DepotOperationResult:
    with_refs = attach_depot_artifact_refs(result, envelope.request.artifact_refs)
    platform_metadata = build_depot_platform_metadata(
        envelope.request.runtime_context,
        platform_metadata=with_refs.platform_metadata or envelope.platform_metadata,
    )
    metadata = {
        **envelope.request.operation.metadata,
        **with_refs.metadata,
        **envelope.metadata,
        "execution_refs": envelope.execution_refs.to_dict(),
    }
    return DepotOperationResult(
        operation_id=with_refs.operation_id,
        operation_type=with_refs.operation_type,
        status=with_refs.status,
        depot_id=with_refs.depot_id,
        asset_id=with_refs.asset_id,
        branch_id=with_refs.branch_id,
        snapshot_id=with_refs.snapshot_id,
        merge_request_id=with_refs.merge_request_id,
        quality_gate_id=with_refs.quality_gate_id,
        release_tag=with_refs.release_tag,
        started_at=with_refs.started_at,
        completed_at=with_refs.completed_at,
        error_code=with_refs.error_code,
        error_message=with_refs.error_message,
        artifact_refs=with_refs.artifact_refs,
        platform_metadata=platform_metadata,
        metadata=metadata,
    )


def submit_depot_operation(
    request: DepotOperationRequest,
    *,
    runtime_context: PlatformRuntimeContext | None,
    client: DepotOperationClient,
    artifact_refs: DepotArtifactRefs | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotOperationResult:
    """Submit one Depot operation through the shared runtime facade."""

    envelope = normalize_depot_runtime_request(
        request,
        runtime_context=runtime_context,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )
    runtime_emitter = _resolve_runtime_emitter(observability, emitter=emitter)
    started = DepotOperationResult(
        operation_id=envelope.request.operation.operation_id,
        operation_type=envelope.request.operation.operation_type,
        status=DepotOperationStatus.PENDING,
        depot_id=envelope.request.operation.depot_id,
        asset_id=envelope.request.operation.asset_id,
        branch_id=envelope.request.operation.branch_id,
        snapshot_id=envelope.request.operation.snapshot_id,
        merge_request_id=envelope.request.operation.merge_request_id,
        release_tag=envelope.request.operation.release_tag,
        artifact_refs=artifact_refs or DepotArtifactRefs(),
        platform_metadata=envelope.platform_metadata,
        metadata={"phase": "submit", **(metadata or {})},
    )
    emit_depot_runtime_event(
        runtime_emitter,
        event_type=DepotOperationEventType.STARTED,
        result=started,
        runtime_context=envelope.request.runtime_context,
        metadata=envelope.metadata,
    )
    try:
        submitted = client.submit_operation(envelope.request.operation)
        finalized = _finalize_depot_result(submitted, envelope=envelope)
        final_event = (
            DepotOperationEventType.WAITING
            if classify_status(finalized.status) == "wait"
            else (
                DepotOperationEventType.FAILED
                if finalized.status == DepotOperationStatus.FAILED
                else DepotOperationEventType.COMPLETED
            )
        )
        emit_depot_runtime_event(
            runtime_emitter,
            event_type=final_event,
            result=finalized,
            runtime_context=envelope.request.runtime_context,
            metadata=envelope.metadata,
        )
        return finalized
    except Exception as exc:
        failed = result_from_runtime_failure(
            envelope.request.operation,
            runtime_context=envelope.request.runtime_context,
            artifact_refs=artifact_refs,
            exc=exc,
        )
        emit_depot_runtime_event(
            runtime_emitter,
            event_type=DepotOperationEventType.FAILED,
            result=failed,
            runtime_context=envelope.request.runtime_context,
            metadata=envelope.metadata,
        )
        raise


def get_depot_operation(
    operation_id: str,
    *,
    runtime_context: PlatformRuntimeContext | None,
    client: DepotOperationClient,
    artifact_refs: DepotArtifactRefs | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotOperationResult:
    """Read and normalize a Depot operation without mutating Depot state."""

    result = client.get_operation(operation_id)
    runtime_context = normalize_runtime_context(
        runtime_context,
        platform=result.platform_metadata.platform if result.platform_metadata is not None else None,
    )
    envelope = normalize_depot_runtime_request(
        DepotOperationRequest(
            operation_id=result.operation_id,
            operation_type=result.operation_type,
            depot_id=result.depot_id,
            asset_id=result.asset_id,
            branch_id=result.branch_id,
            snapshot_id=result.snapshot_id,
            merge_request_id=result.merge_request_id,
            release_tag=result.release_tag,
            platform_metadata=result.platform_metadata,
            metadata=metadata or {},
        ),
        runtime_context=runtime_context,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )
    return _finalize_depot_result(result, envelope=envelope)


def wait_for_depot_operation(
    operation_id: str,
    *,
    runtime_context: PlatformRuntimeContext | None,
    client: DepotOperationClient,
    artifact_refs: DepotArtifactRefs | None = None,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
    initial_result: DepotOperationResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotOperationResult:
    """Poll Depot until a terminal operation result is available."""

    runtime_emitter = _resolve_runtime_emitter(observability, emitter=emitter)
    effective_runtime_context = normalize_runtime_context(
        runtime_context,
        platform=(
            initial_result.platform_metadata.platform
            if initial_result is not None and initial_result.platform_metadata is not None
            else None
        ),
    )
    config = polling or PollingConfig()
    started = time.monotonic()
    polls = 0
    current = initial_result or get_depot_operation(
        operation_id,
        runtime_context=effective_runtime_context,
        client=client,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )
    while not is_terminal_status(current.status):
        polls += 1
        if current.status == DepotOperationStatus.WAITING:
            emit_depot_runtime_event(
                runtime_emitter,
                event_type=DepotOperationEventType.WAITING,
                result=current,
                runtime_context=effective_runtime_context,
                metadata=metadata,
            )
        if polls >= config.max_polls or (time.monotonic() - started) >= config.timeout_seconds:
            raise DepotPollingTimeoutError(
                "Depot operation polling timed out",
                details={
                    "operation_id": operation_id,
                    "polls": polls,
                    "timeout_seconds": config.timeout_seconds,
                    "last_status": current.status.value,
                },
            )
        time.sleep(config.poll_interval_seconds)
        current = client.get_operation(operation_id)
        current = get_depot_operation(
            operation_id,
            runtime_context=effective_runtime_context,
            client=_SingleResultDepotClient(current),
            artifact_refs=artifact_refs,
            metadata=metadata,
        )
    final_event = (
        DepotOperationEventType.FAILED
        if current.status == DepotOperationStatus.FAILED
        else DepotOperationEventType.COMPLETED
    )
    emit_depot_runtime_event(
        runtime_emitter,
        event_type=final_event,
        result=current,
        runtime_context=effective_runtime_context,
        metadata=metadata,
    )
    return current


def execute_depot_operation(
    request: DepotOperationRequest,
    *,
    runtime_context: PlatformRuntimeContext | None,
    client: DepotOperationClient,
    artifact_refs: DepotArtifactRefs | None = None,
    wait: bool = False,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotOperationResult:
    """Execute a shared Depot operation through the parallel runtime facade."""

    submitted = submit_depot_operation(
        request,
        runtime_context=runtime_context,
        client=client,
        artifact_refs=artifact_refs,
        emitter=emitter,
        observability=observability,
        metadata=metadata,
    )
    if not wait:
        return submitted
    if is_terminal_status(submitted.status):
        return submitted
    return wait_for_depot_operation(
        submitted.operation_id,
        runtime_context=runtime_context,
        client=client,
        artifact_refs=artifact_refs,
        polling=polling,
        emitter=emitter,
        observability=observability,
        initial_result=submitted,
        metadata=metadata,
    )


def wait_for_depot_approval(
    operation_id: str,
    *,
    runtime_context: PlatformRuntimeContext | None,
    client: DepotOperationClient,
    artifact_refs: DepotArtifactRefs | None = None,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
    initial_result: DepotOperationResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> DepotOperationResult:
    """Wait for a Depot approval-gated operation without reinterpreting its semantics."""

    return wait_for_depot_operation(
        operation_id,
        runtime_context=runtime_context,
        client=client,
        artifact_refs=artifact_refs,
        polling=polling,
        emitter=emitter,
        observability=observability,
        initial_result=initial_result,
        metadata=metadata,
    )


def run_scheduled_sync_flow(
    flow_request: DepotFlowRequest,
    *,
    client: DepotOperationClient,
    wait: bool = False,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
) -> DepotFlowResult:
    """Run the canonical scheduled sync Depot flow."""

    return _run_single_operation_flow(
        flow_request,
        client=client,
        operation_type=DepotOperationType.SCHEDULED_SYNC,
        step_name="scheduled_sync",
        wait=wait,
        polling=polling,
        emitter=emitter,
        observability=observability,
    )


def run_scheduled_validation_flow(
    flow_request: DepotFlowRequest,
    *,
    client: DepotOperationClient,
    wait: bool = False,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
) -> DepotFlowResult:
    """Run the canonical scheduled validation Depot flow."""

    return _run_single_operation_flow(
        flow_request,
        client=client,
        operation_type=DepotOperationType.VALIDATE_BRANCH,
        step_name="scheduled_validation",
        wait=wait,
        polling=polling,
        emitter=emitter,
        observability=observability,
    )


def run_release_tag_flow(
    flow_request: DepotFlowRequest,
    *,
    client: DepotOperationClient,
    wait: bool = False,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
) -> DepotFlowResult:
    """Run the canonical release tag Depot flow."""

    return _run_single_operation_flow(
        flow_request,
        client=client,
        operation_type=DepotOperationType.RELEASE_TAG,
        step_name="release_tag",
        wait=wait,
        polling=polling,
        emitter=emitter,
        observability=observability,
    )


def run_rollback_flow(
    flow_request: DepotFlowRequest,
    *,
    client: DepotOperationClient,
    wait: bool = False,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
) -> DepotFlowResult:
    """Run the canonical rollback Depot flow."""

    return _run_single_operation_flow(
        flow_request,
        client=client,
        operation_type=DepotOperationType.ROLLBACK_TO_SNAPSHOT,
        step_name="rollback",
        wait=wait,
        polling=polling,
        emitter=emitter,
        observability=observability,
    )


def execute_depot_flow(
    flow_request: DepotFlowRequest,
    *,
    client: DepotOperationClient,
    wait: bool = False,
    polling: PollingConfig | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
) -> DepotFlowResult:
    """Execute a shared Depot orchestration flow through one canonical facade."""

    flow_type = flow_request.flow_type.lower()
    if flow_type == "scheduled_sync":
        return run_scheduled_sync_flow(
            flow_request,
            client=client,
            wait=wait,
            polling=polling,
            emitter=emitter,
            observability=observability,
        )
    if flow_type == "scheduled_validation":
        return run_scheduled_validation_flow(
            flow_request,
            client=client,
            wait=wait,
            polling=polling,
            emitter=emitter,
            observability=observability,
        )
    if flow_type == "release_tag":
        return run_release_tag_flow(
            flow_request,
            client=client,
            wait=wait,
            polling=polling,
            emitter=emitter,
            observability=observability,
        )
    if flow_type == "rollback":
        return run_rollback_flow(
            flow_request,
            client=client,
            wait=wait,
            polling=polling,
            emitter=emitter,
            observability=observability,
        )
    raise ValueError(f"Unsupported Depot flow type: {flow_request.flow_type}")


def _run_single_operation_flow(
    flow_request: DepotFlowRequest,
    *,
    client: DepotOperationClient,
    operation_type: DepotOperationType,
    step_name: str,
    wait: bool,
    polling: PollingConfig | None,
    emitter: ObservabilityEmitter | None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None,
) -> DepotFlowResult:
    runtime_emitter = _resolve_runtime_emitter(observability, emitter=emitter)
    started_flow = _build_flow_started_result(flow_request)
    emit_depot_flow_event(
        runtime_emitter,
        event_type=DepotFlowEventType.FLOW_STARTED,
        result=started_flow,
        runtime_context=flow_request.runtime_context,
        metadata=flow_request.metadata,
    )
    request = _build_flow_operation_request(flow_request, operation_type=operation_type)
    try:
        operation_result = execute_depot_operation(
            request,
            runtime_context=flow_request.runtime_context,
            client=client,
            artifact_refs=flow_request.artifact_refs,
            wait=wait,
            polling=polling,
            emitter=runtime_emitter,
            metadata=flow_request.metadata,
        )
        flow_result = _build_flow_result(
            flow_request,
            operation_result=operation_result,
            step_name=step_name,
        )
    except Exception as exc:
        failed_result = result_from_runtime_failure(
            request,
            runtime_context=flow_request.runtime_context,
            artifact_refs=flow_request.artifact_refs,
            exc=exc,
        )
        flow_result = _build_flow_result(
            flow_request,
            operation_result=failed_result,
            step_name=step_name,
            step_metadata={"raised_exception": type(exc).__name__},
        )
        emit_depot_flow_event(
            runtime_emitter,
            event_type=DepotFlowEventType.FLOW_FAILED,
            result=flow_result,
            runtime_context=flow_request.runtime_context,
            metadata=flow_request.metadata,
        )
        return flow_result

    event_type = _flow_event_type_for_result(flow_result)
    emit_depot_flow_event(
        runtime_emitter,
        event_type=event_type,
        result=flow_result,
        runtime_context=flow_request.runtime_context,
        metadata=flow_request.metadata,
    )
    return flow_result


def _build_flow_started_result(flow_request: DepotFlowRequest) -> DepotFlowResult:
    pending = DepotOperationResult(
        operation_id=f"{flow_request.flow_type}:{flow_request.depot_id}:{flow_request.asset_id}:pending",
        operation_type=DepotOperationType.SCHEDULED_SYNC,
        status=DepotOperationStatus.PENDING,
        depot_id=flow_request.depot_id,
        asset_id=flow_request.asset_id,
        branch_id=flow_request.branch_id,
        release_tag=flow_request.release_tag,
        artifact_refs=flow_request.artifact_refs or DepotArtifactRefs(),
        platform_metadata=build_depot_platform_metadata(flow_request.runtime_context),
        metadata={"phase": "flow_started", **flow_request.metadata},
    )
    return DepotFlowResult(
        flow_type=flow_request.flow_type,
        status=DepotFlowStatus.WAITING,
        depot_id=flow_request.depot_id,
        asset_id=flow_request.asset_id,
        branch_id=flow_request.branch_id,
        release_tag=flow_request.release_tag,
        final_result=pending,
        platform_metadata=pending.platform_metadata,
        artifact_refs=pending.artifact_refs,
        metadata={"phase": "flow_started", **flow_request.metadata},
    )


def _build_flow_operation_request(
    flow_request: DepotFlowRequest,
    *,
    operation_type: DepotOperationType,
) -> DepotOperationRequest:
    host_execution = flow_request.runtime_context.host_execution
    run_identity = (
        host_execution.get("run_id")
        or host_execution.get("flow_run_id")
        or host_execution.get("execution_id")
        or "manual"
    )
    operation_id = (
        f"{operation_type.value}:{flow_request.depot_id}:{flow_request.asset_id}:{run_identity}"
    )
    return DepotOperationRequest(
        operation_id=operation_id,
        operation_type=operation_type,
        depot_id=flow_request.depot_id,
        asset_id=flow_request.asset_id,
        branch_id=flow_request.branch_id,
        snapshot_id=flow_request.snapshot_id,
        target_branch_id=flow_request.target_branch_id,
        release_tag=flow_request.release_tag,
        requested_by=flow_request.requested_by,
        idempotency_key=build_idempotency_key(
            operation_type,
            flow_request.depot_id,
            flow_request.asset_id,
            branch_id=flow_request.branch_id,
            snapshot_id=flow_request.snapshot_id,
            release_tag=flow_request.release_tag,
            target_branch_id=flow_request.target_branch_id,
        ),
        metadata={
            **flow_request.metadata,
            "flow_type": flow_request.flow_type,
        },
    )


def _build_flow_result(
    flow_request: DepotFlowRequest,
    *,
    operation_result: DepotOperationResult,
    step_name: str,
    step_metadata: dict[str, Any] | None = None,
) -> DepotFlowResult:
    step = DepotFlowStepResult(
        step_name=step_name,
        operation_type=operation_result.operation_type.value,
        result=operation_result,
        metadata=step_metadata or {},
    )
    artifact_refs = _aggregate_flow_artifact_refs(
        flow_request.artifact_refs,
        operation_result.artifact_refs,
    )
    return DepotFlowResult(
        flow_type=flow_request.flow_type,
        status=_flow_status_from_operation_result(operation_result),
        depot_id=flow_request.depot_id,
        asset_id=flow_request.asset_id,
        branch_id=flow_request.branch_id,
        release_tag=flow_request.release_tag,
        steps=(step,),
        final_result=operation_result,
        platform_metadata=operation_result.platform_metadata,
        artifact_refs=artifact_refs,
        metadata={
            **flow_request.metadata,
            "step_count": 1,
        },
    )


def _aggregate_flow_artifact_refs(
    caller_refs: DepotArtifactRefs | None,
    final_refs: DepotArtifactRefs,
) -> DepotArtifactRefs:
    if caller_refs is None:
        return final_refs
    return DepotArtifactRefs(
        core_result_ref=final_refs.core_result_ref or caller_refs.core_result_ref,
        core_gate_result_ref=final_refs.core_gate_result_ref or caller_refs.core_gate_result_ref,
        core_evidence_ref=final_refs.core_evidence_ref or caller_refs.core_evidence_ref,
        depot_operation_ref=final_refs.depot_operation_ref or caller_refs.depot_operation_ref,
        merge_request_ref=final_refs.merge_request_ref or caller_refs.merge_request_ref,
        release_ref=final_refs.release_ref or caller_refs.release_ref,
        extras={**caller_refs.extras, **final_refs.extras},
    )


def _flow_status_from_operation_result(result: DepotOperationResult) -> DepotFlowStatus:
    if result.status == DepotOperationStatus.NO_OP:
        return DepotFlowStatus.NO_OP
    if result.status == DepotOperationStatus.WAITING:
        return DepotFlowStatus.WAITING
    if result.status == DepotOperationStatus.CANCELLED:
        return DepotFlowStatus.CANCELLED
    if result.status == DepotOperationStatus.FAILED:
        return DepotFlowStatus.FAILED
    return DepotFlowStatus.SUCCEEDED


def _flow_event_type_for_result(result: DepotFlowResult) -> DepotFlowEventType:
    if result.status == DepotFlowStatus.NO_OP:
        return DepotFlowEventType.FLOW_NO_OP
    if result.status == DepotFlowStatus.WAITING:
        return DepotFlowEventType.FLOW_WAITING
    if result.status == DepotFlowStatus.FAILED:
        return DepotFlowEventType.FLOW_FAILED
    return DepotFlowEventType.FLOW_COMPLETED


@dataclass(frozen=True, slots=True)
class _SingleResultDepotClient:
    """Adapter used to reuse normalization helpers for one already-fetched result."""

    result: DepotOperationResult

    def get_operation(self, operation_id: str) -> DepotOperationResult:
        if operation_id != self.result.operation_id:
            raise ValueError(f"Unexpected operation id {operation_id}")
        return self.result

    def submit_operation(self, request: DepotOperationRequest) -> DepotOperationResult:
        del request
        return self.result


def _serialize_runtime_result(result: Any) -> dict[str, Any]:
    """Serialize shared runtime results, including simple test doubles."""

    if hasattr(result, "to_dict"):
        try:
            return serialize_result_wire(result, include_result_type=True)
        except TypeError:
            payload = cast(dict[str, Any], result.to_dict())
            payload.setdefault("result_type", payload.get("type", "check"))
            return payload

    status = getattr(result, "status", None)
    if status is not None and hasattr(status, "name"):
        status_value = status.name
    elif status is not None and hasattr(status, "value"):
        status_value = str(status.value)
    else:
        status_value = str(status) if status is not None else "unknown"
    return {
        "status": status_value,
        "passed_count": getattr(result, "passed_count", 0),
        "failed_count": getattr(result, "failed_count", 0),
        "warning_count": getattr(result, "warning_count", 0),
        "failure_rate": getattr(result, "failure_rate", 0.0),
        "is_success": getattr(result, "is_success", False),
        "result_type": "check",
    }


def execute_operation(
    operation: OperationKind | str,
    engine: DataQualityEngine,
    *,
    data: Any | None = None,
    rules: Sequence[dict[str, Any]] | None = None,
    baseline: Any | None = None,
    current: Any | None = None,
    runtime_context: PlatformRuntimeContext | None = None,
    source: ResolvedDataSource | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a shared first-party operation with capability and telemetry handling."""

    normalized = normalize_operation_kind(operation)
    ensure_operation_supported(engine, normalized)
    runtime_emitter = _resolve_runtime_emitter(observability, emitter=emitter)

    if source is None and data is not None:
        source = resolve_data_source(data)

    metadata = {"runtime_context": runtime_context.to_dict() if runtime_context is not None else None}
    emit_runtime_event(
        runtime_emitter,
        event_type=OperationEventType.STARTED,
        operation=normalized,
        engine_name=engine.engine_name,
        runtime_context=runtime_context,
        source=source,
        metadata=metadata,
    )

    try:
        if normalized == OperationKind.CHECK:
            effective_rules, effective_kwargs = prepare_check_invocation(engine, rules, **kwargs)
            result: Any = engine.check(data, effective_rules, **effective_kwargs)
        elif normalized == OperationKind.PROFILE:
            result = engine.profile(data, **kwargs)
        elif normalized == OperationKind.LEARN:
            result = engine.learn(data, **kwargs)
        elif normalized == OperationKind.DRIFT:
            result = engine.detect_drift(baseline, current, **kwargs)  # type: ignore[attr-defined]
        elif normalized == OperationKind.ANOMALY:
            result = engine.detect_anomalies(data, **kwargs)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsupported execution operation: {normalized.value}")

        emit_runtime_event(
            runtime_emitter,
            event_type=OperationEventType.COMPLETED,
            operation=normalized,
            engine_name=engine.engine_name,
            runtime_context=runtime_context,
            source=source,
            result=result.to_dict() if hasattr(result, "to_dict") else None,
            metadata=metadata,
        )
        return result
    except Exception as exc:
        emit_runtime_event(
            runtime_emitter,
            event_type=OperationEventType.FAILED,
            operation=normalized,
            engine_name=engine.engine_name,
            runtime_context=runtime_context,
            source=source,
            metadata=metadata,
            error=exc,
        )
        raise


@dataclass(frozen=True, slots=True)
class StreamCheckpointState:
    """Streaming checkpoint state shared across all first-party platforms."""

    batch_index: int = 0
    records_processed: int = 0
    checkpoint_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "records_processed": self.records_processed,
            "checkpoint_token": self.checkpoint_token,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class StreamRequest:
    """A shared streaming execution request."""

    stream: Any
    rules: Sequence[dict[str, Any]] | None = None
    batch_size: int = 1000
    checkpoint: StreamCheckpointState | None = None
    max_batches: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamBatchEnvelope:
    """One batch result in the shared streaming wire contract."""

    batch_index: int
    records_in_batch: int
    checkpoint: StreamCheckpointState
    result: CheckResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "records_in_batch": self.records_in_batch,
            "checkpoint": self.checkpoint.to_dict(),
            "result": _serialize_runtime_result(self.result),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class StreamSummary:
    """Terminal summary for a completed streaming run."""

    total_batches: int
    total_records: int
    passed_batches: int
    failed_batches: int
    warning_batches: int
    final_status: str
    last_checkpoint: StreamCheckpointState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_batches": self.total_batches,
            "total_records": self.total_records,
            "passed_batches": self.passed_batches,
            "failed_batches": self.failed_batches,
            "warning_batches": self.warning_batches,
            "final_status": self.final_status,
            "last_checkpoint": (
                self.last_checkpoint.to_dict() if self.last_checkpoint is not None else None
            ),
            "metadata": self.metadata,
        }


def run_stream_check(
    engine: DataQualityEngine,
    request: StreamRequest,
    *,
    runtime_context: PlatformRuntimeContext | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
) -> Iterator[StreamBatchEnvelope]:
    """Execute streaming validation with bounded memory."""

    ensure_operation_supported(engine, OperationKind.STREAM)
    runtime_emitter = _resolve_runtime_emitter(observability, emitter=emitter)

    source = resolve_data_source(request.stream)
    if request.checkpoint is not None:
        emit_runtime_event(
            runtime_emitter,
            event_type=OperationEventType.RESUMED,
            operation=OperationKind.STREAM,
            engine_name=engine.engine_name,
            runtime_context=runtime_context,
            source=source,
            metadata={"checkpoint": request.checkpoint.to_dict()},
        )
    else:
        emit_runtime_event(
            runtime_emitter,
            event_type=OperationEventType.STARTED,
            operation=OperationKind.STREAM,
            engine_name=engine.engine_name,
            runtime_context=runtime_context,
            source=source,
            metadata={"batch_size": request.batch_size},
        )

    effective_rules, effective_kwargs = prepare_check_invocation(
        engine,
        request.rules,
        **request.kwargs,
    )
    checkpoint = request.checkpoint or StreamCheckpointState()

    if supports_streaming(engine):
        stream_input = _resume_stream(request.stream, skip_records=checkpoint.records_processed)
        batch_index = checkpoint.batch_index
        records_processed = checkpoint.records_processed
        for result in engine.check_stream(  # type: ignore[attr-defined]
            stream_input,
            batch_size=request.batch_size,
            **effective_kwargs,
        ):
            records_in_batch = _infer_batch_record_count(result, request.batch_size)
            batch_index += 1
            records_processed += records_in_batch
            envelope = StreamBatchEnvelope(
                batch_index=batch_index,
                records_in_batch=records_in_batch,
                checkpoint=StreamCheckpointState(
                    batch_index=batch_index,
                    records_processed=records_processed,
                    checkpoint_token=f"batch-{batch_index}",
                    metadata=request.metadata,
                ),
                result=result,
                metadata=dict(request.metadata),
            )
            emit_runtime_event(
                runtime_emitter,
                event_type=OperationEventType.BATCH_COMPLETED,
                operation=OperationKind.STREAM,
                engine_name=engine.engine_name,
                runtime_context=runtime_context,
                source=source,
                result=result,
                metadata={"batch_index": batch_index, "checkpoint": envelope.checkpoint.to_dict()},
            )
            yield envelope
            if request.max_batches is not None and batch_index >= request.max_batches:
                break
    else:
        records_processed = checkpoint.records_processed
        for batch_index, batch in _iter_batches(
            request.stream,
            batch_size=request.batch_size,
            checkpoint=checkpoint,
            max_batches=request.max_batches,
        ):
            payload = _coerce_stream_batch_payload(batch)
            result = execute_operation(
                OperationKind.CHECK,
                engine,
                data=payload,
                rules=effective_rules,
                runtime_context=runtime_context,
                source=source,
                emitter=None,
                observability=None,
                **effective_kwargs,
            )
            records_in_batch = len(batch)
            records_processed += records_in_batch
            envelope = StreamBatchEnvelope(
                batch_index=batch_index,
                records_in_batch=records_in_batch,
                checkpoint=StreamCheckpointState(
                    batch_index=batch_index,
                    records_processed=records_processed,
                    checkpoint_token=f"batch-{batch_index}",
                    metadata=request.metadata,
                ),
                result=result,
                metadata=dict(request.metadata),
            )
            emit_runtime_event(
                runtime_emitter,
                event_type=OperationEventType.BATCH_COMPLETED,
                operation=OperationKind.STREAM,
                engine_name=engine.engine_name,
                runtime_context=runtime_context,
                source=source,
                result=result,
                metadata={"batch_index": batch_index, "checkpoint": envelope.checkpoint.to_dict()},
            )
            yield envelope

    emit_runtime_event(
        runtime_emitter,
        event_type=OperationEventType.COMPLETED,
        operation=OperationKind.STREAM,
        engine_name=engine.engine_name,
        runtime_context=runtime_context,
        source=source,
        metadata={"batch_size": request.batch_size},
    )
    if runtime_emitter is not None:
        runtime_emitter.flush()


async def run_stream_check_async(
    engine: DataQualityEngine,
    request: StreamRequest,
    *,
    runtime_context: PlatformRuntimeContext | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
) -> AsyncIterator[StreamBatchEnvelope]:
    """Async variant of the shared streaming runner."""

    if hasattr(request.stream, "__aiter__"):
        ensure_operation_supported(engine, OperationKind.STREAM)
        runtime_emitter = _resolve_runtime_emitter(observability, emitter=emitter)
        source = resolve_data_source(request.stream)
        emit_runtime_event(
            runtime_emitter,
            event_type=OperationEventType.STARTED,
            operation=OperationKind.STREAM,
            engine_name=engine.engine_name,
            runtime_context=runtime_context,
            source=source,
            metadata={"batch_size": request.batch_size},
        )

        effective_rules, effective_kwargs = prepare_check_invocation(
            engine,
            request.rules,
            **request.kwargs,
        )
        checkpoint = request.checkpoint or StreamCheckpointState()

        batch_index = checkpoint.batch_index
        records_processed = checkpoint.records_processed
        async for batch in _aiter_batches(
            request.stream,
            batch_size=request.batch_size,
            checkpoint=checkpoint,
            max_batches=request.max_batches,
        ):
            batch_index += 1
            payload = _coerce_stream_batch_payload(batch)
            result = execute_operation(
                OperationKind.CHECK,
                engine,
                data=payload,
                rules=effective_rules,
                runtime_context=runtime_context,
                source=source,
                emitter=None,
                observability=None,
                **effective_kwargs,
            )
            records_in_batch = len(batch)
            records_processed += records_in_batch
            envelope = StreamBatchEnvelope(
                batch_index=batch_index,
                records_in_batch=records_in_batch,
                checkpoint=StreamCheckpointState(
                    batch_index=batch_index,
                    records_processed=records_processed,
                    checkpoint_token=f"batch-{batch_index}",
                    metadata=request.metadata,
                ),
                result=result,
                metadata=dict(request.metadata),
            )
            emit_runtime_event(
                runtime_emitter,
                event_type=OperationEventType.BATCH_COMPLETED,
                operation=OperationKind.STREAM,
                engine_name=engine.engine_name,
                runtime_context=runtime_context,
                source=source,
                result=result,
                metadata={"batch_index": batch_index, "checkpoint": envelope.checkpoint.to_dict()},
            )
            yield envelope

        emit_runtime_event(
            runtime_emitter,
            event_type=OperationEventType.COMPLETED,
            operation=OperationKind.STREAM,
            engine_name=engine.engine_name,
            runtime_context=runtime_context,
            source=source,
            metadata={"batch_size": request.batch_size},
        )
        if runtime_emitter is not None:
            runtime_emitter.flush()
        return

    for envelope in run_stream_check(
        engine,
        request,
        runtime_context=runtime_context,
        emitter=emitter,
        observability=observability,
    ):
        yield envelope


def summarize_stream(envelopes: Sequence[StreamBatchEnvelope]) -> StreamSummary:
    """Build a terminal summary from stream batch envelopes."""

    total_records = sum(envelope.records_in_batch for envelope in envelopes)
    passed_batches = sum(envelope.result.status == CheckStatus.PASSED for envelope in envelopes)
    failed_batches = sum(envelope.result.status == CheckStatus.FAILED for envelope in envelopes)
    warning_batches = sum(envelope.result.status == CheckStatus.WARNING for envelope in envelopes)

    final_status = CheckStatus.PASSED.name
    if failed_batches:
        final_status = CheckStatus.FAILED.name
    elif warning_batches:
        final_status = CheckStatus.WARNING.name

    last_checkpoint = envelopes[-1].checkpoint if envelopes else None
    return StreamSummary(
        total_batches=len(envelopes),
        total_records=total_records,
        passed_batches=passed_batches,
        failed_batches=failed_batches,
        warning_batches=warning_batches,
        final_status=final_status,
        last_checkpoint=last_checkpoint,
    )


@dataclass(frozen=True, slots=True)
class QualityGateConfig:
    """Shared config for quality gate evaluation."""

    min_pass_rate: float = 1.0
    min_row_count: int | None = None
    max_failure_count: int | None = None
    continue_on_error: bool = False
    timeout_seconds: float = 300.0


@dataclass(frozen=True, slots=True)
class QualityGateDecision:
    """Result of evaluating a shared quality gate."""

    satisfied: bool
    reason: str
    pass_rate: float
    row_count: int
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "reason": self.reason,
            "pass_rate": self.pass_rate,
            "row_count": self.row_count,
            "result": self.result,
            "metadata": self.metadata,
        }


def evaluate_quality_gate(
    engine: DataQualityEngine,
    data: Any,
    *,
    rules: Sequence[dict[str, Any]] | None = None,
    config: QualityGateConfig | None = None,
    runtime_context: PlatformRuntimeContext | None = None,
    emitter: ObservabilityEmitter | None = None,
    observability: ObservabilityConfig | ObservabilityEmitter | dict[str, Any] | None = None,
    **kwargs: Any,
) -> QualityGateDecision:
    """Evaluate a quality gate using shared check semantics."""

    def _result_payload(result: Any) -> dict[str, Any]:
        if hasattr(result, "to_dict"):
            try:
                return serialize_result_wire(result, include_result_type=True)
            except TypeError:
                pass

        status = getattr(result, "status", None)
        if status is not None and hasattr(status, "name"):
            status_value = status.name
        elif status is not None and hasattr(status, "value"):
            status_value = str(status.value)
        else:
            status_value = str(status) if status is not None else "unknown"
        return {
            "status": status_value,
            "passed_count": getattr(result, "passed_count", 0),
            "failed_count": getattr(result, "failed_count", 0),
            "warning_count": getattr(result, "warning_count", 0),
            "failure_rate": getattr(result, "failure_rate", 0.0),
            "is_success": getattr(result, "is_success", False),
        }

    gate_config = config or QualityGateConfig()
    source = resolve_data_source(data)
    if source is not None and source.kind == DataSourceKind.SYNC_STREAM:
        envelopes = list(
            run_stream_check(
                engine,
                StreamRequest(
                    stream=data,
                    rules=rules,
                    kwargs={"timeout": gate_config.timeout_seconds, **kwargs},
                ),
                runtime_context=runtime_context,
                emitter=emitter,
                observability=observability,
            )
        )
        summary = summarize_stream(envelopes)
        total_batches = summary.total_batches
        pass_rate = 1.0 if total_batches == 0 else (summary.passed_batches / total_batches)
        row_count = summary.total_records

        if gate_config.min_row_count is not None and row_count < gate_config.min_row_count:
            return QualityGateDecision(
                satisfied=False,
                reason=f"row_count {row_count} is below minimum {gate_config.min_row_count}",
                pass_rate=pass_rate,
                row_count=row_count,
                result=summary.to_dict(),
            )

        if gate_config.max_failure_count is not None and summary.failed_batches > gate_config.max_failure_count:
            return QualityGateDecision(
                satisfied=False,
                reason=(
                    f"failed_batches {summary.failed_batches} exceeds maximum "
                    f"{gate_config.max_failure_count}"
                ),
                pass_rate=pass_rate,
                row_count=row_count,
                result=summary.to_dict(),
            )

        if total_batches == 0:
            return QualityGateDecision(
                satisfied=False,
                reason="no stream batches were evaluated",
                pass_rate=pass_rate,
                row_count=row_count,
                result=summary.to_dict(),
            )

        if pass_rate >= gate_config.min_pass_rate:
            return QualityGateDecision(
                satisfied=True,
                reason="quality gate conditions satisfied",
                pass_rate=pass_rate,
                row_count=row_count,
                result=summary.to_dict(),
            )

        return QualityGateDecision(
            satisfied=False,
            reason=(
                f"pass_rate {pass_rate:.4f} is below minimum "
                f"{gate_config.min_pass_rate:.4f}"
            ),
            pass_rate=pass_rate,
            row_count=row_count,
            result=summary.to_dict(),
        )

    try:
        result = execute_operation(
            OperationKind.CHECK,
            engine,
            data=data,
            rules=rules,
            runtime_context=runtime_context,
            source=source,
            emitter=emitter,
            observability=observability,
            timeout=gate_config.timeout_seconds,
            **kwargs,
        )
    except Exception as exc:
        if gate_config.continue_on_error:
            return QualityGateDecision(
                satisfied=False,
                reason=str(exc),
                pass_rate=0.0,
                row_count=0,
                metadata={"error_type": type(exc).__name__},
            )
        raise

    total_rules = getattr(result, "total_count", result.passed_count + result.failed_count)
    pass_rate = 1.0 if total_rules == 0 else (result.passed_count / total_rules)
    row_count = _resolve_row_count(data)

    if gate_config.min_row_count is not None and row_count < gate_config.min_row_count:
        return QualityGateDecision(
            satisfied=False,
            reason=f"row_count {row_count} is below minimum {gate_config.min_row_count}",
            pass_rate=pass_rate,
            row_count=row_count,
            result=_result_payload(result),
        )

    if gate_config.max_failure_count is not None and result.failed_count > gate_config.max_failure_count:
        return QualityGateDecision(
            satisfied=False,
            reason=(
                f"failed_count {result.failed_count} exceeds maximum "
                f"{gate_config.max_failure_count}"
            ),
            pass_rate=pass_rate,
            row_count=row_count,
            result=_result_payload(result),
        )

    if total_rules == 0:
        return QualityGateDecision(
            satisfied=False,
            reason="no rules were evaluated",
            pass_rate=pass_rate,
            row_count=row_count,
            result=_result_payload(result),
        )

    if pass_rate >= gate_config.min_pass_rate:
        return QualityGateDecision(
            satisfied=True,
            reason="quality gate conditions satisfied",
            pass_rate=pass_rate,
            row_count=row_count,
            result=_result_payload(result),
        )

    return QualityGateDecision(
        satisfied=False,
        reason=(
            f"pass_rate {pass_rate:.4f} is below minimum "
            f"{gate_config.min_pass_rate:.4f}"
        ),
        pass_rate=pass_rate,
        row_count=row_count,
        result=_result_payload(result),
    )


def _iter_batches(
    stream: Any,
    *,
    batch_size: int,
    checkpoint: StreamCheckpointState,
    max_batches: int | None,
) -> Iterator[tuple[int, list[Any]]]:
    iterator = iter(_resume_stream(stream, skip_records=checkpoint.records_processed))
    emitted = 0
    batch_index = checkpoint.batch_index

    while True:
        batch: list[Any] = []
        try:
            while len(batch) < batch_size:
                batch.append(next(iterator))
        except StopIteration:
            if not batch:
                break
        batch_index += 1
        emitted += 1
        yield batch_index, batch
        if max_batches is not None and emitted >= max_batches:
            break


async def _aiter_batches(
    stream: Any,
    *,
    batch_size: int,
    checkpoint: StreamCheckpointState,
    max_batches: int | None,
) -> AsyncIterator[list[Any]]:
    emitted = 0
    batch: list[Any] = []
    index = 0
    async for item in _resume_async_stream(stream, skip_records=checkpoint.records_processed):
        batch.append(item)
        if len(batch) == batch_size:
            index += 1
            emitted += 1
            yield list(batch)
            batch = []
            if max_batches is not None and emitted >= max_batches:
                return
    if batch:
        yield list(batch)


def _resume_stream(stream: Any, *, skip_records: int) -> Iterator[Any]:
    if callable(stream):
        stream = stream()
    iterator = iter(stream)
    for _ in range(skip_records):
        try:
            next(iterator)
        except StopIteration:
            return iter(())
    return cast(Iterator[Any], iterator)


async def _resume_async_stream(stream: Any, *, skip_records: int) -> AsyncIterator[Any]:
    async_iterable = stream() if callable(stream) and inspect.iscoroutinefunction(stream) else stream
    iterator = async_iterable.__aiter__()
    skipped = 0
    while skipped < skip_records:
        try:
            await iterator.__anext__()
        except StopAsyncIteration:
            return
        skipped += 1
    while True:
        try:
            yield await iterator.__anext__()
        except StopAsyncIteration:
            return


def _coerce_stream_batch_payload(batch: Sequence[Any]) -> Any:
    if not batch:
        return []
    if all(isinstance(item, dict) for item in batch):
        try:
            import polars as pl

            return pl.DataFrame(batch)
        except Exception:
            return list(batch)
    return list(batch)


def _infer_batch_record_count(result: CheckResult, fallback: int) -> int:
    row_count = result.metadata.get("row_count") if isinstance(result.metadata, dict) else None
    if isinstance(row_count, int):
        return row_count
    return fallback


def _resolve_row_count(data: Any) -> int:
    if hasattr(data, "height"):
        try:
            return int(data.height)
        except Exception:
            return 0
    if hasattr(data, "__len__"):
        try:
            return len(data)
        except Exception:
            return 0
    return 0
