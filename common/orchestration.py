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

from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import inspect
import json
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from common.base import CheckResult, CheckStatus
from common.engines.base import (
    DataQualityEngine,
    supports_anomaly,
    supports_drift,
    supports_streaming,
)
from common.runtime import (
    DataSourceKind,
    ObservabilityBackend,
    ObservabilityConfig,
    PlatformRuntimeContext,
    ResolvedDataSource,
    normalize_observability_config,
    resolve_data_source,
)
from common.serializers import serialize_result_wire


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class OperationKind(str, Enum):
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
        capability_source = engine.get_capabilities()  # type: ignore[attr-defined]
    elif hasattr(engine, "capabilities"):
        capability_source = engine.capabilities  # type: ignore[attr-defined]

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


class OperationEventType(str, Enum):
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


@runtime_checkable
class ObservabilityEmitter(Protocol):
    """Protocol for runtime observability emitters."""

    def emit(self, event: ObservabilityEvent) -> None:
        """Emit a runtime event."""

    def flush(self) -> None:
        """Flush buffered runtime events."""

    def preflight_check(self) -> tuple[bool, str]:
        """Validate that the emitter is ready for execution."""


class NoOpEmitter:
    """Default emitter used when no observability sink is configured."""

    def emit(self, event: ObservabilityEvent) -> None:  # pragma: no cover - intentionally empty
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
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
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


def create_observability_emitter(
    config: ObservabilityConfig | dict[str, Any] | None = None,
) -> ObservabilityEmitter:
    """Create a shared observability emitter from typed config."""

    normalized = normalize_observability_config(config)
    if normalized.backend == ObservabilityBackend.NONE:
        return NoOpEmitter()
    if normalized.backend == ObservabilityBackend.OPENLINEAGE:
        return OpenLineageEmitter(
            namespace=normalized.namespace,
            job_name=normalized.job_name,
            endpoint=normalized.endpoint,
            producer=normalized.producer,
            timeout_seconds=normalized.timeout_seconds,
            retry_count=normalized.retry_count,
            retry_backoff_seconds=normalized.retry_backoff_seconds,
        )
    raise ValueError(f"Unsupported observability backend: {normalized.backend.value}")


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


def _serialize_runtime_result(result: Any) -> dict[str, Any]:
    """Serialize shared runtime results, including simple test doubles."""

    if hasattr(result, "to_dict"):
        try:
            return serialize_result_wire(result, include_result_type=True)
        except TypeError:
            payload = result.to_dict()
            payload.setdefault("result_type", payload.get("type", "check"))
            return payload

    status = getattr(result, "status", None)
    if hasattr(status, "name"):
        status_value = status.name
    elif hasattr(status, "value"):
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
            result = engine.check(data, effective_rules, **effective_kwargs)
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
        if hasattr(status, "name"):
            status_value = status.name
        elif hasattr(status, "value"):
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
    if source.kind == DataSourceKind.SYNC_STREAM:
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
    return iterator


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
            return int(len(data))
        except Exception:
            return 0
    return 0
