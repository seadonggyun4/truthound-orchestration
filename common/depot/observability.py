"""Shared Depot observability helpers, emitters, and redaction rules."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from common.depot.failures import (
    DepotFailureCode,
    build_failure,
    classify_operation_result_failure,
    normalize_failure_code,
)
from common.depot.models import DepotArtifactRefs, DepotOperationStatus, make_json_safe


TRACE_FIELD_NAMES: tuple[str, ...] = (
    "operation_id",
    "operation_type",
    "flow_type",
    "depot_id",
    "asset_id",
    "branch_id",
    "snapshot_id",
    "merge_request_id",
    "release_tag",
    "quality_gate_id",
    "platform",
    "platform_run_id",
    "platform_task_id",
    "platform_job_name",
    "truthound_run_id",
    "request_id",
    "status",
    "error_code",
    "retry_disposition",
)

_SENSITIVE_KEY_PARTS: tuple[str, ...] = (
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "access_key",
    "refresh_token",
    "cookie",
    "set_cookie",
)
_RAW_BODY_FIELDS: frozenset[str] = frozenset(
    {
        "snapshot",
        "snapshot_body",
        "evidence",
        "evidence_blob",
        "raw_rows",
        "dataset",
        "headers",
    }
)
_REDACTED = "[REDACTED]"


def normalize_observability_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


_NORMALIZED_RAW_BODY_FIELDS: frozenset[str] = frozenset(
    normalize_observability_key(name) for name in _RAW_BODY_FIELDS
)


def _is_sensitive_key(key: str) -> bool:
    normalized = normalize_observability_key(key)
    return any(part.replace("_", "") in normalized for part in _SENSITIVE_KEY_PARTS)


def _is_raw_body_key(key: str) -> bool:
    return normalize_observability_key(key) in _NORMALIZED_RAW_BODY_FIELDS


def sanitize_observability_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme and not parsed.netloc and "?" not in value and "#" not in value:
        return value
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def redact_observability_payload(value: Any, *, key: str | None = None) -> Any:
    if key is not None:
        if _is_sensitive_key(key):
            return _REDACTED
        if _is_raw_body_key(key):
            return _REDACTED

    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return sanitize_observability_url(value)
    if isinstance(value, dict):
        return {
            str(item_key): redact_observability_payload(item_value, key=str(item_key))
            for item_key, item_value in make_json_safe(value).items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_observability_payload(item) for item in make_json_safe(value)]
    return redact_observability_payload(make_json_safe(value))


def redact_artifact_refs(artifact_refs: DepotArtifactRefs) -> dict[str, Any]:
    return cast(dict[str, Any], redact_observability_payload(artifact_refs.to_dict()))


def _platform_fields(platform_metadata: Any, host_execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform_run_id": (
            getattr(platform_metadata, "platform_run_id", None)
            or host_execution.get("run_id")
            or host_execution.get("flow_run_id")
        ),
        "platform_task_id": (
            getattr(platform_metadata, "platform_task_id", None)
            or host_execution.get("task_id")
            or host_execution.get("task_run_id")
        ),
        "platform_job_name": (
            getattr(platform_metadata, "platform_job_name", None)
            or host_execution.get("job_name")
            or host_execution.get("dag_id")
            or host_execution.get("flow_name")
            or host_execution.get("op_name")
        ),
    }


def build_trace_fields_for_depot_event(event: Any) -> dict[str, Any]:
    host_execution = redact_observability_payload(event.host_execution)
    failure = build_failure_summary_for_depot_event(event)
    platform_fields = _platform_fields(event.platform_metadata, host_execution)
    trace_fields = {
        "operation_id": event.operation_id,
        "operation_type": event.operation_type.value,
        "flow_type": None,
        "depot_id": event.depot_id,
        "asset_id": event.asset_id,
        "branch_id": event.branch_id,
        "snapshot_id": event.snapshot_id,
        "merge_request_id": event.merge_request_id,
        "release_tag": event.release_tag,
        "quality_gate_id": event.quality_gate_id,
        "platform": event.platform,
        "platform_run_id": platform_fields["platform_run_id"],
        "platform_task_id": platform_fields["platform_task_id"],
        "platform_job_name": platform_fields["platform_job_name"],
        "truthound_run_id": (
            event.metadata.get("truthound_run_id") or host_execution.get("truthound_run_id")
        ),
        "request_id": event.request_id or event.metadata.get("request_id"),
        "status": event.status.value,
        "error_code": event.error_code,
        "retry_disposition": failure["retry_disposition"] if failure else None,
    }
    return {name: trace_fields.get(name) for name in TRACE_FIELD_NAMES}


def build_trace_fields_for_depot_flow_event(event: Any) -> dict[str, Any]:
    host_execution = redact_observability_payload(event.host_execution)
    failure = build_failure_summary_from_result(event.final_result)
    platform_fields = _platform_fields(event.platform_metadata, host_execution)
    final_result = event.final_result
    trace_fields = {
        "operation_id": final_result.operation_id,
        "operation_type": final_result.operation_type.value,
        "flow_type": event.flow_type,
        "depot_id": event.depot_id,
        "asset_id": event.asset_id,
        "branch_id": event.branch_id,
        "snapshot_id": final_result.snapshot_id,
        "merge_request_id": final_result.merge_request_id,
        "release_tag": event.release_tag,
        "quality_gate_id": final_result.quality_gate_id,
        "platform": event.platform,
        "platform_run_id": platform_fields["platform_run_id"],
        "platform_task_id": platform_fields["platform_task_id"],
        "platform_job_name": platform_fields["platform_job_name"],
        "truthound_run_id": (
            event.metadata.get("truthound_run_id") or host_execution.get("truthound_run_id")
        ),
        "request_id": final_result.metadata.get("request_id"),
        "status": event.status.value,
        "error_code": final_result.error_code,
        "retry_disposition": failure["retry_disposition"] if failure else None,
    }
    return {name: trace_fields.get(name) for name in TRACE_FIELD_NAMES}


def build_failure_summary_from_result(result: Any) -> dict[str, Any] | None:
    failure = classify_operation_result_failure(result)
    if failure is None:
        return None
    return failure.to_dict()


def build_failure_summary_for_depot_event(event: Any) -> dict[str, Any] | None:
    if event.status not in {DepotOperationStatus.FAILED, DepotOperationStatus.CANCELLED}:
        return None
    if event.status == DepotOperationStatus.CANCELLED and event.error_code is None:
        failure = build_failure(
            DepotFailureCode.DEPOT_API_FAILURE,
            event.error_message or "operation cancelled",
            details={"status": event.status.value},
        )
        return failure.to_dict()

    original_error_code = event.error_code
    normalized_error_code = normalize_failure_code(original_error_code)
    details: dict[str, Any] = {"status": event.status.value}
    if original_error_code is not None and original_error_code != normalized_error_code.value:
        details["original_error_code"] = original_error_code
    failure = build_failure(
        normalized_error_code,
        event.error_message or "Depot operation failed",
        details=details,
    )
    return failure.to_dict()


def build_structured_log_payload(event: Any) -> dict[str, Any]:
    payload = redact_observability_payload(event.to_dict()) if hasattr(event, "to_dict") else {}
    if hasattr(event, "flow_type"):
        trace_fields = build_trace_fields_for_depot_flow_event(event)
        failure = build_failure_summary_from_result(event.final_result)
        return {
            "event_type": event.event_type.value,
            "platform": event.platform,
            "trace_fields": trace_fields,
            "failure": failure,
            "artifact_refs": redact_artifact_refs(event.artifact_refs),
            "metadata": redact_observability_payload(event.metadata),
            "host_execution": redact_observability_payload(event.host_execution),
            "depot_flow": {
                "flow_type": event.flow_type,
                "status": event.status.value,
                "depot_id": event.depot_id,
                "asset_id": event.asset_id,
                "branch_id": event.branch_id,
                "release_tag": event.release_tag,
                "steps": [redact_observability_payload(step.to_dict()) for step in event.steps],
            },
            "payload": payload,
        }

    trace_fields = build_trace_fields_for_depot_event(event)
    failure = build_failure_summary_for_depot_event(event)
    return {
        "event_type": event.event_type.value,
        "platform": event.platform,
        "trace_fields": trace_fields,
        "failure": failure,
        "artifact_refs": redact_artifact_refs(event.artifact_refs),
        "metadata": redact_observability_payload(event.metadata),
        "host_execution": redact_observability_payload(event.host_execution),
        "depot": {
            "operation_id": event.operation_id,
            "operation_type": event.operation_type.value,
            "status": event.status.value,
            "depot_id": event.depot_id,
            "asset_id": event.asset_id,
            "branch_id": event.branch_id,
            "snapshot_id": event.snapshot_id,
            "merge_request_id": event.merge_request_id,
            "release_tag": event.release_tag,
            "quality_gate_id": event.quality_gate_id,
        },
        "payload": payload,
    }


def build_openlineage_depot_truthound_facet(
    event: Any,
    *,
    producer: str,
) -> dict[str, Any]:
    return {
        "_producer": producer,
        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/RunFacet.json",
        "engine": "depot",
        "platform": event.platform,
        "operation": event.operation_type.value,
        "event_type": event.event_type.value,
        "host_execution": redact_observability_payload(event.host_execution),
        "metadata": redact_observability_payload(event.metadata),
        "trace_fields": build_trace_fields_for_depot_event(event),
        "failure": build_failure_summary_for_depot_event(event),
        "depot": {
            "status": event.status.value,
            "operation_id": event.operation_id,
            "depot_id": event.depot_id,
            "asset_id": event.asset_id,
            "branch_id": event.branch_id,
            "snapshot_id": event.snapshot_id,
            "merge_request_id": event.merge_request_id,
            "quality_gate_id": event.quality_gate_id,
            "release_tag": event.release_tag,
            "artifact_refs": redact_artifact_refs(event.artifact_refs),
            "error_code": event.error_code,
            "error_message": event.error_message,
        },
    }


def build_openlineage_depot_flow_truthound_facet(
    event: Any,
    *,
    producer: str,
) -> dict[str, Any]:
    return {
        "_producer": producer,
        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/RunFacet.json",
        "engine": "depot",
        "platform": event.platform,
        "operation": event.flow_type,
        "event_type": event.event_type.value,
        "host_execution": redact_observability_payload(event.host_execution),
        "metadata": redact_observability_payload(event.metadata),
        "trace_fields": build_trace_fields_for_depot_flow_event(event),
        "failure": build_failure_summary_from_result(event.final_result),
        "depot_flow": {
            "status": event.status.value,
            "flow_type": event.flow_type,
            "depot_id": event.depot_id,
            "asset_id": event.asset_id,
            "branch_id": event.branch_id,
            "release_tag": event.release_tag,
            "artifact_refs": redact_artifact_refs(event.artifact_refs),
            "final_result": {
                "operation_id": event.final_result.operation_id,
                "operation_type": event.final_result.operation_type.value,
                "status": event.final_result.status.value,
                "error_code": event.final_result.error_code,
                "error_message": event.final_result.error_message,
            },
            "steps": [redact_observability_payload(step.to_dict()) for step in event.steps],
        },
    }


@dataclass(slots=True)
class StructuredLogEmitter:
    """Structured JSON logger for shared runtime observability."""

    logger_name: str = "truthound.orchestration.depot"
    log_level: str = "INFO"
    emitted_events: list[dict[str, Any]] = field(default_factory=list)

    def preflight_check(self) -> tuple[bool, str]:
        return True, "structured logger emitter is ready"

    def emit(self, event: Any) -> None:
        payload = redact_observability_payload(event.to_dict()) if hasattr(event, "to_dict") else {}
        self._record(payload)

    def emit_depot(self, event: Any) -> None:
        self._record(build_structured_log_payload(event))

    def emit_depot_flow(self, event: Any) -> None:
        self._record(build_structured_log_payload(event))

    def flush(self) -> None:
        return None

    def _record(self, payload: dict[str, Any]) -> None:
        self.emitted_events.append(payload)
        logger = logging.getLogger(self.logger_name)
        logger.log(getattr(logging, self.log_level.upper(), logging.INFO), json.dumps(payload, sort_keys=True))


@dataclass(slots=True)
class CompositeEmitter:
    """Fan-out emitter that forwards events to multiple concrete emitters."""

    emitters: tuple[Any, ...]

    def preflight_check(self) -> tuple[bool, str]:
        for emitter in self.emitters:
            ready, message = emitter.preflight_check()
            if not ready:
                return False, message
        return True, "composite observability emitter is ready"

    def emit(self, event: Any) -> None:
        for emitter in self.emitters:
            if hasattr(emitter, "emit"):
                emitter.emit(event)

    def emit_depot(self, event: Any) -> None:
        for emitter in self.emitters:
            if hasattr(emitter, "emit_depot"):
                emitter.emit_depot(event)

    def emit_depot_flow(self, event: Any) -> None:
        for emitter in self.emitters:
            if hasattr(emitter, "emit_depot_flow"):
                emitter.emit_depot_flow(event)

    def flush(self) -> None:
        for emitter in self.emitters:
            emitter.flush()
