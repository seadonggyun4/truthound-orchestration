"""Shared runtime contracts for first-party orchestration integrations.

This module centralizes the runtime-facing primitives that platform packages
use to describe host context, zero-config policy, normalized input sources,
and compatibility/preflight reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class AutoConfigPolicy(str, Enum):
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


class ObservabilityBackend(str, Enum):
    """Supported shared observability backends."""

    NONE = "none"
    OPENLINEAGE = "openlineage"


class DataSourceKind(str, Enum):
    """Canonical source kinds understood by the shared runtime."""

    DATAFRAME = "dataframe"
    LOCAL_PATH = "local_path"
    REMOTE_URI = "remote_uri"
    SQL = "sql"
    SYNC_STREAM = "sync_stream"
    ASYNC_STREAM = "async_stream"
    CALLABLE = "callable"
    OBJECT = "object"


class CheckStatus(str, Enum):
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
        try:
            metadata["column_count"] = len(columns)
        except TypeError:
            pass
    height = getattr(value, "height", None)
    if height is not None:
        metadata["row_count"] = int(height)
    elif hasattr(value, "__len__"):
        try:
            metadata["row_count"] = len(value)
        except TypeError:
            pass
    return metadata
