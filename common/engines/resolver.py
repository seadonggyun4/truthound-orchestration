"""Shared engine creation and preflight helpers.

Platform integrations should depend on this module instead of constructing
engine classes directly. It keeps first-party Truthound 3.x defaults,
plugin/runtime discovery, and compatibility checks centralized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
from typing import Any, Callable

from common.engines.base import DataQualityEngine
from common.runtime import (
    AutoConfigPolicy,
    CheckStatus,
    CompatibilityCheck,
    CompatibilityReport,
    PlatformRuntimeContext,
    PreflightReport,
    ResolvedDataSource,
    normalize_auto_config_policy,
    normalize_runtime_context,
    resolve_data_source,
)


@dataclass(frozen=True, slots=True)
class EngineCreationRequest:
    """Structured engine creation request for platform integrations."""

    engine_name: str | None = None
    config: Any | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    runtime_context: PlatformRuntimeContext | None = None
    auto_config_policy: AutoConfigPolicy = AutoConfigPolicy.SAFE_AUTO


@dataclass(frozen=True, slots=True)
class EngineFactorySpec:
    """Factory spec used for built-in and plugin-based engine creation."""

    name: str
    aliases: tuple[str, ...]
    description: str
    factory: Callable[[EngineCreationRequest], DataQualityEngine]
    source: str = "builtin"
    default: bool = False


def _create_truthound_engine(request: EngineCreationRequest) -> DataQualityEngine:
    from common.engines.truthound import TruthoundEngine, TruthoundEngineConfig

    overrides = dict(request.config_overrides)
    if isinstance(request.config, TruthoundEngineConfig):
        config = request.config
        if overrides:
            config = config._copy_with(**overrides)  # type: ignore[attr-defined]
    else:
        config_kwargs = dict(overrides)
        if request.runtime_context is not None:
            policy = normalize_auto_config_policy(request.auto_config_policy)
            if policy == AutoConfigPolicy.SAFE_AUTO:
                config_kwargs.setdefault("context_mode", "ephemeral")
                config_kwargs.setdefault("persist_runs", False)
                config_kwargs.setdefault("persist_docs", False)
                config_kwargs.setdefault("auto_create_workspace", False)
            if request.runtime_context.project_root is not None:
                config_kwargs.setdefault("workspace_root", request.runtime_context.project_root)
        config = TruthoundEngineConfig(**config_kwargs)
    return TruthoundEngine(config=config)


def _create_great_expectations_engine(request: EngineCreationRequest) -> DataQualityEngine:
    from common.engines.great_expectations import (
        GreatExpectationsAdapter,
        GreatExpectationsConfig,
    )

    if isinstance(request.config, GreatExpectationsConfig):
        config = request.config
    else:
        config_kwargs = dict(request.config_overrides)
        config = GreatExpectationsConfig(**config_kwargs) if config_kwargs else request.config
    return GreatExpectationsAdapter(config=config)


def _create_pandera_engine(request: EngineCreationRequest) -> DataQualityEngine:
    from common.engines.pandera import PanderaAdapter, PanderaConfig

    if isinstance(request.config, PanderaConfig):
        config = request.config
    else:
        config_kwargs = dict(request.config_overrides)
        config = PanderaConfig(**config_kwargs) if config_kwargs else request.config
    return PanderaAdapter(config=config)


BUILTIN_ENGINE_SPECS: dict[str, EngineFactorySpec] = {
    "truthound": EngineFactorySpec(
        name="truthound",
        aliases=("truthound3",),
        description="Official Truthound 3.x orchestration runtime",
        factory=_create_truthound_engine,
        default=True,
    ),
    "great_expectations": EngineFactorySpec(
        name="great_expectations",
        aliases=("ge",),
        description="Advanced-tier Great Expectations adapter",
        factory=_create_great_expectations_engine,
    ),
    "pandera": EngineFactorySpec(
        name="pandera",
        aliases=(),
        description="Advanced-tier Pandera adapter",
        factory=_create_pandera_engine,
    ),
}


ENGINE_ALIASES: dict[str, str] = {
    alias: spec.name
    for spec in BUILTIN_ENGINE_SPECS.values()
    for alias in (spec.name, *spec.aliases)
}


def normalize_engine_name(name: str | None) -> str:
    """Normalize engine names and supported aliases."""

    normalized = (name or "truthound").strip().lower()
    return ENGINE_ALIASES.get(normalized, normalized)


def resolve_builtin_engine_spec(name: str | None) -> EngineFactorySpec | None:
    """Resolve a built-in engine factory spec by name or alias."""

    normalized = normalize_engine_name(name)
    return BUILTIN_ENGINE_SPECS.get(normalized)


def create_builtin_engine(
    name: str | None = None,
    **config_overrides: Any,
) -> DataQualityEngine:
    """Create a built-in engine instance."""

    request = _coerce_request(name, config_overrides)
    spec = resolve_builtin_engine_spec(request.engine_name)
    if spec is None:
        from common.engines.registry import EngineNotFoundError

        raise EngineNotFoundError(normalize_engine_name(name))
    return spec.factory(request)


def create_engine(
    request_or_name: EngineCreationRequest | str | None = None,
    **config_overrides: Any,
) -> DataQualityEngine:
    """Create a new engine instance.

    The function accepts either a simple engine name or a structured
    ``EngineCreationRequest`` carrying runtime context and zero-config policy.
    """

    request = _coerce_request(request_or_name, config_overrides)
    spec = resolve_builtin_engine_spec(request.engine_name)
    if spec is not None:
        return spec.factory(request)

    normalized = normalize_engine_name(request.engine_name)
    plugin_engine = _create_plugin_engine(normalized, request)
    if plugin_engine is not None:
        return plugin_engine

    from common.engines.registry import EngineNotFoundError, get_engine

    try:
        return get_engine(normalized)
    except EngineNotFoundError:
        raise


def build_compatibility_report(
    request_or_engine: EngineCreationRequest | DataQualityEngine | str | None = None,
    *,
    runtime_context: PlatformRuntimeContext | None = None,
    resolved_source: ResolvedDataSource | None = None,
    serializer: str = "shared_wire",
) -> CompatibilityReport:
    """Build a compatibility report for the requested engine/runtime pair."""

    context = _resolve_runtime_context(request_or_engine, runtime_context)
    checks: list[CompatibilityCheck] = []
    engine_name = "truthound"
    engine_version: str | None = None

    if context.host_version is not None:
        checks.append(
            CompatibilityCheck(
                name="host_runtime",
                status=CheckStatus.PASSED,
                message=f"{context.platform} runtime metadata provided",
                details={"host_version": context.host_version},
            )
        )

    try:
        if isinstance(request_or_engine, DataQualityEngine):
            engine = request_or_engine
        else:
            engine = create_engine(
                request_or_engine
                if isinstance(request_or_engine, (EngineCreationRequest, str))
                else None,
                runtime_context=context,
            )
        engine_name = engine.engine_name
        engine_version = engine.engine_version
        checks.append(
            CompatibilityCheck(
                name="engine_runtime",
                status=CheckStatus.PASSED,
                message=f"Resolved {engine_name} runtime",
                details={"engine_version": engine_version},
            )
        )
    except Exception as exc:
        checks.append(
            CompatibilityCheck(
                name="engine_runtime",
                status=CheckStatus.FAILED,
                message=f"Failed to resolve engine runtime: {exc}",
                details={"error_type": type(exc).__name__},
            )
        )
        return CompatibilityReport(
            engine_name=engine_name,
            engine_version=engine_version,
            platform=context.platform,
            compatible=False,
            checks=tuple(checks),
            auto_config_policy=context.auto_config_policy,
        )

    if resolved_source is not None:
        if resolved_source.requires_connection and not (
            context.connection_id or context.profile_name
        ):
            checks.append(
                CompatibilityCheck(
                    name="source_resolution",
                    status=CheckStatus.FAILED,
                    message=(
                        f"{resolved_source.kind.value} sources require a connection or profile"
                    ),
                    details=resolved_source.to_dict(),
                )
            )
        else:
            checks.append(
                CompatibilityCheck(
                    name="source_resolution",
                    status=CheckStatus.PASSED,
                    message=f"Resolved {resolved_source.kind.value} source",
                    details=resolved_source.to_dict(),
                )
            )

    try:
        from common.serializers import serialize_result_wire  # noqa: F401

        checks.append(
            CompatibilityCheck(
                name="serializer",
                status=CheckStatus.PASSED,
                message=f"Shared serializer '{serializer}' is available",
            )
        )
    except Exception as exc:
        checks.append(
            CompatibilityCheck(
                name="serializer",
                status=CheckStatus.FAILED,
                message=f"Serializer '{serializer}' is unavailable: {exc}",
                details={"error_type": type(exc).__name__},
            )
        )

    if engine_name == "truthound" and context.auto_config_policy == AutoConfigPolicy.SAFE_AUTO:
        checks.append(
            CompatibilityCheck(
                name="zero_config",
                status=CheckStatus.PASSED,
                message="Truthound safe_auto defaults to ephemeral zero-config runtime",
                details={"context_mode": "ephemeral", "persist_runs": False, "persist_docs": False},
            )
        )

    compatible = all(check.status != CheckStatus.FAILED for check in checks)
    return CompatibilityReport(
        engine_name=engine_name,
        engine_version=engine_version,
        platform=context.platform,
        compatible=compatible,
        checks=tuple(checks),
        auto_config_policy=context.auto_config_policy,
    )


def run_preflight(
    request_or_engine: EngineCreationRequest | DataQualityEngine | str | None = None,
    *,
    runtime_context: PlatformRuntimeContext | None = None,
    serializer: str = "shared_wire",
    data: Any | None = None,
    data_path: str | None = None,
    sql: str | None = None,
    source_factory: Any | None = None,
) -> PreflightReport:
    """Run a shared preflight/compatibility pass before execution."""

    resolved_source = resolve_data_source(
        data,
        data_path=data_path,
        sql=sql,
        source_factory=source_factory,
    )
    context = _resolve_runtime_context(request_or_engine, runtime_context)
    if resolved_source is not None:
        context = context.with_source(resolved_source)

    compatibility = build_compatibility_report(
        request_or_engine,
        runtime_context=context,
        resolved_source=resolved_source,
        serializer=serializer,
    )
    return PreflightReport(
        compatibility=compatibility,
        resolved_source=resolved_source,
        serializer=serializer,
        metadata={"runtime_context": context.to_dict()},
    )


def _coerce_request(
    request_or_name: EngineCreationRequest | str | None,
    config_overrides: dict[str, Any],
) -> EngineCreationRequest:
    overrides = dict(config_overrides)
    runtime_context = overrides.pop("runtime_context", None)
    auto_config_policy = overrides.pop("auto_config_policy", None)
    config = overrides.pop("config", None)

    if isinstance(request_or_name, EngineCreationRequest):
        merged_overrides = {**request_or_name.config_overrides, **overrides}
        merged_context = (
            runtime_context
            if isinstance(runtime_context, PlatformRuntimeContext)
            else request_or_name.runtime_context
        )
        return EngineCreationRequest(
            engine_name=normalize_engine_name(request_or_name.engine_name),
            config=request_or_name.config if config is None else config,
            config_overrides=merged_overrides,
            runtime_context=merged_context,
            auto_config_policy=normalize_auto_config_policy(
                auto_config_policy or request_or_name.auto_config_policy
            ),
        )

    return EngineCreationRequest(
        engine_name=normalize_engine_name(request_or_name),
        config=config,
        config_overrides=overrides,
        runtime_context=runtime_context if isinstance(runtime_context, PlatformRuntimeContext) else None,
        auto_config_policy=normalize_auto_config_policy(auto_config_policy),
    )


def _resolve_runtime_context(
    request_or_engine: EngineCreationRequest | DataQualityEngine | str | None,
    runtime_context: PlatformRuntimeContext | None,
) -> PlatformRuntimeContext:
    if isinstance(request_or_engine, EngineCreationRequest):
        return normalize_runtime_context(
            request_or_engine.runtime_context,
            platform=runtime_context.platform if runtime_context is not None else None,
            auto_config_policy=(
                runtime_context.auto_config_policy if runtime_context is not None else None
            ),
            host_version=runtime_context.host_version if runtime_context is not None else None,
            connection_id=runtime_context.connection_id if runtime_context is not None else None,
            profile_name=runtime_context.profile_name if runtime_context is not None else None,
            source_name=runtime_context.source_name if runtime_context is not None else None,
            project_root=runtime_context.project_root if runtime_context is not None else None,
            host_metadata=runtime_context.host_metadata if runtime_context is not None else None,
            source_descriptors=(
                runtime_context.source_descriptors if runtime_context is not None else None
            ),
            extras=runtime_context.extras if runtime_context is not None else None,
        )
    if runtime_context is not None:
        return normalize_runtime_context(runtime_context)
    return normalize_runtime_context(platform="common")


def _create_plugin_engine(
    normalized_name: str,
    request: EngineCreationRequest,
) -> DataQualityEngine | None:
    try:
        entry_points = importlib.metadata.entry_points(group="truthound.engines")
    except TypeError:
        entry_points = importlib.metadata.entry_points().select(group="truthound.engines")

    for entry_point in entry_points:
        if entry_point.name != normalized_name:
            continue
        engine_class = entry_point.load()
        return _instantiate_engine_class(engine_class, request)
    return None


def _instantiate_engine_class(
    engine_class: type[Any],
    request: EngineCreationRequest,
) -> DataQualityEngine:
    config = request.config
    overrides = dict(request.config_overrides)

    if config is not None:
        try:
            return engine_class(config=config)
        except TypeError:
            return engine_class(config=config, **overrides)

    try:
        return engine_class(**overrides)
    except TypeError:
        return engine_class()
