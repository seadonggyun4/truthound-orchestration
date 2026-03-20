"""Runtime helpers for the Truthound 3.x engine adapter.

This module isolates runtime-specific concerns from the public engine wrapper:
- importing Truthound lazily
- resolving and validating the installed Truthound version
- creating explicit Truthound contexts so orchestration behavior is deterministic
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import inspect
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

from common.exceptions import ValidationExecutionError
from common.engines.version import require_version


if TYPE_CHECKING:
    from collections.abc import Iterator


TRUTHOUND_VERSION_REQUIREMENT = ">=3.0.0,<4.0.0"
TRUTHOUND_CONTEXT_MODES = frozenset({"ephemeral", "project", "provided"})


@dataclass(frozen=True, slots=True)
class _TruthoundVersionProbe:
    """Minimal engine-like object for shared version compatibility checks."""

    version: str

    @property
    def engine_name(self) -> str:
        return "truthound"

    @property
    def engine_version(self) -> str:
        return self.version


def resolve_truthound_version(truthound_module: Any | None = None) -> str:
    """Resolve the installed Truthound version.

    Uses package metadata first so installed wheels/sdists are detected
    consistently, then falls back to the module attribute for editable/local
    source use.
    """

    try:
        return importlib.metadata.version("truthound")
    except importlib.metadata.PackageNotFoundError:
        module = truthound_module
        if module is None:
            try:
                import truthound as module
            except ImportError as e:
                raise ValidationExecutionError(
                    "Truthound library is not installed. "
                    "Install it with: pip install 'truthound>=3.0,<4.0'",
                    cause=e,
                ) from e

        version = getattr(module, "__version__", None)
        if version is None:
            raise ValidationExecutionError(
                "Truthound is installed but its version could not be determined."
            )
        return str(version)


def ensure_truthound_runtime() -> Any:
    """Import Truthound and enforce the supported 3.x version range."""

    try:
        import truthound
    except ImportError as e:
        raise ValidationExecutionError(
            "Truthound library is not installed. "
            "Install it with: pip install 'truthound>=3.0,<4.0'",
            cause=e,
        ) from e

    version = resolve_truthound_version(truthound)
    require_version(_TruthoundVersionProbe(version), TRUTHOUND_VERSION_REQUIREMENT)
    return truthound


def validate_context_mode(mode: str) -> str:
    """Validate and normalize a Truthound context mode."""

    normalized = mode.lower()
    if normalized not in TRUTHOUND_CONTEXT_MODES:
        valid = ", ".join(sorted(TRUTHOUND_CONTEXT_MODES))
        raise ValueError(f"context_mode must be one of {{{valid}}}, got '{mode}'")
    return normalized


def _merge_project_context_config(
    *,
    project_config: Any,
    persist_runs: bool,
    persist_docs: bool,
    auto_create_workspace: bool,
) -> Any:
    """Create a TruthoundContextConfig using project defaults plus orchestration overrides."""

    from truthound.context import TruthoundContextConfig

    return TruthoundContextConfig(
        persist_runs=persist_runs,
        persist_docs=persist_docs,
        auto_create_baseline=getattr(project_config, "auto_create_baseline", True),
        auto_create_workspace=auto_create_workspace,
        default_result_format=getattr(project_config, "default_result_format", "summary"),
        docs_theme=getattr(project_config, "docs_theme", "professional"),
    )


@contextmanager
def prepare_truthound_call(
    engine_config: Any,
    call_kwargs: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Normalize call kwargs and inject an explicit Truthound context when needed."""

    normalized = dict(call_kwargs)
    normalized.pop("context_mode", None)
    normalized.pop("workspace_root", None)
    normalized.pop("persist_runs", None)
    normalized.pop("persist_docs", None)
    normalized.pop("auto_create_workspace", None)
    explicit_context = normalized.get("context")
    if explicit_context is not None:
        yield normalized
        return

    context_mode = validate_context_mode(
        str(call_kwargs.get("context_mode", engine_config.context_mode))
    )
    workspace_root = call_kwargs.get("workspace_root", engine_config.workspace_root)
    persist_runs = bool(call_kwargs.get("persist_runs", engine_config.persist_runs))
    persist_docs = bool(call_kwargs.get("persist_docs", engine_config.persist_docs))
    auto_create_workspace = bool(
        call_kwargs.get("auto_create_workspace", engine_config.auto_create_workspace)
    )

    if context_mode == "provided":
        raise ValidationExecutionError(
            "Truthound context_mode='provided' requires an explicit context argument."
        )

    from truthound.context import TruthoundContext, TruthoundContextConfig, get_context

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if context_mode == "project":
            project_start = Path(workspace_root) if workspace_root is not None else Path.cwd()
            project_context = get_context(project_start)
            context_config = _merge_project_context_config(
                project_config=project_context.config,
                persist_runs=persist_runs,
                persist_docs=persist_docs,
                auto_create_workspace=auto_create_workspace,
            )
            resolved_context = TruthoundContext(
                root_dir=project_context.root_dir,
                config=context_config,
            )
        else:
            if workspace_root is None:
                temp_dir = tempfile.TemporaryDirectory(prefix="truthound-orchestration-")
                root_dir = Path(temp_dir.name)
            else:
                root_dir = Path(workspace_root)

            resolved_context = TruthoundContext(
                root_dir=root_dir,
                config=TruthoundContextConfig(
                    persist_runs=persist_runs,
                    persist_docs=persist_docs,
                    auto_create_workspace=auto_create_workspace,
                ),
            )

        normalized["context"] = resolved_context
        yield normalized
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def filter_supported_truthound_kwargs(
    truthound_callable: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Drop kwargs that the installed Truthound callable does not accept.

    Platform adapters expose richer generic options than the Truthound 3.x
    runtime currently accepts. We keep those platform interfaces stable, but
    only forward the subset that the installed Truthound callable supports.
    """

    try:
        signature = inspect.signature(truthound_callable)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)

    supported = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in kwargs.items() if key in supported}
