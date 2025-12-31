"""Custom sampling strategies for data quality operations.

This module provides specialized samplers for data quality telemetry,
allowing intelligent sampling decisions based on operation characteristics.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Sequence

from common.opentelemetry.config import SamplingConfig
from common.opentelemetry.exceptions import OTelNotInstalledError, OTelSamplingError
from common.opentelemetry.types import SamplingDecision

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import Link, SpanKind
    from opentelemetry.trace.span import TraceState
    from opentelemetry.util.types import Attributes

__all__ = [
    "DataQualitySampler",
    "create_sampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "RatioBasedSampler",
    "ParentBasedSampler",
]

logger = logging.getLogger(__name__)


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.sdk.trace.sampling  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="sampling") from e


class AlwaysOnSampler:
    """Sampler that always samples.

    This sampler records and exports all spans, useful for development
    and debugging.
    """

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> tuple[SamplingDecision, dict[str, Any]]:
        """Determine if a span should be sampled.

        Returns:
            Tuple of (decision, attributes).
        """
        return SamplingDecision.RECORD_AND_SAMPLE, {}

    def get_description(self) -> str:
        """Get sampler description."""
        return "AlwaysOnSampler"


class AlwaysOffSampler:
    """Sampler that never samples.

    This sampler drops all spans, useful for disabling tracing.
    """

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> tuple[SamplingDecision, dict[str, Any]]:
        """Determine if a span should be sampled.

        Returns:
            Tuple of (decision, attributes).
        """
        return SamplingDecision.DROP, {}

    def get_description(self) -> str:
        """Get sampler description."""
        return "AlwaysOffSampler"


class RatioBasedSampler:
    """Sampler based on a probability ratio.

    Samples a percentage of traces based on the trace ID.
    """

    def __init__(self, ratio: float = 1.0) -> None:
        """Initialize RatioBasedSampler.

        Args:
            ratio: Sampling ratio (0.0 to 1.0).
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")
        self._ratio = ratio
        self._bound = int(ratio * (2**63 - 1))

    @property
    def ratio(self) -> float:
        """Get the sampling ratio."""
        return self._ratio

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> tuple[SamplingDecision, dict[str, Any]]:
        """Determine if a span should be sampled.

        Uses the lower 63 bits of the trace ID for deterministic sampling.

        Returns:
            Tuple of (decision, attributes).
        """
        # Use lower 63 bits for deterministic sampling
        if (trace_id & 0x7FFFFFFFFFFFFFFF) < self._bound:
            return SamplingDecision.RECORD_AND_SAMPLE, {}
        return SamplingDecision.DROP, {}

    def get_description(self) -> str:
        """Get sampler description."""
        return f"RatioBasedSampler{{ratio={self._ratio}}}"


class ParentBasedSampler:
    """Sampler that respects parent sampling decisions.

    Uses the parent span's sampling decision when available,
    otherwise falls back to a root sampler.
    """

    def __init__(
        self,
        root_sampler: Any | None = None,
        remote_parent_sampled: Any | None = None,
        remote_parent_not_sampled: Any | None = None,
        local_parent_sampled: Any | None = None,
        local_parent_not_sampled: Any | None = None,
    ) -> None:
        """Initialize ParentBasedSampler.

        Args:
            root_sampler: Sampler for root spans (no parent).
            remote_parent_sampled: Sampler when remote parent is sampled.
            remote_parent_not_sampled: Sampler when remote parent is not sampled.
            local_parent_sampled: Sampler when local parent is sampled.
            local_parent_not_sampled: Sampler when local parent is not sampled.
        """
        self._root_sampler = root_sampler or AlwaysOnSampler()
        self._remote_parent_sampled = remote_parent_sampled or AlwaysOnSampler()
        self._remote_parent_not_sampled = remote_parent_not_sampled or AlwaysOffSampler()
        self._local_parent_sampled = local_parent_sampled or AlwaysOnSampler()
        self._local_parent_not_sampled = local_parent_not_sampled or AlwaysOffSampler()

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> tuple[SamplingDecision, dict[str, Any]]:
        """Determine if a span should be sampled based on parent.

        Returns:
            Tuple of (decision, attributes).
        """
        _check_otel_installed()

        from opentelemetry import trace

        # Check for parent span
        if parent_context is None:
            return self._root_sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )

        parent_span = trace.get_current_span(parent_context)
        parent_span_context = parent_span.get_span_context()

        if not parent_span_context.is_valid:
            return self._root_sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )

        if parent_span_context.is_remote:
            if parent_span_context.trace_flags.sampled:
                return self._remote_parent_sampled.should_sample(
                    parent_context, trace_id, name, kind, attributes, links, trace_state
                )
            return self._remote_parent_not_sampled.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )
        else:
            if parent_span_context.trace_flags.sampled:
                return self._local_parent_sampled.should_sample(
                    parent_context, trace_id, name, kind, attributes, links, trace_state
                )
            return self._local_parent_not_sampled.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )

    def get_description(self) -> str:
        """Get sampler description."""
        return f"ParentBasedSampler{{root={self._root_sampler.get_description()}}}"


class DataQualitySampler:
    """Specialized sampler for data quality operations.

    This sampler implements intelligent sampling decisions based on
    operation characteristics:

    - Always sample error/failure spans
    - Sample slow operations above threshold
    - Apply base rate sampling for normal operations
    - Support per-operation-type sampling rates

    Example:
        config = SamplingConfig(
            sample_rate=0.1,
            error_sample_rate=1.0,
            slow_operation_threshold_ms=500.0,
        )
        sampler = DataQualitySampler(config)
    """

    # Attribute keys for sampling decisions
    ATTR_OPERATION_TYPE = "dq.operation.type"
    ATTR_HAS_ERROR = "dq.has_error"
    ATTR_DURATION_MS = "dq.duration_ms"
    ATTR_CHECK_STATUS = "dq.check.status"

    def __init__(
        self,
        config: SamplingConfig | None = None,
        operation_rates: dict[str, float] | None = None,
    ) -> None:
        """Initialize DataQualitySampler.

        Args:
            config: Sampling configuration.
            operation_rates: Per-operation sampling rates.
                Keys: operation types (check, profile, learn)
                Values: sampling rates (0.0 to 1.0)
        """
        self._config = config or SamplingConfig()
        self._operation_rates = operation_rates or {}
        self._base_sampler = RatioBasedSampler(self._config.sample_rate)
        self._error_sampler = RatioBasedSampler(self._config.error_sample_rate)
        self._slow_sampler = RatioBasedSampler(self._config.slow_operation_sample_rate)

    @property
    def config(self) -> SamplingConfig:
        """Get the sampling configuration."""
        return self._config

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes | None = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> tuple[SamplingDecision, dict[str, Any]]:
        """Determine if a span should be sampled.

        Applies intelligent sampling based on operation characteristics.

        Returns:
            Tuple of (decision, attributes).
        """
        if not self._config.enabled:
            return SamplingDecision.DROP, {}

        attrs = dict(attributes) if attributes else {}

        # Check for error condition
        if self._is_error_span(attrs):
            decision, extra_attrs = self._error_sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )
            extra_attrs["sampled.reason"] = "error"
            return decision, extra_attrs

        # Check for slow operation
        if self._is_slow_operation(attrs):
            decision, extra_attrs = self._slow_sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )
            extra_attrs["sampled.reason"] = "slow_operation"
            return decision, extra_attrs

        # Check for operation-specific rate
        operation_type = attrs.get(self.ATTR_OPERATION_TYPE)
        if operation_type and operation_type in self._operation_rates:
            rate = self._operation_rates[operation_type]
            op_sampler = RatioBasedSampler(rate)
            decision, extra_attrs = op_sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links, trace_state
            )
            extra_attrs["sampled.reason"] = f"operation_rate:{operation_type}"
            return decision, extra_attrs

        # Apply parent-based sampling if configured
        if self._config.parent_based and parent_context is not None:
            _check_otel_installed()
            from opentelemetry import trace

            parent_span = trace.get_current_span(parent_context)
            parent_span_context = parent_span.get_span_context()

            if parent_span_context.is_valid:
                if parent_span_context.trace_flags.sampled:
                    return SamplingDecision.RECORD_AND_SAMPLE, {"sampled.reason": "parent"}
                else:
                    return SamplingDecision.DROP, {"sampled.reason": "parent_not_sampled"}

        # Apply base rate sampling
        decision, extra_attrs = self._base_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )
        extra_attrs["sampled.reason"] = "base_rate"
        return decision, extra_attrs

    def _is_error_span(self, attributes: dict[str, Any]) -> bool:
        """Check if span represents an error.

        Args:
            attributes: Span attributes.

        Returns:
            True if span represents an error.
        """
        # Check explicit error flag
        if attributes.get(self.ATTR_HAS_ERROR):
            return True

        # Check check status
        check_status = attributes.get(self.ATTR_CHECK_STATUS)
        if check_status in ("error", "failed", "ERROR", "FAILED"):
            return True

        return False

    def _is_slow_operation(self, attributes: dict[str, Any]) -> bool:
        """Check if span represents a slow operation.

        Args:
            attributes: Span attributes.

        Returns:
            True if operation exceeds slow threshold.
        """
        duration_ms = attributes.get(self.ATTR_DURATION_MS)
        if duration_ms is None:
            return False

        try:
            return float(duration_ms) > self._config.slow_operation_threshold_ms
        except (ValueError, TypeError):
            return False

    def get_description(self) -> str:
        """Get sampler description."""
        return (
            f"DataQualitySampler{{"
            f"base_rate={self._config.sample_rate}, "
            f"error_rate={self._config.error_sample_rate}, "
            f"slow_threshold={self._config.slow_operation_threshold_ms}ms"
            f"}}"
        )


def create_sampler(
    config: SamplingConfig | None = None,
    operation_rates: dict[str, float] | None = None,
) -> DataQualitySampler:
    """Create a data quality sampler from configuration.

    Factory function for creating a DataQualitySampler.

    Args:
        config: Sampling configuration.
        operation_rates: Per-operation sampling rates.

    Returns:
        Configured DataQualitySampler.

    Example:
        sampler = create_sampler(
            config=SamplingConfig(sample_rate=0.1),
            operation_rates={"check": 0.5, "profile": 0.1},
        )
    """
    return DataQualitySampler(config=config, operation_rates=operation_rates)
