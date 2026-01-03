"""Type definitions for Mage Data Quality blocks.

This module provides common type definitions and data structures
used across the data quality block implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class BlockMetadata:
    """Metadata about block execution.

    Captures timing, identification, and context information
    for a block execution.

    Attributes:
        block_uuid: Unique identifier for the block.
        pipeline_uuid: UUID of the containing pipeline.
        block_type: Type of block (transformer, sensor, condition).
        operation: Operation performed (check, profile, learn).
        started_at: Execution start timestamp.
        completed_at: Execution completion timestamp.
        duration_ms: Execution duration in milliseconds.
        engine_name: Name of the data quality engine used.
        engine_version: Version of the engine.
        tags: Custom tags for the execution.
        extra: Additional metadata.

    Example:
        >>> metadata = BlockMetadata(
        ...     block_uuid="check_1",
        ...     pipeline_uuid="pipeline_1",
        ...     block_type="transformer",
        ...     operation="check",
        ...     started_at=datetime.now(timezone.utc),
        ... )
    """

    block_uuid: str
    pipeline_uuid: str | None = None
    block_type: Literal["transformer", "sensor", "condition"] = "transformer"
    operation: Literal["check", "profile", "learn"] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None
    engine_name: str | None = None
    engine_version: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    extra: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def with_completion(
        self,
        completed_at: datetime | None = None,
    ) -> BlockMetadata:
        """Create new metadata with completion time.

        Args:
            completed_at: Completion timestamp (defaults to now).

        Returns:
            New metadata with completion time and duration calculated.
        """
        if completed_at is None:
            completed_at = datetime.now(timezone.utc)

        duration_ms = None
        if self.started_at is not None:
            delta = completed_at - self.started_at
            duration_ms = delta.total_seconds() * 1000

        return BlockMetadata(
            block_uuid=self.block_uuid,
            pipeline_uuid=self.pipeline_uuid,
            block_type=self.block_type,
            operation=self.operation,
            started_at=self.started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            tags=self.tags,
            extra=self.extra,
        )

    def with_engine(
        self,
        engine_name: str,
        engine_version: str | None = None,
    ) -> BlockMetadata:
        """Create new metadata with engine information.

        Args:
            engine_name: Name of the engine.
            engine_version: Version of the engine.

        Returns:
            New metadata with engine information.
        """
        return BlockMetadata(
            block_uuid=self.block_uuid,
            pipeline_uuid=self.pipeline_uuid,
            block_type=self.block_type,
            operation=self.operation,
            started_at=self.started_at,
            completed_at=self.completed_at,
            duration_ms=self.duration_ms,
            engine_name=engine_name,
            engine_version=engine_version,
            tags=self.tags,
            extra=self.extra,
        )

    def with_extra(self, **kwargs: Any) -> BlockMetadata:
        """Create new metadata with additional fields.

        Args:
            **kwargs: Additional key-value pairs.

        Returns:
            New metadata with extra fields.
        """
        new_extra = dict(self.extra)
        new_extra.update(kwargs)
        return BlockMetadata(
            block_uuid=self.block_uuid,
            pipeline_uuid=self.pipeline_uuid,
            block_type=self.block_type,
            operation=self.operation,
            started_at=self.started_at,
            completed_at=self.completed_at,
            duration_ms=self.duration_ms,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            tags=self.tags,
            extra=tuple(new_extra.items()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary containing all metadata fields.
        """
        result: dict[str, Any] = {
            "block_uuid": self.block_uuid,
            "pipeline_uuid": self.pipeline_uuid,
            "block_type": self.block_type,
            "operation": self.operation,
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "tags": list(self.tags),
        }

        if self.started_at is not None:
            result["started_at"] = self.started_at.isoformat()
        if self.completed_at is not None:
            result["completed_at"] = self.completed_at.isoformat()
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms

        if self.extra:
            result["extra"] = dict(self.extra)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockMetadata:
        """Create from dictionary representation.

        Args:
            data: Dictionary with metadata fields.

        Returns:
            New BlockMetadata instance.
        """
        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])

        completed_at = None
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"])

        return cls(
            block_uuid=data["block_uuid"],
            pipeline_uuid=data.get("pipeline_uuid"),
            block_type=data.get("block_type", "transformer"),
            operation=data.get("operation"),
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=data.get("duration_ms"),
            engine_name=data.get("engine_name"),
            engine_version=data.get("engine_version"),
            tags=frozenset(data.get("tags", [])),
            extra=tuple(data.get("extra", {}).items()),
        )


@dataclass(frozen=True, slots=True)
class DataQualityOutput:
    """Standardized output for data quality operations.

    Wraps the result from data quality operations with metadata
    and provides a consistent interface for downstream blocks.

    Attributes:
        success: Whether the operation succeeded.
        result: The raw result from the engine.
        metadata: Block execution metadata.
        data: Optional output data (for pass-through).
        summary: Optional summary dictionary.

    Example:
        >>> output = DataQualityOutput(
        ...     success=True,
        ...     result=check_result,
        ...     metadata=metadata,
        ...     summary={"passed": 10, "failed": 2},
        ... )
    """

    success: bool
    result: Any
    metadata: BlockMetadata
    data: Any = None
    summary: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def summary_dict(self) -> dict[str, Any]:
        """Get summary as dictionary."""
        return dict(self.summary)

    def with_data(self, data: Any) -> DataQualityOutput:
        """Create new output with data.

        Args:
            data: Data to include in output.

        Returns:
            New output with data.
        """
        return DataQualityOutput(
            success=self.success,
            result=self.result,
            metadata=self.metadata,
            data=data,
            summary=self.summary,
        )

    def with_summary(self, **kwargs: Any) -> DataQualityOutput:
        """Create new output with additional summary fields.

        Args:
            **kwargs: Summary key-value pairs.

        Returns:
            New output with updated summary.
        """
        new_summary = dict(self.summary)
        new_summary.update(kwargs)
        return DataQualityOutput(
            success=self.success,
            result=self.result,
            metadata=self.metadata,
            data=self.data,
            summary=tuple(new_summary.items()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary containing all output fields.
        """
        result_dict: dict[str, Any] = {
            "success": self.success,
            "metadata": self.metadata.to_dict(),
        }

        # Handle result serialization
        if hasattr(self.result, "to_dict"):
            result_dict["result"] = self.result.to_dict()
        elif hasattr(self.result, "__dict__"):
            result_dict["result"] = self.result.__dict__
        else:
            result_dict["result"] = self.result

        if self.summary:
            result_dict["summary"] = dict(self.summary)

        # Note: data is not serialized as it may be a DataFrame
        result_dict["has_data"] = self.data is not None

        return result_dict

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        result: Any = None,
        output_data: Any = None,
    ) -> DataQualityOutput:
        """Create from dictionary representation.

        Args:
            data: Dictionary with output fields.
            result: The result object (not serialized).
            output_data: The output data (not serialized).

        Returns:
            New DataQualityOutput instance.
        """
        return cls(
            success=data["success"],
            result=result if result is not None else data.get("result"),
            metadata=BlockMetadata.from_dict(data["metadata"]),
            data=output_data,
            summary=tuple(data.get("summary", {}).items()),
        )


# Type aliases for common patterns
DataFrameType = Any  # Polars or Pandas DataFrame
RuleList = tuple[dict[str, Any], ...]
TagSet = frozenset[str]
