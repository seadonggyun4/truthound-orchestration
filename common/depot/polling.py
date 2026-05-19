"""Polling helpers for Depot operation status transitions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from common.depot.failures import DepotPollingTimeoutError
from common.depot.models import DepotOperationResult, DepotOperationStatus


class OperationReader(Protocol):
    """Minimal protocol for reading Depot operations."""

    def get_operation(self, operation_id: str) -> DepotOperationResult:
        ...


StatusClass = str


@dataclass(frozen=True, slots=True)
class PollingConfig:
    """Policy controlling Depot operation polling."""

    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 300.0
    max_polls: int = 150
    wait_statuses: tuple[DepotOperationStatus, ...] = field(
        default_factory=lambda: (
            DepotOperationStatus.PENDING,
            DepotOperationStatus.RUNNING,
            DepotOperationStatus.WAITING,
        )
    )
    terminal_statuses: tuple[DepotOperationStatus, ...] = field(
        default_factory=lambda: (
            DepotOperationStatus.SUCCEEDED,
            DepotOperationStatus.FAILED,
            DepotOperationStatus.CANCELLED,
            DepotOperationStatus.NO_OP,
        )
    )


def is_terminal_status(status: DepotOperationStatus | str) -> bool:
    normalized = status if isinstance(status, DepotOperationStatus) else DepotOperationStatus(str(status))
    return normalized in {
        DepotOperationStatus.SUCCEEDED,
        DepotOperationStatus.FAILED,
        DepotOperationStatus.CANCELLED,
        DepotOperationStatus.NO_OP,
    }


def is_waiting_status(status: DepotOperationStatus | str) -> bool:
    normalized = status if isinstance(status, DepotOperationStatus) else DepotOperationStatus(str(status))
    return normalized in {
        DepotOperationStatus.PENDING,
        DepotOperationStatus.RUNNING,
        DepotOperationStatus.WAITING,
    }


def classify_status(status: DepotOperationStatus | str) -> StatusClass:
    return "terminal" if is_terminal_status(status) else "wait"


def wait_for_terminal_operation(
    client: OperationReader,
    operation_id: str,
    config: PollingConfig | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> DepotOperationResult:
    """Poll Depot until the operation reaches a terminal state."""

    effective = config or PollingConfig()
    started = monotonic()
    polls = 0

    while True:
        result = client.get_operation(operation_id)
        polls += 1
        if is_terminal_status(result.status):
            return result
        if polls >= effective.max_polls or (monotonic() - started) >= effective.timeout_seconds:
            raise DepotPollingTimeoutError(
                "Depot operation polling timed out",
                details={
                    "operation_id": operation_id,
                    "polls": polls,
                    "timeout_seconds": effective.timeout_seconds,
                    "last_status": result.status.value,
                },
            )
        sleep(effective.poll_interval_seconds)
