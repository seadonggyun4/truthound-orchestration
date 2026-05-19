"""Dagster resource for Depot operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.depot.client import DepotClient, DepotClientConfig
from common.depot.polling import PollingConfig
from common.orchestration import DepotOperationClient, execute_depot_flow, execute_depot_operation
from common.runtime import DepotFlowRequest, normalize_runtime_context

from truthound_dagster.resources.base import (
    BaseResource,
    ResourceConfig,
)


@dataclass(frozen=True, slots=True)
class DepotResourceConfig(ResourceConfig):
    base_url: str = ""
    api_token: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    wait: bool = False
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 300.0


class DepotResource(BaseResource[DepotResourceConfig]):
    """Dagster-native wrapper over the shared Depot runtime facade."""

    def __init__(
        self,
        config: DepotResourceConfig | None = None,
        client: DepotOperationClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client = client

    @classmethod
    def _default_config(cls) -> DepotResourceConfig:
        return DepotResourceConfig()

    def resolve_client(self) -> DepotOperationClient:
        if self._client is not None:
            return self._client
        if not self.config.base_url or not self.config.api_token:
            raise ValueError("DepotResource requires either a client or base_url/api_token")
        return DepotClient(
            DepotClientConfig(
                base_url=self.config.base_url,
                api_token=self.config.api_token,
                timeout_seconds=self.config.timeout_seconds,
                headers=self.config.headers,
            )
        )

    def execute(
        self,
        request: Any,
        *,
        host_execution: dict[str, Any] | None = None,
        host_metadata: dict[str, Any] | None = None,
        wait: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        runtime_context = normalize_runtime_context(
            platform="dagster",
            host_execution=host_execution or {},
            host_metadata=host_metadata or {"resource_type": type(self).__name__},
        )
        return execute_depot_operation(
            request,
            runtime_context=runtime_context,
            client=self.resolve_client(),
            wait=self.config.wait if wait is None else wait,
            polling=PollingConfig(
                poll_interval_seconds=self.config.poll_interval_seconds,
                timeout_seconds=self.config.poll_timeout_seconds,
            ),
            metadata=metadata,
        )

    def execute_flow(
        self,
        request: DepotFlowRequest,
        *,
        wait: bool | None = None,
    ) -> Any:
        return execute_depot_flow(
            request,
            client=self.resolve_client(),
            wait=self.config.wait if wait is None else wait,
            polling=PollingConfig(
                poll_interval_seconds=self.config.poll_interval_seconds,
                timeout_seconds=self.config.poll_timeout_seconds,
            ),
        )


__all__ = ["DepotResource", "DepotResourceConfig"]
