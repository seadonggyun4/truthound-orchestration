"""Prefect block for Depot operations."""

from __future__ import annotations

try:
    from pydantic.v1 import Field
except ImportError:  # pragma: no cover
    from pydantic import Field  # type: ignore[assignment]

from common.depot.client import DepotClient, DepotClientConfig
from prefect.blocks.core import Block


class DepotBlock(Block):
    """Prefect block storing Depot client configuration."""

    _block_type_name = "Truthound Depot"
    base_url: str = Field(..., description="Depot API base URL")
    api_token: str = Field(..., description="Depot API bearer token")
    timeout_seconds: float = Field(default=10.0)
    poll_interval_seconds: float = Field(default=2.0)
    poll_timeout_seconds: float = Field(default=300.0)
    headers: dict[str, str] = Field(default_factory=dict)

    def create_client(self) -> DepotClient:
        return DepotClient(
            DepotClientConfig(
                base_url=self.base_url,
                api_token=self.api_token,
                timeout_seconds=self.timeout_seconds,
                headers=self.headers,
            )
        )

    def polling_kwargs(self) -> dict[str, float]:
        return {
            "poll_interval_seconds": self.poll_interval_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
        }


__all__ = ["DepotBlock"]
