"""Minimal Kestra Depot flow templates."""

from __future__ import annotations

from typing import Any, cast

import yaml


def generate_depot_validate_flow(
    *,
    flow_id: str,
    namespace: str,
    depot_id: str,
    asset_id: str,
    base_url: str = "{{ secret('DEPOT_BASE_URL') }}",
    api_token: str = "{{ secret('DEPOT_API_TOKEN') }}",
) -> str:
    payload: dict[str, Any] = {
        "id": flow_id,
        "namespace": namespace,
        "tasks": [
            {
                "id": "validate_branch",
                "type": "io.kestra.plugin.scripts.python.Script",
                "script": "\n".join(
                    [
                        "from truthound_kestra.scripts.depot import validate_branch_script",
                        f"result = validate_branch_script(depot_id={depot_id!r}, asset_id={asset_id!r}, base_url={base_url!r}, api_token={api_token!r})",
                        "print(result)",
                    ]
                ),
            }
        ],
    }
    return cast(str, yaml.dump(payload, sort_keys=False, allow_unicode=True))


def generate_depot_scheduled_validate_flow(
    *,
    flow_id: str,
    namespace: str,
    depot_id: str,
    asset_id: str,
    schedule: str,
    base_url: str = "{{ secret('DEPOT_BASE_URL') }}",
    api_token: str = "{{ secret('DEPOT_API_TOKEN') }}",
) -> str:
    payload: dict[str, Any] = {
        "id": flow_id,
        "namespace": namespace,
        "triggers": [{"id": "schedule", "type": "io.kestra.plugin.core.trigger.Schedule", "cron": schedule}],
        "tasks": [
            {
                "id": "scheduled_validate",
                "type": "io.kestra.plugin.scripts.python.Script",
                "script": "\n".join(
                    [
                        "from truthound_kestra.scripts.depot import validate_branch_script",
                        f"result = validate_branch_script(depot_id={depot_id!r}, asset_id={asset_id!r}, base_url={base_url!r}, api_token={api_token!r})",
                        "print(result)",
                    ]
                ),
            }
        ],
    }
    return cast(str, yaml.dump(payload, sort_keys=False, allow_unicode=True))


def generate_depot_release_flow(
    *,
    flow_id: str,
    namespace: str,
    depot_id: str,
    asset_id: str,
    release_tag: str,
    base_url: str = "{{ secret('DEPOT_BASE_URL') }}",
    api_token: str = "{{ secret('DEPOT_API_TOKEN') }}",
) -> str:
    payload: dict[str, Any] = {
        "id": flow_id,
        "namespace": namespace,
        "tasks": [
            {
                "id": "release_tag",
                "type": "io.kestra.plugin.scripts.python.Script",
                "script": "\n".join(
                    [
                        "from truthound_kestra.scripts.depot import release_tag_script",
                        f"result = release_tag_script(depot_id={depot_id!r}, asset_id={asset_id!r}, release_tag={release_tag!r}, base_url={base_url!r}, api_token={api_token!r})",
                        "print(result)",
                    ]
                ),
            }
        ],
    }
    return cast(str, yaml.dump(payload, sort_keys=False, allow_unicode=True))


__all__ = [
    "generate_depot_validate_flow",
    "generate_depot_scheduled_validate_flow",
    "generate_depot_release_flow",
]
