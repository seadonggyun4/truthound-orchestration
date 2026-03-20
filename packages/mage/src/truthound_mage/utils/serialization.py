"""Serialization utilities for Mage data quality blocks."""

from __future__ import annotations

import json
from typing import Any

from common.base import CheckResult, LearnResult, ProfileResult
from common.serializers import detect_result_type, serialize_result_wire


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize any supported result to the shared wire format."""

    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, (CheckResult, ProfileResult, LearnResult)) or hasattr(result, "to_dict"):
        payload = serialize_result_wire(result, include_result_type=True)
        payload["_type"] = {
            "check": "CheckResult",
            "profile": "ProfileResult",
            "learn": "LearnResult",
        }.get(detect_result_type(result), type(result).__name__)
        payload["type"] = detect_result_type(result)
        return payload
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"value": result}


def deserialize_result(
    data: dict[str, Any],
    result_type: str | None = None,
) -> Any:
    """Deserialize dictionary data to a common result object when possible."""

    inferred_type = (result_type or data.get("_type") or data.get("type") or "").lower()
    if inferred_type in ("check", "checkresult"):
        return deserialize_check_result(data)
    if inferred_type in ("profile", "profileresult"):
        return deserialize_profile_result(data)
    if inferred_type in ("learn", "learnresult"):
        return deserialize_learn_result(data)
    return dict(data)


def serialize_check_result(result: CheckResult) -> dict[str, Any]:
    return serialize_result(result)


def deserialize_check_result(data: dict[str, Any]) -> CheckResult:
    normalized = dict(data)
    normalized.pop("_type", None)
    normalized.pop("type", None)
    normalized.pop("result_type", None)
    return CheckResult.from_dict(normalized)


def serialize_profile_result(result: ProfileResult) -> dict[str, Any]:
    return serialize_result(result)


def deserialize_profile_result(data: dict[str, Any]) -> ProfileResult:
    normalized = dict(data)
    normalized.pop("_type", None)
    normalized.pop("type", None)
    normalized.pop("result_type", None)
    return ProfileResult.from_dict(normalized)


def serialize_learn_result(result: LearnResult) -> dict[str, Any]:
    return serialize_result(result)


def deserialize_learn_result(data: dict[str, Any]) -> LearnResult:
    normalized = dict(data)
    normalized.pop("_type", None)
    normalized.pop("type", None)
    normalized.pop("result_type", None)
    return LearnResult.from_dict(normalized)


def to_json(result: Any, **kwargs: Any) -> str:
    return json.dumps(serialize_result(result), **kwargs)


def from_json(data: str, result_type: str | None = None) -> Any:
    return deserialize_result(json.loads(data), result_type)


__all__ = [
    "serialize_result",
    "deserialize_result",
    "serialize_check_result",
    "deserialize_check_result",
    "serialize_profile_result",
    "deserialize_profile_result",
    "serialize_learn_result",
    "deserialize_learn_result",
    "to_json",
    "from_json",
]
