"""Tests for shared Depot observability redaction rules."""

from __future__ import annotations

from common.depot.observability import (
    redact_artifact_refs,
    redact_observability_payload,
    sanitize_observability_url,
)
from common.depot.testing import build_artifact_refs


def test_redaction_masks_secret_bearing_keys_recursively() -> None:
    payload = redact_observability_payload(
        {
            "apiToken": "secret-token",
            "nested": {
                "authorization": "Bearer abc",
                "headers": {"x-api-key": "hidden"},
                "safe": "value",
            },
        }
    )

    assert payload["apiToken"] == "[REDACTED]"
    assert payload["nested"]["authorization"] == "[REDACTED]"
    assert payload["nested"]["headers"] == "[REDACTED]"
    assert payload["nested"]["safe"] == "value"


def test_redaction_strips_query_and_fragment_from_urls() -> None:
    redacted = sanitize_observability_url(
        "https://depot.example/path?token=abc123#fragment"
    )

    assert redacted == "https://depot.example/path"


def test_redaction_drops_raw_body_fields_and_sanitizes_artifact_refs() -> None:
    payload = redact_observability_payload(
        {
            "snapshot_body": {"id": 1},
            "evidence_blob": "raw",
            "dataset": [{"id": 1}],
            "safe_url": "core://results/1?sig=abc",
        }
    )
    artifact_refs = redact_artifact_refs(
        build_artifact_refs(
            core_result_ref="https://core.example/results/1?signature=abc#frag",
        )
    )

    assert payload["snapshot_body"] == "[REDACTED]"
    assert payload["evidence_blob"] == "[REDACTED]"
    assert payload["dataset"] == "[REDACTED]"
    assert payload["safe_url"] == "core://results/1"
    assert artifact_refs["core_result_ref"] == "https://core.example/results/1"
