"""Nightly smoke tests for advanced-tier engines."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_great_expectations_adapter_imports_when_dependency_is_available() -> None:
    pytest.importorskip("great_expectations")

    from common.engines.great_expectations import GreatExpectationsAdapter

    adapter = GreatExpectationsAdapter()
    assert adapter.engine_name == "great_expectations"


@pytest.mark.integration
def test_pandera_adapter_imports_when_dependency_is_available() -> None:
    pytest.importorskip("pandera")

    from common.engines.pandera import PanderaAdapter

    adapter = PanderaAdapter()
    assert adapter.engine_name == "pandera"
