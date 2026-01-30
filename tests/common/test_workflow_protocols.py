"""Tests for ExtendedWorkflowIntegration Protocol (TASK 0-4)."""

from __future__ import annotations

from typing import Any

import pytest

from common.base import (
    AnomalyConfig,
    AnomalyResult,
    AnomalyStatus,
    CheckConfig,
    CheckResult,
    CheckStatus,
    DriftConfig,
    DriftResult,
    DriftStatus,
    ExtendedWorkflowIntegration,
    LearnConfig,
    LearnResult,
    LearnStatus,
    ProfileConfig,
    ProfileResult,
    ProfileStatus,
    WorkflowIntegration,
)


# =============================================================================
# Stub Implementations
# =============================================================================


class BasicWorkflow:
    """Implements only WorkflowIntegration."""

    @property
    def platform_name(self) -> str:
        return "basic"

    @property
    def platform_version(self) -> str:
        return "1.0.0"

    def check(self, data: Any, config: CheckConfig) -> CheckResult:
        return CheckResult(status=CheckStatus.PASSED)

    def profile(self, data: Any, config: ProfileConfig) -> ProfileResult:
        return ProfileResult(status=ProfileStatus.COMPLETED)

    def learn(self, data: Any, config: LearnConfig) -> LearnResult:
        return LearnResult(status=LearnStatus.COMPLETED)


class ExtendedWorkflow(BasicWorkflow):
    """Implements both WorkflowIntegration and ExtendedWorkflowIntegration."""

    def detect_drift(
        self, baseline: Any, current: Any, config: DriftConfig
    ) -> DriftResult:
        return DriftResult(status=DriftStatus.NO_DRIFT, total_columns=1)

    def detect_anomalies(self, data: Any, config: AnomalyConfig) -> AnomalyResult:
        return AnomalyResult(status=AnomalyStatus.NORMAL, total_row_count=1)


# =============================================================================
# Tests
# =============================================================================


class TestWorkflowProtocols:
    def test_basic_is_workflow_integration(self) -> None:
        assert isinstance(BasicWorkflow(), WorkflowIntegration)

    def test_basic_not_extended(self) -> None:
        assert not isinstance(BasicWorkflow(), ExtendedWorkflowIntegration)

    def test_extended_is_workflow_integration(self) -> None:
        assert isinstance(ExtendedWorkflow(), WorkflowIntegration)

    def test_extended_is_extended_workflow(self) -> None:
        assert isinstance(ExtendedWorkflow(), ExtendedWorkflowIntegration)


class TestExtendedWorkflowInvocation:
    def test_detect_drift(self) -> None:
        wf = ExtendedWorkflow()
        result = wf.detect_drift(None, None, DriftConfig())
        assert isinstance(result, DriftResult)
        assert result.status == DriftStatus.NO_DRIFT

    def test_detect_anomalies(self) -> None:
        wf = ExtendedWorkflow()
        result = wf.detect_anomalies(None, AnomalyConfig())
        assert isinstance(result, AnomalyResult)
        assert result.status == AnomalyStatus.NORMAL


class TestConditionalFeatureUse:
    """Test the pattern of checking for extended capabilities before use."""

    def test_safe_drift_check(self) -> None:
        workflow: WorkflowIntegration = BasicWorkflow()
        if isinstance(workflow, ExtendedWorkflowIntegration):
            pytest.fail("BasicWorkflow should not be ExtendedWorkflowIntegration")

    def test_safe_extended_use(self) -> None:
        workflow: WorkflowIntegration = ExtendedWorkflow()
        if isinstance(workflow, ExtendedWorkflowIntegration):
            result = workflow.detect_drift(None, None, DriftConfig())
            assert result.status == DriftStatus.NO_DRIFT
        else:
            pytest.fail("ExtendedWorkflow should be ExtendedWorkflowIntegration")
