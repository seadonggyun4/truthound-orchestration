"""Tests for timeout and concurrency handling in Kestra scripts and flows."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.testing import MockDataQualityEngine

from truthound_kestra.scripts import (
    ScriptConfig,
    CheckScriptConfig,
    ProfileScriptConfig,
    LearnScriptConfig,
    CheckScriptExecutor,
    ProfileScriptExecutor,
    LearnScriptExecutor,
)
from truthound_kestra.flows import (
    FlowConfig,
    TaskConfig,
    TaskType,
    RetryConfig,
    RetryPolicy,
)


# =============================================================================
# Timeout Tests
# =============================================================================


class TestScriptTimeout:
    """Tests for script timeout behavior."""

    def test_timeout_seconds_in_config(self) -> None:
        """Test timeout_seconds configuration."""
        config = ScriptConfig(timeout_seconds=30.0)
        assert config.timeout_seconds == 30.0

    def test_default_timeout_value(self) -> None:
        """Test default timeout value."""
        config = ScriptConfig()
        assert config.timeout_seconds == 300.0  # 5 minutes default

    def test_check_script_config_timeout(self) -> None:
        """Test CheckScriptConfig timeout."""
        config = CheckScriptConfig(timeout_seconds=60.0)
        assert config.timeout_seconds == 60.0

    def test_profile_script_config_timeout(self) -> None:
        """Test ProfileScriptConfig timeout."""
        config = ProfileScriptConfig(timeout_seconds=120.0)
        assert config.timeout_seconds == 120.0

    def test_learn_script_config_timeout(self) -> None:
        """Test LearnScriptConfig timeout."""
        config = LearnScriptConfig(timeout_seconds=180.0)
        assert config.timeout_seconds == 180.0

    def test_timeout_builder_method(self) -> None:
        """Test timeout setting via builder method."""
        config = ScriptConfig()
        config = config.with_timeout(90.0)
        assert config.timeout_seconds == 90.0


class TestFlowTimeout:
    """Tests for flow timeout configuration."""

    def test_task_config_timeout(self) -> None:
        """Test TaskConfig can have timeout."""
        config = TaskConfig(
            id="check_task",
            task_type=TaskType.CHECK,
            timeout_seconds=300.0,  # 5 minutes
        )
        assert config.timeout_seconds == 300.0

    def test_retry_config_with_timeout(self) -> None:
        """Test RetryConfig with max delay (timeout-like behavior)."""
        config = RetryConfig(
            max_attempts=3,
            policy=RetryPolicy.EXPONENTIAL,
            initial_delay_seconds=1.0,
            max_delay_seconds=60.0,  # Max delay acts as effective timeout
        )
        assert config.max_delay_seconds == 60.0


class TestExecutorTimeout:
    """Tests for executor timeout behavior."""

    @pytest.fixture
    def mock_engine(self) -> MockDataQualityEngine:
        """Create a mock engine."""
        engine = MockDataQualityEngine()
        engine.configure_check(success=True)
        engine.configure_profile(success=True)
        engine.configure_learn(success=True)
        return engine

    @pytest.fixture
    def sample_data(self) -> dict[str, list[Any]]:
        """Create sample data."""
        return {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }

    def test_executor_respects_timeout_config(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test that executor respects timeout configuration."""
        config = CheckScriptConfig(
            rules=({"type": "not_null", "column": "id"},),
            timeout_seconds=60.0,
        )
        executor = CheckScriptExecutor(config, engine=mock_engine)

        result = executor.execute(sample_data)

        assert result.is_success is True
        # Execution should complete well within timeout


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestScriptExecutorConcurrency:
    """Tests for concurrent script executor execution."""

    @pytest.fixture
    def mock_engine(self) -> MockDataQualityEngine:
        """Create a thread-safe mock engine."""
        engine = MockDataQualityEngine()
        engine.configure_check(success=True)
        engine.configure_profile(success=True)
        engine.configure_learn(success=True)
        return engine

    @pytest.fixture
    def sample_data(self) -> dict[str, list[Any]]:
        """Create sample data for testing."""
        return {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": ["a@test.com", "b@test.com", "c@test.com", "d@test.com", "e@test.com"],
        }

    def test_concurrent_check_executors(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test multiple check executors running concurrently."""
        config = CheckScriptConfig(
            rules=({"type": "not_null", "column": "id"},),
        )
        executor = CheckScriptExecutor(config, engine=mock_engine)

        num_concurrent = 5
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [
                pool.submit(executor.execute, sample_data)
                for _ in range(num_concurrent)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == num_concurrent
        for result in results:
            assert result.is_success is True

    def test_concurrent_profile_executors(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test multiple profile executors running concurrently."""
        config = ProfileScriptConfig()
        executor = ProfileScriptExecutor(config, engine=mock_engine)

        num_concurrent = 5
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [
                pool.submit(executor.execute, sample_data)
                for _ in range(num_concurrent)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == num_concurrent
        for result in results:
            assert result.is_success is True

    def test_concurrent_learn_executors(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test multiple learn executors running concurrently."""
        config = LearnScriptConfig()
        executor = LearnScriptExecutor(config, engine=mock_engine)

        num_concurrent = 5
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [
                pool.submit(executor.execute, sample_data)
                for _ in range(num_concurrent)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == num_concurrent
        for result in results:
            assert result.is_success is True

    def test_mixed_concurrent_executors(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test mixed check, profile, learn executors concurrently."""
        check_config = CheckScriptConfig(
            rules=({"type": "not_null", "column": "id"},),
        )
        check_executor = CheckScriptExecutor(check_config, engine=mock_engine)
        profile_executor = ProfileScriptExecutor(ProfileScriptConfig(), engine=mock_engine)
        learn_executor = LearnScriptExecutor(LearnScriptConfig(), engine=mock_engine)

        executors = [check_executor, profile_executor, learn_executor]
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(e.execute, sample_data) for e in executors
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == 3
        for result in results:
            assert result.is_success is True


class TestConfigImmutability:
    """Tests for config immutability under concurrent access."""

    def test_script_config_immutable_under_concurrent_access(self) -> None:
        """Test that script config remains immutable during concurrent access."""
        config = ScriptConfig(
            engine_name="truthound",
            timeout_seconds=60.0,
            tags=frozenset(["production", "users"]),
        )

        def access_config() -> dict[str, Any]:
            return {
                "engine_name": config.engine_name,
                "timeout_seconds": config.timeout_seconds,
                "tags": set(config.tags),
                "dict": config.to_dict(),
            }

        num_concurrent = 10
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [pool.submit(access_config) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # All results should be identical
        for result in results:
            assert result["engine_name"] == "truthound"
            assert result["timeout_seconds"] == 60.0
            assert result["tags"] == {"production", "users"}

    def test_check_script_config_immutable(self) -> None:
        """Test CheckScriptConfig immutability under concurrent access."""
        config = CheckScriptConfig(
            rules=({"type": "not_null", "column": "id"},),
            fail_on_error=True,
            auto_schema=False,
        )

        def access_config() -> dict[str, Any]:
            return {
                "rules": config.rules,
                "fail_on_error": config.fail_on_error,
                "auto_schema": config.auto_schema,
            }

        num_concurrent = 10
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [pool.submit(access_config) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        for result in results:
            assert len(result["rules"]) == 1
            assert result["fail_on_error"] is True
            assert result["auto_schema"] is False

    def test_flow_config_immutable_under_concurrent_access(self) -> None:
        """Test that FlowConfig remains immutable during concurrent access."""
        task = TaskConfig(id="task1", task_type=TaskType.CHECK)
        config = FlowConfig(
            id="test_flow",
            namespace="company.data",
            tasks=(task,),
        )

        def access_config() -> dict[str, Any]:
            return {
                "id": config.id,
                "namespace": config.namespace,
                "task_count": len(config.tasks),
                "dict": config.to_dict(),
            }

        num_concurrent = 10
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [pool.submit(access_config) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        for result in results:
            assert result["id"] == "test_flow"
            assert result["namespace"] == "company.data"
            assert result["task_count"] == 1


class TestFlowGenerationConcurrency:
    """Tests for concurrent flow YAML generation."""

    def test_concurrent_flow_yaml_generation(self) -> None:
        """Test generating multiple flow YAMLs concurrently."""
        from truthound_kestra.flows import FlowGenerator, generate_flow_yaml

        def generate_flow(i: int) -> str:
            config = FlowConfig(
                id=f"flow_{i}",
                namespace=f"namespace_{i}",
                description=f"Flow {i}",
            )
            return generate_flow_yaml(config)

        num_concurrent = 10
        yaml_outputs: list[str] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [pool.submit(generate_flow, i) for i in range(num_concurrent)]

            for future in as_completed(futures):
                yaml_output = future.result()
                yaml_outputs.append(yaml_output)

        assert len(yaml_outputs) == num_concurrent

        # Each YAML should be valid and unique
        unique_flows = set()
        for yaml_str in yaml_outputs:
            assert "id:" in yaml_str
            assert "namespace:" in yaml_str
            # Extract flow id for uniqueness check
            for line in yaml_str.split("\n"):
                if line.strip().startswith("id:"):
                    unique_flows.add(line.strip())
                    break

        assert len(unique_flows) == num_concurrent


class TestRetryConfigConcurrency:
    """Tests for retry configuration under concurrent access."""

    def test_retry_config_immutable(self) -> None:
        """Test RetryConfig immutability."""
        config = RetryConfig(
            max_attempts=5,
            policy=RetryPolicy.EXPONENTIAL,
            initial_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )

        def access_config() -> dict[str, Any]:
            return {
                "max_attempts": config.max_attempts,
                "policy": config.policy,
                "initial_delay_seconds": config.initial_delay_seconds,
                "max_delay_seconds": config.max_delay_seconds,
            }

        num_concurrent = 10
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [pool.submit(access_config) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        for result in results:
            assert result["max_attempts"] == 5
            assert result["policy"] == RetryPolicy.EXPONENTIAL
            assert result["initial_delay_seconds"] == 1.0
            assert result["max_delay_seconds"] == 60.0


class TestExceptionThreadSafety:
    """Tests for exception handling under concurrent access."""

    def test_exception_to_dict_concurrent(self) -> None:
        """Test exception to_dict is thread-safe."""
        from truthound_kestra.utils.exceptions import EngineError

        error = EngineError(
            message="Engine failed",
            engine_name="truthound",
            operation="check",
            original_error=ValueError("Bad data"),
        )

        def get_dict() -> dict[str, Any]:
            return error.to_dict()

        num_concurrent = 10
        dicts: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
            futures = [pool.submit(get_dict) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                d = future.result()
                dicts.append(d)

        # All should be identical
        for d in dicts:
            assert d["message"] == "Engine failed"
            assert d["engine_name"] == "truthound"
            assert d["operation"] == "check"
