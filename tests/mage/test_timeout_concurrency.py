"""Tests for timeout and concurrency handling in Mage blocks."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.testing import MockDataQualityEngine
from common.base import CheckStatus

from truthound_mage.blocks.base import BlockConfig, BlockExecutionContext
from truthound_mage.blocks.sensor import (
    BaseSensorBlock,
    SensorBlockConfig,
    SensorResult,
)
from truthound_mage.blocks.transformer import (
    CheckTransformer,
    ProfileTransformer,
    LearnTransformer,
)
from truthound_mage.blocks.base import CheckBlockConfig


# =============================================================================
# Timeout Tests
# =============================================================================


class TestBlockTimeout:
    """Tests for block timeout behavior."""

    def test_timeout_seconds_in_config(self) -> None:
        """Test timeout_seconds configuration."""
        config = BlockConfig(timeout_seconds=30.0)
        assert config.timeout_seconds == 30.0

    def test_default_timeout_value(self) -> None:
        """Test default timeout value."""
        config = BlockConfig()
        assert config.timeout_seconds == 300.0  # 5 minutes default

    def test_timeout_validation_negative(self) -> None:
        """Test that negative timeout is rejected."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            BlockConfig(timeout_seconds=-1.0)

    def test_timeout_validation_zero(self) -> None:
        """Test that zero timeout is rejected."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            BlockConfig(timeout_seconds=0.0)

    def test_sensor_timeout_config(self) -> None:
        """Test sensor timeout configuration."""
        config = SensorBlockConfig(timeout_seconds=60.0)
        assert config.timeout_seconds == 60.0


class TestSensorTimeout:
    """Tests for sensor timeout behavior."""

    def test_sensor_timeout_soft_fail(self) -> None:
        """Test sensor timeout with soft_fail=True."""
        # Create a sensor that never passes
        class NeverPassSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                return False

        config = SensorBlockConfig(
            timeout_seconds=0.1,  # Very short timeout
            poke_interval_seconds=0.05,
            soft_fail=True,
        )
        sensor = NeverPassSensor(config=config)

        result = sensor.execute({"pass_rate": 0.5})

        assert result.passed is False
        assert "timed out" in result.message.lower()
        assert result.poke_count >= 1

    def test_sensor_timeout_hard_fail(self) -> None:
        """Test sensor timeout with soft_fail=False raises exception."""
        from truthound_mage.utils.exceptions import SensorTimeoutError

        class NeverPassSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                return False

        config = SensorBlockConfig(
            timeout_seconds=0.1,
            poke_interval_seconds=0.05,
            soft_fail=False,
        )
        sensor = NeverPassSensor(config=config)

        with pytest.raises(SensorTimeoutError) as exc_info:
            sensor.execute({"pass_rate": 0.5})

        error = exc_info.value
        assert error.timeout_seconds == 0.1
        assert error.poke_count >= 1

    def test_sensor_max_poke_attempts_soft_fail(self) -> None:
        """Test max poke attempts with soft_fail=True."""
        class NeverPassSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                return False

        config = SensorBlockConfig(
            timeout_seconds=300.0,  # Long timeout
            poke_interval_seconds=0.01,
            max_poke_attempts=3,
            soft_fail=True,
        )
        sensor = NeverPassSensor(config=config)

        result = sensor.execute({"pass_rate": 0.5})

        assert result.passed is False
        assert result.poke_count == 3
        assert "max poke attempts" in result.message.lower()

    def test_sensor_max_poke_attempts_hard_fail(self) -> None:
        """Test max poke attempts with soft_fail=False raises exception."""
        from truthound_mage.utils.exceptions import SensorTimeoutError

        class NeverPassSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                return False

        config = SensorBlockConfig(
            timeout_seconds=300.0,
            poke_interval_seconds=0.01,
            max_poke_attempts=2,
            soft_fail=False,
        )
        sensor = NeverPassSensor(config=config)

        with pytest.raises(SensorTimeoutError) as exc_info:
            sensor.execute({"pass_rate": 0.5})

        assert exc_info.value.poke_count == 2

    def test_sensor_exponential_backoff(self) -> None:
        """Test sensor exponential backoff behavior."""
        intervals: list[float] = []

        class TrackingNeverPassSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                return False

        config = SensorBlockConfig(
            timeout_seconds=0.2,
            poke_interval_seconds=0.01,
            exponential_backoff=True,
            soft_fail=True,
        )
        sensor = TrackingNeverPassSensor(config=config)

        result = sensor.execute({"pass_rate": 0.5})

        # Should have completed due to timeout
        assert result.passed is False
        # Multiple pokes should have occurred
        assert result.poke_count >= 1

    def test_sensor_immediate_success_no_timeout(self) -> None:
        """Test that successful sensor doesn't timeout."""
        class AlwaysPassSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                return True

        config = SensorBlockConfig(
            timeout_seconds=0.1,
            poke_interval_seconds=0.05,
        )
        sensor = AlwaysPassSensor(config=config)

        result = sensor.execute({"pass_rate": 0.98})

        assert result.passed is True
        assert result.poke_count == 1


class TestPokeBehavior:
    """Tests for sensor poke behavior."""

    def test_poke_interval_validation(self) -> None:
        """Test that poke interval must be positive."""
        with pytest.raises(ValueError, match="poke_interval_seconds must be positive"):
            SensorBlockConfig(poke_interval_seconds=0)

    def test_poke_interval_validation_negative(self) -> None:
        """Test that negative poke interval is rejected."""
        with pytest.raises(ValueError, match="poke_interval_seconds must be positive"):
            SensorBlockConfig(poke_interval_seconds=-1)

    def test_single_check_no_poke_loop(self) -> None:
        """Test that check() method doesn't loop."""
        poke_count = 0

        class CountingNeverPassSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                nonlocal poke_count
                poke_count += 1
                return False

        config = SensorBlockConfig(
            timeout_seconds=10.0,
            poke_interval_seconds=0.01,
        )
        sensor = CountingNeverPassSensor(config=config)

        # Use check() instead of execute() - should only poke once
        result = sensor.check({"pass_rate": 0.5})

        assert poke_count == 1
        assert result.poke_count == 1
        assert result.passed is False


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestTransformerConcurrency:
    """Tests for concurrent transformer execution."""

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
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }

    def test_concurrent_check_transformers(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test multiple check transformers running concurrently."""
        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
            log_results=False,  # Disable logging to avoid context issues in threads
        )
        transformer = CheckTransformer(config=config, engine=mock_engine)

        num_concurrent = 5
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(transformer.execute, sample_data)
                for _ in range(num_concurrent)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == num_concurrent
        for result in results:
            assert result.success is True

    def test_concurrent_profile_transformers(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test multiple profile transformers running concurrently."""
        from truthound_mage.blocks.base import ProfileBlockConfig
        config = ProfileBlockConfig(log_results=False)
        transformer = ProfileTransformer(config=config, engine=mock_engine)

        num_concurrent = 5
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(transformer.execute, sample_data)
                for _ in range(num_concurrent)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == num_concurrent
        for result in results:
            assert result.success is True

    def test_concurrent_learn_transformers(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test multiple learn transformers running concurrently."""
        from truthound_mage.blocks.base import LearnBlockConfig
        config = LearnBlockConfig(log_results=False)
        transformer = LearnTransformer(config=config, engine=mock_engine)

        num_concurrent = 5
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(transformer.execute, sample_data)
                for _ in range(num_concurrent)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == num_concurrent
        for result in results:
            assert result.success is True

    def test_mixed_concurrent_operations(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, list[Any]]
    ) -> None:
        """Test mixed check, profile, learn operations concurrently."""
        from truthound_mage.blocks.base import ProfileBlockConfig, LearnBlockConfig
        check_config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
            log_results=False,
        )
        profile_config = ProfileBlockConfig(log_results=False)
        learn_config = LearnBlockConfig(log_results=False)
        check_transformer = CheckTransformer(config=check_config, engine=mock_engine)
        profile_transformer = ProfileTransformer(config=profile_config, engine=mock_engine)
        learn_transformer = LearnTransformer(config=learn_config, engine=mock_engine)

        transformers = [check_transformer, profile_transformer, learn_transformer]
        results: list[Any] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(t.execute, sample_data) for t in transformers
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == 3
        for result in results:
            assert result.success is True


class TestSensorConcurrency:
    """Tests for concurrent sensor execution."""

    def test_concurrent_sensor_checks(self) -> None:
        """Test multiple sensors running checks concurrently."""

        class PassRateSensor(BaseSensorBlock):
            def poke(self, check_result: Any, context: Any = None) -> bool:
                if isinstance(check_result, dict):
                    return check_result.get("pass_rate", 0) >= 0.9
                return False

        config = SensorBlockConfig(min_pass_rate=0.9)
        sensor = PassRateSensor(config=config)

        # Different check results
        check_results = [
            {"pass_rate": 0.95, "passed_count": 95, "failed_count": 5},
            {"pass_rate": 0.85, "passed_count": 85, "failed_count": 15},
            {"pass_rate": 0.99, "passed_count": 99, "failed_count": 1},
            {"pass_rate": 0.50, "passed_count": 50, "failed_count": 50},
            {"pass_rate": 0.92, "passed_count": 92, "failed_count": 8},
        ]

        results: list[SensorResult] = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(sensor.check, cr) for cr in check_results
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == 5

        # Count passed and failed
        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(1 for r in results if not r.passed)

        # 0.95, 0.99, 0.92 should pass (>= 0.9)
        # 0.85, 0.50 should fail
        assert passed_count == 3
        assert failed_count == 2


class TestConfigImmutability:
    """Tests for config immutability under concurrent access."""

    def test_block_config_immutable_under_concurrent_access(self) -> None:
        """Test that config remains immutable during concurrent access."""
        config = BlockConfig(
            engine_name="truthound",
            timeout_seconds=60.0,
            tags=frozenset(["tag1", "tag2"]),
        )

        def access_config() -> dict[str, Any]:
            # Read various attributes concurrently
            return {
                "engine_name": config.engine_name,
                "timeout_seconds": config.timeout_seconds,
                "tags": set(config.tags),
                "dict": config.to_dict(),
            }

        num_concurrent = 10
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(access_config) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # All results should be identical
        for result in results:
            assert result["engine_name"] == "truthound"
            assert result["timeout_seconds"] == 60.0
            assert result["tags"] == {"tag1", "tag2"}

    def test_sensor_config_immutable_under_concurrent_access(self) -> None:
        """Test that sensor config remains immutable during concurrent access."""
        config = SensorBlockConfig(
            min_pass_rate=0.95,
            max_failure_rate=0.05,
            poke_interval_seconds=30.0,
            timeout_seconds=120.0,
        )

        def access_config() -> dict[str, Any]:
            return {
                "min_pass_rate": config.min_pass_rate,
                "max_failure_rate": config.max_failure_rate,
                "poke_interval_seconds": config.poke_interval_seconds,
                "timeout_seconds": config.timeout_seconds,
            }

        num_concurrent = 10
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(access_config) for _ in range(num_concurrent)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        for result in results:
            assert result["min_pass_rate"] == 0.95
            assert result["max_failure_rate"] == 0.05


class TestContextThreadSafety:
    """Tests for BlockExecutionContext thread safety."""

    def test_context_creation_concurrent(self) -> None:
        """Test concurrent context creation."""
        def create_context(i: int) -> BlockExecutionContext:
            return BlockExecutionContext(
                block_uuid=f"block_{i}",
                pipeline_uuid=f"pipeline_{i}",
            )

        num_concurrent = 10
        contexts: list[BlockExecutionContext] = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(create_context, i) for i in range(num_concurrent)]

            for future in as_completed(futures):
                context = future.result()
                contexts.append(context)

        assert len(contexts) == num_concurrent

        # Verify each context is unique
        block_uuids = {c.block_uuid for c in contexts}
        assert len(block_uuids) == num_concurrent
