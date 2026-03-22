"""Tests for truthound_kestra.outputs module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.base import CheckResult, CheckStatus, ProfileResult, LearnResult

from truthound_kestra.outputs import (
    # Configuration
    OutputConfig,
    # Handlers
    KestraOutputHandler,
    FileOutputHandler,
    MultiOutputHandler,
    # Functions
    send_outputs,
    send_check_result,
    send_profile_result,
    send_learn_result,
)
from truthound_kestra.utils import OutputFormat


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_output_config_defaults(self) -> None:
        """Test OutputConfig default values."""
        config = OutputConfig()

        assert config.format == OutputFormat.JSON
        assert config.include_metadata is True
        assert config.include_summary is True
        assert config.include_failures is True
        assert config.max_failures == 100
        assert config.file_output is False
        assert config.file_path is None

    def test_output_config_custom(self) -> None:
        """Test OutputConfig with custom values."""
        config = OutputConfig(
            format=OutputFormat.YAML,
            include_metadata=False,
            include_summary=False,
            include_failures=False,
            max_failures=50,
            file_output=True,
            file_path="/tmp/output.yaml",
        )

        assert config.format == OutputFormat.YAML
        assert config.include_metadata is False
        assert config.include_summary is False
        assert config.max_failures == 50
        assert config.file_path == "/tmp/output.yaml"

    def test_output_config_builder(self) -> None:
        """Test OutputConfig builder pattern."""
        config = OutputConfig()
        config = config.with_format(OutputFormat.MARKDOWN)
        config = config.with_file_output("/tmp/output.md")

        assert config.format == OutputFormat.MARKDOWN
        assert config.file_output is True
        assert config.file_path == "/tmp/output.md"

    def test_output_config_with_format_string(self) -> None:
        """Test OutputConfig with_format accepts string."""
        config = OutputConfig()
        config = config.with_format("yaml")

        assert config.format == OutputFormat.YAML

    def test_output_config_immutability(self) -> None:
        """Test that OutputConfig is immutable."""
        config = OutputConfig()
        with pytest.raises(AttributeError):
            config.format = OutputFormat.YAML  # type: ignore


class TestKestraOutputHandler:
    """Tests for KestraOutputHandler."""

    def test_handler_creation(self) -> None:
        """Test KestraOutputHandler creation."""
        config = OutputConfig()
        handler = KestraOutputHandler(config)

        assert handler.config == config

    def test_handler_creation_no_config(self) -> None:
        """Test KestraOutputHandler creation without config."""
        handler = KestraOutputHandler()

        assert handler.config is not None
        assert handler.config.format == OutputFormat.JSON

    def test_handler_send_dict(self) -> None:
        """Test KestraOutputHandler send with dict."""
        handler = KestraOutputHandler()

        outputs = {"status": "passed", "count": 10}

        # Should not raise even without Kestra SDK (fallback to print)
        handler.send(outputs)

    def test_handler_send_check_result(self) -> None:
        """Test KestraOutputHandler send_check_result."""
        handler = KestraOutputHandler()

        # Using ScriptOutput-like dict
        result = {
            "status": "passed",
            "is_success": True,
            "operation": "check",
            "passed_count": 10,
            "failed_count": 0,
            "metadata": {"engine": "mock"},
        }
        handler.send_check_result(result)

    @patch("truthound_kestra.outputs.handlers.logger")
    def test_handler_with_fallback(self, mock_logger: MagicMock) -> None:
        """Test KestraOutputHandler uses fallback when Kestra SDK not available."""
        handler = KestraOutputHandler()

        outputs = {"status": "passed"}
        handler.send(outputs)

        # Should have logged debug message
        mock_logger.debug.assert_called()


class TestFileOutputHandler:
    """Tests for FileOutputHandler."""

    def test_handler_creation(self) -> None:
        """Test FileOutputHandler creation."""
        config = OutputConfig()
        handler = FileOutputHandler(config)

        assert handler.config == config

    def test_handler_send_json(self) -> None:
        """Test FileOutputHandler send with JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.json"
            config = OutputConfig(
                format=OutputFormat.JSON,
                file_output=True,
                file_path=str(output_path),
            )
            handler = FileOutputHandler(config)

            outputs = {"status": "passed", "count": 10}
            handler.send(outputs)

            assert output_path.exists()
            content = json.loads(output_path.read_text())
            assert content["status"] == "passed"

    def test_handler_send_yaml(self) -> None:
        """Test FileOutputHandler send with YAML format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.yaml"
            config = OutputConfig(
                format=OutputFormat.YAML,
                file_output=True,
                file_path=str(output_path),
            )
            handler = FileOutputHandler(config)

            outputs = {"status": "passed", "count": 10}
            handler.send(outputs)

            assert output_path.exists()
            content = output_path.read_text()
            assert "status" in content

    def test_handler_send_markdown(self) -> None:
        """Test FileOutputHandler send with Markdown format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.md"
            config = OutputConfig(
                format=OutputFormat.MARKDOWN,
                file_output=True,
                file_path=str(output_path),
            )
            handler = FileOutputHandler(config)

            outputs = {"status": "passed", "count": 10}
            handler.send(outputs)

            assert output_path.exists()

    def test_handler_no_file_path_raises(self) -> None:
        """Test FileOutputHandler raises when no file_path configured."""
        from truthound_kestra.utils.exceptions import OutputError

        config = OutputConfig()  # No file_path
        handler = FileOutputHandler(config)

        with pytest.raises(OutputError):
            handler.send({"status": "passed"})


class TestMultiOutputHandler:
    """Tests for MultiOutputHandler."""

    def test_handler_creation(self) -> None:
        """Test MultiOutputHandler creation."""
        handler1 = KestraOutputHandler()
        multi = MultiOutputHandler([handler1])

        assert len(multi._handlers) == 1

    def test_handler_send_to_all(self) -> None:
        """Test MultiOutputHandler sends to all handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path1 = Path(tmpdir) / "output1.json"
            output_path2 = Path(tmpdir) / "output2.json"

            config1 = OutputConfig(file_output=True, file_path=str(output_path1))
            config2 = OutputConfig(file_output=True, file_path=str(output_path2))

            handler1 = FileOutputHandler(config1)
            handler2 = FileOutputHandler(config2)

            multi = MultiOutputHandler([handler1, handler2])

            outputs = {"status": "passed"}
            multi.send(outputs)

            assert output_path1.exists()
            assert output_path2.exists()

    def test_handler_send_check_result(self) -> None:
        """Test MultiOutputHandler send_check_result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.json"
            config = OutputConfig(file_output=True, file_path=str(output_path))
            file_handler = FileOutputHandler(config)

            multi = MultiOutputHandler([file_handler])

            result = {
                "status": "passed",
                "is_success": True,
                "passed_count": 10,
                "failed_count": 0,
            }
            multi.send_check_result(result)

            assert output_path.exists()

    def test_handler_partial_failure(self) -> None:
        """Test MultiOutputHandler continues on partial failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # This one will work
            output_path = Path(tmpdir) / "output.json"
            config = OutputConfig(file_output=True, file_path=str(output_path))
            working_handler = FileOutputHandler(config)

            # This one will fail (no file_path)
            failing_handler = FileOutputHandler(OutputConfig())

            multi = MultiOutputHandler([failing_handler, working_handler])

            outputs = {"status": "passed"}
            # Should not raise because at least one handler succeeds
            multi.send(outputs)

            assert output_path.exists()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_send_outputs(self) -> None:
        """Test send_outputs function."""
        outputs = {"status": "passed", "count": 10}

        # Should not raise (uses fallback)
        send_outputs(outputs)

    def test_send_check_result(
        self,
        sample_check_result: CheckResult,
    ) -> None:
        """Test send_check_result function."""
        # Convert to dict for the handler
        result_dict = {
            "status": sample_check_result.status.value,
            "is_success": sample_check_result.is_success,
            "passed_count": sample_check_result.passed_count,
            "failed_count": sample_check_result.failed_count,
        }
        # Should not raise
        send_check_result(result_dict)

    def test_send_check_result_with_name(
        self,
        sample_check_result: CheckResult,
    ) -> None:
        """Test send_check_result with custom output name."""
        result_dict = {
            "status": sample_check_result.status.value,
            "is_success": sample_check_result.is_success,
        }
        send_check_result(result_dict, output_name="custom_check")

    def test_send_profile_result(
        self,
        sample_profile_result: ProfileResult,
    ) -> None:
        """Test send_profile_result function."""
        result_dict = {
            "status": sample_profile_result.status.value,
            "columns": [
                {"column_name": col.column_name, "dtype": col.dtype}
                for col in sample_profile_result.columns
            ],
            "total_rows": sample_profile_result.row_count,
        }
        # Should not raise
        send_profile_result(result_dict)

    def test_send_learn_result(
        self,
        sample_learn_result: LearnResult,
    ) -> None:
        """Test send_learn_result function."""
        result_dict = {
            "status": sample_learn_result.status.value,
            "rules": [
                {
                    "rule_type": rule.rule_type,
                    "column": rule.column,
                    "confidence": rule.confidence,
                }
                for rule in sample_learn_result.rules
            ],
        }
        # Should not raise
        send_learn_result(result_dict)


class TestOutputSerialization:
    """Tests for output serialization."""

    def test_check_result_serialization(
        self,
        sample_check_result: CheckResult,
    ) -> None:
        """Test CheckResult is properly serialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"
            config = OutputConfig(file_output=True, file_path=str(output_path))
            handler = FileOutputHandler(config)

            # Convert to dict for serialization
            data = {
                "status": sample_check_result.status.name.lower(),
                "passed_count": sample_check_result.passed_count,
                "failed_count": sample_check_result.failed_count,
            }
            handler.send(data)

            content = json.loads(output_path.read_text())
            assert content["status"] == "passed"
            assert content["passed_count"] == 10

    def test_failed_check_result_serialization(
        self,
        sample_failed_check_result: CheckResult,
    ) -> None:
        """Test failed CheckResult is properly serialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"
            config = OutputConfig(file_output=True, file_path=str(output_path))
            handler = FileOutputHandler(config)

            data = {
                "status": sample_failed_check_result.status.name.lower(),
                "passed_count": sample_failed_check_result.passed_count,
                "failed_count": sample_failed_check_result.failed_count,
                "failures": [
                    {"column": f.column, "rule_name": f.rule_name, "message": f.message}
                    for f in sample_failed_check_result.failures
                ],
            }
            handler.send(data)

            content = json.loads(output_path.read_text())
            assert content["status"] == "failed"
            assert content["failed_count"] == 2
            assert len(content["failures"]) == 2

    def test_format_check_result(self) -> None:
        """Test BaseOutputHandler format_check_result method."""
        handler = KestraOutputHandler()

        result = {
            "status": "passed",
            "is_success": True,
            "operation": "check",
            "passed_count": 10,
            "failed_count": 0,
            "metadata": {"engine": "mock"},
        }
        formatted = handler.format_check_result(result)

        assert formatted["status"] == "passed"
        assert formatted["is_success"] is True
        assert "summary" in formatted
        assert "metadata" in formatted
