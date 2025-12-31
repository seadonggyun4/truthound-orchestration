"""Tests for manifest and results parsers."""

import json
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from truthound_dbt.parsers import (
    ManifestParser,
    ManifestParserConfig,
    TruthoundTest,
    TruthoundReport,
    CheckCategory,
    ManifestNotFoundError,
    RunResultsParser,
    RunResultsParserConfig,
    TestResult,
    RunSummary,
    RunResultsNotFoundError,
)
from truthound_dbt.testing import (
    MockManifest,
    MockRunResults,
    create_mock_manifest,
    create_mock_run_results,
    sample_manifest_data,
    sample_run_results_data,
)


class TestManifestParser:
    """Tests for ManifestParser."""

    def test_load_manifest_not_found(self):
        """Test loading non-existent manifest raises error."""
        parser = ManifestParser("/nonexistent/manifest.json")
        with pytest.raises(ManifestNotFoundError):
            parser.load()

    def test_load_manifest_success(self, tmp_path):
        """Test loading valid manifest."""
        manifest_path = tmp_path / "manifest.json"
        data = create_mock_manifest()
        manifest_path.write_text(json.dumps(data))

        parser = ManifestParser(manifest_path)
        parser.load()
        # No exception means success

    def test_get_truthound_tests(self, tmp_path):
        """Test getting Truthound tests from manifest."""
        manifest = MockManifest()
        path = manifest.create_temp_file()

        try:
            parser = ManifestParser(path)
            tests = parser.get_truthound_tests()
            assert isinstance(tests, list)
            assert all(isinstance(t, TruthoundTest) for t in tests)
        finally:
            path.unlink()

    def test_get_all_tests(self, tmp_path):
        """Test getting all tests from manifest."""
        manifest_path = tmp_path / "manifest.json"
        data = create_mock_manifest(
            tests=[
                {"type": "not_null", "column": "id", "model": "users"},
                {"type": "unique", "column": "id", "model": "users"},
            ]
        )
        manifest_path.write_text(json.dumps(data))

        parser = ManifestParser(manifest_path)
        tests = parser.get_all_tests()
        assert len(tests) == 2

    def test_get_models(self, tmp_path):
        """Test getting models from manifest."""
        manifest_path = tmp_path / "manifest.json"
        data = create_mock_manifest(models=["model_a", "model_b", "model_c"])
        manifest_path.write_text(json.dumps(data))

        parser = ManifestParser(manifest_path)
        models = parser.get_models()
        assert len(models) == 3

    def test_get_tests_for_model(self, tmp_path):
        """Test getting tests for specific model."""
        manifest_path = tmp_path / "manifest.json"
        data = create_mock_manifest(
            models=["users", "orders"],
            tests=[
                {"type": "not_null", "column": "id", "model": "users"},
                {"type": "unique", "column": "id", "model": "orders"},
            ]
        )
        manifest_path.write_text(json.dumps(data))

        parser = ManifestParser(manifest_path)
        tests = parser.get_tests_for_model("users")
        assert len(tests) == 1

    def test_generate_report(self, tmp_path):
        """Test generating report."""
        manifest = MockManifest()
        path = manifest.create_temp_file()

        try:
            parser = ManifestParser(path)
            report = parser.generate_report()

            assert isinstance(report, TruthoundReport)
            assert isinstance(report.coverage.model_coverage, float)
            assert isinstance(report.distribution.by_type, dict)
        finally:
            path.unlink()

    def test_report_to_markdown(self, tmp_path):
        """Test report markdown generation."""
        manifest = MockManifest()
        path = manifest.create_temp_file()

        try:
            parser = ManifestParser(path)
            report = parser.generate_report()
            markdown = report.to_markdown()

            assert "# Truthound Test Report" in markdown
            assert "## Coverage Summary" in markdown
        finally:
            path.unlink()

    def test_report_to_dict(self, tmp_path):
        """Test report dict conversion."""
        manifest = MockManifest()
        path = manifest.create_temp_file()

        try:
            parser = ManifestParser(path)
            report = parser.generate_report()
            data = report.to_dict()

            assert "tests" in data
            assert "coverage" in data
            assert "distribution" in data
        finally:
            path.unlink()


class TestTruthoundTest:
    """Tests for TruthoundTest dataclass."""

    def test_category_mapping(self):
        """Test category is derived from rule type."""
        test = TruthoundTest(
            name="test_not_null",
            unique_id="test.project.test",
            model="users",
            rule_type="not_null",
        )
        assert test.category == CheckCategory.COMPLETENESS

        test = TruthoundTest(
            name="test_unique",
            unique_id="test.project.test",
            model="users",
            rule_type="unique",
        )
        assert test.category == CheckCategory.UNIQUENESS

    def test_to_rule(self):
        """Test converting to TruthoundRule."""
        test = TruthoundTest(
            name="test_not_null",
            unique_id="test.project.test",
            model="users",
            column="id",
            rule_type="not_null",
        )
        rule = test.to_rule()
        assert rule.rule_type == "not_null"
        assert rule.column == "id"

    def test_to_dict(self):
        """Test converting to dictionary."""
        test = TruthoundTest(
            name="test_not_null",
            unique_id="test.project.test",
            model="users",
            column="id",
            rule_type="not_null",
        )
        data = test.to_dict()
        assert data["name"] == "test_not_null"
        assert data["model"] == "users"
        assert data["column"] == "id"


class TestRunResultsParser:
    """Tests for RunResultsParser."""

    def test_load_results_not_found(self):
        """Test loading non-existent results raises error."""
        parser = RunResultsParser("/nonexistent/run_results.json")
        with pytest.raises(RunResultsNotFoundError):
            parser.load()

    def test_load_results_success(self, tmp_path):
        """Test loading valid results."""
        results_path = tmp_path / "run_results.json"
        data = create_mock_run_results()
        results_path.write_text(json.dumps(data))

        parser = RunResultsParser(results_path)
        parser.load()
        # No exception means success

    def test_get_test_results(self, tmp_path):
        """Test getting test results."""
        results = MockRunResults()
        path = results.create_temp_file()

        try:
            parser = RunResultsParser(path)
            test_results = parser.get_test_results()

            assert isinstance(test_results, list)
            assert all(isinstance(r, TestResult) for r in test_results)
        finally:
            path.unlink()

    def test_get_summary(self, tmp_path):
        """Test getting run summary."""
        results = MockRunResults()
        path = results.create_temp_file()

        try:
            parser = RunResultsParser(path)
            summary = parser.get_summary()

            assert isinstance(summary, RunSummary)
            assert summary.total_tests > 0
            assert 0 <= summary.success_rate <= 100
        finally:
            path.unlink()

    def test_get_failed_tests(self, tmp_path):
        """Test getting failed tests."""
        results_path = tmp_path / "run_results.json"
        data = create_mock_run_results(
            test_results=[
                {"unique_id": "test.p.test_1", "status": "pass"},
                {"unique_id": "test.p.test_2", "status": "fail", "failures": 5},
            ]
        )
        results_path.write_text(json.dumps(data))

        parser = RunResultsParser(results_path)
        failed = parser.get_failed_tests()
        assert len(failed) == 1
        assert failed[0].unique_id == "test.p.test_2"

    def test_get_passed_tests(self, tmp_path):
        """Test getting passed tests."""
        results_path = tmp_path / "run_results.json"
        data = create_mock_run_results(
            test_results=[
                {"unique_id": "test.p.test_1", "status": "pass"},
                {"unique_id": "test.p.test_2", "status": "fail"},
            ]
        )
        results_path.write_text(json.dumps(data))

        parser = RunResultsParser(results_path)
        passed = parser.get_passed_tests()
        assert len(passed) == 1
        assert passed[0].unique_id == "test.p.test_1"

    def test_to_check_results(self, tmp_path):
        """Test converting to CheckResult format."""
        results = MockRunResults()
        path = results.create_temp_file()

        try:
            parser = RunResultsParser(path)
            check_results = parser.to_check_results()

            assert isinstance(check_results, list)
            assert all("status" in r for r in check_results)
        finally:
            path.unlink()


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_passed_property(self):
        """Test passed property."""
        from truthound_dbt.parsers.results import TestStatus

        result = TestResult(
            unique_id="test.p.test",
            status=TestStatus.PASS,
        )
        assert result.passed is True

        result = TestResult(
            unique_id="test.p.test",
            status=TestStatus.FAIL,
        )
        assert result.passed is False

    def test_failed_property(self):
        """Test failed property."""
        from truthound_dbt.parsers.results import TestStatus

        result = TestResult(
            unique_id="test.p.test",
            status=TestStatus.FAIL,
        )
        assert result.failed is True

        result = TestResult(
            unique_id="test.p.test",
            status=TestStatus.ERROR,
        )
        assert result.failed is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        from truthound_dbt.parsers.results import TestStatus

        result = TestResult(
            unique_id="test.p.test",
            status=TestStatus.PASS,
            execution_time=1.5,
        )
        data = result.to_dict()

        assert data["unique_id"] == "test.p.test"
        assert data["status"] == "pass"
        assert data["execution_time"] == 1.5
        assert data["passed"] is True


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_success_rate(self):
        """Test success rate calculation."""
        summary = RunSummary(
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
        )
        assert summary.success_rate == 80.0

    def test_success_rate_zero_tests(self):
        """Test success rate with no tests."""
        summary = RunSummary(total_tests=0)
        assert summary.success_rate == 0.0

    def test_all_passed_true(self):
        """Test all_passed when all pass."""
        summary = RunSummary(
            total_tests=5,
            passed_tests=5,
            failed_tests=0,
            errored_tests=0,
        )
        assert summary.all_passed is True

    def test_all_passed_false(self):
        """Test all_passed when some fail."""
        summary = RunSummary(
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
        )
        assert summary.all_passed is False
