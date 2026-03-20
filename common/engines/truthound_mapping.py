"""Truthound 3.x result mapping helpers.

This module is the only place that understands Truthound-native result shapes
and converts them into orchestration DTOs from ``common.base``.
"""

from __future__ import annotations

from typing import Any

from common.base import (
    CheckResult,
    CheckStatus,
    ColumnProfile,
    LearnResult,
    LearnStatus,
    LearnedRule,
    ProfileResult,
    ProfileStatus,
    Severity,
    ValidationFailure,
)


SEVERITY_MAPPING: dict[str, Severity] = {
    "critical": Severity.CRITICAL,
    "high": Severity.ERROR,
    "medium": Severity.WARNING,
    "low": Severity.INFO,
    "info": Severity.INFO,
}


def _severity_to_common(value: Any) -> Severity:
    """Map Truthound severity values into orchestration severities."""

    raw = getattr(value, "value", value)
    return SEVERITY_MAPPING.get(str(raw).lower(), Severity.ERROR)


def _issue_message(issue: Any) -> str:
    details = getattr(issue, "details", None)
    if details:
        return str(details)

    issue_type = getattr(issue, "issue_type", "validation_issue")
    column = getattr(issue, "column", "_table_")
    return f"{issue_type} detected on {column}"


def _issue_metadata(issue: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("details", "expected", "actual", "validator_name"):
        value = getattr(issue, key, None)
        if value is not None:
            metadata[key] = value
    result = getattr(issue, "result", None)
    if result is not None and hasattr(result, "to_dict"):
        metadata["result"] = result.to_dict()
    exception_info = getattr(issue, "exception_info", None)
    if exception_info is not None and hasattr(exception_info, "to_dict"):
        metadata["exception_info"] = exception_info.to_dict()
    return metadata


def _map_issue(issue: Any, *, total_count: int) -> ValidationFailure:
    return ValidationFailure(
        rule_name=str(
            getattr(issue, "validator_name", None)
            or getattr(issue, "issue_type", None)
            or "validation_issue"
        ),
        column=getattr(issue, "column", None),
        message=_issue_message(issue),
        severity=_severity_to_common(getattr(issue, "severity", "high")),
        failed_count=int(getattr(issue, "count", 0)),
        total_count=total_count,
        sample_values=tuple(getattr(issue, "sample_values", None) or ()),
        metadata=_issue_metadata(issue),
    )


def _map_execution_issue(issue: Any) -> ValidationFailure:
    return ValidationFailure(
        rule_name=str(getattr(issue, "check_name", "execution_issue")),
        message=str(getattr(issue, "message", "Execution issue")),
        severity=Severity.ERROR,
        failed_count=1,
        total_count=1,
        metadata={
            "exception_type": getattr(issue, "exception_type", None),
            "failure_category": getattr(issue, "failure_category", None),
            "retry_count": getattr(issue, "retry_count", 0),
        },
    )


def _classify_check(check: Any) -> CheckStatus:
    issues = tuple(getattr(check, "issues", ()) or ())
    if not issues:
        return CheckStatus.PASSED

    max_severity = max(_severity_to_common(getattr(issue, "severity", "high")) for issue in issues)
    if max_severity >= Severity.ERROR:
        return CheckStatus.FAILED
    return CheckStatus.WARNING


def map_validation_run_result(
    run_result: Any,
    *,
    engine_name: str,
    execution_time_ms: float,
) -> CheckResult:
    """Convert Truthound ``ValidationRunResult`` into orchestration ``CheckResult``."""

    failures = [
        _map_issue(issue, total_count=int(getattr(run_result, "row_count", 0)))
        for issue in getattr(run_result, "issues", ())
    ]
    failures.extend(
        _map_execution_issue(issue) for issue in getattr(run_result, "execution_issues", ())
    )

    passed_count = 0
    failed_count = 0
    warning_count = 0

    for check in getattr(run_result, "checks", ()):
        check_status = _classify_check(check)
        if check_status == CheckStatus.PASSED:
            passed_count += 1
        elif check_status == CheckStatus.WARNING:
            warning_count += 1
        else:
            failed_count += 1

    execution_issue_count = len(getattr(run_result, "execution_issues", ()))
    if execution_issue_count:
        failed_count += execution_issue_count

    if failed_count > 0:
        status = CheckStatus.FAILED
    elif warning_count > 0:
        status = CheckStatus.WARNING
    else:
        status = CheckStatus.PASSED

    result_format = getattr(getattr(run_result, "result_format", None), "value", None)

    return CheckResult(
        status=status,
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=warning_count,
        failures=tuple(failures),
        execution_time_ms=execution_time_ms,
        metadata={
            "engine": engine_name,
            "source": getattr(run_result, "source", "unknown"),
            "suite_name": getattr(run_result, "suite_name", "default"),
            "row_count": getattr(run_result, "row_count", 0),
            "column_count": getattr(run_result, "column_count", 0),
            "run_id": getattr(run_result, "run_id", None),
            "execution_mode": getattr(run_result, "execution_mode", None),
            "result_format": result_format,
            "execution_issue_count": execution_issue_count,
            "metadata": dict(getattr(run_result, "metadata", {}) or {}),
        },
    )


def map_profile_report(
    profile_report: Any,
    *,
    engine_name: str,
    execution_time_ms: float,
    columns_filter: tuple[str, ...] | None = None,
) -> ProfileResult:
    """Convert Truthound ``ProfileReport`` into orchestration ``ProfileResult``."""

    report_dict = profile_report.to_dict() if hasattr(profile_report, "to_dict") else dict(profile_report)
    column_profiles: list[ColumnProfile] = []

    for col_data in report_dict.get("columns", []):
        column_name = col_data.get("name", "unknown")
        if columns_filter and column_name not in columns_filter:
            continue

        null_pct_raw = str(col_data.get("null_pct", "0")).replace("%", "")
        unique_pct_raw = str(col_data.get("unique_pct", "0")).replace("%", "")
        null_pct = float(null_pct_raw) if null_pct_raw else 0.0
        unique_pct = float(unique_pct_raw) if unique_pct_raw else 0.0

        row_count = int(report_dict.get("row_count", 0))
        min_value = col_data.get("min")
        max_value = col_data.get("max")
        if min_value == "-":
            min_value = None
        if max_value == "-":
            max_value = None

        column_profiles.append(
            ColumnProfile(
                column_name=column_name,
                dtype=str(col_data.get("dtype", "unknown")),
                null_count=int(row_count * null_pct / 100) if row_count > 0 else 0,
                null_percentage=null_pct,
                unique_count=int(row_count * unique_pct / 100) if row_count > 0 else 0,
                unique_percentage=unique_pct,
                min_value=min_value,
                max_value=max_value,
            )
        )

    return ProfileResult(
        status=ProfileStatus.COMPLETED,
        row_count=int(report_dict.get("row_count", 0)),
        column_count=len(column_profiles),
        columns=tuple(column_profiles),
        execution_time_ms=execution_time_ms,
        metadata={
            "engine": engine_name,
            "source": report_dict.get("source", "unknown"),
            "size_bytes": report_dict.get("size_bytes"),
        },
    )


def map_schema_to_learn_result(
    schema: Any,
    *,
    engine_name: str,
    execution_time_ms: float,
    columns_filter: tuple[str, ...] | None = None,
    confidence_threshold: float,
) -> LearnResult:
    """Convert Truthound ``Schema`` into orchestration ``LearnResult``."""

    schema_dict = schema.to_dict() if hasattr(schema, "to_dict") else dict(schema)
    row_count = int(schema_dict.get("row_count", 0) or 0)
    columns_data = schema_dict.get("columns", {})
    learned_rules: list[LearnedRule] = []

    for column_name, column_schema in columns_data.items():
        if columns_filter and column_name not in columns_filter:
            continue

        dtype = column_schema.get("dtype")
        if dtype:
            learned_rules.append(
                LearnedRule(
                    rule_type="dtype",
                    column=column_name,
                    parameters={"dtype": dtype},
                    confidence=1.0,
                    sample_size=row_count,
                )
            )

        null_ratio = float(column_schema.get("null_ratio", 0.0) or 0.0)
        if null_ratio < (1 - confidence_threshold):
            learned_rules.append(
                LearnedRule(
                    rule_type="not_null",
                    column=column_name,
                    confidence=1 - null_ratio,
                    sample_size=row_count,
                )
            )

        unique_ratio = float(column_schema.get("unique_ratio", 0.0) or 0.0)
        if unique_ratio >= confidence_threshold:
            learned_rules.append(
                LearnedRule(
                    rule_type="unique",
                    column=column_name,
                    confidence=unique_ratio,
                    sample_size=row_count,
                )
            )

        min_value = column_schema.get("min_value")
        max_value = column_schema.get("max_value")
        if min_value is not None and max_value is not None:
            learned_rules.append(
                LearnedRule(
                    rule_type="in_range",
                    column=column_name,
                    parameters={"min": min_value, "max": max_value},
                    confidence=1.0,
                    sample_size=row_count,
                )
            )

        allowed_values = column_schema.get("allowed_values")
        if allowed_values:
            learned_rules.append(
                LearnedRule(
                    rule_type="in_set",
                    column=column_name,
                    parameters={"values": allowed_values},
                    confidence=1.0,
                    sample_size=row_count,
                )
            )

        min_length = column_schema.get("min_length")
        if min_length is not None:
            learned_rules.append(
                LearnedRule(
                    rule_type="min_length",
                    column=column_name,
                    parameters={"value": min_length},
                    confidence=1.0,
                    sample_size=row_count,
                )
            )

        max_length = column_schema.get("max_length")
        if max_length is not None:
            learned_rules.append(
                LearnedRule(
                    rule_type="max_length",
                    column=column_name,
                    parameters={"value": max_length},
                    confidence=1.0,
                    sample_size=row_count,
                )
            )

    return LearnResult(
        status=LearnStatus.COMPLETED,
        rules=tuple(learned_rules),
        columns_analyzed=len(columns_data),
        execution_time_ms=execution_time_ms,
        metadata={
            "engine": engine_name,
            "row_count": row_count,
            "schema_version": schema_dict.get("version", "1.0"),
            "column_names": list(columns_data.keys()),
        },
    )
