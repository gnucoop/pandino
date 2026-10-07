"""Tests for evaluation.datachat.grading.model, the grading domain model.

Pure unit tests: no I/O, no Maui, no DB, no provider. They pin the frozen
technical vocabulary (grading schema 2), JSON-safe serialization and the
absence of any numeric score or semantic/review record.
"""

import json
from dataclasses import FrozenInstanceError, fields

import pytest

from evaluation.datachat.grading.model import (
    EXECUTION_REASONS,
    GRADING_SCHEMA_VERSION,
    CheckResult,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    RunGrade,
    ScenarioGrade,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
)

F, T = TechnicalFailure, TechnicalOutcome


def _check(**overrides):
    values = dict(
        check_id="fe01.chart_present",
        evidence_class=EvidenceClass.DIRECT,
        status=CheckStatus.MET,
        expected=">= 1",
        observed=1,
        evidence_refs=("scenarios/FE-01/http/datachat.response.bin",),
    )
    values.update(overrides)
    return CheckResult(**values)


def test_frozen_vocabulary():
    assert GRADING_SCHEMA_VERSION == 2
    assert [o.value for o in TechnicalOutcome] == ["TECHNICAL PASS", "FAIL", "INDETERMINATE"]
    assert [f.value for f in TechnicalFailure] == ["EXECUTION", "RESPONSE CONTRACT", "TOOL EXECUTION", "ARTIFACT"]
    assert [e.value for e in EvidenceClass] == ["DIRECT", "INFERRED"]
    assert [s.value for s in CheckStatus] == ["MET", "NOT_MET", "INDETERMINATE", "NOT_APPLICABLE"]
    assert {g.value for g in Gate} == {"GRADEABLE", "NOT_RUN", "NOT_SCOREABLE_INFRA", "NOT_SCOREABLE"}
    assert [e.value for e in ExecutionAssessment] == ["OK", "FAILED", "RESPONSE_CONTRACT_VIOLATION", "INDETERMINATE"]


def test_no_semantic_or_review_vocabulary_remains():
    import evaluation.datachat.grading.model as m

    for name in ("Dimension", "FailureCategory", "Verdict", "RequirementLevel", "ReviewStatus", "ManualReviewItem"):
        assert not hasattr(m, name), name
    names = {f.name for cls in (CheckResult, ScenarioGrade, ScenarioIndexEntry, RunGrade) for f in fields(cls)}
    assert not names & {"dimension", "requirement_level", "verdict", "root_failure", "other_failures", "caveats",
                        "manual_reviews", "pending_reviews", "grading_complete", "counts_by_verdict"}


def test_check_result_serializes_to_json_safe_values():
    data = _check(failure=F.ARTIFACT, status=CheckStatus.NOT_MET).to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data == {"check_id": "fe01.chart_present", "evidence_class": "DIRECT", "status": "NOT_MET",
                    "expected": ">= 1", "observed": 1,
                    "evidence_refs": ["scenarios/FE-01/http/datachat.response.bin"],
                    "failure": "ARTIFACT", "note": None}


def test_check_failure_set_exactly_on_not_met():
    with pytest.raises(ValueError):
        _check(status=CheckStatus.NOT_MET)
    for status in (CheckStatus.MET, CheckStatus.INDETERMINATE, CheckStatus.NOT_APPLICABLE):
        with pytest.raises(ValueError):
            _check(status=status, failure=F.ARTIFACT)


@pytest.mark.parametrize("value", [b"raw", {"nested": {"x": 1}}, [[1]], object()])
def test_check_result_rejects_large_or_raw_values(value):
    with pytest.raises(ValueError):
        _check(observed=value)


@pytest.mark.parametrize("ref", ["/abs/path", "../escape", "scenarios/../../x", ""])
def test_evidence_refs_must_be_run_relative(ref):
    with pytest.raises(ValueError):
        _check(evidence_refs=(ref,))


def test_records_are_immutable():
    with pytest.raises(FrozenInstanceError):
        _check().status = CheckStatus.NOT_MET


_OK = dict(execution_assessment=ExecutionAssessment.OK,
           execution_assessment_reason=ExecutionAssessmentReason.USABLE_OUTCOME)


def test_scenario_grade_nullable_outcome_and_gate_guard():
    grade = ScenarioGrade(scenario_id="FE-03", gate=Gate.NOT_RUN)
    data = grade.to_dict()
    assert data["outcome"] is None and data["failures"] == [] and data["gate"] == "NOT_RUN"
    assert data["schema_version"] == GRADING_SCHEMA_VERSION
    with pytest.raises(ValueError):
        ScenarioGrade(scenario_id="FE-03", gate=Gate.NOT_SCOREABLE_INFRA, outcome=T.TECHNICAL_PASS)
    full = ScenarioGrade(
        scenario_id="FE-01", gate=Gate.GRADEABLE, outcome=T.FAIL, failures=(F.TOOL_EXECUTION, F.ARTIFACT),
        checks=(_check(),), execution_sha256="b" * 64, teardown_verified=False, **_OK,
    )
    data = full.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert (data["execution_assessment"], data["execution_assessment_reason"]) == ("OK", "USABLE_OUTCOME")
    assert data_null_execution(grade)
    assert data["outcome"] == "FAIL" and data["failures"] == ["TOOL EXECUTION", "ARTIFACT"]
    assert data["teardown_verified"] is False
    assert full.label == "FAIL — TOOL EXECUTION, ARTIFACT"


@pytest.mark.parametrize("outcome,failures", [
    (T.FAIL, ()), (T.TECHNICAL_PASS, (F.ARTIFACT,)), (T.INDETERMINATE, (F.ARTIFACT,)),
    (T.FAIL, (F.ARTIFACT, F.ARTIFACT)), (None, ()),
])
def test_gradeable_outcome_and_failures_must_agree(outcome, failures):
    with pytest.raises(ValueError):
        ScenarioGrade(scenario_id="FE-01", gate=Gate.GRADEABLE, outcome=outcome, failures=failures, **_OK)


def test_run_grade_has_no_score_or_run_outcome():
    run = RunGrade(
        run_id="r", official=False, comparable=False,
        scenarios=(ScenarioIndexEntry("FE-01", Gate.GRADEABLE, T.FAIL, (F.ARTIFACT,)),),
        counts_by_outcome={"FAIL": 1}, counts_by_failure={"ARTIFACT": 1},
    )
    data = run.to_dict()
    assert json.loads(json.dumps(data)) == data
    names = {f.name for cls in (RunGrade, ScenarioGrade, CheckResult) for f in fields(cls)}
    assert not {n for n in names if "score" in n}
    assert "outcome" not in {f.name for f in fields(RunGrade)}
    assert data["scenarios"] == [{"scenario_id": "FE-01", "gate": "GRADEABLE", "outcome": "FAIL",
                                  "failures": ["ARTIFACT"]}]
    assert data["schema_version"] == 2 and data["official"] is False
    with pytest.raises(ValueError):
        ScenarioIndexEntry("FE-01", Gate.GRADEABLE, T.FAIL)


def data_null_execution(grade):
    d = grade.to_dict()
    return d["execution_assessment"] is None and d["execution_assessment_reason"] is None


@pytest.mark.parametrize("kwargs", [
    {"outcome": T.TECHNICAL_PASS},  # gradeable without assessment
    {"outcome": T.TECHNICAL_PASS, "execution_assessment": ExecutionAssessment.OK},  # without reason
    {"outcome": T.TECHNICAL_PASS, "execution_assessment_reason": ExecutionAssessmentReason.USABLE_OUTCOME},
    {"outcome": T.TECHNICAL_PASS, "execution_assessment": ExecutionAssessment.OK,
     "execution_assessment_reason": ExecutionAssessmentReason.ADAPTER_FALLBACK},  # impossible pair
])
def test_gradeable_grade_requires_valid_execution_provenance(kwargs):
    with pytest.raises(ValueError):
        ScenarioGrade(scenario_id="FE-01", gate=Gate.GRADEABLE, **kwargs)


@pytest.mark.parametrize("gate", [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE])
def test_non_gradeable_grade_rejects_execution_provenance(gate):
    assert data_null_execution(ScenarioGrade(scenario_id="FE-01", gate=gate))
    with pytest.raises(ValueError):
        ScenarioGrade(scenario_id="FE-01", gate=gate, execution_assessment=ExecutionAssessment.FAILED,
                      execution_assessment_reason=ExecutionAssessmentReason.ERROR_OUTCOME)


def test_every_reason_belongs_to_exactly_one_assessment():
    owners = [a for r in ExecutionAssessmentReason for a, rs in EXECUTION_REASONS.items() if r in rs]
    assert len(owners) == len(ExecutionAssessmentReason) and set(EXECUTION_REASONS) == set(ExecutionAssessment)
