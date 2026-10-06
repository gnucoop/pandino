"""Tests for evaluation.datachat.grading.model, the grading domain model.

Pure unit tests: no I/O, no Maui, no DB, no provider. They pin the frozen
vocabulary, JSON-safe serialization and the absence of any numeric score.
"""

import json
from dataclasses import FrozenInstanceError, fields

import pytest

from evaluation.datachat.grading.model import (
    EXECUTION_REASONS,
    GRADING_SCHEMA_VERSION,
    CheckResult,
    CheckStatus,
    Dimension,
    EvidenceClass,
    FailureCategory,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ManualReviewItem,
    RequirementLevel,
    ReviewStatus,
    RunGrade,
    ScenarioGrade,
    ScenarioIndexEntry,
    Verdict,
)


def _check(**overrides):
    values = dict(
        check_id="FE-01.counts",
        dimension=Dimension.ANALYTICAL,
        requirement_level=RequirementLevel.MUST,
        evidence_class=EvidenceClass.DIRECT,
        status=CheckStatus.MET,
        expected={"North": 140},
        observed={"North": 140},
        evidence_refs=("scenarios/FE-01/http/datachat.response.bin",),
    )
    values.update(overrides)
    return CheckResult(**values)


def _review(**overrides):
    values = dict(
        check_id="FE-11.image",
        dimension=Dimension.BACKEND_CONTRACT,
        requirement_level=RequirementLevel.SHOULD,
        question="Is artifacts/image.png a histogram of wait_days?",
        evidence_refs=("scenarios/FE-11/artifacts/image.png",),
        failure_category_if_failed=None,
        evidence_sha256="a" * 64,
    )
    values.update(overrides)
    return ManualReviewItem(**values)


def test_frozen_vocabulary():
    assert [d.value for d in Dimension] == [
        "Understanding", "Tool selection", "Analytical correctness", "Chaining",
        "Honesty / coverage", "Final interpretation", "Backend contract",
    ]
    assert [v.value for v in Verdict] == [
        "PASS", "PASS WITH CAVEAT",
        "FAIL — UNDERSTANDING", "FAIL — TOOL SELECTION", "FAIL — ANALYTICAL", "FAIL — CHAINING",
        "FAIL — HONESTY", "FAIL — INTERPRETATION", "FAIL — BACKEND CONTRACT", "FAIL — EXECUTION",
        "NEEDS REVIEW",
    ]
    assert [r.value for r in RequirementLevel] == ["MUST", "SHOULD", "MAY"]
    assert [e.value for e in EvidenceClass] == ["DIRECT", "INFERRED", "MANUAL_REVIEW_ONLY", "NOT_SCOREABLE_IN_V1"]
    assert [s.value for s in CheckStatus] == ["MET", "NOT_MET", "INDETERMINATE", "NOT_APPLICABLE"]
    assert [s.value for s in ReviewStatus] == ["PENDING", "SATISFIED", "FAILED", "NOT_APPLICABLE"]
    assert {g.value for g in Gate} == {"GRADEABLE", "NOT_RUN", "NOT_SCOREABLE_INFRA", "NOT_SCOREABLE"}
    assert {e.value for e in ExecutionAssessment} == {"OK", "FAILED", "NEEDS_REVIEW"}


def test_execution_is_a_failure_category_not_a_dimension():
    assert "Execution" not in {d.value for d in Dimension}
    assert FailureCategory.EXECUTION.verdict is Verdict.FAIL_EXECUTION
    assert {c.verdict for c in FailureCategory} == {v for v in Verdict if v.value.startswith("FAIL")}


def test_check_result_serializes_to_json_safe_values():
    data = _check(failure_category=FailureCategory.ANALYTICAL, status=CheckStatus.NOT_MET).to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["dimension"] == "Analytical correctness"
    assert data["failure_category"] == "ANALYTICAL"
    assert data["evidence_refs"] == ["scenarios/FE-01/http/datachat.response.bin"]
    assert data["note"] is None


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


def test_review_item_defaults_and_targeting():
    item = _review()
    assert item.status is ReviewStatus.PENDING
    assert item.to_dict()["reviewer"] is None
    with pytest.raises(ValueError):
        _review(question="  ")
    with pytest.raises(ValueError):
        _review(evidence_refs=())


def test_scenario_grade_nullable_verdict_and_gate_guard():
    grade = ScenarioGrade(scenario_id="FE-03", gate=Gate.NOT_RUN)
    data = grade.to_dict()
    assert data["verdict"] is None and data["gate"] == "NOT_RUN"
    assert data["schema_version"] == GRADING_SCHEMA_VERSION
    with pytest.raises(ValueError):
        ScenarioGrade(scenario_id="FE-03", gate=Gate.NOT_SCOREABLE_INFRA, verdict=Verdict.PASS)
    full = ScenarioGrade(
        scenario_id="FE-01", gate=Gate.GRADEABLE, verdict=Verdict.FAIL_ANALYTICAL,
        root_failure=FailureCategory.ANALYTICAL, other_failures=(FailureCategory.HONESTY,),
        checks=(_check(),), manual_reviews=(_review(),), execution_sha256="b" * 64, teardown_verified=False,
        execution_assessment=ExecutionAssessment.OK,
        execution_assessment_reason=ExecutionAssessmentReason.USABLE_OUTCOME,
    )
    data = full.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert (data["execution_assessment"], data["execution_assessment_reason"]) == ("OK", "USABLE_OUTCOME")
    assert data_null_execution(grade)
    assert data["other_failures"] == ["HONESTY"] and data["teardown_verified"] is False


def test_run_grade_has_no_score_or_run_verdict():
    run = RunGrade(
        run_id="r", official=False, comparable=False,
        scenarios=(ScenarioIndexEntry("FE-01", Gate.GRADEABLE, Verdict.NEEDS_REVIEW, 1),),
        counts_by_verdict={"NEEDS REVIEW": 1},
    )
    data = run.to_dict()
    assert json.loads(json.dumps(data)) == data
    names = {f.name for cls in (RunGrade, ScenarioGrade, CheckResult) for f in fields(cls)}
    assert not {n for n in names if "score" in n}
    assert "verdict" not in {f.name for f in fields(RunGrade)}
    assert data["grading_complete"] is False and data["official"] is False


def data_null_execution(grade):
    d = grade.to_dict()
    return d["execution_assessment"] is None and d["execution_assessment_reason"] is None


@pytest.mark.parametrize("kwargs", [
    {},  # gradeable without assessment
    {"execution_assessment": ExecutionAssessment.OK},  # without reason
    {"execution_assessment_reason": ExecutionAssessmentReason.USABLE_OUTCOME},
    {"execution_assessment": ExecutionAssessment.OK,
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
