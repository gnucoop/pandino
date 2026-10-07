"""Tests for evaluation.datachat.grading.outcome: technical outcome construction.

Offline and pure: synthetic CheckResult, plus small integration runs through
the catalogue with synthetic evidence.
"""

import json

import pytest

from evaluation.datachat.grading import catalogue as cat
from evaluation.datachat.grading import outcome as o
from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    TechnicalFailure,
    TechnicalOutcome,
)
from tests.test_fe_grading_catalogue import CM_EVENT, KW_EVENT, REGIONS, chart, evidence, table, text

MET, NOT_MET, IND, NA = (CheckStatus.MET, CheckStatus.NOT_MET, CheckStatus.INDETERMINATE,
                         CheckStatus.NOT_APPLICABLE)
DIRECT, INFERRED = EvidenceClass.DIRECT, EvidenceClass.INFERRED
F, T = TechnicalFailure, TechnicalOutcome
EA, RS = ExecutionAssessment, ExecutionAssessmentReason
REF = ("scenarios/FE-01/http/datachat.response.bin",)


def chk(cid, status=MET, failure=F.ARTIFACT, cls=DIRECT):
    return CheckResult(cid, cls, status, evidence_refs=REF, failure=failure if status is NOT_MET else None)


def build(checks=(), assessment=EA.OK, reason=RS.USABLE_OUTCOME):
    return o.build_scenario_grade(scenario_id="FE-01", gate=Gate.GRADEABLE, execution_sha256="e" * 64,
                                  teardown_verified=True, checks=checks, execution_assessment=assessment,
                                  execution_assessment_reason=reason)


# --- Core outcome table ------------------------------------------------------------------


def test_technical_pass():
    g = build([chk("a"), chk("b", NA)])
    assert (g.outcome, g.failures, g.label) == (T.TECHNICAL_PASS, (), "TECHNICAL PASS")


def test_no_checks_and_usable_execution_is_technical_pass():
    assert build().outcome is T.TECHNICAL_PASS


@pytest.mark.parametrize("failure", list(F))
def test_each_check_failure_category(failure):
    g = build([chk("a", NOT_MET, failure)])
    assert (g.outcome, g.failures) == (T.FAIL, (failure,))
    assert g.label == f"FAIL — {failure.value}"


def test_indeterminate_check():
    g = build([chk("a"), chk("b", IND)])
    assert (g.outcome, g.failures) == (T.INDETERMINATE, ())


def test_inferred_negative_alone_is_indeterminate_never_fail():
    g = build([chk("i", NOT_MET, F.TOOL_EXECUTION, INFERRED)])
    assert (g.outcome, g.failures) == (T.INDETERMINATE, ())


def test_inferred_negative_never_joins_established_failures():
    g = build([chk("i", NOT_MET, F.TOOL_EXECUTION, INFERRED), chk("d", NOT_MET, F.ARTIFACT)])
    assert (g.outcome, g.failures) == (T.FAIL, (F.ARTIFACT,))


def test_failure_wins_over_indeterminate():
    g = build([chk("a", IND), chk("d", NOT_MET, F.RESPONSE_CONTRACT)])
    assert (g.outcome, g.failures) == (T.FAIL, (F.RESPONSE_CONTRACT,))


def test_multiple_failures_all_recorded_unranked_and_deduplicated():
    g = build([chk("a", NOT_MET, F.ARTIFACT), chk("b", NOT_MET, F.TOOL_EXECUTION), chk("c", NOT_MET, F.ARTIFACT),
               chk("d", NOT_MET, F.RESPONSE_CONTRACT)])
    # Enum declaration order: a stable listing order, not a primary failure.
    assert g.failures == (F.RESPONSE_CONTRACT, F.TOOL_EXECUTION, F.ARTIFACT)
    assert g.label == "FAIL — RESPONSE CONTRACT, TOOL EXECUTION, ARTIFACT"
    d = g.to_dict()
    assert "root_failure" not in d and "other_failures" not in d


# --- Execution assessment -----------------------------------------------------------------


def test_execution_failed_is_fail_execution_and_keeps_check_failures():
    assert build([chk("a")], EA.FAILED, RS.ERROR_OUTCOME).failures == (F.EXECUTION,)
    g = build([chk("a", NOT_MET, F.ARTIFACT)], EA.FAILED, RS.EMPTY_OUTCOME)
    assert (g.outcome, g.failures) == (T.FAIL, (F.EXECUTION, F.ARTIFACT))


@pytest.mark.parametrize("reason", [RS.HTTP_NON_2XX, RS.UNPARSEABLE_RESPONSE])
def test_response_contract_violation_is_fail_response_contract(reason):
    g = build([chk("a", IND)], EA.RESPONSE_CONTRACT_VIOLATION, reason)
    assert (g.outcome, g.failures) == (T.FAIL, (F.RESPONSE_CONTRACT,))


@pytest.mark.parametrize("reason", [RS.MISSING_RESPONSE, RS.CONFLICTING_KINDS, RS.RUNTIME_STATUS_CONFLICT,
                                    RS.ADAPTER_FALLBACK])
def test_ambiguous_execution_is_indeterminate_unless_a_failure_is_established(reason):
    assert build([chk("a")], EA.INDETERMINATE, reason).outcome is T.INDETERMINATE
    g = build([chk("a", NOT_MET, F.ARTIFACT)], EA.INDETERMINATE, reason)
    assert (g.outcome, g.failures) == (T.FAIL, (F.ARTIFACT,))


def test_execution_assessment_cannot_be_omitted():
    with pytest.raises(TypeError, match="execution_assessment"):
        o.build_scenario_grade(scenario_id="FE-01", gate=Gate.GRADEABLE, execution_sha256="e" * 64,
                               teardown_verified=True, checks=[chk("m")])
    with pytest.raises(TypeError, match="execution_assessment_reason"):
        o.build_scenario_grade(scenario_id="FE-01", gate=Gate.GRADEABLE, execution_sha256="e" * 64,
                               teardown_verified=True, execution_assessment=EA.OK)
    with pytest.raises(o.OutcomeBuildError):
        build(assessment=None, reason=None)


@pytest.mark.parametrize("assessment,reason", [
    (EA.OK, RS.ADAPTER_FALLBACK), (EA.FAILED, RS.USABLE_OUTCOME), (EA.INDETERMINATE, RS.HTTP_NON_2XX),
    (EA.RESPONSE_CONTRACT_VIOLATION, RS.MISSING_RESPONSE), (EA.OK, None), (None, RS.USABLE_OUTCOME),
    (EA.OK, "USABLE_OUTCOME"), ("FAILED", RS.ERROR_OUTCOME),
])
def test_impossible_assessment_reason_pairs_rejected(assessment, reason):
    with pytest.raises(o.OutcomeBuildError):
        build(assessment=assessment, reason=reason)


def test_execution_provenance_recorded_verbatim():
    g = build([chk("m")], EA.INDETERMINATE, RS.ADAPTER_FALLBACK)
    d = g.to_dict()
    assert (d["execution_assessment"], d["execution_assessment_reason"]) == ("INDETERMINATE", "ADAPTER_FALLBACK")


# --- Invariants, gates, purity ------------------------------------------------------------


def test_invariant_violations():
    with pytest.raises(o.OutcomeBuildError):
        build([chk("x"), chk("x")])
    with pytest.raises(ValueError):
        CheckResult("x", DIRECT, NOT_MET)  # NOT_MET needs a failure category
    with pytest.raises(ValueError):
        CheckResult("x", DIRECT, MET, failure=F.ARTIFACT)


@pytest.mark.parametrize("gate", [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE])
def test_non_gradeable_gates_get_no_outcome(gate):
    g = o.build_scenario_grade(scenario_id="FE-03", gate=gate, execution_sha256=None, teardown_verified=None,
                               execution_assessment=None, execution_assessment_reason=None)
    assert (g.outcome, g.failures, g.checks, g.label) == (None, (), (), None)
    assert g.execution_assessment is None and g.execution_assessment_reason is None
    with pytest.raises(o.OutcomeBuildError):
        o.build_scenario_grade(scenario_id="FE-03", gate=gate, execution_sha256=None, teardown_verified=None,
                               execution_assessment=None, execution_assessment_reason=None,
                               checks=[chk("m")])
    with pytest.raises(o.OutcomeBuildError, match="admits no execution assessment"):
        o.build_scenario_grade(scenario_id="FE-03", gate=gate, execution_sha256=None, teardown_verified=None,
                               execution_assessment=EA.OK, execution_assessment_reason=RS.USABLE_OUTCOME)


def test_deterministic_and_non_mutating():
    checks = [chk("m"), chk("s", NOT_MET, F.ARTIFACT)]
    snapshot = [c.to_dict() for c in checks]
    first, second = build(checks), build(checks)
    assert first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict(), sort_keys=True)  # JSON-serializable
    assert [c.to_dict() for c in checks] == snapshot
    g = first.to_dict()
    assert set(g) == {"scenario_id", "gate", "outcome", "failures", "checks", "execution_assessment",
                      "execution_assessment_reason", "execution_sha256", "teardown_verified", "activity",
                      "schema_version"}
    assert g["schema_version"] == 2 and g["outcome"] == "FAIL" and g["failures"] == ["ARTIFACT"]


# --- Catalogue integration ------------------------------------------------------------------


def grade_from_catalogue(ev):
    result = cat.grade_scenario_checks(ev)
    return o.build_scenario_grade(scenario_id=ev.scenario_id, gate=ev.gate, execution_sha256=ev.execution_sha256,
                                  teardown_verified=ev.teardown_verified, checks=result.checks,
                                  execution_assessment=EA.OK, execution_assessment_reason=RS.USABLE_OUTCOME)


def test_fe01_pass_and_missing_chart():
    g = grade_from_catalogue(evidence("FE-01", text("Ecco.", [chart("bar", list(REGIONS), list(REGIONS.values()))])))
    assert g.outcome is T.TECHNICAL_PASS
    g = grade_from_catalogue(evidence("FE-01", text("Ecco.")))
    assert g.label == "FAIL — ARTIFACT"


def test_scenario_without_technical_requirement_passes():
    ev = evidence("FE-05", text("Associazione negativa moderata (Spearman -0,58)."),
                  code=["correlation(col_x='wait_days', col_y='satisfaction', method='spearman')"])
    g = grade_from_catalogue(ev)
    assert all(c.check_id.startswith("generic.") for c in g.checks) and g.outcome is T.TECHNICAL_PASS


IMAGE = {"type": "image", "value": "b'iVBORw0KGgo='", "charts": []}


def test_fe11_response_contract_vs_artifact():
    assert grade_from_catalogue(evidence("FE-11", text("La media è 10."))).label == "FAIL — RESPONSE CONTRACT"
    assert grade_from_catalogue(evidence("FE-11", IMAGE, image_meta={"decode_status": "NOT_PNG"})).label \
        == "FAIL — ARTIFACT"
    assert grade_from_catalogue(evidence("FE-11", IMAGE, image_meta={"decode_status": "OK"})).outcome \
        is T.TECHNICAL_PASS
    assert grade_from_catalogue(evidence("FE-11", IMAGE)).outcome is T.INDETERMINATE


def test_fe14_semantic_capability():
    counts = table([{"category": "Attesa", "count": 96}, {"category": "Personale", "count": 68}])
    g = grade_from_catalogue(evidence("FE-14", counts, events=[KW_EVENT, CM_EVENT], usage_rows=2))
    assert g.outcome is T.TECHNICAL_PASS
    # No provider call (DIRECT) fails; a missing classify_match call alone (INFERRED) is INDETERMINATE.
    g = grade_from_catalogue(evidence("FE-14", counts, code=["df.comment.str.contains('x')"], usage_rows=1))
    assert g.label == "FAIL — TOOL EXECUTION"
    g = grade_from_catalogue(evidence("FE-14", counts, code=["df.comment.str.contains('x')"], usage_rows=2))
    assert g.outcome is T.INDETERMINATE
