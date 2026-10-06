"""Tests for evaluation.datachat.grading.verdict: review merge and verdict construction.

Offline and pure: synthetic CheckResult/ManualReviewItem, plus small
integration runs through the Slice 3 catalogue with synthetic evidence.
"""

import json

import pytest

from evaluation.datachat.grading import catalogue as cat
from evaluation.datachat.grading import verdict as v
from evaluation.datachat.grading.model import ExecutionAssessment
from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    Dimension,
    EvidenceClass,
    FailureCategory,
    Gate,
    ManualReviewItem,
    RequirementLevel,
    ReviewStatus,
    Verdict,
)
from tests.test_fe_grading_catalogue import CM_EVENT, GT, KW_EVENT, REGIONS, chart, digest, evidence, table, text

MUST, SHOULD, MAY = RequirementLevel.MUST, RequirementLevel.SHOULD, RequirementLevel.MAY
MET, NOT_MET, IND, NA = (CheckStatus.MET, CheckStatus.NOT_MET, CheckStatus.INDETERMINATE,
                         CheckStatus.NOT_APPLICABLE)
DIRECT, INFERRED = EvidenceClass.DIRECT, EvidenceClass.INFERRED
F = FailureCategory
SHA = "a" * 64
REF = ("scenarios/FE-01/http/datachat.response.bin",)


def chk(cid, status=MET, level=MUST, failure=F.ANALYTICAL, cls=DIRECT):
    return CheckResult(cid, Dimension.ANALYTICAL, level, cls, status, evidence_refs=REF,
                       failure_category=failure if status is NOT_MET and level is MUST else None)


def rev(cid, level=MUST, failure=F.HONESTY, status=ReviewStatus.PENDING):
    return ManualReviewItem(cid, Dimension.HONESTY, level, "Is it so?", REF,
                            failure if level is MUST else None, SHA, status=status)


def ans(cid, status=ReviewStatus.SATISFIED, sha=SHA, note=None):
    return v.ReviewAnswer(cid, status, "reviewer-1", sha, note)


def build(checks=(), reviews=(), answers=()):
    return v.build_scenario_grade(scenario_id="FE-01", gate=Gate.GRADEABLE, execution_sha256="e" * 64,
                                  teardown_verified=True, checks=checks, manual_reviews=reviews,
                                  review_answers=answers, execution_assessment=ExecutionAssessment.OK)


# --- Core verdict table ----------------------------------------------------------------


def test_pass():
    g = build([chk("m"), chk("s", level=SHOULD)])
    assert g.verdict is Verdict.PASS and g.root_failure is None and g.caveats == ()


def test_pass_with_caveat():
    g = build([chk("m"), chk("s", NOT_MET, SHOULD)])
    assert g.verdict is Verdict.PASS_WITH_CAVEAT and g.caveats == ("s",)


def test_should_inferred_negative_is_a_caveat():
    assert build([chk("m"), chk("s", NOT_MET, SHOULD, cls=INFERRED)]).verdict is Verdict.PASS_WITH_CAVEAT


def test_deterministic_fail():
    g = build([chk("m", NOT_MET)])
    assert g.verdict is Verdict.FAIL_ANALYTICAL and g.root_failure is F.ANALYTICAL and g.other_failures == ()


def test_must_indeterminate_needs_review():
    assert build([chk("m", IND)]).verdict is Verdict.NEEDS_REVIEW


def test_should_indeterminate_has_no_effect():
    assert build([chk("m"), chk("s", IND, SHOULD)]).verdict is Verdict.PASS


def test_pending_review_needs_review():
    assert build([chk("m")], [rev("r")]).verdict is Verdict.NEEDS_REVIEW


def test_inferred_must_negative_alone_needs_review():
    g = build([chk("i", NOT_MET, failure=F.TOOL_SELECTION, cls=INFERRED)])
    assert g.verdict is Verdict.NEEDS_REVIEW and g.root_failure is None


def test_direct_failure_stands_on_its_own_same_category():
    g = build([chk("i", NOT_MET, failure=F.TOOL_SELECTION, cls=INFERRED),
               chk("d", NOT_MET, failure=F.TOOL_SELECTION)])
    assert g.verdict is Verdict.FAIL_TOOL_SELECTION and g.other_failures == ()
    # Without the DIRECT check the same INFERRED negative stays unresolved: no confirmation rule exists.
    assert build([chk("i", NOT_MET, failure=F.TOOL_SELECTION, cls=INFERRED)]).verdict is Verdict.NEEDS_REVIEW
    import inspect
    assert "in direct" not in inspect.getsource(v._failures)


def test_inferred_negative_never_joins_other_failures():
    g = build([chk("i", NOT_MET, failure=F.TOOL_SELECTION, cls=INFERRED), chk("d", NOT_MET)])
    assert g.verdict is Verdict.FAIL_ANALYTICAL and g.other_failures == ()
    assert any(c.check_id == "i" for c in g.checks)


# --- Execution assessment ---------------------------------------------------------------

EA = ExecutionAssessment


def build_ea(assessment, checks=(), reviews=()):
    return v.build_scenario_grade(scenario_id="FE-01", gate=Gate.GRADEABLE, execution_sha256="e" * 64,
                                  teardown_verified=True, checks=checks, manual_reviews=reviews,
                                  execution_assessment=assessment)


def test_execution_ok_is_neutral():
    assert build_ea(EA.OK, [chk("m")]).verdict is Verdict.PASS


def test_execution_failed():
    g = build_ea(EA.FAILED, [chk("m")])
    assert g.verdict is Verdict.FAIL_EXECUTION and g.root_failure is F.EXECUTION


def test_execution_failed_is_root_over_other_failures_and_pending_reviews():
    g = build_ea(EA.FAILED, [chk("m", NOT_MET)], [rev("r")])
    assert g.verdict is Verdict.FAIL_EXECUTION and g.other_failures == (F.ANALYTICAL,)
    assert g.manual_reviews[0].status is ReviewStatus.PENDING


def test_execution_needs_review():
    assert build_ea(EA.NEEDS_REVIEW, [chk("m")]).verdict is Verdict.NEEDS_REVIEW
    g = build_ea(EA.NEEDS_REVIEW, [chk("m", NOT_MET)])
    assert g.verdict is Verdict.FAIL_ANALYTICAL and g.root_failure is F.ANALYTICAL


def test_execution_assessment_cannot_be_omitted():
    with pytest.raises(TypeError, match="execution_assessment"):
        v.build_scenario_grade(scenario_id="FE-01", gate=Gate.GRADEABLE, execution_sha256="e" * 64,
                               teardown_verified=True, checks=[chk("m")])


def test_execution_assessment_must_be_the_enum():
    with pytest.raises(v.VerdictBuildError):
        build_ea("FAILED", [chk("m")])


# --- Precedence ----------------------------------------------------------------------


@pytest.mark.parametrize("cats,root,others", [
    ((F.ANALYTICAL, F.UNDERSTANDING), F.UNDERSTANDING, (F.ANALYTICAL,)),
    ((F.ANALYTICAL, F.CHAINING), F.CHAINING, (F.ANALYTICAL,)),
    ((F.INTERPRETATION, F.ANALYTICAL), F.ANALYTICAL, (F.INTERPRETATION,)),
    (tuple(reversed(v.FAILURE_ORDER)), F.EXECUTION, v.FAILURE_ORDER[1:]),
])
def test_root_failure_precedence(cats, root, others):
    g = build([chk(f"c{i}", NOT_MET, failure=cat_) for i, cat_ in enumerate(cats)])
    assert g.root_failure is root and g.other_failures == others and g.verdict is root.verdict


def test_failure_categories_are_not_duplicated():
    g = build([chk("a", NOT_MET), chk("b", NOT_MET)], [rev("r", failure=F.ANALYTICAL)],
              [ans("r", ReviewStatus.FAILED)])
    assert g.root_failure is F.ANALYTICAL and g.other_failures == ()


def test_deterministic_fail_beats_pending_review():
    g = build([chk("m", NOT_MET)], [rev("r", failure=F.INTERPRETATION)])
    assert g.verdict is Verdict.FAIL_ANALYTICAL
    assert g.manual_reviews[0].status is ReviewStatus.PENDING


# --- Reviews ---------------------------------------------------------------------------


def test_should_review_lifecycle():
    reviews = [rev("s", SHOULD)]
    assert build([chk("m")], reviews).verdict is Verdict.NEEDS_REVIEW
    assert build([chk("m")], reviews, [ans("s")]).verdict is Verdict.PASS
    g = build([chk("m")], reviews, [ans("s", ReviewStatus.FAILED)])
    assert g.verdict is Verdict.PASS_WITH_CAVEAT and g.caveats == ("s",) and g.root_failure is None


def test_must_review_failed_and_satisfied():
    g = build([chk("m")], [rev("r", failure=F.HONESTY)], [ans("r", ReviewStatus.FAILED, note="nulls neutral")])
    assert g.verdict is Verdict.FAIL_HONESTY
    item = g.manual_reviews[0]
    assert (item.status, item.reviewer, item.note) == (ReviewStatus.FAILED, "reviewer-1", "nulls neutral")
    assert item.question == "Is it so?" and item.evidence_refs == REF and item.evidence_sha256 == SHA
    assert build([chk("m")], [rev("r")], [ans("r")]).verdict is Verdict.PASS


def test_not_applicable_never_fails_or_caveats():
    g = build([chk("m"), chk("n", NA), chk("s", NA, SHOULD)], [rev("r"), rev("q", SHOULD)],
              [ans("r", ReviewStatus.NOT_APPLICABLE), ans("q", ReviewStatus.NOT_APPLICABLE)])
    assert g.verdict is Verdict.PASS and g.caveats == ()


def test_may_never_changes_the_verdict():
    may = [chk("may1", NOT_MET, MAY, failure=None), chk("may2", IND, MAY)]
    for base, expected in (([chk("m")], Verdict.PASS), ([chk("m"), chk("s", NOT_MET, SHOULD)], Verdict.PASS_WITH_CAVEAT),
                           ([chk("m", NOT_MET)], Verdict.FAIL_ANALYTICAL), ([chk("m", IND)], Verdict.NEEDS_REVIEW)):
        assert build(base + may).verdict is expected


def test_stale_answer_is_ignored_and_reported():
    merge = v.merge_reviews([rev("r")], [ans("r", ReviewStatus.FAILED, sha="b" * 64)])
    assert merge.stale == ("r",) and merge.items[0].status is ReviewStatus.PENDING
    assert build([chk("m")], [rev("r")], [ans("r", ReviewStatus.FAILED, sha="b" * 64)]).verdict is Verdict.NEEDS_REVIEW


def test_matching_answer_applies_and_missing_stays_pending():
    merge = v.merge_reviews([rev("r"), rev("q")], [ans("r")])
    assert [i.status for i in merge.items] == [ReviewStatus.SATISFIED, ReviewStatus.PENDING] and merge.stale == ()


def test_duplicate_and_unknown_answers_rejected():
    with pytest.raises(v.ReviewInputError):
        v.merge_reviews([rev("r")], [ans("r"), ans("r", ReviewStatus.FAILED)])
    with pytest.raises(v.ReviewInputError):
        v.merge_reviews([rev("r")], [ans("nope")])


def test_pending_is_not_a_human_answer():
    with pytest.raises(v.ReviewInputError):
        ans("r", ReviewStatus.PENDING)


# --- Review file -----------------------------------------------------------------------


def _write(tmp_path, payload):
    path = tmp_path / "FE-01.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_load_review_answers(tmp_path):
    path = _write(tmp_path, {"scenario_id": "FE-01", "reviews": {
        "r": {"status": "FAILED", "reviewer": "ac", "note": "x", "evidence_sha256": SHA},
        "q": {"status": "NOT_APPLICABLE", "reviewer": "ac", "evidence_sha256": SHA}}})
    answers = v.load_review_answers(path, "FE-01")
    assert [(a.check_id, a.status) for a in answers] == [("r", ReviewStatus.FAILED), ("q", ReviewStatus.NOT_APPLICABLE)]
    assert v.load_review_answers(tmp_path / "absent.json", "FE-01") == ()


@pytest.mark.parametrize("payload", [
    "{not json",
    '{"scenario_id": "FE-01", "reviews": {"r": {"status": "SATISFIED", "reviewer": "a", "evidence_sha256": "'
    + SHA + '"}, "r": {"status": "FAILED", "reviewer": "a", "evidence_sha256": "' + SHA + '"}}}',
    {"scenario_id": "FE-02", "reviews": {}},
    {"scenario_id": "FE-01", "reviews": []},
    {"scenario_id": "FE-01", "reviews": {}, "extra": 1},
    {"scenario_id": "FE-01", "reviews": {"r": {"status": "PENDING", "reviewer": "a", "evidence_sha256": SHA}}},
    {"scenario_id": "FE-01", "reviews": {"r": {"status": "MAYBE", "reviewer": "a", "evidence_sha256": SHA}}},
    {"scenario_id": "FE-01", "reviews": {"r": {"status": "SATISFIED", "reviewer": "a"}}},
    {"scenario_id": "FE-01", "reviews": {"r": {"status": "SATISFIED", "reviewer": "", "evidence_sha256": SHA}}},
    {"scenario_id": "FE-01", "reviews": {"r": {"status": "SATISFIED", "reviewer": "a", "evidence_sha256": SHA,
                                               "verdict": "PASS"}}},
])
def test_malformed_review_files_rejected(tmp_path, payload):
    with pytest.raises(v.ReviewInputError):
        v.load_review_answers(_write(tmp_path, payload), "FE-01")


# --- Invariants, gates, purity ------------------------------------------------------------


def test_invariant_violations():
    with pytest.raises(v.VerdictBuildError):
        build([chk("x")], [rev("x")])
    with pytest.raises(v.VerdictBuildError):
        build([CheckResult("x", Dimension.ANALYTICAL, MUST, DIRECT, NOT_MET)])
    with pytest.raises(v.VerdictBuildError):
        build([], [ManualReviewItem("r", Dimension.HONESTY, MUST, "q?", REF, None, SHA)])


@pytest.mark.parametrize("gate", [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE])
def test_non_gradeable_gates_get_no_verdict(gate):
    g = v.build_scenario_grade(scenario_id="FE-03", gate=gate, execution_sha256=None, teardown_verified=None,
                               execution_assessment=ExecutionAssessment.OK)
    assert g.verdict is None and g.root_failure is None and g.checks == ()
    with pytest.raises(v.VerdictBuildError):
        v.build_scenario_grade(scenario_id="FE-03", gate=gate, execution_sha256=None, teardown_verified=None,
                               execution_assessment=ExecutionAssessment.OK,
                               checks=[chk("m")])


def test_deterministic_and_non_mutating():
    checks, reviews, answers = [chk("m"), chk("s", NOT_MET, SHOULD)], [rev("r")], [ans("r", ReviewStatus.FAILED)]
    snapshot = ([c.to_dict() for c in checks], [r.to_dict() for r in reviews], list(answers))
    first, second = build(checks, reviews, answers), build(checks, reviews, answers)
    assert first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict(), sort_keys=True)  # JSON-serializable
    assert ([c.to_dict() for c in checks], [r.to_dict() for r in reviews], list(answers)) == snapshot
    assert reviews[0].status is ReviewStatus.PENDING
    g = first.to_dict()
    assert g["schema_version"] == 1 and g["teardown_verified"] is True and g["execution_sha256"] == "e" * 64


# --- Catalogue integration ------------------------------------------------------------------


def grade_from_catalogue(ev, answers=()):
    result = cat.grade_scenario_checks(ev, GT, digest=digest)
    return v.build_scenario_grade(scenario_id=ev.scenario_id, gate=ev.gate, execution_sha256=ev.execution_sha256,
                                  teardown_verified=ev.teardown_verified, checks=result.checks,
                                  manual_reviews=result.manual_reviews, review_answers=answers,
                                  execution_assessment=ExecutionAssessment.OK), result


def answer_all(result, status=ReviewStatus.SATISFIED, **override):
    return [v.ReviewAnswer(r.check_id, override.get(r.check_id, status), "reviewer-1", r.evidence_sha256)
            for r in result.manual_reviews]


def test_fe01_pass():
    g, _ = grade_from_catalogue(evidence("FE-01", text("Ecco.", [chart("bar", list(REGIONS), list(REGIONS.values()))])))
    assert g.verdict is Verdict.PASS


def test_fe05_needs_review_then_caveat_for_missing_n():
    # Spearman record without n, plus prose that never states the sample size.
    ev = evidence("FE-05", text("Associazione negativa moderata (Spearman -0,58)."),
                  code=["correlation(col_x='wait_days', col_y='satisfaction', method='spearman')"])
    g, result = grade_from_catalogue(ev)
    assert g.verdict is Verdict.NEEDS_REVIEW
    g, _ = grade_from_catalogue(ev, answer_all(result))
    assert g.verdict is Verdict.PASS_WITH_CAVEAT and g.caveats == ("fe05.sample_size_reported",)


def test_fe14_taxonomy_review():
    counts = table([{"category": "Attesa", "count": 96}, {"category": "Personale", "count": 68}])
    ev = evidence("FE-14", counts, events=[KW_EVENT, CM_EVENT], usage_rows=2)
    g, result = grade_from_catalogue(ev)
    assert g.verdict is Verdict.NEEDS_REVIEW
    g, _ = grade_from_catalogue(ev, answer_all(result, **{"fe14.taxonomy_coherence": ReviewStatus.FAILED}))
    assert g.verdict is Verdict.FAIL_INTERPRETATION and g.root_failure is F.INTERPRETATION
    g, _ = grade_from_catalogue(ev, answer_all(result))
    assert g.verdict is Verdict.PASS
