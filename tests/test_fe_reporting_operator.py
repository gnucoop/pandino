"""Tests for evaluation.datachat.reporting.operator_report (Reporting v1 Slice 2).

Inputs come from the Slice 1 reader over synthetic graded packages; other
persisted grading states are substituted as typed ``ScenarioGrade`` records,
since the projection consumes typed inputs only. Offline, no writes.
"""

import copy
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    Dimension,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    FailureCategory,
    Gate,
    ManualReviewItem,
    RequirementLevel,
    ReviewStatus,
    ScenarioGrade,
    ScenarioIndexEntry,
    Verdict,
)
from evaluation.datachat.model import SuiteStatus
from evaluation.datachat.reporting.operator_report import (
    AttentionReason,
    OperatorReportError,
    ResultShape,
    build_operator_report,
)
from evaluation.datachat.reporting.reader import load_run_for_reporting
from tests.test_fe_grading_grader import FE01, FE05, grade, repo_root, snapshot  # noqa: F401
from tests.test_fe_grading_reader import build_run

REAL_RUN = Path(__file__).resolve().parents[1] / "evaluation/datachat/runs/20261007T073305Z_5a2cca3"
REF = "scenarios/FE-01/http/datachat.response.bin"
SHA = "a" * 64
MUST, SHOULD = RequirementLevel.MUST, RequirementLevel.SHOULD
OK = (ExecutionAssessment.OK, ExecutionAssessmentReason.USABLE_OUTCOME)


def loaded(tmp_path, repo_root, specs, **kwargs):
    root = build_run(tmp_path, repo_root, specs, **kwargs)
    grade(root, repo_root)
    return root, load_run_for_reporting(root)


def check(cid, status, level=MUST, category=None, **kw):
    return CheckResult(cid, Dimension.ANALYTICAL, level, kw.pop("evidence_class", EvidenceClass.DIRECT), status,
                       evidence_refs=(REF,), failure_category=category, **kw)


def review(cid, status=ReviewStatus.PENDING, level=MUST, category=FailureCategory.INTERPRETATION):
    return ManualReviewItem(cid, Dimension.INTERPRETATION, level, "La risposta è prudente?", (REF,),
                            category, SHA, status)


def regraded(run, sid, assessment=OK, **fields):
    """The same input with one scenario's persisted grade (and index entry) replaced."""
    s = run.scenario(sid)
    g = ScenarioGrade(sid, fields.pop("gate", Gate.GRADEABLE), execution_assessment=assessment[0],
                      execution_assessment_reason=assessment[1], execution_sha256=s.execution_sha256,
                      teardown_verified=fields.pop("teardown_verified", True), **fields)
    pending = sum(r.status is ReviewStatus.PENDING for r in g.manual_reviews)
    entries = tuple(ScenarioIndexEntry(sid, g.gate, g.verdict, pending) if e.scenario_id == sid else e
                    for e in run.run_grade.scenarios)
    scenarios = tuple(replace(x, grade=g, index=entries[i]) if x.scenario_id == sid else x
                      for i, x in enumerate(run.scenarios))
    return replace(run, run_grade=replace(run.run_grade, scenarios=entries), scenarios=scenarios)


# --- PASS / fidelity ---------------------------------------------------------------------


def test_pass_is_compact_and_keeps_detail_and_provenance(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    report = build_operator_report(run)
    s = report.scenarios[0]

    assert (s.question, s.question_sent) == ("domanda?", True)
    assert (s.gate, s.verdict, s.execution_assessment) == (Gate.GRADEABLE, Verdict.PASS, ExecutionAssessment.OK)
    assert not s.attention.required and report.attention_scenarios == ()
    assert (s.execution_issue, s.root_failure, s.failures, s.caveats) == (None, None, (), ())
    assert (s.pending_reviews, s.attention_checks) == ((), ())
    assert s.check_status_counts == ((CheckStatus.MET, len(run.scenario("FE-01").grade.checks)),)
    assert s.detail.checks == s.detail.normal_checks == run.scenario("FE-01").grade.checks
    assert s.provenance.execution_sha256 == run.scenario("FE-01").execution_sha256
    assert s.provenance.request_ids and REF in s.provenance.evidence_refs
    assert report.provenance.run_grade_sha256 == run.run_grade_sha256


def test_agent_result_is_the_slice1_result_unchanged(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    a = build_operator_report(run).scenarios[0].agent_result
    source = run.scenario("FE-01").agent_result
    assert a.source is source and a.shape is ResultShape.TEXT
    assert a.source.content.value == "Ecco i conteggi." and a.source.body == source.body
    assert a.charts == source.content.charts and a.charts[0]["type"] == "bar"


# --- Exceptional verdicts ----------------------------------------------------------------


def test_pass_with_caveat_resolves_persisted_ids_only(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    miss = check("fe01.chart_bar", CheckStatus.NOT_MET, SHOULD)
    soft = review("fe01.soft", ReviewStatus.FAILED, SHOULD, None)
    run = regraded(run, "FE-01", verdict=Verdict.PASS_WITH_CAVEAT, caveats=("fe01.chart_bar", "fe01.soft"),
                   checks=(check("fe01.region_counts", CheckStatus.MET), miss), manual_reviews=(soft,))
    s = build_operator_report(run).scenarios[0]
    assert s.attention.reasons == (AttentionReason.PASS_WITH_CAVEAT,)
    assert [(c.check_id, c.check, c.review) for c in s.caveats] == [
        ("fe01.chart_bar", miss, None), ("fe01.soft", None, soft)]


def test_unresolvable_caveat_is_an_error(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", verdict=Verdict.PASS_WITH_CAVEAT, caveats=("fe01.ghost",))
    with pytest.raises(OperatorReportError, match="fe01.ghost"):
        build_operator_report(run)


def test_fail_exposes_categories_by_equality_only(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    counts = check("fe01.region_counts", CheckStatus.NOT_MET, category=FailureCategory.ANALYTICAL,
                   expected={"north": 140}, observed={"north": 141})
    inferred = check("fe01.inferred", CheckStatus.NOT_MET, category=FailureCategory.ANALYTICAL,
                     evidence_class=EvidenceClass.INFERRED)
    honesty = review("fe01.honest", ReviewStatus.FAILED, category=FailureCategory.HONESTY)
    run = regraded(run, "FE-01", verdict=Verdict.FAIL_ANALYTICAL, root_failure=FailureCategory.ANALYTICAL,
                   other_failures=(FailureCategory.HONESTY,),
                   checks=(counts, inferred, check("fe01.chart_present", CheckStatus.MET)),
                   manual_reviews=(honesty,))
    s = build_operator_report(run).scenarios[0]
    assert s.attention.reasons == (AttentionReason.FAIL_VERDICT,)
    assert (s.root_failure, s.other_failures) == (FailureCategory.ANALYTICAL, (FailureCategory.HONESTY,))
    root, other = s.failures
    # No evidence-class filtering: every item carrying the category is listed, none is called the cause.
    assert root.is_root and root.checks_carrying_category == (counts, inferred)
    assert other.reviews_carrying_category == (honesty,) and other.checks_carrying_category == ()
    assert s.attention_checks == (counts, inferred)
    assert (s.attention_checks[0].expected, s.attention_checks[0].observed) == ({"north": 140}, {"north": 141})
    assert [c.check_id for c in s.detail.normal_checks] == ["fe01.chart_present"]


def test_execution_failure_is_explained_by_assessment_not_checks(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", (ExecutionAssessment.FAILED, ExecutionAssessmentReason.ERROR_OUTCOME),
                   verdict=Verdict.FAIL_EXECUTION, root_failure=FailureCategory.EXECUTION)
    s = build_operator_report(run).scenarios[0]
    assert s.execution_issue.reason is ExecutionAssessmentReason.ERROR_OUTCOME
    assert s.failures[0].checks_carrying_category == s.failures[0].reviews_carrying_category == ()
    assert s.attention.reasons == (AttentionReason.FAIL_VERDICT, AttentionReason.EXECUTION_NOT_OK)


def test_pending_manual_review_is_exposed_not_decided(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-05": FE05})
    report = build_operator_report(run)
    s = report.scenarios[0]
    persisted = [r for r in run.scenario("FE-05").grade.manual_reviews if r.status is ReviewStatus.PENDING]
    assert s.verdict is Verdict.NEEDS_REVIEW and persisted
    assert {AttentionReason.NEEDS_REVIEW_VERDICT, AttentionReason.PENDING_REVIEWS} <= set(s.attention.reasons)
    assert [p.item for p in s.pending_reviews] == persisted
    p = s.pending_reviews[0]
    assert (p.item.question, p.item.dimension, p.item.requirement_level, p.item.evidence_refs) == (
        persisted[0].question, persisted[0].dimension, persisted[0].requirement_level, persisted[0].evidence_refs)
    assert p.answer_file == "grading/reviews/FE-05.json" and p.item.status is ReviewStatus.PENDING
    assert report.incomplete_scenarios == ("FE-05",) and not report.grading_complete


def test_execution_needs_review_has_no_invented_review_item(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", (ExecutionAssessment.NEEDS_REVIEW, ExecutionAssessmentReason.ADAPTER_FALLBACK),
                   verdict=Verdict.NEEDS_REVIEW, checks=(check("fe01.region_counts", CheckStatus.MET),))
    s = build_operator_report(run).scenarios[0]
    assert s.pending_reviews == () and s.attention_checks == ()
    assert (s.execution_issue.assessment, s.execution_issue.resolvable_by_review) == (
        ExecutionAssessment.NEEDS_REVIEW, False)
    assert s.attention.reasons == (AttentionReason.NEEDS_REVIEW_VERDICT, AttentionReason.EXECUTION_NOT_OK)


# --- Non-gradeable gates -----------------------------------------------------------------


def test_not_run_has_no_verdict_and_no_agent_result(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {}, status=SuiteStatus.ABORTED_INFRA, order=["FE-01"])
    report = build_operator_report(run)
    s = report.scenarios[0]
    assert (s.gate, s.verdict, s.has_verdict, s.execution_assessment) == (Gate.NOT_RUN, None, False, None)
    assert s.agent_result.shape is ResultShape.ABSENT and not s.question_sent
    assert s.attention.reasons == (AttentionReason.NOT_GRADEABLE,)
    assert dict(report.gate_counts)[Gate.NOT_RUN] == 1
    assert report.detail.status is SuiteStatus.ABORTED_INFRA


@pytest.mark.parametrize("gate", [Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE])
def test_not_scoreable_keeps_gate_and_execution_context(tmp_path, repo_root, gate):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    failure = run.scenario("FE-01").execution.failure
    run = regraded(run, "FE-01", (None, None), gate=gate)
    s = build_operator_report(run).scenarios[0]
    assert (s.gate, s.verdict, s.execution_issue, s.failures) == (gate, None, None, ())
    assert s.gate_execution_failure is failure and s.detail.execution_status is not None
    assert s.agent_result.shape is ResultShape.TEXT  # captured evidence of a non-gradeable scenario
    assert s.attention.reasons == (AttentionReason.NOT_GRADEABLE,)


def test_teardown_not_verified_draws_attention(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", verdict=Verdict.PASS, teardown_verified=False)
    s = build_operator_report(run).scenarios[0]
    assert s.verdict is Verdict.PASS and s.attention.reasons == (AttentionReason.TEARDOWN_NOT_VERIFIED,)


# --- Run counts / purity -----------------------------------------------------------------


def test_counts_are_split_from_the_persisted_map(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    report = build_operator_report(run)
    persisted = run.run_grade.counts_by_verdict
    assert [v.value for v, _ in report.verdict_counts] == [v.value for v in Verdict]
    assert [g for g, _ in report.gate_counts] == [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE]
    assert {k.value: n for k, n in (*report.verdict_counts, *report.gate_counts)} == persisted
    assert not any(hasattr(report, a) for a in ("verdict", "score", "pass_rate"))


def test_unknown_count_key_is_an_error(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = replace(run, run_grade=replace(run.run_grade, counts_by_verdict={"FAIL": 1}))
    with pytest.raises(OperatorReportError, match="unknown key"):
        build_operator_report(run)


def test_projection_is_pure_and_deterministic(tmp_path, repo_root, monkeypatch):
    root, run = loaded(tmp_path, repo_root, {"FE-01": FE01, "FE-05": FE05})
    before_files, before_input = snapshot(root, exclude=()), copy.deepcopy(run)

    def refuse(*_a, **_k):
        raise AssertionError("I/O attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr("builtins.open", refuse)
    first, second = build_operator_report(run), build_operator_report(run)
    monkeypatch.undo()

    assert first == second and run == before_input
    assert snapshot(root, exclude=()) == before_files and not (root / "reports").exists()


# --- Real FE-01 (read-only) --------------------------------------------------------------


@pytest.mark.skipif(not REAL_RUN.is_dir(), reason="real FE-01 run package not present locally")
def test_real_fe01_projection():
    run = load_run_for_reporting(REAL_RUN)
    report = build_operator_report(run)
    assert (report.run_id, report.official, report.comparable, report.grading_complete) == (
        "20261007T073305Z_5a2cca3", False, False, True)
    s = report.scenarios[0]
    assert s.question == "Quante richieste ci sono per regione? Mostramele anche in un grafico."
    assert s.agent_result.source is run.scenario("FE-01").agent_result and s.agent_result.shape is ResultShape.TEXT
    chart = s.agent_result.charts[0]
    assert chart["type"] == "bar" and chart["datasets"][0]["data"] == [140, 110, 80, 58, 12]
    assert (s.gate, s.verdict, s.execution_assessment) == (Gate.GRADEABLE, Verdict.PASS, ExecutionAssessment.OK)
    assert not s.attention.required and s.check_status_counts == ((CheckStatus.MET, 4),)
    assert (s.caveats, s.pending_reviews, s.detail.resolved_reviews) == ((), (), ())
    assert report.detail.datachat == {"DATACHAT_PROVIDER": "Mistral", "DATACHAT_MODEL": "mistral-small-latest",
                                      "DATACHAT_ENGINE": "smolagents"}
