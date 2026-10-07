"""Tests for evaluation.datachat.reporting.operator_report over grading schema 2.

Inputs come from the Slice 1 reader over synthetic graded packages; other
persisted grading states are substituted as typed ``ScenarioGrade`` records,
since the projection consumes typed inputs only. Offline, no writes.
"""

import copy
import shutil
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.grading import grader as gr
from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ScenarioGrade,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.model import SuiteStatus
from evaluation.datachat.reporting.operator_report import (
    AttentionReason,
    OperatorReportError,
    ResultShape,
    build_operator_report,
)
from evaluation.datachat.reporting.reader import load_run_for_reporting
from tests.test_fe_grading_grader import FE01, FE05, IDENTITY, grade, repo_root, snapshot  # noqa: F401
from tests.test_fe_grading_reader import build_run

REAL_RUN = Path(__file__).resolve().parents[1] / "evaluation/datachat/runs/20261007T073305Z_5a2cca3"
REF = "scenarios/FE-01/http/datachat.response.bin"
OK = (ExecutionAssessment.OK, ExecutionAssessmentReason.USABLE_OUTCOME)
F, T = TechnicalFailure, TechnicalOutcome


def loaded(tmp_path, repo_root, specs, **kwargs):
    root = build_run(tmp_path, repo_root, specs, **kwargs)
    grade(root, repo_root)
    return root, load_run_for_reporting(root)


def check(cid, status, failure=None, **kw):
    return CheckResult(cid, kw.pop("evidence_class", EvidenceClass.DIRECT), status, evidence_refs=(REF,),
                       failure=failure, **kw)


def regraded(run, sid, assessment=OK, **fields):
    """The same input with one scenario's persisted grade (and index entry) replaced."""
    s = run.scenario(sid)
    g = ScenarioGrade(sid, fields.pop("gate", Gate.GRADEABLE), execution_assessment=assessment[0],
                      execution_assessment_reason=assessment[1], execution_sha256=s.execution_sha256,
                      teardown_verified=fields.pop("teardown_verified", True), **fields)
    entries = tuple(ScenarioIndexEntry(sid, g.gate, g.outcome, g.failures) if e.scenario_id == sid else e
                    for e in run.run_grade.scenarios)
    scenarios = tuple(replace(x, grade=g, index=entries[i]) if x.scenario_id == sid else x
                      for i, x in enumerate(run.scenarios))
    return replace(run, run_grade=replace(run.run_grade, scenarios=entries), scenarios=scenarios)


# --- TECHNICAL PASS / fidelity -----------------------------------------------------------


def test_pass_is_compact_and_keeps_detail_and_provenance(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    report = build_operator_report(run)
    s = report.scenarios[0]

    assert (s.question, s.question_sent) == ("domanda?", True)
    assert (s.gate, s.outcome, s.execution_assessment) == (Gate.GRADEABLE, T.TECHNICAL_PASS, ExecutionAssessment.OK)
    assert s.label == "TECHNICAL PASS"
    assert not s.attention.required and report.attention_scenarios == ()
    assert (s.execution_issue, s.failures, s.attention_checks) == (None, (), ())
    assert s.check_status_counts == ((CheckStatus.MET, len(run.scenario("FE-01").grade.checks)),)
    assert s.detail.checks == s.detail.normal_checks == run.scenario("FE-01").grade.checks
    assert s.provenance.execution_sha256 == run.scenario("FE-01").execution_sha256
    assert s.provenance.request_ids and REF in s.provenance.evidence_refs
    assert report.provenance.run_grade_sha256 == run.run_grade_sha256
    assert report.provenance.grading_schema_version == 2


def test_agent_result_is_the_slice1_result_unchanged(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    a = build_operator_report(run).scenarios[0].agent_result
    source = run.scenario("FE-01").agent_result
    assert a.source is source and a.shape is ResultShape.TEXT
    assert a.source.content.value == "Ecco i conteggi." and a.source.body == source.body
    assert a.charts == source.content.charts and a.charts[0]["type"] == "bar"


def test_no_semantic_or_review_projection_remains(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    report = build_operator_report(run)
    s = report.scenarios[0]
    for name in ("verdict", "root_failure", "caveats", "pending_reviews", "other_failures"):
        assert not hasattr(s, name), name
    for name in ("grading_complete", "verdict_counts", "incomplete_scenarios"):
        assert not hasattr(report, name), name
    assert not hasattr(s.detail, "resolved_reviews") and not hasattr(s.provenance, "review_evidence_sha256")


# --- Exceptional outcomes ----------------------------------------------------------------


def test_multiple_failures_exposed_by_equality_only_and_unranked(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    chart = check("fe01.chart_present", CheckStatus.NOT_MET, F.ARTIFACT, expected=">= 1 chart", observed=0)
    inferred = check("fe01.inferred", CheckStatus.NOT_MET, F.TOOL_EXECUTION, evidence_class=EvidenceClass.INFERRED)
    provider = check("fe01.provider", CheckStatus.NOT_MET, F.TOOL_EXECUTION)
    run = regraded(run, "FE-01", outcome=T.FAIL, failures=(F.TOOL_EXECUTION, F.ARTIFACT),
                   checks=(chart, inferred, provider, check("fe01.other", CheckStatus.MET)))
    s = build_operator_report(run).scenarios[0]
    assert s.attention.reasons == (AttentionReason.FAIL_OUTCOME,)
    assert s.label == "FAIL — TOOL EXECUTION, ARTIFACT"
    tool, artifact = s.failures
    assert not hasattr(tool, "is_root")
    # No evidence-class filtering: every check carrying the category is listed, none is called the cause.
    assert (tool.category, tool.checks_carrying_category) == (F.TOOL_EXECUTION, (inferred, provider))
    assert (artifact.category, artifact.checks_carrying_category) == (F.ARTIFACT, (chart,))
    assert s.attention_checks == (chart, inferred, provider)
    assert (s.attention_checks[0].expected, s.attention_checks[0].observed) == (">= 1 chart", 0)
    assert [c.check_id for c in s.detail.normal_checks] == ["fe01.other"]


def test_execution_failure_is_explained_by_assessment_not_checks(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", (ExecutionAssessment.FAILED, ExecutionAssessmentReason.ERROR_OUTCOME),
                   outcome=T.FAIL, failures=(F.EXECUTION,))
    s = build_operator_report(run).scenarios[0]
    assert s.execution_issue.reason is ExecutionAssessmentReason.ERROR_OUTCOME
    assert s.failures[0].checks_carrying_category == ()
    assert s.attention.reasons == (AttentionReason.FAIL_OUTCOME, AttentionReason.EXECUTION_NOT_OK)


def test_response_contract_violation_from_execution(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", (ExecutionAssessment.RESPONSE_CONTRACT_VIOLATION,
                                  ExecutionAssessmentReason.HTTP_NON_2XX),
                   outcome=T.FAIL, failures=(F.RESPONSE_CONTRACT,))
    s = build_operator_report(run).scenarios[0]
    assert (s.execution_issue.assessment, s.label) == (ExecutionAssessment.RESPONSE_CONTRACT_VIOLATION,
                                                       "FAIL — RESPONSE CONTRACT")


def test_indeterminate_execution(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", (ExecutionAssessment.INDETERMINATE, ExecutionAssessmentReason.ADAPTER_FALLBACK),
                   outcome=T.INDETERMINATE, checks=(check("fe01.chart_present", CheckStatus.MET),))
    s = build_operator_report(run).scenarios[0]
    assert s.attention_checks == () and s.failures == ()
    assert s.execution_issue.assessment is ExecutionAssessment.INDETERMINATE
    assert s.attention.reasons == (AttentionReason.INDETERMINATE_OUTCOME, AttentionReason.EXECUTION_NOT_OK)


# --- Non-gradeable gates -----------------------------------------------------------------


def test_not_run_has_no_outcome_and_no_agent_result(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {}, status=SuiteStatus.ABORTED_INFRA, order=["FE-01"])
    report = build_operator_report(run)
    s = report.scenarios[0]
    assert (s.gate, s.outcome, s.has_outcome, s.label, s.execution_assessment) == (
        Gate.NOT_RUN, None, False, None, None)
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
    assert (s.gate, s.outcome, s.execution_issue, s.failures) == (gate, None, None, ())
    assert s.gate_execution_failure is failure and s.detail.execution_status is not None
    assert s.agent_result.shape is ResultShape.TEXT  # captured evidence of a non-gradeable scenario
    assert s.attention.reasons == (AttentionReason.NOT_GRADEABLE,)


def test_teardown_not_verified_draws_attention(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", outcome=T.TECHNICAL_PASS, teardown_verified=False)
    s = build_operator_report(run).scenarios[0]
    assert s.outcome is T.TECHNICAL_PASS and s.attention.reasons == (AttentionReason.TEARDOWN_NOT_VERIFIED,)


# --- Run counts / purity -----------------------------------------------------------------


def test_counts_are_split_from_the_persisted_map(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    report = build_operator_report(run)
    persisted = run.run_grade.counts_by_outcome
    assert [o for o, _ in report.outcome_counts] == list(T)
    assert [g for g, _ in report.gate_counts] == [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE]
    assert {k.value: n for k, n in (*report.outcome_counts, *report.gate_counts)} == persisted
    assert {k.value: n for k, n in report.failure_counts} == run.run_grade.counts_by_failure
    assert [f for f, _ in report.failure_counts] == list(F)
    assert not any(hasattr(report, a) for a in ("outcome", "verdict", "score", "pass_rate"))


def test_unknown_count_key_is_an_error(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    bad = replace(run, run_grade=replace(run.run_grade, counts_by_outcome={"PASS": 1}))
    with pytest.raises(OperatorReportError, match="unknown key"):
        build_operator_report(bad)
    bad = replace(run, run_grade=replace(run.run_grade, counts_by_failure={"UNDERSTANDING": 1}))
    with pytest.raises(OperatorReportError, match="unknown key"):
        build_operator_report(bad)


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


# --- Real FE-01 (regraded scratch copy; the original is never touched) --------------------


def regraded_real_copy(tmp_path):
    """A scratch copy of ``REAL_RUN`` regraded under the current grading schema."""
    root = tmp_path / REAL_RUN.name
    shutil.copytree(REAL_RUN, root, ignore=shutil.ignore_patterns("grading", "reports"))
    gr.grade_run(root, contract.REPO_ROOT, grader_identity=lambda _: dict(IDENTITY))
    return root


@pytest.mark.skipif(not REAL_RUN.is_dir(), reason="real FE-01 run package not present locally")
def test_real_fe01_projection(tmp_path):
    run = load_run_for_reporting(regraded_real_copy(tmp_path))
    report = build_operator_report(run)
    assert (report.run_id, report.official, report.comparable) == ("20261007T073305Z_5a2cca3", False, False)
    s = report.scenarios[0]
    assert s.question == "Quante richieste ci sono per regione? Mostramele anche in un grafico."
    assert s.agent_result.source is run.scenario("FE-01").agent_result and s.agent_result.shape is ResultShape.TEXT
    chart = s.agent_result.charts[0]
    assert chart["type"] == "bar" and chart["datasets"][0]["data"] == [140, 110, 80, 58, 12]
    assert (s.gate, s.outcome, s.execution_assessment) == (Gate.GRADEABLE, T.TECHNICAL_PASS, ExecutionAssessment.OK)
    assert not s.attention.required and s.check_status_counts == ((CheckStatus.MET, 1),)
    assert report.detail.datachat == {"DATACHAT_PROVIDER": "Mistral", "DATACHAT_MODEL": "mistral-small-latest",
                                      "DATACHAT_ENGINE": "smolagents"}
