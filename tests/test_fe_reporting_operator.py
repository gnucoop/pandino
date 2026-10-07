"""Tests for the run-centric projection (evaluation.datachat.reporting.operator_report).

Inputs come from the reader over synthetic graded packages built in a
repository whose ``ground_truth.json`` is the synthetic reference; other
persisted grading states are substituted as typed ``ScenarioGrade`` records.
Offline, no writes.
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
    ActivitySource,
    Gate,
    ObservedActivity,
    ScenarioGrade,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.model import SuiteStatus
from evaluation.datachat.reporting.operator_report import (
    OperatorReportError,
    ResultShape,
    build_operator_report,
)
from evaluation.datachat.reporting.reader import load_run_for_reporting
from evaluation.datachat.reporting.reference import load_evaluation_reference
from tests.test_fe_grading_grader import FE01, FE05, IDENTITY, grade, repo_root, snapshot  # noqa: F401
from tests.test_fe_grading_reader import build_run
from tests.test_fe_reporting_reference import ref_repo  # noqa: F401

REAL_RUN = Path(__file__).resolve().parents[1] / "evaluation/datachat/runs/20261007T073305Z_5a2cca3"
REF = "scenarios/FE-01/http/datachat.response.bin"
OK = (ExecutionAssessment.OK, ExecutionAssessmentReason.USABLE_OUTCOME)
F, T = TechnicalFailure, TechnicalOutcome


def loaded(tmp_path, repo_root, specs, **kwargs):
    root = build_run(tmp_path, repo_root, specs, **kwargs)
    grade(root, repo_root)
    return root, load_run_for_reporting(root)


def references(run, repo):
    return load_evaluation_reference(repo, run.run_grade.contract, [s.scenario_id for s in run.scenarios])


def project(run, repo):
    return build_operator_report(run, references(run, repo))


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


# --- Seven sections -----------------------------------------------------------------------


def test_scenario_fields_follow_the_seven_sections(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    report = project(run, ref_repo)
    s, source = report.scenarios[0], run.scenario("FE-01")
    assert (s.question, s.question_sent) == ("domanda?", True)
    assert s.agent_result.source is source.agent_result and s.agent_result.shape is ResultShape.TEXT
    assert s.activity == source.grade.activity and s.activity
    assert [(c.index, c.filename) for c in s.charts] == [(1, "FE-01_chart_1.png")]
    assert s.charts[0].spec is source.agent_result.content.charts[0]
    v = s.validation
    assert (v.gate, v.outcome, v.failures, v.execution_assessment) == (
        Gate.GRADEABLE, T.TECHNICAL_PASS, (), ExecutionAssessment.OK)
    assert v.checks == source.grade.checks
    assert s.reference.scenario_id == "FE-01" and s.reference.guarantees[0][0] == "G2"
    p = s.provenance
    assert p.execution_sha256 == source.execution_sha256 and p.request_ids and REF in p.evidence_refs
    assert (report.run_grade_sha256, report.grading_schema_version) == (run.run_grade_sha256, 2)


def test_no_grading_centric_or_semantic_projection_remains(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    report = project(run, ref_repo)
    s = report.scenarios[0]
    for name in ("attention", "verdict", "root_failure", "caveats", "pending_reviews", "attention_checks", "label"):
        assert not hasattr(s, name), name
    for name in ("attention_scenarios", "verdict", "score", "pass_rate", "outcome"):
        assert not hasattr(report, name), name


def test_activity_is_the_persisted_summary_in_order(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    activity = (ObservedActivity("classify_match", ActivitySource.RUNTIME, 1, "rejected"),
                ObservedActivity("classify_match", ActivitySource.GENERATED_CODE, 2),
                ObservedActivity("provider_backed_call", ActivitySource.USAGE, 1))
    run = regraded(run, "FE-01", outcome=T.TECHNICAL_PASS, activity=activity)
    assert project(run, ref_repo).scenarios[0].activity == activity


def test_multiple_failures_kept_in_recorded_order(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    chart = check("fe01.chart_present", CheckStatus.NOT_MET, F.ARTIFACT, expected=">= 1 chart", observed=0)
    provider = check("fe01.provider", CheckStatus.NOT_MET, F.TOOL_EXECUTION)
    run = regraded(run, "FE-01", outcome=T.FAIL, failures=(F.TOOL_EXECUTION, F.ARTIFACT), checks=(chart, provider))
    v = project(run, ref_repo).scenarios[0].validation
    assert (v.outcome, v.failures, v.checks) == (T.FAIL, (F.TOOL_EXECUTION, F.ARTIFACT), (chart, provider))


def test_execution_assessment_is_carried_unchanged(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    run = regraded(run, "FE-01", (ExecutionAssessment.INDETERMINATE, ExecutionAssessmentReason.ADAPTER_FALLBACK),
                   outcome=T.INDETERMINATE)
    v = project(run, ref_repo).scenarios[0].validation
    assert (v.outcome, v.execution_assessment, v.execution_assessment_reason) == (
        T.INDETERMINATE, ExecutionAssessment.INDETERMINATE, ExecutionAssessmentReason.ADAPTER_FALLBACK)


# --- Non-gradeable gates -----------------------------------------------------------------


def test_not_run_has_no_outcome_and_no_agent_result(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {}, status=SuiteStatus.ABORTED_INFRA, order=["FE-01"])
    report = project(run, ref_repo)
    s = report.scenarios[0]
    assert (s.validation.gate, s.validation.outcome, s.validation.execution_assessment) == (Gate.NOT_RUN, None, None)
    assert s.agent_result.shape is ResultShape.ABSENT and not s.question_sent and s.question is None
    assert s.activity == () and s.charts == () and s.reference is not None
    assert dict(report.gate_counts)[Gate.NOT_RUN] == 1 and report.status is SuiteStatus.ABORTED_INFRA


@pytest.mark.parametrize("gate", [Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE])
def test_not_scoreable_keeps_gate_and_execution_context(tmp_path, ref_repo, gate):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    failure = run.scenario("FE-01").execution.failure
    run = regraded(run, "FE-01", (None, None), gate=gate)
    s = project(run, ref_repo).scenarios[0]
    assert (s.validation.gate, s.validation.outcome, s.validation.failures) == (gate, None, ())
    assert s.provenance.execution_failure is failure and s.provenance.execution_status is not None
    assert s.agent_result.shape is ResultShape.TEXT  # captured evidence of a non-gradeable scenario


# --- Run counts / purity -----------------------------------------------------------------


def test_counts_are_split_from_the_persisted_map(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    report = project(run, ref_repo)
    persisted = run.run_grade.counts_by_outcome
    assert [o for o, _ in report.outcome_counts] == list(T)
    assert [g for g, _ in report.gate_counts] == [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE]
    assert {k.value: n for k, n in (*report.outcome_counts, *report.gate_counts)} == persisted
    assert {k.value: n for k, n in report.failure_counts} == run.run_grade.counts_by_failure


def test_unknown_count_key_is_an_error(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    refs = references(run, ref_repo)
    bad = replace(run, run_grade=replace(run.run_grade, counts_by_outcome={"PASS": 1}))
    with pytest.raises(OperatorReportError, match="unknown key"):
        build_operator_report(bad, refs)
    bad = replace(run, run_grade=replace(run.run_grade, counts_by_failure={"UNDERSTANDING": 1}))
    with pytest.raises(OperatorReportError, match="unknown key"):
        build_operator_report(bad, refs)


def test_projection_is_pure_and_deterministic(tmp_path, ref_repo, monkeypatch):
    root, run = loaded(tmp_path, ref_repo, {"FE-01": FE01, "FE-05": FE05})
    refs = references(run, ref_repo)
    before_files, before_input = snapshot(root, exclude=()), copy.deepcopy(run)

    def refuse(*_a, **_k):
        raise AssertionError("I/O attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr("builtins.open", refuse)
    first, second = build_operator_report(run, refs), build_operator_report(run, refs)
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
    report = project(run, contract.REPO_ROOT)
    assert (report.run_id, report.official, report.comparable) == ("20261007T073305Z_5a2cca3", False, False)
    s = report.scenarios[0]
    assert s.question == "Quante richieste ci sono per regione? Mostramele anche in un grafico."
    assert s.agent_result.shape is ResultShape.TEXT
    assert s.charts[0].spec["datasets"][0]["data"] == [140, 110, 80, 58, 12]
    assert (s.validation.outcome, s.validation.execution_assessment) == (T.TECHNICAL_PASS, ExecutionAssessment.OK)
    assert report.datachat == {"DATACHAT_PROVIDER": "Mistral", "DATACHAT_MODEL": "mistral-small-latest",
                               "DATACHAT_ENGINE": "smolagents"}
