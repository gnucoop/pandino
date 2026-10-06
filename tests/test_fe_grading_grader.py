"""Tests for evaluation.datachat.grading.grader: run orchestration and result persistence.

Offline only: synthetic run packages (real hashes) built with the reader-test
builders, synthetic ground truth pinned in a temporary repo root. No Maui, no
DB, no provider.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.evidence import RunLayout, sha256_bytes, sha256_file
from evaluation.datachat.grading import catalogue as cat
from evaluation.datachat.grading import grader as gr
from evaluation.datachat.grading.model import ExecutionAssessment, ExecutionAssessmentReason, Gate, Verdict
from evaluation.datachat.grading.reader import load_run_for_grading
from evaluation.datachat.model import Failure, InfraCode, ScenarioStatus, SuiteStatus
from tests.test_fe_grading_catalogue import GT, REGIONS, chart, table, text
from tests.test_fe_grading_reader import AGENT_RECORD, RID, RUN_ID, build_run

EA = ExecutionAssessment
RS = ExecutionAssessmentReason
IDENTITY = {"implementation": gr.GRADER_IMPLEMENTATION, "version": gr.GRADER_VERSION,
            "grading_schema_version": 1, "git_sha": "f" * 40, "dirty": False}


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    """Pinned synthetic evaluation artifacts; ground_truth.json carries the synthetic GT."""
    root = tmp_path / "repo"
    directory = contract.evaluation_dir(root)
    directory.mkdir(parents=True)
    pins = []
    for artifact in contract.PINNED_ARTIFACTS:
        if artifact.name == contract.GROUND_TRUTH_JSON_NAME:
            content = json.dumps(GT).encode()
        else:
            content = f"synthetic {artifact.name}\n".encode()
        (directory / artifact.name).write_bytes(content)
        pins.append(contract.PinnedArtifact(artifact.name, sha256_bytes(content), artifact.required))
    monkeypatch.setattr(contract, "PINNED_ARTIFACTS", tuple(pins))
    return root


def runtime(*tool_lines, final_kind="text", response_kind=None, fallback=False, end="ok", http=200):
    """A runtime log: optional ``(tool, "event=... k=v")`` lines, then the generic markers."""
    pre = f"2026-01-01T00:00:01+00:00 INFO datachat.runtime request_id={RID} app_id=dino"
    lines = [f"2026-01-01T00:00:01+00:00 INFO datachat.tools.{tool}_tool request_id={RID} app_id=dino {msg}"
             for tool, msg in tool_lines]
    lines += [
        f"{pre} final_answer_check request_id={RID} passed=True final_kind={final_kind} reason=OK",
        f"{pre} chat_end request_id={RID} response_kind={response_kind or final_kind}",
        f"{pre} datachat_adapter_status request_id={RID} adapter_fallback_used={fallback}",
        f"{pre} datachat_request_end request_id={RID} status={end} http_status={http}",
    ]
    return "\n".join(lines) + "\n"


def agent(*code, response_kind="text"):
    return {**AGENT_RECORD, "tool_calls": [{"tool": "python_interpreter", "args": c} for c in code],
            "extra": {"channel": "datachat", "response_kind": response_kind}}


def body(response):
    return {"explanation": None, "log_id": 7, "response": response}


FE01 = dict(body=body(text("Ecco i conteggi.", [chart("bar", list(REGIONS), list(REGIONS.values()))])),
            runtime_log=runtime(("chart", "event=tool_call_result kind=bar mode=records")),
            agent_record=agent("r = aggregate(group_by='region', op='count')", "chart(kind='bar', data=r)"))
FE05 = dict(body=body(text("Associazione negativa moderata (Spearman -0,58).")), runtime_log=runtime(),
            agent_record=agent("correlation(col_x='wait_days', col_y='satisfaction', method='spearman')"))
FE14 = dict(
    body=body(table([{"category": "Attesa", "count": 96}, {"category": "Personale", "count": 68}])),
    runtime_log=runtime(("keywords", "event=tool_call_result col=comment language=italian"),
                        ("classify_match", "event=tool_call_result tool=classify_match rows=400 analyzable_rows=340"),
                        final_kind="table"),
    agent_record=agent("keywords(col='comment')", "classify_match(col='comment')", response_kind="table"),
)


def grade(root, repo_root):
    return gr.grade_run(root, repo_root, grader_identity=lambda _: dict(IDENTITY))


def results(root):
    return root / "grading" / "results"


def scenario_json(root, sid):
    return json.loads((results(root) / "scenarios" / f"{sid}.json").read_text())


def write_review(root, sid, answers):
    path = root / "grading" / "reviews" / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenario_id": sid, "reviews": answers}))
    return path


def answer_all(root, sid, overrides=None, *, sha=None):
    overrides = overrides or {}
    return {r["check_id"]: {"status": overrides.get(r["check_id"], "SATISFIED"), "reviewer": "reviewer-1",
                            "evidence_sha256": sha or r["evidence_sha256"]}
            for r in scenario_json(root, sid)["manual_reviews"]}


def snapshot(root: Path, *, exclude=("grading/results/",)) -> dict[str, str]:
    return {rel: sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()
            for rel in [p.relative_to(root).as_posix()] if not rel.startswith(exclude)}


def loaded(tmp_path, repo_root, sid, **spec):
    root = build_run(tmp_path, repo_root, {sid: spec})
    return load_run_for_grading(root, repo_root).scenario(sid)


# --- Execution assessment ---------------------------------------------------------------


# Contract for the execution reason vocabulary: one case per assess_execution branch.
@pytest.mark.parametrize("spec,assessment,reason", [
    (FE01, EA.OK, RS.USABLE_OUTCOME),
    (dict(body=body(text("", [chart("bar", ["North"], [140])]))), EA.OK, RS.USABLE_OUTCOME),
    (dict(runtime_log=runtime(final_kind="error"), agent_record=agent("x", response_kind="error")),
     EA.FAILED, RS.ERROR_OUTCOME),
    (dict(runtime_log=runtime(final_kind="error"), agent_record=None), EA.FAILED, RS.ERROR_OUTCOME),
    (dict(http_status=500, body={"error": "NORMALIZE_FAILED"}, runtime_log=runtime(end="error", http=500)),
     EA.FAILED, RS.HTTP_5XX),
    (dict(body=body(text("   "))), EA.FAILED, RS.EMPTY_OUTCOME),
    (dict(body=body(table([]))), EA.FAILED, RS.EMPTY_OUTCOME),
    (dict(body=body({"type": "image", "value": ""})), EA.FAILED, RS.EMPTY_OUTCOME),
    (dict(runtime_log=runtime(final_kind="error"), agent_record=agent("x", response_kind="text")),
     EA.NEEDS_REVIEW, RS.CONFLICTING_KINDS),
    (dict(http_status=400, body={"error": "bad"}), EA.NEEDS_REVIEW, RS.HTTP_NON_2XX),
    (dict(body=b"{not json"), EA.NEEDS_REVIEW, RS.UNPARSEABLE_RESPONSE),
    (dict(body={"something": "else"}), EA.NEEDS_REVIEW, RS.UNPARSEABLE_RESPONSE),
    (dict(runtime_log=runtime(end="error")), EA.NEEDS_REVIEW, RS.RUNTIME_STATUS_CONFLICT),
    (dict(runtime_log=runtime(fallback=True)), EA.NEEDS_REVIEW, RS.ADAPTER_FALLBACK),
])
def test_execution_assessment_and_reason(tmp_path, repo_root, spec, assessment, reason):
    finding = gr.assess_execution(loaded(tmp_path, repo_root, "FE-01", **spec))
    assert (finding.assessment, finding.reason) == (assessment, reason)


def test_missing_response_needs_review(tmp_path, repo_root):
    ev = loaded(tmp_path, repo_root, "FE-01", **FE01)
    finding = gr.assess_execution(dataclasses.replace(ev, response=None))
    assert (finding.assessment, finding.reason) == (EA.NEEDS_REVIEW, RS.MISSING_RESPONSE)


def test_assessment_refuses_non_gradeable(tmp_path, repo_root):
    with pytest.raises(gr.RunGradingError):
        gr.assess_execution(loaded(tmp_path, repo_root, "FE-01", scoreable=False))


def test_assessment_has_no_scenario_specific_policy():
    source = Path(gr.__file__).read_text()
    assert "FE-0" not in source and "FE-1" not in source


# --- End-to-end orchestration -----------------------------------------------------------


def test_fe01_pass_end_to_end(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    rg = grade(root, repo_root)
    assert scenario_json(root, "FE-01")["verdict"] == "PASS"
    assert rg.grading_complete and rg.counts_by_verdict["PASS"] == 1


def test_fe05_pending_then_caveat(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-05": FE05})
    rg = grade(root, repo_root)
    assert scenario_json(root, "FE-05")["verdict"] == "NEEDS REVIEW" and not rg.grading_complete
    assert rg.scenarios[0].pending_reviews > 0
    write_review(root, "FE-05", answer_all(root, "FE-05"))
    rg = grade(root, repo_root)
    g = scenario_json(root, "FE-05")
    assert g["verdict"] == "PASS WITH CAVEAT" and g["caveats"] == ["fe05.sample_size_reported"]
    assert rg.grading_complete and rg.scenarios[0].pending_reviews == 0


def test_fe14_pending_then_interpretation_failure(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-14": FE14})
    grade(root, repo_root)
    assert scenario_json(root, "FE-14")["verdict"] == "NEEDS REVIEW"
    write_review(root, "FE-14", answer_all(root, "FE-14", {"fe14.taxonomy_coherence": "FAILED"}))
    grade(root, repo_root)
    g = scenario_json(root, "FE-14")
    assert g["verdict"] == "FAIL — INTERPRETATION" and g["root_failure"] == "INTERPRETATION"


def test_execution_failure_is_root_even_when_analysis_passes(tmp_path, repo_root):
    spec = {**FE01, "runtime_log": runtime(("chart", "event=tool_call_result kind=bar"), final_kind="error"),
            "agent_record": None}
    root = build_run(tmp_path, repo_root, {"FE-01": spec})
    grade(root, repo_root)
    g = scenario_json(root, "FE-01")
    assert g["verdict"] == "FAIL — EXECUTION" and g["root_failure"] == "EXECUTION"
    assert all(c["status"] == "MET" for c in g["checks"] if c["requirement_level"] == "MUST")


def test_non_gradeable_scenarios_get_null_verdicts(tmp_path, repo_root, monkeypatch):
    infra = Failure(InfraCode.INFRA_TOKEN_BUDGET, "not enough tokens")
    root = build_run(tmp_path, repo_root, {
        "FE-01": FE01,
        "FE-02": {"status": ScenarioStatus.NOT_SCOREABLE_INFRA, "failure": infra},
        "FE-03": {"scoreable": False},
        "FE-04": {"status": ScenarioStatus.NOT_RUN},
    }, status=SuiteStatus.ABORTED_INFRA)
    graded = []
    real = cat.grade_scenario_checks
    monkeypatch.setattr(cat, "grade_scenario_checks", lambda ev, *a, **k: graded.append(ev.scenario_id) or real(ev, *a, **k))
    rg = grade(root, repo_root)
    assert graded == ["FE-01"]
    for sid, gate in (("FE-02", "NOT_SCOREABLE_INFRA"), ("FE-03", "NOT_SCOREABLE"), ("FE-04", "NOT_RUN")):
        g = scenario_json(root, sid)
        assert (g["gate"], g["verdict"], g["root_failure"], g["checks"], g["manual_reviews"]) == (gate, None, None, [], [])
    assert [(e.scenario_id, e.gate, e.verdict) for e in rg.scenarios] == [
        ("FE-01", Gate.GRADEABLE, Verdict.PASS), ("FE-02", Gate.NOT_SCOREABLE_INFRA, None),
        ("FE-03", Gate.NOT_SCOREABLE, None), ("FE-04", Gate.NOT_RUN, None)]
    c = rg.counts_by_verdict
    assert (c["PASS"], c["NOT_SCOREABLE_INFRA"], c["NOT_SCOREABLE"], c["NOT_RUN"], c["FAIL — EXECUTION"]) == (1, 1, 1, 1, 0)
    assert rg.grading_complete


def test_scenario_order_follows_the_run(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-05": FE05, "FE-01": FE01})
    # Manifest order is the run's frozen order, here deliberately non-alphabetical.
    assert [e.scenario_id for e in grade(root, repo_root).scenarios] == ["FE-05", "FE-01"]


# --- RunGrade --------------------------------------------------------------------------


def test_run_grade_provenance(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    grade(root, repo_root)
    data = json.loads((results(root) / "run_grade.json").read_text())
    assert data["run_id"] == RUN_ID and (data["official"], data["comparable"]) == (False, False)
    assert data["grader"] == IDENTITY and data["schema_version"] == 1
    assert data["contract"]["ground_truth_sha256"] == contract.pinned_artifact(contract.GROUND_TRUTH_JSON_NAME).sha256
    assert data["contract"]["dataset_sha256"] == contract.pinned_artifact(contract.DATASET_NAME).sha256
    assert data["contract"]["scenario_execution_sha256"] == {"FE-01": sha256_file(RunLayout(root).execution("FE-01"))}
    assert "score" not in json.dumps(data) and "run_verdict" not in data


def test_default_grader_identity_reads_git(tmp_path):
    identity = gr.read_grader_identity(contract.REPO_ROOT)
    assert len(identity["git_sha"]) == 40 and isinstance(identity["dirty"], bool)
    with pytest.raises(gr.RunGradingError):
        gr.read_grader_identity(tmp_path)


# --- Reviews -----------------------------------------------------------------------------


def test_review_regeneration_leaves_reviews_untouched(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-05": FE05})
    grade(root, repo_root)
    assert not (root / "grading" / "reviews").exists()  # never auto-created
    path = write_review(root, "FE-05", answer_all(root, "FE-05"))
    before = path.read_bytes()
    grade(root, repo_root)
    assert path.read_bytes() == before


def test_stale_review_stays_pending(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-05": FE05})
    grade(root, repo_root)
    path = write_review(root, "FE-05", answer_all(root, "FE-05", sha="0" * 64))
    before = path.read_bytes()
    grade(root, repo_root)
    g = scenario_json(root, "FE-05")
    assert g["verdict"] == "NEEDS REVIEW" and all(r["status"] == "PENDING" for r in g["manual_reviews"])
    assert path.read_bytes() == before


@pytest.mark.parametrize("content", ["{broken", json.dumps({"scenario_id": "FE-05", "reviews": {
    "unknown.check": {"status": "SATISFIED", "reviewer": "r", "evidence_sha256": "0" * 64}}})])
def test_bad_review_input_fails_grading_without_writing(tmp_path, repo_root, content):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01, "FE-05": FE05})
    path = root / "grading" / "reviews" / "FE-05.json"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    before = snapshot(root, exclude=())
    with pytest.raises(gr.RunGradingError, match="FE-05"):
        grade(root, repo_root)
    assert snapshot(root, exclude=()) == before
    assert not results(root).exists()


# --- Persistence and immutability ------------------------------------------------------


def test_layout_and_immutability(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01, "FE-05": FE05})
    write_review(root, "FE-05", {})
    before = snapshot(root)
    grade(root, repo_root)
    assert snapshot(root) == before
    assert sorted(p.relative_to(results(root)).as_posix() for p in results(root).rglob("*") if p.is_file()) == [
        "run_grade.json", "scenarios/FE-01.json", "scenarios/FE-05.json"]
    assert not list(root.rglob("*.tmp"))


def test_regrading_is_byte_identical_and_replaces(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01, "FE-05": FE05})
    grade(root, repo_root)
    first = snapshot(results(root), exclude=())
    (results(root) / "scenarios" / "FE-01.json").write_text("stale derived output")
    grade(root, repo_root)
    assert snapshot(results(root), exclude=()) == first


def test_contract_mismatch_refused_without_writing(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    (contract.evaluation_dir(repo_root) / contract.DATASET_NAME).write_text("changed\n")
    before = snapshot(root, exclude=())
    with pytest.raises(gr.RunGradingError, match="contract"):
        grade(root, repo_root)
    assert snapshot(root, exclude=()) == before


def test_persistence_failure_is_a_grading_error(tmp_path, repo_root, monkeypatch):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    before = snapshot(root)

    def boom(path, obj):
        raise OSError("disk full")

    monkeypatch.setattr(gr, "write_json_atomic", boom)
    with pytest.raises(gr.RunGradingError, match="persist"):
        grade(root, repo_root)
    assert snapshot(root) == before


def test_grader_imports_no_runtime_dependencies():
    source = Path(gr.__file__).read_text()
    for forbidden in ("psycopg", "runner", "maui_process", "client", "datachat.engine", "requests"):
        assert f"import {forbidden}" not in source and f".{forbidden} import" not in source


# --- Execution provenance in persisted grades ------------------------------------------


def test_persisted_execution_provenance(tmp_path, repo_root):
    failed = {**FE01, "runtime_log": runtime(final_kind="error"), "agent_record": None}
    root = build_run(tmp_path, repo_root, {"FE-01": FE01, "FE-02": failed,
                                           "FE-03": {"status": ScenarioStatus.NOT_RUN}},
                     status=SuiteStatus.ABORTED_INFRA)
    grade(root, repo_root)
    picked = {sid: {k: scenario_json(root, sid)[k] for k in ("execution_assessment", "execution_assessment_reason",
                                                            "verdict")} for sid in ("FE-01", "FE-02", "FE-03")}
    assert picked == {
        "FE-01": {"execution_assessment": "OK", "execution_assessment_reason": "USABLE_OUTCOME", "verdict": "PASS"},
        "FE-02": {"execution_assessment": "FAILED", "execution_assessment_reason": "ERROR_OUTCOME",
                  "verdict": "FAIL — EXECUTION"},
        "FE-03": {"execution_assessment": None, "execution_assessment_reason": None, "verdict": None},
    }
    first = snapshot(results(root), exclude=())
    grade(root, repo_root)
    assert snapshot(results(root), exclude=()) == first


def test_execution_ambiguity_with_and_without_analytical_failure(tmp_path, repo_root):
    wrong = dict(REGIONS, North=999)
    wrong_fe01 = {**FE01, "runtime_log": runtime(("chart", "event=tool_call_result kind=bar"), fallback=True),
                  "body": body(text("Ecco.", [chart("bar", list(wrong), list(wrong.values()))]))}
    ambiguous_fe01 = {**FE01, "runtime_log": runtime(("chart", "event=tool_call_result kind=bar"), fallback=True)}
    root = build_run(tmp_path, repo_root, {"FE-01": ambiguous_fe01})
    grade(root, repo_root)
    g = scenario_json(root, "FE-01")
    assert (g["execution_assessment"], g["execution_assessment_reason"], g["verdict"]) == (
        "NEEDS_REVIEW", "ADAPTER_FALLBACK", "NEEDS REVIEW")
    other = tmp_path / "second"
    root = build_run(other, repo_root, {"FE-01": wrong_fe01})
    grade(root, repo_root)
    g = scenario_json(root, "FE-01")
    assert (g["execution_assessment"], g["execution_assessment_reason"], g["verdict"]) == (
        "NEEDS_REVIEW", "ADAPTER_FALLBACK", "FAIL — ANALYTICAL")


def test_reviews_cannot_override_execution_assessment(tmp_path, repo_root):
    spec = {**FE05, "runtime_log": runtime(fallback=True)}
    root = build_run(tmp_path, repo_root, {"FE-05": spec})
    grade(root, repo_root)
    write_review(root, "FE-05", answer_all(root, "FE-05"))
    grade(root, repo_root)
    g = scenario_json(root, "FE-05")
    assert all(r["status"] == "SATISFIED" for r in g["manual_reviews"])
    assert (g["execution_assessment"], g["execution_assessment_reason"], g["verdict"]) == (
        "NEEDS_REVIEW", "ADAPTER_FALLBACK", "NEEDS REVIEW")
    # An answer keyed by an execution "check id" is an unknown id, not an override.
    write_review(root, "FE-05", {**answer_all(root, "FE-05"), "execution_assessment": {
        "status": "SATISFIED", "reviewer": "r", "evidence_sha256": "0" * 64}})
    with pytest.raises(gr.RunGradingError, match="unknown check ids"):
        grade(root, repo_root)
